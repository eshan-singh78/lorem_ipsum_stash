"""
Correction Layer — v6
Architecture: CONFIDENCE-AWARE LLM CORRECTION → RULE FALLBACK → MANDATORY INVARIANTS

Responsibilities:
  - LLM correction: only touches low-confidence or null fields
  - Rule fallback: fires when LLM fails; same field-ownership constraints
  - Mandatory invariants: panic→low, exp<1→fks≤2, enum validation
  - Updates confidence to "medium" / source to "llm_correction" after correction
  - NO duplicate validation (validation.py owns type/range checks)
  - NO future intent logic (extraction.py owns that)

Exports:
  run_correction(normalized_text, fields) → (fields, correction_log, source)
"""

import json
import re
import requests

from field_registry import (
    FieldValue, LLM_CORRECTION_ALLOWED,
    LLM_CORRECTION_BLOCKED, make_field, assign_confidence,
)

OLLAMA_BASE_URL  = "http://localhost:11434"
CORRECTION_MODEL = "llama3.1:8b"

_CORRECTABLE_CONFIDENCES = {"low"}


def _build_correction_payload(
    fields: dict[str, FieldValue],
) -> tuple[dict, list[str]]:
    correctable: dict = {}
    skipped: list[str] = []
    for fname, fv in fields.items():
        if fname in LLM_CORRECTION_BLOCKED:
            skipped.append(f"{fname}: blocked (rule-owned)")
            continue
        if fname not in LLM_CORRECTION_ALLOWED:
            skipped.append(f"{fname}: not in correction scope")
            continue
        if fv.value is None or fv.confidence in _CORRECTABLE_CONFIDENCES:
            correctable[fname] = {
                "value": fv.value, "confidence": fv.confidence, "source": fv.source,
            }
        else:
            skipped.append(f"{fname}: skipped (conf={fv.confidence})")
    return correctable, skipped


CORRECTION_PROMPT = """You are a financial data correction specialist.
You receive an investor description and a JSON of LOW-CONFIDENCE or NULL fields.
Your job: fill nulls and correct low-confidence values.

STRICT RULES:
- ONLY modify fields provided in the input JSON
- DO NOT invent fields not in the input
- Return null if you cannot confidently determine a value

FINANCIAL KNOWLEDGE SCORE RUBRIC:
1=No knowledge  2=Basic awareness  3=Moderate  4=Strong  5=Expert
Return null if not confident. Do NOT default to 3.

LOSS REACTION — MULTI-LEVEL:
- "panic": ONLY for "stopped investing", "exit", "panic sell", "can't sleep", "sleepless"
- "cautious": "worried", "stressed", "reduced exposure", "paused SIP"
  DO NOT map "worried" → "panic"
- "neutral": "stay calm", "hold steady"
- "aggressive": "buy the dip", "buy more when markets fall"

NEAR-TERM OBLIGATION:
- "high": within ~6 months — wedding, medical, "very soon", "next few months"
- "moderate": 6–24 months — education, home loan, "next year"
- "none": nothing mentioned. If unclear → "none"

MANDATORY CORRECTIONS:
1. loss_reaction == "panic" → risk_behavior MUST be "low"
2. experience_years < 1 → financial_knowledge_score MUST be ≤ 2 (null if unsure)
3. "calculated risks" → risk_behavior = "medium" (NOT "high")
4. peer-driven / tips → decision_autonomy = false
5. financial_knowledge_score == 3 with no clear investing evidence → null

OUTPUT FORMAT:
{"corrected": {"field_name": value_or_null, ...}, "corrections": ["description", ...]}
Return ONLY valid JSON. No markdown."""


def _parse_json(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    return json.loads(text)


def _apply_correction_results(
    fields: dict[str, FieldValue],
    corrected: dict,
    corrections_list: list[str],
    source_label: str,
) -> tuple[dict[str, FieldValue], list[dict]]:
    updated = dict(fields)
    log: list[dict] = []
    for fname, new_val in corrected.items():
        if fname in LLM_CORRECTION_BLOCKED:
            log.append({"field": fname, "action": "blocked",
                        "reason": "rule-owned — rejected"})
            continue
        old_fv  = fields.get(fname)
        old_val = old_fv.value if old_fv else None
        if new_val == old_val:
            continue
        if old_fv and old_fv.confidence not in _CORRECTABLE_CONFIDENCES and old_fv.value is not None:
            log.append({"field": fname, "action": "skipped",
                        "reason": f"conf={old_fv.confidence} — not correctable",
                        "old_value": old_val, "new_value": new_val})
            continue
        new_conf = assign_confidence(fname, new_val, source_label)
        updated[fname] = FieldValue(value=new_val, confidence=new_conf, source=source_label)
        log.append({
            "field": fname, "action": "corrected",
            "old_value": old_val, "new_value": new_val,
            "old_conf": old_fv.confidence if old_fv else "none",
            "new_conf": new_conf, "source": source_label,
            "reason": next((c for c in corrections_list if fname in c), "correction applied"),
        })
    return updated, log


def _rule_fallback_correction(
    text: str,
    fields: dict[str, FieldValue],
) -> tuple[dict[str, FieldValue], list[dict]]:
    updated = dict(fields)
    log: list[dict] = []
    t = text.lower()

    def _set(fname: str, val, reason: str):
        old = updated.get(fname)
        if old and old.value is not None and old.confidence not in _CORRECTABLE_CONFIDENCES:
            return
        new_conf = assign_confidence(fname, val, "rule")
        updated[fname] = FieldValue(value=val, confidence=new_conf, source="rule")
        log.append({"field": fname, "action": "rule_fallback",
                    "old_value": old.value if old else None,
                    "new_value": val, "reason": reason})

    panic_signals = [
        "stopped investing", "exit", "panic sell", "wanted to withdraw completely",
        "can't sleep", "cannot sleep", "sleepless", "considered stopping",
        "stop investing", "pull out everything", "withdraw all",
    ]
    if any(s in t for s in panic_signals):
        _set("loss_reaction", "panic", "panic signal")
        _set("risk_behavior", "low",   "panic → low risk")
    elif any(s in t for s in [
        "worried", "stressed", "stress", "anxious", "anxiety", "nervous",
        "reduced exposure", "paused sip", "shifted to safer", "scared", "fear",
    ]):
        _set("loss_reaction", "cautious", "cautious signal")
    elif any(s in t for s in ["buy the dip", "buy more when", "buy on dips"]):
        _set("loss_reaction", "aggressive", "buy-the-dip signal")

    if any(s in t for s in ["tips from", "friend told", "friends invest",
                              "following others", "peer advice", "someone suggested"]):
        _set("decision_autonomy", False, "peer-driven signal")

    if any(s in t for s in ["wedding", "marriage", "shaadi", "getting married"]):
        _set("near_term_obligation_level", "high",    "wedding signal")
        _set("obligation_type",            "wedding", "wedding signal")

    if any(s in t for s in ["secretly saving", "earmarked", "family expectation",
                              "saving for", "set aside for", "keeping aside"]):
        _set("near_term_obligation_level", "high",   "hidden obligation")
        ot = updated.get("obligation_type")
        if not ot or ot.value is None:
            _set("obligation_type", "family", "hidden obligation")

    if any(s in t for s in ["house purchase", "buying a house", "buying property",
                              "home loan", "property purchase", "flat purchase"]):
        urgent = any(u in t for u in ["this year", "very soon", "next few months", "shortly"])
        _set("near_term_obligation_level", "high" if urgent else "moderate", "house signal")
        _set("obligation_type", "house", "house signal")

    if any(s in t for s in ["freelancer", "freelance", "consultant", "contractor",
                              "project-based", "variable income"]):
        it = updated.get("income_type")
        if not it or it.value in (None, "unknown"):
            _set("income_type", "gig", "gig signal")

    if any(s in t for s in ["business owner", "runs a business", "proprietor",
                              "self-employed", "own business"]):
        it = updated.get("income_type")
        if not it or it.value in (None, "unknown"):
            _set("income_type", "business", "business signal")

    m = re.search(
        r"(\d+\.?\d*)\s*months?\s+(?:of\s+)?(?:experience|investing|in\s+(?:stock|market|mutual))",
        t,
    )
    if m:
        months = float(m.group(1))
        years  = round(months / 12, 2)
        exp_fv = updated.get("experience_years")
        if exp_fv is None or exp_fv.value is None or (
            months > 1 and abs((exp_fv.value or 0) - months) < 0.5
        ):
            _set("experience_years", years, f"{months}mo → {years}y")

    return updated, log


def _apply_mandatory_invariants(
    fields: dict[str, FieldValue],
) -> tuple[dict[str, FieldValue], list[dict]]:
    updated = dict(fields)
    log: list[dict] = []

    def _force(fname: str, val, reason: str):
        old = updated.get(fname)
        updated[fname] = FieldValue(
            value=val,
            confidence=assign_confidence(fname, val, "rule"),
            source="rule",
        )
        log.append({"field": fname, "action": "mandatory_invariant",
                    "old_value": old.value if old else None,
                    "new_value": val, "reason": reason})

    lr = updated.get("loss_reaction")
    rb = updated.get("risk_behavior")
    if lr and lr.value == "panic" and rb and rb.value in ("medium", "high"):
        _force("risk_behavior", "low", f"panic → risk_behavior from '{rb.value}' to 'low'")

    exp_fv = updated.get("experience_years")
    fks_fv = updated.get("financial_knowledge_score")
    if (exp_fv and exp_fv.value is not None and exp_fv.value < 1
            and fks_fv and fks_fv.value is not None and fks_fv.value > 2):
        _force("financial_knowledge_score", 2,
               f"experience={exp_fv.value} < 1 → knowledge capped to 2")

    ntol_fv = updated.get("near_term_obligation_level")
    if ntol_fv and ntol_fv.value not in ("none", "moderate", "high", None):
        _force("near_term_obligation_level", "none",
               f"invalid enum '{ntol_fv.value}' → 'none'")

    ntol_fv = updated.get("near_term_obligation_level")
    ot_fv   = updated.get("obligation_type")
    if ntol_fv and ntol_fv.value in ("none", None) and ot_fv and ot_fv.value is not None:
        _force("obligation_type", None, "obligation_type cleared — level is none")

    return updated, log


def run_correction(
    normalized_text: str,
    fields: dict[str, FieldValue],
) -> tuple[dict[str, FieldValue], list[dict], str]:
    """
    Confidence-aware correction.
    Returns (updated_fields, correction_log, source).
    source: "llm_correction" | "rule_fallback"
    """
    correctable, skipped_log = _build_correction_payload(fields)
    correction_log: list[dict] = [
        {"field": s.split(":")[0].strip(), "action": "skipped", "reason": s}
        for s in skipped_log
    ]

    source = "llm_correction"

    if correctable:
        prompt = (
            f"{CORRECTION_PROMPT}\n\n"
            f"Investor description:\n{normalized_text}\n\n"
            f"Fields to correct:\n{json.dumps(correctable, indent=2)}\n\n"
            f"Return corrected values for ONLY these fields."
        )
        payload = {
            "model":   CORRECTION_MODEL,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": 0, "num_predict": 900},
            "format":  "json",
        }

        last_error = None
        llm_success = False

        for attempt in (1, 2):
            try:
                resp = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=180,
                )
                resp.raise_for_status()
                parsed      = _parse_json(resp.json().get("response", ""))
                corrected   = parsed.get("corrected", {})
                corrections = parsed.get("corrections", [])
                if not isinstance(corrections, list):
                    corrections = [str(corrections)]
                fields, apply_log = _apply_correction_results(
                    fields, corrected, corrections, "llm_correction"
                )
                correction_log.extend(apply_log)
                llm_success = True
                break
            except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
                last_error = e

        if not llm_success:
            source = "rule_fallback"
            correction_log.append({
                "field": "_system", "action": "fallback_activated",
                "reason": f"LLM correction failed: {last_error}",
            })
            fields, fallback_log = _rule_fallback_correction(normalized_text, fields)
            correction_log.extend(fallback_log)

    fields, invariant_log = _apply_mandatory_invariants(fields)
    correction_log.extend(invariant_log)

    return fields, correction_log, source
