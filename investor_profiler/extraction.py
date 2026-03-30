"""
Extraction Layer — v5
Architecture: RULE EXTRACTION → LLM EXTRACTION → CONTROLLED MERGE

Responsibilities:
  - Rule layer: deterministic numeric extraction (income, EMI, savings)
  - LLM layer: soft-signal extraction (behavior, knowledge, obligation)
  - Merge: field-ownership-aware, logged, no silent overwrites
  - Future intent: intent-phrase-gated event detection
  - NO validation logic (moved to validation.py post-correction)
  - NO duplicate behavioral overrides (correction.py owns that)
"""

import json
import re
import requests

from field_registry import (
    FieldValue, FIELD_OWNERSHIP, LLM_EXTRACTION_FIELDS,
    make_field, check_invariants,
)

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "qwen2.5:3b-instruct"


# ---------------------------------------------------------------------------
# Text normalization (pre-processing only — no field extraction here)
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Currency shorthand + experience months → canonical form."""
    # Crore / lakh / k
    text = re.sub(
        r"(\d+\.?\d*)\s*(?:cr|crore|crores)\b",
        lambda m: str(int(float(m.group(1)) * 10_000_000)),
        text, flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(\d+\.?\d*)\s*(?:L|lakh|lakhs)\b",
        lambda m: str(int(float(m.group(1)) * 100_000)),
        text, flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(\d+\.?\d*)\s*[kK]\b",
        lambda m: str(int(float(m.group(1)) * 1_000)),
        text,
    )
    # Months of experience → fractional years
    def _months_to_years(m):
        return f"{round(float(m.group(1)) / 12, 2)} years"
    text = re.sub(
        r"(\d+\.?\d*)\s*months?\s+(?:of\s+)?(?:experience|investing|in\s+(?:stock|market|mutual))",
        _months_to_years, text, flags=re.IGNORECASE,
    )
    return text


def _detect_non_english(text: str) -> bool:
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    return ascii_ratio < 0.4


# ---------------------------------------------------------------------------
# RULE EXTRACTION — deterministic, numeric fields only
# Ownership: monthly_income, emi_amount, emergency_months, income_type (partial)
# ---------------------------------------------------------------------------

def _rule_extract_income(text: str) -> tuple[FieldValue | None, FieldValue | None, str]:
    """
    Returns (monthly_income_fv, income_type_fv, log_msg).
    income_type from rule is always None — LLM owns it unless rule has strong signal.
    Exception: LPA pattern strongly implies salaried; gig/business keywords override.
    """
    t = text.lower()

    # Range: "X–Y lakh per year"
    m = re.search(
        r"(\d+\.?\d*)\s*(?:to|[-–])\s*(\d+\.?\d*)\s*(?:lakh|l)"
        r"\s*(?:per\s*(?:year|annum|pa)|p\.?a\.?|annually)?",
        t,
    )
    if m:
        avg     = (float(m.group(1)) + float(m.group(2))) / 2
        monthly = round((avg * 100_000) / 12)
        log     = f"Rule income: '{m.group(0)}' → monthly_income={monthly} (avg {avg}L/12)"
        return make_field("monthly_income", monthly, "rule"), None, log

    # LPA / lakh per annum
    m = re.search(
        r"(?:earning\s+(?:about|around|approximately)?\s*)?(\d+\.?\d*)\s*"
        r"(?:lpa|lakh\s*(?:per\s*annum|p\.?a\.?|annually|per\s*year))",
        t,
    )
    if m:
        lpa     = float(m.group(1))
        monthly = round((lpa * 100_000) / 12)
        log     = f"Rule income: '{m.group(0)}' → monthly_income={monthly} ({lpa} LPA/12)"
        return make_field("monthly_income", monthly, "rule"), None, log

    return None, None, ""


def _rule_extract_emi(text: str) -> tuple[FieldValue | None, str]:
    t = text.lower()
    m = re.search(
        r"(?:loan\s+)?emi\s+(?:of\s+)?(?:rs\.?|₹|inr)?\s*(\d[\d,]*(?:\.\d+)?)", t
    )
    if m:
        amount = float(m.group(1).replace(",", ""))
        log    = f"Rule EMI: '{m.group(0)}' → emi_amount={amount}"
        return make_field("emi_amount", amount, "rule"), log
    return None, ""


def _rule_extract_savings(text: str) -> tuple[float | None, str]:
    """Returns (total_savings, log). Internal only — used for emergency_months inference."""
    t = text.lower()
    patterns = [
        r"(?:rs\.?|₹|inr)\s*(\d[\d,]*(?:\.\d+)?)\s*(?:in\s+)?"
        r"(?:savings?(?:\s+account)?|rd|fd|fixed\s+deposit|recurring\s+deposit|gold|ppf|nsc)",
        r"(?:keeps?|has|maintains?|saved?)\s+(?:rs\.?|₹|inr)\s*(\d[\d,]*(?:\.\d+)?)"
        r"\s*(?:in\s+)?(?:savings?|rd|fd|gold|ppf)",
        r"(\d[\d,]*(?:\.\d+)?)\s*(?:rs\.?|₹|inr)?\s*(?:in\s+)?"
        r"(?:savings?\s+account|rd|fd|fixed\s+deposit)",
    ]
    # Deduplicate matches by position to avoid double-counting
    seen_positions: set[int] = set()
    total = 0.0
    found = []
    for pat in patterns:
        for m in re.finditer(pat, t):
            if m.start() not in seen_positions:
                seen_positions.add(m.start())
                val = float(m.group(1).replace(",", ""))
                total += val
                found.append(f"{m.group(0).strip()}={val}")

    if found:
        return total, f"Rule savings: {found} → total={total}"
    return None, ""


def run_rule_extraction(text: str) -> tuple[dict[str, FieldValue], list[str]]:
    """
    Deterministic extraction of rule-owned fields.
    Returns {field_name: FieldValue} and log entries.
    Only sets fields this layer owns.
    """
    result: dict[str, FieldValue] = {}
    log: list[str] = []

    income_fv, _, income_log = _rule_extract_income(text)
    if income_fv:
        result["monthly_income"] = income_fv
        log.append(income_log)

    emi_fv, emi_log = _rule_extract_emi(text)
    if emi_fv:
        result["emi_amount"] = emi_fv
        log.append(emi_log)

    savings, savings_log = _rule_extract_savings(text)
    if savings_log:
        log.append(savings_log)

    # Emergency months: inferred from savings + income (both must be present)
    monthly = result.get("monthly_income")
    if savings is not None and monthly is not None and monthly.value and monthly.value > 0:
        em = round(savings / monthly.value, 1)
        result["emergency_months"] = make_field("emergency_months", em, "rule")
        log.append(
            f"Rule emergency_months={em} = savings({savings}) / income({monthly.value})"
        )

    return result, log


# ---------------------------------------------------------------------------
# FUTURE INTENT DETECTION — intent-phrase-gated
# ---------------------------------------------------------------------------

_INTENT_PHRASES = [
    "plans to", "plan to", "planning to",
    "wants to", "want to", "wanting to",
    "considering", "aims to", "aim to",
    "thinking of", "thinking about",
    "intends to", "intend to",
    "looking to", "hoping to",
    "will buy", "will purchase",
    "going to buy", "going to purchase",
]

_FUTURE_EVENT_PATTERNS = [
    (r"(?:buy|purchase|own)\s+(?:a\s+)?(?:house|flat|property|home|apartment)", "housing"),
    (r"(?:home\s+loan|housing\s+loan|mortgage)",                                 "housing"),
    (r"(?:buy|purchase|own)\s+(?:a\s+)?(?:car|vehicle|bike|two.?wheeler)",       "vehicle"),
    (r"(?:start|launch|open|set\s+up)\s+(?:a\s+)?(?:business|startup|venture|shop|firm)", "business"),
    (r"(?:child(?:ren)?'?s?\s+)?(?:education|college|school\s+fees|tuition)",    "education"),
    (r"(?:wedding|marriage|shaadi|get\s+married)",                                "wedding"),
    (r"(?:travel|vacation|trip\s+abroad|foreign\s+trip)",                        "travel"),
    (r"(?:retire|retirement)",                                                    "retirement"),
]

_TIMELINE_HINTS: dict[str, list[str]] = {
    "near": ["next few months", "this year", "very soon", "shortly",
             "within a year", "in 6 months", "in a few months", "next year"],
    "mid":  ["in 2 years", "in 3 years", "in 4 years", "in 5 years",
             "2-3 years", "3-5 years", "couple of years"],
    "long": ["in 10 years", "long term", "eventually", "someday",
             "retirement", "after 5 years", "in 7 years"],
}

_SCALE_MAP: dict[str, dict[str, str]] = {
    # type → {timeline → scale}
    "housing":    {"near": "high", "mid": "high", "long": "medium"},
    "business":   {"near": "high", "mid": "high", "long": "medium"},
    "vehicle":    {"near": "medium", "mid": "low",  "long": "low"},
    "education":  {"near": "medium", "mid": "medium", "long": "low"},
    "wedding":    {"near": "high", "mid": "medium", "long": "low"},
    "travel":     {"near": "low",  "mid": "low",  "long": "low"},
    "retirement": {"near": "low",  "mid": "low",  "long": "low"},
}


def _infer_timeline(context: str) -> str:
    t = context.lower()
    for timeline, hints in _TIMELINE_HINTS.items():
        if any(h in t for h in hints):
            return timeline
    return "mid"


def detect_future_events(text: str) -> tuple[list[dict], float, list[str]]:
    """
    INTENT-GATED: only fires if an intent phrase is present in the text.
    Returns (events, future_obligation_score, reasons).
    """
    t = text.lower()
    reasons: list[str] = []

    # Gate: require at least one intent phrase
    intent_phrase_found = next((p for p in _INTENT_PHRASES if p in t), None)
    if not intent_phrase_found:
        reasons.append("Future intent: no intent phrase detected — skipping event detection")
        return [], 0.0, reasons

    reasons.append(f"Future intent: intent phrase '{intent_phrase_found}' detected")
    events: list[dict] = []

    for pattern, etype in _FUTURE_EVENT_PATTERNS:
        m = re.search(pattern, t)
        if m:
            start    = max(0, m.start() - 100)
            end      = min(len(t), m.end() + 100)
            timeline = _infer_timeline(t[start:end])
            scale    = _SCALE_MAP.get(etype, {}).get(timeline, "low")
            events.append({
                "type": etype, "timeline": timeline,
                "scale": scale, "matched": m.group(0),
            })
            reasons.append(
                f"Future event: '{m.group(0)}' → type={etype}, "
                f"timeline={timeline}, scale={scale}"
            )

    # Score: additive, capped at 20
    score = 0.0
    high_count = sum(1 for e in events if e["scale"] == "high")
    mid_count  = sum(1 for e in events if e["scale"] == "medium")
    low_count  = sum(1 for e in events if e["scale"] == "low")

    if high_count >= 2:
        score = 20.0
        reasons.append("future_obligation_score=20 (multiple high-scale events)")
    elif high_count == 1:
        score = 10.0
        reasons.append("future_obligation_score=10 (single major event)")
    elif mid_count >= 1:
        score = 10.0
        reasons.append("future_obligation_score=10 (moderate future event)")
    elif low_count >= 1:
        score = 5.0
        reasons.append("future_obligation_score=5 (distant/low-scale goal)")

    return events, score, reasons


# ---------------------------------------------------------------------------
# LLM EXTRACTION — soft-signal fields only
# ---------------------------------------------------------------------------

LLM_SYSTEM_PROMPT = """You are a financial data extraction assistant.
Extract ONLY the fields listed in the schema from the investor description.

FINANCIAL KNOWLEDGE SCORE RUBRIC (strict — do NOT default to 3):
1 = No knowledge — never invested, doesn't understand basic terms
2 = Basic awareness — knows FD/savings, heard of mutual funds, no real experience
3 = Moderate understanding — has invested in MF/stocks, understands risk/return basics
4 = Strong knowledge — actively manages portfolio, understands asset allocation
5 = Expert level — professional knowledge, understands derivatives, complex instruments
Return null if not confident.

INCOME TYPE RULES:
- "salaried" = fixed monthly salary from employer
- "business" = owns a business, self-employed with stable revenue
- "gig" = freelancer, consultant, contractor, variable income
- "unknown" if unclear

EXPERIENCE CONVERSION (CRITICAL):
- "X months" of experience → experience_years = X / 12
- NEVER convert "8 months" to 8 years

LOSS REACTION — MULTI-LEVEL:
- "panic": ONLY for "stopped investing", "exit", "panic sell", "wanted to withdraw completely",
  "can't sleep", "sleepless", "considered stopping"
- "cautious": "worried", "stressed", "reduced exposure", "paused SIP", "shifted to safer"
  DO NOT map "worried" → "panic"
- "neutral": "stay calm", "hold steady", "not worried"
- "aggressive": "buy the dip", "buy more when markets fall"

NEAR-TERM OBLIGATION:
- "high": within ~6 months — wedding, medical emergency, "next few months", "very soon"
- "moderate": 6–24 months — education fees, home loan planned, "next year", "upcoming"
- "none": no major upcoming expense. If unclear → "none"

BEHAVIORAL RULES:
- panic signals ALWAYS override aggressive wording in same text
- "calculated risks" → risk_behavior = "medium" (NOT "high")
- peer-driven / tips → decision_autonomy = false
- researches independently → decision_autonomy = true

OUTPUT: Return ONLY valid JSON. Missing or unclear → null. No markdown."""

LLM_SCHEMA = """{
  "income_type": "salaried | business | gig | unknown | null",
  "monthly_income": "number in rupees | null",
  "emergency_months": "number | null",
  "emi_amount": "number in rupees | null",
  "emi_ratio": "number 0-100 | null (ONLY if explicitly stated as %, never derive)",
  "dependents": "integer | null",
  "near_term_obligation_level": "none | moderate | high | null",
  "obligation_type": "wedding | house | education | medical | family | other | null",
  "experience_years": "number | null",
  "financial_knowledge_score": "integer 1-5 | null (prefer null over guessing)",
  "decision_autonomy": "true | false | null",
  "loss_reaction": "panic | cautious | neutral | aggressive | null",
  "risk_behavior": "low | medium | high | null"
}"""


def _parse_json(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    return json.loads(text)


def run_llm_extraction(text: str) -> tuple[dict[str, FieldValue], str | None]:
    """
    LLM extraction of soft-signal fields.
    Returns ({field: FieldValue}, warning_or_None).
    All LLM values get source="llm" with confidence per field_registry rules.
    """
    prompt = (
        f"{LLM_SYSTEM_PROMPT}\n\nSchema:\n{LLM_SCHEMA}\n\n"
        f"Investor description:\n{text}\n\nReturn only the JSON object."
    )
    payload = {
        "model":   OLLAMA_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0, "num_predict": 700},
        "format":  "json",
    }

    warning  = None
    raw_dict = {}

    for attempt in (1, 2):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=90,
            )
            resp.raise_for_status()
            raw_dict = _parse_json(resp.json().get("response", ""))
            break
        except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
            if attempt == 2:
                warning  = f"LLM extraction failed after 2 attempts: {e}"
                raw_dict = {}

    result: dict[str, FieldValue] = {}
    for fname in LLM_EXTRACTION_FIELDS:
        val = raw_dict.get(fname)
        result[fname] = make_field(fname, val, "llm")

    # Knowledge drift fix: score=3 with low confidence → null
    fks_fv = result.get("financial_knowledge_score")
    if fks_fv and fks_fv.value == 3 and fks_fv.confidence == "low":
        result["financial_knowledge_score"] = FieldValue(
            value=None, confidence="low", source="llm"
        )

    return result, warning


# ---------------------------------------------------------------------------
# CONTROLLED MERGE ENGINE
# ---------------------------------------------------------------------------

def merge_fields(
    rule_fields: dict[str, FieldValue],
    llm_fields:  dict[str, FieldValue],
) -> tuple[dict[str, FieldValue], list[dict]]:
    """
    Merge rule and LLM extractions using field ownership contract.
    Returns (merged_fields, merge_log).

    Rules:
      "rule"    → rule_val wins; LLM value logged but discarded
      "llm"     → llm_val wins; rule value logged but discarded
      "mixed"   → higher confidence wins; tie → rule wins
      "derived" → skip (computed later)

    ALL decisions are logged. No silent overwrites.
    """
    all_fields = set(rule_fields) | set(llm_fields)
    merged: dict[str, FieldValue] = {}
    merge_log: list[dict] = []

    _conf_rank = {"low": 0, "medium": 1, "high": 2}

    for fname in all_fields:
        ownership = FIELD_OWNERSHIP.get(fname, "llm")
        rule_fv   = rule_fields.get(fname)
        llm_fv    = llm_fields.get(fname)

        if ownership == "rule":
            chosen = rule_fv or llm_fv
            reason = "rule-owned: rule value authoritative"
            if rule_fv and llm_fv and rule_fv.value != llm_fv.value:
                reason = f"rule-owned: rule={rule_fv.value} overrides llm={llm_fv.value}"

        elif ownership == "llm":
            chosen = llm_fv or rule_fv
            reason = "llm-owned: llm value authoritative"
            if rule_fv and llm_fv and rule_fv.value != llm_fv.value:
                reason = f"llm-owned: llm={llm_fv.value} overrides rule={rule_fv.value}"

        elif ownership == "mixed":
            if rule_fv is None:
                chosen = llm_fv
                reason = "mixed: only llm value present"
            elif llm_fv is None:
                chosen = rule_fv
                reason = "mixed: only rule value present"
            else:
                rule_rank = _conf_rank.get(rule_fv.confidence, 0)
                llm_rank  = _conf_rank.get(llm_fv.confidence, 0)
                if rule_rank >= llm_rank:
                    chosen = rule_fv
                    reason = (
                        f"mixed: rule conf={rule_fv.confidence} ≥ llm conf={llm_fv.confidence}"
                        f" → rule={rule_fv.value} wins"
                    )
                else:
                    chosen = llm_fv
                    reason = (
                        f"mixed: llm conf={llm_fv.confidence} > rule conf={rule_fv.confidence}"
                        f" → llm={llm_fv.value} wins"
                    )
        else:
            # derived — skip
            continue

        if chosen is None:
            chosen = FieldValue.null("default")

        merged[fname] = chosen
        merge_log.append({
            "field":      fname,
            "ownership":  ownership,
            "rule_value": rule_fv.value if rule_fv else None,
            "llm_value":  llm_fv.value  if llm_fv  else None,
            "chosen":     chosen.value,
            "confidence": chosen.confidence,
            "source":     chosen.source,
            "reason":     reason,
        })

    return merged, merge_log


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def extract_investor_data(paragraph: str) -> dict:
    """
    Pipeline:
      [0] Text normalization
      [1] Rule extraction (deterministic)
      [2] LLM extraction (soft signals)
      [3] Controlled merge (ownership-aware, logged)
      [4] Future intent detection (intent-phrase-gated)
      [5] Invariant check

    Returns structured result — NO validation, NO correction (those are downstream).
    """
    if _detect_non_english(paragraph):
        return {
            "fields":              {},
            "merge_log":           [],
            "future_events":       [],
            "future_obligation_score": 0.0,
            "future_intent_reasons": ["Non-English input — skipped"],
            "extraction_warning":  "Non-English input detected; extraction skipped.",
            "non_english":         True,
        }

    text = normalize_text(paragraph)

    # [1] Rule extraction
    rule_fields, rule_log = run_rule_extraction(text)

    # [2] LLM extraction
    llm_fields, llm_warning = run_llm_extraction(text)

    # [3] Controlled merge
    merged, merge_log = merge_fields(rule_fields, llm_fields)

    # [4] Future intent (intent-gated)
    future_events, future_obligation_score, future_reasons = detect_future_events(text)

    # [5] Invariant check
    violations = check_invariants(merged)

    # Auto-fix invariant violations (null+high → low)
    for fname, fv in merged.items():
        if fv.value is None and fv.confidence == "high":
            merged[fname] = FieldValue(value=None, confidence="low", source=fv.source)

    return {
        "fields":                merged,          # {field: FieldValue}
        "normalized_text":       text,
        "merge_log":             merge_log,
        "rule_log":              rule_log,
        "future_events":         future_events,
        "future_obligation_score": future_obligation_score,
        "future_intent_reasons": future_reasons,
        "extraction_warning":    llm_warning,
        "invariant_violations":  violations,
        "non_english":           False,
    }


# ---------------------------------------------------------------------------
# Helpers for downstream stages
# ---------------------------------------------------------------------------

def unwrap_values(fields: dict[str, FieldValue]) -> dict:
    return {k: v.value for k, v in fields.items()}


def unwrap_confidences(fields: dict[str, FieldValue]) -> dict:
    return {k: v.confidence for k, v in fields.items()}


def unwrap_sources(fields: dict[str, FieldValue]) -> dict:
    return {k: v.source for k, v in fields.items()}


def fields_to_dict(fields: dict[str, FieldValue]) -> dict:
    """Serialize all FieldValues to plain dicts for JSON output."""
    return {k: v.to_dict() for k, v in fields.items()}
