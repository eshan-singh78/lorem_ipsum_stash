"""
Extraction Layer — InvestorDNA v7
Architecture: RULE EXTRACTION → SINGLE LLM (llama3.1:8b) → RULE-WINS MERGE

Design principles:
  - Rules own all numeric fields: monthly_income, emi_amount, emergency_months
  - One LLM call per request — no correction pipeline, no second model
  - Rule context is passed INTO the LLM prompt so it never re-derives what rules found
  - Merge is simple: rule_fields always win over llm_fields for rule-owned keys
  - Mandatory invariants enforced in code after merge, not in prompts
  - Future intent detection is deterministic (regex, intent-gated)
"""

import json
import re
import requests

from field_registry import (
    FieldValue, RULE_OWNED_FIELDS, LLM_OWNED_FIELDS,
    DERIVED_FIELDS, make_field, check_invariants,
)

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL       = "llama3.1:8b"


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Currency shorthand + experience months → canonical form."""
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
    # Months of experience → fractional years (rule-level, before LLM sees text)
    text = re.sub(
        r"(\d+\.?\d*)\s*months?\s+(?:of\s+)?(?:experience|investing|in\s+(?:stock|market|mutual))",
        lambda m: f"{round(float(m.group(1)) / 12, 2)} years",
        text, flags=re.IGNORECASE,
    )
    return text


def _detect_non_english(text: str) -> bool:
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    return ascii_ratio < 0.4


# ---------------------------------------------------------------------------
# RULE EXTRACTION — deterministic, numeric fields only
# ---------------------------------------------------------------------------

def _rule_extract_income(text: str) -> tuple[FieldValue | None, str]:
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
        return make_field("monthly_income", monthly, "rule"), \
               f"Rule income: '{m.group(0)}' → {monthly} (avg {avg}L/12)"

    # LPA / lakh per annum
    m = re.search(
        r"(?:earning\s+(?:about|around|approximately)?\s*)?(\d+\.?\d*)\s*"
        r"(?:lpa|lakh\s*(?:per\s*annum|p\.?a\.?|annually|per\s*year))",
        t,
    )
    if m:
        lpa     = float(m.group(1))
        monthly = round((lpa * 100_000) / 12)
        return make_field("monthly_income", monthly, "rule"), \
               f"Rule income: '{m.group(0)}' → {monthly} ({lpa} LPA/12)"

    return None, ""


def _rule_extract_emi(text: str) -> tuple[FieldValue | None, str]:
    t = text.lower()
    m = re.search(
        r"(?:loan\s+)?emi\s+(?:of\s+)?(?:rs\.?|₹|inr)?\s*(\d[\d,]*(?:\.\d+)?)", t
    )
    if m:
        amount = float(m.group(1).replace(",", ""))
        return make_field("emi_amount", amount, "rule"), \
               f"Rule EMI: '{m.group(0)}' → {amount}"
    return None, ""


def _rule_extract_savings(text: str) -> tuple[float | None, str]:
    t = text.lower()
    patterns = [
        r"(?:rs\.?|₹|inr)\s*(\d[\d,]*(?:\.\d+)?)\s*(?:in\s+)?"
        r"(?:savings?(?:\s+account)?|rd|fd|fixed\s+deposit|recurring\s+deposit|gold|ppf|nsc)",
        r"(?:keeps?|has|maintains?|saved?)\s+(?:rs\.?|₹|inr)\s*(\d[\d,]*(?:\.\d+)?)"
        r"\s*(?:in\s+)?(?:savings?|rd|fd|gold|ppf)",
        r"(\d[\d,]*(?:\.\d+)?)\s*(?:rs\.?|₹|inr)?\s*(?:in\s+)?"
        r"(?:savings?\s+account|rd|fd|fixed\s+deposit)",
    ]
    seen: set[int] = set()
    total = 0.0
    found = []
    for pat in patterns:
        for m in re.finditer(pat, t):
            if m.start() not in seen:
                seen.add(m.start())
                val = float(m.group(1).replace(",", ""))
                total += val
                found.append(f"{m.group(0).strip()}={val}")
    if found:
        return total, f"Rule savings: {found} → total={total}"
    return None, ""


def _rule_extract_income_type(text: str) -> tuple[FieldValue | None, str]:
    """Rule-level income type detection from strong keyword signals."""
    t = text.lower()
    if any(s in t for s in ["freelancer", "freelance", "consultant", "contractor",
                              "project-based", "variable income", "gig worker",
                              "zomato", "swiggy", "uber", "ola driver"]):
        return make_field("income_type", "gig", "rule"), "Rule income_type: gig signal"
    if any(s in t for s in ["business owner", "runs a business", "proprietor",
                              "self-employed", "own business", "textile business",
                              "shop owner", "entrepreneur"]):
        return make_field("income_type", "business", "rule"), "Rule income_type: business signal"
    # LPA pattern strongly implies salaried
    if re.search(r"\d+\.?\d*\s*(?:lpa|lakh\s*per\s*(?:annum|year))", t):
        return make_field("income_type", "salaried", "rule"), "Rule income_type: LPA → salaried"
    return None, ""


def run_rule_extraction(text: str) -> tuple[dict[str, FieldValue], list[str]]:
    """
    Deterministic extraction of all rule-owned fields.
    Returns {field_name: FieldValue} and log entries.
    """
    result: dict[str, FieldValue] = {}
    log: list[str] = []

    income_fv, income_log = _rule_extract_income(text)
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

    # Emergency months: derived from savings ÷ income (both must be present)
    monthly = result.get("monthly_income")
    if savings is not None and monthly is not None and monthly.value and monthly.value > 0:
        em = round(savings / monthly.value, 1)
        result["emergency_months"] = make_field("emergency_months", em, "rule")
        log.append(f"Rule emergency_months={em} = savings({savings}) / income({monthly.value})")

    # Income type from strong keyword signals
    it_fv, it_log = _rule_extract_income_type(text)
    if it_fv:
        result["income_type"] = it_fv
        log.append(it_log)

    return result, log


# ---------------------------------------------------------------------------
# FUTURE INTENT DETECTION — deterministic, intent-phrase-gated
# ---------------------------------------------------------------------------

_INTENT_PHRASES = [
    "plans to", "plan to", "planning to", "wants to", "want to", "wanting to",
    "considering", "aims to", "aim to", "thinking of", "thinking about",
    "intends to", "intend to", "looking to", "hoping to",
    "will buy", "will purchase", "going to buy", "going to purchase",
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
    "housing":    {"near": "high", "mid": "high",   "long": "medium"},
    "business":   {"near": "high", "mid": "high",   "long": "medium"},
    "vehicle":    {"near": "medium", "mid": "low",  "long": "low"},
    "education":  {"near": "medium", "mid": "medium", "long": "low"},
    "wedding":    {"near": "high", "mid": "medium", "long": "low"},
    "travel":     {"near": "low",  "mid": "low",    "long": "low"},
    "retirement": {"near": "low",  "mid": "low",    "long": "low"},
}


def detect_future_events(text: str) -> tuple[list[dict], float, list[str]]:
    t = text.lower()
    reasons: list[str] = []

    intent_phrase = next((p for p in _INTENT_PHRASES if p in t), None)
    if not intent_phrase:
        reasons.append("Future intent: no intent phrase — skipping")
        return [], 0.0, reasons

    reasons.append(f"Future intent: '{intent_phrase}' detected")
    events: list[dict] = []

    for pattern, etype in _FUTURE_EVENT_PATTERNS:
        m = re.search(pattern, t)
        if m:
            ctx_start = max(0, m.start() - 100)
            ctx_end   = min(len(t), m.end() + 100)
            ctx       = t[ctx_start:ctx_end]
            timeline  = next(
                (tl for tl, hints in _TIMELINE_HINTS.items() if any(h in ctx for h in hints)),
                "mid",
            )
            scale = _SCALE_MAP.get(etype, {}).get(timeline, "low")
            events.append({"type": etype, "timeline": timeline, "scale": scale, "matched": m.group(0)})
            reasons.append(f"Future event: '{m.group(0)}' → {etype}/{timeline}/{scale}")

    high = sum(1 for e in events if e["scale"] == "high")
    mid  = sum(1 for e in events if e["scale"] == "medium")
    low  = sum(1 for e in events if e["scale"] == "low")

    if high >= 2:   score = 20.0
    elif high == 1: score = 10.0
    elif mid >= 1:  score = 10.0
    elif low >= 1:  score = 5.0
    else:           score = 0.0

    if score > 0:
        reasons.append(f"future_obligation_score={score}")

    return events, score, reasons


# ---------------------------------------------------------------------------
# SINGLE LLM CALL — llama3.1:8b
# Rule context is injected so the model never re-derives numeric fields.
# ---------------------------------------------------------------------------

# The ONE master prompt. No other prompt exists in this codebase.
_LLM_MASTER_PROMPT = """You are a financial behavior analyst extracting investor profile fields.

GROUND TRUTH (already extracted by rules — DO NOT override these):
{known_values_json}

Your job: extract ONLY the interpretive fields below from the investor description.
Do NOT re-extract or modify any field listed in GROUND TRUTH.
Return null for any field you cannot determine with confidence.

━━━ FIELD DEFINITIONS ━━━

income_type: How the investor earns money.
  "salaried"  = fixed monthly salary from employer
  "business"  = owns/runs a business, self-employed with revenue
  "gig"       = freelancer, contractor, platform worker, variable income
  "unknown"   = genuinely unclear
  → Only provide if NOT already in GROUND TRUTH.

dependents: Integer count of people financially dependent on this investor.
  Count family members who rely on this investor's income.
  → null if not mentioned.

experience_years: Years of investment experience as a decimal number.
  → null if not mentioned.

financial_knowledge_score: Rate 1–5 based on demonstrated knowledge:
  1 = Never invested, no understanding of financial terms
  2 = Knows FD/savings/gold, heard of mutual funds, no real investing
  3 = Has invested in MF or stocks, understands risk/return basics
  4 = Actively manages portfolio, understands asset allocation, P/E, NAV
  5 = Expert — understands derivatives, complex instruments, portfolio theory
  → Return null if you cannot clearly place them. Do NOT default to 3.

decision_autonomy: Does this investor make financial decisions independently?
  true  = researches independently, decides alone
  false = follows tips, peer-influenced, family decides, finfluencer-driven
  → null if unclear.

loss_reaction: How does this investor emotionally respond to portfolio losses?
  "panic"     = stopped investing, panic sold, wanted to exit completely,
                can't sleep, sleepless nights, considered stopping SIP
  "cautious"  = worried, stressed, anxious, reduced exposure, paused SIP,
                shifted to safer options — but did NOT exit
  "neutral"   = stays calm, holds steady, not worried about short-term moves
  "aggressive"= buys more when markets fall, sees dips as opportunity
  CRITICAL: "worried" alone → "cautious", NOT "panic"
  CRITICAL: panic signals override any aggressive language in the same text

loss_reaction_description: A 1-2 sentence description of HOW the investor reacted to losses.
  Capture nuance: was it a one-time panic or a pattern? Did they recover? What triggered it?
  Example: "Panicked once during the 2020 crash but resumed investing after 3 months."
  Example: "Consistently cautious — reduces SIP during downturns but never exits fully."
  → null if no loss reaction information is available.

risk_behavior: Overall risk-taking orientation.
  "low"    = avoids risk, prefers guaranteed returns, capital preservation
  "medium" = balanced, accepts some risk for moderate returns
  "high"   = actively seeks high-return opportunities, comfortable with volatility
  CRITICAL: "calculated risks" → "medium", NOT "high"
  CRITICAL: if loss_reaction is "panic", risk_behavior MUST be "low"

near_term_obligation_level: Urgency of upcoming major financial commitment.
  "high"     = within ~6 months: wedding, medical emergency, imminent purchase
  "moderate" = 6–24 months: education fees, planned home loan, upcoming event
  "none"     = no major upcoming expense (use this if unclear)

obligation_type: Type of near-term obligation (only if level is not "none").
  "wedding" | "house" | "education" | "medical" | "family" | "other"
  → null if near_term_obligation_level is "none"

━━━ OUTPUT FORMAT ━━━
Return ONLY a valid JSON object with exactly these keys:
{{
  "income_type": "salaried|business|gig|unknown|null",
  "dependents": integer_or_null,
  "experience_years": number_or_null,
  "financial_knowledge_score": integer_1_to_5_or_null,
  "decision_autonomy": true_or_false_or_null,
  "loss_reaction": "panic|cautious|neutral|aggressive|null",
  "loss_reaction_description": "1-2 sentence description of how they reacted, or null",
  "risk_behavior": "low|medium|high|null",
  "near_term_obligation_level": "none|moderate|high",
  "obligation_type": "wedding|house|education|medical|family|other|null"
}}
No markdown. No explanation. JSON only."""


def _parse_json(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    return json.loads(text)


def run_llm_analysis(
    normalized_text: str,
    rule_fields: dict[str, FieldValue],
) -> tuple[dict[str, FieldValue], str | None]:
    """
    Single LLM call: llama3.1:8b extracts interpretive fields.
    Rule context is injected — model is told not to override rule-owned values.

    Returns ({field: FieldValue}, warning_or_None).
    """
    # Build known_values from rule fields (only non-null values)
    known = {
        k: v.value for k, v in rule_fields.items()
        if v.value is not None and k not in DERIVED_FIELDS
    }
    known_json = json.dumps(known, indent=2) if known else "{}"

    prompt = _LLM_MASTER_PROMPT.format(known_values_json=known_json)
    prompt += f"\n\nInvestor description:\n{normalized_text}"

    payload = {
        "model":   LLM_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0, "num_predict": 512},
        "format":  "json",
    }

    warning  = None
    raw_dict = {}

    for attempt in (1, 2):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=None,
            )
            resp.raise_for_status()
            raw_dict = _parse_json(resp.json().get("response", ""))
            break
        except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
            if attempt == 2:
                warning  = f"LLM analysis failed after 2 attempts: {e}"
                raw_dict = {}

    # Build FieldValue dict for LLM-owned fields only
    llm_fields: dict[str, FieldValue] = {}
    for fname in LLM_OWNED_FIELDS:
        val = raw_dict.get(fname)
        llm_fields[fname] = make_field(fname, val, "llm")

    return llm_fields, warning


# ---------------------------------------------------------------------------
# MERGE — rule wins, always
# Simple precedence: rule_fields override llm_fields for any shared key.
# ---------------------------------------------------------------------------

def merge_fields(
    rule_fields: dict[str, FieldValue],
    llm_fields:  dict[str, FieldValue],
) -> tuple[dict[str, FieldValue], list[dict]]:
    """
    Rule-wins merge. No arbitration needed — ownership is clear.

    Rule-owned fields: always use rule value (LLM was told not to touch these).
    LLM-owned fields:  use LLM value; rule value only as fallback if LLM returned null.
    """
    merged: dict[str, FieldValue] = {}
    log: list[dict] = []

    all_fields = set(rule_fields) | set(llm_fields)

    for fname in all_fields:
        if fname in DERIVED_FIELDS:
            continue  # computed later

        rule_fv = rule_fields.get(fname)
        llm_fv  = llm_fields.get(fname)

        if fname in RULE_OWNED_FIELDS:
            # Rule always wins
            chosen = rule_fv or FieldValue.null("default")
            reason = "rule-owned"
            if rule_fv and llm_fv and llm_fv.value is not None and rule_fv.value != llm_fv.value:
                reason = f"rule-owned: rule={rule_fv.value} (llm={llm_fv.value} discarded)"
        else:
            # LLM-owned: use LLM, fall back to rule if LLM is null
            if llm_fv and llm_fv.value is not None:
                chosen = llm_fv
                reason = "llm-owned"
            elif rule_fv and rule_fv.value is not None:
                chosen = rule_fv
                reason = "llm-owned: llm=null, rule fallback used"
            else:
                chosen = FieldValue.null("default")
                reason = "llm-owned: both null"

        merged[fname] = chosen
        log.append({
            "field":      fname,
            "rule_value": rule_fv.value if rule_fv else None,
            "llm_value":  llm_fv.value  if llm_fv  else None,
            "chosen":     chosen.value,
            "source":     chosen.source,
            "reason":     reason,
        })

    return merged, log


# ---------------------------------------------------------------------------
# POST-MERGE INVARIANTS — deterministic, code-enforced
# These replace the old correction prompt's "MANDATORY CORRECTIONS" section.
# ---------------------------------------------------------------------------

def apply_invariants(
    fields: dict[str, FieldValue],
) -> tuple[dict[str, FieldValue], list[dict]]:
    """
    Enforce structural consistency after merge.
    v12: panic → risk_behavior=low removed (LLM reasons about this in narrative).
    Only structural/data-integrity invariants remain.

    Invariants:
      1. experience_years < 1 → financial_knowledge_score capped at 2
         (factual: cannot have deep knowledge with <1yr experience)
      2. obligation_type must be null when level is "none"
         (structural: type without level is meaningless)
    """
    updated = dict(fields)
    log: list[dict] = []

    def _force(fname: str, val, reason: str):
        old = updated.get(fname)
        updated[fname] = make_field(fname, val, "rule")
        log.append({
            "field": fname, "action": "invariant",
            "old_value": old.value if old else None,
            "new_value": val, "reason": reason,
        })

    # Invariant 1: novice experience → cap knowledge score (factual constraint)
    exp_fv = updated.get("experience_years")
    fks_fv = updated.get("financial_knowledge_score")
    if (exp_fv and exp_fv.value is not None and exp_fv.value < 1
            and fks_fv and fks_fv.value is not None and fks_fv.value > 2):
        _force("financial_knowledge_score", 2,
               f"experience={exp_fv.value}y < 1 → knowledge capped to 2")

    # Invariant 2: obligation_type must be null when level is none (structural)
    ntol = updated.get("near_term_obligation_level")
    ot   = updated.get("obligation_type")
    if ntol and ntol.value in ("none", None) and ot and ot.value is not None:
        _force("obligation_type", None, "obligation_type cleared — level is none")

    # NOTE: panic → risk_behavior=low removed in v12.
    # Whether panic indicates low risk capacity is a contextual judgment
    # made by the narrative layer, not a hard code rule.

    return updated, log


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_investor_data(paragraph: str) -> dict:
    """
    v7 pipeline:
      [0] Non-English detection
      [1] Text normalization (currency, months→years)
      [2] Rule extraction (deterministic numeric fields)
      [3] Future intent detection (deterministic)
      [4] Single LLM call (llama3.1:8b, rule context injected)
      [5] Rule-wins merge
      [6] Post-merge invariants (code-enforced)
      [7] Invariant check

    Returns structured result ready for validation.
    """
    if _detect_non_english(paragraph):
        return {
            "fields":                {},
            "normalized_text":       paragraph,
            "rule_fields":           {},
            "llm_fields":            {},
            "merge_log":             [],
            "rule_log":              [],
            "invariant_log":         [],
            "future_events":         [],
            "future_obligation_score": 0.0,
            "future_intent_reasons": ["Non-English input — skipped"],
            "extraction_warning":    "Non-English input detected; extraction skipped.",
            "non_english":           True,
        }

    text = normalize_text(paragraph)

    # [2] Rule extraction
    rule_fields, rule_log = run_rule_extraction(text)

    # [3] Future intent
    future_events, future_obligation_score, future_reasons = detect_future_events(text)

    # [4] Single LLM call
    llm_fields, llm_warning = run_llm_analysis(text, rule_fields)

    # [5] Merge (rule wins)
    merged, merge_log = merge_fields(rule_fields, llm_fields)

    # [6] Post-merge invariants
    merged, invariant_log = apply_invariants(merged)

    # [7] Invariant check
    violations = check_invariants(merged)
    for fname, fv in merged.items():
        if fv.value is None and fv.confidence == "high":
            merged[fname] = FieldValue(value=None, confidence="low", source=fv.source)

    return {
        "fields":                merged,
        "normalized_text":       text,
        "rule_fields":           {k: v.to_dict() for k, v in rule_fields.items()},
        "llm_fields":            {k: v.to_dict() for k, v in llm_fields.items()},
        "merge_log":             merge_log,
        "rule_log":              rule_log,
        "invariant_log":         invariant_log,
        "future_events":         future_events,
        "future_obligation_score": future_obligation_score,
        "future_intent_reasons": future_reasons,
        "extraction_warning":    llm_warning,
        "invariant_violations":  violations,
        "non_english":           False,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fields_to_dict(fields: dict[str, FieldValue]) -> dict:
    return {k: v.to_dict() for k, v in fields.items()}
