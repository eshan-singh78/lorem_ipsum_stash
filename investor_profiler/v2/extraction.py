"""
Extraction — InvestorDNA v2
RULE EXTRACTION → SINGLE LLM CALL → RULE-WINS MERGE

LLM Call #1 of 3.
Extracts: income_type, dependents, experience_years, financial_knowledge_score,
          decision_autonomy, loss_reaction, risk_behavior, near_term_obligation_level,
          obligation_type
Rules own: monthly_income, emi_amount, emergency_months, income_type (keyword-gated)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from llm_adapter import llm_call

# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

@dataclass
class ExtractedFields:
    # Rule-owned (deterministic)
    monthly_income:    float | None
    emi_amount:        float | None
    emergency_months:  float | None
    emi_ratio:         float | None   # derived

    # LLM-owned (interpretive)
    income_type:                str | None   # salaried|business|gig|unknown
    dependents:                 int | None
    experience_years:           float | None
    financial_knowledge_score:  int | None   # 1-5
    decision_autonomy:          bool | None
    loss_reaction:              str | None   # panic|cautious|neutral|aggressive
    risk_behavior:              str | None   # low|medium|high
    near_term_obligation_level: str          # none|moderate|high
    obligation_type:            str | None   # wedding|house|education|medical|family|other

    # Meta
    data_completeness: int   # 0-100
    missing_fields:    list[str]
    warning:           str | None
    non_english:       bool


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = re.sub(r"(\d+\.?\d*)\s*(?:cr|crore|crores)\b",
                  lambda m: str(int(float(m.group(1)) * 10_000_000)), text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+\.?\d*)\s*(?:L|lakh|lakhs)\b",
                  lambda m: str(int(float(m.group(1)) * 100_000)), text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+\.?\d*)\s*[kK]\b",
                  lambda m: str(int(float(m.group(1)) * 1_000)), text)
    text = re.sub(
        r"(\d+\.?\d*)\s*months?\s+(?:of\s+)?(?:experience|investing|in\s+(?:stock|market|mutual))",
        lambda m: f"{round(float(m.group(1)) / 12, 2)} years",
        text, flags=re.IGNORECASE,
    )
    return text


def _is_non_english(text: str) -> bool:
    return sum(1 for c in text if ord(c) < 128) / max(len(text), 1) < 0.4


# ---------------------------------------------------------------------------
# Rule extraction
# ---------------------------------------------------------------------------

def _rule_income(text: str) -> float | None:
    t = text.lower()
    m = re.search(r"(\d+\.?\d*)\s*(?:to|[-–])\s*(\d+\.?\d*)\s*(?:lakh|l)"
                  r"\s*(?:per\s*(?:year|annum|pa)|p\.?a\.?|annually)?", t)
    if m:
        return round(((float(m.group(1)) + float(m.group(2))) / 2) * 100_000 / 12)
    m = re.search(r"(\d+\.?\d*)\s*(?:lpa|lakh\s*(?:per\s*annum|p\.?a\.?|annually|per\s*year))", t)
    if m:
        return round(float(m.group(1)) * 100_000 / 12)
    return None


def _rule_emi(text: str) -> float | None:
    m = re.search(r"(?:loan\s+)?emi\s+(?:of\s+)?(?:rs\.?|₹|inr)?\s*(\d[\d,]*(?:\.\d+)?)",
                  text.lower())
    return float(m.group(1).replace(",", "")) if m else None


def _rule_savings(text: str) -> float | None:
    t = text.lower()
    patterns = [
        r"(?:rs\.?|₹|inr)\s*(\d[\d,]*(?:\.\d+)?)\s*(?:in\s+)?(?:savings?(?:\s+account)?|rd|fd|fixed\s+deposit|recurring\s+deposit|gold|ppf|nsc)",
        r"(?:keeps?|has|maintains?|saved?)\s+(?:rs\.?|₹|inr)\s*(\d[\d,]*(?:\.\d+)?)\s*(?:in\s+)?(?:savings?|rd|fd|gold|ppf)",
    ]
    total, seen = 0.0, set()
    for pat in patterns:
        for m in re.finditer(pat, t):
            if m.start() not in seen:
                seen.add(m.start())
                total += float(m.group(1).replace(",", ""))
    return total if total > 0 else None


def _rule_income_type(text: str) -> str | None:
    t = text.lower()
    if any(s in t for s in ["freelancer", "freelance", "consultant", "contractor",
                              "project-based", "variable income", "gig worker"]):
        return "gig"
    if any(s in t for s in ["business owner", "runs a business", "proprietor",
                              "self-employed", "own business", "entrepreneur"]):
        return "business"
    if re.search(r"\d+\.?\d*\s*(?:lpa|lakh\s*per\s*(?:annum|year))", t):
        return "salaried"
    return None


# ---------------------------------------------------------------------------
# LLM extraction prompt — Call #1
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
Extract investor profile fields from the description below.

ALREADY KNOWN (do NOT re-extract):
{known_json}

Extract ONLY these fields. Return null if you cannot determine with confidence.

FIELDS:
- income_type: "salaried" | "business" | "gig" | "unknown"
- dependents: integer (people financially dependent on this investor)
- experience_years: decimal number of years investing
- financial_knowledge_score: 1-5
    1=never invested  2=knows FD/gold  3=has MF/stocks  4=manages portfolio  5=expert
    Return null if unclear. Do NOT default to 3.
- decision_autonomy: true (decides alone) | false (peer/family influenced) | null
- loss_reaction: "panic" | "cautious" | "neutral" | "aggressive"
    panic=stopped/exited  cautious=worried but held  neutral=calm  aggressive=bought more
    "worried" alone → "cautious". Panic overrides aggressive language.
- risk_behavior: "low" | "medium" | "high"
    "calculated risks" → "medium". If loss_reaction=panic → risk_behavior MUST be "low".
- near_term_obligation_level: "none" | "moderate" | "high"
    high=within 6 months  moderate=6-24 months  none=nothing imminent
- obligation_type: "wedding"|"house"|"education"|"medical"|"family"|"other" | null
    null when near_term_obligation_level is "none"

Return ONLY valid JSON — no markdown, no explanation:
{{
  "income_type": "...",
  "dependents": null,
  "experience_years": null,
  "financial_knowledge_score": null,
  "decision_autonomy": null,
  "loss_reaction": "...",
  "risk_behavior": "...",
  "near_term_obligation_level": "none",
  "obligation_type": null
}}

Investor description:
{text}
"""


def _run_llm(text: str, known: dict) -> dict:
    prompt = _EXTRACTION_PROMPT.format(
        known_json=json.dumps(known, indent=2) if known else "{}",
        text=text[:1500],
    )
    for attempt in (1, 2):
        try:
            raw = llm_call(prompt, num_predict=384)
            if raw:
                return raw
        except Exception as e:
            if attempt == 2:
                return {"_warning": str(e)}
    return {}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_INCOME_TYPES  = {"salaried", "business", "gig", "unknown"}
_LOSS_REACTIONS = {"panic", "cautious", "neutral", "aggressive"}
_RISK_BEHAVIORS = {"low", "medium", "high"}
_OBL_LEVELS    = {"none", "moderate", "high"}
_OBL_TYPES     = {"wedding", "house", "education", "medical", "family", "other"}


def _cast_int(v) -> int | None:
    try:
        return int(float(v))
    except Exception:
        return None


def _cast_float(v) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def _cast_bool(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.lower() == "true":  return True
        if v.lower() == "false": return False
    return None


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------

_KEY_FIELDS = [
    "income_type", "monthly_income", "emergency_months", "emi_amount",
    "dependents", "experience_years", "financial_knowledge_score",
    "loss_reaction", "risk_behavior", "near_term_obligation_level",
]


def _completeness(fields: dict) -> tuple[int, list[str]]:
    missing = [f for f in _KEY_FIELDS if fields.get(f) is None or fields.get(f) == "unknown"]
    return int(round((len(_KEY_FIELDS) - len(missing)) / len(_KEY_FIELDS) * 100)), missing


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract(paragraph: str) -> ExtractedFields:
    """
    Rule extraction → single LLM call → merge → validate → return.
    """
    if _is_non_english(paragraph):
        return ExtractedFields(
            monthly_income=None, emi_amount=None, emergency_months=None, emi_ratio=None,
            income_type=None, dependents=None, experience_years=None,
            financial_knowledge_score=None, decision_autonomy=None,
            loss_reaction=None, risk_behavior=None,
            near_term_obligation_level="none", obligation_type=None,
            data_completeness=0, missing_fields=_KEY_FIELDS[:],
            warning="Non-English input detected.", non_english=True,
        )

    text = _normalize(paragraph)

    # --- Rule extraction ---
    r_income   = _rule_income(text)
    r_emi      = _rule_emi(text)
    r_savings  = _rule_savings(text)
    r_inc_type = _rule_income_type(text)

    # Emergency months from savings ÷ income
    r_em = None
    if r_savings is not None and r_income and r_income > 0:
        r_em = round(r_savings / r_income, 1)

    known: dict[str, Any] = {}
    if r_income   is not None: known["monthly_income"]   = r_income
    if r_emi      is not None: known["emi_amount"]        = r_emi
    if r_em       is not None: known["emergency_months"]  = r_em
    if r_inc_type is not None: known["income_type"]       = r_inc_type

    # --- LLM call ---
    raw = _run_llm(text, known)
    warning = raw.pop("_warning", None)

    # --- Merge: rules always win ---
    income_type = r_inc_type or (raw.get("income_type") if raw.get("income_type") in _INCOME_TYPES else "unknown")
    dependents  = _cast_int(raw.get("dependents"))
    exp_years   = _cast_float(raw.get("experience_years"))
    fks_raw     = _cast_int(raw.get("financial_knowledge_score"))
    fks         = fks_raw if fks_raw and 1 <= fks_raw <= 5 else None
    autonomy    = _cast_bool(raw.get("decision_autonomy"))
    loss_r      = raw.get("loss_reaction") if raw.get("loss_reaction") in _LOSS_REACTIONS else None
    risk_b      = raw.get("risk_behavior")  if raw.get("risk_behavior")  in _RISK_BEHAVIORS else None
    obl_level   = raw.get("near_term_obligation_level") if raw.get("near_term_obligation_level") in _OBL_LEVELS else "none"
    obl_type    = raw.get("obligation_type") if raw.get("obligation_type") in _OBL_TYPES else None

    # Invariant: experience < 1yr → knowledge capped at 2
    if exp_years is not None and exp_years < 1 and fks and fks > 2:
        fks = 2

    # Invariant: obligation_type must be null when level is none
    if obl_level == "none":
        obl_type = None

    # Derived: emi_ratio
    emi_ratio = None
    if r_emi is not None and r_income and r_income > 0:
        emi_ratio = round(min((r_emi / r_income) * 100, 100), 2)

    fields = {
        "monthly_income": r_income, "emi_amount": r_emi,
        "emergency_months": r_em, "emi_ratio": emi_ratio,
        "income_type": income_type, "dependents": dependents,
        "experience_years": exp_years, "financial_knowledge_score": fks,
        "decision_autonomy": autonomy, "loss_reaction": loss_r,
        "risk_behavior": risk_b, "near_term_obligation_level": obl_level,
        "obligation_type": obl_type,
    }
    completeness, missing = _completeness(fields)

    return ExtractedFields(
        monthly_income=r_income, emi_amount=r_emi,
        emergency_months=r_em, emi_ratio=emi_ratio,
        income_type=income_type, dependents=dependents,
        experience_years=exp_years, financial_knowledge_score=fks,
        decision_autonomy=autonomy, loss_reaction=loss_r,
        risk_behavior=risk_b, near_term_obligation_level=obl_level,
        obligation_type=obl_type,
        data_completeness=completeness, missing_fields=missing,
        warning=warning, non_english=False,
    )
