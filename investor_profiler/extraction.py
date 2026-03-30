"""
NLP Extraction Layer — Stage 1: qwen2.5:3b (fast JSON extraction)
v3: wedding_flag removed, near_term_obligation_level + obligation_type added.
    financial_knowledge_score=3 with low confidence → null (drift fix).
    Freelancer/consultant → gig classification improved.
    Months → years conversion in pre-processing.
"""

import json
import re
import requests

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "qwen2.5:3b-instruct"


# ---------------------------------------------------------------------------
# Pre-processing: currency + time normalization
# ---------------------------------------------------------------------------

def _normalize_currency(text: str) -> str:
    """Convert Indian currency shorthand to plain numbers."""
    text = re.sub(
        r"(\d+\.?\d*)\s*(?:cr|crore|crores)\b",
        lambda m: str(int(float(m.group(1)) * 10_000_000)),
        text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"(\d+\.?\d*)\s*(?:L|lakh|lakhs)\b",
        lambda m: str(int(float(m.group(1)) * 100_000)),
        text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"(\d+\.?\d*)\s*[kK]\b",
        lambda m: str(int(float(m.group(1)) * 1_000)),
        text
    )
    return text


def _normalize_experience(text: str) -> str:
    """Convert 'X months experience/investing' to fractional years."""
    def _months_to_years(m):
        months = float(m.group(1))
        years  = round(months / 12, 2)
        return f"{years} years"

    text = re.sub(
        r"(\d+\.?\d*)\s*months?\s+(?:of\s+)?(?:experience|investing|in\s+(?:stock|market|mutual))",
        _months_to_years,
        text, flags=re.IGNORECASE
    )
    return text


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a financial data extraction assistant.
Extract structured data from the investor description strictly following the schema.

FINANCIAL KNOWLEDGE SCORE RUBRIC (use strictly — do NOT default to 3):
1 = No knowledge — never invested, doesn't understand basic terms
2 = Basic awareness — knows FD/savings, heard of mutual funds, no real experience
3 = Moderate understanding — has invested in MF/stocks, understands risk/return basics
4 = Strong knowledge — actively manages portfolio, understands asset allocation
5 = Expert level — professional knowledge, understands derivatives, complex instruments
If you are not confident about the score, return null.

INCOME TYPE RULES:
- "salaried" = fixed monthly salary from employer (employee, job, company)
- "business" = owns a business, self-employed with stable revenue, proprietor
- "gig" = freelancer, consultant, contractor, variable income, project-based
- If unclear → "unknown"

NEAR-TERM OBLIGATION RULES:
Detect any major upcoming financial commitment and classify:
- "high": urgent or within 6 months — wedding, marriage, shaadi, medical emergency,
          house purchase closing soon, "next few months", "very soon"
- "moderate": planned within 6–24 months — education fees, home loan planned,
              buying property, "planning to", "next year", "upcoming"
- "none": no mention of any major upcoming expense
IMPORTANT: If unclear or not mentioned → "none". Do NOT default to "moderate".

OBLIGATION TYPE (only if near_term_obligation_level is not "none"):
- "wedding" | "house" | "education" | "medical" | "family" | "other"

BEHAVIORAL PRIORITY RULES:
1. panic / anxiety / fear of loss / "can't sleep" → loss_reaction = "panic", risk_behavior = "low"
2. following others / tips / friends' advice → decision_autonomy = false
3. experience < 1 year / "just started" / "new to investing" → experience_years = 0, financial_knowledge_score ≤ 2
4. researches independently / decides themselves → decision_autonomy = true
5. "calculated risks" → risk_behavior = "medium" (NOT "high")
6. "buy more when markets fall" / "buy the dip" → loss_reaction = "aggressive"
7. "stay calm" / "don't panic" → loss_reaction = "neutral"

CONFLICT RULES:
- loss_reaction == "panic" → risk_behavior MUST be "low"
- experience_years < 1 → financial_knowledge_score MUST be ≤ 2
- risk_behavior == "high" AND loss_reaction == "panic" → set risk_behavior = "low"

OUTPUT RULES:
- Return ONLY valid JSON, no explanation, no markdown
- Missing or unclear → null
- DO NOT guess values not grounded in the text
- emi_ratio: ONLY if explicitly stated as a percentage/ratio in the text (NOT derived)
- financial_knowledge_score: return null if you are not confident"""

SCHEMA_DESCRIPTION = """{
  "income_type": "salaried | business | gig | unknown",
  "monthly_income": "number in rupees | null",
  "emergency_months": "number | null",
  "emi_amount": "number in rupees | null",
  "emi_ratio": "number 0-100 | null  (ONLY if explicitly stated as %, never derive it)",
  "dependents": "integer | null",
  "near_term_obligation_level": "none | moderate | high | null",
  "obligation_type": "wedding | house | education | medical | family | other | null",
  "experience_years": "number | null",
  "financial_knowledge_score": "integer 1-5 using rubric above | null (prefer null over guessing)",
  "decision_autonomy": "true | false | null",
  "loss_reaction": "panic | neutral | aggressive | null",
  "risk_behavior": "low | medium | high | null"
}"""


# ---------------------------------------------------------------------------
# Confidence assignment
# ---------------------------------------------------------------------------

# Fields where a non-null value is grounded in explicit text → high confidence
_NUMERIC_FIELDS = {
    "monthly_income", "emergency_months", "emi_amount",
    "emi_ratio", "dependents", "experience_years",
}
# Fields that require interpretation → low confidence
_TONE_INFERRED = {
    "loss_reaction", "risk_behavior", "decision_autonomy",
    "near_term_obligation_level", "obligation_type",
}
# knowledge score: high only if value is 1, 2, 4, or 5 (extremes are unambiguous)
# 3 is the "default guess" — mark as low so drift-fix can null it
_KNOWLEDGE_AMBIGUOUS = {3}


def _assign_confidence(field: str, value) -> str:
    if value is None:
        return "high"   # confident absence
    if field in _NUMERIC_FIELDS:
        return "high"
    if field == "income_type":
        return "high" if value not in ("unknown", None) else "low"
    if field == "financial_knowledge_score":
        return "low" if value in _KNOWLEDGE_AMBIGUOUS else "high"
    if field in _TONE_INFERRED:
        return "low"
    return "medium"


def _wrap_with_confidence(raw: dict) -> dict:
    return {
        field: {"value": value, "confidence": _assign_confidence(field, value)}
        for field, value in raw.items()
    }


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


# ---------------------------------------------------------------------------
# Non-English detection
# ---------------------------------------------------------------------------

def _detect_non_english(paragraph: str) -> str | None:
    ascii_ratio = sum(1 for c in paragraph if ord(c) < 128) / max(len(paragraph), 1)
    if ascii_ratio < 0.4:
        return "Non-English input detected; extraction may be unreliable."
    return None


# ---------------------------------------------------------------------------
# Post-extraction: knowledge score drift fix
# ---------------------------------------------------------------------------

def _fix_knowledge_drift(raw: dict, confidences: dict) -> tuple[dict, list[str]]:
    """
    If financial_knowledge_score == 3 and confidence == low → null it.
    This prevents qwen's default-to-3 bias from inflating context scores.
    """
    fixes = []
    fks = raw.get("financial_knowledge_score")
    if fks == 3 and confidences.get("financial_knowledge_score") == "low":
        raw["financial_knowledge_score"] = None
        confidences["financial_knowledge_score"] = "high"  # confident null
        fixes.append("financial_knowledge_score=3 with low confidence → nulled (drift fix)")
    return raw, fixes


# ---------------------------------------------------------------------------
# Main extraction function (Stage 1 only)
# ---------------------------------------------------------------------------

def extract_investor_data(paragraph: str) -> dict:
    """
    Stage 1: Extract structured features via qwen2.5:3b.

    Returns:
        {
          "fields":             { field: { "value": ..., "confidence": ... } },
          "conflicts_resolved": [...],
          "extraction_warning": str | None,
          "non_english":        bool,
        }
    """
    non_english_warning = _detect_non_english(paragraph)
    if non_english_warning:
        return {
            "fields":             _wrap_with_confidence({}),
            "conflicts_resolved": [],
            "extraction_warning": non_english_warning,
            "non_english":        True,
        }

    normalized_paragraph = _normalize_currency(_normalize_experience(paragraph))

    prompt = (
        f"{SYSTEM_PROMPT}\n\nSchema:\n{SCHEMA_DESCRIPTION}\n\n"
        f"Investor description:\n{normalized_paragraph}\n\nReturn only the JSON object."
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
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=90,
            )
            resp.raise_for_status()
            raw      = resp.json().get("response", "")
            raw_dict = _parse_json(raw)
            break
        except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
            if attempt == 2:
                warning  = f"LLM extraction failed after 2 attempts: {e}."
                raw_dict = {}

    # Build confidence map before drift fix
    wrapped     = _wrap_with_confidence(raw_dict)
    plain_vals  = {k: v["value"] for k, v in wrapped.items()}
    conf_map    = {k: v["confidence"] for k, v in wrapped.items()}

    plain_vals, drift_fixes = _fix_knowledge_drift(plain_vals, conf_map)

    # Rebuild wrapped with corrected confidences
    fields = {
        field: {"value": plain_vals[field], "confidence": conf_map[field]}
        for field in plain_vals
    }

    return {
        "fields":             fields,
        "conflicts_resolved": drift_fixes,
        "extraction_warning": warning,
        "non_english":        False,
    }


def unwrap_values(extracted: dict) -> dict:
    """Flatten fields dict to plain {field: value}."""
    return {k: v["value"] for k, v in extracted.get("fields", {}).items()}


def unwrap_confidences(extracted: dict) -> dict:
    """Return {field: confidence} map."""
    return {k: v["confidence"] for k, v in extracted.get("fields", {}).items()}
