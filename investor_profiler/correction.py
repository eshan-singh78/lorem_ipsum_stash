"""
Stage 2 Correction Layer — llama3.1:8b
v3: wedding_flag removed, near_term_obligation_level + obligation_type added.
    Freelancer/consultant classification improved.
    Months → years conversion enforced.
    knowledge_score=3 low-confidence drift fix applied post-correction.
"""

import json
import re
import requests

OLLAMA_BASE_URL  = "http://localhost:11434"
CORRECTION_MODEL = "llama3.1:8b"

CORRECTION_PROMPT = """You are a financial data correction specialist.
You will receive an investor description and a first-pass JSON extraction.
Your job is to CORRECT errors, fill gaps, and enforce rules.

FINANCIAL KNOWLEDGE SCORE RUBRIC (use strictly):
1 = No knowledge — never invested, doesn't understand basic terms
2 = Basic awareness — knows FD/savings, heard of mutual funds, no real experience
3 = Moderate understanding — has invested in MF/stocks, understands risk/return basics
4 = Strong knowledge — actively manages portfolio, understands asset allocation
5 = Expert level — professional knowledge, understands derivatives, complex instruments
If you cannot confidently assign a score, set it to null.

INCOME TYPE RULES:
- "salaried" = fixed monthly salary from employer
- "business" = owns a business, self-employed with stable revenue
- "gig" = freelancer, consultant, contractor, variable/project-based income
- Correct "unknown" if the text clearly implies one of the above.

NEAR-TERM OBLIGATION RULES:
Classify near_term_obligation_level based on urgency:
- "high": within ~6 months — wedding, marriage, shaadi, medical emergency,
          house purchase closing soon, "next few months", "very soon", "this year"
- "moderate": 6–24 months — education fees, home loan planned, buying property,
              "planning to", "next year", "upcoming", "in a year or two"
- "none": no major upcoming expense mentioned
IMPORTANT: If unclear → "none". Do NOT default to "moderate".

Set obligation_type if near_term_obligation_level is not "none":
- "wedding" | "house" | "education" | "medical" | "family" | "other"

MANDATORY CORRECTION RULES (enforce all):
1. IF loss_reaction == "panic" OR text mentions anxiety/fear/sleepless/nervous about loss:
   → loss_reaction = "panic", risk_behavior = "low"
2. IF experience_years < 1 OR text says "just started"/"new to investing"/"never invested":
   → financial_knowledge_score MUST be ≤ 2 (set to null if unsure)
3. IF risk_behavior == "high" AND loss_reaction == "panic":
   → risk_behavior = "low"
4. IF income_type is "gig" but monthly_income >= 150000:
   → keep income_type = "gig", do NOT change it
5. IF emi_amount is present but monthly_income is missing:
   → do NOT fabricate income; leave monthly_income as null
6. IF experience_years is stated in months (e.g. "6 months experience"):
   → convert to fractional years (6 months = 0.5)
7. IF financial_knowledge_score == 3 and there is no clear evidence of actual investing:
   → set to null

SEMANTIC FIXES:
- "I take calculated risks" → risk_behavior = "medium" (NOT "high")
- "I invest based on tips from friends" → decision_autonomy = false
- "I research before investing" / "I decide myself" → decision_autonomy = true
- "I can't sleep when markets fall" → loss_reaction = "panic"
- "I stay calm during volatility" → loss_reaction = "neutral"
- "I buy more when markets crash" / "buy the dip" → loss_reaction = "aggressive"
- "consultant" / "freelancer" / "contractor" → income_type = "gig"
- "proprietor" / "runs a business" / "business owner" → income_type = "business"

OUTPUT FORMAT:
Return a JSON object with exactly two keys:
{
  "corrected": { ...same schema as input, with corrections applied... },
  "corrections": [ "list of strings describing each change made" ]
}

If no corrections needed, return corrections as empty list.
Return ONLY valid JSON. No explanation. No markdown."""


def _parse_json(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def correct_extraction(paragraph: str, extracted_plain: dict) -> tuple[dict, list[str]]:
    """
    Run llama3.1:8b correction pass on qwen's extraction.

    Returns:
        (corrected_dict, corrections_list)
        On failure, returns (extracted_plain, ["Correction pass failed: <reason>"])
    """
    prompt = f"""{CORRECTION_PROMPT}

Original investor description:
{paragraph}

First-pass extraction (JSON):
{json.dumps(extracted_plain, indent=2)}

Return the corrected JSON object with corrections list."""

    payload = {
        "model":   CORRECTION_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0, "num_predict": 900},
        "format":  "json",
    }

    for attempt in (1, 2):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            raw    = resp.json().get("response", "")
            parsed = _parse_json(raw)

            corrected   = parsed.get("corrected", {})
            corrections = parsed.get("corrections", [])

            if not isinstance(corrections, list):
                corrections = [str(corrections)]

            # Merge: keep original fields not overwritten by corrector
            merged = dict(extracted_plain)
            merged.update({k: v for k, v in corrected.items() if k in corrected})

            return merged, corrections

        except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
            if attempt == 2:
                return extracted_plain, [f"Correction pass failed after 2 attempts: {e}"]

    return extracted_plain, []


def apply_mandatory_rules(data: dict) -> tuple[dict, list[str]]:
    """
    Final deterministic safety net — runs AFTER llama correction.
    Catches anything the LLM missed.
    """
    d       = dict(data)
    applied = []

    # Rule 1: panic → low risk (absolute)
    if d.get("loss_reaction") == "panic" and d.get("risk_behavior") in ("medium", "high"):
        applied.append(
            f"Mandatory: loss_reaction=panic → risk_behavior forced from "
            f"'{d['risk_behavior']}' to 'low'"
        )
        d["risk_behavior"] = "low"

    # Rule 2: experience < 1 → knowledge ≤ 2
    exp = d.get("experience_years")
    fks = d.get("financial_knowledge_score")
    if exp is not None and exp < 1 and fks is not None and fks > 2:
        applied.append(
            f"Mandatory: experience_years={exp} < 1 → "
            f"financial_knowledge_score capped from {fks} to 2"
        )
        d["financial_knowledge_score"] = 2

    # Rule 3: near_term_obligation_level must be valid
    ntol = d.get("near_term_obligation_level")
    if ntol not in ("none", "moderate", "high", None):
        applied.append(
            f"Mandatory: near_term_obligation_level='{ntol}' invalid → set to 'none'"
        )
        d["near_term_obligation_level"] = "none"

    # Rule 4: obligation_type only valid when obligation level is not none
    if d.get("near_term_obligation_level") in ("none", None):
        if d.get("obligation_type") is not None:
            applied.append(
                "Mandatory: obligation_type cleared because near_term_obligation_level is none"
            )
            d["obligation_type"] = None

    return d, applied
