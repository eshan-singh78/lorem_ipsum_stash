"""
Stage 2 Correction Layer — v4
Primary:  llama3.1:8b (2 attempts, 180s timeout each)
Fallback: rule-based deterministic engine (triggers if both LLM attempts fail)

Exports:
  correct_extraction(paragraph, extracted_plain)
    → (corrected_dict, corrections_list, fallback_used, correction_source)
  apply_mandatory_rules(data)
    → (corrected_dict, corrections_list)
"""

import json
import re
import requests

OLLAMA_BASE_URL  = "http://localhost:11434"
CORRECTION_MODEL = "llama3.1:8b"

# ---------------------------------------------------------------------------
# LLM correction prompt
# ---------------------------------------------------------------------------

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

INCOME CONVERSION RULES (CRITICAL):
- "X LPA" means X Lakh Per Annum → monthly_income = (X * 100000) / 12
  Example: "7.5 LPA" → monthly_income = 62500
- "X lakh per year" → same conversion
- Do NOT leave monthly_income null if LPA is mentioned

EXPERIENCE CONVERSION RULES (CRITICAL):
- "X months" of experience → experience_years = X / 12
  Example: "8 months" → experience_years = 0.67
- Do NOT convert "8 months" to 8 years

NEAR-TERM OBLIGATION RULES:
Classify near_term_obligation_level based on urgency:
- "high": within ~6 months — wedding, marriage, shaadi, medical emergency,
          house purchase closing soon, "next few months", "very soon", "this year",
          "secretly saving", "earmarked", "family expectation"
- "moderate": 6–24 months — education fees, home loan planned, buying property,
              "planning to", "next year", "upcoming", "in a year or two"
- "none": no major upcoming expense mentioned
IMPORTANT: If unclear → "none". Do NOT default to "moderate".

Set obligation_type if near_term_obligation_level is not "none":
- "wedding" | "house" | "education" | "medical" | "family" | "other"

MANDATORY CORRECTION RULES (enforce all):
1. IF loss_reaction == "panic" OR text mentions anxiety/fear/sleepless/nervous/
   "considered stopping"/"can't sleep"/"worried about loss":
   → loss_reaction = "panic", risk_behavior = "low"
2. IF experience_years < 1 OR text says "just started"/"new to investing"/"never invested":
   → financial_knowledge_score MUST be ≤ 2 (set to null if unsure)
3. IF risk_behavior == "high" AND loss_reaction == "panic":
   → risk_behavior = "low"
4. IF income_type is "gig" but monthly_income >= 150000:
   → keep income_type = "gig", do NOT change it
5. IF emi_amount is present but monthly_income is missing:
   → do NOT fabricate income; leave monthly_income as null
6. IF financial_knowledge_score == 3 and there is no clear evidence of actual investing:
   → set to null

SEMANTIC FIXES:
- "I take calculated risks" → risk_behavior = "medium" (NOT "high")
- "I invest based on tips from friends" / "peer-driven" → decision_autonomy = false
- "I research before investing" / "I decide myself" → decision_autonomy = true
- "I can't sleep when markets fall" → loss_reaction = "panic"
- "I stay calm during volatility" → loss_reaction = "neutral"
- "I buy more when markets crash" / "buy the dip" → loss_reaction = "aggressive"
- "consultant" / "freelancer" / "contractor" → income_type = "gig"
- "proprietor" / "runs a business" / "business owner" → income_type = "business"
- panic/anxiety language ALWAYS overrides any aggressive wording in the same text

OUTPUT FORMAT:
Return a JSON object with exactly two keys:
{
  "corrected": { ...same schema as input, with corrections applied... },
  "corrections": [ "list of strings describing each change made" ]
}

If no corrections needed, return corrections as empty list.
Return ONLY valid JSON. No explanation. No markdown."""


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
# Rule-based fallback correction engine
# Triggers when llama3.1:8b fails after both attempts.
# Covers the highest-impact signals deterministically.
# ---------------------------------------------------------------------------

def _rule_based_correction(paragraph: str, data: dict) -> tuple[dict, list[str]]:
    """
    Deterministic fallback. Applies regex + keyword rules directly to the
    original paragraph. Returns (corrected_data, corrections_list).
    """
    d       = dict(data)
    applied = []
    text    = paragraph.lower()

    # --- Behavior: panic / anxiety signals ---
    panic_signals = [
        "anxious", "anxiety", "panic", "panicked", "scared", "fear",
        "can't sleep", "cannot sleep", "sleepless", "worried", "nervous",
        "considered stopping", "stop investing", "stressed about",
    ]
    if any(sig in text for sig in panic_signals):
        if d.get("loss_reaction") != "panic":
            applied.append(
                f"Rule fallback: panic/anxiety signal detected → "
                f"loss_reaction='panic', risk_behavior='low'"
            )
            d["loss_reaction"] = "panic"
        if d.get("risk_behavior") in ("medium", "high", None):
            d["risk_behavior"] = "low"

    # --- Behavior: aggressive signals (only if no panic) ---
    if d.get("loss_reaction") != "panic":
        aggressive_signals = ["buy the dip", "buy more when", "buy on dips", "add more when market"]
        if any(sig in text for sig in aggressive_signals):
            if d.get("loss_reaction") is None:
                applied.append("Rule fallback: buy-the-dip signal → loss_reaction='aggressive'")
                d["loss_reaction"] = "aggressive"

    # --- Behavior: peer-driven / tips ---
    peer_signals = ["tips from", "friend told", "friends invest", "following others",
                    "peer advice", "someone suggested", "my friend said"]
    if any(sig in text for sig in peer_signals):
        if d.get("decision_autonomy") is None:
            applied.append("Rule fallback: peer-driven signal → decision_autonomy=false")
            d["decision_autonomy"] = False

    # --- Obligation: wedding / marriage ---
    wedding_signals = [
        "wedding", "marriage", "shaadi", "daughter's wedding", "son's wedding",
        "getting married", "sister's wedding", "brother's wedding",
    ]
    if any(sig in text for sig in wedding_signals):
        if d.get("near_term_obligation_level") is None:
            applied.append(
                "Rule fallback: wedding signal → near_term_obligation_level='high', "
                "obligation_type='wedding'"
            )
            d["near_term_obligation_level"] = "high"
            d["obligation_type"]            = "wedding"

    # --- Obligation: hidden / earmarked savings ---
    hidden_obligation_signals = [
        "secretly saving", "earmarked", "family expectation",
        "saving for", "set aside for", "keeping aside",
    ]
    if any(sig in text for sig in hidden_obligation_signals):
        if d.get("near_term_obligation_level") is None:
            applied.append(
                "Rule fallback: hidden obligation signal → near_term_obligation_level='high'"
            )
            d["near_term_obligation_level"] = "high"
            if d.get("obligation_type") is None:
                d["obligation_type"] = "family"

    # --- Obligation: house purchase ---
    house_signals = ["house purchase", "buying a house", "buying property",
                     "home loan", "property purchase", "flat purchase"]
    if any(sig in text for sig in house_signals):
        if d.get("near_term_obligation_level") is None:
            # Urgency: "this year" / "soon" → high, else moderate
            urgent = any(u in text for u in ["this year", "very soon", "next few months", "shortly"])
            level  = "high" if urgent else "moderate"
            applied.append(
                f"Rule fallback: house purchase signal → "
                f"near_term_obligation_level='{level}', obligation_type='house'"
            )
            d["near_term_obligation_level"] = level
            d["obligation_type"]            = "house"

    # --- Income: LPA conversion ---
    # Matches "7.5 LPA", "7.5lpa", "7.5 lakh per annum", "7.5 lakh pa"
    lpa_match = re.search(
        r"(\d+\.?\d*)\s*(?:lpa|lakh\s*(?:per\s*annum|p\.?a\.?))",
        text
    )
    if lpa_match and d.get("monthly_income") is None:
        lpa    = float(lpa_match.group(1))
        monthly = round((lpa * 100_000) / 12)
        applied.append(
            f"Rule fallback: '{lpa_match.group(0)}' detected → "
            f"monthly_income={monthly} ({lpa} LPA / 12)"
        )
        d["monthly_income"] = monthly
        if d.get("income_type") in (None, "unknown"):
            d["income_type"] = "salaried"   # LPA is almost always salaried

    # --- Experience: months → years ---
    # Only if experience_years is missing or suspiciously large (months misread as years)
    exp_months_match = re.search(
        r"(\d+\.?\d*)\s*months?\s+(?:of\s+)?(?:experience|investing|in\s+(?:stock|market|mutual))",
        text
    )
    if exp_months_match:
        months = float(exp_months_match.group(1))
        years  = round(months / 12, 2)
        current_exp = d.get("experience_years")
        # Fix if missing, or if qwen misread months as years (value == months, not years)
        if current_exp is None or (months > 1 and abs(current_exp - months) < 0.5):
            applied.append(
                f"Rule fallback: '{exp_months_match.group(0)}' → "
                f"experience_years={years} ({months} months / 12)"
            )
            d["experience_years"] = years

    # --- Income type: freelancer / consultant ---
    gig_signals = ["freelancer", "freelance", "consultant", "contractor",
                   "project-based", "project based", "variable income"]
    if any(sig in text for sig in gig_signals):
        if d.get("income_type") in (None, "unknown"):
            applied.append("Rule fallback: freelancer/consultant signal → income_type='gig'")
            d["income_type"] = "gig"

    business_signals = ["business owner", "runs a business", "proprietor",
                        "self-employed", "own business"]
    if any(sig in text for sig in business_signals):
        if d.get("income_type") in (None, "unknown"):
            applied.append("Rule fallback: business owner signal → income_type='business'")
            d["income_type"] = "business"

    return d, applied


# ---------------------------------------------------------------------------
# Post-extraction validation (runs regardless of correction source)
# Catches structural gaps that both LLM and rules may miss.
# ---------------------------------------------------------------------------

def validate_extraction(paragraph: str, data: dict) -> tuple[dict, list[str]]:
    """
    Structural gap checks after correction.
    Fixes cases where a signal is clearly in the text but the field is still null.
    """
    d       = dict(data)
    fixes   = []
    text    = paragraph.lower()

    # 1. LPA present but monthly_income still null
    lpa_match = re.search(
        r"(\d+\.?\d*)\s*(?:lpa|lakh\s*(?:per\s*annum|p\.?a\.?))",
        text
    )
    if lpa_match and d.get("monthly_income") is None:
        lpa     = float(lpa_match.group(1))
        monthly = round((lpa * 100_000) / 12)
        fixes.append(
            f"Extraction validation: LPA '{lpa_match.group(0)}' present but "
            f"monthly_income was null → set to {monthly}"
        )
        d["monthly_income"] = monthly

    # 2. Wedding keyword present but obligation not set
    wedding_kw = ["wedding", "marriage", "shaadi"]
    if any(kw in text for kw in wedding_kw):
        if d.get("near_term_obligation_level") is None:
            fixes.append(
                "Extraction validation: wedding keyword present but "
                "near_term_obligation_level was null → set to 'high'"
            )
            d["near_term_obligation_level"] = "high"
            d["obligation_type"]            = "wedding"

    # 3. Panic keyword present but behavior not set
    panic_kw = ["anxious", "panic", "can't sleep", "cannot sleep", "scared", "fear of loss"]
    if any(kw in text for kw in panic_kw):
        if d.get("loss_reaction") != "panic":
            fixes.append(
                "Extraction validation: panic keyword present but "
                "loss_reaction was not 'panic' → corrected"
            )
            d["loss_reaction"] = "panic"
            d["risk_behavior"] = "low"

    # 4. risk_behavior=high but loss_reaction=panic → mismatch
    if d.get("risk_behavior") == "high" and d.get("loss_reaction") == "panic":
        fixes.append(
            "Extraction validation: risk_behavior=high conflicts with "
            "loss_reaction=panic → risk_behavior corrected to 'low'"
        )
        d["risk_behavior"] = "low"

    return d, fixes


# ---------------------------------------------------------------------------
# Main correction entry point
# ---------------------------------------------------------------------------

def correct_extraction(
    paragraph: str,
    extracted_plain: dict,
) -> tuple[dict, list[str], bool, str]:
    """
    Run llama3.1:8b correction. Falls back to rule-based engine on failure.

    Returns:
        (corrected_dict, corrections_list, fallback_used, correction_source)
        correction_source: "llm" | "rules"
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

    last_error = None

    for attempt in (1, 2):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=180,   # extended from 120s
            )
            resp.raise_for_status()
            raw    = resp.json().get("response", "")
            parsed = _parse_json(raw)

            corrected   = parsed.get("corrected", {})
            corrections = parsed.get("corrections", [])

            if not isinstance(corrections, list):
                corrections = [str(corrections)]

            merged = dict(extracted_plain)
            merged.update({k: v for k, v in corrected.items() if k in corrected})

            return merged, corrections, False, "llm"

        except (requests.RequestException, json.JSONDecodeError, ValueError, KeyError) as e:
            last_error = e
            # attempt 1 failed — loop continues for attempt 2

    # Both LLM attempts failed — activate rule-based fallback
    fallback_note = f"LLM correction failed after 2 attempts ({last_error}) — rule-based fallback activated"
    corrected, rule_corrections = _rule_based_correction(paragraph, extracted_plain)
    rule_corrections = [fallback_note] + rule_corrections

    return corrected, rule_corrections, True, "rules"


# ---------------------------------------------------------------------------
# Mandatory rules — always run after correction (LLM or rules)
# ---------------------------------------------------------------------------

def apply_mandatory_rules(data: dict) -> tuple[dict, list[str]]:
    """
    Final deterministic safety net. Runs after correct_extraction().
    Also calls validate_extraction() for structural gap checks.
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

    # Rule 3: near_term_obligation_level must be valid enum
    ntol = d.get("near_term_obligation_level")
    if ntol not in ("none", "moderate", "high", None):
        applied.append(
            f"Mandatory: near_term_obligation_level='{ntol}' invalid → set to 'none'"
        )
        d["near_term_obligation_level"] = "none"

    # Rule 4: obligation_type only valid when level is not none/null
    if d.get("near_term_obligation_level") in ("none", None):
        if d.get("obligation_type") is not None:
            applied.append(
                "Mandatory: obligation_type cleared because near_term_obligation_level is none"
            )
            d["obligation_type"] = None

    return d, applied
