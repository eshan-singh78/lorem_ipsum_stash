"""
Category-Based Scoring Engine — v3, deterministic, no LLM.
Receives pre-normalized data (normalize() already called in scoring.py).

6 Categories:
  income_stability        — positive (High = good)
  emergency_preparedness  — positive (High = good) | "Unknown" if data missing
  debt_burden             — burden   (High = bad)
  dependency_load         — burden   (High = bad)
  near_term_obligation    — burden   (High = bad)  [replaces cultural_obligation]
  behavioral_risk         — burden   (High = bad)  | "Unknown" if data missing

Changes from v2:
- wedding_flag → near_term_obligation_level (none/moderate/high)
- emergency_preparedness base = 0 (not 20); null → label="Unknown", no flag triggered
- behavioral_risk base = 0 (not 40); null fields → label="Unknown", no flag triggered
- Debt burden EMI thresholds aligned with Axis 3 (scoring.py)
- Balanced Investor: unknown fields do NOT block it (only confirmed HIGH flags block)
- Confidence-aware decision: low confidence → avoid extreme labels
"""


def _clamp(value: float) -> int:
    return max(1, min(100, int(round(value))))


def _label_positive(score: int) -> str:
    if score >= 67:
        return "High"
    elif score >= 34:
        return "Moderate"
    return "Low"


def _label_burden(score: int) -> str:
    if score >= 67:
        return "High"
    elif score >= 34:
        return "Moderate"
    return "Low"


# ---------------------------------------------------------------------------
# Category 1: Income Stability (positive)
# ---------------------------------------------------------------------------

def score_income_stability(data: dict) -> dict:
    score   = 50
    reasons = []

    it = data.get("income_type", "unknown")
    if it == "salaried":
        score += 30
        reasons.append("Salaried income (+30)")
    elif it == "business":
        score += 10
        reasons.append("Business income (+10)")
    elif it == "gig":
        score -= 20
        reasons.append("Gig/freelance income (-20)")
    else:
        score -= 10
        reasons.append("Income type unknown (-10)")

    income = data.get("monthly_income")
    if income is not None:
        if income >= 150_000:
            score += 15
            reasons.append(f"₹{income:,.0f}/mo → high income (+15)")
        elif income >= 75_000:
            score += 8
            reasons.append(f"₹{income:,.0f}/mo → moderate income (+8)")
        elif income >= 30_000:
            score += 2
            reasons.append(f"₹{income:,.0f}/mo → low-moderate (+2)")
        else:
            score -= 10
            reasons.append(f"₹{income:,.0f}/mo → low income (-10)")
    else:
        reasons.append("Monthly income not provided")

    final = _clamp(score)
    return {"label": _label_positive(final), "score": final, "reason": " | ".join(reasons)}


# ---------------------------------------------------------------------------
# Category 2: Emergency Preparedness (positive)
# Base = 0. Null → "Unknown" label, score = None. No flag triggered for unknown.
# ---------------------------------------------------------------------------

def score_emergency_preparedness(data: dict) -> dict:
    em = data.get("emergency_months")

    if em is None:
        return {
            "label":  "Unknown",
            "score":  None,
            "reason": "Emergency fund not reported — cannot assess (not penalized)",
        }

    score   = 0
    reasons = []

    if em >= 6:
        score = 90
        reasons.append(f"{em} months ≥ 6-month benchmark (score: 90)")
    elif em >= 3:
        score = 60
        reasons.append(f"{em} months — adequate (score: 60)")
    elif em >= 1:
        score = 30
        reasons.append(f"{em} months — minimal (score: 30)")
    else:
        score = 5
        reasons.append("Emergency fund = 0 (score: 5)")

    final = _clamp(score)
    return {"label": _label_positive(final), "score": final, "reason": " | ".join(reasons)}


# ---------------------------------------------------------------------------
# Category 3: Debt Burden (burden)
# EMI thresholds aligned with Axis 3 in scoring.py.
# ---------------------------------------------------------------------------

def score_debt_burden(data: dict) -> dict:
    score   = 5
    reasons = []

    emi_ratio = data.get("emi_ratio")
    source    = data.get("emi_ratio_source") or (
        "derived" if data.get("_emi_ratio_derived") else "none"
    )

    if emi_ratio is not None:
        reasons.append(f"EMI ratio {emi_ratio:.1f}% (source: {source})")
        if emi_ratio >= 60:
            score += 80
            reasons.append("≥60% → critically high burden (+80)")
        elif emi_ratio >= 40:
            score += 60
            reasons.append("40–60% → high burden (+60)")
        elif emi_ratio >= 20:
            score += 35
            reasons.append("20–40% → moderate burden (+35)")
        elif emi_ratio > 0:
            score += 12
            reasons.append("<20% → low burden (+12)")
        else:
            reasons.append("EMI ratio = 0 → no debt")
    elif data.get("_incomplete_emi_data"):
        score += 15
        reasons.append("EMI present but income unknown → conservative partial burden (+15)")
    else:
        reasons.append("No EMI data → minimal debt assumed")

    final = _clamp(score)
    return {"label": _label_burden(final), "score": final, "reason": " | ".join(reasons)}


# ---------------------------------------------------------------------------
# Category 4: Dependency Load (burden)
# ---------------------------------------------------------------------------

def score_dependency_load(data: dict) -> dict:
    score   = 5
    reasons = []

    dep = data.get("dependents")
    if dep is not None:
        reasons.append(f"{dep} dependent(s)")
        if dep >= 4:
            score += 75
            reasons.append("4+ dependents → very high load (+75)")
        elif dep >= 3:
            score += 55
            reasons.append("3 dependents → high load (+55)")
        elif dep == 2:
            score += 35
            reasons.append("2 dependents → moderate load (+35)")
        elif dep == 1:
            score += 18
            reasons.append("1 dependent → low-moderate load (+18)")
        else:
            reasons.append("0 dependents → no burden")
    else:
        reasons.append("Dependents not mentioned → no burden assumed")

    final = _clamp(score)
    return {"label": _label_burden(final), "score": final, "reason": " | ".join(reasons)}


# ---------------------------------------------------------------------------
# Category 5: Near-Term Obligation (burden)
# Replaces cultural_obligation / wedding_flag.
# none → 5, moderate → 40, high → 75
# ---------------------------------------------------------------------------

def score_near_term_obligation(data: dict) -> dict:
    ntol    = data.get("near_term_obligation_level", "none")
    otype   = data.get("obligation_type")
    reasons = []

    if ntol == "high":
        score = 75
        label = "High"
        reasons.append(
            f"Near-term obligation: high urgency"
            + (f" ({otype})" if otype else "")
            + " (score: 75)"
        )
    elif ntol == "moderate":
        score = 40
        label = "Moderate"
        reasons.append(
            f"Near-term obligation: moderate"
            + (f" ({otype})" if otype else "")
            + " (score: 40)"
        )
    else:
        score = 5
        label = "Low"
        reasons.append("No near-term obligation detected (score: 5)")

    return {"label": label, "score": score, "reason": " | ".join(reasons)}


# ---------------------------------------------------------------------------
# Category 6: Behavioral Risk (burden)
# Base = 0. If both behavioral fields null → "Unknown", score = None.
# No flag triggered for unknown behavioral data.
# ---------------------------------------------------------------------------

def score_behavioral_risk(data: dict) -> dict:
    rb = data.get("risk_behavior")
    lr = data.get("loss_reaction")

    if rb is None and lr is None:
        return {
            "label":  "Unknown",
            "score":  None,
            "reason": "risk_behavior and loss_reaction not reported — cannot assess (not penalized)",
        }

    score   = 0
    reasons = []

    if rb == "high":
        score += 55
        reasons.append("risk_behavior=high (+55)")
    elif rb == "medium":
        score += 30
        reasons.append("risk_behavior=medium (+30)")
    elif rb == "low":
        score += 5
        reasons.append("risk_behavior=low (+5)")
    else:
        reasons.append("risk_behavior not reported (0)")

    if lr == "panic":
        score += 5
        reasons.append("loss_reaction=panic → actual tolerance low (+5)")
    elif lr == "aggressive":
        score += 35
        reasons.append("loss_reaction=aggressive → high tolerance (+35)")
    elif lr == "neutral":
        reasons.append("loss_reaction=neutral (0)")
    else:
        reasons.append("loss_reaction not reported (0)")

    final = _clamp(score)
    return {"label": _label_burden(final), "score": final, "reason": " | ".join(reasons)}


# ---------------------------------------------------------------------------
# Final Decision Engine
# ---------------------------------------------------------------------------

def compute_final_decision(
    categories: dict,
    financial_capacity: int,
    axis_scores: dict,
    confidence_score: int = 100,
) -> tuple[str, list[str]]:
    """
    Priority:
      1. financial_capacity (PRIMARY — hard rule for < 30)
      2. confirmed HIGH constraint flags (SECONDARY)
      3. behavioral overlay (TERTIARY — blocked for High Constraint)

    Unknown fields (score=None) do NOT trigger constraint flags.
    Low confidence → avoid extreme behavioral labels (Naive Risk Taker).
    """
    debt       = categories["debt_burden"]["score"]
    dep        = categories["dependency_load"]["score"]
    obligation = categories["near_term_obligation"]["score"]
    income     = categories["income_stability"]["score"]
    emergency  = categories["emergency_preparedness"]["score"]   # may be None
    behavioral = categories["behavioral_risk"]["score"]          # may be None
    risk       = axis_scores.get("risk")                         # may be None
    context    = axis_scores.get("context", 30)

    reasoning = []

    # --- Constraint flags — only fire on confirmed (non-None) HIGH scores ---
    flags = set()

    if debt is not None and debt > 65:
        flags.add("high_debt")
        reasoning.append(f"Debt burden high (score: {debt})")

    if dep is not None and dep > 55:
        flags.add("high_dependency")
        reasoning.append(f"Dependency load high (score: {dep})")

    if obligation is not None and obligation > 60:
        flags.add("high_obligation")
        reasoning.append(f"Near-term obligation high (score: {obligation})")

    if income is not None and income < 35:
        flags.add("low_income")
        reasoning.append(f"Income stability low (score: {income})")

    # emergency: only flag if explicitly known to be low (score < 35 AND not None)
    if emergency is not None and emergency < 35:
        flags.add("low_emergency")
        reasoning.append(f"Emergency preparedness low (score: {emergency})")

    # behavioral: only flag if explicitly known to be high
    if behavioral is not None and behavioral > 70:
        flags.add("high_behavioral_risk")
        reasoning.append(f"Behavioral risk high (score: {behavioral})")

    reasoning.append(
        f"Financial capacity: {financial_capacity} = cashflow × (1 - obligation/100)"
    )
    reasoning.append(f"Data confidence score: {confidence_score}%")

    # --- Primary classification ---
    if financial_capacity < 30:
        reasoning.append("Financial capacity < 30 → High Constraint Investor (no override)")
        return "High Constraint Investor", reasoning

    elif financial_capacity < 60:
        base_decision = "Moderate Constraint Investor"
        reasoning.append("Financial capacity 30–59 → Moderate Constraint")
    else:
        base_decision = "Flexible Investor"
        reasoning.append("Financial capacity ≥ 60 → Flexible")

    # --- Behavioral overlay (blocked for High Constraint) ---
    decision = base_decision

    # Low confidence → skip extreme behavioral labels
    low_confidence = confidence_score < 40

    if base_decision == "Flexible Investor":
        if (
            not low_confidence
            and behavioral is not None and behavioral > 70
            and context < 35
        ):
            decision = "Naive Risk Taker"
            reasoning.append("Overlay: high behavioral risk + low context → Naive Risk Taker")
        elif (
            risk is not None and risk >= 65
            and context >= 60
        ):
            decision = "Aggressive Sophisticate"
            reasoning.append("Overlay: high risk + high context → Aggressive Sophisticate")
        elif (
            risk is not None and risk < 35
            and context >= 55
        ):
            decision = "Conservative Analyst"
            reasoning.append("Overlay: low risk + high context → Conservative Analyst")

    elif base_decision == "Moderate Constraint Investor":
        if (
            not low_confidence
            and behavioral is not None and behavioral > 70
            and "low_income" in flags
        ):
            decision = "Naive Risk Taker"
            reasoning.append("Overlay: high behavioral risk + low income → Naive Risk Taker")
        elif not flags:
            # No confirmed constraint flags (unknown fields don't block this)
            decision = "Balanced Investor"
            reasoning.append(
                "Overlay: moderate capacity, no confirmed constraint flags → Balanced Investor"
            )

    if low_confidence:
        reasoning.append(
            f"Note: confidence score {confidence_score}% < 40% — "
            "extreme behavioral labels suppressed; profile marked Low Confidence"
        )

    return decision, reasoning


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_categories(
    normalized_data: dict,
    axis_scores: dict,
    confidence_score: int = 100,
) -> dict:
    """
    Run all 6 category scorers + final decision.
    Expects pre-normalized data (normalize() already called in scoring.py).
    """
    categories = {
        "income_stability":       score_income_stability(normalized_data),
        "emergency_preparedness": score_emergency_preparedness(normalized_data),
        "debt_burden":            score_debt_burden(normalized_data),
        "dependency_load":        score_dependency_load(normalized_data),
        "near_term_obligation":   score_near_term_obligation(normalized_data),
        "behavioral_risk":        score_behavioral_risk(normalized_data),
    }

    final_decision, decision_reasoning = compute_final_decision(
        categories,
        financial_capacity=axis_scores["financial_capacity"],
        axis_scores=axis_scores,
        confidence_score=confidence_score,
    )

    return {
        "categories":         categories,
        "final_decision":     final_decision,
        "decision_reasoning": decision_reasoning,
    }
