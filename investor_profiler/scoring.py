"""
4-Axis Scoring Engine — v3, deterministic, no LLM.
Scores: 1–99.

Changes from v2:
- wedding_flag replaced by near_term_obligation_level in Axis 3
- emergency_months=null → NO penalty (unknown ≠ low)
- behavioral fields missing → score returns None, not a default
- Axis 3 and debt_burden category now use identical EMI thresholds
- Confidence used in decision engine via confidence_score passthrough
- knowledge_score=3 low-confidence → already nulled upstream; no special case needed here
"""


def _clamp(value: float, lo: int = 1, hi: int = 99) -> int:
    return max(lo, min(hi, int(round(value))))


# ---------------------------------------------------------------------------
# Normalization — called ONCE in compute_scores()
# ---------------------------------------------------------------------------

def normalize(data: dict) -> dict:
    """
    Canonical normalization. Returns new dict — does not mutate input.

    emi_ratio logic:
      IF emi_amount AND monthly_income both present → always recompute (authoritative)
      ELIF emi_ratio already present (LLM-provided)  → keep, source="llm"
      ELSE                                            → None, source="none"
    """
    n = dict(data)

    emi    = n.get("emi_amount")
    income = n.get("monthly_income")

    if emi is not None and income and income > 0:
        n["emi_ratio"]        = round(min((emi / income) * 100, 100), 2)
        n["emi_ratio_source"] = "derived"
    elif n.get("emi_ratio") is not None:
        n["emi_ratio_source"] = "llm"
    else:
        n["emi_ratio_source"] = "none"

    n["_emi_ratio_derived"]  = (n["emi_ratio_source"] == "derived")
    n["_incomplete_emi_data"] = (
        emi is not None and (income is None or income <= 0)
    )

    if n.get("income_type") in (None, "unknown"):
        n["income_type"] = "unknown"

    # Normalize obligation level: treat null as "none" for scoring
    ntol = n.get("near_term_obligation_level")
    if ntol not in ("moderate", "high"):
        n["near_term_obligation_level"] = "none"

    return n


# ---------------------------------------------------------------------------
# Consistency checks (pre-scoring)
# ---------------------------------------------------------------------------

def run_consistency_checks(data: dict) -> tuple[dict, list[str]]:
    """
    Detect and auto-correct logical inconsistencies before scoring.
    Returns (corrected_data, list_of_flags).
    """
    d     = dict(data)
    flags = []

    exp = d.get("experience_years")
    fks = d.get("financial_knowledge_score")
    if exp is not None and fks is not None:
        if exp >= 5 and fks <= 2:
            flags.append(
                f"Consistency: experience_years={exp} but knowledge={fks} — raised to 3"
            )
            d["financial_knowledge_score"] = 3
        elif exp < 1 and fks > 2:
            flags.append(
                f"Consistency: experience_years={exp} < 1 but knowledge={fks} — capped to 2"
            )
            d["financial_knowledge_score"] = 2

    rb = d.get("risk_behavior")
    lr = d.get("loss_reaction")
    if rb == "high" and lr == "panic":
        flags.append(
            "Consistency: risk_behavior=high conflicts with loss_reaction=panic — corrected to 'low'"
        )
        d["risk_behavior"] = "low"

    if d.get("_incomplete_emi_data"):
        flags.append(
            "Consistency: emi_amount present but monthly_income missing "
            "— emi_ratio cannot be derived; debt burden may be underestimated"
        )

    return d, flags


# ---------------------------------------------------------------------------
# Confidence weighting
# ---------------------------------------------------------------------------

def _apply_confidence(delta: float, confidence: str) -> float:
    if confidence == "low":
        return delta * 0.5
    if confidence == "medium":
        return delta * 0.85
    return delta


# ---------------------------------------------------------------------------
# Axis 1: Risk Appetite (1–99)
# Returns (score, reasons) — score is None if no behavioral data present
# ---------------------------------------------------------------------------

def score_risk(data: dict, confidences: dict | None = None) -> tuple[int | None, list[str]]:
    conf    = confidences or {}
    reasons = []

    rb = data.get("risk_behavior")
    lr = data.get("loss_reaction")

    # If both behavioral fields are missing → return None (unknown, not penalized)
    if rb is None and lr is None:
        reasons.append("risk_behavior and loss_reaction both unknown — risk axis not scored")
        return None, reasons

    score = 50

    if rb == "low":
        delta = _apply_confidence(-25, conf.get("risk_behavior", "high"))
        score += delta
        reasons.append(f"risk_behavior=low ({delta:+.1f})")
    elif rb == "high":
        delta = _apply_confidence(25, conf.get("risk_behavior", "high"))
        score += delta
        reasons.append(f"risk_behavior=high ({delta:+.1f})")
    elif rb == "medium":
        reasons.append("risk_behavior=medium (0)")

    if lr == "panic":
        delta = _apply_confidence(-20, conf.get("loss_reaction", "high"))
        score += delta
        reasons.append(f"loss_reaction=panic ({delta:+.1f})")
    elif lr == "aggressive":
        delta = _apply_confidence(20, conf.get("loss_reaction", "high"))
        score += delta
        reasons.append(f"loss_reaction=aggressive ({delta:+.1f})")
    elif lr == "neutral":
        reasons.append("loss_reaction=neutral (0)")

    return _clamp(score), reasons


# ---------------------------------------------------------------------------
# Axis 2: Cash Flow Stability (1–99)
# emergency_months=null → no penalty (unknown ≠ bad)
# ---------------------------------------------------------------------------

def score_cashflow(data: dict, confidences: dict | None = None) -> tuple[int, list[str]]:
    score   = 50
    reasons = []
    conf    = confidences or {}

    it      = data.get("income_type")
    it_conf = conf.get("income_type", "high")

    if it == "salaried":
        delta = _apply_confidence(20, it_conf)
        score += delta
        reasons.append(f"salaried income ({delta:+.1f})")
    elif it == "business":
        delta = _apply_confidence(5, it_conf)
        score += delta
        reasons.append(f"business income ({delta:+.1f})")
    elif it == "gig":
        income = data.get("monthly_income")
        if income and income >= 150_000:
            delta = _apply_confidence(-5, it_conf)
            reasons.append(f"gig income, high earner ({delta:+.1f})")
        else:
            delta = _apply_confidence(-15, it_conf)
            reasons.append(f"gig income ({delta:+.1f})")
        score += delta
    else:
        delta = _apply_confidence(-10, it_conf)
        score += delta
        reasons.append(f"income_type unknown ({delta:+.1f})")

    em = data.get("emergency_months")
    if em is not None:
        if em >= 6:
            score += 20
            reasons.append("emergency_months ≥ 6 (+20)")
        elif em >= 3:
            score += 10
            reasons.append(f"emergency_months={em} (+10)")
        elif em >= 1:
            reasons.append(f"emergency_months={em} (0 — minimal but present)")
        else:
            score -= 15
            reasons.append("emergency_months=0 (-15)")
    else:
        # Unknown ≠ bad — no penalty
        reasons.append("emergency_months not reported — no adjustment (unknown ≠ low)")

    return _clamp(score), reasons


# ---------------------------------------------------------------------------
# Axis 3: Financial Obligations (1–99)
# EMI thresholds now match debt_burden category exactly.
# near_term_obligation_level replaces wedding_flag.
# ---------------------------------------------------------------------------

def score_obligations(data: dict) -> tuple[int, list[str]]:
    score   = 10
    reasons = []

    emi_ratio = data.get("emi_ratio")
    source    = data.get("emi_ratio_source", "none")

    if emi_ratio is not None:
        reasons.append(f"emi_ratio={emi_ratio:.1f}% (source: {source})")
        if emi_ratio >= 60:
            score += 55
            reasons.append("emi_ratio ≥ 60% → critical burden (+55)")
        elif emi_ratio >= 40:
            score += 40
            reasons.append("emi_ratio 40–60% → high burden (+40)")
        elif emi_ratio >= 20:
            score += 25
            reasons.append("emi_ratio 20–40% → moderate burden (+25)")
        elif emi_ratio > 0:
            score += 10
            reasons.append("emi_ratio < 20% → low burden (+10)")
    elif data.get("_incomplete_emi_data"):
        score += 15
        reasons.append("emi_amount present but income unknown → conservative partial burden (+15)")

    dep = data.get("dependents")
    if dep is not None:
        if dep >= 4:
            score += 25
            reasons.append(f"{dep} dependents → very high (+25)")
        elif dep >= 2:
            score += 15
            reasons.append(f"{dep} dependents → moderate (+15)")
        elif dep == 1:
            score += 8
            reasons.append("1 dependent (+8)")

    ntol = data.get("near_term_obligation_level")
    if ntol == "high":
        score += 25
        reasons.append(f"near_term_obligation=high (+25)")
    elif ntol == "moderate":
        score += 12
        reasons.append(f"near_term_obligation=moderate (+12)")

    return _clamp(score), reasons


# ---------------------------------------------------------------------------
# Axis 4: Investor Context / Sophistication (1–99)
# ---------------------------------------------------------------------------

def score_context(data: dict, confidences: dict | None = None) -> tuple[int, list[str]]:
    score   = 30
    reasons = []
    conf    = confidences or {}

    exp = data.get("experience_years")
    if exp is not None:
        if exp >= 10:
            score += 30
            reasons.append(f"experience={exp}y (+30)")
        elif exp >= 5:
            score += 20
            reasons.append(f"experience={exp}y (+20)")
        elif exp >= 2:
            score += 10
            reasons.append(f"experience={exp}y (+10)")
        elif exp >= 1:
            score += 5
            reasons.append(f"experience={exp}y (+5)")
        else:
            reasons.append(f"experience={exp}y (<1, no bonus)")

    fks = data.get("financial_knowledge_score")
    if fks is not None:
        bonus = (fks - 1) * 5
        score += bonus
        reasons.append(f"knowledge_score={fks} (+{bonus})")

    da      = data.get("decision_autonomy")
    da_conf = conf.get("decision_autonomy", "high")
    if da is True:
        delta = _apply_confidence(10, da_conf)
        score += delta
        reasons.append(f"decision_autonomy=true ({delta:+.1f})")
    elif da is False:
        delta = _apply_confidence(-5, da_conf)
        score += delta
        reasons.append(f"decision_autonomy=false ({delta:+.1f})")

    return _clamp(score), reasons


# ---------------------------------------------------------------------------
# Financial Capacity
# ---------------------------------------------------------------------------

def compute_financial_capacity(cashflow: int, obligation: int) -> int:
    """cashflow × (1 - obligation/100). Range: 1–99."""
    return _clamp(cashflow * (1 - obligation / 100))


# ---------------------------------------------------------------------------
# Confidence score (0–100) — used by decision engine
# ---------------------------------------------------------------------------

def compute_confidence_score(confidences: dict, validated: dict) -> int:
    """
    Aggregate confidence across key fields.
    high=1.0, medium=0.6, low=0.3, missing=0.0
    """
    key_fields = [
        "income_type", "monthly_income", "emergency_months",
        "emi_amount", "dependents", "experience_years",
        "financial_knowledge_score", "loss_reaction", "risk_behavior",
        "near_term_obligation_level",
    ]
    weight_map = {"high": 1.0, "medium": 0.6, "low": 0.3}
    total = 0.0
    for f in key_fields:
        val  = validated.get(f)
        conf = confidences.get(f, "high")
        if val is None or val == "unknown":
            total += 0.0   # missing field
        else:
            total += weight_map.get(conf, 0.5)

    return int(round((total / len(key_fields)) * 100))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_scores(
    validated_data: dict,
    confidences: dict | None = None,
) -> tuple[dict, dict, dict]:
    """
    Normalize → consistency check → score all 4 axes.

    Returns:
        (axis_scores, normalized_data, debug_info)
    """
    data = normalize(validated_data)
    data, consistency_flags = run_consistency_checks(data)

    conf = confidences or {}

    risk_score,       risk_reasons     = score_risk(data, conf)
    cashflow_score,   cashflow_reasons = score_cashflow(data, conf)
    obligation_score, oblig_reasons    = score_obligations(data)
    context_score,    context_reasons  = score_context(data, conf)

    # Use cashflow=50 as neutral fallback if somehow None (shouldn't happen)
    capacity = compute_financial_capacity(
        cashflow_score or 50,
        obligation_score,
    )

    confidence_score = compute_confidence_score(conf, validated_data)

    derived_fields = []
    if data["emi_ratio_source"] == "derived":
        derived_fields.append(
            f"emi_ratio={data['emi_ratio']:.2f}% derived from "
            f"emi_amount({data.get('emi_amount')}) / monthly_income({data.get('monthly_income')}) × 100"
        )
    if data.get("_incomplete_emi_data"):
        derived_fields.append(
            "WARNING: emi_amount present but monthly_income missing — ratio unverifiable"
        )
    derived_fields.append(
        f"financial_capacity={capacity} = cashflow({cashflow_score}) × "
        f"(1 - obligation({obligation_score})/100)"
    )

    axis_scores = {
        "risk":               risk_score,       # may be None if no behavioral data
        "cashflow":           cashflow_score,
        "obligation":         obligation_score,
        "context":            context_score,
        "financial_capacity": capacity,
    }

    debug_info = {
        "consistency_flags": consistency_flags,
        "derived_fields":    derived_fields,
        "confidence_score":  confidence_score,
        "axis_reasons": {
            "risk":       risk_reasons,
            "cashflow":   cashflow_reasons,
            "obligation": oblig_reasons,
            "context":    context_reasons,
        },
    }

    return axis_scores, data, debug_info
