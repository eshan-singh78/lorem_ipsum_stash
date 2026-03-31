"""
Context-Driven Axis Scoring Engine — InvestorDNA v6

Architecture:
  CategoryAssessment + ProfileContext → AxisScores

CRITICAL DESIGN RULE:
  No axis reads raw fields directly.
  Every axis score is derived from category assessments.
  Categories are derived from ProfileContext.
  ProfileContext is derived from validated fields.

This enforces the InvestorDNA principle:
  "Think like an advisor, not a calculator."

Axis mapping (from InvestorDNA spec):
  Axis 1 (Risk Appetite)  → behavioral_risk + emotional signals from context
  Axis 2 (Cash Flow)      → income_stability + emergency_preparedness
  Axis 3 (Obligations)    → debt_burden + dependency_load + cultural_obligation
  Axis 4 (Context)        → experience + sophistication + autonomy (from ProfileContext)
"""

from dataclasses import dataclass
from context_categories import CategoryAssessment, CategoryResult
from profile_context import ProfileContext


def _clamp(v: float, lo: int = 1, hi: int = 99) -> int:
    return max(lo, min(hi, int(round(v))))


# ---------------------------------------------------------------------------
# Axis score result
# ---------------------------------------------------------------------------

@dataclass
class AxisScores:
    risk: int | None           # Axis 1: 1–99, None if no behavioral data
    cashflow: int              # Axis 2: 1–99
    obligation: int            # Axis 3: 1–99
    context: int               # Axis 4: 1–99
    financial_capacity: int    # derived: cashflow × (1 - obligation/100)

    # Reasoning traces
    risk_reasons: list[str]
    cashflow_reasons: list[str]
    obligation_reasons: list[str]
    context_reasons: list[str]


# ---------------------------------------------------------------------------
# Axis 1: Risk Appetite
# Source: behavioral_risk category + grief/recency modifiers from context
#
# InvestorDNA insight: a high-sophistication investor in grief state has
# SUPPRESSED risk tolerance — their Axis 1 score reflects current state,
# not baseline. The system must flag this distinction.
# ---------------------------------------------------------------------------

def score_axis1_risk(
    categories: CategoryAssessment,
    ctx: ProfileContext,
) -> tuple[int | None, list[str]]:
    br = categories.behavioral_risk
    reasons = []

    if not br.data_available or br.score is None:
        reasons.append(
            "Behavioral data unavailable — Axis 1 (Risk) cannot be scored. "
            "Advisor should conduct behavioral assessment before recommending products."
        )
        return None, reasons

    # Behavioral risk score is a BURDEN score (high = more risk-seeking / volatile behavior)
    # Axis 1 (Risk Appetite) is a POSITIVE score (high = more willing to take risk)
    # Inversion: risk_appetite = 100 - behavioral_risk_burden
    # BUT: panic/fear signals mean LOW risk appetite despite potentially high behavioral score
    # We use the behavioral_risk score as a proxy for risk tolerance level

    lr = ctx.loss_reaction
    rb = ctx.risk_behavior

    # Start from behavioral risk score as base
    base = br.score or 50

    # Map behavioral risk burden → risk appetite
    # High behavioral risk (risk-seeking) → high risk appetite
    # Low behavioral risk (conservative) → low risk appetite
    # Panic → very low risk appetite regardless
    if lr == "panic":
        # Panic overrides everything — very low risk appetite
        score = _clamp(15 + (base * 0.1))
        reasons.append(
            f"Panic loss reaction detected — risk appetite severely suppressed (score: {score})"
        )
    elif lr == "cautious":
        score = _clamp(25 + (base * 0.2))
        reasons.append(
            f"Cautious loss reaction — moderate-low risk appetite (score: {score})"
        )
    elif lr == "aggressive":
        score = _clamp(50 + (base * 0.4))
        reasons.append(
            f"Aggressive loss reaction (buys on dips) — high risk appetite (score: {score})"
        )
    elif rb == "low":
        score = _clamp(20 + (base * 0.15))
        reasons.append(f"Low risk behavior — conservative risk appetite (score: {score})")
    elif rb == "high":
        score = _clamp(55 + (base * 0.3))
        reasons.append(f"High risk behavior — elevated risk appetite (score: {score})")
    else:
        # Neutral / medium — use behavioral score directly as moderate
        score = _clamp(35 + (base * 0.2))
        reasons.append(f"Moderate behavioral profile — balanced risk appetite (score: {score})")

    # Grief state modifier: amplifies loss aversion, suppresses risk appetite
    if ctx.grief_state:
        grief_penalty = 15
        score = _clamp(score - grief_penalty)
        reasons.append(
            f"Grief state modifier: −{grief_penalty} pts. "
            "Current score reflects grief-amplified loss aversion, not baseline personality. "
            "Expected to increase over 6–18 months as grief processing occurs."
        )

    # Recency bias: high apparent risk appetite from inexperience, not conviction
    if ctx.recency_bias_risk:
        reasons.append(
            "RECENCY BIAS WARNING: High risk appetite score reflects inexperience with real losses, "
            "not informed risk acceptance. Do NOT treat this as stable high risk tolerance."
        )

    # Append behavioral modifiers as context
    for mod in br.modifiers:
        reasons.append(f"Context: {mod}")

    return _clamp(score), reasons


# ---------------------------------------------------------------------------
# Axis 2: Cash Flow Stability
# Source: income_stability + emergency_preparedness categories
# ---------------------------------------------------------------------------

def score_axis2_cashflow(
    categories: CategoryAssessment,
    ctx: ProfileContext,
) -> tuple[int, list[str]]:
    inc = categories.income_stability
    em  = categories.emergency_preparedness
    reasons = []

    # Income stability is the primary driver (weight: 0.65)
    inc_score = inc.score if inc.score is not None else 40
    reasons.append(
        f"Income stability: {inc.label} (score: {inc_score}) — {inc.reason}"
    )

    # Emergency preparedness is the resilience modifier (weight: 0.35)
    if em.data_available and em.score is not None:
        em_score = em.score
        reasons.append(
            f"Emergency preparedness: {em.label} (score: {em_score}) — {em.reason}"
        )
        # Weighted combination
        combined = (inc_score * 0.65) + (em_score * 0.35)
    else:
        # Unknown emergency fund: no penalty, but no bonus
        combined = inc_score * 0.85  # slight discount for unknown resilience
        reasons.append(
            "Emergency fund unknown — no penalty applied, but resilience unconfirmed"
        )

    # Context modifiers
    for mod in inc.modifiers + (em.modifiers if em.data_available else []):
        reasons.append(f"Modifier: {mod}")

    return _clamp(combined), reasons


# ---------------------------------------------------------------------------
# Axis 3: Financial Obligations
# Source: debt_burden + dependency_load + cultural_obligation categories
#
# InvestorDNA insight: Axis 3 is the binding constraint axis.
# Even a high-risk-appetite investor cannot invest what they don't have.
# Cultural obligations are FIXED — they cannot be restructured.
# ---------------------------------------------------------------------------

def score_axis3_obligations(
    categories: CategoryAssessment,
    ctx: ProfileContext,
) -> tuple[int, list[str]]:
    debt = categories.debt_burden
    dep  = categories.dependency_load
    cult = categories.cultural_obligation
    reasons = []

    # Debt burden: formal EMI obligations (weight: 0.40)
    debt_score = debt.score if debt.score is not None else 5
    reasons.append(
        f"Debt burden: {debt.label} (score: {debt_score}) — {debt.reason}"
    )

    # Dependency load: people depending on investor (weight: 0.30)
    dep_score = dep.score if dep.score is not None else 5
    reasons.append(
        f"Dependency load: {dep.label} (score: {dep_score}) — {dep.reason}"
    )

    # Cultural obligation: India-specific fixed obligations (weight: 0.30)
    cult_score = cult.score if cult.score is not None else 5
    reasons.append(
        f"Cultural obligation: {cult.label} (score: {cult_score}) — {cult.reason}"
    )

    # Weighted combination
    combined = (debt_score * 0.40) + (dep_score * 0.30) + (cult_score * 0.30)

    # Context modifiers
    all_modifiers = debt.modifiers + dep.modifiers + cult.modifiers
    for mod in all_modifiers:
        reasons.append(f"Modifier: {mod}")

    # Emerging constraint flag
    if ctx.emerging_constraint:
        reasons.append(
            "EMERGING CONSTRAINT: Current obligations are low but future commitments are high. "
            "Classify as Emerging Constraint Investor — do not recommend illiquid products."
        )

    return _clamp(combined), reasons


# ---------------------------------------------------------------------------
# Axis 4: Investor Context / Sophistication
# Source: ProfileContext directly (experience, knowledge, autonomy, city tier)
#
# InvestorDNA insight: Low loss aversion + low sophistication = naive risk-taking (Explorer),
# NOT informed confidence (Strategist). Context distinguishes between these.
# ---------------------------------------------------------------------------

def score_axis4_context(
    categories: CategoryAssessment,
    ctx: ProfileContext,
) -> tuple[int, list[str]]:
    score = 30
    reasons = []

    # Experience years
    exp = ctx.experience_years
    if exp is not None:
        if exp >= 10:
            score += 30
            reasons.append(f"10+ years investment experience (+30)")
        elif exp >= 5:
            score += 20
            reasons.append(f"{exp:.1f} years experience (+20)")
        elif exp >= 2:
            score += 10
            reasons.append(f"{exp:.1f} years experience (+10)")
        elif exp >= 1:
            score += 5
            reasons.append(f"{exp:.1f} years experience (+5)")
        else:
            reasons.append(f"<1 year experience — novice investor")
    else:
        reasons.append("Experience not reported")

    # Financial knowledge score
    fks = ctx.financial_knowledge_score
    if fks is not None:
        bonus = (fks - 1) * 5
        score += bonus
        reasons.append(f"Financial knowledge score: {fks}/5 (+{bonus})")
    else:
        reasons.append("Financial knowledge not assessed")

    # Decision autonomy
    da = ctx.decision_autonomy
    if da is True:
        score += 10
        reasons.append("Independent decision-making (+10)")
    elif da is False:
        score -= 5
        reasons.append("Peer/family-influenced decisions (−5)")

    # City tier modifier (proxy for financial ecosystem access)
    if ctx.city_tier == "metro":
        score += 5
        reasons.append("Metro city — better access to financial products and advice (+5)")
    elif ctx.city_tier == "tier2":
        reasons.append("Tier 2 city — standard access")

    # Peer-driven flag: reduces effective sophistication
    if ctx.peer_driven:
        score = max(1, score - 8)
        reasons.append(
            "Peer-driven investment decisions reduce effective sophistication score (−8). "
            "Investor may not have internalized the reasoning behind their portfolio."
        )

    # Fragmentation risk: high sophistication but fragmented picture
    if ctx.fragmentation_risk:
        reasons.append(
            "FRAGMENTATION RISK: Multiple advisors/platforms detected — "
            "no single advisor has the complete financial picture. "
            "This is the advisor's key value-add opportunity."
        )

    return _clamp(score), reasons


# ---------------------------------------------------------------------------
# Financial Capacity (derived from Axis 2 + Axis 3)
# ---------------------------------------------------------------------------

def compute_financial_capacity(cashflow: int, obligation: int) -> int:
    """
    Financial capacity = how much of the investor's cash flow is actually
    available for investment after obligations are met.

    Formula: cashflow × (1 - obligation/100)
    Range: 1–99
    """
    return _clamp(cashflow * (1 - obligation / 100))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_axis_scores(
    categories: CategoryAssessment,
    ctx: ProfileContext,
) -> AxisScores:
    """
    Score all 4 axes from category assessments and profile context.
    No raw field access — everything flows through categories and context.
    """
    risk_score,    risk_reasons    = score_axis1_risk(categories, ctx)
    cashflow_score, cf_reasons     = score_axis2_cashflow(categories, ctx)
    oblig_score,   oblig_reasons   = score_axis3_obligations(categories, ctx)
    context_score, context_reasons = score_axis4_context(categories, ctx)

    capacity = compute_financial_capacity(cashflow_score, oblig_score)

    return AxisScores(
        risk=risk_score,
        cashflow=cashflow_score,
        obligation=oblig_score,
        context=context_score,
        financial_capacity=capacity,
        risk_reasons=risk_reasons,
        cashflow_reasons=cf_reasons,
        obligation_reasons=oblig_reasons,
        context_reasons=context_reasons,
    )


def axis_scores_to_dict(scores: AxisScores) -> dict:
    return {
        "risk":               scores.risk,
        "cashflow":           scores.cashflow,
        "obligation":         scores.obligation,
        "context":            scores.context,
        "financial_capacity": scores.financial_capacity,
        "reasons": {
            "risk":       scores.risk_reasons,
            "cashflow":   scores.cashflow_reasons,
            "obligation": scores.obligation_reasons,
            "context":    scores.context_reasons,
        },
    }
