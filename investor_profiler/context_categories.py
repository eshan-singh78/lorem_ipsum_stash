"""
Context-Driven Category Assessment — InvestorDNA v6

Architecture:
  ProfileContext → CategoryAssessment

Each category reads ONLY from ProfileContext — never from raw fields.
Each category produces: { score, label, reason, modifiers }

Categories:
  1. income_stability       — how predictable and resilient is income?
  2. emergency_preparedness — buffer against shocks
  3. debt_burden            — formal debt load
  4. dependency_load        — people depending on this investor
  5. cultural_obligation    — India-specific: family role, hidden obligations, religious giving
  6. behavioral_risk        — emotional and cognitive risk patterns

The cultural_obligation category is the InvestorDNA moat —
no Western instrument captures joint family obligations, wedding savings,
or religious giving. This is what separates psychological risk tolerance
from financial capacity to bear losses.
"""

from dataclasses import dataclass
from typing import Any
from profile_context import ProfileContext


# ---------------------------------------------------------------------------
# Category result structure
# ---------------------------------------------------------------------------

@dataclass
class CategoryResult:
    score: int | None          # 1–100, or None if data unavailable
    label: str                 # "low" | "moderate" | "high" | "unknown"
    reason: str                # human-readable explanation for advisor
    modifiers: list[str]       # context-driven adjustments applied
    data_available: bool       # False = cannot assess, not penalized


def _clamp(v: float) -> int:
    return max(1, min(100, int(round(v))))


def _label_positive(score: int) -> str:
    """Higher = better (income, emergency)."""
    if score >= 67:
        return "high"
    if score >= 34:
        return "moderate"
    return "low"


def _label_burden(score: int) -> str:
    """Higher = worse (debt, dependency, obligation, behavioral risk)."""
    if score >= 67:
        return "high"
    if score >= 34:
        return "moderate"
    return "low"


# ---------------------------------------------------------------------------
# Category 1: Income Stability
# Reads: income_type, monthly_income, employment_stability, life_events, flags
# ---------------------------------------------------------------------------

def assess_income_stability(ctx: ProfileContext) -> CategoryResult:
    score = 50
    reasons = []
    modifiers = []

    # Base from income type
    if ctx.income_type == "salaried":
        score += 25
        reasons.append("Salaried employment provides predictable monthly income")
    elif ctx.income_type == "business":
        score += 5
        reasons.append("Business income — moderate predictability, cyclical risk")
    elif ctx.income_type == "gig":
        score -= 20
        reasons.append("Gig/freelance income — high volatility, no employer safety net")
    else:
        score -= 10
        reasons.append("Income type unclear — conservative assessment applied")

    # Income level modifier
    if ctx.monthly_income is not None:
        if ctx.monthly_income >= 150_000:
            score += 15
            reasons.append(f"High income (₹{ctx.monthly_income:,.0f}/mo) provides buffer")
        elif ctx.monthly_income >= 75_000:
            score += 8
            reasons.append(f"Moderate income (₹{ctx.monthly_income:,.0f}/mo)")
        elif ctx.monthly_income >= 30_000:
            score += 2
            reasons.append(f"Low-moderate income (₹{ctx.monthly_income:,.0f}/mo)")
        else:
            score -= 10
            reasons.append(f"Low income (₹{ctx.monthly_income:,.0f}/mo) limits resilience")

    # Life event modifier: recent job change or crisis reduces stability
    for ev in ctx.life_events:
        if ev.event_type == "job_change" and ev.recency == "recent":
            score -= 10
            modifiers.append("Recent job change reduces income stability assessment")
        if ev.event_type == "crisis":
            score -= 8
            modifiers.append("Recent financial/medical crisis impacts income resilience")

    # Responsibility shift: single earner for multiple dependents
    for sig in ctx.cultural_signals:
        if sig.signal_type == "family_role" and "sole_earner" in sig.description:
            score -= 8
            modifiers.append("Sole earner for household — income disruption has amplified impact")

    final = _clamp(score)
    return CategoryResult(
        score=final,
        label=_label_positive(final),
        reason=" | ".join(reasons),
        modifiers=modifiers,
        data_available=ctx.income_type != "unknown" or ctx.monthly_income is not None,
    )


# ---------------------------------------------------------------------------
# Category 2: Emergency Preparedness
# Reads: emergency_months, life_events (crisis depletes buffer)
# ---------------------------------------------------------------------------

def assess_emergency_preparedness(ctx: ProfileContext) -> CategoryResult:
    if ctx.emergency_months is None:
        return CategoryResult(
            score=None,
            label="unknown",
            reason="Emergency fund not reported — cannot assess (not penalized)",
            modifiers=[],
            data_available=False,
        )

    score = 0
    reasons = []
    modifiers = []

    if ctx.emergency_months >= 6:
        score = 90
        reasons.append(f"{ctx.emergency_months:.1f} months reserve meets 6-month benchmark")
    elif ctx.emergency_months >= 3:
        score = 60
        reasons.append(f"{ctx.emergency_months:.1f} months reserve — adequate but below ideal")
    elif ctx.emergency_months >= 1:
        score = 30
        reasons.append(f"{ctx.emergency_months:.1f} months reserve — minimal, vulnerable to shocks")
    else:
        score = 5
        reasons.append("No emergency fund — maximum vulnerability to income disruption")

    # Crisis depletes buffer
    for ev in ctx.life_events:
        if ev.event_type in ("crisis", "death") and ev.recency == "recent":
            score = max(5, score - 20)
            modifiers.append(
                f"Recent {ev.event_type} likely depleted emergency reserves — "
                "stated buffer may be overstated"
            )

    final = _clamp(score)
    return CategoryResult(
        score=final,
        label=_label_positive(final),
        reason=" | ".join(reasons),
        modifiers=modifiers,
        data_available=True,
    )


# ---------------------------------------------------------------------------
# Category 3: Debt Burden
# Reads: emi_ratio, emi_amount, monthly_income
# ---------------------------------------------------------------------------

def assess_debt_burden(ctx: ProfileContext) -> CategoryResult:
    score = 5
    reasons = []
    modifiers = []

    if ctx.emi_ratio is not None:
        if ctx.emi_ratio >= 60:
            score += 80
            reasons.append(
                f"EMI ratio {ctx.emi_ratio:.1f}% — critically high; "
                "less than 40% of income available for living + investing"
            )
        elif ctx.emi_ratio >= 40:
            score += 60
            reasons.append(
                f"EMI ratio {ctx.emi_ratio:.1f}% — high debt load; "
                "significantly constrains investable surplus"
            )
        elif ctx.emi_ratio >= 20:
            score += 35
            reasons.append(
                f"EMI ratio {ctx.emi_ratio:.1f}% — moderate debt; "
                "manageable but limits flexibility"
            )
        elif ctx.emi_ratio > 0:
            score += 12
            reasons.append(
                f"EMI ratio {ctx.emi_ratio:.1f}% — low debt burden"
            )
        else:
            reasons.append("No EMI obligations — debt-free")
    elif ctx.emi_amount is not None and ctx.monthly_income is None:
        score += 15
        reasons.append(
            "EMI present but income unknown — conservative partial burden applied"
        )
        modifiers.append("Cannot compute exact EMI ratio without income data")
    else:
        reasons.append("No debt data — minimal burden assumed")

    final = _clamp(score)
    return CategoryResult(
        score=final,
        label=_label_burden(final),
        reason=" | ".join(reasons),
        modifiers=modifiers,
        data_available=ctx.emi_ratio is not None or ctx.emi_amount is not None,
    )


# ---------------------------------------------------------------------------
# Category 4: Dependency Load
# Reads: dependents, life_events (responsibility shift), cultural_signals
# ---------------------------------------------------------------------------

def assess_dependency_load(ctx: ProfileContext) -> CategoryResult:
    score = 5
    reasons = []
    modifiers = []

    dep = ctx.dependents
    if dep is not None:
        if dep >= 4:
            score += 75
            reasons.append(f"{dep} dependents — very high load; significant income committed to others")
        elif dep >= 3:
            score += 55
            reasons.append(f"{dep} dependents — high load")
        elif dep == 2:
            score += 35
            reasons.append(f"{dep} dependents — moderate load")
        elif dep == 1:
            score += 18
            reasons.append(f"1 dependent — low-moderate load")
        else:
            reasons.append("No dependents — no dependency burden")
    else:
        reasons.append("Dependents not mentioned — no burden assumed")

    # Responsibility shift amplifies dependency impact
    for ev in ctx.life_events:
        if ev.event_type in ("death", "responsibility_shift") and ev.recency == "recent":
            if dep and dep > 0:
                score = min(100, score + 15)
                modifiers.append(
                    f"Recent {ev.event_type} has increased dependency burden — "
                    "previously shared responsibilities now fall on this investor alone"
                )

    # Cultural: joint family obligations not captured in 'dependents' count
    for sig in ctx.cultural_signals:
        if sig.signal_type == "family_role":
            if dep is None or dep == 0:
                score = max(score, 25)
                modifiers.append(
                    "Joint family / family role detected — actual dependency burden "
                    "likely higher than stated dependents count"
                )

    final = _clamp(score)
    return CategoryResult(
        score=final,
        label=_label_burden(final),
        reason=" | ".join(reasons),
        modifiers=modifiers,
        data_available=dep is not None,
    )


# ---------------------------------------------------------------------------
# Category 5: Cultural Obligation
# THE INDIA MOAT — no Western instrument captures this.
# Reads: cultural_signals, near_term_obligation_level, obligation_type,
#        future_obligation_score, life_events
# ---------------------------------------------------------------------------

def assess_cultural_obligation(ctx: ProfileContext) -> CategoryResult:
    """
    Captures what standard risk profiling misses entirely:
    - Joint family financial obligations
    - Wedding/dowry savings (often hidden)
    - Religious/charitable giving (dharmic, non-negotiable)
    - Elder care (parents, in-laws)
    - Social pressure obligations (community expectations)

    These are FIXED obligations — the advisor cannot restructure them.
    They must be treated as binding constraints on investable surplus.
    """
    score = 5
    reasons = []
    modifiers = []

    # Near-term obligation (wedding, house, education, medical)
    ntol = ctx.near_term_obligation_level
    if ntol == "high":
        score += 40
        otype = f" ({ctx.obligation_type})" if ctx.obligation_type else ""
        reasons.append(
            f"High near-term obligation{otype} — significant capital required imminently"
        )
    elif ntol == "moderate":
        score += 20
        otype = f" ({ctx.obligation_type})" if ctx.obligation_type else ""
        reasons.append(
            f"Moderate near-term obligation{otype} — capital commitment expected within 1–2 years"
        )

    # Cultural signals
    for sig in ctx.cultural_signals:
        if sig.signal_type == "hidden_obligation":
            score += 20
            reasons.append(
                "Hidden financial obligation detected — investable surplus is lower than income suggests"
            )
            modifiers.append(
                "Advisor insight: open conversation about goal-based investing for this milestone"
            )
        elif sig.signal_type == "religious":
            score += 10
            reasons.append(
                "Religious/charitable giving commitment — fixed, non-negotiable outflow"
            )
            modifiers.append("Religious giving is culturally fixed — do not suggest reducing it")
        elif sig.signal_type == "social_pressure":
            score += 10
            reasons.append(
                "Social/community financial expectations — implicit obligation burden"
            )
        elif sig.signal_type == "family_role":
            score += 15
            reasons.append(
                f"Family role obligation ({sig.description}) — "
                "financial responsibility extends beyond nuclear family"
            )

    # Future obligation score (from intent detection)
    if ctx.future_obligation_score > 0:
        weighted = round(ctx.future_obligation_score * 0.7, 1)
        score += weighted
        reasons.append(
            f"Future financial commitment detected — adds {weighted:.0f} pts to obligation burden"
        )

    # Life events that create cultural obligations
    for ev in ctx.life_events:
        if ev.event_type == "death" and ev.recency == "recent":
            score += 10
            modifiers.append(
                "Recent bereavement may carry cultural financial obligations "
                "(last rites, ongoing religious commitments, family support)"
            )

    if not reasons:
        reasons.append("No cultural or near-term obligations detected")

    final = _clamp(score)
    return CategoryResult(
        score=final,
        label=_label_burden(final),
        reason=" | ".join(reasons),
        modifiers=modifiers,
        data_available=True,
    )


# ---------------------------------------------------------------------------
# Category 6: Behavioral Risk
# Reads: loss_reaction, risk_behavior, behavioral_signals, flags
# This is where emotional state and cognitive biases are assessed.
# ---------------------------------------------------------------------------

def assess_behavioral_risk(ctx: ProfileContext) -> CategoryResult:
    """
    Behavioral risk is NOT just loss_reaction + risk_behavior.
    It integrates:
    - Emotional state (grief amplifies loss aversion)
    - Cognitive biases (recency bias, overconfidence, peer influence)
    - Decision quality (analytical vs. impulsive)

    A high-sophistication investor in grief state is NOT the same as
    a low-sophistication investor with the same loss_reaction score.
    Context is everything.
    """
    rb = ctx.risk_behavior
    lr = ctx.loss_reaction

    if rb is None and lr is None and not ctx.behavioral_signals:
        return CategoryResult(
            score=None,
            label="unknown",
            reason="No behavioral data available — cannot assess (not penalized)",
            modifiers=[],
            data_available=False,
        )

    score = 0
    reasons = []
    modifiers = []

    # Base from risk_behavior
    if rb == "high":
        score += 55
        reasons.append("High risk-seeking behavior reported")
    elif rb == "medium":
        score += 30
        reasons.append("Moderate risk behavior")
    elif rb == "low":
        score += 5
        reasons.append("Low risk behavior — conservative orientation")

    # Loss reaction overlay
    if lr == "panic":
        score += 5
        reasons.append(
            "Panic response to losses — actual risk tolerance is very low "
            "regardless of stated preference"
        )
    elif lr == "cautious":
        score += 3
        reasons.append("Cautious response to losses — moderate concern")
    elif lr == "aggressive":
        score += 35
        reasons.append("Aggressive response to losses — buys more on dips")
    elif lr == "neutral":
        reasons.append("Neutral loss response — stable emotional baseline")

    # Behavioral signal modifiers
    for sig in ctx.behavioral_signals:
        if sig.signal_type == "fear" and sig.strength == "strong":
            score = max(score, 10)
            modifiers.append(
                "Strong fear/panic signal detected in text — "
                "behavioral risk score floored at 10 regardless of stated preference"
            )
        elif sig.signal_type == "anxiety":
            score = min(100, score + 5)
            modifiers.append(
                "Anxiety-driven monitoring behavior (checking prices obsessively) "
                "indicates higher emotional reactivity than scores suggest"
            )
        elif sig.signal_type == "peer_influence":
            score = min(100, score + 10)
            modifiers.append(
                "Peer-influenced decisions increase behavioral risk — "
                "investor may chase returns or panic-sell based on social signals"
            )
        elif sig.signal_type == "overconfidence":
            score = min(100, score + 15)
            modifiers.append(
                "Overconfidence detected — investor may underestimate downside risk"
            )
        elif sig.signal_type == "analytical":
            score = max(1, score - 10)
            modifiers.append(
                "Analytical decision-making pattern reduces behavioral risk — "
                "investor responds to data, not emotion"
            )

    # Grief state amplification: grief suppresses risk tolerance
    # A high-sophistication investor in grief is NOT a true low-risk investor —
    # their score is temporarily elevated, not baseline
    if ctx.grief_state:
        if rb in ("medium", "high") or lr in ("neutral", "aggressive"):
            score = min(100, score + 15)
            modifiers.append(
                "GRIEF STATE MODIFIER: Recent bereavement amplifies loss aversion by est. 30–40%. "
                "Current behavioral risk score reflects grief-suppressed state, not baseline. "
                "Reassess in 6–12 months."
            )

    # Recency bias: low experience + high risk behavior = naive, not informed
    if ctx.recency_bias_risk:
        modifiers.append(
            "RECENCY BIAS FLAG: High risk tolerance with <1yr experience and low sophistication "
            "suggests recency bias from recent gains, not stable risk acceptance. "
            "First real loss may trigger panic exit."
        )

    final = _clamp(score)
    return CategoryResult(
        score=final,
        label=_label_burden(final),
        reason=" | ".join(reasons),
        modifiers=modifiers,
        data_available=True,
    )


# ---------------------------------------------------------------------------
# Category bundle
# ---------------------------------------------------------------------------

@dataclass
class CategoryAssessment:
    income_stability: CategoryResult
    emergency_preparedness: CategoryResult
    debt_burden: CategoryResult
    dependency_load: CategoryResult
    cultural_obligation: CategoryResult
    behavioral_risk: CategoryResult


def assess_all_categories(ctx: ProfileContext) -> CategoryAssessment:
    """Run all 6 category assessors against the ProfileContext."""
    return CategoryAssessment(
        income_stability=assess_income_stability(ctx),
        emergency_preparedness=assess_emergency_preparedness(ctx),
        debt_burden=assess_debt_burden(ctx),
        dependency_load=assess_dependency_load(ctx),
        cultural_obligation=assess_cultural_obligation(ctx),
        behavioral_risk=assess_behavioral_risk(ctx),
    )


def categories_to_dict(ca: CategoryAssessment) -> dict:
    """Serialize CategoryAssessment to JSON-safe dict."""
    def _result(r: CategoryResult) -> dict:
        return {
            "score": r.score,
            "label": r.label,
            "reason": r.reason,
            "modifiers": r.modifiers,
            "data_available": r.data_available,
        }
    return {
        "income_stability":       _result(ca.income_stability),
        "emergency_preparedness": _result(ca.emergency_preparedness),
        "debt_burden":            _result(ca.debt_burden),
        "dependency_load":        _result(ca.dependency_load),
        "cultural_obligation":    _result(ca.cultural_obligation),
        "behavioral_risk":        _result(ca.behavioral_risk),
    }
