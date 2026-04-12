"""
Context-Driven Category Assessment — InvestorDNA v12

Architecture:
  ProfileContext + InvestorState → CategoryAssessment

v12 change: categories read compound_state from InvestorState,
not raw flags (grief_state, peer_driven, etc.) directly.
The state captures the INTERACTION of signals, not their individual values.

Each category produces: { score, label, reason, modifiers }
"""

from dataclasses import dataclass
from typing import Any
import json
import re
from profile_context import ProfileContext

from llm_adapter import llm_call


_WEIGHT_QUERY_PROMPT = """You are a financial advisor calibrating scoring weights for an investor.

Given the investor's synthesized state below, answer one question:

How constrained is this investor's financial and emotional situation?
Rate each dimension from 1 to 100 where higher = more constrained/burdened.

- obligation_weight: how much should obligation burden be weighted? (1-100)
- behavioral_weight: how much should emotional/behavioral risk be weighted? (1-100)
- income_weight: how much should income stability matter? (1-100)
- reasoning: one sentence explaining your calibration

Return ONLY valid JSON:

{{
  "obligation_weight": integer 1-100,
  "behavioral_weight": integer 1-100,
  "income_weight": integer 1-100,
  "reasoning": "one sentence"
}}

No markdown. JSON only.

Investor state: {compound_state}
State description: {state_description}
Dominant factors: {dominant_factors}
State implications: {state_implications}
"""


def _query_state_weights(investor_state) -> dict:
    """
    Ask LLM: given this investor state, how should scoring weights be calibrated?
    Reads compound_state — not raw flags or individual narrative fields.
    """
    if investor_state is None:
        return {"obligation_weight": 50, "behavioral_weight": 50, "income_weight": 50, "reasoning": "defaults used"}

    prompt = _WEIGHT_QUERY_PROMPT.format(
        compound_state=getattr(investor_state, "compound_state", "unknown"),
        state_description=getattr(investor_state, "state_description", ""),
        dominant_factors=", ".join(getattr(investor_state, "dominant_factors", [])),
        state_implications=", ".join(getattr(investor_state, "state_implications", [])),
    )

    for attempt in (1, 2):
        try:
            raw = llm_call(prompt, num_predict=256)
            return {
                "obligation_weight": max(1, min(100, int(raw.get("obligation_weight", 50)))),
                "behavioral_weight": max(1, min(100, int(raw.get("behavioral_weight", 50)))),
                "income_weight":     max(1, min(100, int(raw.get("income_weight", 50)))),
                "reasoning":         raw.get("reasoning", ""),
            }
        except Exception:
            pass

    return {"obligation_weight": 50, "behavioral_weight": 50, "income_weight": 50, "reasoning": "defaults used"}


# Keep old name as alias for backward compat
_query_narrative_weights = _query_state_weights



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

def assess_cultural_obligation(ctx: ProfileContext, narrative_weight: int = 50) -> CategoryResult:
    """
    Cultural obligation scoring — influenced by narrative_weight.
    narrative_weight (1-100): LLM-derived signal of how constrained this investor is.
    Higher weight → obligation signals amplified proportionally.
    """
    score = 5
    reasons = []
    modifiers = []

    # Scale factor: narrative_weight maps 1-100 → multiplier 0.6-1.4
    # At weight=50 (neutral): multiplier=1.0 (no change)
    # At weight=80 (high burden): multiplier=1.24 (amplified)
    # At weight=20 (low burden): multiplier=0.76 (reduced)
    scale = 0.6 + (narrative_weight / 100) * 0.8
    modifiers.append(
        f"Narrative obligation weight: {narrative_weight}/100 → score multiplier {scale:.2f}"
    )

    ntol = ctx.near_term_obligation_level
    if ntol == "high":
        score += int(round(40 * scale))
        otype = f" ({ctx.obligation_type})" if ctx.obligation_type else ""
        reasons.append(
            f"High near-term obligation{otype} — significant capital required imminently"
        )
    elif ntol == "moderate":
        score += int(round(20 * scale))
        otype = f" ({ctx.obligation_type})" if ctx.obligation_type else ""
        reasons.append(
            f"Moderate near-term obligation{otype} — capital commitment expected within 1–2 years"
        )

    for sig in ctx.cultural_signals:
        if sig.signal_type == "hidden_obligation":
            score += int(round(20 * scale))
            reasons.append(
                "Hidden financial obligation detected — investable surplus is lower than income suggests"
            )
            modifiers.append(
                "Advisor insight: open conversation about goal-based investing for this milestone"
            )
        elif sig.signal_type == "religious":
            score += int(round(10 * scale))
            reasons.append(
                "Religious/charitable giving commitment — fixed, non-negotiable outflow"
            )
            modifiers.append("Religious giving is culturally fixed — do not suggest reducing it")
        elif sig.signal_type == "social_pressure":
            score += int(round(10 * scale))
            reasons.append(
                "Social/community financial expectations — implicit obligation burden"
            )
        elif sig.signal_type == "family_role":
            score += int(round(15 * scale))
            reasons.append(
                f"Family role obligation ({sig.description}) — "
                "financial responsibility extends beyond nuclear family"
            )

    if ctx.future_obligation_score > 0:
        weighted = round(ctx.future_obligation_score * 0.7 * scale, 1)
        score += weighted
        reasons.append(
            f"Future financial commitment detected — adds {weighted:.0f} pts to obligation burden"
        )

    for ev in ctx.life_events:
        if ev.event_type == "death" and ev.recency == "recent":
            score += int(round(10 * scale))
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

def assess_behavioral_risk(ctx: ProfileContext, narrative_weight: int = 50, investor_state=None) -> CategoryResult:
    """
    Behavioral risk scoring — influenced by investor_state compound state.
    v12: reads compound_state from investor_state, not ctx.grief_state directly.
    The state captures the INTERACTION of signals.
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

    # Narrative-derived amplifier from state
    emotional_amp = 0.5 + (narrative_weight / 100)
    modifiers.append(
        f"State behavioral weight: {narrative_weight}/100 → emotional amplifier {emotional_amp:.2f}"
    )

    if rb == "high":
        score += 55
        reasons.append("High risk-seeking behavior reported")
    elif rb == "medium":
        score += 30
        reasons.append("Moderate risk behavior")
    elif rb == "low":
        score += 5
        reasons.append("Low risk behavior — conservative orientation")

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

    for sig in ctx.behavioral_signals:
        if sig.signal_type == "fear" and sig.strength == "strong":
            score = max(score, 10)
            modifiers.append("Strong fear/panic signal — behavioral risk floored at 10")
        elif sig.signal_type == "anxiety":
            score = min(100, score + int(round(5 * emotional_amp)))
            modifiers.append("Anxiety-driven monitoring behavior detected")
        elif sig.signal_type == "peer_influence":
            score = min(100, score + int(round(10 * emotional_amp)))
            modifiers.append("Peer-influenced decisions increase behavioral risk")
        elif sig.signal_type == "overconfidence":
            score = min(100, score + int(round(15 * emotional_amp)))
            modifiers.append("Overconfidence detected — investor may underestimate downside risk")
        elif sig.signal_type == "analytical":
            score = max(1, score - int(round(10 * emotional_amp)))
            modifiers.append("Analytical decision-making reduces behavioral risk")

    # Read grief/emotional state from compound_state, not raw flag
    compound = getattr(investor_state, "compound_state", "").lower() if investor_state else ""
    state_stability = getattr(investor_state, "state_stability", "stable") if investor_state else "stable"

    if compound and any(w in compound for w in ("grief", "crisis", "survival", "burdened")):
        if rb in ("medium", "high") or lr in ("neutral", "aggressive"):
            grief_add = int(round(15 * emotional_amp))
            score = min(100, score + grief_add)
            modifiers.append(
                f"Compound state '{compound}' indicates emotional suppression: "
                f"+{grief_add} pts (state-calibrated). "
                "Current score reflects temporary state, not baseline. Reassess in 6–12 months."
            )
    elif ctx.grief_state and not compound:
        # Fallback to raw flag only when state synthesis unavailable
        if rb in ("medium", "high") or lr in ("neutral", "aggressive"):
            grief_add = int(round(15 * emotional_amp))
            score = min(100, score + grief_add)
            modifiers.append(
                f"Grief state (fallback flag): +{grief_add} pts. Reassess in 6–12 months."
            )

    if ctx.recency_bias_risk:
        modifiers.append(
            "RECENCY BIAS FLAG: High risk tolerance with <1yr experience suggests "
            "recency bias, not stable risk acceptance."
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


def assess_all_categories(ctx: ProfileContext, investor_state=None, narrative=None) -> CategoryAssessment:
    """
    Run all 6 category assessors against the ProfileContext.
    v12: accepts investor_state (compound state) as primary input.
    Reads compound_state — not raw flags.
    narrative accepted for backward compat but investor_state takes precedence.
    """
    # Use investor_state if available, fall back to narrative for weight query
    state_input = investor_state if investor_state is not None else narrative
    weights = _query_state_weights(state_input)
    weight_note = weights.get("reasoning", "")

    income_result  = assess_income_stability(ctx)
    em_result      = assess_emergency_preparedness(ctx)
    debt_result    = assess_debt_burden(ctx)
    dep_result     = assess_dependency_load(ctx)
    cult_result    = assess_cultural_obligation(ctx, narrative_weight=weights["obligation_weight"])
    behav_result   = assess_behavioral_risk(
        ctx,
        narrative_weight=weights["behavioral_weight"],
        investor_state=investor_state,
    )

    if weight_note:
        cult_result.modifiers.append(f"State weight calibration: {weight_note}")

    return CategoryAssessment(
        income_stability=income_result,
        emergency_preparedness=em_result,
        debt_burden=debt_result,
        dependency_load=dep_result,
        cultural_obligation=cult_result,
        behavioral_risk=behav_result,
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
