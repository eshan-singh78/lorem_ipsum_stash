"""
Cross-Axis Union Engine + Narrative Generator — InvestorDNA v12

Architecture:
  AxisScores + CategoryAssessment + ProfileContext
  + NarrativeOutput + DecisionOutput → CrossAxisReport

v12 design:
  Decision is the primary source. Scores are secondary context.
  The suitability classification IS the decision reasoning.
  Numeric equity ceiling is parsed from decision.equity_range — not computed.
  The numeric fallback path is removed.
"""

from dataclasses import dataclass, field
import re
from axis_scoring import AxisScores
from context_categories import CategoryAssessment
from profile_context import ProfileContext
from narrative_layer import NarrativeOutput
from decision_engine import DecisionOutput

# Backward-compat alias
MeaningOutput = NarrativeOutput

# JudgmentOutput removed in v14 — judgment_layer.py deleted


# ---------------------------------------------------------------------------
# Archetype matrix (InvestorDNA 2×2)
# Loss Aversion (Axis 1 inverted) × Financial Sophistication (Axis 4)
# Obligation Burden (Axis 3) as modifier
# ---------------------------------------------------------------------------

def _assign_archetype(
    risk: int | None,
    context: int,
    obligation: int,
    ctx: ProfileContext,
    decision=None,
) -> tuple[str, str]:
    """
    Returns (archetype_name, archetype_description).

    v10: if DecisionOutput is available, use its LLM-derived archetype.
    Matrix fallback retained only when decision engine is unavailable.
    """
    # Prefer LLM-derived archetype from decision engine
    if decision is not None:
        archetype_label = getattr(decision, "archetype", None)
        if archetype_label and archetype_label not in ("Unclassified", ""):
            desc = getattr(decision, "true_risk_profile", "")
            return archetype_label, desc

    # Fallback: matrix-based (used only when decision engine unavailable)
    if risk is None:
        return "Unknown", "Insufficient behavioral data to assign archetype"

    high_sophistication = context >= 55
    low_loss_aversion   = risk >= 55

    if high_sophistication and low_loss_aversion:
        name = "The Strategist"
        desc = (
            "Pursues growth with informed confidence. "
            "Has the knowledge and temperament to handle equity-heavy portfolios."
        )
    elif high_sophistication and not low_loss_aversion:
        name = "The Analyst"
        desc = (
            "High analytical capability but conservative risk orientation. "
            "Makes deliberate, data-driven allocation decisions."
        )
    elif not high_sophistication and low_loss_aversion:
        name = "The Explorer"
        desc = (
            "Open to opportunity and drawn to high-return assets. "
            "Low sophistication + low loss aversion = naive risk-taking, not informed confidence."
        )
    else:
        name = "The Guardian"
        desc = (
            "Protects financial foundation with care and caution. "
            "Education and gradual exposure build confidence over time."
        )

    if ctx.grief_state:
        name += " (Crisis Variant)"
    elif obligation >= 70:
        name += " (Constrained)"

    return name, desc


def _contains(text: str, *phrases: str) -> bool:
    """Case-insensitive check: does text contain ANY of the given phrases?"""
    t = (text or "").lower()
    return any(p.lower() in t for p in phrases)


# ---------------------------------------------------------------------------
# Equity ceiling from narrative signals
# ---------------------------------------------------------------------------

def _equity_ceiling_from_narrative(
    narrative: NarrativeOutput | None,
    capacity: int,
    risk: int | None,
    ctx: ProfileContext,
    context_score: int,
) -> int:
    """
    Derive equity ceiling from narrative text signals only.
    Numeric capacity is used as a soft anchor, not a formula.
    The narrative signals always take precedence.
    """
    # Soft anchor from financial capacity — treated as a hint, not a rule
    if capacity < 25:
        ceiling = 10
    elif capacity < 50:
        ceiling = 25
    elif capacity < 70:
        ceiling = 45
    else:
        ceiling = 65

    # Risk appetite as secondary anchor
    if risk is not None:
        ceiling = min(ceiling, risk)

    # Hard caps from ProfileContext flags (these are factual, not formula)
    if ctx.grief_state:
        ceiling = min(ceiling, 15)
    if ctx.recency_bias_risk:
        ceiling = min(ceiling, 20)
    if context_score < 35:
        ceiling = min(ceiling, 25)

    if narrative is None:
        return ceiling

    # Narrative signals override numeric anchors
    risk_truth   = narrative.risk_truth
    psych        = narrative.psychological_analysis
    financial    = narrative.financial_analysis
    reliability  = narrative.reliability_assessment

    if _contains(psych, "crisis", "survival mode", "overwhelmed", "desperate"):
        ceiling = min(ceiling, 10)
    elif _contains(risk_truth, "suppressed", "overstated", "does not match", "inflated"):
        ceiling = min(ceiling, 20)
    elif _contains(risk_truth, "temporary", "transitional", "not permanent"):
        ceiling = min(ceiling, 30)

    if _contains(financial, "severely constrained", "no investable surplus",
                 "barely covers", "most income committed"):
        ceiling = min(ceiling, 20)

    if _contains(reliability, "unreliable", "distorted", "not trustworthy"):
        ceiling = min(ceiling, 25)

    return max(0, ceiling)



def _detect_mismatch(
    risk: int | None,
    financial_capacity: int,
    obligation: int,
) -> dict | None:
    if risk is None:
        return None

    gap = risk - financial_capacity

    if gap > 25:
        return {
            "type": "risk_exceeds_capacity",
            "severity": "critical" if gap > 40 else "moderate",
            "description": (
                f"Risk appetite ({risk}/99) exceeds financial capacity ({financial_capacity}/99) "
                f"by {gap} points. Investor is psychologically willing to take more risk "
                "than their finances can absorb."
            ),
            "implication": (
                "Do NOT recommend products matching stated risk appetite. "
                "Suitability must be based on financial capacity, not psychological preference."
            ),
        }
    elif financial_capacity - risk > 25:
        return {
            "type": "capacity_exceeds_risk",
            "severity": "moderate",
            "description": (
                f"Financial capacity ({financial_capacity}/99) well above risk appetite ({risk}/99). "
                "Investor is more conservative than their financial situation requires."
            ),
            "implication": (
                "Gradual equity exposure may be appropriate as confidence builds. "
                "Do not push — let the investor lead the pace."
            ),
        }

    return None


# ---------------------------------------------------------------------------
# Binding constraint identification
# ---------------------------------------------------------------------------

def _identify_binding_constraint(
    categories: CategoryAssessment,
    scores: AxisScores,
    ctx: ProfileContext,
) -> dict | None:
    """
    A binding constraint is the single factor that most limits this investor.
    It overrides all other considerations.
    """
    # Obligation burden is the most common binding constraint in Indian context
    if scores.obligation >= 75:
        return {
            "type": "obligation_burden",
            "description": (
                f"Obligation burden ({scores.obligation}/99) is the binding constraint. "
                "Even if risk appetite were moderate, the obligation burden would limit "
                "investable surplus to near-zero after EMIs, dependent costs, and cultural obligations."
            ),
            "priority_actions": _obligation_priority_actions(categories, ctx),
        }

    if scores.financial_capacity < 25:
        return {
            "type": "financial_capacity",
            "description": (
                f"Financial capacity ({scores.financial_capacity}/99) is critically low. "
                "Investment discussion is premature — financial stability must come first."
            ),
            "priority_actions": [
                "Build emergency fund to minimum 3 months",
                "Reduce highest-interest debt first",
                "Stabilize income before any investment commitment",
            ],
        }

    if scores.cashflow < 30:
        return {
            "type": "income_instability",
            "description": (
                f"Cash flow instability ({scores.cashflow}/99) is the binding constraint. "
                "Volatile or insufficient income makes regular investment commitments risky."
            ),
            "priority_actions": [
                "Income stabilization before investment discussion",
                "If investing, use lump-sum during high-income periods rather than fixed SIP",
                "Emergency fund is the first priority",
            ],
        }

    return None


def _obligation_priority_actions(
    categories: CategoryAssessment,
    ctx: ProfileContext,
) -> list[str]:
    actions = []

    # Elder care / uninsured dependents
    for sig in ctx.cultural_signals:
        if sig.signal_type == "family_role":
            actions.append(
                "Priority 1: Health insurance for uninsured dependents "
                "(uninsured elder = maximum financial vulnerability)"
            )
            break

    # Emergency fund
    em = categories.emergency_preparedness
    if em.score is None or (em.score is not None and em.score < 60):
        actions.append("Priority 2: Rebuild emergency fund to 6-month minimum")

    # Debt
    debt = categories.debt_burden
    if debt.score is not None and debt.score > 40:
        actions.append("Priority 3: Prepay highest-interest debt (education loan before home loan)")

    actions.append("Priority 4: Only then — investment allocation discussion")
    return actions


# ---------------------------------------------------------------------------
# Suitability assessment — narrative-driven
# ---------------------------------------------------------------------------

def _compute_suitability(
    scores,
    archetype: str,
    mismatch: dict | None,
    constraint: dict | None,
    ctx: ProfileContext,
    narrative: NarrativeOutput | None = None,
    judgment=None,
    decision=None,
) -> dict:
    """
    v10: suitability is driven by DecisionOutput when available.
    The classification IS the decision engine's reasoning — not a template.
    Falls back to narrative-driven logic when decision engine unavailable.
    """
    # v10: use decision engine output as primary classification
    if decision is not None:
        reasoning   = getattr(decision, "reasoning", "")
        strategy    = getattr(decision, "recommended_strategy", "")
        equity_range = getattr(decision, "equity_range", "")
        advisor_note = getattr(decision, "advisor_note", "")

        # Build narrative-first classification from decision engine
        classification_parts = []
        if reasoning:
            classification_parts.append(reasoning)
        if strategy:
            classification_parts.append(f"Recommended strategy: {strategy}")
        if judgment and judgment.reassessment_recommended:
            classification_parts.append(
                "This recommendation is provisional. "
                "Reassessment in 3–6 months is strongly recommended."
            )

        # Parse equity_range to get a numeric ceiling for guidance
        equity_ceiling = _parse_equity_ceiling(equity_range)

        return {
            "classification":    " ".join(classification_parts),
            "equity_range":      equity_range,
            "equity_ceiling_pct": equity_ceiling,
            "guidance":          _build_guidance(equity_ceiling, ctx),
            "advisor_note":      advisor_note,
        }

    # Fallback: narrative-driven (no decision engine)
    capacity      = scores.financial_capacity
    risk          = scores.risk
    context_score = scores.context

    equity_ceiling = _equity_ceiling_from_narrative(
        narrative, capacity, risk, ctx, context_score
    )

    classification = _generate_suitability_narrative(
        equity_ceiling, narrative, ctx, mismatch, constraint, judgment
    )

    return {
        "classification":    classification,
        "equity_range":      f"0-{equity_ceiling}%",
        "equity_ceiling_pct": equity_ceiling,
        "guidance":          _build_guidance(equity_ceiling, ctx),
        "advisor_note":      "",
    }


def _parse_equity_ceiling(equity_range: str) -> int:
    """Extract the upper bound from an equity range string like '10-20%' or '30%'."""
    if not equity_range:
        return 20
    # Match "X-Y%" → take Y
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*%?", equity_range)
    if m:
        return int(m.group(2))
    # Match single "X%"
    m = re.search(r"(\d+)\s*%?", equity_range)
    if m:
        return int(m.group(1))
    return 20


def _generate_suitability_narrative(
    equity_ceiling: int,
    narrative: NarrativeOutput | None,
    ctx: ProfileContext,
    mismatch: dict | None,
    constraint: dict | None,
    judgment=None,
) -> str:
    """
    Generate a fully synthesized suitability explanation.
    Draws from narrative fields — not template strings.
    """
    parts = []

    # Lead with the investor's current situation from narrative
    if narrative:
        life    = narrative.life_summary
        psych   = narrative.psychological_analysis
        risk_t  = narrative.risk_truth
        fin     = narrative.financial_analysis

        # Situation framing
        if life:
            parts.append(life)

        # Financial reality
        if fin:
            parts.append(fin)

        # Psychological state and its implication
        if psych:
            parts.append(psych)

        # Risk truth — the core of the suitability decision
        if risk_t:
            parts.append(risk_t)
    else:
        # Fallback when narrative unavailable
        if ctx.grief_state:
            parts.append(
                "This investor is navigating a grief state that is suppressing their "
                "natural risk tolerance. Current behavior does not reflect baseline personality."
            )

    # Mismatch
    if mismatch:
        parts.append(mismatch["description"])

    # Binding constraint
    if constraint:
        parts.append(constraint["description"])

    # Judgment overrides
    if judgment and judgment.overrides:
        meaningful = [o for o in judgment.overrides if o.before != o.after]
        if meaningful:
            override_notes = " ".join(
                f"({o.rule}: {o.axis} adjusted from {o.before} to {o.after})"
                for o in meaningful
            )
            parts.append(f"Score adjustments applied: {override_notes}.")

    # Allocation recommendation
    if equity_ceiling == 0:
        parts.append(
            "Investment allocation is premature at this stage. "
            "Priority actions: debt reduction, emergency fund, and insurance coverage."
        )
    elif equity_ceiling <= 15:
        parts.append(
            f"A conservative allocation is appropriate — maximum equity exposure {equity_ceiling}%. "
            "Suitable instruments: liquid funds, short-duration debt, fixed deposits."
        )
    elif equity_ceiling <= 30:
        parts.append(
            f"A moderately conservative allocation is recommended — equity up to {equity_ceiling}% "
            "via balanced advantage or conservative hybrid funds. "
            "Avoid mid/small-cap and illiquid products."
        )
    elif equity_ceiling <= 50:
        parts.append(
            f"A balanced allocation is appropriate — equity up to {equity_ceiling}% "
            "via flexi-cap or multi-cap funds. Maintain a liquidity buffer."
        )
    else:
        parts.append(
            f"A growth-oriented allocation is appropriate — equity up to {equity_ceiling}% "
            "given the investor's capacity and risk profile. "
            "Suitable for flexi-cap, mid-cap, and international diversification."
        )

    # Reassessment note
    if judgment and judgment.reassessment_recommended:
        parts.append(
            "This recommendation is provisional. "
            "Reassessment in 3–6 months is strongly recommended before finalizing long-term allocation."
        )

    return " ".join(parts)


def _build_guidance(equity_ceiling: int, ctx: ProfileContext) -> list[str]:
    guidance = []
    if equity_ceiling == 0:
        guidance.append("No equity exposure recommended at this time")
        guidance.append("Focus: debt reduction, emergency fund, insurance")
    elif equity_ceiling <= 15:
        guidance.append(f"Maximum equity exposure: {equity_ceiling}% (large-cap only)")
        guidance.append("Suitable: liquid funds, short-duration debt funds, FD")
    elif equity_ceiling <= 30:
        guidance.append(f"Maximum equity exposure: {equity_ceiling}% (large-cap / index funds)")
        guidance.append("Suitable: balanced advantage funds, conservative hybrid")
    elif equity_ceiling <= 50:
        guidance.append(f"Equity exposure up to {equity_ceiling}% appropriate")
        guidance.append("Suitable: flexi-cap, multi-cap, balanced hybrid")
    else:
        guidance.append(f"Equity-heavy allocation up to {equity_ceiling}% appropriate")
        guidance.append("Suitable: flexi-cap, mid-cap, international diversification")

    if ctx.fragmentation_risk:
        guidance.append(
            "FRAGMENTATION: Consolidate advisors before adding new products. "
            "Overlap in existing MF holdings likely."
        )
    return guidance


# ---------------------------------------------------------------------------
# Narrative generation
# ---------------------------------------------------------------------------

def _generate_advisor_narrative(
    scores,
    archetype: str,
    archetype_desc: str,
    mismatch: dict | None,
    constraint: dict | None,
    categories: CategoryAssessment,
    ctx: ProfileContext,
    narrative: NarrativeOutput | None = None,
    judgment=None,
    decision=None,
) -> str:
    """
    Generates the advisor insight narrative.
    v10: leads with decision engine's advisor_note when available.
    """
    parts = []

    parts.append(f"Archetype: {archetype}.")

    # v10: lead with decision engine's advisor note — this is the primary insight
    if decision is not None:
        advisor_note = getattr(decision, "advisor_note", "")
        if advisor_note:
            parts.append(f"Advisor note: {advisor_note}")
        true_risk = getattr(decision, "true_risk_profile", "")
        if true_risk:
            parts.append(true_risk)

    # Supplement with narrative advisor insight
    elif narrative and narrative.advisor_insight:
        parts.append(narrative.advisor_insight)

    # Contradictions
    if narrative and narrative.contradictions and \
       not _contains(narrative.contradictions, "none detected", "no contradiction"):
        parts.append(f"Contradictions: {narrative.contradictions}")

    # Mismatch
    if mismatch:
        parts.append(mismatch["description"])
        parts.append(f"Implication: {mismatch['implication']}")

    # Binding constraint
    if constraint:
        parts.append(constraint["description"])

    # Cultural obligation insight
    cult = categories.cultural_obligation
    if cult.score is not None and cult.score > 30:
        parts.append(
            f"Cultural obligation burden is significant (score: {cult.score}/100). "
            + cult.reason
        )
        if ctx.hidden_obligation_detected:
            parts.append(
                "Hidden obligation detected — investable surplus is lower than income suggests."
            )

    # Judgment override summary
    if judgment and judgment.overrides:
        meaningful = [o for o in judgment.overrides if o.before != o.after]
        if meaningful:
            parts.append(
                "Score adjustments: " + "; ".join(
                    f"{o.axis} {o.before}→{o.after} ({o.rule})"
                    for o in meaningful
                ) + "."
            )

    if judgment and judgment.reassessment_recommended:
        parts.append(judgment.judgment_summary)
    elif ctx.grief_state:
        parts.append(
            "Drift expectation: Risk score expected to increase over 6–18 months "
            "as grief processing occurs. Set reassessment at 6-month mark."
        )

    return " ".join(parts)


def _generate_investor_narrative(
    archetype: str,
    scores,
    ctx: ProfileContext,
    suitability: dict,
    narrative: NarrativeOutput | None = None,
    judgment=None,
) -> str:
    """
    Investor-facing narrative: empowering, not clinical.
    v9: draws from narrative.life_summary for personalization.
    """
    parts = []

    # Archetype framing
    if "Strategist" in archetype:
        parts.append(
            "You pursue growth with informed confidence. "
            "Your financial knowledge and experience give you a strong foundation."
        )
    elif "Analyst" in archetype:
        parts.append(
            "You approach investing with careful analysis and a preference for certainty. "
            "Your strength is in understanding the details before committing."
        )
    elif "Explorer" in archetype:
        parts.append(
            "You're open to opportunity and drawn to high-growth possibilities. "
            "Building your financial knowledge will help you make these opportunities work for you."
        )
    else:
        parts.append(
            "You protect your financial foundation with care and caution. "
            "This is a strength — stability is the foundation of long-term wealth."
        )

    # Personalize from narrative life summary if available
    if narrative and narrative.life_summary:
        # Extract the human situation without clinical language
        life = narrative.life_summary
        if _contains(life, "grief", "loss", "bereavement"):
            parts.append(
                "You're navigating a difficult period, and it's natural for financial decisions "
                "to feel heavier right now. There's no rush — building stability step by step is the right approach."
            )
        elif _contains(life, "responsibility", "provider", "family", "dependent"):
            parts.append(
                "You carry significant responsibility for others, which shapes every financial decision you make. "
                "That responsibility is also your motivation — protecting what matters most."
            )
    elif ctx.grief_state:
        parts.append(
            "You're navigating a difficult period, and it's natural for financial decisions "
            "to feel heavier right now. There's no rush — building stability step by step is the right approach."
        )

    # Capacity framing
    cap = scores.financial_capacity
    if cap >= 60:
        parts.append("Your financial capacity gives you real flexibility to invest for the future.")
    elif cap >= 35:
        parts.append(
            "Your financial commitments are significant, but there is room to build "
            "a disciplined investment habit."
        )
    else:
        parts.append(
            "Right now, your priority is building financial resilience — "
            "a strong foundation makes everything else possible."
        )

    # Growth edge
    if "Strategist" in archetype:
        parts.append("Growth edge: Even the best investors benefit from a trusted second opinion.")
    elif "Analyst" in archetype:
        parts.append("Growth edge: As your situation stabilizes, your analytical strength will serve you well.")
    elif "Explorer" in archetype:
        parts.append("Growth edge: Understanding what you own — and why — is the next step.")
    else:
        parts.append("Growth edge: Confidence grows with knowledge. Start with the basics and build from there.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Cross-axis report
# ---------------------------------------------------------------------------

@dataclass
class CrossAxisReport:
    archetype: str
    archetype_description: str
    mismatch: dict | None
    binding_constraint: dict | None
    suitability: dict
    advisor_narrative: str
    investor_narrative: str
    suitability_insights: list[str]
    drift_expectation: str | None


def build_cross_axis_report(
    scores: AxisScores,
    categories: CategoryAssessment,
    ctx: ProfileContext,
    narrative: NarrativeOutput | None = None,
    judgment=None,          # accepted for compat, not used in primary path
    decision: DecisionOutput | None = None,
) -> CrossAxisReport:
    """
    v12: decision is the primary source.
    Scores are secondary context for mismatch detection and insights.
    Judgment is accepted for backward compat but not used.
    """
    # Use raw scores — no judgment adjustment in v12
    eff_risk       = scores.risk
    eff_cashflow   = scores.cashflow
    eff_obligation = scores.obligation
    eff_context    = scores.context
    eff_capacity   = scores.financial_capacity

    class _Eff:
        risk = eff_risk
        cashflow = eff_cashflow
        obligation = eff_obligation
        context = eff_context
        financial_capacity = eff_capacity

    eff = _Eff()

    # Archetype: from decision (LLM-derived), matrix as fallback
    archetype, archetype_desc = _assign_archetype(
        eff.risk, eff.context, eff.obligation, ctx, decision
    )

    # Mismatch and constraint: still computed from scores as secondary signals
    mismatch   = _detect_mismatch(eff.risk, eff.financial_capacity, eff.obligation)
    constraint = _identify_binding_constraint(categories, eff, ctx)

    # Suitability: driven by decision output
    suitability = _compute_suitability(
        eff, archetype, mismatch, constraint, ctx, narrative, None, decision
    )

    advisor_narrative  = _generate_advisor_narrative(
        eff, archetype, archetype_desc, mismatch, constraint,
        categories, ctx, narrative, None, decision
    )
    investor_narrative = _generate_investor_narrative(
        archetype, eff, ctx, suitability, narrative, None
    )

    insights = _build_suitability_insights(eff, ctx, narrative, None, decision)

    # Drift expectation
    drift = None
    if decision and decision.confidence == "low":
        drift = (
            "Decision confidence is low — reassessment recommended once more "
            "information is available."
        )
    elif narrative and _contains(narrative.risk_truth, "suppressed", "temporary",
                                  "not permanent", "transitional"):
        drift = (
            "Risk tolerance appears temporarily suppressed. "
            "Reassess in 3–6 months once situational pressures stabilize."
        )
    elif ctx.grief_state:
        drift = (
            "Risk score expected to increase over 6–18 months as grief processing occurs. "
            "Reassess at 6-month mark."
        )
    elif ctx.recency_bias_risk:
        drift = (
            "Risk tolerance may decrease significantly after first real market loss. "
            "Monitor closely — do not lock into high-risk products."
        )

    return CrossAxisReport(
        archetype=archetype,
        archetype_description=archetype_desc,
        mismatch=mismatch,
        binding_constraint=constraint,
        suitability=suitability,
        advisor_narrative=advisor_narrative,
        investor_narrative=investor_narrative,
        suitability_insights=insights,
        drift_expectation=drift,
    )


def _build_suitability_insights(
    eff,
    ctx: ProfileContext,
    narrative: NarrativeOutput | None,
    judgment,           # accepted for compat, not used
    decision=None,
) -> list[str]:
    """Suitability insight bullets — scores as secondary signals."""
    insights = []

    if decision is not None:
        conf = getattr(decision, "confidence", "") or getattr(decision, "confidence_level", "")
        if conf == "low":
            insights.append("Decision confidence is low — treat recommendation as provisional")

    if eff.obligation > 70:
        insights.append("High obligation burden — restrict aggressive or illiquid exposure")
    if eff.context < 35:
        insights.append("Education-first approach required before complex instruments")
    if eff.cashflow < 40:
        insights.append("Prioritize liquidity — avoid long lock-in products")
    if eff.risk is not None and eff.risk > 70 and eff.financial_capacity < 40:
        insights.append("Risk appetite conflicts with financial capacity — re-evaluation recommended")
    if eff.context >= 70 and eff.risk is not None and eff.risk >= 60:
        insights.append("Suitable for equity-heavy or alternative investment strategies")
    if ctx.fragmentation_risk:
        insights.append("Consolidate advisors — fragmented picture is the primary risk")
    if ctx.grief_state:
        insights.append("Grief state — no major allocation changes until reassessment")
    if ctx.recency_bias_risk:
        insights.append("Recency bias — do not increase market exposure until literacy baseline established")

    if narrative:
        if _contains(narrative.risk_truth, "suppressed", "does not match", "overstated"):
            insights.append("Stated risk preference does not reflect actual capacity — use conservative ceiling")
        if _contains(narrative.contradictions, "contradiction", "inconsistent", "mismatch") and \
           not _contains(narrative.contradictions, "none detected"):
            insights.append("Profile contradictions detected — verify before recommending")
        if _contains(narrative.reliability_assessment, "unreliable", "distorted"):
            insights.append("Profile reliability is low — treat all scores as provisional")

    if not insights:
        insights.append("No critical constraints detected — standard profiling applies")

    return insights


def cross_axis_report_to_dict(report: CrossAxisReport) -> dict:
    return {
        "archetype":             report.archetype,
        "archetype_description": report.archetype_description,
        "mismatch":              report.mismatch,
        "binding_constraint":    report.binding_constraint,
        "suitability":           report.suitability,
        "advisor_narrative":     report.advisor_narrative,
        "investor_narrative":    report.investor_narrative,
        "suitability_insights":  report.suitability_insights,
        "drift_expectation":     report.drift_expectation,
    }
