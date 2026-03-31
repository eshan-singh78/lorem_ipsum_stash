"""
Cross-Axis Union Engine + Narrative Generator — InvestorDNA v6

Architecture:
  AxisScores + CategoryAssessment + ProfileContext → CrossAxisReport

InvestorDNA principle:
  "The 4 axes interact multiplicatively, not additively."

  An investor with Risk Appetite 60/99 and Obligation Burden 80/99 does NOT
  have an 'average' profile. They have a profile where psychological willingness
  to take risk exceeds financial capacity to absorb losses.

This module:
  1. Detects cross-axis mismatches (risk vs. capacity)
  2. Identifies binding constraints (what actually limits this investor)
  3. Assigns the InvestorDNA archetype (2×2: Loss Aversion × Sophistication)
  4. Generates advisor insight narrative (human-readable, actionable)
  5. Generates investor-facing narrative (empowering, not clinical)
  6. Produces suitability assessment with specific recommendations
"""

from dataclasses import dataclass, field
from axis_scoring import AxisScores
from context_categories import CategoryAssessment
from profile_context import ProfileContext


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
) -> tuple[str, str]:
    """
    Returns (archetype_name, archetype_description).

    InvestorDNA archetypes:
      High Sophistication + Low Loss Aversion  → The Strategist
      High Sophistication + High Loss Aversion → The Analyst
      Low Sophistication  + Low Loss Aversion  → The Explorer  (DANGER: naive risk-taking)
      Low Sophistication  + High Loss Aversion → The Guardian

    Obligation modifier: high obligation → append "(Constrained)"
    Grief modifier: → append "(Crisis Variant)"
    """
    if risk is None:
        return "Unknown", "Insufficient behavioral data to assign archetype"

    high_sophistication = context >= 55
    low_loss_aversion   = risk >= 55   # high risk appetite = low loss aversion

    if high_sophistication and low_loss_aversion:
        name = "The Strategist"
        desc = (
            "Pursues growth with informed confidence. "
            "Has the knowledge and temperament to handle equity-heavy portfolios. "
            "Key risk: overconfidence and fragmentation."
        )
    elif high_sophistication and not low_loss_aversion:
        name = "The Analyst"
        desc = (
            "High analytical capability but conservative risk orientation. "
            "Makes deliberate, data-driven allocation decisions. "
            "Responds to spreadsheets and evidence, not reassurance."
        )
    elif not high_sophistication and low_loss_aversion:
        name = "The Explorer"
        desc = (
            "Open to opportunity and drawn to high-return assets. "
            "PROFESSIONAL NOTE: Low sophistication + low loss aversion = naive risk-taking, "
            "not informed confidence. First real loss may trigger panic exit. "
            "Education before exposure."
        )
    else:
        name = "The Guardian"
        desc = (
            "Protects financial foundation with care and caution. "
            "Loss aversion driven by unfamiliarity or obligation burden, not trauma. "
            "Education and gradual exposure build confidence over time."
        )

    # Modifiers
    if ctx.grief_state:
        name += " (Crisis Variant)"
        desc += (
            " CRISIS VARIANT: Current profile reflects grief-suppressed state. "
            "Analytical capability is intact but emotional processing overrides it. "
            "This is a temporary state — reassess in 6–12 months."
        )
    elif obligation >= 70:
        name += " (Constrained)"
        desc += (
            " CONSTRAINED: High obligation burden limits investable surplus. "
            "Psychological risk tolerance exceeds financial capacity to bear losses."
        )

    return name, desc


# ---------------------------------------------------------------------------
# Mismatch detection
# ---------------------------------------------------------------------------

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
# Suitability assessment
# ---------------------------------------------------------------------------

def _compute_suitability(
    scores: AxisScores,
    archetype: str,
    mismatch: dict | None,
    constraint: dict | None,
    ctx: ProfileContext,
) -> dict:
    """
    Produces a suitability classification and specific product guidance.
    """
    capacity = scores.financial_capacity
    risk     = scores.risk
    context  = scores.context

    # Classification
    if capacity < 25:
        classification = "High Constraint — Investment Premature"
        equity_ceiling = 0
    elif capacity < 50:
        classification = "Moderate Constraint — Conservative Allocation"
        equity_ceiling = 20
    elif capacity < 70:
        classification = "Moderate Capacity — Balanced Allocation"
        equity_ceiling = 40
    else:
        classification = "High Capacity — Growth Allocation Possible"
        equity_ceiling = 70

    # Risk appetite cap: never exceed what capacity supports
    if risk is not None and risk < equity_ceiling:
        equity_ceiling = min(equity_ceiling, risk)

    # Grief state: hard cap
    if ctx.grief_state:
        equity_ceiling = min(equity_ceiling, 15)

    # Recency bias: cap naive risk-takers
    if ctx.recency_bias_risk:
        equity_ceiling = min(equity_ceiling, 20)

    # Low sophistication: cap complexity
    if context < 35:
        equity_ceiling = min(equity_ceiling, 25)

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

    return {
        "classification": classification,
        "equity_ceiling_pct": equity_ceiling,
        "guidance": guidance,
    }


# ---------------------------------------------------------------------------
# Narrative generation
# ---------------------------------------------------------------------------

def _generate_advisor_narrative(
    scores: AxisScores,
    archetype: str,
    archetype_desc: str,
    mismatch: dict | None,
    constraint: dict | None,
    categories: CategoryAssessment,
    ctx: ProfileContext,
) -> str:
    """
    Generates the advisor insight narrative.
    This is what an experienced RIA would say after reviewing the full profile.
    """
    parts = []

    # Opening: archetype + core tension
    parts.append(f"Archetype: {archetype}.")

    # Core behavioral insight
    br = categories.behavioral_risk
    if ctx.grief_state:
        parts.append(
            "This investor has high analytical capability but grief-suppressed risk tolerance. "
            "The gap between what they KNOW and what they can emotionally ACCEPT is the key tension. "
            "Approach with data, not reassurance — they will respond to spreadsheets, not sympathy."
        )
    elif ctx.recency_bias_risk:
        parts.append(
            "This investor's apparent risk tolerance is driven by recency bias from recent gains, "
            "not by genuine understanding of downside risk. "
            "They have never experienced a real loss. First significant drawdown may trigger panic exit."
        )
    elif ctx.fragmentation_risk:
        parts.append(
            "This investor is financially sophisticated but operating with a fragmented picture. "
            "No single advisor has the complete view. "
            "The advisor's value proposition is consolidation and holistic planning."
        )
    elif ctx.peer_driven:
        parts.append(
            "Investment decisions are peer-influenced, not research-driven. "
            "The investor may not have internalized the reasoning behind their portfolio. "
            "Education before product complexity."
        )

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
                "Hidden obligation detected — investable surplus is lower than income suggests. "
                "Open the conversation about goal-based investing for this milestone."
            )

    # Drift expectation
    if ctx.grief_state:
        parts.append(
            "Drift expectation: Axis 1 (Risk) score expected to increase over 6–18 months "
            "as grief processing occurs. Set reassessment at 6-month mark."
        )

    return " ".join(parts)


def _generate_investor_narrative(
    archetype: str,
    scores: AxisScores,
    ctx: ProfileContext,
    suitability: dict,
) -> str:
    """
    Investor-facing narrative: empowering, not clinical.
    Uses 'you' language. Acknowledges strengths before constraints.
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
    else:  # Guardian
        parts.append(
            "You protect your financial foundation with care and caution. "
            "This is a strength — stability is the foundation of long-term wealth."
        )

    # Current situation
    if ctx.grief_state:
        parts.append(
            "You're navigating a difficult period, and it's natural for financial decisions "
            "to feel heavier right now. There's no rush — building stability step by step is the right approach."
        )

    # Capacity framing
    cap = scores.financial_capacity
    if cap >= 60:
        parts.append(
            "Your financial capacity gives you real flexibility to invest for the future."
        )
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
) -> CrossAxisReport:
    """
    Full cross-axis union analysis.
    Produces archetype, mismatch detection, constraint identification,
    suitability assessment, and dual narratives.
    """
    archetype, archetype_desc = _assign_archetype(
        scores.risk, scores.context, scores.obligation, ctx
    )

    mismatch   = _detect_mismatch(scores.risk, scores.financial_capacity, scores.obligation)
    constraint = _identify_binding_constraint(categories, scores, ctx)

    suitability = _compute_suitability(scores, archetype, mismatch, constraint, ctx)

    advisor_narrative  = _generate_advisor_narrative(
        scores, archetype, archetype_desc, mismatch, constraint, categories, ctx
    )
    investor_narrative = _generate_investor_narrative(archetype, scores, ctx, suitability)

    # Suitability insights (bullet-point version for UI)
    insights = []
    if scores.obligation > 70:
        insights.append("High obligation burden — restrict aggressive or illiquid exposure")
    if scores.context < 35:
        insights.append("Education-first approach required before complex instruments")
    if scores.cashflow < 40:
        insights.append("Prioritize liquidity — avoid long lock-in products")
    if scores.risk is not None and scores.risk > 70 and scores.financial_capacity < 40:
        insights.append("Risk appetite conflicts with financial capacity — re-evaluation recommended")
    if scores.context >= 70 and scores.risk is not None and scores.risk >= 60:
        insights.append("Suitable for equity-heavy or alternative investment strategies")
    if ctx.fragmentation_risk:
        insights.append("Consolidate advisors — fragmented picture is the primary risk")
    if ctx.grief_state:
        insights.append("Grief state — no major allocation changes until reassessment")
    if ctx.recency_bias_risk:
        insights.append("Recency bias — do not increase market exposure until literacy baseline established")
    if not insights:
        insights.append("No critical constraints detected — standard profiling applies")

    # Drift expectation
    drift = None
    if ctx.grief_state:
        drift = (
            "Axis 1 (Risk) score expected to increase over 6–18 months as grief processing occurs. "
            "Reassess at 6-month mark. If score unchanged at 12 months, this may be baseline personality."
        )
    elif ctx.recency_bias_risk:
        drift = (
            "Risk tolerance score may decrease significantly after first real market loss. "
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
