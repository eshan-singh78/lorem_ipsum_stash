"""
Cross-Axis Analysis — v3, deterministic, no LLM.
Archetype: psychology (risk × context).
Handles risk=None gracefully (missing behavioral data).
"""


def _archetype(risk: int | None, context: int) -> str:
    if risk is None:
        return "Unknown — behavioral data insufficient"
    if risk < 30 and context >= 60:
        return "Analyst"
    if risk >= 70 and context < 35:
        return "Explorer"
    if risk >= 60 and context >= 60:
        return "Aggressive Sophisticate"
    if risk < 40 and context < 40:
        return "Conservative Novice"
    if risk >= 50 and context < 40:
        return "Impulsive Moderate"
    return "Balanced"


def cross_axis_analysis(scores: dict) -> dict:
    risk       = scores.get("risk")        # may be None
    cashflow   = scores["cashflow"]
    obligation = scores["obligation"]
    context    = scores["context"]
    capacity   = scores["financial_capacity"]

    result = {
        "mismatch":             None,
        "constraint":           None,
        "archetype":            _archetype(risk, context),
        "suitability_insights": [],
        "debug": {
            "financial_capacity_formula": (
                f"cashflow({cashflow}) × (1 - obligation({obligation})/100) = {capacity}"
            ),
            "risk_vs_capacity_gap": (risk - capacity) if risk is not None else None,
        },
    }

    if risk is not None:
        gap = risk - capacity
        if gap > 25:
            result["mismatch"] = (
                f"Risk appetite ({risk}) exceeds financial capacity ({capacity}) by {gap} pts — "
                "investor willing to take more risk than finances support."
            )
        elif capacity - risk > 25:
            result["mismatch"] = (
                f"Financial capacity ({capacity}) well above risk appetite ({risk}) — "
                "investor is more conservative than situation requires."
            )

    if obligation > 70:
        result["constraint"] = (
            f"High obligation burden ({obligation}) — EMI/dependents/expenses limit surplus."
        )
    elif capacity < 30:
        result["constraint"] = (
            f"Low financial capacity ({capacity}) — little room for investment."
        )

    insights = []
    if obligation > 70:
        insights.append("Restrict aggressive or illiquid exposure — high obligation burden.")
    if context < 30:
        insights.append("Education-first approach required before complex instruments.")
    if cashflow < 40:
        insights.append("Prioritize liquidity; avoid long lock-in products.")
    if risk is not None and risk > 70 and capacity < 40:
        insights.append(
            "Risk appetite conflicts with financial capacity — re-evaluation recommended."
        )
    if context >= 70 and risk is not None and risk >= 60:
        insights.append("Suitable for equity-heavy or alternative investment strategies.")
    if capacity >= 60 and risk is not None and risk < 40:
        insights.append(
            "Strong capacity but low risk appetite — consider gradual equity exposure."
        )
    if risk is None:
        insights.append(
            "Behavioral data unavailable — risk suitability cannot be fully assessed."
        )
    if not insights:
        insights.append("No critical constraints detected. Standard profiling applies.")

    result["suitability_insights"] = insights
    return result
