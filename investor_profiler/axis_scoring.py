"""
Context-Driven Axis Scoring Engine — InvestorDNA v11

Architecture:
  CategoryAssessment + ProfileContext + NarrativeOutput → AxisScores

v11 change: narrative is passed into scoring.
Fixed constants (grief_penalty=15, experience bonuses, axis weights)
are replaced with narrative-calibrated adjustments via LLM query.

Design principle:
  "Scores are influenced by understanding, not the other way around."
"""

import json
import re
import requests
from dataclasses import dataclass
from context_categories import CategoryAssessment, CategoryResult
from profile_context import ProfileContext

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL       = "llama3.1:8b"

_AXIS_CALIBRATION_PROMPT = """You are a financial advisor calibrating axis scoring for an investor.

Given the narrative below, answer:

1. How much should the investor's RISK SCORE be adjusted? (penalty or bonus, -30 to +30)
   Consider: emotional state, grief, peer influence, reliability of stated preferences.

2. How much should the OBLIGATION AXIS weight be adjusted? (multiplier 0.7 to 1.5)
   Consider: how binding are their obligations relative to their income?

3. How much should the SOPHISTICATION AXIS weight be adjusted? (multiplier 0.7 to 1.3)
   Consider: does their experience reflect genuine understanding or just time?

4. One sentence explaining your calibration.

Return ONLY valid JSON:

{{
  "risk_adjustment": integer from -30 to +30,
  "obligation_multiplier": float from 0.7 to 1.5,
  "sophistication_multiplier": float from 0.7 to 1.3,
  "reasoning": "one sentence"
}}

No markdown. JSON only.

Narrative:
Life situation: {life_summary}
Psychological state: {psychological_analysis}
Risk truth: {risk_truth}
Reliability: {reliability_assessment}
"""


def _query_axis_calibration(investor_state) -> dict:
    """
    Ask LLM: given this investor state, how should axis scores be calibrated?
    Reads compound_state — not raw flags or individual narrative fields.
    """
    defaults = {
        "risk_adjustment": 0,
        "obligation_multiplier": 1.0,
        "sophistication_multiplier": 1.0,
        "reasoning": "defaults used",
    }
    if investor_state is None:
        return defaults

    prompt = _AXIS_CALIBRATION_PROMPT.format(
        life_summary=getattr(investor_state, "state_description", ""),
        psychological_analysis=", ".join(getattr(investor_state, "dominant_factors", [])),
        risk_truth=", ".join(getattr(investor_state, "state_implications", [])),
        reliability_assessment=getattr(investor_state, "compound_state", ""),
    )

    payload = {
        "model":   LLM_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0, "num_predict": 256},
        "format":  "json",
    }

    for attempt in (1, 2):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=None,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                raw = json.loads(m.group(0))
                return {
                    "risk_adjustment":          max(-30, min(30, int(raw.get("risk_adjustment", 0)))),
                    "obligation_multiplier":    max(0.7, min(1.5, float(raw.get("obligation_multiplier", 1.0)))),
                    "sophistication_multiplier": max(0.7, min(1.3, float(raw.get("sophistication_multiplier", 1.0)))),
                    "reasoning":                raw.get("reasoning", ""),
                }
        except Exception:
            pass

    return defaults


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
    risk_adjustment: int = 0,
    investor_state=None,
    signals=None,
) -> tuple[int | None, list[str]]:
    """
    Axis 1: Risk Appetite.
    v14: reads loss_response from signals.behavior (not ctx.loss_reaction from extraction LLM).
    risk_adjustment: state-derived calibration (-30 to +30).
    """
    br = categories.behavioral_risk
    reasons = []

    if not br.data_available or br.score is None:
        reasons.append(
            "Behavioral data unavailable — Axis 1 (Risk) cannot be scored. "
            "Advisor should conduct behavioral assessment before recommending products."
        )
        return None, reasons

    # v14: prefer signals.behavior.loss_response over ctx.loss_reaction
    if signals is not None:
        lr = signals.behavior.loss_response
        reasons.append(f"Loss response from signal extraction: {lr}")
    else:
        lr = ctx.loss_reaction

    rb   = ctx.risk_behavior
    base = br.score or 50

    if lr == "panic":
        score = _clamp(15 + (base * 0.1))
        reasons.append(f"Panic loss response — risk appetite severely suppressed (score: {score})")
    elif lr == "cautious":
        score = _clamp(25 + (base * 0.2))
        reasons.append(f"Cautious loss response — moderate-low risk appetite (score: {score})")
    elif lr == "aggressive":
        score = _clamp(50 + (base * 0.4))
        reasons.append(f"Aggressive loss response — high risk appetite (score: {score})")
    elif rb == "low":
        score = _clamp(20 + (base * 0.15))
        reasons.append(f"Low risk behavior — conservative risk appetite (score: {score})")
    elif rb == "high":
        score = _clamp(55 + (base * 0.3))
        reasons.append(f"High risk behavior — elevated risk appetite (score: {score})")
    else:
        score = _clamp(35 + (base * 0.2))
        reasons.append(f"Moderate behavioral profile — balanced risk appetite (score: {score})")

    # Resilience modifier — from signals, never dropped
    if signals is not None and signals.behavior.resilience_level == "high":
        score = min(99, score + 5)
        reasons.append(
            f"Resilience modifier: +5 (evidence: {signals.behavior.resilience_evidence or 'high resilience detected'})"
        )

    # Apply state-derived adjustment
    if risk_adjustment != 0:
        before = score
        score  = _clamp(score + risk_adjustment)
        direction = "reduced" if risk_adjustment < 0 else "increased"
        compound = getattr(investor_state, "compound_state", "") if investor_state else ""
        reasons.append(
            f"State calibration ({compound!r}): risk score {direction} by "
            f"{abs(risk_adjustment)} pts ({before} → {score})."
        )
    elif ctx.grief_state and investor_state is None:
        score = _clamp(score - 15)
        reasons.append("Grief state (fallback): risk score reduced by 15 pts.")

    if ctx.recency_bias_risk:
        reasons.append(
            "RECENCY BIAS WARNING: High risk appetite score reflects inexperience with real losses."
        )

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
    obligation_multiplier: float = 1.0,
) -> tuple[int, list[str]]:
    """
    Axis 3: Financial Obligations.
    obligation_multiplier: narrative-derived (0.7-1.5).
    Replaces fixed weights 0.40/0.30/0.30 with narrative-calibrated combination.
    """
    debt = categories.debt_burden
    dep  = categories.dependency_load
    cult = categories.cultural_obligation
    reasons = []

    debt_score = debt.score if debt.score is not None else 5
    dep_score  = dep.score  if dep.score  is not None else 5
    cult_score = cult.score if cult.score is not None else 5

    reasons.append(f"Debt burden: {debt.label} (score: {debt_score}) — {debt.reason}")
    reasons.append(f"Dependency load: {dep.label} (score: {dep_score}) — {dep.reason}")
    reasons.append(f"Cultural obligation: {cult.label} (score: {cult_score}) — {cult.reason}")

    # Base weighted combination (fixed structural weights)
    combined = (debt_score * 0.40) + (dep_score * 0.30) + (cult_score * 0.30)

    # Apply narrative-derived multiplier
    if obligation_multiplier != 1.0:
        before   = combined
        combined = combined * obligation_multiplier
        reasons.append(
            f"Narrative obligation multiplier: {obligation_multiplier:.2f} "
            f"({before:.1f} → {combined:.1f}) — narrative indicates "
            f"{'higher' if obligation_multiplier > 1.0 else 'lower'} binding constraint."
        )

    all_modifiers = debt.modifiers + dep.modifiers + cult.modifiers
    for mod in all_modifiers:
        reasons.append(f"Modifier: {mod}")

    if ctx.emerging_constraint:
        reasons.append(
            "EMERGING CONSTRAINT: Current obligations are low but future commitments are high. "
            "Do not recommend illiquid products."
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
    sophistication_multiplier: float = 1.0,
    investor_state=None,
) -> tuple[int, list[str]]:
    """
    Axis 4: Investor Context / Sophistication.
    sophistication_multiplier: state-derived (0.7-1.3).
    Reads compound_state for peer/fragmentation signals, not raw flags.
    """
    score = 30
    reasons = []

    exp = ctx.experience_years
    if exp is not None:
        if exp >= 10:
            bonus = int(round(30 * sophistication_multiplier))
            score += bonus
            reasons.append(f"10+ years investment experience (+{bonus})")
        elif exp >= 5:
            bonus = int(round(20 * sophistication_multiplier))
            score += bonus
            reasons.append(f"{exp:.1f} years experience (+{bonus})")
        elif exp >= 2:
            bonus = int(round(10 * sophistication_multiplier))
            score += bonus
            reasons.append(f"{exp:.1f} years experience (+{bonus})")
        elif exp >= 1:
            bonus = int(round(5 * sophistication_multiplier))
            score += bonus
            reasons.append(f"{exp:.1f} years experience (+{bonus})")
        else:
            reasons.append("<1 year experience — novice investor")
    else:
        reasons.append("Experience not reported")

    fks = ctx.financial_knowledge_score
    if fks is not None:
        bonus = int(round((fks - 1) * 5 * sophistication_multiplier))
        score += bonus
        reasons.append(f"Financial knowledge score: {fks}/5 (+{bonus})")
    else:
        reasons.append("Financial knowledge not assessed")

    if sophistication_multiplier != 1.0:
        compound = getattr(investor_state, "compound_state", "") if investor_state else ""
        reasons.append(
            f"State sophistication multiplier: {sophistication_multiplier:.2f} "
            f"(state: {compound!r})"
        )

    da = ctx.decision_autonomy
    if da is True:
        score += 10
        reasons.append("Independent decision-making (+10)")
    elif da is False:
        score -= 5
        reasons.append("Peer/family-influenced decisions (−5)")

    if ctx.city_tier == "metro":
        score += 5
        reasons.append("Metro city — better access to financial products (+5)")
    elif ctx.city_tier == "tier2":
        reasons.append("Tier 2 city — standard access")

    # Read peer/fragmentation from compound_state, not raw flags
    compound = getattr(investor_state, "compound_state", "").lower() if investor_state else ""
    if compound and any(w in compound for w in ("peer", "speculator", "influenced", "herd")):
        score = max(1, score - 8)
        reasons.append(f"Compound state '{compound}' indicates peer-driven decisions (−8)")
    elif ctx.peer_driven and not compound:
        score = max(1, score - 8)
        reasons.append("Peer-driven decisions reduce effective sophistication (−8) [fallback flag]")

    if compound and any(w in compound for w in ("fragmented", "multiple advisor")):
        reasons.append(f"Compound state '{compound}' — fragmentation risk detected")
    elif ctx.fragmentation_risk and not compound:
        reasons.append("FRAGMENTATION RISK: Multiple advisors/platforms [fallback flag]")

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
    narrative=None,
    investor_state=None,
    signals=None,
) -> AxisScores:
    """
    Score all 4 axes from category assessments and profile context.
    v14: signals parameter routes loss_response from signal_extraction (not extraction LLM).
    investor_state is primary calibration input.
    """
    state_input = investor_state if investor_state is not None else narrative
    calib = _query_axis_calibration(state_input)

    risk_score,    risk_reasons    = score_axis1_risk(
        categories, ctx,
        risk_adjustment=calib["risk_adjustment"],
        investor_state=investor_state,
        signals=signals,
    )
    cashflow_score, cf_reasons     = score_axis2_cashflow(categories, ctx)
    oblig_score,   oblig_reasons   = score_axis3_obligations(
        categories, ctx, obligation_multiplier=calib["obligation_multiplier"]
    )
    context_score, context_reasons = score_axis4_context(
        categories, ctx,
        sophistication_multiplier=calib["sophistication_multiplier"],
        investor_state=investor_state,
    )

    capacity = compute_financial_capacity(cashflow_score, oblig_score)

    if calib.get("reasoning"):
        risk_reasons.append(f"Axis calibration: {calib['reasoning']}")

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
