"""
Scoring Engine — InvestorDNA v2
FULLY DETERMINISTIC. Zero LLM calls.

All scores derived from ExtractedFields + Signals using explicit rules.
Same input → same scores, always.

Axes (1-99):
  risk       — behavioral risk appetite
  cashflow   — income stability
  obligation — financial burden
  context    — sophistication / experience
  capacity   — cashflow × (1 - obligation/100)  [derived]

Archetype assigned deterministically from risk × context matrix.
"""
from __future__ import annotations

import importlib.util as _ilu
import os
import sys
from dataclasses import dataclass

# Resolve ExtractedFields from v2/extraction.py explicitly — avoids collision
# with v1/extraction.py when both directories are on sys.path.
_v2_dir = os.path.dirname(os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location(
    "investor_profiler.v2.extraction",
    os.path.join(_v2_dir, "extraction.py"),
)
_v2_extraction_mod = sys.modules.get("investor_profiler.v2.extraction")
if _v2_extraction_mod is None:
    _v2_extraction_mod = _ilu.module_from_spec(_spec)
    sys.modules["investor_profiler.v2.extraction"] = _v2_extraction_mod
    _spec.loader.exec_module(_v2_extraction_mod)
ExtractedFields = _v2_extraction_mod.ExtractedFields
ExtractedFields = _v2_extraction_mod.ExtractedFields

from signals import Signals


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class Scores:
    risk:             int | None   # None = no behavioral data
    cashflow:         int
    obligation:       int
    context:          int
    capacity:         int          # derived

    archetype:        str          # from ARCHETYPE_ENUM
    archetype_reason: str

    risk_reasons:       list[str]
    cashflow_reasons:   list[str]
    obligation_reasons: list[str]
    context_reasons:    list[str]

    # Flags for downstream use
    grief_state:        bool
    recency_bias:       bool
    peer_driven:        bool
    emerging_constraint: bool
    hidden_obligation:  bool


ARCHETYPE_ENUM = frozenset({
    "Strategist",          # high sophistication + low loss aversion
    "Analyst",             # high sophistication + high loss aversion
    "Explorer",            # low sophistication + low loss aversion (naive risk)
    "Guardian",            # low sophistication + high loss aversion
    "Constrained Builder", # high obligation burden (overrides matrix)
    "Transitional",        # behavioral shift detected
    "Crisis Mode",         # grief/crisis state
    "Provisional",         # insufficient data
})


# ---------------------------------------------------------------------------
# Clamp helper
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: int = 1, hi: int = 99) -> int:
    return max(lo, min(hi, int(round(v))))


# ---------------------------------------------------------------------------
# Axis 1: Risk
# Source: signals.loss_response + signals.dominant_trait + fields.risk_behavior
# Fully deterministic — no LLM multipliers.
# ---------------------------------------------------------------------------

def _score_risk(f: ExtractedFields, s: Signals) -> tuple[int | None, list[str]]:
    reasons: list[str] = []

    # No behavioral data at all
    if f.loss_reaction is None and f.risk_behavior is None:
        reasons.append("No behavioral data — risk axis cannot be scored.")
        return None, reasons

    # Base from loss_reaction (signals take precedence over extraction field)
    lr = s.loss_response if s.valid else f.loss_reaction

    if lr == "panic":
        base = 12
        reasons.append("Panic loss response — risk appetite severely suppressed.")
    elif lr == "cautious":
        base = 28
        reasons.append("Cautious loss response — moderate-low risk appetite.")
    elif lr == "aggressive":
        base = 68
        reasons.append("Aggressive loss response — high risk appetite.")
    elif lr == "neutral":
        base = 45
        reasons.append("Neutral loss response — balanced risk appetite.")
    else:
        # Fall back to risk_behavior
        rb = f.risk_behavior
        if rb == "low":
            base = 20
            reasons.append("Low risk behavior stated.")
        elif rb == "high":
            base = 60
            reasons.append("High risk behavior stated.")
        else:
            base = 40
            reasons.append("Moderate risk behavior stated.")

    score = base

    # Resilience modifier (+5 for high, -5 for low) — deterministic from signals
    if s.valid:
        if s.resilience_level == "high":
            score = min(99, score + 5)
            reasons.append("High resilience: +5.")
        elif s.resilience_level == "low":
            score = max(1, score - 5)
            reasons.append("Low resilience: -5.")

    # Grief state: hard cap at 20
    grief = s.valid and s.life_event_type == "death" and s.life_event_recency == "recent"
    if grief:
        score = min(score, 20)
        reasons.append("Recent bereavement: risk capped at 20.")

    # Dominant trait override (from contradiction resolution)
    if s.valid and s.dominant_trait == "panic" and score > 20:
        score = min(score, 20)
        reasons.append("Dominant trait=panic: risk capped at 20.")
    elif s.valid and s.dominant_trait == "cautious" and score > 35:
        score = min(score, 35)
        reasons.append("Dominant trait=cautious: risk capped at 35.")

    # Recency bias warning (no score change — just flag)
    recency = (f.experience_years is not None and f.experience_years < 1
               and f.risk_behavior == "high"
               and (f.financial_knowledge_score is None or f.financial_knowledge_score <= 2))
    if recency:
        reasons.append("RECENCY BIAS: High risk appetite with <1yr experience — may not be stable.")

    return _clamp(score), reasons


# ---------------------------------------------------------------------------
# Axis 2: Cashflow
# Source: income_type + monthly_income + emergency_months
# ---------------------------------------------------------------------------

def _score_cashflow(f: ExtractedFields, s: Signals) -> tuple[int, list[str]]:
    reasons: list[str] = []
    score = 50

    # Income type base
    if f.income_type == "salaried":
        score += 20
        reasons.append("Salaried: stable income base.")
    elif f.income_type == "business":
        score += 5
        reasons.append("Business income: moderate stability.")
    elif f.income_type == "gig":
        score -= 20
        reasons.append("Gig income: high volatility.")
    else:
        score -= 5
        reasons.append("Income type unknown: conservative adjustment.")

    # Income level
    if f.monthly_income is not None:
        if f.monthly_income >= 150_000:
            score += 15; reasons.append(f"High income ₹{f.monthly_income:,.0f}/mo: +15.")
        elif f.monthly_income >= 75_000:
            score += 8;  reasons.append(f"Moderate income ₹{f.monthly_income:,.0f}/mo: +8.")
        elif f.monthly_income >= 30_000:
            score += 2;  reasons.append(f"Low-moderate income: +2.")
        else:
            score -= 10; reasons.append(f"Low income ₹{f.monthly_income:,.0f}/mo: -10.")

    # Emergency fund
    if f.emergency_months is not None:
        if f.emergency_months >= 6:
            score += 10; reasons.append(f"{f.emergency_months:.1f}mo emergency fund: +10.")
        elif f.emergency_months >= 3:
            score += 5;  reasons.append(f"{f.emergency_months:.1f}mo emergency fund: +5.")
        elif f.emergency_months < 1:
            score -= 10; reasons.append("No emergency fund: -10.")

    # Financial pressure from signals
    if s.valid:
        if s.financial_pressure == "high":
            score -= 10; reasons.append("High financial pressure (signals): -10.")
        elif s.financial_pressure == "low":
            score += 5;  reasons.append("Low financial pressure (signals): +5.")

    # Recent crisis depletes buffer
    if s.valid and s.life_event_type in ("crisis", "death") and s.life_event_recency == "recent":
        score -= 8; reasons.append("Recent crisis/bereavement: -8.")

    return _clamp(score), reasons


# ---------------------------------------------------------------------------
# Axis 3: Obligation
# Source: emi_ratio + dependents + near_term_obligation + cultural signals
# ---------------------------------------------------------------------------

def _score_obligation(f: ExtractedFields, s: Signals) -> tuple[int, list[str]]:
    reasons: list[str] = []
    score = 5

    # EMI burden
    if f.emi_ratio is not None:
        if f.emi_ratio >= 60:
            score += 75; reasons.append(f"EMI ratio {f.emi_ratio:.1f}%: critical burden.")
        elif f.emi_ratio >= 40:
            score += 55; reasons.append(f"EMI ratio {f.emi_ratio:.1f}%: high burden.")
        elif f.emi_ratio >= 20:
            score += 30; reasons.append(f"EMI ratio {f.emi_ratio:.1f}%: moderate burden.")
        elif f.emi_ratio > 0:
            score += 10; reasons.append(f"EMI ratio {f.emi_ratio:.1f}%: low burden.")
    elif f.emi_amount is not None:
        score += 15; reasons.append("EMI present, income unknown: partial burden.")

    # Dependents
    dep = f.dependents or 0
    if dep >= 4:
        score += 30; reasons.append(f"{dep} dependents: very high load.")
    elif dep >= 3:
        score += 22; reasons.append(f"{dep} dependents: high load.")
    elif dep == 2:
        score += 14; reasons.append(f"{dep} dependents: moderate load.")
    elif dep == 1:
        score += 7;  reasons.append("1 dependent: low-moderate load.")

    # Near-term obligation
    if f.near_term_obligation_level == "high":
        score += 20; reasons.append(f"High near-term obligation ({f.obligation_type or 'unspecified'}).")
    elif f.near_term_obligation_level == "moderate":
        score += 10; reasons.append(f"Moderate near-term obligation ({f.obligation_type or 'unspecified'}).")

    # Cultural / hidden obligations from signals
    if s.valid:
        if s.cultural_obligation:
            score += 10; reasons.append(f"Cultural obligation: {s.cultural_obligation[:60]}.")
        if s.hidden_obligations:
            score += 8;  reasons.append("Hidden obligations detected.")
        if s.constraint_level == "high":
            score += 10; reasons.append("High financial constraint (signals): +10.")

    # Provider role
    if s.valid and s.provider_role == "primary" and dep > 0:
        score += 8; reasons.append("Primary provider with dependents: +8.")

    return _clamp(score), reasons


# ---------------------------------------------------------------------------
# Axis 4: Context / Sophistication
# Source: experience_years + financial_knowledge_score + decision_autonomy + signals
# ---------------------------------------------------------------------------

def _score_context(f: ExtractedFields, s: Signals) -> tuple[int, list[str]]:
    reasons: list[str] = []
    score = 25

    # Experience
    exp = f.experience_years
    if exp is not None:
        if exp >= 10:
            score += 30; reasons.append(f"10+ years experience: +30.")
        elif exp >= 5:
            score += 20; reasons.append(f"{exp:.1f}yr experience: +20.")
        elif exp >= 2:
            score += 12; reasons.append(f"{exp:.1f}yr experience: +12.")
        elif exp >= 1:
            score += 6;  reasons.append(f"{exp:.1f}yr experience: +6.")
        else:
            reasons.append("<1yr experience: novice.")
    else:
        reasons.append("Experience not reported.")

    # Financial knowledge
    fks = f.financial_knowledge_score
    if fks is not None:
        bonus = (fks - 1) * 5
        score += bonus; reasons.append(f"Knowledge score {fks}/5: +{bonus}.")

    # Decision autonomy
    if f.decision_autonomy is True:
        score += 10; reasons.append("Independent decision-making: +10.")
    elif f.decision_autonomy is False:
        score -= 5;  reasons.append("Peer/family-influenced decisions: -5.")

    # Peer influence from signals
    if s.valid:
        if s.peer_influence == "high":
            score -= 8; reasons.append("High peer influence (signals): -8.")
        elif s.peer_influence == "medium":
            score -= 3; reasons.append("Medium peer influence (signals): -3.")
        if s.analytical == "high":
            score += 5; reasons.append("High analytical tendency: +5.")

    return _clamp(score), reasons


# ---------------------------------------------------------------------------
# Capacity (derived)
# ---------------------------------------------------------------------------

def _capacity(cashflow: int, obligation: int) -> int:
    return _clamp(cashflow * (1 - obligation / 100))


# ---------------------------------------------------------------------------
# Archetype assignment — deterministic matrix
# ---------------------------------------------------------------------------

def _assign_archetype(
    risk: int | None,
    context: int,
    obligation: int,
    grief: bool,
    has_shift: bool,
    capacity: int,
) -> tuple[str, str]:
    # Crisis Mode: grief overrides everything
    if grief:
        return "Crisis Mode", "Investor is navigating a grief/crisis state. Current preferences are unreliable."

    # Transitional: behavioral shift detected
    if has_shift:
        return "Transitional", "Investor is in a behavioral transition. Current allocation should be conservative."

    # Constrained Builder: obligation burden dominates
    if obligation >= 70 or capacity < 20:
        return "Constrained Builder", "High obligation burden limits investable surplus. Stability before growth."

    # Provisional: no behavioral data
    if risk is None:
        return "Provisional", "Insufficient behavioral data to assign archetype. Advisor assessment required."

    high_soph = context >= 55
    low_loss  = risk >= 55

    if high_soph and low_loss:
        return "Strategist", "Pursues growth with informed confidence. Suitable for equity-heavy portfolios."
    elif high_soph and not low_loss:
        return "Analyst", "High analytical capability, conservative risk orientation. Deliberate allocation decisions."
    elif not high_soph and low_loss:
        return "Explorer", "Open to opportunity but low sophistication. Naive risk-taking, not informed confidence."
    else:
        return "Guardian", "Protects financial foundation with care. Education and gradual exposure recommended."


# ---------------------------------------------------------------------------
# Equity ceiling — deterministic from scores + flags
# ---------------------------------------------------------------------------

def equity_ceiling(scores: "Scores") -> int:
    """
    Deterministic equity ceiling from scores and flags.
    No LLM involved.
    """
    cap = 65  # start permissive, apply constraints

    # Capacity anchor
    if scores.capacity < 20:   cap = min(cap, 5)
    elif scores.capacity < 35: cap = min(cap, 15)
    elif scores.capacity < 50: cap = min(cap, 30)
    elif scores.capacity < 65: cap = min(cap, 45)

    # Risk anchor
    if scores.risk is not None:
        cap = min(cap, scores.risk)

    # Hard caps from flags
    if scores.grief_state:    cap = min(cap, 15)
    if scores.recency_bias:   cap = min(cap, 20)
    if scores.context < 35:   cap = min(cap, 25)
    if scores.obligation > 70: cap = min(cap, 20)

    return max(0, cap)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_scores(f: ExtractedFields, s: Signals) -> Scores:
    """
    Fully deterministic scoring. No LLM calls.
    Same ExtractedFields + Signals → same Scores, always.
    """
    grief = s.valid and s.life_event_type == "death" and s.life_event_recency == "recent"
    recency = (f.experience_years is not None and f.experience_years < 1
               and f.risk_behavior == "high"
               and (f.financial_knowledge_score is None or f.financial_knowledge_score <= 2))
    peer = (f.decision_autonomy is False
            or (s.valid and s.peer_influence in ("medium", "high")))
    hidden = bool(s.valid and s.hidden_obligations)
    emerging = (
        (f.emi_ratio is None or f.emi_ratio < 20)
        and (f.dependents or 0) < 2
        and s.valid and s.primary_intent in ("housing", "business", "wedding")
        and s.intent_timeline == "near"
    )

    risk_score,    risk_r    = _score_risk(f, s)
    cashflow_score, cf_r     = _score_cashflow(f, s)
    oblig_score,   oblig_r   = _score_obligation(f, s)
    context_score, context_r = _score_context(f, s)
    cap_score                = _capacity(cashflow_score, oblig_score)

    archetype, arch_reason = _assign_archetype(
        risk_score, context_score, oblig_score,
        grief, s.valid and s.has_shift, cap_score,
    )

    return Scores(
        risk=risk_score, cashflow=cashflow_score,
        obligation=oblig_score, context=context_score, capacity=cap_score,
        archetype=archetype, archetype_reason=arch_reason,
        risk_reasons=risk_r, cashflow_reasons=cf_r,
        obligation_reasons=oblig_r, context_reasons=context_r,
        grief_state=grief, recency_bias=recency,
        peer_driven=peer, emerging_constraint=emerging,
        hidden_obligation=hidden,
    )


def scores_to_dict(sc: Scores) -> dict:
    return {
        "risk": sc.risk, "cashflow": sc.cashflow,
        "obligation": sc.obligation, "context": sc.context, "capacity": sc.capacity,
        "archetype": sc.archetype, "archetype_reason": sc.archetype_reason,
        "flags": {
            "grief_state": sc.grief_state, "recency_bias": sc.recency_bias,
            "peer_driven": sc.peer_driven, "emerging_constraint": sc.emerging_constraint,
            "hidden_obligation": sc.hidden_obligation,
        },
        "reasons": {
            "risk": sc.risk_reasons, "cashflow": sc.cashflow_reasons,
            "obligation": sc.obligation_reasons, "context": sc.context_reasons,
        },
    }
