"""
Decision Engine — InvestorDNA v2
LLM Call #3 of 3.

Input:  ExtractedFields + Signals + Scores
Output: DecisionOutput

Design:
- Archetype is ENUM-constrained (from scoring.py — deterministic)
- LLM generates: reasoning, advisor_note, first_step, reassessment_trigger
- LLM does NOT determine: archetype, equity range, allocation mode
- All allocation numbers come from deterministic scoring.equity_ceiling()
- Guardrails applied deterministically AFTER LLM call
"""
from __future__ import annotations

import importlib.util as _ilu
import os
import sys
from dataclasses import dataclass, field

# Resolve ExtractedFields from v2/extraction.py explicitly — avoids collision
# with v1/extraction.py when both directories are on sys.path
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

from signals import Signals
from scoring import Scores, equity_ceiling, ARCHETYPE_ENUM
from llm_adapter import llm_call

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class DecisionOutput:
    # Deterministic fields (from scoring)
    archetype:          str   # from ARCHETYPE_ENUM
    current_allocation: str   # e.g. "10-20%"
    baseline_allocation: str  # e.g. "20-35%"
    allocation_mode:    str   # normal|conservative|defensive|transitional
    equity_ceiling_pct: int
    dominant_trait:     str   # from signals

    # LLM-generated fields (narrative only — no numbers)
    reasoning:          str
    advisor_note:       str
    first_step:         str
    reassessment_trigger: str

    # Strategy
    primary_instrument: str
    sip_recommended:    bool

    # Meta
    confidence:         str   # high|medium|low
    fallback_used:      bool  = False
    warning:            str | None = None
    guardrail_notes:    list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Allocation range builder — deterministic
# ---------------------------------------------------------------------------

def _allocation_range(ceiling: int) -> str:
    if ceiling <= 0:   return "0%"
    if ceiling <= 10:  return f"0-{ceiling}%"
    floor = max(0, ceiling - 15)
    return f"{floor}-{ceiling}%"


def _allocation_mode(scores: Scores, signals: Signals) -> str:
    trait = signals.dominant_trait if signals.valid else "unknown"
    if trait in ("panic",) or scores.grief_state:
        return "defensive"
    if trait in ("cautious", "constrained") or scores.obligation > 70:
        return "conservative"
    if signals.valid and signals.has_shift:
        return "transitional"
    return "normal"


def _primary_instrument(ceiling: int, scores: Scores) -> str:
    if ceiling == 0:   return "liquid fund / FD"
    if ceiling <= 15:  return "liquid fund / short-duration debt"
    if ceiling <= 30:  return "balanced advantage fund / conservative hybrid"
    if ceiling <= 50:  return "flexi-cap / multi-cap fund"
    return "flexi-cap / mid-cap / international diversification"


def _baseline_ceiling(scores: Scores, signals: Signals) -> int:
    """Long-term ceiling once temporary state resolves."""
    if not signals.valid or not signals.has_shift:
        return equity_ceiling(scores)
    # If shift is temporary, baseline is higher than current
    if signals.shift_permanence == "likely_temporary":
        return min(99, equity_ceiling(scores) + 20)
    return equity_ceiling(scores)


# ---------------------------------------------------------------------------
# Decision prompt — Call #3
# ---------------------------------------------------------------------------

_DECISION_PROMPT = """\
You are a senior financial advisor. Generate a concise investment recommendation.

INVESTOR PROFILE (already computed — do NOT change these numbers):
Archetype: {archetype}
Current equity allocation: {current_allocation}
Dominant trait: {dominant_trait}
Risk score: {risk}/99
Cashflow score: {cashflow}/99
Obligation score: {obligation}/99
Sophistication score: {context}/99
Financial capacity: {capacity}/99
Grief state: {grief}
Signals summary: {signals_summary}

Your task — write ONLY these 4 fields. Do NOT suggest different numbers.

Return ONLY valid JSON:
{{
  "reasoning": "3-4 sentences explaining WHY this allocation fits this investor. Reference dominant_trait and key scores.",
  "advisor_note": "The single most important caution for the advisor. One sentence.",
  "first_step": "One concrete actionable first step with a specific amount or percentage.",
  "reassessment_trigger": "Specific condition that should trigger a reassessment (not just 'annual review')."
}}

No markdown. JSON only.
"""


def _signals_summary(f: ExtractedFields, s: Signals, sc: Scores) -> str:
    parts = []
    if s.valid:
        parts.append(f"loss_response={s.loss_response}")
        parts.append(f"resilience={s.resilience_level}")
        parts.append(f"constraint={s.constraint_level}")
        if s.life_event_type != "none":
            parts.append(f"life_event={s.life_event_type}({s.life_event_recency})")
        if s.cultural_obligation:
            parts.append(f"cultural_obligation=yes")
        if s.has_shift:
            parts.append(f"behavioral_shift={s.shift_permanence}")
        if s.contradiction_note:
            parts.append(f"contradiction: {s.contradiction_note[:80]}")
    if f.monthly_income:
        parts.append(f"income=₹{f.monthly_income:,.0f}/mo")
    if f.emi_ratio:
        parts.append(f"emi_ratio={f.emi_ratio:.1f}%")
    if f.emergency_months is not None:
        parts.append(f"emergency={f.emergency_months:.1f}mo")
    return " | ".join(parts) or "minimal data available"


def _run_decision_llm(f: ExtractedFields, s: Signals, sc: Scores,
                      current_alloc: str) -> dict:
    prompt = _DECISION_PROMPT.format(
        archetype=sc.archetype,
        current_allocation=current_alloc,
        dominant_trait=s.dominant_trait if s.valid else "unknown",
        risk=sc.risk if sc.risk is not None else "N/A",
        cashflow=sc.cashflow, obligation=sc.obligation,
        context=sc.context, capacity=sc.capacity,
        grief="yes" if sc.grief_state else "no",
        signals_summary=_signals_summary(f, s, sc),
    )
    for attempt in (1, 2):
        try:
            raw = llm_call(prompt, num_predict=512)
            if raw.get("reasoning") and raw.get("advisor_note"):
                return raw
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# Guardrails — deterministic, applied after LLM
# ---------------------------------------------------------------------------

def _apply_guardrails(decision: DecisionOutput, scores: Scores, signals: Signals) -> list[str]:
    """
    Hard constraint enforcement. Returns list of notes for any adjustments made.
    Mutates decision in-place.
    """
    notes: list[str] = []
    ceiling = scores.capacity  # use capacity as the hard ceiling

    def _parse_upper(s: str) -> int | None:
        import re
        m = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*%?", s)
        if m: return int(m.group(2))
        m = re.search(r"(\d+)\s*%?", s)
        if m: return int(m.group(1))
        return None

    upper = _parse_upper(decision.current_allocation)

    # R1: panic/grief → max 20%
    if (scores.grief_state or (signals.valid and signals.dominant_trait == "panic")) and upper and upper > 20:
        decision.current_allocation = f"0-20%"
        notes.append("R1: grief/panic → current_allocation capped at 20%")

    # R2: cautious/constrained → max 30%
    elif signals.valid and signals.dominant_trait in ("cautious", "constrained") and upper and upper > 30:
        decision.current_allocation = f"0-30%"
        notes.append("R2: cautious/constrained → current_allocation capped at 30%")

    # R3: obligation > 70 → max 20%
    if scores.obligation > 70:
        upper2 = _parse_upper(decision.current_allocation)
        if upper2 and upper2 > 20:
            decision.current_allocation = "0-20%"
            notes.append("R3: obligation>70 → current_allocation capped at 20%")

    # R4: transitional → baseline must exceed current
    if signals.valid and signals.has_shift:
        cur_u  = _parse_upper(decision.current_allocation) or 0
        base_u = _parse_upper(decision.baseline_allocation) or 0
        if base_u <= cur_u:
            decision.baseline_allocation = f"{cur_u+5}-{cur_u+20}%"
            notes.append("R4: shift detected → baseline bumped above current")

    # R5: allocation_mode sync
    cur_u  = _parse_upper(decision.current_allocation) or 0
    base_u = _parse_upper(decision.baseline_allocation) or 0
    if cur_u != base_u and decision.allocation_mode == "normal":
        decision.allocation_mode = "transitional"
        notes.append("R5: current≠baseline → allocation_mode set to transitional")

    return notes


# ---------------------------------------------------------------------------
# Fallback decision — used when LLM fails entirely
# ---------------------------------------------------------------------------

def _fallback_decision(scores: Scores, signals: Signals) -> DecisionOutput:
    ceiling = equity_ceiling(scores)
    current = _allocation_range(ceiling)
    return DecisionOutput(
        archetype=scores.archetype,
        current_allocation=current,
        baseline_allocation=current,
        allocation_mode=_allocation_mode(scores, signals),
        equity_ceiling_pct=ceiling,
        dominant_trait=signals.dominant_trait if signals.valid else "unknown",
        reasoning="Provisional recommendation based on profile data. Advisor review required.",
        advisor_note="Manual advisor assessment required — automated reasoning unavailable.",
        first_step="Consult a financial advisor before making any investment decisions.",
        reassessment_trigger="Once advisor assessment is complete.",
        primary_instrument=_primary_instrument(ceiling, scores),
        sip_recommended=ceiling >= 20,
        confidence="low",
        fallback_used=True,
        warning="LLM reasoning unavailable — conservative defaults applied.",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_decision(f: ExtractedFields, s: Signals, sc: Scores) -> DecisionOutput:
    """
    LLM Call #3 — generate reasoning narrative for a pre-computed allocation.

    The allocation numbers are ALREADY determined by scoring.py (deterministic).
    The LLM only writes the explanation.
    """
    ceiling      = equity_ceiling(sc)
    base_ceiling = _baseline_ceiling(sc, s)
    current      = _allocation_range(ceiling)
    baseline     = _allocation_range(base_ceiling)
    mode         = _allocation_mode(sc, s)
    instrument   = _primary_instrument(ceiling, sc)
    sip          = ceiling >= 20

    # Confidence from data completeness
    if f.data_completeness >= 70:
        confidence = "high"
    elif f.data_completeness >= 40:
        confidence = "medium"
    else:
        confidence = "low"

    # LLM call for narrative only
    raw = _run_decision_llm(f, s, sc, current)

    if not raw:
        d = _fallback_decision(sc, s)
        d.baseline_allocation = baseline
        return d

    decision = DecisionOutput(
        archetype=sc.archetype,
        current_allocation=current,
        baseline_allocation=baseline,
        allocation_mode=mode,
        equity_ceiling_pct=ceiling,
        dominant_trait=s.dominant_trait if s.valid else "unknown",
        reasoning=raw.get("reasoning", "Reasoning unavailable."),
        advisor_note=raw.get("advisor_note", "No specific advisor note."),
        first_step=raw.get("first_step", "Consult advisor for first step."),
        reassessment_trigger=raw.get("reassessment_trigger", "Standard annual review."),
        primary_instrument=instrument,
        sip_recommended=sip,
        confidence=confidence,
        fallback_used=False,
    )

    # Apply deterministic guardrails
    notes = _apply_guardrails(decision, sc, s)
    decision.guardrail_notes = notes

    return decision


def decision_to_dict(d: DecisionOutput) -> dict:
    return {
        "archetype":           d.archetype,
        "current_allocation":  d.current_allocation,
        "baseline_allocation": d.baseline_allocation,
        "allocation_mode":     d.allocation_mode,
        "equity_ceiling_pct":  d.equity_ceiling_pct,
        "dominant_trait":      d.dominant_trait,
        "reasoning":           d.reasoning,
        "advisor_note":        d.advisor_note,
        "first_step":          d.first_step,
        "reassessment_trigger": d.reassessment_trigger,
        "primary_instrument":  d.primary_instrument,
        "sip_recommended":     d.sip_recommended,
        "confidence":          d.confidence,
        "fallback_used":       d.fallback_used,
        "warning":             d.warning,
        "guardrail_notes":     d.guardrail_notes,
    }
