"""
Decision Guardrails — InvestorDNA v15
Hard constraint enforcement on LLM-proposed decisions.

Architecture:
  DecisionOutput + InvestorState + SignalOutput → DecisionOutput (adjusted)

Design principle:
  "The LLM reasons freely. The guardrails ensure safety.
   No rejection — only adjustment with full audit trail."

Rules are deterministic. They fire on structured state fields, not on
narrative text. Each adjustment records before/after/rule/reason.

RULES (in priority order):
  R1 — PANIC DOMINANCE:       dominant_trait == "panic"  → current_allocation ≤ 20%
  R2 — CRISIS STATE:          compound_state contains crisis signals → current_allocation ≤ 10%
  R3 — HIGH OBLIGATION + LOW RESILIENCE: constraint_level == "high" AND resilience_level == "low" → current_allocation ≤ 15%
  R4 — TRANSITIONAL ORDERING: shift_detected == True → current_allocation < baseline_allocation
  R5 — TEMPORARY ORDERING:    is_temporary == True → baseline_allocation > current_allocation
  R6 — ALLOCATION MODE SYNC:  current ≠ baseline → allocation_mode must be "transitional"
  R7 — GRIEF STATE:           grief life event (recent) → current_allocation ≤ 15%
  R8 — PEER SPECULATOR:       peer_influence == "high" AND resilience_level == "low" → current_allocation ≤ 25%
  R9 — RECENCY BIAS:          shift_detected AND shift_permanence == "likely_temporary" AND baseline > 40% → cap baseline at 40%
"""

import re
from dataclasses import dataclass, field
from decision_engine import DecisionOutput, StateContext, TemporalStrategy, RiskAssessment


# ---------------------------------------------------------------------------
# Guardrail adjustment record
# ---------------------------------------------------------------------------

@dataclass
class GuardrailAdjustment:
    rule: str           # rule identifier, e.g. "R1_PANIC_DOMINANCE"
    field: str          # which field was adjusted: "current_allocation" | "baseline_allocation" | "allocation_mode"
    before: str         # value before adjustment
    after: str          # value after adjustment
    reason: str         # human-readable explanation


# ---------------------------------------------------------------------------
# Allocation range parsing and capping
# ---------------------------------------------------------------------------

def _parse_upper(allocation: str) -> int | None:
    """
    Extract the upper bound from an allocation string.
    "10-20%" → 20,  "0%" → 0,  "30-40%" → 40,  "20%" → 20
    Returns None if unparseable.
    """
    if not allocation:
        return None
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*%?", allocation)
    if m:
        return int(m.group(2))
    m = re.search(r"(\d+)\s*%?", allocation)
    if m:
        return int(m.group(1))
    return None


def _parse_lower(allocation: str) -> int | None:
    """Extract the lower bound from an allocation string."""
    if not allocation:
        return None
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*%?", allocation)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*%?", allocation)
    if m:
        return int(m.group(1))
    return None


def _cap_allocation(allocation: str, cap: int) -> str:
    """
    Cap an allocation string so its upper bound does not exceed `cap`.
    "30-40%" capped at 20 → "0-20%"
    "15-25%" capped at 20 → "0-20%"
    "10-20%" capped at 20 → "10-20%"  (no change)
    "25%"    capped at 20 → "20%"
    "0%"     capped at 20 → "0%"      (no change)
    """
    upper = _parse_upper(allocation)
    if upper is None or upper <= cap:
        return allocation

    # Check if it's a range or a single value
    is_range = bool(re.search(r"\d+\s*[-–]\s*\d+", allocation))
    if not is_range:
        # Single value like "25%" → just cap it
        return f"{cap}%"

    lower = _parse_lower(allocation) or 0
    # If lower is also above cap, reset to 0; if lower straddles cap, also reset to 0
    new_lower = lower if lower < cap else 0
    # If the range straddles the cap (lower < cap but upper > cap), reset lower to 0
    # so the range clearly communicates the cap is the ceiling
    if lower > 0 and lower < cap and upper > cap:
        new_lower = 0
    new_upper = cap

    if new_lower == new_upper:
        return f"{new_upper}%"
    return f"{new_lower}-{new_upper}%"


def _ensure_baseline_gt_current(current: str, baseline: str) -> str:
    """
    Ensure baseline_allocation upper bound > current_allocation upper bound.
    If not, bump baseline up by 10 percentage points.
    """
    cur_upper  = _parse_upper(current)  or 0
    base_upper = _parse_upper(baseline) or 0

    if base_upper > cur_upper:
        return baseline  # already correct

    # Bump baseline: new range starts at cur_upper+5, ends at cur_upper+15
    new_lower = cur_upper + 5
    new_upper = cur_upper + 15
    return f"{new_lower}-{new_upper}%"


# ---------------------------------------------------------------------------
# Individual rule evaluators
# ---------------------------------------------------------------------------

def _r1_panic_dominance(
    decision: DecisionOutput,
    investor_state,
    signals,
    adjustments: list[GuardrailAdjustment],
) -> DecisionOutput:
    """R1: If dominant_trait is panic, current_allocation must be ≤ 20%."""
    dominant = (investor_state.dominant_trait or "").lower() if investor_state else ""
    if dominant != "panic":
        return decision

    cap = 20
    upper = _parse_upper(decision.current_allocation)
    if upper is not None and upper <= cap:
        return decision

    before = decision.current_allocation
    after  = _cap_allocation(before, cap)
    adjustments.append(GuardrailAdjustment(
        rule="R1_PANIC_DOMINANCE",
        field="current_allocation",
        before=before,
        after=after,
        reason=(
            f"dominant_trait='{dominant}' — panic-driven investors must not exceed "
            f"{cap}% equity in current allocation regardless of stated preference."
        ),
    ))
    decision.current_allocation = after
    return decision


def _r2_crisis_state(
    decision: DecisionOutput,
    investor_state,
    signals,
    adjustments: list[GuardrailAdjustment],
) -> DecisionOutput:
    """R2: If compound_state contains crisis signals, current_allocation ≤ 10%."""
    compound = (investor_state.compound_state or "").lower() if investor_state else ""
    crisis_words = ("crisis", "survival mode", "collapsed", "insolvent", "bankruptcy")
    if not any(w in compound for w in crisis_words):
        # Also check life events
        if signals is None:
            return decision
        has_crisis = any(
            ev.type == "crisis" and ev.recency == "recent"
            for ev in signals.life_events
        )
        if not has_crisis:
            return decision

    cap = 10
    upper = _parse_upper(decision.current_allocation)
    if upper is not None and upper <= cap:
        return decision

    before = decision.current_allocation
    after  = _cap_allocation(before, cap)
    adjustments.append(GuardrailAdjustment(
        rule="R2_CRISIS_STATE",
        field="current_allocation",
        before=before,
        after=after,
        reason=(
            f"Crisis state detected (compound_state='{compound}') — "
            f"equity exposure must not exceed {cap}% during active crisis."
        ),
    ))
    decision.current_allocation = after
    return decision


def _r3_high_obligation_low_resilience(
    decision: DecisionOutput,
    investor_state,
    signals,
    adjustments: list[GuardrailAdjustment],
) -> DecisionOutput:
    """R3: constraint_level == "high" AND resilience_level == "low" → current_allocation ≤ 15%."""
    if signals is None or investor_state is None:
        return decision

    constraint  = signals.financial_state.constraint_level
    resilience  = investor_state.resilience_level

    if not (constraint == "high" and resilience == "low"):
        return decision

    cap = 15
    upper = _parse_upper(decision.current_allocation)
    if upper is not None and upper <= cap:
        return decision

    before = decision.current_allocation
    after  = _cap_allocation(before, cap)
    adjustments.append(GuardrailAdjustment(
        rule="R3_HIGH_OBLIGATION_LOW_RESILIENCE",
        field="current_allocation",
        before=before,
        after=after,
        reason=(
            f"constraint_level='{constraint}' AND resilience_level='{resilience}' — "
            f"severely constrained investor with low resilience must not exceed {cap}% equity."
        ),
    ))
    decision.current_allocation = after
    return decision


def _r4_transitional_ordering(
    decision: DecisionOutput,
    investor_state,
    signals,
    adjustments: list[GuardrailAdjustment],
) -> DecisionOutput:
    """R4: shift_detected == True → current_allocation < baseline_allocation."""
    if investor_state is None or not investor_state.shift_detected:
        return decision

    cur_upper  = _parse_upper(decision.current_allocation)  or 0
    base_upper = _parse_upper(decision.baseline_allocation) or 0

    if cur_upper < base_upper:
        return decision  # already correct

    before = decision.baseline_allocation
    after  = _ensure_baseline_gt_current(decision.current_allocation, decision.baseline_allocation)
    adjustments.append(GuardrailAdjustment(
        rule="R4_TRANSITIONAL_ORDERING",
        field="baseline_allocation",
        before=before,
        after=after,
        reason=(
            "shift_detected=True — baseline_allocation must exceed current_allocation "
            "to reflect true long-term capacity beyond the current transitional state."
        ),
    ))
    decision.baseline_allocation = after
    return decision


def _r5_temporary_ordering(
    decision: DecisionOutput,
    investor_state,
    signals,
    adjustments: list[GuardrailAdjustment],
) -> DecisionOutput:
    """R5: is_temporary == True → baseline_allocation > current_allocation."""
    if not decision.temporal_strategy.is_temporary:
        return decision

    cur_upper  = _parse_upper(decision.current_allocation)  or 0
    base_upper = _parse_upper(decision.baseline_allocation) or 0

    if base_upper > cur_upper:
        return decision  # already correct

    before = decision.baseline_allocation
    after  = _ensure_baseline_gt_current(decision.current_allocation, decision.baseline_allocation)
    adjustments.append(GuardrailAdjustment(
        rule="R5_TEMPORARY_ORDERING",
        field="baseline_allocation",
        before=before,
        after=after,
        reason=(
            "temporal_strategy.is_temporary=True — baseline_allocation must exceed "
            "current_allocation to represent the investor's capacity once the temporary "
            "state resolves."
        ),
    ))
    decision.baseline_allocation = after
    return decision


def _r6_allocation_mode_sync(
    decision: DecisionOutput,
    investor_state,
    signals,
    adjustments: list[GuardrailAdjustment],
) -> DecisionOutput:
    """R6: If current_allocation ≠ baseline_allocation, allocation_mode must be 'transitional'."""
    cur_upper  = _parse_upper(decision.current_allocation)  or 0
    base_upper = _parse_upper(decision.baseline_allocation) or 0

    if cur_upper == base_upper:
        return decision  # same — mode can be static

    if decision.allocation_mode == "transitional":
        return decision  # already correct

    before = decision.allocation_mode
    after  = "transitional"
    adjustments.append(GuardrailAdjustment(
        rule="R6_ALLOCATION_MODE_SYNC",
        field="allocation_mode",
        before=before,
        after=after,
        reason=(
            f"current_allocation ({decision.current_allocation}) ≠ "
            f"baseline_allocation ({decision.baseline_allocation}) — "
            "allocation_mode must be 'transitional' when current and baseline differ."
        ),
    ))
    decision.allocation_mode = after
    return decision


def _r7_grief_state(
    decision: DecisionOutput,
    investor_state,
    signals,
    adjustments: list[GuardrailAdjustment],
) -> DecisionOutput:
    """R7: Recent grief life event → cap based on resilience_level."""
    if signals is None:
        return decision

    has_recent_grief = any(
        ev.type == "death" and ev.recency == "recent"
        for ev in signals.life_events
    )
    if not has_recent_grief:
        return decision

    resilience = (investor_state.resilience_level or "medium").lower() if investor_state else "medium"

    if resilience == "high":
        # No hard cap — add a warning modifier only
        adjustments.append(GuardrailAdjustment(
            rule="R7_GRIEF_STATE",
            field="current_allocation",
            before=decision.current_allocation,
            after=decision.current_allocation,
            reason=(
                "Recent bereavement detected but resilience_level='high' — "
                "no hard cap applied. Advisor should monitor emotional state."
            ),
        ))
        return decision

    cap = 15 if resilience == "low" else 20  # low → 15%, medium → 20%

    upper = _parse_upper(decision.current_allocation)
    if upper is not None and upper <= cap:
        return decision

    before = decision.current_allocation
    after  = _cap_allocation(before, cap)
    adjustments.append(GuardrailAdjustment(
        rule="R7_GRIEF_STATE",
        field="current_allocation",
        before=before,
        after=after,
        reason=(
            f"Recent bereavement detected with resilience_level='{resilience}' — "
            f"equity exposure capped at {cap}%. Current preferences are unreliable."
        ),
    ))
    decision.current_allocation = after
    return decision


def _r8_peer_speculator(
    decision: DecisionOutput,
    investor_state,
    signals,
    adjustments: list[GuardrailAdjustment],
) -> DecisionOutput:
    """R8: peer_influence == "high" AND resilience_level == "low" → current_allocation ≤ 25%."""
    if signals is None or investor_state is None:
        return decision

    peer_influence = signals.decision_style.peer_influence
    resilience     = investor_state.resilience_level

    if not (peer_influence == "high" and resilience == "low"):
        return decision

    cap = 25
    upper = _parse_upper(decision.current_allocation)
    if upper is not None and upper <= cap:
        return decision

    before = decision.current_allocation
    after  = _cap_allocation(before, cap)
    adjustments.append(GuardrailAdjustment(
        rule="R8_PEER_SPECULATOR",
        field="current_allocation",
        before=before,
        after=after,
        reason=(
            f"peer_influence='{peer_influence}' AND resilience_level='{resilience}' — "
            f"peer-driven investor with low resilience must not exceed {cap}% equity. "
            "Stated risk appetite is inflated by social influence, not genuine capacity."
        ),
    ))
    decision.current_allocation = after
    return decision


def _r9_recency_bias_baseline_cap(
    decision: DecisionOutput,
    investor_state,
    signals,
    adjustments: list[GuardrailAdjustment],
) -> DecisionOutput:
    """R9: shift_detected AND shift_permanence == "likely_temporary" → baseline_allocation ≤ 40%."""
    if investor_state is None:
        return decision

    if not investor_state.shift_detected:
        return decision

    if investor_state.shift_permanence != "likely_temporary":
        return decision

    cap = 40
    base_upper = _parse_upper(decision.baseline_allocation)
    if base_upper is not None and base_upper <= cap:
        return decision

    before = decision.baseline_allocation
    after  = _cap_allocation(before, cap)
    adjustments.append(GuardrailAdjustment(
        rule="R9_RECENCY_BIAS_BASELINE_CAP",
        field="baseline_allocation",
        before=before,
        after=after,
        reason=(
            f"shift_detected=True AND shift_permanence='likely_temporary' — "
            f"baseline_allocation capped at {cap}% because the behavioral shift "
            "may not represent stable long-term capacity."
        ),
    ))
    decision.baseline_allocation = after
    return decision


def _r10_inconsistent_behavior(
    decision: DecisionOutput,
    investor_state,
    signals,
    adjustments: list[GuardrailAdjustment],
) -> DecisionOutput:
    """R10: Inconsistent behavior + unknown dominant_trait → current_allocation ≤ 20%."""
    if signals is None or investor_state is None:
        return decision

    consistency    = signals.behavior.consistency
    dominant_trait = (investor_state.dominant_trait or "").strip().lower()

    if not (consistency == "inconsistent" and dominant_trait in ("unknown", "")):
        return decision

    cap = 20
    upper = _parse_upper(decision.current_allocation)
    if upper is not None and upper <= cap:
        return decision

    before = decision.current_allocation
    after  = _cap_allocation(before, cap)
    adjustments.append(GuardrailAdjustment(
        rule="R10_INCONSISTENT_BEHAVIOR",
        field="current_allocation",
        before=before,
        after=after,
        reason=(
            "Inconsistent behavior with no clear dominant trait → unreliable execution, "
            "conservative cap applied."
        ),
    ))
    decision.current_allocation = after
    return decision


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Ordered rule pipeline — rules fire in priority order
_RULES = [
    _r2_crisis_state,               # highest priority — crisis overrides everything
    _r1_panic_dominance,            # panic is the next strongest signal
    _r7_grief_state,                # grief is a hard constraint
    _r3_high_obligation_low_resilience,
    _r8_peer_speculator,
    _r10_inconsistent_behavior,     # inconsistent behavior with no dominant trait
    _r4_transitional_ordering,      # structural ordering rules
    _r5_temporary_ordering,
    _r9_recency_bias_baseline_cap,
    _r6_allocation_mode_sync,       # always last — syncs mode after all value changes
]


def apply_guardrails(
    decision: DecisionOutput,
    investor_state,
    signals,
) -> tuple[DecisionOutput, list[GuardrailAdjustment]]:
    """
    Apply all hard constraint rules to the LLM-proposed decision.

    Rules fire in priority order. Each rule may adjust current_allocation,
    baseline_allocation, or allocation_mode. Every adjustment is recorded.

    Returns (adjusted_decision, adjustments).
    adjustments is empty if no rules fired.

    IMPORTANT: This function mutates decision in-place for efficiency.
    The caller should treat the returned decision as the authoritative output.
    """
    adjustments: list[GuardrailAdjustment] = []

    for rule_fn in _RULES:
        decision = rule_fn(decision, investor_state, signals, adjustments)

    return decision, adjustments


def guardrail_adjustments_to_dict(adjustments: list[GuardrailAdjustment]) -> list[dict]:
    return [
        {
            "rule":   a.rule,
            "field":  a.field,
            "before": a.before,
            "after":  a.after,
            "reason": a.reason,
        }
        for a in adjustments
    ]
