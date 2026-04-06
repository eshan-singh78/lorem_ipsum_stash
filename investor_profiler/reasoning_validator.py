"""
Reasoning Validator — InvestorDNA v17 (8B-optimised)

Checks:
  C1 — TRACE COMPLETENESS:    signals + dominant_factors + state_inference must be present
  C2 — DOMINANT TRAIT ENUM:   dominant_trait must be from DOMINANT_TRAIT_ENUM (not verbatim match)
  C3 — CONTRADICTIONS:        warning only (8B may not resolve all contradictions)
  C4 — ALLOCATION FOLLOWS:    panic/cautious dominant_trait → allocation ≤ 25%
  C5 — TEMPORAL CONSISTENCY:  shift_detected → allocation_mode not static
  C6 — LOGIC DEPTH:           decision_logic must have ≥ 1 step (relaxed for 8B)
  C7 — REMOVED:               replaced by C2 enum check
"""

from dataclasses import dataclass
from decision_engine import DecisionOutput, ReasoningTrace, DOMINANT_TRAIT_ENUM
import re


@dataclass
class TraceViolation:
    check: str
    description: str
    severity: str   # "blocking" | "warning"


@dataclass
class TraceValidationResult:
    is_valid: bool
    violations: list[TraceViolation]
    correction_feedback: str
    warnings: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_upper(allocation: str) -> int | None:
    if not allocation:
        return None
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*%?", allocation)
    if m:
        return int(m.group(2))
    m = re.search(r"(\d+)\s*%?", allocation)
    if m:
        return int(m.group(1))
    return None


_CONSERVATIVE_TRAITS = {"panic", "cautious"}
_CONSERVATIVE_THRESHOLD = 25


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _c1_trace_completeness(trace, violations, warnings):
    """C1: trace must have signals, dominant_factors, state_inference."""
    if not trace.signals_considered:
        violations.append(TraceViolation(
            check="C1_TRACE_COMPLETENESS",
            description="reasoning_trace.signals_considered is empty.",
            severity="blocking",
        ))
    if not trace.dominant_factors:
        violations.append(TraceViolation(
            check="C1_TRACE_COMPLETENESS",
            description="reasoning_trace.dominant_factors is empty.",
            severity="blocking",
        ))
    if not trace.state_inference:
        violations.append(TraceViolation(
            check="C1_TRACE_COMPLETENESS",
            description="reasoning_trace.state_inference is missing.",
            severity="blocking",
        ))


def _c2_dominant_trait_enum(decision, violations, warnings):
    """C2: dominant_trait must be from DOMINANT_TRAIT_ENUM."""
    trait = (decision.state_context.dominant_trait or "").lower().strip()
    if trait in ("", "unknown"):
        return  # unknown is allowed
    if trait not in DOMINANT_TRAIT_ENUM:
        violations.append(TraceViolation(
            check="C2_DOMINANT_TRAIT_ENUM",
            description=(
                f"state_context.dominant_trait='{trait}' is not in the allowed enum: "
                f"{sorted(DOMINANT_TRAIT_ENUM)}. "
                "Use the closest matching enum value."
            ),
            severity="blocking",
        ))


def _c3_contradictions_resolved(trace, signals, violations, warnings):
    """C3: warning only — 8B may not resolve all contradictions."""
    if signals is None or not signals.contradictions:
        return
    if not trace.contradictions:
        unresolved = [
            f"{c.dominant_trait} vs {c.suppressed_trait}"
            for c in signals.contradictions
        ]
        warnings.append(
            f"C3_WARNING: {len(signals.contradictions)} signal contradiction(s) not resolved "
            f"in reasoning_trace: {unresolved}."
        )


def _c4_allocation_follows_dominant(trace, decision, violations, warnings):
    """C4: panic/cautious dominant_trait → current_allocation ≤ 25%."""
    dominant = (decision.state_context.dominant_trait or "").lower()
    if dominant not in _CONSERVATIVE_TRAITS:
        return
    upper = _parse_upper(decision.current_allocation)
    if upper is None:
        return
    if upper > _CONSERVATIVE_THRESHOLD:
        violations.append(TraceViolation(
            check="C4_ALLOCATION_FOLLOWS_DOMINANT",
            description=(
                f"dominant_trait='{dominant}' requires conservative allocation but "
                f"current_allocation='{decision.current_allocation}' (upper={upper}%) "
                f"exceeds {_CONSERVATIVE_THRESHOLD}%."
            ),
            severity="blocking",
        ))


def _c5_temporal_consistency(trace, decision, investor_state, violations, warnings):
    """C5: shift_detected → allocation_mode must not be static/normal."""
    shift = getattr(investor_state, "shift_detected", False) if investor_state else False
    if not shift:
        return
    if decision.allocation_mode in ("static", "normal"):
        violations.append(TraceViolation(
            check="C5_TEMPORAL_CONSISTENCY",
            description=(
                f"shift_detected=True but allocation_mode='{decision.allocation_mode}'. "
                "Use 'conservative' or 'defensive' when a behavioral shift is detected."
            ),
            severity="blocking",
        ))


def _c6_decision_logic_depth(trace, violations, warnings):
    """C6: decision_logic must have ≥ 1 step (relaxed for 8B)."""
    if len(trace.decision_logic) < 1:
        violations.append(TraceViolation(
            check="C6_DECISION_LOGIC_DEPTH",
            description="reasoning_trace.decision_logic is empty — at least 1 step required.",
            severity="blocking",
        ))


# ---------------------------------------------------------------------------
# Correction feedback
# ---------------------------------------------------------------------------

def _build_correction_feedback(violations: list[TraceViolation], signals=None) -> str:
    blocking = [v for v in violations if v.severity == "blocking"]
    if not blocking:
        return ""

    lines = ["Fix these issues in your output:"]
    for v in blocking:
        lines.append(f"  [{v.check}] {v.description}")

    lines.append("")
    lines.append(f"dominant_trait MUST be one of: {', '.join(sorted(DOMINANT_TRAIT_ENUM))}.")
    lines.append("dominant_factors must not be empty.")
    lines.append("current_allocation must follow from dominant_trait.")

    # Inject actual contradiction data for C3-related retries
    if signals and signals.contradictions:
        contra_list = [
            f"'{c.dominant_trait}' vs '{c.suppressed_trait}'"
            for c in signals.contradictions
        ]
        lines.append(f"Signal contradictions to resolve: {contra_list}.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate_reasoning_trace(
    decision: DecisionOutput,
    investor_state=None,
    signals=None,
) -> TraceValidationResult:
    trace      = decision.reasoning_trace
    violations: list[TraceViolation] = []
    warnings:   list[str]            = []

    _c1_trace_completeness(trace, violations, warnings)
    _c2_dominant_trait_enum(decision, violations, warnings)
    _c3_contradictions_resolved(trace, signals, violations, warnings)
    _c4_allocation_follows_dominant(trace, decision, violations, warnings)
    _c5_temporal_consistency(trace, decision, investor_state, violations, warnings)
    _c6_decision_logic_depth(trace, violations, warnings)

    blocking = [v for v in violations if v.severity == "blocking"]
    is_valid  = len(blocking) == 0

    return TraceValidationResult(
        is_valid=is_valid,
        violations=violations,
        correction_feedback=_build_correction_feedback(violations, signals),
        warnings=warnings,
    )


def trace_validation_to_dict(r: TraceValidationResult) -> dict:
    return {
        "is_valid": r.is_valid,
        "violations": [
            {"check": v.check, "description": v.description, "severity": v.severity}
            for v in r.violations
        ],
        "warnings": r.warnings,
        "correction_feedback": r.correction_feedback,
    }
