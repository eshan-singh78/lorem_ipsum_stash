"""
Reasoning Validator — InvestorDNA v15
Deterministic consistency checks on the LLM reasoning trace.

Architecture:
  ReasoningTrace + DecisionOutput + InvestorState → ValidationResult

Design principle:
  "The trace must be internally consistent.
   The allocation must follow from the dominant factors.
   Contradictions must be resolved — not ignored."

Checks (deterministic — no LLM calls):
  C1 — TRACE COMPLETENESS:   reasoning_trace must have signals, dominant_factors, state_inference
  C2 — DOMINANT MATCH:       state_context.dominant_trait must appear in reasoning_trace.dominant_factors
  C3 — CONTRADICTION RESOLVED: every structured contradiction must appear in reasoning_trace.contradictions
  C4 — ALLOCATION FOLLOWS:   if dominant_trait is panic/cautious, current_allocation must be conservative
  C5 — TEMPORAL CONSISTENCY: if shift_detected, allocation_mode must not be "static"
  C6 — DECISION LOGIC DEPTH: decision_logic must have ≥ 2 steps

Auto-correction feedback is returned as a structured hint for the decision engine to re-run.
"""

from dataclasses import dataclass, field
from decision_engine import DecisionOutput, ReasoningTrace
import re


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass
class TraceViolation:
    check: str          # check identifier, e.g. "C1_TRACE_COMPLETENESS"
    description: str    # what is wrong
    severity: str       # "blocking" | "warning"
                        # blocking → must re-run; warning → surface to advisor


@dataclass
class TraceValidationResult:
    is_valid: bool
    violations: list[TraceViolation]
    correction_feedback: str    # injected into re-run prompt if not valid
    warnings: list[str]         # non-blocking issues for advisor


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


_CONSERVATIVE_TRAITS = {"panic", "grief", "crisis", "cautious", "fear"}
_CONSERVATIVE_THRESHOLD = 25   # current_allocation upper bound must be ≤ this


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _c1_trace_completeness(
    trace: ReasoningTrace,
    violations: list[TraceViolation],
    warnings: list[str],
) -> None:
    """C1: reasoning_trace must have signals, dominant_factors, state_inference."""
    if not trace.signals_considered:
        violations.append(TraceViolation(
            check="C1_TRACE_COMPLETENESS",
            description="reasoning_trace.signals_considered is empty — no signals were identified.",
            severity="blocking",
        ))
    if not trace.dominant_factors:
        violations.append(TraceViolation(
            check="C1_TRACE_COMPLETENESS",
            description="reasoning_trace.dominant_factors is empty — no dominant factor was identified.",
            severity="blocking",
        ))
    if not trace.state_inference:
        violations.append(TraceViolation(
            check="C1_TRACE_COMPLETENESS",
            description="reasoning_trace.state_inference is missing — overall state was not inferred.",
            severity="blocking",
        ))


def _c2_dominant_match(
    trace: ReasoningTrace,
    decision: DecisionOutput,
    violations: list[TraceViolation],
    warnings: list[str],
) -> None:
    """C2: state_context.dominant_trait must appear in reasoning_trace.dominant_factors."""
    sc_dominant = (decision.state_context.dominant_trait or "").lower()
    if sc_dominant in ("unknown", ""):
        return  # can't check without a dominant trait

    trace_dominant_lower = [f.lower() for f in trace.dominant_factors]
    # Check if any dominant factor contains the trait as a substring
    match = any(sc_dominant in f or f in sc_dominant for f in trace_dominant_lower)
    if not match:
        violations.append(TraceViolation(
            check="C2_DOMINANT_MATCH",
            description=(
                f"state_context.dominant_trait='{sc_dominant}' does not appear in "
                f"reasoning_trace.dominant_factors={trace.dominant_factors}. "
                "The dominant trait must be derived from the reasoning trace."
            ),
            severity="blocking",
        ))


def _c3_contradictions_resolved(
    trace: ReasoningTrace,
    signals,
    violations: list[TraceViolation],
    warnings: list[str],
) -> None:
    """C3: every structured contradiction from signals must be addressed in the trace."""
    if signals is None or not signals.contradictions:
        return  # nothing to check

    if not trace.contradictions:
        # Contradictions exist in signals but none resolved in trace
        unresolved = [
            f"{c.dominant_trait} vs {c.suppressed_trait}"
            for c in signals.contradictions
        ]
        violations.append(TraceViolation(
            check="C3_CONTRADICTIONS_RESOLVED",
            description=(
                f"Structured contradictions exist but reasoning_trace.contradictions is empty. "
                f"Unresolved: {unresolved}. "
                "Every contradiction must be explicitly resolved with a dominant_trait chosen."
            ),
            severity="blocking",
        ))
        return

    # Check that each signal contradiction has a corresponding trace resolution
    for sig_c in signals.contradictions:
        sig_dominant = (sig_c.dominant_trait or "").lower()
        resolved = any(
            sig_dominant in (tc.dominant_trait or "").lower() or
            sig_dominant in (tc.signal_1 or "").lower() or
            sig_dominant in (tc.signal_2 or "").lower()
            for tc in trace.contradictions
        )
        if not resolved:
            violations.append(TraceViolation(
                check="C3_CONTRADICTIONS_RESOLVED",
                description=(
                    f"Signal contradiction with dominant_trait='{sig_dominant}' "
                    "was not explicitly addressed in reasoning_trace.contradictions."
                ),
                severity="blocking",
            ))


def _c4_allocation_follows_dominant(
    trace: ReasoningTrace,
    decision: DecisionOutput,
    violations: list[TraceViolation],
    warnings: list[str],
) -> None:
    """C4: if dominant_trait is a conservative signal, current_allocation must be conservative."""
    dominant = (decision.state_context.dominant_trait or "").lower()
    if not any(t in dominant for t in _CONSERVATIVE_TRAITS):
        return  # dominant trait is not conservative — no constraint

    upper = _parse_upper(decision.current_allocation)
    if upper is None:
        return

    if upper > _CONSERVATIVE_THRESHOLD:
        violations.append(TraceViolation(
            check="C4_ALLOCATION_FOLLOWS_DOMINANT",
            description=(
                f"dominant_trait='{dominant}' is a conservative signal but "
                f"current_allocation='{decision.current_allocation}' (upper={upper}%) "
                f"exceeds the conservative threshold of {_CONSERVATIVE_THRESHOLD}%. "
                "The allocation must follow from the dominant factor."
            ),
            severity="blocking",
        ))


def _c5_temporal_consistency(
    trace: ReasoningTrace,
    decision: DecisionOutput,
    investor_state,
    violations: list[TraceViolation],
    warnings: list[str],
) -> None:
    """C5: if shift_detected, allocation_mode must not be 'static'."""
    shift_detected = getattr(investor_state, "shift_detected", False) if investor_state else False
    if not shift_detected:
        return

    if decision.allocation_mode == "static":
        violations.append(TraceViolation(
            check="C5_TEMPORAL_CONSISTENCY",
            description=(
                "shift_detected=True but allocation_mode='static'. "
                "When a behavioral shift is detected, allocation_mode must be "
                "'transitional' or 'conditional' to reflect the current vs baseline distinction."
            ),
            severity="blocking",
        ))


def _c6_decision_logic_depth(
    trace: ReasoningTrace,
    violations: list[TraceViolation],
    warnings: list[str],
) -> None:
    """C6: decision_logic must have ≥ 2 steps."""
    if len(trace.decision_logic) < 2:
        violations.append(TraceViolation(
            check="C6_DECISION_LOGIC_DEPTH",
            description=(
                f"reasoning_trace.decision_logic has only {len(trace.decision_logic)} step(s). "
                "At least 2 steps are required to demonstrate non-trivial reasoning."
            ),
            severity="blocking",
        ))


def _c7_dominant_trait_grounded(
    trace: ReasoningTrace,
    decision: DecisionOutput,
    violations: list[TraceViolation],
    warnings: list[str],
) -> None:
    """
    C7: dominant_trait must be verbatim or substring-matched from:
      - reasoning_trace.dominant_factors, OR
      - a contradiction resolution dominant_trait in reasoning_trace.contradictions
    Inventing abstract traits outside these sources is a blocking violation.
    """
    sc_dominant = (decision.state_context.dominant_trait or "").lower().strip()
    if sc_dominant in ("unknown", ""):
        return  # C2 already handles missing dominant_trait

    # Build allowed pool: dominant_factors + contradiction dominant_traits
    allowed_pool = [f.lower() for f in trace.dominant_factors]
    allowed_pool += [c.dominant_trait.lower() for c in trace.contradictions if c.dominant_trait]

    if not allowed_pool:
        return  # C1/C2 will catch empty trace

    # Check: dominant_trait must appear in or contain a token from the allowed pool
    matched = any(
        sc_dominant in source or source in sc_dominant
        for source in allowed_pool
    )
    if not matched:
        violations.append(TraceViolation(
            check="C7_DOMINANT_TRAIT_GROUNDED",
            description=(
                f"state_context.dominant_trait='{sc_dominant}' is not derived from "
                f"reasoning_trace.dominant_factors={trace.dominant_factors} "
                f"or any contradiction resolution dominant_trait={[c.dominant_trait for c in trace.contradictions]}. "
                "dominant_trait MUST be selected from these sources only — "
                "inventing abstract traits is forbidden."
            ),
            severity="blocking",
        ))


# ---------------------------------------------------------------------------
# Correction feedback builder
# ---------------------------------------------------------------------------

def _build_correction_feedback(violations: list[TraceViolation]) -> str:
    """Build a structured correction hint for the decision engine re-run."""
    if not violations:
        return ""

    blocking = [v for v in violations if v.severity == "blocking"]
    if not blocking:
        return ""

    lines = [
        "⚠️ REASONING TRACE VALIDATION FAILED. Your previous output had these problems:",
    ]
    for v in blocking:
        lines.append(f"  [{v.check}] {v.description}")

    lines.append("")
    lines.append("You MUST fix ALL of the above before outputting your decision.")
    lines.append("Re-run your reasoning from Step 1. Do not skip steps.")
    lines.append("Ensure reasoning_trace.dominant_factors is non-empty.")
    lines.append("Ensure every contradiction is resolved in reasoning_trace.contradictions.")
    lines.append("Ensure current_allocation follows from the dominant_trait.")
    lines.append(
        "CRITICAL: state_context.dominant_trait MUST be copied verbatim from "
        "reasoning_trace.dominant_factors[0] or a contradiction resolution dominant_trait. "
        "Do NOT invent new abstract traits."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate_reasoning_trace(
    decision: DecisionOutput,
    investor_state=None,
    signals=None,
) -> TraceValidationResult:
    """
    Deterministic validation of the reasoning trace.
    No LLM calls — pure structural checks.

    Returns TraceValidationResult with:
      - is_valid: False if any blocking violation found
      - violations: list of all violations
      - correction_feedback: injected into re-run prompt if not valid
      - warnings: non-blocking issues for advisor
    """
    trace      = decision.reasoning_trace
    violations: list[TraceViolation] = []
    warnings:   list[str]            = []

    _c1_trace_completeness(trace, violations, warnings)
    _c2_dominant_match(trace, decision, violations, warnings)
    _c3_contradictions_resolved(trace, signals, violations, warnings)
    _c4_allocation_follows_dominant(trace, decision, violations, warnings)
    _c5_temporal_consistency(trace, decision, investor_state, violations, warnings)
    _c6_decision_logic_depth(trace, violations, warnings)
    _c7_dominant_trait_grounded(trace, decision, violations, warnings)

    blocking = [v for v in violations if v.severity == "blocking"]
    is_valid = len(blocking) == 0

    return TraceValidationResult(
        is_valid=is_valid,
        violations=violations,
        correction_feedback=_build_correction_feedback(violations),
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
