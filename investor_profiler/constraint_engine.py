"""
Constraint Engine — InvestorDNA v16
Unified constraint enforcement: guardrails + trace validation + re-check.

Architecture:
  DecisionOutput + InvestorState + SignalOutput → (DecisionOutput, ConstraintReport)

Replaces the separate guardrails → trace_validation sequence in main.py.
Single entry point for all constraint enforcement.

Flow:
  1. validate_reasoning_trace  (deterministic — no LLM)
  2. apply_guardrails          (deterministic — no LLM)
  3. re-check trace after guardrails (some guardrail adjustments may fix trace violations)
  4. hard_enforce              (final safety net — no escape)
  5. return ConstraintReport   (full audit trail)
"""

from dataclasses import dataclass, field
from decision_engine import DecisionOutput
from decision_guardrails import apply_guardrails, GuardrailAdjustment, guardrail_adjustments_to_dict
from reasoning_validator import validate_reasoning_trace, TraceValidationResult, trace_validation_to_dict


@dataclass
class ConstraintReport:
    # Trace validation (before guardrails)
    pre_guardrail_trace_valid: bool
    pre_guardrail_violations: list[str]

    # Guardrail adjustments
    guardrail_adjustments: list[GuardrailAdjustment]

    # Trace validation (after guardrails)
    post_guardrail_trace_valid: bool
    post_guardrail_violations: list[str]

    # Hard enforcement
    hard_enforced: bool
    hard_enforce_note: str

    # Warnings (non-blocking)
    warnings: list[str]


def run_constraint_engine(
    decision: DecisionOutput,
    investor_state,
    signals,
) -> tuple[DecisionOutput, ConstraintReport]:
    """
    Unified constraint enforcement pipeline.

    Returns (adjusted_decision, constraint_report).
    The decision is mutated in-place — caller should use the returned decision.
    """
    # Step 1: validate trace before guardrails
    pre_result: TraceValidationResult = validate_reasoning_trace(
        decision, investor_state=investor_state, signals=signals
    )
    pre_violations = [v.check for v in pre_result.violations]

    # Step 2: apply guardrails
    decision, guardrail_log = apply_guardrails(decision, investor_state, signals)
    decision.guardrail_adjustments = guardrail_log

    # Step 3: re-check trace after guardrails
    # Guardrail adjustments (e.g. R6 fixing allocation_mode) may resolve C5
    post_result: TraceValidationResult = validate_reasoning_trace(
        decision, investor_state=investor_state, signals=signals
    )
    post_violations = [v.check for v in post_result.violations]

    # Step 4: hard enforcement (final safety net)
    from decision_engine import _hard_enforce
    before_alloc = decision.current_allocation
    decision = _hard_enforce(decision, investor_state)
    hard_enforced = decision.current_allocation != before_alloc
    hard_note = (
        f"HARD_ENFORCE applied: {before_alloc} → {decision.current_allocation}"
        if hard_enforced else ""
    )

    report = ConstraintReport(
        pre_guardrail_trace_valid=pre_result.is_valid,
        pre_guardrail_violations=pre_violations,
        guardrail_adjustments=guardrail_log,
        post_guardrail_trace_valid=post_result.is_valid,
        post_guardrail_violations=post_violations,
        hard_enforced=hard_enforced,
        hard_enforce_note=hard_note,
        warnings=pre_result.warnings + post_result.warnings,
    )

    return decision, report


def constraint_report_to_dict(r: ConstraintReport) -> dict:
    return {
        "pre_guardrail_trace_valid":   r.pre_guardrail_trace_valid,
        "pre_guardrail_violations":    r.pre_guardrail_violations,
        "guardrail_adjustments":       guardrail_adjustments_to_dict(r.guardrail_adjustments),
        "post_guardrail_trace_valid":  r.post_guardrail_trace_valid,
        "post_guardrail_violations":   r.post_guardrail_violations,
        "hard_enforced":               r.hard_enforced,
        "hard_enforce_note":           r.hard_enforce_note,
        "warnings":                    r.warnings,
    }
