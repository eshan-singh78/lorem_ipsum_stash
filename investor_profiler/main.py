"""
Main Pipeline Orchestrator — InvestorDNA v17
Architecture:
  INPUT
  → extraction         (facts only — numbers, no interpretation)
  → signal_extraction  (LLM extracts ALL structured signals — single source of truth)
  → narrative_layer    (LLM understands raw text, grounded by signals)
  → profile_context    (built from signals — no regex)
  → state_synthesis    (signals → structured InvestorState)
  → decision_engine    (LLM with multi-retry + dynamic priority + hard enforcement)
  → constraint_engine  (unified: trace validation + guardrails + re-check + hard enforce)
  → scoring_layer      (categories + axis — reads from signals)
  → validation_layer   (validates current/baseline/temporal consistency)
  → cross_axis         (decision is primary, scores are context)
  → trace_store        (lightweight learning — logs every run)
  → OUTPUT

v16 principle: retry until correct → enforce deterministically → log everything
"""

import json
import argparse

from field_registry import FieldValue, check_invariants
from extraction import extract_investor_data, fields_to_dict
from signal_extraction import extract_signals, signals_to_dict
from validation import (
    validate_and_cast, compute_derived_fields,
    is_all_null, compute_data_completeness,
    build_field_sources, final_confidence_check,
)
from profile_context import build_profile_context, context_to_dict
from narrative_layer import generate_narrative, narrative_to_dict
from state_synthesis import synthesize_state, state_to_dict
from decision_engine import generate_decision, decision_to_dict, DEBUG_REASONING
from constraint_engine import run_constraint_engine, constraint_report_to_dict
from reasoning_validator import validate_reasoning_trace, trace_validation_to_dict
from trace_store import record_trace, analyze_traces
from context_categories import assess_all_categories, categories_to_dict
from axis_scoring import compute_axis_scores, axis_scores_to_dict
from validation_layer import validate_scores_vs_decision, validation_to_dict
from cross_axis import build_cross_axis_report, cross_axis_report_to_dict

_KEY_FIELDS = [
    "income_type", "monthly_income", "emergency_months",
    "emi_amount", "dependents", "experience_years",
    "financial_knowledge_score", "loss_reaction", "risk_behavior",
    "near_term_obligation_level",
]


def _narrative_fallback_state():
    """Minimal InvestorState-like object with safe defaults for narrative failure path."""
    from state_synthesis import InvestorState
    return InvestorState(
        compound_state="unknown — narrative unavailable",
        state_description="",
        dominant_factors=[],
        state_implications=[],
        state_stability="stable",
        confidence="low",
        dominant_trait="unknown",
        suppressed_traits=[],
        resilience_level="medium",
        resilience_evidence="",
        shift_detected=False,
        baseline_behavior="",
        current_behavior="",
        shift_permanence="unknown",
        raw={},
        warning="Narrative unavailable",
    )


def _fv_plain(fields: dict[str, FieldValue]) -> dict:
    return {k: v.value for k, v in fields.items()}


def _fv_conf(fields: dict[str, FieldValue]) -> dict:
    return {k: v.confidence for k, v in fields.items()}


def _compute_confidence_score(confidences: dict, validated: dict) -> int:
    weight_map = {"high": 1.0, "medium": 0.6, "low": 0.3}
    total = 0.0
    for f in _KEY_FIELDS:
        val  = validated.get(f)
        conf = confidences.get(f, "low")
        if val is None or val == "unknown":
            total += 0.0
        else:
            total += weight_map.get(conf, 0.3)
    return int(round((total / len(_KEY_FIELDS)) * 100))


def run_pipeline(paragraph: str, verbose: bool = False) -> dict:
    """Full pipeline: investor paragraph → structured InvestorDNA profile."""

    # -----------------------------------------------------------------------
    # Stage 1: Extraction
    #   - normalize_text
    #   - rule_extraction (deterministic)
    #   - single LLM call (llama3.1:8b)
    #   - rule-wins merge
    #   - post-merge invariants
    #   - future intent detection
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[1/6] Extraction (rules + single LLM + merge)...")

    extraction_result = extract_investor_data(paragraph)

    if extraction_result.get("non_english"):
        return {
            "profile_context":    None,
            "category_scores":    None,
            "axis_scores":        None,
            "cross_axis":         None,
            "final_decision":     "Error: Non-English Input",
            "decision_reasoning": ["Input is not in English."],
            "confidence_score":   0,
            "data_completeness":  0,
            "debug": {"missing_fields": _KEY_FIELDS},
        }

    fields: dict[str, FieldValue] = extraction_result["fields"]
    normalized_text: str          = extraction_result["normalized_text"]
    future_events: list[dict]     = extraction_result["future_events"]
    future_obligation_score: float = extraction_result["future_obligation_score"]

    if verbose:
        rule_f = extraction_result.get("rule_fields", {})
        llm_f  = extraction_result.get("llm_fields", {})
        print(f"  Rule fields:  {list(rule_f.keys())}")
        print(f"  LLM fields:   {list(llm_f.keys())}")
        inv_log = extraction_result.get("invariant_log", [])
        if inv_log:
            print(f"  Invariants applied: {[e['field'] for e in inv_log]}")

    # -----------------------------------------------------------------------
    # Stage 2: Validation (type/range/enum only — no business logic)
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[2/6] Validation...")

    fields = validate_and_cast(fields)

    # -----------------------------------------------------------------------
    # Stage 3: Derived field computation
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[3/6] Derived fields...")

    fields, derivation_log = compute_derived_fields(fields, future_obligation_score)
    fields, conf_violations = final_confidence_check(fields)
    invariant_violations = check_invariants(fields)

    plain_validated = _fv_plain(fields)
    confidences     = _fv_conf(fields)
    field_sources   = build_field_sources(fields)

    data_completeness, missing_fields = compute_data_completeness(fields)

    if is_all_null(fields):
        return {
            "profile_context":  None,
            "narrative":        None,
            "decision":         None,
            "category_scores":  None,
            "axis_scores":      None,
            "validation":       None,
            "cross_axis":       None,
            "final_decision":   "Insufficient Data",
            "confidence_score": 0,
            "data_completeness": data_completeness,
            "debug": {
                "missing_fields":         missing_fields,
                "rule_fields":            extraction_result.get("rule_fields", {}),
                "llm_fields":             extraction_result.get("llm_fields", {}),
                "merge_log":              extraction_result.get("merge_log", []),
                "invariant_log":          extraction_result.get("invariant_log", []),
                "derived_fields":         derivation_log,
                "future_events_detected": future_events,
                "extraction_warning":     extraction_result.get("extraction_warning"),
            },
        }

    # -----------------------------------------------------------------------
    # Stage 4: Signal Extraction — single LLM call, ALL signals extracted once
    # This is the single source of truth for all behavioral/contextual signals.
    # Replaces all regex in profile_context, state_classifier, etc.
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[4] Signal extraction (LLM — unified signal layer)...")

    signals = extract_signals(paragraph)

    if not signals.signals_valid:
        return {
            "status": "error",
            "message": "Unable to extract reliable signals from investor description.",
            "recommendation": "Insufficient data for profiling — please provide more detail.",
            "signals_warning": signals.warning,
        }

    if verbose:
        print(f"  Life events:    {[e.type for e in signals.life_events]}")
        print(f"  Responsibility: {signals.responsibility.role}, pressure={signals.responsibility.financial_pressure}")
        print(f"  Behavior:       loss={signals.behavior.loss_response}, resilience={signals.behavior.resilience_level}")
        print(f"  Contradictions: {len(signals.contradictions)}")
        print(f"  Behavioral shift: {signals.temporal_context.has_behavioral_shift}")
        if signals.warning:
            print(f"  WARNING: {signals.warning}")

    # -----------------------------------------------------------------------
    # Stage 5: Narrative — understand the investor, grounded by signals
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[5] Narrative (LLM — understanding, grounded by signals)...")

    narrative = generate_narrative(normalized_text, plain_validated, signals=signals)

    if not narrative.narrative_valid:
        # Switch to conservative fallback mode — do not pass placeholder text downstream
        from decision_engine import _fallback_decision
        decision = _fallback_decision(retry_count=0)
        decision.warning = f"Narrative generation failed — conservative fallback applied. {narrative.warning or ''}".strip()
        investor_state = _narrative_fallback_state()
        # Build profile_ctx for scoring/cross_axis (still needed downstream)
        profile_ctx = build_profile_context(
            validated_fields=plain_validated,
            raw_text=paragraph,
            future_obligation_score=future_obligation_score,
            signals=signals,
        )
        # Skip decision_engine and synthesize_state — go directly to constraint_engine
    else:
        if verbose:
            print(f"  Life summary:    {narrative.life_summary[:80]}...")
            print(f"  Risk truth:      {narrative.risk_truth[:80]}...")
            if narrative.warning:
                print(f"  WARNING: {narrative.warning}")

        # -----------------------------------------------------------------------
        # Stage 6: Profile Context — built from signals, no regex
        # -----------------------------------------------------------------------
        if verbose:
            print("\n[6] Profile context (signal-driven, no regex)...")

        profile_ctx = build_profile_context(
            validated_fields=plain_validated,
            raw_text=paragraph,
            future_obligation_score=future_obligation_score,
            signals=signals,
        )

        # -----------------------------------------------------------------------
        # Stage 7: State Synthesis — convert signals into ONE coherent state
        # -----------------------------------------------------------------------
        if verbose:
            print("\n[7] State synthesis (LLM — signals → coherent state)...")

        investor_state = synthesize_state(paragraph, narrative, profile_ctx, signals=signals)

        if verbose:
            print(f"  Compound state:  {investor_state.compound_state}")
            print(f"  Stability:       {investor_state.state_stability}")
            print(f"  Dominant factors: {investor_state.dominant_factors}")
            if investor_state.warning:
                print(f"  WARNING: {investor_state.warning}")

        # -----------------------------------------------------------------------
        # Stage 8: Decision — from narrative + signals + synthesized state, NO scores
        # -----------------------------------------------------------------------
        if verbose:
            print("\n[8] Decision engine (LLM — state + narrative + signals)...")

        decision = generate_decision(
            narrative, raw_text=paragraph,
            investor_state=investor_state, signals=signals,
        )

        if verbose:
            print(f"  Archetype:    {decision.archetype}")
            print(f"  Equity range: {decision.equity_range}")
            print(f"  Confidence:   {decision.confidence}")
            print(f"  Reasoning:    {decision.reasoning[:80]}...")
            if decision.warning:
                print(f"  WARNING: {decision.warning}")

    # -----------------------------------------------------------------------
    # Stage 8b+8c: Constraint Engine — unified enforcement
    # Combines: trace validation + guardrails + re-check + hard enforce
    # Single entry point — no invalid output escapes.
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[8b] Constraint engine (unified enforcement)...")

    decision, constraint_report = run_constraint_engine(decision, investor_state, signals)

    if verbose:
        print(f"  Pre-guardrail trace valid:  {constraint_report.pre_guardrail_trace_valid}")
        print(f"  Post-guardrail trace valid: {constraint_report.post_guardrail_trace_valid}")
        if constraint_report.guardrail_adjustments:
            for adj in constraint_report.guardrail_adjustments:
                print(f"  [{adj.rule}] {adj.field}: {adj.before} → {adj.after}")
        if constraint_report.hard_enforced:
            print(f"  HARD ENFORCE: {constraint_report.hard_enforce_note}")

    # Trace validation result for output (post-guardrail is the authoritative one)
    trace_validation = validate_reasoning_trace(decision, investor_state=investor_state, signals=signals)

    # -----------------------------------------------------------------------
    # Stage 8: Scoring — categories + axis AFTER decision
    # investor_state drives calibration, not raw flags
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[8] Scoring layer (state-driven, secondary signals)...")

    categories  = assess_all_categories(profile_ctx, investor_state=investor_state)
    axis_scores = compute_axis_scores(categories, profile_ctx, investor_state=investor_state, signals=signals)

    if verbose:
        print(f"  Risk: {axis_scores.risk}  Cashflow: {axis_scores.cashflow}  "
              f"Obligation: {axis_scores.obligation}  Context: {axis_scores.context}  "
              f"Capacity: {axis_scores.financial_capacity}")

    # -----------------------------------------------------------------------
    # Stage 9: Validation — do scores support the decision?
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[9] Validation layer (scores vs decision)...")

    validation = validate_scores_vs_decision(decision, axis_scores, narrative, investor_state=investor_state)

    if verbose:
        print(f"  Scores support decision: {validation.scores_support_decision}")
        print(f"  Alignment: {validation.overall_alignment}")
        if validation.mismatch_note:
            print(f"  Mismatch: {validation.mismatch_note[:80]}...")

    # -----------------------------------------------------------------------
    # Stage 10: Cross-Axis — decision is primary, scores are context
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[10] Cross-axis union...")

    cross_axis = build_cross_axis_report(
        axis_scores, categories, profile_ctx, narrative, None, decision
    )

    if verbose:
        print(f"  Archetype: {cross_axis.archetype}")

    # -----------------------------------------------------------------------
    # Trace store — record this run for analysis
    # -----------------------------------------------------------------------
    record_trace(paragraph, decision, trace_validation)

    # -----------------------------------------------------------------------
    # Assemble output
    # -----------------------------------------------------------------------
    final_decision   = decision.reasoning if decision.reasoning else cross_axis.suitability["classification"]
    confidence_score = _compute_confidence_score(confidences, plain_validated)

    if data_completeness < 40:
        final_decision = f"Low Confidence Profile — {final_decision}"

    return {
        "profile_context":   context_to_dict(profile_ctx),
        "signals":           signals_to_dict(signals),
        "narrative":         narrative_to_dict(narrative),
        "investor_state":    state_to_dict(investor_state),
        "decision":          decision_to_dict(decision),
        "constraint_report": constraint_report_to_dict(constraint_report),
        "trace_validation":  trace_validation_to_dict(trace_validation),
        "category_scores":   categories_to_dict(categories),
        "axis_scores": {
            "risk":               axis_scores.risk,
            "cashflow":           axis_scores.cashflow,
            "obligation":         axis_scores.obligation,
            "context":            axis_scores.context,
            "financial_capacity": axis_scores.financial_capacity,
        },
        "validation":        validation_to_dict(validation),
        "cross_axis":        cross_axis_report_to_dict(cross_axis),
        "final_decision":    final_decision,
        "advisor_narrative": cross_axis.advisor_narrative,
        "investor_narrative": cross_axis.investor_narrative,
        "suitability_insights": cross_axis.suitability_insights,
        "confidence_score":  confidence_score,
        "data_completeness": data_completeness,
        "extracted_data":    fields_to_dict(fields),
        "debug": {
            "rule_fields":            extraction_result.get("rule_fields", {}),
            "llm_fields":             extraction_result.get("llm_fields", {}),
            "merge_log":              extraction_result.get("merge_log", []),
            "rule_log":               extraction_result.get("rule_log", []),
            "invariant_log":          extraction_result.get("invariant_log", []),
            "field_sources":          field_sources,
            "derived_fields":         derivation_log,
            "confidence_corrections": conf_violations,
            "invariant_violations":   invariant_violations,
            "missing_fields":         missing_fields,
            "axis_reasons":           axis_scores_to_dict(axis_scores)["reasons"],
            "future_events_detected": future_events,
            "future_obligation_score": future_obligation_score,
            "extraction_warning":     extraction_result.get("extraction_warning"),
            "signal_warning":         signals.warning,
            "narrative_warning":      narrative.warning,
            "decision_warning":       decision.warning,
            "guardrail_adjustments":  constraint_report_to_dict(constraint_report)["guardrail_adjustments"],
            "trace_validation":       trace_validation_to_dict(trace_validation),
            "constraint_report":      constraint_report_to_dict(constraint_report),
            "state_synthesis_warning": investor_state.warning,
            "validation_warning":     validation.warning,
        },
        "extraction_warning": extraction_result.get("extraction_warning"),
    }


def main():
    parser = argparse.ArgumentParser(description="InvestorDNA Profiling Engine v7")
    parser.add_argument("--paragraph", "-p", type=str, help="Investor description")
    parser.add_argument("--file",      "-f", type=str, help="Path to text file")
    parser.add_argument("--verbose",   "-v", action="store_true")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            paragraph = f.read().strip()
    elif args.paragraph:
        paragraph = args.paragraph
    else:
        print("Enter investor description (blank line to submit):")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        paragraph = " ".join(lines)

    if not paragraph:
        print("Error: No input provided.")
        return

    result = run_pipeline(paragraph, verbose=args.verbose)
    print("\n" + "=" * 60)
    print("INVESTORDNA PROFILE OUTPUT v7")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
