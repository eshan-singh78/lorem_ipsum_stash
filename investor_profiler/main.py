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
import sys
import threading
import time


# ---------------------------------------------------------------------------
# Live stage logger — shows current stage + elapsed time in real time
# ---------------------------------------------------------------------------

class StageLogger:
    """Prints a live ticker showing the current stage and elapsed seconds."""

    def __init__(self):
        self._stage = ""
        self._start = 0.0
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def start(self, stage: str):
        with self._lock:
            self._stage = stage
            self._start = time.time()
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._tick, daemon=True)
            self._thread.start()
        else:
            # Just update stage — ticker keeps running
            sys.stderr.write("\n")

    def _tick(self):
        spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while self._running:
            with self._lock:
                stage = self._stage
                elapsed = time.time() - self._start
            sys.stderr.write(f"\r{spinner[i % len(spinner)]}  {stage}  ({elapsed:.1f}s)   ")
            sys.stderr.flush()
            time.sleep(0.1)
            i += 1

    def done(self, stage: str = ""):
        with self._lock:
            elapsed = time.time() - self._start
            label = stage or self._stage
        sys.stderr.write(f"\r✓  {label}  ({elapsed:.1f}s)          \n")
        sys.stderr.flush()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)


_logger = StageLogger()

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
from report_layer import generate_report
from report_formatter import build_pdf_payload, render_pdf

_KEY_FIELDS = [
    "income_type", "monthly_income", "emergency_months",
    "emi_amount", "dependents", "experience_years",
    "financial_knowledge_score", "loss_reaction", "risk_behavior",
    "near_term_obligation_level",
]


# ---------------------------------------------------------------------------
# Unified output wrapper — ALL exits go through this
# ---------------------------------------------------------------------------

def _wrap(
    report: dict,
    status: str = "success",
    reason: str = "",
    confidence: str = "high",
    fallback_used: bool = False,
) -> dict:
    """
    Wrap any report dict into the unified production schema:
      { "status": ..., "report": {...}, "meta": {...} }
    """
    return {
        "status": status,
        "report": report,
        "meta": {
            "reason":        reason,
            "confidence":    confidence,
            "fallback_used": fallback_used,
        },
    }


def _minimal_pipeline_payload(
    reason: str,
    paragraph: str = "",
    signals=None,
    narrative=None,
    investor_state=None,
    decision=None,
    data_completeness: int = 0,
    missing_fields: list | None = None,
    future_obligation_score: float = 0.0,
) -> dict:
    """
    Build the minimum pipeline_output dict needed by report_layer
    when the full pipeline cannot complete.
    """
    from decision_engine import _fallback_decision, decision_to_dict
    from signal_extraction import signals_to_dict
    from narrative_layer import narrative_to_dict
    from state_synthesis import state_to_dict

    safe_decision = decision or _fallback_decision(retry_count=0)

    return {
        "status":             "partial",
        "partial_reason":     reason,
        "signals":            signals_to_dict(signals) if signals else {},
        "narrative":          narrative_to_dict(narrative) if narrative else {},
        "investor_state":     state_to_dict(investor_state) if investor_state else {},
        "decision":           decision_to_dict(safe_decision),
        "decision_confidence": "low",
        "constraint_report":  {"guardrail_adjustments": [], "pre_guardrail_trace_valid": False,
                               "post_guardrail_trace_valid": False, "hard_enforced": False,
                               "hard_enforce_note": "", "warnings": [],
                               "pre_guardrail_violations": [], "post_guardrail_violations": []},
        "trace_validation":   {"is_valid": False, "violations": [], "warnings": [],
                               "correction_feedback": "", "not_applicable": True},
        "category_scores":    {},
        "axis_scores":        {"risk": 0, "cashflow": 0, "obligation": 0,
                               "context": 0, "financial_capacity": 0},
        "validation":         {"scores_support_decision": False, "overall_alignment": "unknown",
                               "mismatch_note": "", "warning": ""},
        "cross_axis":         {"suitability": {"classification": "Insufficient data"},
                               "suitability_insights": [], "advisor_narrative": "",
                               "investor_narrative": "", "archetype": ""},
        "profile_context":    {},
        "final_decision":     "Insufficient data — fallback applied.",
        "advisor_narrative":  "",
        "investor_narrative": "",
        "suitability_insights": [],
        "confidence_score":   0,
        "data_completeness":  data_completeness,
        "extracted_data":     {},
        "debug": {
            "missing_fields":          missing_fields or [],
            "future_obligation_score": future_obligation_score,
            "blocking_violations":     [],
        },
    }


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

    _logger.start("Stage 1/9 — Extraction (rules + LLM)")
    extraction_result = extract_investor_data(paragraph)
    _logger.done()

    if extraction_result.get("non_english"):
        _logger.stop()
        payload = _minimal_pipeline_payload("non_english_input")
        return _wrap(
            generate_report(payload),
            status="failed",
            reason="non_english_input",
            confidence="low",
            fallback_used=True,
        )

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

    _logger.start("Stage 2/9 — Validation")
    fields = validate_and_cast(fields)
    _logger.done()

    # -----------------------------------------------------------------------
    # Stage 3: Derived field computation
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[3/6] Derived fields...")

    _logger.start("Stage 3/9 — Derived fields")
    fields, derivation_log = compute_derived_fields(fields, future_obligation_score)
    fields, conf_violations = final_confidence_check(fields)
    invariant_violations = check_invariants(fields)
    _logger.done()

    plain_validated = _fv_plain(fields)
    confidences     = _fv_conf(fields)
    field_sources   = build_field_sources(fields)

    data_completeness, missing_fields = compute_data_completeness(fields)

    if is_all_null(fields):
        _logger.stop()
        payload = _minimal_pipeline_payload(
            "all_fields_null",
            paragraph=paragraph,
            data_completeness=data_completeness,
            missing_fields=missing_fields,
            future_obligation_score=future_obligation_score,
        )
        return _wrap(
            generate_report(payload),
            status="partial",
            reason="all_fields_null",
            confidence="low",
            fallback_used=True,
        )

    # -----------------------------------------------------------------------
    # Stage 4: Signal Extraction — single LLM call, ALL signals extracted once
    # This is the single source of truth for all behavioral/contextual signals.
    # Replaces all regex in profile_context, state_classifier, etc.
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[4] Signal extraction (LLM — unified signal layer)...")

    _logger.start("Stage 4/9 — Signal extraction (LLM)")
    signals = extract_signals(paragraph)
    _logger.done()

    if not signals.signals_valid:
        # Signals failed — build a minimal fallback path and continue to report
        if verbose:
            print(f"  [!] Signal extraction failed — fallback path active. {signals.warning or ''}")
        from decision_engine import _fallback_decision
        from state_synthesis import InvestorState
        _signal_fallback_state = InvestorState(
            compound_state="unknown — signal extraction failed",
            state_description="", dominant_factors=[], state_implications=[],
            state_stability="stable", confidence="low", dominant_trait="unknown",
            suppressed_traits=[], resilience_level="medium", resilience_evidence="",
            shift_detected=False, baseline_behavior="", current_behavior="",
            shift_permanence="unknown", raw={}, warning=signals.warning,
        )
        _signal_fallback_narrative = type("N", (), {
            "narrative_valid": False, "life_summary": "", "financial_analysis": "",
            "psychological_analysis": "", "contradictions": "", "risk_truth": "",
            "reliability_assessment": "", "advisor_insight": "", "warning": signals.warning,
        })()
        payload = _minimal_pipeline_payload(
            "signal_extraction_failed",
            paragraph=paragraph,
            signals=signals,
            investor_state=_signal_fallback_state,
            data_completeness=data_completeness,
            missing_fields=missing_fields,
            future_obligation_score=future_obligation_score,
        )
        _logger.stop()
        return _wrap(
            generate_report(payload),
            status="partial",
            reason="signal_extraction_failed",
            confidence="low",
            fallback_used=True,
        )

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

    _logger.start("Stage 5/9 — Narrative (LLM)")
    narrative = generate_narrative(normalized_text, plain_validated, signals=signals)
    _logger.done()

    if not narrative.narrative_valid:
        # Narrative failed — use fallback state and decision, skip to constraint_engine
        if verbose:
            print(f"  [!] Narrative failed — fallback applied. {narrative.warning or ''}")
        from decision_engine import _fallback_decision
        decision = _fallback_decision(retry_count=0)
        decision.warning = f"Narrative generation failed — conservative fallback applied. {narrative.warning or ''}".strip()
        investor_state = _narrative_fallback_state()
        profile_ctx = build_profile_context(
            validated_fields=plain_validated,
            raw_text=paragraph,
            future_obligation_score=future_obligation_score,
            signals=signals,
        )
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

        _logger.start("Stage 6/9 — Profile context")
        profile_ctx = build_profile_context(
            validated_fields=plain_validated,
            raw_text=paragraph,
            future_obligation_score=future_obligation_score,
            signals=signals,
        )
        _logger.done()

        # -----------------------------------------------------------------------
        # Stage 7: State Synthesis — convert signals into ONE coherent state
        # -----------------------------------------------------------------------
        if verbose:
            print("\n[7] State synthesis (LLM — signals → coherent state)...")

        _logger.start("Stage 7/9 — State synthesis (LLM)")
        investor_state = synthesize_state(paragraph, narrative, profile_ctx, signals=signals)
        _logger.done()

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

        _logger.start("Stage 8/9 — Decision engine (LLM)")
        decision = generate_decision(
            narrative, raw_text=paragraph,
            investor_state=investor_state, signals=signals,
        )
        _logger.done()

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

    _logger.start("Stage 8b/9 — Constraint engine + scoring + validation")
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
    # On fallback paths (narrative/decision failed), mark trace as not_applicable
    # to suppress artificial C1/C2/C6 violations from an intentionally empty trace.
    is_fallback_path = getattr(decision, "fallback_used", False)
    trace_validation = validate_reasoning_trace(decision, investor_state=investor_state, signals=signals)

    # If reasoning is invalid, inject a safe fallback decision and continue.
    # The report layer ALWAYS runs — decision failure degrades confidence, not insight.
    blocking_violations = [v for v in trace_validation.violations if v.severity == "blocking"]
    decision_confidence = "high"
    if not trace_validation.is_valid and blocking_violations:
        from decision_engine import _fallback_decision
        fallback = _fallback_decision(retry_count=getattr(decision, "retry_count", 0))
        fallback.warning = (
            "Fallback applied due to reasoning inconsistency. "
            + (decision.warning or "")
        ).strip()
        fallback.guardrail_adjustments = getattr(decision, "guardrail_adjustments", [])
        decision = fallback
        decision_confidence = "low"
        is_fallback_path = True
        if verbose:
            print(
                f"\n[!] Reasoning validation failed "
                f"({len(blocking_violations)} blocking violation(s)) — "
                "safe fallback applied, pipeline continues."
            )
    elif is_fallback_path:
        decision_confidence = "low"

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
    _logger.done()

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

    _logger.start("Stage 9/9 — Cross-axis union")
    cross_axis = build_cross_axis_report(
        axis_scores, categories, profile_ctx, narrative, None, decision
    )
    _logger.done()

    if verbose:
        print(f"  Archetype: {cross_axis.archetype}")

    # -----------------------------------------------------------------------
    # Trace store — record this run for analysis
    # -----------------------------------------------------------------------
    record_trace(paragraph, decision, trace_validation)
    _logger.stop()

    # -----------------------------------------------------------------------
    # Assemble output
    # -----------------------------------------------------------------------
    final_decision   = decision.reasoning if decision.reasoning else cross_axis.suitability["classification"]
    confidence_score = _compute_confidence_score(confidences, plain_validated)

    if data_completeness < 40:
        final_decision = f"Low Confidence Profile — {final_decision}"

    pipeline_output = {
        "profile_context":   context_to_dict(profile_ctx),
        "signals":           signals_to_dict(signals),
        "narrative":         narrative_to_dict(narrative),
        "investor_state":    state_to_dict(investor_state),
        "decision":          decision_to_dict(decision),
        "constraint_report": constraint_report_to_dict(constraint_report),
        "trace_validation":  {**trace_validation_to_dict(trace_validation),
                              "not_applicable": is_fallback_path},
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
        "decision_confidence": decision_confidence,
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
            "blocking_violations":    [
                {"check": v.check, "description": v.description, "severity": v.severity}
                for v in blocking_violations
            ],
            "state_synthesis_warning": investor_state.warning,
            "validation_warning":     validation.warning,
        },
        "extraction_warning": extraction_result.get("extraction_warning"),
    }

    return _wrap(
        generate_report(pipeline_output),
        status="success" if decision_confidence == "high" else "partial",
        reason="" if decision_confidence == "high" else "decision_fallback_applied",
        confidence=decision_confidence,
        fallback_used=is_fallback_path,
    )


def main():
    parser = argparse.ArgumentParser(description="InvestorDNA Profiling Engine v7")
    parser.add_argument("--paragraph",    "-p", type=str, help="Investor description")
    parser.add_argument("--file",         "-f", type=str, help="Path to text file")
    parser.add_argument("--verbose",      "-v", action="store_true")
    parser.add_argument("--generate-pdf", action="store_true", help="Generate PDF report")
    parser.add_argument("--pdf-name",     type=str, default=None, help="Custom PDF filename")
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

    if args.generate_pdf:
        try:
            payload  = build_pdf_payload(result)
            filename = args.pdf_name or f"investor_report_{int(time.time())}.pdf"
            render_pdf(payload, filename)
            print(f"\n📄 PDF generated: {filename}")
        except Exception as e:
            print(f"\n❌ PDF generation failed: {e}")


if __name__ == "__main__":
    main()
