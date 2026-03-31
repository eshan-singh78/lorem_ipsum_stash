"""
Main Pipeline Orchestrator — InvestorDNA v7
Architecture:
  INPUT
  → normalize_text
  → rule_extraction (deterministic)
  → single LLM call (llama3.1:8b, rule context injected)
  → rule-wins merge + post-merge invariants
  → validation (type/range/enum)
  → derived field computation
  → profile_context (rich context object)
  → category_assessment (context → categories)
  → axis_scoring (categories → 4 axes)
  → cross_axis_union (mismatch, constraint, archetype, narratives)
  → OUTPUT

v7 changes vs v6:
  - Correction pipeline entirely removed (was: 3B extraction + 8B correction)
  - Single model: llama3.1:8b only
  - One LLM call per request
  - Rule context injected into LLM prompt — no post-hoc correction needed
  - Post-merge invariants enforced in code (not in prompts)
  - Dead files removed: scoring.py, categories.py, analysis.py, correction.py
"""

import json
import argparse

from field_registry import FieldValue, check_invariants
from extraction import extract_investor_data, fields_to_dict
from validation import (
    validate_and_cast, compute_derived_fields,
    is_all_null, compute_data_completeness,
    build_field_sources, final_confidence_check,
)
from profile_context import build_profile_context, context_to_dict
from context_categories import assess_all_categories, categories_to_dict
from axis_scoring import compute_axis_scores, axis_scores_to_dict
from cross_axis import build_cross_axis_report, cross_axis_report_to_dict

_KEY_FIELDS = [
    "income_type", "monthly_income", "emergency_months",
    "emi_amount", "dependents", "experience_years",
    "financial_knowledge_score", "loss_reaction", "risk_behavior",
    "near_term_obligation_level",
]


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
            "profile_context":    None,
            "category_scores":    None,
            "axis_scores":        None,
            "cross_axis":         None,
            "final_decision":     "Insufficient Data",
            "decision_reasoning": ["All extracted fields are null — cannot profile investor."],
            "confidence_score":   0,
            "data_completeness":  data_completeness,
            "debug": {
                "missing_fields":          missing_fields,
                "rule_fields":             extraction_result.get("rule_fields", {}),
                "llm_fields":              extraction_result.get("llm_fields", {}),
                "merge_log":               extraction_result.get("merge_log", []),
                "rule_log":                extraction_result.get("rule_log", []),
                "invariant_log":           extraction_result.get("invariant_log", []),
                "derived_fields":          derivation_log,
                "confidence_corrections":  conf_violations,
                "invariant_violations":    invariant_violations,
                "future_events_detected":  future_events,
                "future_obligation_score": future_obligation_score,
                "extraction_warning":      extraction_result.get("extraction_warning"),
            },
        }

    # -----------------------------------------------------------------------
    # Stage 4: Profile Context
    # Single boundary where validated fields are read.
    # All downstream stages read from ProfileContext only.
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[4/6] Profile context...")

    profile_ctx = build_profile_context(
        validated_fields=plain_validated,
        raw_text=paragraph,
        future_obligation_score=future_obligation_score,
    )

    if verbose:
        print(f"  Life events:        {[e.event_type for e in profile_ctx.life_events]}")
        print(f"  Cultural signals:   {[s.signal_type for s in profile_ctx.cultural_signals]}")
        print(f"  Behavioral signals: {[s.signal_type for s in profile_ctx.behavioral_signals]}")
        print(f"  Flags: grief={profile_ctx.grief_state}, "
              f"peer={profile_ctx.peer_driven}, "
              f"hidden_obligation={profile_ctx.hidden_obligation_detected}, "
              f"recency_bias={profile_ctx.recency_bias_risk}")

    # -----------------------------------------------------------------------
    # Stage 5: Category Assessment → Axis Scoring
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[5/6] Categories + axis scoring...")

    categories  = assess_all_categories(profile_ctx)
    axis_scores = compute_axis_scores(categories, profile_ctx)

    if verbose:
        print(f"  income_stability:    {categories.income_stability.label} ({categories.income_stability.score})")
        print(f"  debt_burden:         {categories.debt_burden.label} ({categories.debt_burden.score})")
        print(f"  cultural_obligation: {categories.cultural_obligation.label} ({categories.cultural_obligation.score})")
        print(f"  behavioral_risk:     {categories.behavioral_risk.label} ({categories.behavioral_risk.score})")
        print(f"  Axis 1 Risk:         {axis_scores.risk}")
        print(f"  Axis 2 Cashflow:     {axis_scores.cashflow}")
        print(f"  Axis 3 Obligation:   {axis_scores.obligation}")
        print(f"  Axis 4 Context:      {axis_scores.context}")
        print(f"  Financial Capacity:  {axis_scores.financial_capacity}")

    # -----------------------------------------------------------------------
    # Stage 6: Cross-Axis Union
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[6/6] Cross-axis union...")

    cross_axis = build_cross_axis_report(axis_scores, categories, profile_ctx)

    if verbose:
        print(f"  Archetype:   {cross_axis.archetype}")
        print(f"  Mismatch:    {cross_axis.mismatch['type'] if cross_axis.mismatch else None}")
        print(f"  Constraint:  {cross_axis.binding_constraint['type'] if cross_axis.binding_constraint else None}")
        print(f"  Suitability: {cross_axis.suitability['classification']}")

    # -----------------------------------------------------------------------
    # Assemble output
    # -----------------------------------------------------------------------
    final_decision   = cross_axis.suitability["classification"]
    confidence_score = _compute_confidence_score(confidences, plain_validated)

    if data_completeness < 40:
        final_decision = f"Low Confidence Profile ({final_decision})"

    return {
        "profile_context":   context_to_dict(profile_ctx),
        "category_scores":   categories_to_dict(categories),
        "axis_scores": {
            "risk":               axis_scores.risk,
            "cashflow":           axis_scores.cashflow,
            "obligation":         axis_scores.obligation,
            "context":            axis_scores.context,
            "financial_capacity": axis_scores.financial_capacity,
        },
        "cross_axis":         cross_axis_report_to_dict(cross_axis),
        "final_decision":     final_decision,
        "decision_reasoning": cross_axis.suitability_insights,
        "advisor_narrative":  cross_axis.advisor_narrative,
        "investor_narrative": cross_axis.investor_narrative,
        "confidence_score":   confidence_score,
        "data_completeness":  data_completeness,
        "extracted_data":     fields_to_dict(fields),
        "debug": {
            # v7 debug: rule/llm/final field visibility
            "rule_fields":             extraction_result.get("rule_fields", {}),
            "llm_fields":              extraction_result.get("llm_fields", {}),
            "merge_log":               extraction_result.get("merge_log", []),
            "rule_log":                extraction_result.get("rule_log", []),
            "invariant_log":           extraction_result.get("invariant_log", []),
            "field_sources":           field_sources,
            "derived_fields":          derivation_log,
            "confidence_corrections":  conf_violations,
            "invariant_violations":    invariant_violations,
            "missing_fields":          missing_fields,
            "axis_reasons":            axis_scores_to_dict(axis_scores)["reasons"],
            "future_events_detected":  future_events,
            "future_obligation_score": future_obligation_score,
            "future_intent_reasons":   extraction_result.get("future_intent_reasons", []),
            "extraction_warning":      extraction_result.get("extraction_warning"),
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
