"""
Main Pipeline Orchestrator — InvestorDNA v6
Architecture:
  INPUT
  → RULE EXTRACTION → LLM EXTRACTION → MERGE (CONTROLLED)
  → CORRECTION (CONFIDENCE-AWARE) → VALIDATION (SINGLE LAYER)
  → DERIVED COMPUTATION
  → PROFILE CONTEXT (rich context object — demographics, life events, cultural, behavioral)
  → CATEGORY ASSESSMENT (context → categories, no raw field access)
  → AXIS SCORING (categories → 4 axes, no raw field access)
  → CROSS-AXIS UNION (mismatch, constraint, archetype, narratives)
  → OUTPUT

Key guarantees:
  - Raw fields are ONLY read in profile_context.py (single boundary)
  - All scoring reads from ProfileContext or CategoryAssessment
  - Cultural and behavioral context captured before scoring
  - Narratives generated from full context, not just numbers
  - Full audit trail preserved
"""

import json
import argparse

from field_registry import FieldValue, check_invariants
from extraction import (
    extract_investor_data, unwrap_values, unwrap_confidences,
    unwrap_sources, fields_to_dict,
)
from correction import run_correction
from validation import (
    validate_and_cast, compute_derived_fields,
    is_all_null, compute_data_completeness,
    build_field_sources, final_confidence_check,
)

# New v6 pipeline stages
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


def run_pipeline(paragraph: str, verbose: bool = False) -> dict:
    """Full pipeline: investor paragraph → structured InvestorDNA profile."""

    # -----------------------------------------------------------------------
    # Stage 1: Extraction (rule + LLM + controlled merge)
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[1/7] Extraction (rule + LLM + merge)...")

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
            "debug": {
                "missing_fields": _KEY_FIELDS,
                "merge_log": [],
                "correction_log": [],
                "field_sources": {},
            },
        }

    fields: dict[str, FieldValue] = extraction_result["fields"]
    normalized_text: str          = extraction_result["normalized_text"]
    merge_log: list[dict]         = extraction_result["merge_log"]
    future_events: list[dict]     = extraction_result["future_events"]
    future_obligation_score: float = extraction_result["future_obligation_score"]

    # -----------------------------------------------------------------------
    # Stage 2: Correction (confidence-aware)
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[2/7] Correction (confidence-aware)...")

    fields, correction_log, correction_source = run_correction(normalized_text, fields)

    # -----------------------------------------------------------------------
    # Stage 3: Validation (type/range/enum only)
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[3/7] Validation...")

    fields = validate_and_cast(fields)

    # -----------------------------------------------------------------------
    # Stage 4: Derived field computation
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[4/7] Derived field computation...")

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
                "missing_fields":         missing_fields,
                "merge_log":              merge_log,
                "correction_log":         correction_log,
                "correction_source":      correction_source,
                "field_sources":          field_sources,
                "derived_fields":         derivation_log,
                "confidence_corrections": conf_violations,
                "invariant_violations":   invariant_violations,
                "future_events_detected": future_events,
                "future_obligation_score": future_obligation_score,
            },
        }

    # -----------------------------------------------------------------------
    # Stage 5: Profile Context (NEW — rich context object)
    # This is the boundary: raw fields are read HERE and only here.
    # All downstream stages read from ProfileContext.
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[5/7] Building profile context...")

    profile_ctx = build_profile_context(
        validated_fields=plain_validated,
        raw_text=paragraph,
        future_obligation_score=future_obligation_score,
    )

    if verbose:
        print(f"  Life events: {[e.event_type for e in profile_ctx.life_events]}")
        print(f"  Cultural signals: {[s.signal_type for s in profile_ctx.cultural_signals]}")
        print(f"  Behavioral signals: {[s.signal_type for s in profile_ctx.behavioral_signals]}")
        print(f"  Flags: grief={profile_ctx.grief_state}, peer={profile_ctx.peer_driven}, "
              f"hidden_obligation={profile_ctx.hidden_obligation_detected}")

    # -----------------------------------------------------------------------
    # Stage 6: Category Assessment (context → categories)
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[6/7] Category assessment...")

    categories = assess_all_categories(profile_ctx)

    if verbose:
        print(f"  income_stability: {categories.income_stability.label} ({categories.income_stability.score})")
        print(f"  debt_burden: {categories.debt_burden.label} ({categories.debt_burden.score})")
        print(f"  cultural_obligation: {categories.cultural_obligation.label} ({categories.cultural_obligation.score})")
        print(f"  behavioral_risk: {categories.behavioral_risk.label} ({categories.behavioral_risk.score})")

    # -----------------------------------------------------------------------
    # Stage 7: Axis Scoring + Cross-Axis Union
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[7/7] Axis scoring + cross-axis union...")

    axis_scores = compute_axis_scores(categories, profile_ctx)
    cross_axis  = build_cross_axis_report(axis_scores, categories, profile_ctx)

    if verbose:
        print(f"  Axis 1 (Risk): {axis_scores.risk}")
        print(f"  Axis 2 (Cashflow): {axis_scores.cashflow}")
        print(f"  Axis 3 (Obligation): {axis_scores.obligation}")
        print(f"  Axis 4 (Context): {axis_scores.context}")
        print(f"  Financial Capacity: {axis_scores.financial_capacity}")
        print(f"  Archetype: {cross_axis.archetype}")

    # -----------------------------------------------------------------------
    # Final decision (from suitability classification)
    # -----------------------------------------------------------------------
    final_decision = cross_axis.suitability["classification"]
    if data_completeness < 40:
        final_decision = f"Low Confidence Profile ({final_decision})"

    # Confidence score (simple: based on data completeness + key field presence)
    confidence_score = _compute_confidence_score(confidences, plain_validated)

    # -----------------------------------------------------------------------
    # Assemble output
    # -----------------------------------------------------------------------
    return {
        # Rich context (new in v6)
        "profile_context": context_to_dict(profile_ctx),

        # Category assessment (new in v6 — replaces flat category scores)
        "category_scores": categories_to_dict(categories),

        # Axis scores
        "axis_scores": {
            "risk":               axis_scores.risk,
            "cashflow":           axis_scores.cashflow,
            "obligation":         axis_scores.obligation,
            "context":            axis_scores.context,
            "financial_capacity": axis_scores.financial_capacity,
        },

        # Cross-axis union (new in v6 — full narrative output)
        "cross_axis": cross_axis_report_to_dict(cross_axis),

        # Decision
        "final_decision":     final_decision,
        "decision_reasoning": cross_axis.suitability_insights,

        # Narratives (new in v6)
        "advisor_narrative":  cross_axis.advisor_narrative,
        "investor_narrative": cross_axis.investor_narrative,

        # Metadata
        "confidence_score":  confidence_score,
        "data_completeness": data_completeness,

        # Extracted data (for audit)
        "extracted_data": fields_to_dict(fields),

        # Debug / audit trail
        "debug": {
            "merge_log":              merge_log,
            "correction_log":         correction_log,
            "correction_source":      correction_source,
            "field_sources":          field_sources,
            "derived_fields":         derivation_log,
            "confidence_corrections": conf_violations,
            "invariant_violations":   invariant_violations,
            "missing_fields":         missing_fields,
            "axis_reasons":           axis_scores_to_dict(axis_scores)["reasons"],
            "future_events_detected": future_events,
            "future_obligation_score": future_obligation_score,
            "future_intent_reasons":  extraction_result.get("future_intent_reasons", []),
            "rule_log":               extraction_result.get("rule_log", []),
            "extraction_warning":     extraction_result.get("extraction_warning"),
        },

        "extraction_warning": extraction_result.get("extraction_warning"),
    }


def _compute_confidence_score(confidences: dict, validated: dict) -> int:
    key_fields = [
        "income_type", "monthly_income", "emergency_months",
        "emi_amount", "dependents", "experience_years",
        "financial_knowledge_score", "loss_reaction", "risk_behavior",
        "near_term_obligation_level",
    ]
    weight_map = {"high": 1.0, "medium": 0.6, "low": 0.3}
    total = 0.0
    for f in key_fields:
        val  = validated.get(f)
        conf = confidences.get(f, "low")
        if val is None or val == "unknown":
            total += 0.0
        else:
            total += weight_map.get(conf, 0.3)
    return int(round((total / len(key_fields)) * 100))


def main():
    parser = argparse.ArgumentParser(description="InvestorDNA Profiling Engine v6")
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
    print("INVESTORDNA PROFILE OUTPUT")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
