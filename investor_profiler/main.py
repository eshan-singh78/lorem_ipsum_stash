"""
Main Pipeline Orchestrator — v5
Architecture:
  INPUT → RULE EXTRACTION → LLM EXTRACTION → MERGE (CONTROLLED)
        → CORRECTION (CONFIDENCE-AWARE) → VALIDATION (SINGLE LAYER)
        → DERIVED COMPUTATION → SCORING → DECISION

Key guarantees:
  - Each field has ONE authoritative source (field_registry.py)
  - No silent overwrites — all merges logged
  - Confidence reflects real certainty (no "high" defaults)
  - LLM correction only touches low-confidence / null fields
  - Derived fields recomputed post-correction from fresh inputs
  - Single validation layer (validation.py) — no duplicate logic
  - Full audit trail in debug output
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
from scoring import compute_scores
from analysis import cross_axis_analysis
from categories import compute_categories

_KEY_FIELDS = [
    "income_type", "monthly_income", "emergency_months",
    "emi_amount", "dependents", "experience_years",
    "financial_knowledge_score", "loss_reaction", "risk_behavior",
    "near_term_obligation_level",
]


def _fv_plain(fields: dict[str, FieldValue]) -> dict:
    """Extract plain {field: value} from FieldValue dict."""
    return {k: v.value for k, v in fields.items()}


def _fv_conf(fields: dict[str, FieldValue]) -> dict:
    """Extract {field: confidence} from FieldValue dict."""
    return {k: v.confidence for k, v in fields.items()}


def run_pipeline(paragraph: str, verbose: bool = False) -> dict:
    """Full pipeline: investor paragraph → structured profile."""

    # -----------------------------------------------------------------------
    # Stage 1: Extraction (rule + LLM + controlled merge)
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[1/6] Extraction (rule + LLM + merge)...")

    extraction_result = extract_investor_data(paragraph)

    if extraction_result.get("non_english"):
        return {
            "extracted_data":     {},
            "axis_scores":        None,
            "category_scores":    None,
            "financial_capacity": None,
            "final_decision":     "Error: Non-English Input",
            "decision_reasoning": ["Input is not in English."],
            "confidence_score":   0,
            "data_completeness":  0,
            "debug": {
                "missing_fields":    _KEY_FIELDS,
                "merge_log":         [],
                "correction_log":    [],
                "field_sources":     {},
                "derived_fields":    [],
                "confidence_corrections": [],
                "invariant_violations":   [],
            },
            "extraction_warning": extraction_result.get("extraction_warning"),
        }

    fields: dict[str, FieldValue] = extraction_result["fields"]
    normalized_text: str          = extraction_result["normalized_text"]
    merge_log: list[dict]         = extraction_result["merge_log"]
    future_events: list[dict]     = extraction_result["future_events"]
    future_obligation_score: float = extraction_result["future_obligation_score"]

    if verbose:
        print("Merged fields:", json.dumps(_fv_plain(fields), indent=2))
        print(f"Future events: {len(future_events)}, score={future_obligation_score}")

    # -----------------------------------------------------------------------
    # Stage 2: Correction (confidence-aware, normalized_text used)
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[2/6] Correction (confidence-aware)...")

    fields, correction_log, correction_source = run_correction(normalized_text, fields)

    if verbose:
        corrected_count = sum(1 for e in correction_log if e.get("action") == "corrected")
        print(f"Corrections applied: {corrected_count}, source: {correction_source}")

    # -----------------------------------------------------------------------
    # Stage 3: Validation (single layer — type/range/enum only)
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[3/6] Validation...")

    fields = validate_and_cast(fields)

    # -----------------------------------------------------------------------
    # Stage 4: Derived field computation (post-correction, fresh inputs)
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[4/6] Derived field computation...")

    fields, derivation_log = compute_derived_fields(fields, future_obligation_score)

    # Final confidence invariant check
    fields, conf_violations = final_confidence_check(fields)

    # Full invariant check for audit
    invariant_violations = check_invariants(fields)

    # Build plain dicts for downstream (scoring expects plain values + confidences)
    plain_validated = _fv_plain(fields)
    confidences     = _fv_conf(fields)
    field_sources   = build_field_sources(fields)

    data_completeness, missing_fields = compute_data_completeness(fields)

    if verbose:
        print(f"Completeness: {data_completeness}%  Missing: {missing_fields}")

    if is_all_null(fields):
        return {
            "extracted_data":     fields_to_dict(fields),
            "axis_scores":        None,
            "category_scores":    None,
            "financial_capacity": None,
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
                "future_intent_reasons":  extraction_result.get("future_intent_reasons", []),
            },
            "extraction_warning": extraction_result.get("extraction_warning"),
        }

    # -----------------------------------------------------------------------
    # Stage 5: Scoring
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[5/6] Scoring...")

    axis_scores, normalized_data, score_debug = compute_scores(
        plain_validated, confidences,
        future_obligation_score=future_obligation_score,
    )
    confidence_score = score_debug["confidence_score"]

    if verbose:
        print("Axis scores:", json.dumps(
            {k: v for k, v in axis_scores.items() if not k.startswith("_")}, indent=2
        ))

    # -----------------------------------------------------------------------
    # Stage 6: Analysis + Decision
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[6/6] Analysis + Decision...")

    analysis        = cross_axis_analysis(axis_scores)
    category_report = compute_categories(
        normalized_data, axis_scores, confidence_score=confidence_score,
    )

    final_decision = category_report["final_decision"]
    if data_completeness < 40:
        final_decision = f"Low Confidence Profile ({final_decision})"
        category_report["decision_reasoning"].append(
            f"Data completeness {data_completeness}% < 40% — profile marked Low Confidence"
        )

    if verbose:
        print("Final decision:", final_decision)

    # Obligation reason
    ntol  = normalized_data.get("near_term_obligation_level", "none")
    otype = normalized_data.get("obligation_type")
    obligation_reason = (
        f"near_term_obligation_level={ntol}"
        + (f", type={otype}" if otype else "")
        if ntol in ("moderate", "high")
        else "No near-term obligation detected"
    )

    # -----------------------------------------------------------------------
    # Assemble output
    # -----------------------------------------------------------------------
    return {
        "extracted_data": fields_to_dict(fields),
        "axis_scores": {
            "risk":       axis_scores["risk"],
            "cashflow":   axis_scores["cashflow"],
            "obligation": axis_scores["obligation"],
            "context":    axis_scores["context"],
        },
        "category_scores":    category_report["categories"],
        "financial_capacity": axis_scores["financial_capacity"],
        "final_decision":     final_decision,
        "decision_reasoning": category_report["decision_reasoning"],
        "confidence_score":   confidence_score,
        "data_completeness":  data_completeness,
        "analysis": {
            "archetype":            analysis["archetype"],
            "mismatch":             analysis["mismatch"],
            "constraint":           analysis["constraint"],
            "suitability_insights": analysis["suitability_insights"],
        },
        "debug": {
            # Audit trail
            "merge_log":              merge_log,
            "correction_log":         correction_log,
            "correction_source":      correction_source,
            "field_sources":          field_sources,
            "derived_fields":         derivation_log,
            "confidence_corrections": conf_violations,
            "invariant_violations":   invariant_violations,
            # Scoring debug
            "missing_fields":         missing_fields,
            "consistency_flags":      score_debug["consistency_flags"],
            "axis_reasons":           score_debug["axis_reasons"],
            "capacity_formula":       analysis["debug"]["financial_capacity_formula"],
            "risk_vs_capacity_gap":   analysis["debug"]["risk_vs_capacity_gap"],
            "emi_ratio_source":       normalized_data.get("emi_ratio_source", "none"),
            "obligation_reason":      obligation_reason,
            "emerging_constraint":    score_debug.get("emerging_constraint", False),
            # Future intent
            "future_events_detected": future_events,
            "future_obligation_score": future_obligation_score,
            "future_intent_reasons":  extraction_result.get("future_intent_reasons", []),
            # Extraction
            "rule_log":               extraction_result.get("rule_log", []),
            "extraction_warning":     extraction_result.get("extraction_warning"),
        },
        "extraction_warning": extraction_result.get("extraction_warning"),
    }


def main():
    parser = argparse.ArgumentParser(description="Investor Profiling Engine v5")
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
    print("INVESTOR PROFILE OUTPUT")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
