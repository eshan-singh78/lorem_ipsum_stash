"""
Main pipeline orchestrator — v3.
Flow:
  [1] Stage 1 extraction  (qwen2.5:3b  — fast JSON)
  [2] Stage 2 correction  (llama3.1:8b — semantic fix + behavioral rules)
  [3] Validation + data completeness
  [4] Scoring             (normalize + 4-axis + consistency checks)
  [5] Cross-axis analysis
  [6] Category scoring + final decision

Changes from v2:
- wedding_flag removed; near_term_obligation_level used throughout
- data_completeness + confidence_score added to output
- missing_fields surfaced in debug
- Low Confidence Profile label when completeness < 40%
- normalize() called ONCE inside compute_scores(); result threaded everywhere
"""

import json
import argparse

from extraction import extract_investor_data, unwrap_values, unwrap_confidences
from correction import correct_extraction, apply_mandatory_rules
from validation import validate, is_all_null, compute_data_completeness
from scoring import compute_scores
from analysis import cross_axis_analysis
from categories import compute_categories

# Key fields for missing-field reporting
_KEY_FIELDS = [
    "income_type", "monthly_income", "emergency_months",
    "emi_amount", "dependents", "experience_years",
    "financial_knowledge_score", "loss_reaction", "risk_behavior",
    "near_term_obligation_level",
]


def _build_missing_fields(validated: dict) -> list[str]:
    return [
        f for f in _KEY_FIELDS
        if validated.get(f) is None or validated.get(f) == "unknown"
    ]


def run_pipeline(paragraph: str, verbose: bool = False) -> dict:
    """Full pipeline: investor paragraph → structured profile."""

    # -----------------------------------------------------------------------
    # Step 1: Stage 1 — qwen2.5:3b extraction
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[1/6] Stage 1: qwen2.5:3b extraction...")

    extraction_result = extract_investor_data(paragraph)

    if extraction_result.get("non_english"):
        return {
            "extracted_data":     {},
            "axis_scores":        None,
            "category_scores":    None,
            "financial_capacity": None,
            "final_decision":     "Error: Non-English Input",
            "decision_reasoning": ["Input is not in English. Please provide description in English."],
            "confidence_score":   0,
            "data_completeness":  0,
            "debug": {
                "missing_fields":      _KEY_FIELDS,
                "derived_fields":      [],
                "obligation_reason":   "N/A",
                "corrections_applied": [],
            },
            "extraction_warning": extraction_result.get("extraction_warning"),
        }

    extracted_fields = extraction_result["fields"]
    plain_values     = unwrap_values(extraction_result)
    confidences      = unwrap_confidences(extraction_result)

    if verbose:
        print("Stage 1 extracted:", json.dumps(plain_values, indent=2))

    # -----------------------------------------------------------------------
    # Step 2: Stage 2 — llama3.1:8b correction
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[2/6] Stage 2: llama3.1:8b correction...")

    corrected_values, llm_corrections, fallback_used, correction_source = correct_extraction(
        paragraph, plain_values
    )
    corrected_values, mandatory_corrections = apply_mandatory_rules(corrected_values)
    all_corrections = llm_corrections + mandatory_corrections

    # Track confidence adjustments for fields that changed
    confidence_adjustments = []
    for field, orig_val in plain_values.items():
        new_val = corrected_values.get(field)
        if new_val != orig_val and confidences.get(field) == "low":
            confidence_adjustments.append(
                f"{field}: low-confidence field corrected ({orig_val!r} → {new_val!r})"
            )

    if verbose:
        if all_corrections:
            print("Corrections applied:", all_corrections)
        else:
            print("No corrections needed.")

    # -----------------------------------------------------------------------
    # Step 3: Validation + completeness
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[3/6] Validating...")

    validated = validate(corrected_values)
    data_completeness, missing_fields = compute_data_completeness(validated)

    if verbose:
        print("Validated:", json.dumps(validated, indent=2))
        print(f"Data completeness: {data_completeness}%  Missing: {missing_fields}")

    if is_all_null(validated):
        return {
            "extracted_data":     extracted_fields,
            "axis_scores":        None,
            "category_scores":    None,
            "financial_capacity": None,
            "final_decision":     "Insufficient Data",
            "decision_reasoning": [
                "All extracted fields are null or missing — cannot profile investor."
            ],
            "confidence_score":   0,
            "data_completeness":  data_completeness,
            "debug": {
                "missing_fields":      missing_fields,
                "derived_fields":      [],
                "obligation_reason":   "No obligation data extracted.",
                "corrections_applied": all_corrections,
                "fallback_used":       fallback_used,
                "correction_source":   correction_source,
            },
            "extraction_warning": (
                extraction_result.get("extraction_warning")
                or "Input did not contain enough financial information."
            ),
        }

    # -----------------------------------------------------------------------
    # Step 4: Scoring
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[4/6] Computing 4-axis scores...")

    axis_scores, normalized_data, score_debug = compute_scores(validated, confidences)
    confidence_score = score_debug["confidence_score"]

    if verbose:
        print("Axis scores:", json.dumps(axis_scores, indent=2))
        print(f"Confidence score: {confidence_score}%")

    # -----------------------------------------------------------------------
    # Step 5: Cross-axis analysis
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[5/6] Cross-axis analysis...")

    analysis = cross_axis_analysis(axis_scores)

    # -----------------------------------------------------------------------
    # Step 6: Category scoring + final decision
    # -----------------------------------------------------------------------
    if verbose:
        print("\n[6/6] Category scoring + final decision...")

    category_report = compute_categories(
        normalized_data,
        axis_scores,
        confidence_score=confidence_score,
    )

    # Low Confidence Profile label
    final_decision = category_report["final_decision"]
    if data_completeness < 40:
        final_decision = f"Low Confidence Profile ({final_decision})"
        category_report["decision_reasoning"].append(
            f"Data completeness {data_completeness}% < 40% — profile marked Low Confidence"
        )

    if verbose:
        print("Final decision:", final_decision)

    # -----------------------------------------------------------------------
    # Obligation reason for debug
    # -----------------------------------------------------------------------
    ntol  = normalized_data.get("near_term_obligation_level", "none")
    otype = normalized_data.get("obligation_type")
    if ntol in ("moderate", "high"):
        obligation_reason = (
            f"near_term_obligation_level={ntol}"
            + (f", type={otype}" if otype else "")
            + f" — detected from text"
        )
    else:
        obligation_reason = "No near-term obligation detected"

    # -----------------------------------------------------------------------
    # Assemble output
    # -----------------------------------------------------------------------
    return {
        "extracted_data": extracted_fields,
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
            "missing_fields":         missing_fields,
            "derived_fields":         score_debug["derived_fields"],
            "obligation_reason":      obligation_reason,
            "corrections_applied":    all_corrections,
            "confidence_adjustments": confidence_adjustments,
            "consistency_flags":      score_debug["consistency_flags"],
            "axis_reasons":           score_debug["axis_reasons"],
            "capacity_formula":       analysis["debug"]["financial_capacity_formula"],
            "risk_vs_capacity_gap":   analysis["debug"]["risk_vs_capacity_gap"],
            "emi_ratio_source":       normalized_data.get("emi_ratio_source", "none"),
            "fallback_used":          fallback_used,
            "correction_source":      correction_source,
        },
        "extraction_warning": extraction_result.get("extraction_warning"),
    }


def main():
    parser = argparse.ArgumentParser(description="Investor Profiling Engine v3")
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
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
