"""
Validation Layer — v5 (Single Layer)
Runs ONCE after correction. Responsibilities:
  - Type casting and range enforcement
  - Enum validation
  - Derived field computation (emi_ratio, future_obligation_score)
  - Confidence consistency final check
  - NO business logic, NO behavioral overrides, NO duplicate rules

This is the ONLY validation stage in the pipeline.
"""

from typing import Any
from field_registry import FieldValue, DERIVED_FIELDS, make_field, check_invariants

VALID_INCOME_TYPES      = {"salaried", "business", "gig", "unknown"}
VALID_LOSS_REACTIONS    = {"panic", "cautious", "neutral", "aggressive", None}
VALID_RISK_BEHAVIORS    = {"low", "medium", "high", None}
VALID_OBLIGATION_LEVELS = {"none", "moderate", "high", None}
VALID_OBLIGATION_TYPES  = {"wedding", "house", "education", "medical", "family", "other", None}


def _cast_number(val: Any, allow_float: bool = True) -> float | int | None:
    if val is None:
        return None
    try:
        return float(val) if allow_float else int(float(val))
    except (TypeError, ValueError):
        return None


def _cast_bool(val: Any) -> bool | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        if val.lower() == "true":
            return True
        if val.lower() == "false":
            return False
    return None


def validate_and_cast(fields: dict[str, FieldValue]) -> dict[str, FieldValue]:
    """
    Type-cast and range-check all fields.
    Invalid values become null (FieldValue.null) — confidence preserved where possible.
    Returns new dict — does not mutate input.
    """
    out: dict[str, FieldValue] = {}

    def _keep_or_null(fname: str, fv: FieldValue, new_val) -> FieldValue:
        """If cast succeeded, keep original confidence/source. If failed, null with low conf."""
        if new_val is None:
            return FieldValue.null(fv.source)
        return FieldValue(value=new_val, confidence=fv.confidence, source=fv.source)

    for fname, fv in fields.items():
        if fname in DERIVED_FIELDS:
            out[fname] = fv  # derived fields validated separately
            continue

        val = fv.value

        if fname == "income_type":
            out[fname] = FieldValue(
                value=val if val in VALID_INCOME_TYPES else "unknown",
                confidence=fv.confidence, source=fv.source,
            )

        elif fname == "monthly_income":
            n = _cast_number(val)
            out[fname] = _keep_or_null(fname, fv, n if n and n > 0 else None)

        elif fname == "emergency_months":
            n = _cast_number(val)
            out[fname] = _keep_or_null(fname, fv, n if n is not None and n >= 0 else None)

        elif fname == "emi_amount":
            n = _cast_number(val)
            out[fname] = _keep_or_null(fname, fv, n if n is not None and n >= 0 else None)

        elif fname == "emi_ratio":
            n = _cast_number(val)
            out[fname] = _keep_or_null(fname, fv, n if n is not None and 0 <= n <= 100 else None)

        elif fname == "dependents":
            n = _cast_number(val, allow_float=False)
            out[fname] = _keep_or_null(fname, fv, int(n) if n is not None and n >= 0 else None)

        elif fname == "near_term_obligation_level":
            out[fname] = FieldValue(
                value=val if val in VALID_OBLIGATION_LEVELS else None,
                confidence=fv.confidence, source=fv.source,
            )

        elif fname == "obligation_type":
            out[fname] = FieldValue(
                value=val if val in VALID_OBLIGATION_TYPES else None,
                confidence=fv.confidence, source=fv.source,
            )

        elif fname == "experience_years":
            n = _cast_number(val)
            out[fname] = _keep_or_null(fname, fv, n if n is not None and n >= 0 else None)

        elif fname == "financial_knowledge_score":
            n = _cast_number(val, allow_float=False)
            out[fname] = _keep_or_null(
                fname, fv, int(n) if n is not None and 1 <= n <= 5 else None
            )

        elif fname == "decision_autonomy":
            out[fname] = _keep_or_null(fname, fv, _cast_bool(val))

        elif fname == "loss_reaction":
            out[fname] = FieldValue(
                value=val if val in VALID_LOSS_REACTIONS else None,
                confidence=fv.confidence, source=fv.source,
            )

        elif fname == "risk_behavior":
            out[fname] = FieldValue(
                value=val if val in VALID_RISK_BEHAVIORS else None,
                confidence=fv.confidence, source=fv.source,
            )

        else:
            out[fname] = fv  # pass through unknown fields unchanged

    # obligation_type must be null when level is none/null
    ntol = out.get("near_term_obligation_level")
    if ntol and ntol.value in ("none", None):
        ot = out.get("obligation_type")
        if ot and ot.value is not None:
            out["obligation_type"] = FieldValue.null(ot.source)

    return out


def compute_derived_fields(
    fields: dict[str, FieldValue],
    future_obligation_score: float = 0.0,
) -> tuple[dict[str, FieldValue], list[str]]:
    """
    Compute all derived fields from validated inputs.
    Called AFTER correction and validation — uses fresh inputs only.
    Returns (updated_fields, derivation_log).
    """
    out  = dict(fields)
    log: list[str] = []

    emi_fv    = out.get("emi_amount")
    income_fv = out.get("monthly_income")
    emi_val   = emi_fv.value    if emi_fv    else None
    income_val = income_fv.value if income_fv else None

    if emi_val is not None and income_val and income_val > 0:
        ratio = round(min((emi_val / income_val) * 100, 100), 2)
        out["emi_ratio"]        = make_field("emi_ratio", ratio, "derived")
        out["emi_ratio_source"] = FieldValue(value="derived", confidence="medium", source="derived")
        log.append(f"emi_ratio={ratio:.2f}% = emi({emi_val}) / income({income_val}) × 100")
    elif out.get("emi_ratio") and out["emi_ratio"].value is not None:
        out["emi_ratio_source"] = FieldValue(value="llm", confidence="low", source="derived")
        log.append(f"emi_ratio={out['emi_ratio'].value} (LLM-stated, not derived)")
    else:
        out["emi_ratio"]        = FieldValue.null("derived")
        out["emi_ratio_source"] = FieldValue(value="none", confidence="medium", source="derived")

    # _incomplete_emi_data flag
    incomplete = emi_val is not None and (income_val is None or income_val <= 0)
    out["_incomplete_emi_data"] = FieldValue(
        value=incomplete, confidence="high", source="derived"
    )
    if incomplete:
        log.append("WARNING: emi_amount present but monthly_income missing — ratio unverifiable")

    # future_obligation_score
    out["future_obligation_score"] = make_field(
        "future_obligation_score", future_obligation_score, "derived"
    )
    if future_obligation_score > 0:
        log.append(f"future_obligation_score={future_obligation_score} (from intent detection)")

    return out, log


def is_all_null(fields: dict[str, FieldValue]) -> bool:
    meaningful = [
        "monthly_income", "emergency_months", "emi_amount", "emi_ratio",
        "dependents", "experience_years", "financial_knowledge_score",
        "loss_reaction", "risk_behavior", "near_term_obligation_level",
    ]
    return all(
        fields.get(f) is None or fields[f].value is None
        for f in meaningful
    )


def compute_data_completeness(fields: dict[str, FieldValue]) -> tuple[int, list[str]]:
    key_fields = [
        "income_type", "monthly_income", "emergency_months",
        "emi_amount", "dependents", "experience_years",
        "financial_knowledge_score", "loss_reaction", "risk_behavior",
        "near_term_obligation_level",
    ]
    missing = []
    for f in key_fields:
        fv = fields.get(f)
        if fv is None or fv.value is None or fv.value == "unknown":
            missing.append(f)
    present = len(key_fields) - len(missing)
    return int(round((present / len(key_fields)) * 100)), missing


def build_field_sources(fields: dict[str, FieldValue]) -> dict[str, str]:
    """Return {field: source} map for audit output."""
    return {k: v.source for k, v in fields.items() if not k.startswith("_")}


def final_confidence_check(
    fields: dict[str, FieldValue],
) -> tuple[dict[str, FieldValue], list[str]]:
    """
    Final invariant: null value must never have high confidence.
    Runs after all other stages as a safety net.
    """
    out = dict(fields)
    violations: list[str] = []
    for fname, fv in fields.items():
        if fv.value is None and fv.confidence == "high":
            out[fname] = FieldValue(value=None, confidence="low", source=fv.source)
            violations.append(
                f"Final check: {fname}=null had confidence='high' → forced to 'low'"
            )
    return out, violations
