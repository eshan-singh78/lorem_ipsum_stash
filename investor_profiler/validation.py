"""
Validation Layer — v3
Type enforcement and range checks.
Operates on plain {field: value} dicts (after unwrapping confidence layer).

Changes from v2:
- wedding_flag removed
- near_term_obligation_level + obligation_type added
- is_all_null updated to reflect new schema
"""

from typing import Any

VALID_INCOME_TYPES      = {"salaried", "business", "gig", "unknown"}
VALID_LOSS_REACTIONS    = {"panic", "neutral", "aggressive", None}
VALID_RISK_BEHAVIORS    = {"low", "medium", "high", None}
VALID_OBLIGATION_LEVELS = {"none", "moderate", "high", None}
VALID_OBLIGATION_TYPES  = {"wedding", "house", "education", "medical", "family", "other", None}


def _cast_number(val: Any, allow_float: bool = True):
    if val is None:
        return None
    try:
        return float(val) if allow_float else int(float(val))
    except (TypeError, ValueError):
        return None


def _cast_bool(val: Any):
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


def validate(plain: dict) -> dict:
    """
    Validate and sanitize a plain {field: value} dict.
    Returns clean dict; invalid fields become null.
    """
    v = {}

    it = plain.get("income_type")
    v["income_type"] = it if it in VALID_INCOME_TYPES else "unknown"

    v["monthly_income"] = _cast_number(plain.get("monthly_income"))
    if v["monthly_income"] is not None and v["monthly_income"] <= 0:
        v["monthly_income"] = None

    v["emergency_months"] = _cast_number(plain.get("emergency_months"))
    if v["emergency_months"] is not None and v["emergency_months"] < 0:
        v["emergency_months"] = None

    v["emi_amount"] = _cast_number(plain.get("emi_amount"))
    if v["emi_amount"] is not None and v["emi_amount"] < 0:
        v["emi_amount"] = None

    emi_ratio = _cast_number(plain.get("emi_ratio"))
    v["emi_ratio"] = emi_ratio if emi_ratio is not None and 0 <= emi_ratio <= 100 else None

    dep = _cast_number(plain.get("dependents"), allow_float=False)
    v["dependents"] = int(dep) if dep is not None and dep >= 0 else None

    ntol = plain.get("near_term_obligation_level")
    v["near_term_obligation_level"] = ntol if ntol in VALID_OBLIGATION_LEVELS else None

    ot = plain.get("obligation_type")
    v["obligation_type"] = ot if ot in VALID_OBLIGATION_TYPES else None
    if v["near_term_obligation_level"] in ("none", None):
        v["obligation_type"] = None

    exp = _cast_number(plain.get("experience_years"))
    v["experience_years"] = exp if exp is not None and exp >= 0 else None

    fks = _cast_number(plain.get("financial_knowledge_score"), allow_float=False)
    v["financial_knowledge_score"] = int(fks) if fks is not None and 1 <= fks <= 5 else None

    v["decision_autonomy"] = _cast_bool(plain.get("decision_autonomy"))

    lr = plain.get("loss_reaction")
    v["loss_reaction"] = lr if lr in VALID_LOSS_REACTIONS else None

    rb = plain.get("risk_behavior")
    v["risk_behavior"] = rb if rb in VALID_RISK_BEHAVIORS else None

    return v


def is_all_null(validated: dict) -> bool:
    """Return True if every meaningful field is null/unknown — insufficient data."""
    meaningful = [
        "monthly_income", "emergency_months", "emi_amount", "emi_ratio",
        "dependents", "experience_years", "financial_knowledge_score",
        "loss_reaction", "risk_behavior", "near_term_obligation_level",
    ]
    return all(validated.get(f) is None for f in meaningful)


def compute_data_completeness(validated: dict) -> tuple[int, list[str]]:
    """
    Returns (completeness_pct, missing_fields_list).
    Key fields that matter for profiling.
    """
    key_fields = [
        "income_type", "monthly_income", "emergency_months",
        "emi_amount", "dependents", "experience_years",
        "financial_knowledge_score", "loss_reaction", "risk_behavior",
        "near_term_obligation_level",
    ]
    missing = []
    for f in key_fields:
        val = validated.get(f)
        if val is None or val == "unknown":
            missing.append(f)

    present = len(key_fields) - len(missing)
    pct     = int(round((present / len(key_fields)) * 100))
    return pct, missing
