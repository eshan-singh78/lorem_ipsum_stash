"""
Field Ownership Registry — v1
Single source of truth for field ownership, confidence rules, and data structure.

Every field in the system has:
  - ONE authoritative owner: "rule" | "llm" | "derived" | "mixed"
  - A canonical FieldValue structure: {value, confidence, source}
  - A defined confidence assignment rule

Pipeline stages MUST consult this registry before writing any field.
"""

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Canonical data structure — ALL fields must use this
# ---------------------------------------------------------------------------

@dataclass
class FieldValue:
    value:      Any
    confidence: str   # "low" | "medium" | "high"
    source:     str   # "rule" | "llm" | "llm_correction" | "derived" | "default"

    def to_dict(self) -> dict:
        return {"value": self.value, "confidence": self.confidence, "source": self.source}

    @staticmethod
    def null(source: str = "default") -> "FieldValue":
        """A null field always has low confidence."""
        return FieldValue(value=None, confidence="low", source=source)


# ---------------------------------------------------------------------------
# Field ownership contract
# ---------------------------------------------------------------------------

# "rule"    → LLM cannot override; rule value is authoritative
# "llm"     → rules cannot overwrite; LLM is authoritative; rules only validate
# "derived" → computed from other fields; never set by LLM or rules directly
# "mixed"   → arbitration required; higher confidence wins; logged

FIELD_OWNERSHIP: dict[str, str] = {
    # Rule-owned (deterministic extraction)
    "monthly_income":   "rule",
    "emi_amount":       "rule",
    "emergency_months": "rule",
    "income_type":      "rule",   # rule sets from LPA/gig/business keywords; LLM refines only if rule=None

    # LLM-owned (soft signal, semantic interpretation)
    "loss_reaction":             "llm",
    "risk_behavior":             "llm",
    "decision_autonomy":         "llm",
    "financial_knowledge_score": "llm",
    "experience_years":          "llm",   # LLM extracts; rule corrects months→years only
    "dependents":                "llm",

    # Mixed (both sources contribute; arbitration by confidence)
    "near_term_obligation_level": "mixed",
    "obligation_type":            "mixed",

    # Derived (computed post-correction; never set directly)
    "emi_ratio":          "derived",
    "financial_capacity": "derived",
    "future_obligation_score": "derived",
}

# Fields the LLM extraction prompt should populate
LLM_EXTRACTION_FIELDS = {
    "income_type", "monthly_income", "emergency_months", "emi_amount", "emi_ratio",
    "dependents", "near_term_obligation_level", "obligation_type",
    "experience_years", "financial_knowledge_score",
    "decision_autonomy", "loss_reaction", "risk_behavior",
}

# Fields the correction LLM is ALLOWED to modify (only low/null confidence)
LLM_CORRECTION_ALLOWED = {
    "loss_reaction", "risk_behavior", "decision_autonomy",
    "financial_knowledge_score", "near_term_obligation_level",
    "obligation_type", "experience_years", "income_type", "dependents",
}

# Fields the correction LLM must NEVER touch (rule-owned with medium+ confidence)
LLM_CORRECTION_BLOCKED = {"monthly_income", "emi_amount", "emergency_months"}

# Derived fields — recomputed after correction, never trusted from LLM
DERIVED_FIELDS = {"emi_ratio", "financial_capacity", "future_obligation_score"}

# ---------------------------------------------------------------------------
# Confidence assignment rules
# ---------------------------------------------------------------------------

# Fields where LLM value is inherently interpretive → always low
_LLM_ALWAYS_LOW = {
    "loss_reaction", "risk_behavior", "decision_autonomy",
    "near_term_obligation_level", "obligation_type",
}

# knowledge score = 3 is the LLM's default guess → low confidence
_KNOWLEDGE_AMBIGUOUS_VALUES = {3}


def assign_confidence(field_name: str, value: Any, source: str) -> str:
    """
    Strict confidence assignment. NEVER returns "high" for null values.

    Rules:
      null value          → always "low"
      source == "rule"    → "medium" (deterministic but pattern-matched)
      source == "derived" → "medium" (computed from other fields)
      LLM interpretive    → "low"
      LLM numeric (explicit) → "high"
      LLM knowledge=3     → "low" (drift guard)
      LLM correction      → "medium" (upgraded from low)
    """
    if value is None:
        return "low"   # INVARIANT: null → never high

    if source == "rule":
        return "medium"

    if source == "derived":
        return "medium"

    if source == "llm_correction":
        return "medium"   # correction upgraded the confidence

    # source == "llm"
    if field_name in _LLM_ALWAYS_LOW:
        return "low"

    if field_name == "financial_knowledge_score":
        return "low" if value in _KNOWLEDGE_AMBIGUOUS_VALUES else "high"

    if field_name == "income_type":
        return "high" if value not in ("unknown", None) else "low"

    # Numeric fields from LLM with explicit value
    if field_name in {"monthly_income", "emi_amount", "emergency_months",
                      "emi_ratio", "dependents", "experience_years"}:
        return "high"

    return "medium"


def make_field(field_name: str, value: Any, source: str) -> FieldValue:
    """Create a FieldValue with correct confidence for the given source."""
    conf = assign_confidence(field_name, value, source)
    return FieldValue(value=value, confidence=conf, source=source)


# ---------------------------------------------------------------------------
# Invariant checker — call at pipeline boundaries
# ---------------------------------------------------------------------------

def check_invariants(fields: dict[str, FieldValue]) -> list[str]:
    """
    Verify system invariants. Returns list of violations (empty = clean).

    Invariants:
      1. No null value with high confidence
      2. No derived field set by non-derived source
      3. All FieldValue instances (not raw dicts)
    """
    violations = []
    for fname, fv in fields.items():
        if not isinstance(fv, FieldValue):
            violations.append(f"INVARIANT: {fname} is not a FieldValue instance")
            continue
        if fv.value is None and fv.confidence == "high":
            violations.append(
                f"INVARIANT VIOLATION: {fname}=null has confidence='high' — forced to 'low'"
            )
        if fname in DERIVED_FIELDS and fv.source not in ("derived", "default"):
            violations.append(
                f"INVARIANT VIOLATION: derived field '{fname}' has source='{fv.source}' "
                f"(must be 'derived')"
            )
    return violations
