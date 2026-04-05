"""
Field Ownership Registry — InvestorDNA v7
Single source of truth for field ownership, confidence rules, and data structure.

v7 changes:
  - Removed LLM_CORRECTION_ALLOWED / LLM_CORRECTION_BLOCKED (correction pipeline deleted)
  - Removed LLM_EXTRACTION_FIELDS (replaced by LLM_OWNED_FIELDS)
  - Added RULE_OWNED_FIELDS for explicit merge precedence
  - assign_confidence simplified: no "llm_correction" source
"""

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Canonical data structure
# ---------------------------------------------------------------------------

@dataclass
class FieldValue:
    value:      Any
    confidence: str   # "low" | "medium" | "high"
    source:     str   # "rule" | "llm" | "derived" | "default"

    def to_dict(self) -> dict:
        return {"value": self.value, "confidence": self.confidence, "source": self.source}

    @staticmethod
    def null(source: str = "default") -> "FieldValue":
        return FieldValue(value=None, confidence="low", source=source)


# ---------------------------------------------------------------------------
# Field ownership — drives merge precedence
# ---------------------------------------------------------------------------

# Rule-owned: extracted deterministically; LLM must not override
RULE_OWNED_FIELDS: set[str] = {
    "monthly_income",
    "emi_amount",
    "emergency_months",
    "income_type",      # rule sets from LPA/gig/business keywords; LLM fills only if rule=None
}

# LLM-owned: interpretive/semantic; rules cannot reliably extract these
LLM_OWNED_FIELDS: set[str] = {
    "loss_reaction",
    "loss_reaction_description",
    "risk_behavior",
    "decision_autonomy",
    "financial_knowledge_score",
    "experience_years",
    "dependents",
    "near_term_obligation_level",
    "obligation_type",
}

# Derived: computed post-merge from other fields; never set by LLM or rules
DERIVED_FIELDS: set[str] = {
    "emi_ratio",
    "financial_capacity",
    "future_obligation_score",
}

# All known fields (for invariant checks)
ALL_KNOWN_FIELDS: set[str] = RULE_OWNED_FIELDS | LLM_OWNED_FIELDS | DERIVED_FIELDS


# ---------------------------------------------------------------------------
# Confidence assignment
# ---------------------------------------------------------------------------

# LLM interpretive fields — always low confidence (semantic, not factual)
_LLM_INTERPRETIVE: set[str] = {
    "loss_reaction",
    "loss_reaction_description",
    "risk_behavior",
    "decision_autonomy",
    "near_term_obligation_level",
    "obligation_type",
}


def assign_confidence(field_name: str, value: Any, source: str) -> str:
    """
    Strict confidence assignment. NEVER returns "high" for null values.

    Rules:
      null value          → always "low"
      source == "rule"    → "medium" (deterministic regex, not infallible)
      source == "derived" → "medium" (computed from other fields)
      LLM interpretive    → "low"  (semantic judgment, inherently uncertain)
      LLM numeric/factual → "high" (explicit value stated in text)
      LLM knowledge=3     → "low"  (ambiguous default guard)
    """
    if value is None:
        return "low"

    if source == "rule":
        return "medium"

    if source == "derived":
        return "medium"

    # source == "llm"
    if field_name in _LLM_INTERPRETIVE:
        return "low"

    if field_name == "financial_knowledge_score":
        return "low" if value == 3 else "high"

    if field_name == "income_type":
        return "high" if value not in ("unknown", None) else "low"

    # Numeric LLM fields with explicit value
    if field_name in {"monthly_income", "emi_amount", "emergency_months",
                      "dependents", "experience_years"}:
        return "high"

    return "medium"


def make_field(field_name: str, value: Any, source: str) -> FieldValue:
    conf = assign_confidence(field_name, value, source)
    return FieldValue(value=value, confidence=conf, source=source)


# ---------------------------------------------------------------------------
# Invariant checker
# ---------------------------------------------------------------------------

def check_invariants(fields: dict[str, FieldValue]) -> list[str]:
    """
    Verify system invariants at pipeline boundaries.

    Invariants:
      1. No null value with high confidence
      2. No derived field set by non-derived source
      3. All values are FieldValue instances
    """
    violations = []
    for fname, fv in fields.items():
        if not isinstance(fv, FieldValue):
            violations.append(f"INVARIANT: {fname} is not a FieldValue instance")
            continue
        if fv.value is None and fv.confidence == "high":
            violations.append(
                f"INVARIANT: {fname}=null has confidence='high'"
            )
        if fname in DERIVED_FIELDS and fv.source not in ("derived", "default"):
            violations.append(
                f"INVARIANT: derived field '{fname}' has source='{fv.source}'"
            )
    return violations
