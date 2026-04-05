"""
Validation Layer — InvestorDNA v14
Validates structured decision against scores and investor state.

Architecture:
  DecisionOutput + AxisScores + NarrativeOutput + InvestorState → ValidationResult

v14 changes:
  - Validates current_allocation vs current state
  - Validates baseline_allocation vs true capacity
  - Validates temporal_strategy consistency (is_temporary vs state_stability)
  - Validates dominant_trait is reflected in current_allocation
"""

import json
import re
import requests
from dataclasses import dataclass, field

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL       = "llama3.1:8b"

_VALIDATION_PROMPT = """You are a senior financial advisor reviewing a structured investment decision.

The DECISION was made from narrative understanding alone (no scores).
Guardrail rules have already been applied to enforce hard constraints.
The SCORES were computed afterward as secondary signals.

Your task: determine whether the decision is INTERNALLY CONSISTENT and whether scores support it.

Check ALL of the following:
1. Does current_allocation match the investor's current state (compound_state, dominant_trait)?
2. Does baseline_allocation match true long-term capacity?
3. Is temporal_strategy consistent? (if state_stability is transitional/temporary, is_temporary should be true)
4. Does dominant_trait appear in the reasoning? (e.g. if dominant_trait=panic, allocation should be conservative)
5. Do the computed scores broadly support the decision?
6. Are guardrail adjustments appropriate? (if any were applied, do they make sense given the state?)

Return ONLY valid JSON:

{{
  "scores_support_decision": true or false,
  "mismatches": [
    {{
      "axis": "risk | cashflow | obligation | context | financial_capacity | temporal | dominant_trait | guardrail",
      "score": integer or null,
      "decision_implies": integer or null,
      "reason": "why this is inconsistent"
    }}
  ],
  "guardrail_compliance": "compliant | over_restricted | under_restricted",
  "guardrail_note": "note about guardrail application, or empty string if compliant",
  "mismatch_note": "what the advisor should know about this mismatch, or empty string if none",
  "overall_alignment": "strong | partial | weak"
}}

No markdown. JSON only.

━━━ STRUCTURED DECISION (after guardrails) ━━━
Current allocation: {current_allocation}
Baseline allocation: {baseline_allocation}
Allocation mode: {allocation_mode}
Dominant trait: {dominant_trait}
State stability: {state_stability}
Is temporary: {is_temporary}
Reassessment trigger: {reassessment_trigger}
Reasoning: {reasoning}
Confidence: {confidence}

━━━ GUARDRAIL ADJUSTMENTS APPLIED ━━━
{guardrail_summary}

━━━ COMPUTED SCORES (secondary signals) ━━━
Risk appetite: {risk}
Cash flow stability: {cashflow}
Obligation burden: {obligation}
Investor sophistication: {context}
Financial capacity: {financial_capacity}

━━━ NARRATIVE CONTEXT ━━━
Life situation: {life_summary}
Financial reality: {financial_analysis}
Risk truth: {risk_truth}
"""


@dataclass
class ValidationResult:
    scores_support_decision: bool
    mismatches: list[dict]
    mismatch_note: str
    overall_alignment: str          # strong | partial | weak
    guardrail_compliance: str = "compliant"   # compliant | over_restricted | under_restricted
    guardrail_note: str = ""
    warning: str | None = None


def _build_guardrail_summary(decision) -> str:
    adjustments = getattr(decision, "guardrail_adjustments", []) or []
    if not adjustments:
        return "No guardrail adjustments were applied."
    lines = []
    for a in adjustments:
        lines.append(f"[{a.rule}] {a.field}: {a.before} → {a.after} — {a.reason}")
    return "\n".join(lines)


def validate_scores_vs_decision(
    decision,
    axis_scores,
    narrative,
    investor_state=None,
) -> ValidationResult:
    """
    v15: validates structured decision (current/baseline/temporal) against scores and state.
    Also verifies guardrail compliance.
    """
    current_allocation  = getattr(decision, "current_allocation", getattr(decision, "equity_range", "unknown"))
    baseline_allocation = getattr(decision, "baseline_allocation", current_allocation)
    allocation_mode     = getattr(decision, "allocation_mode", "static")

    sc = getattr(decision, "state_context", None)
    dominant_trait  = getattr(sc, "dominant_trait", "unknown") if sc else "unknown"
    state_stability = getattr(sc, "state_stability", "stable") if sc else "stable"

    ts = getattr(decision, "temporal_strategy", None)
    is_temporary         = getattr(ts, "is_temporary", False) if ts else False
    reassessment_trigger = getattr(ts, "reassessment_trigger", "") if ts else ""

    prompt = _VALIDATION_PROMPT.format(
        current_allocation=current_allocation,
        baseline_allocation=baseline_allocation,
        allocation_mode=allocation_mode,
        dominant_trait=dominant_trait,
        state_stability=state_stability,
        is_temporary=str(is_temporary).lower(),
        reassessment_trigger=reassessment_trigger,
        reasoning=getattr(decision, "reasoning", "Not available"),
        confidence=getattr(decision, "confidence", "Not available"),
        guardrail_summary=_build_guardrail_summary(decision),
        risk=axis_scores.risk,
        cashflow=axis_scores.cashflow,
        obligation=axis_scores.obligation,
        context=axis_scores.context,
        financial_capacity=axis_scores.financial_capacity,
        life_summary=getattr(narrative, "life_summary", "Not available"),
        financial_analysis=getattr(narrative, "financial_analysis", "Not available"),
        risk_truth=getattr(narrative, "risk_truth", "Not available"),
    )

    payload = {
        "model":   LLM_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0, "num_predict": 512},
        "format":  "json",
    }

    warning  = None
    raw_dict = {}

    for attempt in (1, 2):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=None,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                raw_dict = json.loads(m.group(0))
                break
        except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
            if attempt == 2:
                warning = f"Validation layer failed: {e}"

    if not raw_dict:
        return ValidationResult(
            scores_support_decision=True,
            mismatches=[],
            mismatch_note="",
            overall_alignment="unknown",
            warning=warning,
        )

    mismatches = raw_dict.get("mismatches", [])
    if not isinstance(mismatches, list):
        mismatches = []

    mismatch_note = raw_dict.get("mismatch_note", "")
    if mismatches and not mismatch_note:
        mismatch_note = "Decision inconsistency detected — requires advisor review."

    alignment = raw_dict.get("overall_alignment", "strong")
    if alignment not in ("strong", "partial", "weak"):
        alignment = "strong"

    guardrail_compliance = raw_dict.get("guardrail_compliance", "compliant")
    if guardrail_compliance not in ("compliant", "over_restricted", "under_restricted"):
        guardrail_compliance = "compliant"

    return ValidationResult(
        scores_support_decision=bool(raw_dict.get("scores_support_decision", True)),
        mismatches=mismatches,
        mismatch_note=mismatch_note,
        overall_alignment=alignment,
        guardrail_compliance=guardrail_compliance,
        guardrail_note=raw_dict.get("guardrail_note", ""),
        warning=warning,
    )


def validation_to_dict(v: ValidationResult) -> dict:
    return {
        "scores_support_decision": v.scores_support_decision,
        "mismatches":              v.mismatches,
        "mismatch_note":           v.mismatch_note,
        "overall_alignment":       v.overall_alignment,
        "guardrail_compliance":    v.guardrail_compliance,
        "guardrail_note":          v.guardrail_note,
        "warning":                 v.warning,
    }
