"""
Score Sanity Check — InvestorDNA v11
Validates that computed scores are consistent with the narrative.

Architecture:
  NarrativeOutput + AxisScores → SanityResult

Design principle:
  "If the scores contradict the story, trust the story."

The LLM is asked: "Do these scores match the narrative?"
If not → scores are flagged as unreliable and the decision engine
is instructed to weight narrative more heavily.
"""

import json
import re
import requests
from dataclasses import dataclass, field

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL       = "llama3.1:8b"

_SANITY_PROMPT = """You are a senior financial advisor reviewing computed scores against a narrative.

Your task: determine whether the scores below are CONSISTENT with the narrative.

A score is inconsistent if it contradicts what the narrative clearly describes.
Examples of inconsistency:
- Risk score = 70 but narrative says "investor is in grief and avoiding all risk"
- Obligation score = 20 but narrative says "most income is committed to family obligations"
- Cashflow score = 80 but narrative says "investor is severely financially constrained"

Answer:
1. Are the scores broadly consistent with the narrative? (yes / no)
2. Which specific scores are inconsistent, and why?
3. What should the scores approximately be, based on the narrative alone?

Return ONLY valid JSON:

{{
  "scores_consistent": true or false,
  "inconsistencies": [
    {{
      "axis": "risk | cashflow | obligation | context",
      "computed": integer,
      "narrative_suggests": integer,
      "reason": "why the computed score contradicts the narrative"
    }}
  ],
  "overall_assessment": "one sentence summary"
}}

If scores are consistent, return empty list for inconsistencies.
No markdown. JSON only.

━━━ NARRATIVE ━━━
Life situation: {life_summary}
Financial reality: {financial_analysis}
Psychological state: {psychological_analysis}
Risk truth: {risk_truth}

━━━ COMPUTED SCORES ━━━
Risk appetite: {risk}
Cash flow stability: {cashflow}
Obligation burden: {obligation}
Investor sophistication: {context}
Financial capacity: {financial_capacity}
"""


@dataclass
class SanityResult:
    scores_consistent: bool
    inconsistencies: list[dict]     # [{axis, computed, narrative_suggests, reason}]
    overall_assessment: str
    corrected_scores: dict          # axis → corrected value (only for inconsistent axes)
    warning: str | None = None


def check_score_sanity(narrative, axis_scores) -> SanityResult:
    """
    Ask LLM: do the computed scores match the narrative?
    Returns SanityResult with inconsistencies and suggested corrections.
    """
    prompt = _SANITY_PROMPT.format(
        life_summary=getattr(narrative, "life_summary", "Not available"),
        financial_analysis=getattr(narrative, "financial_analysis", "Not available"),
        psychological_analysis=getattr(narrative, "psychological_analysis", "Not available"),
        risk_truth=getattr(narrative, "risk_truth", "Not available"),
        risk=axis_scores.risk,
        cashflow=axis_scores.cashflow,
        obligation=axis_scores.obligation,
        context=axis_scores.context,
        financial_capacity=axis_scores.financial_capacity,
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
                warning = f"Score sanity check failed: {e}"

    if not raw_dict:
        return SanityResult(
            scores_consistent=True,
            inconsistencies=[],
            overall_assessment="Sanity check unavailable — assuming scores are consistent.",
            corrected_scores={},
            warning=warning,
        )

    scores_consistent = bool(raw_dict.get("scores_consistent", True))
    inconsistencies   = raw_dict.get("inconsistencies", [])
    if not isinstance(inconsistencies, list):
        inconsistencies = []

    # Build corrected_scores dict from inconsistencies
    corrected_scores = {}
    for item in inconsistencies:
        axis = item.get("axis", "")
        suggested = item.get("narrative_suggests")
        if axis and suggested is not None:
            try:
                corrected_scores[axis] = max(1, min(99, int(suggested)))
            except (TypeError, ValueError):
                pass

    return SanityResult(
        scores_consistent=scores_consistent,
        inconsistencies=inconsistencies,
        overall_assessment=raw_dict.get("overall_assessment", ""),
        corrected_scores=corrected_scores,
        warning=warning,
    )


def apply_sanity_corrections(axis_scores, sanity: SanityResult):
    """
    If sanity check found inconsistencies, return corrected axis scores.
    Only corrects axes flagged as inconsistent — leaves others unchanged.
    Returns a new AxisScores-like dict (not mutating the original).
    """
    if sanity.scores_consistent or not sanity.corrected_scores:
        return None  # no corrections needed

    from axis_scoring import AxisScores
    corrections = sanity.corrected_scores

    risk       = corrections.get("risk",       axis_scores.risk)
    cashflow   = corrections.get("cashflow",   axis_scores.cashflow)
    obligation = corrections.get("obligation", axis_scores.obligation)
    context    = corrections.get("context",    axis_scores.context)

    from axis_scoring import compute_financial_capacity
    capacity = compute_financial_capacity(cashflow, obligation)

    return AxisScores(
        risk=risk,
        cashflow=cashflow,
        obligation=obligation,
        context=context,
        financial_capacity=capacity,
        risk_reasons=axis_scores.risk_reasons + [
            f"SANITY CORRECTION: narrative suggested {corrections.get('risk', risk)} "
            f"(was {axis_scores.risk})"
        ] if "risk" in corrections else axis_scores.risk_reasons,
        cashflow_reasons=axis_scores.cashflow_reasons,
        obligation_reasons=axis_scores.obligation_reasons + [
            f"SANITY CORRECTION: narrative suggested {corrections.get('obligation', obligation)} "
            f"(was {axis_scores.obligation})"
        ] if "obligation" in corrections else axis_scores.obligation_reasons,
        context_reasons=axis_scores.context_reasons,
    )


def sanity_to_dict(s: SanityResult) -> dict:
    return {
        "scores_consistent":  s.scores_consistent,
        "inconsistencies":    s.inconsistencies,
        "overall_assessment": s.overall_assessment,
        "corrected_scores":   s.corrected_scores,
        "warning":            s.warning,
    }
