"""
Narrative Layer — InvestorDNA v13
LLM-driven narrative understanding. FIRST reasoning step in the pipeline.

Architecture:
  raw_text + minimal_facts + SignalOutput → NarrativeOutput

v13 change: narrative receives SignalOutput as grounding context.
Signals are structured facts the LLM already extracted — the narrative
uses them as anchors to reason DEEPER, not to re-derive them.

Design principle:
  "Understand first. Compute second. Decide third."
"""

import json
import re
import requests
from dataclasses import dataclass, field

from llm_adapter import llm_call

_NARRATIVE_PROMPT = """You are an experienced financial advisor analyzing an investor.

Your task is NOT to classify. Your task is to UNDERSTAND and EXPLAIN the situation.

You have been given structured signals already extracted from the investor description.
Use these signals as ANCHORS — do not re-derive them. Reason DEEPER from them.

Instructions:
- Describe the investor's LIFE SITUATION in 2-3 sentences. Use the life events and responsibility signals as anchors. Explain what these signals MEAN for this investor's financial life — not just what they are.
- Describe their FINANCIAL REALITY: use the financial state signals and numeric facts. What is their ACTUAL investable capacity given all constraints?
- Describe their PSYCHOLOGICAL STATE: use the behavior and decision style signals. How do they behave under stress? Is their resilience genuine or fragile? What drives their decisions?
- Address the CONTRADICTIONS: if contradictions are listed, explain which trait DOMINATES under stress and why. Do not average them out.
- Explain whether their STATED RISK preference matches their REAL RISK capacity. Use the dominant_trait from contradictions if present.
- Assess the RELIABILITY of their profile: is this a stable picture or distorted by emotion, recency, or peer influence?
- Write an ADVISOR INSIGHT: the single most important thing an advisor must understand before recommending.

CRITICAL:
- If resilience_level is "high" with evidence, acknowledge it — do not suppress it
- If contradictions list a dominant_trait, that trait must appear in your risk_truth
- Do NOT output categories or enums
- Do NOT summarize — reason
- Be specific and direct

Return ONLY valid JSON with exactly these keys:

{{
  "life_summary": "2-3 sentences describing who this investor is and what they carry",
  "financial_analysis": "specific analysis of constraints, strengths, and real investable capacity",
  "psychological_analysis": "how they behave under stress, what drives decisions, fear vs conviction",
  "contradictions": "specific inconsistencies between stated preferences and actual situation, or 'none detected'",
  "risk_truth": "whether stated risk matches real risk capacity and behavior — be specific, use dominant_trait if present",
  "reliability_assessment": "is this profile stable and trustworthy, or distorted — explain why",
  "advisor_insight": "the single most important thing an advisor must understand before recommending"
}}

No markdown. No explanation outside the JSON. JSON only.

Investor description:
{text}

Extracted numeric facts (use as anchors):
{minimal_facts}

Structured signals (use as grounding — reason deeper from these):
{signal_context}
"""


@dataclass
class NarrativeOutput:
    life_summary: str           # who they are, what they carry
    financial_analysis: str     # real constraints, strengths, investable capacity
    psychological_analysis: str # behavior under stress, fear vs conviction
    contradictions: str         # inconsistencies between stated and actual
    risk_truth: str             # stated risk vs real risk capacity
    reliability_assessment: str # stable/trustworthy vs distorted
    advisor_insight: str        # single most important thing for advisor
    raw: dict = field(default_factory=dict)
    warning: str | None = None
    narrative_valid: bool = True


def _build_minimal_facts(plain_fields: dict) -> str:
    lines = []
    if plain_fields.get("monthly_income"):
        lines.append(f"Monthly income: ₹{plain_fields['monthly_income']:,.0f}")
    if plain_fields.get("emi_ratio") is not None:
        lines.append(f"EMI as % of income: {plain_fields['emi_ratio']:.1f}%")
    elif plain_fields.get("emi_amount") is not None:
        lines.append(f"EMI amount: ₹{plain_fields['emi_amount']:,.0f}/month")
    if plain_fields.get("emergency_months") is not None:
        lines.append(f"Emergency fund: {plain_fields['emergency_months']:.1f} months of expenses")
    if plain_fields.get("dependents") is not None:
        lines.append(f"Dependents: {plain_fields['dependents']}")
    if plain_fields.get("experience_years") is not None:
        lines.append(f"Investment experience: {plain_fields['experience_years']:.1f} years")
    if plain_fields.get("financial_knowledge_score") is not None:
        lines.append(f"Financial knowledge score: {plain_fields['financial_knowledge_score']}/5")
    if plain_fields.get("near_term_obligation_level") not in (None, "none"):
        otype = f" ({plain_fields['obligation_type']})" if plain_fields.get("obligation_type") else ""
        lines.append(f"Near-term obligation: {plain_fields['near_term_obligation_level']}{otype}")
    return "\n".join(lines) if lines else "No numeric facts extracted."


def _build_signal_context(signals) -> str:
    """
    Serialize SignalOutput into a concise grounding context for the narrative LLM.
    Only includes non-empty, meaningful signals.
    """
    if signals is None:
        return "No structured signals available."

    lines = []

    # Life events
    for ev in signals.life_events:
        lines.append(f"Life event: {ev.type} ({ev.recency}, impact={ev.impact}) — {ev.description}")

    # Responsibility
    resp = signals.responsibility
    if resp.role != "independent" or resp.dependents_count:
        lines.append(
            f"Responsibility: {resp.role} provider, "
            f"dependents={resp.dependents_count}, "
            f"financial_pressure={resp.financial_pressure}"
        )
        if resp.dependents_description:
            lines.append(f"  Dependents detail: {resp.dependents_description}")
    for ob in resp.cultural_obligations:
        lines.append(f"  Cultural obligation: {ob}")

    # Behavior
    beh = signals.behavior
    lines.append(
        f"Behavior: loss_response={beh.loss_response}, "
        f"resilience={beh.resilience_level}, "
        f"consistency={beh.consistency}"
    )
    if beh.loss_response_detail:
        lines.append(f"  Loss response detail: {beh.loss_response_detail}")
    if beh.resilience_evidence:
        lines.append(f"  Resilience evidence: {beh.resilience_evidence}")

    # Decision style
    ds = signals.decision_style
    lines.append(
        f"Decision style: autonomy={ds.autonomy}, "
        f"peer_influence={ds.peer_influence}, "
        f"analytical={ds.analytical_tendency}"
    )
    if ds.peer_influence_detail:
        lines.append(f"  Peer influence detail: {ds.peer_influence_detail}")

    # Financial state
    fs = signals.financial_state
    lines.append(
        f"Financial state: constraint={fs.constraint_level}, "
        f"obligation_scale={fs.obligation_scale}"
    )
    if fs.constraint_detail:
        lines.append(f"  Constraint detail: {fs.constraint_detail}")
    if fs.hidden_obligations:
        lines.append(f"  Hidden obligations: {fs.hidden_obligations}")

    # Contradictions — always included
    for c in signals.contradictions:
        lines.append(
            f"CONTRADICTION ({c.type}): dominant_trait='{c.dominant_trait}', "
            f"suppressed_trait='{c.suppressed_trait}' — {c.explanation}"
        )

    # Temporal context
    tc = signals.temporal_context
    if tc.has_behavioral_shift:
        lines.append(
            f"Behavioral shift: {tc.baseline_orientation} → {tc.current_orientation}"
        )
        if tc.shift_cause:
            lines.append(f"  Shift cause: {tc.shift_cause} (recency={tc.shift_recency}, permanence={tc.shift_permanence})")

    # Intent
    for intent in signals.intent:
        lines.append(
            f"Intent: {intent.goal} ({intent.category}, timeline={intent.timeline}, "
            f"firmness={intent.firmness}) — {intent.firmness_evidence}"
        )

    return "\n".join(lines) if lines else "No structured signals extracted."


def _parse_json(text: str) -> dict:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    return json.loads(text)


def _safe_str(val, default: str = "") -> str:
    if isinstance(val, str) and val.strip():
        return val.strip()
    return default


_REQUIRED_KEYS = {
    "life_summary", "financial_analysis", "psychological_analysis",
    "contradictions", "risk_truth", "reliability_assessment", "advisor_insight",
}


def generate_narrative(
    normalized_text: str,
    plain_fields: dict,
    signals=None,
) -> NarrativeOutput:
    """
    Call LLM to generate a free-text narrative understanding of the investor.

    v13: receives raw text + minimal numeric facts + SignalOutput.
    Signals are structured grounding context — the LLM reasons deeper from them.

    plain_fields: dict of validated field values (numbers only).
    signals: SignalOutput from signal_extraction (optional, degrades gracefully).
    """
    minimal_facts  = _build_minimal_facts(plain_fields)
    signal_context = _build_signal_context(signals)

    # Keep prompt size manageable for smaller cloud models
    if len(signal_context) > 1500:
        signal_context = signal_context[:1500] + "\n... [truncated for brevity]"

    prompt = _NARRATIVE_PROMPT.format(
        text=normalized_text[:1200],   # cap raw text too
        minimal_facts=minimal_facts,
        signal_context=signal_context,
    )

    warning  = None
    raw_dict = {}

    for attempt in (1, 2):
        try:
            raw_dict = llm_call(prompt, num_predict=2048)
            if not _REQUIRED_KEYS.issubset(raw_dict.keys()):
                missing = _REQUIRED_KEYS - raw_dict.keys()
                if attempt == 2:
                    warning = f"Narrative incomplete — missing keys: {missing}"
            else:
                break
        except (Exception,) as e:
            if attempt == 2:
                warning = f"Narrative generation failed after 2 attempts: {e}"
                raw_dict = {}

    narrative_valid = not (
        (warning and "failed" in warning) or
        (not raw_dict or not _REQUIRED_KEYS.issubset(raw_dict.keys()))
    )

    return NarrativeOutput(
        life_summary=_safe_str(
            raw_dict.get("life_summary"),
            "Life situation could not be assessed — insufficient information.",
        ),
        financial_analysis=_safe_str(
            raw_dict.get("financial_analysis"),
            "Financial analysis unavailable.",
        ),
        psychological_analysis=_safe_str(
            raw_dict.get("psychological_analysis"),
            "Psychological state could not be assessed.",
        ),
        contradictions=_safe_str(
            raw_dict.get("contradictions"),
            "none detected",
        ),
        risk_truth=_safe_str(
            raw_dict.get("risk_truth"),
            "Risk alignment could not be assessed.",
        ),
        reliability_assessment=_safe_str(
            raw_dict.get("reliability_assessment"),
            "Reliability unknown — insufficient data.",
        ),
        advisor_insight=_safe_str(
            raw_dict.get("advisor_insight"),
            "No specific advisor insight available.",
        ),
        raw=raw_dict,
        warning=warning,
        narrative_valid=narrative_valid,
    )


def narrative_to_dict(n: NarrativeOutput) -> dict:
    return {
        "life_summary":          n.life_summary,
        "financial_analysis":    n.financial_analysis,
        "psychological_analysis": n.psychological_analysis,
        "contradictions":        n.contradictions,
        "risk_truth":            n.risk_truth,
        "reliability_assessment": n.reliability_assessment,
        "advisor_insight":       n.advisor_insight,
        "warning":               n.warning,
    }


# ---------------------------------------------------------------------------
# Backward-compat shim: expose as extract_meaning / MeaningOutput alias
# so existing imports in main.py / cross_axis.py continue to work
# during the transition period.
# ---------------------------------------------------------------------------
extract_meaning  = generate_narrative
MeaningOutput    = NarrativeOutput
meaning_to_dict  = narrative_to_dict
