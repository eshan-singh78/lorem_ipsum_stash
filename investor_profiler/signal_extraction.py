"""
Signal Extraction Layer — InvestorDNA v13
Unified LLM-driven signal extraction. Single source of truth for all signals.

Architecture:
  raw_text → SignalOutput

Design principle:
  ONE LLM call extracts ALL structured signals from raw text.
  Every downstream layer (profile_context, narrative, scoring, decision)
  reads from this single signal source — no regex, no re-derivation.

Signals captured:
  - life_events: what happened, recency, emotional impact
  - responsibility: role, dependents, financial pressure
  - behavior: loss response, consistency, resilience_level
  - decision_style: autonomy, peer influence degree
  - financial_state: constraint level, obligation scale
  - contradictions: structured with dominant_trait (never dropped)
  - temporal_context: behavioral transitions with timeline
  - intent: future plans with firmness level
"""

import json
import re
from dataclasses import dataclass, field

from llm_adapter import llm_call

_SIGNAL_PROMPT = """You are a financial behavior analyst.

Extract ALL meaningful signals from this investor description.
DO NOT summarize. DO NOT classify broadly. Extract specific structured signals.

CRITICAL RULES:
- resilience_level must ALWAYS be set — never omit it
- contradictions must ALWAYS be populated if any inconsistency exists
- dominant_trait in contradictions is the behavior that ACTUALLY drives decisions under stress
- temporal_context must capture behavioral shifts with their cause and timeline
- intent entries must distinguish firm plans from vague aspirations

Return ONLY valid JSON with exactly this structure:

{{
  "life_events": [
    {{
      "type": "death | job_change | marriage | responsibility_shift | crisis | other",
      "description": "specific description of what happened",
      "recency": "recent | past | unknown",
      "impact": "low | medium | high"
    }}
  ],
  "responsibility": {{
    "role": "primary | shared | independent",
    "dependents_count": integer or null,
    "dependents_description": "who depends on them and how",
    "financial_pressure": "low | medium | high",
    "cultural_obligations": ["list of specific obligations detected, empty if none"]
  }},
  "behavior": {{
    "loss_response": "panic | cautious | neutral | aggressive",
    "loss_response_detail": "specific description of how they responded — what they did, when, did they recover",
    "consistency": "stable | inconsistent",
    "resilience_level": "low | medium | high",
    "resilience_evidence": "specific text evidence for resilience level"
  }},
  "decision_style": {{
    "autonomy": "independent | influenced",
    "peer_influence": "low | medium | high",
    "peer_influence_detail": "specific description of how peer influence manifests",
    "analytical_tendency": "low | medium | high"
  }},
  "financial_state": {{
    "constraint_level": "low | medium | high",
    "constraint_detail": "specific constraints identified",
    "obligation_scale": "low | medium | high",
    "hidden_obligations": "description of any hidden or unspoken financial obligations, or null"
  }},
  "contradictions": [
    {{
      "type": "knowledge_vs_behavior | stated_vs_actual | risk_vs_capacity | other",
      "dominant_trait": "the behavior that ACTUALLY drives decisions under stress",
      "suppressed_trait": "the trait that is stated but overridden under pressure",
      "explanation": "specific explanation of the contradiction and which trait dominates"
    }}
  ],
  "temporal_context": {{
    "has_behavioral_shift": true or false,
    "baseline_orientation": "description of historical or natural risk orientation",
    "current_orientation": "description of current behavior",
    "shift_cause": "what caused the shift, or null if no shift",
    "shift_recency": "recent | past | unknown | null",
    "shift_permanence": "likely_permanent | likely_temporary | unknown | null"
  }},
  "intent": [
    {{
      "goal": "description of the future plan",
      "category": "housing | vehicle | business | education | wedding | travel | retirement | other",
      "timeline": "near | mid | long",
      "firmness": "firm | tentative",
      "firmness_evidence": "what in the text indicates firmness or tentativeness"
    }}
  ]
}}

If a section has no signals, return empty list [] or null values — do not omit the key.
No markdown. JSON only.

Investor description:
{text}
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LifeEventSignal:
    type: str           # death | job_change | marriage | responsibility_shift | crisis | other
    description: str
    recency: str        # recent | past | unknown
    impact: str         # low | medium | high


@dataclass
class ResponsibilitySignal:
    role: str                       # primary | shared | independent
    dependents_count: int | None
    dependents_description: str
    financial_pressure: str         # low | medium | high
    cultural_obligations: list[str]


@dataclass
class BehaviorSignal:
    loss_response: str              # panic | cautious | neutral | aggressive
    loss_response_detail: str       # specific narrative of how they responded
    consistency: str                # stable | inconsistent
    resilience_level: str           # low | medium | high  — NEVER dropped
    resilience_evidence: str        # text evidence


@dataclass
class DecisionStyleSignal:
    autonomy: str                   # independent | influenced
    peer_influence: str             # low | medium | high
    peer_influence_detail: str
    analytical_tendency: str        # low | medium | high


@dataclass
class FinancialStateSignal:
    constraint_level: str           # low | medium | high
    constraint_detail: str
    obligation_scale: str           # low | medium | high
    hidden_obligations: str | None


@dataclass
class ContradictionSignal:
    type: str                       # knowledge_vs_behavior | stated_vs_actual | risk_vs_capacity | other
    dominant_trait: str             # what ACTUALLY drives decisions under stress
    suppressed_trait: str           # what is stated but overridden
    explanation: str


@dataclass
class TemporalContext:
    has_behavioral_shift: bool
    baseline_orientation: str
    current_orientation: str
    shift_cause: str | None
    shift_recency: str | None       # recent | past | unknown | null
    shift_permanence: str | None    # likely_permanent | likely_temporary | unknown | null


@dataclass
class IntentSignal:
    goal: str
    category: str                   # housing | vehicle | business | education | wedding | travel | retirement | other
    timeline: str                   # near | mid | long
    firmness: str                   # firm | tentative
    firmness_evidence: str


@dataclass
class SignalOutput:
    life_events: list[LifeEventSignal]
    responsibility: ResponsibilitySignal
    behavior: BehaviorSignal
    decision_style: DecisionStyleSignal
    financial_state: FinancialStateSignal
    contradictions: list[ContradictionSignal]   # ALWAYS passed downstream — never dropped
    temporal_context: TemporalContext
    intent: list[IntentSignal]
    raw: dict = field(default_factory=dict)
    warning: str | None = None
    signals_valid: bool = True


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

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
    return val.strip() if isinstance(val, str) and val.strip() else default


def _safe_enum(val, valid: set, default: str) -> str:
    return val if val in valid else default


def _parse_life_events(raw: list) -> list[LifeEventSignal]:
    if not isinstance(raw, list):
        return []
    events = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        events.append(LifeEventSignal(
            type=_safe_enum(item.get("type"), {
                "death", "job_change", "marriage", "responsibility_shift", "crisis", "other"
            }, "other"),
            description=_safe_str(item.get("description"), "Life event detected"),
            recency=_safe_enum(item.get("recency"), {"recent", "past", "unknown"}, "unknown"),
            impact=_safe_enum(item.get("impact"), {"low", "medium", "high"}, "medium"),
        ))
    return events


def _parse_responsibility(raw: dict | None) -> ResponsibilitySignal:
    if not isinstance(raw, dict):
        return ResponsibilitySignal(
            role="independent", dependents_count=None,
            dependents_description="", financial_pressure="low",
            cultural_obligations=[],
        )
    deps = raw.get("dependents_count")
    if deps is not None:
        try:
            deps = int(deps)
        except (TypeError, ValueError):
            deps = None
    return ResponsibilitySignal(
        role=_safe_enum(raw.get("role"), {"primary", "shared", "independent"}, "independent"),
        dependents_count=deps,
        dependents_description=_safe_str(raw.get("dependents_description")),
        financial_pressure=_safe_enum(raw.get("financial_pressure"), {"low", "medium", "high"}, "low"),
        cultural_obligations=raw.get("cultural_obligations", []) if isinstance(raw.get("cultural_obligations"), list) else [],
    )


def _parse_behavior(raw: dict | None) -> BehaviorSignal:
    if not isinstance(raw, dict):
        return BehaviorSignal(
            loss_response="neutral", loss_response_detail="",
            consistency="stable", resilience_level="medium",
            resilience_evidence="",
        )
    return BehaviorSignal(
        loss_response=_safe_enum(raw.get("loss_response"), {
            "panic", "cautious", "neutral", "aggressive"
        }, "neutral"),
        loss_response_detail=_safe_str(raw.get("loss_response_detail")),
        consistency=_safe_enum(raw.get("consistency"), {"stable", "inconsistent"}, "stable"),
        resilience_level=_safe_enum(raw.get("resilience_level"), {"low", "medium", "high"}, "medium"),
        resilience_evidence=_safe_str(raw.get("resilience_evidence")),
    )


def _parse_decision_style(raw: dict | None) -> DecisionStyleSignal:
    if not isinstance(raw, dict):
        return DecisionStyleSignal(
            autonomy="independent", peer_influence="low",
            peer_influence_detail="", analytical_tendency="low",
        )
    return DecisionStyleSignal(
        autonomy=_safe_enum(raw.get("autonomy"), {"independent", "influenced"}, "independent"),
        peer_influence=_safe_enum(raw.get("peer_influence"), {"low", "medium", "high"}, "low"),
        peer_influence_detail=_safe_str(raw.get("peer_influence_detail")),
        analytical_tendency=_safe_enum(raw.get("analytical_tendency"), {"low", "medium", "high"}, "low"),
    )


def _parse_financial_state(raw: dict | None) -> FinancialStateSignal:
    if not isinstance(raw, dict):
        return FinancialStateSignal(
            constraint_level="low", constraint_detail="",
            obligation_scale="low", hidden_obligations=None,
        )
    ho = raw.get("hidden_obligations")
    return FinancialStateSignal(
        constraint_level=_safe_enum(raw.get("constraint_level"), {"low", "medium", "high"}, "low"),
        constraint_detail=_safe_str(raw.get("constraint_detail")),
        obligation_scale=_safe_enum(raw.get("obligation_scale"), {"low", "medium", "high"}, "low"),
        hidden_obligations=_safe_str(ho) if ho and ho != "null" else None,
    )


def _parse_contradictions(raw: list) -> list[ContradictionSignal]:
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        result.append(ContradictionSignal(
            type=_safe_enum(item.get("type"), {
                "knowledge_vs_behavior", "stated_vs_actual",
                "risk_vs_capacity", "other"
            }, "other"),
            dominant_trait=_safe_str(item.get("dominant_trait"), "unknown"),
            suppressed_trait=_safe_str(item.get("suppressed_trait"), "unknown"),
            explanation=_safe_str(item.get("explanation")),
        ))
    return result


def _parse_temporal_context(raw: dict | None) -> TemporalContext:
    if not isinstance(raw, dict):
        return TemporalContext(
            has_behavioral_shift=False,
            baseline_orientation="",
            current_orientation="",
            shift_cause=None,
            shift_recency=None,
            shift_permanence=None,
        )
    return TemporalContext(
        has_behavioral_shift=bool(raw.get("has_behavioral_shift", False)),
        baseline_orientation=_safe_str(raw.get("baseline_orientation")),
        current_orientation=_safe_str(raw.get("current_orientation")),
        shift_cause=_safe_str(raw.get("shift_cause")) or None,
        shift_recency=_safe_enum(
            raw.get("shift_recency"),
            {"recent", "past", "unknown", None}, None
        ),
        shift_permanence=_safe_enum(
            raw.get("shift_permanence"),
            {"likely_permanent", "likely_temporary", "unknown", None}, None
        ),
    )


def _parse_intent(raw: list) -> list[IntentSignal]:
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        result.append(IntentSignal(
            goal=_safe_str(item.get("goal"), "Future plan detected"),
            category=_safe_enum(item.get("category"), {
                "housing", "vehicle", "business", "education",
                "wedding", "travel", "retirement", "other"
            }, "other"),
            timeline=_safe_enum(item.get("timeline"), {"near", "mid", "long"}, "mid"),
            firmness=_safe_enum(item.get("firmness"), {"firm", "tentative"}, "tentative"),
            firmness_evidence=_safe_str(item.get("firmness_evidence")),
        ))
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_signals(raw_text: str) -> SignalOutput:
    """
    Single LLM call to extract ALL structured signals from investor text.
    This is the ONLY place signals are extracted — no regex anywhere else.

    Returns SignalOutput — the single source of truth for all downstream layers.
    """
    prompt = _SIGNAL_PROMPT.format(text=raw_text)

    warning  = None
    raw_dict = {}

    for attempt in (1, 2):
        try:
            raw_dict = llm_call(prompt, num_predict=2048)
            if raw_dict:
                break
        except (Exception,) as e:
            if attempt == 2:
                warning = f"Signal extraction failed after 2 attempts: {e}"
                raw_dict = {}

    if not raw_dict:
        return SignalOutput(
            life_events=_parse_life_events([]),
            responsibility=_parse_responsibility(None),
            behavior=_parse_behavior(None),
            decision_style=_parse_decision_style(None),
            financial_state=_parse_financial_state(None),
            contradictions=_parse_contradictions([]),
            temporal_context=_parse_temporal_context(None),
            intent=_parse_intent([]),
            raw={},
            warning=warning,
            signals_valid=False,
        )

    return SignalOutput(
        life_events=_parse_life_events(raw_dict.get("life_events", [])),
        responsibility=_parse_responsibility(raw_dict.get("responsibility")),
        behavior=_parse_behavior(raw_dict.get("behavior")),
        decision_style=_parse_decision_style(raw_dict.get("decision_style")),
        financial_state=_parse_financial_state(raw_dict.get("financial_state")),
        contradictions=_parse_contradictions(raw_dict.get("contradictions", [])),
        temporal_context=_parse_temporal_context(raw_dict.get("temporal_context")),
        intent=_parse_intent(raw_dict.get("intent", [])),
        raw=raw_dict,
        warning=warning,
        signals_valid=True,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def signals_to_dict(s: SignalOutput) -> dict:
    return {
        "life_events": [
            {"type": e.type, "description": e.description,
             "recency": e.recency, "impact": e.impact}
            for e in s.life_events
        ],
        "responsibility": {
            "role": s.responsibility.role,
            "dependents_count": s.responsibility.dependents_count,
            "dependents_description": s.responsibility.dependents_description,
            "financial_pressure": s.responsibility.financial_pressure,
            "cultural_obligations": s.responsibility.cultural_obligations,
        },
        "behavior": {
            "loss_response": s.behavior.loss_response,
            "loss_response_detail": s.behavior.loss_response_detail,
            "consistency": s.behavior.consistency,
            "resilience_level": s.behavior.resilience_level,
            "resilience_evidence": s.behavior.resilience_evidence,
        },
        "decision_style": {
            "autonomy": s.decision_style.autonomy,
            "peer_influence": s.decision_style.peer_influence,
            "peer_influence_detail": s.decision_style.peer_influence_detail,
            "analytical_tendency": s.decision_style.analytical_tendency,
        },
        "financial_state": {
            "constraint_level": s.financial_state.constraint_level,
            "constraint_detail": s.financial_state.constraint_detail,
            "obligation_scale": s.financial_state.obligation_scale,
            "hidden_obligations": s.financial_state.hidden_obligations,
        },
        "contradictions": [
            {"type": c.type, "dominant_trait": c.dominant_trait,
             "suppressed_trait": c.suppressed_trait, "explanation": c.explanation}
            for c in s.contradictions
        ],
        "temporal_context": {
            "has_behavioral_shift": s.temporal_context.has_behavioral_shift,
            "baseline_orientation": s.temporal_context.baseline_orientation,
            "current_orientation": s.temporal_context.current_orientation,
            "shift_cause": s.temporal_context.shift_cause,
            "shift_recency": s.temporal_context.shift_recency,
            "shift_permanence": s.temporal_context.shift_permanence,
        },
        "intent": [
            {"goal": i.goal, "category": i.category, "timeline": i.timeline,
             "firmness": i.firmness, "firmness_evidence": i.firmness_evidence}
            for i in s.intent
        ],
        "warning": s.warning,
    }
