"""
Signal Extraction — InvestorDNA v2
LLM Call #2 of 3.

Single source of truth for ALL behavioral signals.
Simplified schema — designed for reliable output from 8B models.

Key design changes from v1:
- Flattened schema (no deep nesting)
- Reduced required fields
- num_predict capped at 768 (was 2048)
- Contradictions as a single string, not an array of objects
- All enums validated post-parse with safe defaults
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from llm_adapter import llm_call

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class Signals:
    # Life context
    life_event_type:    str   # none|death|job_change|marriage|crisis|other
    life_event_recency: str   # recent|past|unknown
    life_event_impact:  str   # low|medium|high

    # Responsibility
    provider_role:      str   # primary|shared|independent
    financial_pressure: str   # low|medium|high
    cultural_obligation: str  # description or ""

    # Behavior
    loss_response:      str   # panic|cautious|neutral|aggressive
    resilience_level:   str   # low|medium|high
    consistency:        str   # stable|inconsistent

    # Decision style
    autonomy:           str   # independent|influenced
    peer_influence:     str   # low|medium|high
    analytical:         str   # low|medium|high

    # Financial state
    constraint_level:   str   # low|medium|high
    hidden_obligations: str   # description or ""

    # Behavioral shift
    has_shift:          bool
    shift_permanence:   str   # likely_permanent|likely_temporary|unknown

    # Contradiction (dominant trait under stress)
    dominant_trait:     str   # panic|cautious|constrained|stable|aggressive|inconsistent|unknown
    contradiction_note: str   # one sentence or ""

    # Intent
    primary_intent:     str   # none|housing|vehicle|business|education|wedding|retirement|other
    intent_timeline:    str   # near|mid|long|none
    intent_firmness:    str   # firm|tentative|none

    valid:   bool = True
    warning: str | None = None


# ---------------------------------------------------------------------------
# Prompt — Call #2
# ---------------------------------------------------------------------------

_SIGNAL_PROMPT = """\
You are a financial behavior analyst. Extract signals from this investor description.

Return ONLY valid JSON with exactly these keys. Use the exact enum values shown.

{{
  "life_event_type":    "none|death|job_change|marriage|crisis|other",
  "life_event_recency": "recent|past|unknown",
  "life_event_impact":  "low|medium|high",
  "provider_role":      "primary|shared|independent",
  "financial_pressure": "low|medium|high",
  "cultural_obligation": "one sentence describing obligation, or empty string",
  "loss_response":      "panic|cautious|neutral|aggressive",
  "resilience_level":   "low|medium|high",
  "consistency":        "stable|inconsistent",
  "autonomy":           "independent|influenced",
  "peer_influence":     "low|medium|high",
  "analytical":         "low|medium|high",
  "constraint_level":   "low|medium|high",
  "hidden_obligations": "one sentence or empty string",
  "has_shift":          true or false,
  "shift_permanence":   "likely_permanent|likely_temporary|unknown",
  "dominant_trait":     "panic|cautious|constrained|stable|aggressive|inconsistent|unknown",
  "contradiction_note": "one sentence describing the main contradiction, or empty string",
  "primary_intent":     "none|housing|vehicle|business|education|wedding|retirement|other",
  "intent_timeline":    "near|mid|long|none",
  "intent_firmness":    "firm|tentative|none"
}}

Rules:
- dominant_trait is the behavior that ACTUALLY drives decisions under stress
- If loss_response is panic, dominant_trait MUST be "panic"
- If no life event, set life_event_type to "none"
- If no intent, set primary_intent to "none" and intent_timeline/firmness to "none"
- resilience_level MUST always be set — never omit

Investor description:
{text}
"""

# ---------------------------------------------------------------------------
# Enum sets for validation
# ---------------------------------------------------------------------------

_LIFE_EVENT_TYPES  = {"none", "death", "job_change", "marriage", "crisis", "other"}
_RECENCY           = {"recent", "past", "unknown"}
_IMPACT            = {"low", "medium", "high"}
_PROVIDER_ROLES    = {"primary", "shared", "independent"}
_PRESSURE          = {"low", "medium", "high"}
_LOSS_RESPONSES    = {"panic", "cautious", "neutral", "aggressive"}
_RESILIENCE        = {"low", "medium", "high"}
_CONSISTENCY       = {"stable", "inconsistent"}
_AUTONOMY          = {"independent", "influenced"}
_PEER              = {"low", "medium", "high"}
_ANALYTICAL        = {"low", "medium", "high"}
_CONSTRAINT        = {"low", "medium", "high"}
_SHIFT_PERM        = {"likely_permanent", "likely_temporary", "unknown"}
_DOMINANT_TRAITS   = {"panic", "cautious", "constrained", "stable", "aggressive", "inconsistent", "unknown"}
_INTENT_TYPES      = {"none", "housing", "vehicle", "business", "education", "wedding", "retirement", "other"}
_TIMELINES         = {"near", "mid", "long", "none"}
_FIRMNESS          = {"firm", "tentative", "none"}


def _e(val, valid_set: set, default: str) -> str:
    return val if isinstance(val, str) and val in valid_set else default


def _s(val, default: str = "") -> str:
    return val.strip() if isinstance(val, str) and val.strip() else default


def _b(val, default: bool = False) -> bool:
    if isinstance(val, bool): return val
    if isinstance(val, str):  return val.lower() == "true"
    return default


# ---------------------------------------------------------------------------
# Dominant trait normalization (fuzzy map for common LLM deviations)
# ---------------------------------------------------------------------------

_TRAIT_MAP = {
    "fear": "panic", "grief": "panic", "crisis": "panic", "anxious": "cautious",
    "conservative": "cautious", "risk_averse": "cautious", "risk averse": "cautious",
    "financially constrained": "constrained", "obligation": "constrained",
    "high constraint": "constrained", "no savings": "constrained",
    "disciplined": "stable", "structured": "stable", "balanced": "stable",
    "growth": "aggressive", "high risk": "aggressive", "speculative": "aggressive",
    "inconsistency": "inconsistent", "behavioral inconsistency": "inconsistent",
}


def _normalize_trait(raw: str) -> str:
    r = (raw or "").lower().strip()
    if r in _DOMINANT_TRAITS:
        return r
    for key, mapped in _TRAIT_MAP.items():
        if key in r:
            return mapped
    return "unknown"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_signals(text: str) -> Signals:
    """LLM Call #2 — extract all behavioral signals from raw text."""
    prompt = _SIGNAL_PROMPT.format(text=text[:1500])

    raw: dict = {}
    warning = None
    for attempt in (1, 2):
        try:
            raw = llm_call(prompt, num_predict=768)
            if raw:
                break
        except Exception as e:
            if attempt == 2:
                warning = f"Signal extraction failed: {e}"
                raw = {}

    if not raw:
        return Signals(
            life_event_type="none", life_event_recency="unknown", life_event_impact="low",
            provider_role="independent", financial_pressure="low", cultural_obligation="",
            loss_response="neutral", resilience_level="medium", consistency="stable",
            autonomy="independent", peer_influence="low", analytical="low",
            constraint_level="low", hidden_obligations="",
            has_shift=False, shift_permanence="unknown",
            dominant_trait="unknown", contradiction_note="",
            primary_intent="none", intent_timeline="none", intent_firmness="none",
            valid=False, warning=warning,
        )

    # Enforce: if loss_response=panic, dominant_trait must be panic
    loss_r = _e(raw.get("loss_response"), _LOSS_RESPONSES, "neutral")
    raw_trait = _normalize_trait(raw.get("dominant_trait", "unknown"))
    if loss_r == "panic":
        raw_trait = "panic"

    return Signals(
        life_event_type=_e(raw.get("life_event_type"),    _LIFE_EVENT_TYPES, "none"),
        life_event_recency=_e(raw.get("life_event_recency"), _RECENCY,        "unknown"),
        life_event_impact=_e(raw.get("life_event_impact"),   _IMPACT,         "low"),
        provider_role=_e(raw.get("provider_role"),           _PROVIDER_ROLES, "independent"),
        financial_pressure=_e(raw.get("financial_pressure"), _PRESSURE,       "low"),
        cultural_obligation=_s(raw.get("cultural_obligation")),
        loss_response=loss_r,
        resilience_level=_e(raw.get("resilience_level"),     _RESILIENCE,     "medium"),
        consistency=_e(raw.get("consistency"),               _CONSISTENCY,    "stable"),
        autonomy=_e(raw.get("autonomy"),                     _AUTONOMY,       "independent"),
        peer_influence=_e(raw.get("peer_influence"),         _PEER,           "low"),
        analytical=_e(raw.get("analytical"),                 _ANALYTICAL,     "low"),
        constraint_level=_e(raw.get("constraint_level"),     _CONSTRAINT,     "low"),
        hidden_obligations=_s(raw.get("hidden_obligations")),
        has_shift=_b(raw.get("has_shift"), False),
        shift_permanence=_e(raw.get("shift_permanence"),     _SHIFT_PERM,     "unknown"),
        dominant_trait=raw_trait,
        contradiction_note=_s(raw.get("contradiction_note")),
        primary_intent=_e(raw.get("primary_intent"),         _INTENT_TYPES,   "none"),
        intent_timeline=_e(raw.get("intent_timeline"),       _TIMELINES,      "none"),
        intent_firmness=_e(raw.get("intent_firmness"),       _FIRMNESS,       "none"),
        valid=True, warning=warning,
    )


def signals_to_dict(s: Signals) -> dict:
    return {
        "life_event_type": s.life_event_type, "life_event_recency": s.life_event_recency,
        "life_event_impact": s.life_event_impact, "provider_role": s.provider_role,
        "financial_pressure": s.financial_pressure, "cultural_obligation": s.cultural_obligation,
        "loss_response": s.loss_response, "resilience_level": s.resilience_level,
        "consistency": s.consistency, "autonomy": s.autonomy,
        "peer_influence": s.peer_influence, "analytical": s.analytical,
        "constraint_level": s.constraint_level, "hidden_obligations": s.hidden_obligations,
        "has_shift": s.has_shift, "shift_permanence": s.shift_permanence,
        "dominant_trait": s.dominant_trait, "contradiction_note": s.contradiction_note,
        "primary_intent": s.primary_intent, "intent_timeline": s.intent_timeline,
        "intent_firmness": s.intent_firmness,
        "valid": s.valid, "warning": s.warning,
    }
