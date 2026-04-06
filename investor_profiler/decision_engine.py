"""
Decision Engine — InvestorDNA v17 (8B-optimised)

Architecture:
  NarrativeOutput + InvestorState + SignalOutput → DecisionOutput

v17 changes:
  - Split single overloaded prompt into CALL A (reasoning) + CALL B (decision)
  - dominant_trait validated against enum, not verbatim substring match
  - strategy is structured dict with actionable fields
  - risk_assessment includes explicit checklist
  - Correction loop injects actual contradiction data
  - Prompts stripped to task + schema + 2-3 rules max

Design principle: Break one complex task into multiple simple tasks.
"""

import json
import re
import requests
from dataclasses import dataclass, field

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL       = "llama3.1:8b"
MAX_RETRIES     = 3
DEBUG_REASONING = False

# ---------------------------------------------------------------------------
# Allowed enum for dominant_trait — validated deterministically
# ---------------------------------------------------------------------------

DOMINANT_TRAIT_ENUM = frozenset({
    "panic", "cautious", "constrained", "stable",
    "aggressive", "inconsistent", "unknown",
})

# ---------------------------------------------------------------------------
# CALL A — Reasoning prompt (signals → trace only, no allocation)
# ---------------------------------------------------------------------------

_REASONING_PROMPT = """Analyze this investor and extract reasoning signals.

INVESTOR NARRATIVE:
Life: {life_summary}
Financial: {financial_analysis}
Psychology: {psychological_analysis}
Risk truth: {risk_truth}

INVESTOR STATE:
Compound state: {compound_state}
Dominant factors: {dominant_factors}
Shift detected: {shift_detected}

SIGNAL CONTRADICTIONS:
{structured_contradictions}

Return JSON only:
{{
  "signals_considered": ["list key signals, max 6"],
  "dominant_factors": ["1-3 short phrases, max 5 words each"],
  "secondary_factors": ["real but overridden"],
  "contradictions": [
    {{"signal_1": "...", "signal_2": "...", "resolution": "...", "dominant_trait": "..."}}
  ],
  "state_inference": "one sentence summary of investor condition",
  "decision_logic": ["reason 1", "reason 2"]
}}

Rules:
- dominant_factors must be specific, not abstract
- Do NOT generate allocation or strategy
- contradictions list may be empty if none exist
"""

# ---------------------------------------------------------------------------
# CALL B — Decision prompt (trace → allocation + strategy)
# ---------------------------------------------------------------------------

_DECISION_PROMPT = """Generate an investment decision from this reasoning.

REASONING TRACE:
Signals: {signals_considered}
Dominant factors: {dominant_factors}
State inference: {state_inference}
Contradictions resolved: {contradictions_summary}

PRIORITY: {priority_order}

Return JSON only:
{{
  "current_allocation": "X-Y% equity RIGHT NOW",
  "baseline_allocation": "X-Y% equity long-term",
  "allocation_mode": "normal | conservative | defensive",
  "state_context": {{
    "compound_state": "short phrase",
    "dominant_trait": "one of: panic|cautious|constrained|stable|aggressive|inconsistent|unknown",
    "resilience_level": "low | medium | high",
    "state_stability": "stable | transitional | unstable"
  }},
  "temporal_strategy": {{
    "is_temporary": true or false,
    "reassessment_trigger": "specific condition",
    "reassessment_timeline": "e.g. 6-12 months",
    "expected_shift": "how allocation changes or none"
  }},
  "risk_assessment": {{
    "identified_risks": ["check each: no emergency fund, income instability, no savings, upcoming major expense, no investment experience"],
    "true_risk_capacity": "1-2 sentences",
    "stated_vs_actual_gap": "description or aligned",
    "reliability": "low | medium | high"
  }},
  "strategy": {{
    "primary_instrument": "e.g. liquid fund / equity SIP / FD",
    "equity_pct": 0,
    "debt_pct": 0,
    "sip_recommended": true or false,
    "first_step": "one clear actionable step with amount or %"
  }},
  "archetype": "short descriptive label",
  "confidence": "high | medium | low",
  "reasoning": "3-4 sentences referencing dominant factors",
  "advisor_note": "single most important caution"
}}

Rules:
- dominant_trait MUST be from the enum above
- If dominant_trait is panic or cautious, current_allocation must be ≤ 25%
- allocation_mode must be conservative or defensive when dominant_trait is panic/cautious/constrained
"""

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ContradictionResolution:
    signal_1: str
    signal_2: str
    resolution: str
    dominant_trait: str


@dataclass
class ReasoningTrace:
    signals_considered: list[str]
    dominant_factors: list[str]
    secondary_factors: list[str]
    contradictions: list[ContradictionResolution]
    state_inference: str
    decision_logic: list[str]


@dataclass
class StateContext:
    compound_state: str
    dominant_trait: str
    resilience_level: str
    state_stability: str


@dataclass
class TemporalStrategy:
    is_temporary: bool
    reassessment_trigger: str
    reassessment_timeline: str
    expected_shift: str


@dataclass
class RiskAssessment:
    identified_risks: list[str]
    true_risk_capacity: str
    stated_vs_actual_gap: str
    reliability: str


@dataclass
class Strategy:
    primary_instrument: str
    equity_pct: int
    debt_pct: int
    sip_recommended: bool
    first_step: str


@dataclass
class DecisionOutput:
    reasoning_trace: ReasoningTrace
    current_allocation: str
    baseline_allocation: str
    allocation_mode: str
    state_context: StateContext
    temporal_strategy: TemporalStrategy
    risk_assessment: RiskAssessment
    strategy_structured: Strategy
    archetype: str
    strategy: str           # legacy string form — derived from strategy_structured
    confidence: str
    reasoning: str
    advisor_note: str
    raw: dict = field(default_factory=dict)
    warning: str | None = None
    guardrail_adjustments: list = field(default_factory=list)
    retry_count: int = 0
    fallback_used: bool = False

    @property
    def equity_range(self) -> str:
        return self.current_allocation

    @property
    def true_risk_profile(self) -> str:
        return self.risk_assessment.true_risk_capacity

    @property
    def recommended_strategy(self) -> str:
        return self.strategy

    @property
    def confidence_level(self) -> str:
        return self.confidence


# ---------------------------------------------------------------------------
# Helpers
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


def _safe(val, default: str = "") -> str:
    return val.strip() if isinstance(val, str) and val.strip() else default


def _safe_bool(val, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() == "true"
    return default


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _normalise_dominant_trait(raw: str) -> str:
    """Map raw LLM output to the closest allowed enum value."""
    raw_lower = raw.lower().strip()
    if raw_lower in DOMINANT_TRAIT_ENUM:
        return raw_lower
    # Fuzzy map common synonyms
    _MAP = {
        "fear": "panic", "grief": "panic", "crisis": "panic", "anxious": "cautious",
        "conservative": "cautious", "risk_averse": "cautious", "risk averse": "cautious",
        "financially constrained": "constrained", "obligation": "constrained",
        "high constraint": "constrained", "no savings": "constrained",
        "disciplined": "stable", "structured": "stable", "balanced": "stable",
        "growth": "aggressive", "high risk": "aggressive", "speculative": "aggressive",
        "inconsistency": "inconsistent", "behavioral inconsistency": "inconsistent",
    }
    for key, mapped in _MAP.items():
        if key in raw_lower:
            return mapped
    return "unknown"


def _build_structured_contradictions(signals) -> str:
    if signals is None or not signals.contradictions:
        return "none"
    lines = []
    for c in signals.contradictions:
        lines.append(
            f"- dominant: '{c.dominant_trait}' vs suppressed: '{c.suppressed_trait}' "
            f"({c.explanation})"
        )
    return "\n".join(lines)


def _contradictions_summary(trace_raw: dict) -> str:
    contras = trace_raw.get("contradictions", [])
    if not contras:
        return "none"
    parts = []
    for c in contras:
        if isinstance(c, dict):
            parts.append(
                f"{c.get('signal_1','?')} vs {c.get('signal_2','?')} "
                f"→ '{c.get('dominant_trait','?')}' wins"
            )
    return "; ".join(parts) if parts else "none"


def _parse_reasoning_trace(raw: dict) -> ReasoningTrace:
    rt = raw.get("reasoning_trace") or raw  # Call A returns trace at top level

    def _lst(key):
        v = rt.get(key, [])
        return v if isinstance(v, list) else ([str(v)] if v else [])

    contradictions = []
    for c in _lst("contradictions"):
        if isinstance(c, dict):
            contradictions.append(ContradictionResolution(
                signal_1=_safe(c.get("signal_1"), "unknown"),
                signal_2=_safe(c.get("signal_2"), "unknown"),
                resolution=_safe(c.get("resolution"), ""),
                dominant_trait=_safe(c.get("dominant_trait"), "unknown"),
            ))

    return ReasoningTrace(
        signals_considered=_lst("signals_considered"),
        dominant_factors=_lst("dominant_factors"),
        secondary_factors=_lst("secondary_factors"),
        contradictions=contradictions,
        state_inference=_safe(rt.get("state_inference"), ""),
        decision_logic=_lst("decision_logic"),
    )


def _parse_strategy(raw: dict) -> Strategy:
    s = raw.get("strategy") or {}
    if isinstance(s, str):
        # Legacy string — wrap it
        return Strategy(
            primary_instrument=s,
            equity_pct=0, debt_pct=0,
            sip_recommended=False,
            first_step=s,
        )
    return Strategy(
        primary_instrument=_safe(s.get("primary_instrument"), "liquid fund"),
        equity_pct=_safe_int(s.get("equity_pct"), 0),
        debt_pct=_safe_int(s.get("debt_pct"), 0),
        sip_recommended=_safe_bool(s.get("sip_recommended"), False),
        first_step=_safe(s.get("first_step"), "Consult advisor for first step."),
    )


def _parse_risk_assessment(raw: dict) -> RiskAssessment:
    ra = raw.get("risk_assessment") or {}
    identified = ra.get("identified_risks", [])
    if not isinstance(identified, list):
        identified = []
    return RiskAssessment(
        identified_risks=identified,
        true_risk_capacity=_safe(ra.get("true_risk_capacity"), "Could not be determined."),
        stated_vs_actual_gap=_safe(ra.get("stated_vs_actual_gap"), "unknown"),
        reliability=_safe(ra.get("reliability"), "medium"),
    )


def _parse_decision(trace_raw: dict, decision_raw: dict) -> DecisionOutput:
    """Merge Call A (trace) + Call B (decision) into DecisionOutput."""
    sc_raw = decision_raw.get("state_context") or {}
    ts_raw = decision_raw.get("temporal_strategy") or {}

    reasoning_trace = _parse_reasoning_trace(trace_raw)

    raw_dominant = _safe(sc_raw.get("dominant_trait"), "unknown")
    dominant_trait = _normalise_dominant_trait(raw_dominant)

    state_context = StateContext(
        compound_state=_safe(sc_raw.get("compound_state"), "unknown"),
        dominant_trait=dominant_trait,
        resilience_level=_safe(sc_raw.get("resilience_level"), "medium"),
        state_stability=_safe(sc_raw.get("state_stability"), "stable"),
    )

    temporal_strategy = TemporalStrategy(
        is_temporary=_safe_bool(ts_raw.get("is_temporary"), False),
        reassessment_trigger=_safe(ts_raw.get("reassessment_trigger"), "standard annual review"),
        reassessment_timeline=_safe(ts_raw.get("reassessment_timeline"), "12 months"),
        expected_shift=_safe(ts_raw.get("expected_shift"), "none"),
    )

    risk_assessment = _parse_risk_assessment(decision_raw)
    strategy_structured = _parse_strategy(decision_raw)

    # Derive legacy strategy string from structured
    strategy_str = (
        f"{strategy_structured.primary_instrument}; "
        f"equity {strategy_structured.equity_pct}%, debt {strategy_structured.debt_pct}%. "
        f"{strategy_structured.first_step}"
    )

    allocation_mode = _safe(decision_raw.get("allocation_mode"), "normal")
    if allocation_mode not in ("normal", "conservative", "defensive",
                               "static", "transitional", "conditional"):
        allocation_mode = "normal"

    confidence = _safe(decision_raw.get("confidence"), "low")
    if confidence not in ("high", "medium", "low"):
        confidence = "low"

    warning = None
    if confidence == "low":
        warning = "Provisional — requires further advisor validation."

    return DecisionOutput(
        reasoning_trace=reasoning_trace,
        current_allocation=_safe(decision_raw.get("current_allocation"), "0-10%"),
        baseline_allocation=_safe(decision_raw.get("baseline_allocation"), "0-10%"),
        allocation_mode=allocation_mode,
        state_context=state_context,
        temporal_strategy=temporal_strategy,
        risk_assessment=risk_assessment,
        strategy_structured=strategy_structured,
        archetype=_safe(decision_raw.get("archetype"), "Unclassified"),
        strategy=strategy_str,
        confidence=confidence,
        reasoning=_safe(decision_raw.get("reasoning"), "Reasoning unavailable."),
        advisor_note=_safe(decision_raw.get("advisor_note"), "No specific advisor note."),
        raw={**trace_raw, **decision_raw},
        warning=warning,
    )


def _empty_trace() -> ReasoningTrace:
    return ReasoningTrace(
        signals_considered=[], dominant_factors=[], secondary_factors=[],
        contradictions=[], state_inference="", decision_logic=[],
    )


def _fallback_decision(retry_count: int = 0) -> DecisionOutput:
    return DecisionOutput(
        reasoning_trace=_empty_trace(),
        current_allocation="5-10%",
        baseline_allocation="15-20%",
        allocation_mode="conservative",
        state_context=StateContext(
            compound_state="unknown — LLM unavailable",
            dominant_trait="unknown",
            resilience_level="medium",
            state_stability="stable",
        ),
        temporal_strategy=TemporalStrategy(
            is_temporary=False,
            reassessment_trigger="Retry when LLM is available",
            reassessment_timeline="immediate",
            expected_shift="none",
        ),
        risk_assessment=RiskAssessment(
            identified_risks=[],
            true_risk_capacity="Could not be determined — LLM unavailable.",
            stated_vs_actual_gap="unknown",
            reliability="low",
        ),
        strategy_structured=Strategy(
            primary_instrument="liquid fund",
            equity_pct=5, debt_pct=95,
            sip_recommended=False,
            first_step="Park funds in liquid fund until advisor assessment is complete.",
        ),
        archetype="Unclassified",
        strategy="Conservative fallback — manual advisor assessment required.",
        confidence="low",
        reasoning="Decision engine failed after all retries — conservative fallback applied.",
        advisor_note="Manual advisor assessment required. LLM was unavailable.",
        raw={},
        warning=f"LLM failed after {retry_count} retries — conservative fallback used.",
        retry_count=retry_count,
        fallback_used=True,
    )


# ---------------------------------------------------------------------------
# Priority resolution
# ---------------------------------------------------------------------------

_DEFAULT_PRIORITY = (
    "behavioral > financial constraint > life events > knowledge"
)
_FINANCIAL_FIRST_PRIORITY = (
    "financial constraint (HIGH) > behavioral > life events > knowledge"
)


def resolve_priority(signals, investor_state) -> str:
    constraint = "low"
    if signals is not None:
        constraint = getattr(getattr(signals, "financial_state", None), "constraint_level", "low")
    if constraint == "high":
        return _FINANCIAL_FIRST_PRIORITY
    return _DEFAULT_PRIORITY


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

def _llm_call(prompt: str, num_predict: int = 1024) -> dict:
    """Single Ollama call. Returns parsed dict or raises."""
    payload = {
        "model":   LLM_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0, "num_predict": num_predict},
        "format":  "json",
    }
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=None,
    )
    resp.raise_for_status()
    return _parse_json(resp.json().get("response", ""))


def _call_reasoning(narrative, investor_state, signals) -> dict:
    """Call A — extract reasoning trace only."""
    prompt = _REASONING_PROMPT.format(
        life_summary=getattr(narrative, "life_summary", "Not available"),
        financial_analysis=getattr(narrative, "financial_analysis", "Not available"),
        psychological_analysis=getattr(narrative, "psychological_analysis", "Not available"),
        risk_truth=getattr(narrative, "risk_truth", "Not available"),
        compound_state=getattr(investor_state, "compound_state", "unknown") if investor_state else "unknown",
        dominant_factors=", ".join(getattr(investor_state, "dominant_factors", [])) if investor_state else "",
        shift_detected=str(getattr(investor_state, "shift_detected", False)).lower() if investor_state else "false",
        structured_contradictions=_build_structured_contradictions(signals),
    )
    for attempt in (1, 2):
        try:
            raw = _llm_call(prompt, num_predict=768)
            if raw.get("dominant_factors") and raw.get("state_inference"):
                return raw
        except (requests.RequestException, json.JSONDecodeError, ValueError):
            if attempt == 2:
                return {}
    return {}


def _call_decision(trace_raw: dict, priority_str: str, correction_hint: str = "") -> dict:
    """Call B — generate allocation + strategy from trace."""
    prompt = _DECISION_PROMPT.format(
        signals_considered=", ".join(trace_raw.get("signals_considered", [])),
        dominant_factors=", ".join(trace_raw.get("dominant_factors", [])),
        state_inference=trace_raw.get("state_inference", ""),
        contradictions_summary=_contradictions_summary(trace_raw),
        priority_order=priority_str,
    )
    if correction_hint:
        prompt += (
            f"\n\n⚠️ CORRECTION: Fix these issues from your previous output:\n"
            f"{correction_hint}\n"
            f"dominant_trait MUST be one of: {', '.join(sorted(DOMINANT_TRAIT_ENUM))}.\n"
            f"Do NOT invent new traits."
        )
    for attempt in (1, 2):
        try:
            raw = _llm_call(prompt, num_predict=1024)
            if raw.get("current_allocation") and raw.get("state_context"):
                return raw
        except (requests.RequestException, json.JSONDecodeError, ValueError):
            if attempt == 2:
                return {}
    return {}


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _auto_fix_dominant_trait(decision: DecisionOutput) -> None:
    """If dominant_trait is not in enum after retries, map it or set unknown."""
    raw = decision.state_context.dominant_trait
    normalised = _normalise_dominant_trait(raw)
    if normalised != raw:
        decision.state_context.dominant_trait = normalised
        decision.warning = (
            ((decision.warning or "") + f" | AUTO_FIX: dominant_trait '{raw}' → '{normalised}'.")
        ).lstrip(" | ")


def _hard_enforce(decision: DecisionOutput, investor_state) -> DecisionOutput:
    from decision_guardrails import _parse_upper, _cap_allocation
    dominant = decision.state_context.dominant_trait
    if dominant not in ("panic", "cautious"):
        return decision
    upper = _parse_upper(decision.current_allocation)
    if upper is None or upper <= 25:
        return decision
    before = decision.current_allocation
    decision.current_allocation = _cap_allocation(before, 20)
    decision.warning = (
        ((decision.warning or "") +
         f" | HARD_ENFORCE: {dominant} with {before} → {decision.current_allocation}.")
    ).lstrip(" | ")
    return decision


def _calibrate_confidence(decision: DecisionOutput, retry_count: int, violations: list) -> DecisionOutput:
    blocking = [v for v in violations if getattr(v, "severity", "") == "blocking"]
    if retry_count >= 2 or blocking:
        if decision.confidence == "high":
            decision.confidence = "medium"
        elif decision.confidence == "medium":
            decision.confidence = "low"
        note = f"Confidence downgraded (retries={retry_count}, blocking={len(blocking)})."
        decision.warning = (((decision.warning or "") + " | " + note)).lstrip(" | ")
    return decision


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_decision(
    narrative,
    raw_text: str = "",
    investor_state=None,
    signals=None,
    profile_context=None,   # legacy — ignored
    axis_scores=None,        # legacy — ignored
    judgment=None,           # legacy — ignored
    compound_state=None,
) -> DecisionOutput:
    """
    Two-call decision engine optimised for 8B models.

    Flow:
      Call A: narrative + state → reasoning trace
      Call B: trace → allocation + strategy
      Validate → retry with correction if needed
      Hard enforce → confidence calibrate
    """
    from reasoning_validator import validate_reasoning_trace

    state        = investor_state or compound_state
    priority_str = resolve_priority(signals, state)
    retry_count  = 0
    last_decision: DecisionOutput | None = None
    last_violations: list = []
    correction_hint = ""

    # Call A — reasoning (done once, reused across retries)
    trace_raw = _call_reasoning(narrative, state, signals)
    if not trace_raw:
        return _fallback_decision(retry_count=0)

    for attempt in range(MAX_RETRIES):
        retry_count = attempt

        # Call B — decision from trace
        decision_raw = _call_decision(trace_raw, priority_str, correction_hint)
        if not decision_raw:
            break

        decision = _parse_decision(trace_raw, decision_raw)

        trace_result = validate_reasoning_trace(decision, investor_state=state, signals=signals)
        last_decision   = decision
        last_violations = trace_result.violations

        if trace_result.is_valid:
            break

        correction_hint = trace_result.correction_feedback
        if not correction_hint:
            break

    if last_decision is None:
        return _fallback_decision(retry_count=retry_count)

    decision = last_decision
    decision.retry_count = retry_count

    _auto_fix_dominant_trait(decision)
    decision = _hard_enforce(decision, state)
    decision = _calibrate_confidence(decision, retry_count, last_violations)

    return decision


# ---------------------------------------------------------------------------
# LLM consistency check (lightweight — runs after valid trace)
# ---------------------------------------------------------------------------

_VALIDATION_PROMPT = """Is this investment recommendation consistent with the investor profile?

Profile: {life_summary} | Risk truth: {risk_truth}
Recommendation: allocation={current_allocation}, dominant_trait={dominant_trait}, mode={allocation_mode}

Return JSON: {{"is_consistent": true/false, "inconsistency": "...", "correction_hint": "..."}}
"""


def validate_reasoning(narrative, decision: DecisionOutput) -> tuple[bool, str, str]:
    prompt = _VALIDATION_PROMPT.format(
        life_summary=getattr(narrative, "life_summary", "")[:200],
        risk_truth=getattr(narrative, "risk_truth", "")[:200],
        current_allocation=decision.current_allocation,
        dominant_trait=decision.state_context.dominant_trait,
        allocation_mode=decision.allocation_mode,
    )
    try:
        raw = _llm_call(prompt, num_predict=128)
        return (
            bool(raw.get("is_consistent", True)),
            raw.get("inconsistency", ""),
            raw.get("correction_hint", ""),
        )
    except (requests.RequestException, json.JSONDecodeError, ValueError):
        return True, "", ""


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def decision_to_dict(d: DecisionOutput) -> dict:
    return {
        "reasoning_trace": {
            "signals_considered": d.reasoning_trace.signals_considered,
            "dominant_factors":   d.reasoning_trace.dominant_factors,
            "secondary_factors":  d.reasoning_trace.secondary_factors,
            "contradictions": [
                {"signal_1": c.signal_1, "signal_2": c.signal_2,
                 "resolution": c.resolution, "dominant_trait": c.dominant_trait}
                for c in d.reasoning_trace.contradictions
            ],
            "state_inference": d.reasoning_trace.state_inference,
            "decision_logic":  d.reasoning_trace.decision_logic,
        },
        "current_allocation":  d.current_allocation,
        "baseline_allocation": d.baseline_allocation,
        "allocation_mode":     d.allocation_mode,
        "state_context": {
            "compound_state":   d.state_context.compound_state,
            "dominant_trait":   d.state_context.dominant_trait,
            "resilience_level": d.state_context.resilience_level,
            "state_stability":  d.state_context.state_stability,
        },
        "temporal_strategy": {
            "is_temporary":          d.temporal_strategy.is_temporary,
            "reassessment_trigger":  d.temporal_strategy.reassessment_trigger,
            "reassessment_timeline": d.temporal_strategy.reassessment_timeline,
            "expected_shift":        d.temporal_strategy.expected_shift,
        },
        "risk_assessment": {
            "identified_risks":     d.risk_assessment.identified_risks,
            "true_risk_capacity":   d.risk_assessment.true_risk_capacity,
            "stated_vs_actual_gap": d.risk_assessment.stated_vs_actual_gap,
            "reliability":          d.risk_assessment.reliability,
        },
        "strategy": {
            "primary_instrument": d.strategy_structured.primary_instrument,
            "equity_pct":         d.strategy_structured.equity_pct,
            "debt_pct":           d.strategy_structured.debt_pct,
            "sip_recommended":    d.strategy_structured.sip_recommended,
            "first_step":         d.strategy_structured.first_step,
        },
        "archetype":    d.archetype,
        "confidence":   d.confidence,
        "reasoning":    d.reasoning,
        "advisor_note": d.advisor_note,
        "warning":      d.warning,
        "retry_count":  d.retry_count,
        "fallback_used": d.fallback_used,
        "guardrail_adjustments": [
            {"rule": a.rule, "field": a.field, "before": a.before,
             "after": a.after, "reason": a.reason}
            for a in (d.guardrail_adjustments or [])
        ],
        # Legacy compat
        "equity_range":      d.current_allocation,
        "true_risk_profile": d.risk_assessment.true_risk_capacity,
    }
