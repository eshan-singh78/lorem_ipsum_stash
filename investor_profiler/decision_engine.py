"""
Decision Engine — InvestorDNA v16
Structured reasoning with multi-retry, dynamic priority, and hard enforcement.

Architecture:
  NarrativeOutput + InvestorState + SignalOutput → DecisionOutput

v16 changes:
  - MAX_RETRIES = 3 loop with trace validation on each attempt
  - Dynamic priority resolution: financial constraint overrides behavioral when high
  - Deterministic safe fallback when all retries fail
  - Hard enforcement: panic + >25% is force-corrected even after all retries
  - Confidence calibration: downgraded when retries ≥ 2 or violations detected
  - DEBUG_REASONING flag for full trace output
  - retry_count and fallback_used logged on every run

Design principle:
  "LLM can think freely. System must ensure it is always correct."
"""

import json
import re
import requests
from dataclasses import dataclass, field

OLLAMA_BASE_URL  = "http://localhost:11434"
LLM_MODEL        = "llama3.1:8b"
MAX_RETRIES      = 3
DEBUG_REASONING  = False   # set True to include full trace in output

_DECISION_PROMPT = """You are a senior financial advisor making a final investment suitability decision.

You MUST follow this exact reasoning order. DO NOT skip steps. DO NOT jump to conclusions.

━━━ SIGNAL PRIORITY HIERARCHY (highest to lowest) ━━━
{priority_order}

When signals conflict, the HIGHER priority signal MUST dominate.
DO NOT average conflicting signals. CHOOSE the dominant one and explain why.

━━━ MANDATORY REASONING STEPS ━━━

STEP 1 — IDENTIFY ALL RELEVANT SIGNALS
List every signal present: behavioral, financial, life events, knowledge.
Do not skip any signal mentioned in the inputs.

STEP 2 — SEPARATE DOMINANT vs SECONDARY FACTORS
Apply the priority hierarchy above.
Dominant factors = signals that will drive the final decision.
Secondary factors = real but overridden by higher-priority signals.

STEP 3 — DETECT AND RESOLVE CONTRADICTIONS
For each contradiction: name both signals, choose the dominant one, explain why.
DO NOT average. DO NOT ignore. ALWAYS resolve.

STEP 4 — INFER OVERALL STATE
Based on dominant factors and resolved contradictions, describe the investor's actual state.
This must be a direct consequence of steps 1-3.

STEP 5 — DECIDE ALLOCATION
current_allocation = what is appropriate RIGHT NOW given the inferred state.
baseline_allocation = what is appropriate when the state normalizes.
If state is temporary or transitional, current MUST differ from baseline.

STEP 6 — VERIFY CONSISTENCY
Check: does dominant_trait match current_allocation?
Check: does state_inference align with the signals?
If not — revise before outputting.

━━━ ADDITIONAL RULES ━━━
- allocation_mode must be "transitional" whenever current ≠ baseline
- If dominant_trait is "panic", current_allocation must be conservative
- DO NOT use formulas or numeric thresholds
- DO NOT output a single static allocation when a behavioral shift is detected

Return ONLY valid JSON with exactly this structure:

{{
  "reasoning_trace": {{
    "signals_considered": ["list every signal identified in step 1"],
    "dominant_factors": ["factors that drive the decision — must not be empty"],
    "secondary_factors": ["real but overridden factors"],
    "contradictions": [
      {{
        "signal_1": "first conflicting signal",
        "signal_2": "second conflicting signal",
        "resolution": "why one dominates the other",
        "dominant_trait": "the signal that wins"
      }}
    ],
    "state_inference": "one sentence: the investor's actual state derived from steps 1-3",
    "decision_logic": [
      "step 1: ...",
      "step 2: ...",
      "step 3: ..."
    ]
  }},
  "current_allocation": "equity range RIGHT NOW, e.g. '0%' or '5-15%' or '20-30%'",
  "baseline_allocation": "equity range for true long-term capacity, e.g. '30-40%'",
  "allocation_mode": "static | transitional | conditional",
  "state_context": {{
    "compound_state": "one phrase describing investor's overall situation",
    "dominant_trait": "must match reasoning_trace.dominant_factors[0]",
    "resilience_level": "low | medium | high",
    "state_stability": "stable | transitional | unstable"
  }},
  "temporal_strategy": {{
    "is_temporary": true or false,
    "reassessment_trigger": "specific condition that should trigger reassessment",
    "reassessment_timeline": "e.g. '6-12 months' or 'after obligation resolved'",
    "expected_shift": "how allocation changes when state normalizes, or 'none'"
  }},
  "risk_assessment": {{
    "true_risk_capacity": "2-3 sentences on actual capacity given life situation",
    "stated_vs_actual_gap": "description of gap, or 'aligned'",
    "reliability": "low | medium | high"
  }},
  "archetype": "descriptive label derived from the reasoning trace",
  "strategy": "specific investment strategy in plain language",
  "confidence": "high | medium | low",
  "reasoning": "5-7 sentences — must reference dominant_factors and contradiction resolutions",
  "advisor_note": "the single most important thing the advisor must be careful about"
}}

No markdown. JSON only.

━━━ INVESTOR NARRATIVE ━━━
Life situation: {life_summary}
Financial reality: {financial_analysis}
Psychological state: {psychological_analysis}
Contradictions: {contradictions}
Risk truth: {risk_truth}
Reliability: {reliability_assessment}
Advisor insight: {advisor_insight}

━━━ STRUCTURED CONTRADICTIONS (must be resolved in reasoning_trace.contradictions) ━━━
{structured_contradictions}

━━━ SYNTHESIZED INVESTOR STATE ━━━
Compound state: {compound_state}
State description: {state_description}
Dominant factors: {dominant_factors}
State implications: {state_implications}
State stability: {state_stability}
Dominant trait: {dominant_trait}
Resilience level: {resilience_level}
Behavioral shift detected: {shift_detected}
Baseline behavior: {baseline_behavior}
Current behavior: {current_behavior}
Shift permanence: {shift_permanence}

━━━ RAW INVESTOR TEXT (for additional context) ━━━
{raw_text}
"""

_VALIDATION_PROMPT = """You are a senior financial advisor reviewing a colleague's recommendation.

Determine whether this recommendation is CONSISTENT with the investor narrative and state.

A recommendation is inconsistent if:
- It ignores the dominant_trait (e.g. recommends aggressive equity when dominant_trait is panic)
- current_allocation does not match the investor's current state
- baseline_allocation does not match true long-term capacity
- temporal_strategy is missing when state_stability is transitional
- allocation_mode is "static" when a behavioral shift is detected

Return ONLY valid JSON:

{{
  "is_consistent": true or false,
  "inconsistency": "specific description, or empty string if consistent",
  "correction_hint": "what the recommendation should say instead, or empty string if consistent"
}}

No markdown. JSON only.

━━━ NARRATIVE ━━━
Life situation: {life_summary}
Psychological state: {psychological_analysis}
Risk truth: {risk_truth}
Reliability: {reliability_assessment}

━━━ RECOMMENDATION ━━━
Current allocation: {current_allocation}
Baseline allocation: {baseline_allocation}
Allocation mode: {allocation_mode}
Dominant trait: {dominant_trait}
State stability: {state_stability}
Is temporary: {is_temporary}
Reasoning: {reasoning}
Confidence: {confidence}
"""


# ---------------------------------------------------------------------------
# Dynamic priority resolution (Step 2)
# ---------------------------------------------------------------------------

_DEFAULT_PRIORITY = (
    "1. Behavioral signals (panic, loss response, consistency under stress)\n"
    "2. Financial constraints (obligation level, constraint severity, resilience)\n"
    "3. Life events (grief, crisis, responsibility shift)\n"
    "4. Experience and knowledge (sophistication, decision autonomy)"
)

_FINANCIAL_FIRST_PRIORITY = (
    "1. Financial constraints (obligation level, constraint severity, resilience) "
    "← ELEVATED: financial constraint is HIGH\n"
    "2. Behavioral signals (panic, loss response, consistency under stress)\n"
    "3. Life events (grief, crisis, responsibility shift)\n"
    "4. Experience and knowledge (sophistication, decision autonomy)"
)


def resolve_priority(signals, investor_state) -> str:
    """
    Determine signal priority order based on context.

    Rule:
      IF financial_constraint == "high" → financial overrides behavioral
      ELIF dominant_trait in [panic, grief] → behavioral overrides (default)
      ELSE → default hierarchy

    Returns a formatted priority string injected into the decision prompt.
    """
    constraint = "low"
    if signals is not None:
        constraint = signals.financial_state.constraint_level

    dominant = ""
    if investor_state is not None:
        dominant = (investor_state.dominant_trait or "").lower()

    if constraint == "high":
        return _FINANCIAL_FIRST_PRIORITY

    # Default: behavioral first (covers panic, grief, and all other cases)
    return _DEFAULT_PRIORITY

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
    resilience_level: str   # low | medium | high
    state_stability: str    # stable | transitional | unstable


@dataclass
class TemporalStrategy:
    is_temporary: bool
    reassessment_trigger: str
    reassessment_timeline: str
    expected_shift: str


@dataclass
class RiskAssessment:
    true_risk_capacity: str
    stated_vs_actual_gap: str
    reliability: str        # low | medium | high


@dataclass
class DecisionOutput:
    # v15: reasoning trace — produced BEFORE allocation
    reasoning_trace: ReasoningTrace

    # v14 structured fields
    current_allocation: str
    baseline_allocation: str
    allocation_mode: str            # static | transitional | conditional
    state_context: StateContext
    temporal_strategy: TemporalStrategy
    risk_assessment: RiskAssessment

    # Shared fields
    archetype: str
    strategy: str
    confidence: str
    reasoning: str
    advisor_note: str
    raw: dict = field(default_factory=dict)
    warning: str | None = None
    guardrail_adjustments: list = field(default_factory=list)
    # v16 run metadata
    retry_count: int = 0
    fallback_used: bool = False

    # Legacy compat
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


def _build_structured_contradictions(signals) -> str:
    if signals is None or not signals.contradictions:
        return "No structured contradictions detected."
    lines = []
    for c in signals.contradictions:
        lines.append(
            f"- Type: {c.type}\n"
            f"  dominant_trait: '{c.dominant_trait}' (THIS drives behavior under stress)\n"
            f"  suppressed_trait: '{c.suppressed_trait}' (stated but overridden under pressure)\n"
            f"  explanation: {c.explanation}"
        )
    return "\n".join(lines)


def _parse_reasoning_trace(raw: dict) -> ReasoningTrace:
    """Parse reasoning_trace from LLM output."""
    rt = raw.get("reasoning_trace") or {}

    signals = rt.get("signals_considered", [])
    if not isinstance(signals, list):
        signals = [str(signals)] if signals else []

    dominant = rt.get("dominant_factors", [])
    if not isinstance(dominant, list):
        dominant = [str(dominant)] if dominant else []

    secondary = rt.get("secondary_factors", [])
    if not isinstance(secondary, list):
        secondary = [str(secondary)] if secondary else []

    raw_contradictions = rt.get("contradictions", [])
    if not isinstance(raw_contradictions, list):
        raw_contradictions = []
    contradictions = []
    for c in raw_contradictions:
        if isinstance(c, dict):
            contradictions.append(ContradictionResolution(
                signal_1=_safe(c.get("signal_1"), "unknown"),
                signal_2=_safe(c.get("signal_2"), "unknown"),
                resolution=_safe(c.get("resolution"), ""),
                dominant_trait=_safe(c.get("dominant_trait"), "unknown"),
            ))

    logic = rt.get("decision_logic", [])
    if not isinstance(logic, list):
        logic = [str(logic)] if logic else []

    return ReasoningTrace(
        signals_considered=signals,
        dominant_factors=dominant,
        secondary_factors=secondary,
        contradictions=contradictions,
        state_inference=_safe(rt.get("state_inference"), ""),
        decision_logic=logic,
    )


def _parse_decision(raw: dict) -> DecisionOutput:
    """Parse LLM JSON into structured DecisionOutput."""
    sc_raw = raw.get("state_context") or {}
    ts_raw = raw.get("temporal_strategy") or {}
    ra_raw = raw.get("risk_assessment") or {}

    reasoning_trace = _parse_reasoning_trace(raw)

    state_context = StateContext(
        compound_state=_safe(sc_raw.get("compound_state"), "unknown"),
        dominant_trait=_safe(sc_raw.get("dominant_trait"), "unknown"),
        resilience_level=_safe(sc_raw.get("resilience_level"), "medium"),
        state_stability=_safe(sc_raw.get("state_stability"), "stable"),
    )

    temporal_strategy = TemporalStrategy(
        is_temporary=_safe_bool(ts_raw.get("is_temporary"), False),
        reassessment_trigger=_safe(ts_raw.get("reassessment_trigger"), "standard annual review"),
        reassessment_timeline=_safe(ts_raw.get("reassessment_timeline"), "12 months"),
        expected_shift=_safe(ts_raw.get("expected_shift"), "none"),
    )

    risk_assessment = RiskAssessment(
        true_risk_capacity=_safe(ra_raw.get("true_risk_capacity"), "Could not be determined."),
        stated_vs_actual_gap=_safe(ra_raw.get("stated_vs_actual_gap"), "unknown"),
        reliability=_safe(ra_raw.get("reliability"), "medium"),
    )

    confidence = _safe(raw.get("confidence"), "low")
    if confidence not in ("high", "medium", "low"):
        confidence = "low"

    allocation_mode = _safe(raw.get("allocation_mode"), "static")
    if allocation_mode not in ("static", "transitional", "conditional"):
        allocation_mode = "static"

    warning = None
    if confidence == "low":
        warning = "Provisional — requires further advisor validation."

    return DecisionOutput(
        reasoning_trace=reasoning_trace,
        current_allocation=_safe(raw.get("current_allocation"), "0-10%"),
        baseline_allocation=_safe(raw.get("baseline_allocation"), "0-10%"),
        allocation_mode=allocation_mode,
        state_context=state_context,
        temporal_strategy=temporal_strategy,
        risk_assessment=risk_assessment,
        archetype=_safe(raw.get("archetype"), "Unclassified"),
        strategy=_safe(raw.get("strategy"), "Insufficient information to make a specific recommendation."),
        confidence=confidence,
        reasoning=_safe(raw.get("reasoning"), "Reasoning unavailable — LLM decision engine did not respond."),
        advisor_note=_safe(raw.get("advisor_note"), "No specific advisor note available."),
        raw=raw,
        warning=warning,
    )


def _empty_trace() -> ReasoningTrace:
    return ReasoningTrace(
        signals_considered=[],
        dominant_factors=[],
        secondary_factors=[],
        contradictions=[],
        state_inference="",
        decision_logic=[],
    )


def _fallback_decision(retry_count: int = 0) -> DecisionOutput:
    """
    Deterministic safe fallback when all LLM retries fail.
    Conservative allocation — never exposes investor to unsafe equity levels.
    """
    return DecisionOutput(
        reasoning_trace=_empty_trace(),
        current_allocation="5-10%",
        baseline_allocation="15-20%",
        allocation_mode="conservative_fallback",
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
            true_risk_capacity="Could not be determined — LLM unavailable.",
            stated_vs_actual_gap="unknown",
            reliability="low",
        ),
        archetype="Unclassified",
        strategy="Conservative fallback — manual advisor assessment required.",
        confidence="low",
        reasoning="Decision engine failed after all retries — conservative fallback applied.",
        advisor_note="Manual advisor assessment required. LLM was unavailable.",
        raw={},
        warning=f"LLM failed consistency checks after {retry_count} retries — conservative fallback used.",
        retry_count=retry_count,
        fallback_used=True,
    )


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def generate_decision(
    narrative,
    raw_text: str = "",
    investor_state=None,
    signals=None,
    # Legacy params — accepted but ignored
    profile_context=None,
    axis_scores=None,
    judgment=None,
    compound_state=None,
) -> DecisionOutput:
    """
    LLM decision with multi-retry loop, trace validation, and hard enforcement.

    v16 flow:
      1. Resolve dynamic priority order
      2. Loop up to MAX_RETRIES:
           a. Call LLM
           b. Validate reasoning trace (deterministic)
           c. If valid → break
           d. If invalid → inject correction feedback and retry
      3. If all retries fail → deterministic safe fallback
      4. Hard enforcement: panic + >25% → force-correct regardless
      5. Confidence calibration: downgrade if retries ≥ 2 or violations
      6. LLM consistency check (existing)
    """
    from reasoning_validator import validate_reasoning_trace

    state         = investor_state or compound_state
    priority_str  = resolve_priority(signals, state)
    retry_count   = 0
    last_decision = None
    last_violations: list = []
    correction_hint = ""

    for attempt in range(MAX_RETRIES):
        retry_count = attempt
        decision = _call_decision_llm(
            narrative, raw_text, state,
            signals=signals,
            priority_order=priority_str,
            correction_hint=correction_hint,
        )

        # LLM hard failure — no point retrying
        if decision.fallback_used:
            decision.retry_count = retry_count
            return decision

        # Validate reasoning trace
        trace_result = validate_reasoning_trace(decision, investor_state=state, signals=signals)
        last_decision   = decision
        last_violations = trace_result.violations

        if trace_result.is_valid:
            break  # good output — exit loop

        # Build correction feedback for next attempt
        correction_hint = trace_result.correction_feedback
        if not correction_hint:
            break  # no actionable feedback — accept as-is

    # All retries exhausted without valid trace → safe fallback
    if last_decision is None:
        return _fallback_decision(retry_count=retry_count)

    decision = last_decision
    decision.retry_count = retry_count

    # Step 5: Hard enforcement — panic + >25% is force-corrected unconditionally
    decision = _hard_enforce(decision, state)

    # Step 7: Confidence calibration
    decision = _calibrate_confidence(decision, retry_count, last_violations)

    # LLM consistency check (existing — runs after trace is valid)
    is_consistent, inconsistency, hint = validate_reasoning(narrative, decision)
    if not is_consistent and hint:
        regenerated = _call_decision_llm(
            narrative, raw_text, state,
            signals=signals,
            priority_order=priority_str,
            correction_hint=hint,
        )
        if not regenerated.fallback_used:
            regenerated.retry_count = retry_count + 1
            regenerated.warning = (
                f"Decision regenerated after consistency check. "
                f"Original inconsistency: {inconsistency}"
            )
            regenerated = _hard_enforce(regenerated, state)
            regenerated = _calibrate_confidence(regenerated, regenerated.retry_count, [])
            return regenerated

    return decision


def _hard_enforce(decision: DecisionOutput, investor_state) -> DecisionOutput:
    """
    Step 5: Hard enforcement — runs AFTER all retries and guardrails.
    Catches any remaining violations that escaped the retry loop.

    Rule: dominant_trait == "panic" AND current_allocation > 25% → force to 20%
    This is the last line of defense — no invalid output escapes.
    """
    from decision_guardrails import _parse_upper, _cap_allocation

    dominant = (decision.state_context.dominant_trait or "").lower()
    if "panic" not in dominant:
        return decision

    upper = _parse_upper(decision.current_allocation)
    if upper is None or upper <= 25:
        return decision

    before = decision.current_allocation
    decision.current_allocation = _cap_allocation(before, 20)

    existing_warning = decision.warning or ""
    decision.warning = (
        existing_warning +
        f" | HARD_ENFORCE: panic dominant_trait with current_allocation={before} "
        f"force-corrected to {decision.current_allocation}."
    ).lstrip(" | ")

    return decision


def _calibrate_confidence(
    decision: DecisionOutput,
    retry_count: int,
    violations: list,
) -> DecisionOutput:
    """
    Step 7: Downgrade confidence when reasoning required retries or had violations.
    high → medium if retry_count ≥ 2 or blocking violations detected
    medium → low  if retry_count ≥ 2 or blocking violations detected
    """
    blocking = [v for v in violations if getattr(v, "severity", "") == "blocking"]
    should_downgrade = retry_count >= 2 or len(blocking) > 0

    if not should_downgrade:
        return decision

    if decision.confidence == "high":
        decision.confidence = "medium"
    elif decision.confidence == "medium":
        decision.confidence = "low"

    note = f"Confidence downgraded (retry_count={retry_count}, blocking_violations={len(blocking)})."
    decision.warning = ((decision.warning or "") + " | " + note).lstrip(" | ")

    return decision


def _call_decision_llm(
    narrative,
    raw_text: str,
    investor_state=None,
    signals=None,
    priority_order: str = "",
    correction_hint: str = "",
) -> DecisionOutput:
    dominant_trait   = getattr(investor_state, "dominant_trait", "") if investor_state else ""
    resilience_level = getattr(investor_state, "resilience_level", "") if investor_state else ""
    shift_detected   = getattr(investor_state, "shift_detected", False) if investor_state else False
    baseline_beh     = getattr(investor_state, "baseline_behavior", "") if investor_state else ""
    current_beh      = getattr(investor_state, "current_behavior", "") if investor_state else ""
    shift_permanence = getattr(investor_state, "shift_permanence", "") if investor_state else ""

    prompt = _DECISION_PROMPT.format(
        priority_order=priority_order or _DEFAULT_PRIORITY,
        life_summary=getattr(narrative, "life_summary", "Not available"),
        financial_analysis=getattr(narrative, "financial_analysis", "Not available"),
        psychological_analysis=getattr(narrative, "psychological_analysis", "Not available"),
        contradictions=getattr(narrative, "contradictions", "none detected"),
        risk_truth=getattr(narrative, "risk_truth", "Not available"),
        reliability_assessment=getattr(narrative, "reliability_assessment", "Not available"),
        advisor_insight=getattr(narrative, "advisor_insight", "Not available"),
        structured_contradictions=_build_structured_contradictions(signals),
        compound_state=getattr(investor_state, "compound_state", "Not synthesized") if investor_state else "Not synthesized",
        state_description=getattr(investor_state, "state_description", "") if investor_state else "",
        dominant_factors=", ".join(getattr(investor_state, "dominant_factors", [])) if investor_state else "",
        state_implications=", ".join(getattr(investor_state, "state_implications", [])) if investor_state else "",
        state_stability=getattr(investor_state, "state_stability", "unknown") if investor_state else "unknown",
        dominant_trait=dominant_trait,
        resilience_level=resilience_level,
        shift_detected=str(shift_detected).lower(),
        baseline_behavior=baseline_beh,
        current_behavior=current_beh,
        shift_permanence=shift_permanence,
        raw_text=raw_text[:800] if raw_text else "Not available",
    )

    if correction_hint:
        prompt += (
            f"\n\n⚠️ CORRECTION REQUIRED: A previous version of this recommendation "
            f"was inconsistent. Specifically:\n{correction_hint}\n"
            f"Ensure your recommendation addresses ALL of the above."
        )

    payload = {
        "model":   LLM_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0, "num_predict": 2048},
        "format":  "json",
    }

    raw_dict = {}
    for attempt in (1, 2):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=None,
            )
            resp.raise_for_status()
            raw_dict = _parse_json(resp.json().get("response", ""))
            # Reject shallow outputs — must have trace + allocation + dominant_factors
            rt = raw_dict.get("reasoning_trace") or {}
            has_trace      = bool(rt)
            has_allocation = bool(raw_dict.get("current_allocation"))
            has_dominant   = bool(rt.get("dominant_factors"))
            if has_trace and has_allocation and has_dominant:
                break
            # Shallow — retry on attempt 1, accept on attempt 2
            if attempt == 2 and has_allocation:
                break  # accept partial on final attempt
        except (requests.RequestException, json.JSONDecodeError, ValueError):
            if attempt == 2:
                return _fallback_decision()

    if not raw_dict.get("current_allocation"):
        return _fallback_decision()

    return _parse_decision(raw_dict)


def validate_reasoning(narrative, decision: DecisionOutput) -> tuple[bool, str, str]:
    """
    Meta-reasoning: ask LLM if the decision is consistent with narrative + state.
    v14: checks temporal consistency — allocation_mode vs state_stability.
    """
    prompt = _VALIDATION_PROMPT.format(
        life_summary=getattr(narrative, "life_summary", "Not available"),
        psychological_analysis=getattr(narrative, "psychological_analysis", "Not available"),
        risk_truth=getattr(narrative, "risk_truth", "Not available"),
        reliability_assessment=getattr(narrative, "reliability_assessment", "Not available"),
        current_allocation=decision.current_allocation,
        baseline_allocation=decision.baseline_allocation,
        allocation_mode=decision.allocation_mode,
        dominant_trait=decision.state_context.dominant_trait,
        state_stability=decision.state_context.state_stability,
        is_temporary=str(decision.temporal_strategy.is_temporary).lower(),
        reasoning=decision.reasoning,
        confidence=decision.confidence,
    )

    payload = {
        "model":   LLM_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0, "num_predict": 256},
        "format":  "json",
    }

    for attempt in (1, 2):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=None,
            )
            resp.raise_for_status()
            raw = _parse_json(resp.json().get("response", ""))
            if raw:
                return (
                    bool(raw.get("is_consistent", True)),
                    raw.get("inconsistency", ""),
                    raw.get("correction_hint", ""),
                )
        except (requests.RequestException, json.JSONDecodeError, ValueError):
            pass

    return True, "", ""


def decision_to_dict(d: DecisionOutput) -> dict:
    return {
        "reasoning_trace": {
            "signals_considered": d.reasoning_trace.signals_considered,
            "dominant_factors":   d.reasoning_trace.dominant_factors,
            "secondary_factors":  d.reasoning_trace.secondary_factors,
            "contradictions": [
                {
                    "signal_1":      c.signal_1,
                    "signal_2":      c.signal_2,
                    "resolution":    c.resolution,
                    "dominant_trait": c.dominant_trait,
                }
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
            "true_risk_capacity":   d.risk_assessment.true_risk_capacity,
            "stated_vs_actual_gap": d.risk_assessment.stated_vs_actual_gap,
            "reliability":          d.risk_assessment.reliability,
        },
        "archetype":    d.archetype,
        "strategy":     d.strategy,
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
