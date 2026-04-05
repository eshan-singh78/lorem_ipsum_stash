"""
State Synthesis — InvestorDNA v14
Converts signals + narrative into a fully structured investor state.

Architecture:
  raw_text + NarrativeOutput + ProfileContext + SignalOutput → InvestorState

v14 changes:
  - InvestorState now carries ALL structured factors explicitly:
      dominant_trait, suppressed_traits, resilience_level, resilience_evidence,
      shift_detected, baseline_behavior, current_behavior, shift_permanence
  - Signals are mapped directly into state fields — no compression
  - contradictions → dominant_trait (not just compound_state string)
  - temporal_context → shift_detected, baseline_behavior, current_behavior, shift_permanence
  - LLM synthesizes the INTERACTION; structured fields come from signals directly

Design principle:
  "The state is not a label. It is a structured record of who this investor
   is, what changed, what drives them, and how resilient they are."
"""

import json
import re
import requests
from dataclasses import dataclass, field

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL       = "llama3.1:8b"

_SYNTHESIS_PROMPT = """You are a senior financial advisor.

Given this investor profile, your task is to UNDERSTAND the overall situation.

Do NOT list signals. Do NOT classify into categories. Do NOT enumerate flags.

Instead:
1. Describe the investor's situation in ONE concise phrase (the compound state)
2. Explain what is actually happening in their life right now (2-3 sentences)
3. Identify the 2-3 dominant factors that are most affecting their financial decisions
4. Explain what this state implies for financial behavior and risk capacity
5. Assess how stable or temporary this state is

IMPORTANT:
- Think holistically — how do the signals INTERACT, not just what they are individually
- If signals contradict each other, explain which one dominates and why
- The compound state phrase should be specific enough to guide an advisor
- Dominant factors should explain WHY the investor behaves as they do
- State implications should tell the advisor what to expect from this investor

Return ONLY valid JSON:

{{
  "compound_state": "one concise phrase describing the investor's overall situation",
  "state_description": "2-3 sentences explaining what is actually happening in their life",
  "dominant_factors": ["factor 1", "factor 2", "factor 3 (optional)"],
  "state_implications": [
    "implication for financial behavior",
    "implication for risk capacity",
    "implication for advisor approach"
  ],
  "state_stability": "temporary | transitional | stable",
  "confidence": "high | medium | low"
}}

No markdown. JSON only.

━━━ INVESTOR NARRATIVE ━━━
Life situation: {life_summary}
Financial reality: {financial_analysis}
Psychological state: {psychological_analysis}
Contradictions: {contradictions}
Risk truth: {risk_truth}
Reliability: {reliability_assessment}

━━━ EXTRACTED SIGNALS ━━━
{signals}
"""


@dataclass
class InvestorState:
    # LLM-synthesized fields
    compound_state: str
    state_description: str
    dominant_factors: list[str]
    state_implications: list[str]
    state_stability: str            # temporary | transitional | stable
    confidence: str                 # high | medium | low

    # v14: structured fields mapped directly from signals — never compressed
    dominant_trait: str             # from contradictions[0].dominant_trait or behavior.loss_response
    suppressed_traits: list[str]    # from contradictions[*].suppressed_trait
    resilience_level: str           # from signals.behavior.resilience_level
    resilience_evidence: str        # from signals.behavior.resilience_evidence
    shift_detected: bool            # from signals.temporal_context.has_behavioral_shift
    baseline_behavior: str          # from signals.temporal_context.baseline_orientation
    current_behavior: str           # from signals.temporal_context.current_orientation
    shift_permanence: str           # from signals.temporal_context.shift_permanence or "unknown"

    raw: dict = field(default_factory=dict)
    warning: str | None = None

    # ---------------------------------------------------------------------------
    # Legacy compat properties — used by modules that still read these
    # ---------------------------------------------------------------------------
    @property
    def severity(self) -> str:
        s = self.compound_state.lower()
        if any(w in s for w in ("crisis", "grief", "survival", "collapsed")):
            return "critical" if "crisis" in s else "high"
        if any(w in s for w in ("constrained", "pressure", "stressed", "transitional")):
            return "moderate"
        return "low"

    @property
    def stability_first(self) -> bool:
        return self.severity in ("critical", "high")

    @property
    def reason(self) -> str:
        return self.state_description

    @property
    def state_divergence(self) -> bool:
        return self.shift_detected or self.state_stability in ("temporary", "transitional")

    @property
    def divergence_reason(self) -> str:
        return self.state_description

    @property
    def is_transition(self) -> bool:
        return self.state_divergence

    @property
    def baseline_risk(self) -> str:
        return "unknown"

    @property
    def current_risk(self) -> str:
        return "unknown"


# ---------------------------------------------------------------------------
# Signal summary builder — reads from SignalOutput
# ---------------------------------------------------------------------------

def _build_signals_from_signal_output(signals) -> str:
    """Build signal summary from SignalOutput (v14 primary path)."""
    if signals is None:
        return "No structured signals available."
    lines = []
    for ev in signals.life_events:
        lines.append(f"Life event: {ev.type} ({ev.recency}, impact={ev.impact}) — {ev.description}")
    resp = signals.responsibility
    lines.append(
        f"Responsibility: {resp.role}, dependents={resp.dependents_count}, "
        f"pressure={resp.financial_pressure}"
    )
    if resp.dependents_description:
        lines.append(f"  Dependents: {resp.dependents_description}")
    for ob in resp.cultural_obligations:
        lines.append(f"  Cultural obligation: {ob}")
    beh = signals.behavior
    lines.append(
        f"Behavior: loss_response={beh.loss_response}, resilience={beh.resilience_level}, "
        f"consistency={beh.consistency}"
    )
    if beh.loss_response_detail:
        lines.append(f"  Loss detail: {beh.loss_response_detail}")
    if beh.resilience_evidence:
        lines.append(f"  Resilience evidence: {beh.resilience_evidence}")
    ds = signals.decision_style
    lines.append(
        f"Decision style: autonomy={ds.autonomy}, peer_influence={ds.peer_influence}, "
        f"analytical={ds.analytical_tendency}"
    )
    fs = signals.financial_state
    lines.append(
        f"Financial state: constraint={fs.constraint_level}, "
        f"obligation_scale={fs.obligation_scale}"
    )
    if fs.hidden_obligations:
        lines.append(f"  Hidden obligations: {fs.hidden_obligations}")
    for c in signals.contradictions:
        lines.append(
            f"Contradiction: dominant='{c.dominant_trait}', "
            f"suppressed='{c.suppressed_trait}' — {c.explanation}"
        )
    tc = signals.temporal_context
    if tc.has_behavioral_shift:
        lines.append(
            f"Behavioral shift: {tc.baseline_orientation} → {tc.current_orientation}"
        )
        if tc.shift_cause:
            lines.append(
                f"  Cause: {tc.shift_cause} "
                f"(recency={tc.shift_recency}, permanence={tc.shift_permanence})"
            )
    for intent in signals.intent:
        lines.append(
            f"Intent: {intent.goal} ({intent.category}, timeline={intent.timeline}, "
            f"firmness={intent.firmness})"
        )
    return "\n".join(lines)


def _build_signals_from_profile_context(profile_context) -> str:
    """Fallback: build signal summary from ProfileContext (when SignalOutput unavailable)."""
    lines = []
    if profile_context.monthly_income:
        lines.append(f"Monthly income: ₹{profile_context.monthly_income:,.0f}")
    if profile_context.emi_ratio is not None:
        lines.append(f"EMI ratio: {profile_context.emi_ratio:.1f}% of income")
    if profile_context.emergency_months is not None:
        lines.append(f"Emergency fund: {profile_context.emergency_months:.1f} months")
    if profile_context.dependents is not None:
        lines.append(f"Dependents: {profile_context.dependents}")
    if profile_context.experience_years is not None:
        lines.append(f"Investment experience: {profile_context.experience_years:.1f} years")
    # v14: prefer signals.behavior.loss_response; fall back to ctx.loss_reaction
    if profile_context.loss_reaction:
        lines.append(f"Loss reaction: {profile_context.loss_reaction}")
    if profile_context.risk_behavior:
        lines.append(f"Stated risk behavior: {profile_context.risk_behavior}")
    if profile_context.near_term_obligation_level not in (None, "none"):
        otype = f" ({profile_context.obligation_type})" if profile_context.obligation_type else ""
        lines.append(f"Near-term obligation: {profile_context.near_term_obligation_level}{otype}")
    for ev in profile_context.life_events:
        lines.append(f"Life event: {ev.event_type} ({ev.recency}, weight: {ev.emotional_weight})")
    for sig in profile_context.cultural_signals:
        lines.append(f"Cultural: {sig.signal_type} — {sig.description}")
    for sig in profile_context.behavioral_signals:
        lines.append(f"Behavioral: {sig.signal_type} ({sig.strength})")
    # v14: resilience from ProfileContext
    if hasattr(profile_context, "resilience_level") and profile_context.resilience_level:
        lines.append(f"Resilience: {profile_context.resilience_level}")
    if hasattr(profile_context, "resilience_evidence") and profile_context.resilience_evidence:
        lines.append(f"Resilience evidence: {profile_context.resilience_evidence}")
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


def _extract_structured_from_signals(signals) -> dict:
    """
    Map SignalOutput fields directly into InvestorState structured fields.
    This is the v14 key change: structured fields come from signals, not LLM inference.
    """
    if signals is None:
        return {
            "dominant_trait": "unknown",
            "suppressed_traits": [],
            "resilience_level": "medium",
            "resilience_evidence": "",
            "shift_detected": False,
            "baseline_behavior": "",
            "current_behavior": "",
            "shift_permanence": "unknown",
        }

    # dominant_trait: from first contradiction's dominant_trait, else from loss_response
    dominant_trait = "unknown"
    suppressed_traits = []
    if signals.contradictions:
        dominant_trait = signals.contradictions[0].dominant_trait
        suppressed_traits = [c.suppressed_trait for c in signals.contradictions]
    elif signals.behavior.loss_response not in ("neutral", ""):
        dominant_trait = signals.behavior.loss_response

    tc = signals.temporal_context
    shift_permanence = tc.shift_permanence or "unknown"
    if shift_permanence not in ("likely_permanent", "likely_temporary", "unknown"):
        shift_permanence = "unknown"

    return {
        "dominant_trait":     dominant_trait,
        "suppressed_traits":  suppressed_traits,
        "resilience_level":   signals.behavior.resilience_level,
        "resilience_evidence": signals.behavior.resilience_evidence,
        "shift_detected":     tc.has_behavioral_shift,
        "baseline_behavior":  tc.baseline_orientation or "",
        "current_behavior":   tc.current_orientation or "",
        "shift_permanence":   shift_permanence,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def synthesize_state(
    raw_text: str,
    narrative,
    profile_context,
    signals=None,
) -> InvestorState:
    """
    Synthesize all signals into a fully structured investor state.

    v14: structured fields (dominant_trait, resilience, shift) are mapped
    directly from SignalOutput — not inferred by the LLM.
    The LLM synthesizes the compound_state and implications only.
    """
    # Build signal summary for LLM
    if signals is not None:
        signal_summary = _build_signals_from_signal_output(signals)
    else:
        signal_summary = _build_signals_from_profile_context(profile_context)

    # Extract structured fields directly from signals (no LLM needed for these)
    structured = _extract_structured_from_signals(signals)

    prompt = _SYNTHESIS_PROMPT.format(
        life_summary=getattr(narrative, "life_summary", "Not available"),
        financial_analysis=getattr(narrative, "financial_analysis", "Not available"),
        psychological_analysis=getattr(narrative, "psychological_analysis", "Not available"),
        contradictions=getattr(narrative, "contradictions", "none detected"),
        risk_truth=getattr(narrative, "risk_truth", "Not available"),
        reliability_assessment=getattr(narrative, "reliability_assessment", "Not available"),
        signals=signal_summary,
    )

    payload = {
        "model":   LLM_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0, "num_predict": 768},
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
            raw_dict = _parse_json(resp.json().get("response", ""))
            if raw_dict.get("compound_state"):
                break
        except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
            if attempt == 2:
                warning = f"State synthesis failed: {e}"
                raw_dict = {}

    if not raw_dict.get("compound_state"):
        return _fallback_state(profile_context, narrative, warning, structured)

    dominant = raw_dict.get("dominant_factors", [])
    if not isinstance(dominant, list):
        dominant = [str(dominant)]

    implications = raw_dict.get("state_implications", [])
    if not isinstance(implications, list):
        implications = [str(implications)]

    stability = raw_dict.get("state_stability", "stable")
    if stability not in ("temporary", "transitional", "stable"):
        stability = "stable"

    confidence = raw_dict.get("confidence", "medium")
    if confidence not in ("high", "medium", "low"):
        confidence = "medium"

    return InvestorState(
        compound_state=raw_dict["compound_state"].strip(),
        state_description=raw_dict.get("state_description", "").strip(),
        dominant_factors=dominant,
        state_implications=implications,
        state_stability=stability,
        confidence=confidence,
        # Structured fields from signals — not from LLM
        dominant_trait=structured["dominant_trait"],
        suppressed_traits=structured["suppressed_traits"],
        resilience_level=structured["resilience_level"],
        resilience_evidence=structured["resilience_evidence"],
        shift_detected=structured["shift_detected"],
        baseline_behavior=structured["baseline_behavior"],
        current_behavior=structured["current_behavior"],
        shift_permanence=structured["shift_permanence"],
        raw=raw_dict,
        warning=warning,
    )


def _fallback_state(
    profile_context,
    narrative,
    warning: str | None,
    structured: dict,
) -> InvestorState:
    grief    = profile_context.grief_state
    peer     = profile_context.peer_driven
    recency  = profile_context.recency_bias_risk
    emerging = profile_context.emerging_constraint
    deps     = profile_context.dependents or 0

    has_provider = any(s.signal_type == "family_role" for s in profile_context.cultural_signals)
    has_crisis   = any(ev.event_type == "crisis" for ev in profile_context.life_events)

    if (grief or has_crisis) and (deps > 0 or has_provider):
        state = "grief-burdened primary provider"
        desc  = "Investor is navigating grief while carrying primary financial responsibility."
        factors = ["grief suppressing risk tolerance", "primary financial responsibility", "constrained investable surplus"]
        implications = ["avoid investment product discussion", "focus on stability and insurance", "reassess in 6-12 months"]
        stability = "temporary"
    elif grief:
        state = "grief-suppressed investor"
        desc  = "Grief is temporarily suppressing natural risk tolerance."
        factors = ["grief", "emotional distortion of preferences"]
        implications = ["current preferences unreliable", "conservative approach required"]
        stability = "temporary"
    elif peer and recency:
        state = "peer-driven speculator"
        desc  = "Investment decisions driven by peer influence and recency bias."
        factors = ["peer influence", "recency bias", "inflated risk appetite"]
        implications = ["stated risk preference unreliable", "education before product complexity"]
        stability = "transitional"
    elif emerging:
        state = "constrained investor with emerging obligations"
        desc  = "Current obligations are manageable but future commitments will constrain capacity."
        factors = ["emerging financial obligations", "limited future investable surplus"]
        implications = ["avoid illiquid products", "plan for upcoming obligations"]
        stability = "transitional"
    else:
        state = "stable moderate-capacity investor"
        desc  = "No critical compound state detected — investor appears stable."
        factors = ["stable income", "moderate obligations"]
        implications = ["standard profiling applies"]
        stability = "stable"

    return InvestorState(
        compound_state=state,
        state_description=desc,
        dominant_factors=factors,
        state_implications=implications,
        state_stability=stability,
        confidence="medium",
        dominant_trait=structured["dominant_trait"],
        suppressed_traits=structured["suppressed_traits"],
        resilience_level=structured["resilience_level"],
        resilience_evidence=structured["resilience_evidence"],
        shift_detected=structured["shift_detected"],
        baseline_behavior=structured["baseline_behavior"],
        current_behavior=structured["current_behavior"],
        shift_permanence=structured["shift_permanence"],
        raw={},
        warning=warning,
    )


def state_to_dict(s: InvestorState) -> dict:
    return {
        "compound_state":     s.compound_state,
        "state_description":  s.state_description,
        "dominant_factors":   s.dominant_factors,
        "state_implications": s.state_implications,
        "state_stability":    s.state_stability,
        "confidence":         s.confidence,
        # v14 structured fields
        "dominant_trait":     s.dominant_trait,
        "suppressed_traits":  s.suppressed_traits,
        "resilience_level":   s.resilience_level,
        "resilience_evidence": s.resilience_evidence,
        "shift_detected":     s.shift_detected,
        "baseline_behavior":  s.baseline_behavior,
        "current_behavior":   s.current_behavior,
        "shift_permanence":   s.shift_permanence,
        "warning":            s.warning,
    }
