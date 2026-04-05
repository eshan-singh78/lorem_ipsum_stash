"""
Profile Context Layer — InvestorDNA v13
Transforms validated fields + SignalOutput into a structured ProfileContext.

Architecture:
  validated_fields + SignalOutput → ProfileContext

v13 change: ALL regex pattern matching removed.
Profile context is built entirely from SignalOutput (LLM-extracted signals).
No _DEATH_PATTERNS, no _FEAR_PATTERNS, no cultural regex.

The ProfileContext is the single source of truth for all downstream scoring.
No axis or category scorer reads raw fields or raw text directly.
"""

from dataclasses import dataclass, field
from signal_extraction import (
    SignalOutput, LifeEventSignal, ContradictionSignal,
    BehaviorSignal, DecisionStyleSignal, ResponsibilitySignal,
)


# ---------------------------------------------------------------------------
# Context data structures — unchanged interface for downstream compatibility
# ---------------------------------------------------------------------------

@dataclass
class LifeEvent:
    event_type: str
    description: str
    recency: str        # "recent" | "past" | "unknown"
    emotional_weight: str  # "high" | "medium" | "low"


@dataclass
class CulturalSignal:
    signal_type: str       # "family_role" | "hidden_obligation" | "religious" | "social_pressure"
    description: str
    negotiability: str     # "fixed" | "flexible" | "unknown"


@dataclass
class BehavioralSignal:
    signal_type: str       # "fear" | "peer_influence" | "anxiety" | "overconfidence" | "analytical" | "resilience"
    description: str
    strength: str          # "strong" | "moderate" | "weak"


@dataclass
class ProfileContext:
    # Core demographics
    income_type: str
    monthly_income: float | None
    city_tier: str                  # "metro" | "tier2" | "tier3" | "unknown"
    employment_stability: str       # "stable" | "moderate" | "volatile" | "unknown"

    # Financial snapshot
    emergency_months: float | None
    emi_amount: float | None
    emi_ratio: float | None
    dependents: int | None
    has_insurance: bool | None

    # Experience & sophistication
    experience_years: float | None
    financial_knowledge_score: int | None
    decision_autonomy: bool | None

    # Behavioral
    loss_reaction: str | None
    risk_behavior: str | None

    # Obligation
    near_term_obligation_level: str
    obligation_type: str | None
    future_obligation_score: float

    # Rich context layers
    life_events: list[LifeEvent] = field(default_factory=list)
    cultural_signals: list[CulturalSignal] = field(default_factory=list)
    behavioral_signals: list[BehavioralSignal] = field(default_factory=list)

    # Derived flags
    grief_state: bool = False
    peer_driven: bool = False
    hidden_obligation_detected: bool = False
    fragmentation_risk: bool = False
    recency_bias_risk: bool = False
    emerging_constraint: bool = False

    # v13: resilience and contradiction signals — preserved, never dropped
    resilience_level: str = "medium"        # low | medium | high
    resilience_evidence: str = ""
    contradictions: list = field(default_factory=list)  # list[ContradictionSignal]


# ---------------------------------------------------------------------------
# Build from signals — no regex
# ---------------------------------------------------------------------------

def _impact_to_weight(impact: str) -> str:
    return {"high": "high", "medium": "medium", "low": "low"}.get(impact, "medium")


def _life_events_from_signals(signals: SignalOutput) -> list[LifeEvent]:
    events = []
    for ev in signals.life_events:
        events.append(LifeEvent(
            event_type=ev.type,
            description=ev.description,
            recency=ev.recency,
            emotional_weight=_impact_to_weight(ev.impact),
        ))
    return events


def _cultural_signals_from_signals(signals: SignalOutput) -> list[CulturalSignal]:
    cultural = []

    # Family role from responsibility
    resp = signals.responsibility
    if resp.role == "primary":
        cultural.append(CulturalSignal(
            signal_type="family_role",
            description=f"Primary financial provider — {resp.dependents_description or 'family dependent on this investor'}",
            negotiability="fixed",
        ))

    # Cultural obligations from responsibility
    for obligation in resp.cultural_obligations:
        if obligation:
            # Classify the obligation type
            ob_lower = obligation.lower()
            if any(w in ob_lower for w in ["religious", "temple", "mosque", "church", "zakat", "tithe", "daan"]):
                cultural.append(CulturalSignal(
                    signal_type="religious",
                    description=obligation,
                    negotiability="fixed",
                ))
            elif any(w in ob_lower for w in ["hidden", "secret", "wedding", "marriage", "dowry"]):
                cultural.append(CulturalSignal(
                    signal_type="hidden_obligation",
                    description=obligation,
                    negotiability="fixed",
                ))
            elif any(w in ob_lower for w in ["social", "community", "expectation", "reputation"]):
                cultural.append(CulturalSignal(
                    signal_type="social_pressure",
                    description=obligation,
                    negotiability="fixed",
                ))
            else:
                cultural.append(CulturalSignal(
                    signal_type="family_role",
                    description=obligation,
                    negotiability="fixed",
                ))

    # Hidden obligations from financial state
    if signals.financial_state.hidden_obligations:
        cultural.append(CulturalSignal(
            signal_type="hidden_obligation",
            description=signals.financial_state.hidden_obligations,
            negotiability="fixed",
        ))

    return cultural


def _behavioral_signals_from_signals(signals: SignalOutput) -> list[BehavioralSignal]:
    behavioral = []
    beh = signals.behavior
    ds  = signals.decision_style

    # Loss response → fear or resilience signal
    if beh.loss_response == "panic":
        behavioral.append(BehavioralSignal(
            signal_type="fear",
            description=beh.loss_response_detail or "Panic response to portfolio losses",
            strength="strong",
        ))
    elif beh.loss_response == "cautious":
        behavioral.append(BehavioralSignal(
            signal_type="anxiety",
            description=beh.loss_response_detail or "Cautious response to losses",
            strength="moderate",
        ))

    # Resilience — always captured, never dropped
    if beh.resilience_level == "high":
        behavioral.append(BehavioralSignal(
            signal_type="resilience",
            description=beh.resilience_evidence or "High resilience demonstrated",
            strength="strong",
        ))
    elif beh.resilience_level == "medium":
        behavioral.append(BehavioralSignal(
            signal_type="resilience",
            description=beh.resilience_evidence or "Moderate resilience",
            strength="moderate",
        ))

    # Peer influence
    if ds.peer_influence in ("medium", "high"):
        behavioral.append(BehavioralSignal(
            signal_type="peer_influence",
            description=ds.peer_influence_detail or "Investment decisions influenced by peers",
            strength="strong" if ds.peer_influence == "high" else "moderate",
        ))

    # Analytical tendency
    if ds.analytical_tendency in ("medium", "high"):
        behavioral.append(BehavioralSignal(
            signal_type="analytical",
            description="Systematic, research-driven decision making detected",
            strength="strong" if ds.analytical_tendency == "high" else "moderate",
        ))

    return behavioral


def _infer_city_tier(raw_text: str) -> str:
    """Minimal city tier inference — kept as a lightweight lookup, not behavioral regex."""
    t = raw_text.lower()
    metros = {"mumbai", "delhi", "bangalore", "bengaluru", "hyderabad", "chennai",
              "kolkata", "pune", "noida", "gurgaon", "gurugram", "ahmedabad"}
    tier2s = {"surat", "indore", "jaipur", "lucknow", "nagpur", "bhopal",
              "visakhapatnam", "patna", "vadodara", "coimbatore", "chandigarh"}
    for city in metros:
        if city in t:
            return "metro"
    for city in tier2s:
        if city in t:
            return "tier2"
    if any(w in t for w in ["metro city", "metropolitan", "tier 1", "tier-1"]):
        return "metro"
    if any(w in t for w in ["tier 2", "tier-2", "tier 3", "tier-3", "small town"]):
        return "tier2"
    return "unknown"


def _infer_employment_stability(income_type: str, signals: SignalOutput) -> str:
    if income_type == "salaried":
        return "stable"
    elif income_type == "business":
        return "moderate"
    elif income_type == "gig":
        return "volatile"
    return "unknown"


def _compute_flags(
    life_events: list[LifeEvent],
    cultural_signals: list[CulturalSignal],
    behavioral_signals: list[BehavioralSignal],
    signals: SignalOutput,
    fields: dict,
    future_obligation_score: float,
) -> dict[str, bool]:
    flags = {
        "grief_state": False,
        "peer_driven": False,
        "hidden_obligation_detected": False,
        "fragmentation_risk": False,
        "recency_bias_risk": False,
        "emerging_constraint": False,
    }

    # Grief: death event with recent recency
    for ev in life_events:
        if ev.event_type == "death" and ev.recency == "recent":
            flags["grief_state"] = True

    # Peer driven: from signal, not regex
    if signals.decision_style.peer_influence in ("medium", "high"):
        flags["peer_driven"] = True
    if fields.get("decision_autonomy") is False:
        flags["peer_driven"] = True

    # Hidden obligation
    for sig in cultural_signals:
        if sig.signal_type == "hidden_obligation":
            flags["hidden_obligation_detected"] = True
    if signals.financial_state.hidden_obligations:
        flags["hidden_obligation_detected"] = True

    # Recency bias: low experience + high risk behavior
    exp = fields.get("experience_years")
    fks = fields.get("financial_knowledge_score")
    rb  = fields.get("risk_behavior")
    flags["recency_bias_risk"] = (
        exp is not None and exp < 1
        and rb == "high"
        and (fks is None or fks <= 2)
    )

    # Emerging constraint: low current obligations but high future
    emi_ratio = fields.get("emi_ratio")
    current_low = (emi_ratio is None or emi_ratio < 20) and (fields.get("dependents") or 0) < 2
    flags["emerging_constraint"] = current_low and future_obligation_score >= 10

    return flags


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_profile_context(
    validated_fields: dict,
    raw_text: str,
    future_obligation_score: float = 0.0,
    signals: SignalOutput | None = None,
) -> "ProfileContext":
    """
    Build ProfileContext from validated fields + SignalOutput.

    v13: signals parameter is the primary source for all behavioral/contextual data.
    raw_text is used only for city tier inference (lightweight lookup).
    If signals is None, context is built from fields only (degraded mode).
    """
    f = validated_fields

    if signals is not None:
        life_events        = _life_events_from_signals(signals)
        cultural_signals   = _cultural_signals_from_signals(signals)
        behavioral_signals = _behavioral_signals_from_signals(signals)
        resilience_level   = signals.behavior.resilience_level
        resilience_evidence = signals.behavior.resilience_evidence
        contradictions     = signals.contradictions
    else:
        # Degraded mode: no signals available
        life_events        = []
        cultural_signals   = []
        behavioral_signals = []
        resilience_level   = "medium"
        resilience_evidence = ""
        contradictions     = []

    city_tier = _infer_city_tier(raw_text)
    income_type = f.get("income_type", "unknown") or "unknown"
    employment_stability = _infer_employment_stability(income_type, signals) if signals else "unknown"

    flags = _compute_flags(
        life_events, cultural_signals, behavioral_signals,
        signals, f, future_obligation_score,
    ) if signals else {
        "grief_state": False, "peer_driven": False,
        "hidden_obligation_detected": False, "fragmentation_risk": False,
        "recency_bias_risk": False, "emerging_constraint": False,
    }

    return ProfileContext(
        income_type=income_type,
        monthly_income=f.get("monthly_income"),
        city_tier=city_tier,
        employment_stability=employment_stability,
        emergency_months=f.get("emergency_months"),
        emi_amount=f.get("emi_amount"),
        emi_ratio=f.get("emi_ratio"),
        dependents=f.get("dependents"),
        has_insurance=None,
        experience_years=f.get("experience_years"),
        financial_knowledge_score=f.get("financial_knowledge_score"),
        decision_autonomy=f.get("decision_autonomy"),
        loss_reaction=f.get("loss_reaction"),
        risk_behavior=f.get("risk_behavior"),
        near_term_obligation_level=f.get("near_term_obligation_level") or "none",
        obligation_type=f.get("obligation_type"),
        future_obligation_score=future_obligation_score,
        life_events=life_events,
        cultural_signals=cultural_signals,
        behavioral_signals=behavioral_signals,
        grief_state=flags["grief_state"],
        peer_driven=flags["peer_driven"],
        hidden_obligation_detected=flags["hidden_obligation_detected"],
        fragmentation_risk=flags["fragmentation_risk"],
        recency_bias_risk=flags["recency_bias_risk"],
        emerging_constraint=flags["emerging_constraint"],
        resilience_level=resilience_level,
        resilience_evidence=resilience_evidence,
        contradictions=contradictions,
    )


def context_to_dict(ctx: ProfileContext) -> dict:
    return {
        "demographics": {
            "income_type": ctx.income_type,
            "monthly_income": ctx.monthly_income,
            "city_tier": ctx.city_tier,
            "employment_stability": ctx.employment_stability,
        },
        "financial_snapshot": {
            "emergency_months": ctx.emergency_months,
            "emi_amount": ctx.emi_amount,
            "emi_ratio": ctx.emi_ratio,
            "dependents": ctx.dependents,
        },
        "experience": {
            "experience_years": ctx.experience_years,
            "financial_knowledge_score": ctx.financial_knowledge_score,
            "decision_autonomy": ctx.decision_autonomy,
        },
        "behavioral": {
            "loss_reaction": ctx.loss_reaction,
            "risk_behavior": ctx.risk_behavior,
            "resilience_level": ctx.resilience_level,
            "resilience_evidence": ctx.resilience_evidence,
        },
        "obligation": {
            "near_term_obligation_level": ctx.near_term_obligation_level,
            "obligation_type": ctx.obligation_type,
            "future_obligation_score": ctx.future_obligation_score,
        },
        "life_events": [
            {"type": e.event_type, "description": e.description,
             "recency": e.recency, "emotional_weight": e.emotional_weight}
            for e in ctx.life_events
        ],
        "cultural_signals": [
            {"type": s.signal_type, "description": s.description,
             "negotiability": s.negotiability}
            for s in ctx.cultural_signals
        ],
        "behavioral_signals": [
            {"type": s.signal_type, "description": s.description, "strength": s.strength}
            for s in ctx.behavioral_signals
        ],
        "contradictions": [
            {"type": c.type, "dominant_trait": c.dominant_trait,
             "suppressed_trait": c.suppressed_trait, "explanation": c.explanation}
            for c in ctx.contradictions
        ],
        "flags": {
            "grief_state": ctx.grief_state,
            "peer_driven": ctx.peer_driven,
            "hidden_obligation_detected": ctx.hidden_obligation_detected,
            "fragmentation_risk": ctx.fragmentation_risk,
            "recency_bias_risk": ctx.recency_bias_risk,
            "emerging_constraint": ctx.emerging_constraint,
        },
    }
