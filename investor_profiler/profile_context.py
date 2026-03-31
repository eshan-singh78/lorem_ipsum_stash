"""
Profile Context Layer — InvestorDNA v6
Transforms validated fields + raw text into a structured context object.

Architecture:
  validated_fields + raw_text → ProfileContext

The ProfileContext is the single source of truth for all downstream reasoning.
No axis or category scorer should ever read raw fields directly.

Context captures:
  - demographics: age, income, city tier, employment
  - life_events: death, job change, marriage, responsibility shift
  - financial_snapshot: assets, debts, savings, insurance
  - cultural_context: family role, hidden obligations, religious commitments
  - behavioral_signals: fear, peer influence, decision patterns, anxiety markers
"""

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Context data structures
# ---------------------------------------------------------------------------

@dataclass
class LifeEvent:
    event_type: str          # "death", "job_change", "marriage", "responsibility_shift", "crisis"
    description: str
    recency: str             # "recent" | "past" | "unknown"
    emotional_weight: str    # "high" | "medium" | "low"


@dataclass
class CulturalSignal:
    signal_type: str         # "family_role", "hidden_obligation", "religious", "social_pressure"
    description: str
    negotiability: str       # "fixed" | "flexible" | "unknown"


@dataclass
class BehavioralSignal:
    signal_type: str         # "fear", "peer_influence", "anxiety", "overconfidence", "analytical"
    description: str
    strength: str            # "strong" | "moderate" | "weak"


@dataclass
class ProfileContext:
    # Core demographics
    income_type: str                    # "salaried" | "business" | "gig" | "unknown"
    monthly_income: float | None
    city_tier: str                      # "metro" | "tier2" | "tier3" | "unknown"
    employment_stability: str           # "stable" | "moderate" | "volatile" | "unknown"

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
    loss_reaction: str | None           # "panic" | "cautious" | "neutral" | "aggressive"
    risk_behavior: str | None           # "low" | "medium" | "high"

    # Obligation
    near_term_obligation_level: str     # "none" | "moderate" | "high"
    obligation_type: str | None
    future_obligation_score: float

    # Rich context layers
    life_events: list[LifeEvent] = field(default_factory=list)
    cultural_signals: list[CulturalSignal] = field(default_factory=list)
    behavioral_signals: list[BehavioralSignal] = field(default_factory=list)

    # Derived flags (set during context building)
    grief_state: bool = False
    peer_driven: bool = False
    hidden_obligation_detected: bool = False
    fragmentation_risk: bool = False
    recency_bias_risk: bool = False
    emerging_constraint: bool = False


# ---------------------------------------------------------------------------
# Life event detection from raw text
# ---------------------------------------------------------------------------

_DEATH_PATTERNS = [
    r"(?:father|mother|parent|spouse|husband|wife|sibling)\s+(?:passed|died|death|expired|no more)",
    r"(?:lost|losing)\s+(?:my|his|her|their)\s+(?:father|mother|parent|spouse)",
    r"(?:death|demise|passing)\s+of\s+(?:father|mother|parent|spouse)",
    r"(?:recently\s+)?(?:bereaved|widowed)",
]

_JOB_CHANGE_PATTERNS = [
    r"(?:recently\s+)?(?:changed|switched|left|quit|resigned from|lost)\s+(?:job|work|employment)",
    r"(?:new\s+job|new\s+role|joined|started\s+working\s+at)",
    r"(?:laid off|retrenchment|redundancy|fired)",
]

_MARRIAGE_PATTERNS = [
    r"(?:recently\s+)?(?:married|got\s+married|wedding\s+recently)",
    r"(?:newly\s+wed|just\s+married)",
]

_RESPONSIBILITY_SHIFT_PATTERNS = [
    r"(?:eldest\s+son|only\s+son|primary\s+earner|sole\s+earner|breadwinner)",
    r"(?:took\s+over|taking\s+over|responsible\s+for|now\s+responsible)",
    r"(?:entire\s+(?:financial\s+)?responsibility|all\s+(?:financial\s+)?responsibility)",
]

_CRISIS_PATTERNS = [
    r"(?:financial\s+crisis|debt\s+trap|bankruptcy|insolvent)",
    r"(?:medical\s+emergency|hospital\s+bills|medical\s+debt)",
    r"(?:covid|pandemic)\s+(?:hit|affected|impacted|crisis)",
]


def _detect_life_events(text: str) -> list[LifeEvent]:
    t = text.lower()
    events: list[LifeEvent] = []

    for pat in _DEATH_PATTERNS:
        if re.search(pat, t):
            recency = "recent" if any(w in t for w in ["recently", "just", "last year", "this year", "2020", "2021", "2022", "2023", "2024"]) else "past"
            events.append(LifeEvent(
                event_type="death",
                description="Loss of close family member detected",
                recency=recency,
                emotional_weight="high",
            ))
            break

    for pat in _JOB_CHANGE_PATTERNS:
        if re.search(pat, t):
            events.append(LifeEvent(
                event_type="job_change",
                description="Recent employment change detected",
                recency="recent",
                emotional_weight="medium",
            ))
            break

    for pat in _MARRIAGE_PATTERNS:
        if re.search(pat, t):
            events.append(LifeEvent(
                event_type="marriage",
                description="Recent marriage detected",
                recency="recent",
                emotional_weight="medium",
            ))
            break

    for pat in _RESPONSIBILITY_SHIFT_PATTERNS:
        if re.search(pat, t):
            events.append(LifeEvent(
                event_type="responsibility_shift",
                description="Sudden increase in financial responsibility detected",
                recency="recent",
                emotional_weight="high",
            ))
            break

    for pat in _CRISIS_PATTERNS:
        if re.search(pat, t):
            events.append(LifeEvent(
                event_type="crisis",
                description="Financial or medical crisis detected",
                recency="recent",
                emotional_weight="high",
            ))
            break

    return events


# ---------------------------------------------------------------------------
# Cultural signal detection
# ---------------------------------------------------------------------------

_FAMILY_ROLE_PATTERNS = {
    "eldest_son": [r"eldest\s+son", r"older\s+son", r"first\s+son"],
    "joint_family": [r"joint\s+family", r"living\s+with\s+(?:parents|family)", r"parents\s+at\s+home"],
    "sole_earner": [r"sole\s+earner", r"only\s+earner", r"primary\s+earner", r"breadwinner"],
    "patriarch": [r"head\s+of\s+(?:the\s+)?family", r"family\s+head"],
}

_HIDDEN_OBLIGATION_PATTERNS = [
    r"(?:secretly|secretly\s+saving|saving\s+secretly|parents\s+(?:don't|do\s+not)\s+know)",
    r"(?:hidden|undisclosed|unspoken)\s+(?:savings?|obligation|commitment)",
    r"(?:wedding|marriage)\s+(?:savings?|fund|corpus|money)",
    r"(?:dowry|dahej)",
    r"(?:earmarked|set\s+aside|keeping\s+aside)\s+for",
]

_RELIGIOUS_PATTERNS = [
    r"(?:religious|dharmic|spiritual)\s+(?:giving|donation|commitment|obligation)",
    r"(?:temple|mosque|church|gurudwara)\s+(?:donation|giving|contribution)",
    r"(?:charity|charitable)\s+(?:giving|donation)",
    r"(?:zakat|tithe|daan|dakshina)",
]

_SOCIAL_PRESSURE_PATTERNS = [
    r"(?:social\s+expectation|community\s+expectation|family\s+expectation)",
    r"(?:shame|embarrassment|face|izzat|reputation)\s+(?:if|when|about)",
    r"(?:what\s+will\s+people\s+say|log\s+kya\s+kahenge)",
    r"(?:community\s+standard|society\s+expects|expected\s+to)",
]


def _detect_cultural_signals(text: str) -> list[CulturalSignal]:
    t = text.lower()
    signals: list[CulturalSignal] = []

    for role, patterns in _FAMILY_ROLE_PATTERNS.items():
        if any(re.search(p, t) for p in patterns):
            signals.append(CulturalSignal(
                signal_type="family_role",
                description=f"Family role: {role.replace('_', ' ')}",
                negotiability="fixed",
            ))

    for pat in _HIDDEN_OBLIGATION_PATTERNS:
        if re.search(pat, t):
            signals.append(CulturalSignal(
                signal_type="hidden_obligation",
                description="Hidden or unspoken financial obligation detected",
                negotiability="fixed",
            ))
            break

    for pat in _RELIGIOUS_PATTERNS:
        if re.search(pat, t):
            signals.append(CulturalSignal(
                signal_type="religious",
                description="Religious or charitable giving commitment detected",
                negotiability="fixed",
            ))
            break

    for pat in _SOCIAL_PRESSURE_PATTERNS:
        if re.search(pat, t):
            signals.append(CulturalSignal(
                signal_type="social_pressure",
                description="Social or community financial pressure detected",
                negotiability="fixed",
            ))
            break

    return signals


# ---------------------------------------------------------------------------
# Behavioral signal detection
# ---------------------------------------------------------------------------

_FEAR_PATTERNS = [
    r"(?:can't\s+sleep|cannot\s+sleep|sleepless|sleepless\s+nights)",
    r"(?:scared|terrified|frightened)\s+(?:of|about)\s+(?:loss|losing|market)",
    r"(?:panic|panicked|panicking)\s+(?:sell|sold|selling|exit)",
    r"(?:stopped\s+investing|pulled\s+out|withdrew\s+all|exit\s+market)",
]

_ANXIETY_PATTERNS = [
    r"(?:checks?\s+(?:nav|price|portfolio|market))\s+(?:daily|every\s+day|multiple\s+times|constantly|obsessively)",
    r"(?:worried|anxious|nervous|stressed)\s+(?:about|when|if)\s+(?:market|loss|portfolio|investment)",
    r"(?:considered\s+stopping|thinking\s+of\s+stopping|wanted\s+to\s+stop)\s+(?:sip|investing)",
]

_PEER_INFLUENCE_PATTERNS = [
    r"(?:friend|colleague|relative|neighbour)\s+(?:told|suggested|recommended|showed|said)",
    r"(?:because\s+(?:my\s+)?friend|after\s+seeing\s+friend|friend\s+(?:posted|shared|showed))",
    r"(?:instagram|youtube|twitter|social\s+media)\s+(?:finfluencer|influencer|tip|advice)",
    r"(?:tips?\s+from|following\s+(?:tips?|advice|others))",
    r"(?:peer\s+(?:pressure|comparison|influence)|keeping\s+up\s+with)",
]

_OVERCONFIDENCE_PATTERNS = [
    r"(?:always\s+(?:right|correct|profitable)|never\s+(?:wrong|lost))",
    r"(?:confident\s+in\s+(?:my\s+)?(?:picks?|ability|analysis|stock))",
    r"(?:beat\s+the\s+market|outperform|alpha\s+generation)",
    r"(?:15[-–]20\s+trades?|active\s+(?:trader|trading))",
]

_ANALYTICAL_PATTERNS = [
    r"(?:researches?\s+(?:before|independently|thoroughly)|does\s+own\s+research)",
    r"(?:spreadsheet|excel|model|analysis|calculated)",
    r"(?:reverse.engineer|negotiated|systematic|prioritized\s+high.interest)",
]


def _detect_behavioral_signals(text: str) -> list[BehavioralSignal]:
    t = text.lower()
    signals: list[BehavioralSignal] = []

    for pat in _FEAR_PATTERNS:
        if re.search(pat, t):
            signals.append(BehavioralSignal(
                signal_type="fear",
                description="Strong fear/panic response to market loss detected",
                strength="strong",
            ))
            break

    for pat in _ANXIETY_PATTERNS:
        if re.search(pat, t):
            signals.append(BehavioralSignal(
                signal_type="anxiety",
                description="Anxiety-driven monitoring behavior detected",
                strength="moderate",
            ))
            break

    for pat in _PEER_INFLUENCE_PATTERNS:
        if re.search(pat, t):
            signals.append(BehavioralSignal(
                signal_type="peer_influence",
                description="Investment decisions influenced by peers or social media",
                strength="strong",
            ))
            break

    for pat in _OVERCONFIDENCE_PATTERNS:
        if re.search(pat, t):
            signals.append(BehavioralSignal(
                signal_type="overconfidence",
                description="Overconfidence in investment ability detected",
                strength="moderate",
            ))
            break

    for pat in _ANALYTICAL_PATTERNS:
        if re.search(pat, t):
            signals.append(BehavioralSignal(
                signal_type="analytical",
                description="Systematic, research-driven decision making detected",
                strength="strong",
            ))
            break

    return signals


# ---------------------------------------------------------------------------
# City tier inference
# ---------------------------------------------------------------------------

_METRO_CITIES = {
    "mumbai", "delhi", "bangalore", "bengaluru", "hyderabad",
    "chennai", "kolkata", "pune", "noida", "gurgaon", "gurugram",
    "ahmedabad", "navi mumbai", "thane",
}
_TIER2_CITIES = {
    "surat", "indore", "jaipur", "lucknow", "kanpur", "nagpur",
    "bhopal", "visakhapatnam", "patna", "vadodara", "coimbatore",
    "agra", "nashik", "faridabad", "meerut", "rajkot", "varanasi",
    "amritsar", "allahabad", "prayagraj", "ranchi", "howrah",
    "jabalpur", "gwalior", "vijayawada", "jodhpur", "madurai",
    "raipur", "kota", "chandigarh", "guwahati", "solapur",
}


def _infer_city_tier(text: str) -> str:
    t = text.lower()
    for city in _METRO_CITIES:
        if city in t:
            return "metro"
    for city in _TIER2_CITIES:
        if city in t:
            return "tier2"
    if any(w in t for w in ["tier 1", "tier-1", "metro city", "metropolitan"]):
        return "metro"
    if any(w in t for w in ["tier 2", "tier-2", "tier 3", "tier-3", "small town", "village"]):
        return "tier2"
    return "unknown"


# ---------------------------------------------------------------------------
# Employment stability inference
# ---------------------------------------------------------------------------

def _infer_employment_stability(income_type: str, text: str) -> str:
    t = text.lower()
    if income_type == "salaried":
        if any(w in t for w in ["government", "govt", "psu", "public sector", "bank employee"]):
            return "stable"
        return "stable"
    elif income_type == "business":
        if any(w in t for w in ["cyclical", "seasonal", "irregular", "volatile"]):
            return "moderate"
        return "moderate"
    elif income_type == "gig":
        return "volatile"
    return "unknown"


# ---------------------------------------------------------------------------
# Derived flag computation
# ---------------------------------------------------------------------------

def _compute_flags(
    life_events: list[LifeEvent],
    cultural_signals: list[CulturalSignal],
    behavioral_signals: list[BehavioralSignal],
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

    # Grief state: recent death + high emotional weight
    for ev in life_events:
        if ev.event_type == "death" and ev.recency == "recent":
            flags["grief_state"] = True

    # Peer driven: peer influence signal OR decision_autonomy=False
    for sig in behavioral_signals:
        if sig.signal_type == "peer_influence":
            flags["peer_driven"] = True
    if fields.get("decision_autonomy") is False:
        flags["peer_driven"] = True

    # Hidden obligation
    for sig in cultural_signals:
        if sig.signal_type == "hidden_obligation":
            flags["hidden_obligation_detected"] = True

    # Fragmentation risk: multiple advisors / multiple platforms mentioned
    # (detected via text patterns in context building)
    flags["fragmentation_risk"] = fields.get("_fragmentation_risk", False)

    # Recency bias: low experience + crypto/recent bull run gains
    exp = fields.get("experience_years")
    fks = fields.get("financial_knowledge_score")
    rb  = fields.get("risk_behavior")
    flags["recency_bias_risk"] = (
        exp is not None and exp < 1
        and rb == "high"
        and (fks is None or fks <= 2)
    )

    # Emerging constraint: current obligation low but future high
    emi_ratio = fields.get("emi_ratio")
    current_low = (emi_ratio is None or emi_ratio < 20) and (fields.get("dependents") or 0) < 2
    flags["emerging_constraint"] = current_low and future_obligation_score >= 10

    return flags


# ---------------------------------------------------------------------------
# Fragmentation risk detection
# ---------------------------------------------------------------------------

_FRAGMENTATION_PATTERNS = [
    r"(?:multiple\s+(?:advisors?|mfds?|brokers?|platforms?))",
    r"(?:two\s+(?:different|separate)\s+(?:mfds?|advisors?|brokers?))",
    r"(?:three\s+(?:different|separate)\s+(?:financial|advisors?|professionals?))",
    r"(?:none\s+(?:aware|knows?)\s+(?:of\s+)?(?:the\s+)?(?:complete|full|other|each\s+other))",
    r"(?:different\s+(?:mfds?|advisors?|brokers?|platforms?))",
]


def _detect_fragmentation(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in _FRAGMENTATION_PATTERNS)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_profile_context(
    validated_fields: dict,
    raw_text: str,
    future_obligation_score: float = 0.0,
) -> ProfileContext:
    """
    Build a rich ProfileContext from validated fields + raw text.

    This is the ONLY place that reads validated_fields.
    All downstream reasoning reads from ProfileContext.
    """
    f = validated_fields  # shorthand

    # Detect rich context from raw text
    life_events       = _detect_life_events(raw_text)
    cultural_signals  = _detect_cultural_signals(raw_text)
    behavioral_signals = _detect_behavioral_signals(raw_text)
    city_tier         = _infer_city_tier(raw_text)
    fragmentation     = _detect_fragmentation(raw_text)

    income_type = f.get("income_type", "unknown") or "unknown"
    employment_stability = _infer_employment_stability(income_type, raw_text)

    # Inject fragmentation flag into fields for flag computation
    fields_with_flags = dict(f)
    fields_with_flags["_fragmentation_risk"] = fragmentation

    flags = _compute_flags(
        life_events, cultural_signals, behavioral_signals,
        fields_with_flags, future_obligation_score,
    )

    return ProfileContext(
        # Demographics
        income_type=income_type,
        monthly_income=f.get("monthly_income"),
        city_tier=city_tier,
        employment_stability=employment_stability,

        # Financial snapshot
        emergency_months=f.get("emergency_months"),
        emi_amount=f.get("emi_amount"),
        emi_ratio=f.get("emi_ratio"),
        dependents=f.get("dependents"),
        has_insurance=None,  # not yet extracted — future field

        # Experience & sophistication
        experience_years=f.get("experience_years"),
        financial_knowledge_score=f.get("financial_knowledge_score"),
        decision_autonomy=f.get("decision_autonomy"),

        # Behavioral
        loss_reaction=f.get("loss_reaction"),
        risk_behavior=f.get("risk_behavior"),

        # Obligation
        near_term_obligation_level=f.get("near_term_obligation_level") or "none",
        obligation_type=f.get("obligation_type"),
        future_obligation_score=future_obligation_score,

        # Rich context
        life_events=life_events,
        cultural_signals=cultural_signals,
        behavioral_signals=behavioral_signals,

        # Derived flags
        grief_state=flags["grief_state"],
        peer_driven=flags["peer_driven"],
        hidden_obligation_detected=flags["hidden_obligation_detected"],
        fragmentation_risk=flags["fragmentation_risk"],
        recency_bias_risk=flags["recency_bias_risk"],
        emerging_constraint=flags["emerging_constraint"],
    )


def context_to_dict(ctx: ProfileContext) -> dict:
    """Serialize ProfileContext to a JSON-safe dict for output."""
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
        },
        "obligation": {
            "near_term_obligation_level": ctx.near_term_obligation_level,
            "obligation_type": ctx.obligation_type,
            "future_obligation_score": ctx.future_obligation_score,
        },
        "life_events": [
            {
                "type": e.event_type,
                "description": e.description,
                "recency": e.recency,
                "emotional_weight": e.emotional_weight,
            }
            for e in ctx.life_events
        ],
        "cultural_signals": [
            {
                "type": s.signal_type,
                "description": s.description,
                "negotiability": s.negotiability,
            }
            for s in ctx.cultural_signals
        ],
        "behavioral_signals": [
            {
                "type": s.signal_type,
                "description": s.description,
                "strength": s.strength,
            }
            for s in ctx.behavioral_signals
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
