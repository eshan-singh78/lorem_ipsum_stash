"""
Report Formatter — InvestorDNA PDF Pipeline
Converts raw pipeline JSON → structured PDF-ready payload.

Architecture:
  pipeline_output (status + report + meta) → PDF payload dict

Design principle: Structure, clean, and present intelligence — never alter it.
No LLM calls. No new reasoning. Pure deterministic transformation.
"""

from __future__ import annotations
import re
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Cleaning rules — strip system language from all advisor-facing text
# ---------------------------------------------------------------------------

_CLEAN_RULES = [
    (r"\bLLM\b",                          "system"),
    (r"\bllm\b",                          "system"),
    (r"\bvalidation failed\b",            "requires verification"),
    (r"\bvalidation\b",                   "verification"),
    (r"\bconstraint engine\b",            "risk assessment system"),
    (r"\bconstraint violation\b",         "profile adjustment required"),
    (r"\bconstraint\b",                   "data limitation"),
    (r"\bguardrail adjustment\b",         "risk-based adjustment"),
    (r"\bguardrail\b",                    "safety check"),
    (r"\btrace validation\b",             "profile verification"),
    (r"\btrace valid\b",                  "profile verified"),
    (r"\btrace\b",                        "profile analysis"),
    (r"\bfallback used\b",                "system limitation applied"),
    (r"\bfallback\b",                     "conservative default"),
    (r"\bblocking violation\b",           "profile inconsistency"),
    (r"\bscore mismatch\b",               "profile signal mismatch"),
    (r"\breasoning trace\b",              "profile analysis"),
    (r"\bLLM unavailable\b",              "system limitation"),
    (r"\bunavailable\b",                  "not available"),
]

_PROVISIONAL_PHRASES = {
    "fallback", "llm unavailable", "validation failed",
    "conservative default", "system limitation",
}


def _clean(text: str) -> str:
    """Strip all system/internal language from advisor-facing text."""
    if not text:
        return ""
    for pattern, replacement in _CLEAN_RULES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text.strip()


def _is_system_phrase(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in _PROVISIONAL_PHRASES)


def _clean_list(items: list) -> list[str]:
    """Clean a list of strings, removing pure system phrases."""
    result = []
    for item in (items or []):
        cleaned = _clean(str(item))
        if cleaned and not _is_system_phrase(cleaned):
            result.append(cleaned)
    return result


# ---------------------------------------------------------------------------
# Safe accessors
# ---------------------------------------------------------------------------

def _get(d: dict, *keys, default=""):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


def _lst(d: dict, *keys) -> list:
    val = _get(d, *keys, default=[])
    return val if isinstance(val, list) else []


def _num(val, default=None):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _score_label(score) -> str:
    s = _num(score)
    if s is None:
        return "unknown"
    if s < 35:
        return "low"
    if s < 65:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_cover_page(report: dict, meta: dict, client_id: str | None) -> dict:
    cid = client_id or report.get("client_id") or f"INV-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    fallback_used = meta.get("fallback_used", False)
    status = "Provisional" if fallback_used else "Final"

    return {
        "client_id":    cid,
        "report_title": "Investor Suitability Profile",
        "version":      "InvestorDNA v1",
        "date":         datetime.now(timezone.utc).strftime("%d %B %Y"),
        "status":       status,
        "language":     "English",
        "ai_disclosure": (
            "This report was generated with the assistance of an AI-based profiling system. "
            "All recommendations are subject to advisor review and must not be acted upon "
            "without independent professional validation."
        ),
    }


def _infer_primary_objective(report: dict, full_profile: dict) -> str:
    """Derive a single primary objective from the profile state."""
    axis = _get(full_profile, "axis_scores", default={})
    ctx  = _get(full_profile, "profile_context", default={})
    flags = _get(ctx, "flags", default={})
    fin   = _get(ctx, "financial_snapshot", default={})

    capacity   = _num(_get(axis, "financial_capacity"), 50)
    obligation = _num(_get(axis, "obligation"), 0)
    cashflow   = _num(_get(axis, "cashflow"), 50)
    em         = _num(fin.get("emergency_months"), 0)

    if flags.get("grief_state"):
        return "Preserve financial stability during a period of personal transition"
    if capacity is not None and capacity < 25:
        return "Build financial resilience before initiating any investment activity"
    if em is not None and em < 3:
        return "Establish an emergency fund as the immediate financial priority"
    if obligation is not None and obligation > 70:
        return "Reduce obligation burden to create investable surplus"
    if cashflow is not None and cashflow < 35:
        return "Stabilize income and cash flow before committing to investments"
    if capacity is not None and capacity < 50:
        return "Build financial stability before investing"
    return "Grow wealth through a disciplined, risk-appropriate investment strategy"


def _build_executive_summary(report: dict, meta: dict, full_profile: dict | None = None) -> str:
    raw = _get(report, "executive_summary")
    if not raw:
        archetype = _get(report, "full_profile", "cross_axis", "archetype")
        decision  = _get(report, "full_profile", "final_decision")
        parts = []
        if archetype and not _is_system_phrase(archetype):
            parts.append(f"Investor archetype: {archetype}.")
        if decision and not _is_system_phrase(decision):
            parts.append(_clean(decision))
        if not parts:
            parts.append("Insufficient data to generate a complete executive summary.")
        raw = " ".join(parts)

    summary = _clean(raw)

    # Prepend primary objective (guard against double-prepend)
    fp = full_profile or _get(report, "full_profile", default={})
    if not summary.lower().startswith("primary objective"):
        objective = _infer_primary_objective(report, fp)
        summary = f"Primary objective: {objective}. {summary}"

    if meta.get("fallback_used"):
        summary = (
            "PROVISIONAL REPORT — Automated decision requires manual advisor validation. "
            + summary
        )

    return summary


def _build_profile_context(full_profile: dict) -> dict:
    ctx = _get(full_profile, "profile_context", default={})
    if not isinstance(ctx, dict):
        ctx = {}

    demo    = _get(ctx, "demographics",       default={})
    fin     = _get(ctx, "financial_snapshot", default={})
    flags   = _get(ctx, "flags",              default={})
    beh_raw = _lst(ctx, "behavioral_signals")
    contra  = _lst(ctx, "contradictions")

    # 3.1 Demographics
    demographics = {
        "income_type":           _get(demo, "income_type") or "Not reported",
        "employment_stability":  _get(demo, "employment_stability") or "Not reported",
        "dependents":            fin.get("dependents", "Not reported"),
        "city_tier":             _get(demo, "city_tier") or "Not reported",
    }

    # 3.2 Financial snapshot
    emi = fin.get("emi_amount")
    savings_raw = fin.get("emergency_months")
    financial_snapshot = {
        "emi":             f"₹{emi:,.0f}/month" if _num(emi) else "Not reported",
        "savings":         f"{savings_raw} months emergency fund" if _num(savings_raw) else "Not reported",
        "emergency_months": savings_raw if _num(savings_raw) else "Not reported",
        "emi_ratio":       f"{fin.get('emi_ratio', 'Not reported')}%"
                           if _num(fin.get("emi_ratio")) else "Not reported",
    }

    # 3.3 Life events
    life_events = []
    for ev in _lst(ctx, "life_events"):
        life_events.append({
            "event":          ev.get("type", "unknown"),
            "description":    _clean(ev.get("description", "")),
            "recency":        ev.get("recency", "unknown"),
            "emotional_weight": ev.get("emotional_weight", "unknown"),
        })

    # 3.4 Cultural context
    cultural_context = []
    for sig in _lst(ctx, "cultural_signals"):
        cultural_context.append({
            "type":          sig.get("type", "unknown"),
            "description":   _clean(sig.get("description", "")),
            "negotiability": sig.get("negotiability", "unknown"),
        })

    # 3.5 Behavioral signals + contradictions
    behavioral_signals = []
    for sig in beh_raw:
        behavioral_signals.append({
            "type":        sig.get("type", "unknown"),
            "description": _clean(sig.get("description", "")),
            "strength":    sig.get("strength", "unknown"),
        })

    contradictions = []
    for c in contra:
        contradictions.append({
            "dominant":   c.get("dominant_trait", "unknown"),
            "suppressed": c.get("suppressed_trait", "unknown"),
            "explanation": _clean(c.get("explanation", "")),
        })

    return {
        "demographics":       demographics,
        "financial_snapshot": financial_snapshot,
        "life_events":        life_events,
        "cultural_context":   cultural_context,
        "behavioral_signals": behavioral_signals,
        "contradictions":     contradictions,
        "flags":              flags,
    }


def _build_axis_assessment(full_profile: dict) -> dict:
    axis   = _get(full_profile, "axis_scores",    default={})
    cats   = _get(full_profile, "category_scores", default={})

    def _axis_entry(key: str, label: str) -> dict:
        score = axis.get(key)
        # Try to get reason from category_scores
        reason = ""
        for cat_key, cat_val in (cats.items() if isinstance(cats, dict) else []):
            if isinstance(cat_val, dict):
                cat_reason = cat_val.get("reason", "")
                # Match category to axis by keyword
                if key in cat_key.lower() or cat_key.lower() in key:
                    reason = _clean(cat_reason)
                    break
        if not reason:
            # Fallback: collect all reasons for this axis
            reasons = []
            for cat_val in (cats.values() if isinstance(cats, dict) else []):
                if isinstance(cat_val, dict) and cat_val.get("reason"):
                    reasons.append(_clean(cat_val["reason"]))
            reason = reasons[0] if reasons else ""

        return {
            "score":   score,
            "label":   _score_label(score),
            "insight": reason or f"{label} assessment based on available profile data.",
        }

    # Build per-axis with category-specific reasons
    def _axis_with_cat_reason(axis_key: str, cat_keys: list[str], label: str) -> dict:
        score = axis.get(axis_key)
        reason = ""
        for ck in cat_keys:
            cat = cats.get(ck, {}) if isinstance(cats, dict) else {}
            if isinstance(cat, dict) and cat.get("reason"):
                reason = _clean(cat["reason"])
                break
        return {
            "score":   score,
            "label":   _score_label(score),
            "insight": reason or f"{label} assessment based on available profile data.",
        }

    return {
        "risk": _axis_with_cat_reason(
            "risk",
            ["loss_aversion", "risk_behavior", "behavioral_risk"],
            "Risk tolerance",
        ),
        "cashflow": _axis_with_cat_reason(
            "cashflow",
            ["income_stability", "cashflow", "income"],
            "Cash flow stability",
        ),
        "obligation": _axis_with_cat_reason(
            "obligation",
            ["obligation", "debt_burden", "emi_burden"],
            "Obligation burden",
        ),
        "context": _axis_with_cat_reason(
            "context",
            ["financial_knowledge", "experience", "sophistication"],
            "Financial sophistication",
        ),
        "financial_capacity": _axis_with_cat_reason(
            "financial_capacity",
            ["emergency_preparedness", "savings", "capacity"],
            "Financial capacity",
        ),
    }


def _build_suitability(full_profile: dict) -> dict:
    cross = _get(full_profile, "cross_axis", default={})
    suit  = _get(cross, "suitability", default={})

    archetype   = _clean(_get(cross, "archetype"))
    binding     = cross.get("binding_constraint") or {}
    advisor_nar = _clean(_get(cross, "advisor_narrative"))

    classification = _clean(_get(suit, "classification"))
    if _is_system_phrase(classification):
        classification = "Suitability assessment requires manual advisor review."

    return {
        "archetype":            archetype or "Unclassified",
        "classification":       classification,
        "equity_range":         _get(suit, "equity_range") or "Not determined",
        "equity_ceiling_pct":   suit.get("equity_ceiling_pct"),
        "guidance":             _clean_list(_lst(suit, "guidance")),
        "binding_constraint":   {
            "type":        binding.get("type", "none"),
            "description": _clean(binding.get("description", "")),
            "actions":     _clean_list(binding.get("priority_actions", [])),
        } if binding else None,
        "advisor_narrative":    advisor_nar,
        "suitability_insights": _clean_list(_lst(full_profile, "suitability_insights")),
    }


# ---------------------------------------------------------------------------
# Risk filter — only factual financial risks, no advisory/meta statements
# ---------------------------------------------------------------------------

# Patterns that indicate a genuine financial risk statement
_RISK_ALLOW_PATTERNS = [
    r"\black of\b",
    r"\bno \w+\b",
    r"\binsufficient\b",
    r"\binstability\b",
    r"\bexposure to\b",
    r"\bhigh \w+ burden\b",
    r"\blow \w+ (fund|reserve|savings|capacity)\b",
    r"\birregular\b",
    r"\bvolatile\b",
    r"\buninsured\b",
    r"\bno emergency\b",
    r"\bno savings\b",
    r"\bno insurance\b",
    r"\bdebt burden\b",
    r"\bemi burden\b",
    r"\bnear.term obligation\b",
]

# Patterns that mark advisory insights, not risks — must be excluded from risks section
_RISK_EXCLUDE_PATTERNS = [
    r"\bprioritize\b",
    r"\brecommend\b",
    r"\bapproach\b",
    r"\beducation.first\b",
    r"\bprofile reliability\b",
    r"\btreat.*provisional\b",
    r"\bconsolidate\b",
    r"\bdo not push\b",
    r"\bgradual\b",
    r"\bsuitable for\b",
    r"\bdecision confidence\b",
    r"\bconfidence is low\b",
    r"\bprovision\b",
]

# Language normalization for known bad phrases
_RISK_LANGUAGE_MAP = [
    (r"profile reliability is low",
     "Data completeness is limited — verify before execution"),
    (r"reliability.*low",
     "Data completeness is limited — verify before execution"),
    (r"stated risk preference does not reflect",
     "Stated risk preference may not reflect actual financial capacity"),
]


def _is_valid_risk(text: str) -> bool:
    """Return True only if text is a factual financial risk statement."""
    t = text.lower()
    # Must not be an advisory/meta statement
    if any(re.search(p, t) for p in _RISK_EXCLUDE_PATTERNS):
        return False
    # Must match at least one risk pattern
    return any(re.search(p, t) for p in _RISK_ALLOW_PATTERNS)


def _normalize_risk_language(text: str) -> str:
    for pattern, replacement in _RISK_LANGUAGE_MAP:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _build_risk_bias(report: dict, full_profile: dict) -> dict:
    ctx  = _get(full_profile, "profile_context", default={})
    fin  = _get(ctx, "financial_snapshot", default={})
    demo = _get(ctx, "demographics", default={})

    # --- Key risks: factual financial risks only ---
    raw_risks = _lst(report, "key_risks")
    risks = []
    for item in raw_risks:
        text = _normalize_risk_language(_clean(str(item)))
        if text and _is_valid_risk(text):
            risks.append(text)

    # Deduplicate (case-insensitive)
    seen = set()
    deduped = []
    for r in risks:
        key = r.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    risks = deduped

    # Fallback: derive from profile data when no valid risks found
    if not risks:
        em = _num(fin.get("emergency_months"), 0)
        if em < 3:
            risks.append("Insufficient emergency fund — less than 3 months of expenses covered")
        if demo.get("income_type") in ("gig", "business", "freelance"):
            risks.append("Income instability — irregular cash flow limits consistent investment capacity")
        obligation = _num(_get(full_profile, "axis_scores", "obligation"), 0)
        if obligation > 60:
            risks.append("High obligation burden — future financial commitments may limit investable surplus")
        emi_ratio = _num(fin.get("emi_ratio"), 0)
        if emi_ratio > 40:
            risks.append("High EMI burden — debt servicing exceeds 40% of income")
        if not risks:
            risks.append("Insufficient data to enumerate specific risks — advisor review required")

    # --- Bias summary ---
    raw_bias = _get(report, "bias_summary")
    bias = _normalize_risk_language(_clean(raw_bias)) \
        if raw_bias and not _is_system_phrase(raw_bias) else ""

    # --- Behavioral bias flags ---
    flags = _get(ctx, "flags", default={})
    bias_flags = []
    if flags.get("recency_bias_risk"):
        bias_flags.append(
            "Recency bias — recent market performance may be distorting risk perception"
        )
    if flags.get("peer_driven"):
        bias_flags.append(
            "Peer-influenced decisions — choices may not reflect personal risk capacity"
        )
    if flags.get("grief_state"):
        bias_flags.append(
            "Grief state — current risk aversion may be temporary, not baseline behavior"
        )

    return {
        "key_risks":    risks,
        "bias_summary": bias or "No specific behavioral bias identified.",
        "bias_flags":   bias_flags,
    }


def _build_strategy(report: dict, full_profile: dict, meta: dict) -> dict:
    decision = _get(full_profile, "decision", default={})
    strategy = _get(decision, "strategy", default={})

    equity_pct  = strategy.get("equity_pct")
    debt_pct    = strategy.get("debt_pct")
    instrument  = _clean(_get(strategy, "primary_instrument"))
    first_step  = _clean(_get(strategy, "first_step"))
    sip         = strategy.get("sip_recommended", False)

    temporal = _get(decision, "temporal_strategy", default={})
    is_temp  = temporal.get("is_temporary", False)
    trigger  = _clean(_get(temporal, "reassessment_trigger"))
    timeline = _clean(_get(temporal, "reassessment_timeline"))
    shift    = _clean(_get(temporal, "expected_shift"))

    fallback_used = meta.get("fallback_used") or decision.get("fallback_used", False)

    result = {
        "equity_pct":          equity_pct,
        "debt_pct":            debt_pct,
        "primary_instrument":  instrument or "To be determined by advisor",
        "sip_recommended":     sip,
        "first_step":          first_step or "Consult advisor for first step.",
        "temporal_strategy": {
            "is_temporary":           is_temp,
            "reassessment_trigger":   trigger or "Standard annual review",
            "reassessment_timeline":  timeline or "12 months",
            "expected_shift":         shift or "None anticipated",
        },
    }

    if fallback_used:
        result["advisory_note"] = (
            "Manual advisor validation required before execution. "
            "This strategy was generated under data limitations and must be reviewed."
        )

    return result


# ---------------------------------------------------------------------------
# Action normalization — dedup, prioritize, format
# ---------------------------------------------------------------------------

# Markers that disqualify a string from being an executable action
_NON_ACTION_MARKERS = [
    "note:", "disclaimer", "subject to", "this report", "generated by",
    "ai-based", "not for client", "advisor review", "manual validation",
    "requires verification", "data limitation",
]

# Verb prefixes that confirm something is an executable action
_ACTION_VERB_PREFIXES = (
    "build", "start", "open", "invest", "reduce", "prepay", "purchase",
    "review", "consolidate", "increase", "decrease", "allocate", "set up",
    "create", "establish", "transfer", "close", "switch", "add", "remove",
    "park", "maintain", "ensure", "obtain", "get", "buy", "sell",
)

# Semantic intent clusters — actions sharing a cluster key are near-duplicates.
# The first match wins; subsequent matches in the same cluster are dropped.
_ACTION_INTENT_CLUSTERS = [
    ("emergency_fund",   ["emergency fund", "contingency fund", "liquid reserve", "3 month", "6 month"]),
    ("sip_start",        ["start sip", "start a sip", "begin sip", "initiate sip", "set up sip"]),
    ("debt_reduce",      ["reduce debt", "prepay", "pay off", "clear debt", "repay"]),
    ("insurance",        ["insurance", "health cover", "life cover", "term plan"]),
    ("equity_invest",    ["equity", "mutual fund", "flexi-cap", "index fund", "large-cap"]),
    ("liquid_park",      ["liquid fund", "park funds", "park in liquid"]),
    ("budget_review",    ["budget", "expense review", "track expense"]),
    ("advisor_consult",  ["consult advisor", "meet advisor", "advisor review"]),
]

# Time-horizon classification rules — checked in order, first match wins
_ACTION_HORIZON_RULES = [
    # Immediate (0–30 days): urgent financial safety actions
    ("immediate", [
        "emergency fund", "contingency", "liquid reserve",
        "insurance", "health cover", "term plan",
        "park funds", "liquid fund",
        "stop sip", "pause", "halt",
    ]),
    # Near-term (1–3 months): structural fixes
    ("near_term", [
        "reduce debt", "prepay", "pay off", "repay",
        "start sip", "begin sip", "set up sip",
        "budget", "track expense",
        "open account", "nominee",
    ]),
    # Medium-term (3–6 months): growth / optimization
    ("medium_term", [
        "equity", "mutual fund", "flexi-cap", "index fund",
        "increase sip", "step up",
        "review portfolio", "rebalance",
        "consult advisor",
    ]),
]

# Canonical verb normalization — maps common synonyms to a single preferred verb
_VERB_NORMALIZATIONS = [
    (r"^(create|establish|set up|build up|maintain)\s+(an?\s+)?emergency fund",
     "Build an emergency fund"),
    (r"^(create|establish|set up)\s+(an?\s+)?sip",
     "Start a SIP"),
    (r"^(pay off|clear|repay)\s+(the\s+)?debt",
     "Reduce debt"),
    (r"^(get|purchase|take)\s+(an?\s+)?(health|life|term)\s+(insurance|cover|plan)",
     "Obtain health/life insurance"),
]


def _normalize_action_verb(text: str) -> str:
    """Apply canonical verb normalization to an action string."""
    for pattern, replacement in _VERB_NORMALIZATIONS:
        new = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        if new != text:
            return new
    # Ensure first letter is uppercase
    return text[0].upper() + text[1:] if text else text


def _action_intent_cluster(text: str) -> str | None:
    """Return the intent cluster key for an action, or None if no match."""
    lower = text.lower()
    for cluster_key, keywords in _ACTION_INTENT_CLUSTERS:
        if any(kw in lower for kw in keywords):
            return cluster_key
    return None


def _action_horizon(text: str) -> str:
    """Classify action into time horizon bucket."""
    lower = text.lower()
    for horizon, keywords in _ACTION_HORIZON_RULES:
        if any(kw in lower for kw in keywords):
            return horizon
    return "near_term"  # default


def _build_actions(report: dict) -> dict:
    """
    Build a structured action plan with three time-horizon buckets.

    Returns:
        {
            "immediate":   [...],   # 0–30 days
            "near_term":   [...],   # 1–3 months
            "medium_term": [...],   # 3–6 months
        }
    """
    raw = _lst(report, "recommended_actions")
    candidates = []

    for item in raw:
        text = _clean(str(item)).strip()
        if not text or _is_system_phrase(text):
            continue
        lower = text.lower()
        if any(m in lower for m in _NON_ACTION_MARKERS):
            continue
        starts_with_verb = lower.startswith(_ACTION_VERB_PREFIXES)
        has_measurable   = any(w in lower for w in [
            "%", "month", "fund", "sip", "emi", "insurance",
            "₹", "rs", "amount", "step",
        ])
        if not starts_with_verb and not has_measurable:
            continue
        candidates.append(_normalize_action_verb(text))

    # Semantic dedup — one action per intent cluster
    seen_clusters: set[str] = set()
    seen_keys: set[str] = set()
    deduped = []
    for action in candidates:
        cluster = _action_intent_cluster(action)
        # Strip amounts/numbers for key comparison
        key = re.sub(r"rs\.?\s*[\d,]+(/month)?|\d+\s*%|\d+[\s-]*months?|[^\w\s]", "", action.lower())
        key = re.sub(r"\s+", " ", key).strip()
        if cluster and cluster in seen_clusters:
            continue
        if key in seen_keys:
            continue
        if cluster:
            seen_clusters.add(cluster)
        seen_keys.add(key)
        deduped.append(action)

    # Bucket by time horizon
    buckets: dict[str, list[str]] = {"immediate": [], "near_term": [], "medium_term": []}
    for action in deduped:
        buckets[_action_horizon(action)].append(action)

    return buckets


# ---------------------------------------------------------------------------
# Suitability insights — advisory guidance, separated from risks
# ---------------------------------------------------------------------------

# Patterns that belong in insights (advisory/guidance), not risks
_INSIGHT_PATTERNS = [
    r"\bprioritize\b",
    r"\beducation.first\b",
    r"\bconsolidate\b",
    r"\bdo not\b",
    r"\bavoid\b",
    r"\bsuitable for\b",
    r"\breassess\b",
    r"\bgradual\b",
    r"\bconfidence.*low\b",
    r"\bprovisional\b",
    r"\bliquidity\b",
    r"\block.in\b",
    r"\bcomplex instrument\b",
    r"\bfragment\b",
]

_INSIGHT_LANGUAGE_MAP = [
    (r"profile reliability is low",
     "Data completeness is limited — verify before execution"),
    (r"reliability.*low",
     "Data completeness is limited — verify before execution"),
    (r"treat.*provisional",
     "Treat recommendation as provisional — additional data required"),
    (r"decision confidence is low.*",
     "Data completeness is limited — verify before execution"),
]


def _normalize_insight_language(text: str) -> str:
    """Apply language normalization — stop after first match to avoid chained replacements."""
    for pattern, replacement in _INSIGHT_LANGUAGE_MAP:
        new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        if new_text != text:
            return new_text.strip()
    return text


def _build_suitability_insights(full_profile: dict, report: dict) -> list[str]:
    """
    Collect advisory guidance items — things that belong in insights, not risks.
    Sources: suitability_insights from cross_axis + filtered items from key_risks.
    """
    raw_insights = _lst(full_profile, "suitability_insights")

    # Also pull any items from key_risks that are actually advisory guidance
    raw_risks = _lst(report, "key_risks")
    for item in raw_risks:
        text = _clean(str(item))
        lower = text.lower()
        if any(re.search(p, lower) for p in _INSIGHT_PATTERNS):
            raw_insights.append(text)

    # Clean, normalize, deduplicate
    # Use 2-word prefix key so near-synonyms like "Education-first approach recommended"
    # and "Education-first approach required" collapse to the same entry.
    seen_keys: list[str] = []
    insights = []
    for item in raw_insights:
        text = _normalize_insight_language(_clean(str(item)))
        if not text or _is_system_phrase(text):
            continue
        words = re.sub(r"[^\w\s]", "", text.lower()).split()
        key = " ".join(words[:2])
        # Skip if any existing key starts with this key or vice versa
        if any(k.startswith(key) or key.startswith(k) for k in seen_keys):
            continue
        seen_keys.append(key)
        insights.append(text)

    return insights


def _build_restrictions(report: dict) -> list[str]:
    raw = _lst(report, "do_not_recommend")
    result = []
    for item in raw:
        text = _clean(str(item))
        if text and not _is_system_phrase(text):
            result.append(text)
    return result


def _build_assumptions(report: dict, full_profile: dict, meta: dict) -> dict:
    note = _clean(_get(report, "completeness_note"))
    debug = _get(full_profile, "debug", default={})
    missing = debug.get("missing_fields", []) if isinstance(debug, dict) else []
    confidence_score = full_profile.get("confidence_score", 0)
    data_completeness = full_profile.get("data_completeness", 0)

    return {
        "completeness_note":  note or "Profile completeness assessment not available.",
        "missing_fields":     missing,
        "confidence_score":   confidence_score,
        "data_completeness":  data_completeness,
        "meta_confidence":    meta.get("confidence", "low"),
        "assumptions": [
            "All data is self-reported and has not been independently verified.",
            "Income and expense figures are based on investor-provided information.",
            "Behavioral signals are inferred from narrative and may not capture full context.",
            "This profile reflects the investor's situation at the time of assessment.",
        ],
    }


def _build_disclosures() -> dict:
    return {
        "sebi_compliance": (
            "This report is prepared in accordance with SEBI (Investment Advisers) "
            "Regulations, 2013 and subsequent amendments. The recommendations herein "
            "are based on the investor's risk profile and suitability assessment."
        ),
        "ai_usage": (
            "This report was generated with the assistance of an AI-based profiling system "
            "(InvestorDNA v1). The AI system extracts and structures investor signals from "
            "natural language input. All outputs are subject to mandatory advisor review "
            "before being presented to or acted upon by the client."
        ),
        "data_retention": (
            "Client data and profile records are retained for a minimum of 5 years "
            "in accordance with SEBI regulatory requirements."
        ),
        "methodology_version": "InvestorDNA v1 — Multi-axis behavioral profiling engine",
        "disclaimer": (
            "This document is intended for use by SEBI-registered Investment Advisers only. "
            "It does not constitute investment advice and must not be shared with clients "
            "without advisor review and sign-off. Past performance is not indicative of "
            "future results. Investments are subject to market risk."
        ),
    }


def _build_appendix(full_profile: dict) -> dict:
    debug = full_profile.get("debug", {})
    trace = full_profile.get("trace_validation", {})
    constraint = full_profile.get("constraint_report", {})

    return {
        "label": "Advisor Technical Appendix — Not for client-facing use",
        "debug":             debug if isinstance(debug, dict) else {},
        "trace_validation":  trace if isinstance(trace, dict) else {},
        "constraint_report": constraint if isinstance(constraint, dict) else {},
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_pdf_payload(pipeline_output: dict, client_id: str | None = None) -> dict:
    """
    Convert raw pipeline JSON → structured PDF-ready payload.

    Args:
        pipeline_output: The full output from run_pipeline() — {status, report, meta}
        client_id: Optional client identifier; auto-generated if missing.

    Returns:
        Structured dict matching the 12-section PDF schema.
    """
    report       = pipeline_output.get("report", {})
    meta         = pipeline_output.get("meta", {})
    full_profile = report.get("full_profile", {})

    # If report doesn't have full_profile at top level, it may be nested differently
    # Handle both flat and nested structures
    if not full_profile and isinstance(report, dict):
        # Try to reconstruct from report keys that map to pipeline_output keys
        full_profile = {
            "profile_context":   report.get("profile_context", {}),
            "axis_scores":       report.get("axis_scores", {}),
            "category_scores":   report.get("category_scores", {}),
            "cross_axis":        report.get("cross_axis", {}),
            "decision":          report.get("decision", {}),
            "debug":             report.get("debug", {}),
            "trace_validation":  report.get("trace_validation", {}),
            "constraint_report": report.get("constraint_report", {}),
            "final_decision":    report.get("final_decision", ""),
            "confidence_score":  report.get("confidence_score", 0),
            "data_completeness": report.get("data_completeness", 0),
            "suitability_insights": report.get("suitability_insights", []),
        }

    risks    = _build_risk_bias(report, full_profile)
    insights = _build_suitability_insights(full_profile, report)
    actions  = _build_actions(report)

    # Cross-section redundancy filter:
    # Remove from insights any item whose meaning already appears in risks,
    # and vice versa. Use 2-word prefix key for fuzzy matching.
    def _prefix_key(text: str) -> str:
        words = re.sub(r"[^\w\s]", "", text.lower()).split()
        return " ".join(words[:2])

    risk_keys    = {_prefix_key(r) for r in risks["key_risks"]}
    insight_keys = {_prefix_key(i) for i in insights}

    # Drop insights that duplicate a risk
    insights = [i for i in insights
                if not any(ik.startswith(rk) or rk.startswith(ik)
                           for ik in [_prefix_key(i)] for rk in risk_keys)]

    # Drop risks that duplicate an insight (shouldn't happen after filtering, but safety net)
    clean_risks = [r for r in risks["key_risks"]
                   if not any(rk.startswith(ik) or ik.startswith(rk)
                              for rk in [_prefix_key(r)] for ik in insight_keys)]
    risks = {**risks, "key_risks": clean_risks}

    # Flatten all action texts for cross-section check
    all_action_texts = [
        a.lower() for bucket in actions.values() for a in bucket
    ]
    # Remove insights that are already expressed as actions
    insights = [i for i in insights
                if not any(_prefix_key(i) in a for a in all_action_texts)]

    return {
        "cover_page":           _build_cover_page(report, meta, client_id),
        "executive_summary":    _build_executive_summary(report, meta, full_profile),
        "profile_context":      _build_profile_context(full_profile),
        "axis_assessment":      _build_axis_assessment(full_profile),
        "suitability":          _build_suitability(full_profile),
        "suitability_insights": insights,
        "risk_bias":            risks,
        "strategy":             _build_strategy(report, full_profile, meta),
        "actions":              actions,
        "restrictions":         _build_restrictions(report),
        "assumptions":          _build_assumptions(report, full_profile, meta),
        "disclosures":          _build_disclosures(),
        "appendix":             _build_appendix(full_profile),
    }


# ---------------------------------------------------------------------------
# PDF renderer — HTML → PDF via weasyprint (recommended) or reportlab fallback
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  body {{ font-family: Arial, sans-serif; font-size: 11pt; color: #1a1a1a; margin: 40px; }}
  h1   {{ font-size: 18pt; color: #003366; border-bottom: 2px solid #003366; padding-bottom: 6px; }}
  h2   {{ font-size: 13pt; color: #003366; margin-top: 24px; }}
  h3   {{ font-size: 11pt; color: #444; margin-top: 16px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
  th   {{ background: #e8eef4; font-weight: bold; }}
  .provisional {{ background: #fff3cd; border: 1px solid #ffc107; padding: 10px; margin: 10px 0; }}
  .disclaimer  {{ font-size: 9pt; color: #666; border-top: 1px solid #ccc; margin-top: 30px; padding-top: 10px; }}
  .appendix    {{ background: #f8f8f8; border: 1px solid #ddd; padding: 12px; font-size: 9pt; }}
  ul {{ margin: 6px 0; padding-left: 20px; }}
  li {{ margin: 3px 0; }}
  .score-high   {{ color: #1a7a1a; font-weight: bold; }}
  .score-medium {{ color: #b36b00; font-weight: bold; }}
  .score-low    {{ color: #cc0000; font-weight: bold; }}
</style>
</head>
<body>

<!-- COVER PAGE -->
<h1>Investor Suitability Profile</h1>
<table>
  <tr><th>Client ID</th><td>{client_id}</td><th>Date</th><td>{date}</td></tr>
  <tr><th>Version</th><td>{version}</td><th>Status</th><td>{status}</td></tr>
</table>
<p><em>{ai_disclosure}</em></p>

{provisional_banner}

<!-- EXECUTIVE SUMMARY -->
<h2>Executive Summary</h2>
<p>{executive_summary}</p>

<!-- PROFILE CONTEXT -->
<h2>Profile Context</h2>
<h3>Demographics</h3>
<table>
  <tr><th>Income Type</th><td>{income_type}</td><th>Employment</th><td>{employment_stability}</td></tr>
  <tr><th>Dependents</th><td>{dependents}</td><th>City Tier</th><td>{city_tier}</td></tr>
</table>
<h3>Financial Snapshot</h3>
<table>
  <tr><th>EMI</th><td>{emi}</td><th>Emergency Fund</th><td>{savings}</td></tr>
  <tr><th>EMI Ratio</th><td>{emi_ratio}</td><td colspan="2"></td></tr>
</table>
{life_events_html}
{cultural_html}
{behavioral_html}

<!-- 4-AXIS ASSESSMENT -->
<h2>4-Axis Assessment</h2>
<table>
  <tr><th>Axis</th><th>Score</th><th>Level</th><th>Insight</th></tr>
  {axis_rows}
</table>

<!-- CROSS-AXIS SUITABILITY -->
<h2>Cross-Axis Suitability</h2>
<table>
  <tr><th>Archetype</th><td colspan="3">{archetype}</td></tr>
  <tr><th>Equity Range</th><td>{equity_range}</td><th>Status</th><td>{suitability_status}</td></tr>
</table>
<p>{suitability_classification}</p>
{binding_constraint_html}
{suitability_insights_html}

<!-- RISK & BIAS ANALYSIS -->
<h2>Risk &amp; Bias Analysis</h2>
<h3>Key Risks</h3>
<ul>{risks_html}</ul>
<h3>Behavioral Bias</h3>
<p>{bias_summary}</p>
{bias_flags_html}

<!-- RECOMMENDED STRATEGY -->
<h2>Recommended Strategy</h2>
<table>
  <tr><th>Equity</th><td>{equity_pct}%</td><th>Debt</th><td>{debt_pct}%</td></tr>
  <tr><th>Primary Instrument</th><td>{instrument}</td><th>SIP</th><td>{sip}</td></tr>
  <tr><th>First Step</th><td colspan="3">{first_step}</td></tr>
</table>
{strategy_note_html}
<h3>Temporal Strategy</h3>
<table>
  <tr><th>Temporary?</th><td>{is_temporary}</td><th>Timeline</th><td>{timeline}</td></tr>
  <tr><th>Trigger</th><td>{trigger}</td><th>Expected Shift</th><td>{expected_shift}</td></tr>
</table>

<!-- ACTION PLAN -->
<h2>Action Plan</h2>
<ul>{actions_html}</ul>

<!-- DO NOT RECOMMEND -->
<h2>Do Not Recommend</h2>
<ul>{restrictions_html}</ul>

<!-- ASSUMPTIONS & LIMITATIONS -->
<h2>Assumptions &amp; Limitations</h2>
<p>{completeness_note}</p>
<table>
  <tr><th>Data Completeness</th><td>{data_completeness}%</td><th>Confidence Score</th><td>{confidence_score}</td></tr>
</table>
{missing_fields_html}
<h3>Assumptions</h3>
<ul>{assumptions_html}</ul>

<!-- REGULATORY DISCLOSURES -->
<h2>Regulatory Disclosures</h2>
<p><strong>SEBI Compliance:</strong> {sebi_compliance}</p>
<p><strong>AI Usage:</strong> {ai_usage}</p>
<p><strong>Data Retention:</strong> {data_retention}</p>
<p><strong>Methodology:</strong> {methodology_version}</p>
<div class="disclaimer">{disclaimer}</div>

<!-- APPENDIX -->
<h2>Appendix — Advisor Technical Reference</h2>
<div class="appendix">
  <p><strong>{appendix_label}</strong></p>
  <pre>{appendix_json}</pre>
</div>

</body>
</html>"""


def _li(items: list) -> str:
    if not items:
        return "<li>None</li>"
    return "".join(f"<li>{i}</li>" for i in items)


def _score_class(label: str) -> str:
    return {"high": "score-high", "medium": "score-medium", "low": "score-low"}.get(label, "")


def _build_html(payload: dict) -> str:
    import json as _json

    cover    = payload["cover_page"]
    ctx      = payload["profile_context"]
    axis     = payload["axis_assessment"]
    suit     = payload["suitability"]
    rb       = payload["risk_bias"]
    strat    = payload["strategy"]
    assum    = payload["assumptions"]
    disc     = payload["disclosures"]
    appendix = payload["appendix"]

    demo = ctx.get("demographics", {})
    fin  = ctx.get("financial_snapshot", {})

    # Provisional banner
    provisional_banner = ""
    if cover.get("status") == "Provisional":
        provisional_banner = (
            '<div class="provisional"><strong>PROVISIONAL REPORT</strong> — '
            'Automated decision requires manual advisor validation before use.</div>'
        )

    # Life events
    life_events = ctx.get("life_events", [])
    if life_events:
        rows = "".join(
            f"<tr><td>{e['event']}</td><td>{e['description']}</td>"
            f"<td>{e['recency']}</td><td>{e['emotional_weight']}</td></tr>"
            for e in life_events
        )
        life_events_html = (
            "<h3>Life Events</h3>"
            "<table><tr><th>Event</th><th>Description</th><th>Recency</th><th>Weight</th></tr>"
            + rows + "</table>"
        )
    else:
        life_events_html = "<h3>Life Events</h3><p>None reported.</p>"

    # Cultural context
    cultural = ctx.get("cultural_context", [])
    if cultural:
        rows = "".join(
            f"<tr><td>{s['type']}</td><td>{s['description']}</td><td>{s['negotiability']}</td></tr>"
            for s in cultural
        )
        cultural_html = (
            "<h3>Cultural Context</h3>"
            "<table><tr><th>Type</th><th>Description</th><th>Negotiability</th></tr>"
            + rows + "</table>"
        )
    else:
        cultural_html = "<h3>Cultural Context</h3><p>None detected.</p>"

    # Behavioral signals
    behavioral = ctx.get("behavioral_signals", [])
    if behavioral:
        rows = "".join(
            f"<tr><td>{s['type']}</td><td>{s['description']}</td><td>{s['strength']}</td></tr>"
            for s in behavioral
        )
        behavioral_html = (
            "<h3>Behavioral Signals</h3>"
            "<table><tr><th>Type</th><th>Description</th><th>Strength</th></tr>"
            + rows + "</table>"
        )
    else:
        behavioral_html = "<h3>Behavioral Signals</h3><p>None detected.</p>"

    # Axis rows
    axis_names = {
        "risk": "Risk Tolerance",
        "cashflow": "Cash Flow Stability",
        "obligation": "Obligation Burden",
        "context": "Financial Sophistication",
        "financial_capacity": "Financial Capacity",
    }
    axis_rows = ""
    for key, label in axis_names.items():
        entry = axis.get(key, {})
        score = entry.get("score", "—")
        lbl   = entry.get("label", "unknown")
        insight = entry.get("insight", "")
        css = _score_class(lbl)
        axis_rows += (
            f"<tr><td>{label}</td>"
            f"<td class='{css}'>{score}</td>"
            f"<td class='{css}'>{lbl.upper()}</td>"
            f"<td>{insight}</td></tr>"
        )

    # Binding constraint
    bc = suit.get("binding_constraint")
    if bc and bc.get("type") != "none":
        bc_actions = _li(bc.get("actions", []))
        binding_constraint_html = (
            f"<h3>Binding Constraint: {bc['type'].replace('_', ' ').title()}</h3>"
            f"<p>{bc.get('description', '')}</p>"
            f"<ul>{bc_actions}</ul>"
        )
    else:
        binding_constraint_html = ""

    # Suitability insights
    insights = suit.get("suitability_insights", [])
    suitability_insights_html = (
        "<h3>Suitability Insights</h3><ul>" + _li(insights) + "</ul>"
        if insights else ""
    )

    # Bias flags
    bias_flags = rb.get("bias_flags", [])
    bias_flags_html = (
        "<ul>" + _li(bias_flags) + "</ul>" if bias_flags else ""
    )

    # Strategy note
    advisory_note = strat.get("advisory_note", "")
    strategy_note_html = (
        f'<div class="provisional"><strong>Advisory Note:</strong> {advisory_note}</div>'
        if advisory_note else ""
    )

    # Missing fields
    missing = assum.get("missing_fields", [])
    missing_fields_html = (
        "<p><strong>Missing fields:</strong> " + ", ".join(missing) + "</p>"
        if missing else ""
    )

    temporal = strat.get("temporal_strategy", {})

    # Appendix — truncate to avoid massive PDFs
    appendix_data = {
        "debug":            appendix.get("debug", {}),
        "trace_validation": appendix.get("trace_validation", {}),
    }
    appendix_json = _json.dumps(appendix_data, indent=2, default=str)
    if len(appendix_json) > 8000:
        appendix_json = appendix_json[:8000] + "\n... [truncated — see full debug output]"

    return _HTML_TEMPLATE.format(
        client_id=cover["client_id"],
        date=cover["date"],
        version=cover["version"],
        status=cover["status"],
        ai_disclosure=cover["ai_disclosure"],
        provisional_banner=provisional_banner,
        executive_summary=payload["executive_summary"],
        income_type=demo.get("income_type", "—"),
        employment_stability=demo.get("employment_stability", "—"),
        dependents=demo.get("dependents", "—"),
        city_tier=demo.get("city_tier", "—"),
        emi=fin.get("emi", "—"),
        savings=fin.get("savings", "—"),
        emi_ratio=fin.get("emi_ratio", "—"),
        life_events_html=life_events_html,
        cultural_html=cultural_html,
        behavioral_html=behavioral_html,
        axis_rows=axis_rows,
        archetype=suit.get("archetype", "—"),
        equity_range=suit.get("equity_range", "—"),
        suitability_status=cover["status"],
        suitability_classification=suit.get("classification", "—"),
        binding_constraint_html=binding_constraint_html,
        suitability_insights_html=suitability_insights_html,
        risks_html=_li(rb.get("key_risks", [])),
        bias_summary=rb.get("bias_summary", "—"),
        bias_flags_html=bias_flags_html,
        equity_pct=strat.get("equity_pct", "—"),
        debt_pct=strat.get("debt_pct", "—"),
        instrument=strat.get("primary_instrument", "—"),
        sip="Yes" if strat.get("sip_recommended") else "No",
        first_step=strat.get("first_step", "—"),
        strategy_note_html=strategy_note_html,
        is_temporary="Yes" if temporal.get("is_temporary") else "No",
        timeline=temporal.get("reassessment_timeline", "—"),
        trigger=temporal.get("reassessment_trigger", "—"),
        expected_shift=temporal.get("expected_shift", "—"),
        actions_html=_li(payload.get("actions", [])),
        restrictions_html=_li(payload.get("restrictions", [])),
        completeness_note=assum.get("completeness_note", "—"),
        data_completeness=assum.get("data_completeness", "—"),
        confidence_score=assum.get("confidence_score", "—"),
        missing_fields_html=missing_fields_html,
        assumptions_html=_li(assum.get("assumptions", [])),
        sebi_compliance=disc["sebi_compliance"],
        ai_usage=disc["ai_usage"],
        data_retention=disc["data_retention"],
        methodology_version=disc["methodology_version"],
        disclaimer=disc["disclaimer"],
        appendix_label=appendix.get("label", "Advisor Technical Appendix"),
        appendix_json=appendix_json,
    )


def render_pdf(payload: dict, output_path: str = "investor_report.pdf") -> str:
    """
    Render PDF payload → PDF file using reportlab (pure Python, no system deps).

    Args:
        payload:     Output from build_pdf_payload()
        output_path: Destination file path

    Returns:
        Absolute path to the generated PDF.
    """
    import os
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether,
    )

    # -----------------------------------------------------------------------
    # Document setup
    # -----------------------------------------------------------------------
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=22*mm, rightMargin=22*mm,
        topMargin=20*mm, bottomMargin=20*mm,
        title="Investor Suitability Profile",
        author="InvestorDNA v1",
    )

    # -----------------------------------------------------------------------
    # Style definitions
    # -----------------------------------------------------------------------
    base = getSampleStyleSheet()

    S = {
        "h1": ParagraphStyle("h1", parent=base["Heading1"],
                             fontSize=18, textColor=colors.HexColor("#003366"),
                             spaceAfter=4, spaceBefore=0),
        "h2": ParagraphStyle("h2", parent=base["Heading2"],
                             fontSize=13, textColor=colors.HexColor("#003366"),
                             spaceBefore=14, spaceAfter=4),
        "h3": ParagraphStyle("h3", parent=base["Heading3"],
                             fontSize=11, textColor=colors.HexColor("#444444"),
                             spaceBefore=8, spaceAfter=2),
        "body": ParagraphStyle("body", parent=base["Normal"],
                               fontSize=9.5, leading=14, spaceAfter=4),
        "bullet": ParagraphStyle("bullet", parent=base["Normal"],
                                 fontSize=9.5, leading=14, spaceAfter=2,
                                 leftIndent=12, bulletIndent=0),
        "warn": ParagraphStyle("warn", parent=base["Normal"],
                               fontSize=9.5, leading=14, spaceAfter=4,
                               backColor=colors.HexColor("#fff3cd"),
                               borderColor=colors.HexColor("#ffc107"),
                               borderWidth=1, borderPadding=6),
        "small": ParagraphStyle("small", parent=base["Normal"],
                                fontSize=8, textColor=colors.HexColor("#666666"),
                                leading=11, spaceAfter=4),
        "meta": ParagraphStyle("meta", parent=base["Normal"],
                               fontSize=9, textColor=colors.HexColor("#333333"),
                               leading=13),
    }

    # -----------------------------------------------------------------------
    # Builder helpers
    # -----------------------------------------------------------------------
    story = []

    def h1(text):
        story.append(Paragraph(text, S["h1"]))
        story.append(HRFlowable(width="100%", thickness=2,
                                color=colors.HexColor("#003366"), spaceAfter=6))

    def h2(text):
        story.append(Paragraph(text, S["h2"]))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#cccccc"), spaceAfter=3))

    def h3(text):
        story.append(Paragraph(text, S["h3"]))

    def p(text):
        if text:
            story.append(Paragraph(str(text), S["body"]))

    def bullets(items: list, fallback: str = "None"):
        if not items:
            story.append(Paragraph(f"\u2022  {fallback}", S["bullet"]))
        else:
            for item in items:
                story.append(Paragraph(f"\u2022  {item}", S["bullet"]))

    def warn(text):
        story.append(Paragraph(f"\u26a0  {text}", S["warn"]))
        story.append(Spacer(1, 2*mm))

    def gap(n=4):
        story.append(Spacer(1, n*mm))

    def kv_table(rows: list[tuple], col_widths=None):
        """Two-column key-value table."""
        w = col_widths or [55*mm, 105*mm]
        data = [[Paragraph(f"<b>{k}</b>", S["meta"]),
                 Paragraph(str(v), S["meta"])] for k, v in rows]
        t = Table(data, colWidths=w)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8eef4")),
            ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("VALIGN",     (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 3*mm))

    def grid_table(headers: list, rows: list, col_widths=None):
        """Multi-column data table with header row."""
        data = [[Paragraph(f"<b>{h}</b>", S["meta"]) for h in headers]]
        for row in rows:
            data.append([Paragraph(str(c), S["meta"]) for c in row])
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#003366")),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f4f7fb")]),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ]))
        story.append(t)
        story.append(Spacer(1, 3*mm))

    # -----------------------------------------------------------------------
    # Unpack payload
    # -----------------------------------------------------------------------
    cover  = payload["cover_page"]
    ctx    = payload["profile_context"]
    axis   = payload["axis_assessment"]
    suit   = payload["suitability"]
    rb     = payload["risk_bias"]
    strat  = payload["strategy"]
    assum  = payload["assumptions"]
    disc   = payload["disclosures"]
    app    = payload["appendix"]
    demo   = ctx.get("demographics", {})
    fin    = ctx.get("financial_snapshot", {})
    temp   = strat.get("temporal_strategy", {})

    # -----------------------------------------------------------------------
    # COVER PAGE
    # -----------------------------------------------------------------------
    h1("Investor Suitability Profile")
    kv_table([
        ("Client ID",  cover["client_id"]),
        ("Date",       cover["date"]),
        ("Version",    cover["version"]),
        ("Status",     cover["status"]),
    ])
    p(cover["ai_disclosure"])

    if cover["status"] == "Provisional":
        warn("PROVISIONAL REPORT — This report was generated under data limitations. "
             "Manual advisor validation is required before use.")
    gap()

    # -----------------------------------------------------------------------
    # EXECUTIVE SUMMARY
    # -----------------------------------------------------------------------
    h2("Executive Summary")
    p(payload["executive_summary"])
    gap()

    # -----------------------------------------------------------------------
    # PROFILE CONTEXT
    # -----------------------------------------------------------------------
    h2("Profile Context")

    h3("Demographics")
    kv_table([
        ("Income Type",          demo.get("income_type", "—")),
        ("Employment Stability", demo.get("employment_stability", "—")),
        ("Dependents",           demo.get("dependents", "—")),
        ("City Tier",            demo.get("city_tier", "—")),
    ])

    h3("Financial Snapshot")
    kv_table([
        ("EMI",            fin.get("emi", "—")),
        ("Emergency Fund", fin.get("savings", "—")),
        ("EMI Ratio",      fin.get("emi_ratio", "—")),
    ])

    life_events = ctx.get("life_events", [])
    h3("Life Events")
    if life_events:
        grid_table(
            ["Event", "Description", "Recency", "Weight"],
            [[e["event"], e["description"], e["recency"], e["emotional_weight"]]
             for e in life_events],
            col_widths=[28*mm, 80*mm, 24*mm, 24*mm],
        )
    else:
        p("None reported.")

    cultural = ctx.get("cultural_context", [])
    h3("Cultural Context")
    if cultural:
        grid_table(
            ["Type", "Description", "Negotiability"],
            [[s["type"], s["description"], s["negotiability"]] for s in cultural],
            col_widths=[35*mm, 100*mm, 25*mm],
        )
    else:
        p("None detected.")

    behavioral = ctx.get("behavioral_signals", [])
    h3("Behavioral Signals")
    if behavioral:
        grid_table(
            ["Type", "Description", "Strength"],
            [[s["type"], s["description"], s["strength"]] for s in behavioral],
            col_widths=[35*mm, 100*mm, 25*mm],
        )
    else:
        p("None detected.")

    contradictions = ctx.get("contradictions", [])
    if contradictions:
        h3("Contradictions")
        grid_table(
            ["Dominant", "Suppressed", "Explanation"],
            [[c["dominant"], c["suppressed"], c["explanation"]] for c in contradictions],
            col_widths=[35*mm, 35*mm, 90*mm],
        )
    gap()

    # -----------------------------------------------------------------------
    # 4-AXIS ASSESSMENT
    # -----------------------------------------------------------------------
    h2("4-Axis Assessment")
    axis_names = [
        ("risk",               "Risk Tolerance"),
        ("cashflow",           "Cash Flow Stability"),
        ("obligation",         "Obligation Burden"),
        ("context",            "Financial Sophistication"),
        ("financial_capacity", "Financial Capacity"),
    ]
    axis_rows = []
    for key, label in axis_names:
        e = axis.get(key, {})
        score = e.get("score", "—")
        lbl   = (e.get("label") or "unknown").upper()
        insight = (e.get("insight") or "")[:80]
        axis_rows.append([label, str(score), lbl, insight])

    grid_table(
        ["Axis", "Score", "Level", "Insight"],
        axis_rows,
        col_widths=[42*mm, 18*mm, 22*mm, 78*mm],
    )
    gap()

    # -----------------------------------------------------------------------
    # CROSS-AXIS SUITABILITY
    # -----------------------------------------------------------------------
    h2("Cross-Axis Suitability")
    kv_table([
        ("Archetype",    suit.get("archetype", "—")),
        ("Equity Range", suit.get("equity_range", "—")),
    ])
    p(suit.get("classification", ""))

    bc = suit.get("binding_constraint")
    if bc and bc.get("type") not in (None, "none", ""):
        h3(f"Binding Constraint: {bc['type'].replace('_', ' ').title()}")
        p(bc.get("description", ""))
        bullets(bc.get("actions", []))
    gap()

    # -----------------------------------------------------------------------
    # SUITABILITY INSIGHTS  (advisory guidance — separate from risks)
    # -----------------------------------------------------------------------
    suit_insights = payload.get("suitability_insights", [])
    if suit_insights:
        h2("Suitability Insights")
        bullets(suit_insights)
        gap()

    # -----------------------------------------------------------------------
    # RISK & BIAS ANALYSIS  (factual financial risks only)
    # -----------------------------------------------------------------------
    h2("Risk & Bias Analysis")
    h3("Key Risks")
    bullets(rb.get("key_risks", []))

    h3("Behavioral Bias")
    p(rb.get("bias_summary", "No specific behavioral bias identified."))
    bias_flags = rb.get("bias_flags", [])
    if bias_flags:
        bullets(bias_flags)
    gap()

    # -----------------------------------------------------------------------
    # RECOMMENDED STRATEGY
    # -----------------------------------------------------------------------
    h2("Recommended Strategy")
    kv_table([
        ("Equity Allocation",  f"{strat.get('equity_pct', '—')}%"),
        ("Debt Allocation",    f"{strat.get('debt_pct', '—')}%"),
        ("Primary Instrument", strat.get("primary_instrument", "—")),
        ("SIP Recommended",    "Yes" if strat.get("sip_recommended") else "No"),
        ("First Step",         strat.get("first_step", "—")),
    ])

    if strat.get("advisory_note"):
        warn(strat["advisory_note"])

    h3("Temporal Strategy")
    kv_table([
        ("Temporary Allocation",   "Yes" if temp.get("is_temporary") else "No"),
        ("Reassessment Trigger",   temp.get("reassessment_trigger", "—")),
        ("Reassessment Timeline",  temp.get("reassessment_timeline", "—")),
        ("Expected Shift",         temp.get("expected_shift", "—")),
    ])
    gap()

    # -----------------------------------------------------------------------
    # ACTION PLAN  (bucketed by time horizon)
    # -----------------------------------------------------------------------
    h2("Action Plan")
    actions = payload.get("actions", {})
    if isinstance(actions, dict):
        immediate   = actions.get("immediate", [])
        near_term   = actions.get("near_term", [])
        medium_term = actions.get("medium_term", [])
        has_any = immediate or near_term or medium_term
        if not has_any:
            bullets([], fallback="No specific actions identified.")
        else:
            if immediate:
                h3("Immediate (0–30 days)")
                bullets(immediate)
            if near_term:
                h3("Near-term (1–3 months)")
                bullets(near_term)
            if medium_term:
                h3("Medium-term (3–6 months)")
                bullets(medium_term)
    else:
        # Fallback for flat list (backward compat)
        bullets(actions if isinstance(actions, list) else [],
                fallback="No specific actions identified.")
    gap()

    # -----------------------------------------------------------------------
    # DO NOT RECOMMEND
    # -----------------------------------------------------------------------
    h2("Do Not Recommend")
    bullets(payload.get("restrictions", []), fallback="No specific restrictions identified.")
    gap()

    # -----------------------------------------------------------------------
    # ASSUMPTIONS & LIMITATIONS
    # -----------------------------------------------------------------------
    h2("Assumptions & Limitations")
    p(assum.get("completeness_note", ""))
    kv_table([
        ("Data Completeness", f"{assum.get('data_completeness', '—')}%"),
        ("Confidence Score",  str(assum.get("confidence_score", "—"))),
    ])
    missing = assum.get("missing_fields", [])
    if missing:
        p(f"Missing fields: {', '.join(str(m) for m in missing)}")
    h3("Assumptions")
    bullets(assum.get("assumptions", []))
    gap()

    # -----------------------------------------------------------------------
    # REGULATORY DISCLOSURES
    # -----------------------------------------------------------------------
    h2("Regulatory Disclosures")
    kv_table([
        ("SEBI Compliance",    disc["sebi_compliance"]),
        ("AI Usage",           disc["ai_usage"]),
        ("Data Retention",     disc["data_retention"]),
        ("Methodology",        disc["methodology_version"]),
    ], col_widths=[40*mm, 120*mm])
    story.append(Paragraph(disc["disclaimer"], S["small"]))
    gap()

    # -----------------------------------------------------------------------
    # APPENDIX — Advisor Technical Reference
    # -----------------------------------------------------------------------
    h2("Appendix — Advisor Technical Reference")
    warn("Not for client-facing use. For advisor review only.")

    import json as _json
    appendix_data = {
        "debug":            app.get("debug", {}),
        "trace_validation": app.get("trace_validation", {}),
    }
    raw_json = _json.dumps(appendix_data, indent=2, default=str)
    if len(raw_json) > 6000:
        raw_json = raw_json[:6000] + "\n... [truncated]"

    # Render as monospace paragraphs (reportlab has no <pre> equivalent)
    mono = ParagraphStyle("mono", parent=base["Code"],
                          fontSize=7, leading=10, spaceAfter=1,
                          fontName="Courier")
    for line in raw_json.splitlines():
        story.append(Paragraph(line.replace(" ", "&nbsp;").replace("<", "&lt;"), mono))

    # -----------------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------------
    doc.build(story)
    return os.path.abspath(output_path)
