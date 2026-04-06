"""
Report Layer — InvestorDNA (Consulting Grade)
Transforms pipeline output into advisor-ready, prioritized RIA intelligence.

Design principle: Do not add intelligence — only refine clarity, priority, usability.
No LLM calls. No new reasoning. Pure deterministic aggregation + transformation.

Input:  full pipeline output dict from run_pipeline()
Output: consulting-grade RIA report dict
"""

from __future__ import annotations
import re


# ---------------------------------------------------------------------------
# Helpers — safe accessors
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


def _list(d: dict, *keys) -> list:
    val = _get(d, *keys, default=[])
    return val if isinstance(val, list) else []


def _str(val) -> str:
    return str(val).strip() if val else ""


def _num(val, default=None):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _extracted_val(pipeline: dict, key: str):
    extracted = _get(pipeline, "extracted_data", default={})
    if not isinstance(extracted, dict):
        return None
    v = extracted.get(key)
    if isinstance(v, dict):
        return v.get("value")
    return v


def _sentence(s: str) -> str:
    """Ensure string ends with a period and is properly capitalised."""
    s = s.strip().rstrip(".")
    if not s:
        return ""
    return s[0].upper() + s[1:] + "."


def _normalise_risk_phrase(raw: str) -> str:
    """
    Rewrite raw factor text into standardised risk phrasing.
    Pattern: 'Lack of X' | 'Exposure to Y' | 'Risk of Z'
    """
    s = raw.strip().rstrip(".")
    lower = s.lower()

    # Already well-formed
    if lower.startswith(("lack of", "exposure to", "risk of", "absence of")):
        return _sentence(s)

    # Common rewrites
    rewrites = [
        (r"^no savings",            "Lack of savings discipline"),
        (r"^no emergency",          "Absence of an emergency fund"),
        (r"^no plan",               "Lack of a structured financial plan"),
        (r"^no experience",         "Lack of investment experience"),
        (r"^no investment",         "Lack of investment activity"),
        (r"^variable income",       "Income instability due to variable earnings"),
        (r"^irregular income",      "Income instability due to irregular earnings"),
        (r"^freelance income",      "Income instability due to freelance earnings"),
        (r"^unstable income",       "Income instability due to unstable earnings"),
        (r"^inconsistent",          "Risk of inconsistent financial behaviour"),
        (r"^unstructured",          "Lack of financial structure and planning discipline"),
        (r"^high loss.aversion",    "Exposure to panic-driven exit risk during market drawdowns"),
        (r"^panic",                 "Exposure to panic-driven decision-making under market stress"),
        (r"^behavioural contradiction", "Risk of behavioural contradiction"),
        (r"^guardrail triggered",   "Constraint violation requiring allocation adjustment"),
    ]
    for pattern, replacement in rewrites:
        m = re.match(pattern, lower)
        if m:
            remainder = s[m.end():].strip(" :-—")
            # Only append remainder if it adds meaningful context (not a single word echo)
            if remainder and len(remainder.split()) > 2:
                return _sentence(f"{replacement} — {remainder}")
            return _sentence(replacement)

    # Fallback: prefix with "Risk of"
    return _sentence(f"Risk of {s[0].lower()}{s[1:]}")


# ---------------------------------------------------------------------------
# Risk Flag (computed first — used by executive summary and other sections)
# ---------------------------------------------------------------------------

def _risk_flag(pipeline: dict) -> str:
    dominant_trait    = _str(_get(pipeline, "decision", "state_context", "dominant_trait")).lower()
    state_stability   = _str(_get(pipeline, "investor_state", "state_stability")).lower()
    income_type       = _str(_extracted_val(pipeline, "income_type")).lower()
    emergency_months  = _num(_extracted_val(pipeline, "emergency_months"), default=0)
    has_guardrails    = bool(_list(pipeline, "constraint_report", "guardrail_adjustments"))
    has_contradictions = bool(_list(pipeline, "signals", "contradictions"))
    constraint_level  = _str(_get(pipeline, "signals", "financial_state", "constraint_level")).lower()
    data_completeness = _num(_get(pipeline, "data_completeness", default=0), default=0)

    no_savings       = emergency_months < 1
    unstable_income  = income_type in ("variable", "freelance", "irregular", "contract", "self_employed")
    behavioral_issue = dominant_trait in ("panic", "grief", "crisis", "fear", "inconsistent")
    high_constraint  = constraint_level == "high"

    if (no_savings and unstable_income) or behavioral_issue or has_guardrails or high_constraint:
        return "High"

    has_savings   = emergency_months >= 3
    stable_income = income_type in ("salaried", "salary", "fixed")
    stable_state  = state_stability == "stable"
    good_data     = data_completeness >= 70

    if has_savings and stable_income and stable_state and not has_contradictions and good_data:
        return "Low"

    return "Moderate"


# ---------------------------------------------------------------------------
# Section 1 — Executive Summary
# ---------------------------------------------------------------------------

def _executive_summary(pipeline: dict, risk_flag: str, decision_confidence: str = "high") -> str:
    archetype       = _str(_get(pipeline, "decision", "archetype"))
    compound_state  = _str(_get(pipeline, "investor_state", "compound_state"))
    dominant_trait  = _str(_get(pipeline, "decision", "state_context", "dominant_trait"))
    constraint      = _str(_get(pipeline, "signals", "financial_state", "constraint_level"))
    life_summary    = _str(_get(pipeline, "narrative", "life_summary"))
    risk_truth      = _str(_get(pipeline, "narrative", "risk_truth"))
    state_stability = _str(_get(pipeline, "investor_state", "state_stability"))
    income_type     = _str(_extracted_val(pipeline, "income_type"))

    # Sentence 1 — single complete classification sentence
    s1_parts: list[str] = []
    if archetype and archetype not in ("Unclassified", ""):
        s1_parts.append(archetype)
    if compound_state and compound_state not in ("unknown", ""):
        s1_parts.append(compound_state)
    if risk_flag:
        s1_parts.append(f"presenting a {risk_flag.lower()} overall risk profile")

    if s1_parts:
        sentence1 = _sentence(", ".join(s1_parts))
    elif life_summary:
        sentence1 = _sentence(life_summary)
        life_summary = ""  # consumed — don't repeat
    else:
        sentence1 = ""

    # Sentence 2 — financial and behavioural state
    s2_parts: list[str] = []
    if life_summary:
        s2_parts.append(life_summary.rstrip("."))
    if dominant_trait and dominant_trait not in ("unknown", ""):
        trait_clause = f"dominant behavioural trait is '{dominant_trait}'"
        if constraint and constraint not in ("low", ""):
            trait_clause += f" with {constraint} financial constraint"
        if income_type:
            trait_clause += f" and {income_type} income"
        s2_parts.append(trait_clause)
    sentence2 = _sentence("; ".join(s2_parts)) if s2_parts else ""

    # Sentence 3 — risk truth or stability note (pick the more informative one)
    sentence3 = ""
    if risk_truth:
        sentence3 = _sentence(risk_truth)
    elif state_stability and state_stability not in ("stable", ""):
        sentence3 = _sentence(
            f"The investor's state is currently {state_stability} — "
            "recommendations should account for near-term behavioural volatility"
        )

    parts = [s for s in (sentence1, sentence2, sentence3) if s]

    if not parts:
        return "Insufficient data to generate an executive summary."

    # Prepend a decision reliability note when fallback was applied
    if decision_confidence == "low":
        parts.insert(0,
            "Note: the allocation decision could not be fully validated — "
            "a conservative fallback has been applied; all other profile insights remain valid."
        )

    # Cap at 3 sentences
    return " ".join(parts[:3])


# ---------------------------------------------------------------------------
# Section 2 — Key Risks (prioritised, standardised phrasing, max 3)
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _severity_from_context(text: str) -> str:
    lower = text.lower()
    if any(w in lower for w in (
        "no savings", "unstable", "guardrail", "panic", "crisis", "grief",
        "obligation", "constraint", "no emergency", "variable income",
        "irregular income", "loss-aversion", "panic-driven",
    )):
        return "high"
    if any(w in lower for w in (
        "inconsistent", "no plan", "lack of", "unstructured", "contradiction",
        "mismatch", "low knowledge", "no experience", "behavioral",
    )):
        return "medium"
    return "low"


def _key_risks(pipeline: dict) -> list[dict]:
    candidates: list[dict] = []

    # From reasoning_trace dominant_factors
    for factor in _list(pipeline, "decision", "reasoning_trace", "dominant_factors"):
        s = _str(factor)
        if s:
            candidates.append({
                "risk":     _normalise_risk_phrase(s),
                "severity": _severity_from_context(s),
                "reason":   "Identified as a primary driver in the reasoning trace.",
            })

    # From signal contradictions
    for c in _list(pipeline, "signals", "contradictions"):
        if not isinstance(c, dict):
            continue
        dominant    = _str(c.get("dominant_trait"))
        suppressed  = _str(c.get("suppressed_trait"))
        explanation = _str(c.get("explanation"))
        if dominant and suppressed:
            candidates.append({
                "risk":     _sentence(
                    f"Risk of behavioural contradiction — '{dominant}' overrides "
                    f"'{suppressed}' under stress"
                ),
                "severity": _severity_from_context(dominant + " " + explanation),
                "reason":   explanation or "Detected via signal contradiction analysis.",
            })

    # From guardrail adjustments
    for adj in _list(pipeline, "constraint_report", "guardrail_adjustments"):
        if not isinstance(adj, dict):
            continue
        before  = _str(adj.get("before"))
        after   = _str(adj.get("after"))
        reason  = _str(adj.get("reason"))
        if before:
            candidates.append({
                "risk":     _sentence(
                    f"Constraint violation — allocation reduced from {before} to {after}"
                ),
                "severity": "high",
                "reason":   reason or "Safety adjustment enforced by the constraint engine.",
            })

    # From signals.behavior loss_response
    loss_response = _str(_get(pipeline, "signals", "behavior", "loss_response")).lower()
    if loss_response in ("panic_sell", "panic", "exit_all", "freeze"):
        candidates.append({
            "risk":     _sentence(
                "Exposure to panic-driven exit risk — investor is likely to liquidate "
                "positions during market drawdowns"
            ),
            "severity": "high",
            "reason":   f"Behavioral signal: loss_response = '{loss_response}'.",
        })

    # Deduplicate by normalised risk prefix (first 50 chars)
    seen: set[str] = set()
    unique: list[dict] = []
    for r in candidates:
        key = r["risk"][:50].lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)

    # Sort high → medium → low
    unique.sort(key=lambda x: _SEVERITY_ORDER.get(x["severity"], 3))

    if not unique:
        return [{
            "risk":     "No material risks identified — additional data may be required.",
            "severity": "low",
            "reason":   "Insufficient signal data to derive specific risks.",
        }]

    return unique[:3]


# ---------------------------------------------------------------------------
# Section 3 — Recommended Actions (executable, merged, max 3–4)
# ---------------------------------------------------------------------------

_LIQUIDITY_KEYWORDS = (
    "liquid fund", "emergency fund", "liquidity buffer",
    "build emergency", "start saving", "savings",
)


def _overlaps_liquidity(action_text: str) -> bool:
    lower = action_text.lower()
    return any(kw in lower for kw in _LIQUIDITY_KEYWORDS)


def _recommended_actions(pipeline: dict) -> list[dict]:
    raw: list[dict] = []

    surplus_raw = _investable_surplus_raw(pipeline)

    # From suitability_insights
    for insight in _list(pipeline, "cross_axis", "suitability_insights"):
        s = _str(insight)
        if s:
            raw.append({
                "action":           _sentence(s),
                "timeline":         "within 30–60 days",
                "expected_outcome": "Align portfolio with the assessed suitability profile.",
                "_group":           "suitability",
            })

    # From advisor_insight
    advisor_insight = _str(_get(pipeline, "narrative", "advisor_insight"))
    if advisor_insight:
        raw.append({
            "action":           _sentence(advisor_insight),
            "timeline":         "immediate",
            "expected_outcome": "Address the primary concern identified in the investor profile.",
            "_group":           "advisor",
        })

    # From state_implications
    for imp in _list(pipeline, "investor_state", "state_implications"):
        s = _str(imp)
        if s:
            raw.append({
                "action":           _sentence(s),
                "timeline":         "within 60–90 days",
                "expected_outcome": "Stabilise investor state and reduce behavioural risk.",
                "_group":           "state",
            })

    # Emergency fund action
    emergency_months = _num(_extracted_val(pipeline, "emergency_months"), default=0)
    if emergency_months < 3 and surplus_raw and surplus_raw > 0:
        monthly_save = round(min(surplus_raw * 0.5, surplus_raw) / 500) * 500
        monthly_income = _num(_extracted_val(pipeline, "monthly_income"), 0) or 0
        months_to_goal = round((3 * monthly_income) / max(monthly_save, 1))
        raw.append({
            "action":           (
                f"Build an emergency fund by investing ₹{monthly_save:,.0f}/month "
                f"into a liquid fund."
            ),
            "timeline":         "within the next 30 days",
            "expected_outcome": (
                f"Establish a 3-month income buffer in approximately "
                f"{months_to_goal} months."
            ),
            "_group":           "liquidity",
        })

    # Reassessment trigger
    trigger       = _str(_get(pipeline, "decision", "temporal_strategy", "reassessment_trigger"))
    timeline_hint = _str(_get(pipeline, "decision", "temporal_strategy", "reassessment_timeline"))
    if trigger and trigger not in ("standard annual review", ""):
        raw.append({
            "action":           _sentence(f"Schedule a portfolio reassessment once: {trigger}"),
            "timeline":         timeline_hint or "as triggered",
            "expected_outcome": "Recalibrate allocation to reflect the investor's updated state.",
            "_group":           "reassessment",
        })

    # --- Merge liquidity-related actions into one ---
    liquidity_actions = [a for a in raw if _overlaps_liquidity(a["action"])]
    other_actions     = [a for a in raw if not _overlaps_liquidity(a["action"])]

    merged: list[dict] = []
    if liquidity_actions:
        # Use the most specific one (emergency fund with ₹ amount preferred)
        best = next(
            (a for a in liquidity_actions if "₹" in a["action"]),
            liquidity_actions[0],
        )
        merged.append({k: v for k, v in best.items() if k != "_group"})

    # Deduplicate other actions by prefix
    seen: set[str] = set()
    for a in other_actions:
        key = a["action"][:50].lower()
        if key not in seen:
            seen.add(key)
            merged.append({k: v for k, v in a.items() if k != "_group"})

    if not merged:
        merged.append({
            "action":           "Conduct a detailed advisor consultation before making any investment recommendations.",
            "timeline":         "immediate",
            "expected_outcome": "Establish a baseline financial plan.",
        })

    return merged[:4]


# ---------------------------------------------------------------------------
# Section 4 — Do Not Recommend (concise, advisor-friendly)
# ---------------------------------------------------------------------------

def _humanise_field(field: str) -> str:
    """Convert snake_case field names to readable allocation labels."""
    mapping = {
        "current_allocation": "equity allocation",
        "allocation":         "equity allocation",
        "equity_range":       "equity allocation",
        "baseline_allocation": "baseline equity allocation",
    }
    return mapping.get(field.lower(), field.replace("_", " "))


def _do_not_recommend(pipeline: dict) -> list[dict]:
    dnr: list[dict] = []

    # From guardrail adjustments
    for adj in _list(pipeline, "constraint_report", "guardrail_adjustments"):
        if not isinstance(adj, dict):
            continue
        field  = _humanise_field(_str(adj.get("field")))
        before = _str(adj.get("before"))
        reason = _str(adj.get("reason"))
        if before and reason:
            dnr.append({
                "restriction":     _sentence(f"Avoid {before} {field} at the current stage"),
                "reason":          _sentence(reason),
                "risk_if_ignored": "Investor may be exposed to an allocation level that the constraint engine has flagged as unsafe.",
            })

    # From dominant_trait
    dominant_trait = _str(_get(pipeline, "decision", "state_context", "dominant_trait")).lower()
    if dominant_trait in ("panic", "grief", "crisis", "fear"):
        dnr.append({
            "restriction":     "Avoid high-equity, leveraged, or high-volatility products.",
            "reason":          _sentence(
                f"Dominant trait is '{dominant_trait}', indicating low loss tolerance "
                "and a high likelihood of stress-driven exit"
            ),
            "risk_if_ignored": "Investor is likely to panic-exit during drawdowns, crystallising losses.",
        })

    # From risk mismatch
    mismatch_note = _str(_get(pipeline, "validation", "mismatch_note"))
    if mismatch_note:
        dnr.append({
            "restriction":     "Avoid products that exceed the investor's assessed risk capacity.",
            "reason":          _sentence(mismatch_note),
            "risk_if_ignored": "Investor may be exposed to risk beyond their actual tolerance.",
        })

    # From stated_vs_actual_gap
    gap = _str(_get(pipeline, "decision", "risk_assessment", "stated_vs_actual_gap")).lower()
    if gap and gap not in ("aligned", "unknown", ""):
        dnr.append({
            "restriction":     "Do not rely solely on the investor's stated risk preference.",
            "reason":          _sentence(f"A stated-vs-actual gap has been identified: {gap}"),
            "risk_if_ignored": "Recommendations anchored to stated preference will be misaligned with actual behaviour under stress.",
        })

    # From allocation_mode
    allocation_mode = _str(_get(pipeline, "decision", "allocation_mode")).lower()
    if allocation_mode == "transitional":
        dnr.append({
            "restriction":     "Avoid long lock-in periods or illiquid products at this stage.",
            "reason":          "The investor is in a transitional state — allocation is expected to shift in the near term.",
            "risk_if_ignored": "Investor may be locked into a position that no longer reflects their financial state.",
        })

    if not dnr:
        dnr.append({
            "restriction":     "No specific product exclusions identified at this time.",
            "reason":          "Profile does not trigger any hard restrictions.",
            "risk_if_ignored": "N/A",
        })

    return dnr[:4]


# ---------------------------------------------------------------------------
# Section 5 — Bias Summary (consultative, real-world labels)
# ---------------------------------------------------------------------------

_BIAS_TYPE_MAP = {
    "inconsistency":            "Behavioral Inconsistency",
    "behavioral_inconsistency": "Behavioral Inconsistency",
    "panic":                    "Emotional Override",
    "knowledge_gap":            "Emotional Override",
    "peer_influence":           "Herd Bias",
    "herd":                     "Herd Bias",
    "no_structure":             "Lack of Financial Discipline",
    "unstructured":             "Lack of Financial Discipline",
    "overconfidence":           "Overconfidence Bias",
    "loss_aversion":            "Loss Aversion Bias",
    "recency":                  "Recency Bias",
}


def _map_bias_type(raw_type: str, dominant: str, suppressed: str) -> str:
    raw_lower = raw_type.lower().replace(" ", "_")
    for key, label in _BIAS_TYPE_MAP.items():
        if key in raw_lower:
            return label
    combined = (dominant + " " + suppressed).lower()
    if any(w in combined for w in ("panic", "fear", "grief")):
        return "Emotional Override"
    if any(w in combined for w in ("knowledge", "experience")):
        return "Emotional Override"
    if any(w in combined for w in ("peer", "herd")):
        return "Herd Bias"
    if any(w in combined for w in ("inconsistent", "unstructured")):
        return "Behavioral Inconsistency"
    return raw_type.replace("_", " ").title() + " Bias"


def _bias_summary(pipeline: dict) -> list[dict]:
    biases: list[dict] = []

    # From signal contradictions
    for c in _list(pipeline, "signals", "contradictions"):
        if not isinstance(c, dict):
            continue
        dominant    = _str(c.get("dominant_trait"))
        suppressed  = _str(c.get("suppressed_trait"))
        explanation = _str(c.get("explanation"))
        ctype       = _str(c.get("type", ""))
        if dominant:
            bias_label = _map_bias_type(ctype, dominant, suppressed)
            biases.append({
                "bias":         bias_label,
                "impact":       _sentence(
                    f"Under stress, '{dominant}' overrides '{suppressed}' — "
                    "stated preferences are unlikely to reflect actual behaviour"
                ),
                "advisor_note": _sentence(
                    explanation or
                    "Probe how the investor has actually behaved during past market stress, "
                    "not how they report they would behave"
                ),
            })

    # From reasoning_trace contradictions
    for c in _list(pipeline, "decision", "reasoning_trace", "contradictions"):
        if not isinstance(c, dict):
            continue
        s1       = _str(c.get("signal_1"))
        s2       = _str(c.get("signal_2"))
        resolution = _str(c.get("resolution"))
        dominant = _str(c.get("dominant_trait"))
        if s1 and s2:
            biases.append({
                "bias":         "Conflicting Signal Bias",
                "impact":       _sentence(
                    f"Conflicting signals detected ({s1} vs {s2}); "
                    f"'{dominant}' was selected as the dominant driver"
                ),
                "advisor_note": _sentence(
                    resolution or
                    "Verify that the dominant signal reflects the investor's actual "
                    "decision-making pattern, not a one-time response"
                ),
            })

    # From suppressed_traits — renamed to "Latent growth bias" per spec
    for trait in _list(pipeline, "investor_state", "suppressed_traits"):
        s = _str(trait)
        if s:
            biases.append({
                "bias":         "Latent Growth Bias",
                "impact":       _sentence(
                    f"A growth orientation ('{s}') is present but currently suppressed "
                    "by the dominant behavioural pattern"
                ),
                "advisor_note": _sentence(
                    f"Monitor for re-emergence of '{s}' as the investor's financial "
                    "situation stabilises or market conditions improve"
                ),
            })

    # Deduplicate
    seen: set[str] = set()
    unique: list[dict] = []
    for b in biases:
        key = b["bias"] + "|" + b["impact"][:40]
        if key not in seen:
            seen.add(key)
            unique.append(b)

    if not unique:
        unique.append({
            "bias":         "None detected",
            "impact":       "No significant behavioural contradictions identified in this profile.",
            "advisor_note": "Continue monitoring for emerging biases as the investor gains market experience.",
        })

    return unique


# ---------------------------------------------------------------------------
# Section 6 — Investable Surplus
# ---------------------------------------------------------------------------

def _investable_surplus_raw(pipeline: dict):
    """Returns numeric surplus or None. Used internally by actions section."""
    monthly_income = _num(_extracted_val(pipeline, "monthly_income"))
    emi_amount     = _num(_extracted_val(pipeline, "emi_amount"), default=0)
    dependents     = _num(_extracted_val(pipeline, "dependents"), default=0)
    income_type    = _str(_extracted_val(pipeline, "income_type")).lower()
    future_score   = _num(_get(pipeline, "debug", "future_obligation_score", default=0), default=0.0)

    if monthly_income is None:
        return None

    discretionary     = monthly_income * 0.30
    emi               = emi_amount or 0
    dependent_cost    = dependents * 4000
    is_variable       = income_type in ("variable", "freelance", "irregular", "contract", "self_employed", "")
    volatility_buffer = monthly_income * (0.15 if is_variable else 0.10)
    obligation_buffer = monthly_income * future_score * 0.15

    return monthly_income - discretionary - emi - dependent_cost - volatility_buffer - obligation_buffer


def _investable_surplus(pipeline: dict) -> str:
    monthly_income = _num(_extracted_val(pipeline, "monthly_income"))
    if monthly_income is None:
        return "Cannot compute — monthly income not available."

    surplus = _investable_surplus_raw(pipeline)
    if surplus is None:
        return "Cannot compute — monthly income not parseable."

    if surplus <= 0:
        return "₹0 — income is fully absorbed by expenses, obligations, and buffers."

    income_type = _str(_extracted_val(pipeline, "income_type")).lower()
    is_variable = income_type in ("variable", "freelance", "irregular", "contract", "self_employed", "")

    low  = max(0, round(surplus * 0.9 / 500) * 500)
    high = round(surplus * 1.1 / 500) * 500

    buffer_note = (
        "15% volatility buffer applied for variable income"
        if is_variable else
        "10% income buffer applied"
    )
    return (
        f"₹{low:,.0f} – ₹{high:,.0f} per month "
        f"(estimated after expenses and buffers; based on inferred spending patterns; {buffer_note})"
    )


# ---------------------------------------------------------------------------
# Section 7 — Completeness Note
# ---------------------------------------------------------------------------

def _completeness_note(pipeline: dict) -> str:
    completeness = _num(_get(pipeline, "data_completeness", default=None))
    missing      = _list(pipeline, "debug", "missing_fields")

    if completeness is None:
        return "Data completeness could not be assessed."

    pct = int(completeness)

    if pct >= 80:
        level = "High"
        note  = "Recommendations can be made with reasonable confidence."
    elif pct >= 50:
        level = "Moderate"
        note  = "Sufficient for a provisional recommendation — advisor should verify missing fields before finalising."
    else:
        level = "Low"
        note  = "Profile is incomplete — treat all recommendations as indicative only."

    if missing:
        humanised   = [str(f).replace("_", " ") for f in missing[:5]]
        missing_str = ", ".join(humanised)
        suffix = f" Missing: {missing_str}" + (" and more." if len(missing) > 5 else ".")
    else:
        suffix = ""

    return f"{level} completeness ({pct}%). {note}{suffix}"


# ---------------------------------------------------------------------------
# Final validation pass — enforce max limits and cross-section deduplication
# ---------------------------------------------------------------------------

def _validate_report(report: dict) -> dict:
    """
    Post-generation quality pass:
    - Enforce max limits (risks ≤ 3, actions ≤ 4)
    - Remove any action text that duplicates a risk
    - Ensure no sentence fragments (strings shorter than 10 chars)
    """
    # Enforce limits
    report["key_risks"]           = report["key_risks"][:3]
    report["recommended_actions"] = report["recommended_actions"][:4]
    report["do_not_recommend"]    = report["do_not_recommend"][:4]

    # Cross-section dedup: remove actions whose text substantially overlaps a risk
    risk_tokens = set()
    for r in report["key_risks"]:
        for word in r.get("risk", "").lower().split():
            if len(word) > 5:
                risk_tokens.add(word)

    clean_actions: list[dict] = []
    for a in report["recommended_actions"]:
        action_lower = a.get("action", "").lower()
        overlap = sum(1 for t in risk_tokens if t in action_lower)
        # Allow if overlap is low (actions naturally reference risks)
        # Only drop if the action IS the risk reworded (high overlap + short action)
        if overlap >= 4 and len(a.get("action", "")) < 80:
            continue
        clean_actions.append(a)
    report["recommended_actions"] = clean_actions[:4]

    # Fragment guard — replace any string value shorter than 10 chars with a fallback
    for key in ("executive_summary", "investable_surplus", "completeness_note"):
        if isinstance(report.get(key), str) and len(report[key]) < 10:
            report[key] = "Not available."

    return report


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_report(pipeline_output: dict) -> dict:
    """
    Convert full pipeline output into a consulting-grade RIA-ready report.
    No LLM calls. No pipeline modification. Pure deterministic transformation.

    Always generates a full report. Decision failure degrades confidence, not insight.
    """
    # Only bail on truly unrecoverable upstream errors (non-english, all-null, signal failure)
    # "invalid_reasoning" is no longer a hard stop — fallback decision is already injected upstream
    status = pipeline_output.get("status")
    if status == "error":
        return {
            "risk_flag":           "High",
            "decision_confidence": "low",
            "executive_summary":   _sentence(
                f"Report generation blocked — pipeline returned an unrecoverable error. "
                f"{pipeline_output.get('message', '')}"
            ),
            "key_risks": [{
                "risk":     _sentence(pipeline_output.get("message", "Unknown pipeline error")),
                "severity": "high",
                "reason":   "Pipeline did not complete successfully.",
            }],
            "recommended_actions": [{
                "action":           "Resolve all pipeline errors before generating an investor report.",
                "timeline":         "immediate",
                "expected_outcome": "Valid, unblocked pipeline output.",
            }],
            "do_not_recommend": [{
                "restriction":     "Do not make any investment recommendations at this stage.",
                "reason":          "Pipeline output is invalid or incomplete.",
                "risk_if_ignored": "Any recommendations would be based on unvalidated reasoning.",
            }],
            "bias_summary":      [],
            "investable_surplus": "N/A",
            "completeness_note":  "N/A — pipeline did not complete successfully.",
            "full_profile":       pipeline_output,
        }

    risk_flag           = _risk_flag(pipeline_output)
    decision_confidence = _str(pipeline_output.get("decision_confidence", "high")) or "high"

    report = {
        "risk_flag":           risk_flag,
        "decision_confidence": decision_confidence,
        "executive_summary":   _executive_summary(pipeline_output, risk_flag, decision_confidence),
        "key_risks":           _key_risks(pipeline_output),
        "recommended_actions": _recommended_actions(pipeline_output),
        "do_not_recommend":    _do_not_recommend(pipeline_output),
        "bias_summary":        _bias_summary(pipeline_output),
        "investable_surplus":  _investable_surplus(pipeline_output),
        "completeness_note":   _completeness_note(pipeline_output),
        "full_profile":        pipeline_output,
    }

    return _validate_report(report)
