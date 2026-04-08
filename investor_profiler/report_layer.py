"""
Report Layer — InvestorDNA (Consulting Grade v2)

Output structure:
  {
    "report":        advisor-facing, clean, no system terms
    "advisor_debug": full system transparency, all internal signals
  }

Design principle: Do not hide intelligence — present it at the right abstraction level.
No LLM calls. No new reasoning. Pure deterministic aggregation + transformation.
"""

from __future__ import annotations
import re


# ---------------------------------------------------------------------------
# Helpers
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
    s = s.strip().rstrip(".")
    if not s:
        return ""
    return s[0].upper() + s[1:] + "."


# ---------------------------------------------------------------------------
# Internal-to-human translation table
# Strips system language from advisor-facing text
# ---------------------------------------------------------------------------

_SYSTEM_TERM_REPLACEMENTS = [
    (r"\bconstraint violation\b",        "profile adjustment required"),
    (r"\bguardrail\b",                   "safety check"),
    (r"\ballocation mismatch\b",         "allocation does not match current profile"),
    (r"\btrace validation\b",            "profile verification"),
    (r"\btrace valid\b",                 "profile verified"),
    (r"\ballocation_mode\b",             "investment approach"),
    (r"\bblocking violation\b",          "profile inconsistency"),
    (r"\bconstraint engine\b",           "risk assessment system"),
    (r"\bguardrail adjustment\b",        "risk-based adjustment"),
    (r"\bscore mismatch\b",              "profile signal mismatch"),
    (r"\breasoning trace\b",             "profile analysis"),
    (r"\bdominant_factors\b",            "key drivers"),
    (r"\bstate_context\b",               "investor state"),
    (r"\bfallback\b",                    "conservative default"),
]

def _clean(text: str) -> str:
    """Remove all system/internal language from advisor-facing text."""
    for pattern, replacement in _SYSTEM_TERM_REPLACEMENTS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text



# ---------------------------------------------------------------------------
# generate_report — assembles the advisor-facing report dict from pipeline output
# This is the single function called by main.py.
# It wraps the raw pipeline_output into a clean report + full_profile structure.
# ---------------------------------------------------------------------------

def generate_report(pipeline_output: dict) -> dict:
    """
    Convert raw pipeline_output dict → advisor-facing report dict.

    The returned dict becomes pipeline_output["report"] in the final _wrap() call.
    It exposes:
      - Top-level advisor fields (executive_summary, key_risks, etc.)
      - full_profile: the complete structured profile for downstream use
    """
    p = pipeline_output  # shorthand

    # -----------------------------------------------------------------------
    # Executive summary — synthesized from decision reasoning + cross-axis
    # -----------------------------------------------------------------------
    decision    = _get(p, "decision", default={})
    cross_axis  = _get(p, "cross_axis", default={})
    suitability = _get(cross_axis, "suitability", default={})
    archetype   = _str(_get(cross_axis, "archetype"))
    reasoning   = _str(_get(decision, "reasoning"))
    advisor_note = _str(_get(decision, "advisor_note"))
    equity_range = _str(_get(decision, "current_allocation") or _get(decision, "equity_range"))

    summary_parts = []
    if archetype and archetype not in ("Unclassified", ""):
        summary_parts.append(f"Investor archetype: {archetype}.")
    if reasoning:
        summary_parts.append(_clean(reasoning))
    if advisor_note:
        summary_parts.append(_clean(advisor_note))
    if equity_range:
        summary_parts.append(f"Recommended equity range: {equity_range}.")

    executive_summary = " ".join(summary_parts) or "Profile assessment complete. Advisor review required."

    # -----------------------------------------------------------------------
    # Key risks — from decision risk_assessment + profile flags
    # -----------------------------------------------------------------------
    risk_assessment = _get(decision, "risk_assessment", default={})
    identified_risks = _list(risk_assessment, "identified_risks")
    key_risks = [_clean(r) for r in identified_risks if r and not _clean(r).startswith("check")]

    # Supplement from cross-axis insights
    for insight in _list(p, "suitability_insights"):
        cleaned = _clean(_str(insight))
        if cleaned and cleaned not in key_risks:
            key_risks.append(cleaned)

    # -----------------------------------------------------------------------
    # Bias summary — from narrative contradictions + behavioral signals
    # -----------------------------------------------------------------------
    narrative = _get(p, "narrative", default={})
    contradictions_text = _str(_get(narrative, "contradictions"))
    reliability = _str(_get(narrative, "reliability_assessment"))

    bias_parts = []
    if contradictions_text and "none" not in contradictions_text.lower():
        bias_parts.append(_clean(contradictions_text))
    if reliability:
        bias_parts.append(_clean(reliability))
    bias_summary = " ".join(bias_parts) or ""

    # -----------------------------------------------------------------------
    # Recommended actions — from decision strategy + binding constraint
    # -----------------------------------------------------------------------
    strategy = _get(decision, "strategy", default={})
    first_step = _str(_get(strategy, "first_step"))
    binding = _get(cross_axis, "binding_constraint", default={})
    priority_actions = _list(binding, "priority_actions") if isinstance(binding, dict) else []

    recommended_actions = []
    if first_step:
        recommended_actions.append(_clean(first_step))
    for action in priority_actions:
        cleaned = _clean(_str(action))
        if cleaned and cleaned not in recommended_actions:
            recommended_actions.append(cleaned)

    # -----------------------------------------------------------------------
    # Do not recommend — from suitability guidance (negative constraints)
    # -----------------------------------------------------------------------
    guidance = _list(suitability, "guidance")
    do_not_recommend = []
    for g in guidance:
        g_lower = g.lower()
        if any(w in g_lower for w in ["avoid", "do not", "not recommended", "no equity", "restrict"]):
            do_not_recommend.append(_clean(g))

    # Obligation-based restrictions
    obligation_score = _num(_get(p, "axis_scores", "obligation"), 0)
    if obligation_score and obligation_score > 70:
        do_not_recommend.append("Illiquid or long lock-in investment products")
        do_not_recommend.append("Aggressive or speculative equity instruments")

    # -----------------------------------------------------------------------
    # Completeness note
    # -----------------------------------------------------------------------
    data_completeness = _num(_get(p, "data_completeness"), 0)
    missing_fields = _list(p, "debug", "missing_fields")

    if data_completeness and data_completeness < 40:
        completeness_note = (
            f"Profile completeness is low ({int(data_completeness)}%). "
            f"Missing fields: {', '.join(missing_fields[:5]) if missing_fields else 'multiple fields'}. "
            "Recommendations should be treated as provisional."
        )
    elif data_completeness and data_completeness < 70:
        completeness_note = (
            f"Profile completeness is moderate ({int(data_completeness)}%). "
            "Some fields were inferred or unavailable."
        )
    else:
        completeness_note = (
            f"Profile completeness: {int(data_completeness) if data_completeness else 'unknown'}%."
        )

    # -----------------------------------------------------------------------
    # full_profile — the complete structured profile for report_formatter
    # -----------------------------------------------------------------------
    full_profile = {
        "profile_context":    _get(p, "profile_context", default={}),
        "axis_scores":        _get(p, "axis_scores", default={}),
        "category_scores":    _get(p, "category_scores", default={}),
        "cross_axis":         cross_axis,
        "decision":           decision,
        "debug":              _get(p, "debug", default={}),
        "trace_validation":   _get(p, "trace_validation", default={}),
        "constraint_report":  _get(p, "constraint_report", default={}),
        "final_decision":     _str(_get(p, "final_decision")),
        "confidence_score":   _num(_get(p, "confidence_score"), 0),
        "data_completeness":  data_completeness,
        "suitability_insights": _list(p, "suitability_insights"),
    }

    return {
        "executive_summary":   executive_summary,
        "key_risks":           key_risks,
        "bias_summary":        bias_summary,
        "recommended_actions": recommended_actions,
        "do_not_recommend":    do_not_recommend,
        "completeness_note":   completeness_note,
        "full_profile":        full_profile,
        # Pass-through fields for direct access
        "archetype":           archetype,
        "equity_range":        equity_range,
        "advisor_note":        _clean(advisor_note),
        "profile_context":     _get(p, "profile_context", default={}),
        "axis_scores":         _get(p, "axis_scores", default={}),
        "cross_axis":          cross_axis,
        "decision":            decision,
        "suitability_insights": _list(p, "suitability_insights"),
        "confidence_score":    _num(_get(p, "confidence_score"), 0),
        "data_completeness":   data_completeness,
    }
