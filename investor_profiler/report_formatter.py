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


def _build_executive_summary(report: dict, meta: dict) -> str:
    raw = _get(report, "executive_summary")
    if not raw:
        # Compose from available fields
        archetype = _get(report, "full_profile", "cross_axis", "archetype")
        decision  = _get(report, "full_profile", "final_decision")
        confidence = meta.get("confidence", "low")
        parts = []
        if archetype and not _is_system_phrase(archetype):
            parts.append(f"Investor archetype: {archetype}.")
        if decision and not _is_system_phrase(decision):
            parts.append(_clean(decision))
        if not parts:
            parts.append("Insufficient data to generate a complete executive summary.")
        raw = " ".join(parts)

    summary = _clean(raw)

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


def _build_risk_bias(report: dict, full_profile: dict) -> dict:
    # Risks
    raw_risks = _lst(report, "key_risks")
    risks = _clean_list(raw_risks)

    if not risks:
        # Generate fallback risks from profile signals
        ctx = _get(full_profile, "profile_context", default={})
        fin = _get(ctx, "financial_snapshot", default={})
        demo = _get(ctx, "demographics", default={})
        fallback_risks = []
        if not _num(fin.get("emergency_months")) or _num(fin.get("emergency_months"), 0) < 3:
            fallback_risks.append("Insufficient emergency fund — financial vulnerability to unexpected events")
        if demo.get("income_type") in ("gig", "business", "freelance"):
            fallback_risks.append("Income instability — irregular cash flow limits consistent investment capacity")
        obligation = _get(full_profile, "axis_scores", "obligation")
        if _num(obligation, 0) > 60:
            fallback_risks.append("High obligation burden — future financial commitments may limit investable surplus")
        if not fallback_risks:
            fallback_risks.append("Insufficient data to enumerate specific risks — advisor review required")
        risks = fallback_risks

    # Bias
    raw_bias = _get(report, "bias_summary")
    bias = _clean(raw_bias) if raw_bias and not _is_system_phrase(raw_bias) else ""

    # Behavioral flags from profile
    flags = _get(full_profile, "profile_context", "flags", default={})
    bias_flags = []
    if flags.get("recency_bias_risk"):
        bias_flags.append("Recency bias detected — recent market performance may be distorting risk perception")
    if flags.get("peer_driven"):
        bias_flags.append("Peer-influenced decision making — investment choices may not reflect personal risk capacity")
    if flags.get("grief_state"):
        bias_flags.append("Grief state — current risk aversion may be temporary and not reflective of baseline behavior")

    return {
        "key_risks":   risks,
        "bias_summary": bias or "No specific behavioral bias identified.",
        "bias_flags":  bias_flags,
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


def _build_actions(report: dict) -> list[str]:
    raw = _lst(report, "recommended_actions")
    actions = []
    for item in raw:
        text = _clean(str(item))
        # Keep only executable steps — filter out meta/system statements
        if not text:
            continue
        if _is_system_phrase(text):
            continue
        # Must look like an action (starts with verb or contains actionable language)
        lower = text.lower()
        non_action_markers = [
            "note:", "disclaimer", "subject to", "this report",
            "generated by", "ai-based", "not for client",
        ]
        if any(m in lower for m in non_action_markers):
            continue
        actions.append(text)
    return actions


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

    return {
        "cover_page":        _build_cover_page(report, meta, client_id),
        "executive_summary": _build_executive_summary(report, meta),
        "profile_context":   _build_profile_context(full_profile),
        "axis_assessment":   _build_axis_assessment(full_profile),
        "suitability":       _build_suitability(full_profile),
        "risk_bias":         _build_risk_bias(report, full_profile),
        "strategy":          _build_strategy(report, full_profile, meta),
        "actions":           _build_actions(report),
        "restrictions":      _build_restrictions(report),
        "assumptions":       _build_assumptions(report, full_profile, meta),
        "disclosures":       _build_disclosures(),
        "appendix":          _build_appendix(full_profile),
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
    Render PDF payload → PDF file.

    Tries weasyprint first (HTML → PDF, recommended).
    Falls back to reportlab if weasyprint is not installed.

    Args:
        payload:     Output from build_pdf_payload()
        output_path: Destination file path

    Returns:
        Absolute path to the generated PDF.
    """
    import os

    html_content = _build_html(payload)

    # --- Option 1: weasyprint (recommended) ---
    try:
        from weasyprint import HTML
        HTML(string=html_content).write_pdf(output_path)
        return os.path.abspath(output_path)
    except ImportError:
        pass  # fall through to reportlab

    # --- Option 2: reportlab (plain text fallback) ---
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors

        doc = SimpleDocTemplate(output_path, pagesize=A4,
                                leftMargin=20*mm, rightMargin=20*mm,
                                topMargin=20*mm, bottomMargin=20*mm)
        styles = getSampleStyleSheet()
        story  = []

        def _h1(text):
            story.append(Paragraph(text, styles["Heading1"]))
            story.append(Spacer(1, 4*mm))

        def _h2(text):
            story.append(Paragraph(text, styles["Heading2"]))
            story.append(Spacer(1, 2*mm))

        def _p(text):
            story.append(Paragraph(str(text), styles["Normal"]))
            story.append(Spacer(1, 2*mm))

        cover = payload["cover_page"]
        _h1("Investor Suitability Profile")
        _p(f"Client ID: {cover['client_id']}  |  Date: {cover['date']}  |  Status: {cover['status']}")
        _p(cover["ai_disclosure"])

        if cover["status"] == "Provisional":
            _p("⚠ PROVISIONAL REPORT — Manual advisor validation required.")

        _h2("Executive Summary")
        _p(payload["executive_summary"])

        ctx  = payload["profile_context"]
        demo = ctx.get("demographics", {})
        fin  = ctx.get("financial_snapshot", {})
        _h2("Profile Context")
        _p(f"Income: {demo.get('income_type','—')}  |  Employment: {demo.get('employment_stability','—')}  "
           f"|  Dependents: {demo.get('dependents','—')}  |  City: {demo.get('city_tier','—')}")
        _p(f"EMI: {fin.get('emi','—')}  |  Emergency Fund: {fin.get('savings','—')}")

        _h2("4-Axis Assessment")
        axis = payload["axis_assessment"]
        axis_data = [["Axis", "Score", "Level", "Insight"]]
        for key, label in [("risk","Risk"), ("cashflow","Cash Flow"),
                            ("obligation","Obligation"), ("context","Sophistication"),
                            ("financial_capacity","Capacity")]:
            e = axis.get(key, {})
            axis_data.append([label, str(e.get("score","—")), e.get("label","—").upper(), e.get("insight","")[:60]])
        t = Table(axis_data, colWidths=[35*mm, 18*mm, 22*mm, 85*mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#003366")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("GRID",       (0,0), (-1,-1), 0.5, colors.grey),
            ("FONTSIZE",   (0,0), (-1,-1), 9),
        ]))
        story.append(t)
        story.append(Spacer(1, 4*mm))

        suit = payload["suitability"]
        _h2("Cross-Axis Suitability")
        _p(f"Archetype: {suit.get('archetype','—')}  |  Equity Range: {suit.get('equity_range','—')}")
        _p(suit.get("classification", ""))

        rb = payload["risk_bias"]
        _h2("Risk & Bias Analysis")
        for r in rb.get("key_risks", []):
            _p(f"• {r}")
        _p(f"Bias: {rb.get('bias_summary','—')}")

        strat = payload["strategy"]
        _h2("Recommended Strategy")
        _p(f"Equity: {strat.get('equity_pct','—')}%  |  Debt: {strat.get('debt_pct','—')}%  "
           f"|  Instrument: {strat.get('primary_instrument','—')}")
        _p(f"First step: {strat.get('first_step','—')}")
        if strat.get("advisory_note"):
            _p(f"⚠ {strat['advisory_note']}")

        _h2("Action Plan")
        for a in payload.get("actions", []):
            _p(f"• {a}")

        _h2("Do Not Recommend")
        for r in payload.get("restrictions", []):
            _p(f"• {r}")

        assum = payload["assumptions"]
        _h2("Assumptions & Limitations")
        _p(assum.get("completeness_note", ""))
        for a in assum.get("assumptions", []):
            _p(f"• {a}")

        disc = payload["disclosures"]
        _h2("Regulatory Disclosures")
        _p(disc["sebi_compliance"])
        _p(disc["ai_usage"])
        _p(disc["disclaimer"])

        doc.build(story)
        return os.path.abspath(output_path)

    except ImportError:
        raise RuntimeError(
            "No PDF library available. Install weasyprint (recommended) or reportlab:\n"
            "  pip install weasyprint\n"
            "  pip install reportlab"
        )
