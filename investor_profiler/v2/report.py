"""
Report Layer — InvestorDNA v2
Pure deterministic aggregation. Zero LLM calls.

Assembles the final advisor-facing report from pipeline outputs.
Strips all system language. Marks provisional results clearly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from extraction import ExtractedFields
from signals import Signals
from scoring import Scores, equity_ceiling
from decision import DecisionOutput

# ---------------------------------------------------------------------------
# System language cleanup
# ---------------------------------------------------------------------------

_CLEAN_RULES = [
    (r"\bLLM\b",                  "system"),
    (r"\bguardrail\b",            "safety check"),
    (r"\bfallback\b",             "conservative default"),
    (r"\btrace\b",                "profile analysis"),
    (r"\bblocking violation\b",   "profile inconsistency"),
    (r"\bconstraint engine\b",    "risk assessment"),
]


def _clean(text: str) -> str:
    if not text:
        return ""
    for pat, rep in _CLEAN_RULES:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    return text.strip()


# ---------------------------------------------------------------------------
# Suitability guidance — deterministic from scores
# ---------------------------------------------------------------------------

def _guidance(sc: Scores, d: DecisionOutput) -> list[str]:
    ceiling = d.equity_ceiling_pct
    g = []
    if ceiling == 0:
        g.append("No equity exposure recommended at this time.")
        g.append("Focus: emergency fund, debt reduction, insurance.")
    elif ceiling <= 15:
        g.append(f"Maximum equity exposure: {ceiling}% (large-cap / index funds only).")
        g.append("Suitable: liquid funds, short-duration debt, FD.")
    elif ceiling <= 30:
        g.append(f"Maximum equity exposure: {ceiling}% via balanced advantage or conservative hybrid.")
        g.append("Avoid mid/small-cap and illiquid products.")
    elif ceiling <= 50:
        g.append(f"Equity up to {ceiling}% via flexi-cap or multi-cap funds.")
        g.append("Maintain a liquidity buffer of at least 3 months.")
    else:
        g.append(f"Equity-heavy allocation up to {ceiling}% appropriate.")
        g.append("Suitable: flexi-cap, mid-cap, international diversification.")

    if sc.grief_state:
        g.append("Grief state: no major allocation changes until reassessment.")
    if sc.recency_bias:
        g.append("Recency bias detected: do not increase market exposure until literacy baseline established.")
    if sc.peer_driven:
        g.append("Peer-influenced decisions: verify choices reflect personal capacity, not social pressure.")
    if sc.emerging_constraint:
        g.append("Emerging obligation: avoid illiquid or long lock-in products.")
    if sc.hidden_obligation:
        g.append("Hidden obligations detected: investable surplus may be lower than income suggests.")
    return g


def _do_not_recommend(sc: Scores, d: DecisionOutput) -> list[str]:
    dnr = []
    if d.equity_ceiling_pct < 20:
        dnr.append("Equity mutual funds above conservative allocation")
        dnr.append("Mid-cap, small-cap, or thematic funds")
    if sc.obligation > 70:
        dnr.append("Illiquid or long lock-in investment products")
        dnr.append("Aggressive or speculative equity instruments")
    if sc.grief_state:
        dnr.append("Any major financial commitment until reassessment")
    if sc.recency_bias:
        dnr.append("High-risk products until investment literacy is established")
    return dnr


def _key_risks(f: ExtractedFields, sc: Scores) -> list[str]:
    risks = []
    if f.emergency_months is not None and f.emergency_months < 3:
        risks.append(f"Insufficient emergency fund ({f.emergency_months:.1f} months — minimum 3 required).")
    if f.income_type in ("gig", "business"):
        risks.append("Income instability — irregular cash flow limits consistent investment capacity.")
    if sc.obligation > 70:
        risks.append("High obligation burden — financial commitments limit investable surplus.")
    if f.emi_ratio and f.emi_ratio > 40:
        risks.append(f"High EMI burden ({f.emi_ratio:.1f}% of income).")
    if sc.grief_state:
        risks.append("Grief state — current risk preferences are unreliable and temporary.")
    if sc.recency_bias:
        risks.append("Recency bias — high risk appetite with <1yr experience may not be stable.")
    if sc.peer_driven:
        risks.append("Peer-influenced decisions — stated preferences may not reflect actual capacity.")
    if not risks:
        risks.append("No critical financial risks identified from available data.")
    return risks


def _completeness_note(f: ExtractedFields) -> str:
    if f.data_completeness < 40:
        return (f"Profile completeness is low ({f.data_completeness}%). "
                f"Missing: {', '.join(f.missing_fields[:5])}. "
                "Recommendations are provisional.")
    if f.data_completeness < 70:
        return (f"Profile completeness is moderate ({f.data_completeness}%). "
                "Some fields were inferred or unavailable.")
    return f"Profile completeness: {f.data_completeness}%."


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class Report:
    # Executive
    archetype:         str
    equity_range:      str
    allocation_mode:   str
    confidence:        str
    is_provisional:    bool

    # Narrative (LLM-generated, cleaned)
    reasoning:         str
    advisor_note:      str
    first_step:        str
    reassessment_trigger: str

    # Structured
    key_risks:         list[str]
    guidance:          list[str]
    do_not_recommend:  list[str]
    completeness_note: str

    # Scores
    scores: dict

    # Signals summary
    signals: dict

    # Guardrail notes (internal — shown in debug only)
    guardrail_notes: list[str]

    # Warning (shown when provisional)
    warning: str | None


def build_report(
    f: ExtractedFields,
    s: Signals,
    sc: Scores,
    d: DecisionOutput,
) -> Report:
    is_provisional = d.fallback_used or d.confidence == "low" or f.data_completeness < 40

    return Report(
        archetype=sc.archetype,
        equity_range=d.current_allocation,
        allocation_mode=d.allocation_mode,
        confidence=d.confidence,
        is_provisional=is_provisional,
        reasoning=_clean(d.reasoning),
        advisor_note=_clean(d.advisor_note),
        first_step=_clean(d.first_step),
        reassessment_trigger=_clean(d.reassessment_trigger),
        key_risks=_key_risks(f, sc),
        guidance=_guidance(sc, d),
        do_not_recommend=_do_not_recommend(sc, d),
        completeness_note=_completeness_note(f),
        scores={
            "risk": sc.risk, "cashflow": sc.cashflow,
            "obligation": sc.obligation, "context": sc.context,
            "capacity": sc.capacity,
        },
        signals={
            "dominant_trait": s.dominant_trait if s.valid else "unknown",
            "loss_response":  s.loss_response  if s.valid else "unknown",
            "resilience":     s.resilience_level if s.valid else "unknown",
            "constraint":     s.constraint_level if s.valid else "unknown",
            "grief_state":    sc.grief_state,
            "peer_driven":    sc.peer_driven,
        },
        guardrail_notes=d.guardrail_notes,
        warning=(
            "PROVISIONAL — insufficient data or automated reasoning unavailable. "
            "Manual advisor review required before acting on this report."
            if is_provisional else None
        ),
    )


def report_to_dict(r: Report) -> dict:
    return {
        "archetype":          r.archetype,
        "equity_range":       r.equity_range,
        "allocation_mode":    r.allocation_mode,
        "confidence":         r.confidence,
        "is_provisional":     r.is_provisional,
        "warning":            r.warning,
        "reasoning":          r.reasoning,
        "advisor_note":       r.advisor_note,
        "first_step":         r.first_step,
        "reassessment_trigger": r.reassessment_trigger,
        "key_risks":          r.key_risks,
        "guidance":           r.guidance,
        "do_not_recommend":   r.do_not_recommend,
        "completeness_note":  r.completeness_note,
        "scores":             r.scores,
        "signals":            r.signals,
        "guardrail_notes":    r.guardrail_notes,
    }
