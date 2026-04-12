"""
Pipeline Orchestrator — InvestorDNA v2

Architecture (3 LLM calls max):
  INPUT
  → Stage 1: extract()          LLM Call #1 — fields (rules + single LLM)
  → Stage 2: extract_signals()  LLM Call #2 — behavioral signals
  → Stage 3: compute_scores()   DETERMINISTIC — zero LLM
  → Stage 4: generate_decision() LLM Call #3 — reasoning narrative only
  → Stage 5: build_report()     DETERMINISTIC — zero LLM
  → OUTPUT

Total: 3 LLM calls (happy path). Never more than 3 + 2 retries per call = 9 max.

Removed from v1:
  - narrative_layer (LLM call)
  - state_synthesis (LLM call)
  - validation_layer (LLM call)
  - axis_calibration (LLM call)
  - category_weight_query (LLM call)
  - cross_axis (merged into report.py)
  - judgment_layer (deleted in v1 already)
  - score_sanity (LLM call)
"""
from __future__ import annotations

import json
import sys
import threading
import time
import argparse

from llm_adapter import configure as configure_llm, get_config as get_llm_config
from extraction import extract, ExtractedFields
from signals import extract_signals, Signals, signals_to_dict
from scoring import compute_scores, scores_to_dict
from decision import generate_decision, decision_to_dict
from report import build_report, report_to_dict


# ---------------------------------------------------------------------------
# Stage logger
# ---------------------------------------------------------------------------

class _Logger:
    def __init__(self):
        self._stage = ""
        self._start = 0.0
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

    def start(self, stage: str):
        with self._lock:
            self._stage = stage
            self._start = time.time()
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._tick, daemon=True)
            self._thread.start()
        else:
            sys.stderr.write("\n")

    def _tick(self):
        sp = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while self._running:
            with self._lock:
                stage = self._stage
                elapsed = time.time() - self._start
            sys.stderr.write(f"\r{sp[i % len(sp)]}  {stage}  ({elapsed:.1f}s)   ")
            sys.stderr.flush()
            time.sleep(0.1)
            i += 1

    def done(self, label: str = ""):
        with self._lock:
            elapsed = time.time() - self._start
            lbl = label or self._stage
        sys.stderr.write(f"\r✓  {lbl}  ({elapsed:.1f}s)          \n")
        sys.stderr.flush()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)


_log = _Logger()


# ---------------------------------------------------------------------------
# Unified output wrapper
# ---------------------------------------------------------------------------

def _wrap(report_dict: dict, status: str, reason: str = "",
          confidence: str = "high", fallback_used: bool = False) -> dict:
    return {
        "status":  status,
        "report":  report_dict,
        "meta": {
            "reason":        reason,
            "confidence":    confidence,
            "fallback_used": fallback_used,
        },
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(paragraph: str, verbose: bool = False) -> dict:
    """
    3-call pipeline: extract → signals → scores → decision → report.
    """

    # -----------------------------------------------------------------------
    # Stage 1: Extraction (rules + LLM Call #1)
    # -----------------------------------------------------------------------
    _log.start("1/4 — Extraction")
    fields: ExtractedFields = extract(paragraph)
    _log.done()

    if fields.non_english:
        _log.stop()
        return _wrap(
            {"warning": "Non-English input — extraction skipped.",
             "archetype": "Provisional", "equity_range": "N/A",
             "confidence": "low", "is_provisional": True},
            status="failed", reason="non_english_input",
            confidence="low", fallback_used=True,
        )

    if verbose:
        print(f"  completeness={fields.data_completeness}%  missing={fields.missing_fields}")
        if fields.warning:
            print(f"  WARNING: {fields.warning}")

    # -----------------------------------------------------------------------
    # Stage 2: Signal Extraction (LLM Call #2)
    # -----------------------------------------------------------------------
    _log.start("2/4 — Signal extraction")
    signals: Signals = extract_signals(paragraph)
    _log.done()

    if verbose:
        print(f"  dominant_trait={signals.dominant_trait}  loss={signals.loss_response}"
              f"  resilience={signals.resilience_level}  valid={signals.valid}")
        if signals.warning:
            print(f"  WARNING: {signals.warning}")

    # -----------------------------------------------------------------------
    # Stage 3: Scoring (DETERMINISTIC — no LLM)
    # -----------------------------------------------------------------------
    _log.start("3/4 — Scoring (deterministic)")
    scores = compute_scores(fields, signals)
    _log.done()

    if verbose:
        print(f"  archetype={scores.archetype}  risk={scores.risk}  cashflow={scores.cashflow}"
              f"  obligation={scores.obligation}  context={scores.context}  capacity={scores.capacity}")

    # -----------------------------------------------------------------------
    # Stage 4: Decision (LLM Call #3 — narrative only)
    # -----------------------------------------------------------------------
    _log.start("4/4 — Decision (LLM narrative)")
    decision = generate_decision(fields, signals, scores)
    _log.done()

    if verbose:
        print(f"  allocation={decision.current_allocation}  mode={decision.allocation_mode}"
              f"  confidence={decision.confidence}  fallback={decision.fallback_used}")
        if decision.guardrail_notes:
            for note in decision.guardrail_notes:
                print(f"  [guardrail] {note}")
        if decision.warning:
            print(f"  WARNING: {decision.warning}")

    # -----------------------------------------------------------------------
    # Stage 5: Report assembly (DETERMINISTIC — no LLM)
    # -----------------------------------------------------------------------
    report = build_report(fields, signals, scores, decision)
    _log.stop()

    status     = "success" if not report.is_provisional else "partial"
    confidence = decision.confidence

    return _wrap(
        report_to_dict(report),
        status=status,
        reason="" if not report.is_provisional else "provisional_profile",
        confidence=confidence,
        fallback_used=decision.fallback_used,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="InvestorDNA v2 Pipeline")
    parser.add_argument("--paragraph", "-p", type=str)
    parser.add_argument("--file",      "-f", type=str)
    parser.add_argument("--verbose",   "-v", action="store_true")
    parser.add_argument("--provider",  type=str)
    parser.add_argument("--model",     type=str)
    parser.add_argument("--generate-pdf", action="store_true",
                        help="Generate a PDF report after pipeline completes")
    parser.add_argument("--pdf-name",  type=str, default=None,
                        help="Custom PDF filename (default: investor_report_<timestamp>.pdf)")
    args = parser.parse_args()

    if args.provider or args.model:
        configure_llm(provider=args.provider, model=args.model)

    if args.verbose:
        cfg = get_llm_config()
        print(f"[LLM] provider={cfg['provider']}  model={cfg['model']}  "
              f"url={cfg['base_url']}  timeout={cfg['timeout']}s")

    if args.file:
        with open(args.file) as fh:
            paragraph = fh.read().strip()
    elif args.paragraph:
        paragraph = args.paragraph
    else:
        print("Enter investor description (blank line to submit):")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        paragraph = " ".join(lines)

    result = run_pipeline(paragraph, verbose=args.verbose)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.generate_pdf:
        try:
            import sys as _sys, os as _os
            # Add parent dir so v1 report_formatter is importable
            _parent = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
            if _parent not in _sys.path:
                _sys.path.insert(0, _parent)
            from report_formatter import build_pdf_payload, render_pdf
            payload  = build_pdf_payload(result)
            filename = args.pdf_name or f"investor_report_{int(time.time())}.pdf"
            render_pdf(payload, filename)
            print(f"\n📄 PDF generated: {filename}", file=sys.stderr)
        except Exception as e:
            print(f"\n❌ PDF generation failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
