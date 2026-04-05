"""
Trace Store — InvestorDNA v16
Lightweight in-memory trace storage with file persistence.

Architecture:
  run_pipeline output → TraceStore → analysis utilities

Stores per-run records:
  - input text (truncated)
  - reasoning_trace
  - final_decision (allocations, dominant_trait, confidence)
  - violations from trace validation
  - retry_count, fallback_used

Keeps last MAX_TRACES records in memory.
Persists to JSONL file when flush() is called or MAX_TRACES reached.

analyze_traces() returns:
  - most common violations
  - most frequent dominant_traits
  - retry frequency distribution
  - fallback rate
"""

import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any

MAX_TRACES   = 500
_STORE_FILE  = os.path.join(os.path.dirname(__file__), "trace_store.jsonl")


# ---------------------------------------------------------------------------
# Record structure
# ---------------------------------------------------------------------------

@dataclass
class TraceRecord:
    timestamp: float
    input_preview: str          # first 200 chars of raw input
    dominant_trait: str
    resilience_level: str
    current_allocation: str
    baseline_allocation: str
    allocation_mode: str
    confidence: str
    retry_count: int
    fallback_used: bool
    violations: list[str]       # list of check names that fired
    warnings: list[str]
    reasoning_trace_summary: dict   # signals_considered, dominant_factors, state_inference


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class TraceStore:
    def __init__(self, max_traces: int = MAX_TRACES, store_file: str = _STORE_FILE):
        self._records: list[TraceRecord] = []
        self._max     = max_traces
        self._file    = store_file
        self._dirty   = False

    def record(
        self,
        raw_input: str,
        decision,
        trace_validation,
    ) -> None:
        """
        Store a trace record from a pipeline run.

        decision: DecisionOutput
        trace_validation: TraceValidationResult
        """
        violations = [v.check for v in (trace_validation.violations or [])]
        warnings   = list(trace_validation.warnings or [])

        rt = decision.reasoning_trace
        rec = TraceRecord(
            timestamp=time.time(),
            input_preview=raw_input[:200],
            dominant_trait=decision.state_context.dominant_trait,
            resilience_level=decision.state_context.resilience_level,
            current_allocation=decision.current_allocation,
            baseline_allocation=decision.baseline_allocation,
            allocation_mode=decision.allocation_mode,
            confidence=decision.confidence,
            retry_count=decision.retry_count,
            fallback_used=decision.fallback_used,
            violations=violations,
            warnings=warnings,
            reasoning_trace_summary={
                "signals_considered": rt.signals_considered[:5],  # cap for storage
                "dominant_factors":   rt.dominant_factors,
                "state_inference":    rt.state_inference,
            },
        )

        self._records.append(rec)
        self._dirty = True

        # Auto-flush when at capacity
        if len(self._records) >= self._max:
            self.flush()
            self._records = self._records[-50:]  # keep last 50 in memory

    def flush(self) -> None:
        """Append all dirty records to the JSONL store file."""
        if not self._dirty or not self._records:
            return
        try:
            with open(self._file, "a", encoding="utf-8") as f:
                for rec in self._records:
                    f.write(json.dumps(asdict(rec)) + "\n")
            self._dirty = False
        except OSError:
            pass  # non-fatal — trace storage is best-effort

    def analyze_traces(self) -> dict:
        """
        Analyze traces from both in-memory and JSONL file.
        Returns patterns useful for debugging and improvement.
        """
        # Load all records from JSONL file
        file_records: list[dict] = []
        try:
            if os.path.exists(self._file):
                with open(self._file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                file_records.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
        except OSError:
            pass

        # Merge: deduplicate by timestamp (in-memory takes precedence)
        in_memory_timestamps = {r.timestamp for r in self._records}
        merged_dicts = [r for r in file_records if r.get("timestamp") not in in_memory_timestamps]
        merged_dicts += [asdict(r) for r in self._records]

        if not merged_dicts:
            return {"message": "No traces recorded yet."}

        violation_counter  = Counter()
        dominant_counter   = Counter()
        retry_counter      = Counter()
        confidence_counter = Counter()
        fallback_count     = 0
        violation_run_count = 0

        for rec in merged_dicts:
            violations = rec.get("violations", [])
            for v in violations:
                violation_counter[v] += 1
            if violations:
                violation_run_count += 1
            dominant_counter[rec.get("dominant_trait", "unknown")] += 1
            retry_counter[rec.get("retry_count", 0)] += 1
            confidence_counter[rec.get("confidence", "unknown")] += 1
            if rec.get("fallback_used"):
                fallback_count += 1

        total = len(merged_dicts)
        fallback_rate = round(fallback_count / total, 3)

        return {
            "total_traces":          total,
            "fallback_rate":         fallback_rate,
            "fallback_rate_warning": (
                "WARNING: fallback_rate exceeds 5% threshold"
                if fallback_rate > 0.05 else None
            ),
            "violation_rate":        round(violation_run_count / total, 3),
            "most_common_violations": violation_counter.most_common(10),
            "dominant_trait_distribution": dominant_counter.most_common(10),
            "retry_distribution":    dict(sorted(retry_counter.items())),
            "confidence_distribution": dict(confidence_counter),
            "avg_retry_count":       round(
                sum(r * c for r, c in retry_counter.items()) / total, 2
            ),
        }

    def get_recent(self, n: int = 10) -> list[dict]:
        """Return the n most recent trace records as dicts."""
        return [asdict(r) for r in self._records[-n:]]

    def __len__(self) -> int:
        return len(self._records)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store = TraceStore()


def record_trace(raw_input: str, decision, trace_validation) -> None:
    """Module-level convenience function."""
    _store.record(raw_input, decision, trace_validation)


def analyze_traces() -> dict:
    """Module-level convenience function."""
    return _store.analyze_traces()


def get_recent_traces(n: int = 10) -> list[dict]:
    return _store.get_recent(n)


def flush_traces() -> None:
    _store.flush()
