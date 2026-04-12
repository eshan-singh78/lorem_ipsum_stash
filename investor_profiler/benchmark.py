#!/usr/bin/env python3
"""
InvestorDNA Benchmark Runner

Runs the FULL v1 pipeline (main.py) for every model × profile combination.
The v1 flow is unchanged — all stages, all data, full RIA PDF output.

What v2 contributes:
  • Model list  — Ollama Cloud models from v2/llm_config.json
  • Three flags — --v2-signals --v2-scoring --v2-decision
    (better signal schema, deterministic scoring, deterministic validation)
  • Nothing else changes — narrative, state synthesis, decision engine,
    cross-axis, report formatter, PDF — all v1.

Usage:
    python benchmark.py                          # all models × all profiles
    python benchmark.py --dry-run                # print plan, no execution
    python benchmark.py --models qwen3.5-397b    # one model
    python benchmark.py --profiles rohan priya   # subset of profiles
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR  = Path(__file__).parent           # investor_profiler/
MAIN_PY     = SCRIPT_DIR / "main.py"          # v1 entry point — always used
# The benchmark writes only the "model" field here before each run.
# llm_adapter.py merges root llm_config.json + this file, so the model
# switch propagates into every LLM call inside main.py automatically.
CONFIG_PATH = SCRIPT_DIR / "v2" / "llm_config.json"
RESULTS_DIR = SCRIPT_DIR / "results"

# ---------------------------------------------------------------------------
# Models (Ollama Cloud)
# ---------------------------------------------------------------------------

MODELS = [
    {"id": "cogito-2.1:671b-cloud",  "slug": "cogito-2.1-671b"},
    {"id": "gpt-oss:120b-cloud",     "slug": "gpt-oss-120b"},
    {"id": "nemotron-3-super:cloud", "slug": "nemotron-3-super"},
    {"id": "qwen3.5:397b-cloud",     "slug": "qwen3.5-397b"},
    {"id": "gemma4:31b-cloud",       "slug": "gemma4-31b"},
]

# ---------------------------------------------------------------------------
# Investor profiles
# ---------------------------------------------------------------------------

PROFILES = {
    "rohan": (
        "Rohan is a 22-year-old independent contractor currently earning around ₹4–5 lakh per year "
        "through freelance and project-based work, with income that can vary month to month. He lives "
        "with his family and does not have any fixed financial responsibilities or loans at present. "
        "His spending is mostly discretionary, including frequent food orders, gadgets, and occasional "
        "travel. He does not have any formal savings, investments, SIPs, or fixed deposits yet. "
        "Over the next year, he plans to move out and rent a 3BHK apartment where he also wants to "
        "set up his own office space. He is also considering purchasing a car under ₹20 lakh in the "
        "near future. His long-term goal is to become financially stable and settled by the age of "
        "32–33. He has not actively participated in financial markets so far and does not have prior "
        "investing experience. His financial decisions are currently self-driven but largely "
        "unstructured, and he has not yet developed a clear investment strategy or discipline around "
        "saving and budgeting."
    ),
    "ananya": (
        "Ananya Gupta is a 34-year-old self-employed boutique owner in Jaipur earning roughly "
        "₹6–8 lakh annually depending on seasonal sales; her income fluctuates significantly across "
        "months, with peak earnings during wedding seasons and low cash flow during off months. She "
        "is married with one young child and financially supports her parents occasionally. She does "
        "not have any formal investments but keeps around ₹2.5 lakh in a savings account and invests "
        "intermittently in gold jewellery and chit funds recommended by relatives. She has no formal "
        "loans but often uses informal borrowing from family during low-income months. Recently, she "
        "started a ₹3,000 monthly SIP in a balanced mutual fund after seeing a friend's portfolio but "
        "feels unsure about market movements and tends to pause contributions when business income "
        "drops. She does not track markets actively but becomes worried when she hears about losses "
        "from others and prefers 'safe' options suggested by family. She plans to expand her boutique "
        "within the next year, which may require a significant investment, and also wants to save for "
        "her child's education. While she is comfortable using WhatsApp and UPI, she avoids reading "
        "financial documents and relies heavily on advice from her husband and relatives for financial "
        "decisions."
    ),
    "priya": (
        "Priya Sharma is a 26-year-old woman working in an IT services company in Indore, earning "
        "about 7.5 LPA and living with her parents; she is unmarried, primarily Hindi-speaking with "
        "professional English fluency, and highly comfortable with digital platforms like Groww and "
        "Instagram finance content. She began investing 8 months ago with a ₹5,000 per month SIP "
        "into a large-cap fund via Groww, has no EMIs, keeps about ₹1.2 lakh in her savings account, "
        "and maintains an additional ₹80,000 in a recurring deposit that she is secretly earmarking "
        "for her own wedding without her parents' knowledge. She checks her NAV daily, feels anxious "
        "enough to have considered stopping her SIP after a 6% dip, frequently takes cues from Hindi "
        "YouTube finfluencers, and was originally motivated to start investing when a college friend "
        "posted impressive returns on Instagram, making peer comparison her main driver. Her parents "
        "strongly disapprove of the stock market and see it as gambling, so she does not talk to them "
        "about her investments, and although her financial literacy is low, her digital fluency is "
        "high, which allows her to act on peer and social media influences while silently carrying a "
        "culturally shaped sense of obligation to save for her wedding."
    ),
    "rohit": (
        "Rohit Verma is a 29-year-old male working as a mid-level operations manager in a logistics "
        "company in Nagpur, earning about 9 LPA; he is engaged, lives in a rented apartment with a "
        "friend, and plans to marry within the next 18 months. He is Hindi-primary with functional "
        "English, comfortable with UPI, banking apps, and YouTube but not very confident reading long "
        "English reports or blogs. He started two SIPs 14 months ago after his HR pushed an NPS and "
        "his colleague helped him open a demat account: ₹7,000 per month into a mid-cap mutual fund "
        "and ₹3,000 per month into a sectoral fund recommended by a popular Hindi YouTube finfluencer. "
        "He has an outstanding bike loan with an EMI of ₹4,500, keeps about ₹90,000 in his savings "
        "account, and has around ₹60,000 in a gold jewellery scheme in his mother's name, which he "
        "mentally counts as part of his wedding fund. During a recent 10–12% correction in mid-caps "
        "he shifted one SIP to a 'safer' large-cap fund on the advice of a colleague but did not stop "
        "investing, and he often checks portfolio value late at night on his phone, feeling stressed "
        "yet proud when it is higher than his friends' screenshots in their WhatsApp group. His "
        "parents strongly prefer fixed deposits and gold, view equities as risky 'timepass', and "
        "expect him to prioritise buying a small flat after marriage, which creates a quiet tension "
        "between his desire for faster growth and his fear of disappointing them."
    ),
}

# Archetypes that indicate a GENUINELY degraded result
# Guardian is a valid archetype — only flag it as fallback when combined with low confidence
FALLBACK_ARCHETYPES  = {"Provisional", "Unclassified"}
FALLBACK_CONFIDENCES = {"low"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log(msg: str):
    print(msg, flush=True)


def switch_model(model_id: str):
    """
    Write only the 'model' field to v2/llm_config.json.
    llm_adapter.py merges root llm_config.json + v2/llm_config.json,
    so this single-field write is enough to switch the active model
    for every LLM call inside main.py.
    """
    # Read existing v2 config to preserve any other fields
    existing: dict = {}
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass
    existing["model"] = model_id
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")


def ensure_dir(model_slug: str) -> Path:
    d = RESULTS_DIR / model_slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def find_new_pdf(since: float) -> Path | None:
    """
    Scan SCRIPT_DIR for any .pdf whose mtime >= since.
    main.py writes PDFs into its own working directory (SCRIPT_DIR).
    Returns the newest match, or None.
    """
    candidates = []
    for p in SCRIPT_DIR.glob("*.pdf"):
        try:
            if p.stat().st_mtime >= since:
                candidates.append(p)
        except OSError:
            pass
    # Also check one level deep (results/ subfolders excluded via name check)
    for p in SCRIPT_DIR.glob("**/*.pdf"):
        if "results" in p.parts:
            continue
        try:
            if p.stat().st_mtime >= since and p not in candidates:
                candidates.append(p)
        except OSError:
            pass
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def parse_json_output(stdout: str) -> dict:
    """
    Extract the JSON block from main.py stdout.

    main.py stdout structure:
        \\n============================================================
        INVESTORDNA PROFILE OUTPUT v7
        ============================================================
        { ...JSON... }

        📄 PDF generated: investor_report_1234567890.pdf

    Strategy:
      1. Find the first '{' (start of JSON)
      2. Find the last '}' before any non-JSON trailing lines
      3. Parse that slice directly — avoids O(n²) fallback loop
    """
    idx = stdout.find("{")
    if idx == -1:
        return {}

    # Find the last '}' in the string — JSON ends there
    # Trailing lines like "📄 PDF generated: ..." come after the closing brace
    last_brace = stdout.rfind("}")
    if last_brace == -1 or last_brace < idx:
        return {}

    candidate = stdout[idx:last_brace + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Fallback: the JSON itself may be malformed (truncated model output)
        # Try progressively shorter slices from the right — but cap iterations
        for end in range(last_brace, idx, -1):
            if stdout[end] != "}":
                continue
            try:
                return json.loads(stdout[idx:end + 1])
            except json.JSONDecodeError:
                continue
        return {}


def classify(result: dict, pdf_found: bool, returncode: int) -> str:
    """
    Classify a run as 'success', 'fallback', or 'failed'.

    'failed'  = pipeline crashed, JSON invalid, or subprocess error
    'fallback' = pipeline ran but produced a degraded/provisional result
    'success'  = pipeline ran and produced a confident, non-provisional result

    NOTE: PDF missing does NOT classify as 'failed' — it is tracked separately
    via pdf_generated=False. A run can succeed without a PDF (e.g. reportlab error).
    """
    # Hard failures: process crashed or produced no parseable output
    if returncode != 0 or not result:
        return "failed"

    report     = result.get("report", {})
    confidence = result.get("meta", {}).get("confidence", "")
    status     = result.get("status", "")
    fallback_used = result.get("meta", {}).get("fallback_used", False)

    # Archetype — check top-level first, then nested v1 path
    arch = report.get("archetype", "")
    if not arch:
        full  = report.get("full_profile", {}) or {}
        cross = full.get("cross_axis", {}) or {}
        arch  = cross.get("archetype", "") if isinstance(cross, dict) else ""

    # Explicit pipeline failure status
    if status == "failed":
        return "failed"

    # Degraded / provisional results
    if arch in FALLBACK_ARCHETYPES:
        return "fallback"
    if confidence in FALLBACK_CONFIDENCES:
        return "fallback"
    if status == "partial":
        return "fallback"
    if fallback_used:
        return "fallback"

    return "success"


# ---------------------------------------------------------------------------
# Single run — always main.py, always full v1 pipeline
# ---------------------------------------------------------------------------

def run_one(model: dict, profile_name: str, profile_text: str,
            out_dir: Path, dry_run: bool) -> dict:
    """
    Execute one benchmark run.

    Always calls:
        python main.py
            --paragraph  <profile_text>
            --generate-pdf
            --v2-signals   (flat signal schema — fixes truncation on large models)
            --v2-scoring   (deterministic scoring — removes 2 LLM calibration calls)
            --v2-decision  (deterministic validation — removes 1 LLM validation call)

    Everything else is pure v1:
        extraction → validation → narrative → state synthesis → decision engine
        → constraint engine → cross-axis → report formatter → PDF
    """
    stamp      = ts()
    model_id   = model["id"]
    model_slug = model["slug"]

    log(f"\n{'─' * 62}")
    log(f"  ▶  {model_slug}  /  {profile_name}  [{stamp}]")
    log(f"{'─' * 62}")

    if dry_run:
        log("  [dry-run] skipping")
        return {
            "model": model_id, "model_slug": model_slug,
            "profile": profile_name, "timestamp": stamp,
            "status": "dry_run", "latency_s": 0,
            "pdf_generated": False, "pdf_path": None,
            "json_path": None, "log_path": None, "errors": [],
            "archetype": "", "confidence": "", "pipeline_status": "",
        }

    # 1. Switch model in v2/llm_config.json
    switch_model(model_id)
    log(f"  model  → {model_id}")

    # 2. Snapshot time for PDF detection
    t_start      = time.time()
    before_mtime = t_start - 1.0   # 1s buffer

    # 3. Build command — full v1 pipeline + v2 improvement flags
    cmd = [
        sys.executable, str(MAIN_PY),
        "--paragraph", profile_text,
        "--generate-pdf",
        "--v2-signals",   # flat 21-field signal schema (768 tokens, no truncation)
        "--v2-scoring",   # deterministic axis scoring (no LLM calibration calls)
        "--v2-decision",  # deterministic validation (no LLM validation call)
    ]

    errors: list[str] = []
    stdout_text = stderr_text = ""
    returncode  = -1

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(SCRIPT_DIR),
            timeout=600,        # 10-minute hard ceiling per run
        )
        stdout_text = proc.stdout or ""
        stderr_text = proc.stderr or ""
        returncode  = proc.returncode
    except subprocess.TimeoutExpired:
        errors.append("Timed out after 600s")
    except Exception as exc:
        errors.append(f"Subprocess error: {exc}")

    latency = round(time.time() - t_start, 2)
    log(f"  done   → {latency}s  rc={returncode}")

    # 4. Parse JSON from stdout
    result = parse_json_output(stdout_text)
    if not result and stdout_text.strip():
        errors.append("JSON parse failed — stdout was not valid JSON")

    # 5. Detect and move PDF
    pdf_src  = find_new_pdf(before_mtime)
    pdf_dest = None
    if pdf_src:
        # FIX 1: timestamp captured AFTER PDF is found — reflects actual generation time
        pdf_stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        # FIX 2: guard against empty model_slug
        safe_slug  = model_slug or "unknown_model"
        pdf_name   = f"{safe_slug}_{profile_name}_{pdf_stamp}.pdf"
        pdf_dest   = out_dir / pdf_name
        try:
            shutil.move(str(pdf_src), str(pdf_dest))
            log(f"  PDF    → {pdf_dest.relative_to(SCRIPT_DIR)}")
            # FIX 3: explicit save confirmation
            print(f"  Saved PDF: {pdf_dest}", flush=True)
        except Exception as exc:
            errors.append(f"PDF move failed: {exc}")
            pdf_dest = None
    else:
        errors.append("No PDF generated")
        log("  ⚠  No PDF found")

    # 6. Save JSON output
    json_path = out_dir / f"{profile_name}_{stamp}.json"
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            if result:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            else:
                f.write(stdout_text)
        log(f"  JSON   → {json_path.relative_to(SCRIPT_DIR)}")
    except Exception as exc:
        errors.append(f"JSON save failed: {exc}")
        json_path = None

    # 7. Save stderr log
    log_path = out_dir / f"{profile_name}_{stamp}.log"
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=== STDERR ===\n")
            f.write(stderr_text)
            if errors:
                f.write("\n=== BENCHMARK ERRORS ===\n")
                f.write("\n".join(errors) + "\n")
        log(f"  LOG    → {log_path.relative_to(SCRIPT_DIR)}")
    except Exception as exc:
        errors.append(f"Log save failed: {exc}")
        log_path = None

    # 8. Classify
    status = classify(result, pdf_dest is not None, returncode)
    icon   = {"success": "✓", "fallback": "⚠", "failed": "✗"}.get(status, "?")
    log(f"  {icon}  {status}")
    for e in errors:
        log(f"     ! {e}")

    return {
        "model":           model_id,
        "model_slug":      model_slug,
        "profile":         profile_name,
        "timestamp":       stamp,
        "status":          status,
        "latency_s":       latency,
        "pdf_generated":   pdf_dest is not None,
        "pdf_path":        str(pdf_dest.relative_to(SCRIPT_DIR)) if pdf_dest else None,
        "json_path":       str(json_path.relative_to(SCRIPT_DIR)) if json_path else None,
        "log_path":        str(log_path.relative_to(SCRIPT_DIR)) if log_path else None,
        "errors":          errors,
        "archetype":       result.get("report", {}).get("archetype", ""),
        "confidence":      result.get("meta", {}).get("confidence", ""),
        "pipeline_status": result.get("status", ""),
        "fallback_used":   result.get("meta", {}).get("fallback_used", False),
        "equity_range":    result.get("report", {}).get("equity_range", ""),
        "data_completeness": result.get("report", {}).get("data_completeness", None),
        "dominant_trait":  result.get("report", {}).get("signals", {}).get("dominant_trait", ""),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(runs: list[dict]):
    log("\n" + "=" * 62)
    log("BENCHMARK SUMMARY")
    log("=" * 62)

    total    = len(runs)
    success  = sum(1 for r in runs if r["status"] == "success")
    fallback = sum(1 for r in runs if r["status"] == "fallback")
    failed   = sum(1 for r in runs if r["status"] == "failed")
    pdfs     = sum(1 for r in runs if r["pdf_generated"])
    avg_lat  = round(sum(r["latency_s"] for r in runs) / total, 1) if total else 0

    log(f"  Total runs  : {total}")
    log(f"  ✓ Success   : {success}")
    log(f"  ⚠ Fallback  : {fallback}")
    log(f"  ✗ Failed    : {failed}")
    log(f"  PDFs saved  : {pdfs}/{total}")
    log(f"  Avg latency : {avg_lat}s")
    log("")

    # Per-model row
    seen_slugs = list(dict.fromkeys(r["model_slug"] for r in runs))
    for slug in seen_slugs:
        mr  = [r for r in runs if r["model_slug"] == slug]
        s   = sum(1 for r in mr if r["status"] == "success")
        fb  = sum(1 for r in mr if r["status"] == "fallback")
        fx  = sum(1 for r in mr if r["status"] == "failed")
        lat = round(sum(r["latency_s"] for r in mr) / len(mr), 1)
        log(f"  {slug:<24}  ✓{s}  ⚠{fb}  ✗{fx}  avg={lat}s")

    log("")
    log(f"  {'MODEL':<24}  {'PROFILE':<8}  {'STATUS':<10}  {'LAT':>7}  {'PDF':<5}  {'CONF':<7}  ARCHETYPE")
    log(f"  {'-'*24}  {'-'*8}  {'-'*10}  {'-'*7}  {'-'*5}  {'-'*7}  {'-'*22}")
    for r in runs:
        icon = {"success": "✓", "fallback": "⚠", "failed": "✗", "dry_run": "–"}.get(r["status"], "?")
        pdf  = "yes" if r["pdf_generated"] else "no"
        arch = r.get("archetype", "")[:22]
        conf = r.get("confidence", "")[:7]
        log(f"  {r['model_slug']:<24}  {r['profile']:<8}  "
            f"{icon} {r['status']:<8}  {r['latency_s']:>6.1f}s  {pdf:<5}  {conf:<7}  {arch}")


def save_summary(runs: list[dict]):
    path = RESULTS_DIR / "results_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "pipeline":     "v1 full (main.py) + v2 flags (--v2-signals --v2-scoring --v2-decision)",
            "total_runs":   len(runs),
            "success":      sum(1 for r in runs if r["status"] == "success"),
            "fallback":     sum(1 for r in runs if r["status"] == "fallback"),
            "failed":       sum(1 for r in runs if r["status"] == "failed"),
            "pdfs_saved":   sum(1 for r in runs if r["pdf_generated"]),
            "runs":         runs,
        }, f, indent=2, ensure_ascii=False, default=str)
    log(f"\n  Summary → {path.relative_to(SCRIPT_DIR)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="InvestorDNA Benchmark — full v1 pipeline, Ollama Cloud models"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Run only these model slugs, e.g. --models qwen3.5-397b gemma4-31b",
    )
    parser.add_argument(
        "--profiles", nargs="+", default=None,
        help="Run only these profiles, e.g. --profiles rohan priya",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the run plan without executing anything",
    )
    args = parser.parse_args()

    # Filter
    models   = MODELS
    profiles = PROFILES

    if args.models:
        slugs  = {s.lower() for s in args.models}
        models = [m for m in MODELS
                  if m["slug"].lower() in slugs
                  or any(s in m["slug"].lower() for s in slugs)]
        if not models:
            log(f"ERROR: no models matched {args.models}")
            sys.exit(1)

    if args.profiles:
        names    = {p.lower() for p in args.profiles}
        profiles = {k: v for k, v in PROFILES.items() if k.lower() in names}
        if not profiles:
            log(f"ERROR: no profiles matched {args.profiles}")
            sys.exit(1)

    total = len(models) * len(profiles)

    log("=" * 62)
    log("InvestorDNA Benchmark Runner")
    log("=" * 62)
    log(f"  Pipeline : v1 full (main.py) — all stages, full RIA PDF")
    log(f"  v2 flags : --v2-signals  --v2-scoring  --v2-decision")
    log(f"  Models   : {[m['slug'] for m in models]}")
    log(f"  Profiles : {list(profiles.keys())}")
    log(f"  Runs     : {total}")
    log(f"  Config   : {CONFIG_PATH}")
    log(f"  Results  : {RESULTS_DIR}")
    if args.dry_run:
        log("  Mode     : DRY RUN — no execution")
    log("")

    # Pre-flight checks
    if not MAIN_PY.exists():
        log(f"ERROR: main.py not found at {MAIN_PY}")
        sys.exit(1)
    if not CONFIG_PATH.exists():
        log(f"ERROR: v2/llm_config.json not found at {CONFIG_PATH}")
        sys.exit(1)

    # Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for m in models:
        ensure_dir(m["slug"])

    # Execute
    all_runs: list[dict] = []
    for i, model in enumerate(models):
        out_dir = ensure_dir(model["slug"])
        for profile_name, profile_text in profiles.items():
            run_num = i * len(profiles) + list(profiles).index(profile_name) + 1
            log(f"\n[{run_num}/{total}]")
            metrics = run_one(
                model=model,
                profile_name=profile_name,
                profile_text=profile_text,
                out_dir=out_dir,
                dry_run=args.dry_run,
            )
            all_runs.append(metrics)

    print_summary(all_runs)
    if not args.dry_run:
        save_summary(all_runs)


if __name__ == "__main__":
    main()
