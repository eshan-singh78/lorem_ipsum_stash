# InvestorDNA v2 — Production-Grade Refactor

## Architecture

```
INPUT (free-text paragraph)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 1: EXTRACTION                    LLM Call #1      │
│  • Rule extraction (income, EMI, savings) — deterministic│
│  • Single LLM call for interpretive fields              │
│  • Rule-wins merge                                      │
│  Output: ExtractedFields                                │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 2: SIGNAL EXTRACTION             LLM Call #2      │
│  • Flat schema (21 fields, no deep nesting)             │
│  • num_predict=768 (was 2048)                           │
│  • All enums validated post-parse                       │
│  • Single source of truth for behavioral signals        │
│  Output: Signals                                        │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 3: SCORING                       ZERO LLM CALLS   │
│  • Axis 1 (Risk): from loss_response + dominant_trait   │
│  • Axis 2 (Cashflow): from income_type + emergency fund │
│  • Axis 3 (Obligation): from emi_ratio + dependents     │
│  • Axis 4 (Context): from experience + knowledge        │
│  • Capacity: cashflow × (1 - obligation/100)            │
│  • Archetype: deterministic matrix                      │
│  • Equity ceiling: deterministic from scores + flags    │
│  Output: Scores                                         │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 4: DECISION                      LLM Call #3      │
│  • Allocation numbers already set by scoring (no LLM)   │
│  • LLM writes: reasoning, advisor_note, first_step      │
│  • Guardrails applied deterministically after LLM       │
│  • Fallback: conservative defaults if LLM fails         │
│  Output: DecisionOutput                                 │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 5: REPORT                        ZERO LLM CALLS   │
│  • Deterministic aggregation                            │
│  • Provisional flag set explicitly                      │
│  • System language stripped                             │
│  Output: Report → JSON                                  │
└─────────────────────────────────────────────────────────┘
```

## LLM Call Reduction

| Version | LLM Calls (happy path) | LLM Calls (max with retries) |
|---------|------------------------|-------------------------------|
| v1      | 9                      | 15+                           |
| v2      | 3                      | 9 (3 calls × 2 retries each)  |

## What Was Removed

| v1 Component | Why Removed |
|---|---|
| `narrative_layer.py` | LLM call that re-derived signals already in `signal_extraction.py`. Conflicting source. |
| `state_synthesis.py` | LLM call to synthesize a "compound state" that was then fed into another LLM. Redundant. |
| `validation_layer.py` | LLM validating LLM output. Circular. Replaced with deterministic guardrails. |
| `_query_axis_calibration()` | LLM injecting ±30pt random adjustments into deterministic scoring. Root cause of inconsistency. |
| `_query_state_weights()` | LLM injecting random multipliers into category scoring. Same problem. |
| `score_sanity.py` | LLM checking LLM scores. Circular. Not called in v1 main.py anyway. |
| `cross_axis.py` | Merged into `report.py`. Was pure aggregation with no new logic. |
| `context_categories.py` | Merged into `scoring.py`. Category scoring is now direct axis scoring. |
| `profile_context.py` | Merged into `scoring.py`. Signal-to-flag mapping is now inline. |
| `reasoning_validator.py` | Replaced by deterministic guardrails in `decision.py`. |
| `constraint_engine.py` | Replaced by `_apply_guardrails()` in `decision.py`. |
| `judgment_layer.py` | Already deleted in v1. |

## What Is Rule-Based vs LLM-Based

### Rule-Based (deterministic, same input → same output always)
- `monthly_income`, `emi_amount`, `emergency_months` extraction (regex)
- `income_type` detection (keyword matching)
- `emi_ratio` computation (arithmetic)
- All axis scores (Axes 1-4)
- Financial capacity (derived from axes)
- Equity ceiling
- Archetype assignment (2×2 matrix + override conditions)
- Allocation range construction
- Allocation mode selection
- Guardrail enforcement (R1-R5)
- Report assembly

### LLM-Based (3 calls)
- **Call #1**: Interpretive field extraction (loss_reaction, risk_behavior, dependents, etc.)
- **Call #2**: Behavioral signal extraction (flat schema, 21 fields)
- **Call #3**: Reasoning narrative (explains pre-computed allocation — does NOT set numbers)

## Non-Determinism Eliminated

| v1 Source of Variance | v2 Fix |
|---|---|
| `_query_axis_calibration()` returning `risk_adjustment` ±30 | Removed. Risk adjustment is deterministic from `dominant_trait`. |
| `_query_state_weights()` returning random multipliers | Removed. Weights are fixed constants. |
| `_assign_archetype()` preferring LLM-derived archetype | Removed. Archetype is always from deterministic matrix. |
| `validate_scores_vs_decision()` LLM call | Removed. Replaced with deterministic checks. |
| `generate_narrative()` affecting equity ceiling | Removed. Equity ceiling is purely from scores. |
| `synthesize_state()` affecting dominant_trait | Removed. `dominant_trait` comes directly from signals. |

## Archetype Enum

```
Strategist          — high sophistication + low loss aversion
Analyst             — high sophistication + high loss aversion
Explorer            — low sophistication + low loss aversion (naive risk)
Guardian            — low sophistication + high loss aversion
Constrained Builder — obligation > 70 or capacity < 20 (overrides matrix)
Transitional        — behavioral shift detected
Crisis Mode         — grief/crisis state (overrides everything)
Provisional         — insufficient behavioral data
```

## Fallback Behavior

| Failure | v1 Behavior | v2 Behavior |
|---|---|---|
| Signal extraction fails | Full fallback, generic report | `signals.valid=False`, scoring uses extraction fields only, marked provisional |
| Decision LLM fails | "Unclassified" archetype, fallback decision | Archetype from deterministic scoring, fallback narrative, marked provisional |
| All fields null | "Guardian" fallback | `Provisional` archetype, explicit warning |
| Non-English input | Silent fallback | Explicit `status=failed`, `reason=non_english_input` |

## Running

```bash
# Local Ollama
python pipeline.py -p "I earn 8 LPA, have 2 dependents..."

# With verbose output
python pipeline.py -p "..." --verbose

# From file
python pipeline.py -f investor_profile.txt

# Override provider
python pipeline.py -p "..." --provider openrouter --model meta-llama/llama-3.1-8b-instruct:free

# Environment variables (recommended for production)
export LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=your-key-here
python pipeline.py -p "..."
```

## Configuration

Use environment variables — never put API keys in `llm_config.json`:

```bash
export LLM_PROVIDER=ollama          # ollama | ollama_cloud | openrouter
export LLM_MODEL=llama3.1:8b
export OLLAMA_BASE_URL=http://localhost:11434
export LLM_TIMEOUT=120              # always set a timeout
```

`llm_config.json` is gitignored and should only be used for local development.
