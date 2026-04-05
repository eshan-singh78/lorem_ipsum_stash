# InvestorDNA Profiling Engine — v8

A meaning-driven behavioral investor profiling system built for the Indian market. Takes a free-text investor description and produces a structured 4-axis profile, archetype classification, suitability assessment, and advisor-grade narratives.

**v8 key change:** The system no longer treats scores as truth. It interprets context first (via LLM meaning extraction), applies judgment overrides when context demands, and distinguishes temporary vs permanent behavior via state divergence detection.

---

## Quick Start

```bash
pip install -r requirements.txt

# Run with inline text
python main.py -p "Ravi earns 18 LPA at TCS. Home loan EMI 25000/month. Has 2L in FD. Worried about market crashes, considered stopping SIP during last correction."

# Run from file
python main.py -f investor_description.txt

# Verbose mode (shows each pipeline stage)
python main.py -p "..." --verbose
```

**Requirement:** [Ollama](https://ollama.ai) running locally with `llama3.1:8b` pulled.

```bash
ollama pull llama3.1:8b
ollama serve   # starts on http://localhost:11434
```

---

## Architecture Overview

```
Raw Text Input
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 1 — EXTRACTION  (extraction.py)                  │
│  normalize_text → rule_extraction → future_intent       │
│  → single LLM call (llama3.1:8b, rule context injected) │
│  → rule-wins merge → post-merge invariants              │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2–3 — VALIDATION + DERIVED FIELDS (validation.py)│
│  Type casting, range checks, emi_ratio computation      │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 4 — PROFILE CONTEXT  (profile_context.py)        │
│  Single boundary: validated fields read here only.      │
│  Detects life_events, cultural_signals, behavioral_     │
│  signals, city_tier, grief_state, peer_driven, etc.     │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 5 — MEANING LAYER  (meaning_layer.py)  ← NEW v8  │
│                                                         │
│  LLM (llama3.1:8b) interprets the investor's situation. │
│  Does NOT extract fields — interprets context.          │
│                                                         │
│  Output (MeaningOutput):                                │
│  → life_situation: independent_individual |             │
│       family_dependent | primary_financial_provider |   │
│       crisis_driven_responsibility                      │
│  → financial_state: stable | constrained | unstable     │
│  → psychological_state: stable | cautious |             │
│       stressed | grief_impacted                         │
│  → decision_mode: independent | influenced |            │
│       reactive | analytical                             │
│  → constraint_level: low | moderate | high              │
│  → state_type: stable | transitional | crisis           │
│  → bias_flags: [loss_aversion, recency_bias,            │
│       herd_behavior, overconfidence, none]              │
│  → score_reliability: high | medium | low               │
│  → confidence_notes: "..."                              │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 6 — STATE MODEL  (state_model.py)  ← NEW v8      │
│                                                         │
│  Detects baseline vs current risk divergence.           │
│  "used to invest aggressively but now avoids risk"      │
│  → baseline_risk=high, current_risk=low, divergence=True│
│                                                         │
│  Output (StateModel):                                   │
│  → baseline_risk: high | medium | low | unknown         │
│  → current_risk:  high | medium | low | unknown         │
│  → state_divergence: bool                               │
│  → divergence_reason: "..."                             │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 7a — CATEGORY ASSESSMENT (context_categories.py) │
│  6 categories from ProfileContext:                      │
│  income_stability, emergency_preparedness, debt_burden, │
│  dependency_load, cultural_obligation, behavioral_risk  │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 7b — AXIS SCORING  (axis_scoring.py)             │
│  4 axes from CategoryAssessment + ProfileContext:       │
│  Axis1=Risk, Axis2=Cashflow, Axis3=Obligation,          │
│  Axis4=Context, FinancialCapacity=derived               │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 8 — JUDGMENT LAYER  (judgment_layer.py) ← NEW v8 │
│                                                         │
│  Treats axis scores as INPUTS, not truth.               │
│  Applies safe overrides when context demands.           │
│                                                         │
│  RULE 1 — Low reliability:  risk −25%, mark provisional │
│  RULE 2 — Crisis state:     cap risk ≤ 30               │
│  RULE 3 — Transitional:     widen interpretation,       │
│                             mark for reassessment       │
│  RULE 4 — High constraint:  obligation +15%, cap equity │
│  RULE 5 — State divergence: mark unstable, no long-term │
│                                                         │
│  All caps use min(original, cap) — never blind overwrite│
│                                                         │
│  Output (JudgmentOutput):                               │
│  → final_axis: { risk, cashflow, obligation,            │
│       context, financial_capacity }                     │
│  → overrides_applied: [RULE1_..., RULE2_..., ...]       │
│  → score_reliability: high | medium | low               │
│  → is_provisional: bool                                 │
│  → judgment_summary: advisor-grade reasoning paragraph  │
│  → reassessment_recommended: bool                       │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 9 — CROSS-AXIS UNION  (cross_axis.py)            │
│                                                         │
│  Uses judgment.final_axis (post-override) scores.       │
│  Consumes meaning.state_type + meaning.constraint_level │
│  for suitability narrative.                             │
│                                                         │
│  Archetype, mismatch, binding constraint, suitability,  │
│  advisor_narrative, investor_narrative, drift_expectation│
│                                                         │
│  Suitability output is advisor-grade narrative:         │
│  "This investor is in a transitional emotional state    │
│   with high obligation load. Current risk tolerance is  │
│   suppressed and may not reflect baseline behavior.     │
│   Recommend conservative allocation with reassessment   │
│   in 3–6 months."                                       │
└─────────────────────────────────────────────────────────┘
     │
     ▼
  JSON Output
```

---

## File Structure

```
investor_profiler/
├── main.py                 Pipeline orchestrator — run this
├── extraction.py           Stage 1: rules + single LLM call + merge
├── field_registry.py       Field ownership, FieldValue dataclass, confidence rules
├── validation.py           Stage 2–3: type casting, derived fields
├── profile_context.py      Stage 4: ProfileContext builder
├── meaning_layer.py        Stage 5: LLM behavioral interpretation (NEW v8)
├── state_model.py          Stage 6: baseline vs current risk divergence (NEW v8)
├── context_categories.py   Stage 7a: 6 category assessors
├── axis_scoring.py         Stage 7b: 4-axis scorer
├── judgment_layer.py       Stage 8: context-driven override engine (NEW v8)
├── cross_axis.py           Stage 9: union engine + advisor-grade narratives
└── requirements.txt
```

---

## Field Ownership

Every field has exactly one authoritative source. The merge is deterministic — no arbitration.

| Field | Owner | How extracted |
|-------|-------|---------------|
| `monthly_income` | Rule | LPA/range regex → ÷12 |
| `emi_amount` | Rule | "EMI of ₹X" regex |
| `emergency_months` | Rule | savings ÷ income |
| `income_type` | Rule | gig/business keywords, LPA → salaried |
| `loss_reaction` | LLM | panic/cautious/neutral/aggressive |
| `risk_behavior` | LLM | low/medium/high |
| `decision_autonomy` | LLM | true/false |
| `financial_knowledge_score` | LLM | 1–5 rubric |
| `experience_years` | LLM | decimal years |
| `dependents` | LLM | integer count |
| `near_term_obligation_level` | LLM | none/moderate/high |
| `obligation_type` | LLM | wedding/house/education/medical/family/other |
| `emi_ratio` | Derived | emi_amount / monthly_income × 100 |
| `future_obligation_score` | Derived | intent detection score |

**Rule fields always win in merge.** LLM is told the rule values upfront and instructed not to re-derive them.

---

## The LLM Call

One call per request. No correction pipeline. No second model.

- **Model:** `llama3.1:8b` via Ollama
- **Endpoint:** `http://localhost:11434/api/generate`
- **Temperature:** 0 (deterministic)
- **Max tokens:** 512
- **Timeout:** None (waits indefinitely)
- **Retries:** 2 attempts on failure
- **Fallback:** If both attempts fail, LLM fields are all null; rule fields still populate the profile

The prompt injects rule-extracted values as `GROUND TRUTH` so the model never re-derives numeric fields. It only handles interpretive/semantic fields.

---

## Post-Merge Invariants (Code-Enforced)

These run after merge, in Python — not in the prompt.

| Invariant | Condition | Action |
|-----------|-----------|--------|
| Panic override | `loss_reaction == "panic"` AND `risk_behavior in ("medium", "high")` | Force `risk_behavior = "low"` |
| Novice cap | `experience_years < 1` AND `financial_knowledge_score > 2` | Cap `financial_knowledge_score = 2` |
| Obligation cleanup | `near_term_obligation_level == "none"` AND `obligation_type != null` | Clear `obligation_type = null` |

---

## Confidence System

Every field carries a confidence level that flows through the pipeline.

| Source | Confidence | Rationale |
|--------|-----------|-----------|
| `rule` | `medium` | Deterministic regex, not infallible |
| `llm` (interpretive) | `low` | Semantic judgment, inherently uncertain |
| `llm` (numeric/factual) | `high` | Explicit value stated in text |
| `llm` (knowledge=3) | `low` | Ambiguous default guard |
| `derived` | `medium` | Computed from other fields |
| `null` (any source) | `low` | Invariant: null never has high confidence |

---

## Output Structure

```json
{
  "profile_context": { "demographics", "financial_snapshot", "experience",
                       "behavioral", "obligation", "life_events",
                       "cultural_signals", "behavioral_signals", "flags" },
  "meaning": {
    "life_situation":     "crisis_driven_responsibility",
    "financial_state":    "constrained",
    "psychological_state": "grief_impacted",
    "decision_mode":      "reactive",
    "constraint_level":   "high",
    "state_type":         "crisis",
    "bias_flags":         ["loss_aversion"],
    "score_reliability":  "low",
    "confidence_notes":   "..."
  },
  "state_model": {
    "baseline_risk":    "medium",
    "current_risk":     "low",
    "state_divergence": true,
    "divergence_reason": "..."
  },
  "category_scores": {
    "income_stability":       { "score", "label", "reason", "modifiers", "data_available" },
    "emergency_preparedness": { ... },
    "debt_burden":            { ... },
    "dependency_load":        { ... },
    "cultural_obligation":    { ... },
    "behavioral_risk":        { ... }
  },
  "axis_scores": {
    "raw":   { "risk", "cashflow", "obligation", "context", "financial_capacity" },
    "final": { "risk", "cashflow", "obligation", "context", "financial_capacity" }
  },
  "judgment": {
    "final_axis":              { "risk", "cashflow", "obligation", "context", "financial_capacity" },
    "overrides_applied":       ["RULE1_LOW_RELIABILITY", "RULE2_CRISIS_STATE"],
    "score_reliability":       "low",
    "is_provisional":          true,
    "judgment_summary":        "This investor is in a crisis state...",
    "reassessment_recommended": true
  },
  "cross_axis": {
    "archetype":             "The Guardian (Crisis Variant)",
    "archetype_description": "...",
    "mismatch":              { "type", "severity", "description", "implication" } or null,
    "binding_constraint":    { "type", "description", "priority_actions" } or null,
    "suitability": {
      "classification":    "This investor is in a crisis state with a grief-impacted emotional state...",
      "equity_ceiling_pct": 10,
      "guidance":          [ "..." ]
    },
    "advisor_narrative":  "...",
    "investor_narrative": "...",
    "suitability_insights": [ "Provisional decision — scores adjusted for emotional/crisis state" ],
    "drift_expectation":  "..."
  },
  "final_decision":     "...",
  "advisor_narrative":  "...",
  "investor_narrative": "...",
  "confidence_score":   0–100,
  "data_completeness":  0–100,
  "extracted_data":     { field: { "value", "confidence", "source" } }
}
```

---

## The 4 Axes

### Axis 1 — Risk Appetite (1–99)
Measures psychological willingness to take risk. Derived from `behavioral_risk` category, modulated by grief state and recency bias flags. Can be `null` if no behavioral data is present.

### Axis 2 — Cash Flow Stability (1–99)
Measures how predictable and resilient the investor's income is. Weighted combination of `income_stability` (65%) and `emergency_preparedness` (35%).

### Axis 3 — Obligation Burden (1–99)
Measures total financial commitments. Weighted combination of `debt_burden` (40%), `dependency_load` (30%), and `cultural_obligation` (30%). This is the India moat — captures joint family obligations, wedding savings, religious giving, and elder care that no Western instrument measures.

### Axis 4 — Investor Context (1–99)
Measures financial sophistication and decision quality. Driven by experience years, knowledge score, decision autonomy, and city tier. Penalized for peer-driven decisions.

### Financial Capacity (derived)
```
financial_capacity = Axis2 × (1 − Axis3 / 100)
```
This is the actual investable capacity — what remains after obligations are met. It is the primary suitability driver, not Axis 1.

---

## Archetypes

| Axis 4 (Sophistication) | Axis 1 (Risk Appetite) | Archetype |
|------------------------|----------------------|-----------|
| High (≥55) | High (≥55) | The Strategist |
| High (≥55) | Low (<55) | The Analyst |
| Low (<55) | High (≥55) | The Explorer ⚠️ |
| Low (<55) | Low (<55) | The Guardian |

The Explorer is the most dangerous profile — low sophistication + high apparent risk appetite = naive risk-taking, not informed confidence. The system flags this explicitly and caps equity exposure.

Modifiers appended to archetype name:
- `(Crisis Variant)` — grief state detected
- `(Constrained)` — obligation burden ≥ 70

---

## Cultural Obligation Category

This is what separates InvestorDNA from Western risk profiling tools. The `cultural_obligation` category captures:

- **Joint family obligations** — financial responsibility extends beyond nuclear family
- **Hidden wedding savings** — often undisclosed, reduces investable surplus
- **Religious/charitable giving** — dharmic commitments, non-negotiable fixed outflows
- **Elder care** — parents/in-laws without insurance = maximum financial vulnerability
- **Social pressure** — community expectations create implicit financial obligations

These are treated as **fixed constraints** — the advisor cannot restructure them. They must be factored into investable surplus before any product recommendation.

---

## Debug Visibility

Every request produces a full audit trail in the `debug` key:

```
debug.rule_fields    — what rules extracted (before LLM)
debug.llm_fields     — what LLM extracted
debug.merge_log      — field-by-field merge decisions with reasons
debug.invariant_log  — post-merge corrections applied
debug.axis_reasons   — scoring rationale for each axis
debug.future_events_detected — intent-gated future obligations
```

This makes every scoring decision traceable and auditable.

---

## Design Principles

1. **Meaning before scoring.** The LLM interprets life situation, psychological state, and constraint level before any axis scoring happens. Context is not an afterthought.

2. **Scores are inputs, not truth.** The judgment layer treats axis scores as starting points. It overrides them when context demands — crisis state, low reliability, high constraint, state divergence.

3. **Safe overrides only.** All caps use `min(original, cap)` — never blind overwrite. The system degrades gracefully.

4. **Temporary vs permanent behavior.** The state model detects when current behavior diverges from baseline personality. A grief-driven shift to low risk is not the same as a permanently conservative investor.

5. **Rules own numbers, LLM owns interpretation.** Numeric fields (income, EMI, savings) are never trusted from the LLM. The LLM is told the rule values upfront.

6. **One LLM call per stage, two stages total.** Extraction uses one LLM call; meaning extraction uses a second. No correction pipeline, no retry-on-low-confidence.

7. **Invariants in code, not prompts.** Business rules like `panic → risk_behavior=low` are enforced deterministically after merge.

8. **Context before scoring.** ProfileContext captures life events, cultural signals, and behavioral patterns from raw text before any scoring. A grief-state investor is not the same as a baseline investor with the same loss_reaction score.

9. **Categories before axes.** No axis reads raw fields directly. Every axis score is derived from category assessments, which are derived from ProfileContext.

10. **Axes interact multiplicatively.** Financial capacity = cashflow × (1 − obligation/100). High risk appetite + high obligation = mismatch, not average.
