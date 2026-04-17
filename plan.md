# EM FX Trade Structuring & Sizing Tool — Build Plan

## Status: Phase D complete — 131 tests passing

---

## Phases

| Phase | Description | Status |
|---|---|---|
| A | Foundation — schemas, config system, synthetic data | **Done** |
| B | Pricing engine — vanilla, spreads, RR, exotics, scenario matrix | **Done** |
| C | Knowledge engine — conventions, structure selector, sizing, critique | **Done** |
| D | Conversation layer — LLM integration, context builder, flow controller | **Done** |
| E | Streamlit web app — chat UI, Plotly heatmaps, sizing cards | **Done** |
| E | CLI interface — REPL, session management | After D |

---

## What's built

### Foundation (`config/`, `data/`, `knowledge/`)

**Layered config system** — three layers merge at runtime:
1. `knowledge/defaults/*.json` — base judgment layer (Kelly fractions, vol thresholds, structure rules)
2. `config/user_profile.json` — persists user preferences across sessions
3. `SessionOverrides` (in-memory) — on-the-fly changes detected from conversation

`config/override_detector.py` parses `[PREF_CHANGE: {...}]` tags emitted by the LLM. `config/resolver.py` merges layers and populates `source_trace` so the tool can explain where any config value came from.

**Market data** — `data/market_snapshot.json` contains synthetic vol surfaces (5 deltas × 6 tenors), forward curves, and precomputed vol regime stats for USDBRL, USDTRY, EURPLN.

**Convention facts** — `knowledge/facts/` holds immutable NDF/deliverable conventions for each pair. Never overridden at runtime.

**Knowledge defaults** — `knowledge/defaults/` holds the full judgment layer: sizing rules (Kelly, vol-adj, stops, TP, tranches), vol regime thresholds, structure selection decision rules + catalog, critique framework. All editable as JSON with no code changes.

---

### Pricing engine (`pricing/`)

| Module | What it does |
|---|---|
| `forwards.py` | Tenor→years, forward interpolation, CIP rate derivation, `RateContext` |
| `black_scholes.py` | Black-76 GK: call/put, call spread, put spread, risk reversal |
| `rko.py` | Up-and-out call / down-and-out put — full Haug A-B+C-D formula |
| `scenario.py` | P&L matrix: spot × vol × time grid, plain-text formatter for LLM injection |
| `digital.py` | Cash-or-nothing digital call/put |
| `digital_rko.py` | Digital+RKO via finite-difference on the Haug formula |

**Key technical note:** The simplified reflection formula `c_uo = c - (H/S)^α × c_reflected` breaks down for large rate differentials (BRL carry ≈ 7.7%). The Haug formula is used throughout.

---

### Knowledge engine (`knowledge_engine/`)

| Module | What it does |
|---|---|
| `models.py` | `TradeView`, `StructureShortlistItem`, `SizingOutput`, `ResolvedConventions`, `CritiqueOutput` |
| `loader.py` | Reads all JSON files with `lru_cache`; clean accessors |
| `conventions.py` | `resolve(pair)` → typed conventions; `format_for_context()` → LLM-ready text |
| `structure_selector.py` | Deterministic rules engine: view + vol regime → ordered shortlist |
| `sizing_engine.py` | Kelly × vol-adj, notional from budget, vol-derived stops, tranches, TPs |
| `critique_engine.py` | Five-dimension evaluation: EV, scenario weakness, execution, gamma, hedge |

**Structure selector** evaluates all rules in order; conditions include direct fields (`direction_conviction`, `timing_conviction`) and derived fields (`budget_type`, `view_aligns_with_skew`, `skew_magnitude`, `target_level_known`). Direction-resolution maps generic IDs (`vanilla_call` + `base_lower` → `vanilla_put`). Session exclusions respected.

**Verified behaviours (from tests):**
- High conviction + topside skew + base_higher → SR-03 fires → `call_spread` prioritised
- Budget constrained + target known → SR-02 fires → spread structures
- Low direction conviction → SR-05 fires → spot/forward only
- BRL elevated vol → `vol_adjustment = 0.75` → sizing reduced
- PLN normal vol → `vol_adjustment = 1.0` → no reduction
- Session override (`quarter-Kelly`) propagates end-to-end through resolver → sizing engine
- Excluded structures never appear in shortlist
- RKO call correctly flagged as path-dependent in critique
- Digital correctly flagged for binary gamma in critique

---

## Phase D — Conversation layer (next)

### Files to build

```
conversation/
├── client.py          # Anthropic SDK wrapper (streaming optional)
├── context_builder.py # Assembles system prompt from pre-computed knowledge blocks
├── flow.py            # State machine: INTAKE → VALIDATION → REC → SIZING → EXIT
├── modes/
│   ├── recommend.py   # Four-step recommend orchestrator
│   └── critique.py    # Critique mode orchestrator
└── prompts/
    ├── system_base.txt
    ├── view_extraction.txt
    ├── view_validation.txt
    ├── structure_rec.txt
    ├── sizing.txt
    ├── entry_exit.txt
    └── critique.txt
```

### LLM's role at each step

The LLM **never does quantitative reasoning**. Pre-computed outputs are injected into the system prompt as structured text; the LLM narrates and explains them.

| Step | Knowledge engine runs | LLM does |
|---|---|---|
| INTAKE | — | Extract `TradeView` from PM's natural language; emit `[VIEW: {...}]` tag |
| VALIDATION | conventions.resolve(), vol regime note | Contextualise the view against what the market is pricing; flag crowded/differentiated |
| STRUCTURE_REC | structure_selector.select(), scenario matrix built for each shortlisted structure | Explain shortlist and scenario matrix in PM-facing terms; detect `[EXOTIC_REQUEST]` |
| SIZING | sizing_engine.compute_sizing() | Narrate Kelly fraction, vol adjustment, stop, tranches; explain the reasoning chain |
| ENTRY_EXIT | — (stop/TP already in sizing output) | Catalyst-based time exit; tie everything into a complete trade specification |
| CRITIQUE | critique_engine.evaluate_structure() | Lead with verdict; narrate each dimension specifically; apply same sizing logic |

### Key design constraint

`context_builder.py` is the separation boundary. Everything above it is deterministic Python. The LLM sees structured text blocks, never raw JSON or raw Python objects. `source_trace` is available for "why are you using X?" responses.

### Session override flow

```
PM says "use quarter-Kelly for this one"
  → LLM emits [PREF_CHANGE: {"field": "sizing.kelly.default_fraction", "value": 0.25, "scope": "session"}]
  → override_detector.extract_overrides() strips tag, returns SessionOverride
  → flow.py re-runs resolver with updated SessionOverrides
  → next step uses updated ResolvedConfig
  → LLM confirms the change in its response
```

---

## Phase E — Streamlit web app (after D)

Chat interface in `interface/app.py`. Shareable via Streamlit Community Cloud (free). All Python — no JS frontend.

```
interface/
├── app.py           # Streamlit entry point: chat loop, session state, streaming
├── charts.py        # Plotly renderers: scenario heatmap, sizing summary cards
└── session.py       # Wraps conversation/flow.py; bridges Streamlit state ↔ flow state
```

**Key visual components:**
- `st.chat_message` / `st.chat_input` — native chat UI with streamed LLM responses
- Scenario P&L matrix → interactive Plotly heatmap (spot × vol, colour = P&L %, time horizon selector)
- Sizing output → `st.metric` cards (Kelly fraction, notional, stop level, tranches)
- Structure shortlist → `st.expander` cards with rationale and caution badges

**Deployment:** `streamlit run interface/app.py` locally; push to Streamlit Community Cloud for a shareable URL. Requires adding `streamlit` and `plotly` to `pyproject.toml`.

---

## Open questions (from spec)

- **Outcomes logging** — SQLite log of sessions + subsequent market outcomes. Not Phase 1.
- **Bloomberg integration** — Replace `data/market_snapshot.json` with live OVML/FXFA feed. `MACROTOOL_SNAPSHOT_PATH` env var already provides the swap point.
- **Vol surface beyond POC** — Currently the scenario matrix uses a flat vol perturbation. A proper vol surface interpolator (delta → strike conversion, smile fitting) would improve accuracy.

---

## Running the tests

```bash
.venv/bin/python -m pytest                    # full suite (131 tests)
.venv/bin/python -m pytest tests/test_foundation.py   # Phase A
.venv/bin/python -m pytest tests/test_pricing.py      # Phase B
.venv/bin/python -m pytest tests/test_knowledge.py    # Phase C
```

Python 3.13 via Homebrew (`/opt/homebrew/bin/python3.13`). Venv at `.venv/`.
