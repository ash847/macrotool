# Agentic Workflow — Build Plan

Branch: `agentic-workflow` (off `main` @ c4d3697). Worktree:
`/Users/ash/Documents/Coding work/agentic-workflow`. Entry point: `interface/app.py`.

## Motivation

The current conversational path (`conversation/flow.py`) is a fixed state machine
(`INTAKE → DONE`, 3 hardcoded API calls). Responses are one-dimensional and intolerant
of topic drift, because the *control flow* is hardcoded — every question must route back
through INTAKE or the narrow DONE follow-up. The follow-up call always re-feeds the same
frozen explanation pack, so a question that changes an input ("what if I move the strike
to 40Δ?") can't actually be priced; the LLM can only hand-wave.

**Goal:** replace the rigid graph with a tool-calling agent loop that handles topic drift,
*without* giving up the load-bearing invariant that makes the tool trustworthy:

> **The LLM never produces a number. It only orchestrates (picks which tool to call) and
> narrates (explains numbers a tool returned).**

This is safe here specifically because the codebase already obeys strict layer separation
and a clean "numbers are pure, narration is separate" discipline. That discipline is the
precondition that makes agentic tool-use safe.

## Design decisions (settled)

### 1. Additive, not a rewrite
- `analytics/` + `knowledge_engine/` are untouched — they're already pure.
- The agent loop is built *next to* `flow.py`, not replacing it. The deterministic
  pipeline stays as (a) the mandatory first stage and (b) the regression oracle.

### 2. Standard pack runs first, enforced in Python (not by prompt)
- `flow._run_engines()` IS the "standard pack" (`compute_market_state → score_structures
  → compute_sizing → distributions → evaluate`).
- On a new view, *our code* calls `_run_engines()` unconditionally and seeds the agent
  context with the structured results. The agent starts every conversation already holding
  the full baseline; the first response is the current deterministic output, verbatim.
- "Only rerun for new queries" becomes a **cache key**:
  `(pair, direction, horizon, target, structure_constraint, primary_objective,
  trade_management)`. Same key → agent narrates from context, zero tool calls, zero new
  numbers. The agent has **no tool** that recomputes the baseline from scratch; only our
  code calls `_run_engines`.

### 3. The LLM describes a structure; Python constructs it
This is the key safety move. The dangerous levers are *already* isolated in
`analytics/structure_pricer.py::price_variants(ms, structure_id, target, is_call,
stop_price, loss_budget, smile, warnings)`:
- `is_call` (direction) is derived from `view.direction == "base_higher"` — never ad hoc.
- weights (`1.5`, `2.0`) are hardcoded inside `_1x1p5` / `_1x2` — the LLM can't express them.
- strikes are resolved internally from deltas (`otm_call_strike` / vol surface) or solved
  by bisection (digital). The caller passes *deltas or premium targets*, not strikes.
- `target` / `loss_budget` come from the session view + sizing.

So the agent emits a structure **specification** in a tight grammar; the parser resolves it
into the synthetic variant dict `price_variants` already consumes. The LLM names the trade
the way a PM says it out loud — it never sets direction, weights, signs, or notionals.

Example: `"34 vs 25 1x1.5"` →
```python
StructureRequest(family="1x1.5_spread", legs=[Delta(34), Delta(25)])
```
fed to the existing `_resolve_ratio_spread_strikes` path (which already accepts
`long_delta` / `short_delta`). Python supplies ratio `1.5` (family pricer), `is_call`
(view), `target`/`loss_budget`/`smile` (session). The agent supplied only family + leg refs.

### 4. "Narration over numbers in context" = the existing comparator/explanation pack
- `_build_explanation_pack_context()` → `build_comparator_inputs` →
  `build_recommendation_pack` → `render_explanation_pack(pack)`, stashed on
  `self.explanation_pack_context` and threaded into `build_followup_prompt`.
- The agentic loop reuses this exact pack as the seed context. "Why 1x1.5?" narrates over it
  (no tool call). "What if 40Δ?" triggers `price_structure("40Δ ...")` → fresh
  `PricedVariant` appended to context → agent narrates over the new structured result.
- Likely rendering tweak: expose the pack in a more structured/labelled form so the agent
  can cleanly distinguish "baseline variant" from "PM-requested variant" in a comparison.

### 5. The agent's mental model — a two-tier tool surface (NOT the quant pipeline)

The agent does **not** carry a model of "market state → ranker → scoring → sizing." It
never touches a `MarketState`, never decides how to rank/score. That dependency chain lives
in Python and runs as one atomic unit — orchestrating it step-by-step from the agent would
let it desync (rank against one state, size against another) and would mean the agent is
reasoning about numbers, which is forbidden.

The agent's *actual* mental model is shallow: **conversation/context state + a routing
decision.** "What is the current view (pair, tenor, direction, target, prefs)? Did the PM
change a view input, ask about a specific structure, or just ask a question?"

The routing axis is a single question: **does this change the `MarketState`?**

| Tier | Triggered by | Tool | Who computes |
|---|---|---|---|
| **Tier 1 — coarse** | a change to the *view inputs*: pair, tenor, direction, **target/magnitude**, prefs | `run_standard_pack(view…)` → **our** `_run_engines` | Python runs the **entire** chain wholesale (market state → ranker → scoring → sizing → distributions) |
| **Tier 2 — fine** | pricing/evaluating a *specific structure* within the **current** market state | `price_structure(request)`, `evaluate_scenarios`, `size` | Python prices that one payoff against the **frozen** market state |

Key consequences:
- **Changing pair/tenor IS allowed** — it's a Tier 1 call. The agent extracts the new view
  from plain English (exactly what `[VIEW: {…}]` extraction already does) and hands the
  *inputs* to the tool; Python rebuilds everything deterministically. The agent reconstructs
  nothing and isn't allowed to look inside `_run_engines`.
- **`target` / `magnitude` is Tier 1, not Tier 2.** It feeds the scorer (`target_z`, the
  `target_z_abs` gate/dimension), so moving the target must re-score → full pack rebuild.
  Tier 2 prices a structure *against the already-computed target*, holding market state fixed.
- Tier 2 is the dangerous tier (agent naming a payoff) → it goes through the Phase 1 strict
  grammar. Tier 1 is coarse and safe → the agent only passes view parameters.
- This rejects Pole A (agent owns/updates the quant model) AND Pole B (frozen single pack,
  no pair/tenor change). It's tool-mediated pack rebuilds: "rebuild" is one coarse Python
  call the agent can't see inside.

The three PM actions map cleanly: (1) change a view input → Tier 1 → fresh pack → narrate;
(2) ask about a structure → Tier 2 → new `PricedVariant` appended → narrate; (3) ask
"why/what" → no tool → narrate over the pack in context.

## Tool boundary (the whole agent-facing surface)

| Agent-facing tool | Wraps | LLM supplies | Python supplies (locked) |
|---|---|---|---|
| `run_standard_pack` (auto, not a real tool) | `_run_engines` | nothing | everything |
| `price_structure(request)` | parser → `price_variants` | family + leg refs | weights, `is_call`, target, loss_budget, smile |
| `evaluate_scenarios(request)` | `price_scenarios` | structure ref | scenario grid, surface |
| `size(request)` | `compute_sizing` | structure ref | Kelly fractions, config |
| (narration) | — | prose only | — |

Start by exposing only the **read/evaluate** tools. Keep sizing on a more constrained path
until the loop is trusted.

## The grammar (`StructureRequest`)

Small, closed vocabulary. Anything outside → **structured rejection, never a guess.**

- **family**: from a closed registry = `structure_profiles.json` keys
  (`vanilla`, `1x1_spread`, `1x1.5_spread`, `1x2_spread`, `seagull`, `european_digital`,
  `european_rko`, ...). Reject unknown families.
- **leg reference vocabulary**: `NΔ` (delta), `ATMF`, `±Nσ`, `target`, `premium=N%`.
- **resolution policy**:
  - Allow **free deltas** for spreads/vanillas (low risk — the pricer handles arbitrary
    delta pairs; strictly more flexible than today's curated `structure_variants.json` menu).
  - Keep **digital / RKO** families on their curated premium-target variants (they go
    through bisection + arb guards — don't let the agent request arbitrary barriers/strikes).
- **direction**: NOT in the grammar. `is_call` always comes from the session view. (This is
  where the `base_higher`/`base_lower` flip risk goes to die — the agent literally cannot
  express direction.)
- **ambiguity**: return a structured clarification request, never silently pick.
- **auditability**: the canonical request string lands in the Langfuse trace, far cleaner
  than free-form tool args ("agent asked for `34v25 1x1.5`").

## Build phases

### Phase 1 — `StructureRequest` grammar + parser (FIRST, load-bearing) — DEFINED

A standalone, pure module. **No LLM, no MarketState, no pricing** — string in, validated
spec out. Fully unit-testable in isolation. This is the safety boundary: everything the LLM
is allowed to say about a structure passes through here and is either resolved to a known
shape or rejected.

#### Location
New isolated package `agentic/` (keeps Phases 1–3 cleanly separated from `flow.py`):
- `agentic/structure_request.py` — data model, parser, `to_variant_dict`.
- `agentic/family_registry.py` — closed family table (synonyms, leg arity, allowed leg
  kinds, free-vs-curated policy), derived from `structure_profiles.json` + `structure_variants.json`.
- `tests/test_structure_request.py`.

#### Scope boundary (what Phase 1 does NOT do)
- Does not price anything, does not touch `price_variants` (that's Phase 2).
- Does not call the vol surface or resolve strikes (the pricer does that from deltas).
- Does not decide `is_call` (session view does, in Phase 2).
- Does not need a target value present (it only records *that* a leg references "target").

#### Data model
```python
LegKind = Literal["delta", "atmf", "sigma", "target", "premium"]

@dataclass(frozen=True)
class LegRef:
    kind: LegKind
    value: float | None     # delta∈(0,1); sigma multiple (signed); premium fraction∈(0,1)
                            # None for atmf / target

@dataclass(frozen=True)
class StructureRequest:
    family: str             # canonical family id, e.g. "1x1.5_spread"
    legs: tuple[LegRef, ...]
    canonical: str          # normalized echo for the Langfuse trace, e.g. "1x1.5_spread 34Δ/25Δ"

class StructureRequestError(ValueError):
    """Structured rejection. Carries .reason (enum) and .detail (human string)."""

@dataclass(frozen=True)
class ClarificationNeeded:
    question: str           # returned (not raised) when the request is ambiguous
```

#### Family registry (closed — the whole allowed surface)
| family | leg arity | leg kinds accepted | policy | variant dict produced |
|---|---|---|---|---|
| `vanilla` | 1 | delta / atmf / sigma | free | `{delta}` |
| `1x1_spread` | 2 | delta (both) | free | `{long_delta, short_delta}` |
| `1x1.5_spread` | 2 | delta+delta, **or** {atmf\|sigma}+target | free | delta-pair → `{long_delta, short_delta}`; anchored → `{long_type: "atmf"\|"half_sigma", min_target_z?}` |
| `1x2_spread` | 2 | same as 1x1.5 | free | same as 1x1.5 |
| `seagull` | 3 | delta×3 (spread_long, spread_short, wing) | free | `{spread_long, spread_short, wing_delta}` |
| `european_rko` | 2 | delta (long) + delta (barrier) | free | `{long_delta, barrier}` |
| `european_digital` | 1 | premium **only** | **curated** | `{target_prem_pct}` |
| `european_digital_rko` | — | — | **disabled** (`enabled:false`) → reject | — |
| `rko` | — | — | **disabled** (`enabled:false`) → reject | — |

Notes:
- **Free** families accept arbitrary deltas (strictly more flexible than the curated
  `structure_variants.json` menu — the pricer handles any delta pair).
- **Curated** families (`european_digital`) only accept a `premium=N%` leg — they bisect on
  premium and carry the smile-arb guard; the LLM may not request arbitrary strikes/barriers.
- **Disabled** families are rejected with reason `FAMILY_DISABLED` (mirrors the
  `enabled:false` gate in `structure_profiles.json`).
- For the two ratio families the **short leg is the family ratio**, applied in the pricer
  (`_1x1p5` uses 1.5×, `_1x2` uses 2×) — never in the request. The second delta leg is the
  *short strike placement*, not a weight.

#### Leg-reference grammar (tokenizer)
Tolerant of PM phrasing; case-insensitive; whitespace-flexible.
- **delta**: `34Δ`, `34d`, `34 delta`, `0.34Δ`. Integers 1–99 → `/100`; floats in (0,1) kept.
  Out of (0,1) → `BAD_DELTA`.
- **atmf**: `ATMF`, `ATM`, and `50Δ` is normalized to `atmf` for the long leg.
- **sigma**: `+1σ`, `-0.5σ`, `1.5sigma`, `half sigma`→`0.5σ`. Signed multiple.
- **target**: `target`, `tgt`.
- **premium**: `10%`, `prem=10%`, `~10% prem` → fraction 0.10. Range (0,1) else `BAD_PREMIUM`.
- **separators**: `vs`, `/`, `,` between legs. Family token (`1x1.5`, `spread`, `digital`,
  `seagull`, `erko`, `vanilla`, `call/put option`) anywhere in the string.
- **direction words** (`call`, `put`, `long`, `short`): parsed but **not used for
  construction**. They are *cross-checked* against the session view in Phase 2; a direct
  contradiction there → clarification. Phase 1 just records/strips them.

#### Family inference
- Explicit family token present → use it (after synonym normalization).
- No family token but leg shape is unambiguous (e.g. exactly a `premium=` leg → digital) →
  infer.
- Ambiguous (e.g. two bare deltas could be `1x1_spread` / `1x1.5_spread` / `1x2_spread`) →
  return `ClarificationNeeded("1x1, 1x1.5 or 1x2 spread?")`. Never silently pick.

#### Validation → structured rejection
`StructureRequestError.reason` enum (each maps to a clean agent-facing message):
`UNKNOWN_FAMILY`, `FAMILY_DISABLED`, `WRONG_LEG_COUNT`, `BAD_DELTA`, `BAD_PREMIUM`,
`BAD_LEG_KIND_FOR_FAMILY`, `MIXED_ANCHOR` (e.g. a digital given a delta leg, or a ratio
family given target on a non-anchor leg), `EMPTY_REQUEST`.

#### Public API
```python
def parse_structure_request(text: str) -> StructureRequest | ClarificationNeeded
def to_variant_dict(req: StructureRequest) -> dict   # the synthetic variant dict + "label"
```
`to_variant_dict` output is exactly the per-variant dict shape `price_variants` already
consumes (validated against `structure_variants.json` shapes above), plus a `"label"` key
built from `canonical`.

#### Tests (`tests/test_structure_request.py`)
1. **Happy path per family** — one canonical request each → expected `StructureRequest` +
   `to_variant_dict` shape (assert keys match the JSON variant schema exactly).
2. **Delta normalization** — `34Δ`, `0.34`, `34 delta` all → `0.34`; `50Δ` long leg → atmf.
3. **Ratio-family dual form** — `"34 vs 25 1x1.5"` → delta-pair dict; `"ATMF vs target
   1x1.5"` → `{long_type:"atmf"}`; `"½σ vs target 1x2"` → `{long_type:"half_sigma",
   min_target_z:0.5}`.
4. **Curated guard** — `european_digital` with a delta leg → `BAD_LEG_KIND_FOR_FAMILY`;
   with `premium=10%` → `{target_prem_pct:0.10}`.
5. **Disabled families** — `rko`, `european_digital_rko` → `FAMILY_DISABLED`.
6. **Rejections** — unknown family, wrong leg count, delta=120, premium=0%.
7. **Ambiguity** — two bare deltas, no family token → `ClarificationNeeded`.
8. **Direction words ignored** — `"25Δ put"` and `"25Δ call"` parse identically (direction
   stripped, not stored for construction).
9. **Parity with curated menu** — for each entry in `structure_variants.json`, a request
   phrased to match it round-trips to the *same* variant dict (proves the grammar is a
   superset of today's menu, breaking nothing).

#### Done criteria
- All tests green; `parse_structure_request` + `to_variant_dict` importable with zero
  dependency on `analytics` / `conversation` / any LLM.
- The 9 curated-menu entries each have a request string that reproduces their variant dict.
- No change to any existing file — Phase 1 is purely additive (`agentic/` + one test file).

### Phase 2 — `price_structure` tool wrapper
- `price_structure(request: str)` → parse → `to_variant_dict` → `price_variants(ms, family,
  target=<session>, is_call=<view>, loss_budget=<session>, smile=<MarketState.surface>)`
  → return the `PricedVariant` as a structured result.
- Pulls `ms`, view direction, target, surface from the session — agent supplies only the
  request string. Hard-validate args; reject/echo on ambiguity.

### Phase 3 — Agent loop (next to `flow.py`, not replacing it)
- New module (e.g. `conversation/agent_flow.py`) with a tool-calling loop.
- On new view: call `_run_engines()`, seed context with the (re-rendered, labelled)
  explanation pack.
- System prompt carries domain conventions (direction, carry, base-ccy payoff, digital =
  100%-at-target) and the hard rule: *no number that didn't come from a tool / the pack.*
- Tools registered: `price_structure` (+ later `evaluate_scenarios`, `size`).
- Cache keyed as in decision #2; same key → narrate, no recompute.
- Reuse Langfuse tracing (one generation per tool call / turn).

### Phase 4 — Wire into Streamlit + validate
- Add an entry point / toggle in `interface/app.py` to drive `agent_flow` instead of the
  fixed state machine (keep the deterministic path available as "full evaluation" fallback).
- Validate: baseline first-response matches the deterministic output; topic-drift questions
  price correctly; "why" questions make zero tool calls.

## Costs / risks to keep in view
1. **Determinism of *which* numbers appear** — agent may call tools in different orders or
   stop early. Mitigation: deterministic standard pack first; retain a "show full standard
   evaluation" path; Langfuse tracing for audit/reproducibility.
2. **Arg-level correctness** — mitigated structurally: the grammar removes direction/weights/
   strikes/notionals from the LLM's reach. Hard-validate the rest in Python; reject on
   ambiguity rather than trusting the model.
3. **Cost / latency** — multi-turn tool loops are more calls than the fixed 3. Mitigated by
   the cache (most follow-ups are zero-tool narration).

## Invariants to preserve (from CLAUDE.md)
- LLM narrates only; all numbers pre-computed by the engine.
- Messages must end in a user turn before any API call.
- `base_higher`/`base_lower` is relative to ccy1; `with_carry` formula is correct — do not
  change.
- Distributions are non-blocking enrichment.
- Vol surface is built once in `_run_engines` and reused everywhere a vanilla is priced.

## Status log
- [x] Worktree `agentic-workflow` created, venv synced, pushed to `origin`.
- [x] Phase 1 — grammar + parser. `agentic/structure_request.py` +
      `agentic/family_registry.py` + `tests/test_structure_request.py` (28 tests, all
      green). Pure (no analytics/conversation/LLM deps), additive (no existing file
      touched). Parity test confirms the grammar is a superset of the curated
      `structure_variants.json` menu.
      **Strict-token policy (chosen):** any unrecognized alphabetic token (unknown
      structure name, typo, junk) → `ClarificationNeeded` naming the offending term, never
      a silent guess. Only legitimate leg markers (`d/atmf/atm/sigma/target/tgt/prem/vs`)
      and a resolved family word are allowed residue. `UNKNOWN_FAMILY` remains a defensive
      guard in `to_variant_dict` (unreachable via parse).
- [x] Phase 2 — `price_structure` tool. `agentic/price_structure.py` +
      `tests/test_price_structure.py` (9 tests). One minimal seam added to
      `analytics/structure_pricer.py`: optional `variants_override` so a caller can price a
      single synthetic variant instead of the curated JSON menu (default None → every
      existing caller unchanged). Tool pulls `is_call` / `target` / `loss_budget` / `smile`
      from the session, agent supplies only the request string. Three tagged outcomes:
      `PricedStructure` / `ClarificationNeeded` / `PricingUnavailable`; hard-malformed
      requests still raise `StructureRequestError`. Full suite: 423 pass, 1 fail
      (pre-existing scorer-tuning failure, unrelated). Phase 1+2 = 37 tests green.
- [ ] Phase 3 — agent loop.
- [ ] Phase 4 — Streamlit wiring + validation.
