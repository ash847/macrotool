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

### Phase 3 — Agent loop (next to `flow.py`, not replacing it) — DEFINED

The first LLM-touching phase. A tool-calling loop where the LLM routes the PM's turn to
Tier-1 (rebuild the pack) / Tier-2 (price a structure) / no-tool (narrate), and Python does
all computation. `flow.py` is left untouched; this is a parallel driver.

#### Grounding constraint (from the code)
`conversation/client.py: MacroToolClient` is **text-streaming only** (`stream(messages,
system)` → text chunks). It has **no tool-use path**. So Phase 3 must add a tool-calling
seam for Anthropic (`tool_use` / `tool_result` content blocks via `messages.create(tools=
[...])`). Anthropic-only at first; **Gemini tool-use is deferred** (project defaults to
anthropic). Responses are short → use non-streaming `create` for the tool loop (simpler than
streaming tool_use assembly); stream only the final text turn if desired.

#### Location (all additive)
- `agentic/agent_llm.py`     — **provider-neutral tool seam (explicit deliverable).** A
                               small interface that normalizes, across providers: (a) our
                               tool schema → the provider's tool-declaration format, (b) the
                               model's "call tool X(args)" → a normalized `ToolCall`, (c) our
                               tool result → the provider's result message. `agent_flow.py`
                               never branches on provider. **Anthropic adapter built + tested
                               first.** Mirrors/reuses the retry logic in
                               `AnthropicProviderClient`. The fake-client tests are
                               provider-agnostic, so they cover any adapter for free.
                               **Adapter order: OpenAI before Gemini.** OpenAI is the closer
                               cousin to Anthropic (JSON-Schema tools, `tool_calls` carrying
                               per-call **IDs**, one-result-per-ID) — only friction is
                               `arguments` arriving as a JSON *string* (`json.loads` it).
                               Gemini is the awkward one: **no per-call ID** (calls/results
                               matched by function *name*), which gets fiddly with parallel
                               calls. **Seam design constraint:** the normalized `ToolCall`
                               carries a call ID from day one; Anthropic and OpenAI supply it,
                               the Gemini adapter must *manufacture* one and map it back to
                               the function name. Both adapters are later drop-ins (one class,
                               no loop/dispatch changes).
- `agentic/standard_pack.py` — `build_pack(view, snapshot, cfg, prefs) -> StandardPack`,
                               the deterministic chain. **Light refactor:** factor the body
                               of `flow._run_engines` into this shared function and have
                               `flow.py` call it too (DRY; keeps one source of truth). The
                               pack bundles `market_state`, `selector_result`, `sizing`,
                               distributions, and the rendered explanation pack.
- `agentic/tools.py`         — tool JSON schemas + the Python dispatch table.
- `agentic/session.py`       — `AgentSession`: current view, current `StandardPack`, the
                               Tier-1 cache, the conversation messages, and the set of
                               priced structures requested this session.
- `agentic/agent_flow.py`    — the loop + system-prompt assembly.
- `tests/test_agent_flow.py`, `tests/test_agent_tools.py`.

#### Scope boundary (what Phase 3 does NOT do)
- No Streamlit wiring (Phase 4). The loop is drivable headlessly for tests.
- No `evaluate_scenarios` / `size` tools yet — `run_standard_pack` + `price_structure` only.
  (Scenario/sizing tools are a fast follow once the loop is trusted.)
- The **provider-neutral seam is in scope**; the **Gemini adapter implementation is not**
  (Anthropic adapter only — Gemini/OpenAI adapters are later drop-ins behind the seam).

#### Tool schemas (the whole agent-facing surface)
**Tier 1 — `run_standard_pack`** (coarse; triggers the full deterministic chain):
- params: `pair`, `direction` (`base_higher`|`base_lower`), `horizon_days`,
  `magnitude_pct` (optional), `primary_objective`, `structure_constraint`,
  `trade_management`, `mode` (`recommend`|`critique`).
- These are exactly the `TradeView` fields — view extraction from NL *is* the LLM populating
  these args. Python validates them (`TradeView` Pydantic), then runs `build_pack`. The LLM
  produces the *view*, never numbers — identical in spirit to today's `[VIEW: {…}]` tag.
- returns: the rendered pack (market context, shortlist + scores, sizing, distributions),
  labelled so the agent can cite it.

**Tier 2 — `price_structure`** (fine; against the frozen pack):
- params: `request` (string, e.g. `"34 vs 25 1x1.5"`, `"digital 10%"`).
- dispatch: pulls `ms` / `is_call` / `target` / `loss_budget` / `smile` from the current
  pack+view (NOT from the LLM) and calls `agentic.price_structure.price_structure`.
- **guard:** if no pack exists yet → return a tool error telling the agent to call
  `run_standard_pack` first (this is how decision #2 — "standard pack first" — is enforced:
  the agent has no numbers and cannot price until the Python pack exists).
- returns: `PricedStructure` rendered as labelled text, or the `ClarificationNeeded` /
  `PricingUnavailable` message for the agent to relay.

#### Model choice (decided)
- **Workhorse: Sonnet 4.6 (`claude-sonnet-4-6`).** Already the project default in
  `conversation/client.py`; solid tool-use, 1M context, right cost/quality balance for a
  low-volume interactive tool ($3 / $15 per 1M in/out).
- **Fallback one config flag away: Opus 4.8 (`claude-opus-4-8`)** ($5 / $25). Reach for it if
  testing shows tier mis-routing, direction flips, or drift on the narrate-only rule — Opus
  has the strongest instruction adherence + tool triggering, and at this volume the price
  delta is immaterial. Also the better pick if/when the agent *critiques* a PM structure.
- **Not Haiku for the loop.** Weakest at tool adherence and at honoring "no number that
  didn't come from a tool" — the one place not to economize, given the safety model rests on
  reliable tool-calling. (A narrow forced-`tool_choice` view-extraction sub-call on Haiku is a
  *possible* later micro-optimization — premature until the single-model loop works.)
- Rationale: the loop's work is routing + arg population + narration — **not** heavy quant
  reasoning (all numbers are Python). So tool-calling reliability and rule adherence dominate
  the choice, not raw reasoning depth. Model ID is just an adapter param → Sonnet↔Opus (or a
  later provider) is a config change, not code.

#### The loop (`agent_flow.advance(user_message)`)
1. Append the user turn.
2. Call the LLM with the tool schemas + system prompt.
3. If the response has `tool_use` blocks → dispatch each to the Python table, append a
   `tool_result` block (user-role), go to 2.
4. If the response is text (`end_turn`) → that's the narration; emit and stop.
5. Bound iterations (e.g. ≤ 6 tool rounds/turn) to prevent runaway; on overflow, emit a
   graceful "couldn't complete" and stop.

#### Tier-1 cache (decision #2)
Key = `(pair, direction, horizon_days, magnitude_pct, structure_constraint,
primary_objective, trade_management, mode)`. `run_standard_pack` computes the key; cache hit
→ reuse the pack (zero recompute, the agent narrates from context); miss → `build_pack`,
store. There is **no** tool that recomputes the pack piecemeal — only `build_pack`, only via
this one coarse door.

#### System prompt (the hard rules)
- Domain conventions: direction (`base_higher`/`base_lower` rel. ccy1), carry/`with_carry`,
  base-ccy payoff, **European digital = 100% at target**.
- **No number that didn't come from a tool result or the pack in context.** Never compute,
  never interpolate, never invent a strike/premium/level.
- Routing: change the view (pair/tenor/direction/**target**/prefs) → `run_standard_pack`;
  price a specific structure the PM names → `price_structure`; a "why/what/explain" question
  about numbers already shown → **no tool**, narrate.
- On `ClarificationNeeded` → ask the PM the returned question; on `PricingUnavailable` →
  relay the reason (e.g. "needs a target").

#### Rendering for the agent
- Reuse `render_explanation_pack` for the pack, but add light **labels** so the agent can
  distinguish "baseline (from the pack)" vs "PM-requested" structures.
- New `render_priced_structure(pv)` — a compact labelled block for a single `PricedVariant`
  (strikes, premium %, payoff@target, RR, max loss, and ccy fields when sized).

#### Invariants preserved
- **Messages end in a user turn before every API call** — satisfied for free: Anthropic
  `tool_result` blocks are user-role, so `user → assistant(tool_use) → user(tool_result) →
  …` always ends on user before the next `create`.
- LLM narrates only; numbers come from Python. Tier-2 danger goes through the Phase-1 grammar.

#### Testing (no API burn)
- `FakeToolClient` returning a *queued* script of responses (`tool_use` blocks then a final
  text turn) → drive the loop deterministically, asserting: correct dispatch, cache
  hit/miss + pack reuse, the `price_structure`-needs-pack guard, the ClarificationNeeded
  relay, and iteration bounding.
- `tests/test_agent_tools.py` — dispatch table in isolation (schema validation, view→pack,
  request→priced structure) without the loop.
- One **live** smoke test gated on `ANTHROPIC_API_KEY` (skipped if absent): a real "long
  3m USDBRL, target +6%" → assert a `run_standard_pack` call happens and a coherent pack
  comes back.

#### Done criteria
- Loop runs headlessly under the fake client; all unit tests green.
- `flow._run_engines` and `build_pack` share one implementation (no divergence).
- `price_structure` refuses before a pack exists; Tier-1 cache demonstrably avoids recompute
  on identical view inputs.
- No existing behavior changed except the `_run_engines` → `build_pack` extraction (covered
  by the existing flow tests, which must stay green).

### Phase 4 — Wire into Streamlit + validate
- Add an entry point / toggle in `interface/app.py` to drive `agent_flow` instead of the
  fixed state machine (keep the deterministic path available as "full evaluation" fallback).
- Validate: baseline first-response matches the deterministic output; topic-drift questions
  price correctly; "why" questions make zero tool calls.

### Phase 5 — Langfuse observability for the agent loop (NEXT — not started)

Goal: durable, searchable traces for the Agent page, and ideally a "paste a trace ID here →
Claude reads it directly" debug loop (replaces copy-pasting the on-screen Engine trace).

Current state: the Agent page emits **no** Langfuse traces — `agent_flow` was never
instrumented (only the legacy `conversation.flow` traces). The on-screen "🔍 Engine trace"
expander is the only observability today, and it's ephemeral (`st.session_state`).

Steps:
1. **Instrument the loop (prerequisite, do first).** Reuse `conversation.tracing`: one
   trace/generation per `llm.create()` turn and one span per tool dispatch in `agent_flow` /
   `tools.dispatch` (record tool name, args, result text, is_error). Useful on its own — you
   get the Langfuse UI even without sharing keys.
2. **Direct-query path (optional, to stop copy-pasting to Claude).** Feasibility already
   checked this session: **network egress to `cloud.langfuse.com` works (HTTP 200)**; **no
   Langfuse MCP connector exists** in the registry; my shell has **no** Langfuse keys (they
   live in Streamlit secrets). So the path is the **Langfuse REST API** (`GET
   /api/public/traces`, `/traces/{id}`) from Bash — needs `LANGFUSE_PUBLIC_KEY` /
   `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` dropped into Claude's shell env (confirm
   cloud vs self-hosted; only cloud host was reachability-tested). Then: user pastes a trace
   ID/URL → Claude fetches + analyzes it.
   - **Security note:** the secret key is a real project-scoped credential; if shared, rotate
     it afterwards or use a read-only/scoped key if the plan supports one.
3. Workflow once both done: PM pastes a Langfuse trace ID/URL in chat → Claude pulls the
   tool calls / inputs / outputs / latency and diagnoses, no manual paste of contents.

## Costs / risks to keep in view
1. **Determinism of *which* numbers appear** — agent may call tools in different orders or
   stop early. Mitigation: deterministic standard pack first; retain a "show full standard
   evaluation" path; Langfuse tracing for audit/reproducibility.
2. **Arg-level correctness** — mitigated structurally: the grammar removes direction/weights/
   strikes/notionals from the LLM's reach. Hard-validate the rest in Python; reject on
   ambiguity rather than trusting the model.
3. **Cost / latency** — multi-turn tool loops are more calls than the fixed 3. Mitigated by
   the cache (most follow-ups are zero-tool narration).

## Known limitations (revisit when needed)

### Grammar ↔ delta-resolver range mismatch
The Phase 1 grammar accepts **free deltas in `(0, 1)`**, but the strike resolver
(`analytics/strike_resolver.py`) only supports **`[0.10, 0.50]`** — a σ-approximation that
looks up `z = N⁻¹(Δ)` over the standard broker pillars (10/15/25/50Δ) and interpolates
between them. Consequences and rationale:
- A request like `60Δ` or `5Δ` **parses cleanly** but would **raise `ValueError` at pricing
  time**, not at parse time. The two layers disagree on the valid band.
- The `[0.10, 0.50]` cap is intentional, not arbitrary: `0.50` = ATM-forward (above it is
  ITM → express from the other side via `is_call`/`otm_put_strike`); below `0.10` the
  approximation degrades and quotes are thin. It was scoped to exactly cover the curated
  `structure_variants.json` menu (which only uses 10–50Δ).
- Resolution logic is **separable** (one small module, stable signature `fwd, vol, T, delta
  → strike`) but wired by **direct import**, not an injection seam like the vol surface
  (`build_vol_surface`). So "improve the math" is a drop-in edit; "swap resolvers at
  runtime" would need a small `StrikeResolver` Protocol + factory.
- **When we come back to it** — two clean fixes: (a) tighten the grammar to reject deltas
  outside the supported band at *parse* time (cheap, immediate), or (b) replace the
  σ-approximation with a continuous, exact `N⁻¹`-based (premium-adjusted where appropriate)
  inversion so any delta in `(0, 0.50]` resolves — which also makes the grammar's free-delta
  promise real. (b) is the "add complexity to delta resolution" upgrade; both close the gap.
- Not urgent: every curated structure and every realistic PM request sits in 10–50Δ today.

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
- [x] Phase 3 — agent loop. `agentic/standard_pack.py` (build_pack, refactored out of
      flow._run_engines), `agentic/session.py` (AgentSession + Tier-1 cache),
      `agentic/render.py` (labelled pack / priced-structure renderers), `agentic/tools.py`
      (run_standard_pack + price_structure schemas + dispatch, with the needs-pack guard and
      SUPPORTED_PAIRS validation), `agentic/agent_llm.py` (provider-neutral `ToolLLM` seam +
      `AnthropicToolLLM` + `FakeToolLLM`), `agentic/agent_flow.py` (the loop + system prompt).
      Tests: `tests/test_agent_tools.py` (7) + `tests/test_agent_flow.py` (6, fake-LLM
      driven: pack→narrate, price-within-pack, needs-pack guard, cache reuse, clarification
      relay, iteration bound) + `tests/test_agent_live.py` (skipped without ANTHROPIC_API_KEY).
      Full suite: 436 pass, 1 pre-existing scorer fail. Model default Sonnet 4.6.
      **Not yet built (deferred to Phase 3.5 / Phase 4):** evaluate_scenarios + size tools;
      OpenAI/Gemini adapters; richer render via the comparator explanation pack (current
      render is a self-contained labelled summary); Streamlit wiring.
- [ ] Phase 5 — Langfuse observability for the agent loop (NEXT). Instrument agent_flow/tools
      with `conversation.tracing` (one trace/turn, one span/tool call); then optional
      REST-API direct-query path (egress OK, no MCP, needs keys in Claude's shell). See the
      Phase 5 section above for the full plan + security note.
- [~] Phase 4 — Streamlit wiring. `interface/app.py`: new **Agent** nav page (user-accessible)
      driving `AgentFlow` over `AnthropicToolLLM`, with `st.chat_input` chat UI, a "New
      conversation" reset, a live-view caption, and error capture via `log_error`. Agent
      session/flow cached in `st.session_state`; prefs seeded from the sidebar pref widgets.
      pyproject bumped 0.1.28 → 0.1.29+agent (+ uv.lock) for Streamlit Cloud reinstall.
      **Pending: live validation on Streamlit** (does the model route well end-to-end) and
      the deterministic-first-response parity check.
