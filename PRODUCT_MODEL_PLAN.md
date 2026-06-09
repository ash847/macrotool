# Product Model Refactor — `Structure` / `Leg` / `PricedStructure`

Engine refactor (not agent work) — replaces the flat `PricedVariant` with a first-class
product model. Standalone build doc; execute against this when green-lit.

**Status: deferred, pattern finalized.** The flat `PricedVariant` + per-family pricers work and
are fully tested (~440 green, 1 pre-existing scorer fail). Take this on when bespoke-package
composition / notional round-tripping / Kelly-on-arbitrary-legs becomes a recurring need.

See also: `AGENTIC_WORKFLOW_PLAN.md` → "Candidate refactor: the Structure / Leg product model"
for the *why* (the recurring LLM-fabrication bugs: deltas, wing ratio, leg notionals, strikes —
all "render couldn't see it → model invented it"). The durable cure is a **complete object + a
total renderer**, and a re-priceable `Structure` as the engine's shared currency.

---

## 1. The pattern (finalized)

Three objects + one function. Separate **definition** from **context** from **valuation**.

- **`Leg`** — definition of one instrument, market-independent:
  - `instrument` (enum): `Vanilla | Digital | …(future Barrier)` — selects the leg's pricing
    model + payoff shape.
  - `right` (enum): `Call | Put`. Orthogonal to position direction.
  - `signed_notional` (float): **+ long / − short**; magnitude is the weight. A package is then
    a literal weighted sum `Σ signed_notionalᵢ × unitᵢ`. Subsumes `wing_ratio` (wing = a leg at
    −0.55) and ratio-spread weights (a leg at −1.5 / −2). **Decision: signed notionals on the
    leg** (not a separate weights array).
  - `spec` (anchor): how the strike is named — `Delta(x) | ATMF | Sigma(±x) | Strike(K) |
    PremiumTarget(pct)` (last for digitals). **Spec is authoritative on the definition; the
    resolved strike is an output** (resolution needs `MarketState`, happens at price time).
  - (future) `expiry` — single tenor today; per-leg enables calendars/diagonals.
- **`Structure`** — package definition: `family` tag (display/provenance) + `legs: list[Leg]`
  + structure-level definitional fields (e.g. `barrier` for RKO; binary payout on the digital
  leg). Market-independent, **serializable, immutable**. What the grammar resolves a request into.
- **`MarketState`** — context (spot/fwd/vols/carry/regime). **Not** part of the product object;
  Class-B framing (carry direction, regime) lives here / on a context view-model.
- **`PricedStructure`** — valuation output of `price(structure, ms)`: per-leg *resolved* data
  (resolved strike, realized delta, unit price base/term, greeks) + linearly-aggregated package
  metrics (net premium, max loss, payoff@target, RR, breakeven, greeks), common currency, with
  provenance (which `Structure` + which `MarketState`).

### Pricing pattern (leg-level model polymorphism, uniform linear aggregation)
`price(structure, ms) -> PricedStructure`:
1. Per leg: resolve strike from spec (needs `ms`), price it with **the model appropriate to
   that leg** — vanilla on BS + smile (`vol_at_strike`), digital on the digital primitive
   (skew-corrected), future barrier on local-vol/Dupire/PDE. Produce unit price + greeks.
2. **Aggregate uniformly linearly** — value/greeks = `Σ signed_notional × unit`. Nonlinear
   structure metrics (max loss, breakeven) are operations on the linearly-aggregated payoff
   curve (`payoff(S) = Σ signed_notionalᵢ × leg_payoffᵢ(S)`, linear in legs at each `S`; then
   extremum). **No structure-type dispatch in the aggregator.** A leg may be *replicated* into
   sub-instruments for pricing (e.g. `european_rko` = vanillas + a digital strip) — internal to
   that leg's pricer; the result is still a per-leg price summed linearly.

**The one fenced corner:** continuously-monitored path dependence at *expiry scenarios* — a
continuous knock-out's expiry value depends on path, not terminal `S`. Complicates that leg's
*scenario* valuation, not the aggregation; already out of scope (CLAUDE.md keeps path-dependent
RKO scenario legs flat; `european_rko` is expiry-only → fully decomposable). Everything currently
priced fits the per-leg-model + linear-aggregation pattern.

### Agent boundary (preserved — no Phase-1 regression)
The agent **never authors** `Structure`/`Leg`. It requests via the Phase-1 grammar; the engine
resolves the request into a `Structure` and returns a `PricedStructure` for narration. Agent sets
*spec* only; the engine resolves strikes / realized deltas / premiums / call-put.

---

## 2. Blast radius — what's touched vs insulated

| Layer | Touched? | Why |
|---|---|---|
| **Pricing** (`analytics/structure_pricer.py`, `pricing/`) | **Heavily** — the refactor | `PricedVariant` → `Structure`/`PricedStructure`; per-family fns → per-leg model + linear aggregator |
| **Structure *selection* scoring** (`knowledge_engine/structure_scorer.py`, `affinity_scores.json`) | **No** | Scores *families* off `MarketState` → `structure_id` shortlist. Never sees a leg or priced variant. Insulated. |
| **Scenario / comparator eval** (`comparator.py`, `scenario_scorer`, `scenario_weighter`) | **Seam only** | Logic (weighting, `score_ccy`) unchanged; only the *input type* flips `PricedVariant → PricedStructure`. Mechanical. |
| **Sizing / Kelly** (`sizing_engine`, `interface/kelly_v2/pricing.py`) | **Seam only** | Payoff bridge reads the priced result; same. |
| **Scenario pricer** (`analytics/scenario_pricer.py`) | **Yes** | Becomes `price(structure, ms_scenarioᵢ)` over the grid |
| **UI** (`interface/structure_eval.py`) | **Seam** | Reads `PricedStructure` instead of `PricedVariant` |
| **Agentic** (`agentic/render.py`, `price_structure.py`, `standard_pack.py`, grammar) | **Yes** | Total renderer over `PricedStructure`; grammar emits `Structure` |

**The scoring intelligence barely moves.** Affinity selection is untouched; scenario scoring
logic is untouched (input type only). The real surface is **pricing + the data interface its
consumers read.**

---

## 3. Implementation phases (strangler-fig, parity-guarded, green at every step)

Core de-risking device: a **byte-parity harness** pinning the new pricer to the old before any
consumer migrates. Same spirit as the existing `test_premium_basis` / flat-identical pins.

### Phase A — Model + parity harness (ZERO behavior change)
- Define `Leg` / `Structure` / `PricedStructure` dataclasses (pure data; nothing wired).
- `build_structure(family, variant_dict, ms) → Structure` — convert each existing curated
  variant (`long_delta` / `spread_long`/`spread_short`/`wing_delta` / `target_prem_pct` …) into a
  leg list. No pricing yet.
- **Parity harness** (`tests/test_product_model_parity.py`): for every curated variant ×
  {USDBRL, USDTRY, EURPLN, GBPUSD} × a couple of targets/tenors, assert (once Phase B lands) the
  new `price(build_structure(...), ms)` matches legacy `price_variants(...)` on `strikes`,
  `net_premium_pct`, `payoff_at_target_pct`, `max_loss_pct` byte-for-byte (or < 1e-9).
- **Exit:** harness compiles and enumerates all variant×market combos; legacy side captured as
  the golden reference.

### Phase B — New pricer behind the old interface (parity-locked)
- Implement `price(structure, ms)`: per-leg model-appropriate pricing + linear aggregation +
  structure-metric operations on the aggregated payoff.
- Make it pass the parity harness **byte-for-byte** vs the legacy per-family fns. This is where
  the care goes — the digital skew correction, the seagull wing funding, the european_rko strip,
  the `/spot` base-ccy premium basis (CLAUDE.md load-bearing invariant) must all reproduce.
- Flip `price_variants()` to delegate to `price()` internally, returning a `PricedStructure` with
  a **back-compat shim** exposing `.strikes`, `.net_premium_pct`, `.wing_ratio`, etc.
- **Exit:** all ~440 existing tests green, no consumer changed. New engine live but invisible.

### Phase C — Migrate consumers to `PricedStructure` (one per commit, green between)
1. **Agentic renderer** (`agentic/render.py`) — read legs/deltas/notionals directly → **kills the
   Class-A fabrication class** (deltas, wing ratio, leg notionals are now first-class). Highest
   value, do first.
2. **Comparator** (`comparator.py`) — variant evaluation consumes `PricedStructure`.
3. **Scenario pricer** (`scenario_pricer.py`) — `price(structure, ms_scenarioᵢ)` over the grid.
4. **Kelly bridge** (`interface/kelly_v2/pricing.py`) — payoff from `Structure` (no per-family
   bridge).
5. **UI** (`interface/structure_eval.py`).
- Retire shim fields as each consumer stops using them.
- **Exit:** shim removed; all consumers on `PricedStructure`.

### Phase D — Grammar emits `Structure` + arbitrary-leg construction
- Phase-1 grammar resolves a request straight into a `Structure` (not a variant dict).
- Add the arbitrary-leg construction path → the agent composes bespoke packages and re-prices /
  scenario-scores them vs the standard pack (the capability payoff). Re-price under a new
  `MarketState` is the Tier-1 rebuild for free.
- **Exit:** agent can price a non-curated combination end-to-end, ranked against the pack.

### Phase E — JSON migration (optional, last)
- Migrate `structure_variants.json` to leg-list schemas. Deferrable indefinitely — the Phase-A
  `build_structure` adapter keeps the old JSON working until converted.

---

## 4. Parity-harness spec (the linchpin)

- **Coverage:** every entry in `structure_variants.json`, across the 4 supported pairs, at ≥2
  targets (one near, one extended) and ≥2 tenors. Include the deep-carry / inverted-carry cases
  that `tests/test_pricing.py` already exercises.
- **Asserted fields:** `strikes` (and `barrier`), `net_premium_pct`, `payoff_at_target_pct`,
  `rr_at_target`, `max_loss_pct`, `is_zero_cost`, `wing_ratio` (→ derived from leg notionals).
- **Tolerance:** byte-for-byte where the math is identical; < 1e-9 otherwise. Any intentional
  divergence must be explicitly whitelisted with a reason (there should be none).
- **Smile + flat:** run under both a `FlatSurface` and a built smile surface (mirror the existing
  flat-identical guards in `test_vol_surface_refactor.py`).
- **Runs in CI** for the whole of Phases B–C so no consumer migration silently shifts a number.

---

## 5. Risks & mitigations

- **Silent number drift** → the parity harness + back-compat shim (Phases A–B) make the internal
  swap invisible and verified before any consumer moves.
- **Big-bang temptation** → forbidden; consumers migrate one commit at a time (Phase C).
- **Digital / barrier edge cases** (skew correction, arb guard, european_rko strip) → these are
  the hard part of Phase B; the harness covers them explicitly, and the `SmileArbitrageError`
  guard must survive the move.
- **Premium-basis invariant** (`black76/spot`, base-ccy %) → pinned by the existing
  `test_premium_basis`; keep it green throughout.
- **Scope creep into scoring** → out of scope. Affinity scoring stays as-is; only input *types*
  change at the scenario seam.

---

## 6. Sequencing & estimate

- Phases A–B (model + parity-locked pricer) are the bulk of the care — a few focused days; this
  is where correctness is won or lost.
- Phases C–E are mechanical and parallelizable once parity holds.
- Sequence the grammar arbitrary-leg extension (Phase D) *with* this; it's the capability the
  whole refactor unlocks.
- It is the data half of CLAUDE.md's deferred `PricingContext` seam — consider folding that
  ergonomic change in during Phase B while the pricer signatures are already being reworked.
