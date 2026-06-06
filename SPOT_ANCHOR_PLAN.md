# Spot-Anchored Scoring & Scenarios — Build Plan

**Branch:** `feature/spot-anchored-scoring` (worktree `/Users/ash/Documents/Coding work/codex`, off `main`).
**Status:** planning. Nothing implemented yet.

## Goal

Make the **evaluation layer** (σ-distance scoring + scenario grids) center on **spot**
instead of the **forward**, because forward-anchoring is unintuitive on high-carry pairs
(the forward sits far from spot, so a modest move-from-spot reads as a large/small σ-move
depending on the carry). PMs think in **moves from where spot is now**.

## Core principle — two anchors, deliberately separated

The forward keeps two jobs; spot takes over one of them:

| Concern | Anchor | Changes? |
|---|---|---|
| **Pricing** (Black-76 martingale, premiums, deltas, scenario MtM) | **Forward** | NO — no-arb requires it |
| **Trade construction** (put vs call, strike OTM-ness, variant eligibility) | **Forward moneyness** | NO — market convention; puts/calls stay forward-relative |
| **Scoring σ-distance** (`target_z` → affinity buckets/gates) | **Spot** | YES |
| **Scenario grid centering** (where could spot be at the checkpoints) | **Spot** | YES |

So the forward never disappears — it stays the pricing center and the construction anchor.
We only re-anchor *how far is the target* and *where do scenarios sit* to spot.

**Explicit non-goal:** do NOT change `put_call`, strike resolution, digital/RKO bounds,
or any `pricing/` code. Those remain forward moneyness.

## Where the forward is the anchor today (surface to change)

- `analytics/market_state.py`
  - `target_z = ln(target/fwd)/(σ√T)` → scoring distance, **forward-relative**. ← move to spot.
  - `put_call = "Call" if target > fwd else "Put"` → **KEEP forward** (construction).
  - `c = ln(fwd/spot)/(σ√T)` → the carry itself, KEEP.
  - `atmfsratio`, `with_carry` → inherently fwd-vs-spot, KEEP.
- `knowledge/defaults/affinity_scores.json`
  - `target_z_abs` buckets + `target_z_abs_min/max` gates calibrated to **forward-relative** σ.
    → must be **re-tuned** for spot-relative σ (shift differs by carry regime).
- `analytics/scenario_generator.py`
  - Grid built in forward space: columns `F`, `K`, `±σ` offsets from `F`/`K`;
    `scenario_spot = scenario_fwd · e^{-(r_d-r_f)τ}`. → re-anchor centering to spot.
- `analytics/structure_pricer.py`
  - `min_target_z` variant-eligibility checks (½σ / ratio-spread variants) use **forward**
    σ-distance → KEEP forward (this is construction).
- `knowledge/defaults/scenario_definitions.json`
  - Weightings reference `target_z_abs` (and fire conditions). → re-tune against spot σ.

## Key design decisions (to lock before/while building)

1. **Keep BOTH σ-distances on MarketState.** Add `target_z_spot = ln(target/spot)/(σ√T)`
   as a new field; keep existing `target_z` (forward) for construction/eligibility/`put_call`.
   Affinity scoring switches to `target_z_spot`. This is the clean separation:
   *construction stays forward, scoring moves to spot.*

2. **Scenario centering = zero-drift around spot (subjective view), priced at per-cell forward.**
   The grid's "no-move" column = spot unchanged; σ-offset columns measured from spot with
   `σ_t = vol·√(elapsed)`. Each cell still derives a forward for MtM pricing via CIP
   (`scenario_fwd = scenario_spot · e^{(r_d-r_f)τ}`). So the *grid* is spot-centered (where
   spot could be), each *point* is priced forward-correctly. The `K`/target columns are
   unchanged (target is target).

3. **Carry discipline / don't flatter carry trades.** Forward-anchoring quietly reminded us
   the market already prices the drift. Spot-centered scenarios weight a carry trade by the
   PM's *subjective* "spot stays put" — which can inflate carry trades that are ~zero-EV under
   risk-neutral. **Resolved:** (a) keep carry as an explicit *scoring* input (`carry_regime`,
   `with_carry`, `carry_alignment`) — engine, unchanged; (b) keep the carry *visible* to the PM
   via the grid's retained `F` reference column **and** the forward shown in each time-horizon
   row label (`25% T (fwd = X.XX)`). The forward never leaves the screen; it's just no longer the
   grid's center.

4. **Re-tune, don't reuse, the affinity buckets.** Spot-relative `target_z` values differ from
   forward-relative by ~`c` (large on USDTRY/USDBRL, tiny on GBPUSD). The bucket boundaries
   (`near/moderate/extended/far`) and gates must be re-derived, likely per carry regime.

## Phased plan

### Phase 0 — Non-destructive spike (decide before touching live path)
- [ ] Compute `target_z_spot` alongside `target_z` (forward) in a throwaway script.
- [ ] Build a spot-anchored scenario grid alongside the forward grid.
- [ ] Run both across all 4 pairs (USDBRL, USDTRY, EURPLN, GBPUSD) at a few tenors/targets.
- [ ] Diff: how much do σ-distances shift per pair? How does the scenario grid's spot
      coverage change? Where would affinity buckets reclassify? (put_call unchanged.)
- [ ] Output a short findings note → confirm approach + size the re-tune. **Gate to Phase 1.**

### Phase 1 — MarketState
- [ ] Add `target_z_spot` field to `MarketState`; compute in `compute_market_state`.
- [ ] Keep `target_z` (forward), `put_call`, `c`, `with_carry`, `atmfsratio` unchanged.
- [ ] Tests: pin both σ-distances; assert `put_call` still forward-derived.

### Phase 2 — Affinity scoring
- [ ] Point `target_z_abs` bucket + gate at `target_z_spot` in `structure_scorer.py`.
- [ ] Re-tune `affinity_scores.json` `target_z_abs` buckets + `target_z_abs_min/max` gates
      for spot-relative σ (informed by Phase 0). Keep carry dimensions explicit.
- [ ] Tests: update `test_structure_scorer.py` expectations; new cases on a high-carry pair.

### Phase 3 — Scenario grid
- [ ] Re-anchor `scenario_generator.py` centering to spot (no-move = spot; σ-offsets from spot,
      `σ_t` scaling unchanged). Derive `scenario_fwd` per cell via CIP for MtM pricing.
- [ ] **Retain the `F` column** as a labeled forward-reference (demoted from "no-move center" to a
      reference, NOT removed) — preserves the carry cue in-grid.
- [ ] Keep `K`/target columns; keep pricing forward-based in `scenario_pricer.py`.
- [ ] Define + expose a per-row forward (the remaining-time / roll-down forward at each checkpoint)
      so the row labels can show it (see Phase 5). Decide exactly which forward: remaining-time
      outright `spot·e^{(r_d-r_f)·remaining}` (rolls down to spot at expiry).
- [ ] Tests: update scenario-grid + `test_scenario_pricer.py` for spot-centered levels.

### Phase 4 — Scenario weighting re-tune
- [ ] Re-tune `scenario_definitions.json` weightings against spot-centered scenarios + spot σ.
- [ ] Tests: `test_scenario_weighter.py`.

### Phase 5 — UI / display
- [ ] Surface the spot-relative move (σ-from-spot) in the Trade View market-state caption.
- [ ] **Time-horizon row labels show the forward at each checkpoint**, e.g. `25% T (fwd = X.XX)` —
      gives the carry / forward-drift context inline (replaces a separate caption). Forward def
      from Phase 3 (remaining-time roll-down).
- [ ] Keep forward-OTM strike labels (construction is still forward moneyness).

### Phase 6 — Integration + docs
- [ ] `demo.py` smoke across 4 pairs; full `pytest`.
- [ ] Update `CLAUDE.md` (market-state / affinity / scenario sections) + memory.
- [ ] Version bump, PR.

## Invariants (do not break)
- Pricing stays forward-anchored (Black-76). `pricing/` untouched.
- `put_call`, strike OTM-ness, variant eligibility (`min_target_z`) stay **forward moneyness**.
- Flat-vol byte-identical guards still pass.
- Carry stays an explicit scoring dimension.
- All three evaluation layers (market state, affinity, scenarios) move together — no half-state.

## Open questions / risks
- Carry-EV discipline (decision #3) — keep an explicit carry/drift term, or accept spot-centered
  weighting flatters carry? Resolve in Phase 0.
- Re-tune scope: are spot-relative buckets carry-regime-dependent? Likely yes.
- Test surface is large (target_z, scenario levels, affinity ranks). Budget for it.
- Backward-compat of saved Supabase configs (affinity_scores / scenario_definitions) tuned to
  forward σ — re-tuned values are a config change, not just code.

## Phase 0 findings (spike: `spikes/phase0_spot_vs_fwd.py`, 6M)

**The whole shift is exactly the carry `c`.** Identity confirmed: `target_z_fwd = target_z_spot − c`,
and the evaluation grid is the spot grid scaled by `fwd/spot = e^{c·σ_T}`. So one number — the
normalised carry — drives every divergence. Per-pair `c` (6M): USDBRL +0.31, **USDTRY +1.20**,
EURPLN +0.39, GBPUSD −0.05.

**[A] Selection (target_z buckets/gates) — material on carry pairs, negligible on G10.**
- **USDTRY (c≈1.2):** a target placed **1σ above spot** maps to `z_fwd ≈ −0.2` → bucket `near`
  AND fails the `1x1_spread` gate (|z|≥0.5), while spot-anchored it's `moderate` and passes. A
  target **0.5σ above spot** is a forward *put* (below fwd). Buckets/gates flip on **6 of 8** sampled
  targets. This is exactly the unintuitive behaviour.
- **USDBRL / EURPLN (c≈0.3–0.4):** flips on ~half the rows, mostly one bucket over, near boundaries.
- **GBPUSD (c≈−0.05):** essentially identical — spot vs fwd agree (one boundary nudge).
- `put_call` is forward-derived throughout (unchanged by design) — confirmed in the table.

**[B] Evaluation (scenario grid) — the headline.** The forward-centered grid's **"no-move" cell is
spot ending at the forward**, empirically validated: USDTRY Expiry/F `scenario_spot = 44.76 = fwd =
**+16.4%** above spot`. So today's "no move" scenario models USDTRY *rising 16%*. Spot-centered
"no-move" = spot unchanged (0%). The entire grid is shifted up by the carry:
  - USDTRY no-move +16.4% → 0%; −1σ +2.5% → −11.9%; +1σ +32.2% → +13.6%.
  - USDBRL/EURPLN shift ~2–4%; GBPUSD ~0.3% (negligible).

**Conclusions / decisions surfaced:**
1. The change is real and worth it on the carry pairs (USDTRY especially); near-noise on G10 — so
   nothing breaks for GBPUSD, the carry pairs get the intuition fix.
2. Re-tune IS carry-regime-dependent (the shift = c, which *is* the regime axis). Bucket boundaries
   will need re-derivation primarily for regime 1–2.
3. **Carry-discipline decision (plan §3) is now concrete:** spot-centering makes a with-carry target
   look like a smaller move (good intuition) AND the "no-move" scenario stops crediting a 16% drift.
   That removes the implicit double-count — but means the carry tailwind must be visible elsewhere
   (explicit carry dimension / a drift annotation), else carry trades lose the "it's already moving
   our way" signal entirely. Recommend: keep `c`/`carry_alignment` weighting, add a visible
   "forward drift = +X%" annotation in the UI so the PM sees both lenses.

## Progress log
- (init) Plan written; codex worktree reset to clean copy of main on `feature/spot-anchored-scoring`.
- Phase 0 spike done (`spikes/phase0_spot_vs_fwd.py`): confirmed `z_fwd = z_spot − c`; quantified
  bucket/gate flips (heavy on USDTRY, negligible on GBPUSD) and the +16.4% USDTRY "no-move" grid
  artefact. Findings above. **Gate cleared — approach confirmed.** Next: Phase 1 (MarketState).
