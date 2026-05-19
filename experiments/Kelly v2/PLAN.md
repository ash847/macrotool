# Kelly v2 — Subjective Distribution → Edge Prototype

Standalone Streamlit prototype. Lives in `experiments/Kelly v2/`. No coupling to the main MacroTool app yet — integrate later once UI and math feel right.

**Status:** v2 build complete (steps 1–9 + several UX iterations). Engine + UI shipped; all 95 tests pass.

## Goal of this prototype

Let a PM input a subjective view of where an FX rate will be at expiry as N anchor points, smooth it into a valid PDF, price a target option structure against both the PM's distribution and the market's smile-implied distribution, and report **edge vs market-implied price** for that structure — decomposed into pure view-divergence and elicitation-scheme cost.

Kelly sizing itself is deferred. This prototype stops at edge articulation.

## Scope (in)

- Streamlit UI for N-anchor distribution elicitation in **both** modes:
  - **Option 1 (CDF, "fixed probability bins")** — fixed quantiles, PM enters prices. Built first.
  - **Option 2 (PDF, "fixed spot ranges")** — fixed σ-anchored buckets, PM enters integer percent probabilities summing to 100. Built second.
- PCHIP spline construction on the anchors with validity enforcement.
- Variable bucket count (5/7/9/11), no `7` literals in the engine.
- Up-sampling to 200 micro-bins on a shared grid for PM and baseline.
- Market baseline: synthetic lognormal (closed-form) or loaded from a v1 JSON snapshot fixture.
- Pricing for vanilla call/put on both distributions, with discount factor folded into the pricing math (not surfaced in UI).
- **Three-way edge decomposition** — Full edge / View edge / Anchoring cost (added late in v2 after the truncation-bias question surfaced; see decisions section).
- Side-by-side visualisations of PM vs market in both modes — strip plot with coloured inter-quantile bands (Option 1), grouped bars + stacked allocation bar (Option 2).
- Sanity tests: Black-Scholes match on synthetic baseline; engine identity with wide anchors; default-anchor truncation directionality; cross-mode pricing consistency; decomposition identity.

## Scope (out, for now)

- Kelly fraction calculation (continuous or discrete). Defer until we trust the upstream pipeline.
- Multi-leg structures (spreads, seagulls, RKOs). Vanilla only until pricing is trusted (engine handles all linear combinations of vanillas trivially).
- Path-dependent structures (continuously-monitored barriers, lookbacks, Asians). Out of scope by construction — terminal PDF alone doesn't carry enough information; would need smile-parametrisation + dynamic model.
- Implied vol smile derivation (`PDF → σ_imp(K)`). Deferred to integration; PDF-level edge is sufficient for the prototype's purposes.
- Integration with MacroTool's `compute_smile_distribution()`. v1 JSON snapshot loader is in place — real smile data slots into the same schema.
- Persistence, auth, multi-pair support, scenario weights.

## Architecture

```
experiments/Kelly v2/
  app.py                # Streamlit entrypoint
  elicitation.py        # 7-point → PCHIP CDF → 200-bin PDF
  pricing.py            # Option payoff on bins; expected value under arbitrary PDF
  baseline.py           # Load / construct market-implied PDF (mock or real snapshot)
  edge.py               # Compare PM vs baseline pricing; emit edge metric
  tests/
    test_elicitation.py
    test_pricing.py
    test_edge.py
  fixtures/
    snapshot_usdbrl.json  # Saved market snapshot for baseline
  PLAN.md
  NOTES.md              # Running log of decisions / surprises
```

## Inputs (UI)

Both modes supported. Build Option 1 first; add Option 2 once Option 1 is trusted.

### Option 1 — CDF (fixed quantiles)

**7 anchor points**, fixed probability levels: `[2%, 10%, 25%, 50%, 75%, 90%, 98%]`.
- PM enters the price level at each quantile.
- UI enforces strict monotonicity: `p_2 < p_10 < p_25 < ... < p_98`.
- Median (50%) defaults to current forward; PM can override.
- Tails beyond 2% / 98% — see Grid extent decision below.

### Option 2 — PDF (fixed ranges)

**7 bucket probabilities**, fixed price-range buckets (boundaries derived from the market smile, e.g. ±0.5σ, ±1σ, ±1.5σ around forward, with outer-tail buckets).
- PM enters the probability mass in each bucket.
- UI enforces sum-to-100% via proportional rescaling or stacked-bar input.
- PCHIP fits a smooth PDF through bucket midpoints; renormalised post-fit.
- Bucket boundaries are deliberately not symmetric in price — they're symmetric in σ on the market smile. This makes Option 2 comparable to Option 1 by construction.

**Pair / expiry / structure target:** dropdowns or fixed for the prototype. Start with USDBRL 3M vanilla call at a chosen strike.

## Engine

### Spline construction (`elicitation.py`)

1. Take 7 (quantile, price) pairs.
2. Fit `scipy.interpolate.PchipInterpolator` on (price, quantile) — i.e., the CDF as a function of price.
3. Validate monotonicity post-fit (PCHIP guarantees it given monotone input, but assert anyway).
4. Evaluate CDF on a fine price grid (200 points spanning from p_2 minus a buffer to p_98 plus a buffer; buffer = e.g. 10% of the p_98 − p_2 range, but tail probability mass beyond the anchors is zero — document this).
5. Derive PDF as finite difference of CDF over the grid. Renormalise to sum to 1 over the 200 bins.
6. Return: `bins` (array of 200 prices), `probs` (array of 200 probabilities summing to 1).

Edge cases to handle:
- Two anchors very close → CDF nearly vertical → PDF spike. Cap PDF density or warn.
- PM picks tails inside the median range → reject at UI level (monotonicity check).

### Pricing (`pricing.py`)

```python
def expected_payoff(bins, probs, payoff_fn):
    return sum(probs * payoff_fn(bins))
```

Where `payoff_fn` for a vanilla call is `max(S - K, 0)`. No discounting in v2 (or apply a flat discount factor — declare which).

### Baseline (`baseline.py`)

Two implementations behind one interface:
- `synthetic_lognormal_baseline(spot, vol, T, forward, n_bins=200)` — for development and sanity checks.
- `smile_baseline_from_snapshot(snapshot_path, pair, tenor, n_bins=200)` — read a saved MacroTool snapshot and produce a smile-implied PDF using the same Breeden-Litzenberger or scenario-weight logic the main app uses. Can be a thin port of `analytics/distributions.py` or just load a pre-computed array from JSON to avoid the dependency.

In both cases, output schema is identical: `(bins, probs)` aligned to the *same* grid as the elicited PDF (re-interpolate to the same 200 bin centres so subtraction is meaningful).

### Edge (`edge.py`)

Initial draft was a single "edge = price(pm) − price(market)". The build surfaced a real interpretation problem: under the truncate-to-anchors policy, PM matching market still shows a non-zero edge (the "anchoring cost" — ~11% of an ATM call for default 7 anchors at q=[2, 98]). v2 ships with a three-way decomposition:

```
Full edge      = price(pm) − price(market_full)        # what your trade earns vs market
View edge      = price(pm) − price(shadow_market)      # pure view-divergence
Anchoring cost = price(shadow_market) − price(market)  # cost of the 4% truncation policy

Full edge = View edge + Anchoring cost
```

Where `shadow_market` is the market re-elicited through PM's own anchor / bucket scheme (built via `shadow_market_from_cdf_anchors` for Option 1, `shadow_market_from_pdf_buckets` for Option 2). The shadow cancels truncation and PCHIP smoothing, isolating view-divergence.

UI displays all three metrics with one-line captions; Kelly sizing (when added) operates on **Full edge** because that's what trade economics actually realise.

## Sanity checks (build these as tests)

1. **Identity test:** if PM's anchors are read from the market's smile-implied CDF, edge ≈ 0 to within < 1 bp. This is the build-order step 7 from the prior chat.
2. **Vanilla closed-form:** synthetic lognormal baseline priced via 200-bin expected value matches Black-Scholes to < 1 bp for ATM and ±1σ strikes.
3. **CDF validity:** spline output is monotonic non-decreasing on the full grid, starts ≤ 0.02, ends ≥ 0.98.
4. **PDF validity:** all probs ≥ 0, sum to 1.0 within float tolerance.
5. **Tail behaviour:** moving the 2% anchor lower (heavier left tail) increases the price of an OTM put on the PM distribution but not on the baseline.
6. **Monotone in obvious direction:** if PM shifts the entire distribution right by Δ, edge on an OTM call increases.

## Build order

1. **`elicitation.py` Option 1 + tests** — pure functions, no UI. CDF anchors → PCHIP → 200-bin PDF, validity-checked.
2. **`pricing.py` + tests** — vanilla payoff, expected value under arbitrary PDF, discount factor applied.
3. **`baseline.py`** — synthetic lognormal only. Run sanity check 2 (closed-form match).
4. **`edge.py`** — wire Option 1 elicitation + pricing + baseline. Run sanity checks 1, 5, 6.
5. **`app.py` v1** — Streamlit UI for Option 1: 7 price inputs, strike picker, edge readout (abs + % of mid). Tune UX until elicitation feels natural.
6. **`elicitation.py` Option 2 + tests** — PDF bucket anchors → PCHIP → 200-bin PDF, sum-to-1 enforced.
7. **`app.py` v2** — add Option 2 UI (stacked-bar or 7 % inputs with sum lock). Mode toggle between Option 1 and Option 2.
8. **Cross-mode sanity** — feeding the same belief through both modes should produce ~the same edge. Add as a test.
9. **Real baseline** — port or load smile-implied PDF from a saved MacroTool snapshot. Re-run all sanity checks.
10. **(Future, not v2)** Multi-leg structures, Kelly fraction, integration into MacroTool.

## Decisions locked in

- **Discount factor:** included in pricing math (default 1.0 in the prototype since it cancels out of edge under consistent application), not surfaced in UI.
- **Grid extent:** truncate to the outer anchors; tail mass outside is dropped. UI shows a hard warning when a structure strike falls outside this range. No parametric tail in v2. **Revisit candidate** — see `NOTES.md`. If PMs routinely care about strikes outside their anchors, switch to a parametric tail or expand the anchor count.
- **Edge decomposition:** Full edge / View edge / Anchoring cost displayed in both modes. The 4% tail truncation effect is surfaced explicitly rather than hidden — labelled in the UI as the anchoring cost.
- **Option 1 input increments:** prices as floats with step=0.01, format `.4f`.
- **Option 2 input increments:** integer percent (0–100), step=1, format `%d`. Largest-remainder (Hamilton) rounding ensures defaults and renormalisation sum to exactly 100.
- **Option 2 bucket boundaries:** σ-anchored linearly across [−2.5σ, +2.5σ] on the market smile. Options 1 and 2 are directly comparable.
- **Edge display:** absolute (quote ccy per unit notional) + % of mid premium, three metrics (Full / Anchoring / View) side-by-side.
- **Baseline format on disk:** v1 JSON schema with `bins`, `probs`, `pair`, `forward`, `tenor_years`, `source`. Roundtrips via `save_snapshot` / `load_snapshot`. Real smile-implied data can slot in by writing to this format.
- **"Edge vs market-implied" labelling** — explicit in the UI; never claim to isolate pure forecasting edge.
- **Visualisations:** Altair (Streamlit-native; no extra dep). Option 1 strip plot with coloured inter-quantile bands; Option 2 grouped bars + stacked allocation bar. Same blueorange palette across modes so a given probability band reads the same colour everywhere.
- **Mode labels in UI:** plain English — "Use fixed probability bins" / "Use fixed spot ranges" (rather than "Option 1 (CDF)" / "Option 2 (PDF)").

## Flexibility requirement — variable bucket count

The number of anchors / buckets must be a **runtime parameter**, not hardcoded. v2 ships with **7** as the default, but the code must support changing N without refactors. Potentially exposed as a UI control in a later version (e.g. "use 5 / 7 / 11 anchors").

Concretely:
- `elicitation.py` functions take an `anchors` array of arbitrary length (Option 1) or a `buckets` config of arbitrary length (Option 2). No `7` literal anywhere in the engine.
- Default anchor sets (the `[2, 10, 25, 50, 75, 90, 98]` quantiles for Option 1, and the σ-band boundaries for Option 2) live in a config dict at the top of `elicitation.py` — easy to swap or extend.
- The UI builds inputs by iterating over the active anchor set, not by hardcoding 7 widgets.
- Tests parametrise over N ∈ {5, 7, 11} to confirm the engine doesn't silently depend on a specific count.
- Sanity checks (validity, monotonicity, sum-to-1) all phrased in terms of N, not 7.

## Definition of done for v2 — met

- ✅ Streamlit app runs locally with mode toggle.
- ✅ Either mode accepts N anchors (5/7/9/11), displays a three-way edge decomposition for a chosen vanilla strike.
- ✅ All sanity tests pass in both modes (engine identity, BS closed-form match, tail behaviour, monotone direction, decomposition identity, cross-mode consistency).
- ✅ Out-of-range strike triggers a warning, not a silent zero.
- ✅ `NOTES.md` captures non-trivial decisions made during the build.
- ✅ Both modes have side-by-side visualisations comparing PM vs market.
- 🟡 UX validation by PM — pending hands-on use.

**Tests:** 95/95 passing.

**Test breakdown:**
- `test_elicitation.py` — Option 1 spline + binning, parametrised over N (17 tests)
- `test_elicitation_option2.py` — Option 2 bucket → CDF construction, σ-boundary helpers (16 tests)
- `test_pricing.py` — vanilla payoff arithmetic, DF, put-call parity, ATM equivalence (13 tests)
- `test_baseline.py` — synthetic lognormal, BS closed-form match across F/σ/T/K and call/put (14 tests)
- `test_edge.py` — engine identity, default-anchor truncation directionality, tail/skew sanity, shadow-market construction, three-way decomposition identity (20 tests)
- `test_cross_mode.py` — same belief → same pricing across modes (5 tests)
- `test_snapshot.py` — JSON round-trip + validation (7 tests)

## What's next (out of scope for v2)

- **Kelly fraction** — closed-form continuous (Thorp) + discrete log-utility solver. Operates on Full edge.
- **Path-dependent payoff handling** — terminal PDF is insufficient; needs smile parametrisation + dynamic model (local vol or stochastic vol).
- **PDF → implied vol smile transformation** — `pdf_to_implied_vol(dist, strikes, F, T)` via root-find on BS. Useful for integrating with MacroTool's smile-based pipeline.
- **Multi-leg structure pricing** — vanilla payoff combinations (spreads, RR, seagulls, butterflies, European RKO at expiry, European digital RKO at expiry). Engine handles these trivially via linear combinations of vanilla prices; just needs a `structure_payoff(name, strikes)` registry.
- **Real smile-implied snapshots** — exporter on the main MacroTool side that writes `compute_smile_distribution()` output into the v1 JSON schema in `fixtures/`.
- **Slider input mode** — drag-segments on the stacked allocation bar (instead of/alongside number inputs). Requires a custom Streamlit component.
