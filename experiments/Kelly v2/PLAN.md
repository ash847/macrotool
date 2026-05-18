# Kelly v2 — Subjective Distribution → Edge Prototype

Standalone Streamlit prototype. Lives in `experiments/Kelly v2/`. No coupling to the main MacroTool app yet — integrate later once UI and math feel right.

## Goal of this prototype

Let a PM input a subjective view of where an FX rate will be at expiry as 7 anchor points, smooth it into a valid PDF, price a target option structure against both the PM's distribution and the market's smile-implied distribution, and report **edge vs market-implied price** for that structure.

Kelly sizing itself is deferred. This prototype stops at edge articulation.

## Scope (in)

- Streamlit UI for 7-point distribution elicitation in **both** modes:
  - **Option 1 (CDF)** — fixed quantiles, PM enters prices. Built first.
  - **Option 2 (PDF)** — fixed price ranges, PM enters probabilities summing to 100%. Built second, after Option 1 is trusted.
- PCHIP spline construction on the anchors with validity enforcement.
- Up-sampling to 200 micro-bins on a shared grid for PM and baseline.
- Market baseline distribution loaded from a saved snapshot (synthetic lognormal first, then real smile-implied).
- Pricing for vanilla call/put on both distributions, with discount factor folded into the pricing math (not surfaced in UI).
- Edge readout: absolute (quote ccy per unit notional) **and** % of mid premium.
- Zero-edge sanity test (vanilla priced under baseline ≈ Black-Scholes).

## Scope (out, for now)

- Kelly fraction calculation (continuous or discrete). Defer until we trust the upstream pipeline.
- Multi-leg structures (spreads, seagulls). Vanilla only until pricing is trusted.
- Integration with MacroTool's `compute_smile_distribution()`. Use a saved snapshot or synthetic baseline.
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

```python
edge_value = expected_payoff(bins, pm_probs, payoff) - expected_payoff(bins, mkt_probs, payoff)
edge_bps   = edge_value / spot * 10_000
```

UI displays both absolute (in quote currency per unit notional) and bp-of-spot.

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

- **Discount factor:** included in pricing math, not surfaced in UI.
- **Grid extent:** truncate to the outer anchors; tail mass outside is dropped. UI shows a hard warning when a structure strike falls outside this range. No parametric tail in v2. **Revisit candidate** — flagged in `NOTES.md` for later reconsideration once we see real PM usage. If PMs routinely care about strikes outside their anchors, switch to a parametric tail (Option b) or expand the anchor count.
- **Option 2 bucket boundaries:** σ-anchored on the market smile (e.g. ±0.5σ, ±1σ, ±1.5σ around forward, with outer-tail buckets), so Options 1 and 2 are directly comparable.
- **Edge display:** absolute (quote ccy per unit notional) + % of mid premium.
- **Baseline format on disk:** store the pre-computed (price, prob) array per snapshot/tenor. Recomputation from smile deferred to integration.
- **"Edge vs market-implied" labelling** — explicit in the UI; never claim to isolate pure forecasting edge.

## Flexibility requirement — variable bucket count

The number of anchors / buckets must be a **runtime parameter**, not hardcoded. v2 ships with **7** as the default, but the code must support changing N without refactors. Potentially exposed as a UI control in a later version (e.g. "use 5 / 7 / 11 anchors").

Concretely:
- `elicitation.py` functions take an `anchors` array of arbitrary length (Option 1) or a `buckets` config of arbitrary length (Option 2). No `7` literal anywhere in the engine.
- Default anchor sets (the `[2, 10, 25, 50, 75, 90, 98]` quantiles for Option 1, and the σ-band boundaries for Option 2) live in a config dict at the top of `elicitation.py` — easy to swap or extend.
- The UI builds inputs by iterating over the active anchor set, not by hardcoding 7 widgets.
- Tests parametrise over N ∈ {5, 7, 11} to confirm the engine doesn't silently depend on a specific count.
- Sanity checks (validity, monotonicity, sum-to-1) all phrased in terms of N, not 7.

## Definition of done for v2

- Streamlit app runs locally with mode toggle between **Option 1 (CDF)** and **Option 2 (PDF)**.
- Either mode accepts 7 anchors, displays edge for a chosen vanilla strike (absolute + % of mid premium).
- All 6 sanity tests pass for each mode.
- Cross-mode sanity test passes (same belief expressed two ways → comparable edge).
- Out-of-range strike triggers the warning, not a silent zero.
- `NOTES.md` captures every non-trivial decision made during the build.
- The elicitation UX is good enough in both modes that you'd actually use it on a real view. If not, iterate before declaring done.
