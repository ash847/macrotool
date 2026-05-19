# Kelly v2 — Build Notes

Running log of decisions, surprises, and revisit candidates.

## Environment

- Shared `.venv` lives at the parent MacroTool repo root, not in this worktree. Run Python via the absolute path `/Users/ash/Documents/Coding work/MacroTool/.venv/bin/python`. Streamlit and pytest binaries similarly absolute. Not a problem, just noting.
- Folder name `Kelly v2` contains a space. Pytest and Streamlit handle it fine with quoted paths.

## Design choices made during build

### Step 1 — elicitation.py Option 1

- **Bin model:** bin *edges* (n_bins + 1 of them) over [p_outer_low, p_outer_high], `probs[i] = CDF(edge[i+1]) − CDF(edge[i])`, `bins[i]` = bin centre. Cleaner than centre-based finite differences and means renormalisation only needs to absorb the truncated tail mass (~`quantiles[0] + (1 − quantiles[-1])`), not floating-point noise.
- **Renormalisation:** clip raw probs to ≥ 0 (kills tiny PCHIP overshoot), then divide by sum. After renormalisation `probs` sums to exactly 1 in float.
- **Minimum N:** 3 anchors. Fewer than 3 doesn't make sense for a non-trivial CDF.
- **No tail extension:** grid truncates to [prices[0], prices[-1]]; mass outside the outer anchors is dropped. Matches PLAN.md decision. **Revisit candidate** — log here if real usage suggests we need a parametric tail.

### Step 4 — edge.py and sanity-check 1 finding

PLAN.md's identity test specified `edge ≈ 0 to within < 1 bp` when PM anchors are extracted from baseline at the default quantiles. **Unreachable** with the truncate-to-anchors policy. Reasoning:

- Default anchors span [p_2, p_98] of baseline.
- PM's elicited distribution covers only [p_2, p_98] and renormalises mass-sum to 1 (i.e. each interior bin gets scaled by ~1/0.96).
- Baseline covers full support, with ~4% mass in tails outside [p_2, p_98].
- For a vanilla call ATM under F=5, σ=0.10, T=0.25: baseline right-tail above p_98 contributes ~10% of the option value. PM has no mass there, but interior bins are scaled up ~4%. Net effect on ATM call edge is ~−0.6% of forward.

This is a **predictable, policy-induced edge**, not engine error. Two ways the engine could test sanity-check 1 cleanly:

- **Wide-anchor variant:** extract anchors at very wide quantiles (e.g. [0.001, 0.005, ..., 0.999]). Truncation drops < 0.2% of mass; engine error dominates and is < 5 bp.
- **Default-anchor variant:** accept the ~0.5–2% of forward edge as expected; assert it's in the right direction (typically negative for ATM call) and shrinks as quantile coverage widens.

Implemented both in `tests/test_edge.py`.

**Implication for the UI:** the "edge vs market-implied" label is still honest, but PMs should know that a chunk of the displayed edge for in-support strikes comes from the tail-truncation policy, not just view-divergence. Loud disclosure in the UI when this matters.

**Revisit candidate, reinforced:** if PMs find this annoying or misleading, switch to parametric-tail extension (option b from the PLAN). The cost is asking PMs to commit to a tail decay assumption they didn't make.

### Step 5 — Streamlit UI self-test

UI renders cleanly. Verified at initial render:
- All N anchor inputs render (N selector default 7; supports 5/7/9/11).
- Sidebar Forward/Vol/Tenor wired, anchor preset selector wired, reset button wired.
- Strike + Call/Put radio render.
- Edge readout displays PM price, market price, abs edge, and % of mid.
- "Edge vs market-implied" caption present.
- For F=5.0, σ=0.10, T=0.25, ATM call: PM=0.0888, mkt=0.0997, edge=-0.0110, -11% of mid. Matches expected truncation-policy bias from Step 4 analysis.
- Initial Streamlit warning about `value=` + `key=` clash on widgets fixed by removing `value=` arguments and seeding session state instead.

**Synthetic-event interaction testing didn't work.** Streamlit's number inputs use a debounced commit-on-blur pattern that doesn't pick up dispatched `input`/`change`/`keydown(Enter)` events from `javascript_tool`. Tried the React `Object.getOwnPropertyDescriptor` setter trick and Enter dispatch — neither triggered Streamlit's rerun.

Consequence: live UX interaction (sliders moving, edge updating, warnings firing) is for human verification, not automated self-test. Engine correctness is covered by unit tests (58/58 across elicitation, pricing, baseline, edge); the UI is only doing input wiring and display, and those render correctly at startup.

### Post-v2 — UI iteration and edge decomposition (PM-driven changes)

After the initial 9-step build, several UX issues surfaced during PM testing:

- **Mode labels in plain English.** "Option 1 / Option 2 (CDF) / (PDF)" replaced with "Use fixed probability bins" / "Use fixed spot ranges" — closer to how PMs think about the input mode.
- **Option 1 chart switched from histogram to strip plot.** The grouped-bar histogram conflated bin width with bar height (only market bars moved when PM adjusted prices, but the visual implied both market and user were changing). The strip plot shows what the PM is actually doing: positioning fixed quantile markers along a price axis. Single-dimension change.
- **Coloured inter-quantile bands on the Option 1 strip plot.** Same blueorange palette as Option 2's stacked allocation bar — a given probability band sits in the same colour across modes.
- **Option 2 charts stay visible while editing.** Initial code returned early when sum ≠ 1, blanking the grouped bar chart. Fixed so both charts render at every keystroke; only pricing is gated on sum-to-1.
- **Option 2 inputs as integer percent (step 1).** Switched from fractional (0–1) to integer (0–100), `format="%d"`. Streamlit coerces fractional input to nearest integer. Default seeds and "Renormalise to 100%" use largest-remainder (Hamilton) rounding so sums are exactly 100.
- **`st.session_state` modification rule.** The "Renormalise to 1" button initially failed with `StreamlitAPIException` because we wrote to `bucket_i` session-state keys *after* their widgets had been instantiated. Fixed by routing through `on_click` callbacks — Streamlit applies those before the next rerun.
- **Three-way edge decomposition.** PM raised: "the displayed edge isn't zero even when I exactly match the market — that's because of the 4% truncation, right? Adjust for it." Implemented:
  - `Full edge = price(pm) − price(market_full)`
  - `View edge = price(pm) − price(shadow_market)` where shadow = market re-elicited through PM's own anchor scheme
  - `Anchoring cost = price(shadow_market) − price(market_full)`
  - Identity: Full = View + Anchoring. All three shown in the UI with one-line captions.
  - **Kelly sizing (when added) operates on Full edge** — that's what the trade actually realises. View edge is the cognitive load-bearing number for "is my elicitation distinct from the market?".

### Conceptual exchanges worth preserving

- **Validity of the PDF in both modes.** The PCHIP monotonicity guarantee applies to both Option 1 and Option 2 because in both cases the engine fits a spline through (price, cumulative_probability) anchors where cumulative is monotone non-decreasing from 0 to 1. Per-bucket mass in Option 2 is preserved *exactly* at the boundary level (because the PCHIP spline passes through the boundary anchors); only the *within-bucket* mass distribution is smoothed.
- **PDF approach is European-only.** Mathematically exact (up to bin resolution) for any European payoff that's a function of S_T alone — vanilla, all spreads, RR, strangle, butterfly, seagull, 1×2, European digital, European RKO at expiry, European digital RKO at expiry. *Not* sufficient for continuously-monitored barriers (path-dependent); those need smile + dynamic model.
- **Smile parametrisation requires a model.** The smile at one tenor is mathematically equivalent to the marginal PDF at that tenor (via Breeden–Litzenberger). To unlock path-dependence you need a smile *plus a dynamic model* (Dupire local vol or stochastic vol); the smile alone doesn't conjure path info.

## Revisit candidates

- Grid extent (truncate-with-warning vs parametric tail) — see PLAN.md decisions section. Reinforced after step 4 (above). Three-way edge decomposition now makes the truncation cost explicit in the UI, which may reduce the need to change the policy itself.
