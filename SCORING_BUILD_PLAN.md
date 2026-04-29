# Two-Tier Scenario Scoring — Build Plan

## Goal

Convert the current scenario evaluation output (raw P&L per scenario) into a single weighted score per shortlisted structure, where:

1. The **scenario family weights** are derived from market context (Tier 1) and optionally adjusted by PM intent (Tier 2).
2. The score is a **weighted expected P&L**, denominated in % of spot and base ccy — sign and magnitude both meaningful.
3. The score complements but does not replace the affinity score. Affinity gates and orders the shortlist; scenario score reranks within the shortlist and flags when no structure has positive expected P&L.

---

## Core Principle: Baseline + Scaled Adjustments

Family weights start from a neutral baseline (1/8 each) and are scaled up or down by rules conditional on market state. Same architecture as the affinity scorer:

- Rules live in JSON, tunable without code changes
- Each rule fires a +/- adjustment to a single family weight
- After all rules fire, weights are renormalised to sum to 1
- A floor (e.g. 0.02) prevents any family from collapsing to zero

This pattern is already proven in `affinity_scores.json`. Reusing it gives one mental model and one tuning UI.

---

## Two Passes, Not One Composite

The system produces **two independent rankings** that are displayed side by side:

| Rank | Structure | Affinity | Weighted P&L | Recommended? |
|---|---|---|---|---|
| 1 | Vanilla call | +12 | +2.1% | ✓ |
| 2 | 1×1 spread | +9 | +0.4% | ✓ |
| 3 | 1×2 spread | +7 | −0.8% | ⚠ |

- **Affinity** answers: *Is this structure type the right fit for this market?*
- **Weighted P&L** answers: *Will this specific structure make money under expected paths?*

A disagreement between the two is information, not a bug. The PM keeps the judgment call. We never blend them into a single composite — that destroys interpretability.

---

## Tier 1 — Context-Derived Base Weights

Build now. Self-contained, no PM input required.

### Inputs

Already computed per query in `MarketState`:
- Carry regime (0/1/2) and direction (with/counter)
- Target z-score (target_z_abs)
- Tenor (T)
- ATM/FS ratio
- Vol level (ms.vol)

### Rule structure

A rule has a condition, a target family, and an adjustment magnitude. Example shape (illustrative only — actual numbers tuned via testing):

| Condition | Family | Adjustment |
|---|---|---|
| target_z_abs > 1.5 | OVERSHOOT | +0.05 |
| target_z_abs > 1.5 | EARLY_TARGET | −0.04 |
| target_z_abs < 0.5 | EARLY_TARGET | +0.05 |
| target_z_abs < 0.5 | OVERSHOOT | −0.04 |
| carry_regime == 2 AND with_carry | CORRECT_PATH | +0.05 |
| carry_regime == 2 AND with_carry | WRONG_WAY | −0.04 |
| carry_regime == 2 AND counter_carry | WRONG_WAY | +0.06 |
| carry_regime == 2 AND counter_carry | NO_MOVE | +0.04 |
| carry_regime == 0 | NO_MOVE | −0.03 |
| carry_regime == 0 | VOL_SENSITIVITY | +0.03 |
| T < 0.083 (≤ 1m) | EXPIRY | +0.04 |
| T < 0.083 (≤ 1m) | EARLY_TARGET | −0.03 |
| T > 0.5 (> 6m) | EARLY_TARGET | +0.03 |
| ms.vol > 0.20 | VOL_SENSITIVITY | +0.04 |

Start with **8–12 rules**. Resist the urge to over-specify before testing.

### Output

A weights dict: `{family_name: weight}` summing to 1, with a floor of 0.02 per family.

### Files

- `knowledge/defaults/scenario_weights.json` — rule definitions
- `knowledge_engine/scenario_weighter.py` — pure function loading rules and computing weights from MarketState
- `tests/test_scenario_weighter.py` — verify normalisation, floor, individual rules

---

## Tier 2 — PM Intent Overlay

Add later, after Tier 1 is in real use and feedback identifies the gap. Three explicit inputs at trade entry, not a long form:

| Question | Options |
|---|---|
| Conviction | high · tactical |
| Plan | hold to expiry · monetise on first move |
| Primary concern | wrong-way drawdown · capping the upside · theta bleed · vol move |

Each answer is a small set of weight adjustments applied **on top of** the Tier 1 base. Same rule pattern in JSON.

### Why not now

PMs without a calibrated mental model of weight dollar-equivalents will give noisy answers. Tier 1 produces a sensible default; Tier 2 only earns its place when testers consistently disagree with Tier 1's output for explainable reasons.

### Files (when built)

- `knowledge/defaults/scenario_intent_overlays.json` — intent-driven adjustments
- `interface/intent_picker.py` — three-question UI component
- Extend `scenario_weighter.py` to optionally apply intent overlays

---

## Scoring the Structures

Once weights exist, compute one score per shortlisted structure.

### Per-structure score

For a given structure (with priced variant) and weights `w_f` per family:

```
family_pnl[f]  = mean of pnl_pct across scenarios in family f
score          = Σ (w_f × family_pnl[f])
score_ccy      = score × structure_notional
```

Notes:
- Score is in **% of spot**, base ccy equivalent shown alongside.
- Sign is meaningful — negative score = expected loss across weighted scenarios.
- Magnitude is meaningful — directly comparable across structures and across queries.

### Aggregation choice within a family

Default: simple mean of `pnl_pct` across scenarios in the family. Alternatives we may consider later:
- Median (more robust to outliers)
- Worst-case (CVaR — only the bottom N% of P&L outcomes)
- Asymmetric: penalise downside more than upside (loss aversion)

Start with mean. Add alternatives only when an actual case demands it.

### Files

- `knowledge_engine/scenario_scorer.py` — pure function: `score_structure(scenario_rows, weights) -> dict`
- `tests/test_scenario_scorer.py`

---

## Display & UX

### Trade View — extend the Structure Evaluation section

Add a new sub-section above the per-family tables:

> **Weighted P&L summary**
>
> | Structure | Affinity | Weighted P&L | Recommended? |
> | --- | --- | --- | --- |
> | Vanilla call | +12 | +2.1% (€2.10) | ✓ |
> | 1×1 spread | +9 | +0.4% (€0.40) | ✓ |
> | 1×2 spread | +7 | −0.8% (−€0.80) | ⚠ |

The ⚠ flag fires on negative score. If **all** structures have negative weighted P&L, surface a warning above the table: *"No shortlisted structure has positive expected P&L under the weighted scenarios — consider not trading."*

### Show the weights themselves

A small expander labelled "Scenario weights for this trade" that shows the family weights as a horizontal bar — lets PM see what context-derived weights are doing without scrolling through rules.

### Make weights editable in the UI (later)

Once Tier 2 is in, expose sliders inside the same expander so the PM can override individual family weights. Override session-only (don't persist) — same pattern as affinity overrides.

---

## Architecture Summary

```
analytics/scenario_generator.py        ← unchanged (pure scenarios)
analytics/scenario_pricer.py           ← unchanged (per-scenario P&L)
knowledge_engine/scenario_weighter.py  ← NEW: MarketState → family weights (Tier 1)
knowledge_engine/scenario_scorer.py    ← NEW: weights × P&L rows → structure score
knowledge/defaults/scenario_weights.json     ← NEW: Tier 1 rules
knowledge/defaults/scenario_intent_overlays.json ← NEW (Tier 2, later)
interface/app.py                       ← extend Structure Evaluation
```

The data flow:

```
trade view → MarketState → scenario_weighter → family weights
                       ↓
              shortlist of structures
                       ↓
            for each: priced variant + scenarios
                       ↓
            scenario_pricer → per-scenario P&L
                       ↓
       scenario_scorer(rows, weights) → structure score
```

Each module is pure and independently testable. Adding Tier 2 is one new function call between weighter and scorer.

---

## What This Is *Not*

- Not a replacement for affinity scoring. Both run, both shown.
- Not a black-box ML score. Every weight and rule is inspectable in JSON and surfaced in the UI.
- Not a sizing recommendation. Sizing already comes from the linear-trade convention (`LINEAR_NOTIONAL × stop_pct`); the scenario score informs whether the trade is worth doing at that size.
- Not a single number ranking. PM sees affinity + weighted P&L side by side and makes the call.

---

## Build Order

1. **Tier 1 weighter** — `scenario_weighter.py` + JSON + tests. ~8 rules to start.
2. **Scenario scorer** — `scenario_scorer.py` + tests. Pure function.
3. **App integration** — weighted P&L summary table above per-family tables in Structure Evaluation.
4. **Weights expander** — show derived weights in UI.
5. **Iterate on rules** — testers will surface cases where weights feel wrong; tune the JSON.
6. **Tier 2 intent overlay** — only after step 5 has been running long enough to know it's needed.

Steps 1–4 are roughly the same work envelope as the original scenario generator + pricer build. Aim for a single sitting.

---

## Open Questions for Later

- **Cross-trade calibration.** Once we have many queries logged, can we calibrate weights by correlating "weighted P&L" with realised P&L of similar trades? Out of scope until there's data.
- **Variant-level scoring.** Currently we evaluate one variant per structure. Does scoring all variants and surfacing the best within each structure add value, or noise?
- **Path-dependent P&L.** Current scenarios are point-in-time snapshots. A real PM cares about the path: did the trade ever drawdown more than X% along the way? Path scenarios would need a new generator family.
- **Skew shocks in pricing.** Skew rules are recorded but ignored. When the pricer becomes smile-aware, the SKEW_SENSITIVITY family becomes meaningful and may need re-weighting.
