# Scenario Grid Evaluation Spec

## Purpose

Replace the current family-based Structure Evaluation weighting model with a
direct scenario-grid model.

Instead of assigning weights to broad scenario families, each context will
assign numeric multipliers to explicit scenario grid points. The final score for
a structure variant will be the weighted average of the P&L across the active
grid points.

This spec is the agreed artifact for the refactor.


## High-Level Change

Current model:

- Scenario generator emits scenarios grouped into families
- Contexts assign weights to families
- Structure score is derived from family-weighted average P&L

New model:

- Scenario generator emits explicit scenario grid points
- Contexts retain their current names
- Each context stores numeric multipliers per valid grid cell
- Invalid cells do not exist for scoring and are shown as `-` in the UI
- Structure score is the weighted average of per-cell P&L


## Sigma Convention

All `σ` labels in the scenario grid are **time-scaled**:

`σ_t = vol × sqrt(elapsed_time)`

This applies to:

- `K+½σ`
- `−½σ`
- `−1σ`

This is intentionally different from the old bugged behaviour where full-tenor
`vol × sqrt(T)` was reused for every row.


## Scenario Grid

Rows:

- `1w`
- `25%T`
- `50%T`
- `Expiry`

Columns:

- `F`
- `t%→K`
- `K`
- `K+½σ`
- `−½σ`
- `−1σ`
- `Δvol`

Grid availability:

| Row / Column | F | t%→K | K | K+½σ | −½σ | −1σ | Δvol |
|---|---|---|---|---|---|---|---|
| `1w` | valid* | - | - | - | - | - | valid* |
| `25%T` | valid | valid | valid | valid | valid | valid | valid |
| `50%T` | valid | valid | valid | valid | valid | valid | valid |
| `Expiry` | valid | - | valid | valid | - | valid | - |

\* `1w` row exists only when `T > 6 weeks`.


## Column Definitions

### `F`

Forward remains unchanged.

- Scenario forward = current forward for that expiry point
- This is the no-move / carry / theta anchor

### `t%→K`

Forward has moved the same fraction of the log-distance to target as elapsed
time.

Examples:

- At `25%T`: forward is 25% of the log-distance from `F` to `K`
- At `50%T`: forward is 50% of the log-distance from `F` to `K`

This is the "on-pace path" scenario.

This column is invalid at expiry.

### `K`

Forward is exactly at target.

- Pre-expiry: early touch / monetisation style scenario
- At expiry: terminal target-hit scenario

### `K+½σ`

Target overshot by half a time-scaled sigma.

Definition:

- Bullish direction: `scenario_fwd = K × exp(+0.5 × σ_t)`
- Bearish direction: `scenario_fwd = K × exp(-0.5 × σ_t)`

### `−½σ`

Adverse half-sigma move from the **current forward**, not from the
proportional-progress point.

Definition:

- Bullish direction: `scenario_fwd = F × exp(-0.5 × σ_t)`
- Bearish direction: `scenario_fwd = F × exp(+0.5 × σ_t)`

Invalid at expiry.

### `−1σ`

Adverse full-sigma move from the **current forward**, not from the
proportional-progress point.

Definition:

- Bullish direction: `scenario_fwd = F × exp(-1.0 × σ_t)`
- Bearish direction: `scenario_fwd = F × exp(+1.0 × σ_t)`

Valid at expiry.

### `Δvol`

Vol shocked up/down by 1 vol point, with the forward held fixed at the row’s
proportional-progress anchor.

Definition:

- Two sub-scenarios are priced:
  - `vol - 0.01`
  - `vol + 0.01`
- Final cell P&L is the arithmetic average of the two

Forward anchor by row:

- `1w`: forward unchanged (`F`)
- `25%T`: proportional-progress anchor (`25%→K`)
- `50%T`: proportional-progress anchor (`50%→K`)

Invalid at expiry.


## Row Definitions

### `1w`

Only included when tenor `T > 6 weeks`.

Purpose:

- Pure short-horizon carry/theta diagnostic

Valid cells:

- `F`
- `Δvol`

For this row:

- `elapsed_time = 1 week`
- `remaining_time = T - 1 week`

### `25%T`

- `elapsed_time = 0.25 × T`
- `remaining_time = 0.75 × T`

### `50%T`

- `elapsed_time = 0.50 × T`
- `remaining_time = 0.50 × T`

### `Expiry`

- `elapsed_time = T`
- `remaining_time = 0`

At expiry:

- Vol shocks are invalid because only intrinsic value matters
- `t%→K` is invalid because there is no remaining path concept
- `−½σ` is intentionally omitted because it adds little beyond `F` and `−1σ`


## Expiry Blank-Cell Reasoning

### `t%→K` at expiry

Meaningless: there is no remaining path by expiry.

### `−½σ` at expiry

Intentionally omitted. The terminal downside is already adequately represented
by:

- `F`
- `−1σ`

and the upside by:

- `K`
- `K+½σ`

### `Δvol` at expiry

Invalid because vol does not affect intrinsic value at expiry.


## Context Model

Current context names are retained.

Examples:

- `classic_carry`
- `cheap_carry`
- `conservative_carry`
- `delta_carry`
- `big_move`

The internal representation changes from family weights to per-cell multipliers.


## Weighting Model

Each valid cell has a **continuous numeric multiplier**.

Rules:

- Minimum value: `0.1`
- No upper bound imposed by this spec
- Invalid cells are stored/displayed as unavailable and are not editable

Baseline:

- Universal baseline weight for all valid cells
- Context values are treated as multipliers on that baseline

Because baseline is universal, it cancels out in the final weighted-average
score. In practice the context multipliers define relative importance.


## Scoring Formula

For a given structure variant and a given context:

- Let `pnl[cell]` be the P&L result for that cell
- Let `m[cell]` be the context multiplier for that cell
- Let valid cells be the cells that exist for the tenor and grid definition

Final score:

`score = sum(m[cell] × pnl[cell]) / sum(m[cell])`

Notes:

- This is a **weighted average**, not a raw summation
- Invalid cells are excluded entirely
- If `Δvol` is present, its `pnl[cell]` is already the average of the up/down
  vol sub-scenarios before entering the weighted-average formula


## UI Requirements

The context editor should render the fixed grid.

Rules:

- Valid cells:
  - editable numeric inputs
  - minimum `0.1`
- Invalid cells:
  - show `-`
  - no editing allowed

`1w` row behaviour:

- Show only when `T > 6 weeks`
- Otherwise omit it from both generation and editing logic


## Generator Requirements

The scenario generator should move from family-first output to explicit
cell-first output.

Each emitted scenario row should carry enough metadata to identify:

- row label
- column label
- elapsed time
- remaining time
- scenario forward
- scenario spot
- scenario vol
- whether the row is a direct scenario or a `Δvol` sub-scenario

For `Δvol`, either:

- emit two internal sub-scenarios and average them in the scorer, or
- emit one logical grid cell with pre-averaged P&L after pricing

Implementation choice is open as long as the exposed scoring semantics match
this spec.


## Migration Requirements

The following changes will be required:

1. Remove family-weight-based scoring from Structure Evaluation
2. Replace family-weight context config with per-grid-cell multipliers
3. Update the context editor to render the explicit grid
4. Update scenario generation to emit the explicit grid points
5. Update scenario scoring to use weighted average across explicit cells
6. Preserve current context names and the overall context-selection logic


## Non-Goals

- No redesign of context names in this iteration
- No change to the PM preference routing logic that selects the active context
- No change to the meaning of target-based path scenarios (`t%→K`, `K`)
- No change to the sigma-based direction classification threshold logic beyond
  retaining full-tenor sigma for `_compute_direction`
