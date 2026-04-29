# Scenario Generation & Structure Evaluation Spec

## Objective

Add a deterministic scenario module that, given a live trade view, produces:

1. A canonical pack of scenario JSON objects (forward, vol, skew shifts at fixed time fractions).
2. A repriced value of a chosen structure variant under each scenario (price + P&L vs entry premium).

The scenario module itself is pure computation — no LLM, no ranking. It is wired into the existing Streamlit Trade View output and (in a follow-up step) into a new "Structure Evaluation" page.

---

## Trade Inputs (already exist upstream)

- currency_pair
- spot
- forward (F)
- implied_vol (base_vol)
- tenor_years (T)
- target (K)
- r_d, r_f (from existing rate_context)

Strike, tenor, and entry premium are **fixed at trade entry** and do not change across scenarios.

---

## Pricing Invariants Across Scenarios

Across every scenario:

- **Vol surface:** invariant. We apply only an absolute vol shift (`vol_shift`) on top of base_vol.
- **Skew:** invariant for now (`skew_rule` is recorded in JSON but ignored by the pricer).
- **Discount factors / rates:** invariant. r_d, r_f are unchanged; DF(τ) = exp(−r_d × τ).
- **Strike, structure, expiry:** fixed at trade entry.

What moves across scenarios:

- **Forward** (per the scenario's spot_rule, applied to F).
- **Spot** is *back-derived* from the new forward: `scenario_spot = new_fwd × exp(−(r_d − r_f) × remaining_time)`.
- **Time:** `remaining_time = T × (1 − time_fraction)`.
- **Vol:** `scenario_vol = max(base_vol + vol_shift, 0.01)`.

At expiry (time_fraction = 1.0): `remaining_time = 0`, so `scenario_spot = new_fwd` and the option value is intrinsic only — no pricing needed.

---

## Derived Values (per trade)

- F = forward
- K = target
- T = tenor_years
- sigma_T = base_vol × sqrt(T)
- direction = sign(log(K / F))
- direction = 0 if `abs(log(K / F)) < 0.05 × sigma_T` (neutral threshold)

If direction = 0, generate the neutral scenario pack instead of the directional pack.

---

## Scenario JSON Shape

```json
{
  "id": "CORRECT_PATH_25",
  "family": "CORRECT_PATH",
  "time_fraction": 0.25,
  "fwd_rule": "PARTIAL_TO_TARGET_25",
  "vol_rule": "VOL_FLAT",
  "skew_rule": "SKEW_UNCHANGED",
  "tags": ["path_pnl", "correct_direction"],
  "derived": {
    "elapsed_time": 0.0625,
    "remaining_time": 0.1875,
    "scenario_fwd": 1.0409,
    "scenario_spot": 1.0395,
    "vol_shift": 0.0,
    "scenario_vol": 0.09,
    "skew_multiplier": 1.0,
    "sigma_T": 0.045,
    "direction": 1
  }
}
```

Notes:
- No `name` field — `id` is self-descriptive.
- `fwd_rule` (formerly `spot_rule`) — all rules now operate on the forward.
- `derived` contains everything needed to reprice: scenario_fwd, scenario_vol, remaining_time. scenario_spot is included for display only.

---

## Enumerations

### Families
- CORRECT_PATH
- EARLY_TARGET
- NO_MOVE
- WRONG_WAY
- OVERSHOOT
- VOL_SENSITIVITY
- SKEW_SENSITIVITY
- EXPIRY  *(time_fraction = 1.0; option value is intrinsic, no pricing required)*

### Time fractions
- 0.25, 0.50, 0.75, 1.00

### Forward rules (directional)
- FORWARD, TARGET
- PARTIAL_TO_TARGET_25, PARTIAL_TO_TARGET_50, PARTIAL_TO_TARGET_75
- ADVERSE_0_5SIGMA, ADVERSE_1SIGMA
- OVERSHOOT_0_5SIGMA, OVERSHOOT_1SIGMA

### Forward rules (neutral, used when direction = 0)
- NEUTRAL_FORWARD
- NEUTRAL_DOWN_0_5SIGMA, NEUTRAL_DOWN_1SIGMA
- NEUTRAL_UP_0_5SIGMA, NEUTRAL_UP_1SIGMA

### Vol rules
- VOL_FLAT (vol_shift = 0)
- VOL_DOWN (vol_shift = −0.01 absolute)
- VOL_UP (vol_shift = +0.01 absolute)
- Floor: `scenario_vol = max(base_vol + vol_shift, 0.01)`

### Skew rules (recorded only, downstream ignores for now)
- SKEW_UNCHANGED (multiplier = 1.0)
- SKEW_FLATTER (multiplier = 0.75)
- SKEW_STEEPER (multiplier = 1.25)

---

## Forward Rule Formulas

Let F = forward, K = target, sigma_T = base_vol × sqrt(T), direction = sign(log(K/F)).

| Rule | Formula |
|---|---|
| FORWARD | `new_fwd = F` |
| TARGET | `new_fwd = K` |
| PARTIAL_TO_TARGET_25 | `new_fwd = exp(0.75·ln F + 0.25·ln K)` |
| PARTIAL_TO_TARGET_50 | `new_fwd = exp(0.50·ln F + 0.50·ln K)` |
| PARTIAL_TO_TARGET_75 | `new_fwd = exp(0.25·ln F + 0.75·ln K)` |
| ADVERSE_0_5SIGMA | `new_fwd = F × exp(−direction × 0.5 × sigma_T)` |
| ADVERSE_1SIGMA | `new_fwd = F × exp(−direction × 1.0 × sigma_T)` |
| OVERSHOOT_0_5SIGMA | `new_fwd = K × exp(direction × 0.5 × sigma_T)` |
| OVERSHOOT_1SIGMA | `new_fwd = K × exp(direction × 1.0 × sigma_T)` |
| NEUTRAL_FORWARD | `new_fwd = F` |
| NEUTRAL_DOWN_0_5SIGMA | `new_fwd = F × exp(−0.5 × sigma_T)` |
| NEUTRAL_DOWN_1SIGMA | `new_fwd = F × exp(−1.0 × sigma_T)` |
| NEUTRAL_UP_0_5SIGMA | `new_fwd = F × exp(+0.5 × sigma_T)` |
| NEUTRAL_UP_1SIGMA | `new_fwd = F × exp(+1.0 × sigma_T)` |

Then in every case:
```
scenario_spot = new_fwd × exp(−(r_d − r_f) × remaining_time)
```

`sigma_T` in the offset formulas always uses **base_vol** (vol_shift only affects pricing, not anchor placement).

---

## Standard Scenario Packs

### Directional Pack (~21 scenarios)

**Correct Path** (4): `CORRECT_PATH_25/50/75` use `PARTIAL_TO_TARGET_*`; `CORRECT_PATH_100` uses `TARGET` (family = EXPIRY internally — terminal at K).

**Early Target** (3): `EARLY_TARGET_25/50/75` — fwd_rule = TARGET at each pre-expiry time fraction.

**No Move** (3): `NO_MOVE_25/50/75` — fwd_rule = FORWARD, theta bleed.

**Wrong Way** (2): `WRONG_SMALL_50` (ADVERSE_0_5SIGMA), `WRONG_LARGE_50` (ADVERSE_1SIGMA), at time_fraction = 0.50.

**Overshoot** (2): `OVERSHOOT_50` (mid-life), `OVERSHOOT_100` (terminal/EXPIRY family).

**Vol Sensitivity** (4): VOL_DOWN/UP × {PARTIAL_TO_TARGET_50, TARGET}, all at time_fraction = 0.50.

**Skew Sensitivity** (2): SKEW_FLATTER and SKEW_STEEPER at TARGET, time_fraction = 0.50.

**Expiry pack** (additional terminals beyond CORRECT_PATH_100 / OVERSHOOT_100):
- `EXPIRY_FORWARD` (no move at expiry)
- `EXPIRY_ADVERSE_1SIGMA` (large miss at expiry)

### Neutral Pack (when direction = 0, ~9 scenarios)
- NEUTRAL_DOWN_1SIGMA_50, NEUTRAL_DOWN_0_5SIGMA_50
- NEUTRAL_FORWARD_25, NEUTRAL_FORWARD_50, NEUTRAL_FORWARD_75
- NEUTRAL_UP_0_5SIGMA_50, NEUTRAL_UP_1SIGMA_50
- VOL_DOWN_FORWARD_50, VOL_UP_FORWARD_50

---

## Module API

```python
# analytics/scenario_generator.py
def generate_scenarios(trade_inputs: dict) -> list[dict]
def get_enumerations() -> dict
```

```python
# analytics/scenario_pricer.py
def price_scenarios(
    variant: PricedVariant,            # the chosen variant priced at entry
    structure_id: str,                 # e.g. "vanilla"
    scenarios: list[dict],             # output of generate_scenarios
    trade_inputs: dict,
    is_call: bool,
) -> list[dict]                        # one row per scenario with price_pct + pnl_pct
```

The pricer reuses the existing Black-76 routines (and digital/RKO routines later). At expiry it returns intrinsic value directly. Skew rule is read but ignored.

---

## Streamlit Integration

### Trade View (existing page)
After the **Feedback** section, add a new section **Structure Evaluation**:
- Take the **first variant** of the **first ranked structure** from `flow.selector_result.shortlist[0]`.
- Generate the scenario pack from the live trade inputs.
- Reprice that one variant in every scenario.
- Display a table with columns:
  - `id`, `family`, `time_fraction`, `fwd_rule`, `vol_rule`, `skew_rule`
  - `remaining_time`, `scenario_fwd`, `scenario_spot`
  - `vol_shift`, `scenario_vol`
  - `price_pct` (option price as % of spot in that scenario)
  - `pnl_pct` (price_pct − entry_premium_pct)

### Page rename
Rename the existing **"Decision Parameters"** page to **"Structure Selection"**.

### New page: "Structure Evaluation"
Deferred to a follow-up iteration. Will read the latest Trade View result from `st.session_state` and show the full scenario JSON for inspection. Out of scope for this build.

---

## Test Case (live, no separate fixture)

We test by running the existing Trade View flow end-to-end. No standalone test trade — the integration hangs off whatever the user enters.

Unit tests (`tests/test_scenario_generator.py` + `tests/test_scenario_pricer.py`):
- Forward formulas at every rule (directional + neutral).
- direction detection at the neutral threshold boundary.
- `scenario_spot = new_fwd × exp(−(r_d − r_f) × τ)` round-trip.
- Vol floor at 1%.
- Expiry returns intrinsic value (no Black-76 call).
- P&L = scenario_price − entry_premium.

---

## Acceptance Criteria

1. `generate_scenarios()` returns the directional pack for non-neutral targets and the neutral pack when `|log(K/F)| < 0.05 × sigma_T`.
2. Each scenario contains template fields + `derived` block with scenario_fwd, scenario_spot, scenario_vol, remaining_time, sigma_T, direction.
3. All forward levels and times are computed from formulas, never hardcoded.
4. Vol floor enforced at 1%.
5. EXPIRY scenarios price as intrinsic only (no Black-76 call).
6. The Trade View page shows the scenario table for the first variant of the first ranked structure, with price and P&L columns.
7. "Decision Parameters" page is renamed to "Structure Selection".
8. Tests pass alongside the existing 205-test suite.
9. `pyproject.toml` version bumped (Streamlit Cloud package cache).
