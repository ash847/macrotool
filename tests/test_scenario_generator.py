import math

from analytics.scenario_generator import (
    GRID_COLS,
    GRID_ROWS,
    ONE_WEEK_YEARS,
    SIX_WEEKS_YEARS,
    cell_id,
    generate_scenarios,
    get_enumerations,
    valid_grid_cells_for_tenor,
)

_INPUTS = {
    "spot": 1.035,
    "forward": 1.038,
    "target": 1.12,
    "tenor_years": 0.333,
    "implied_vol": 0.09,
    "r_d": 0.05,
    "r_f": 0.03,
}


class TestGridStructure:
    def test_enumerations_expose_grid(self):
        enums = get_enumerations()
        assert enums["grid_rows"] == GRID_ROWS
        assert enums["grid_cols"] == GRID_COLS

    def test_1w_row_only_when_tenor_gt_6w(self):
        short = valid_grid_cells_for_tenor(SIX_WEEKS_YEARS)
        long = valid_grid_cells_for_tenor(SIX_WEEKS_YEARS + 0.01)
        assert ("1w", "F") not in short
        assert ("1w", "F") in long

    def test_generated_ids_follow_row_col_scheme(self):
        scenarios = generate_scenarios(_INPUTS)
        ids = {s["id"] for s in scenarios}
        assert cell_id("25%T", "K") in ids
        assert cell_id("50%T", "Δvol") in ids
        assert cell_id("Expiry", "−1σ") in ids


class TestSigmaScaling:
    def test_half_time_adverse_uses_half_time_sigma(self):
        scenarios = generate_scenarios(_INPUTS)
        sc = next(s for s in scenarios if s["id"] == cell_id("50%T", "−1σ"))
        sigma_half = _INPUTS["implied_vol"] * math.sqrt(_INPUTS["tenor_years"] * 0.50)
        expected = _INPUTS["forward"] * math.exp(-sigma_half)
        assert abs(sc["derived"]["scenario_fwd"] - expected) < 1e-6

    def test_expiry_adverse_uses_full_tenor_sigma(self):
        scenarios = generate_scenarios(_INPUTS)
        sc = next(s for s in scenarios if s["id"] == cell_id("Expiry", "−1σ"))
        sigma_full = _INPUTS["implied_vol"] * math.sqrt(_INPUTS["tenor_years"])
        expected = _INPUTS["forward"] * math.exp(-sigma_full)
        assert abs(sc["derived"]["scenario_fwd"] - expected) < 1e-6

    def test_overshoot_is_defined_off_target(self):
        scenarios = generate_scenarios(_INPUTS)
        sc = next(s for s in scenarios if s["id"] == cell_id("50%T", "K+½σ"))
        sigma_half = _INPUTS["implied_vol"] * math.sqrt(_INPUTS["tenor_years"] * 0.50)
        expected = _INPUTS["target"] * math.exp(0.5 * sigma_half)
        assert abs(sc["derived"]["scenario_fwd"] - expected) < 1e-6


class TestRowSemantics:
    def test_1w_f_uses_one_week_elapsed_time(self):
        scenarios = generate_scenarios(_INPUTS)
        sc = next(s for s in scenarios if s["id"] == cell_id("1w", "F"))
        assert sc["derived"]["elapsed_time"] == round(ONE_WEEK_YEARS, 8)

    def test_progress_row_moves_partway_to_target_in_log_space(self):
        scenarios = generate_scenarios(_INPUTS)
        sc = next(s for s in scenarios if s["id"] == cell_id("25%T", "t%→K"))
        expected = math.exp(0.75 * math.log(_INPUTS["forward"]) + 0.25 * math.log(_INPUTS["target"]))
        assert abs(sc["derived"]["scenario_fwd"] - expected) < 1e-6

    def test_delta_vol_marks_average_vol_cell(self):
        scenarios = generate_scenarios(_INPUTS)
        sc = next(s for s in scenarios if s["id"] == cell_id("25%T", "Δvol"))
        assert sc["vol_shifts"] == [-0.01, 0.01]
        assert sc["vol_rule"] == "VOL_AVG"

    def test_expiry_spot_equals_fwd(self):
        scenarios = generate_scenarios(_INPUTS)
        sc = next(s for s in scenarios if s["id"] == cell_id("Expiry", "F"))
        assert abs(sc["derived"]["scenario_spot"] - sc["derived"]["scenario_fwd"]) < 1e-8
