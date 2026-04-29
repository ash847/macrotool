"""
Tests for analytics.scenario_generator.
"""

import math
import pytest

from analytics.scenario_generator import (
    generate_scenarios,
    get_enumerations,
    _apply_fwd_rule,
    _compute_direction,
    _DIRECTIONAL_PACK,
    _NEUTRAL_PACK,
)

# ---------------------------------------------------------------------------
# Shared trade inputs
# ---------------------------------------------------------------------------

_INPUTS = {
    "spot": 1.035,
    "forward": 1.038,
    "target": 1.12,
    "tenor_years": 0.333,
    "implied_vol": 0.09,
    "r_d": 0.05,
    "r_f": 0.03,
}

_F = _INPUTS["forward"]
_K = _INPUTS["target"]
_T = _INPUTS["tenor_years"]
_VOL = _INPUTS["implied_vol"]
_SIGMA_T = _VOL * math.sqrt(_T)
_DIR = 1  # K > F


# ---------------------------------------------------------------------------
# Direction detection
# ---------------------------------------------------------------------------

class TestDirectionDetection:
    def test_positive_direction(self):
        # K clearly above F
        assert _compute_direction(1.038, 1.12, 0.05) == 1

    def test_negative_direction(self):
        # K clearly below F
        assert _compute_direction(1.038, 0.95, 0.05) == -1

    def test_neutral_inside_threshold(self):
        # |log(K/F)| < 0.05 * sigma_T
        F, T, vol = 1.038, 0.25, 0.09
        sigma_T = vol * math.sqrt(T)
        # tiny move, well inside neutral band
        K = F * math.exp(0.001 * sigma_T)
        assert _compute_direction(F, K, sigma_T) == 0

    def test_just_outside_neutral_threshold(self):
        # |log(K/F)| exactly at 0.06 * sigma_T → directional
        F, T, vol = 1.038, 0.25, 0.09
        sigma_T = vol * math.sqrt(T)
        K = F * math.exp(0.06 * sigma_T)
        assert _compute_direction(F, K, sigma_T) == 1


# ---------------------------------------------------------------------------
# Forward rule formulas
# ---------------------------------------------------------------------------

class TestFwdRules:
    def test_forward_rule(self):
        assert _apply_fwd_rule("FORWARD", _F, _K, _SIGMA_T, _DIR) == _F

    def test_target_rule(self):
        assert _apply_fwd_rule("TARGET", _F, _K, _SIGMA_T, _DIR) == _K

    def test_partial_25(self):
        expected = math.exp(0.75 * math.log(_F) + 0.25 * math.log(_K))
        assert abs(_apply_fwd_rule("PARTIAL_TO_TARGET_25", _F, _K, _SIGMA_T, _DIR) - expected) < 1e-10

    def test_partial_50(self):
        expected = math.exp(0.50 * math.log(_F) + 0.50 * math.log(_K))
        assert abs(_apply_fwd_rule("PARTIAL_TO_TARGET_50", _F, _K, _SIGMA_T, _DIR) - expected) < 1e-10

    def test_partial_75(self):
        expected = math.exp(0.25 * math.log(_F) + 0.75 * math.log(_K))
        assert abs(_apply_fwd_rule("PARTIAL_TO_TARGET_75", _F, _K, _SIGMA_T, _DIR) - expected) < 1e-10

    def test_partial_ordering(self):
        p25 = _apply_fwd_rule("PARTIAL_TO_TARGET_25", _F, _K, _SIGMA_T, _DIR)
        p50 = _apply_fwd_rule("PARTIAL_TO_TARGET_50", _F, _K, _SIGMA_T, _DIR)
        p75 = _apply_fwd_rule("PARTIAL_TO_TARGET_75", _F, _K, _SIGMA_T, _DIR)
        assert _F < p25 < p50 < p75 < _K

    def test_adverse_0_5sigma_direction_positive(self):
        # direction=+1: adverse means fwd moves down
        expected = _F * math.exp(-1 * 0.5 * _SIGMA_T)
        result = _apply_fwd_rule("ADVERSE_0_5SIGMA", _F, _K, _SIGMA_T, 1)
        assert abs(result - expected) < 1e-10
        assert result < _F

    def test_adverse_1sigma_direction_negative(self):
        # direction=-1: adverse means fwd moves up
        F, K = 1.038, 0.95
        sigma_T = 0.05
        expected = F * math.exp(-(-1) * 1.0 * sigma_T)
        result = _apply_fwd_rule("ADVERSE_1SIGMA", F, K, sigma_T, -1)
        assert abs(result - expected) < 1e-10
        assert result > F

    def test_overshoot_0_5sigma(self):
        expected = _K * math.exp(_DIR * 0.5 * _SIGMA_T)
        result = _apply_fwd_rule("OVERSHOOT_0_5SIGMA", _F, _K, _SIGMA_T, _DIR)
        assert abs(result - expected) < 1e-10
        assert result > _K

    def test_neutral_forward(self):
        assert _apply_fwd_rule("NEUTRAL_FORWARD", _F, _K, _SIGMA_T, 0) == _F

    def test_neutral_down(self):
        expected = _F * math.exp(-0.5 * _SIGMA_T)
        result = _apply_fwd_rule("NEUTRAL_DOWN_0_5SIGMA", _F, _K, _SIGMA_T, 0)
        assert abs(result - expected) < 1e-10

    def test_unknown_rule_raises(self):
        with pytest.raises(ValueError):
            _apply_fwd_rule("MADE_UP", _F, _K, _SIGMA_T, _DIR)


# ---------------------------------------------------------------------------
# Scenario spot back-derivation
# ---------------------------------------------------------------------------

class TestScenarioSpot:
    def test_spot_round_trip(self):
        """scenario_spot = new_fwd * exp(-(r_d-r_f)*tau) → back to fwd correctly."""
        scenarios = generate_scenarios(_INPUTS)
        r_d = _INPUTS["r_d"]
        r_f = _INPUTS["r_f"]
        for sc in scenarios:
            d = sc["derived"]
            tau = d["remaining_time"]
            reconstructed_fwd = d["scenario_spot"] * math.exp((r_d - r_f) * tau)
            assert abs(reconstructed_fwd - d["scenario_fwd"]) < 1e-6, (
                f"{sc['id']}: fwd mismatch {reconstructed_fwd:.8f} vs {d['scenario_fwd']:.8f}"
            )

    def test_expiry_spot_equals_fwd(self):
        """At time_fraction=1.0, tau=0 so scenario_spot == scenario_fwd."""
        scenarios = generate_scenarios(_INPUTS)
        expiry = [s for s in scenarios if s["time_fraction"] == 1.0]
        assert expiry, "no expiry scenarios found"
        for sc in expiry:
            d = sc["derived"]
            assert abs(d["scenario_spot"] - d["scenario_fwd"]) < 1e-8, sc["id"]


# ---------------------------------------------------------------------------
# Vol rules
# ---------------------------------------------------------------------------

class TestVolRules:
    def test_vol_flat(self):
        scenarios = generate_scenarios(_INPUTS)
        flat = next(s for s in scenarios if s["vol_rule"] == "VOL_FLAT")
        assert flat["derived"]["vol_shift"] == 0.0
        assert abs(flat["derived"]["scenario_vol"] - _VOL) < 1e-10

    def test_vol_down(self):
        scenarios = generate_scenarios(_INPUTS)
        down = next(s for s in scenarios if s["vol_rule"] == "VOL_DOWN")
        assert abs(down["derived"]["vol_shift"] - (-0.01)) < 1e-10
        assert abs(down["derived"]["scenario_vol"] - (_VOL - 0.01)) < 1e-10

    def test_vol_up(self):
        scenarios = generate_scenarios(_INPUTS)
        up = next(s for s in scenarios if s["vol_rule"] == "VOL_UP")
        assert abs(up["derived"]["vol_shift"] - 0.01) < 1e-10
        assert abs(up["derived"]["scenario_vol"] - (_VOL + 0.01)) < 1e-10

    def test_vol_floor_at_one_percent(self):
        inputs = {**_INPUTS, "implied_vol": 0.005}  # base vol 0.5%
        scenarios = generate_scenarios(inputs)
        down = next(s for s in scenarios if s["vol_rule"] == "VOL_DOWN")
        assert down["derived"]["scenario_vol"] == 0.01


# ---------------------------------------------------------------------------
# Pack selection
# ---------------------------------------------------------------------------

class TestPackSelection:
    def test_directional_pack_selected(self):
        scenarios = generate_scenarios(_INPUTS)
        ids = {s["id"] for s in scenarios}
        assert "CORRECT_PATH_25" in ids
        assert "NEUTRAL_FORWARD_50" not in ids

    def test_neutral_pack_selected_when_direction_zero(self):
        # K very close to F → direction = 0
        inputs = {**_INPUTS, "target": _INPUTS["forward"] * 1.0001}
        scenarios = generate_scenarios(inputs)
        ids = {s["id"] for s in scenarios}
        assert "NEUTRAL_FORWARD_50" in ids
        assert "CORRECT_PATH_25" not in ids

    def test_directional_pack_count(self):
        scenarios = generate_scenarios(_INPUTS)
        assert len(scenarios) == len(_DIRECTIONAL_PACK)

    def test_neutral_pack_count(self):
        inputs = {**_INPUTS, "target": _INPUTS["forward"] * 1.0001}
        scenarios = generate_scenarios(inputs)
        assert len(scenarios) == len(_NEUTRAL_PACK)


# ---------------------------------------------------------------------------
# Derived block structure
# ---------------------------------------------------------------------------

class TestDerivedBlock:
    def test_all_derived_keys_present(self):
        required = {
            "elapsed_time", "remaining_time", "scenario_fwd", "scenario_spot",
            "vol_shift", "scenario_vol", "skew_multiplier", "sigma_T", "direction",
        }
        scenarios = generate_scenarios(_INPUTS)
        for sc in scenarios:
            assert required.issubset(sc["derived"].keys()), sc["id"]

    def test_time_fractions_sum_to_tenor(self):
        T = _INPUTS["tenor_years"]
        scenarios = generate_scenarios(_INPUTS)
        for sc in scenarios:
            d = sc["derived"]
            assert abs(d["elapsed_time"] + d["remaining_time"] - T) < 1e-6, sc["id"]

    def test_sigma_T_consistent(self):
        sigma_T = _VOL * math.sqrt(_T)
        scenarios = generate_scenarios(_INPUTS)
        for sc in scenarios:
            assert abs(sc["derived"]["sigma_T"] - sigma_T) < 1e-8

    def test_skew_multiplier_unchanged(self):
        scenarios = generate_scenarios(_INPUTS)
        unch = [s for s in scenarios if s["skew_rule"] == "SKEW_UNCHANGED"]
        for sc in unch:
            assert sc["derived"]["skew_multiplier"] == 1.0


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TestEnumerations:
    def test_get_enumerations_keys(self):
        enums = get_enumerations()
        for key in ("families", "time_fractions", "fwd_rules", "vol_rules", "skew_rules"):
            assert key in enums

    def test_expiry_in_families(self):
        assert "EXPIRY" in get_enumerations()["families"]
