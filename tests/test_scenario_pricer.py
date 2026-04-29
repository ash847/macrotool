"""
Tests for analytics.scenario_pricer.
"""

import math
import pytest

from analytics.scenario_generator import generate_scenarios
from analytics.scenario_pricer import price_scenarios
from analytics.structure_pricer import PricedVariant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPOT = 1.035
_FWD = 1.038
_K_STRIKE = 1.038   # ATMF vanilla
_TARGET = 1.12
_T = 0.333
_VOL = 0.09
_R_D = 0.05
_R_F = 0.03
_ENTRY_PREM_PCT = 0.02  # 2% entry premium for vanilla

_TRADE_INPUTS = {
    "spot": _SPOT,
    "forward": _FWD,
    "target": _TARGET,
    "tenor_years": _T,
    "implied_vol": _VOL,
    "r_d": _R_D,
    "r_f": _R_F,
}


def _vanilla_variant(prem_pct=_ENTRY_PREM_PCT) -> PricedVariant:
    return PricedVariant(
        variant_label="ATMF",
        strikes=[_K_STRIKE],
        barrier=None,
        net_premium_pct=prem_pct,
        breakeven=_K_STRIKE + prem_pct * _SPOT,
        payoff_at_target_pct=(_TARGET - _K_STRIKE) / _SPOT,
        rr_at_target=((_TARGET - _K_STRIKE) / _SPOT) / prem_pct,
        max_loss_pct=prem_pct,
        wing_ratio=None,
        is_zero_cost=False,
    )


def _single_scenario(scenario_spot, remaining_time, scenario_vol=_VOL, scenario_fwd=None) -> dict:
    if scenario_fwd is None:
        scenario_fwd = scenario_spot * math.exp((_R_D - _R_F) * remaining_time)
    return [{
        "id": "TEST",
        "family": "TEST",
        "time_fraction": 1.0 - remaining_time / _T,
        "fwd_rule": "FORWARD",
        "vol_rule": "VOL_FLAT",
        "skew_rule": "SKEW_UNCHANGED",
        "tags": [],
        "derived": {
            "elapsed_time": _T - remaining_time,
            "remaining_time": remaining_time,
            "scenario_fwd": scenario_fwd,
            "scenario_spot": scenario_spot,
            "vol_shift": 0.0,
            "scenario_vol": scenario_vol,
            "skew_multiplier": 1.0,
            "sigma_T": _VOL * math.sqrt(_T),
            "direction": 1,
        },
    }]


# ---------------------------------------------------------------------------
# Vanilla call — expiry
# ---------------------------------------------------------------------------

class TestVanillaCallExpiry:
    def test_itm_returns_intrinsic(self):
        # At expiry (tau=0), call value = max(spot - K, 0) / entry_spot
        scenario_spot = 1.10   # above K_STRIKE=1.038
        scenarios = _single_scenario(scenario_spot, remaining_time=0.0)
        v = _vanilla_variant()
        rows = price_scenarios(v, "vanilla", scenarios, _TRADE_INPUTS, is_call=True)
        expected_price_pct = (scenario_spot - _K_STRIKE) / _SPOT
        assert abs(rows[0]["price_pct"] - expected_price_pct) < 1e-6

    def test_otm_returns_zero(self):
        scenario_spot = 1.00   # below K_STRIKE=1.038
        scenarios = _single_scenario(scenario_spot, remaining_time=0.0)
        v = _vanilla_variant()
        rows = price_scenarios(v, "vanilla", scenarios, _TRADE_INPUTS, is_call=True)
        assert abs(rows[0]["price_pct"]) < 1e-8

    def test_pnl_equals_price_minus_entry_premium(self):
        scenario_spot = 1.10
        scenarios = _single_scenario(scenario_spot, remaining_time=0.0)
        v = _vanilla_variant(prem_pct=0.02)
        rows = price_scenarios(v, "vanilla", scenarios, _TRADE_INPUTS, is_call=True)
        assert abs(rows[0]["pnl_pct"] - (rows[0]["price_pct"] - 0.02)) < 1e-10


class TestVanillaPutExpiry:
    def test_itm_put_returns_intrinsic(self):
        K = 1.038
        scenario_spot = 0.98   # below K
        v = PricedVariant(
            variant_label="ATMF put",
            strikes=[K],
            barrier=None,
            net_premium_pct=0.02,
            breakeven=K - 0.02,
            payoff_at_target_pct=None,
            rr_at_target=None,
            max_loss_pct=0.02,
            wing_ratio=None,
            is_zero_cost=False,
        )
        inputs = {**_TRADE_INPUTS, "target": 0.95}
        scenarios = _single_scenario(scenario_spot, remaining_time=0.0)
        rows = price_scenarios(v, "vanilla", scenarios, inputs, is_call=False)
        expected_price_pct = (K - scenario_spot) / _SPOT
        assert abs(rows[0]["price_pct"] - expected_price_pct) < 1e-6

    def test_otm_put_returns_zero(self):
        K = 1.038
        scenario_spot = 1.10   # above K → put OTM
        v = PricedVariant(
            variant_label="ATMF put",
            strikes=[K], barrier=None, net_premium_pct=0.02,
            breakeven=K - 0.02, payoff_at_target_pct=None, rr_at_target=None,
            max_loss_pct=0.02, wing_ratio=None, is_zero_cost=False,
        )
        inputs = {**_TRADE_INPUTS, "target": 0.95}
        scenarios = _single_scenario(scenario_spot, remaining_time=0.0)
        rows = price_scenarios(v, "vanilla", scenarios, inputs, is_call=False)
        assert abs(rows[0]["price_pct"]) < 1e-8


# ---------------------------------------------------------------------------
# 1x1 spread — expiry
# ---------------------------------------------------------------------------

class TestSpreadExpiry:
    def test_itm_capped_payoff(self):
        K_long, K_short = 1.038, 1.08
        scenario_spot = 1.09   # above K_short → capped
        v = PricedVariant(
            variant_label="ATMF / 25Δ",
            strikes=[K_long, K_short], barrier=None, net_premium_pct=0.01,
            breakeven=K_long + 0.01, payoff_at_target_pct=None, rr_at_target=None,
            max_loss_pct=0.01, wing_ratio=None, is_zero_cost=False,
        )
        scenarios = _single_scenario(scenario_spot, remaining_time=0.0)
        rows = price_scenarios(v, "1x1_spread", scenarios, _TRADE_INPUTS, is_call=True)
        # At expiry, both legs are intrinsic; net = (K_short - K_long) / spot
        expected = (K_short - K_long) / _SPOT
        assert abs(rows[0]["price_pct"] - expected) < 1e-6

    def test_between_strikes(self):
        K_long, K_short = 1.038, 1.08
        scenario_spot = 1.06   # between strikes
        v = PricedVariant(
            variant_label="test",
            strikes=[K_long, K_short], barrier=None, net_premium_pct=0.01,
            breakeven=None, payoff_at_target_pct=None, rr_at_target=None,
            max_loss_pct=0.01, wing_ratio=None, is_zero_cost=False,
        )
        scenarios = _single_scenario(scenario_spot, remaining_time=0.0)
        rows = price_scenarios(v, "1x1_spread", scenarios, _TRADE_INPUTS, is_call=True)
        expected = (scenario_spot - K_long) / _SPOT
        assert abs(rows[0]["price_pct"] - expected) < 1e-6


# ---------------------------------------------------------------------------
# 1x2 spread — expiry
# ---------------------------------------------------------------------------

class TestOneByTwoExpiry:
    def test_at_short_strike_net_long_value(self):
        K1, K2 = 1.038, 1.12
        scenario_spot = K2   # at both short strikes
        v = PricedVariant(
            variant_label="ATMF / 2× target",
            strikes=[K1, K2], barrier=None, net_premium_pct=0.005,
            breakeven=None, payoff_at_target_pct=None, rr_at_target=None,
            max_loss_pct=0.005, wing_ratio=None, is_zero_cost=False,
        )
        scenarios = _single_scenario(scenario_spot, remaining_time=0.0)
        rows = price_scenarios(v, "1x2_spread", scenarios, _TRADE_INPUTS, is_call=True)
        # long K1 call: scenario_spot - K1; short 2× K2 call: 0 (at K2)
        expected = (scenario_spot - K1) / _SPOT
        assert abs(rows[0]["price_pct"] - expected) < 1e-6


# ---------------------------------------------------------------------------
# Seagull — expiry
# ---------------------------------------------------------------------------

class TestSeagullExpiry:
    def test_above_short_call_strike(self):
        K1, K2, K3 = 1.038, 1.08, 1.00  # long call, short call, short put
        scenario_spot = 1.09   # above K2 → spread at max, wing OTM
        v = PricedVariant(
            variant_label="ATMF / 25Δ spread + 25Δ put",
            strikes=[K1, K2, K3], barrier=None, net_premium_pct=0.0,
            breakeven=None, payoff_at_target_pct=None, rr_at_target=None,
            max_loss_pct=0.0, wing_ratio=0.5, is_zero_cost=True,
        )
        scenarios = _single_scenario(scenario_spot, remaining_time=0.0)
        rows = price_scenarios(v, "seagull", scenarios, _TRADE_INPUTS, is_call=True)
        # spread: K2 - K1; wing put OTM: 0
        expected = (K2 - K1) / _SPOT
        assert abs(rows[0]["price_pct"] - expected) < 1e-6

    def test_below_put_wing_strike(self):
        K1, K2, K3 = 1.038, 1.08, 1.00
        scenario_spot = 0.98   # below K3 → call spread worthless, put wing ITM
        v = PricedVariant(
            variant_label="test",
            strikes=[K1, K2, K3], barrier=None, net_premium_pct=0.0,
            breakeven=None, payoff_at_target_pct=None, rr_at_target=None,
            max_loss_pct=0.0, wing_ratio=0.5, is_zero_cost=True,
        )
        scenarios = _single_scenario(scenario_spot, remaining_time=0.0)
        rows = price_scenarios(v, "seagull", scenarios, _TRADE_INPUTS, is_call=True)
        # spread: 0; wing: -(0.5 * (K3 - scenario_spot) / spot)
        expected = -(0.5 * (K3 - scenario_spot)) / _SPOT
        assert abs(rows[0]["price_pct"] - expected) < 1e-6


# ---------------------------------------------------------------------------
# Digital — expiry
# ---------------------------------------------------------------------------

class TestDigitalExpiry:
    def test_itm_digital_call_returns_one(self):
        K = 1.08
        scenario_spot = 1.09   # above K
        v = PricedVariant(
            variant_label="~20% prem",
            strikes=[K], barrier=None, net_premium_pct=0.20,
            breakeven=K, payoff_at_target_pct=1.0, rr_at_target=5.0,
            max_loss_pct=0.20, wing_ratio=None, is_zero_cost=False,
        )
        scenarios = _single_scenario(scenario_spot, remaining_time=0.0)
        rows = price_scenarios(v, "european_digital", scenarios, _TRADE_INPUTS, is_call=True)
        # payout = entry_spot; price = entry_spot; price_pct = 1.0
        assert abs(rows[0]["price_pct"] - 1.0) < 1e-8

    def test_otm_digital_call_returns_zero(self):
        K = 1.12
        scenario_spot = 1.10   # below K → OTM
        v = PricedVariant(
            variant_label="~10% prem",
            strikes=[K], barrier=None, net_premium_pct=0.10,
            breakeven=K, payoff_at_target_pct=1.0, rr_at_target=10.0,
            max_loss_pct=0.10, wing_ratio=None, is_zero_cost=False,
        )
        scenarios = _single_scenario(scenario_spot, remaining_time=0.0)
        rows = price_scenarios(v, "european_digital", scenarios, _TRADE_INPUTS, is_call=True)
        assert abs(rows[0]["price_pct"]) < 1e-8


# ---------------------------------------------------------------------------
# Integration: generate_scenarios then price
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_directional_pipeline(self):
        v = _vanilla_variant()
        scenarios = generate_scenarios(_TRADE_INPUTS)
        rows = price_scenarios(v, "vanilla", scenarios, _TRADE_INPUTS, is_call=True)
        assert len(rows) == len(scenarios)
        for r in rows:
            assert "price_pct" in r
            assert "pnl_pct" in r
            assert abs(r["pnl_pct"] - (r["price_pct"] - v.net_premium_pct)) < 1e-10

    def test_expiry_scenarios_price_as_intrinsic(self):
        v = _vanilla_variant()
        scenarios = generate_scenarios(_TRADE_INPUTS)
        expiry = [s for s in scenarios if s["derived"]["remaining_time"] == 0.0]
        rows = price_scenarios(v, "vanilla", expiry, _TRADE_INPUTS, is_call=True)
        for r in rows:
            sspot = r["scenario_spot"]
            expected = max(sspot - _K_STRIKE, 0.0) / _SPOT
            assert abs(r["price_pct"] - expected) < 1e-6, r["scenario_id"]

    def test_output_row_has_required_keys(self):
        required = {
            "structure_id", "variant_label", "scenario_id", "family",
            "time_fraction", "fwd_rule", "vol_rule", "skew_rule", "tags",
            "elapsed_time", "remaining_time", "scenario_fwd", "scenario_spot",
            "vol_shift", "scenario_vol", "skew_multiplier", "price_pct", "pnl_pct",
        }
        v = _vanilla_variant()
        scenarios = generate_scenarios(_TRADE_INPUTS)
        rows = price_scenarios(v, "vanilla", scenarios, _TRADE_INPUTS, is_call=True)
        for r in rows:
            assert required.issubset(r.keys())
