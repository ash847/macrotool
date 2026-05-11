"""
Tests for analytics.scenario_pricer.
"""

import math
import pytest

from analytics.scenario_generator import generate_scenarios
from analytics.scenario_pricer import price_scenarios
from analytics.structure_pricer import PricedVariant
from pricing.black_scholes import call_mtm


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
        "row": "TEST",
        "col": "TEST",
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
# 1x1.5 spread — expiry
# ---------------------------------------------------------------------------

class TestOneByOnePointFiveExpiry:
    def test_at_short_strike_net_long_value(self):
        K1, K2 = 1.038, 1.12
        scenario_spot = K2
        v = PricedVariant(
            variant_label="ATMF / 1.5x target",
            strikes=[K1, K2], barrier=None, net_premium_pct=0.005,
            breakeven=None, payoff_at_target_pct=None, rr_at_target=None,
            max_loss_pct=0.005, wing_ratio=None, is_zero_cost=False,
        )
        scenarios = _single_scenario(scenario_spot, remaining_time=0.0)
        rows = price_scenarios(v, "1x1.5_spread", scenarios, _TRADE_INPUTS, is_call=True)
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
# European RKO — expiry
# ---------------------------------------------------------------------------

class TestEuropeanRKOExpiry:
    def test_call_between_strike_and_barrier_behaves_like_vanilla(self):
        K = 1.08
        H = 1.12
        scenario_spot = 1.10
        v = PricedVariant(
            variant_label="ATMF / 2× target",
            strikes=[K, H], barrier=H, net_premium_pct=0.03,
            breakeven=None, payoff_at_target_pct=None, rr_at_target=None,
            max_loss_pct=0.03, wing_ratio=None, is_zero_cost=False,
        )
        rows = price_scenarios(v, "european_rko", _single_scenario(scenario_spot, remaining_time=0.0), _TRADE_INPUTS, is_call=True)
        expected = (scenario_spot - K) / _SPOT
        assert abs(rows[0]["price_pct"] - expected) < 1e-8

    def test_call_zero_beyond_barrier(self):
        K = 1.08
        H = 1.12
        scenario_spot = 1.13
        v = PricedVariant(
            variant_label="ATMF / 2× target",
            strikes=[K, H], barrier=H, net_premium_pct=0.03,
            breakeven=None, payoff_at_target_pct=None, rr_at_target=None,
            max_loss_pct=0.03, wing_ratio=None, is_zero_cost=False,
        )
        rows = price_scenarios(v, "european_rko", _single_scenario(scenario_spot, remaining_time=0.0), _TRADE_INPUTS, is_call=True)
        assert abs(rows[0]["price_pct"]) < 1e-8


class TestDeltaVolScenarios:
    def test_delta_vol_uses_worse_of_up_down_vol_shocks(self):
        v = _vanilla_variant(prem_pct=0.02)
        scenario_spot = 1.06
        remaining_time = 0.2
        scenario_fwd = scenario_spot * math.exp((_R_D - _R_F) * remaining_time)
        vol_bump = _VOL * 0.04
        scenarios = [{
            "id": "25%T|Δvol",
            "row": "25%T",
            "col": "Δvol",
            "time_fraction": 0.25,
            "fwd_rule": "Δvol",
            "vol_rule": "VOL_AVG",
            "skew_rule": "SKEW_UNCHANGED",
            "tags": ["25%T", "Δvol"],
            "vol_shifts": [-vol_bump, vol_bump],
            "derived": {
                "elapsed_time": _T - remaining_time,
                "remaining_time": remaining_time,
                "scenario_fwd": scenario_fwd,
                "scenario_spot": scenario_spot,
                "vol_shift": None,
                "scenario_vol": _VOL,
                "skew_multiplier": 1.0,
                "sigma_T": _VOL * math.sqrt(max(_T - remaining_time, 0.0)),
                "direction": 1,
            },
        }]
        rows = price_scenarios(v, "vanilla", scenarios, _TRADE_INPUTS, is_call=True)
        down = call_mtm(scenario_spot, _K_STRIKE, remaining_time, _VOL - vol_bump, _R_D, _R_F)
        up = call_mtm(scenario_spot, _K_STRIKE, remaining_time, _VOL + vol_bump, _R_D, _R_F)
        expected = min(down, up) / _SPOT
        assert abs(rows[0]["price_pct"] - expected) < 1e-6
        assert rows[0]["vol_shift"] == "±4% vol"


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
            "structure_id", "variant_label", "scenario_id", "row", "col",
            "time_fraction", "fwd_rule", "vol_rule", "skew_rule", "tags",
            "elapsed_time", "remaining_time", "scenario_fwd", "scenario_spot",
            "vol_shift", "scenario_vol", "skew_multiplier", "price_pct", "pnl_pct",
            "structure_notional", "price_ccy", "pnl_ccy",
        }
        v = _vanilla_variant()
        scenarios = generate_scenarios(_TRADE_INPUTS)
        rows = price_scenarios(v, "vanilla", scenarios, _TRADE_INPUTS, is_call=True)
        for r in rows:
            assert required.issubset(r.keys())


# ---------------------------------------------------------------------------
# Sizing — dollar amounts from loss budget
# ---------------------------------------------------------------------------

class TestSizing:
    def test_variant_sizing_basic(self):
        """structure_notional = loss_budget / max_loss_pct."""
        from analytics.structure_pricer import _size_variant
        v = _vanilla_variant(prem_pct=0.02)
        v.payoff_at_target_pct = 0.12
        _size_variant(v, loss_budget=4.0)
        assert abs(v.structure_notional - 200.0) < 1e-9   # 4.0 / 0.02
        assert abs(v.net_premium_ccy - 4.0) < 1e-9
        assert abs(v.max_loss_ccy - 4.0) < 1e-9            # = loss_budget
        assert abs(v.payoff_at_target_ccy - 24.0) < 1e-9   # 0.12 * 200

    def test_size_variant_skips_zero_max_loss(self):
        from analytics.structure_pricer import _size_variant
        v = _vanilla_variant()
        v.max_loss_pct = 0.0
        _size_variant(v, loss_budget=4.0)
        assert v.structure_notional == 500.0
        assert abs(v.net_premium_ccy - 10.0) < 1e-9
        assert abs(v.max_loss_ccy - 0.0) < 1e-9

    def test_size_variant_defaults_negative_premium_to_capped_notional(self):
        from analytics.structure_pricer import _size_variant
        v = _vanilla_variant(prem_pct=-0.003)
        v.max_loss_pct = 0.0
        _size_variant(v, loss_budget=4.0)
        assert v.structure_notional == 500.0
        assert abs(v.net_premium_ccy + 1.5) < 1e-9
        assert abs(v.max_loss_ccy - 0.0) < 1e-9

    def test_size_variant_caps_very_low_premium_notional(self):
        from analytics.structure_pricer import _size_variant
        v = _vanilla_variant(prem_pct=0.001)
        v.payoff_at_target_pct = 0.12
        _size_variant(v, loss_budget=4.0)
        assert abs(v.structure_notional - 500.0) < 1e-9
        assert abs(v.net_premium_ccy - 0.5) < 1e-9
        assert abs(v.max_loss_ccy - 0.5) < 1e-9
        assert abs(v.payoff_at_target_ccy - 60.0) < 1e-9

    def test_price_variants_passes_loss_budget(self):
        """price_variants populates dollar fields when loss_budget given."""
        from analytics.structure_pricer import price_variants
        from analytics.market_state import compute_market_state
        ms = compute_market_state(
            spot=5.0, fwd=5.05, vol=0.15, T=0.25, r_d=0.05, r_f=0.04,
            target=5.30, direction="base_higher",
        )
        pvs = price_variants(ms, "vanilla", target=5.30, is_call=True, loss_budget=4.0)
        assert pvs
        for pv in pvs:
            assert pv.structure_notional is not None
            assert pv.structure_notional > 0
            assert pv.structure_notional <= 500.0
            assert pv.max_loss_ccy <= 4.0 + 1e-6
            # Premium $ = premium_pct * notional, where notional = budget / max_loss_pct.
            # For vanilla, max_loss_pct == net_premium_pct, so premium_ccy == max_loss_ccy.
            assert abs(pv.net_premium_ccy - pv.max_loss_ccy) < 1e-6

    def test_price_variants_caps_cheap_1x2_base_leg_notional(self):
        from analytics.structure_pricer import price_variants
        from analytics.market_state import compute_market_state
        ms = compute_market_state(
            spot=5.0, fwd=5.05, vol=0.15, T=0.25, r_d=0.05, r_f=0.04,
            target=5.30, direction="base_higher",
        )
        pvs = price_variants(ms, "1x2_spread", target=5.30, is_call=True, loss_budget=4.0)
        assert pvs
        assert any(pv.structure_notional is not None and pv.structure_notional <= 500.0 for pv in pvs)

    def test_price_variants_no_budget_leaves_fields_none(self):
        from analytics.structure_pricer import price_variants
        from analytics.market_state import compute_market_state
        ms = compute_market_state(
            spot=5.0, fwd=5.05, vol=0.15, T=0.25, r_d=0.05, r_f=0.04,
            target=5.30, direction="base_higher",
        )
        pvs = price_variants(ms, "vanilla", target=5.30, is_call=True)  # no loss_budget
        for pv in pvs:
            assert pv.structure_notional is None
            assert pv.net_premium_ccy is None

    def test_scenario_rows_carry_dollar_amounts(self):
        v = _vanilla_variant(prem_pct=0.02)
        v.structure_notional = 200.0  # manually sized
        scenarios = generate_scenarios(_TRADE_INPUTS)
        rows = price_scenarios(v, "vanilla", scenarios, _TRADE_INPUTS, is_call=True)
        for r in rows:
            assert r["structure_notional"] == 200.0
            assert r["price_ccy"] is not None
            assert r["pnl_ccy"] is not None
            assert abs(r["price_ccy"] - r["price_pct"] * 200.0) < 1e-9
            assert abs(r["pnl_ccy"] - r["pnl_pct"] * 200.0) < 1e-9

    def test_scenario_rows_dollar_none_when_notional_none(self):
        v = _vanilla_variant()  # structure_notional defaults to None
        scenarios = generate_scenarios(_TRADE_INPUTS)
        rows = price_scenarios(v, "vanilla", scenarios, _TRADE_INPUTS, is_call=True)
        for r in rows:
            assert r["structure_notional"] is None
            assert r["price_ccy"] is None
            assert r["pnl_ccy"] is None
