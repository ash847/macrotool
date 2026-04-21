"""
Unit tests for the pricing engine.

All tests use concrete, analytically-verifiable values. Test parameters are
chosen so the results can be checked against known Black-Scholes identities
and boundary conditions.

Base test parameters (USDBRL-like):
  S = 5.785, K = 5.785 (ATM), T = 0.25 (3M), σ = 0.182, r_d = 0.043, r_f = 0.173
  (r_f derived from 3M forward 5.897 via CIP: r_f = r_d - ln(5.897/5.785)/0.25)
"""

import math
import pytest
import numpy as np

from pricing.forwards import (
    tenor_to_years,
    interpolate_forward,
    implied_r_f,
    discount_factor,
    build_rate_context,
    rate_context_for_snapshot,
)
from pricing.black_scholes import (
    black76_call,
    black76_put,
    call_value,
    put_value,
    call_spread,
    put_spread,
    risk_reversal,
)
from pricing.rko import rko_call, rko_put
from pricing.digital import digital_call, digital_put
from pricing.digital_rko import digital_rko_call, digital_rko_put
from pricing.scenario import build_scenario_matrix, ScenarioConfig, format_scenario_table
from data.snapshot_loader import load_snapshot


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def snapshot():
    return load_snapshot()


@pytest.fixture(scope="module")
def brl(snapshot):
    return snapshot.get("USDBRL")


# Standard test parameters
S   = 5.785
K   = 5.785       # ATM
T   = 0.25        # 3 months
sig = 0.182
r_d = 0.043
# r_f derived from CIP with 3M forward 5.897
r_f = r_d - math.log(5.897 / 5.785) / T   # ≈ 0.1199

F   = S * math.exp((r_d - r_f) * T)        # should be close to 5.897
DF  = math.exp(-r_d * T)


# ---------------------------------------------------------------------------
# Forwards
# ---------------------------------------------------------------------------

class TestForwards:
    def test_tenor_to_years_1m(self):
        assert tenor_to_years("1M") == pytest.approx(30 / 365)

    def test_tenor_to_years_3m(self):
        assert tenor_to_years("3M") == pytest.approx(91 / 365)

    def test_tenor_to_years_1y(self):
        assert tenor_to_years("1Y") == pytest.approx(1.0)

    def test_unknown_tenor_raises(self):
        with pytest.raises(ValueError):
            tenor_to_years("5Y")

    def test_interpolate_forward_at_tenor(self, brl):
        # At exact 3M tenor, should return the snapshot value
        T_3m = tenor_to_years("3M")
        fwd = interpolate_forward(brl, T_3m)
        assert fwd == pytest.approx(5.8970, rel=0.01)

    def test_interpolate_forward_between_tenors(self, brl):
        # Between 1M and 2M should be between their values
        T_1m = tenor_to_years("1M")
        T_2m = tenor_to_years("2M")
        T_mid = (T_1m + T_2m) / 2
        fwd_1m = interpolate_forward(brl, T_1m)
        fwd_2m = interpolate_forward(brl, T_2m)
        fwd_mid = interpolate_forward(brl, T_mid)
        assert fwd_1m < fwd_mid < fwd_2m

    def test_interpolate_forward_beyond_last_tenor_flat(self, brl):
        # Beyond 1Y should extrapolate flat from last point
        fwd_1y = interpolate_forward(brl, 1.0)
        fwd_2y = interpolate_forward(brl, 2.0)
        assert fwd_2y == pytest.approx(fwd_1y, rel=0.001)

    def test_discount_factor(self):
        df = discount_factor(0.043, 0.25)
        assert df == pytest.approx(math.exp(-0.043 * 0.25))
        assert df < 1.0

    def test_implied_r_f_round_trips(self):
        # Given spot, forward, r_d → r_f, check CIP consistency
        _r_f = implied_r_f(S, F, T, r_d)
        F_check = S * math.exp((r_d - _r_f) * T)
        assert F_check == pytest.approx(F, rel=1e-6)

    def test_build_rate_context(self, brl):
        ctx = rate_context_for_snapshot(brl, tenor_to_years("3M"))
        assert ctx.spot == pytest.approx(5.785, rel=0.01)
        assert ctx.forward == pytest.approx(5.897, rel=0.01)
        assert ctx.T == pytest.approx(tenor_to_years("3M"))
        assert ctx.discount_factor < 1.0
        # r_d = BRL (implied), r_f = USD (direct from usd_df_curve)
        # BRL rates > USD rates, so rate_differential (r_d - r_f) > 0
        assert ctx.rate_differential > 0
        assert ctx.r_f == pytest.approx(0.045, abs=0.01)   # USD rate ~4-5%
        assert ctx.r_d > 0.10   # BRL implied ~12%


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------

class TestBlackScholes:
    def test_put_call_parity(self):
        """c - p = DF * (F - K)"""
        c = black76_call(F, K, T, sig, DF)
        p = black76_put(F, K, T, sig, DF)
        assert c - p == pytest.approx(DF * (F - K), abs=1e-8)

    def test_atm_call_equals_atm_put_when_f_equals_k(self):
        """When F = K: ATM call = ATM put (put-call parity becomes c = p)"""
        c = black76_call(F, F, T, sig, DF)
        p = black76_put(F, F, T, sig, DF)
        assert c == pytest.approx(p, rel=1e-6)

    def test_deep_itm_call_approaches_intrinsic(self):
        """Very low strike: call ≈ DF * (F - K)"""
        low_K = F * 0.5
        c = black76_call(F, low_K, T, sig, DF)
        assert c == pytest.approx(DF * (F - low_K), rel=0.01)

    def test_deep_otm_call_approaches_zero(self):
        high_K = F * 2.0
        c = black76_call(F, high_K, T, sig, DF)
        assert c < 0.001 * F

    def test_call_increases_with_vol(self):
        c_low  = black76_call(F, K, T, 0.10, DF)
        c_high = black76_call(F, K, T, 0.30, DF)
        assert c_high > c_low

    def test_call_increases_with_time(self):
        c_short = black76_call(F, K, 0.1, sig, DF)
        c_long  = black76_call(F, K, 1.0, sig, math.exp(-r_d * 1.0))
        assert c_long > c_short

    def test_call_spread_cheaper_than_vanilla(self):
        high_K = K * 1.05
        vanilla_c = call_value(S, K, T, sig, r_d, r_f)
        spread_res = call_spread(S, K, high_K, T, sig, sig, r_d, r_f)
        assert spread_res.net_premium < vanilla_c

    def test_call_spread_premium_positive(self):
        """Long low strike, short high strike: always costs money."""
        spread_res = call_spread(S, K * 0.95, K * 1.05, T, sig, sig, r_d, r_f)
        assert spread_res.net_premium > 0

    def test_put_spread_premium_positive(self):
        spread_res = put_spread(S, K * 0.95, K * 1.05, T, sig, sig, r_d, r_f)
        assert spread_res.net_premium > 0

    def test_risk_reversal_bullish_net_premium(self):
        """In a topside-skewed market (sigma_call > sigma_put), bullish RR costs money."""
        sigma_call = sig * 1.10   # topside skew: call vol > put vol
        sigma_put  = sig * 0.95
        rr = risk_reversal(S, K * 1.05, K * 0.95, T, sigma_call, sigma_put, r_d, r_f,
                           direction="bullish")
        assert rr.net_premium > 0   # buying the expensive call side

    def test_risk_reversal_legs_positive(self):
        """Both legs should have positive individual value."""
        rr = risk_reversal(S, K * 1.05, K * 0.95, T, sig, sig, r_d, r_f)
        assert rr.call_leg.premium > 0
        assert rr.put_leg.premium > 0

    def test_call_value_at_zero_time_is_intrinsic(self):
        """At T=0 (approximately), call value ≈ max(F-K, 0) * DF."""
        itm_call = call_value(S, K * 0.9, 1e-6, sig, r_d, r_f)
        expected = max(S * math.exp((r_d - r_f) * 1e-6) - K * 0.9, 0) * math.exp(-r_d * 1e-6)
        assert itm_call == pytest.approx(expected, rel=0.01)


# ---------------------------------------------------------------------------
# RKO
# ---------------------------------------------------------------------------

class TestRKO:
    def test_rko_call_less_than_vanilla(self):
        """RKO call must be cheaper than vanilla call."""
        vanilla = call_value(S, K, T, sig, r_d, r_f)
        rko = rko_call(S, K, barrier=K * 1.15, T=T, sigma=sig, r_d=r_d, r_f=r_f)
        assert rko < vanilla

    def test_rko_call_approaches_zero_near_barrier(self):
        """As spot approaches barrier from below, RKO call → 0."""
        barrier = K * 1.10
        spot_near_barrier = barrier * 0.999
        rko = rko_call(spot_near_barrier, K * 0.98, barrier, T, sig, r_d, r_f)
        assert rko < 0.001 * K

    def test_rko_call_zero_when_spot_at_barrier(self):
        barrier = K * 1.10
        rko = rko_call(barrier, K, barrier, T, sig, r_d, r_f)
        assert rko == 0.0

    def test_rko_call_zero_when_barrier_at_or_below_strike(self):
        rko = rko_call(S, K, barrier=K * 0.95, T=T, sigma=sig, r_d=r_d, r_f=r_f)
        assert rko == 0.0

    def test_rko_call_approaches_vanilla_for_distant_barrier(self):
        """Far barrier: RKO price converges toward vanilla. Use 2x barrier (not 10x)
        since the convergence is asymptotic and high-carry currencies need more distance."""
        vanilla = call_value(S, K, T, sig, r_d, r_f)
        rko_far = rko_call(S, K, barrier=K * 2.0, T=T, sigma=sig, r_d=r_d, r_f=r_f)
        rko_nearer = rko_call(S, K, barrier=K * 1.2, T=T, sigma=sig, r_d=r_d, r_f=r_f)
        # Far barrier should be closer to vanilla than near barrier
        assert abs(vanilla - rko_far) < abs(vanilla - rko_nearer)
        # And far barrier should be reasonably close to vanilla
        assert rko_far == pytest.approx(vanilla, rel=0.10)  # within 10%

    def test_rko_call_non_negative(self):
        for barrier_mult in [1.05, 1.10, 1.20, 2.0]:
            rko = rko_call(S, K, barrier=K * barrier_mult, T=T, sigma=sig, r_d=r_d, r_f=r_f)
            assert rko >= 0.0

    def test_rko_put_less_than_vanilla(self):
        vanilla = put_value(S, K, T, sig, r_d, r_f)
        rko = rko_put(S, K, barrier=K * 0.85, T=T, sigma=sig, r_d=r_d, r_f=r_f)
        assert rko < vanilla

    def test_rko_put_zero_when_spot_at_barrier(self):
        barrier = K * 0.85
        rko = rko_put(barrier, K, barrier, T, sig, r_d, r_f)
        assert rko == 0.0

    def test_rko_call_monotone_in_barrier_distance(self):
        """RKO call increases as barrier moves further away from spot."""
        barriers = [K * m for m in [1.05, 1.10, 1.20, 1.50]]
        prices = [rko_call(S, K, b, T, sig, r_d, r_f) for b in barriers]
        for i in range(len(prices) - 1):
            assert prices[i] <= prices[i + 1]


# ---------------------------------------------------------------------------
# Digital
# ---------------------------------------------------------------------------

class TestDigital:
    def test_digital_call_plus_put_equals_discount_factor(self):
        """Put-call parity for cash-or-nothing: dc + dp = DF."""
        dc = digital_call(S, K, T, sig, r_d, r_f)
        dp = digital_put(S, K, T, sig, r_d, r_f)
        assert dc + dp == pytest.approx(DF, rel=1e-6)

    def test_digital_call_between_zero_and_df(self):
        dc = digital_call(S, K, T, sig, r_d, r_f)
        assert 0 < dc < DF

    def test_digital_call_deep_itm_approaches_df(self):
        """Deep ITM digital call ≈ DF (almost certain to expire ITM)."""
        dc = digital_call(S, K * 0.3, T, sig, r_d, r_f)
        assert dc == pytest.approx(DF, rel=0.01)

    def test_digital_call_deep_otm_approaches_zero(self):
        dc = digital_call(S, K * 3.0, T, sig, r_d, r_f)
        assert dc < 0.01

    def test_digital_call_increases_with_spot(self):
        """Higher spot → higher digital call probability."""
        dc_low  = digital_call(S * 0.9, K, T, sig, r_d, r_f)
        dc_high = digital_call(S * 1.1, K, T, sig, r_d, r_f)
        assert dc_high > dc_low

    def test_digital_atm_call_approx_half_df(self):
        """ATM digital call ≈ DF/2 when F ≈ K (symmetric vol)."""
        dc = digital_call(S, F, T, sig, r_d, r_f)
        assert dc == pytest.approx(DF / 2, abs=0.05 * DF)

    def test_digital_payout_scaling(self):
        """Digital call with payout=0.05 should be 5x the payout=0.01 case."""
        dc_1 = digital_call(S, K, T, sig, r_d, r_f, payout=0.01)
        dc_5 = digital_call(S, K, T, sig, r_d, r_f, payout=0.05)
        assert dc_5 == pytest.approx(5 * dc_1, rel=1e-6)


# ---------------------------------------------------------------------------
# Digital + RKO
# ---------------------------------------------------------------------------

class TestDigitalRKO:
    def test_digital_rko_less_than_plain_digital(self):
        """Digital+RKO always cheaper than plain digital (same direction)."""
        plain = digital_call(S, K, T, sig, r_d, r_f)
        drko  = digital_rko_call(S, K, barrier=K * 1.15, T=T, sigma=sig, r_d=r_d, r_f=r_f)
        assert drko < plain

    def test_digital_rko_call_zero_at_barrier(self):
        barrier = K * 1.15
        drko = digital_rko_call(barrier, K, barrier, T, sig, r_d, r_f)
        assert drko == 0.0

    def test_digital_rko_call_zero_barrier_below_strike(self):
        drko = digital_rko_call(S, K, barrier=K * 0.9, T=T, sigma=sig, r_d=r_d, r_f=r_f)
        assert drko == 0.0

    def test_digital_rko_call_approaches_plain_digital_for_distant_barrier(self):
        """Far barrier: digital+RKO converges toward plain digital."""
        plain = digital_call(S, K, T, sig, r_d, r_f)
        drko_far    = digital_rko_call(S, K, barrier=K * 2.0, T=T, sigma=sig, r_d=r_d, r_f=r_f)
        drko_nearer = digital_rko_call(S, K, barrier=K * 1.2, T=T, sigma=sig, r_d=r_d, r_f=r_f)
        # Far barrier should be closer to plain digital than near barrier
        assert abs(plain - drko_far) < abs(plain - drko_nearer)
        assert drko_far == pytest.approx(plain, rel=0.15)

    def test_digital_rko_non_negative(self):
        for barrier_mult in [1.05, 1.10, 1.20, 2.0]:
            drko = digital_rko_call(S, K, K * barrier_mult, T, sig, r_d, r_f)
            assert drko >= 0.0

    def test_digital_rko_call_monotone_in_barrier(self):
        """Digital+RKO increases as barrier moves further away."""
        barriers = [K * m for m in [1.05, 1.10, 1.20, 1.50]]
        prices = [digital_rko_call(S, K, b, T, sig, r_d, r_f) for b in barriers]
        for i in range(len(prices) - 1):
            assert prices[i] <= prices[i + 1]

    def test_digital_rko_put_less_than_plain_digital(self):
        plain = digital_put(S, K, T, sig, r_d, r_f)
        drko  = digital_rko_put(S, K, barrier=K * 0.85, T=T, sigma=sig, r_d=r_d, r_f=r_f)
        assert drko < plain


# ---------------------------------------------------------------------------
# Scenario matrix
# ---------------------------------------------------------------------------

class TestScenarioMatrix:
    def _call_pricer(self, strike, r_d, r_f):
        def pricer(spot, T_rem, sigma):
            return call_value(spot, strike, T_rem, sigma, r_d, r_f)
        return pricer

    def test_output_shape(self):
        cfg = ScenarioConfig(
            spot_range_pct=10,
            spot_steps=5,
            vol_range_pct=25,
            vol_steps=3,
            time_horizons_years=[1/12, 2/12, 3/12],
            tenor_labels=["1M", "2M", "3M"],
        )
        pricer = self._call_pricer(K, r_d, r_f)
        initial = call_value(S, K, T, sig, r_d, r_f)
        result = build_scenario_matrix(pricer, S, sig, T, initial, cfg)
        assert result.pnl_matrix.shape == (3, 5, 3)

    def test_at_base_spot_and_vol_pnl_is_time_decay(self):
        """
        At zero spot/vol shift: P&L should be negative (time decay)
        for a long option as time passes.
        """
        cfg = ScenarioConfig(
            spot_steps=5, vol_steps=1,
            time_horizons_years=[1/12],
            tenor_labels=["1M"],
        )
        pricer = self._call_pricer(K, r_d, r_f)
        initial = call_value(S, K, T, sig, r_d, r_f)
        result = build_scenario_matrix(pricer, S, sig, T, initial, cfg)
        # Centre index for 5 spot steps = index 2, vol step 0
        centre_pnl = result.pnl_matrix[0, 2, 0]
        assert centre_pnl < 0   # theta decay: option worth less after 1M, all else equal

    def test_pnl_increases_with_spot_for_call(self):
        """For a long call, P&L increases as spot increases (positive delta)."""
        cfg = ScenarioConfig(spot_steps=5, vol_steps=1,
                             time_horizons_years=[1/12], tenor_labels=["1M"])
        pricer = self._call_pricer(K, r_d, r_f)
        initial = call_value(S, K, T, sig, r_d, r_f)
        result = build_scenario_matrix(pricer, S, sig, T, initial, cfg)
        pnl_by_spot = result.pnl_matrix[0, :, 0]
        for i in range(len(pnl_by_spot) - 1):
            assert pnl_by_spot[i] < pnl_by_spot[i + 1]

    def test_pnl_increases_with_vol_for_long_option(self):
        """Long option: P&L increases with vol (positive vega)."""
        cfg = ScenarioConfig(spot_steps=1, vol_steps=3,
                             spot_range_pct=0,
                             time_horizons_years=[1/12], tenor_labels=["1M"])
        pricer = self._call_pricer(K, r_d, r_f)
        initial = call_value(S, K, T, sig, r_d, r_f)
        result = build_scenario_matrix(pricer, S, sig, T, initial, cfg)
        pnl_by_vol = result.pnl_matrix[0, 0, :]
        for i in range(len(pnl_by_vol) - 1):
            assert pnl_by_vol[i] < pnl_by_vol[i + 1]

    def test_format_scenario_table_produces_string(self):
        cfg = ScenarioConfig(
            spot_steps=3, vol_steps=3,
            time_horizons_years=[1/12, 3/12],
            tenor_labels=["1M", "3M"],
        )
        pricer = self._call_pricer(K, r_d, r_f)
        initial = call_value(S, K, T, sig, r_d, r_f)
        result = build_scenario_matrix(pricer, S, sig, T, initial, cfg)
        table_str = format_scenario_table(result, notional_scale=1_000_000)
        assert "1M" in table_str
        assert "3M" in table_str
        assert "Spot" in table_str
        assert "Vol" in table_str

    def test_scenario_with_real_snapshot(self, snapshot):
        """End-to-end: build scenario matrix from real snapshot data."""
        brl = snapshot.get("USDBRL")
        ctx = rate_context_for_snapshot(brl, tenor_to_years("3M"))
        strike = ctx.forward  # ATM forward

        def pricer(spot, T_rem, sigma):
            return call_value(spot, strike, T_rem, sigma, ctx.r_d, ctx.r_f)

        initial = call_value(ctx.spot, strike, ctx.T, brl.get_atm_vol("3M"), ctx.r_d, ctx.r_f)
        cfg = ScenarioConfig(
            spot_steps=5, vol_steps=3,
            time_horizons_years=[1/12, 2/12, 3/12],
            tenor_labels=["1M", "2M", "3M"],
        )
        result = build_scenario_matrix(pricer, ctx.spot, brl.get_atm_vol("3M"), ctx.T, initial, cfg)
        assert result.pnl_matrix.shape == (3, 5, 3)
        # At the end horizon (3M), ATM call at expiry: ITM scenarios positive, OTM negative
        expiry_pnl = result.pnl_matrix[2, :, 1]  # base vol, all spot shifts, at 3M
        assert expiry_pnl[-1] > expiry_pnl[0]    # higher spot = better P&L for call
