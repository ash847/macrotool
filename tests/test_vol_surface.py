"""
Tests for:
  - delta_to_strike / strike_to_delta (pricing/black_scholes.py)
  - SmileInterpolator (analytics/vol_surface.py)
  - price_delta_spread / price_strike_spread (pricing/spreads.py)
"""
import math
import pytest
from data.snapshot_loader import load_snapshot
from pricing.black_scholes import delta_to_strike, strike_to_delta, black76_delta_call, black76_delta_put
from analytics.vol_surface import SmileInterpolator
from pricing.spreads import price_delta_spread, price_strike_spread
from pricing.forwards import rate_context_for_snapshot

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def snapshot():
    return load_snapshot()

@pytest.fixture(scope="module")
def brl(snapshot):
    return snapshot.get("USDBRL")

@pytest.fixture(scope="module")
def smile_brl(brl):
    return SmileInterpolator(brl)

# ---------------------------------------------------------------------------
# delta_to_strike / strike_to_delta
# ---------------------------------------------------------------------------

class TestDeltaStrikeRoundtrip:

    def test_call_roundtrip(self, brl):
        F, T, sigma = 5.897, 0.25, 0.175
        for delta in [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]:
            K = delta_to_strike(delta, F, T, sigma)
            c_delta, _ = strike_to_delta(K, F, T, sigma)
            assert abs(c_delta - delta) < 1e-10, f"call roundtrip failed at delta={delta}"

    def test_put_roundtrip(self, brl):
        F, T, sigma = 5.897, 0.25, 0.175
        for put_delta in [-0.10, -0.25, -0.40, -0.50]:
            K = delta_to_strike(put_delta, F, T, sigma)
            _, p_delta = strike_to_delta(K, F, T, sigma)
            assert abs(p_delta - put_delta) < 1e-10, f"put roundtrip failed at delta={put_delta}"

    def test_put_call_parity_strikes(self):
        # 25DC and 25DP should produce different strikes
        F, T, sigma = 5.897, 0.25, 0.175
        K_call = delta_to_strike(0.25, F, T, sigma)
        K_put  = delta_to_strike(-0.25, F, T, sigma)
        assert K_call > F, "25d call strike should be above forward"
        assert K_put  < F, "25d put strike should be below forward"

    def test_50d_call_near_forward(self):
        # 50d call strike is close to but not exactly the forward (ATM forward delta ≈ N(0.5σ√T))
        F, T, sigma = 5.897, 0.25, 0.175
        K = delta_to_strike(0.50, F, T, sigma)
        # K should be slightly below F (because N(d1)=0.5 implies d1=0, so K = F*exp(0.5σ²T) > F... wait)
        # Actually: d1=0 → K = F * exp(0.5σ²T), so K > F for 50d call
        assert K > F * 0.99 and K < F * 1.02

    def test_deeper_otm_call_has_higher_strike(self):
        F, T, sigma = 5.897, 0.25, 0.175
        K_25 = delta_to_strike(0.25, F, T, sigma)
        K_10 = delta_to_strike(0.10, F, T, sigma)
        assert K_10 > K_25, "10d call (deeper OTM) should have higher strike than 25d call"

    def test_deeper_otm_put_has_lower_strike(self):
        F, T, sigma = 5.897, 0.25, 0.175
        K_25 = delta_to_strike(-0.25, F, T, sigma)
        K_10 = delta_to_strike(-0.10, F, T, sigma)
        assert K_10 < K_25, "10d put (deeper OTM) should have lower strike than 25d put"

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError):
            delta_to_strike(0.0, 5.897, 0.25, 0.175)
        with pytest.raises(ValueError):
            delta_to_strike(1.0, 5.897, 0.25, 0.175)
        with pytest.raises(ValueError):
            delta_to_strike(-1.0, 5.897, 0.25, 0.175)

    def test_strike_to_delta_recovers_known_values(self):
        # At K=F: d1 = 0.5σ√T, so call delta = N(0.5σ√T) > 0.5
        F, T, sigma = 5.897, 0.25, 0.175
        c, p = strike_to_delta(F, F, T, sigma)
        expected_d1 = 0.5 * sigma * math.sqrt(T)
        from scipy.stats import norm
        expected_c = norm.cdf(expected_d1)
        assert abs(c - expected_c) < 1e-10
        assert abs(c + abs(p) - 1.0) < 1e-10  # call + |put| = 1 (put delta is negative)

    def test_consistent_with_existing_delta_functions(self):
        F, T, sigma = 5.897, 0.25, 0.175
        K = delta_to_strike(0.25, F, T, sigma)
        c, p = strike_to_delta(K, F, T, sigma)
        assert abs(c - black76_delta_call(F, K, T, sigma)) < 1e-12
        assert abs(p - black76_delta_put(F, K, T, sigma)) < 1e-12


# ---------------------------------------------------------------------------
# SmileInterpolator
# ---------------------------------------------------------------------------

class TestSmileInterpolator:

    def test_pillar_vols_reproduced_at_tenor(self, brl, smile_brl):
        # Vol at a pillar delta and exact tenor should match the surface exactly
        for tenor, days in [("1M", 30), ("3M", 91), ("6M", 182)]:
            for delta_label, delta_val in [("25DC", 0.25), ("ATM", 0.50), ("25DP", -0.25)]:
                expected = brl.get_vol(tenor, delta_label)  # type: ignore[arg-type]
                if expected is None:
                    continue
                got = smile_brl.vol_at_delta(delta_val, days)
                assert abs(got - expected) < 5e-4, (
                    f"Pillar vol mismatch at {tenor}/{delta_label}: "
                    f"expected {expected:.4f}, got {got:.4f}"
                )

    def test_interpolated_tenor_is_between_brackets(self, brl, smile_brl):
        # 4M is between 3M and 6M; ATM vol should be between 3M and 6M ATM vols
        vol_3m = brl.get_vol("3M", "ATM")
        vol_6m = brl.get_vol("6M", "ATM")
        vol_4m = smile_brl.vol_at_delta(0.50, horizon_days=120)
        assert min(vol_3m, vol_6m) <= vol_4m <= max(vol_3m, vol_6m)

    def test_smile_has_positive_vols(self, smile_brl):
        for delta in [-0.40, -0.25, -0.10, 0.10, 0.25, 0.40, 0.50]:
            for days in [30, 91, 120, 182]:
                v = smile_brl.vol_at_delta(delta, days)
                assert v > 0, f"Non-positive vol at delta={delta}, days={days}"

    def test_skew_direction_brl(self, brl, smile_brl):
        # USDBRL has topside skew: 25DC (USD call) more expensive than 25DP at same tenor
        for days in [30, 91, 182]:
            v_25dc = brl.get_vol("3M" if days == 91 else ("1M" if days == 30 else "6M"), "25DC")
            v_25dp = brl.get_vol("3M" if days == 91 else ("1M" if days == 30 else "6M"), "25DP")
            if v_25dc and v_25dp:
                assert v_25dc > v_25dp, f"Expected topside skew at {days}d"

    def test_vol_at_strike_atm_consistent(self, brl, smile_brl):
        # vol_at_strike at the ATM forward should be close to ATM pillar vol
        from pricing.forwards import rate_context_for_snapshot
        T = 91 / 365.0
        rate_ctx = rate_context_for_snapshot(brl, T)
        F = rate_ctx.forward
        vol_at_fwd = smile_brl.vol_at_strike(F, F, 91)
        atm_pillar = brl.get_vol("3M", "ATM")
        assert abs(vol_at_fwd - atm_pillar) < 0.005, (
            f"vol_at_strike at ATM forward should be close to ATM pillar: "
            f"{vol_at_fwd:.4f} vs {atm_pillar:.4f}"
        )

    def test_vol_at_strike_otm_call_reflects_skew(self, brl, smile_brl):
        # For USDBRL topside skew: OTM call (high strike) should have higher vol than ATM
        from pricing.forwards import rate_context_for_snapshot
        T = 91 / 365.0
        rate_ctx = rate_context_for_snapshot(brl, T)
        F = rate_ctx.forward
        K_otm_call = delta_to_strike(0.25, F, T, smile_brl.vol_at_delta(0.25, 91))
        vol_otm = smile_brl.vol_at_strike(K_otm_call, F, 91)
        vol_atm = smile_brl.vol_at_strike(F, F, 91)
        assert vol_otm > vol_atm, "OTM call should have higher vol than ATM for USDBRL topside skew"

    def test_all_pairs_build_without_error(self, snapshot):
        for pair in snapshot.currencies:
            ccy = snapshot.get(pair)
            interp = SmileInterpolator(ccy)
            v = interp.vol_at_delta(0.25, 91)
            assert v > 0


# ---------------------------------------------------------------------------
# Spread pricing
# ---------------------------------------------------------------------------

class TestDeltaSpread:

    def test_put_spread_basic(self, brl):
        result = price_delta_spread(-0.40, -0.20, brl, horizon_days=120)
        assert result.pair == "USDBRL"
        assert result.net_premium > 0, "Put spread should cost net premium"
        # Long leg is deeper ITM (higher delta abs) → higher premium
        assert result.long_leg.premium > result.short_leg.premium
        # Long put strike should be above short put strike (long is closer to spot)
        assert result.long_leg.strike > result.short_leg.strike

    def test_put_spread_strikes_below_forward(self, brl):
        result = price_delta_spread(-0.40, -0.20, brl, horizon_days=120)
        assert result.long_leg.strike < result.forward
        assert result.short_leg.strike < result.forward

    def test_call_spread_basic(self, brl):
        result = price_delta_spread(0.40, 0.20, brl, horizon_days=120)
        assert result.net_premium > 0
        # Long call (higher delta = lower strike OTM call) costs more
        assert result.long_leg.premium > result.short_leg.premium
        assert result.long_leg.strike < result.short_leg.strike

    def test_call_spread_strikes_above_forward(self, brl):
        result = price_delta_spread(0.40, 0.20, brl, horizon_days=91)
        assert result.long_leg.strike > result.forward
        assert result.short_leg.strike > result.forward

    def test_net_premium_equals_legs_difference(self, brl):
        result = price_delta_spread(-0.25, -0.10, brl, horizon_days=91)
        assert abs(result.net_premium - (result.long_leg.premium - result.short_leg.premium)) < 1e-10

    def test_premium_pct_consistent_with_absolute(self, brl):
        notional = 1_000_000.0
        result = price_delta_spread(-0.25, -0.10, brl, horizon_days=91, notional=notional)
        assert abs(result.net_premium_pct - result.net_premium / notional) < 1e-12

    def test_mixed_signs_raises(self, brl):
        with pytest.raises(ValueError):
            price_delta_spread(0.25, -0.25, brl, horizon_days=91)

    def test_delta_outputs_are_consistent_with_strikes(self, brl):
        result = price_delta_spread(-0.25, -0.10, brl, horizon_days=91)
        # Call delta + |put delta| should equal 1
        assert abs(result.long_leg.call_delta + abs(result.long_leg.put_delta) - 1.0) < 1e-10
        assert abs(result.short_leg.call_delta + abs(result.short_leg.put_delta) - 1.0) < 1e-10

    def test_specific_example_40d_20d_put_4m(self, brl):
        # The motivating example from the design discussion
        result = price_delta_spread(-0.40, -0.20, brl, horizon_days=120)
        # Sanity: strikes and deltas in right ballpark
        assert 0.35 < abs(result.long_leg.put_delta) < 0.45
        assert 0.15 < abs(result.short_leg.put_delta) < 0.25
        assert result.net_premium_pct > 0
        assert result.net_premium_pct < 0.20  # spread < 20% of notional (synthetic surface has wide vols)


class TestStrikeSpread:

    def test_put_spread_from_strikes(self, brl):
        from pricing.forwards import rate_context_for_snapshot
        T = 120 / 365.0
        ctx = rate_context_for_snapshot(brl, T)
        F = ctx.forward
        # Use strikes either side of 25d put
        K_long  = F * 0.97   # closer to spot
        K_short = F * 0.94   # further OTM
        result = price_strike_spread(K_long, K_short, brl, horizon_days=120, option_type="put")
        assert result.net_premium > 0
        assert result.long_leg.strike == K_long
        assert result.short_leg.strike == K_short

    def test_call_spread_from_strikes(self, brl):
        from pricing.forwards import rate_context_for_snapshot
        T = 91 / 365.0
        ctx = rate_context_for_snapshot(brl, T)
        F = ctx.forward
        K_long  = F * 1.02
        K_short = F * 1.05
        result = price_strike_spread(K_long, K_short, brl, horizon_days=91, option_type="call")
        assert result.net_premium > 0
        assert result.long_leg.strike < result.short_leg.strike

    def test_delta_outputs_in_valid_range(self, brl):
        from pricing.forwards import rate_context_for_snapshot
        T = 91 / 365.0
        ctx = rate_context_for_snapshot(brl, T)
        F = ctx.forward
        result = price_strike_spread(F * 0.97, F * 0.94, brl, horizon_days=91, option_type="put")
        for leg in (result.long_leg, result.short_leg):
            assert 0 < leg.call_delta < 1
            assert -1 < leg.put_delta < 0

    def test_invalid_option_type_raises(self, brl):
        with pytest.raises(ValueError):
            price_strike_spread(5.50, 5.70, brl, horizon_days=91, option_type="straddle")

    def test_strike_and_delta_entry_consistent(self, brl):
        # price_delta_spread and price_strike_spread should agree when given equivalent inputs
        long_d, short_d = -0.25, -0.10
        T = 91 / 365.0
        ctx = rate_context_for_snapshot(brl, T)
        F = ctx.forward

        delta_result = price_delta_spread(long_d, short_d, brl, horizon_days=91)
        K_long  = delta_result.long_leg.strike
        K_short = delta_result.short_leg.strike

        strike_result = price_strike_spread(K_long, K_short, brl, horizon_days=91, option_type="put")

        assert abs(delta_result.net_premium - strike_result.net_premium) / delta_result.net_premium < 0.01
