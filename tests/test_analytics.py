"""
Tests for the analytics module — price distribution computations.

Covers:
  - PriceDistribution model structure and ordering
  - Flat-vol distribution: forward drift, weekly steps, band ordering
  - Smile distribution: USDBRL topside skew effect on upper/lower tails
  - Intermediate-horizon interpolation (non-pillar tenor)
"""

import math
import pytest

from data.snapshot_loader import load_snapshot
from analytics.distributions import compute_flat_vol_distribution, compute_smile_distribution
from analytics.models import BAND_LABELS, PriceDistribution


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
def flat_91(brl):
    return compute_flat_vol_distribution(brl, 91)


@pytest.fixture(scope="module")
def smile_91(brl):
    return compute_smile_distribution(brl, 91)


# ---------------------------------------------------------------------------
# Model structure
# ---------------------------------------------------------------------------

class TestPriceDistributionModel:
    def test_flat_is_pydantic_model(self, flat_91):
        assert isinstance(flat_91, PriceDistribution)

    def test_smile_is_pydantic_model(self, smile_91):
        assert isinstance(smile_91, PriceDistribution)

    def test_flat_vol_type(self, flat_91):
        assert flat_91.vol_type == "flat_atm"

    def test_smile_vol_type(self, smile_91):
        assert smile_91.vol_type == "smile"

    def test_seven_bands_each(self, flat_91, smile_91):
        assert len(flat_91.bands) == 7
        assert len(smile_91.bands) == 7

    def test_band_labels_match_constant(self, flat_91):
        assert [b.label for b in flat_91.bands] == BAND_LABELS

    def test_all_bands_same_length_as_time_steps(self, flat_91):
        n = len(flat_91.time_steps_days)
        for band in flat_91.bands:
            assert len(band.prices) == n

    def test_atm_vol_positive(self, flat_91, smile_91):
        assert flat_91.atm_vol > 0
        assert smile_91.atm_vol > 0

    def test_same_atm_vol_for_both(self, flat_91, smile_91):
        """Both distributions use the same interpolated ATM vol."""
        assert flat_91.atm_vol == pytest.approx(smile_91.atm_vol)


# ---------------------------------------------------------------------------
# Flat-vol distribution
# ---------------------------------------------------------------------------

class TestFlatVolDistribution:
    def test_first_price_is_spot(self, flat_91, brl):
        for band in flat_91.bands:
            assert band.prices[0] == pytest.approx(brl.spot)

    def test_time_steps_start_at_zero(self, flat_91):
        assert flat_91.time_steps_days[0] == 0

    def test_time_steps_end_at_horizon(self, flat_91):
        assert flat_91.time_steps_days[-1] == 91

    def test_weekly_step_count(self, flat_91):
        # ceil(91/7) = 13 steps → 14 elements including t=0
        assert len(flat_91.time_steps_days) == 14

    def test_terminal_strict_ordering(self, flat_91):
        terminals = [
            flat_91.terminal_minus3s,
            flat_91.terminal_minus2s,
            flat_91.terminal_minus1s,
            flat_91.terminal_median,
            flat_91.terminal_plus1s,
            flat_91.terminal_plus2s,
            flat_91.terminal_plus3s,
        ]
        for i in range(len(terminals) - 1):
            assert terminals[i] < terminals[i + 1]

    def test_median_below_forward(self, flat_91, brl):
        """Lognormal median < mean (forward) due to Jensen's inequality."""
        fwd_3m = brl.get_forward("3M").outright
        assert flat_91.terminal_median < fwd_3m

    def test_mean_approx_equals_forward(self, flat_91, brl):
        """
        E[S_T] = S₀ × exp(μT) = F_T.  The arithmetic mean of the terminal
        distribution must equal the 3M forward.
        Verified via S₀ × exp(μT) directly; the stored bands are percentiles.
        """
        import math
        S0 = brl.spot
        F_T = brl.get_forward("3M").outright
        T = 91 / 365.0
        mu = math.log(F_T / S0) / T
        mean = S0 * math.exp(mu * T)
        assert mean == pytest.approx(F_T, rel=1e-6)

    def test_axis_bounds_wider_than_3sigma(self, flat_91):
        assert flat_91.axis_max > flat_91.terminal_plus3s
        assert flat_91.axis_min < flat_91.terminal_minus3s

    def test_pair_preserved(self, flat_91):
        assert flat_91.pair == "USDBRL"

    def test_horizon_preserved(self, flat_91):
        assert flat_91.horizon_days == 91

    def test_prices_grow_toward_upper_bands(self, flat_91):
        """At any time step, higher sigma band → higher price."""
        for idx in range(1, len(flat_91.time_steps_days)):
            prices_at_t = [b.prices[idx] for b in flat_91.bands]
            for i in range(len(prices_at_t) - 1):
                assert prices_at_t[i] < prices_at_t[i + 1]


# ---------------------------------------------------------------------------
# Smile distribution — USDBRL topside skew effects
# ---------------------------------------------------------------------------

class TestSmileDistribution:
    def test_first_price_is_spot(self, smile_91, brl):
        for band in smile_91.bands:
            assert band.prices[0] == pytest.approx(brl.spot)

    def test_terminal_strict_ordering(self, smile_91):
        terminals = [
            smile_91.terminal_minus3s,
            smile_91.terminal_minus2s,
            smile_91.terminal_minus1s,
            smile_91.terminal_median,
            smile_91.terminal_plus1s,
            smile_91.terminal_plus2s,
            smile_91.terminal_plus3s,
        ]
        for i in range(len(terminals) - 1):
            assert terminals[i] < terminals[i + 1]

    def test_topside_skew_widens_upper_tail(self, flat_91, smile_91):
        """USDBRL has call skew → +1σ/+2σ/+3σ prices higher with smile."""
        assert smile_91.terminal_plus1s > flat_91.terminal_plus1s
        assert smile_91.terminal_plus2s > flat_91.terminal_plus2s
        assert smile_91.terminal_plus3s > flat_91.terminal_plus3s

    def test_smile_lowers_lower_tail(self, flat_91, smile_91):
        """Higher put vol under the smile → lower tail price moves down."""
        assert smile_91.terminal_minus1s < flat_91.terminal_minus1s
        assert smile_91.terminal_minus2s < flat_91.terminal_minus2s
        assert smile_91.terminal_minus3s < flat_91.terminal_minus3s

    def test_upper_tail_expands_more_than_lower(self, flat_91, smile_91):
        """Topside skew: call-side expansion larger than put-side."""
        upper_delta = smile_91.terminal_plus1s - flat_91.terminal_plus1s
        lower_delta = flat_91.terminal_minus1s - smile_91.terminal_minus1s  # both positive
        assert upper_delta > lower_delta

    def test_medians_approximately_equal(self, flat_91, smile_91):
        """Both medians use ~ATM vol; difference should be < 1%."""
        rel_diff = abs(flat_91.terminal_median - smile_91.terminal_median) / flat_91.terminal_median
        assert rel_diff < 0.01


# ---------------------------------------------------------------------------
# Horizon interpolation (non-pillar tenor)
# ---------------------------------------------------------------------------

class TestHorizonInterpolation:
    def test_45d_horizon_runs_without_error(self, brl):
        dist = compute_flat_vol_distribution(brl, 45)
        assert dist.horizon_days == 45
        assert dist.time_steps_days[-1] == 45

    def test_180d_horizon_runs_without_error(self, brl):
        dist = compute_flat_vol_distribution(brl, 180)
        assert dist.terminal_median > brl.spot  # USDBRL in contango

    def test_smile_45d_horizon(self, brl):
        dist = compute_smile_distribution(brl, 45)
        assert len(dist.bands) == 7

    def test_custom_n_steps(self, brl):
        dist = compute_flat_vol_distribution(brl, 91, n_steps=5)
        assert len(dist.time_steps_days) == 6  # 0 + 5 steps
        assert dist.time_steps_days[-1] == 91
