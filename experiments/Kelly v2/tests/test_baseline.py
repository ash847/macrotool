import numpy as np
import pytest

from baseline import black_scholes_vanilla, synthetic_lognormal_baseline
from elicitation import Distribution
from pricing import forward_of, price_vanilla


def test_synthetic_baseline_shape():
    dist = synthetic_lognormal_baseline(forward=5.0, sigma=0.10, tenor_years=0.25)
    assert isinstance(dist, Distribution)
    assert dist.n_bins == 200
    assert dist.probs.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(dist.probs >= 0.0)
    assert np.all(np.diff(dist.bins) > 0)


def test_synthetic_baseline_forward_recovered():
    """Mean of S under the lognormal should equal the input forward."""
    F = 5.0
    dist = synthetic_lognormal_baseline(forward=F, sigma=0.10, tenor_years=0.25, n_bins=400, n_stdev=8.0)
    assert forward_of(dist) == pytest.approx(F, rel=1e-3)


def test_synthetic_baseline_rejects_bad_inputs():
    with pytest.raises(ValueError, match="forward"):
        synthetic_lognormal_baseline(forward=-1.0, sigma=0.10, tenor_years=0.25)
    with pytest.raises(ValueError, match="sigma"):
        synthetic_lognormal_baseline(forward=5.0, sigma=-0.10, tenor_years=0.25)
    with pytest.raises(ValueError, match="tenor"):
        synthetic_lognormal_baseline(forward=5.0, sigma=0.10, tenor_years=0.0)
    with pytest.raises(ValueError, match="n_bins"):
        synthetic_lognormal_baseline(forward=5.0, sigma=0.10, tenor_years=0.25, n_bins=1)


@pytest.mark.parametrize(
    "F, sigma, T, K, is_call, df",
    [
        # ATM
        (5.0, 0.10, 0.25, 5.0, True, 1.00),
        (5.0, 0.10, 0.25, 5.0, False, 1.00),
        # ITM call / OTM put
        (5.0, 0.10, 0.25, 4.5, True, 0.99),
        (5.0, 0.10, 0.25, 4.5, False, 0.99),
        # OTM call / ITM put
        (5.0, 0.10, 0.25, 5.5, True, 0.98),
        (5.0, 0.10, 0.25, 5.5, False, 0.98),
        # Longer tenor, higher vol
        (5.0, 0.20, 1.00, 5.0, True, 0.95),
        # Larger spot scale
        (100.0, 0.15, 0.50, 105.0, True, 0.97),
    ],
)
def test_discrete_pricing_matches_black_scholes(F, sigma, T, K, is_call, df):
    """Sanity check 2: synthetic baseline priced on bins matches BS closed form."""
    dist = synthetic_lognormal_baseline(forward=F, sigma=sigma, tenor_years=T, n_bins=400, n_stdev=8.0)
    discrete = price_vanilla(dist, strike=K, is_call=is_call, discount_factor=df)
    closed = black_scholes_vanilla(forward=F, strike=K, sigma=sigma, tenor_years=T, is_call=is_call, discount_factor=df)

    # Tolerance: 0.5 bp of forward. Discretisation error is dominated by the
    # ATM gamma; 400 bins over +/- 8 stdev keeps this well under 1 bp for
    # realistic FX parameters.
    bp_of_forward = 1e-4 * F
    assert discrete == pytest.approx(closed, abs=0.5 * bp_of_forward)


def test_black_scholes_put_call_parity():
    """BS prices must satisfy C - P = DF * (F - K)."""
    F, sigma, T, df = 5.0, 0.10, 0.25, 0.97
    for K in [4.0, 4.5, 5.0, 5.5, 6.0]:
        c = black_scholes_vanilla(F, K, sigma, T, is_call=True, discount_factor=df)
        p = black_scholes_vanilla(F, K, sigma, T, is_call=False, discount_factor=df)
        assert c - p == pytest.approx(df * (F - K), abs=1e-12)


def test_black_scholes_degenerate_strike():
    F, sigma, T, df = 5.0, 0.10, 0.25, 0.97
    assert black_scholes_vanilla(F, 0.0, sigma, T, is_call=True, discount_factor=df) == pytest.approx(df * F)
    assert black_scholes_vanilla(F, 0.0, sigma, T, is_call=False, discount_factor=df) == pytest.approx(0.0)


def test_black_scholes_validation():
    with pytest.raises(ValueError):
        black_scholes_vanilla(-5.0, 5.0, 0.10, 0.25, is_call=True)
    with pytest.raises(ValueError):
        black_scholes_vanilla(5.0, 5.0, -0.10, 0.25, is_call=True)
    with pytest.raises(ValueError):
        black_scholes_vanilla(5.0, 5.0, 0.10, 0.0, is_call=True)
    with pytest.raises(ValueError, match="discount_factor"):
        black_scholes_vanilla(5.0, 5.0, 0.10, 0.25, is_call=True, discount_factor=0.0)
