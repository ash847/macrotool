import numpy as np
import pytest

from interface.kelly_v2.elicitation import Distribution, elicit_from_cdf_anchors
from interface.kelly_v2.pricing import (
    base_ccy_payoff_for_trade_rec,
    call_payoff,
    expected_payoff,
    forward_of,
    price_option,
    price_vanilla,
    put_payoff,
)


def standard_dist(p_min: float = 4.0, p_max: float = 6.0) -> Distribution:
    q = np.array([0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98])
    prices = np.linspace(p_min, p_max, 7)
    return elicit_from_cdf_anchors(prices, q)


def point_mass(price: float, n_bins: int = 51) -> Distribution:
    """Discrete dist with all mass at exactly `price` (placed as the middle bin)."""
    half = 0.5
    bins = np.linspace(price - half, price + half, n_bins)
    probs = np.zeros(n_bins)
    probs[n_bins // 2] = 1.0
    return Distribution(bins=bins, probs=probs)


def test_call_payoff_basic():
    payoff = call_payoff(strike=5.0)
    prices = np.array([4.0, 5.0, 6.0, 7.0])
    np.testing.assert_allclose(payoff(prices), [0.0, 0.0, 1.0, 2.0])


def test_put_payoff_basic():
    payoff = put_payoff(strike=5.0)
    prices = np.array([4.0, 5.0, 6.0, 7.0])
    np.testing.assert_allclose(payoff(prices), [1.0, 0.0, 0.0, 0.0])


def test_expected_payoff_on_point_mass():
    dist = point_mass(price=5.7)
    assert expected_payoff(dist, call_payoff(5.0)) == pytest.approx(0.7, abs=1e-9)
    assert expected_payoff(dist, put_payoff(6.0)) == pytest.approx(0.3, abs=1e-9)


def test_expected_payoff_linearity():
    dist = standard_dist()
    a = expected_payoff(dist, call_payoff(5.0))
    b = expected_payoff(dist, lambda p: 2.0 * call_payoff(5.0)(p))
    assert b == pytest.approx(2.0 * a, abs=1e-12)


def test_discount_factor_applied():
    dist = standard_dist()
    undisc = price_option(dist, call_payoff(5.0), discount_factor=1.0)
    disc = price_option(dist, call_payoff(5.0), discount_factor=0.95)
    assert disc == pytest.approx(0.95 * undisc, abs=1e-12)


def test_discount_factor_validation():
    dist = standard_dist()
    with pytest.raises(ValueError, match="discount_factor"):
        price_option(dist, call_payoff(5.0), discount_factor=0.0)
    with pytest.raises(ValueError, match="discount_factor"):
        price_option(dist, call_payoff(5.0), discount_factor=1.1)
    with pytest.raises(ValueError, match="discount_factor"):
        price_option(dist, call_payoff(5.0), discount_factor=-0.5)


def test_call_price_non_negative():
    dist = standard_dist()
    for k in np.linspace(3.0, 8.0, 11):
        assert price_vanilla(dist, k, is_call=True) >= 0.0
        assert price_vanilla(dist, k, is_call=False) >= 0.0


def test_deep_otm_call_is_zero():
    dist = standard_dist(p_min=4.0, p_max=6.0)
    # Strike well above the support — no mass above it.
    assert price_vanilla(dist, strike=10.0, is_call=True) == pytest.approx(0.0, abs=1e-12)


def test_deep_otm_put_is_zero():
    dist = standard_dist(p_min=4.0, p_max=6.0)
    assert price_vanilla(dist, strike=0.0, is_call=False) == pytest.approx(0.0, abs=1e-12)


def test_deep_itm_call_approaches_forward_minus_strike():
    dist = standard_dist(p_min=4.0, p_max=6.0)
    fwd = forward_of(dist)
    strike = 0.0  # all mass above the strike
    df = 0.97
    expected = df * (fwd - strike)
    assert price_vanilla(dist, strike, is_call=True, discount_factor=df) == pytest.approx(expected, abs=1e-9)


def test_put_call_parity():
    """C - P = DF * (F - K) where F is the dist's expected value."""
    dist = standard_dist()
    fwd = forward_of(dist)
    df = 0.98
    for strike in [4.5, 5.0, 5.5]:
        c = price_vanilla(dist, strike, is_call=True, discount_factor=df)
        p = price_vanilla(dist, strike, is_call=False, discount_factor=df)
        assert c - p == pytest.approx(df * (fwd - strike), abs=1e-9)


def test_forward_of_point_mass():
    dist = point_mass(price=5.3)
    assert forward_of(dist) == pytest.approx(5.3, abs=1e-9)


def test_atm_call_equals_put_at_forward_strike():
    dist = standard_dist()
    fwd = forward_of(dist)
    df = 0.99
    c = price_vanilla(dist, fwd, is_call=True, discount_factor=df)
    p = price_vanilla(dist, fwd, is_call=False, discount_factor=df)
    assert c == pytest.approx(p, abs=1e-9)


def test_trade_rec_call_spread_payoff_in_base_ccy():
    payoff = base_ccy_payoff_for_trade_rec(
        "1x1_spread",
        strikes=[5.0, 6.0],
        barrier=None,
        is_call=True,
        entry_spot=5.0,
    )
    prices = np.array([4.5, 5.5, 7.0])
    np.testing.assert_allclose(
        payoff(prices),
        [0.0, 0.5 / 5.5, 1.0 / 7.0],
        atol=1e-12,
    )


def test_trade_rec_digital_payoff_scales_by_entry_spot():
    payoff = base_ccy_payoff_for_trade_rec(
        "european_digital",
        strikes=[5.5],
        barrier=None,
        is_call=True,
        entry_spot=5.0,
    )
    prices = np.array([5.4, 5.6])
    np.testing.assert_allclose(payoff(prices), [0.0, 5.0 / 5.6], atol=1e-12)
