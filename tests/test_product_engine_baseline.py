"""Independent fixed-contract invariants for the first product migration slice.

These are mathematical checks, not approval of the legacy sizing policy.
Markets and constructions are deliberately independent of production JSON.
"""

import math
from copy import deepcopy

import numpy as np
import pytest

from analytics.market_state import compute_market_state
from analytics.payoffs import base_ccy_payoff_for_trade_rec
from analytics.product_model import Right
from analytics.product_pricer import build_structure, price
from analytics.scenario_pricer import _value_variant


CASES = [
    ("vanilla", {"label": "fixed vanilla", "delta": 0.5}, [1.0]),
    ("1x1_spread", {"label": "fixed spread", "long_delta": 0.5, "short_delta": 0.25}, [1.0, -1.0]),
    ("1x1.5_spread", {"label": "fixed ratio", "long_delta": 0.5, "short_delta": 0.25}, [1.0, -1.5]),
    ("1x2_spread", {"label": "fixed ratio", "long_delta": 0.5, "short_delta": 0.25}, [1.0, -2.0]),
    ("1x2x1_spread", {"label": "fixed butterfly", "long_delta": 0.5, "short_delta": 0.25}, [1.0, -2.0, 1.0]),
]


def _normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _black76(forward, strike, tau, vol, rd, is_call):
    """Independent formula: do not share the production pricing implementation."""
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * tau) / (vol * math.sqrt(tau))
    d2 = d1 - vol * math.sqrt(tau)
    sign = 1 if is_call else -1
    return math.exp(-rd * tau) * sign * (
        forward * _normal_cdf(sign * d1) - strike * _normal_cdf(sign * d2)
    )


@pytest.mark.parametrize("rd,rf", [(0.12, 0.04), (0.03, 0.04)], ids=["positive-carry", "negative-carry"])
@pytest.mark.parametrize("is_call", [True, False], ids=["call", "put"])
@pytest.mark.parametrize("family,definition,quantities", CASES, ids=[c[0] for c in CASES])
def test_fixed_contract_premium_payoff_and_no_shock(family, definition, quantities, is_call, rd, rf):
    spot, vol, tau = 5.0, 0.15, 0.25
    forward = spot * math.exp((rd - rf) * tau)
    target = forward * (1.12 if is_call else 0.88)
    ms = compute_market_state(
        spot=spot, fwd=forward, vol=vol, T=tau, r_d=rd, r_f=rf,
        target=target, direction="base_higher" if is_call else "base_lower",
    )
    structure = build_structure(family, definition, is_call)
    assert structure is not None
    priced = price(structure, ms, target=target)
    before = deepcopy(priced)
    assert [leg.notional for leg in priced.priced_legs] == quantities
    assert all(leg.leg.right == (Right.CALL if is_call else Right.PUT)
               for leg in priced.priced_legs)

    expected_premium = sum(
        q * _black76(forward, k, tau, vol, rd, is_call)
        for q, k in zip(quantities, priced.strikes)
    ) / spot
    assert priced.net_premium_pct == pytest.approx(expected_premium, abs=1e-12)

    no_shock = _value_variant(family, priced, spot, vol, tau, rd, rf, spot, is_call) / spot
    assert no_shock == pytest.approx(expected_premium, abs=1e-12)

    # Include exact strikes and both tails; conversion is at terminal spot.
    grid = np.unique(np.r_[np.linspace(spot * 0.5, spot * 1.5, 31), priced.strikes])
    sign = 1 if is_call else -1
    expected_payoff = sum(q * np.maximum(sign * (grid - k), 0.0)
                          for q, k in zip(quantities, priced.strikes)) / grid
    payoff = base_ccy_payoff_for_trade_rec(
        family, strikes=priced.strikes, barrier=None, is_call=is_call, entry_spot=spot,
    )
    np.testing.assert_allclose(payoff(grid), expected_payoff, atol=1e-12)
    scenario_expiry = [
        _value_variant(family, priced, s, vol, 0.0, rd, rf, spot, is_call) / s
        for s in grid
    ]
    np.testing.assert_allclose(scenario_expiry, expected_payoff, atol=1e-12)
    assert priced == before, "Valuation must not mutate the contract"
