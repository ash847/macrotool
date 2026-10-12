from types import SimpleNamespace

import pytest

from knowledge_engine.payoff_risk import payoff_risk_note


def inputs(strikes, weights, is_call=True, premium=0.01, barrier=None):
    variant = SimpleNamespace(strikes=strikes, net_premium_pct=premium, barrier=barrier)
    product = SimpleNamespace(priced_legs=[
        SimpleNamespace(notional=weight, strike=strike, leg=SimpleNamespace(right=SimpleNamespace(value="call" if is_call else "put")))
        for weight, strike in zip(weights, strikes)
    ])
    return variant, product


@pytest.mark.parametrize("ratio", [1.5, 2])
@pytest.mark.parametrize("premium", [0.01, 0.0, -0.01])
@pytest.mark.parametrize("is_call", [True, False])
def test_ratio_threshold_is_net_of_premium_and_matches_scenario_engine(ratio, premium, is_call):
    from analytics.scenario_pricer import _value_variant
    strikes = [1.4, 1.45] if is_call else [1.4, 1.35]
    variant, product = inputs(strikes, [1, -ratio], is_call, premium)
    family = "1x2_spread" if ratio == 2 else "1x1.5_spread"
    intercept = ratio * strikes[1] - strikes[0] if is_call else strikes[0] - ratio * strikes[1]
    slope = 1 - ratio if is_call else ratio - 1
    crossing = -intercept / (slope - premium)
    note = payoff_risk_note(family, variant, product, is_call)
    assert f"net {'short' if is_call else 'long'} {'above' if is_call else 'below'} {strikes[1]:.4f}" in note
    assert f"net losses {'above' if is_call else 'below'} {crossing:.4f}" in note
    assert "target" not in note and "unlimited" not in note
    value = _value_variant(family, variant, crossing, 0.1, 0, 0.03, 0.02, 1.4, is_call)
    assert value / crossing - premium == pytest.approx(0, abs=1e-12)


def test_butterfly_zero_outer_payoff_only_for_balanced_wings():
    variant, product = inputs([1.3, 1.4, 1.5], [1, -2, 1])
    note = payoff_risk_note("1x2x1_spread", variant, product, True)
    assert "peaks at 1.4000" in note and "net P&L is -1.00%" in note
    variant, product = inputs([1.3, 1.4, 1.55], [1, -2, 1])
    assert "Unequal wings" in payoff_risk_note("1x2x1_spread", variant, product, True)


@pytest.mark.parametrize("is_call", [True, False])
def test_barrier_and_digital_conditions(is_call):
    barrier = 1.5 if is_call else 1.3
    variant, product = inputs([1.4], [1], is_call, barrier=barrier)
    side = "above" if is_call else "below"
    expiry = payoff_risk_note("european_rko", variant, product, is_call)
    assert f"at or {side} {barrier:.4f}" in expiry
    assert "earlier barrier touch does not" in expiry
    assert "during its life" in payoff_risk_note("rko", variant, product, is_call)
    assert "never touched" in payoff_risk_note("european_digital_rko", variant, product, is_call)
    assert "100% of base-currency notional" in payoff_risk_note("european_digital", variant, product, is_call)


def test_spread_and_missing_legs():
    variant, product = inputs([1.4, 1.5], [1, -1])
    assert "Maximum net loss: 1.00%" in payoff_risk_note("1x1_spread", variant, product, True)
    assert "actual option legs not retained" in payoff_risk_note("1x2_spread", variant, None, True)


def test_seagull_uses_actual_opposite_wing():
    variant, product = inputs([1.4, 1.5, 1.3], [1, -1, -0.5], premium=0)
    product.priced_legs[2].leg.right.value = "put"
    note = payoff_risk_note("seagull", variant, product, True)
    assert "net long below 1.3000" in note
    assert "net losses below 1.3000" in note
    assert "capped beyond 1.5000" in note


def test_tail_without_a_positive_spot_breakeven_does_not_invent_one():
    variant, product = inputs([1.4, 1.35], [1, -2], False, premium=1.1)
    note = payoff_risk_note("1x2_spread", variant, product, False)
    assert "net losses throughout that tail region" in note
    variant, product = inputs([1.4, 1.45], [1, -2], premium=-1.1)
    assert "no net-loss crossing" in payoff_risk_note("1x2_spread", variant, product, True)


def test_linear_and_invalid_terms():
    variant, product = inputs([1.4], [1])
    assert "moves lower" in payoff_risk_note("linear", variant, None, True)
    assert "moves higher" in payoff_risk_note("linear", variant, None, False)
    variant.net_premium_pct = float("nan")
    assert "unavailable" in payoff_risk_note("vanilla", variant, product, True)
