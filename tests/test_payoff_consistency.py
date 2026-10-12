import numpy as np
import pytest

from analytics.expiry_breakeven import expiry_breakevens
from analytics.payoffs import base_ccy_payoff_for_trade_rec
from analytics.product_pricer import build_structure, price
from analytics.scenario_pricer import _value_variant
from analytics.structure_pricer import price_variants
from tests.test_product_model_parity import _make_ms


@pytest.mark.parametrize("is_call", [True, False])
def test_vanilla_breakeven_uses_base_currency(is_call):
    roots, flat = expiry_breakevens([(1, 100, is_call)], 0.02)
    assert roots == pytest.approx([100 / (0.98 if is_call else 1.02)])
    assert not flat


def test_ratio_has_both_crossings():
    roots, flat = expiry_breakevens([(1, 100, True), (-2, 110, True)], 0.02)
    assert roots == pytest.approx([100 / 0.98, 120 / 1.02])
    assert not flat


def test_credit_ratio_has_only_tail_crossing():
    roots, _ = expiry_breakevens([(1, 100, True), (-2, 110, True)], -0.01)
    assert roots == pytest.approx([120 / 0.99])


def test_zero_cost_regions_are_not_isolated_roots():
    roots, flat = expiry_breakevens([(1, 100, True), (-2, 110, True)], 0)
    assert roots == pytest.approx([120])
    assert flat


def test_digital_jump_is_not_breakeven():
    assert expiry_breakevens([(1, 100, True)], 0.1, digital=True) == ([], False)


def test_barrier_jump_cannot_create_a_root():
    roots, _ = expiry_breakevens([(1, 100, True)], 1 / 6, barrier=120)
    assert not roots
    roots, _ = expiry_breakevens([(1, 100, True)], 0.02, barrier=120)
    assert roots == pytest.approx([100 / 0.98])


@pytest.mark.parametrize("family,ratio", [("1x1.5_spread", 1.5), ("1x2_spread", 2)])
@pytest.mark.parametrize("direction", ["base_higher", "base_lower"])
def test_ratio_target_all_legs_across_strikes(family, ratio, direction):
    ms, target = _make_ms("GBPUSD", 90, 6, direction, None)
    is_call = direction == "base_higher"
    config = {"label": "audit", "long_delta": 0.4, "short_delta": 0.2}
    seed = price_variants(ms, family, target=target, is_call=is_call, variants_override=[config])[0]
    for spot in [seed.strikes[0] * 0.95, *seed.strikes, seed.strikes[0] * 1.1]:
        variant = price_variants(ms, family, target=spot, is_call=is_call, variants_override=[config])[0]
        product = price(build_structure(family, config, is_call), ms, target=spot)
        intrinsic = lambda strike: max(spot - strike if is_call else strike - spot, 0)
        expected = (intrinsic(variant.strikes[0]) - ratio * intrinsic(variant.strikes[1])) / spot
        assert variant.payoff_at_target_pct == pytest.approx(expected)
        assert product.payoff_at_target_pct == pytest.approx(expected)
        assert variant.economics.target_net_pnl_pct == pytest.approx(expected - variant.net_premium_pct)
        if variant.rr_at_target is not None:
            assert variant.rr_at_target == pytest.approx(expected / variant.net_premium_pct)
        assert product.breakevens == pytest.approx(variant.breakevens)
        for root in variant.breakevens:
            value = _value_variant(family, variant, root, ms.vol, 0, ms.r_d, ms.r_f, ms.spot, is_call)
            assert value / root - variant.net_premium_pct == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("direction", ["base_higher", "base_lower"])
def test_seagull_full_precision_shared_with_kelly_and_scenarios(direction):
    ms, target = _make_ms("GBPUSD", 90, 6, direction, None)
    is_call = direction == "base_higher"
    config = {"label": "audit", "spread_long": 0.5, "spread_short": 0.25, "wing_delta": 0.25}
    variant = price_variants(ms, "seagull", target=target, is_call=is_call, variants_override=[config])[0]
    product = price(build_structure("seagull", config, is_call), ms, target=target)
    assert variant.wing_ratio == pytest.approx(-product.priced_legs[2].notional, abs=1e-14)
    assert product.wing_ratio == variant.wing_ratio
    assert abs(variant.wing_ratio - round(variant.wing_ratio, 2)) > 1e-8
    assert sum(leg.notional * leg.unit_premium for leg in product.priced_legs) == pytest.approx(0, abs=1e-12)
    bridge = base_ccy_payoff_for_trade_rec("seagull", strikes=variant.strikes, barrier=None,
                                         is_call=is_call, entry_spot=ms.spot, wing_ratio=variant.wing_ratio)
    for spot in [min(variant.strikes) * 0.8, max(variant.strikes) * 1.2]:
        expected = sum(leg.notional * max(spot - leg.strike if leg.leg.right.value == "call" else leg.strike - spot, 0)
                       for leg in product.priced_legs) / spot
        assert bridge(np.array([spot]))[0] == pytest.approx(expected)
        assert _value_variant("seagull", variant, spot, ms.vol, 0, ms.r_d, ms.r_f, ms.spot, is_call) / spot == pytest.approx(expected)


def test_capped_quote_payoff_can_have_two_base_currency_breakevens():
    roots, _ = expiry_breakevens([(1, 100, True), (-1, 110, True)], 0.02)
    assert roots == pytest.approx([100 / 0.98, 10 / 0.02])


def test_no_breakeven_when_premium_exceeds_maximum_payoff():
    assert expiry_breakevens([(1, 100, True)], 1.1) == ([], False)
