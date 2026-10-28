"""Main's strike grammar and risk guards alongside retained flexible contracts."""

import math

import pytest

from agentic.price_structure import PricedStructure, PricingUnavailable, price_structure
from agentic.structure_request import parse_structure_request, to_variant_dict
from knowledge_engine.tail_policy import assign_variant_tails
from pricing.black_scholes import black76_call, black76_put
from tests.test_product_model_parity import _make_ms, _surface_for


@pytest.mark.parametrize("text,kinds", [
    ("1x1.3 5.00/5.20", ("strike", "strike")),
    ("1x1.3 25d/15d", ("delta", "delta")),
    ("1x1.3 ATMF/25d", ("atmf", "delta")),
    ("1x1.3 ATMF/5.2", ("atmf", "strike")),
    ("1x1.3 ATMF/target", ("atmf", "target")),
])
def test_number_roles_are_distinct(text, kinds):
    request = parse_structure_request(text)
    assert request.short_ratio == 1.3
    assert tuple(leg.kind for leg in request.legs) == kinds
    assert to_variant_dict(request)["short_ratio"] == 1.3


@pytest.mark.parametrize("is_call", [True, False])
def test_explicit_strikes_price_at_own_vol_and_keep_ratio_without_target(is_call):
    surface = _surface_for("USDBRL", False)
    ms, _ = _make_ms("USDBRL", 90, 12, "base_higher" if is_call else "base_lower", surface)
    k1, k2 = (ms.fwd * 0.99, ms.fwd * 1.04) if is_call else (ms.fwd * 1.01, ms.fwd * 0.96)
    result = price_structure(f"1x1.3 {k1!r}/{k2!r}", ms, is_call=is_call,
                             loss_budget=100, linear_notional=10000)
    assert isinstance(result, PricedStructure)
    pv = result.variant
    assert pv.strikes == pytest.approx([k1, k2])
    assert [leg.signed_notional for leg in pv.resolved_contract.legs] == [1, -1.3]
    fn = black76_call if is_call else black76_put
    values = [fn(ms.fwd, k, ms.T, surface.vol_at_strike(k, ms.fwd, 90),
                 math.exp(-ms.r_d * ms.T)) for k in (k1, k2)]
    assert pv.net_premium_pct == pytest.approx((values[0] - 1.3 * values[1]) / ms.spot)
    assert pv.structure_notional is not None
    assign_variant_tails("ratio_spread", pv, is_call, to_variant_dict(result.request))
    assert pv.higher_spot_tail is is_call
    assert pv.lower_spot_tail is (not is_call)
    assert pv.can_lose_beyond_premium is True
    reversed_result = price_structure(f"1x1.3 {k2!r}/{k1!r}", ms, is_call=is_call)
    assert isinstance(reversed_result, PricingUnavailable)


def test_bare_number_is_not_silently_retried_as_delta():
    ms, target = _make_ms("USDBRL", 90, 12, "base_higher", None)
    bare = price_structure("1x1.3 ATMF/25", ms, is_call=True, target=target)
    assert isinstance(bare, PricingUnavailable)
    assert "Strike 25" in bare.detail
    marked = price_structure("1x1.3 ATMF/25d", ms, is_call=True, target=target)
    assert isinstance(marked, PricedStructure)
    assert marked.variant.strikes[0] == ms.fwd
