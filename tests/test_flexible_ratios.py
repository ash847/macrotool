"""Explicit ratios survive parsing, valuation, sizing and agent rendering."""

import math

import numpy as np
import pytest

from agentic.price_structure import PricedStructure, PricingUnavailable, price_structure
from agentic.structure_request import StructureRequest, StructureRequestError, parse_structure_request, to_variant_dict
from analytics.payoffs import base_ccy_payoff_for_trade_rec
from analytics.scenario_pricer import _value_variant
from analytics.sizing import SizingSpec, kelly_fraction_per_notional, per_notional_pnl
from analytics.structure_pricer import price_variants
from tests.test_product_model_parity import _make_ms, _surface_for


@pytest.mark.parametrize("text", ["1x1.3 spread ATMF/25", "34 vs 25 1 x 1.3", "1x1.3 ATMF vs target"])
def test_explicit_ratio_is_not_parsed_as_one_to_one(text):
    from agentic.family_registry import resolve_family
    assert resolve_family(text.lower())[0] == "ratio_spread"
    request = parse_structure_request(text)
    assert isinstance(request, StructureRequest)
    assert request.family == "ratio_spread"
    assert to_variant_dict(request)["short_ratio"] == 1.3
    assert "1x1.3" in request.canonical


@pytest.mark.parametrize("text", ["1x0 spread 34/25", "1x-1.3 34/25", "1x0.8 34/25"])
def test_unsupported_ratio_rejected(text):
    with pytest.raises(StructureRequestError):
        parse_structure_request(text)


@pytest.mark.parametrize("ratio,family", [(1, "1x1_spread"), (1.5, "1x1.5_spread"), (2, "1x2_spread")])
def test_existing_ratio_names_keep_their_family(ratio, family):
    assert parse_structure_request(f"1x{ratio} 34/25").family == family


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("flat", [True, False])
def test_ratio_entry_payoff_scenario_and_fixed_sizing(is_call, flat):
    surface = _surface_for("USDBRL", flat)
    ms, target = _make_ms("USDBRL", 90, 12, "base_higher" if is_call else "base_lower", surface)
    result = price_structure("1x1.3 spread ATMF/25", ms, is_call=is_call,
                             target=target, smile=surface, loss_budget=100, linear_notional=10000)
    assert isinstance(result, PricedStructure)
    pv = result.variant
    contract = pv.resolved_contract
    assert [leg.signed_notional for leg in contract.legs] == [1, -1.3]
    quotes = result.priced_structure.priced_legs
    assert pv.net_premium_pct == pytest.approx((quotes[0].unit_premium - 1.3 * quotes[1].unit_premium) / ms.spot)
    assert pv.max_loss_pct == abs(pv.net_premium_pct)
    assert pv.structure_notional == pytest.approx(min(100 / pv.max_loss_pct, 100000))
    assert pv.economics.target_net_pnl_pct is not None
    assert pv.economics.contractual_loss_status == "unknown"
    payoff = base_ccy_payoff_for_trade_rec("ratio_spread", strikes=pv.strikes,
                                          barrier=None, is_call=is_call, entry_spot=ms.spot,
                                          resolved_contract=contract)
    prices = np.array([ms.spot * 0.5, ms.spot, ms.spot * 1.5])
    sign = 1 if is_call else -1
    expected = (np.maximum(sign * (prices - pv.strikes[0]), 0)
                - 1.3 * np.maximum(sign * (prices - pv.strikes[1]), 0)) / prices
    np.testing.assert_allclose(payoff(prices), expected)
    for spot, value in zip(prices, expected):
        assert _value_variant("ratio_spread", pv, spot, ms.vol, 0, ms.r_d, ms.r_f,
                              ms.spot, is_call, surface) / spot == pytest.approx(value)
    assert contract is result.priced_structure.resolved_contract


@pytest.mark.parametrize("is_call", [True, False])
def test_kelly_uses_full_requested_ratio(is_call):
    ms, target = _make_ms("USDBRL", 90, 12, "base_higher" if is_call else "base_lower", None)
    bins = np.array([ms.fwd * 0.7, ms.fwd, ms.fwd * 1.08, ms.fwd * 1.4])
    probs = np.array([0.1, 0.1, 0.7, 0.1])
    spec = SizingSpec(method="kelly", bankroll=10000, kelly_lambda=0.5,
                      kelly_probs=tuple(probs), kelly_bins=tuple(bins))
    result = price_structure("1x1.3 ATMF/25", ms, is_call=is_call, target=target,
                             sizing_spec=spec, linear_notional=10000)
    pv = result.variant
    sign = 1 if is_call else -1
    raw = (np.maximum(sign * (bins - pv.strikes[0]), 0)
           - 1.3 * np.maximum(sign * (bins - pv.strikes[1]), 0)) / bins
    pnl = per_notional_pnl(raw, pv.net_premium_pct, math.exp(-ms.r_f * ms.T))
    fraction = kelly_fraction_per_notional(probs, pnl)
    assert pv.kelly_fraction == pytest.approx(fraction)
    assert pv.structure_notional == pytest.approx(min(0.5 * fraction * 10000, 100000))
    assert pv.sizing_trace.effective_method == "kelly"


def test_custom_ratios_do_not_expand_recommendation_menu_or_bypass_no_tails():
    ms, target = _make_ms("USDBRL", 90, 12, "base_higher", None)
    assert price_variants(ms, "ratio_spread", target=target) == []
    result = price_structure("1x1.3 ATMF/25", ms, is_call=True, target=target,
                             structure_constraint="Avoid tail-risky structures")
    assert isinstance(result, PricingUnavailable)


def test_agent_tool_sizes_and_displays_requested_ratio():
    from agentic.session import AgentSession
    from agentic.tools import dispatch
    from config.loader import load_config
    from tests.test_product_model_parity import _frozen_snapshot
    session = AgentSession(snapshot=_frozen_snapshot(), cfg=load_config(), linear_notional=10000)
    _, error = dispatch(session, "run_standard_pack", {"pair": "USDBRL", "horizon_days": 90,
                  "direction": "base_higher", "magnitude_pct": 12})
    assert not error
    text, error = dispatch(session, "price_structure", {"request": "1x1.3 spread ATMF/25"})
    assert not error, text
    assert session.priced[-1].variant.structure_notional is not None
    assert session.priced[-1].variant.resolved_contract.legs[1].signed_notional == -1.3
    assert "1x1.3" in text


@pytest.mark.parametrize("ratio,family", [(1.5, "1x1.5_spread"), (2.0, "1x2_spread")])
@pytest.mark.parametrize("is_call", [True, False])
def test_flexible_path_matches_existing_ratio_policy(ratio, family, is_call):
    from tests.test_product_model_parity import _assert_parity
    ms, target = _make_ms("USDBRL", 90, 12, "base_higher" if is_call else "base_lower", None)
    definition = dict(label="same", long_delta=0.34, short_delta=0.25)
    fixed = price_variants(ms, family, target=target, is_call=is_call,
                           variants_override=[definition], loss_budget=100)[0]
    flexible = price_variants(ms, "ratio_spread", target=target, is_call=is_call,
                              variants_override=[dict(definition, short_ratio=ratio)], loss_budget=100)[0]
    _assert_parity(flexible, fixed, family)
    assert flexible.structure_notional == pytest.approx(fixed.structure_notional)


def test_ratio_target_requires_target_without_reseeding():
    ms, _ = _make_ms("USDBRL", 90, 12, "base_higher", None)
    assert isinstance(price_structure("1x1.3 ATMF/target", ms, is_call=True), PricingUnavailable)
