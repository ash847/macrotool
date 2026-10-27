"""Exact forward-delta conversion and smile entry/revaluation consistency."""

import math

import pytest
from scipy.stats import norm

from agentic.structure_request import parse_structure_request, to_variant_dict
from analytics.product_pricer import build_structure, price
from analytics.scenario_pricer import _value_variant
from analytics.strike_resolver import otm_call_strike, otm_put_strike
from analytics.structure_pricer import price_variants
from tests.test_product_model_parity import _make_ms, _surface_for


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("delta", [0.1, 0.15, 0.25, 0.30, 0.34, 0.40, 0.5])
@pytest.mark.parametrize("vol,T", [(0.09, 30 / 365), (0.15, 91 / 365), (0.4, 1.0)])
def test_exact_forward_delta_round_trip(is_call, delta, vol, T):
    forward = 5.2
    resolve = otm_call_strike if is_call else otm_put_strike
    strike = resolve(forward, vol, T, delta)
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * T) / (vol * math.sqrt(T))
    actual = norm.cdf(d1) if is_call else norm.cdf(-d1)
    assert actual == pytest.approx(delta, abs=1e-12)


@pytest.mark.parametrize("is_call", [True, False])
def test_atmf_and_true_fifty_delta_are_distinct(is_call):
    surface = _surface_for("USDBRL", False)
    ms, target = _make_ms("USDBRL", 90, 12, "base_higher" if is_call else "base_lower", surface)
    atmf = to_variant_dict(parse_structure_request("vanilla ATMF"))
    fifty = to_variant_dict(parse_structure_request("vanilla 50 delta"))
    assert atmf != fifty
    atmf_trade = price_variants(ms, "vanilla", target=target, is_call=is_call,
                                smile=surface, variants_override=[atmf])[0]
    fifty_trade = price_variants(ms, "vanilla", target=target, is_call=is_call,
                                 smile=surface, variants_override=[fifty])[0]
    assert atmf_trade.strikes[0] == ms.fwd
    assert fifty_trade.strikes[0] == pytest.approx(ms.fwd * math.exp(0.5 * ms.vol ** 2 * ms.T))
    assert fifty_trade.strikes[0] != pytest.approx(ms.fwd)
    product = price(build_structure("vanilla", atmf, is_call), ms, target=target, smile=surface)
    assert product.strikes == atmf_trade.strikes
    assert product.priced_legs[0].vol == pytest.approx(surface.vol_at_strike(ms.fwd, ms.fwd, 90))


@pytest.mark.parametrize("family", ["vanilla", "1x1_spread", "1x1.5_spread", "1x2_spread", "1x2x1_spread", "seagull"])
@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("flat", [True, False], ids=["flat", "smile"])
def test_unchanged_market_reproduces_entry_value(family, is_call, flat):
    surface = _surface_for("USDBRL", flat)
    ms, target = _make_ms("USDBRL", 90, 12, "base_higher" if is_call else "base_lower", surface)
    variants = price_variants(ms, family, target=target, is_call=is_call, smile=surface)
    assert variants
    for trade in variants:
        mark = _value_variant(family, trade, ms.spot, ms.vol, ms.T, ms.r_d, ms.r_f,
                              ms.spot, is_call, surface) / ms.spot
        assert mark == pytest.approx(trade.net_premium_pct, abs=5e-10), trade.variant_label


@pytest.mark.parametrize("family,request_text", [
    ("1x1_spread", "1x1 ATMF/25 delta"),
    ("1x1.5_spread", "1x1.5 ATMF/25 delta"),
    ("1x2_spread", "1x2 ATMF/25 delta"),
    ("seagull", "seagull ATMF/25 delta/25 delta"),
    ("european_rko", "erko ATMF/25 delta"),
])
def test_atmf_marker_survives_parser_and_product_builder(family, request_text):
    from analytics.product_model import AnchorKind
    definition = to_variant_dict(parse_structure_request(request_text))
    assert definition["atmf_legs"] == [0]
    assert build_structure(family, definition, True).legs[0].anchor.kind == AnchorKind.ATMF


def _steep_surface():
    from analytics.vol_surface import build_vol_surface
    from tests.test_product_model_parity import _frozen_snapshot
    ccy = _frozen_snapshot().get("USDBRL")
    # Fixed steep pillar shape, independent of future snapshot updates.
    vols = {"10DP": 0.0875125, "25DP": 0.079725, "ATM": 0.11305,
            "25DC": 0.178875, "10DC": 0.2942875}
    for node in ccy.vol_surface:
        node.vol = vols[node.delta]
    return build_vol_surface(ccy)


@pytest.mark.parametrize("delta", [0.1, 0.2, 0.25, 0.30, 0.34, 0.4, 0.5])
@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("max_iter", [0, 10])
def test_steep_smile_lookup_converges_even_when_iteration_budget_is_exhausted(delta, is_call, max_iter):
    surface = _steep_surface()
    vol = surface.vol_at_delta(delta if is_call else -delta, 91)
    resolve = otm_call_strike if is_call else otm_put_strike
    strike = resolve(52.5, vol, 91 / 365, delta)
    recovered = surface.vol_at_strike(strike, 52.5, 91, max_iter=max_iter)
    assert recovered == pytest.approx(vol, abs=1e-9)


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("family,count", [("european_rko", 6), ("european_digital", 3)])
def test_steep_smile_rejects_negative_packages_instead_of_clipping_them(family, count, is_call):
    from analytics.market_state import compute_market_state
    surface = _steep_surface()
    T = 91 / 365
    target = 52.5 * (1.15 if is_call else 0.85)
    ms = compute_market_state(
        spot=50, fwd=52.5, vol=surface.vol_at_delta(0.5, 91), T=T,
        r_d=0.04 + math.log(52.5 / 50) / T, r_f=0.04,
        direction="base_higher" if is_call else "base_lower", target=target, surface=surface,
    )
    warnings = []
    trades = price_variants(ms, family, target=target, is_call=is_call,
                            smile=surface, warnings=warnings)
    if family == "european_rko" and is_call:
        # Convergence fixes digital lookup, but this smile still has invalid
        # package values. They must not reappear as zero-premium trades.
        assert len(warnings) == 3
        assert all("negative European RKO package value" in w for w in warnings)
        assert len(trades) == count - 3
    else:
        assert not warnings, warnings
        assert len(trades) == count
    assert all(v.net_premium_pct > 0 for v in trades)


@pytest.mark.parametrize("resolve", [otm_call_strike, otm_put_strike])
@pytest.mark.parametrize("forward,vol,T,delta", [
    (0, .15, .25, .25), (5, 0, .25, .25), (5, .15, 0, .25),
    (5, .15, .25, 0), (5, .15, .25, 1), (5, float("nan"), .25, .25),
])
def test_invalid_delta_resolution_inputs_fail_explicitly(resolve, forward, vol, T, delta):
    with pytest.raises(ValueError):
        resolve(forward, vol, T, delta)
