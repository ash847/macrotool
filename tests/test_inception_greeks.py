"""Fixed-contract numerical checks; never bump a construction anchor."""

from dataclasses import replace
import json
import math

import pytest

from analytics.inception_greeks import inception_greeks
from analytics.product_model import Right
from analytics.structure_pricer import price_variants
from pricing.black_scholes import black76_call, black76_put
from tests.test_product_model_parity import _make_ms, _surface_for


@pytest.mark.parametrize("family", ["vanilla", "1x1_spread", "1x1.5_spread", "1x2_spread", "1x2x1_spread", "seagull", "ratio_spread"])
@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("flat", [True, False])
@pytest.mark.parametrize("pair", ["USDBRL", "EURPLN"])
def test_analytic_risks_match_fixed_contract_bumps(family, is_call, flat, pair, monkeypatch):
    surface = _surface_for(pair, flat)
    ms, target = _make_ms(pair, 90, 12, "base_higher" if is_call else "base_lower", surface)
    extra = {"variants_override": [dict(label="ratio", long_delta=0.34, short_delta=0.25, short_ratio=1.3)]} if family == "ratio_spread" else {}
    variants = price_variants(ms, family, target=target, is_call=is_call,
                              smile=surface, loss_budget=100, **extra)
    assert variants

    def fail(*args, **kwargs):
        raise AssertionError("Greeks must not reconstruct the trade")

    monkeypatch.setattr("analytics.product_pricer.construct_package", fail)
    for variant in variants:
        product = variant.product_price
        n = 1_000_000
        kwargs = dict(spot=ms.spot, forward=ms.fwd, quote_rate=ms.r_d)
        result = inception_greeks(product, notional=n, **kwargs)
        doubled = inception_greeks(product, notional=2*n, **kwargs)
        zero = inception_greeks(product, notional=0, **kwargs)
        for field in result.package.__dataclass_fields__:
            assert getattr(result.package, field) == pytest.approx(sum(getattr(leg, field) for leg in result.legs))
            assert getattr(doubled.package, field) == pytest.approx(2 * getattr(result.package, field))
            assert getattr(zero.package, field) == 0

        def pv(spot, shift=0):
            return n * sum(
                leg.signed_notional * (black76_call if leg.right == Right.CALL else black76_put)(
                    ms.fwd * spot / ms.spot, leg.strike, ms.T, quote.vol + shift, math.exp(-ms.r_d * ms.T))
                for leg, quote in zip(product.resolved_contract.legs, product.priced_legs))

        center = pv(ms.spot)
        for bump in (0.00005, 0.0001, 0.0002):
            h = ms.spot * bump
            delta = (pv(ms.spot+h) - pv(ms.spot-h)) / (2*h)
            half_delta = (pv(ms.spot+h/2) - pv(ms.spot-h/2)) / h
            gamma = (pv(ms.spot+h) - 2*center + pv(ms.spot-h)) / h**2 * ms.spot * 0.01
            half_gamma = (pv(ms.spot+h/2) - 2*center + pv(ms.spot-h/2)) / (h/2)**2 * ms.spot * 0.01
            # Richardson cancellation removes O(h^2) truncation when net
            # spread delta is near zero; tolerance is 0.05 units per million.
            assert result.package.delta_base == pytest.approx((4*half_delta-delta)/3, rel=2e-4, abs=0.05)
            assert result.package.gamma_base_per_1pct_spot == pytest.approx((4*half_gamma-gamma)/3, rel=2e-4, abs=0.05)
        # 0.001 absolute vol = 0.1 volatility point, reported per 1 point.
        for h in (0.0005, 0.001, 0.002):
            vega = (pv(ms.spot, h) - pv(ms.spot, -h)) / (2*h) * 0.01 / ms.spot
            assert result.package.vega_base_per_vol_point == pytest.approx(vega, rel=0.005, abs=0.5)


def test_rejects_missing_or_inconsistent_entry_quotes():
    ms, target = _make_ms("USDBRL", 90, 12, "base_higher", None)
    product = price_variants(ms, "vanilla", target=target)[0].product_price
    kwargs = dict(spot=ms.spot, forward=ms.fwd, quote_rate=ms.r_d, notional=100)
    with pytest.raises(ValueError, match="vanilla-leg"):
        inception_greeks(replace(product, resolved_contract=None), **kwargs)
    with pytest.raises(ValueError, match="match"):
        inception_greeks(replace(product, priced_legs=[]), **kwargs)
    with pytest.raises(ValueError, match="differs"):
        inception_greeks(replace(product, priced_legs=[replace(product.priced_legs[0], strike=1)]), **kwargs)
    with pytest.raises(ValueError, match="finite"):
        inception_greeks(product, **dict(kwargs, notional=float("nan")))


def test_agent_greeks_are_on_demand_and_context_bound():
    from agentic.session import AgentSession
    from agentic.tools import dispatch
    from agentic.shortlist import shortlist_reference
    from config.loader import load_config
    from tests.test_product_model_parity import _frozen_snapshot

    session = AgentSession(snapshot=_frozen_snapshot(), cfg=load_config(), linear_notional=10000)
    assert dispatch(session, "inception_greeks", {})[1]
    text, error = dispatch(session, "run_standard_pack", {"pair": "USDBRL", "horizon_days": 90,
                           "direction": "base_higher", "magnitude_pct": 12})
    assert not error, text
    assert "vega_base_per_vol_point" not in text
    ref = shortlist_reference(session.pack, session.view)
    rec = next(r for r in session.pack.recommended if r.variant.product_price is not None
               and r.variant.resolved_contract is not None)
    text, error = dispatch(session, "inception_greeks", {"shortlist_ref": ref, "rank": rec.rank})
    assert not error, text
    data = json.loads(text)
    assert data["base_currency"] == "USD"
    assert data["greeks"]["notional_base"] == rec.variant.structure_notional
    # The provider-neutral loop must pass precomputed Greeks back to the model
    # without automatically adding a dashboard/shortlist to this answer.
    from agentic.agent_flow import AgentFlow
    from agentic.agent_llm import FakeToolLLM, LLMTurn, ToolCall
    llm = FakeToolLLM(script=[
        LLMTurn(text="", tool_calls=[ToolCall("greeks", "inception_greeks",
                {"shortlist_ref": ref, "rank": rec.rank})], stop_reason="tool_use"),
        LLMTurn(text="These are the requested inception risks.", tool_calls=[], stop_reason="end_turn"),
    ])
    flow = AgentFlow(llm, session)
    assert flow.advance("What are the Greeks for that recommendation?") == "These are the requested inception risks."
    assert any("vega_base_per_vol_point" in str(message) for message in session.messages)
    for args in ({"rank": rec.rank}, {"rank": True}, {"rank": 9999},
                 {"rank": rec.rank, "custom_index": 1}, {"custom_index": 9999}):
        if args == {"rank": rec.rank}:
            assert dispatch(session, "inception_greeks", args)[1]
        else:
            assert dispatch(session, "inception_greeks", dict(args, shortlist_ref=ref))[1]
    text, error = dispatch(session, "price_structure", {"request": "1x1.3 ATMF/25d"})
    assert not error, text
    assert "CUSTOM TRADE INDEX: 1" in text
    args = {"shortlist_ref": ref, "custom_index": 1}
    assert not dispatch(session, "inception_greeks", args)[1]
    # Unsupported product must not silently receive vanilla Greeks.
    variant = session.priced[0].variant
    retained = variant.product_price
    variant.product_price = None
    assert dispatch(session, "inception_greeks", args)[1]
    variant.product_price = retained
    # Even with a valid current reference an old custom trade is rejected.
    session.priced[0] = replace(session.priced[0], inception_market_state=object())
    assert dispatch(session, "inception_greeks", args)[1]
