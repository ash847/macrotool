from types import SimpleNamespace

import pytest

from agentic.render import _carry_explanation, _legs_breakdown
from analytics.market_state import compute_market_state
from analytics.product_pricer import build_structure, price


@pytest.mark.parametrize("forward,direction,alignment,action,relation,yield_ccy", [
    (6.5, "base_higher", "WITH", "buy", "below", "USD"),
    (6.5, "base_lower", "COUNTER", "sell", "below", "USD"),
    (6.9, "base_higher", "COUNTER", "buy", "above", "CNH"),
    (6.9, "base_lower", "WITH", "sell", "above", "CNH"),
])
def test_selected_carry_case(forward, direction, alignment, action, relation, yield_ccy):
    state = compute_market_state(spot=6.7, fwd=forward, vol=0.033, T=1, r_d=0.01,
                                 r_f=0.04, direction=direction)
    text = _carry_explanation(state, SimpleNamespace(pair="USDCNH", direction=direction, horizon_days=365))
    assert ("WITH the carry" if alignment == "WITH" else "COUNTER to the carry") in text
    assert f"You {action} USD forward {relation}" in text
    assert f"{yield_ccy} has the higher implied interest rate" in text
    assert ("would favour" if alignment == "WITH" else "would work against") in text
    assert "not a forecast of spot" in text
    assert f"forward={forward:.4f}" in text


@pytest.mark.parametrize("direction", ["base_higher", "base_lower"])
def test_equal_forward_is_neutral(direction):
    state = compute_market_state(spot=6.7, fwd=6.7, vol=0.033, T=1, r_d=0.04,
                                 r_f=0.04, direction=direction)
    text = _carry_explanation(state, SimpleNamespace(pair="USDCNH", direction=direction, horizon_days=365))
    assert "no implied carry advantage" in text
    assert "WITH the carry" not in text and "COUNTER" not in text


@pytest.mark.parametrize("is_call", [True, False])
def test_geometric_wing_never_inherits_short_delta(is_call):
    state = compute_market_state(spot=6.7125, fwd=6.511122, vol=0.0329, T=1,
                                 r_d=0.0139, r_f=0.0444,
                                 direction="base_higher" if is_call else "base_lower")
    structure = build_structure("1x2x1_spread", {"long_delta": 0.25, "short_delta": 0.10}, is_call)
    priced = price(structure, state, target=6.85 if is_call else 6.0)
    before = (priced.strikes[:], priced.net_premium_pct)
    lines = _legs_breakdown(priced, 1_000_000, "USD")
    assert "25Δ" in lines[0] and "10Δ" in lines[1]
    assert "long wing" in lines[2] and "Δ" not in lines[2]
    assert "delta not supplied" in lines[2]
    assert f"{priced.strikes[2]:.4f}" in lines[2]
    assert priced.strikes[2] == pytest.approx(2 * priced.strikes[1] - priced.strikes[0])
    assert before == (priced.strikes, priced.net_premium_pct)


def test_actual_cnh_pack_contains_correct_case_and_wing_label():
    from agentic.render import render_pack
    from agentic.standard_pack import build_pack
    from config.loader import load_config
    from data.snapshot_loader import load_snapshot
    from knowledge_engine.models import TradeView
    from pricing.forwards import rate_context_for_snapshot

    currency = load_snapshot().get("USDCNH")
    forward = rate_context_for_snapshot(currency, 1).forward
    view = TradeView(pair="USDCNH", direction="base_higher", direction_conviction="medium",
                     horizon_days=365, magnitude_pct=(6.85 / forward - 1) * 100)
    pack = build_pack(view, currency, load_config())
    text = render_pack(pack, view)
    assert "WITH the carry" in text and "You buy USD forward below" in text
    assert "sell forward at a premium" not in text
    butterfly = next(rec for rec in pack.recommended if rec.structure_id == "1x2x1_spread")
    from agentic.render import render_recommended
    assert "long wing Call" in render_recommended(butterfly, "USD")
