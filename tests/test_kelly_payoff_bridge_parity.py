"""Component C — payoff-bridge parity vs the product model (engine source of truth).

`analytics.payoffs.base_ccy_payoff_for_trade_rec` must reproduce the engine's terminal
payoff. For leg-based families we compare against the product model's per-leg intrinsic
sum (parity-locked to structure_pricer). Digitals are pinned to the base-ccy convention.
"""
import numpy as np
import pytest

from analytics.market_state import compute_market_state
from analytics.payoffs import base_ccy_payoff_for_trade_rec
from analytics.product_model import Right
from analytics.product_pricer import build_structure, price
from analytics.structure_pricer import _load_variants, price_variants

_LEG_FAMILIES = ["vanilla", "1x1_spread", "1x1.5_spread", "1x2_spread", "1x2x1_spread", "seagull"]

_MARKETS = [
    # (spot, fwd, vol, T, r_d, r_f, target)  — high-carry and near-flat
    (5.0, 5.20, 0.15, 0.25, 0.12, 0.04, 5.45),
    (5.0, 5.05, 0.12, 0.50, 0.05, 0.04, 5.40),
]


def _ms(mkt, direction):
    spot, fwd, vol, T, r_d, r_f, tgt = mkt
    return compute_market_state(spot=spot, fwd=fwd, vol=vol, T=T, r_d=r_d, r_f=r_f,
                                target=tgt, direction=direction), tgt


def _product_terminal(ps, prices):
    """Σ signed_notional · intrinsic(S) / S from the product model's priced legs."""
    total = np.zeros_like(prices, dtype=float)
    for pl in ps.priced_legs:
        n = pl.effective_notional if getattr(pl, "effective_notional", None) is not None else pl.notional
        if pl.leg.right == Right.CALL:
            intr = np.maximum(prices - pl.strike, 0.0)
        else:
            intr = np.maximum(pl.strike - prices, 0.0)
        total += n * intr
    return total / prices


@pytest.mark.parametrize("direction", ["base_higher", "base_lower"])
@pytest.mark.parametrize("mkt", _MARKETS)
@pytest.mark.parametrize("family", _LEG_FAMILIES)
def test_bridge_matches_product_model(family, mkt, direction):
    ms, target = _ms(mkt, direction)
    is_call = direction == "base_higher"
    stop = ms.fwd * (1 - 0.05) if is_call else ms.fwd * (1 + 0.05)   # seagull max-loss anchor
    pvs = price_variants(ms, family, target=target, is_call=is_call, stop_price=stop)
    if not pvs:
        pytest.skip(f"{family}: no eligible variants for this market")
    by_label = {v.variant_label: v for v in pvs}
    grid = np.linspace(0.7 * ms.spot, 1.5 * ms.spot, 21)
    checked = 0
    for variant in _load_variants()[family]:
        pv = by_label.get(variant["label"])
        if pv is None:
            continue
        st = build_structure(family, variant, is_call)
        if st is None:
            continue
        ps = price(st, ms, target=target, smile=None, stop_price=stop)
        # The seagull wing ratio is display-rounded to 2dp on PricedVariant; use the
        # product model's full-precision solved ratio so this isolates the bridge's
        # leg/sign/strike structure from that rounding (a separate, ~0.3% concern).
        wing_ratio = pv.wing_ratio
        if family == "seagull":
            wing_ratio = abs(ps.priced_legs[2].effective_notional)
        bridge = base_ccy_payoff_for_trade_rec(
            family, strikes=pv.strikes, barrier=pv.barrier, is_call=is_call,
            entry_spot=ms.spot, wing_ratio=wing_ratio,
        )
        np.testing.assert_allclose(
            bridge(grid), _product_terminal(ps, grid), atol=1e-9,
            err_msg=f"{family} '{variant['label']}' {direction}",
        )
        checked += 1
    assert checked > 0


def test_1x2x1_bridge_is_long_short2_long():
    # +1@K1 −2@K2 +1@K3, with the wing at K3 = 2·K2 − K1.
    k1, k2 = 5.0, 5.3
    k3 = 2 * k2 - k1
    payoff = base_ccy_payoff_for_trade_rec(
        "1x2x1_spread", strikes=[k1, k2, k3], barrier=None, is_call=True, entry_spot=5.0,
    )
    prices = np.array([4.8, k2, 6.0])
    expected = (np.maximum(prices - k1, 0) - 2 * np.maximum(prices - k2, 0)
                + np.maximum(prices - k3, 0)) / prices
    np.testing.assert_allclose(payoff(prices), expected, atol=1e-12)
    # peak at K2, zero outside the wings (≤K1 or ≥K3)
    assert payoff(np.array([k2]))[0] > 0
    assert payoff(np.array([4.8]))[0] == 0.0
    assert payoff(np.array([k3 + 0.5]))[0] == pytest.approx(0.0, abs=1e-12)


def test_digital_base_ccy_convention():
    call = base_ccy_payoff_for_trade_rec("european_digital", strikes=[5.5], barrier=None,
                                         is_call=True, entry_spot=5.0)
    put = base_ccy_payoff_for_trade_rec("european_digital", strikes=[5.5], barrier=None,
                                        is_call=False, entry_spot=5.0)
    np.testing.assert_allclose(call(np.array([5.4, 5.6])), [0.0, 1.0])
    np.testing.assert_allclose(put(np.array([5.4, 5.6])), [1.0, 0.0])


def test_european_rko_knocks_out():
    payoff = base_ccy_payoff_for_trade_rec("european_rko", strikes=[5.2], barrier=5.8,
                                           is_call=True, entry_spot=5.0)
    prices = np.array([5.0, 5.5, 6.0])   # below K, ITM in-band, knocked out above barrier
    got = payoff(prices)
    assert got[0] == 0.0
    assert got[1] == pytest.approx((5.5 - 5.2) / 5.5)
    assert got[2] == 0.0
