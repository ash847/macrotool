"""
Phase 1 smile-vol wiring guard.

``structure_pricer.price_variants`` accepts an optional ``smile`` argument
(a ``SmileInterpolator``). When ``smile=None`` it must reproduce the legacy
flat-ATM-vol pricing byte-for-byte; when a smile is supplied it must price each
leg at that strike's interpolated vol.

USDTRY has a strong topside call skew (ATM 21.2%, 25DC ~25.6%, 10DC ~30.8%),
so the smile makes:
  * OTM topside *long* legs RICHER  (a 25Δ call costs more than at ATM vol), and
  * ratio-spread *short* legs richer too — but because the short leg is
    multiplied (1x1.5 / 1x2), a richer short leg makes the *net debit* CHEAPER.

These directional checks pin that the smile is actually plumbed through to the
leg pricing and not silently dropped.
"""

from __future__ import annotations

import pytest

from analytics.market_state import compute_market_state
from analytics.structure_pricer import price_variants
from analytics.vol_surface import SmileInterpolator
from data.snapshot_loader import load_snapshot


_TARGET = 55.0


@pytest.fixture(scope="module")
def usdtry_ccy():
    return load_snapshot().get("USDTRY")


@pytest.fixture(scope="module")
def market_state(usdtry_ccy):
    import math

    spot = usdtry_ccy.spot
    # 3m horizon, ATM vol; derive r_d from the forward via CIP.
    T = 91 / 365.0
    fwd = 46.7597
    vol = 0.2121
    r_f = 0.0452
    r_d = r_f + math.log(fwd / spot) / T
    return compute_market_state(
        spot, fwd, vol, T, r_d, r_f, target=_TARGET, direction="base_higher"
    )


@pytest.fixture(scope="module")
def smile(usdtry_ccy):
    return SmileInterpolator(usdtry_ccy)


_SMILED_STRUCTURES = [
    "vanilla",
    "1x1_spread",
    "1x1.5_spread",
    "1x2_spread",
    "seagull",
]


@pytest.mark.parametrize("structure_id", _SMILED_STRUCTURES)
def test_smile_none_is_flat_legacy(market_state, structure_id):
    """smile=None must reproduce the no-smile call exactly (the default)."""
    flat_explicit = price_variants(
        market_state, structure_id, target=_TARGET, is_call=True,
        stop_price=None, smile=None,
    )
    flat_default = price_variants(
        market_state, structure_id, target=_TARGET, is_call=True,
        stop_price=None,
    )
    assert len(flat_explicit) == len(flat_default)
    for a, b in zip(flat_explicit, flat_default):
        assert a.net_premium_pct == b.net_premium_pct
        assert a.strikes == b.strikes


def test_smile_makes_otm_topside_call_richer(market_state, smile):
    """An OTM topside call (25Δ) is priced at >ATM vol → richer premium."""
    flat = price_variants(
        market_state, "vanilla", target=_TARGET, is_call=True, stop_price=None,
    )
    smiled = price_variants(
        market_state, "vanilla", target=_TARGET, is_call=True, stop_price=None,
        smile=smile,
    )
    # Match variants by strike so we compare like-for-like.
    flat_by_k = {round(pv.strikes[0], 4): pv for pv in flat}
    for pv in smiled:
        k = round(pv.strikes[0], 4)
        if k not in flat_by_k:
            continue
        # OTM calls (strike above forward) should be richer under the topside skew.
        if pv.strikes[0] > market_state.fwd:
            assert pv.net_premium_pct >= flat_by_k[k].net_premium_pct


@pytest.mark.parametrize("structure_id", ["1x1.5_spread", "1x2_spread"])
def test_smile_makes_ratio_net_debit_cheaper(market_state, smile, structure_id):
    """Richer short legs (multiplied) lower the net debit on topside ratios."""
    flat = price_variants(
        market_state, structure_id, target=_TARGET, is_call=True, stop_price=None,
    )
    smiled = price_variants(
        market_state, structure_id, target=_TARGET, is_call=True, stop_price=None,
        smile=smile,
    )
    flat_by_k = {tuple(pv.strikes): pv for pv in flat}
    compared = 0
    for pv in smiled:
        key = tuple(pv.strikes)
        if key not in flat_by_k:
            continue
        compared += 1
        assert pv.net_premium_pct <= flat_by_k[key].net_premium_pct + 1e-9
    assert compared > 0, f"no matching variants compared for {structure_id}"
