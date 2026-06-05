"""
Phase 2 smile refactor guards.

The refactor introduces a pluggable ``VolSurface`` interface and pushes the
smile through every place a *vanilla* option is priced — atmfsratio (in
``compute_market_state``) and scenario mark-to-market — while keeping the
digital / RKO / european_rko legs on flat ATM vol.

Two invariants are pinned here:

  1. The flat path is byte-for-byte identical to legacy. ``surface=None`` and an
     explicit ``FlatSurface(vol)`` must reproduce the scalar-vol numbers exactly,
     for both atmfsratio and scenario MtM.

  2. A real smile actually moves vanilla pricing. USDTRY has a strong topside
     call skew, so the smile must shift atmfsratio and the vanilla scenario MtM
     rows, while leaving the digital scenario rows untouched.
"""

from __future__ import annotations

import math

import pytest

from analytics.market_state import compute_market_state
from analytics.scenario_generator import generate_scenarios
from analytics.scenario_pricer import price_scenarios
from analytics.structure_pricer import price_variants
from analytics.vol_surface import (
    FlatSurface,
    SmileInterpolator,
    VolSurface,
    build_vol_surface,
    smile_skew_spread,
)
from data.snapshot_loader import load_snapshot


_TARGET = 55.0
_VOL = 0.2121
_R_F = 0.0452
_T = 91 / 365.0
_FWD = 46.7597


@pytest.fixture(scope="module")
def usdtry_ccy():
    return load_snapshot().get("USDTRY")


@pytest.fixture(scope="module")
def smile(usdtry_ccy):
    return SmileInterpolator(usdtry_ccy)


@pytest.fixture(scope="module")
def trade_inputs(usdtry_ccy):
    spot = usdtry_ccy.spot
    r_d = _R_F + math.log(_FWD / spot) / _T
    return {
        "spot": spot,
        "forward": _FWD,
        "implied_vol": _VOL,
        "tenor_years": _T,
        "target": _TARGET,
        "r_d": r_d,
        "r_f": _R_F,
    }


def _market_state(spot, surface=None):
    r_d = _R_F + math.log(_FWD / spot) / _T
    return compute_market_state(
        spot, _FWD, _VOL, _T, r_d, _R_F,
        target=_TARGET, direction="base_higher", surface=surface,
    )


# ---------------------------------------------------------------------------
# VolSurface abstraction
# ---------------------------------------------------------------------------

def test_implementations_satisfy_protocol(usdtry_ccy):
    assert isinstance(FlatSurface(0.2), VolSurface)
    assert isinstance(SmileInterpolator(usdtry_ccy), VolSurface)


def test_build_vol_surface_returns_surface(usdtry_ccy):
    surface = build_vol_surface(usdtry_ccy)
    assert isinstance(surface, VolSurface)


def test_build_vol_surface_rejects_unknown_method(usdtry_ccy):
    with pytest.raises(ValueError):
        build_vol_surface(usdtry_ccy, method="not_a_method")


def test_flat_surface_returns_constant():
    fs = FlatSurface(0.18)
    assert fs.vol_at_strike(40.0, 46.0, 91) == 0.18
    assert fs.vol_at_delta(0.25, 91) == 0.18
    assert smile_skew_spread(fs, 40.0, 46.0, 91) == 0.0
    assert smile_skew_spread(None, 40.0, 46.0, 91) == 0.0


# ---------------------------------------------------------------------------
# atmfsratio
# ---------------------------------------------------------------------------

def test_atmfsratio_flat_matches_none(usdtry_ccy):
    """surface=None and FlatSurface(vol) give the identical atmfsratio."""
    spot = usdtry_ccy.spot
    ms_none = _market_state(spot, surface=None)
    ms_flat = _market_state(spot, surface=FlatSurface(_VOL))
    assert ms_flat.atmfsratio == pytest.approx(ms_none.atmfsratio, rel=1e-12)


def test_atmfsratio_smile_differs(usdtry_ccy, smile):
    """A real topside-skew smile moves atmfsratio off the flat value."""
    spot = usdtry_ccy.spot
    ms_flat = _market_state(spot, surface=None)
    ms_smile = _market_state(spot, surface=smile)
    assert ms_smile.atmfsratio is not None
    assert ms_smile.atmfsratio == pytest.approx(ms_smile.atmfsratio)  # finite
    assert ms_smile.atmfsratio != pytest.approx(ms_flat.atmfsratio, rel=1e-6)


def test_market_state_carries_surface(usdtry_ccy, smile):
    ms = _market_state(usdtry_ccy.spot, surface=smile)
    assert ms.surface is smile
    assert _market_state(usdtry_ccy.spot, surface=None).surface is None


# ---------------------------------------------------------------------------
# Scenario MtM
# ---------------------------------------------------------------------------

def _vanilla_variant(ms):
    return price_variants(ms, "vanilla", target=_TARGET, is_call=True)[0]


def _scenario_rows(variant, structure_id, trade_inputs, surface):
    scenarios = generate_scenarios(trade_inputs)
    return price_scenarios(variant, structure_id, scenarios, trade_inputs, True, surface=surface)


def test_scenario_flat_matches_none(usdtry_ccy, trade_inputs):
    """surface=None and FlatSurface reproduce identical scenario MtM rows."""
    ms = _market_state(usdtry_ccy.spot, surface=None)
    pv = _vanilla_variant(ms)
    none_rows = _scenario_rows(pv, "vanilla", trade_inputs, None)
    flat_rows = _scenario_rows(pv, "vanilla", trade_inputs, FlatSurface(_VOL))
    assert len(none_rows) == len(flat_rows)
    for a, b in zip(none_rows, flat_rows):
        assert a["price_pct"] == pytest.approx(b["price_pct"], rel=1e-12, abs=1e-15)


def test_scenario_smile_moves_vanilla(usdtry_ccy, trade_inputs, smile):
    """A real smile shifts at least one non-expiry vanilla scenario row."""
    ms = _market_state(usdtry_ccy.spot, surface=smile)
    pv = _vanilla_variant(ms)
    flat_rows = _scenario_rows(pv, "vanilla", trade_inputs, None)
    smile_rows = _scenario_rows(pv, "vanilla", trade_inputs, smile)
    diffs = [
        abs(a["price_pct"] - b["price_pct"])
        for a, b in zip(flat_rows, smile_rows)
        if a["remaining_time"] > 0
    ]
    assert any(d > 1e-9 for d in diffs), "smile did not move any vanilla scenario row"


def test_scenario_smile_leaves_digital_flat(usdtry_ccy, trade_inputs, smile):
    """Digital legs stay on flat vol — smile must not change their MtM rows."""
    ms = _market_state(usdtry_ccy.spot, surface=smile)
    pvs = price_variants(ms, "european_digital", target=_TARGET, is_call=True, smile=smile)
    if not pvs:
        pytest.skip("no european_digital variant priced")
    pv = pvs[0]
    flat_rows = _scenario_rows(pv, "european_digital", trade_inputs, None)
    smile_rows = _scenario_rows(pv, "european_digital", trade_inputs, smile)
    for a, b in zip(flat_rows, smile_rows):
        assert a["price_pct"] == pytest.approx(b["price_pct"], rel=1e-12, abs=1e-15)


# ---------------------------------------------------------------------------
# European-package vol-provider seam (step 1): european_rko entry pricing.
# The two vanilla legs now price at their own strike vol; the digital strip is
# still level-only (slope correction lands in step 2). Flat must stay identical.
# ---------------------------------------------------------------------------

def test_european_rko_entry_flat_matches_none(usdtry_ccy):
    """surface=None and FlatSurface(vol) give identical ERKO entry premiums."""
    spot = usdtry_ccy.spot
    ms_none = _market_state(spot, surface=None)
    ms_flat = _market_state(spot, surface=FlatSurface(_VOL))
    none_pvs = price_variants(ms_none, "european_rko", target=_TARGET, is_call=True)
    flat_pvs = price_variants(ms_flat, "european_rko", target=_TARGET, is_call=True, smile=FlatSurface(_VOL))
    assert none_pvs and len(none_pvs) == len(flat_pvs)
    for a, b in zip(none_pvs, flat_pvs):
        assert a.variant_label == b.variant_label
        assert a.strikes == pytest.approx(b.strikes, rel=1e-12, abs=1e-15)
        assert a.net_premium_pct == pytest.approx(b.net_premium_pct, rel=1e-12, abs=1e-15)


def test_european_rko_entry_smile_moves_premium(usdtry_ccy, smile):
    """A real topside-skew smile shifts ERKO entry premium off the flat value."""
    spot = usdtry_ccy.spot
    flat_pvs = price_variants(_market_state(spot, surface=None), "european_rko", target=_TARGET, is_call=True)
    smile_pvs = price_variants(_market_state(spot, surface=smile), "european_rko", target=_TARGET, is_call=True, smile=smile)
    assert flat_pvs and len(flat_pvs) == len(smile_pvs)
    diffs = [abs(a.net_premium_pct - b.net_premium_pct) for a, b in zip(flat_pvs, smile_pvs)]
    assert any(d > 1e-9 for d in diffs), "smile did not move any european_rko entry premium"


def test_european_digital_entry_flat_matches_none(usdtry_ccy):
    """surface=None and FlatSurface(vol) give identical european_digital entries."""
    spot = usdtry_ccy.spot
    none_pvs = price_variants(_market_state(spot, surface=None), "european_digital", target=_TARGET, is_call=True)
    flat_pvs = price_variants(
        _market_state(spot, surface=FlatSurface(_VOL)), "european_digital",
        target=_TARGET, is_call=True, smile=FlatSurface(_VOL),
    )
    assert none_pvs and len(none_pvs) == len(flat_pvs)
    for a, b in zip(none_pvs, flat_pvs):
        assert a.strikes == pytest.approx(b.strikes, rel=1e-12, abs=1e-15)
        assert a.net_premium_pct == pytest.approx(b.net_premium_pct, rel=1e-12, abs=1e-15)


def test_european_digital_entry_smile_moves_strike(usdtry_ccy, smile):
    """A real smile (level + skew-slope) shifts the solved digital strike/premium."""
    spot = usdtry_ccy.spot
    flat_pvs = price_variants(_market_state(spot, surface=None), "european_digital", target=_TARGET, is_call=True)
    smile_pvs = price_variants(_market_state(spot, surface=smile), "european_digital", target=_TARGET, is_call=True, smile=smile)
    assert flat_pvs and len(flat_pvs) == len(smile_pvs)
    diffs = [abs(a.strikes[0] - b.strikes[0]) for a, b in zip(flat_pvs, smile_pvs)]
    assert any(d > 1e-6 for d in diffs), "smile did not move any european_digital strike"


class _SteepSkewSurface:
    """Duck-typed VolSurface with a skew steep enough to manufacture a butterfly
    arbitrage (negative density) above the forward — used to exercise the drop path."""

    def vol_at_strike(self, K, F, horizon_days):
        return max(_VOL + 2.0 * (K - F), 0.01)

    def vol_at_delta(self, delta, horizon_days):
        return _VOL


def test_smile_arbitrage_drops_european_digital_with_warning(usdtry_ccy):
    """A steep-skew surface drops the un-priceable digital variants and records a note."""
    ms = _market_state(usdtry_ccy.spot, surface=None)  # surface passed to price_variants below
    warns: list[str] = []
    pvs = price_variants(
        ms, "european_digital", target=_TARGET, is_call=True,
        smile=_SteepSkewSurface(), warnings=warns,
    )
    assert warns, "steep-skew smile should drop at least one digital variant with a note"
    assert len(pvs) < 3, "arbitrage variants should be omitted from the result"


def test_smile_arbitrage_european_rko_propagates(usdtry_ccy):
    """The ERKO digital strip inherits the guard — steep skew drops ERKO variants too."""
    ms = _market_state(usdtry_ccy.spot, surface=None)
    warns: list[str] = []
    price_variants(
        ms, "european_rko", target=_TARGET, is_call=True,
        smile=_SteepSkewSurface(), warnings=warns,
    )
    assert warns, "steep-skew smile should drop at least one european_rko variant"
