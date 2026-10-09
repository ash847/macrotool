"""Market-distribution overlay: sweep, properties, fallback ladder, and isolation from the P&L.

The overlay is enrichment. The contract these tests pin down:
  * every pair × tenor in the snapshot yields a reconciled density (smile, or labelled flat);
  * it is a valid density (increasing support, non-negative, right mass) that reproduces the
    engine's own published percentiles exactly and carries the live skew;
  * when it can't be trusted it degrades smile → flat → nothing, never to a wrong picture;
  * drawing it cannot change a single point of the P&L chart.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import analytics.market_density as md
from analytics.distributions import compute_smile_distribution
from analytics.market_density import market_density
from analytics.payoff_curve import build_payoff_curve
from data.snapshot_loader import load_snapshot
from interface.payoff_chart import payoff_figure

_TENORS = (7, 30, 91, 180, 365)


@pytest.fixture(scope="module")
def snap():
    return load_snapshot()


def _cases():
    return [(p, h) for p in load_snapshot().currencies.keys() for h in _TENORS]


def _quantile(d, p: float) -> float:
    """Spot at cumulative probability p (p measured within the grid's 0.0005 … 0.9995 span)."""
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (d.pdf[1:] + d.pdf[:-1]) * np.diff(d.spots))])
    return float(np.interp(p - 0.0005, cum, d.spots))


# ---------------------------------------------------------------------------
# sweep + properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair,horizon", _cases())
def test_every_pair_and_tenor_gets_a_reconciled_density(snap, pair, horizon):
    ccy = snap.get(pair)
    d = market_density(ccy, horizon, compute_smile_distribution(ccy, horizon))
    assert d is not None, f"{pair} {horizon}d: no density at all"
    assert d.ok, [c for c in d.checks if not c.ok]
    assert np.all(np.diff(d.spots) > 0)
    assert np.all(d.pdf >= 0.0) and np.all(np.isfinite(d.pdf))
    assert d.mass == pytest.approx(0.999, abs=1e-3)
    # shape sanity: the density's mode sits inside the central 90% of its own mass
    mode = float(d.spots[int(np.argmax(d.pdf))])
    assert _quantile(d, 0.05) < mode < _quantile(d, 0.95)


def test_reproduces_the_engines_published_percentiles_exactly(snap):
    ccy = snap.get("USDBRL")
    ref = compute_smile_distribution(ccy, 91)
    d = market_density(ccy, 91, ref)
    assert d is not None and d.kind == "smile"
    chk = next(c for c in d.checks if c.name == "matches_published_percentiles")
    assert chk.ok, chk.detail


def test_smile_density_carries_the_live_skew(snap):
    """USDBRL has topside skew: the smile density's right tail is fatter than flat-vol."""
    ccy = snap.get("USDBRL")
    smile = market_density(ccy, 91, compute_smile_distribution(ccy, 91))
    flat = md._build(ccy, 91, "flat", None)
    assert smile.kind == "smile" and flat is not None
    assert _quantile(smile, 0.99) > _quantile(flat, 0.99)


# ---------------------------------------------------------------------------
# fallback ladder: smile → flat → nothing
# ---------------------------------------------------------------------------

def test_non_monotone_smile_mapping_falls_back_to_flat(snap, monkeypatch):
    ccy = snap.get("USDBRL")
    # an absurdly steep vol-vs-percentile ramp makes S(p) fold back on itself
    monkeypatch.setattr(md, "_smile_vol_for_percentile", lambda c, h, a, q: 0.05 + 40.0 * (q - 0.5) ** 2 * (1 if q > 0.5 else -1))
    d = market_density(ccy, 91, None)
    assert d is not None and d.kind == "flat" and d.ok


def test_published_percentile_mismatch_falls_back_to_flat(snap):
    ccy = snap.get("USDBRL")
    ref = compute_smile_distribution(ccy, 91)
    tampered = ref.model_copy(update={"terminal_plus1s": ref.terminal_plus1s * 1.01})
    d = market_density(ccy, 91, tampered)
    assert d is not None and d.kind == "flat"


def test_nan_smile_falls_back_to_flat(snap, monkeypatch):
    ccy = snap.get("USDBRL")
    monkeypatch.setattr(md, "_smile_vol_for_percentile", lambda *a: float("nan"))
    d = market_density(ccy, 91, None)
    assert d is not None and d.kind == "flat"


def test_nothing_reconciles_means_no_overlay(snap, monkeypatch):
    ccy = snap.get("USDBRL")
    monkeypatch.setattr(md, "_interp_vol", lambda *a: 0.0)          # σ = 0 ⇒ S(p) is constant ⇒ not increasing
    monkeypatch.setattr(md, "_smile_vol_for_percentile", lambda *a: 0.0)
    assert market_density(ccy, 91, None) is None


def test_bad_horizon_means_no_overlay(snap):
    assert market_density(snap.get("USDBRL"), 0, None) is None


# ---------------------------------------------------------------------------
# isolation: the overlay cannot touch the P&L
# ---------------------------------------------------------------------------

def _curve():
    v = SimpleNamespace(
        variant_label="t", strikes=[105.0], barrier=None, net_premium_pct=0.02, breakeven=None,
        payoff_at_target_pct=None, rr_at_target=None, max_loss_pct=0.02, wing_ratio=None,
        is_zero_cost=False, structure_notional=1000.0,
    )
    return build_payoff_curve("vanilla", v, None, pair="USDXXX", spot=100.0, fwd=101.0,
                              vol=0.10, T=0.25, is_call=True, target=None)


def _pnl_traces(fig):
    return [t for t in fig.data if t.yaxis != "y2"]


def test_overlay_does_not_change_any_pnl_trace(snap):
    ccy = snap.get("USDBRL")
    d = market_density(ccy, 91, compute_smile_distribution(ccy, 91))
    c = _curve()
    with_d, without = payoff_figure(c, density=d), payoff_figure(c, density=None)
    assert len(_pnl_traces(with_d)) == len(_pnl_traces(without))
    for a, b in zip(_pnl_traces(with_d), _pnl_traces(without)):
        np.testing.assert_array_equal(np.asarray(a.x), np.asarray(b.x))
        np.testing.assert_array_equal(np.asarray(a.y), np.asarray(b.y))
    assert any(t.yaxis == "y2" for t in with_d.data) and not any(t.yaxis == "y2" for t in without.data)


def test_density_axis_is_hidden_fixed_and_scaled_to_a_fraction_of_the_plot(snap):
    ccy = snap.get("USDBRL")
    d = market_density(ccy, 91, compute_smile_distribution(ccy, 91))
    fig = payoff_figure(_curve(), density=d)
    y2 = fig.layout.yaxis2
    assert y2.visible is False and y2.fixedrange is True
    assert y2.range[1] == pytest.approx(float(d.pdf.max()) / 0.30)


def test_failed_density_is_silently_ignored(snap):
    ccy = snap.get("USDBRL")
    d = market_density(ccy, 91, compute_smile_distribution(ccy, 91))
    d.checks.append(md.CurveCheck("forced", False, "forced failure"))
    fig = payoff_figure(_curve(), density=d)
    assert fig is not None and not any(t.yaxis == "y2" for t in fig.data)


def test_failed_pnl_curve_still_returns_no_chart_even_with_a_good_density(snap):
    ccy = snap.get("USDBRL")
    d = market_density(ccy, 91, compute_smile_distribution(ccy, 91))
    c = _curve()
    c.checks.append(md.CurveCheck("forced", False, "forced failure"))
    assert payoff_figure(c, density=d) is None
