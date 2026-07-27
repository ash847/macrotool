"""Sizing-regime awareness: per-structure Kelly f* on the variant, and the agent pack
stating (and locking to) the active regime with per-structure f* under Kelly."""

from __future__ import annotations

import pytest

from agentic.render import render_pack
from agentic.standard_pack import build_pack
from analytics.distributions import interpolate_atm_vol
from analytics.sizing import view_implied_distribution
from config.loader import load_config
from data.snapshot_loader import load_snapshot
from knowledge_engine.models import TradeView
from pricing.forwards import rate_context_for_snapshot


@pytest.fixture(scope="module")
def ctx():
    snap = load_snapshot()
    cfg = load_config()
    view = TradeView(pair="USDBRL", direction="base_higher", direction_conviction="medium",
                     horizon_days=90, magnitude_pct=6.0, mode="recommend")
    ccy = snap.get("USDBRL")
    rc = rate_context_for_snapshot(ccy, 90 / 365)
    vol = interpolate_atm_vol(ccy, 90)
    probs, bins = view_implied_distribution(
        spot=ccy.spot, fwd=rc.forward, vol=vol, T=90 / 365,
        target=rc.forward * 1.06, conviction="medium",
    )
    return snap, cfg, view, ccy, probs, bins


def test_fixed_loss_pack_states_fixed_regime_and_no_kelly(ctx):
    snap, cfg, view, ccy, _, _ = ctx
    pack = build_pack(view, ccy, cfg, linear_notional=100_000_000)
    assert pack.sizing_method == "fixed_loss"
    assert pack.recommended[0].variant.kelly_fraction is None
    txt = render_pack(pack, view)
    assert "SIZING REGIME: FIXED-LOSS" in txt
    assert "Kelly f*" not in txt


def test_kelly_pack_states_kelly_regime_with_per_structure_fstar(ctx):
    snap, cfg, view, ccy, probs, bins = ctx
    pack = build_pack(view, ccy, cfg, linear_notional=100_000_000,
                      sizing_method="kelly", kelly_lambda=0.5,
                      kelly_probs=probs, kelly_bins=bins)
    assert pack.sizing_method == "kelly"
    top = pack.recommended[0].variant
    assert top.kelly_fraction is not None and top.kelly_fraction >= 0.0
    txt = render_pack(pack, view)
    assert "SIZING REGIME: KELLY" in txt
    assert "FIXED-LOSS" not in txt          # the other regime is never mentioned
    assert "Kelly f* =" in txt


def test_kelly_notional_is_lambda_times_fstar_times_w(ctx):
    snap, cfg, view, ccy, probs, bins = ctx
    W, lam = 100_000_000.0, 0.5
    pack = build_pack(view, ccy, cfg, linear_notional=W,
                      sizing_method="kelly", kelly_lambda=lam,
                      kelly_probs=probs, kelly_bins=bins)
    for r in pack.recommended:
        pv = r.variant
        if pv.kelly_fraction is None or pv.structure_notional is None:
            continue
        cap = 10.0 * W
        expected = min(lam * pv.kelly_fraction * W, cap)
        assert pv.structure_notional == pytest.approx(expected, rel=1e-9)


def test_missing_distribution_falls_back_to_fixed_loss(ctx):
    # sizing_method="kelly" but no distribution → fixed-loss (no crash, no bogus f*).
    snap, cfg, view, ccy, _, _ = ctx
    pack = build_pack(view, ccy, cfg, linear_notional=100_000_000, sizing_method="kelly")
    assert pack.sizing_method == "fixed_loss"
    assert "SIZING REGIME: FIXED-LOSS" in render_pack(pack, view)
