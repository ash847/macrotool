"""Sizing-regime awareness: per-structure Kelly f* on the variant, and the agent pack
stating (and locking to) the active regime with per-structure f* under Kelly."""

from __future__ import annotations

import pytest

from agentic.render import render_pack
from agentic.standard_pack import build_pack
from analytics.distributions import interpolate_atm_vol
from tests._curves import stated_lognormal
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
    probs, bins = stated_lognormal(rc.forward * 1.04, vol, 90 / 365)   # PM-stated curve
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
    assert "capital at risk" in txt         # headline Kelly figure (fraction of W)
    assert "f* =" in txt                    # notional multiple, as context


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


def test_kelly_without_stated_curve_sizes_on_market_distribution(ctx):
    # Kelly with no stated curve → the MARKET distribution (no edge), stated as such —
    # never a synthesised edge, and never a silent switch to fixed-loss.
    snap, cfg, view, ccy, _, _ = ctx
    pack = build_pack(view, ccy, cfg, linear_notional=100_000_000, sizing_method="kelly")
    assert pack.sizing_method == "kelly"
    assert pack.kelly_distribution_source == "market"
    txt = render_pack(pack, view)
    assert "SIZING REGIME: KELLY" in txt and "DISTRIBUTION: MARKET" in txt
    assert "FIXED-LOSS" not in txt


def test_stated_curve_is_labelled_explicit(ctx):
    snap, cfg, view, ccy, probs, bins = ctx
    pack = build_pack(view, ccy, cfg, linear_notional=100_000_000, sizing_method="kelly",
                      kelly_probs=probs, kelly_bins=bins)
    assert pack.kelly_distribution_source == "explicit"
    assert "DISTRIBUTION: MARKET" not in render_pack(pack, view)


def _agent_session(snap, cfg, probs, bins, key):
    from agentic.session import AgentSession
    return AgentSession(snapshot=snap, cfg=cfg, linear_notional=100_000_000,
                        sizing_method="kelly", kelly_probs=probs, kelly_bins=bins,
                        kelly_curve_key=key)


def test_agent_passes_the_sizing_regime_to_the_engine(ctx):
    # Regression: run_standard_pack used to drop sizing_method → always fixed-loss.
    from agentic.tools import dispatch
    snap, cfg, _, _, probs, bins = ctx
    s = _agent_session(snap, cfg, probs, bins, ("USDBRL", 90))
    out, err = dispatch(s, "run_standard_pack",
                        {"pair": "USDBRL", "horizon_days": 90, "direction": "base_higher",
                         "magnitude_pct": 6.0})
    assert not err
    assert s.pack.sizing_method == "kelly" and s.pack.kelly_distribution_source == "explicit"


def test_agent_never_uses_a_curve_stated_for_another_trade(ctx):
    from agentic.tools import dispatch
    snap, cfg, _, _, probs, bins = ctx
    for other_key, args in [
        (("USDBRL", 90), {"pair": "USDJPY", "horizon_days": 90, "direction": "base_higher",
                          "magnitude_pct": 3.0}),                       # other pair
        (("USDBRL", 60), {"pair": "USDBRL", "horizon_days": 90, "direction": "base_higher",
                          "magnitude_pct": 6.0}),                       # other horizon
        (None, {"pair": "USDBRL", "horizon_days": 90, "direction": "base_higher",
                "magnitude_pct": 6.0}),                                 # untagged curve
    ]:
        s = _agent_session(snap, cfg, probs, bins, other_key)
        out, err = dispatch(s, "run_standard_pack", args)
        assert not err
        assert s.pack.kelly_distribution_source == "market", other_key
        assert "DISTRIBUTION: MARKET" in out
