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


def test_kelly_without_distribution_falls_back_to_fixed_loss_and_flags_it(ctx):
    # Kelly selected but no distribution for this trade → sized FIXED-LOSS, and the
    # pack tells the model to say so and ask the PM to set one up. Never a synthesised
    # (conviction/market) edge.
    snap, cfg, view, ccy, _, _ = ctx
    pack = build_pack(view, ccy, cfg, linear_notional=100_000_000, sizing_method="kelly")
    assert pack.sizing_method == "fixed_loss" and pack.kelly_fallback is True
    assert pack.recommended[0].variant.kelly_fraction is None
    txt = render_pack(pack, view)
    assert "the PM selected KELLY, but has NOT set up a distribution" in txt
    assert pack.expiry is not None and pack.expiry.strftime("%d-%b-%y") in txt
    assert "Kelly f*" not in txt


def test_stated_distribution_sizes_kelly(ctx):
    snap, cfg, view, ccy, probs, bins = ctx
    pack = build_pack(view, ccy, cfg, linear_notional=100_000_000, sizing_method="kelly",
                      kelly_probs=probs, kelly_bins=bins)
    assert pack.sizing_method == "kelly" and pack.kelly_fallback is False
    assert pack.sizing_spec is not None and pack.sizing_spec.method == "kelly"
    assert "has NOT set up a distribution" not in render_pack(pack, view)


def _agent_session(snap, cfg, probs, bins, key):
    from agentic.session import AgentSession
    return AgentSession(snapshot=snap, cfg=cfg, linear_notional=100_000_000,
                        sizing_method="kelly",
                        kelly_curves={key: (probs, bins)} if key else {})


def _key(snap, pair, days):
    from analytics.sizing import curve_key, expiry_for
    return curve_key(pair, expiry_for(snap.snapshot_date, days))


def test_agent_passes_the_sizing_regime_to_the_engine(ctx):
    # Regression: run_standard_pack used to drop sizing_method → always fixed-loss.
    from agentic.tools import dispatch
    snap, cfg, _, _, probs, bins = ctx
    s = _agent_session(snap, cfg, probs, bins, _key(snap, "USDBRL", 90))
    out, err = dispatch(s, "run_standard_pack",
                        {"pair": "USDBRL", "horizon_days": 90, "direction": "base_higher",
                         "magnitude_pct": 6.0})
    assert not err
    assert s.pack.sizing_method == "kelly" and not s.pack.kelly_fallback


def test_agent_never_uses_a_distribution_stated_for_another_trade(ctx):
    from agentic.tools import dispatch
    snap, cfg, _, _, probs, bins = ctx
    brl = {"pair": "USDBRL", "horizon_days": 90, "direction": "base_higher", "magnitude_pct": 6.0}
    for stated_key, args in [
        (_key(snap, "USDBRL", 90), {"pair": "USDJPY", "horizon_days": 90,
                                    "direction": "base_higher", "magnitude_pct": 3.0}),
        (_key(snap, "USDBRL", 60), brl),                 # same pair, other expiry
        (None, brl),                                     # nothing stated
    ]:
        s = _agent_session(snap, cfg, probs, bins, stated_key)
        out, err = dispatch(s, "run_standard_pack", args)
        assert not err
        assert s.pack.sizing_method == "fixed_loss" and s.pack.kelly_fallback, stated_key
        assert "has NOT set up a distribution" in out


def test_distribution_key_is_the_expiry_date_not_the_horizon(ctx):
    # A saved view reopened later is a shorter-horizon trade to the SAME expiry: the
    # PM's distribution must still apply.
    from analytics.sizing import curve_for_trade, curve_key, expiry_for
    from datetime import timedelta
    snap, *_ , probs, bins = ctx
    exp = expiry_for(snap.snapshot_date, 90)
    curves = {curve_key("USDBRL", exp): (probs, bins)}
    later = snap.snapshot_date + timedelta(days=20)
    assert curve_for_trade(curves, "USDBRL", expiry_for(later, 70)) is not None
    assert curve_for_trade(curves, "USDBRL", expiry_for(later, 90)) is None


def test_price_structure_uses_the_packs_kelly_sizing(ctx):
    # Tier-2: a PM-named structure is sized under the same regime as the pack.
    from agentic.tools import dispatch
    snap, cfg, _, _, probs, bins = ctx
    s = _agent_session(snap, cfg, probs, bins, _key(snap, "USDBRL", 90))
    dispatch(s, "run_standard_pack", {"pair": "USDBRL", "horizon_days": 90,
                                      "direction": "base_higher", "magnitude_pct": 6.0})
    out, err = dispatch(s, "price_structure", {"request": "25Δ vanilla"})
    assert not err
    assert s.priced and s.priced[-1].variant.kelly_fraction is not None
