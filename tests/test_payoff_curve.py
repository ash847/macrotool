"""Payoff-curve data layer + figure: golden values, engine-wide sweep, failure injection.

Three layers of protection, matching the "a missing chart beats a wrong one" contract:
  1. Golden — hand-computed expiry P&L for every enabled family on synthetic variants.
  2. Sweep  — every recommended structure the real engine produces, across every snapshot
              pair × both directions × two tenors, must yield a curve that passes its own
              hard checks (this is what catches edge cases nobody thought of).
  3. Gate   — corrupt an engine number / a leg and prove the curve is withheld.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from analytics.payoff_curve import PayoffCurve, build_payoff_curve, curves_for_pack
from analytics.product_model import (
    Anchor, AnchorKind, Instrument, Leg, PricedLeg, PricedStructure, Right, Structure,
)
from interface.payoff_chart import payoff_figure

SPOT, FWD, VOL, T = 100.0, 101.0, 0.10, 0.25


# ---------------------------------------------------------------------------
# helpers — synthetic variants shaped like PricedVariant / PricedStructure
# ---------------------------------------------------------------------------

def _variant(**kw):
    base = dict(
        variant_label="test", strikes=[], barrier=None, net_premium_pct=0.0, breakeven=None,
        payoff_at_target_pct=None, rr_at_target=None, max_loss_pct=0.0, wing_ratio=None,
        is_zero_cost=False, structure_notional=1000.0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _legs(spec, family="x"):
    """spec: [(signed_notional, strike, 'call'|'put')] → a real PricedStructure."""
    plegs = []
    for n, k, right in spec:
        leg = Leg(Instrument.VANILLA, Right(right), n, Anchor(AnchorKind.STRIKE, k))
        plegs.append(PricedLeg(leg=leg, strike=k, vol=VOL, unit_premium=0.0))
    return PricedStructure(
        structure=Structure(family, tuple(p.leg for p in plegs)), priced_legs=plegs,
        net_premium_pct=0.0, payoff_at_target_pct=None, rr_at_target=None,
        max_loss_pct=0.0, breakeven=None, is_zero_cost=False,
    )


def _curve(structure_id, variant, ps=None, *, is_call=True, target=None):
    return build_payoff_curve(
        structure_id, variant, ps, pair="USDXXX", spot=SPOT, fwd=FWD, vol=VOL, T=T,
        is_call=is_call, target=target,
    )


def _at(curve: PayoffCurve, s: float) -> float:
    return float(np.interp(s, curve.spots, curve.pnl_pct))


# ---------------------------------------------------------------------------
# 1 — golden values
# ---------------------------------------------------------------------------

def test_vanilla_call_golden():
    prem = 0.02
    v = _variant(strikes=[105.0], net_premium_pct=prem, max_loss_pct=prem,
                 breakeven=105 + prem * SPOT)
    c = _curve("vanilla", v, _legs([(1.0, 105.0, "call")]))
    assert c.ok, c.failures
    assert _at(c, 90.0) == pytest.approx(-prem)                       # OTM: lose the premium
    assert _at(c, 110.0) == pytest.approx(5.0 / 110.0 - prem, abs=1e-6)  # payoff ÷ terminal spot
    assert len(c.breakevens) == 1 and 105 < c.breakevens[0] < 108


def test_vanilla_put_golden():
    prem = 0.015
    v = _variant(strikes=[95.0], net_premium_pct=prem, max_loss_pct=prem)
    c = _curve("vanilla", v, _legs([(1.0, 95.0, "put")]), is_call=False)
    assert c.ok, c.failures
    assert _at(c, 110.0) == pytest.approx(-prem)
    assert _at(c, 90.0) == pytest.approx(5.0 / 90.0 - prem, abs=1e-6)


def test_call_spread_golden_capped_both_sides():
    prem = 0.012
    v = _variant(strikes=[102.0, 108.0], net_premium_pct=prem, max_loss_pct=prem)
    c = _curve("1x1_spread", v, _legs([(1.0, 102.0, "call"), (-1.0, 108.0, "call")]))
    assert c.ok, c.failures
    assert _at(c, 114.0) == pytest.approx(6.0 / 114.0 - prem, abs=1e-6)   # capped payoff
    assert _at(c, 95.0) == pytest.approx(-prem)


def test_1x2_golden_open_tail_is_flagged_not_hidden():
    prem = -0.001                                                      # tiny credit
    v = _variant(strikes=[103.0, 107.0], net_premium_pct=prem, max_loss_pct=abs(prem))
    c = _curve("1x2_spread", v, _legs([(1.0, 103.0, "call"), (-2.0, 107.0, "call")]))
    assert c.ok, c.failures                                            # chart still drawn
    assert _at(c, 107.0) == pytest.approx(4.0 / 107.0 - prem, abs=1e-6)  # peak at K2
    assert _at(c, 115.0) < -0.03                                       # loss far past the engine's "max loss"
    assert any(n.name == "max_loss_open_tail" for n in c.notes)        # …and the disagreement is surfaced


def test_1x2x1_golden_bounded_by_premium():
    prem = 0.003
    v = _variant(strikes=[103.0, 107.0, 111.0], net_premium_pct=prem, max_loss_pct=prem)
    c = _curve("1x2x1_spread", v,
               _legs([(1.0, 103.0, "call"), (-2.0, 107.0, "call"), (1.0, 111.0, "call")]))
    assert c.ok, c.failures
    assert float(c.pnl_pct.min()) == pytest.approx(-prem, abs=1e-6)    # max loss = premium
    assert not c.notes or all(n.name != "max_loss_open_tail" for n in c.notes)


def test_seagull_uses_priced_wing_not_rounded_ratio():
    exact = 1.6842279
    v = _variant(strikes=[102.0, 106.0, 96.0], net_premium_pct=0.0, wing_ratio=1.68,
                 max_loss_pct=0.01, is_zero_cost=True)
    ps = _legs([(1.0, 102.0, "call"), (-1.0, 106.0, "call"), (-exact, 96.0, "put")])
    c = _curve("seagull", v, ps)
    assert c.ok, c.failures                                            # routes agree on the *priced* wing
    assert _at(c, 90.0) == pytest.approx(-exact * 6.0 / 90.0, abs=1e-6)
    assert any(n.name == "seagull_wing_rounded" for n in c.notes)


def test_european_digital_golden_base_ccy_cash_or_nothing():
    prem = 0.30
    v = _variant(strikes=[106.0], net_premium_pct=prem, max_loss_pct=prem,
                 payoff_at_target_pct=1.0, breakeven=106.0)
    c = _curve("european_digital", v, None, target=108.0)
    assert c.ok, c.failures
    assert _at(c, 100.0) == pytest.approx(-prem)
    assert _at(c, 112.0) == pytest.approx(1.0 - prem)                  # fixed 100% base-ccy payout
    assert c.breakevens == ()                                          # a digital jumps; it never "crosses"


def test_european_rko_golden_knocks_out_beyond_barrier():
    prem = 0.0014
    v = _variant(strikes=[105.0], barrier=108.0, net_premium_pct=prem, max_loss_pct=prem)
    c = _curve("european_rko", v, None)
    assert c.ok, c.failures
    assert _at(c, 107.0) == pytest.approx(2.0 / 107.0 - prem, abs=1e-6)
    assert _at(c, 109.0) == pytest.approx(-prem)                       # knocked out at expiry


# ---------------------------------------------------------------------------
# 3 — the gate: corrupt something, prove the chart is withheld
# ---------------------------------------------------------------------------

def test_gate_wrong_max_loss_on_bounded_family_withholds_chart():
    v = _variant(strikes=[105.0], net_premium_pct=0.02, max_loss_pct=0.05)   # engine says 5%, truth is 2%
    c = _curve("vanilla", v, _legs([(1.0, 105.0, "call")]))
    assert not c.ok and any(f.name == "max_loss_bounded" for f in c.failures)
    assert payoff_figure(c) is None


def test_gate_leg_strike_disagreeing_with_strikes_withholds_chart():
    v = _variant(strikes=[105.0], net_premium_pct=0.02, max_loss_pct=0.02)
    c = _curve("vanilla", v, _legs([(1.0, 106.0, "call")]))                  # legs say 106, strikes say 105
    assert not c.ok and any(f.name == "routes_agree" for f in c.failures)
    assert payoff_figure(c) is None


def test_gate_digital_payoff_at_target_mismatch_withholds_chart():
    v = _variant(strikes=[106.0], net_premium_pct=0.30, max_loss_pct=0.30,
                 payoff_at_target_pct=1.10)                                  # the legacy spot/target basis
    c = _curve("european_digital", v, None, target=108.0)
    assert not c.ok and any(f.name == "payoff_at_target" for f in c.failures)


def test_gate_nan_premium_withholds_chart():
    v = _variant(strikes=[105.0], net_premium_pct=float("nan"), max_loss_pct=0.02)
    c = _curve("vanilla", v, _legs([(1.0, 105.0, "call")]))
    assert not c.ok and any(f.name == "finite" for f in c.failures)


def test_no_strikes_means_no_curve():
    assert _curve("vanilla", _variant(strikes=[]), None) is None


# ---------------------------------------------------------------------------
# figure + units
# ---------------------------------------------------------------------------

def test_figure_units_are_a_pure_rescale_of_the_same_curve():
    prem = 0.02
    v = _variant(strikes=[105.0], net_premium_pct=prem, max_loss_pct=prem, structure_notional=1000.0)
    c = _curve("vanilla", v, _legs([(1.0, 105.0, "call")]))
    line = lambda f: next(t for t in f.data if t.name == "P&L at expiry")
    pct, ccy = line(payoff_figure(c, "pct")), line(payoff_figure(c, "notional"))
    np.testing.assert_allclose(np.asarray(ccy.y), np.asarray(pct.y) * 10.0)   # 1000 notional ÷ 100 (pct → fraction)
    np.testing.assert_allclose(np.asarray(pct.y), c.pnl_pct * 100.0)
    np.testing.assert_allclose(np.asarray(pct.x), c.spots)


def test_figure_falls_back_to_pct_without_a_sized_notional():
    v = _variant(strikes=[105.0], net_premium_pct=0.02, max_loss_pct=0.02, structure_notional=None)
    c = _curve("vanilla", v, _legs([(1.0, 105.0, "call")]))
    fig = payoff_figure(c, "notional")
    assert fig is not None and "% of" in fig.layout.yaxis.title.text


def test_figure_marks_spot_forward_target_strikes_and_breakevens():
    prem = 0.02
    v = _variant(strikes=[105.0], net_premium_pct=prem, max_loss_pct=prem)
    c = _curve("vanilla", v, _legs([(1.0, 105.0, "call")]), target=108.0)
    fig = payoff_figure(c)
    labels = " ".join(a.text for a in fig.layout.annotations)
    for name in ("Spot", "Fwd", "Target", "K "):
        assert name in labels
    assert any(t.name == "Breakeven" for t in fig.data)


# ---------------------------------------------------------------------------
# 2 — engine-wide sweep (real build_pack; no LLM, no network)
# ---------------------------------------------------------------------------

def _sweep_cases():
    from data.snapshot_loader import load_snapshot
    pairs = list(load_snapshot().currencies.keys())
    return [(p, d, h) for p in pairs for d in ("base_higher", "base_lower") for h in (30, 91)]


@pytest.fixture(scope="module")
def _snap_cfg():
    from config.loader import load_config
    from data.snapshot_loader import load_snapshot
    return load_snapshot(), load_config()


def _pack(snap_cfg, pair, direction, horizon, magnitude=3.0):
    from agentic.standard_pack import build_pack
    from knowledge_engine.models import TradeView
    snap, cfg = snap_cfg
    view = TradeView(pair=pair, direction=direction, direction_conviction="high",
                     horizon_days=horizon, magnitude_pct=magnitude, mode="recommend")
    return build_pack(view, snap.get(pair), cfg)


@pytest.mark.parametrize("pair,direction,horizon", _sweep_cases())
def test_every_recommended_structure_gets_a_reconciled_curve(_snap_cfg, pair, direction, horizon):
    pack = _pack(_snap_cfg, pair, direction, horizon)
    curves = curves_for_pack(pack, pair)
    assert len(curves) == len(pack.recommended)
    for rec, c in zip(pack.recommended, curves):
        tag = f"{pair} {direction} {horizon}d #{rec.rank} {rec.structure_id} {rec.variant.variant_label}"
        assert c is not None, f"{tag}: no curve"
        assert c.ok, f"{tag}: " + "; ".join(f"{f.name}: {f.detail}" for f in c.failures)
        assert np.all(np.diff(c.spots) > 0), f"{tag}: spots not strictly increasing"
        assert payoff_figure(c) is not None, tag
        # every reported zero-crossing really is one
        for be in c.breakevens:
            bridge_val = float(np.interp(be, c.spots, c.pnl_pct))
            assert abs(bridge_val) < 1e-5, f"{tag}: breakeven {be} has pnl {bridge_val}"


def test_chart_package_payoff_is_the_full_package_not_the_long_leg(_snap_cfg):
    """GBPUSD 6% / 180d: the 30Δ/10Δ ratio spreads have the target *beyond* K2. The package
    payoff there must include the short leg(s); the engine's headline payoff_at_target
    currently counts the long leg only (a surfaced note, not a chart error)."""
    pack = _pack(_snap_cfg, "GBPUSD", "base_higher", 180, magnitude=6.0)
    curves = curves_for_pack(pack, "GBPUSD")
    t = pack.target
    checked = 0
    for rec, c in zip(pack.recommended, curves):
        if rec.structure_id not in ("1x1.5_spread", "1x2_spread") or len(rec.variant.strikes) < 2:
            continue
        k1, k2 = rec.variant.strikes[:2]
        if t <= k2:
            continue
        ratio = 1.5 if rec.structure_id == "1x1.5_spread" else 2.0
        full = (max(t - k1, 0) - ratio * max(t - k2, 0)) / t
        assert float(np.interp(t, c.spots, c.pnl_pct + c.net_premium_pct)) == pytest.approx(full, abs=1e-6)
        checked += 1
    if checked == 0:
        pytest.skip("no ratio spread with target beyond K2 in this snapshot")
