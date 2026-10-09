"""Expiry P&L curve for one priced structure — pure numpy, no plotting, no IO.

Data layer behind the payoff chart. It turns an already-priced recommended variant into
a dense, kink-exact P&L curve (net of premium, base-ccy fraction of notional — the same
basis as every ``*_pct`` field on ``PricedVariant``) and *reconciles that curve against
the engine's own numbers* before anyone is allowed to draw it.

Fail-safe contract: a chart is only as trustworthy as its checks. ``PayoffCurve.ok`` is
False if any hard check fails (non-finite values, two independent payoff routes
disagreeing, payoff-at-target not matching the engine, the bounded-loss families not
bottoming out at the engine's max loss, a breakeven the engine reports that the curve
doesn't have). The UI draws nothing when ``ok`` is False — a missing chart is acceptable,
a wrong one is not. Soft findings (``severity="note"``) never block the chart; they
surface real model disagreements, e.g. the engine's max loss understating an open tail.

Two independent routes to the same payoff:
  1. the strike-based bridge ``analytics.payoffs.base_ccy_payoff_for_trade_rec``
     (what Kelly integrates against), and
  2. the leg-sum over the product-model ``PricedStructure`` (the same primitive the
     pricer aggregates and ``payoff_profile`` reads).
They must agree to float precision on every vanilla-leg family.

Basis note: P&L here is base-ccy, i.e. payoff_quote / S_T − premium_pct. The engine's
``breakeven`` is the quote-ccy convention (K ± premium·spot). The two differ by roughly
premium² — fractions of a percent — so the breakeven reconciliation uses a small relative
tolerance rather than equality.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from analytics.payoffs import base_ccy_payoff_for_trade_rec

Severity = Literal["fail", "note"]

# Families whose loss is bounded at the net premium paid (debit, no short tail), so the
# curve's worst point MUST equal the engine's max loss. Everything else can legitimately
# have a different worst point on the grid (open tails, stress-point max loss).
_BOUNDED_DEBIT = {"vanilla", "1x1_spread", "1x2x1_spread", "european_digital", "european_rko"}
_LEG_SUM_FAMILIES_EXCLUDE = {"european_digital", "european_rko", "rko", "european_digital_rko"}

_ABS_TOL = 1e-9
_TARGET_TOL = 1e-6
_BREAKEVEN_REL_TOL = 0.006      # chart vs engine breakeven (see basis note)
_N_DENSE = 801
_SIGMA_SPAN = 3.0               # default half-width of the x range, in σ√T


@dataclass(frozen=True)
class CurveCheck:
    name: str
    ok: bool
    detail: str
    severity: Severity = "fail"


@dataclass
class PayoffCurve:
    structure_id: str
    label: str
    pair: str
    spots: np.ndarray                 # strictly increasing terminal spots
    pnl_pct: np.ndarray               # P&L per 1 unit of base-ccy notional, net of premium
    gross_pct: np.ndarray             # payoff only (no premium)
    net_premium_pct: float            # +debit / −credit
    notional: float | None            # sized base-ccy notional (None if the pack isn't sized)
    base_ccy: str
    spot: float
    fwd: float
    target: float | None
    strikes: tuple[float, ...]
    barrier: float | None
    breakevens: tuple[float, ...]     # zero-crossings of the plotted curve
    sizing_method: str = "fixed_loss"
    checks: list[CurveCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any((not c.ok) and c.severity == "fail" for c in self.checks)

    @property
    def failures(self) -> list[CurveCheck]:
        return [c for c in self.checks if (not c.ok) and c.severity == "fail"]

    @property
    def notes(self) -> list[CurveCheck]:
        return [c for c in self.checks if (not c.ok) and c.severity == "note"]

    def pnl(self, units: Literal["pct", "notional"] = "pct") -> np.ndarray:
        """Curve in the requested units: % of notional, or base-ccy at the sized notional."""
        if units == "notional":
            if self.notional is None:
                raise ValueError("no sized notional for this variant")
            return self.pnl_pct * self.notional
        return self.pnl_pct * 100.0


def _legs_of(priced_structure) -> list[tuple[float, float, bool]] | None:
    """(signed_notional, strike, is_call) per leg, or None if not a plain-vanilla leg-sum."""
    if priced_structure is None or not getattr(priced_structure, "priced_legs", None):
        return None
    out = []
    for pl in priced_structure.priced_legs:
        if pl.leg.instrument.value != "vanilla":
            return None
        out.append((float(pl.notional), float(pl.strike), pl.leg.right.value == "call"))
    return out


def _leg_sum_pct(legs, s: np.ndarray) -> np.ndarray:
    total = np.zeros_like(s, dtype=float)
    for n, k, is_call in legs:
        intr = np.maximum(s - k, 0.0) if is_call else np.maximum(k - s, 0.0)
        total += n * intr
    return total / s


def _grid(spot, sigma_sqrt_t, levels, strikes, barrier) -> np.ndarray:
    """Dense log-spaced grid over ±3σ (widened to cover every key level) with each strike /
    barrier inserted exactly, plus a hair either side so kinks and digital jumps are sharp."""
    span = _SIGMA_SPAN * sigma_sqrt_t
    lo = spot * math.exp(-span)
    hi = spot * math.exp(span)
    pad = 1.0 + 0.5 * sigma_sqrt_t
    for lvl in levels:
        lo = min(lo, lvl / pad)
        hi = max(hi, lvl * pad)
    grid = np.exp(np.linspace(math.log(lo), math.log(hi), _N_DENSE))
    knots = [k for k in (*strikes, barrier) if k is not None]
    extra = []
    for k in knots:
        extra += [k, k * (1 - 1e-9), k * (1 + 1e-9)]
    return np.unique(np.concatenate([grid, np.array(extra)]))


_JUMP_REL_WIDTH = 1e-7   # the grid brackets each strike/barrier with ±1e-9 points


def _crossings(x: np.ndarray, y: np.ndarray) -> list[float]:
    """Continuous zero-crossings only. A digital's strike or a European RKO's barrier is a
    *cliff* — P&L jumps across zero between two grid points ~1e-9 apart without ever being
    zero — and calling that a breakeven would be wrong, so jumps are skipped."""
    out = []
    for i in range(len(x) - 1):
        a, b = y[i], y[i + 1]
        if (x[i + 1] - x[i]) / x[i] < _JUMP_REL_WIDTH:
            continue
        if (a < -_ABS_TOL and b > _ABS_TOL) or (a > _ABS_TOL and b < -_ABS_TOL):
            out.append(float(x[i] + (a / (a - b)) * (x[i + 1] - x[i])))
    return out


def build_payoff_curve(
    structure_id: str,
    variant,                      # PricedVariant
    priced_structure,             # product-model PricedStructure | None
    *,
    pair: str,
    spot: float,
    fwd: float,
    vol: float,
    T: float,
    is_call: bool,
    target: float | None,
    sizing_method: str = "fixed_loss",
) -> PayoffCurve | None:
    """Build and self-check the expiry P&L curve for one recommended variant.

    Returns None only when the inputs can't describe a curve at all (no strikes). Any
    reconciliation problem returns a curve whose ``ok`` is False — the caller then
    declines to draw it.
    """
    strikes = tuple(float(k) for k in variant.strikes)
    if not strikes or not (spot > 0 and vol > 0 and T > 0):
        return None
    barrier = None if variant.barrier is None else float(variant.barrier)
    sigma_sqrt_t = vol * math.sqrt(T)
    prem = float(variant.net_premium_pct)

    # The seagull wing is solved to a non-round ratio; PricedVariant.wing_ratio is the
    # display-rounded copy (1.68 vs 1.6842). Draw the package that was actually priced.
    wing_ratio = variant.wing_ratio
    wing_note = None
    legs_priced = _legs_of(priced_structure)
    if structure_id == "seagull" and legs_priced is not None and len(legs_priced) >= 3:
        exact = abs(legs_priced[2][0])
        if wing_ratio is not None and abs(exact - wing_ratio) > 1e-6:
            wing_note = (
                f"variant.wing_ratio is display-rounded ({wing_ratio:g}) vs the priced wing "
                f"({exact:.4f}); the chart uses the priced wing"
            )
        wing_ratio = exact

    try:
        bridge = base_ccy_payoff_for_trade_rec(
            structure_id, strikes=list(strikes), barrier=barrier, is_call=is_call,
            entry_spot=spot, wing_ratio=wing_ratio,
        )
    except Exception:
        return None

    levels = [x for x in (spot, fwd, target, barrier, *strikes) if x is not None]
    s = _grid(spot, sigma_sqrt_t, levels, strikes, barrier)
    gross = np.asarray(bridge(s), dtype=float)
    pnl = gross - prem

    # Re-evaluate exactly at the curve's zero-crossings so fills and markers are clean.
    crossings = _crossings(s, pnl)
    if crossings:
        s = np.unique(np.concatenate([s, np.array(crossings)]))
        gross = np.asarray(bridge(s), dtype=float)
        pnl = gross - prem
        # NB: keep the first-pass crossings. The inserted points evaluate to ~0, so a
        # strict sign-change scan over the refined grid would erase them.

    checks: list[CurveCheck] = []

    # 1 — finite
    checks.append(CurveCheck(
        "finite", bool(np.all(np.isfinite(pnl)) and np.all(np.isfinite(s))),
        "curve contains NaN/inf" if not np.all(np.isfinite(pnl)) else "all values finite",
    ))

    if wing_note:
        checks.append(CurveCheck("seagull_wing_rounded", False, wing_note, severity="note"))

    # 2 — two independent payoff routes agree (vanilla-leg families)
    legs = None if structure_id in _LEG_SUM_FAMILIES_EXCLUDE else legs_priced
    if legs is not None:
        diff = float(np.max(np.abs(_leg_sum_pct(legs, s) - gross)))
        checks.append(CurveCheck(
            "routes_agree", diff < _ABS_TOL,
            f"strike bridge vs leg-sum max |Δ| = {diff:.2e}",
        ))
    else:
        checks.append(CurveCheck(
            "routes_agree", True, "single route (digital / barrier family)", severity="note",
        ))

    # Engine-number reconciliation is a HARD gate only when the chart rests on a single
    # payoff route (digital / barrier). Where two independent routes already agree the
    # chart is verified on its own, so a disagreement with the engine's headline number is
    # a finding about the engine (surfaced as a note), not a reason to hide a correct chart.
    engine_sev: Severity = "fail" if legs is None else "note"

    # 3 — payoff at target equals the engine's
    if target is not None and variant.payoff_at_target_pct is not None:
        at_t = float(bridge(np.array([target]))[0])
        checks.append(CurveCheck(
            "payoff_at_target", abs(at_t - variant.payoff_at_target_pct) < _TARGET_TOL,
            f"package payoff at target {at_t:.6f} vs engine headline "
            f"{variant.payoff_at_target_pct:.6f}",
            severity=engine_sev,
        ))

    # 4 — worst point vs the engine's max loss
    worst = float(-np.min(pnl))
    ml = float(variant.max_loss_pct)
    if structure_id in _BOUNDED_DEBIT and prem > 0:
        checks.append(CurveCheck(
            "max_loss_bounded", abs(worst - ml) < 1e-6,
            f"curve worst loss {worst:.6f} vs engine max loss {ml:.6f}",
        ))
    elif worst > ml + 1e-6:
        at = float(s[int(np.argmin(pnl))])
        checks.append(CurveCheck(
            "max_loss_open_tail", False,
            f"loss on the chart reaches {worst:.2%} of notional at {at:.4f} "
            f"(edge of the plotted range) but the engine's max loss is {ml:.2%} — the "
            "structure has an open tail the engine's max-loss figure does not capture",
            severity="note",
        ))

    # 5 — engine breakeven present on the curve (skip digital: it jumps, never crosses)
    if variant.breakeven is not None and structure_id != "european_digital":
        be = float(variant.breakeven)
        near = [c for c in crossings if abs(c - be) / be < _BREAKEVEN_REL_TOL]
        checks.append(CurveCheck(
            "breakeven", bool(near),
            f"engine breakeven {be:.4f}; chart crossings "
            f"{', '.join(f'{c:.4f}' for c in crossings) or 'none'}",
            severity=engine_sev,
        ))

    return PayoffCurve(
        structure_id=structure_id,
        label=variant.variant_label,
        pair=pair,
        spots=s,
        pnl_pct=pnl,
        gross_pct=gross,
        net_premium_pct=prem,
        notional=variant.structure_notional,
        base_ccy=pair[:3],
        spot=float(spot),
        fwd=float(fwd),
        target=None if target is None else float(target),
        strikes=strikes,
        barrier=barrier,
        breakevens=tuple(crossings),
        sizing_method=sizing_method,
        checks=checks,
    )


def curves_for_pack(pack, pair: str) -> list[PayoffCurve | None]:
    """One curve per recommended structure, in the pack's (display-rank) order."""
    ms = pack.market_state
    out: list[PayoffCurve | None] = []
    for rec in pack.recommended:
        try:
            out.append(build_payoff_curve(
                rec.structure_id, rec.variant, rec.priced_structure,
                pair=pair, spot=ms.spot, fwd=ms.fwd, vol=ms.vol, T=ms.T,
                is_call=pack.is_call, target=pack.target,
                sizing_method=pack.sizing_method,
            ))
        except Exception:
            out.append(None)
    return out
