"""Market (smile) terminal density for the payoff chart's background overlay.

Built from the engine's *own* definition of the market distribution — the percentile →
smile-vol quantile mapping behind ``analytics.distributions.compute_smile_distribution`` —
evaluated on a fine probability grid instead of the seven published percentile points. The
overlay therefore matches the distribution the rest of the tool already shows, and the
density is simply ``dp/dS`` along a quantile function that is monotone by construction.

Why not Breeden–Litzenberger on the vol surface? It was tried first and is the wrong tool
here: the cubic-spline smile is clamped flat beyond the outermost pillars, and that slope
kink becomes a spike in ∂²C/∂K² — a *negative* spike on the live snapshot, i.e. the surface
genuinely implies a small butterfly arbitrage in its extrapolated wings (the same class of
defect the digital smile-arb guard exists for). A background picture must not inherit that.

Self-checks (any hard failure → fall back to flat-vol lognormal, then to nothing):
  * finite everywhere
  * the quantile function S(p) is strictly increasing (⇒ a valid, non-negative density; the
    percentile→vol mapping can break this under a steep skew)
  * the density integrates to the probability span of the grid (0.999)
  * the grid reproduces the engine's published ±1/2/3σ terminal percentiles
    (``PriceDistribution.terminal_*``) to 1e-9 when a reference distribution is supplied

Fallback ladder: smile → flat-vol lognormal (labelled) → None. The overlay is enrichment;
it must never be the reason a chart is wrong or missing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.stats import norm

from analytics.distributions import (
    _gbm_price,
    _interp_vol,
    _smile_vol_for_percentile,
)
from analytics.models import BAND_PERCENTILES
from analytics.payoff_curve import CurveCheck

_P_LO, _P_HI, _N = 0.0005, 0.9995, 2001
_MASS_TOL = 1e-3
_REF_REL_TOL = 1e-9

# (z, PriceDistribution attribute) for the engine's published terminal points. NB the
# published "±1/2/3σ" bands sit at the *rounded* percentiles in ``BAND_PERCENTILES``
# (e.g. 0.0013 → z = −3.0115, not −3), so the reference z's come from those, not from
# integers. The grid contains each of them exactly, so the comparison is exact.
_REFERENCE_Z = list(zip(
    (float(norm.ppf(q)) for q in BAND_PERCENTILES),
    ("terminal_minus3s", "terminal_minus2s", "terminal_minus1s", "terminal_median",
     "terminal_plus1s", "terminal_plus2s", "terminal_plus3s"),
))


@dataclass
class MarketDensity:
    spots: np.ndarray                 # strictly increasing terminal spots
    pdf: np.ndarray                   # density per unit of spot (≥ 0)
    kind: Literal["smile", "flat"]
    mass: float                       # ∫ pdf dS  (≈ _P_HI − _P_LO)
    checks: list[CurveCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any((not c.ok) and c.severity == "fail" for c in self.checks)

    @property
    def label(self) -> str:
        return ("Market distribution at expiry (smile)" if self.kind == "smile"
                else "Market distribution at expiry (flat vol — smile unavailable)")


def _build(ccy, horizon_days, kind, reference) -> MarketDensity | None:
    from pricing.forwards import interpolate_forward

    T = horizon_days / 365.0
    S0 = ccy.spot
    mu = math.log(interpolate_forward(ccy, T) / S0) / T
    sigma_atm = _interp_vol(ccy, horizon_days, "ATM")

    z_lo, z_hi = norm.ppf(_P_LO), norm.ppf(_P_HI)
    z = np.unique(np.concatenate([np.linspace(z_lo, z_hi, _N), [q for q, _ in _REFERENCE_Z]]))
    # A reference z can land within float epsilon of a grid point; that near-duplicate would
    # read as a flat/decreasing step in S(p). Drop points closer than 1e-9 in z.
    z = z[np.concatenate([[True], np.diff(z) > 1e-9])]
    p = norm.cdf(z)
    if kind == "smile":
        sig = np.array([_smile_vol_for_percentile(ccy, horizon_days, sigma_atm, float(q)) for q in p])
    else:
        sig = np.full_like(p, sigma_atm)
    s = S0 * np.exp((mu - sig * sig / 2.0) * T + sig * math.sqrt(T) * z)

    checks: list[CurveCheck] = []
    finite = bool(np.all(np.isfinite(s)))
    checks.append(CurveCheck("density_finite", finite, "all values finite" if finite else "NaN/inf in quantiles"))
    if not finite:
        return None

    steps = np.diff(s)
    mono = bool(np.all(steps > 0))
    checks.append(CurveCheck(
        "quantile_monotone", mono,
        "S(p) strictly increasing" if mono
        else f"S(p) decreases at {int((steps <= 0).sum())} grid step(s) — percentile→vol mapping too steep",
    ))
    if not mono:
        return MarketDensity(s, np.zeros_like(s), kind, float("nan"), checks)

    pdf = np.gradient(p, s)                                   # dp/dS ≥ 0 since S(p) is increasing
    mass = float(np.trapezoid(pdf, s))
    checks.append(CurveCheck(
        "density_mass", abs(mass - (_P_HI - _P_LO)) < _MASS_TOL,
        f"∫f = {mass:.4f} vs probability span {_P_HI - _P_LO:.4f}",
    ))

    if reference is not None and kind == "smile":
        worst = 0.0
        for q, attr in _REFERENCE_Z:
            ref = getattr(reference, attr, None)
            if ref is None:
                continue
            got = float(np.interp(q, z, s))
            worst = max(worst, abs(got / ref - 1.0))
        checks.append(CurveCheck(
            "matches_published_percentiles", worst < _REF_REL_TOL,
            f"max |grid/PriceDistribution − 1| over ±1/2/3σ + median = {worst:.2e}",
        ))
    return MarketDensity(s, np.maximum(pdf, 0.0), kind, mass, checks)


def market_density(ccy, horizon_days: int, reference=None) -> MarketDensity | None:
    """Smile density if it reconciles; else flat-vol lognormal if *that* reconciles; else None.

    ``reference`` is the pack's smile ``PriceDistribution`` (optional) — when given, the
    smile grid must reproduce its published percentiles.
    """
    if horizon_days <= 0:
        return None
    for kind in ("smile", "flat"):
        try:
            d = _build(ccy, horizon_days, kind, reference)
        except Exception:
            d = None
        if d is not None and d.ok:
            return d
    return None


def density_for_pack(pack, ccy) -> MarketDensity | None:
    try:
        return market_density(ccy, int(round(pack.market_state.T * 365)), pack.smile_distribution)
    except Exception:
        return None
