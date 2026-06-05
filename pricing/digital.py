"""
European digital (cash-or-nothing) option pricer.

A European digital call pays a fixed cash amount R if S_T > K at expiry.
A European digital put pays R if S_T < K at expiry.

Pricing formula (Black-76 form):

    Digital call = DF * R * N(d2)
    Digital put  = DF * R * N(-d2)

where:
    d2 = [ln(F/K) - 0.5*σ²*T] / (σ√T)
    DF = e^(-r_d * T)
    F  = outright forward = S * e^((r_d - r_f)*T)
    R  = fixed payout amount (in premium currency units per unit of notional)

Put-call parity check:
    digital_call + digital_put = DF * R    [exhaustive events]

Key risk: binary gamma near expiry. As T→0, the digital behaves like a Dirac
delta at K — the delta and gamma become very large near the strike. The scenario
matrix will make this visible.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from scipy.stats import norm  # type: ignore

from pricing.black_scholes import black76_vega

_N = norm.cdf

# Central-difference step for the smile slope dσ/dK, as a fraction of strike.
_SLOPE_REL_H = 1e-3

# Tolerance on the [0, DF] bound below/above which a smile digital is treated as
# an arbitrage (genuine violations are O(1e-3)+; smile-slope FD noise is < 1e-9).
_ARB_ATOL = 1e-7


class SmileArbitrageError(ValueError):
    """The smile-implied digital fell outside ``[0, DF]``.

    A digital is ``DF·Q(S_T > K)`` — a discounted probability — so a value outside
    ``[0, DF]`` means the interpolated smile carries a local butterfly arbitrage
    (negative risk-neutral density) at this strike, typically cubic-spline
    overshoot in the extrapolated wings. The price is not trustworthy; callers
    should drop the affected variant rather than clamp or emit it.
    """


def _d2(F: float, K: float, T: float, sigma: float) -> float:
    """d2 from Black-76."""
    if T <= 0:
        return math.inf if F > K else -math.inf
    return (math.log(F / K) - 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))


def _smile_slope(vol_fn: Callable[[float], float], K: float) -> float:
    """Central-difference estimate of dσ/dK from the smile seam at strike K."""
    h = max(abs(K) * _SLOPE_REL_H, 1e-6)
    return (vol_fn(K + h) - vol_fn(K - h)) / (2.0 * h)


def _guard_digital_unit(unit: float, DF: float, strike: float, side: str) -> None:
    """Raise SmileArbitrageError if a unit digital value escapes [0, DF]."""
    if unit < -_ARB_ATOL or unit > DF + _ARB_ATOL:
        raise SmileArbitrageError(
            f"digital {side} unit value {unit:.4g} outside [0, {DF:.4g}] at "
            f"K={strike:.4f} — smile butterfly arbitrage (negative density)"
        )


def digital_call(
    spot: float,
    strike: float,
    T: float,
    sigma: float,
    r_d: float,
    r_f: float,
    payout: float = 1.0,
    notional: float = 1.0,
    vol_fn: Callable[[float], float] | None = None,
) -> float:
    """
    European cash-or-nothing digital call.

    Pays `payout` units of premium currency per unit of notional if S_T > strike.

    Args:
        payout: Fixed payout as a fraction of notional (e.g. 0.05 = 5% of notional).
        vol_fn: Optional strike→vol callable (the smile seam). When supplied, the
            digital is the smile-consistent value ``DF·N(d2(σ(K))) − vega·σ′(K)``
            — the strike-derivative of the call price including the skew-slope
            correction. When ``None`` the scalar ``sigma`` is used, reproducing
            the legacy flat-vol price ``DF·N(d2)`` byte-for-byte.
    """
    F = spot * math.exp((r_d - r_f) * T)
    DF = math.exp(-r_d * T)
    if vol_fn is None:
        d2_val = _d2(F, strike, T, sigma)
        return DF * payout * _N(d2_val) * notional
    sig = vol_fn(strike)
    d2_val = _d2(F, strike, T, sig)
    vega = black76_vega(F, strike, T, sig, DF)
    unit = DF * _N(d2_val) - vega * _smile_slope(vol_fn, strike)
    _guard_digital_unit(unit, DF, strike, "call")
    return payout * notional * unit


def digital_put(
    spot: float,
    strike: float,
    T: float,
    sigma: float,
    r_d: float,
    r_f: float,
    payout: float = 1.0,
    notional: float = 1.0,
    vol_fn: Callable[[float], float] | None = None,
) -> float:
    """European cash-or-nothing digital put. Pays payout if S_T < strike.

    ``vol_fn`` is the optional strike→vol smile seam. Smile-consistent value is
    ``DF·N(−d2(σ(K))) + vega·σ′(K)`` (put parity flips the sign of the skew-slope
    term vs the call); ``None`` reproduces the legacy flat-vol price exactly.
    """
    F = spot * math.exp((r_d - r_f) * T)
    DF = math.exp(-r_d * T)
    if vol_fn is None:
        d2_val = _d2(F, strike, T, sigma)
        return DF * payout * _N(-d2_val) * notional
    sig = vol_fn(strike)
    d2_val = _d2(F, strike, T, sig)
    vega = black76_vega(F, strike, T, sig, DF)
    unit = DF * _N(-d2_val) + vega * _smile_slope(vol_fn, strike)
    _guard_digital_unit(unit, DF, strike, "put")
    return payout * notional * unit


def digital_call_mtm(
    spot: float,
    strike: float,
    T_remaining: float,
    sigma: float,
    r_d: float,
    r_f: float,
    payout: float = 1.0,
    notional: float = 1.0,
) -> float:
    """Mark-to-market value of a long digital call with T_remaining years left."""
    if T_remaining <= 0:
        return (payout * notional) if spot > strike else 0.0
    return digital_call(spot, strike, T_remaining, sigma, r_d, r_f, payout, notional)


def digital_put_mtm(
    spot: float,
    strike: float,
    T_remaining: float,
    sigma: float,
    r_d: float,
    r_f: float,
    payout: float = 1.0,
    notional: float = 1.0,
) -> float:
    """Mark-to-market value of a long digital put with T_remaining years left."""
    if T_remaining <= 0:
        return (payout * notional) if spot < strike else 0.0
    return digital_put(spot, strike, T_remaining, sigma, r_d, r_f, payout, notional)
