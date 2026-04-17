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

from scipy.stats import norm  # type: ignore

_N = norm.cdf


def _d2(F: float, K: float, T: float, sigma: float) -> float:
    """d2 from Black-76."""
    if T <= 0:
        return math.inf if F > K else -math.inf
    return (math.log(F / K) - 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))


def digital_call(
    spot: float,
    strike: float,
    T: float,
    sigma: float,
    r_d: float,
    r_f: float,
    payout: float = 1.0,
    notional: float = 1.0,
) -> float:
    """
    European cash-or-nothing digital call.

    Pays `payout` units of premium currency per unit of notional if S_T > strike.

    Args:
        payout: Fixed payout as a fraction of notional (e.g. 0.05 = 5% of notional).
    """
    F = spot * math.exp((r_d - r_f) * T)
    DF = math.exp(-r_d * T)
    d2_val = _d2(F, strike, T, sigma)
    return DF * payout * _N(d2_val) * notional


def digital_put(
    spot: float,
    strike: float,
    T: float,
    sigma: float,
    r_d: float,
    r_f: float,
    payout: float = 1.0,
    notional: float = 1.0,
) -> float:
    """European cash-or-nothing digital put. Pays payout if S_T < strike."""
    F = spot * math.exp((r_d - r_f) * T)
    DF = math.exp(-r_d * T)
    d2_val = _d2(F, strike, T, sigma)
    return DF * payout * _N(-d2_val) * notional


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
