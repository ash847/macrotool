"""
European digital option with Reverse Knock-Out (digital + RKO) pricer.

This pays a fixed cash amount R if:
  - S_T > K at expiry (for a digital call), AND
  - Spot has NOT touched the barrier H during the option's life.

The barrier H is in the profit direction (reverse knock-out):
  - Digital call + RKO: H is ABOVE spot (up-and-out)
  - Digital put  + RKO: H is BELOW spot (down-and-out)

Pricing method:
    The digital+RKO price is computed as the negative derivative of the vanilla
    barrier call price with respect to strike:

        dc_uo(K) = -∂c_uo/∂K

    This is implemented via central finite difference on the Haug vanilla barrier
    formula, which is exact and avoids re-deriving the complex binary-barrier
    closed-form from scratch. Accuracy is sufficient for POC comparison purposes.

    The payout is normalised: dc_uo returns the value per unit payout × notional.
"""

from __future__ import annotations

from pricing.rko import rko_call, rko_put


_BUMP = 1e-5   # relative strike bump for finite difference


def digital_rko_call(
    spot: float,
    strike: float,
    barrier: float,
    T: float,
    sigma: float,
    r_d: float,
    r_f: float,
    payout: float = 1.0,
    notional: float = 1.0,
) -> float:
    """
    European digital call with up-and-out barrier (RKO).
    Pays `payout * notional` if S_T > strike AND spot never touched barrier H.

    Barrier H must be above current spot.
    Returns 0 if spot >= barrier or barrier <= strike.
    """
    if spot >= barrier:
        return 0.0
    if barrier <= strike:
        return 0.0

    dK = strike * _BUMP
    c_lo = rko_call(spot, strike - dK, barrier, T, sigma, r_d, r_f, notional=1.0)
    c_hi = rko_call(spot, strike + dK, barrier, T, sigma, r_d, r_f, notional=1.0)
    dc = (c_lo - c_hi) / (2 * dK)   # -∂c/∂K
    return max(dc * payout, 0.0) * notional


def digital_rko_put(
    spot: float,
    strike: float,
    barrier: float,
    T: float,
    sigma: float,
    r_d: float,
    r_f: float,
    payout: float = 1.0,
    notional: float = 1.0,
) -> float:
    """
    European digital put with down-and-out barrier (RKO).
    Pays `payout * notional` if S_T < strike AND spot never touched barrier H.

    Barrier H must be below current spot.
    Returns 0 if spot <= barrier or barrier >= strike.
    """
    if spot <= barrier:
        return 0.0
    if barrier >= strike:
        return 0.0

    dK = strike * _BUMP
    p_lo = rko_put(spot, strike - dK, barrier, T, sigma, r_d, r_f, notional=1.0)
    p_hi = rko_put(spot, strike + dK, barrier, T, sigma, r_d, r_f, notional=1.0)
    dp = (p_hi - p_lo) / (2 * dK)   # -∂p/∂K (put value increases with strike)
    return max(dp * payout, 0.0) * notional


def digital_rko_call_mtm(
    spot: float,
    strike: float,
    barrier: float,
    T_remaining: float,
    sigma: float,
    r_d: float,
    r_f: float,
    payout: float = 1.0,
    notional: float = 1.0,
) -> float:
    """Mark-to-market value of a long digital+RKO call with T_remaining years left."""
    if T_remaining <= 0:
        if spot >= barrier:
            return 0.0
        return (payout * notional) if spot > strike else 0.0
    return digital_rko_call(spot, strike, barrier, T_remaining, sigma, r_d, r_f, payout, notional)


def digital_rko_put_mtm(
    spot: float,
    strike: float,
    barrier: float,
    T_remaining: float,
    sigma: float,
    r_d: float,
    r_f: float,
    payout: float = 1.0,
    notional: float = 1.0,
) -> float:
    """Mark-to-market value of a long digital+RKO put with T_remaining years left."""
    if T_remaining <= 0:
        if spot <= barrier:
            return 0.0
        return (payout * notional) if spot < strike else 0.0
    return digital_rko_put(spot, strike, barrier, T_remaining, sigma, r_d, r_f, payout, notional)
