"""
Reverse Knock-Out (RKO) option pricer — closed-form under GBM.

An RKO option ceases to exist (knocks out) if spot touches the barrier at any
point during the option's life. "Reverse" means the barrier is in the
profit direction:
  - RKO call: up-and-out call — barrier H is ABOVE current spot
  - RKO put:  down-and-out put — barrier H is BELOW current spot

Implementation uses the full Reiner-Rubinstein (1991) / Haug (2006) formula,
expressed in terms of A, B, C, D terms. This is correct for arbitrary cost of
carry (r_d ≠ r_f) — unlike the simplified reflection formula which only holds
when r_d = r_f.

Reference: Haug, "The Complete Guide to Option Pricing Formulas", 2nd ed., Ch. 4.

Notation (Haug):
    b = r_d - r_f   (cost of carry)
    μ = (b - σ²/2) / σ²
    φ = 1 (call), -1 (put)
    η = -1 (up barrier), 1 (down barrier)

Formula: vanilla_barrier = A - B + C - D  (for the cases where the option has value)
"""

from __future__ import annotations

import math

from scipy.stats import norm  # type: ignore

_N = norm.cdf
_n = norm.pdf


def _barrier_terms(
    S: float,
    K: float,
    H: float,
    T: float,
    sigma: float,
    r_d: float,
    r_f: float,
    phi: float,   # 1=call, -1=put
    eta: float,   # -1=up barrier, 1=down barrier
) -> tuple[float, float, float, float]:
    """
    Compute the A, B, C, D terms from the Haug single-barrier formula.

    Returns (A, B, C, D).
    The option price for the relevant barrier type is then a combination of these.
    """
    b = r_d - r_f
    r = r_d
    sqrt_T = math.sqrt(T)
    mu = (b - 0.5 * sigma ** 2) / sigma ** 2

    x1 = math.log(S / K) / (sigma * sqrt_T) + (1 + mu) * sigma * sqrt_T
    x2 = math.log(S / H) / (sigma * sqrt_T) + (1 + mu) * sigma * sqrt_T
    y1 = math.log(H ** 2 / (S * K)) / (sigma * sqrt_T) + (1 + mu) * sigma * sqrt_T
    y2 = math.log(H / S) / (sigma * sqrt_T) + (1 + mu) * sigma * sqrt_T

    # Coefficients for the barrier reflection terms
    h_s_2mu_plus2 = (H / S) ** (2 * (mu + 1))
    h_s_2mu       = (H / S) ** (2 * mu)

    carry_factor = math.exp((b - r) * T)   # = e^(-r_f * T) in GK terms
    disc_factor  = math.exp(-r * T)         # = e^(-r_d * T)

    A = phi * (S * carry_factor * _N(phi * x1) - K * disc_factor * _N(phi * (x1 - sigma * sqrt_T)))
    B = phi * (S * carry_factor * _N(phi * x2) - K * disc_factor * _N(phi * (x2 - sigma * sqrt_T)))
    C = phi * (S * carry_factor * h_s_2mu_plus2 * _N(eta * y1)
               - K * disc_factor * h_s_2mu * _N(eta * (y1 - sigma * sqrt_T)))
    D = phi * (S * carry_factor * h_s_2mu_plus2 * _N(eta * y2)
               - K * disc_factor * h_s_2mu * _N(eta * (y2 - sigma * sqrt_T)))

    return A, B, C, D


def rko_call(
    spot: float,
    strike: float,
    barrier: float,
    T: float,
    sigma: float,
    r_d: float,
    r_f: float,
    notional: float = 1.0,
) -> float:
    """
    Up-and-out call (RKO call). Barrier H must be above current spot.

    Returns option value in premium currency per unit of notional.
    Returns 0 if spot >= barrier (already knocked out) or barrier <= strike.
    """
    if spot >= barrier:
        return 0.0
    if barrier <= strike:
        # Barrier at or below strike: knocked out before becoming ITM
        return 0.0

    A, B, C, D = _barrier_terms(spot, strike, barrier, T, sigma, r_d, r_f, phi=1.0, eta=-1.0)
    price = A - B + C - D
    return max(price, 0.0) * notional


def rko_put(
    spot: float,
    strike: float,
    barrier: float,
    T: float,
    sigma: float,
    r_d: float,
    r_f: float,
    notional: float = 1.0,
) -> float:
    """
    Down-and-out put (RKO put). Barrier H must be below current spot.

    Returns option value in premium currency per unit of notional.
    Returns 0 if spot <= barrier (already knocked out) or barrier >= strike.
    """
    if spot <= barrier:
        return 0.0
    if barrier >= strike:
        return 0.0

    A, B, C, D = _barrier_terms(spot, strike, barrier, T, sigma, r_d, r_f, phi=-1.0, eta=1.0)
    price = A - B + C - D
    return max(price, 0.0) * notional


def rko_call_mtm(
    spot: float,
    strike: float,
    barrier: float,
    T_remaining: float,
    sigma: float,
    r_d: float,
    r_f: float,
    notional: float = 1.0,
) -> float:
    """Mark-to-market value of a long RKO call with T_remaining years left."""
    if T_remaining <= 0:
        if spot >= barrier:
            return 0.0
        return max(spot - strike, 0.0) * notional
    return rko_call(spot, strike, barrier, T_remaining, sigma, r_d, r_f, notional)


def rko_put_mtm(
    spot: float,
    strike: float,
    barrier: float,
    T_remaining: float,
    sigma: float,
    r_d: float,
    r_f: float,
    notional: float = 1.0,
) -> float:
    """Mark-to-market value of a long RKO put with T_remaining years left."""
    if T_remaining <= 0:
        if spot <= barrier:
            return 0.0
        return max(strike - spot, 0.0) * notional
    return rko_put(spot, strike, barrier, T_remaining, sigma, r_d, r_f, notional)
