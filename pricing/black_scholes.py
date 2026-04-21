"""
Garman-Kohlhagen (GK) pricer for vanilla FX options.

All prices use the Black-76 forward-based form:

    d1 = [ln(F/K) + 0.5*σ²*T] / (σ√T)
    d2 = d1 - σ√T
    Call = DF * [F*N(d1) - K*N(d2)]
    Put  = DF * [K*N(-d2) - F*N(-d1)]

where F is the outright forward and DF = e^(-r_d * T).

Delta convention: forward delta (undiscounted), premium not included.
  Call forward delta: Δ_c = N(d1)          range (0, 1)
  Put  forward delta: Δ_p = N(d1) - 1      range (-1, 0)
  Put-call parity:    Δ_c + |Δ_p| = 1

delta_to_strike is closed-form under this convention:
  K = F × exp(0.5σ²T − N⁻¹(Δ_c) × σ√T)

strike_to_delta is also direct — compute d1 and apply N().

This formulation is correct for both NDF currencies (where the settlement is in
USD and the forward already encodes the full interest differential) and for
deliverable FX options.

Structures implemented:
  - Vanilla call / put
  - Call spread / put spread (long near strike, short far strike)
  - Risk reversal (long OTM call, short OTM put — or reverse)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

from scipy.stats import norm  # type: ignore

_N = norm.cdf
_n = norm.pdf


# ---------------------------------------------------------------------------
# Core formula
# ---------------------------------------------------------------------------

def _d1_d2(F: float, K: float, T: float, sigma: float) -> tuple[float, float]:
    """Compute d1 and d2 for Black-76."""
    if T <= 0:
        # At expiry: intrinsic only
        return (math.inf if F > K else -math.inf), (math.inf if F > K else -math.inf)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def black76_call(F: float, K: float, T: float, sigma: float, discount_factor: float) -> float:
    """Black-76 call price."""
    d1, d2 = _d1_d2(F, K, T, sigma)
    return discount_factor * (F * _N(d1) - K * _N(d2))


def black76_put(F: float, K: float, T: float, sigma: float, discount_factor: float) -> float:
    """Black-76 put price."""
    d1, d2 = _d1_d2(F, K, T, sigma)
    return discount_factor * (K * _N(-d2) - F * _N(-d1))


def black76_delta_call(F: float, K: float, T: float, sigma: float) -> float:
    """Forward delta of a call (dV/dF * 1/DF, i.e. undiscounted)."""
    d1, _ = _d1_d2(F, K, T, sigma)
    return _N(d1)


def black76_delta_put(F: float, K: float, T: float, sigma: float) -> float:
    """Forward delta of a put."""
    d1, _ = _d1_d2(F, K, T, sigma)
    return _N(d1) - 1.0


def black76_vega(F: float, K: float, T: float, sigma: float, discount_factor: float) -> float:
    """Vega (dV/dσ) for a call or put (same for both)."""
    d1, _ = _d1_d2(F, K, T, sigma)
    return discount_factor * F * _n(d1) * math.sqrt(T)


def delta_to_strike(delta: float, F: float, T: float, sigma: float) -> float:
    """
    Convert a forward delta to a strike (closed-form, forward delta / no premium adjustment).

    Args:
        delta: call delta in (0, 1) or put delta in (-1, 0).
               A 25-delta put is passed as -0.25.
    Returns:
        Strike K such that black76_delta_call(F, K, T, sigma) == delta  (for calls)
        or     black76_delta_put(F, K, T, sigma) == delta               (for puts).
    """
    if not (-1.0 < delta < 1.0) or delta == 0.0:
        raise ValueError(f"delta must be in (-1, 1) excluding 0, got {delta}")
    # Convert put delta to call delta equivalent: Δ_c = 1 + Δ_p for puts
    call_delta = delta if delta > 0 else 1.0 + delta
    d1 = norm.ppf(call_delta)
    return F * math.exp(0.5 * sigma ** 2 * T - d1 * sigma * math.sqrt(T))


def strike_to_delta(K: float, F: float, T: float, sigma: float) -> tuple[float, float]:
    """
    Compute both call and put forward deltas for a given strike.

    Returns:
        (call_delta, put_delta) where call_delta ∈ (0,1), put_delta ∈ (-1,0).
    """
    d1, _ = _d1_d2(F, K, T, sigma)
    call_delta = _N(d1)
    put_delta = call_delta - 1.0
    return call_delta, put_delta


# ---------------------------------------------------------------------------
# Convenience wrappers that take (spot, r_d, r_f) instead of forward
# ---------------------------------------------------------------------------

def call_value(
    spot: float,
    strike: float,
    T: float,
    sigma: float,
    r_d: float,
    r_f: float,
    notional: float = 1.0,
) -> float:
    """
    GK call price in premium currency units per unit of notional.
    F = spot * e^((r_d - r_f)*T), DF = e^(-r_d*T).
    """
    F = spot * math.exp((r_d - r_f) * T)
    DF = math.exp(-r_d * T)
    return black76_call(F, strike, T, sigma, DF) * notional


def put_value(
    spot: float,
    strike: float,
    T: float,
    sigma: float,
    r_d: float,
    r_f: float,
    notional: float = 1.0,
) -> float:
    """GK put price."""
    F = spot * math.exp((r_d - r_f) * T)
    DF = math.exp(-r_d * T)
    return black76_put(F, strike, T, sigma, DF) * notional


# ---------------------------------------------------------------------------
# Multi-leg structures
# ---------------------------------------------------------------------------

@dataclass
class SpreadResult:
    net_premium: float    # positive = net cost to buyer
    long_leg_premium: float
    short_leg_premium: float
    long_strike: float
    short_strike: float


def call_spread(
    spot: float,
    low_strike: float,
    high_strike: float,
    T: float,
    sigma_low: float,
    sigma_high: float,
    r_d: float,
    r_f: float,
    notional: float = 1.0,
) -> SpreadResult:
    """
    Long call at low_strike, short call at high_strike.
    Net premium > 0 means the spread costs money (typical: long spread costs less than vanilla).
    """
    long_prem = call_value(spot, low_strike, T, sigma_low, r_d, r_f, notional)
    short_prem = call_value(spot, high_strike, T, sigma_high, r_d, r_f, notional)
    return SpreadResult(
        net_premium=long_prem - short_prem,
        long_leg_premium=long_prem,
        short_leg_premium=short_prem,
        long_strike=low_strike,
        short_strike=high_strike,
    )


def put_spread(
    spot: float,
    low_strike: float,
    high_strike: float,
    T: float,
    sigma_low: float,
    sigma_high: float,
    r_d: float,
    r_f: float,
    notional: float = 1.0,
) -> SpreadResult:
    """
    Long put at high_strike, short put at low_strike (standard put spread).
    Net premium > 0 means net cost.
    """
    long_prem = put_value(spot, high_strike, T, sigma_high, r_d, r_f, notional)
    short_prem = put_value(spot, low_strike, T, sigma_low, r_d, r_f, notional)
    return SpreadResult(
        net_premium=long_prem - short_prem,
        long_leg_premium=long_prem,
        short_leg_premium=short_prem,
        long_strike=high_strike,
        short_strike=low_strike,
    )


class RiskReversalLeg(NamedTuple):
    type: str          # "call" or "put"
    strike: float
    premium: float     # positive = cost of buying that leg
    delta: float       # forward delta


@dataclass
class RiskReversalResult:
    """
    Standard risk reversal: long call + short put (bullish RR) or long put + short call (bearish RR).
    Net premium > 0 means net cost to the long-call / short-put buyer.
    """
    net_premium: float
    call_leg: RiskReversalLeg
    put_leg: RiskReversalLeg
    direction: str     # "bullish" (long call / short put) or "bearish"


def risk_reversal(
    spot: float,
    call_strike: float,
    put_strike: float,
    T: float,
    sigma_call: float,
    sigma_put: float,
    r_d: float,
    r_f: float,
    notional: float = 1.0,
    direction: str = "bullish",
) -> RiskReversalResult:
    """
    Bullish RR: long call at call_strike, short put at put_strike.
    Bearish RR: long put at put_strike, short call at call_strike.

    Convention: net_premium > 0 means you pay net premium.
    """
    F = spot * math.exp((r_d - r_f) * T)
    DF = math.exp(-r_d * T)

    call_prem = black76_call(F, call_strike, T, sigma_call, DF) * notional
    put_prem  = black76_put(F, put_strike, T, sigma_put, DF) * notional
    call_delta = black76_delta_call(F, call_strike, T, sigma_call)
    put_delta  = black76_delta_put(F, put_strike, T, sigma_put)

    if direction == "bullish":
        net = call_prem - put_prem  # long call, short put
    else:
        net = put_prem - call_prem  # long put, short call

    return RiskReversalResult(
        net_premium=net,
        call_leg=RiskReversalLeg("call", call_strike, call_prem, call_delta),
        put_leg=RiskReversalLeg("put", put_strike, put_prem, put_delta),
        direction=direction,
    )


# ---------------------------------------------------------------------------
# Mark-to-market value (used by scenario matrix)
# ---------------------------------------------------------------------------

def call_mtm(
    spot: float,
    strike: float,
    T_remaining: float,
    sigma: float,
    r_d: float,
    r_f: float,
    notional: float = 1.0,
) -> float:
    """
    Current mark-to-market value of a long call with T_remaining years to expiry.
    At T_remaining=0 returns intrinsic value.
    """
    if T_remaining <= 0:
        F = spot * math.exp((r_d - r_f) * max(T_remaining, 0))
        return max(F - strike, 0.0) * math.exp(-r_d * max(T_remaining, 0)) * notional
    return call_value(spot, strike, T_remaining, sigma, r_d, r_f, notional)


def put_mtm(
    spot: float,
    strike: float,
    T_remaining: float,
    sigma: float,
    r_d: float,
    r_f: float,
    notional: float = 1.0,
) -> float:
    """Current mark-to-market value of a long put."""
    if T_remaining <= 0:
        F = spot * math.exp((r_d - r_f) * max(T_remaining, 0))
        return max(strike - F, 0.0) * math.exp(-r_d * max(T_remaining, 0)) * notional
    return put_value(spot, strike, T_remaining, sigma, r_d, r_f, notional)
