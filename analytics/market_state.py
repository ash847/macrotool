"""
Market state computation for structure selection.

Derives quantitative metrics from raw market inputs (spot, fwd, vol, T, target).
All computations are deterministic; no domain judgment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricing.black_scholes import (
    call_spread as bs_call_spread,
    put_spread as bs_put_spread,
)


@dataclass
class MarketState:
    """Fully-derived market state for a single trade horizon."""

    # --- raw inputs ---
    spot: float
    fwd: float
    vol: float    # ATM vol, annualised (e.g. 0.15 = 15%)
    T: float      # years to expiry
    r_d: float    # domestic continuously-compounded rate
    r_f: float    # foreign continuously-compounded rate

    # --- derived ---
    c: float             # ln(fwd/spot) / (σ√T) — normalised carry
    carry_regime: int    # 0 = noisy (<0.4), 1 = potential (0.4–0.8), 2 = high (>0.8)
    target_z: float | None      # ln(target/fwd) / (σ√T); None if no target supplied
    atmfsratio: float | None    # |fwd-spot| / carry-spread cost; None if carry_regime == 0
    put_call: str | None        # "Call" if target > fwd, "Put" if target < fwd; None if no target
    with_carry: bool            # True if view direction aligns with the carry (sign of c)


def compute_market_state(
    spot: float,
    fwd: float,
    vol: float,
    T: float,
    r_d: float,
    r_f: float,
    target: float | None = None,
    direction: str | None = None,
    carry_regime_cuts: list[float] | None = None,
) -> MarketState:
    """
    Compute all derived market state metrics from raw inputs.

    atmfsratio is the payout ratio of the spread that exactly captures the forward
    drift — long ATM-spot leg, short ATM-fwd leg, in the direction of the carry.
    Only computed when carry_regime >= 1 (|c| >= 0.4).

    Args:
        spot:   Spot rate.
        fwd:    Outright forward at expiry T.
        vol:    ATM implied vol, annualised.
        T:      Time to expiry in years.
        r_d:    Domestic continuously-compounded rate.
        r_f:    Foreign continuously-compounded rate.
        target: Optional target spot level.
    """
    vol_sqrt_T = vol * math.sqrt(T)

    c = math.log(fwd / spot) / vol_sqrt_T

    cuts = carry_regime_cuts if carry_regime_cuts is not None else [0.4, 0.8]
    abs_c = abs(c)
    if abs_c < cuts[0]:
        carry_regime = 0
    elif abs_c < cuts[1]:
        carry_regime = 1
    else:
        carry_regime = 2

    target_z = math.log(target / fwd) / vol_sqrt_T if target is not None else None
    put_call = ("Call" if target > fwd else "Put") if target is not None else None
    with_carry = (c > 0) == (direction == "base_lower") if direction else (c > 0)

    atmfsratio = None
    carry_pips = abs(fwd - spot)
    if carry_pips > 0:
        if fwd >= spot:
            result = bs_call_spread(spot, spot, fwd, T, vol, vol, r_d, r_f)
        else:
            result = bs_put_spread(spot, fwd, spot, T, vol, vol, r_d, r_f)
        if result.net_premium > 0:
            atmfsratio = carry_pips / result.net_premium

    return MarketState(
        spot=spot,
        fwd=fwd,
        vol=vol,
        T=T,
        r_d=r_d,
        r_f=r_f,
        c=c,
        carry_regime=carry_regime,
        target_z=target_z,
        atmfsratio=atmfsratio,
        put_call=put_call,
        with_carry=with_carry,
    )
