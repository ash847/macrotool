"""
Forward pricing utilities.

Handles:
  - Tenor string → years conversion
  - Outright forward interpolation from the snapshot term structure
  - Implied rate differential from spot/forward
  - Discount factor computation

NDF note: NDF outrights in the snapshot already reflect the full interest rate
differential (including fixing-lag adjustments). For pricing purposes we treat
the outright forward as the GBM drift anchor and derive r_d - r_f implicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from data.schema import CurrencySnapshot

# Tenor string → approximate year fraction (Act/365 convention for POC)
_TENOR_YEARS: dict[str, float] = {
    "1W":  7 / 365,
    "2W": 14 / 365,
    "1M": 30 / 365,
    "2M": 60 / 365,
    "3M": 91 / 365,
    "6M": 182 / 365,
    "9M": 273 / 365,
    "1Y": 365 / 365,
}


def tenor_to_years(tenor: str) -> float:
    """Convert tenor string to year fraction (Act/365)."""
    if tenor not in _TENOR_YEARS:
        raise ValueError(f"Unknown tenor '{tenor}'. Supported: {list(_TENOR_YEARS)}")
    return _TENOR_YEARS[tenor]


@dataclass
class RateContext:
    """
    Everything downstream pricers need about rates for a single pair and tenor.

    r_d and r_f are derived from the snapshot's spot/forward relationship
    together with a supplied settlement currency rate (r_d).
    """
    spot: float
    forward: float        # outright forward at T
    T: float              # time in years
    r_d: float            # domestic (settlement currency) continuously-compounded rate
    r_f: float            # foreign rate implied from CIP: F = S * e^((r_d - r_f)*T)
    discount_factor: float  # e^(-r_d * T)

    @property
    def rate_differential(self) -> float:
        return self.r_d - self.r_f


def interpolate_forward(snapshot: CurrencySnapshot, T_years: float) -> float:
    """
    Linear interpolation of the outright forward from the snapshot term structure.

    For T beyond the last tenor, flat extrapolation from the last point.
    For T before the first tenor, flat extrapolation from spot.
    """
    tenors = [(tenor_to_years(f.tenor), f.outright) for f in snapshot.forwards]
    tenors.sort(key=lambda x: x[0])

    if T_years <= tenors[0][0]:
        # Interpolate between spot and first forward
        t0, t1 = 0.0, tenors[0][0]
        f0, f1 = snapshot.spot, tenors[0][1]
        if t1 == t0:
            return f1
        return f0 + (f1 - f0) * T_years / t1

    if T_years >= tenors[-1][0]:
        return tenors[-1][1]

    for i in range(len(tenors) - 1):
        t0, f0 = tenors[i]
        t1, f1 = tenors[i + 1]
        if t0 <= T_years <= t1:
            weight = (T_years - t0) / (t1 - t0)
            return f0 + (f1 - f0) * weight

    return tenors[-1][1]  # fallback


def implied_r_f(spot: float, forward: float, T: float, r_d: float) -> float:
    """
    Derive the foreign (base currency) rate from CIP:
      F = S * e^((r_d - r_f) * T)
      => r_f = r_d - ln(F/S) / T
    """
    if T <= 0:
        raise ValueError("T must be positive")
    return r_d - math.log(forward / spot) / T


def discount_factor(rate: float, T: float) -> float:
    """e^(-rate * T)"""
    return math.exp(-rate * T)


def build_rate_context(
    snapshot: CurrencySnapshot,
    T_years: float,
    r_d: float,
) -> RateContext:
    """
    Build a RateContext for a given pair, tenor, and domestic rate.

    Args:
        snapshot:  The CurrencySnapshot for the pair.
        T_years:   Time to expiry in years.
        r_d:       Domestic (settlement currency) continuously-compounded rate.
                   For NDF pairs (USD settlement): USD rate.
                   For EURPLN: PLN rate (PLN is the term/settlement currency).

    Returns:
        RateContext with all rate inputs resolved.
    """
    forward = interpolate_forward(snapshot, T_years)
    r_f = implied_r_f(snapshot.spot, forward, T_years, r_d)
    df = discount_factor(r_d, T_years)
    return RateContext(
        spot=snapshot.spot,
        forward=forward,
        T=T_years,
        r_d=r_d,
        r_f=r_f,
        discount_factor=df,
    )


# Default settlement currency rates for the synthetic snapshot (approximate, POC only)
DEFAULT_SETTLEMENT_RATES: dict[str, float] = {
    "USDBRL": 0.043,   # USD rate (settlement currency for NDF)
    "USDTRY": 0.043,   # USD rate (settlement currency for NDF)
    "EURPLN": 0.058,   # PLN rate (term/settlement currency for deliverable)
}
