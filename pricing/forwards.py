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


def interpolate_df_rate(df_curve: list, T_years: float) -> float:
    """
    Linearly interpolate a continuously-compounded rate from a DF curve.

    df_curve entries must have .tenor (TenorLabel) and .df (float) attributes.
    """
    points = sorted([(tenor_to_years(d.tenor), d.df) for d in df_curve])
    if T_years <= points[0][0]:
        t, df = points[0]
        return -math.log(df) / t if t > 0 else 0.0
    if T_years >= points[-1][0]:
        t, df = points[-1]
        return -math.log(df) / t
    for i in range(len(points) - 1):
        t0, df0 = points[i]
        t1, df1 = points[i + 1]
        if t0 <= T_years <= t1:
            w = (T_years - t0) / (t1 - t0)
            df_interp = df0 + (df1 - df0) * w
            return -math.log(df_interp) / T_years
    t, df = points[-1]
    return -math.log(df) / t


def build_rate_context(
    snapshot: CurrencySnapshot,
    T_years: float,
    r_d: float,
    r_f: float | None = None,
) -> RateContext:
    """
    Build a RateContext for a given pair and tenor.

    For NDF pairs: pass r_d (USD rate); r_f is CIP-derived from the forward.
    For EURPLN: pass r_f (EUR rate from eur_df_curve); r_d (PLN) is CIP-derived.

    Args:
        snapshot:  CurrencySnapshot for the pair.
        T_years:   Time to expiry in years.
        r_d:       Domestic rate (settlement currency). Ignored when r_f is supplied.
        r_f:       Foreign (base) rate. When provided, r_d is derived from CIP
                   instead of the other way around.
    """
    forward = interpolate_forward(snapshot, T_years)
    if r_f is not None:
        # r_f is the known anchor; derive r_d from CIP: r_d = r_f + ln(F/S)/T
        r_d = r_f + math.log(forward / snapshot.spot) / T_years
    else:
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


# Default settlement currency rates used for NDF pairs (USD is the settlement currency)
DEFAULT_SETTLEMENT_RATES: dict[str, float] = {
    "USDBRL": 0.043,
    "USDTRY": 0.043,
}


def rate_context_for_snapshot(snapshot: CurrencySnapshot, T_years: float) -> RateContext:
    """
    Build a RateContext using the correct rate anchor for the pair.

    NDF pairs (USDBRL, USDTRY): USD rate from DEFAULT_SETTLEMENT_RATES is the
    anchor; the EM rate is CIP-derived from the forward.

    EURPLN (deliverable): EUR rate is read directly from eur_df_curve and is
    the anchor; PLN rate is CIP-derived from the forward.
    """
    if snapshot.instrument_type == "Deliverable":
        r_f = interpolate_df_rate(snapshot.eur_df_curve, T_years)
        return build_rate_context(snapshot, T_years, r_d=0.0, r_f=r_f)
    r_d = DEFAULT_SETTLEMENT_RATES.get(snapshot.pair, 0.043)
    return build_rate_context(snapshot, T_years, r_d=r_d)
