"""Revalue fixed vanilla legs without building, resizing or solving a trade."""

from dataclasses import dataclass
import math

import numpy as np

from analytics.product_model import ResolvedContract, Right
from analytics.vol_surface import VolSurface, smile_skew_spread
from pricing.black_scholes import call_mtm, put_mtm


@dataclass(frozen=True)
class PricingContext:
    spot: float
    remaining_years: float
    atm_vol: float
    r_d: float
    r_f: float
    surface: VolSurface | None = None

    def __post_init__(self):
        if not all(math.isfinite(x) for x in
                   (self.spot, self.remaining_years, self.atm_vol, self.r_d, self.r_f)):
            raise ValueError("Pricing inputs must be finite")
        if self.spot <= 0 or self.remaining_years < 0 or self.atm_vol < 0:
            raise ValueError("Invalid spot, remaining maturity or volatility")


def value_resolved(contract: ResolvedContract, context: PricingContext) -> float:
    """Quote-currency PV per unit base notional, using the existing sticky-delta rule.

    Divide by the valuation spot for base-currency value. Inception delta-pillar
    pricing is intentionally separate: this is revaluation, not reconstruction.
    """
    c = context
    if c.remaining_years > contract.tenor_years + 1e-12:
        raise ValueError("Remaining maturity exceeds the contract's original tenor")
    forward = c.spot * math.exp((c.r_d - c.r_f) * c.remaining_years)
    total = 0.0
    for leg in contract.legs:
        vol = c.atm_vol
        if c.surface is not None and c.remaining_years > 0:
            vol = max(vol + smile_skew_spread(
                c.surface, leg.strike, forward, max(round(c.remaining_years * 365), 1)
            ), 0.01)
        fn = call_mtm if leg.right == Right.CALL else put_mtm
        total += leg.signed_notional * fn(
            c.spot, leg.strike, c.remaining_years, vol, c.r_d, c.r_f
        )
    return total


def terminal_payoff(contract: ResolvedContract, spots) -> np.ndarray:
    """Gross terminal payoff in base currency, at terminal (not inception) spot."""
    prices = np.asarray(spots, dtype=float)
    if np.any(~np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError("Terminal spots must be finite and positive")
    total = np.zeros_like(prices)
    for leg in contract.legs:
        sign = 1 if leg.right == Right.CALL else -1
        total += leg.signed_notional * np.maximum(sign * (prices - leg.strike), 0.0)
    return total / prices
