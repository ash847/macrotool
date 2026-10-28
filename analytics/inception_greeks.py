"""Conventional inception risks of fixed vanilla contracts, without IO or sizing.

Delta is d(quote PV)/d(spot), naturally base units. Gamma is the change
in that delta for a 1% spot rise. Vega is quote PV sensitivity converted at
fixed inception spot, per one vol point. These are not derivatives of PV/S.
"""

from dataclasses import dataclass
import math

from scipy.special import ndtr

from analytics.product_model import Instrument, PricedStructure, Right


@dataclass(frozen=True)
class Greeks:
    delta_base: float
    gamma_base_per_1pct_spot: float
    vega_base_per_vol_point: float


@dataclass(frozen=True)
class InceptionGreeks:
    legs: tuple[Greeks, ...]
    package: Greeks
    notional_base: float


def inception_greeks(product: PricedStructure, *, spot: float, forward: float,
                     quote_rate: float, notional: float) -> InceptionGreeks:
    """Use retained entry vols and fixed quantities; never resolve anchors again.

Forward scales proportionally with spot (fixed rates/forward points ratio).
All returned leg risks include signed ratios and the actual package notional.
"""
    contract = product.resolved_contract
    if contract is None or product.barrier is not None:
        raise ValueError("Inception Greeks support resolved vanilla-leg packages only")
    if not all(math.isfinite(x) for x in (spot, forward, quote_rate, notional)):
        raise ValueError("Greek inputs must be finite")
    if spot <= 0 or forward <= 0 or notional < 0:
        raise ValueError("Spot/forward must be positive and notional non-negative")
    if len(product.priced_legs) != len(contract.legs):
        raise ValueError("Entry quotes do not match the resolved contract")
    t = contract.tenor_years
    root_t = math.sqrt(t)
    df = math.exp(-quote_rate * t)
    risks = []
    for leg, quote in zip(contract.legs, product.priced_legs):
        if (quote.leg.instrument != Instrument.VANILLA
                or quote.strike != leg.strike or quote.leg.right != leg.right
                or quote.notional != leg.signed_notional):
            raise ValueError("Entry quote differs from the fixed contract")
        vol = quote.vol
        if not math.isfinite(vol) or vol <= 0:
            raise ValueError("Entry vol must be finite and positive")
        d1 = (math.log(forward / leg.strike) + 0.5 * vol * vol * t) / (vol * root_t)
        density = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
        scale = notional * leg.signed_notional
        delta = df * forward / spot * (ndtr(d1) if leg.right == Right.CALL else -ndtr(-d1))
        gamma_1pct = df * forward / spot * density / (vol * root_t) * 0.01
        vega = df * forward * density * root_t / spot * 0.01
        risks.append(Greeks(scale * delta, scale * gamma_1pct, scale * vega))
    total = Greeks(*(math.fsum(getattr(r, field) for r in risks)
                     for field in Greeks.__dataclass_fields__))
    return InceptionGreeks(tuple(risks), total, notional)
