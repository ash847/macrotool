"""
Spread pricing with full smile interpolation.

Two entry points:

  price_delta_spread(long_delta, short_delta, ...)
    → input: deltas (e.g. -0.40 and -0.20 for a put spread)
    → output: strikes, vols, and premiums for each leg

  price_strike_spread(long_strike, short_strike, ...)
    → input: strikes (e.g. 5.50 and 5.70)
    → output: deltas, vols, and premiums for each leg

Both return a SpreadPricingResult. Premium is in base currency units
(USD for USD/EM pairs, EUR for EURPLN) as a fraction of notional,
and as an absolute amount given notional.

Convention
----------
"Long" means you own the option. For a put spread:
  - long_leg is the higher-strike put (more expensive, closer to spot)
  - short_leg is the lower-strike put (cheaper, further OTM)
  Net premium > 0 means you pay net.

Delta sign:
  - Call delta: positive (0, 1)
  - Put delta: negative (-1, 0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from analytics.vol_surface import SmileInterpolator
from data.schema import CurrencySnapshot
from pricing.black_scholes import (
    black76_call,
    black76_put,
    delta_to_strike,
    strike_to_delta,
)
from pricing.forwards import rate_context_for_snapshot


@dataclass
class SpreadLeg:
    strike: float
    call_delta: float    # always positive (0, 1)
    put_delta: float     # always negative (-1, 0)
    vol: float           # implied vol used to price this leg
    premium: float       # absolute premium in base currency
    premium_pct: float   # premium as % of notional (e.g. 0.0123 = 1.23%)


@dataclass
class SpreadPricingResult:
    pair: str
    horizon_days: int
    T: float
    spot: float
    forward: float
    long_leg: SpreadLeg
    short_leg: SpreadLeg
    net_premium: float      # absolute, base currency (positive = you pay)
    net_premium_pct: float  # as % of notional


def price_delta_spread(
    long_delta: float,
    short_delta: float,
    ccy: CurrencySnapshot,
    horizon_days: int,
    notional: float = 1_000_000.0,
) -> SpreadPricingResult:
    """
    Price a spread given two forward deltas.

    Args:
        long_delta:  delta of the leg you buy  (e.g. -0.40 for 40d put).
        short_delta: delta of the leg you sell (e.g. -0.20 for 20d put).
        ccy:         CurrencySnapshot for the pair.
        horizon_days: days to expiry.
        notional:    base currency notional.

    Both deltas must have the same sign (both calls or both puts).
    """
    _validate_same_type(long_delta, short_delta)

    T = horizon_days / 365.0
    rate_ctx = rate_context_for_snapshot(ccy, T)
    smile = SmileInterpolator(ccy)

    long_vol = smile.vol_at_delta(long_delta, horizon_days)
    short_vol = smile.vol_at_delta(short_delta, horizon_days)

    long_K  = delta_to_strike(long_delta,  rate_ctx.forward, T, long_vol)
    short_K = delta_to_strike(short_delta, rate_ctx.forward, T, short_vol)

    return _build_result(
        ccy, horizon_days, T, rate_ctx,
        long_delta, short_delta,
        long_K, short_K,
        long_vol, short_vol,
        notional,
    )


def price_strike_spread(
    long_strike: float,
    short_strike: float,
    ccy: CurrencySnapshot,
    horizon_days: int,
    option_type: str = "put",
    notional: float = 1_000_000.0,
) -> SpreadPricingResult:
    """
    Price a spread given two strikes.

    Args:
        long_strike:  strike of the leg you buy.
        short_strike: strike of the leg you sell.
        ccy:          CurrencySnapshot for the pair.
        horizon_days: days to expiry.
        option_type:  "call" or "put".
        notional:     base currency notional.
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    T = horizon_days / 365.0
    rate_ctx = rate_context_for_snapshot(ccy, T)
    smile = SmileInterpolator(ccy)

    long_vol  = smile.vol_at_strike(long_strike,  rate_ctx.forward, horizon_days)
    short_vol = smile.vol_at_strike(short_strike, rate_ctx.forward, horizon_days)

    long_c_delta,  long_p_delta  = strike_to_delta(long_strike,  rate_ctx.forward, T, long_vol)
    short_c_delta, short_p_delta = strike_to_delta(short_strike, rate_ctx.forward, T, short_vol)

    long_delta  = long_p_delta  if option_type == "put" else long_c_delta
    short_delta = short_p_delta if option_type == "put" else short_c_delta

    return _build_result(
        ccy, horizon_days, T, rate_ctx,
        long_delta, short_delta,
        long_strike, short_strike,
        long_vol, short_vol,
        notional,
    )


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _build_result(
    ccy: CurrencySnapshot,
    horizon_days: int,
    T: float,
    rate_ctx,
    long_delta: float,
    short_delta: float,
    long_K: float,
    short_K: float,
    long_vol: float,
    short_vol: float,
    notional: float,
) -> SpreadPricingResult:
    F  = rate_ctx.forward
    DF = rate_ctx.discount_factor
    is_put = long_delta < 0

    if is_put:
        long_prem  = black76_put(F, long_K,  T, long_vol,  DF) * notional
        short_prem = black76_put(F, short_K, T, short_vol, DF) * notional
    else:
        long_prem  = black76_call(F, long_K,  T, long_vol,  DF) * notional
        short_prem = black76_call(F, short_K, T, short_vol, DF) * notional

    long_c,  long_p  = strike_to_delta(long_K,  F, T, long_vol)
    short_c, short_p = strike_to_delta(short_K, F, T, short_vol)

    net = long_prem - short_prem

    long_leg = SpreadLeg(
        strike=long_K,
        call_delta=long_c,
        put_delta=long_p,
        vol=long_vol,
        premium=long_prem,
        premium_pct=long_prem / notional,
    )
    short_leg = SpreadLeg(
        strike=short_K,
        call_delta=short_c,
        put_delta=short_p,
        vol=short_vol,
        premium=short_prem,
        premium_pct=short_prem / notional,
    )

    return SpreadPricingResult(
        pair=ccy.pair,
        horizon_days=horizon_days,
        T=T,
        spot=ccy.spot,
        forward=F,
        long_leg=long_leg,
        short_leg=short_leg,
        net_premium=net,
        net_premium_pct=net / notional,
    )


def _validate_same_type(d1: float, d2: float) -> None:
    if (d1 > 0) != (d2 > 0):
        raise ValueError(
            f"Both legs must be same type (both calls >0 or both puts <0). "
            f"Got {d1} and {d2}."
        )
