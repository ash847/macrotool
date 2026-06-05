"""
Option and structure pricing on a discrete distribution.

Pricing is expected payoff at expiry under a given distribution, scaled by the
discount factor for the quote currency.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .elicitation import Distribution


PayoffFn = Callable[[np.ndarray], np.ndarray]


def call_payoff(strike: float) -> PayoffFn:
    return lambda prices: np.maximum(prices - strike, 0.0)


def put_payoff(strike: float) -> PayoffFn:
    return lambda prices: np.maximum(strike - prices, 0.0)


def base_ccy_payoff_for_trade_rec(
    structure_id: str,
    *,
    strikes: list[float],
    barrier: float | None,
    is_call: bool,
    entry_spot: float,
    wing_ratio: float | None = None,
) -> PayoffFn:
    """Return terminal payoff in base-ccy units for one Trade Rec variant.

    The returned payoff is per 1 unit of structure notional and matches the
    `*_pct` fields on `PricedVariant`, which are quoted in base currency units.
    """
    if structure_id == "vanilla":
        strike = strikes[0]
        if is_call:
            return lambda prices: np.maximum(prices - strike, 0.0) / prices
        return lambda prices: np.maximum(strike - prices, 0.0) / prices

    if structure_id == "1x1_spread":
        k1, k2 = strikes[:2]
        if is_call:
            return lambda prices: (np.maximum(prices - k1, 0.0) - np.maximum(prices - k2, 0.0)) / prices
        return lambda prices: (np.maximum(k1 - prices, 0.0) - np.maximum(k2 - prices, 0.0)) / prices

    if structure_id == "1x1.5_spread":
        k1, k2 = strikes[:2]
        if is_call:
            return lambda prices: (np.maximum(prices - k1, 0.0) - 1.5 * np.maximum(prices - k2, 0.0)) / prices
        return lambda prices: (np.maximum(k1 - prices, 0.0) - 1.5 * np.maximum(k2 - prices, 0.0)) / prices

    if structure_id == "1x2_spread":
        k1, k2 = strikes[:2]
        if is_call:
            return lambda prices: (np.maximum(prices - k1, 0.0) - 2.0 * np.maximum(prices - k2, 0.0)) / prices
        return lambda prices: (np.maximum(k1 - prices, 0.0) - 2.0 * np.maximum(k2 - prices, 0.0)) / prices

    if structure_id == "seagull":
        k1, k2, k3 = strikes[:3]
        ratio = wing_ratio or 0.0
        if is_call:
            return lambda prices: (
                np.maximum(prices - k1, 0.0)
                - np.maximum(prices - k2, 0.0)
                - ratio * np.maximum(k3 - prices, 0.0)
            ) / prices
        return lambda prices: (
            np.maximum(k1 - prices, 0.0)
            - np.maximum(k2 - prices, 0.0)
            - ratio * np.maximum(prices - k3, 0.0)
        ) / prices

    if structure_id == "european_digital":
        strike = strikes[0]
        if is_call:
            return lambda prices: np.where(prices > strike, entry_spot / prices, 0.0)
        return lambda prices: np.where(prices < strike, entry_spot / prices, 0.0)

    if structure_id == "european_digital_rko":
        strike = strikes[0]
        if barrier is None:
            raise ValueError("Digital RKO payoff requires a barrier")
        if is_call:
            return lambda prices: np.where(
                (prices > strike) & (prices < barrier),
                entry_spot / prices,
                0.0,
            )
        return lambda prices: np.where(
            (prices < strike) & (prices > barrier),
            entry_spot / prices,
            0.0,
        )

    if structure_id == "european_rko":
        strike = strikes[0]
        if barrier is None:
            raise ValueError("European RKO payoff requires a barrier")
        if is_call:
            return lambda prices: np.where(
                prices < barrier,
                np.maximum(prices - strike, 0.0) / prices,
                0.0,
            )
        return lambda prices: np.where(
            prices > barrier,
            np.maximum(strike - prices, 0.0) / prices,
            0.0,
        )

    raise ValueError(f"Unsupported structure for Kelly payoff bridge: {structure_id}")


def expected_payoff(dist: Distribution, payoff: PayoffFn) -> float:
    return float(np.dot(dist.probs, payoff(dist.bins)))


def price_option(
    dist: Distribution,
    payoff: PayoffFn,
    discount_factor: float = 1.0,
) -> float:
    if not (0.0 < discount_factor <= 1.0):
        raise ValueError(
            f"discount_factor must lie in (0, 1]; got {discount_factor}"
        )
    return discount_factor * expected_payoff(dist, payoff)


def price_vanilla(
    dist: Distribution,
    strike: float,
    is_call: bool,
    discount_factor: float = 1.0,
) -> float:
    payoff = call_payoff(strike) if is_call else put_payoff(strike)
    return price_option(dist, payoff, discount_factor)


def forward_of(dist: Distribution) -> float:
    """Expected spot under the distribution — the implied forward."""
    return float(np.dot(dist.probs, dist.bins))
