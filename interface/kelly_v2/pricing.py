"""
Vanilla option pricing on a discrete distribution.

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
