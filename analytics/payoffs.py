"""Terminal payoff functions per structure family — pure, market-independent.

Each `base_ccy_payoff_for_trade_rec` payoff is **per 1 unit of structure notional**
in BASE-currency units, matching the `*_pct` fields on `PricedVariant` (which are
base-ccy fractions). These are terminal (expiry) payoffs over a vector of terminal
spots, used by:
  - Kelly sizing (`analytics/sizing.py`) — integrate against the PM's distribution.
  - The Kelly Sizing screen (`interface/kelly_v2/pricing.py` re-exports these).

Lives in `analytics/` (not `interface/`) so the engine can size without importing
the UI layer. `interface/kelly_v2/pricing.py` re-exports the payoff constructors for
back-compat; the distribution-based helpers (expected_payoff / price_*) stay there.

Convention notes (must match `analytics/structure_pricer.py`):
  - european_digital is a **base-ccy cash-or-nothing** trade: terminal payoff is
    1.0 (100% of notional) if ITM, else 0.0 — NOT the legacy spot/target basis.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

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
    """Terminal base-ccy payoff (per 1 unit of structure notional) for one variant.

    ``entry_spot`` is retained for signature back-compat; the digital families now
    settle base-ccy cash-or-nothing and ignore it.
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

    if structure_id == "1x2x1_spread":
        k1, k2, k3 = strikes[:3]   # K3 = 2·K2 − K1 (equidistant long wing)
        if is_call:
            return lambda prices: (
                np.maximum(prices - k1, 0.0)
                - 2.0 * np.maximum(prices - k2, 0.0)
                + np.maximum(prices - k3, 0.0)
            ) / prices
        return lambda prices: (
            np.maximum(k1 - prices, 0.0)
            - 2.0 * np.maximum(k2 - prices, 0.0)
            + np.maximum(k3 - prices, 0.0)
        ) / prices

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
        # Base-ccy cash-or-nothing: 1.0 (100% of notional) if ITM, else 0.0.
        strike = strikes[0]
        if is_call:
            return lambda prices: np.where(prices > strike, 1.0, 0.0)
        return lambda prices: np.where(prices < strike, 1.0, 0.0)

    if structure_id == "european_digital_rko":
        # Base-ccy cash-or-nothing within the live band (gated/disabled family).
        strike = strikes[0]
        if barrier is None:
            raise ValueError("Digital RKO payoff requires a barrier")
        if is_call:
            return lambda prices: np.where((prices > strike) & (prices < barrier), 1.0, 0.0)
        return lambda prices: np.where((prices < strike) & (prices > barrier), 1.0, 0.0)

    if structure_id == "european_rko":
        strike = strikes[0]
        if barrier is None:
            raise ValueError("European RKO payoff requires a barrier")
        if is_call:
            return lambda prices: np.where(prices < barrier, np.maximum(prices - strike, 0.0) / prices, 0.0)
        return lambda prices: np.where(prices > barrier, np.maximum(strike - prices, 0.0) / prices, 0.0)

    raise ValueError(f"Unsupported structure for payoff bridge: {structure_id}")
