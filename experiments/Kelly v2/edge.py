"""
Edge calculation: compare PM's subjective distribution against a market
baseline by pricing the same option under both and reporting the difference.

The edge is **vs market-implied** pricing — not pure forecasting edge — since
the baseline is risk-neutral. See PLAN.md for the labelling decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from elicitation import Distribution
from pricing import price_vanilla


@dataclass(frozen=True)
class EdgeReport:
    pm_price: float
    mkt_price: float
    edge_absolute: float
    edge_pct_of_mid: float | None
    out_of_range: bool


def compute_edge(
    pm_dist: Distribution,
    mkt_dist: Distribution,
    strike: float,
    is_call: bool,
    discount_factor: float = 1.0,
) -> EdgeReport:
    pm_price = price_vanilla(pm_dist, strike, is_call, discount_factor)
    mkt_price = price_vanilla(mkt_dist, strike, is_call, discount_factor)
    edge_abs = pm_price - mkt_price

    edge_pct = (edge_abs / mkt_price * 100.0) if mkt_price > 1e-10 else None

    lo, hi = pm_dist.support
    out_of_range = strike < lo or strike > hi

    return EdgeReport(
        pm_price=pm_price,
        mkt_price=mkt_price,
        edge_absolute=edge_abs,
        edge_pct_of_mid=edge_pct,
        out_of_range=out_of_range,
    )


def quantile(dist: Distribution, q: float) -> float:
    """Inverse CDF of a discrete distribution by linear interpolation between bins."""
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must lie in (0, 1); got {q}")
    cum = np.cumsum(dist.probs)
    # Linear interp on (cum, bins). cum is monotone non-decreasing.
    return float(np.interp(q, cum, dist.bins))


def anchors_from_baseline(
    mkt_dist: Distribution,
    quantiles: np.ndarray | list[float],
) -> np.ndarray:
    """Extract anchor prices from a baseline distribution at the given quantiles."""
    return np.array([quantile(mkt_dist, q) for q in quantiles])
