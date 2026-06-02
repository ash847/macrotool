"""
Edge calculation: compare PM's subjective distribution against a market
baseline by pricing the same option under both and reporting the difference.

The edge is **vs market-implied** pricing — not pure forecasting edge — since
the baseline is risk-neutral. See PLAN.md for the labelling decision.

We compute three numbers that decompose the total edge:

    full_edge      = price(pm) - price(market)
    view_edge      = price(pm) - price(shadow_market)
    anchoring_cost = price(shadow_market) - price(market)

    full_edge = view_edge + anchoring_cost

where `shadow_market` is the market distribution re-elicited through PM's
own anchor/bucket scheme. The shadow cancels truncation and PCHIP smoothing,
isolating pure view-divergence in `view_edge`. The `anchoring_cost` is the
pricing impact of dropping the tails outside PM's outer anchors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .elicitation import (
    Distribution,
    elicit_from_cdf_anchors,
    elicit_from_pdf_buckets,
)
from .pricing import PayoffFn, price_option, price_vanilla


@dataclass(frozen=True)
class EdgeReport:
    pm_price: float
    mkt_price: float
    shadow_price: float
    full_edge: float
    view_edge: float
    anchoring_cost: float
    full_edge_pct_of_mid: float | None
    view_edge_pct_of_mid: float | None
    out_of_range: bool


def compute_edge(
    pm_dist: Distribution,
    mkt_dist: Distribution,
    shadow_dist: Distribution | None = None,
    strike: float = 0.0,
    is_call: bool = True,
    discount_factor: float = 1.0,
) -> EdgeReport:
    # If no shadow is supplied, fall back to the market itself — that gives
    # anchoring_cost == 0 and view_edge == full_edge.
    if shadow_dist is None:
        shadow_dist = mkt_dist
    pm_price = price_vanilla(pm_dist, strike, is_call, discount_factor)
    mkt_price = price_vanilla(mkt_dist, strike, is_call, discount_factor)
    shadow_price = price_vanilla(shadow_dist, strike, is_call, discount_factor)

    full_edge = pm_price - mkt_price
    view_edge = pm_price - shadow_price
    anchoring_cost = shadow_price - mkt_price

    if mkt_price > 1e-10:
        full_pct = full_edge / mkt_price * 100.0
        view_pct = view_edge / mkt_price * 100.0
    else:
        full_pct = None
        view_pct = None

    lo, hi = pm_dist.support
    out_of_range = strike < lo or strike > hi

    return EdgeReport(
        pm_price=pm_price,
        mkt_price=mkt_price,
        shadow_price=shadow_price,
        full_edge=full_edge,
        view_edge=view_edge,
        anchoring_cost=anchoring_cost,
        full_edge_pct_of_mid=full_pct,
        view_edge_pct_of_mid=view_pct,
        out_of_range=out_of_range,
    )


def compute_edge_for_payoff(
    pm_dist: Distribution,
    mkt_dist: Distribution,
    payoff: PayoffFn,
    shadow_dist: Distribution | None = None,
    discount_factor: float = 1.0,
    reference_levels: list[float] | None = None,
) -> EdgeReport:
    """Generic edge decomposition for an arbitrary payoff function."""
    if shadow_dist is None:
        shadow_dist = mkt_dist
    pm_price = price_option(pm_dist, payoff, discount_factor)
    mkt_price = price_option(mkt_dist, payoff, discount_factor)
    shadow_price = price_option(shadow_dist, payoff, discount_factor)

    full_edge = pm_price - mkt_price
    view_edge = pm_price - shadow_price
    anchoring_cost = shadow_price - mkt_price

    if mkt_price > 1e-10:
        full_pct = full_edge / mkt_price * 100.0
        view_pct = view_edge / mkt_price * 100.0
    else:
        full_pct = None
        view_pct = None

    lo, hi = pm_dist.support
    levels = reference_levels or []
    out_of_range = any(level < lo or level > hi for level in levels)

    return EdgeReport(
        pm_price=pm_price,
        mkt_price=mkt_price,
        shadow_price=shadow_price,
        full_edge=full_edge,
        view_edge=view_edge,
        anchoring_cost=anchoring_cost,
        full_edge_pct_of_mid=full_pct,
        view_edge_pct_of_mid=view_pct,
        out_of_range=out_of_range,
    )


def shadow_market_from_cdf_anchors(
    mkt_dist: Distribution,
    quantiles: np.ndarray | list[float],
    n_bins: int | None = None,
) -> Distribution:
    """
    Option 1 shadow: extract market prices at PM's quantile anchors and feed
    them back through `elicit_from_cdf_anchors`. The result is the market view
    re-elicited under PM's own scheme — same truncation, same PCHIP smoothing.
    """
    anchors = anchors_from_baseline(mkt_dist, quantiles)
    kwargs = {"n_bins": n_bins} if n_bins is not None else {}
    return elicit_from_cdf_anchors(anchors, quantiles, **kwargs)


def shadow_market_from_pdf_buckets(
    mkt_dist: Distribution,
    boundaries: np.ndarray,
    n_bins: int | None = None,
) -> Distribution:
    """
    Option 2 shadow: read the market's mass in each of PM's σ-buckets and feed
    those back through `elicit_from_pdf_buckets`. The result is the market
    re-elicited under PM's bucket scheme.
    """
    boundaries = np.asarray(boundaries, dtype=float)
    masses = np.array([
        float(mkt_dist.probs[(mkt_dist.bins >= boundaries[i]) & (mkt_dist.bins < boundaries[i + 1])].sum())
        for i in range(len(boundaries) - 1)
    ])
    total = masses.sum()
    if total <= 0:
        raise ValueError("market distribution has no mass within the supplied boundaries")
    masses = masses / total
    kwargs = {"n_bins": n_bins} if n_bins is not None else {}
    return elicit_from_pdf_buckets(boundaries, masses, **kwargs)


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
