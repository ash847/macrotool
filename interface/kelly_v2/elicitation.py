"""
Elicit a discrete 200-bin distribution from N anchor points.

Option 1 (CDF mode): N (quantile, price) pairs. PM gives the price at fixed
quantile levels. Truncates to outer anchors and renormalises.

Option 2 (PDF mode): N+1 price boundaries + N bucket probabilities. PM gives
the probability mass per bucket. Buckets define the full support; renormali-
sation only absorbs PCHIP shape error.

Both reduce to a CDF-anchor problem internally — see `_distribution_from_cdf_anchors`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator


DEFAULT_OPTION1_QUANTILES: tuple[float, ...] = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98)
DEFAULT_N_BINS: int = 200
MIN_ANCHORS: int = 3
SUM_TO_ONE_TOL: float = 1e-6


@dataclass(frozen=True)
class Distribution:
    bins: np.ndarray   # bin centres (prices), strictly increasing
    probs: np.ndarray  # probability mass per bin, sums to 1

    @property
    def support(self) -> tuple[float, float]:
        return float(self.bins[0]), float(self.bins[-1])

    @property
    def n_bins(self) -> int:
        return int(self.bins.size)


def _distribution_from_cdf_anchors(
    prices: np.ndarray,
    cdf_values: np.ndarray,
    n_bins: int,
) -> Distribution:
    """PCHIP-fit the CDF, sample on n_bins edges, derive bin probabilities."""
    cdf_fn = PchipInterpolator(prices, cdf_values)
    edges = np.linspace(prices[0], prices[-1], n_bins + 1)
    cdf_at_edges = cdf_fn(edges)
    bin_centres = 0.5 * (edges[:-1] + edges[1:])
    raw_probs = np.clip(np.diff(cdf_at_edges), 0.0, None)
    total = float(raw_probs.sum())
    if total <= 0.0:
        raise ValueError("CDF differences sum to zero; check anchors")
    return Distribution(bins=bin_centres, probs=raw_probs / total)


def elicit_from_cdf_anchors(
    prices: Sequence[float],
    quantiles: Sequence[float] = DEFAULT_OPTION1_QUANTILES,
    n_bins: int = DEFAULT_N_BINS,
) -> Distribution:
    """
    Option 1 — Build a discrete distribution from N (quantile, price) anchors.

    Truncates to [prices[0], prices[-1]]; tail mass outside the outer anchors
    is dropped and the remaining mass is renormalised to sum to 1.
    """
    p = np.asarray(prices, dtype=float)
    q = np.asarray(quantiles, dtype=float)

    if p.shape != q.shape:
        raise ValueError(
            f"prices and quantiles must have the same length; got {p.size} vs {q.size}"
        )
    if p.size < MIN_ANCHORS:
        raise ValueError(f"need at least {MIN_ANCHORS} anchors; got {p.size}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2; got {n_bins}")
    if not np.all(np.diff(p) > 0):
        raise ValueError(f"prices must be strictly increasing; got {p.tolist()}")
    if not np.all(np.diff(q) > 0):
        raise ValueError(f"quantiles must be strictly increasing; got {q.tolist()}")
    if q[0] <= 0.0 or q[-1] >= 1.0:
        raise ValueError(
            f"quantiles must lie strictly in (0, 1); got [{q[0]}, {q[-1]}]"
        )

    return _distribution_from_cdf_anchors(p, q, n_bins)


def elicit_from_pdf_buckets(
    boundaries: Sequence[float],
    bucket_probs: Sequence[float],
    n_bins: int = DEFAULT_N_BINS,
) -> Distribution:
    """
    Option 2 — Build a discrete distribution from N+1 price boundaries and N
    bucket probabilities.

    `boundaries[i]` and `boundaries[i+1]` are the lower and upper price edges
    of bucket i. `bucket_probs[i]` is the probability mass in bucket i; they
    must sum to 1 (within float tolerance).

    Internally we form the cumulative CDF at each boundary (starting at 0,
    ending at 1) and pass through the same PCHIP-then-bin engine as Option 1.
    """
    b = np.asarray(boundaries, dtype=float)
    pb = np.asarray(bucket_probs, dtype=float)

    if b.size != pb.size + 1:
        raise ValueError(
            f"need exactly N+1 boundaries for N bucket_probs; got {b.size} boundaries and {pb.size} probs"
        )
    if pb.size < MIN_ANCHORS:
        raise ValueError(f"need at least {MIN_ANCHORS} buckets; got {pb.size}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2; got {n_bins}")
    if not np.all(np.diff(b) > 0):
        raise ValueError(f"boundaries must be strictly increasing; got {b.tolist()}")
    if np.any(pb < 0):
        raise ValueError(f"bucket_probs must be non-negative; got {pb.tolist()}")
    total = float(pb.sum())
    if abs(total - 1.0) > SUM_TO_ONE_TOL:
        raise ValueError(
            f"bucket_probs must sum to 1 (within {SUM_TO_ONE_TOL}); got sum={total}"
        )

    cdf_at_boundaries = np.concatenate([[0.0], np.cumsum(pb)])
    # Snap the last value to exactly 1 to remove float drift from cumsum.
    cdf_at_boundaries[-1] = 1.0
    return _distribution_from_cdf_anchors(b, cdf_at_boundaries, n_bins)


def default_sigma_boundaries(n_buckets: int, sigma_extent: float = 2.5) -> np.ndarray:
    """
    Linearly-spaced σ-anchor boundaries for Option 2.

    Returns N+1 σ-space offsets in [-sigma_extent, +sigma_extent]. Map to price
    via `forward * exp(sigma * sqrt(T) * offset)`.
    """
    if n_buckets < MIN_ANCHORS:
        raise ValueError(f"need at least {MIN_ANCHORS} buckets; got {n_buckets}")
    return np.linspace(-sigma_extent, sigma_extent, n_buckets + 1)


def sigma_boundaries_to_prices(
    sigma_offsets: np.ndarray,
    forward: float,
    sigma: float,
    tenor_years: float,
) -> np.ndarray:
    """Map σ-space offsets to price-space boundaries on the lognormal market smile."""
    return forward * np.exp(sigma_offsets * sigma * np.sqrt(tenor_years))
