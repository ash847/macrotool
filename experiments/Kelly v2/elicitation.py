"""
Elicit a discrete 200-bin distribution from N anchor points.

Option 1 (CDF mode): N (quantile, price) pairs. PM gives the price at fixed
quantile levels. Implemented here.

Option 2 (PDF mode): N (bucket, probability) pairs. To be added later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator


DEFAULT_OPTION1_QUANTILES: tuple[float, ...] = (0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98)
DEFAULT_N_BINS: int = 200
MIN_ANCHORS: int = 3


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


def elicit_from_cdf_anchors(
    prices: Sequence[float],
    quantiles: Sequence[float] = DEFAULT_OPTION1_QUANTILES,
    n_bins: int = DEFAULT_N_BINS,
) -> Distribution:
    """
    Build a discrete distribution from N (quantile, price) anchors.

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

    cdf_fn = PchipInterpolator(p, q)

    edges = np.linspace(p[0], p[-1], n_bins + 1)
    cdf_at_edges = cdf_fn(edges)
    bin_centres = 0.5 * (edges[:-1] + edges[1:])
    raw_probs = np.clip(np.diff(cdf_at_edges), 0.0, None)

    total = float(raw_probs.sum())
    if total <= 0.0:
        raise ValueError("CDF differences sum to zero; check anchors")
    probs = raw_probs / total

    return Distribution(bins=bin_centres, probs=probs)
