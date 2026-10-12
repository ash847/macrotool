"""Test-only explicit Kelly curves. Production code never synthesises an edge; tests
that need one state it here explicitly (a lognormal at a stated median)."""
import math

import numpy as np


def stated_lognormal(median: float, vol: float, T: float, n_bins: int = 41,
                     sigma_extent: float = 4.0):
    sig = max(vol * math.sqrt(T), 1e-6)
    bins = np.linspace(median * math.exp(-sigma_extent * sig),
                       median * math.exp(sigma_extent * sig), n_bins)
    z = (np.log(bins) - math.log(median)) / sig
    dens = np.exp(-0.5 * z * z) / bins
    probs = dens / dens.sum()
    return tuple(float(p) for p in probs), tuple(float(b) for b in bins)
