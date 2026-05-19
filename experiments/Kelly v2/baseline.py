"""
Baseline market distributions.

For v2 we support a synthetic lognormal baseline (closed-form, used for the
Black-Scholes sanity check). A real smile-implied baseline from a saved
MacroTool snapshot is added in build step 9.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from elicitation import Distribution


def synthetic_lognormal_baseline(
    forward: float,
    sigma: float,
    tenor_years: float,
    n_bins: int = 200,
    n_stdev: float = 5.0,
) -> Distribution:
    """
    Lognormal distribution centred on the forward.

    Grid spans ±n_stdev of log-spot around log(F); bin centres are geometric
    midpoints of the log-grid (i.e. midpoints of the log-edges, exponentiated).
    Truncation tail mass is renormalised away.
    """
    if forward <= 0:
        raise ValueError(f"forward must be positive; got {forward}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive; got {sigma}")
    if tenor_years <= 0:
        raise ValueError(f"tenor_years must be positive; got {tenor_years}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2; got {n_bins}")

    sqrt_T = np.sqrt(tenor_years)
    log_F = np.log(forward)
    mu_log = log_F - 0.5 * sigma**2 * tenor_years
    s_log = sigma * sqrt_T

    log_edges = np.linspace(log_F - n_stdev * s_log, log_F + n_stdev * s_log, n_bins + 1)
    cdf_at_edges = norm.cdf(log_edges, loc=mu_log, scale=s_log)
    raw_probs = np.diff(cdf_at_edges)
    probs = raw_probs / raw_probs.sum()

    bin_centres = np.exp(0.5 * (log_edges[:-1] + log_edges[1:]))
    return Distribution(bins=bin_centres, probs=probs)


def black_scholes_vanilla(
    forward: float,
    strike: float,
    sigma: float,
    tenor_years: float,
    is_call: bool,
    discount_factor: float = 1.0,
) -> float:
    """Black-Scholes price of a vanilla on the forward (Black '76 form)."""
    if forward <= 0 or sigma <= 0 or tenor_years <= 0:
        raise ValueError("forward, sigma, tenor_years must all be positive")
    if not (0.0 < discount_factor <= 1.0):
        raise ValueError(f"discount_factor must lie in (0, 1]; got {discount_factor}")
    if strike <= 0:
        # Degenerate: a call with K=0 pays the spot; a put with K=0 pays 0.
        return discount_factor * forward if is_call else 0.0

    sqrt_T = np.sqrt(tenor_years)
    d1 = (np.log(forward / strike) + 0.5 * sigma**2 * tenor_years) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if is_call:
        return discount_factor * (forward * norm.cdf(d1) - strike * norm.cdf(d2))
    return discount_factor * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1))
