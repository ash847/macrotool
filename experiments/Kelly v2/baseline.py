"""
Baseline market distributions.

Synthetic lognormal — closed-form, used for the BS sanity check.
Snapshot loader — reads a saved smile-implied PDF from a self-contained
JSON fixture. The fixture format is documented in fixtures/README.md;
the integration step in the main MacroTool app will export to this format.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from elicitation import Distribution


SNAPSHOT_SCHEMA_VERSION: int = 1


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


def save_snapshot(
    dist: Distribution,
    path: Path | str,
    *,
    pair: str,
    forward: float,
    tenor_years: float,
    source: str,
) -> None:
    """Write a Distribution to JSON in the v1 snapshot schema."""
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "pair": pair,
        "forward": forward,
        "tenor_years": tenor_years,
        "source": source,
        "bins": dist.bins.tolist(),
        "probs": dist.probs.tolist(),
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_snapshot(path: Path | str) -> tuple[Distribution, dict]:
    """
    Load a Distribution and its metadata from a v1 snapshot JSON.
    Returns (distribution, metadata) where metadata excludes the bins/probs arrays.
    """
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {payload.get('schema_version')!r}; "
            f"expected {SNAPSHOT_SCHEMA_VERSION}"
        )
    bins = np.asarray(payload["bins"], dtype=float)
    probs = np.asarray(payload["probs"], dtype=float)
    if bins.shape != probs.shape:
        raise ValueError(f"bins and probs must have the same shape; got {bins.shape} vs {probs.shape}")
    if not np.all(np.diff(bins) > 0):
        raise ValueError("bins must be strictly increasing")
    if np.any(probs < 0):
        raise ValueError("probs must be non-negative")
    total = probs.sum()
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"probs must sum to 1; got {total}")
    dist = Distribution(bins=bins, probs=probs)
    metadata = {k: v for k, v in payload.items() if k not in {"bins", "probs"}}
    return dist, metadata
