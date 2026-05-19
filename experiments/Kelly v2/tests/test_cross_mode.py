"""
Cross-mode sanity: the same underlying belief encoded in Option 1 (CDF mode)
versus Option 2 (PDF mode) should price options consistently.

The two modes have different default tail-coverage policies:
- Option 1 default anchors span [p_2, p_98] of baseline (4% truncated).
- Option 2 default sigma-buckets span ±2.5σ (~1% truncated).

So a tight equivalence test must use matched coverage. We use:
- Option 1 with very wide quantiles (q in [0.001, 0.999]).
- Option 2 with matching sigma extent (±~3.09σ for 99.8% coverage).

Under matched coverage, pricing under the two modes must agree tightly.
"""

import numpy as np
import pytest
from scipy.stats import norm

from baseline import black_scholes_vanilla, synthetic_lognormal_baseline
from edge import anchors_from_baseline
from elicitation import (
    DEFAULT_OPTION1_QUANTILES,
    default_sigma_boundaries,
    elicit_from_cdf_anchors,
    elicit_from_pdf_buckets,
    sigma_boundaries_to_prices,
)
from pricing import price_vanilla


F, SIGMA, T = 5.0, 0.10, 0.25


def _baseline():
    return synthetic_lognormal_baseline(forward=F, sigma=SIGMA, tenor_years=T, n_bins=400, n_stdev=8.0)


def _bucket_masses_from(dist, boundaries):
    masses = []
    for i in range(len(boundaries) - 1):
        mask = (dist.bins >= boundaries[i]) & (dist.bins < boundaries[i + 1])
        masses.append(float(dist.probs[mask].sum()))
    return np.array(masses)


def test_cross_mode_pricing_matches_under_matched_coverage():
    """
    Encode the same lognormal belief in Option 1 with wide quantiles and
    Option 2 with matched sigma extent. Vanilla pricing should agree tightly.
    """
    base = _baseline()

    # Option 1: very wide quantiles → small truncation
    q_wide = np.array([0.001, 0.01, 0.10, 0.50, 0.90, 0.99, 0.999])
    anchors = anchors_from_baseline(base, q_wide)
    pm_opt1 = elicit_from_cdf_anchors(anchors, q_wide)

    # Option 2: matching sigma extent. q=0.001 corresponds to sigma=ppf(0.001) ≈ −3.09.
    sigma_extent = -norm.ppf(0.001)
    offsets = default_sigma_boundaries(7, sigma_extent=sigma_extent)
    boundaries = sigma_boundaries_to_prices(offsets, F, SIGMA, T)
    masses = _bucket_masses_from(base, boundaries)
    masses = masses / masses.sum()
    pm_opt2 = elicit_from_pdf_buckets(boundaries, masses)

    for strike in [4.5, 4.8, 5.0, 5.2, 5.5]:
        for is_call in [True, False]:
            p1 = price_vanilla(pm_opt1, strike, is_call)
            p2 = price_vanilla(pm_opt2, strike, is_call)
            # Tolerance: 20 bp of forward. The two modes have different anchor
            # placements (quantile-spaced vs sigma-spaced) so PCHIP shape error
            # differs; absolute agreement to ~20 bp is realistic.
            assert abs(p1 - p2) < 20e-4 * F, (
                f"K={strike} {'call' if is_call else 'put'}: opt1={p1:.6f}, opt2={p2:.6f}, "
                f"diff={p1 - p2:.6f}"
            )


def test_both_modes_track_baseline_within_their_policy():
    """
    Default coverage settings: Option 1 should under-price (drops more tail),
    Option 2 should be closer to closed-form (drops less).
    """
    base = _baseline()

    # Option 1 with default 7 anchors
    anchors = anchors_from_baseline(base, DEFAULT_OPTION1_QUANTILES)
    pm_opt1 = elicit_from_cdf_anchors(anchors, DEFAULT_OPTION1_QUANTILES)

    # Option 2 with default sigma 7 buckets
    offsets = default_sigma_boundaries(7)
    boundaries = sigma_boundaries_to_prices(offsets, F, SIGMA, T)
    masses = _bucket_masses_from(base, boundaries)
    masses = masses / masses.sum()
    pm_opt2 = elicit_from_pdf_buckets(boundaries, masses)

    closed = black_scholes_vanilla(F, F, SIGMA, T, is_call=True)
    p1 = price_vanilla(pm_opt1, F, is_call=True)
    p2 = price_vanilla(pm_opt2, F, is_call=True)

    # Both below the closed-form due to truncated tail mass.
    assert p1 < closed
    assert p2 < closed
    # Option 2's truncation is smaller, so it's closer.
    assert abs(p2 - closed) < abs(p1 - closed)


@pytest.mark.parametrize("n", [5, 7, 11])
def test_cross_mode_consistency_parametric_over_n(n):
    """Cross-mode consistency holds across N in {5, 7, 11}."""
    base = _baseline()

    # Option 1 with wide-coverage quantiles, N anchors
    q = np.linspace(0.001, 0.999, n)
    anchors = anchors_from_baseline(base, q)
    pm_opt1 = elicit_from_cdf_anchors(anchors, q)

    # Option 2 with matched extent, N buckets
    sigma_extent = -norm.ppf(0.001)
    offsets = default_sigma_boundaries(n, sigma_extent=sigma_extent)
    boundaries = sigma_boundaries_to_prices(offsets, F, SIGMA, T)
    masses = _bucket_masses_from(base, boundaries)
    masses = masses / masses.sum()
    pm_opt2 = elicit_from_pdf_buckets(boundaries, masses)

    p1 = price_vanilla(pm_opt1, F, is_call=True)
    p2 = price_vanilla(pm_opt2, F, is_call=True)
    # Tolerance loosens slightly with fewer anchors.
    tol = 30e-4 * F if n < 7 else 20e-4 * F
    assert abs(p1 - p2) < tol, f"N={n}: opt1={p1:.6f}, opt2={p2:.6f}, diff={p1 - p2:.6f}"
