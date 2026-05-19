import numpy as np
import pytest

from baseline import synthetic_lognormal_baseline
from edge import (
    anchors_from_baseline,
    compute_edge,
    quantile,
    shadow_market_from_cdf_anchors,
    shadow_market_from_pdf_buckets,
)
from elicitation import (
    DEFAULT_OPTION1_QUANTILES,
    default_sigma_boundaries,
    elicit_from_cdf_anchors,
    elicit_from_pdf_buckets,
    sigma_boundaries_to_prices,
)


def _baseline(F=5.0, sigma=0.10, T=0.25):
    return synthetic_lognormal_baseline(forward=F, sigma=sigma, tenor_years=T, n_bins=400, n_stdev=8.0)


def test_compute_edge_returns_zero_when_distributions_identical():
    dist = _baseline()
    rep = compute_edge(dist, dist, strike=5.0, is_call=True, discount_factor=0.97)
    assert rep.full_edge == pytest.approx(0.0, abs=1e-12)
    assert rep.pm_price == pytest.approx(rep.mkt_price, abs=1e-12)
    assert rep.out_of_range is False


def test_out_of_range_flag_set_for_strike_outside_support():
    F = 5.0
    base = _baseline(F=F)
    quantiles = DEFAULT_OPTION1_QUANTILES
    anchors = anchors_from_baseline(base, quantiles)
    pm = elicit_from_cdf_anchors(anchors, quantiles)

    lo, hi = pm.support
    out_strike = hi + 0.5 * (hi - lo)
    rep = compute_edge(pm, base, strike=out_strike, is_call=True)
    assert rep.out_of_range is True

    in_strike = 0.5 * (lo + hi)
    rep_in = compute_edge(pm, base, strike=in_strike, is_call=True)
    assert rep_in.out_of_range is False


def test_edge_pct_none_when_market_price_near_zero():
    base = _baseline(F=5.0)
    # Strike well above support — both call prices ~ 0.
    rep = compute_edge(base, base, strike=100.0, is_call=True)
    assert rep.full_edge_pct_of_mid is None


# --- Sanity check 1: zero-edge identity ---
# PLAN.md called for edge < 1 bp when PM anchors are extracted from the
# baseline. With the truncate-to-anchors policy this is unreachable — PM drops
# the ~4% of mass outside [p_2, p_98] and renormalises. See NOTES.md "Step 4"
# for the full derivation. We split the identity test in two:
#   * wide-anchor variant — isolates engine error (truncation negligible)
#   * default-anchor variant — accepts truncation-induced edge as expected


@pytest.mark.parametrize(
    "F, sigma, T, K, is_call",
    [
        (5.0, 0.10, 0.25, 5.0, True),
        (5.0, 0.10, 0.25, 4.7, False),
        (5.0, 0.10, 0.25, 5.3, True),
        (5.0, 0.20, 1.00, 5.0, True),
        (100.0, 0.15, 0.50, 100.0, True),
    ],
)
def test_engine_identity_with_wide_anchors(F, sigma, T, K, is_call):
    """Engine identity: with very wide anchors (q in [0.001, 0.999]), truncation
    drops < 0.2% of mass and edge reflects engine error only."""
    base = synthetic_lognormal_baseline(forward=F, sigma=sigma, tenor_years=T, n_bins=400, n_stdev=8.0)
    quantiles = np.array([0.001, 0.01, 0.10, 0.50, 0.90, 0.99, 0.999])
    anchors = anchors_from_baseline(base, quantiles)
    pm = elicit_from_cdf_anchors(anchors, quantiles)

    rep = compute_edge(pm, base, strike=K, is_call=is_call, discount_factor=0.97)
    tol = 1e-3 * F  # 10 bp of forward
    assert abs(rep.full_edge) < tol, (
        f"edge {rep.full_edge:.6f} exceeds {tol:.6f} "
        f"(PM={rep.pm_price:.6f}, mkt={rep.mkt_price:.6f})"
    )


def test_default_anchor_identity_edge_is_bounded_and_directional():
    """With default [2, 98] anchors, truncation creates a small but bounded edge.
    For an ATM call, the dropped upper tail (above p_98) was net positive — so
    PM should price BELOW baseline (negative edge)."""
    F = 5.0
    base = _baseline(F=F)
    quantiles = DEFAULT_OPTION1_QUANTILES
    anchors = anchors_from_baseline(base, quantiles)
    pm = elicit_from_cdf_anchors(anchors, quantiles)

    rep = compute_edge(pm, base, strike=F, is_call=True)
    assert rep.full_edge < 0.0
    assert abs(rep.full_edge) < 0.02 * F  # within 2% of forward


def test_zero_edge_tightens_as_anchor_coverage_widens():
    """Widening anchor quantiles toward (0, 1) shrinks the truncation-induced edge."""
    base = _baseline()
    K = 5.0

    edges_by_coverage = []
    for outer in [0.02, 0.01, 0.005, 0.001]:
        qs = np.array([outer, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0 - outer])
        anchors = anchors_from_baseline(base, qs)
        pm = elicit_from_cdf_anchors(anchors, qs)
        rep = compute_edge(pm, base, strike=K, is_call=True)
        edges_by_coverage.append(abs(rep.full_edge))

    # Edge must shrink as we widen the anchor coverage.
    assert edges_by_coverage[-1] < edges_by_coverage[0]


# --- Sanity check 5: tail behaviour ---


def test_lower_left_tail_anchor_raises_otm_put_price():
    """Moving the 2% anchor lower should make an OTM put more valuable under PM."""
    base = _baseline()
    quantiles = np.array(DEFAULT_OPTION1_QUANTILES)
    base_anchors = anchors_from_baseline(base, quantiles)

    # Lower the leftmost anchor (heavier left tail in PM's view).
    heavy_left = base_anchors.copy()
    heavy_left[0] -= 0.2  # nudge leftmost anchor down

    pm_baseline = elicit_from_cdf_anchors(base_anchors, quantiles)
    pm_heavy_left = elicit_from_cdf_anchors(heavy_left, quantiles)

    # OTM put strike sits just inside the elicited support of the baseline-PM.
    # Both PMs must contain it (heavy_left support starts even further left, so OK).
    strike = float(base_anchors[1])  # at p_10

    put_baseline = compute_edge(pm_baseline, base, strike=strike, is_call=False).pm_price
    put_heavy_left = compute_edge(pm_heavy_left, base, strike=strike, is_call=False).pm_price

    assert put_heavy_left > put_baseline


# --- Sanity check 6: monotone in obvious direction ---


def test_shifting_pm_distribution_right_increases_call_edge():
    """If PM's whole distribution shifts up by delta, call edge should increase."""
    base = _baseline()
    quantiles = np.array(DEFAULT_OPTION1_QUANTILES)
    base_anchors = anchors_from_baseline(base, quantiles)

    shifted_anchors = base_anchors + 0.1  # uniform right-shift

    pm_baseline = elicit_from_cdf_anchors(base_anchors, quantiles)
    pm_shifted = elicit_from_cdf_anchors(shifted_anchors, quantiles)

    strike = 5.0
    edge_base = compute_edge(pm_baseline, base, strike=strike, is_call=True).full_edge
    edge_shift = compute_edge(pm_shifted, base, strike=strike, is_call=True).full_edge

    assert edge_shift > edge_base


# --- quantile helper ---


def test_quantile_recovers_anchors_on_baseline():
    """Inverse CDF on the baseline near its anchor quantiles."""
    base = _baseline()
    for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
        recovered = quantile(base, q)
        # Compare to closed-form lognormal quantile. F=5, sigma=0.1, T=0.25.
        from scipy.stats import norm

        s = 0.10 * np.sqrt(0.25)
        mu = np.log(5.0) - 0.5 * 0.10**2 * 0.25
        expected = np.exp(mu + s * norm.ppf(q))
        assert recovered == pytest.approx(expected, rel=5e-3)


def test_quantile_validation():
    base = _baseline()
    with pytest.raises(ValueError):
        quantile(base, 0.0)
    with pytest.raises(ValueError):
        quantile(base, 1.0)


# --- shadow market + decomposition ---


def test_edge_decomposition_identity():
    """Full edge == view edge + anchoring cost, by construction."""
    F = 5.0
    base = _baseline(F=F)
    quantiles = DEFAULT_OPTION1_QUANTILES
    anchors = anchors_from_baseline(base, quantiles)
    pm = elicit_from_cdf_anchors(anchors, quantiles)
    shadow = shadow_market_from_cdf_anchors(base, list(quantiles))

    rep = compute_edge(pm, base, shadow, strike=F, is_call=True)
    assert rep.full_edge == pytest.approx(rep.view_edge + rep.anchoring_cost, abs=1e-12)


def test_view_edge_near_zero_when_pm_matches_shadow_option1():
    """If PM's anchors come from the market CDF, PM equals the shadow exactly
    and the view edge is zero — leaving the full edge as the anchoring cost."""
    F = 5.0
    base = _baseline(F=F)
    quantiles = DEFAULT_OPTION1_QUANTILES
    anchors = anchors_from_baseline(base, quantiles)
    pm = elicit_from_cdf_anchors(anchors, quantiles)
    shadow = shadow_market_from_cdf_anchors(base, list(quantiles))

    for K in [4.7, 5.0, 5.3]:
        for is_call in (True, False):
            rep = compute_edge(pm, base, shadow, strike=K, is_call=is_call)
            assert abs(rep.view_edge) < 1e-9
            # Anchoring cost soaks up the full edge.
            assert rep.anchoring_cost == pytest.approx(rep.full_edge, abs=1e-9)


def test_view_edge_near_zero_when_pm_matches_shadow_option2():
    """Same property in Option 2: PM bucket masses = market masses → view edge ≈ 0."""
    F = 5.0
    base = _baseline(F=F)
    offsets = default_sigma_boundaries(7)
    boundaries = sigma_boundaries_to_prices(offsets, F, sigma=0.10, tenor_years=0.25)

    # PM uses the baseline masses in each bucket — should equal the shadow.
    masses = np.array([
        float(base.probs[(base.bins >= boundaries[i]) & (base.bins < boundaries[i + 1])].sum())
        for i in range(len(boundaries) - 1)
    ])
    masses = masses / masses.sum()
    pm = elicit_from_pdf_buckets(boundaries, masses)
    shadow = shadow_market_from_pdf_buckets(base, boundaries)

    for K in [4.7, 5.0, 5.3]:
        for is_call in (True, False):
            rep = compute_edge(pm, base, shadow, strike=K, is_call=is_call)
            assert abs(rep.view_edge) < 1e-9


def test_anchoring_cost_negative_for_atm_call_default_anchors():
    """Default Option 1 anchors drop ~4% tail mass; ATM call anchoring cost is negative."""
    F = 5.0
    base = _baseline(F=F)
    quantiles = DEFAULT_OPTION1_QUANTILES
    anchors = anchors_from_baseline(base, quantiles)
    pm = elicit_from_cdf_anchors(anchors, quantiles)
    shadow = shadow_market_from_cdf_anchors(base, list(quantiles))

    rep = compute_edge(pm, base, shadow, strike=F, is_call=True)
    assert rep.anchoring_cost < 0.0


def test_shadow_market_from_pdf_buckets_rejects_zero_mass():
    """Boundaries that capture no baseline mass should raise."""
    base = _baseline(F=5.0)
    # Boundaries far outside the baseline support
    boundaries = np.array([100.0, 110.0, 120.0, 130.0])
    with pytest.raises(ValueError, match="no mass"):
        shadow_market_from_pdf_buckets(base, boundaries)


def test_default_compute_edge_falls_back_to_market_as_shadow():
    """If no shadow_dist is passed, view_edge == full_edge and anchoring_cost == 0."""
    F = 5.0
    base = _baseline(F=F)
    quantiles = DEFAULT_OPTION1_QUANTILES
    anchors = anchors_from_baseline(base, quantiles)
    pm = elicit_from_cdf_anchors(anchors, quantiles)

    rep = compute_edge(pm, base, strike=F, is_call=True)
    assert rep.anchoring_cost == pytest.approx(0.0, abs=1e-12)
    assert rep.view_edge == pytest.approx(rep.full_edge, abs=1e-12)
