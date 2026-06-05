import numpy as np
import pytest

from interface.kelly_v2.elicitation import (
    DEFAULT_N_BINS,
    Distribution,
    default_sigma_boundaries,
    elicit_from_pdf_buckets,
    sigma_boundaries_to_prices,
)


def uniform_buckets(n: int) -> np.ndarray:
    """Equal-mass buckets summing to 1."""
    return np.full(n, 1.0 / n)


def linear_boundaries(n_buckets: int, lo: float = 4.0, hi: float = 6.0) -> np.ndarray:
    return np.linspace(lo, hi, n_buckets + 1)


@pytest.mark.parametrize("n", [5, 7, 11])
def test_output_shape_and_validity(n):
    dist = elicit_from_pdf_buckets(linear_boundaries(n), uniform_buckets(n))
    assert isinstance(dist, Distribution)
    assert dist.bins.shape == (DEFAULT_N_BINS,)
    assert dist.probs.shape == (DEFAULT_N_BINS,)
    assert np.all(dist.probs >= 0.0)
    assert dist.probs.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(dist.bins) > 0)


def test_support_matches_outer_boundaries():
    boundaries = np.array([4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5])  # 7 buckets
    dist = elicit_from_pdf_buckets(boundaries, uniform_buckets(7))
    lo, hi = dist.support
    bin_width = (boundaries[-1] - boundaries[0]) / DEFAULT_N_BINS
    assert lo == pytest.approx(boundaries[0] + 0.5 * bin_width)
    assert hi == pytest.approx(boundaries[-1] - 0.5 * bin_width)


@pytest.mark.parametrize("n", [5, 7, 11])
def test_bucket_probabilities_approximately_recovered(n):
    """The mass in each bucket of the resulting distribution should be close
    to the input bucket probability. PCHIP smoothing means we don't get exact
    recovery but it should be tight (within ~1% of the per-bucket prob).
    """
    rng = np.random.default_rng(seed=42)
    boundaries = linear_boundaries(n)
    raw = rng.uniform(0.1, 1.0, size=n)
    bp = raw / raw.sum()

    dist = elicit_from_pdf_buckets(boundaries, bp, n_bins=400)

    for i in range(n):
        mask = (dist.bins >= boundaries[i]) & (dist.bins < boundaries[i + 1])
        recovered = dist.probs[mask].sum()
        assert recovered == pytest.approx(bp[i], abs=0.02), (
            f"bucket {i}: input {bp[i]:.4f}, recovered {recovered:.4f}"
        )


def test_skewed_buckets_shift_mass():
    """Loading the right-hand buckets concentrates mass on the right."""
    boundaries = linear_boundaries(7)
    bp = np.array([0.01, 0.01, 0.02, 0.05, 0.10, 0.30, 0.51])
    dist = elicit_from_pdf_buckets(boundaries, bp)

    mid = 0.5 * (boundaries[0] + boundaries[-1])
    left = dist.probs[dist.bins < mid].sum()
    right = dist.probs[dist.bins >= mid].sum()
    assert right > left * 4.0


# --- validation ---


def test_rejects_mismatched_lengths():
    # 5 boundaries with 3 probs — should be 4 probs.
    with pytest.raises(ValueError, match="N\\+1 boundaries"):
        elicit_from_pdf_buckets([4.0, 5.0, 6.0, 7.0, 8.0], [1.0 / 3, 1.0 / 3, 1.0 / 3])


def test_rejects_non_monotonic_boundaries():
    with pytest.raises(ValueError, match="strictly increasing"):
        elicit_from_pdf_buckets([4.0, 5.0, 4.5, 6.0], [0.3, 0.3, 0.4])


def test_rejects_negative_bucket_probs():
    boundaries = linear_boundaries(3)
    with pytest.raises(ValueError, match="non-negative"):
        elicit_from_pdf_buckets(boundaries, [0.5, -0.1, 0.6])


def test_rejects_bucket_probs_not_summing_to_one():
    boundaries = linear_boundaries(3)
    with pytest.raises(ValueError, match="sum to 1"):
        elicit_from_pdf_buckets(boundaries, [0.3, 0.3, 0.3])
    with pytest.raises(ValueError, match="sum to 1"):
        elicit_from_pdf_buckets(boundaries, [0.4, 0.4, 0.4])


def test_rejects_too_few_buckets():
    with pytest.raises(ValueError, match="at least"):
        elicit_from_pdf_buckets([4.0, 5.0, 6.0], [0.5, 0.5])


def test_accepts_sum_within_float_tolerance():
    """Tiny float drift in the sum should be tolerated."""
    boundaries = linear_boundaries(3)
    bp = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])  # not exactly 1
    dist = elicit_from_pdf_buckets(boundaries, bp)
    assert dist.probs.sum() == pytest.approx(1.0, abs=1e-12)


# --- σ-anchored boundaries ---


@pytest.mark.parametrize("n", [5, 7, 11])
def test_default_sigma_boundaries_shape_and_symmetry(n):
    offsets = default_sigma_boundaries(n)
    assert offsets.shape == (n + 1,)
    assert offsets[0] == -offsets[-1]
    assert np.all(np.diff(offsets) > 0)


def test_sigma_boundaries_to_prices_lognormal():
    """+/- 1 σ should map to F * exp(+/- σ√T)."""
    F, sigma, T = 5.0, 0.10, 0.25
    offsets = np.array([-1.0, 0.0, 1.0])
    prices = sigma_boundaries_to_prices(offsets, F, sigma, T)
    expected = F * np.exp(np.array([-1, 0, 1]) * sigma * np.sqrt(T))
    np.testing.assert_allclose(prices, expected, rtol=1e-12)


def test_default_sigma_boundaries_validation():
    with pytest.raises(ValueError, match="at least"):
        default_sigma_boundaries(2)
