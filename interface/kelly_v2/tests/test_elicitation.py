import numpy as np
import pytest

from interface.kelly_v2.elicitation import (
    DEFAULT_OPTION1_QUANTILES,
    Distribution,
    elicit_from_cdf_anchors,
)


def anchor_set(n: int, low: float = 0.02, high: float = 0.98) -> np.ndarray:
    """Equally-spaced quantile anchors in (0, 1) for parametrised tests."""
    return np.linspace(low, high, n)


def linear_prices(quantiles: np.ndarray, p_min: float = 5.0, p_max: float = 6.0) -> np.ndarray:
    """Prices laid out linearly in quantile space — implies a roughly uniform PDF."""
    return p_min + (p_max - p_min) * (quantiles - quantiles[0]) / (quantiles[-1] - quantiles[0])


@pytest.mark.parametrize("n", [5, 7, 11])
def test_output_shape_and_validity(n):
    q = anchor_set(n)
    prices = linear_prices(q)
    dist = elicit_from_cdf_anchors(prices, q)

    assert isinstance(dist, Distribution)
    assert dist.bins.shape == (200,)
    assert dist.probs.shape == (200,)
    assert np.all(dist.probs >= 0.0)
    assert dist.probs.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(dist.bins) > 0)


@pytest.mark.parametrize("n", [5, 7, 11])
def test_support_matches_outer_anchors(n):
    q = anchor_set(n)
    prices = linear_prices(q, p_min=4.0, p_max=8.0)
    dist = elicit_from_cdf_anchors(prices, q)

    lo, hi = dist.support
    # Bin centres are inside the outer edges by half a bin width.
    bin_width = (prices[-1] - prices[0]) / 200
    assert lo == pytest.approx(prices[0] + 0.5 * bin_width)
    assert hi == pytest.approx(prices[-1] - 0.5 * bin_width)


def test_default_quantiles_is_seven_and_works():
    assert len(DEFAULT_OPTION1_QUANTILES) == 7
    prices = np.linspace(5.0, 6.0, 7)
    dist = elicit_from_cdf_anchors(prices)  # default quantiles
    assert dist.n_bins == 200
    assert dist.probs.sum() == pytest.approx(1.0, abs=1e-12)


def test_custom_n_bins():
    q = anchor_set(7)
    prices = linear_prices(q)
    dist = elicit_from_cdf_anchors(prices, q, n_bins=50)
    assert dist.n_bins == 50
    assert dist.probs.sum() == pytest.approx(1.0, abs=1e-12)


def test_median_anchor_recovered():
    """50th-percentile anchor should be near the centre of the cumulative mass."""
    q = np.array([0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98])
    prices = np.array([4.0, 4.5, 4.9, 5.0, 5.1, 5.5, 6.0])
    dist = elicit_from_cdf_anchors(prices, q)

    cum = np.cumsum(dist.probs)
    median_idx = int(np.searchsorted(cum, 0.5))
    median_price = dist.bins[median_idx]
    # The 50%-quantile price was 5.0; after renormalisation and truncation it
    # shifts slightly because the dropped 4% tail mass isn't symmetric across
    # this asymmetric distribution. Wide tolerance is fine — we're testing
    # that the engine doesn't drastically misplace the median.
    assert abs(median_price - 5.0) < 0.15


def test_uniform_quantiles_give_near_uniform_pdf():
    """If prices are linear in quantile, PDF should be approximately flat."""
    q = np.linspace(0.02, 0.98, 7)
    prices = linear_prices(q, p_min=5.0, p_max=6.0)
    dist = elicit_from_cdf_anchors(prices, q)

    # Coefficient of variation of probs should be small for a near-uniform PDF.
    cv = dist.probs.std() / dist.probs.mean()
    assert cv < 0.05


def test_skewed_anchors_produce_skewed_pdf():
    """Anchors bunched on the left should put mass on the left."""
    q = np.array([0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98])
    prices = np.array([5.00, 5.05, 5.10, 5.20, 5.40, 5.80, 6.50])
    dist = elicit_from_cdf_anchors(prices, q)

    median_price = float(np.median(dist.bins))
    left_mass = dist.probs[dist.bins < median_price].sum()
    right_mass = dist.probs[dist.bins >= median_price].sum()
    assert left_mass > right_mass  # tighter on the left -> more mass on the left


def test_rejects_non_monotonic_prices():
    with pytest.raises(ValueError, match="strictly increasing"):
        elicit_from_cdf_anchors([5.0, 4.9, 5.5, 6.0, 6.5, 7.0, 7.5])


def test_rejects_non_monotonic_quantiles():
    with pytest.raises(ValueError, match="strictly increasing"):
        elicit_from_cdf_anchors(
            [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
            [0.02, 0.10, 0.30, 0.25, 0.75, 0.90, 0.98],
        )


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        elicit_from_cdf_anchors([5.0, 6.0, 7.0], [0.1, 0.5, 0.9, 0.99])


def test_rejects_too_few_anchors():
    with pytest.raises(ValueError, match="at least"):
        elicit_from_cdf_anchors([5.0, 6.0], [0.1, 0.9])


def test_rejects_quantiles_out_of_range():
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        elicit_from_cdf_anchors([5.0, 6.0, 7.0], [0.0, 0.5, 0.9])
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        elicit_from_cdf_anchors([5.0, 6.0, 7.0], [0.1, 0.5, 1.0])


def test_rejects_too_few_bins():
    q = anchor_set(7)
    prices = linear_prices(q)
    with pytest.raises(ValueError, match="n_bins"):
        elicit_from_cdf_anchors(prices, q, n_bins=1)
