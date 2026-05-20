import numpy as np
import pytest

from baseline import synthetic_lognormal_baseline
from elicitation import Distribution
from kelly import (
    KellyReport,
    SAFETY_F_MAX,
    UNBOUNDED_LOSS_THRESHOLD,
    compute_kelly,
    kelly_continuous,
    kelly_discrete,
    kelly_growth_curve,
)

from pricing import call_payoff, put_payoff


# --- helpers ---


def binary_distribution(p_win: float, win_price: float, lose_price: float) -> Distribution:
    """Two-bin distribution: lose with prob (1-p_win) at lose_price, win at win_price."""
    bins = np.array([lose_price, win_price])
    probs = np.array([1.0 - p_win, p_win])
    return Distribution(bins=bins, probs=probs)


# --- analytical / boundary cases ---


def test_zero_edge_returns_zero_fraction():
    """Fair bet (E[r] = 0): Kelly says don't bet."""
    dist = binary_distribution(p_win=0.5, win_price=6.0, lose_price=4.0)
    # Call at K=5 priced at exactly its fair value under the dist.
    payoff = call_payoff(strike=5.0)
    expected_payoff = 0.5 * 0  # bin 0: max(4-5, 0) = 0
    expected_payoff += 0.5 * 1.0  # bin 1: max(6-5, 0) = 1.0
    cost = expected_payoff  # 0.5

    assert kelly_continuous(dist, payoff, cost=cost) == 0.0
    assert kelly_discrete(dist, payoff, cost=cost) == 0.0


def test_negative_edge_returns_zero_fraction():
    """Overpriced bet (E[r] < 0): Kelly says don't bet."""
    dist = binary_distribution(p_win=0.5, win_price=6.0, lose_price=4.0)
    payoff = call_payoff(strike=5.0)
    cost = 0.6  # PM expects 0.5; market wants 0.6 — losing bet under PM's view

    assert kelly_continuous(dist, payoff, cost=cost) == 0.0
    assert kelly_discrete(dist, payoff, cost=cost) == 0.0


def test_discrete_matches_binary_kelly_formula():
    """For a strictly binary bet, kelly_discrete must match p − q/b."""
    p_win = 0.6
    win_price = 6.0
    lose_price = 4.0
    strike = 5.0
    cost = 0.4  # below the fair value 0.5×1 = 0.5

    dist = binary_distribution(p_win, win_price, lose_price)
    payoff = call_payoff(strike=strike)

    # Returns: lose → -1 (lose premium); win → (1.0 - 0.4)/0.4 = 1.5
    b = (1.0 - cost) / cost  # 1.5
    f_analytic = p_win - (1 - p_win) / b  # 0.6 - 0.4/1.5 = 0.333…

    f_num = kelly_discrete(dist, payoff, cost=cost)
    assert f_num == pytest.approx(f_analytic, abs=1e-4)


def test_higher_variance_lowers_kelly():
    """Two trades with same E[r] but higher Var[r] → smaller f*."""
    # Trade A: deterministic +100% on win, -100% on loss, p_win = 0.6
    dist_a = binary_distribution(p_win=0.6, win_price=6.0, lose_price=4.0)
    payoff_a = call_payoff(strike=5.0)
    cost_a = 0.5  # gives win-return = +100%

    # Trade B: a 3-bin spread on the win side with the same mean win-return
    # but higher variance.
    bins_b = np.array([4.0, 5.5, 6.5])
    # Mean: 0.4*0 + 0.3*0.5 + 0.3*1.5 = 0.6 → cost = 0.5 → E[r] = 0.2
    # Wait — we want the same E[r] as trade A. Trade A: 0.4*(-1) + 0.6*1 = 0.2. Match.
    probs_b = np.array([0.4, 0.3, 0.3])
    dist_b = Distribution(bins=bins_b, probs=probs_b)
    payoff_b = call_payoff(strike=5.0)
    # cost = mean payoff = 0.4*0 + 0.3*0.5 + 0.3*1.5 = 0.6 — but that gives E[r] = 0
    # Use cost = 0.5 to match trade A's expected return at +20%.
    cost_b = 0.5

    f_a = kelly_discrete(dist_a, payoff_a, cost=cost_a)
    f_b = kelly_discrete(dist_b, payoff_b, cost=cost_b)
    assert f_b < f_a


# --- continuous vs discrete agreement ---


def test_continuous_close_to_discrete_for_small_edge():
    """When |r| is small, Thorp's Taylor approximation should agree with discrete."""
    # Binary bet with tiny win/lose magnitudes so |r| stays small.
    # We construct it as a custom Distribution at two bins close together.
    bins = np.array([4.99, 5.01])
    probs = np.array([0.45, 0.55])
    dist = Distribution(bins=bins, probs=probs)

    # Payoff that returns small magnitudes: call at K=5.0
    payoff = call_payoff(strike=5.0)
    # win payoff = 0.01, lose payoff = 0
    # cost slightly below fair value 0.55*0.01 = 0.0055 → modest edge
    cost = 0.0050
    # returns: lose → -1, win → (0.01 - 0.005)/0.005 = 1.0
    # |r| isn't small here actually. Let me use partial fractional payoff instead.

    # Better: use a distribution where both outcomes have positive payoff
    # so the "lose" return isn't -1.
    bins = np.array([5.10, 5.20])
    probs = np.array([0.45, 0.55])
    dist = Distribution(bins=bins, probs=probs)
    payoff = call_payoff(strike=5.0)
    # payoffs: 0.10 and 0.20; fair = 0.155; pick cost giving small edge
    cost = 0.150
    # r: (0.10 - 0.15)/0.15 = -0.333, (0.20 - 0.15)/0.15 = 0.333. |r| ≈ 1/3.

    f_c = kelly_continuous(dist, payoff, cost=cost)
    f_d = kelly_discrete(dist, payoff, cost=cost)
    # Allow 50% relative tolerance — Thorp is an approximation; |r| ≈ 0.33
    # is on the edge of where the Taylor expansion is reliable.
    assert f_c == pytest.approx(f_d, rel=0.5) or abs(f_c - f_d) < 0.2


# --- KellyReport assembly ---


def test_compute_kelly_returns_full_report():
    dist = binary_distribution(p_win=0.6, win_price=6.0, lose_price=4.0)
    payoff = call_payoff(strike=5.0)
    rep = compute_kelly(dist, payoff, cost=0.4, multiplier=0.5)

    assert isinstance(rep, KellyReport)
    assert rep.f_discrete > 0
    assert rep.f_displayed == pytest.approx(rep.f_raw * 0.5, abs=1e-9)
    # r_lose = -1, r_win = (1.0 - 0.4)/0.4 = 1.5 → E[r] = 0.4*(-1) + 0.6*1.5 = 0.5
    expected_r = 0.4 * (-1.0) + 0.6 * 1.5
    assert rep.expected_return == pytest.approx(expected_r, abs=1e-12)


def test_displayed_fraction_respects_multiplier():
    """f_displayed = f_raw × multiplier."""
    dist = binary_distribution(p_win=0.6, win_price=6.0, lose_price=4.0)
    payoff = call_payoff(strike=5.0)
    cost = 0.4

    rep_full = compute_kelly(dist, payoff, cost=cost, multiplier=1.0)
    rep_half = compute_kelly(dist, payoff, cost=cost, multiplier=0.5)

    assert rep_half.f_displayed == pytest.approx(rep_full.f_displayed * 0.5, abs=1e-9)


def test_compute_kelly_validation():
    dist = binary_distribution(p_win=0.6, win_price=6.0, lose_price=4.0)
    payoff = call_payoff(strike=5.0)

    with pytest.raises(ValueError, match="multiplier"):
        compute_kelly(dist, payoff, cost=0.4, multiplier=0.0)
    with pytest.raises(ValueError, match="multiplier"):
        compute_kelly(dist, payoff, cost=0.4, multiplier=1.5)


def test_zero_cost_raises():
    dist = binary_distribution(p_win=0.6, win_price=6.0, lose_price=4.0)
    payoff = call_payoff(strike=5.0)
    with pytest.raises(ValueError, match="cost"):
        kelly_discrete(dist, payoff, cost=0.0)


# --- risk metrics ---


def test_prob_loss_and_total_loss():
    dist = binary_distribution(p_win=0.6, win_price=6.0, lose_price=4.0)
    payoff = call_payoff(strike=5.0)
    rep = compute_kelly(dist, payoff, cost=0.4, multiplier=0.5)

    # 40% of mass below the strike → total loss
    assert rep.prob_loss == pytest.approx(0.4, abs=1e-9)
    assert rep.prob_total_loss == pytest.approx(0.4, abs=1e-9)


def test_expected_log_growth_positive_for_favourable_bet():
    dist = binary_distribution(p_win=0.6, win_price=6.0, lose_price=4.0)
    payoff = call_payoff(strike=5.0)
    rep = compute_kelly(dist, payoff, cost=0.4, multiplier=0.5)
    assert rep.expected_log_growth > 0


def test_expected_log_growth_zero_at_zero_fraction():
    """f_displayed = 0 → log(1) = 0 contribution every bin."""
    dist = binary_distribution(p_win=0.5, win_price=6.0, lose_price=4.0)
    payoff = call_payoff(strike=5.0)
    cost = 0.5  # E[r] = 0 → f_displayed = 0
    rep = compute_kelly(dist, payoff, cost=cost)
    assert rep.f_displayed == 0.0
    assert rep.expected_log_growth == 0.0


# --- growth curve ---


def test_growth_curve_zero_at_origin():
    """E[log(1 + 0·r)] = 0 for any distribution."""
    dist = binary_distribution(p_win=0.6, win_price=6.0, lose_price=4.0)
    payoff = call_payoff(strike=5.0)
    f_vals, geo_vals, er_vals = kelly_growth_curve(dist, payoff, cost=0.4, f_star=0.3)
    assert f_vals[0] == pytest.approx(0.0)
    assert geo_vals[0] == pytest.approx(0.0, abs=1e-10)


def test_growth_curve_peaks_at_f_star():
    """The maximum of the curve should be at approximately f_star."""
    dist = binary_distribution(p_win=0.6, win_price=6.0, lose_price=4.0)
    payoff = call_payoff(strike=5.0)
    cost = 0.4
    f_star = kelly_discrete(dist, payoff, cost=cost)
    f_vals, geo_vals, er_vals = kelly_growth_curve(dist, payoff, cost=cost, f_star=f_star)
    peak_idx = int(np.argmax(geo_vals))
    f_at_peak = f_vals[peak_idx]
    assert f_at_peak == pytest.approx(f_star, abs=0.02)  # within 2% of f*


def test_growth_curve_decreasing_past_peak():
    """Geometric growth should be strictly decreasing well past the peak."""
    dist = binary_distribution(p_win=0.6, win_price=6.0, lose_price=4.0)
    payoff = call_payoff(strike=5.0)
    cost = 0.4
    f_star = kelly_discrete(dist, payoff, cost=cost)
    f_vals, geo_vals, er_vals = kelly_growth_curve(dist, payoff, cost=cost, f_star=f_star)
    # Values in the upper half of the range should be decreasing
    upper_half = geo_vals[len(geo_vals) // 2:]
    assert np.all(np.diff(upper_half) < 0)


def test_er_curve_linear_and_above_geo():
    """E[r] overlay must be linear in f and lie above geometric growth past the peak."""
    dist = binary_distribution(p_win=0.6, win_price=6.0, lose_price=4.0)
    payoff = call_payoff(strike=5.0)
    cost = 0.4
    f_star = kelly_discrete(dist, payoff, cost=cost)
    f_vals, geo_vals, er_vals = kelly_growth_curve(dist, payoff, cost=cost, f_star=f_star)
    # er_curve is linear: second differences should be ~0
    assert np.allclose(np.diff(er_vals, 2), 0.0, atol=1e-10)
    # Past the peak, expected return exceeds geometric return (variance drag)
    past_peak = f_vals > f_star
    assert np.all(er_vals[past_peak] >= geo_vals[past_peak])


# --- put symmetry ---


def test_kelly_works_for_puts():
    """Put with positive edge should also produce f* > 0."""
    dist = binary_distribution(p_win=0.6, win_price=4.0, lose_price=6.0)  # bearish dist
    payoff = put_payoff(strike=5.0)
    # payoff at win (S=4): max(5-4,0) = 1; at lose (S=6): 0
    fair = 0.6 * 1.0  # 0.6
    cost = 0.4
    f = kelly_discrete(dist, payoff, cost=cost)
    assert f > 0


# --- unbounded loss guard ---


def test_unbounded_loss_flag_set():
    """Payoff with r_min << -1 should set unbounded_loss=True and zero all fractions."""
    bins = np.array([4.0, 5.0, 7.0])
    probs = np.array([0.3, 0.4, 0.3])
    dist = Distribution(bins=bins, probs=probs)
    cost = 0.5

    # At bin 7.0: payoff = -50 → r = (-50 - 0.5)/0.5 = -101 << UNBOUNDED_LOSS_THRESHOLD
    def bad_payoff(s: np.ndarray) -> np.ndarray:
        return np.where(s > 5.5, -50.0, 1.0)

    rep = compute_kelly(dist, bad_payoff, cost=cost)
    assert rep.unbounded_loss is True
    assert rep.f_displayed == 0.0
    assert rep.f_raw == 0.0


def test_bounded_excess_loss_still_computes():
    """r_min between -1 and UNBOUNDED_LOSS_THRESHOLD: Kelly runs, upper bound is correct."""
    # r_min = -3: loss is 3× the premium but well within threshold.
    # Distribution: two bins, one gives r = -3, other gives r = +5 with p=0.8.
    bins = np.array([4.0, 6.0])
    probs = np.array([0.2, 0.8])
    dist = Distribution(bins=bins, probs=probs)
    cost = 0.5

    def bounded_payoff(s: np.ndarray) -> np.ndarray:
        # payoff at 4.0 → -1.0  → r = (-1.0 - 0.5)/0.5 = -3.0
        # payoff at 6.0 →  3.0  → r = (3.0 - 0.5)/0.5 = +5.0
        return np.where(s < 5.0, -1.0, 3.0)

    rep = compute_kelly(dist, bounded_payoff, cost=cost)
    assert rep.unbounded_loss is False
    assert rep.f_raw > 0
    # Upper bound for optimizer was 1/3 - ε ≈ 0.333; f* must be below that.
    assert rep.f_discrete < 1.0 / 3.0 + 1e-6
