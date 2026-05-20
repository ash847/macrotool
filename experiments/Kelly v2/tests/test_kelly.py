import numpy as np
import pytest

from baseline import synthetic_lognormal_baseline
from elicitation import Distribution
from kelly import (
    KellyReport,
    SAFETY_F_MAX,
    compute_kelly,
    kelly_continuous,
    kelly_discrete,
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
