"""Component B — per-notional Kelly primitive (analytics/sizing.py)."""
import numpy as np
import pytest

from analytics.sizing import (
    SizingSpec,
    kelly_fraction_per_notional,
    kelly_notional,
    per_notional_pnl,
)


def _two_point(p_up, r_up, r_dn):
    """Simple 2-outcome dist: prob p_up of return r_up, else r_dn (per notional)."""
    return np.array([p_up, 1.0 - p_up]), np.array([r_up, r_dn])


class TestKellyFraction:
    def test_no_edge_zero(self):
        # E[π] <= 0 → don't bet.
        probs, pnl = _two_point(0.5, 0.10, -0.10)   # zero mean
        assert kelly_fraction_per_notional(probs, pnl) == 0.0
        probs, pnl = _two_point(0.3, 0.10, -0.10)   # negative mean
        assert kelly_fraction_per_notional(probs, pnl) == 0.0

    def test_positive_edge_positive_fraction(self):
        probs, pnl = _two_point(0.6, 0.10, -0.10)
        assert kelly_fraction_per_notional(probs, pnl) > 0.0

    def test_monotonic_in_edge(self):
        f_lo = kelly_fraction_per_notional(*_two_point(0.55, 0.10, -0.10))
        f_hi = kelly_fraction_per_notional(*_two_point(0.70, 0.10, -0.10))
        assert f_hi > f_lo

    def test_ruin_bound_long_option(self):
        # Long-option-like: worst outcome loses the premium (π_min = -prem).
        prem = 0.02
        probs = np.array([0.5, 0.5])
        pnl = np.array([0.30, -prem])      # ITM gain vs lose premium
        x = kelly_fraction_per_notional(probs, pnl)
        assert x < 1.0 / prem              # leverage bounded by 1/|π_min|
        # premium actually spent = x·W·prem stays within bankroll W
        W = 100.0
        assert x * W * prem <= W + 1e-6

    def test_zero_cost_structure_well_defined(self):
        # Zero premium, both-signed outcomes → finite x*, bound from worst loss.
        probs = np.array([0.5, 0.5])
        pnl = np.array([0.05, -0.03])      # net_premium_pct = 0 baked in
        x = kelly_fraction_per_notional(probs, pnl)
        assert 0.0 < x < 1.0 / 0.03

    def test_net_credit_structure_well_defined(self):
        # Credit received (per_notional_pnl with negative premium) still finite.
        payoff = np.array([0.0, 0.08])      # base-ccy payoff per notional
        pnl = per_notional_pnl(payoff, net_premium_pct=-0.01)   # credit
        assert np.isclose(pnl[0], 0.01)     # OTM: keep the credit
        probs = np.array([0.4, 0.6])
        x = kelly_fraction_per_notional(probs, pnl)
        assert x > 0.0

    def test_no_loss_outcome_runs_to_ceiling(self):
        # All outcomes non-negative + positive mean → unbounded; clamps at f_max.
        probs = np.array([0.5, 0.5])
        pnl = np.array([0.02, 0.05])
        x = kelly_fraction_per_notional(probs, pnl, f_max=10.0)
        assert x == pytest.approx(10.0, abs=1e-3)

    def test_thorp_closed_form_sanity(self):
        # Small-edge near-Gaussian: x* ≈ E[π]/Var[π].
        rng = np.random.default_rng(0)
        bins = rng.normal(0.0, 0.05, size=2000)
        pnl = bins + 0.01                  # small positive drift
        probs = np.full(pnl.size, 1.0 / pnl.size)
        x = kelly_fraction_per_notional(probs, pnl)
        e = float(np.mean(pnl)); v = float(np.var(pnl))
        assert x == pytest.approx(e / v, rel=0.15)


class TestPerNotionalPnl:
    def test_pnl_is_payoff_minus_premium(self):
        payoff = np.array([0.0, 0.10, 0.20])
        pnl = per_notional_pnl(payoff, net_premium_pct=0.03, discount_factor=1.0)
        np.testing.assert_allclose(pnl, [-0.03, 0.07, 0.17])

    def test_discount_applies_to_payoff_only(self):
        payoff = np.array([0.10])
        pnl = per_notional_pnl(payoff, net_premium_pct=0.02, discount_factor=0.9)
        assert pnl[0] == pytest.approx(0.9 * 0.10 - 0.02)


class TestKellyNotional:
    def test_caps_at_cap(self):
        probs = np.array([0.5, 0.5])
        pnl = np.array([0.50, 0.40])        # no-loss → x runs to ceiling
        spec = SizingSpec(method="kelly", kelly_lambda=0.5, bankroll=100.0)
        n = kelly_notional(probs, pnl, spec, cap=1000.0)
        assert n == pytest.approx(1000.0)

    def test_scales_with_lambda(self):
        probs = np.array([0.5, 0.5])
        pnl = np.array([0.30, -0.02])
        n_half = kelly_notional(probs, pnl, SizingSpec(method="kelly", kelly_lambda=0.5, bankroll=100.0), cap=1e9)
        n_quarter = kelly_notional(probs, pnl, SizingSpec(method="kelly", kelly_lambda=0.25, bankroll=100.0), cap=1e9)
        assert n_half == pytest.approx(2.0 * n_quarter)

    def test_spec_defaults_are_fixed_loss(self):
        s = SizingSpec()
        assert s.method == "fixed_loss"
        assert s.target_rr == 3.0
        assert not s.has_distribution()
