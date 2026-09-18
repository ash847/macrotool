"""Phase 3 — ranking stability under fractional Kelly (plan §9).

Because `score_ccy = notional · score_pct` and Kelly notional = `λ · x* · W`, the
fractional-Kelly multiplier λ cancels in the ranking — so the ordinal ranking is
invariant to λ *unless the notional cap binds asymmetrically*. These tests guard
that property (it's an implementation guarantee: fractional Kelly must be `λ·x*`,
not a re-solve) and demonstrate the one cap-driven exception.
"""
import numpy as np
import pytest

from analytics.market_state import compute_market_state
from analytics.sizing import (
    SizingSpec,
    kelly_fraction_per_notional,
    market_distribution,
)
from analytics.structure_pricer import price_variants
from tests._curves import stated_lognormal

_LAMBDAS = [0.1, 0.25, 0.5, 1.0]


def _ranking(score_ccys):
    """Indices sorted by score_ccy descending (the variant ordering)."""
    return tuple(int(i) for i in np.argsort(-np.asarray(score_ccys, dtype=float)))


def _kendall_tau(a, b):
    n = len(a); ra = {v: i for i, v in enumerate(a)}; rb = {v: i for i, v in enumerate(b)}
    conc = disc = 0
    items = list(a)
    for i in range(n):
        for j in range(i + 1, n):
            x, y = items[i], items[j]
            s = np.sign(ra[x] - ra[y]) * np.sign(rb[x] - rb[y])
            conc += s > 0; disc += s < 0
    return (conc - disc) / (0.5 * n * (n - 1))


class TestLambdaInvariance:
    def test_ranking_identical_across_lambda_offcap(self):
        # 3 variants with distinct (x*, score_pct); score_ccy = λ·x*·W·score_pct.
        x_star = np.array([0.8, 1.4, 0.5])
        score_pct = np.array([0.020, 0.012, 0.030])
        W = 100.0
        base = None
        for lam in _LAMBDAS:
            score_ccy = lam * x_star * W * score_pct           # uncapped
            r = _ranking(score_ccy)
            if base is None:
                base = r
            assert r == base
            assert _kendall_tau(r, base) == pytest.approx(1.0)

    def test_lambda_is_pure_multiplier_on_notional(self):
        # Off-cap, a variant's notional scales linearly with λ (N = λ·x*·W).
        ms = compute_market_state(spot=5.0, fwd=5.05, vol=0.15, T=0.25, r_d=0.05, r_f=0.04,
                                  target=5.30, direction="base_higher")
        probs, bins = stated_lognormal(5.215, ms.vol, ms.T)   # explicit bullish curve
        n_by_lambda = {}
        for lam in [0.1, 0.2]:
            spec = SizingSpec(method="kelly", kelly_lambda=lam, bankroll=100.0,
                              kelly_probs=probs, kelly_bins=bins)
            # large cap so it can't bind
            pv = price_variants(ms, "vanilla", target=5.30, is_call=True, sizing_spec=spec,
                                linear_notional=1e9)[0]
            n_by_lambda[lam] = pv.structure_notional
        assert n_by_lambda[0.2] == pytest.approx(2.0 * n_by_lambda[0.1], rel=1e-6)

    def test_cap_binding_can_flip_ranking(self):
        # Variant A has the higher uncapped score but caps; B overtakes at high λ.
        W, cap = 100.0, 1000.0
        # A: very large x* (caps at high λ), modest score_pct. B: moderate x* (also
        # caps at λ=1), higher score_pct. Off-cap A leads; once both cap, B wins.
        xa, sa = 50.0, 0.020
        xb, sb = 12.0, 0.030
        rankings = set()
        for lam in [0.1, 1.0]:
            na = min(lam * xa * W, cap); nb = min(lam * xb * W, cap)
            rankings.add(_ranking([na * sa, nb * sb]))
        assert len(rankings) == 2  # ranking flips → cap is the (only) flip driver


class TestMarketDistribution:
    def test_sums_to_one_and_positive(self):
        probs, bins = market_distribution(5.0, 5.05, 0.15, 0.25)
        assert abs(sum(probs) - 1.0) < 1e-9
        assert all(b > 0 for b in bins) and all(p >= 0 for p in probs)

    def test_centred_on_the_forward(self):
        probs, bins = market_distribution(5.0, 5.05, 0.15, 0.25)
        cdf = np.cumsum(probs)
        median = float(np.interp(0.5, cdf, bins))
        assert median == pytest.approx(5.05, rel=0.01)

    def test_takes_no_target_or_conviction(self):
        import inspect
        params = set(inspect.signature(market_distribution).parameters)
        assert not params & {"target", "conviction"}

    def test_kelly_sizes_nothing_without_a_stated_edge(self):
        ms = compute_market_state(spot=5.0, fwd=5.05, vol=0.15, T=0.25, r_d=0.05, r_f=0.04,
                                  target=5.30, direction="base_higher")
        probs, bins = market_distribution(ms.spot, ms.fwd, ms.vol, ms.T)
        spec = SizingSpec(method="kelly", kelly_probs=probs, kelly_bins=bins,
                          distribution_source="market")
        pv = price_variants(ms, "vanilla", target=5.30, is_call=True, sizing_spec=spec,
                            linear_notional=1e9)[0]
        assert (pv.structure_notional or 0.0) == pytest.approx(0.0, abs=1e-6)
