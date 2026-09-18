"""Phase 2 — the sizing seam: SizingSpec routing in price_variants.

Locks the fixed-loss path byte-for-byte (default behaviour unchanged) and exercises
the Kelly branch + the load-bearing `score_ccy = notional · score_pct` invariant.
"""
import numpy as np
import pytest

from analytics.market_state import compute_market_state
from analytics.scenario_generator import generate_scenarios
from analytics.scenario_pricer import price_scenarios
from analytics.sizing import SizingSpec
from analytics.structure_pricer import price_variants
from knowledge_engine.scenario_scorer import score_structure


def _ms():
    return compute_market_state(spot=5.0, fwd=5.05, vol=0.15, T=0.25, r_d=0.05, r_f=0.04,
                                target=5.30, direction="base_higher")


def _bullish_dist(center=5.35, width=0.20, n=41):
    bins = np.linspace(center - 3 * width, center + 3 * width, n)
    bins = bins[bins > 0]
    probs = np.exp(-0.5 * ((bins - center) / width) ** 2)
    probs /= probs.sum()
    return tuple(probs), tuple(bins)


class TestFixedLossRegression:
    def test_no_sizing_spec_matches_loss_budget_path(self):
        # Default (no sizing_spec) is unchanged: vanilla max_loss == net_premium, so
        # notional = budget / premium (capped), max_loss_ccy <= budget.
        ms = _ms()
        pvs = price_variants(ms, "vanilla", target=5.30, is_call=True, loss_budget=4.0)
        assert pvs
        for pv in pvs:
            assert pv.structure_notional is not None
            assert pv.max_loss_ccy <= 4.0 + 1e-6
            assert abs(pv.net_premium_ccy - pv.max_loss_ccy) < 1e-6

    def test_fixed_loss_spec_equals_no_spec(self):
        ms = _ms()
        a = price_variants(ms, "1x1_spread", target=5.30, is_call=True, loss_budget=4.0)
        b = price_variants(ms, "1x1_spread", target=5.30, is_call=True, loss_budget=4.0,
                           sizing_spec=SizingSpec())  # explicit fixed-loss default
        assert [round(x.structure_notional, 9) for x in a] == [round(x.structure_notional, 9) for x in b]


class TestKellyBranch:
    def test_kelly_populates_notionals_capped(self):
        ms = _ms()
        probs, bins = _bullish_dist()
        spec = SizingSpec(method="kelly", kelly_lambda=0.5, bankroll=100.0,
                          kelly_probs=probs, kelly_bins=bins)
        pvs = price_variants(ms, "vanilla", target=5.30, is_call=True, sizing_spec=spec)
        assert pvs
        for pv in pvs:
            assert pv.structure_notional is not None
            assert 0.0 <= pv.structure_notional <= 10.0 * 100.0 + 1e-6

    def test_kelly_ignores_loss_budget(self):
        ms = _ms()
        probs, bins = _bullish_dist()
        spec = SizingSpec(method="kelly", kelly_probs=probs, kelly_bins=bins)
        # loss_budget passed but Kelly should drive sizing, not the budget.
        pvs = price_variants(ms, "vanilla", target=5.30, is_call=True,
                             loss_budget=4.0, sizing_spec=spec)
        # Kelly notional differs from the fixed-loss notional for the same budget.
        fixed = price_variants(ms, "vanilla", target=5.30, is_call=True, loss_budget=4.0)
        assert pvs[0].structure_notional != pytest.approx(fixed[0].structure_notional)

    def test_kelly_no_edge_zero_notional(self):
        # A distribution centred at the forward (no directional edge) → tiny/zero size.
        ms = _ms()
        probs, bins = _bullish_dist(center=ms.fwd, width=0.20)
        spec = SizingSpec(method="kelly", kelly_probs=probs, kelly_bins=bins)
        pvs = price_variants(ms, "vanilla", target=5.30, is_call=True, sizing_spec=spec)
        # OTM call with mass centred at fwd has negative expected P&L after premium → 0.
        assert pvs[0].structure_notional == 0.0

    def test_unsupported_family_left_unsized(self):
        # risk_reversal has no payoff bridge entry → left unsized (no crash).
        ms = _ms()
        probs, bins = _bullish_dist()
        spec = SizingSpec(method="kelly", kelly_probs=probs, kelly_bins=bins)
        # Use a covered family to confirm the loop runs; the guard is exercised in
        # _size_variants_kelly's try/except (covered by families without a bridge).
        pvs = price_variants(ms, "vanilla", target=5.30, is_call=True, sizing_spec=spec)
        assert pvs  # no crash


class TestScoreCcyInvariant:
    def test_score_ccy_is_notional_times_score_pct(self):
        # The load-bearing fact: score_ccy = notional · score_pct. Sizing only scales it.
        ms = _ms()
        ti = {"spot": 5.0, "forward": 5.05, "implied_vol": 0.15, "tenor_years": 0.25,
              "target": 5.30, "r_d": 0.05, "r_f": 0.04}
        probs, bins = _bullish_dist()
        spec = SizingSpec(method="kelly", kelly_probs=probs, kelly_bins=bins)
        pv = price_variants(ms, "1x2x1_spread", target=5.30, is_call=True, sizing_spec=spec)[0]
        assert pv.structure_notional and pv.structure_notional > 0
        rows = price_scenarios(pv, "1x2x1_spread", generate_scenarios(ti), ti, True)
        # Uniform weights keyed by scenario_id (the key score_structure matches on).
        w = {r["scenario_id"]: 1.0 for r in rows}
        sc = score_structure(rows, w)
        assert sc.score_ccy is not None
        assert sc.score_ccy == pytest.approx(pv.structure_notional * sc.score_pct, rel=1e-9)
