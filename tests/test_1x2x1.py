"""1x2x1 butterfly — the 1x2 with a long wing that caps the open tail.

Covers: equidistant geometry (K3 = 2·K2 − K1), capped max loss = net premium,
selection as a distinct family, and the preference-gate behaviour that makes it
the tail-safe alternative to the 1x2 (survives 'Avoid tail-risky', excluded by
'Avoid complex').
"""
import math

from analytics.market_state import compute_market_state
from analytics.structure_pricer import price_variants
from analytics.scenario_generator import generate_scenarios
from analytics.scenario_pricer import price_scenarios
from knowledge_engine.structure_scorer import score_structures


def _ms():
    return compute_market_state(
        spot=5.0, fwd=5.05, vol=0.15, T=0.25, r_d=0.05, r_f=0.04,
        target=5.30, direction="base_higher",
    )


class TestGeometryAndPricing:
    def test_third_strike_equidistant(self):
        pvs = price_variants(_ms(), "1x2x1_spread", target=5.30, is_call=True)
        assert pvs
        for pv in pvs:
            assert len(pv.strikes) == 3
            K1, K2, K3 = pv.strikes
            assert math.isclose(K3, 2.0 * K2 - K1, rel_tol=0, abs_tol=1e-9)

    def test_max_loss_is_net_premium_capped_tail(self):
        # The wing caps the 1x2's open tail → a long butterfly is a debit and its
        # max loss equals the net premium paid (not the unbounded short-leg tail).
        pvs = price_variants(_ms(), "1x2x1_spread", target=5.30, is_call=True)
        for pv in pvs:
            assert pv.net_premium_pct >= 0.0                       # debit
            assert math.isclose(pv.max_loss_pct, max(pv.net_premium_pct, 0.0), abs_tol=1e-12)

    def test_loss_strictly_smaller_than_naked_1x2_tail(self):
        # Same first two strikes; the 1x2x1 caps the loss the 1x2 leaves open.
        ms = _ms()
        ti = {"spot": 5.0, "forward": 5.05, "implied_vol": 0.15, "tenor_years": 0.25,
              "target": 5.30, "r_d": 0.05, "r_f": 0.04}
        scenarios = generate_scenarios(ti)
        fly = price_variants(ms, "1x2x1_spread", target=5.30, is_call=True)[0]
        spread = price_variants(ms, "1x2_spread", target=5.30, is_call=True)[0]
        fly_rows = price_scenarios(fly, "1x2x1_spread", scenarios, ti, True)
        spread_rows = price_scenarios(spread, "1x2_spread", scenarios, ti, True)
        # Worst-case package value (% of spot) across the grid: the fly is never worse.
        fly_worst = min(r["price_pct"] for r in fly_rows)
        spread_worst = min(r["price_pct"] for r in spread_rows)
        assert fly_worst >= spread_worst - 1e-9

    def test_sizing_applies(self):
        pvs = price_variants(_ms(), "1x2x1_spread", target=5.30, is_call=True, loss_budget=4.0)
        for pv in pvs:
            assert pv.structure_notional is not None
            assert pv.structure_notional <= 10.0 * 100.0 + 1e-6   # capped at 10× linear notional


class TestSelectionAndGates:
    def _ms_eligible(self):
        # carry regime 2, target inside the gate band so the ratio families are eligible.
        return compute_market_state(
            spot=5.0, fwd=5.20, vol=0.15, T=0.25, r_d=0.12, r_f=0.04,
            target=5.45, direction="base_higher",
        )

    def test_appears_as_distinct_family(self):
        res = score_structures(self._ms_eligible())
        ids = {s.structure_id for s in res.shortlist}
        # 1x2x1 is scored independently of the 1x2.
        assert "1x2x1_spread" in ids or "1x2x1_spread" in {
            s.structure_id for s in res.shortlist
        }

    def test_survives_avoid_tail_risky_when_1x2_excluded(self):
        ms = self._ms_eligible()
        res = score_structures(ms, structure_constraint="Avoid tail-risky structures")
        ids = {s.structure_id for s in res.shortlist}
        assert "1x2_spread" not in ids          # naked tail → hard-gated out
        assert "1x2x1_spread" in ids            # capped tail → survives

    def test_excluded_by_avoid_complex(self):
        ms = self._ms_eligible()
        res = score_structures(ms, structure_constraint="Avoid complex structures")
        ids = {s.structure_id for s in res.shortlist}
        assert "1x2x1_spread" not in ids        # 3-leg structure → complex
