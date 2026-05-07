import pytest

from knowledge_engine.scenario_scorer import score_structure


def _row(scenario_id: str, row: str, col: str, pnl_pct: float, pnl_ccy: float | None = None) -> dict:
    return {
        "scenario_id": scenario_id,
        "row": row,
        "col": col,
        "pnl_pct": pnl_pct,
        "pnl_ccy": pnl_ccy,
    }


class TestScenarioGridScoring:
    def test_empty_rows_returns_zero(self):
        r = score_structure([], {"25%T|F": 1.0})
        assert r.score_pct == 0.0
        assert r.score_ccy is None
        assert r.cells == []

    def test_weighted_average_uses_multipliers(self):
        rows = [
            _row("25%T|F", "25%T", "F", 0.02),
            _row("25%T|K", "25%T", "K", 0.08),
        ]
        r = score_structure(rows, {"25%T|F": 1.0, "25%T|K": 3.0})
        assert r.score_pct == pytest.approx((1.0 * 0.02 + 3.0 * 0.08) / 4.0)

    def test_missing_multiplier_treated_as_zero(self):
        rows = [
            _row("25%T|F", "25%T", "F", 0.02),
            _row("25%T|K", "25%T", "K", 0.08),
        ]
        r = score_structure(rows, {"25%T|F": 1.0})
        assert r.score_pct == pytest.approx(0.02)

    def test_ccy_weighted_average(self):
        rows = [
            _row("25%T|F", "25%T", "F", 0.02, 2.0),
            _row("25%T|K", "25%T", "K", 0.08, 8.0),
        ]
        r = score_structure(rows, {"25%T|F": 1.0, "25%T|K": 3.0})
        assert r.score_ccy == pytest.approx((1.0 * 2.0 + 3.0 * 8.0) / 4.0)
