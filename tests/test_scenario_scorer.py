"""Tests for knowledge_engine.scenario_scorer."""

from __future__ import annotations

import pytest

from knowledge_engine.scenario_scorer import score_structure


def _row(family: str, pnl_pct: float, pnl_ccy: float | None = None) -> dict:
    return {"family": family, "pnl_pct": pnl_pct, "pnl_ccy": pnl_ccy}


class TestBasicScoring:
    def test_empty_rows_returns_zero(self):
        r = score_structure([], {"CORRECT_PATH": 1.0})
        assert r.score_pct == 0.0
        assert r.score_ccy is None
        assert r.families == []

    def test_uniform_weights_means_simple_average(self):
        rows = [
            _row("CORRECT_PATH", 0.02),
            _row("CORRECT_PATH", 0.04),
            _row("WRONG_WAY",   -0.03),
            _row("WRONG_WAY",   -0.01),
        ]
        weights = {"CORRECT_PATH": 0.5, "WRONG_WAY": 0.5}
        r = score_structure(rows, weights)
        # avg CP = 0.03, avg WW = -0.02, weighted = 0.5*0.03 + 0.5*-0.02 = 0.005
        assert r.score_pct == pytest.approx(0.005)

    def test_unequal_weights_shift_score(self):
        rows = [
            _row("CORRECT_PATH", 0.10),
            _row("WRONG_WAY",   -0.10),
        ]
        # Weighting CORRECT_PATH heavily makes the score positive.
        r1 = score_structure(rows, {"CORRECT_PATH": 0.9, "WRONG_WAY": 0.1})
        # Weighting WRONG_WAY heavily makes it negative.
        r2 = score_structure(rows, {"CORRECT_PATH": 0.1, "WRONG_WAY": 0.9})
        assert r1.score_pct > 0
        assert r2.score_pct < 0
        assert r1.score_pct == pytest.approx(0.08)
        assert r2.score_pct == pytest.approx(-0.08)


class TestCcyHandling:
    def test_ccy_score_when_all_rows_have_ccy(self):
        rows = [
            _row("CORRECT_PATH", 0.02, 2.0),
            _row("WRONG_WAY",   -0.01, -1.0),
        ]
        r = score_structure(rows, {"CORRECT_PATH": 0.5, "WRONG_WAY": 0.5})
        assert r.score_ccy == pytest.approx(0.5 * 2.0 + 0.5 * -1.0)

    def test_ccy_score_none_when_no_ccy_anywhere(self):
        rows = [_row("CORRECT_PATH", 0.02), _row("WRONG_WAY", -0.01)]
        r = score_structure(rows, {"CORRECT_PATH": 0.5, "WRONG_WAY": 0.5})
        assert r.score_ccy is None


class TestFamilyBreakdown:
    def test_breakdown_one_per_family(self):
        rows = [
            _row("CORRECT_PATH", 0.02),
            _row("CORRECT_PATH", 0.04),
            _row("WRONG_WAY",   -0.03),
        ]
        r = score_structure(rows, {"CORRECT_PATH": 0.5, "WRONG_WAY": 0.5})
        families = {b.family for b in r.families}
        assert families == {"CORRECT_PATH", "WRONG_WAY"}
        cp = next(b for b in r.families if b.family == "CORRECT_PATH")
        assert cp.n_scenarios == 2
        assert cp.avg_pnl_pct == pytest.approx(0.03)
        assert cp.weight == 0.5
        assert cp.contrib_pct == pytest.approx(0.5 * 0.03)

    def test_missing_weight_treated_as_zero(self):
        rows = [_row("CORRECT_PATH", 0.05), _row("EXPIRY", 0.10)]
        # Only CORRECT_PATH has a weight — EXPIRY contributes 0.
        r = score_structure(rows, {"CORRECT_PATH": 1.0})
        assert r.score_pct == pytest.approx(0.05)
        expiry = next(b for b in r.families if b.family == "EXPIRY")
        assert expiry.weight == 0.0
        assert expiry.contrib_pct == 0.0
