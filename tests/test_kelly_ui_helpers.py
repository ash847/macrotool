"""Component E — UI behaviour logic (pure helpers, no Streamlit)."""
import pytest

from analytics.market_state import compute_market_state
from interface.kelly_sizing_ui import (
    build_sizing_spec,
    kelly_row_flag,
    meaning_banner,
    notional_column_label,
    should_reseed,
)


def _ms():
    return compute_market_state(spot=5.0, fwd=5.05, vol=0.15, T=0.25, r_d=0.05, r_f=0.04,
                                target=5.30, direction="base_higher")


class TestBuildSizingSpec:
    def test_fixed_loss_round_trips_rr(self):
        spec = build_sizing_spec({"sizing_method": "fixed_loss", "target_rr": 2.5})
        assert spec.method == "fixed_loss"
        assert spec.target_rr == 2.5

    def test_kelly_uses_elicited_distribution(self):
        state = {"sizing_method": "kelly", "kelly_lambda": 0.25,
                 "kelly_probs": (0.5, 0.5), "kelly_bins": (5.0, 5.5)}
        spec = build_sizing_spec(state)
        assert spec.method == "kelly"
        assert spec.kelly_lambda == 0.25
        assert spec.has_distribution()

    def test_kelly_falls_back_to_view_implied_seed(self):
        spec = build_sizing_spec({"sizing_method": "kelly"}, ms=_ms(), target=5.30)
        assert spec.method == "kelly"
        assert spec.has_distribution()           # seeded, not an error
        assert abs(sum(spec.kelly_probs) - 1.0) < 1e-9

    def test_kelly_without_target_falls_back_to_fixed(self):
        spec = build_sizing_spec({"sizing_method": "kelly"}, ms=_ms(), target=None)
        assert spec.method == "fixed_loss"       # nothing to seed → fixed-loss for the table

    def test_kelly_bin_count_flows(self):
        spec = build_sizing_spec({"sizing_method": "kelly", "kelly_n_bins": 11},
                                 ms=_ms(), target=5.30)
        assert len(spec.kelly_bins) == 11


class TestShouldReseed:
    def test_first_time(self):
        assert should_reseed(None, {"pair": "USDTRY"}) is True

    def test_mode_switch_to_kelly(self):
        prev = {"sizing_method": "fixed_loss", "pair": "USDTRY", "horizon": 90, "target": 30, "conviction": "medium"}
        new = {**prev, "sizing_method": "kelly"}
        assert should_reseed(prev, new) is True

    def test_context_change_reseeds(self):
        prev = {"sizing_method": "kelly", "pair": "USDTRY", "horizon": 90, "target": 30, "conviction": "medium"}
        assert should_reseed(prev, {**prev, "target": 32}) is True
        assert should_reseed(prev, {**prev, "conviction": "high"}) is True

    def test_lambda_change_does_not_reseed(self):
        prev = {"sizing_method": "kelly", "pair": "USDTRY", "horizon": 90, "target": 30,
                "conviction": "medium", "kelly_lambda": 0.5}
        new = {**prev, "kelly_lambda": 0.25}     # λ not a reseed key
        assert should_reseed(prev, new) is False

    def test_plain_rerun_does_not_reseed(self):
        prev = {"sizing_method": "kelly", "pair": "USDTRY", "horizon": 90, "target": 30, "conviction": "medium"}
        assert should_reseed(prev, dict(prev)) is False


class TestLabels:
    def test_column_label(self):
        assert "Kelly" in notional_column_label("kelly")
        assert "max-loss" in notional_column_label("fixed_loss")

    def test_banner_differs(self):
        assert meaning_banner("kelly") != meaning_banner("fixed_loss")

    def test_row_flags(self):
        assert kelly_row_flag(0.0, cap=1000.0) == "no Kelly edge"
        assert kelly_row_flag(1000.0, cap=1000.0) == "capped"
        assert kelly_row_flag(500.0, cap=1000.0) == ""
        assert kelly_row_flag(None, cap=1000.0) == ""
