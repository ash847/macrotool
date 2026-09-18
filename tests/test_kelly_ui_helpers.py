"""Component E — UI behaviour logic (pure helpers, no Streamlit)."""
import pytest

from analytics.market_state import compute_market_state
from interface.kelly_sizing_ui import (
    build_sizing_spec,
    kelly_row_flag,
    meaning_banner,
    notional_column_label,
)


def _ms():
    return compute_market_state(spot=5.0, fwd=5.05, vol=0.15, T=0.25, r_d=0.05, r_f=0.04,
                                target=5.30, direction="base_higher")


class TestBuildSizingSpec:
    def test_fixed_loss_round_trips_rr(self):
        spec = build_sizing_spec({"sizing_method": "fixed_loss", "target_rr": 2.5})
        assert spec.method == "fixed_loss"
        assert spec.target_rr == 2.5

    def test_kelly_uses_curve_stated_for_this_trade(self):
        state = {"sizing_method": "kelly", "kelly_lambda": 0.25,
                 "kelly_probs": (0.5, 0.5), "kelly_bins": (5.0, 5.5),
                 "kelly_curve_key": ("USDBRL", 90)}
        spec = build_sizing_spec(state, ms=_ms(), trade_key=("USDBRL", 90))
        assert spec.method == "kelly" and spec.kelly_lambda == 0.25
        assert spec.distribution_source == "explicit"
        assert spec.kelly_bins == (5.0, 5.5)

    @pytest.mark.parametrize("stated_key", [("USDJPY", 90), ("USDBRL", 60), None])
    def test_curve_for_another_trade_is_ignored(self, stated_key):
        state = {"sizing_method": "kelly", "kelly_probs": (0.5, 0.5), "kelly_bins": (5.0, 5.5),
                 "kelly_curve_key": stated_key}
        spec = build_sizing_spec(state, ms=_ms(), trade_key=("USDBRL", 90))
        assert spec.method == "kelly" and spec.distribution_source == "market"
        assert spec.kelly_bins != (5.0, 5.5)

    def test_no_stated_curve_sizes_on_market_distribution(self):
        spec = build_sizing_spec({"sizing_method": "kelly"}, ms=_ms(), trade_key=("USDBRL", 90))
        assert spec.method == "kelly" and spec.distribution_source == "market"
        assert abs(sum(spec.kelly_probs) - 1.0) < 1e-9

    def test_kelly_without_market_state_cannot_size(self):
        spec = build_sizing_spec({"sizing_method": "kelly"}, ms=None, trade_key=("USDBRL", 90))
        assert spec.method == "fixed_loss"


class TestLabels:
    def test_column_label(self):
        assert "Kelly" in notional_column_label("kelly")
        assert "max-loss" in notional_column_label("fixed_loss")

    def test_banner_differs(self):
        assert meaning_banner("kelly") != meaning_banner("fixed_loss")

    def test_market_banner_says_no_edge_stated(self):
        msg = meaning_banner("kelly", "market")
        assert "MARKET" in msg and "haven't stated" in msg

    def test_row_flags(self):
        assert kelly_row_flag(0.0, cap=1000.0) == "no Kelly edge"
        assert kelly_row_flag(1000.0, cap=1000.0) == "capped"
        assert kelly_row_flag(500.0, cap=1000.0) == ""
        assert kelly_row_flag(None, cap=1000.0) == ""
