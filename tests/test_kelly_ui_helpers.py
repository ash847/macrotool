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

    def _key(self, pair="USDBRL", exp="2026-06-23"):
        from analytics.sizing import curve_key
        return curve_key(pair, exp)

    def test_kelly_uses_curve_stated_for_this_trade(self):
        state = {"sizing_method": "kelly", "kelly_lambda": 0.25,
                 "kelly_probs": (0.5, 0.5), "kelly_bins": (5.0, 5.5),
                 "kelly_curve_key": self._key()}
        spec = build_sizing_spec(state, ms=_ms(), trade_key=("USDBRL", "2026-06-23"))
        assert spec.method == "kelly" and spec.kelly_lambda == 0.25
        assert not spec.kelly_fallback and spec.kelly_bins == (5.0, 5.5)

    @pytest.mark.parametrize("stated", [("USDJPY", "2026-06-23"), ("USDBRL", "2026-05-24"), None])
    def test_no_distribution_for_this_trade_falls_back_to_fixed_loss(self, stated):
        state = {"sizing_method": "kelly", "target_rr": 2.5,
                 "kelly_probs": (0.5, 0.5), "kelly_bins": (5.0, 5.5),
                 "kelly_curve_key": self._key(*stated) if stated else None}
        spec = build_sizing_spec(state, ms=_ms(), trade_key=("USDBRL", "2026-06-23"))
        assert spec.method == "fixed_loss" and spec.kelly_fallback
        assert spec.target_rr == 2.5 and not spec.has_distribution()


class TestLabels:
    def test_column_label(self):
        assert "Kelly" in notional_column_label("kelly")
        assert "max-loss" in notional_column_label("fixed_loss")

    def test_banner_differs(self):
        assert meaning_banner("kelly") != meaning_banner("fixed_loss")

    def test_fallback_message_says_fixed_loss(self):
        from interface.kelly_sizing_ui import KELLY_FALLBACK_MSG
        assert "FIXED-LOSS" in KELLY_FALLBACK_MSG and "distribution" in KELLY_FALLBACK_MSG

    def test_row_flags(self):
        assert kelly_row_flag(0.0, cap=1000.0) == "no Kelly edge"
        assert kelly_row_flag(1000.0, cap=1000.0) == "capped"
        assert kelly_row_flag(500.0, cap=1000.0) == ""
        assert kelly_row_flag(None, cap=1000.0) == ""
