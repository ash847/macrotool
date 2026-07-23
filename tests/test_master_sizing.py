"""Master sizing control support: the capped flag (engine) and the merged
preference mapping + sizing_capital fallback (interface)."""

from __future__ import annotations

from analytics.structure_pricer import PricedVariant, _size_variant
from interface.prefs import (
    DEFAULT_MERGED_PREF,
    MERGED_PREF_OPTIONS,
    merged_pref_fields,
    merged_pref_label,
)
from interface.structure_eval import LINEAR_NOTIONAL, sizing_capital


def _pv(net_premium_pct: float, max_loss_pct: float) -> PricedVariant:
    return PricedVariant(
        variant_label="t", strikes=[1.0], barrier=None,
        net_premium_pct=net_premium_pct, breakeven=None,
        payoff_at_target_pct=0.05, rr_at_target=None,
        max_loss_pct=max_loss_pct, wing_ratio=None, is_zero_cost=False,
    )


class TestCappedFlag:
    def test_debit_within_budget_not_capped(self):
        pv = _pv(0.02, 0.02)
        _size_variant(pv, loss_budget=2.0, linear_notional=100.0)
        assert pv.structure_notional == 100.0          # 2.0 / 0.02
        assert pv.capped is False
        assert pv.max_loss_ccy == 2.0                  # achieves the budget

    def test_low_premium_hits_cap(self):
        pv = _pv(0.0005, 0.0005)                       # tiny max loss → notional explodes
        _size_variant(pv, loss_budget=2.0, linear_notional=100.0)
        assert pv.structure_notional == 1000.0         # 10 × linear
        assert pv.capped is True
        # achieved max loss sits BELOW the budget on capped rows
        assert pv.max_loss_ccy < 2.0

    def test_net_credit_pinned_and_flagged(self):
        pv = _pv(-0.01, 0.01)
        _size_variant(pv, loss_budget=2.0, linear_notional=100.0)
        assert pv.structure_notional == 1000.0
        assert pv.capped is True

    def test_zero_max_loss_pinned_and_flagged(self):
        pv = _pv(0.0, 0.0)
        _size_variant(pv, loss_budget=2.0, linear_notional=100.0)
        assert pv.structure_notional == 1000.0
        assert pv.capped is True

    def test_cap_scales_with_linear_notional(self):
        pv = _pv(0.0005, 0.0005)
        _size_variant(pv, loss_budget=2_000_000.0, linear_notional=100_000_000.0)
        assert pv.structure_notional == 1_000_000_000.0  # 10 × W
        assert pv.capped is True


class TestMergedPrefs:
    def test_every_label_maps_to_valid_engine_fields(self):
        constraints = {"No restriction", "Avoid capped structures",
                       "Avoid complex structures", "Avoid tail-risky structures"}
        managements = {"Standard hold", "May monetise early",
                       "Need defendable mark-to-market"}
        for label in MERGED_PREF_OPTIONS:
            sc, tm = merged_pref_fields(label)
            assert sc in constraints and tm in managements

    def test_roundtrip(self):
        for label in MERGED_PREF_OPTIONS:
            sc, tm = merged_pref_fields(label)
            assert merged_pref_label(sc, tm) == label

    def test_unknown_label_falls_back_to_default(self):
        assert merged_pref_fields("nonsense") == MERGED_PREF_OPTIONS[DEFAULT_MERGED_PREF]

    def test_unknown_combination_falls_back_to_default_label(self):
        assert merged_pref_label("Avoid capped structures",
                                 "Need defendable mark-to-market") == DEFAULT_MERGED_PREF


def test_sizing_capital_falls_back_to_nominal_outside_app():
    # No Streamlit session / no sizing_capital key → the nominal engine default,
    # so tests and scripts are unaffected by the UI's W.
    assert sizing_capital() == LINEAR_NOTIONAL
