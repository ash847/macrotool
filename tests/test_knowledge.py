"""
Tests for the knowledge engine.

Verifies:
  - Convention resolver returns correct typed data for all three pairs
  - Structure selector produces correct shortlists for key view types
  - Sizing engine applies Kelly, vol adjustment, and stop correctly
  - Critique engine produces appropriate verdicts
  - Config overrides propagate through to engine outputs
"""

import math
import pytest

from config.loader import load_config
from config.schema import SessionOverrides
from config.resolver import resolve
from data.snapshot_loader import load_snapshot
from knowledge_engine.conventions import resolve as resolve_conventions, format_for_context
from knowledge_engine.loader import get_decision_rules, get_structure_catalog
from knowledge_engine.models import TradeView
from knowledge_engine.structure_selector import select_structures
from knowledge_engine.sizing_engine import compute_sizing
from knowledge_engine.critique_engine import evaluate_structure


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def snapshot():
    return load_snapshot()

@pytest.fixture(scope="module")
def cfg():
    return load_config()

@pytest.fixture(scope="module")
def brl_snapshot(snapshot):
    return snapshot.get("USDBRL")

@pytest.fixture(scope="module")
def try_snapshot(snapshot):
    return snapshot.get("USDTRY")

@pytest.fixture(scope="module")
def pln_snapshot(snapshot):
    return snapshot.get("EURPLN")

def _brl_view(**kwargs) -> TradeView:
    defaults = dict(
        pair="USDBRL", direction="base_higher",
        direction_conviction="high", timing_conviction="high",
        horizon_days=91,
    )
    defaults.update(kwargs)
    return TradeView(**defaults)

def _pln_view(**kwargs) -> TradeView:
    defaults = dict(
        pair="EURPLN", direction="base_higher",
        direction_conviction="medium", timing_conviction="medium",
        horizon_days=91,
    )
    defaults.update(kwargs)
    return TradeView(**defaults)


# ---------------------------------------------------------------------------
# Conventions resolver
# ---------------------------------------------------------------------------

class TestConventions:
    def test_usdbrl_is_ndf(self):
        conv = resolve_conventions("USDBRL")
        assert conv.instrument_type == "NDF"

    def test_usdbrl_fixing_source(self):
        conv = resolve_conventions("USDBRL")
        assert conv.fixing_source == "PTAX"

    def test_usdbrl_settlement_currency_usd(self):
        conv = resolve_conventions("USDBRL")
        assert conv.settlement_currency == "USD"

    def test_usdbrl_premium_currency_usd(self):
        conv = resolve_conventions("USDBRL")
        assert conv.premium_currency == "USD"

    def test_usdtry_is_ndf(self):
        conv = resolve_conventions("USDTRY")
        assert conv.instrument_type == "NDF"

    def test_usdtry_fixing_source(self):
        conv = resolve_conventions("USDTRY")
        assert conv.fixing_source == "CBRT"

    def test_eurpln_is_deliverable(self):
        conv = resolve_conventions("EURPLN")
        assert conv.instrument_type == "Deliverable"

    def test_eurpln_no_fixing_source(self):
        conv = resolve_conventions("EURPLN")
        assert conv.fixing_source is None

    def test_eurpln_settlement_pln(self):
        conv = resolve_conventions("EURPLN")
        assert conv.settlement_currency == "PLN"

    def test_all_pairs_have_risk_notes(self):
        for pair in ["USDBRL", "USDTRY", "EURPLN"]:
            conv = resolve_conventions(pair)
            assert len(conv.risk_notes) > 0

    def test_all_pairs_have_liquid_tenors(self):
        for pair in ["USDBRL", "USDTRY", "EURPLN"]:
            conv = resolve_conventions(pair)
            assert len(conv.liquid_tenors) > 0
            assert "1M" in conv.liquid_tenors
            assert "3M" in conv.liquid_tenors

    def test_format_for_context_produces_string(self):
        conv = resolve_conventions("USDBRL")
        text = format_for_context(conv)
        assert "USDBRL" in text
        assert "PTAX" in text
        assert "NDF" in text


# ---------------------------------------------------------------------------
# Structure selector
# ---------------------------------------------------------------------------

class TestStructureSelector:
    def test_high_conviction_base_higher_includes_vanilla_call(self, brl_snapshot, cfg):
        view = _brl_view(direction_conviction="high", timing_conviction="high")
        result = select_structures(view, cfg)
        ids = [s.structure_id for s in result.shortlist]
        assert "vanilla_call" in ids

    def test_high_conviction_base_lower_includes_vanilla_put(self, brl_snapshot, cfg):
        view = _brl_view(direction="base_lower", direction_conviction="high")
        result = select_structures(view, cfg)
        ids = [s.structure_id for s in result.shortlist]
        assert "vanilla_put" in ids

    def test_no_vanilla_call_for_base_lower_view(self, brl_snapshot, cfg):
        view = _brl_view(direction="base_lower", direction_conviction="high")
        result = select_structures(view, cfg)
        ids = [s.structure_id for s in result.shortlist]
        assert "vanilla_call" not in ids

    def test_budget_constrained_with_target_suggests_spread(self, brl_snapshot, cfg):
        view = _brl_view(
            direction_conviction="high",
            budget_usd=200_000,
            magnitude_pct=5.0,
        )
        result = select_structures(view, cfg)
        ids = [s.structure_id for s in result.shortlist]
        assert "call_spread" in ids

    def test_skew_rule_sr03_does_not_fire_without_regime(self, brl_snapshot, cfg):
        """SR-03 requires skew classification — does not fire without vol regime data."""
        view = _brl_view(direction_conviction="high")
        result = select_structures(view, cfg)
        assert "SR-03" not in result.rules_fired

    def test_low_direction_conviction_suggests_spot_or_forward(self, brl_snapshot, cfg):
        view = _brl_view(direction_conviction="low")
        result = select_structures(view, cfg)
        ids = [s.structure_id for s in result.shortlist]
        assert "spot" in ids or "forward" in ids

    def test_low_timing_conviction_fires_sr04(self, brl_snapshot, cfg):
        view = _brl_view(timing_conviction="low", direction_conviction="high")
        result = select_structures(view, cfg)
        assert "SR-04" in result.rules_fired

    def test_low_timing_conviction_has_tranche_modifier(self, brl_snapshot, cfg):
        view = _brl_view(timing_conviction="low", direction_conviction="high")
        result = select_structures(view, cfg)
        tranche_items = [s for s in result.shortlist if s.sizing_modifier == "tranche_entry"]
        assert len(tranche_items) > 0

    def test_sr03_does_not_fire_for_pln(self, pln_snapshot, cfg):
        """SR-03 requires skew classification — does not fire."""
        view = _pln_view(direction_conviction="high")
        result = select_structures(view, cfg)
        assert "SR-03" not in result.rules_fired

    def test_shortlist_respects_max_size(self, brl_snapshot, cfg):
        view = _brl_view(direction_conviction="high")
        result = select_structures(view, cfg)
        non_exotic = [s for s in result.shortlist if not s.is_exotic]
        assert len(non_exotic) <= cfg.structures.max_shortlist_size

    def test_excluded_structure_not_in_shortlist(self, brl_snapshot):
        session = SessionOverrides()
        session.apply("structures.excluded_structures", ["call_spread"], "session", "no spreads")
        cfg_with_exclusion = resolve(load_config(), None, session)
        view = _brl_view(
            direction_conviction="high", budget_usd=200_000, magnitude_pct=5.0
        )
        result = select_structures(view, cfg_with_exclusion)
        ids = [s.structure_id for s in result.shortlist]
        assert "call_spread" not in ids

    def test_shortlist_has_rationale(self, brl_snapshot, cfg):
        view = _brl_view()
        result = select_structures(view, cfg)
        for item in result.shortlist:
            assert len(item.rationale) > 0

    def test_result_skew_context_is_blank(self, brl_snapshot, cfg):
        """skew_context is blank until vol regime classification is re-introduced."""
        view = _brl_view()
        result = select_structures(view, cfg)
        assert result.skew_context == ""
        assert result.vol_regime_note == ""


# ---------------------------------------------------------------------------
# Sizing engine
# ---------------------------------------------------------------------------

class TestSizingEngine:
    def _top_structure(self, view, snapshot, cfg):
        result = select_structures(view, cfg)
        return result.shortlist[0]

    def test_kelly_fraction_high_conviction(self, brl_snapshot, cfg):
        view = _brl_view(direction_conviction="high", budget_usd=500_000)
        top = self._top_structure(view, brl_snapshot, cfg)
        sizing = compute_sizing(view, brl_snapshot, top, cfg)
        assert sizing.kelly_fraction == pytest.approx(cfg.sizing.kelly.high_conviction_fraction)

    def test_kelly_fraction_low_conviction(self, brl_snapshot, cfg):
        view = _brl_view(direction_conviction="low", budget_usd=500_000)
        top = self._top_structure(view, brl_snapshot, cfg)
        sizing = compute_sizing(view, brl_snapshot, top, cfg)
        assert sizing.kelly_fraction == pytest.approx(cfg.sizing.kelly.low_conviction_fraction)

    def test_vol_adjustment_is_one_without_regime(self, brl_snapshot, cfg):
        """vol_adjustment is 1.0 until regime classification is re-introduced."""
        view = _brl_view(direction_conviction="medium", budget_usd=500_000)
        top = self._top_structure(view, brl_snapshot, cfg)
        sizing = compute_sizing(view, brl_snapshot, top, cfg)
        assert sizing.vol_adjustment == pytest.approx(1.0)

    def test_adjusted_kelly_is_product(self, brl_snapshot, cfg):
        view = _brl_view(direction_conviction="high", budget_usd=500_000)
        top = self._top_structure(view, brl_snapshot, cfg)
        sizing = compute_sizing(view, brl_snapshot, top, cfg)
        assert sizing.adjusted_kelly == pytest.approx(
            sizing.kelly_fraction * sizing.vol_adjustment
        )

    def test_budget_produces_notional(self, brl_snapshot, cfg):
        view = _brl_view(direction_conviction="high", budget_usd=200_000)
        top = self._top_structure(view, brl_snapshot, cfg)
        sizing = compute_sizing(view, brl_snapshot, top, cfg)
        assert sizing.base_notional_usd is not None
        assert sizing.base_notional_usd > 0
        assert sizing.kelly_notional_usd is not None

    def test_stop_on_correct_side_for_base_higher(self, brl_snapshot, cfg):
        """Long base position: stop should be BELOW current spot."""
        view = _brl_view(direction="base_higher")
        top = self._top_structure(view, brl_snapshot, cfg)
        sizing = compute_sizing(view, brl_snapshot, top, cfg)
        assert sizing.stop_level < brl_snapshot.spot

    def test_stop_on_correct_side_for_base_lower(self, brl_snapshot, cfg):
        """Short base position: stop should be ABOVE current spot."""
        view = _brl_view(direction="base_lower")
        top = self._top_structure(view, brl_snapshot, cfg)
        sizing = compute_sizing(view, brl_snapshot, top, cfg)
        assert sizing.stop_level > brl_snapshot.spot

    def test_stop_distance_is_vol_derived(self, brl_snapshot, cfg):
        """Stop distance = ATR multiple × daily range (uses 1M ATM vol from surface)."""
        view = _brl_view()
        top = self._top_structure(view, brl_snapshot, cfg)
        sizing = compute_sizing(view, brl_snapshot, top, cfg)
        expected_daily_range = brl_snapshot.spot * brl_snapshot.get_atm_vol("1M") / math.sqrt(252)
        expected_stop_dist = expected_daily_range * cfg.sizing.stop.atr_multiple
        assert sizing.daily_range_est == pytest.approx(expected_daily_range, rel=1e-5)
        assert abs(brl_snapshot.spot - sizing.stop_level) == pytest.approx(expected_stop_dist, rel=1e-5)

    def test_low_timing_conviction_produces_tranche_schedule(self, brl_snapshot, cfg):
        view = _brl_view(timing_conviction="low")
        top = self._top_structure(view, brl_snapshot, cfg)
        sizing = compute_sizing(view, brl_snapshot, top, cfg)
        assert sizing.tranche_schedule is not None
        assert sizing.tranche_count == cfg.sizing.tranche_entry.low_timing_conviction.count
        assert abs(sum(sizing.tranche_schedule) - 1.0) < 1e-6

    def test_take_profit_levels_when_target_known(self, brl_snapshot, cfg):
        view = _brl_view(magnitude_pct=5.0)
        top = self._top_structure(view, brl_snapshot, cfg)
        sizing = compute_sizing(view, brl_snapshot, top, cfg)
        assert len(sizing.take_profit_levels) == 3   # scale_1, scale_2, runner

    def test_take_profit_levels_absent_without_target(self, brl_snapshot, cfg):
        view = _brl_view(magnitude_pct=None)
        top = self._top_structure(view, brl_snapshot, cfg)
        sizing = compute_sizing(view, brl_snapshot, top, cfg)
        assert len(sizing.take_profit_levels) == 0

    def test_sizing_notes_are_non_empty(self, brl_snapshot, cfg):
        view = _brl_view(budget_usd=200_000)
        top = self._top_structure(view, brl_snapshot, cfg)
        sizing = compute_sizing(view, brl_snapshot, top, cfg)
        assert len(sizing.notes) > 0

    def test_session_override_kelly_propagates(self, brl_snapshot):
        """Quarter-Kelly override from session should flow through to sizing."""
        session = SessionOverrides()
        session.apply("sizing.kelly.default_fraction", 0.25, "session", "use quarter-Kelly")
        cfg_override = resolve(load_config(), None, session)

        view = _brl_view(direction_conviction="medium", budget_usd=200_000)
        top = select_structures(view, cfg_override).shortlist[0]
        sizing = compute_sizing(view, brl_snapshot, top, cfg_override)
        assert sizing.kelly_fraction == pytest.approx(0.25)

    def test_higher_atr_multiple_produces_wider_stop(self, brl_snapshot):
        """Overriding ATR multiple to 3.0 should give a wider stop."""
        session = SessionOverrides()
        session.apply("sizing.stop.atr_multiple", 3.0, "session", "wider stops")
        cfg_wide = resolve(load_config(), None, session)
        cfg_base = load_config()

        view = _brl_view()
        top = select_structures(view, cfg_base).shortlist[0]
        sz_wide = compute_sizing(view, brl_snapshot, top, cfg_wide)
        sz_base = compute_sizing(view, brl_snapshot, top, cfg_base)

        assert sz_wide.stop_distance_pct > sz_base.stop_distance_pct


# ---------------------------------------------------------------------------
# Critique engine
# ---------------------------------------------------------------------------

class TestCritiqueEngine:
    def _run_critique(self, pm_id, view, snapshot, cfg):
        from knowledge_engine.structure_selector import select_structures
        selector_result = select_structures(view, cfg)
        return evaluate_structure(view, pm_id, selector_result)

    def test_matching_recommended_structure_is_appropriate(self, brl_snapshot, cfg):
        view = _brl_view(direction_conviction="high")
        selector_result = select_structures(view, cfg)
        rec_id = selector_result.shortlist[0].structure_id
        critique = evaluate_structure(view, rec_id, selector_result)
        assert critique.verdict == "appropriate_for_view"

    def test_rko_call_is_suboptimal_not_misaligned(self, brl_snapshot, cfg):
        """RKO call is path dependent — should be flagged but not materially misaligned."""
        view = _brl_view(direction_conviction="high")
        selector_result = select_structures(view, cfg)
        critique = evaluate_structure(view, "rko_call", selector_result)
        assert critique.verdict in ("suboptimal_but_defensible", "materially_misaligned")

    def test_rko_call_flags_barrier_path_risk(self, brl_snapshot, cfg):
        view = _brl_view(direction_conviction="high")
        selector_result = select_structures(view, cfg)
        critique = evaluate_structure(view, "rko_call", selector_result)
        assert "barrier" in critique.scenario_weakness.lower() or "knock" in critique.scenario_weakness.lower()

    def test_digital_flags_binary_gamma(self, brl_snapshot, cfg):
        view = _brl_view(direction_conviction="high")
        selector_result = select_structures(view, cfg)
        critique = evaluate_structure(view, "european_digital", selector_result)
        assert "binary" in critique.gamma_notes.lower() or "expiry" in critique.gamma_notes.lower()

    def test_critique_has_recommended_alternative(self, brl_snapshot, cfg):
        view = _brl_view(direction_conviction="high")
        selector_result = select_structures(view, cfg)
        critique = evaluate_structure(view, "rko_call", selector_result)
        assert critique.recommended_alternative is not None

    def test_all_dimensions_populated(self, brl_snapshot, cfg):
        view = _brl_view(direction_conviction="high")
        selector_result = select_structures(view, cfg)
        critique = evaluate_structure(view, "vanilla_call", selector_result)
        assert len(critique.ev_comparison_note) > 0
        assert len(critique.scenario_weakness) > 0
        assert len(critique.execution_notes) > 0
        assert len(critique.gamma_notes) > 0
        assert len(critique.hedge_effectiveness) > 0

    def test_risk_reversal_notes_unlimited_loss(self, brl_snapshot, cfg):
        view = _brl_view(direction_conviction="high")
        selector_result = select_structures(view, cfg)
        critique = evaluate_structure(view, "risk_reversal", selector_result)
        assert "unlimited" in critique.ev_comparison_note.lower() or "sold leg" in critique.ev_comparison_note.lower()
