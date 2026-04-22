"""
Smoke tests for Phase A — foundation layer.

Verifies:
  - Market snapshot loads and validates correctly
  - Config layers load and resolve without error
  - Session overrides propagate through to ResolvedConfig
  - Override detector parses [PREF_CHANGE] tags correctly
"""

import pytest
from data.snapshot_loader import load_snapshot
from config.loader import load_config, load_base_config
from config.schema import SessionOverrides
from config.resolver import resolve, explain_source
from config.override_detector import extract_overrides


# ---------------------------------------------------------------------------
# Market snapshot
# ---------------------------------------------------------------------------

class TestMarketSnapshot:
    def test_loads_without_error(self):
        snap = load_snapshot()
        assert snap is not None

    def test_all_three_pairs_present(self):
        snap = load_snapshot()
        assert "USDBRL" in snap.currencies
        assert "USDTRY" in snap.currencies
        assert "EURPLN" in snap.currencies

    def test_vol_surface_has_correct_nodes(self):
        snap = load_snapshot()
        brl = snap.get("USDBRL")
        # 5 deltas × 6 tenors = 30 nodes per currency
        assert len(brl.vol_surface) == 30

    def test_get_atm_vol(self):
        snap = load_snapshot()
        brl = snap.get("USDBRL")
        vol = brl.get_atm_vol("1M")
        assert vol == pytest.approx(0.182)

    def test_get_forward(self):
        snap = load_snapshot()
        brl = snap.get("USDBRL")
        fwd = brl.get_forward("3M")
        assert fwd is not None
        assert fwd.outright == pytest.approx(5.8970)

    def test_pln_is_deliverable(self):
        snap = load_snapshot()
        pln = snap.get("EURPLN")
        assert pln.instrument_type == "Deliverable"

    def test_brl_try_are_ndf(self):
        snap = load_snapshot()
        assert snap.get("USDBRL").instrument_type == "NDF"
        assert snap.get("USDTRY").instrument_type == "NDF"

    def test_df_curves_present(self):
        snap = load_snapshot()
        brl = snap.get("USDBRL")
        assert len(brl.usd_df_curve) == 6
        assert len(brl.eur_df_curve) == 6
        assert brl.get_usd_df("3M") == pytest.approx(0.9888)

    def test_missing_pair_raises(self):
        snap = load_snapshot()
        with pytest.raises(KeyError):
            snap.get("USDXXX")


# ---------------------------------------------------------------------------
# Config loading and resolution
# ---------------------------------------------------------------------------

class TestConfigLoading:
    def test_base_config_loads(self):
        cfg = load_base_config()
        assert cfg is not None

    def test_default_kelly_fraction(self):
        cfg = load_base_config()
        assert cfg.sizing.kelly.default_fraction == pytest.approx(0.5)

    def test_default_vol_adjustment_elevated(self):
        cfg = load_base_config()
        assert cfg.sizing.vol_regime_size_adjustment.elevated == pytest.approx(0.75)

    def test_full_load_with_profile(self):
        cfg = load_config()
        assert cfg is not None
        assert cfg.sizing is not None
        assert cfg.structures is not None

    def test_source_trace_set(self):
        cfg = load_base_config()
        assert cfg.source_trace.get("*") == "default"

    def test_get_kelly_fraction_by_conviction(self):
        cfg = load_base_config()
        assert cfg.get_kelly_fraction("high") == pytest.approx(0.75)
        assert cfg.get_kelly_fraction("low") == pytest.approx(0.25)
        assert cfg.get_kelly_fraction("medium") == pytest.approx(0.5)

    def test_get_vol_size_multiplier(self):
        cfg = load_base_config()
        assert cfg.get_vol_size_multiplier("elevated") == pytest.approx(0.75)
        assert cfg.get_vol_size_multiplier("stressed") == pytest.approx(0.5)

    def test_tranche_weights_sum_to_one(self):
        cfg = load_base_config()
        for attr in ("low_timing_conviction", "medium_timing_conviction", "high_timing_conviction"):
            schedule = getattr(cfg.sizing.tranche_entry, attr)
            assert abs(sum(schedule.weights) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Session overrides
# ---------------------------------------------------------------------------

class TestSessionOverrides:
    def test_session_override_changes_kelly(self):
        base = load_base_config()
        session = SessionOverrides()
        session.apply(
            field_path="sizing.kelly.default_fraction",
            value=0.25,
            scope="session",
            raw_text="use quarter-Kelly for this trade",
        )
        cfg = resolve(base, None, session)
        assert cfg.sizing.kelly.default_fraction == pytest.approx(0.25)

    def test_session_override_tracked_in_trace(self):
        base = load_base_config()
        session = SessionOverrides()
        session.apply("sizing.stop.atr_multiple", 3.0, "session", "tighter stops please")
        cfg = resolve(base, None, session)
        assert cfg.source_trace.get("sizing.stop.atr_multiple") == "session"

    def test_explain_source_session(self):
        base = load_base_config()
        session = SessionOverrides()
        session.apply("sizing.kelly.default_fraction", 0.25, "session", "use quarter-Kelly")
        cfg = resolve(base, None, session)
        explanation = explain_source(cfg, "sizing.kelly.default_fraction")
        assert "conversation" in explanation

    def test_explain_source_default(self):
        cfg = load_base_config()
        explanation = explain_source(cfg, "sizing.kelly.default_fraction")
        assert "default" in explanation

    def test_later_override_wins(self):
        base = load_base_config()
        session = SessionOverrides()
        session.apply("sizing.kelly.default_fraction", 0.25, "session", "first")
        session.apply("sizing.kelly.default_fraction", 0.75, "session", "second")
        cfg = resolve(base, None, session)
        assert cfg.sizing.kelly.default_fraction == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Override detector
# ---------------------------------------------------------------------------

class TestOverrideDetector:
    def test_detects_kelly_override(self):
        response = 'I will use quarter-Kelly for this. [PREF_CHANGE: {"field": "sizing.kelly.default_fraction", "value": 0.25, "scope": "session"}]'
        clean, overrides = extract_overrides(response)
        assert len(overrides) == 1
        assert overrides[0].field_path == "sizing.kelly.default_fraction"
        assert overrides[0].value == pytest.approx(0.25)
        assert overrides[0].scope == "session"

    def test_tag_stripped_from_response(self):
        response = 'Understood. [PREF_CHANGE: {"field": "sizing.kelly.default_fraction", "value": 0.25, "scope": "session"}] I will adjust accordingly.'
        clean, _ = extract_overrides(response)
        assert "[PREF_CHANGE" not in clean

    def test_unknown_field_ignored(self):
        response = '[PREF_CHANGE: {"field": "unknown.field.path", "value": 99, "scope": "session"}]'
        _, overrides = extract_overrides(response)
        assert len(overrides) == 0

    def test_malformed_json_ignored(self):
        response = '[PREF_CHANGE: {bad json here}]'
        _, overrides = extract_overrides(response)
        assert len(overrides) == 0

    def test_profile_scope_preserved(self):
        response = '[PREF_CHANGE: {"field": "sizing.kelly.default_fraction", "value": 0.5, "scope": "profile"}]'
        _, overrides = extract_overrides(response)
        assert overrides[0].scope == "profile"

    def test_out_of_range_value_ignored(self):
        response = '[PREF_CHANGE: {"field": "sizing.kelly.default_fraction", "value": 5.0, "scope": "session"}]'
        _, overrides = extract_overrides(response)
        assert len(overrides) == 0

    def test_excluded_structures_override(self):
        response = '[PREF_CHANGE: {"field": "structures.excluded_structures", "value": ["risk_reversal"], "scope": "profile"}]'
        _, overrides = extract_overrides(response)
        assert overrides[0].value == ["risk_reversal"]
