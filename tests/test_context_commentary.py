"""Context commentary (scoring-philosophy verbalization) + driver decomposition.

Covers: commentary/glossary resolution from the local config and per-profile;
driver_contribs exhaustiveness (sums back to score_pct) after moving into the engine;
and _migrate_config_shape tolerating configs without the new keys.
"""

import pytest

import knowledge_engine.scenario_weighter as sw
from knowledge_engine.scenario_scorer import (
    CellBreakdown,
    DRIVER_BUCKETS,
    ScoreResult,
    driver_contribs,
)


def test_local_commentary_and_glossary_present():
    sw.clear_scenario_weights_cache()
    c = sw.get_context_commentary("classic_carry")
    assert c.get("market_behavior") and c.get("trade_guidance")
    g = sw.get_driver_glossary()
    assert {"Carry", "Directional", "Adverse", "Vega"}.issubset(g.keys())
    sw.clear_scenario_weights_cache()


def test_commentary_absent_context_is_empty():
    sw.clear_scenario_weights_cache()
    assert sw.get_context_commentary("does_not_exist") == {}
    assert sw.get_context_commentary(None) == {}
    sw.clear_scenario_weights_cache()


def test_driver_contribs_is_exhaustive():
    """Every grid column maps to exactly one bucket, so the bucket totals sum back to
    the score — the property the agent relies on to explain P&L."""
    cols = [c for cols in DRIVER_BUCKETS.values() for c in cols]
    cells = [
        CellBreakdown(
            scenario_id=c, row="25%T", col=c, pnl_pct=0.0, pnl_ccy=None,
            multiplier=1.0, normalized_weight=0.1, contrib_pct=0.01 * (i + 1), contrib_ccy=None,
        )
        for i, c in enumerate(cols)
    ]
    score = ScoreResult(score_pct=sum(c.contrib_pct for c in cells), score_ccy=None, cells=cells)
    d = driver_contribs(score)
    assert set(d) == set(DRIVER_BUCKETS)
    assert sum(d.values()) == pytest.approx(score.score_pct)


def test_migrate_tolerates_missing_commentary():
    cfg = sw._migrate_config_shape({
        "base_weightings": [{"id": "x", "when": [], "multipliers": {}}],
        "preference_overlays": [],
    })
    assert cfg["base_weightings"][0]["id"] == "x"          # no crash without commentary
    assert sw.get_context_commentary("x", None) is not None  # absent → {} not error


def test_commentary_is_profile_aware(monkeypatch):
    sw.clear_scenario_weights_cache()
    monkeypatch.setattr("interface.security.can_have_personal_weights", lambda e: e == "vip@x.com")
    monkeypatch.setattr("interface.supabase_logger.personal_weights_key", lambda e: f"scenario_definitions::{e}")

    def cfg(marker):
        return {
            "baseline": 1.0, "min_multiplier": 0.1, "preference_overlays": [],
            "base_weightings": [{"id": "classic_carry", "when": [], "multipliers": {},
                                 "commentary": {"market_behavior": marker}}],
        }

    store = {"scenario_definitions": cfg("GLOBAL"), "scenario_definitions::vip@x.com": cfg("PERSONAL")}
    monkeypatch.setattr(
        "interface.supabase_logger.fetch_config_for_engine_with_meta",
        lambda k: (store.get(k), "supabase" if k in store else "missing"),
    )
    assert sw.get_context_commentary("classic_carry", "vip@x.com")["market_behavior"] == "PERSONAL"
    assert sw.get_context_commentary("classic_carry", "other@x.com")["market_behavior"] == "GLOBAL"
    sw.clear_scenario_weights_cache()
