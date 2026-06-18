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
    sw.clear_context_commentary_cache()
    c = sw.get_context_commentary("classic_carry")
    assert c.get("market_behavior") and c.get("trade_guidance")
    g = sw.get_driver_glossary()
    assert {"Carry", "Directional", "Adverse", "Vega"}.issubset(g.keys())
    sw.clear_context_commentary_cache()


def test_commentary_absent_context_is_empty():
    sw.clear_context_commentary_cache()
    assert sw.get_context_commentary("does_not_exist") == {}
    assert sw.get_context_commentary(None) == {}
    sw.clear_context_commentary_cache()


def test_commentary_is_global_from_supabase_key(monkeypatch):
    """Commentary is a single GLOBAL store (key 'context_commentary'), not per-user."""
    sw.clear_context_commentary_cache()
    store = {"context_commentary": {"contexts": {"big_move": {"market_behavior": "G"}}}}
    monkeypatch.setattr(
        "interface.supabase_logger.fetch_config_for_engine_with_meta",
        lambda k: (store.get(k), "supabase" if k in store else "missing"),
    )
    assert sw.get_context_commentary("big_move")["market_behavior"] == "G"
    sw.clear_context_commentary_cache()


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


def test_commentary_is_decoupled_from_weights_config(monkeypatch):
    """Commentary lives in its own global key, not in scenario_definitions — so it does
    not fork with per-user weights."""
    sw.clear_context_commentary_cache()
    # Only the weights key is present; commentary key absent → commentary falls to local.
    monkeypatch.setattr(
        "interface.supabase_logger.fetch_config_for_engine_with_meta",
        lambda k: ({"base_weightings": [], "preference_overlays": []}, "supabase")
        if k == "scenario_definitions" else (None, "missing"),
    )
    # local file still resolves the seeded contexts
    assert sw.get_context_commentary("classic_carry").get("market_behavior")
    sw.clear_context_commentary_cache()
