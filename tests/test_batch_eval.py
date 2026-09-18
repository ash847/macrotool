"""Tests for the batch cross-trade pivot compute layer.

Covers the pure `compute_structure_evaluation` extraction and the invariants the
Batch pivot relies on: exhaustive driver decomposition, stable cross-trade
variant identity, and gated variants being absent (not zero) for a trade.
"""

from __future__ import annotations

import math

import pytest

from conversation.flow import ConversationFlow
from data.snapshot_loader import load_snapshot
from knowledge_engine.models import TradeView
from pricing.forwards import rate_context_for_snapshot
from interface.structure_eval import (
    DRIVER_BUCKETS,
    compute_structure_evaluation,
    driver_contribs,
    target_price,
)


def _run(pair: str, horizon_days: int, target: float) -> ConversationFlow:
    """Build + run one trade exactly as the Batch page does, without the LLM."""
    snapshot = load_snapshot()
    ccy = snapshot.get(pair)
    rate_ctx = rate_context_for_snapshot(ccy, horizon_days / 365.0)
    fwd = rate_ctx.forward
    direction = "base_higher" if target > fwd else "base_lower"
    magnitude_pct = abs(target / fwd - 1.0) * 100.0

    flow = ConversationFlow(snapshot=snapshot)
    flow.view = TradeView(
        pair=pair,
        direction=direction,
        direction_conviction="medium",
        horizon_days=horizon_days,
        magnitude_pct=magnitude_pct,
    )
    flow.ccy = ccy
    flow.structure_constraint = "No restriction"
    flow.primary_objective = "Balanced"
    flow.trade_management = "Standard hold"
    flow.target_rr = 3.0
    flow._run_engines()
    return flow


def _eval(pair: str, horizon_days: int, target: float):
    flow = _run(pair, horizon_days, target)
    res = compute_structure_evaluation(flow, target_price(flow))
    assert res is not None, f"no evaluation for {pair} {horizon_days}d {target}"
    return res


def test_driver_buckets_are_exhaustive():
    """Every grid column maps to exactly one driver bucket — no overlap, no gaps."""
    all_cols = [c for cols in DRIVER_BUCKETS.values() for c in cols]
    assert len(all_cols) == len(set(all_cols)), "a column appears in two buckets"
    from analytics.scenario_generator import GRID_COLS
    assert set(all_cols) == set(GRID_COLS), "buckets do not cover GRID_COLS exactly"


def test_drivers_sum_back_to_weighted_pnl():
    """Carry + Directional + Adverse + Vega must reconstruct the context P&L."""
    res = _eval("USDBRL", 91, 5.60)
    assert res.variants
    for ve in res.variants:
        buckets = ve.drivers
        assert set(buckets) == set(DRIVER_BUCKETS)
        assert math.isclose(sum(buckets.values()), ve.score_pct, abs_tol=1e-9)


def test_delta_is_context_minus_baseline():
    res = _eval("USDBRL", 91, 5.60)
    for ve in res.variants:
        assert math.isclose(ve.delta_pct, ve.score_pct - ve.score_base_pct, abs_tol=1e-12)


def test_variant_identity_is_stable_across_trades():
    """The (struct · variant_label) key aligns the same variant across trades —
    this is what makes the by-variant pivot meaningful."""
    a = _eval("USDBRL", 91, 5.60)
    b = _eval("USDBRL", 182, 6.00)
    keys_a = {(ve.structure_id, ve.variant_label) for ve in a.variants}
    keys_b = {(ve.structure_id, ve.variant_label) for ve in b.variants}
    assert keys_a & keys_b, "no shared variant identity across two USDBRL trades"


def test_driver_contribs_on_empty_score():
    """A degenerate score with no cells returns zeroed buckets, not a crash."""
    class _Empty:
        cells: list = []
    out = driver_contribs(_Empty())
    assert out == {b: 0.0 for b in DRIVER_BUCKETS}
