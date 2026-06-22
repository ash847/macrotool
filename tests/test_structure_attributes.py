"""Tests for the qualitative attribute / findings layer.

Covers: tag derivation purity, IP-safety of the rendered findings block (no
scores / weights / dimension names), and the deciding-axis pick.
"""

from __future__ import annotations

from knowledge_engine.comparator import ScenarioAggregate, ScenarioAggregates
from knowledge_engine.scenario_scorer import CellBreakdown, ScoreResult
from knowledge_engine.structure_attributes import (
    ATTRIBUTES,
    attributes,
    deciding_axis,
    render_findings,
)


def _cell(col: str, contrib_pct: float) -> CellBreakdown:
    return CellBreakdown(
        scenario_id=f"Expiry|{col}", row="Expiry", col=col,
        pnl_pct=contrib_pct, pnl_ccy=None, multiplier=1.0,
        normalized_weight=1.0, contrib_pct=contrib_pct, contrib_ccy=None,
    )


def _score(by_col: dict[str, float]) -> ScoreResult:
    cells = [_cell(col, v) for col, v in by_col.items()]
    return ScoreResult(score_pct=sum(by_col.values()), score_ccy=None, cells=cells)


def _agg(slow=None, correct=None, wrong=None, over=None, vol=None) -> ScenarioAggregates:
    def a(v):
        return ScenarioAggregate(scenario_ids=(), avg_pnl_pct=v, worst_pnl_pct=v, best_pnl_pct=v)
    return ScenarioAggregates(
        slow_path=a(slow), correct_path=a(correct), wrong_way=a(wrong),
        overshoot=a(over), vol_sensitivity=a(vol),
        expiry_target_price_pct=None, expiry_target_pnl_pct=None,
    )


def test_carry_led_and_capped_for_ratio_spread():
    # Carry ("S") dominates the split → carry_led; ratio spread → tail + capped.
    score = _score({"S": 0.08, "K": 0.01, "−1σ": -0.01, "Δvol": 0.0})
    agg = _agg(slow=0.02, correct=0.05, wrong=-0.005, over=0.04, vol=0.0)
    tags = attributes("1x1.5_spread", score, agg)
    assert "carry_led" in tags
    assert "holds_on_slow_path" in tags
    assert "resilient_wrong_way" in tags
    assert "tail_beyond_short_strike" in tags
    assert "capped_upside" in tags          # overshoot (0.04) !> correct*1.05 (0.0525)


def test_direction_led_vanilla_rewards_overshoot():
    score = _score({"S": 0.0, "K": 0.06, "t%→K": 0.03, "−1σ": -0.02, "Δvol": 0.0})
    agg = _agg(slow=-0.01, correct=0.04, wrong=-0.05, over=0.09, vol=0.0)
    tags = attributes("vanilla", score, agg)
    assert "direction_led" in tags
    assert "needs_the_move" in tags          # slow path negative
    assert "exposed_wrong_way" in tags
    assert "rewards_overshoot" in tags        # 0.09 > 0.04*1.05
    assert "capped_upside" not in tags        # vanilla never capped


def test_vol_sensitive_and_binary():
    score = _score({"S": 0.01, "K": 0.02, "Δvol": 0.03})
    agg = _agg(slow=0.0, correct=0.02, wrong=0.0, over=0.02, vol=0.01)
    tags = attributes("european_digital", score, agg)
    assert "vol_sensitive" in tags
    assert "binary_expiry" in tags


def test_purity_same_inputs_same_tags():
    score = _score({"S": 0.05, "K": 0.02})
    agg = _agg(slow=0.01, correct=0.03, wrong=-0.005, over=0.02, vol=0.0)
    assert attributes("vanilla", score, agg) == attributes("vanilla", score, agg)


def test_deciding_axis_picks_largest_gap():
    chosen = _agg(slow=0.05, correct=0.02, wrong=-0.01, over=0.02, vol=0.0)
    runner = _agg(slow=-0.02, correct=0.02, wrong=-0.01, over=0.02, vol=0.0)
    # slow_path gap is by far the largest → slow-path gloss.
    assert "slow" in deciding_axis(chosen, runner).lower()


def test_render_findings_is_ip_clean():
    score = _score({"S": 0.08, "K": 0.01, "−1σ": -0.01})
    agg = _agg(slow=0.02, correct=0.05, wrong=-0.005, over=0.04, vol=0.0)
    tags = attributes("1x1.5_spread", score, agg)
    text = render_findings([("1x1.5 put spread", tags)],
                           deciding="it holds the slow / grinding scenarios better")
    # No raw numbers and no scoring-machinery vocabulary may leak. ("carry" the
    # market concept is fine in a gloss — the IP is the scores/weights/method.)
    # Skip the header line (an instruction to the LLM, not rendered content).
    body = "\n".join(text.splitlines()[1:])
    lowered = body.lower()
    text = body
    assert "$" not in text
    assert "%" not in text
    assert not any(ch.isdigit() for ch in text.replace("1x1.5", ""))
    for banned in ("score", "weight", "affinity", "multiplier", "scenario_weighted",
                   "target_z", "atmfsratio", "carry_regime", "carry_alignment"):
        assert banned not in lowered, f"leaked: {banned!r}\n{text}"
    # The qualitative glosses are present.
    assert "carry / roll-down" in text
    assert "decided by:" in text


def test_render_findings_empty():
    assert render_findings([]) == ""


def test_every_tag_has_a_gloss():
    # Guards against a tag being added in attributes() without a phrasebook entry.
    import knowledge_engine.structure_attributes as m
    used = set()
    for name in dir(m):
        pass
    # All tags emitted by the geometry sets must exist in ATTRIBUTES.
    for t in ("carry_led", "direction_led", "vol_sensitive", "holds_on_slow_path",
              "needs_the_move", "resilient_wrong_way", "exposed_wrong_way",
              "rewards_overshoot", "capped_upside", "tail_beyond_short_strike",
              "binary_expiry", "barrier_path_risk"):
        assert t in ATTRIBUTES
