from dataclasses import replace
from types import SimpleNamespace

import pytest

from agentic.standard_pack import build_pack
from agentic.shortlist import present_shortlist, render_shortlist, shortlist_reference
from agentic.session import AgentSession
from agentic.tools import dispatch
from analytics.distributions import interpolate_atm_vol
from config.loader import load_config
from data.snapshot_loader import load_snapshot
from knowledge_engine.models import TradeView
from pricing.forwards import rate_context_for_snapshot
from tests._curves import stated_lognormal


@pytest.mark.parametrize("direction", ["base_higher", "base_lower"])
@pytest.mark.parametrize("method", ["fixed_loss", "kelly"])
def test_all_ranked_variants_match_trade_view(direction, method, monkeypatch):
    import interface.structure_eval as trade_view
    snapshot = load_snapshot()
    currency = snapshot.get("USDBRL")
    config = load_config()
    view = TradeView(pair="USDBRL", direction=direction, direction_conviction="medium",
                     horizon_days=90, magnitude_pct=6, mode="recommend")
    forward = rate_context_for_snapshot(currency, 90 / 365).forward
    probs, bins = stated_lognormal(forward * (1.04 if direction == "base_higher" else 0.96),
                                   interpolate_atm_vol(currency, 90), 90 / 365)
    pack = build_pack(view, currency, config, linear_notional=1_000_000,
                      sizing_method=method, kelly_probs=probs, kelly_bins=bins)
    monkeypatch.setattr(trade_view, "sizing_capital", lambda: 1_000_000)
    flow = SimpleNamespace(market_state=pack.market_state, selector_result=pack.selector_result,
                           view=view, ccy=currency, target_rr=3, sizing_spec=pack.sizing_spec)
    evaluation = trade_view.compute_structure_evaluation(flow, pack.target)
    assert pack.variants_ranked
    assert len(pack.recommended) == len(evaluation.variants)
    assert len({rec.structure_id for rec in pack.recommended}) < len(pack.recommended)
    for rec, expected in zip(pack.recommended, evaluation.variants):
        assert (rec.structure_id, rec.variant.variant_label) == (expected.structure_id, expected.variant_label)
        assert rec.variant.strikes == pytest.approx(expected.pv.strikes)
        assert rec.variant.structure_notional == pytest.approx(expected.pv.structure_notional)
        assert rec.variant.net_premium_pct == pytest.approx(expected.pv.net_premium_pct)
        assert rec.score_ccy == pytest.approx(expected.score.score_ccy)
        assert rec.score_pct == pytest.approx(expected.score.score_pct)
        assert rec.absolute_contribution_total_pct == pytest.approx(
            sum(abs(cell.contrib_pct) for cell in expected.score.cells)
        )
    assert any(rec.structure_id == "linear" for rec in pack.recommended)
    table = render_shortlist(pack, view)
    assert "### Market state" in table and "### Shortlisted structures" in table
    assert "### Top structures" in table
    assert "Net P&L at target" not in table and "Additional loss beyond premium?" not in table
    reply = present_shortlist("A regime explanation. Selection fits that regime.", pack, view, automatic=True)
    assert reply.index("Market state") < reply.index("A regime explanation") < reply.index("Shortlisted structures") < reply.index("Top structures")
    session = AgentSession(snapshot=snapshot, cfg=config, view=view, pack=pack)
    reference = shortlist_reference(pack, view)
    family = next(rec.structure_id for rec in pack.recommended if sum(other.structure_id == rec.structure_id for other in pack.recommended) > 1)
    content, error = dispatch(session, "inspect_recommendations", {"shortlist_ref": reference, "family": family})
    assert not error
    for rec in pack.recommended:
        if rec.structure_id == family:
            assert f"Engine rank {rec.rank}:" in content


def test_no_target_is_not_labelled_as_scenario_ranked():
    snapshot = load_snapshot()
    view = TradeView(pair="USDBRL", direction="base_higher", direction_conviction="medium",
                     horizon_days=90, mode="recommend")
    pack = build_pack(view, snapshot.get(view.pair), load_config())
    assert not pack.variants_ranked
    assert "Scenario ranking unavailable" in render_shortlist(pack, view)
