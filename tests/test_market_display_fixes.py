from dataclasses import replace
from types import SimpleNamespace

import pytest

from agentic.agent_flow import SYSTEM_PROMPT
from agentic.render import _sizing_explanation, render_pack
from agentic.shortlist import render_market_state
from agentic.standard_pack import build_pack
from analytics.structure_pricer import PricedVariant, _size_variant
from config.loader import load_config
from data.snapshot_loader import load_snapshot
from knowledge_engine.models import TradeView


@pytest.fixture(scope="module")
def context():
    snapshot = load_snapshot()
    view = TradeView(pair="USDJPY", direction="base_lower", direction_conviction="medium",
                     horizon_days=90, magnitude_pct=3.3, mode="recommend")
    return view, build_pack(view, snapshot.get("USDJPY"), load_config(), linear_notional=1_000_000)


def test_market_target_is_from_spot_and_budget_is_common_input(context):
    view, pack = context
    text = render_market_state(pack, view)
    assert "Target (%)" in text
    assert f"{pack.target / pack.market_state.spot - 1:+.2%}" in text
    assert "Loss budget*" in text
    assert "common input" in text
    assert "notional caps and sizing policies" in text
    assert "N/A — Kelly sizing" in render_market_state(replace(pack, sizing_method="kelly"), view)


def test_remote_commentary_cannot_be_treated_as_exclusion_evidence(context, monkeypatch):
    view, pack = context
    monkeypatch.setattr("knowledge_engine.scenario_weighter.get_context_commentary",
                        lambda _: {"trade_guidance": "Avoid seagulls."})
    pack = replace(pack, active_context="directional_low_carry",
                   selector_result=SimpleNamespace(shortlist=[SimpleNamespace(display_name="Seagull")]))
    text = render_pack(pack, view)
    assert "NOT an exclusion log" in text
    assert "NOT regime-excluded): Seagull" in text
    assert "ignore any legacy touch/retrace" in text
    assert "no path-dependent products" in SYSTEM_PROMPT


def test_net_credit_is_single_sizing_policy():
    trade = PricedVariant(variant_label="credit", strikes=[100, 110], barrier=None,
                          net_premium_pct=-0.01, breakeven=None, payoff_at_target_pct=None,
                          rr_at_target=None, max_loss_pct=0.01, wing_ratio=None, is_zero_cost=False)
    _size_variant(trade, 100, 1000)
    text = _sizing_explanation(trade, "USD")
    assert trade.structure_notional == 10000
    assert "INSTEAD OF budget-based sizing" in text
    assert "not an additional sizing adjustment" in text


def settings_app():
    import streamlit as st
    from datetime import date
    from types import SimpleNamespace
    from interface.agent_settings_ui import render_agent_settings
    from workspace.settings import ChatSettings

    st.session_state.ws_conv = SimpleNamespace(id="test-settings", settings=ChatSettings().to_dict(), distributions={})
    pack = SimpleNamespace(sizing_method="fixed_loss", loss_budget=12345, kelly_fallback=False)
    session = SimpleNamespace(pack=pack, view=SimpleNamespace(pair="USDJPY"), expiry_for=lambda _: date(2027, 1, 1))
    st.session_state.agent_flow = SimpleNamespace(session=session)
    render_agent_settings(None, busy=False, capital=1_000_000, set_capital=lambda _: None,
                          capital_ccy="USD", on_error=lambda *args: None)


def test_settings_displays_applied_budget_and_footnote():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(settings_app).run()
    assert not app.exception
    assert any("12,345 USD" in item.value for item in app.markdown)
    assert any("notional caps and sizing policies" in item.value for item in app.caption)
