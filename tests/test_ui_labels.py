"""Plain-English labels: one wording file feeds the screen and the Agent."""

import re

import pytest

from agentic.agent_flow import build_system_prompt
from agentic.session import AgentSession
from agentic.tools import dispatch
from config.loader import load_config
from data.snapshot_loader import load_snapshot
from knowledge_engine import ui_labels as UL
from knowledge_engine.loader import load_ui_labels

# Every key the UI / agent code looks up. If you add a UL.label("x") call, add "x" here.
USED_KEYS = (
    "spot", "forward", "implied_vol", "horizon", "target_distance_fwd",
    "target_distance_spot", "carry", "carry_vs_vol", "carry_payout_ratio",
    "rate_base", "rate_quote", "skew", "smile_curvature", "move_to_target",
    "stop_distance", "stop_level", "loss_budget", "bankroll", "fit_score",
    "pnl_score", "kelly_risk", "premium", "variant", "strikes", "notional",
)


@pytest.mark.parametrize("key", USED_KEYS)
def test_every_used_key_has_a_label_and_a_tip(key):
    entry = load_ui_labels()["labels"][key]
    assert entry["label"].strip() and entry["tip"].strip()


def test_placeholders_format_and_nothing_is_left_unfilled():
    assert UL.label("rate_base", ccy="USD") == "USD rate"
    assert UL.label("rate_quote", ccy="BRL") == "BRL rate (implied)"
    assert "3.0:1" in UL.label("stop_distance", rr=3.0)
    for key in ("spot", "skew", "kelly_risk"):
        assert "{" not in UL.label(key)


def test_tip_appends_the_market_term_only_where_there_is_one():
    assert "Market term: 25Δ risk reversal" in UL.tip("skew")
    assert "Market term" not in UL.tip("spot")


def test_carry_vs_vol_words_cover_every_regime():
    assert [UL.carry_vs_vol_label(r) for r in (0, 1, 2)] == ["Low", "Moderate", "High"]


def test_unknown_key_fails_loudly():
    with pytest.raises(KeyError):
        UL.label("not_a_real_label")


def test_glossary_lists_plain_label_with_its_market_term():
    glossary = UL.glossary_text()
    assert "- Skew (market term: 25Δ risk reversal (25d RR))" in glossary
    assert "- Carry vs vol (market term: carry regime)" in glossary
    assert "{ccy}" not in glossary
    assert "Spot" not in glossary          # no market term -> not a glossary entry


def test_agent_system_prompt_carries_audience_rules_and_terms():
    prompt = build_system_prompt(["USDBRL"])
    assert "AUDIENCE AND STYLE:" in prompt
    assert "new to FX-options structuring" in prompt
    assert "Do not teach unprompted" in prompt
    assert "TERMS (" in prompt
    assert "Smile curvature (market term: 25Δ butterfly" in prompt


def test_agent_pack_uses_plain_labels_not_engine_shorthand():
    session = AgentSession(snapshot=load_snapshot(), cfg=load_config())
    text, is_error = dispatch(session, "run_standard_pack", {
        "pair": "USDBRL", "horizon_days": 90, "target_level": 4.5,
    })
    assert not is_error
    assert "implied vol (ATM)=" in text
    assert "target distance from forward=" in text
    assert "carry vs vol:" in text
    for shorthand in ("atm_vol=", "target_z(fwd)", "put_call=", "regime="):
        assert shorthand not in text
    # Labels stay in step with the screen: no stray placeholder braces.
    assert not re.search(r"\{(ccy|rr)", text)
