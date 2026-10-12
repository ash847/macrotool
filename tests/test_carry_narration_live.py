"""Opt-in real-model narration checks; responses need semantic review as well."""

import os
from types import SimpleNamespace

import pytest

from agentic.agent_flow import build_system_prompt
from agentic.agent_llm import AnthropicToolLLM
from agentic.render import _carry_explanation
from analytics.market_state import compute_market_state


@pytest.mark.skipif(os.getenv("RUN_LIVE_CARRY_TESTS") != "1", reason="Opt-in paid model calls")
@pytest.mark.parametrize("days,forward,target,direction", [
    (365, 6.5111, 6.75, "base_higher"),
    (90, 6.6676, 6.7250, "base_higher"),
    (365, 6.5111, 6.4, "base_lower"),
    (90, 6.6676, 6.6, "base_lower"),
])
def test_live_carry_narration(days, forward, target, direction):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key and os.getenv("LIVE_SECRETS_FILE"):
        import tomllib
        with open(os.environ["LIVE_SECRETS_FILE"], "rb") as source:
            key = tomllib.load(source).get("ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("No Anthropic key configured")
    state = compute_market_state(spot=6.7125, fwd=forward, vol=0.033 if days == 365 else 0.023,
                                 T=days / 365, r_d=0.01, r_f=0.04, direction=direction)
    view = SimpleNamespace(pair="USDCNH", direction=direction, horizon_days=days)
    context = _carry_explanation(state, view)
    llm = AnthropicToolLLM(api_key=key)
    request = (
        f"PM request: $cnh {days}d {target}.\nEngine market context: {context}\n"
        f"ATM vol: {state.vol:.1%}. No historical vol observations or path assumptions supplied.\n"
        "Give only the initial 2–3 sentence market commentary, using these supplied engine facts. "
        "No tools or tables are required for this narration check."
    )
    response = llm.create([llm.format_user(request)], build_system_prompt(["USDCNH"]), [])
    assert response.stop_reason == "end_turn"
    assert response.text.strip() and not response.tool_calls
    print(f"\nLIVE CASE {days}d {direction} target={target}\n{response.text}\n")
    lower = response.text.lower()
    for phrase in ("overcoming that roll-down", "creates meaningful friction", "vol compresses",
                   "premiums are compressed", "compresses option premiums", "slow-grinding",
                   "low-initial-delta", "decay-resistant", "market already prices in"):
        assert phrase not in lower
    assert f"vol is {forward}" not in lower
