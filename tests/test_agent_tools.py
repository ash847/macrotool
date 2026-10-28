"""Phase 3 tests — the tool dispatch layer in isolation (no LLM)."""

import pytest

from agentic.session import AgentSession
from agentic.tools import dispatch
from config.loader import load_config
from data.snapshot_loader import load_snapshot

_VIEW = {"pair": "USDBRL", "direction": "base_higher", "horizon_days": 60, "magnitude_pct": 6.0}


def _session():
    return AgentSession(snapshot=load_snapshot(), cfg=load_config())


def _usdbrl_forward(horizon_days: int = 90) -> float:
    """The USDBRL outright forward — so direction-inference tests derive their
    targets from the live forward rather than hardcoding a level."""
    from pricing.forwards import rate_context_for_snapshot
    ccy = load_snapshot().get("USDBRL")
    return rate_context_for_snapshot(ccy, horizon_days / 365.0).forward


def test_run_standard_pack_builds_and_sets_state():
    s = _session()
    content, is_error = dispatch(s, "run_standard_pack", dict(_VIEW))
    assert not is_error
    assert "RECOMMENDED STRUCTURES" in content
    assert s.pack is not None and s.view is not None
    # specific priced structures, not just family names
    assert len(s.pack.recommended) >= 1
    assert s.pack.recommended[0].variant.net_premium_pct is not None


def test_family_only_request_returns_recommended():
    s = _session()
    dispatch(s, "run_standard_pack", dict(_VIEW))
    fam = s.pack.recommended[0].structure_id
    content, is_error = dispatch(s, "price_structure", {"request": fam.replace("_", " ")})
    assert not is_error
    assert "RECOMMENDED" in content      # answered from the pack, no strikes demanded


def test_target_level_infers_direction_below_forward():
    # A level 5% below the forward → direction must be base_lower (a put), NOT
    # base_higher, and the target must be that level (not a % move).
    target = round(_usdbrl_forward(90) * 0.95, 4)
    s = _session()
    content, is_error = dispatch(
        s, "run_standard_pack", {"pair": "USDBRL", "horizon_days": 90, "target_level": target}
    )
    assert not is_error
    assert s.view.direction == "base_lower"
    assert abs(s.pack.target - target) < 1e-6


def test_target_level_infers_direction_above_forward():
    target = round(_usdbrl_forward(90) * 1.05, 4)
    s = _session()
    dispatch(s, "run_standard_pack", {"pair": "USDBRL", "horizon_days": 90, "target_level": target})
    assert s.view.direction == "base_higher"
    assert abs(s.pack.target - target) < 1e-6


def test_magnitude_requires_direction():
    s = _session()
    content, is_error = dispatch(
        s, "run_standard_pack", {"pair": "USDBRL", "horizon_days": 90, "magnitude_pct": 6.0}
    )
    assert is_error
    assert "direction" in content.lower()


@pytest.mark.parametrize("direction,sign", [("base_higher", 1), ("base_lower", -1)])
def test_percentage_move_defaults_to_spot(direction, sign):
    s = _session()
    _, error = dispatch(s, "run_standard_pack", {**_VIEW, "direction": direction})
    assert not error
    assert s.pack.target == pytest.approx(s.snapshot.get("USDBRL").spot * (1 + sign * .06))
    from agentic.shortlist import render_market_state
    assert f"{sign * .06:+.2%}" in render_market_state(s.pack, s.view)


def test_explicit_forward_percentage():
    s = _session()
    _, error = dispatch(s, "run_standard_pack", {**_VIEW, "move_reference": "forward"})
    assert not error
    assert s.pack.target == pytest.approx(_usdbrl_forward(60) * 1.06)


def test_percentage_target_preserved_when_tail_preference_changes():
    s = _session()
    dispatch(s, "run_standard_pack", dict(_VIEW))
    target = s.pack.target
    from agentic.tools import _set_tail_constraint
    _set_tail_constraint(s, {"tail_constraint": "none"})
    assert s.pack.target == pytest.approx(target)


def test_cache_reuse_no_recompute():
    s = _session()
    dispatch(s, "run_standard_pack", dict(_VIEW))
    first_pack = s.pack
    content, is_error = dispatch(s, "run_standard_pack", dict(_VIEW))
    assert not is_error
    assert "reused cached pack" in content
    assert s.pack is first_pack          # same object → no recompute
    assert len(s._cache) == 1


def test_price_structure_requires_pack():
    s = _session()
    content, is_error = dispatch(s, "price_structure", {"request": "34Δ vs 25Δ 1x1.5"})
    assert is_error
    assert "run_standard_pack" in content


def test_price_structure_after_pack():
    s = _session()
    dispatch(s, "run_standard_pack", dict(_VIEW))
    content, is_error = dispatch(s, "price_structure", {"request": "34Δ vs 25Δ 1x1.5"})
    assert not is_error
    assert "PM-REQUESTED STRUCTURE" in content
    assert len(s.priced) == 1


def test_price_structure_clarification_is_not_error():
    s = _session()
    dispatch(s, "run_standard_pack", dict(_VIEW))
    content, is_error = dispatch(s, "price_structure", {"request": "34Δ vs 25Δ"})
    assert not is_error                  # ambiguity → ask the PM
    assert "1x1.5" in content


def test_unsupported_pair_errors():
    s = _session()
    content, is_error = dispatch(
        s, "run_standard_pack", {"pair": "USDZZZ", "direction": "base_higher", "horizon_days": 30}
    )
    assert is_error
    assert "Unsupported pair" in content


def test_malformed_structure_request_errors():
    s = _session()
    dispatch(s, "run_standard_pack", dict(_VIEW))
    content, is_error = dispatch(s, "price_structure", {"request": "vanilla 120Δ"})
    assert is_error


# ---------------------------------------------------------------------------
# Strikes by default (a bare number is a strike, never guessed as a delta)
# ---------------------------------------------------------------------------

def _pack(pair, direction, target):
    s = _session()
    _, is_error = dispatch(s, "run_standard_pack", {
        "pair": pair, "direction": direction, "horizon_days": 90, "target_level": target})
    assert not is_error
    return s


def _strikes(content):
    line = next(l for l in content.splitlines() if l.strip().startswith("strikes="))
    return line.split("strikes=[", 1)[1].split("]", 1)[0]


def test_brl_reprice_at_strike_levels_prices_those_strikes_and_says_so():
    """The MCP failure: '1x1 5.00 / 4.80' on USDBRL was read as 5Δ/4.8Δ and crashed."""
    s = _pack("USDBRL", "base_lower", 4.8)
    content, is_error = dispatch(s, "price_structure", {"request": "1x1 5.00 / 4.80"})
    assert not is_error
    assert content.startswith("LEG INPUTS: 5 = strike; 4.8 = strike")
    assert _strikes(content) == "5.0000, 4.8000"


@pytest.mark.parametrize("pair, target, ask, strikes", [
    ("USDMXN", 16.5, "1x1 17.2 / 16.7", "17.2000, 16.7000"),   # was silently 17Δ/17Δ
    ("USDTRY", 47.0, "1x1 48.5 / 47.5", "48.5000, 47.5000"),   # was silently 48Δ/47Δ
])
def test_strike_levels_that_look_like_deltas_are_priced_as_strikes(pair, target, ask, strikes):
    s = _pack(pair, "base_lower", target)
    content, is_error = dispatch(s, "price_structure", {"request": ask})
    assert not is_error and _strikes(content) == strikes


def test_marked_deltas_still_place_on_the_smile():
    s = _pack("USDBRL", "base_lower", 4.8)
    content, _ = dispatch(s, "price_structure", {"request": "1x1 40Δ / 20Δ"})
    assert "LEG INPUTS: 40Δ = delta; 20Δ = delta" in content
    assert "PM-REQUESTED STRUCTURE: 1x1_spread 40Δ/20Δ" in content


def test_implausible_strike_is_refused_with_a_delta_hint():
    s = _pack("USDJPY", "base_higher", 162.0)
    content, is_error = dispatch(s, "price_structure", {"request": "25 vanilla"})
    assert not is_error and "PM-REQUESTED STRUCTURE" not in content
    assert "Strike 25 is" in content and "write it as 25Δ" in content
    assert not s.priced


def test_reversed_spread_strikes_are_refused():
    s = _pack("USDBRL", "base_lower", 4.8)
    content, _ = dispatch(s, "price_structure", {"request": "1x1 4.80 / 5.00"})
    assert "long strike must be above the short strike" in content
    assert not s.priced


def test_unexpected_tool_failure_is_an_error_result_not_an_exception(monkeypatch):
    """Any exception inside a tool becomes an is_error result, so the agent loop
    always appends a tool_result and the conversation history stays valid."""
    import agentic.tools as T
    s = _pack("USDBRL", "base_lower", 4.8)

    def boom(*_a, **_k):
        raise ValueError("delta 0.048 outside supported range [0.10, 0.50]")
    monkeypatch.setattr(T, "price_structure", boom)
    content, is_error = dispatch(s, "price_structure", {"request": "1x1 40Δ / 20Δ"})
    assert is_error
    assert content == "Tool 'price_structure' failed: ValueError: delta 0.048 outside supported range [0.10, 0.50]"
