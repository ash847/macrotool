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
    content, is_error = dispatch(s, "price_structure", {"request": "34 vs 25 1x1.5"})
    assert is_error
    assert "run_standard_pack" in content


def test_price_structure_after_pack():
    s = _session()
    dispatch(s, "run_standard_pack", dict(_VIEW))
    content, is_error = dispatch(s, "price_structure", {"request": "34 vs 25 1x1.5"})
    assert not is_error
    assert "PM-REQUESTED STRUCTURE" in content
    assert len(s.priced) == 1


def test_price_structure_clarification_is_not_error():
    s = _session()
    dispatch(s, "run_standard_pack", dict(_VIEW))
    content, is_error = dispatch(s, "price_structure", {"request": "34 vs 25"})
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
