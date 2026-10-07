from types import SimpleNamespace

import pytest

from agentic.standard_pack import build_pack
from config.loader import load_config
from data.snapshot_loader import load_snapshot
from interface.prefs import MERGED_PREF_OPTIONS
from knowledge_engine.models import TradeView
from knowledge_engine.preference_policy import preference_exclusion_reason
from agentic.price_structure import price_structure, PricingUnavailable


def test_menu_removes_early_monetisation():
    assert len(MERGED_PREF_OPTIONS) == 3
    assert all("May monetise early" not in fields for fields in MERGED_PREF_OPTIONS.values())


def test_custom_nonvanilla_cannot_bypass_vanilla_only():
    result = price_structure("1x1 40Δ/20Δ", SimpleNamespace(), is_call=True,
                             structure_constraint="Avoid capped structures")
    assert isinstance(result, PricingUnavailable)
    assert "vanilla-only" in result.detail


@pytest.mark.parametrize("status,bound,allowed", [
    ("bounded", 0.05, True), ("bounded", 0.0, True), ("unknown", None, False),
    ("unbounded", None, False), ("bounded", float("inf"), False),
    ("bounded", float("nan"), False), ("bounded", -1, False),
])
def test_defined_loss_requires_finite_verified_bound(status, bound, allowed):
    variant = SimpleNamespace(economics=SimpleNamespace(contractual_loss_status=status,
                              contractual_max_loss_pct=bound), can_lose_beyond_premium=True)
    assert (preference_exclusion_reason("1x2x1_spread", variant, "Avoid tail-risky structures") is None) == allowed


@pytest.mark.parametrize("direction", ["base_higher", "base_lower"])
@pytest.mark.parametrize("constraint", ["Avoid capped structures", "Avoid tail-risky structures"])
def test_ranked_and_fallback_packs_obey_preferences(direction, constraint):
    snapshot = load_snapshot()
    config = load_config()
    for magnitude in (6.0, None):
        view = TradeView(pair="GBPUSD", direction=direction, direction_conviction="medium",
                         horizon_days=90, magnitude_pct=magnitude, mode="recommend")
        pack = build_pack(view, snapshot.get("GBPUSD"), config, structure_constraint=constraint)
        assert pack.recommended
        assert all(preference_exclusion_reason(rec.structure_id, rec.variant, constraint) is None
                   for rec in pack.recommended)
        assert all(rec.structure_id != "linear" for rec in pack.recommended)
        if constraint == "Avoid capped structures":
            assert {rec.structure_id for rec in pack.recommended} == {"vanilla"}
        else:
            assert not {"1x2_spread", "1x1.5_spread", "seagull"} & {rec.structure_id for rec in pack.recommended}
