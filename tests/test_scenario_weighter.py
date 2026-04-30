"""Tests for knowledge_engine.scenario_weighter.

Context structure (top-down priority):
  1.  classic_carry           with_carry, carry∈{1,2}, target≤1.25, prim∈{Bal,Hold}, trade∈{Std,Def}
  2.  cheap_carry              with_carry, carry∈{1,2}, target≤1.25, prim=Cost, trade∈{Std,Def}
  3.  conservative_carry      with_carry, carry∈{1,2}, target≤1.25, prim=Risk
  4.  delta_carry              with_carry, carry∈{1,2}, target≤1.25, prim∈{Bal,Cost,Hold}, trade=Early
  5.  big_move                 target>1.25, trade∈{Std,Early}
  6+. (existing market-state-only contexts as fallback for everything else)

Old contexts that required `with_carry=true` (carry_capture, directional_with_carry,
carry_momentum_extended) are now **unreachable** for typical PM preferences — every
combination of the 4 primary objectives pairs with the 3 trade-management values,
and one of the 4 carry-preference contexts always fires first. They are retained in
the JSON as fallbacks but are not exercised in tests.
"""

from __future__ import annotations

import pytest

from analytics.market_state import MarketState
from analytics.scenario_generator import FAMILIES
from knowledge_engine.scenario_weighter import (
    FiredContext,
    WeighterResult,
    compute_family_weights,
    load_scenario_weights_config,
)


def _ms(
    *,
    target_z: float | None = 0.7,
    carry_regime: int = 1,
    with_carry: bool = True,
    T: float = 0.25,
    vol: float = 0.10,
    atmfsratio: float | None = 1.5,
) -> MarketState:
    """Default state: carry=1, target=0.7, with-carry, vol=10%, T=3m."""
    return MarketState(
        spot=1.0, fwd=1.0, vol=vol, T=T, r_d=0.02, r_f=0.02,
        c=0.0, carry_regime=carry_regime, target_z=target_z,
        atmfsratio=atmfsratio, put_call=None, with_carry=with_carry,
    )


# Bypass token: this trade_management value lets old contexts fire because:
#   - none of classic/cheap/conservative/delta require trade=Defendable
#     specifically, so it doesn't change those — but combined with
#     with_carry=False or carry=0, those four are excluded anyway.
#   - big_move requires trade∈{Std,Early}, so this value bypasses big_move.
_BYPASS_TM = "Need defendable mark-to-market"


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------

class TestStructuralInvariants:
    def test_default_state_fires_classic_with_uniform_weights(self):
        """Default ms (carry=1, with-carry, target=0.7) + default prefs
        → classic_carry fires; its empty adjustments leave weights uniform."""
        result = compute_family_weights(_ms())
        assert isinstance(result, WeighterResult)
        for fam in FAMILIES:
            assert result.weights[fam] == pytest.approx(1.0 / len(FAMILIES))
        assert len(result.fired) == 1
        assert result.fired[0].id == "classic_carry"

    def test_at_most_one_context_fires(self):
        states = [
            _ms(target_z=2.0, carry_regime=2, with_carry=True),
            _ms(target_z=0.2, carry_regime=0, with_carry=False, T=0.75, vol=0.25),
            _ms(target_z=None, carry_regime=2, with_carry=False),
            _ms(T=0.05),
        ]
        for ms in states:
            r = compute_family_weights(ms)
            assert len(r.fired) <= 1, f"Expected ≤1 context, got {[c.id for c in r.fired]}"

    def test_weights_always_sum_to_one(self):
        states = [
            _ms(target_z=2.0, carry_regime=2, with_carry=True, T=0.05, vol=0.30),
            _ms(target_z=0.2, carry_regime=0, with_carry=False, T=0.75, vol=0.15),
            _ms(target_z=None, carry_regime=2, with_carry=False, T=0.10, vol=0.25),
            _ms(),
        ]
        for ms in states:
            r = compute_family_weights(ms)
            assert sum(r.weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_floor_enforced_after_any_context(self):
        cfg = load_scenario_weights_config()
        floor = cfg["floor"]
        # short_dated has EARLY_TARGET -0.04; trade=Def + target>1.25 to bypass new 5.
        r = compute_family_weights(
            _ms(T=0.05, target_z=2.0),
            trade_management=_BYPASS_TM,
        )
        for fam, w in r.weights.items():
            assert w >= floor - 1e-12, f"{fam}={w} below floor {floor}"

    def test_every_family_present(self):
        r = compute_family_weights(_ms(target_z=2.0, carry_regime=2, with_carry=True))
        assert set(r.weights) == set(FAMILIES)


# ---------------------------------------------------------------------------
# New preference-aware contexts (top 5)
# ---------------------------------------------------------------------------

class TestNewContexts:
    def test_classic_carry_default_prefs(self):
        r = compute_family_weights(_ms(carry_regime=2))
        assert len(r.fired) == 1
        assert r.fired[0].id == "classic_carry"

    def test_classic_carry_hold_up_path(self):
        r = compute_family_weights(
            _ms(carry_regime=1),
            primary_objective="Hold up if the path is slow/noisy",
        )
        assert r.fired[0].id == "classic_carry"

    def test_classic_carry_defendable(self):
        r = compute_family_weights(
            _ms(carry_regime=1),
            trade_management="Need defendable mark-to-market",
        )
        assert r.fired[0].id == "classic_carry"

    def test_cheap_carry_keep_cost_low(self):
        r = compute_family_weights(
            _ms(carry_regime=2),
            primary_objective="Keep cost low",
        )
        assert r.fired[0].id == "cheap_carry"

    def test_conservative_carry_keep_risk_clean(self):
        r = compute_family_weights(
            _ms(carry_regime=2),
            primary_objective="Keep risk clean",
        )
        assert r.fired[0].id == "conservative_carry"

    def test_conservative_carry_overrides_trade_management(self):
        """Conservative has no trade_management condition — fires for any value."""
        for tm in ("Standard hold", "May monetise early", "Need defendable mark-to-market"):
            r = compute_family_weights(
                _ms(carry_regime=2),
                primary_objective="Keep risk clean",
                trade_management=tm,
            )
            assert r.fired[0].id == "conservative_carry"

    def test_delta_carry_may_monetise(self):
        r = compute_family_weights(
            _ms(carry_regime=2),
            trade_management="May monetise early",
        )
        # primary defaults to Balanced, which is in {Bal, Cost, Hold} → delta fires
        assert r.fired[0].id == "delta_carry"

    def test_big_move_default(self):
        # target>1.25, default prefs (Bal, Standard hold) → big_move (target>1.25 fails new carry contexts)
        r = compute_family_weights(_ms(target_z=1.5))
        assert r.fired[0].id == "big_move"

    def test_big_move_against_carry(self):
        r = compute_family_weights(_ms(target_z=2.0, with_carry=False))
        assert r.fired[0].id == "big_move"

    def test_big_move_carry_zero(self):
        # carry=0 bypasses the 4 carry-pref contexts; target>1.25 fires big_move.
        r = compute_family_weights(_ms(carry_regime=0, target_z=2.0))
        assert r.fired[0].id == "big_move"


# ---------------------------------------------------------------------------
# Old market-state-only contexts as fallbacks
# ---------------------------------------------------------------------------

class TestOldContextFallback:
    def test_short_dated_fires_when_new5_bypassed(self):
        # T<0.083 + target>1.25 + trade=Def + with_carry=True:
        # classic/cheap/conservative/delta all need target<=1.25 (excluded);
        # big_move needs trade∈{Std,Early} (excluded). Falls to short_dated.
        r = compute_family_weights(
            _ms(T=0.05, target_z=2.0, carry_regime=2),
            trade_management=_BYPASS_TM,
        )
        assert r.fired[0].id == "short_dated"

    def test_counter_carry_fires_for_against_carry(self):
        # with_carry=False bypasses the 4 carry-pref contexts.
        r = compute_family_weights(_ms(carry_regime=2, with_carry=False))
        assert r.fired[0].id == "counter_carry"

    def test_vol_dominated_fires_for_carry_zero_high_vol(self):
        r = compute_family_weights(_ms(carry_regime=0, vol=0.25))
        assert r.fired[0].id == "vol_dominated"

    def test_directional_low_carry_fires_for_carry_zero(self):
        r = compute_family_weights(_ms(carry_regime=0, vol=0.10))
        assert r.fired[0].id == "directional_low_carry"

    def test_long_dated_fires_when_new5_bypassed(self):
        # T>0.5, with_carry=False, target<=1.25 (no big_move) → long_dated
        r = compute_family_weights(_ms(T=0.75, carry_regime=1, with_carry=False))
        assert r.fired[0].id == "long_dated"

    def test_speculative_far_fires_when_new5_bypassed(self):
        # target>1.5, with_carry=False, trade=Def (bypass big_move):
        # all new5 excluded → speculative_far
        r = compute_family_weights(
            _ms(carry_regime=1, target_z=2.0, with_carry=False),
            trade_management=_BYPASS_TM,
        )
        assert r.fired[0].id == "speculative_far"

    def test_speculative_near_fires_when_new5_bypassed(self):
        # target<0.5, with_carry=False, target<=1.25 (no big_move) → speculative_near
        r = compute_family_weights(_ms(carry_regime=1, target_z=0.2, with_carry=False))
        assert r.fired[0].id == "speculative_near"

    def test_high_vol_fires_when_new5_bypassed(self):
        # vol>0.20, carry=1 (so not vol_dominated), with_carry=False, target<=1.25
        r = compute_family_weights(
            _ms(carry_regime=1, target_z=0.7, vol=0.25, with_carry=False),
        )
        assert r.fired[0].id == "high_vol"


# ---------------------------------------------------------------------------
# `in` operator behaviour
# ---------------------------------------------------------------------------

class TestInOperator:
    def test_classic_fires_for_carry_1_or_2(self):
        # `carry_regime in [1, 2]` — both should fire classic
        for cr in (1, 2):
            r = compute_family_weights(_ms(carry_regime=cr))
            assert r.fired[0].id == "classic_carry"

    def test_classic_does_not_fire_for_carry_0(self):
        r = compute_family_weights(_ms(carry_regime=0))
        # carry=0 doesn't match any of the 4 carry-pref contexts
        # nor big_move (target=0.7 <= 1.25). Falls to directional_low_carry.
        assert r.fired[0].id == "directional_low_carry"

    def test_classic_in_primary_objective_match_either(self):
        for po in ("Balanced", "Hold up if the path is slow/noisy"):
            r = compute_family_weights(_ms(), primary_objective=po)
            assert r.fired[0].id == "classic_carry"


# ---------------------------------------------------------------------------
# None handling
# ---------------------------------------------------------------------------

class TestNoneHandling:
    def test_target_z_none_skips_all_target_dependent_contexts(self):
        # No target → carry-pref contexts fail (target_z_abs <= 1.25 is False
        # when target_z is None). big_move also fails (target>1.25 needs target).
        # Falls to old contexts. carry=1, vol=0.10 → nothing in old list either.
        r = compute_family_weights(_ms(target_z=None, carry_regime=1))
        assert r.fired == []
        for fam in FAMILIES:
            assert r.weights[fam] == pytest.approx(1.0 / len(FAMILIES))

    def test_atmfsratio_none_does_not_crash(self):
        r = compute_family_weights(_ms(atmfsratio=None))
        assert sum(r.weights.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# FiredContext transparency
# ---------------------------------------------------------------------------

class TestFiredContext:
    def test_fired_is_FiredContext_type(self):
        r = compute_family_weights(_ms(carry_regime=2))
        assert len(r.fired) == 1
        assert isinstance(r.fired[0], FiredContext)

    def test_fired_exposes_adjustments_dict(self):
        # New contexts have empty adjustments by default — assert the field is a dict.
        r = compute_family_weights(_ms(carry_regime=2))
        assert isinstance(r.fired[0].adjustments, dict)

    def test_fired_carries_comment(self):
        r = compute_family_weights(_ms(carry_regime=2))
        assert r.fired[0].comment != ""


# ---------------------------------------------------------------------------
# Config sanity
# ---------------------------------------------------------------------------

class TestConfig:
    def test_all_contexts_target_known_families(self):
        cfg = load_scenario_weights_config()
        for ctx in cfg["contexts"]:
            for fam in ctx["adjustments"]:
                assert fam in FAMILIES, f"Context '{ctx['id']}' targets unknown family '{fam}'"

    def test_conditions_use_supported_fields_and_ops(self):
        from knowledge_engine.scenario_weighter import _FIELD_GETTERS, _OPS
        cfg = load_scenario_weights_config()
        for ctx in cfg["contexts"]:
            for cond in ctx.get("when", []):
                assert cond["field"] in _FIELD_GETTERS, f"Unknown field '{cond['field']}'"
                assert cond["op"]    in _OPS,           f"Unknown op '{cond['op']}'"

    def test_baseline_value_is_sane(self):
        cfg = load_scenario_weights_config()
        assert cfg["baseline"] * len(FAMILIES) <= 1.0 + 1e-9
