"""Tests for knowledge_engine.scenario_weighter."""

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
    """Build a minimal MarketState for weighter tests.

    Defaults chosen so no context fires:
      carry_regime=1  → no carry-regime contexts (they require 0 or 2)
      target_z=0.7    → between 0.5 and 1.5, so neither speculative_far
                         nor speculative_near fires
      T=0.25          → between 1m and 6m, so neither short/long tenor fires
      vol=0.10        → below 0.20 threshold, so high_vol doesn't fire
    """
    return MarketState(
        spot=1.0,
        fwd=1.0,
        vol=vol,
        T=T,
        r_d=0.02,
        r_f=0.02,
        c=0.0,
        carry_regime=carry_regime,
        target_z=target_z,
        atmfsratio=atmfsratio,
        put_call=None,
        with_carry=with_carry,
    )


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------

class TestStructuralInvariants:
    def test_no_context_returns_uniform_weights(self):
        """When no context fires the weights should be exactly uniform."""
        result = compute_family_weights(_ms())
        assert isinstance(result, WeighterResult)
        for fam in FAMILIES:
            assert result.weights[fam] == pytest.approx(1.0 / len(FAMILIES))
        assert result.fired == []

    def test_at_most_one_context_fires(self):
        """First-match selection: fired list has at most one entry."""
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
        # short_dated has EARLY_TARGET -0.04; verify floor is still honoured.
        r = compute_family_weights(_ms(T=0.05))
        for fam, w in r.weights.items():
            assert w >= floor - 1e-12, f"{fam}={w} below floor {floor}"

    def test_every_family_present(self):
        r = compute_family_weights(_ms(target_z=2.0, carry_regime=2, with_carry=True))
        assert set(r.weights) == set(FAMILIES)


# ---------------------------------------------------------------------------
# Context priority — does the right (first-match) context fire?
# ---------------------------------------------------------------------------

class TestContextPriority:
    def test_short_dated_fires_before_carry_contexts(self):
        """short_dated is first in the list; it should beat carry contexts."""
        r = compute_family_weights(_ms(T=0.05, carry_regime=2, with_carry=True, target_z=2.0))
        assert len(r.fired) == 1
        assert r.fired[0].id == "short_dated"

    def test_counter_carry_fires_for_high_carry_counter(self):
        r = compute_family_weights(_ms(carry_regime=2, with_carry=False))
        assert len(r.fired) == 1
        assert r.fired[0].id == "counter_carry"

    def test_carry_momentum_extended_fires_for_high_carry_far_target(self):
        r = compute_family_weights(_ms(carry_regime=2, with_carry=True, target_z=2.0))
        assert len(r.fired) == 1
        assert r.fired[0].id == "carry_momentum_extended"

    def test_carry_capture_fires_for_high_carry_near_target(self):
        r = compute_family_weights(_ms(carry_regime=2, with_carry=True, target_z=0.2))
        assert len(r.fired) == 1
        assert r.fired[0].id == "carry_capture"

    def test_directional_with_carry_fires_for_moderate_target(self):
        """carry_regime=2, with-carry, target between 0.5 and 1.5 → directional_with_carry."""
        r = compute_family_weights(_ms(carry_regime=2, with_carry=True, target_z=0.9))
        assert len(r.fired) == 1
        assert r.fired[0].id == "directional_with_carry"

    def test_vol_dominated_fires_before_directional_low_carry(self):
        r = compute_family_weights(_ms(carry_regime=0, vol=0.25))
        assert len(r.fired) == 1
        assert r.fired[0].id == "vol_dominated"

    def test_directional_low_carry_fires_for_noisy_low_vol(self):
        r = compute_family_weights(_ms(carry_regime=0, vol=0.10))
        assert len(r.fired) == 1
        assert r.fired[0].id == "directional_low_carry"

    def test_long_dated_fires_for_long_tenor_moderate_carry(self):
        r = compute_family_weights(_ms(T=0.75, carry_regime=1))
        assert len(r.fired) == 1
        assert r.fired[0].id == "long_dated"

    def test_speculative_far_fires_for_far_target_mid_carry(self):
        r = compute_family_weights(_ms(carry_regime=1, target_z=2.0))
        assert len(r.fired) == 1
        assert r.fired[0].id == "speculative_far"

    def test_speculative_near_fires_for_near_target_mid_carry(self):
        r = compute_family_weights(_ms(carry_regime=1, target_z=0.2))
        assert len(r.fired) == 1
        assert r.fired[0].id == "speculative_near"

    def test_high_vol_fires_for_elevated_vol_mid_carry(self):
        # No other context should fire: carry_regime=1, target_z=0.7 (neutral), T=0.25
        r = compute_family_weights(_ms(carry_regime=1, target_z=0.7, vol=0.25))
        assert len(r.fired) == 1
        assert r.fired[0].id == "high_vol"

    def test_no_context_fires_for_neutral_state(self):
        """carry=1, target between 0.5–1.5, mid tenor, low vol → nothing fires."""
        r = compute_family_weights(_ms())
        assert r.fired == []


# ---------------------------------------------------------------------------
# Weight direction — does each context bend families the expected way?
# ---------------------------------------------------------------------------

class TestWeightDirections:
    def test_short_dated_lifts_expiry_drops_early(self):
        baseline = compute_family_weights(_ms())
        r = compute_family_weights(_ms(T=0.05))
        assert r.weights["EXPIRY"]       > baseline.weights["EXPIRY"]
        assert r.weights["EARLY_TARGET"] < baseline.weights["EARLY_TARGET"]

    def test_counter_carry_lifts_wrongway_and_nomove(self):
        baseline = compute_family_weights(_ms())
        r = compute_family_weights(_ms(carry_regime=2, with_carry=False))
        assert r.weights["WRONG_WAY"] > baseline.weights["WRONG_WAY"]
        assert r.weights["NO_MOVE"]   > baseline.weights["NO_MOVE"]
        assert r.weights["CORRECT_PATH"] < baseline.weights["CORRECT_PATH"]

    def test_carry_momentum_extended_lifts_correct_and_overshoot(self):
        baseline = compute_family_weights(_ms())
        r = compute_family_weights(_ms(carry_regime=2, with_carry=True, target_z=2.0))
        assert r.weights["CORRECT_PATH"] > baseline.weights["CORRECT_PATH"]
        assert r.weights["OVERSHOOT"]    > baseline.weights["OVERSHOOT"]
        assert r.weights["WRONG_WAY"]    < baseline.weights["WRONG_WAY"]

    def test_carry_capture_lifts_correct_and_early(self):
        baseline = compute_family_weights(_ms())
        r = compute_family_weights(_ms(carry_regime=2, with_carry=True, target_z=0.2))
        assert r.weights["CORRECT_PATH"] > baseline.weights["CORRECT_PATH"]
        assert r.weights["EARLY_TARGET"] > baseline.weights["EARLY_TARGET"]

    def test_vol_dominated_lifts_vol_sensitivity(self):
        baseline = compute_family_weights(_ms())
        r = compute_family_weights(_ms(carry_regime=0, vol=0.25))
        assert r.weights["VOL_SENSITIVITY"] > baseline.weights["VOL_SENSITIVITY"]

    def test_speculative_far_lifts_overshoot_drops_early(self):
        baseline = compute_family_weights(_ms())
        r = compute_family_weights(_ms(carry_regime=1, target_z=2.0))
        assert r.weights["OVERSHOOT"]    > baseline.weights["OVERSHOOT"]
        assert r.weights["EARLY_TARGET"] < baseline.weights["EARLY_TARGET"]

    def test_speculative_near_lifts_early_drops_overshoot(self):
        baseline = compute_family_weights(_ms())
        r = compute_family_weights(_ms(carry_regime=1, target_z=0.2))
        assert r.weights["EARLY_TARGET"] > baseline.weights["EARLY_TARGET"]
        assert r.weights["OVERSHOOT"]    < baseline.weights["OVERSHOOT"]


# ---------------------------------------------------------------------------
# None-handling
# ---------------------------------------------------------------------------

class TestNoneHandling:
    def test_target_z_none_skips_target_dependent_contexts(self):
        """No target → speculative_far, speculative_near, carry contexts with
        target conditions all skip. With carry_regime=1 no context fires."""
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
        r = compute_family_weights(_ms(carry_regime=2, with_carry=True, target_z=2.0))
        assert len(r.fired) == 1
        assert isinstance(r.fired[0], FiredContext)

    def test_fired_exposes_adjustments_dict(self):
        r = compute_family_weights(_ms(carry_regime=2, with_carry=True, target_z=2.0))
        ctx = r.fired[0]
        assert "CORRECT_PATH" in ctx.adjustments
        assert "WRONG_WAY"    in ctx.adjustments

    def test_fired_carries_comment(self):
        r = compute_family_weights(_ms(carry_regime=2, with_carry=True, target_z=2.0))
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
