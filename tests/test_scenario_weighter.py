"""Tests for knowledge_engine.scenario_weighter."""

from __future__ import annotations

import math

import pytest

from analytics.market_state import MarketState
from analytics.scenario_generator import FAMILIES
from knowledge_engine.scenario_weighter import (
    WeighterResult,
    compute_family_weights,
    load_scenario_weights_config,
)


def _ms(
    *,
    target_z: float | None = 0.0,
    carry_regime: int = 1,
    with_carry: bool = True,
    T: float = 0.25,
    vol: float = 0.10,
    atmfsratio: float | None = 1.5,
) -> MarketState:
    """Build a MarketState with only the fields the weighter uses, defaults
    chosen so no rule fires (carry_regime=1 has no rule, target_z=0 fails
    both <0.5 and >1.5, T=0.25 inside neither tenor band, vol below 0.20)."""
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
    def test_baseline_state_returns_uniform_weights(self):
        # carry_regime=1, target_z just inside neutral band (0.6), vol low,
        # T mid-tenor — no rule fires, so weights should be exactly uniform.
        result = compute_family_weights(_ms(target_z=0.6, carry_regime=1))
        assert isinstance(result, WeighterResult)
        for fam in FAMILIES:
            assert result.weights[fam] == pytest.approx(1.0 / len(FAMILIES))
        assert result.fired == []

    def test_weights_always_sum_to_one(self):
        states = [
            _ms(target_z=2.0, carry_regime=2, with_carry=True, T=0.05, vol=0.30),
            _ms(target_z=0.2, carry_regime=0, with_carry=False, T=0.75, vol=0.15),
            _ms(target_z=None, carry_regime=2, with_carry=False, T=0.10, vol=0.25),
        ]
        for ms in states:
            r = compute_family_weights(ms)
            assert sum(r.weights.values()) == pytest.approx(1.0, abs=1e-9)
            assert all(w >= r.floor - 1e-12 for w in r.weights.values())

    def test_every_family_present_after_normalisation(self):
        r = compute_family_weights(_ms(target_z=2.5, carry_regime=2))
        assert set(r.weights) == set(FAMILIES)

    def test_floor_prevents_negative_weights(self):
        # Construct a state where many negative rules target one family.
        # Target far AND short tenor both trim EARLY_TARGET (-0.04, -0.03).
        # 0.125 - 0.07 = 0.055 — still above floor — but we verify the
        # floor itself by reading config and checking no family is below it.
        cfg = load_scenario_weights_config()
        floor = cfg["floor"]
        r = compute_family_weights(_ms(target_z=2.5, T=0.05, carry_regime=2))
        for fam, w in r.weights.items():
            assert w >= floor - 1e-12, f"{fam}={w} below floor {floor}"


# ---------------------------------------------------------------------------
# Individual rules — does each one bend weights in the expected direction?
# ---------------------------------------------------------------------------

class TestIndividualRules:
    def test_far_target_bumps_overshoot_drops_early(self):
        baseline = compute_family_weights(_ms(target_z=0.7))
        far = compute_family_weights(_ms(target_z=2.0))
        assert far.weights["OVERSHOOT"] > baseline.weights["OVERSHOOT"]
        assert far.weights["EARLY_TARGET"] < baseline.weights["EARLY_TARGET"]

    def test_near_target_bumps_early_drops_overshoot(self):
        baseline = compute_family_weights(_ms(target_z=0.7))
        near = compute_family_weights(_ms(target_z=0.2))
        assert near.weights["EARLY_TARGET"] > baseline.weights["EARLY_TARGET"]
        assert near.weights["OVERSHOOT"] < baseline.weights["OVERSHOOT"]

    def test_high_carry_with_lifts_correct_path(self):
        baseline = compute_family_weights(_ms(carry_regime=1, with_carry=True))
        with_high = compute_family_weights(_ms(carry_regime=2, with_carry=True))
        assert with_high.weights["CORRECT_PATH"] > baseline.weights["CORRECT_PATH"]
        assert with_high.weights["WRONG_WAY"] < baseline.weights["WRONG_WAY"]

    def test_high_carry_counter_lifts_wrongway_and_nomove(self):
        baseline = compute_family_weights(_ms(carry_regime=1, with_carry=False))
        counter = compute_family_weights(_ms(carry_regime=2, with_carry=False))
        assert counter.weights["WRONG_WAY"] > baseline.weights["WRONG_WAY"]
        assert counter.weights["NO_MOVE"]   > baseline.weights["NO_MOVE"]

    def test_noisy_carry_lifts_vol_sensitivity(self):
        baseline = compute_family_weights(_ms(carry_regime=1))
        noisy = compute_family_weights(_ms(carry_regime=0))
        assert noisy.weights["VOL_SENSITIVITY"] > baseline.weights["VOL_SENSITIVITY"]
        assert noisy.weights["NO_MOVE"]         < baseline.weights["NO_MOVE"]

    def test_short_tenor_lifts_expiry(self):
        baseline = compute_family_weights(_ms(T=0.25))
        short = compute_family_weights(_ms(T=0.05))   # ~18 days
        assert short.weights["EXPIRY"]       > baseline.weights["EXPIRY"]
        assert short.weights["EARLY_TARGET"] < baseline.weights["EARLY_TARGET"]

    def test_long_tenor_lifts_early_target(self):
        baseline = compute_family_weights(_ms(T=0.25))
        long_ = compute_family_weights(_ms(T=0.75))   # 9 months
        assert long_.weights["EARLY_TARGET"] > baseline.weights["EARLY_TARGET"]

    def test_high_vol_lifts_vol_sensitivity(self):
        baseline = compute_family_weights(_ms(vol=0.10))
        hi = compute_family_weights(_ms(vol=0.25))
        assert hi.weights["VOL_SENSITIVITY"] > baseline.weights["VOL_SENSITIVITY"]


# ---------------------------------------------------------------------------
# None-handling — fields that may be None on MarketState
# ---------------------------------------------------------------------------

class TestNoneHandling:
    def test_target_z_none_skips_target_rules(self):
        # No target → no target_z rules should fire. Weights should equal a
        # baseline state (with no other rules firing) — uniform.
        r = compute_family_weights(_ms(target_z=None, carry_regime=1))
        for fam in FAMILIES:
            assert r.weights[fam] == pytest.approx(1.0 / len(FAMILIES))

    def test_atmfsratio_none_does_not_break(self):
        # No atmfsratio rules in starter set, but verify None doesn't crash.
        r = compute_family_weights(_ms(atmfsratio=None))
        assert sum(r.weights.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Fired-rules transparency
# ---------------------------------------------------------------------------

class TestFiredRules:
    def test_fired_records_match_weight_changes(self):
        r = compute_family_weights(_ms(target_z=2.0, carry_regime=2, with_carry=True, vol=0.25))
        fired_ids = {rule.id for rule in r.fired}
        # Expected rules: far_target_overshoot_up, far_target_early_down,
        # high_carry_with_correct_up, high_carry_with_wrongway_down,
        # high_vol_volsens_up.
        assert "far_target_overshoot_up"     in fired_ids
        assert "far_target_early_down"       in fired_ids
        assert "high_carry_with_correct_up"  in fired_ids
        assert "high_carry_with_wrongway_down" in fired_ids
        assert "high_vol_volsens_up"         in fired_ids

    def test_fired_carries_comment(self):
        r = compute_family_weights(_ms(target_z=2.0))
        any_with_comment = any(rule.comment for rule in r.fired)
        assert any_with_comment, "Fired rules should expose human-readable comments"


# ---------------------------------------------------------------------------
# Config sanity
# ---------------------------------------------------------------------------

class TestConfig:
    def test_config_targets_only_known_families(self):
        cfg = load_scenario_weights_config()
        for rule in cfg["rules"]:
            assert rule["family"] in FAMILIES, f"Rule {rule['id']} targets unknown family"

    def test_config_uses_supported_fields_and_ops(self):
        from knowledge_engine.scenario_weighter import _FIELD_GETTERS, _OPS
        cfg = load_scenario_weights_config()
        for rule in cfg["rules"]:
            for cond in rule.get("when", []):
                assert cond["field"] in _FIELD_GETTERS
                assert cond["op"]    in _OPS

    def test_baseline_times_families_under_one(self):
        cfg = load_scenario_weights_config()
        # Sanity: total of baselines < 1 leaves room for adjustments to do
        # something. (Not strict — just guards against accidentally setting
        # baseline = 1.0 and breaking the architecture.)
        assert cfg["baseline"] * len(FAMILIES) <= 1.0 + 1e-9
