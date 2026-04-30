"""
Tests for the pure-function helpers in interface/context_rules.py.

Covers:
  - _parse_condition_value
  - _conditions_to_df / _df_to_conditions round-trip
  - _check_shadowing
  - _simulate_context_fire
  - _validate_contexts
"""

from __future__ import annotations

import pandas as pd
import pytest

from interface.context_rules import (
    _check_shadowing,
    _conditions_to_df,
    _df_to_conditions,
    _parse_condition_value,
    _simulate_context_fire,
    _validate_contexts,
)


# ---------------------------------------------------------------------------
# _parse_condition_value
# ---------------------------------------------------------------------------

class TestParseConditionValue:
    def test_true(self):
        assert _parse_condition_value("true") is True

    def test_false(self):
        assert _parse_condition_value("false") is False

    def test_true_case_insensitive(self):
        assert _parse_condition_value("True") is True
        assert _parse_condition_value("TRUE") is True

    def test_integer(self):
        v = _parse_condition_value("2")
        assert v == 2
        assert isinstance(v, int)

    def test_integer_zero(self):
        v = _parse_condition_value("0")
        assert v == 0
        assert isinstance(v, int)

    def test_float(self):
        v = _parse_condition_value("0.5")
        assert abs(v - 0.5) < 1e-9
        assert isinstance(v, float)

    def test_float_decimal(self):
        v = _parse_condition_value("0.083")
        assert abs(v - 0.083) < 1e-9

    def test_unparseable_returns_string(self):
        # Strings that aren't bool/int/float are accepted as-is — used for
        # enum fields like primary_objective ("Balanced", "Keep cost low" …).
        v = _parse_condition_value("Balanced")
        assert v == "Balanced"
        assert isinstance(v, str)

    def test_whitespace_stripped(self):
        assert _parse_condition_value("  2  ") == 2


# ---------------------------------------------------------------------------
# _conditions_to_df
# ---------------------------------------------------------------------------

class TestConditionsToDf:
    def test_empty(self):
        df = _conditions_to_df([])
        assert list(df.columns) == ["field", "op", "value"]
        assert len(df) == 0

    def test_float_value(self):
        when = [{"field": "T", "op": "<", "value": 0.083}]
        df = _conditions_to_df(when)
        assert df.iloc[0]["field"] == "T"
        assert df.iloc[0]["op"] == "<"
        assert df.iloc[0]["value"] == "0.083"

    def test_bool_true(self):
        when = [{"field": "with_carry", "op": "==", "value": True}]
        df = _conditions_to_df(when)
        assert df.iloc[0]["value"] == "true"

    def test_bool_false(self):
        when = [{"field": "with_carry", "op": "==", "value": False}]
        df = _conditions_to_df(when)
        assert df.iloc[0]["value"] == "false"

    def test_int_value(self):
        when = [{"field": "carry_regime", "op": "==", "value": 2}]
        df = _conditions_to_df(when)
        assert df.iloc[0]["value"] == "2"

    def test_multiple(self):
        when = [
            {"field": "carry_regime", "op": "==", "value": 2},
            {"field": "with_carry",   "op": "==", "value": True},
        ]
        df = _conditions_to_df(when)
        assert len(df) == 2


# ---------------------------------------------------------------------------
# _df_to_conditions (round-trip and error handling)
# ---------------------------------------------------------------------------

class TestDfToConditions:
    def test_empty_df(self):
        df = pd.DataFrame(columns=["field", "op", "value"])
        conds, errors = _df_to_conditions(df)
        assert conds == []
        assert errors == []

    def test_round_trip_float(self):
        when = [{"field": "T", "op": "<", "value": 0.083}]
        df = _conditions_to_df(when)
        conds, errors = _df_to_conditions(df)
        assert errors == []
        assert len(conds) == 1
        assert conds[0]["field"] == "T"
        assert conds[0]["op"] == "<"
        assert abs(conds[0]["value"] - 0.083) < 1e-9

    def test_round_trip_bool_true(self):
        when = [{"field": "with_carry", "op": "==", "value": True}]
        df = _conditions_to_df(when)
        conds, errors = _df_to_conditions(df)
        assert errors == []
        assert conds[0]["value"] is True

    def test_round_trip_bool_false(self):
        when = [{"field": "with_carry", "op": "==", "value": False}]
        df = _conditions_to_df(when)
        conds, errors = _df_to_conditions(df)
        assert errors == []
        assert conds[0]["value"] is False

    def test_round_trip_int(self):
        when = [{"field": "carry_regime", "op": "==", "value": 2}]
        df = _conditions_to_df(when)
        conds, errors = _df_to_conditions(df)
        assert errors == []
        assert conds[0]["value"] == 2
        assert isinstance(conds[0]["value"], int)

    def test_unknown_field_error(self):
        df = pd.DataFrame([{"field": "nonexistent", "op": "==", "value": "1"}])
        conds, errors = _df_to_conditions(df)
        assert conds == []
        assert any("nonexistent" in e for e in errors)

    def test_unknown_op_error(self):
        df = pd.DataFrame([{"field": "carry_regime", "op": "??", "value": "2"}])
        conds, errors = _df_to_conditions(df)
        assert conds == []
        assert any("??" in e for e in errors)

    def test_string_value_accepted_for_enum_fields(self):
        # primary_objective accepts string values like "Balanced".
        df = pd.DataFrame([{
            "field": "primary_objective", "op": "==", "value": "Balanced",
        }])
        conds, errors = _df_to_conditions(df)
        assert errors == []
        assert conds[0]["value"] == "Balanced"

    def test_in_op_parses_comma_separated_list(self):
        df = pd.DataFrame([{
            "field": "carry_regime", "op": "in", "value": "1, 2",
        }])
        conds, errors = _df_to_conditions(df)
        assert errors == []
        assert conds[0]["value"] == [1, 2]

    def test_in_op_parses_string_list(self):
        df = pd.DataFrame([{
            "field": "primary_objective", "op": "in",
            "value": "Balanced, Keep cost low",
        }])
        conds, errors = _df_to_conditions(df)
        assert errors == []
        assert conds[0]["value"] == ["Balanced", "Keep cost low"]

    def test_blank_row_skipped(self):
        df = pd.DataFrame([{"field": "", "op": "", "value": ""}])
        conds, errors = _df_to_conditions(df)
        assert conds == []
        assert errors == []

    def test_multiple_conditions(self):
        when = [
            {"field": "carry_regime",  "op": "==", "value": 2},
            {"field": "with_carry",    "op": "==", "value": True},
            {"field": "target_z_abs",  "op": ">",  "value": 1.5},
        ]
        df = _conditions_to_df(when)
        conds, errors = _df_to_conditions(df)
        assert errors == []
        assert len(conds) == 3


# ---------------------------------------------------------------------------
# _check_shadowing
# ---------------------------------------------------------------------------

class TestCheckShadowing:
    def _ctx(self, ctx_id: str, when: list[dict]) -> dict:
        return {"id": ctx_id, "when": when, "adjustments": {}}

    def test_no_shadowing(self):
        contexts = [
            self._ctx("a", [{"field": "carry_regime", "op": "==", "value": 2}]),
            self._ctx("b", [{"field": "T", "op": "<", "value": 0.1}]),
        ]
        assert _check_shadowing(contexts) == []

    def test_no_conditions_always_fires(self):
        contexts = [
            self._ctx("a", []),   # no conditions → always fires
            self._ctx("b", [{"field": "carry_regime", "op": "==", "value": 2}]),
        ]
        warnings = _check_shadowing(contexts)
        assert len(warnings) == 1
        assert "b" in warnings[0]
        assert "a" in warnings[0]

    def test_subset_shadowing(self):
        # a has {c1}, b has {c1, c2} → a shadows b
        c1 = {"field": "carry_regime", "op": "==", "value": 2}
        c2 = {"field": "with_carry",   "op": "==", "value": True}
        contexts = [
            self._ctx("a", [c1]),
            self._ctx("b", [c1, c2]),
        ]
        warnings = _check_shadowing(contexts)
        assert len(warnings) == 1
        assert "b" in warnings[0]

    def test_equal_conditions_not_shadowed(self):
        # Same condition sets — not a strict subset, no warning
        c1 = {"field": "carry_regime", "op": "==", "value": 2}
        contexts = [
            self._ctx("a", [c1]),
            self._ctx("b", [c1]),
        ]
        assert _check_shadowing(contexts) == []

    def test_lower_priority_more_specific_not_shadowed(self):
        # b has MORE conditions than a but b appears first — check order
        c1 = {"field": "carry_regime", "op": "==", "value": 2}
        c2 = {"field": "with_carry",   "op": "==", "value": True}
        contexts = [
            self._ctx("b", [c1, c2]),  # more specific, higher priority → fine
            self._ctx("a", [c1]),      # less specific, lower priority → can still fire
        ]
        # a is NOT shadowed by b (b is stricter; when only c1 is true, b won't fire but a will)
        assert _check_shadowing(contexts) == []

    def test_single_context_no_warning(self):
        c = [{"field": "T", "op": "<", "value": 0.083}]
        assert _check_shadowing([self._ctx("a", c)]) == []

    def test_empty_list(self):
        assert _check_shadowing([]) == []


# ---------------------------------------------------------------------------
# _simulate_context_fire
# ---------------------------------------------------------------------------

class TestSimulateContextFire:
    def _ctx(self, ctx_id: str, when: list[dict]) -> dict:
        return {"id": ctx_id, "when": when, "adjustments": {}}

    def _fv(self, **kwargs) -> dict:
        base = {
            "carry_regime": 1, "with_carry": True, "T": 0.25,
            "vol": 0.12, "target_z_abs": 1.0, "atmfsratio": 1.0,
        }
        base.update(kwargs)
        return base

    def test_no_contexts_returns_none(self):
        assert _simulate_context_fire([], self._fv()) is None

    def test_no_match_returns_none(self):
        contexts = [self._ctx("a", [{"field": "carry_regime", "op": "==", "value": 2}])]
        fv = self._fv(carry_regime=0)
        assert _simulate_context_fire(contexts, fv) is None

    def test_single_match(self):
        contexts = [self._ctx("a", [{"field": "carry_regime", "op": "==", "value": 2}])]
        fv = self._fv(carry_regime=2)
        assert _simulate_context_fire(contexts, fv) == "a"

    def test_first_match_wins(self):
        c_regime2 = [{"field": "carry_regime", "op": "==", "value": 2}]
        c_high_vol = [{"field": "vol", "op": ">", "value": 0.20}]
        contexts = [
            self._ctx("first", c_regime2),
            self._ctx("second", c_high_vol),
        ]
        # Both match → first wins
        fv = self._fv(carry_regime=2, vol=0.25)
        assert _simulate_context_fire(contexts, fv) == "first"

    def test_second_fires_when_first_does_not(self):
        c_regime2 = [{"field": "carry_regime", "op": "==", "value": 2}]
        c_high_vol = [{"field": "vol", "op": ">", "value": 0.20}]
        contexts = [
            self._ctx("first", c_regime2),
            self._ctx("second", c_high_vol),
        ]
        fv = self._fv(carry_regime=0, vol=0.25)
        assert _simulate_context_fire(contexts, fv) == "second"

    def test_no_conditions_always_fires(self):
        contexts = [self._ctx("always", [])]
        assert _simulate_context_fire(contexts, self._fv()) == "always"

    def test_bool_condition_true(self):
        cond = [{"field": "with_carry", "op": "==", "value": True}]
        contexts = [self._ctx("wc", cond)]
        assert _simulate_context_fire(contexts, self._fv(with_carry=True)) == "wc"
        assert _simulate_context_fire(contexts, self._fv(with_carry=False)) is None

    def test_bool_condition_false(self):
        cond = [{"field": "with_carry", "op": "==", "value": False}]
        contexts = [self._ctx("cc", cond)]
        assert _simulate_context_fire(contexts, self._fv(with_carry=False)) == "cc"
        assert _simulate_context_fire(contexts, self._fv(with_carry=True)) is None

    def test_short_dated(self):
        cond = [{"field": "T", "op": "<", "value": 0.083}]
        contexts = [self._ctx("short_dated", cond)]
        assert _simulate_context_fire(contexts, self._fv(T=0.05)) == "short_dated"
        assert _simulate_context_fire(contexts, self._fv(T=0.25)) is None

    def test_multi_condition_all_must_match(self):
        conds = [
            {"field": "carry_regime", "op": "==", "value": 2},
            {"field": "with_carry",   "op": "==", "value": True},
            {"field": "target_z_abs", "op": ">",  "value": 1.5},
        ]
        contexts = [self._ctx("x", conds)]
        # All match
        assert _simulate_context_fire(
            contexts, self._fv(carry_regime=2, with_carry=True, target_z_abs=2.0)
        ) == "x"
        # One fails
        assert _simulate_context_fire(
            contexts, self._fv(carry_regime=2, with_carry=True, target_z_abs=1.0)
        ) is None

    def test_gt_op(self):
        cond = [{"field": "target_z_abs", "op": ">", "value": 1.5}]
        contexts = [self._ctx("far", cond)]
        assert _simulate_context_fire(contexts, self._fv(target_z_abs=2.0)) == "far"
        assert _simulate_context_fire(contexts, self._fv(target_z_abs=1.5)) is None
        assert _simulate_context_fire(contexts, self._fv(target_z_abs=1.0)) is None

    def test_gte_op(self):
        cond = [{"field": "target_z_abs", "op": ">=", "value": 1.5}]
        contexts = [self._ctx("far", cond)]
        assert _simulate_context_fire(contexts, self._fv(target_z_abs=1.5)) == "far"

    def test_ne_op(self):
        cond = [{"field": "carry_regime", "op": "!=", "value": 0}]
        contexts = [self._ctx("some_carry", cond)]
        assert _simulate_context_fire(contexts, self._fv(carry_regime=1)) == "some_carry"
        assert _simulate_context_fire(contexts, self._fv(carry_regime=0)) is None


# ---------------------------------------------------------------------------
# _validate_contexts
# ---------------------------------------------------------------------------

class TestValidateContexts:
    def _ctx(self, ctx_id: str) -> dict:
        return {"id": ctx_id, "when": [], "adjustments": {}}

    def test_valid(self):
        contexts = [self._ctx("a"), self._ctx("b")]
        assert _validate_contexts(contexts) == []

    def test_empty_id(self):
        contexts = [self._ctx(""), self._ctx("b")]
        errors = _validate_contexts(contexts)
        assert any("empty" in e.lower() for e in errors)

    def test_duplicate_id(self):
        contexts = [self._ctx("a"), self._ctx("a")]
        errors = _validate_contexts(contexts)
        assert any("duplicate" in e.lower() for e in errors)

    def test_empty_list(self):
        assert _validate_contexts([]) == []

    def test_single_valid(self):
        assert _validate_contexts([self._ctx("x")]) == []
