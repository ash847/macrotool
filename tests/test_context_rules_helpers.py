import pandas as pd

from interface.context_rules import (
    _conditions_to_df,
    _df_to_conditions,
    _parse_condition_value,
    _simulate_context_fire,
    _validate_contexts,
)


class TestParseConditionValue:
    def test_bool(self):
        assert _parse_condition_value("true") is True
        assert _parse_condition_value("false") is False

    def test_numbers(self):
        assert _parse_condition_value("2") == 2
        assert _parse_condition_value("0.5") == 0.5

    def test_string_passthrough(self):
        assert _parse_condition_value("Balanced") == "Balanced"


class TestConditionRoundTrip:
    def test_conditions_to_df_and_back(self):
        when = [
            {"field": "carry_regime", "op": "in", "value": [1, 2]},
            {"field": "with_carry", "op": "==", "value": True},
        ]
        df = _conditions_to_df(when)
        conds, errors = _df_to_conditions(df)
        assert errors == []
        assert conds == when

    def test_blank_rows_skipped(self):
        df = pd.DataFrame([{"field": "", "op": "", "value": ""}])
        conds, errors = _df_to_conditions(df)
        assert conds == []
        assert errors == []


class TestSimulateContextFire:
    def test_first_match_wins(self):
        contexts = [
            {"id": "first", "when": [{"field": "carry_regime", "op": "==", "value": 1}], "multipliers": {}},
            {"id": "second", "when": [{"field": "vol", "op": ">", "value": 0.2}], "multipliers": {}},
        ]

        class _MS:
            carry_regime = 1
            with_carry = True
            T = 0.25
            vol = 0.25
            target_z = 1.0
            atmfsratio = 1.0

        fired = _simulate_context_fire(contexts, {
            "ms": _MS,
            "prefs": {"primary_objective": "Balanced", "trade_management": "Standard hold"},
        })
        assert fired == "first"


class TestValidateContexts:
    def test_duplicate_id(self):
        errors = _validate_contexts([
            {"id": "a", "when": [], "multipliers": {}},
            {"id": "a", "when": [], "multipliers": {}},
        ])
        assert any("Duplicate" in e for e in errors)
