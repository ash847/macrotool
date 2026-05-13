import pandas as pd

from interface.context_rules import (
    _compact_multipliers,
    _conditions_to_df,
    _df_to_conditions,
    _merge_contexts_with_latest_multipliers,
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
        conds, errors = _df_to_conditions(df, ["carry_regime", "with_carry"])
        assert errors == []
        assert conds == when

    def test_blank_rows_skipped(self):
        df = pd.DataFrame([{"field": "", "op": "", "value": ""}])
        conds, errors = _df_to_conditions(df, ["carry_regime"])
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

    def test_all_matches_mode_returns_all_matching_ids(self):
        contexts = [
            {"id": "cost", "when": [{"field": "primary_objective", "op": "==", "value": "Keep cost low"}], "multipliers": {}},
            {"id": "mtm", "when": [{"field": "trade_management", "op": "==", "value": "Need defendable mark-to-market"}], "multipliers": {}},
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
            "prefs": {
                "primary_objective": "Keep cost low",
                "trade_management": "Need defendable mark-to-market",
            },
        }, all_matches=True)
        assert fired == ["cost", "mtm"]


class TestValidateContexts:
    def test_duplicate_id(self):
        errors = _validate_contexts([
            {"id": "a", "when": [], "multipliers": {}},
            {"id": "a", "when": [], "multipliers": {}},
        ])
        assert any("Duplicate" in e for e in errors)


class TestScenarioWeightPersistenceHelpers:
    def test_compact_multipliers_drops_baseline_values(self):
        compact = _compact_multipliers(
            {"Expiry|F": 1.0, "Expiry|K": 1.4, "50%T|K": 1.0},
            baseline=1.0,
        )
        assert compact == {"Expiry|K": 1.4}

    def test_merge_contexts_preserves_multipliers_across_rename(self):
        contexts = [{
            "_uid": "1",
            "_original_id": "classic_carry",
            "id": "classic_carry_v2",
            "comment": "renamed",
            "when": [],
            "multipliers": {},
        }]
        latest_cfg = {
            "base_weightings": [{
                "id": "classic_carry",
                "multipliers": {"Expiry|K": 1.6},
            }]
        }
        merged = _merge_contexts_with_latest_multipliers(contexts, latest_cfg, config_key="base_weightings")
        assert merged[0]["id"] == "classic_carry_v2"
        assert merged[0]["multipliers"] == {"Expiry|K": 1.6}
