from analytics.market_state import MarketState
from analytics.scenario_generator import cell_id, valid_grid_cells_for_tenor
from knowledge_engine.scenario_weighter import (
    FiredWeighting,
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
    return MarketState(
        spot=1.0, fwd=1.0, vol=vol, T=T, r_d=0.02, r_f=0.02,
        c=0.0, carry_regime=carry_regime, target_z=target_z,
        atmfsratio=atmfsratio, put_call=None, with_carry=with_carry,
    )


class TestStructuralInvariants:
    def test_returns_grid_weights(self):
        result = compute_family_weights(_ms())
        assert isinstance(result, WeighterResult)
        assert set(result.weights) == {cell_id(r, c) for r, c in valid_grid_cells_for_tenor(0.25)}
        assert abs(sum(result.weights.values()) - 1.0) < 1e-9

    def test_returns_raw_multipliers(self):
        result = compute_family_weights(_ms())
        assert set(result.multipliers) == set(result.weights)
        assert all(v > 0 for v in result.multipliers.values())

    def test_at_most_one_base_context_fires(self):
        states = [
            _ms(target_z=2.0, carry_regime=2, with_carry=True),
            _ms(target_z=0.2, carry_regime=0, with_carry=False, T=0.75, vol=0.25),
            _ms(target_z=None, carry_regime=2, with_carry=False),
        ]
        for ms in states:
            r = compute_family_weights(ms)
            assert r.base_fired is None or isinstance(r.base_fired, FiredWeighting)
            assert len([ctx for ctx in r.fired if ctx == r.base_fired]) <= 1


class TestSelection:
    def test_default_state_fires_classic(self):
        r = compute_family_weights(_ms())
        assert r.base_fired is not None
        assert r.base_fired.id == "classic_carry"
        assert r.overlay_fired == []

    def test_keep_cost_low_applies_overlay(self):
        r = compute_family_weights(_ms(carry_regime=2), primary_objective="Keep cost low")
        assert r.base_fired is not None
        assert r.base_fired.id == "classic_carry"
        assert [ctx.id for ctx in r.overlay_fired] == ["objective_keep_cost_low"]

    def test_big_move_still_fires(self):
        r = compute_family_weights(_ms(target_z=1.5))
        assert r.base_fired is not None
        assert r.base_fired.id == "big_move"

    def test_target_none_falls_back_to_no_context(self):
        r = compute_family_weights(_ms(target_z=None, carry_regime=1))
        assert r.base_fired is None
        assert r.fired == []

    def test_trade_management_overlay_stacks(self):
        r = compute_family_weights(_ms(), trade_management="May monetise early")
        assert r.base_fired is not None
        assert r.base_fired.id == "classic_carry"
        assert [ctx.id for ctx in r.overlay_fired] == ["management_monetise_early"]


class TestTransparencyAndConfig:
    def test_fired_context_type(self):
        r = compute_family_weights(_ms())
        assert isinstance(r.base_fired, FiredWeighting)

    def test_config_uses_multipliers_shape(self):
        cfg = load_scenario_weights_config()
        assert "baseline" in cfg
        assert "min_multiplier" in cfg
        assert "base_weightings" in cfg
        assert "preference_overlays" in cfg
        for ctx in cfg["base_weightings"] + cfg["preference_overlays"]:
            assert "multipliers" in ctx
