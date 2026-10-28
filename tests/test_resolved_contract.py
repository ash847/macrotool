"""Migration guards against the retained legacy consumer paths."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from analytics.payoffs import base_ccy_payoff_for_trade_rec
from analytics.product_model import ResolvedContract, ResolvedLeg, Right
from analytics.resolved_pricing import PricingContext, terminal_payoff, value_resolved
from analytics.scenario_pricer import _value_variant
from analytics.structure_pricer import price_variants
from analytics.trade_economics import compute_trade_economics
from tests.test_product_model_parity import _make_ms, _surface_for


@pytest.mark.parametrize("family", ["vanilla", "1x1_spread", "1x1.5_spread", "1x2_spread", "1x2x1_spread", "seagull"])
@pytest.mark.parametrize("is_call", [True, False], ids=["call", "put"])
@pytest.mark.parametrize("flat", [True, False], ids=["flat", "smile"])
def test_migrated_consumers_match_legacy(family, is_call, flat, monkeypatch):
    surface = _surface_for("USDBRL", flat)
    ms, target = _make_ms("USDBRL", 90, 12, "base_higher" if is_call else "base_lower", surface)
    variants = price_variants(ms, family, target=target, is_call=is_call, smile=surface, loss_budget=100)
    assert variants

    def cannot_construct(*args, **kwargs):
        raise AssertionError("A valuation consumer must not reconstruct the trade")

    monkeypatch.setattr("analytics.product_pricer.build_structure", cannot_construct)
    monkeypatch.setattr("analytics.product_pricer._resolve_leg", cannot_construct)
    for priced in variants:
        contract = priced.resolved_contract
        assert contract is not None
        assert [leg.strike for leg in contract.legs] == priced.strikes
        legacy = replace(priced, resolved_contract=None)
        before = repr(contract)
        for spot in (ms.spot * 0.7, ms.spot, ms.spot * 1.3):
            for tau in (0.0, ms.T / 2, ms.T):
                for vol in (ms.vol * 0.8, ms.vol, ms.vol * 1.2):
                    args = (spot, vol, tau, ms.r_d, ms.r_f, ms.spot, is_call, surface)
                    assert _value_variant(family, priced, *args) == pytest.approx(
                        _value_variant(family, legacy, *args), abs=1e-12,
                    )
        grid = np.unique(np.r_[np.linspace(ms.spot * 0.5, ms.spot * 1.5, 51), priced.strikes])
        kwargs = dict(strikes=priced.strikes, barrier=None, is_call=is_call, entry_spot=ms.spot,
                      wing_ratio=priced.wing_ratio)
        new_payoff = base_ccy_payoff_for_trade_rec(family, **kwargs, resolved_contract=contract)
        old_payoff = base_ccy_payoff_for_trade_rec(family, **kwargs)
        np.testing.assert_allclose(new_payoff(grid), old_payoff(grid), atol=1e-12)
        for days in (30, 90):
            kwargs = dict(target=target, is_call=is_call, surface=surface, evaluation_days=days)
            new = compute_trade_economics(priced, family, ms, **kwargs)
            old = compute_trade_economics(legacy, family, ms, **kwargs)
            assert new.target_net_pnl_pct == pytest.approx(old.target_net_pnl_pct, abs=1e-12)
        assert repr(contract) == before


def test_contract_is_immutable_and_quantity_drives_valuation():
    # The valuation engine uses quantities, not the family name's implied ratio.
    contract = ResolvedContract("1x2_spread", "fixed", (
        ResolvedLeg(Right.CALL, 5.0, 1.0), ResolvedLeg(Right.CALL, 5.5, -1.25),
    ), 0.5)
    with pytest.raises(FrozenInstanceError):
        contract.legs[0].strike = 6
    with pytest.raises(FrozenInstanceError):
        contract.legs = ()
    assert terminal_payoff(contract, [6.0])[0] == pytest.approx((1.0 - 1.25 * 0.5) / 6)
    assert value_resolved(contract, PricingContext(6, 0, 0.2, 0.05, 0.03)) == pytest.approx(0.375)


@pytest.mark.parametrize("strike,quantity", [(0, 1), (-1, 1), (float("nan"), 1), (5, float("inf"))])
def test_invalid_legs_are_rejected(strike, quantity):
    with pytest.raises(ValueError):
        ResolvedLeg(Right.CALL, strike, quantity)


def test_invalid_context_and_terminal_spots_are_rejected():
    contract = ResolvedContract("vanilla", "fixed", (ResolvedLeg(Right.PUT, 5, 1),), 0.5)
    with pytest.raises(ValueError):
        value_resolved(contract, PricingContext(5, 0.6, 0.15, 0.05, 0.03))
    with pytest.raises(ValueError):
        PricingContext(0, 0.5, 0.15, 0.05, 0.03)
    with pytest.raises(ValueError):
        terminal_payoff(contract, [0])


def test_specialised_products_are_not_frozen_as_vanilla_contracts():
    ms, target = _make_ms("USDBRL", 90, 12, "base_higher", None)
    for family in ("european_rko", "european_digital"):
        variants = price_variants(ms, family, target=target)
        assert variants
        assert all(v.resolved_contract is None for v in variants)
