"""Construction is not permitted inside aggregation or market revaluation."""

import pytest

from analytics.product_model import ResolvedContract, ResolvedLeg, Right
from analytics.product_policies import sizing_loss_proxy
from analytics.product_pricer import build_structure, construct_package, _price_constructed
from analytics.resolved_pricing import PricingContext, terminal_payoff, value_resolved
from analytics.structure_pricer import _load_variants
from tests.product_entry_reference import legacy_entry
from tests.test_product_model_parity import _assert_parity, _make_ms, _surface_for


@pytest.mark.parametrize("family", ["vanilla", "1x1_spread", "1x1.5_spread",
                                    "1x2_spread", "1x2x1_spread", "seagull"])
@pytest.mark.parametrize("is_call", [True, False])
def test_aggregation_cannot_resolve_or_construct(family, is_call, monkeypatch):
    surface = _surface_for("USDBRL", False)
    ms, target = _make_ms("USDBRL", 90, 12, "base_higher" if is_call else "base_lower", surface)
    definition = _load_variants()[family][0]
    structure = build_structure(family, definition, is_call)
    package = construct_package(structure, ms, target, surface)
    stop = ms.fwd * (0.96 if is_call else 1.04)
    expected = legacy_entry(ms, family, [definition], target, is_call, surface, stop_price=stop)[0]

    def fail(*args, **kwargs):
        raise AssertionError("Only construction may solve strikes or quantities")

    monkeypatch.setattr("analytics.product_pricer.construct_package", fail)
    monkeypatch.setattr("analytics.product_pricer._resolve_leg", fail)
    monkeypatch.setattr("analytics.product_pricer.freeze_contract", fail)
    actual = _price_constructed(structure, package, ms, target, surface, stop)
    _assert_parity(actual, expected, family)
    assert actual.resolved_contract is package.contract
    before = repr(package.contract)
    value_resolved(package.contract, PricingContext(ms.spot * 1.2, ms.T / 2,
                                                    ms.vol * 1.2, ms.r_d, ms.r_f, surface))
    assert repr(package.contract) == before


def test_four_leg_valuation_does_not_imply_approved_sizing_policy():
    contract = ResolvedContract("custom", "four legs", (
        ResolvedLeg(Right.CALL, 4, 1), ResolvedLeg(Right.CALL, 5, -1.3),
        ResolvedLeg(Right.PUT, 4, -0.4), ResolvedLeg(Right.PUT, 3, 0.4),
    ), 0.5)
    assert float(terminal_payoff(contract, 6)) == pytest.approx((2 - 1.3) / 6)
    assert value_resolved(contract, PricingContext(6, 0, 0.2, 0.05, 0.03)) == pytest.approx(0.7)
    with pytest.raises(ValueError, match="No approved sizing policy"):
        sizing_loss_proxy(contract, 0.01, None)


def test_entry_breakevens_are_calculated_once(monkeypatch):
    from analytics import expiry_breakeven
    from analytics.structure_pricer import price_variants
    ms, target = _make_ms("USDBRL", 90, 12, "base_higher", None)
    original = expiry_breakeven.expiry_breakevens
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(expiry_breakeven, "expiry_breakevens", counted)
    rows = price_variants(ms, "seagull", target=target)
    assert len(calls) == len(rows)
    assert all(row.breakevens == row.product_price.breakevens for row in rows)
