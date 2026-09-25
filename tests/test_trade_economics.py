from dataclasses import asdict
from types import SimpleNamespace

import pytest

from agentic.render import _ccy_summary, _variant_summary, render_pack
from agentic.standard_pack import build_pack
from analytics.structure_pricer import PricedVariant
from analytics.trade_economics import compute_trade_economics
from config.loader import load_config
from data.snapshot_loader import load_snapshot
from knowledge_engine.models import TradeView
from pricing.black_scholes import call_mtm


@pytest.fixture
def market():
    return SimpleNamespace(T=90 / 365, spot=100.0, vol=0.2, r_d=0.05, r_f=0.03)


def variant(premium=0.02, zero_cost=False):
    return PricedVariant(
        variant_label="test", strikes=[100.0, 110.0], barrier=None,
        net_premium_pct=premium, breakeven=None, payoff_at_target_pct=0.9,
        rr_at_target=45.0, max_loss_pct=abs(premium), wing_ratio=None,
        is_zero_cost=zero_cost,
    )


def test_net_target_return_is_not_legacy_gross_rr(market):
    trade = variant()
    before = asdict(trade)
    result = compute_trade_economics(trade, "vanilla", market, target=120, is_call=True)
    assert result.target_net_pnl_pct == pytest.approx(20 / 120 - 0.02)
    assert result.target_return_on_premium == pytest.approx((20 / 120 - 0.02) / 0.02)
    assert result.contractual_max_loss_pct == 0.02
    assert result.evaluation_days == 90
    assert result.valuation_kind == "expiry_payoff"
    assert asdict(trade) == before


@pytest.mark.parametrize("premium,zero_cost", [(0.0, True), (-0.01, False)])
def test_no_outlay_keeps_net_pnl_without_ratio(market, premium, zero_cost):
    result = compute_trade_economics(
        variant(premium, zero_cost), "1x2_spread", market, target=120, is_call=True,
    )
    assert result.ratio_status == "not_applicable"
    assert result.ratio_reason == "Not applicable — no premium outlay"
    assert result.target_return_on_premium is None
    assert result.target_net_pnl_pct == pytest.approx(-premium)


def test_ratio_spread_uses_all_legs_and_does_not_claim_premium_is_max_loss(market):
    result = compute_trade_economics(
        variant(), "1x2_spread", market, target=130, is_call=True, loss_budget=1000,
    )
    assert result.target_net_pnl_pct == pytest.approx((30 - 2 * 20) / 130 - 0.02)
    assert result.sizing_loss_pct == 0.02
    assert result.loss_budget == 1000
    assert result.contractual_loss_status == "unknown"
    assert result.contractual_max_loss_pct is None


def test_before_expiry_uses_mtm_and_same_premium_basis(market):
    result = compute_trade_economics(
        variant(), "vanilla", market, target=105, is_call=True, evaluation_days=30,
    )
    expected = call_mtm(105, 100, 60 / 365, 0.2, 0.05, 0.03) / 105 - 0.02
    assert result.target_net_pnl_pct == pytest.approx(expected)
    assert result.valuation_kind == "mark_to_market"
    assert result.evaluation_days == 30


@pytest.mark.parametrize("target,reason", [(None, "No target specified"), (0, "Target must be")])
def test_missing_invalid_target_has_explicit_reason(market, target, reason):
    result = compute_trade_economics(variant(), "vanilla", market, target=target, is_call=True)
    assert result.ratio_status == "unavailable"
    assert reason in result.ratio_reason


def test_unsupported_path_product_does_not_return_fake_zero(market):
    result = compute_trade_economics(variant(), "rko", market, target=120, is_call=True)
    assert result.target_net_pnl_pct is None
    assert "path state" in result.target_pnl_reason


def test_seagull_proxy_and_budget_are_not_a_loss_bound(market):
    trade = variant(0.0, True)
    trade.strikes = [100.0, 110.0, 90.0]
    trade.wing_ratio = 1.5
    trade.max_loss_pct = 0.015
    result = compute_trade_economics(
        trade, "seagull", market, target=120, is_call=True,
        loss_budget=1000, stop_price=95,
    )
    assert result.sizing_loss_pct == 0.015
    assert result.sizing_reference == 95
    assert result.loss_budget == 1000
    assert result.contractual_max_loss_pct is None
    assert result.ratio_status == "not_applicable"


def test_target_pricing_failure_is_unavailable_not_zero(market, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("cannot price")
    monkeypatch.setattr("analytics.scenario_pricer._value_variant", fail)
    result = compute_trade_economics(variant(), "vanilla", market, target=120, is_call=True)
    assert result.target_net_pnl_pct is None
    assert result.ratio_status == "unavailable"
    assert "unavailable" in result.ratio_reason


def test_render_scales_once_and_hides_legacy_fields(market):
    trade = variant()
    trade.structure_notional = 1_000_000
    trade.economics = compute_trade_economics(
        trade, "1x2_spread", market, target=120, is_call=True, loss_budget=20_000,
    )
    text = _variant_summary(trade)
    amounts = _ccy_summary(trade, "EUR")
    assert "rr=" not in text
    assert "max_loss=" not in text
    assert "90-day horizon" in text
    assert "target return on premium=-1.00×" in text
    assert "net P&L at target≈-20,000 EUR" in amounts
    assert "loss budget=20,000 EUR" in amounts
    assert "contractual maximum loss≈" not in amounts


@pytest.mark.parametrize("direction", ["base_higher", "base_lower"])
@pytest.mark.parametrize("sizing_method", ["fixed_loss", "kelly"])
def test_pack_numbers_and_rankings_unchanged(monkeypatch, direction, sizing_method):
    snapshot = load_snapshot()
    config = load_config()
    view = TradeView(
        pair="USDBRL", direction=direction, direction_conviction="medium",
        horizon_days=90, magnitude_pct=6.0, mode="recommend",
    )
    options = dict(linear_notional=1_000_000, sizing_method=sizing_method)
    if sizing_method == "kelly":
        from tests._curves import stated_lognormal
        from pricing.forwards import rate_context_for_snapshot
        from analytics.distributions import interpolate_atm_vol
        currency = snapshot.get("USDBRL")
        context = rate_context_for_snapshot(currency, 90 / 365)
        probs, bins = stated_lognormal(context.forward * 1.04, interpolate_atm_vol(currency, 90), 90 / 365)
        options.update(kelly_probs=probs, kelly_bins=bins)
    updated = build_pack(view, snapshot.get("USDBRL"), config, **options)
    assert updated.recommended
    for rec in updated.recommended:
        assert rec.variant.economics is not None
    text = render_pack(updated, view)
    assert "target return on premium" in text
    assert "max loss =" not in text
    monkeypatch.setattr("analytics.structure_pricer.compute_trade_economics", lambda *args, **kwargs: None)
    legacy = build_pack(view, snapshot.get("USDBRL"), config, **options)
    def numeric_results(pack):
        results = []
        for rec in pack.recommended:
            data = asdict(rec.variant)
            data.pop("economics")
            results.append((rec.structure_id, rec.rank, rec.score_ccy, data))
        return results
    assert numeric_results(updated) == numeric_results(legacy)
