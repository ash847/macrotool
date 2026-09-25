from dataclasses import replace

import pytest

from agentic.render import _ccy_summary
from agentic.session import AgentSession
from agentic.tools import dispatch
from analytics.sizing import SizingSpec
from analytics.sizing_trace import distribution_fingerprint
from analytics.structure_pricer import _size_variant, _size_variants_kelly, price_variants
from analytics.market_state import compute_market_state
from config.loader import load_config
from data.snapshot_loader import load_snapshot
from tests.test_trade_economics import variant


@pytest.mark.parametrize("premium,proxy,budget,wanted,final,rule", [
    (0.02, 0.02, 100, 5000, 5000, "loss_budget"),
    (0.02, 0.02, 1000, 50000, 10000, "notional_cap"),
    (-0.01, 0.01, 100, None, 10000, "net_credit_policy"),
    (0.0, 0.0, 100, None, 10000, "missing_or_near_zero_proxy"),
    (0.0, None, 100, None, 10000, "missing_or_near_zero_proxy"),
])
def test_fixed_loss_records_actual_branch(premium, proxy, budget, wanted, final, rule):
    trade = variant(premium)
    trade.max_loss_pct = proxy
    _size_variant(trade, budget, 1000)
    trace = trade.sizing_trace
    assert trace.effective_method == "fixed_loss"
    assert trace.loss_budget == budget
    assert trace.per_unit_loss_proxy == proxy
    assert trace.uncapped_notional == wanted
    assert trace.final_notional == final == trade.structure_notional
    assert trace.binding_constraint == rule
    assert trace.notional_cap == 10000
    assert trade.net_premium_ccy == premium * final


@pytest.mark.parametrize("fraction,final,rule,status", [
    (3.0, 1500, "fractional_kelly", "sized"),
    (30.0, 10000, "notional_cap", "sized"),
    (0.0, 0, "fractional_kelly", "zero"),
])
def test_kelly_records_fraction_lambda_and_cap(monkeypatch, fraction, final, rule, status):
    monkeypatch.setattr("analytics.sizing.kelly_fraction_per_notional", lambda *args: fraction)
    trade = variant()
    spec = SizingSpec(method="kelly", bankroll=1000, kelly_lambda=0.5,
                      kelly_probs=(0.5, 0.5), kelly_bins=(90.0, 120.0))
    _size_variants_kelly([trade], "vanilla", True, 100, 0.03, 0.25, spec, 1000)
    trace = trade.sizing_trace
    assert trace.status == status
    assert trace.full_kelly_fraction == fraction
    assert trace.kelly_lambda == 0.5
    assert trace.uncapped_notional == 0.5 * fraction * 1000
    assert trace.final_notional == final == trade.structure_notional
    assert trace.binding_constraint == rule
    assert trace.distribution_id == distribution_fingerprint(spec.kelly_probs, spec.kelly_bins)
    assert trace.distribution_points == 2


def test_error_is_not_zero_or_fixed_loss_fallback(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("Unsupported payoff")
    monkeypatch.setattr("analytics.payoffs.base_ccy_payoff_for_trade_rec", fail)
    trade = variant()
    spec = SizingSpec(method="kelly", kelly_probs=(1.0,), kelly_bins=(120.0,))
    _size_variants_kelly([trade], "vanilla", True, 100, 0.03, 0.25, spec, 1000)
    assert trade.structure_notional is None
    assert trade.sizing_trace.status == "error"
    assert trade.sizing_trace.effective_method == "kelly"
    assert "no fixed-loss fallback" in _ccy_summary(trade, "EUR")


def test_unsized_trade_has_explicit_reason():
    market = compute_market_state(5.0, 5.1, 0.15, 0.25, 0.05, 0.03, target=5.5)
    trades = price_variants(market, "vanilla", target=5.5)
    assert trades
    for trade in trades:
        assert trade.structure_notional is None
        assert trade.sizing_trace.status == "unavailable"
        assert "No positive sizing budget" in _ccy_summary(trade, "USD")


def test_direct_pricer_missing_distribution_records_requested_method():
    market = compute_market_state(5.0, 5.1, 0.15, 0.25, 0.05, 0.03, target=5.5)
    trades = price_variants(market, "vanilla", target=5.5, loss_budget=1,
                            sizing_spec=SizingSpec(method="kelly"))
    assert trades
    for trade in trades:
        assert trade.sizing_trace.requested_method == "kelly"
        assert trade.sizing_trace.effective_method == "fixed_loss"
        assert trade.sizing_trace.fallback_reason == "No stated Kelly distribution supplied"


def test_empty_kelly_distribution_is_unavailable_not_valid_zero():
    trade = variant()
    spec = SizingSpec(method="kelly", kelly_probs=(), kelly_bins=())
    _size_variants_kelly([trade], "vanilla", True, 100, 0.03, 0.25, spec, 1000)
    assert trade.structure_notional == 0
    assert trade.sizing_trace.status == "unavailable"


@pytest.mark.parametrize("capital", [1000, 1_000_000, 1_000_000_000])
def test_explanation_uses_full_currency_units(capital):
    trade = variant()
    _size_variant(trade, capital * 0.02, capital)
    text = _ccy_summary(trade, "EUR")
    assert f"Final notional={capital:,.2f} EUR" in text
    assert f"Reference capital={capital:,.2f} EUR" in text


def test_audit_text_ignores_sub_display_precision_round_trip_noise():
    trade = variant()
    _size_variant(trade, 1.0, 100)
    trade.sizing_trace = replace(
        trade.sizing_trace, budget_distance=0.01, budget_reference=155.500475,
        budget_target=160.16549, budget_input_rr=3,
    )
    before = _ccy_summary(trade, "USD")
    trade.sizing_trace = replace(
        trade.sizing_trace, budget_distance=0.0100000007,
        per_unit_loss_proxy=0.0200000002,
    )
    assert _ccy_summary(trade, "USD") == before


def test_pack_and_custom_trade_retain_disclosed_kelly_fallback():
    snapshot = load_snapshot()
    session = AgentSession(snapshot=snapshot, cfg=load_config(), sizing_method="kelly", linear_notional=1_000_000)
    text, error = dispatch(session, "run_standard_pack", {
        "pair": "USDBRL", "direction": "base_higher", "horizon_days": 90, "magnitude_pct": 6.0,
    })
    assert not error
    assert session.pack.kelly_fallback
    assert "SIZING AUDIT" in text
    for rec in session.pack.recommended:
        trace = rec.variant.sizing_trace
        assert trace.requested_method == "kelly"
        assert trace.effective_method == "fixed_loss"
        assert trace.fallback_reason
        assert trace.loss_budget == pytest.approx(trace.reference_capital * trace.budget_distance)
        assert trace.budget_input_rr == 3
    reply, error = dispatch(session, "price_structure", {"request": "vanilla 31Δ"})
    assert not error
    assert "Fallback reason: No stated Kelly distribution" in reply
    assert "Budget origin:" in reply
    assert session.priced[-1].variant.sizing_trace.requested_method == "kelly"
