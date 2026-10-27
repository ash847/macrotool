"""Single entry-pricing path, independent legacy parity, and retained-object identity."""

import pytest

from analytics.structure_pricer import _load_variants, price_variants
from tests.product_entry_reference import legacy_entry
from tests.test_product_model_parity import _assert_parity, _make_ms, _surface_for


@pytest.mark.parametrize("family", ["vanilla", "1x1_spread", "1x1.5_spread", "1x2_spread"])
@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("flat", [True, False])
def test_public_entry_matches_independent_legacy(family, is_call, flat):
    surface = _surface_for("USDBRL", flat)
    ms, target = _make_ms("USDBRL", 90, 12, "base_higher" if is_call else "base_lower", surface)
    definitions = _load_variants()[family]
    new = price_variants(ms, family, target=target, is_call=is_call, smile=surface,
                         variants_override=definitions, loss_budget=100)
    old = legacy_entry(ms, family, definitions, target, is_call, surface)
    assert len(new) == len(old)
    for actual, expected in zip(new, old):
        _assert_parity(actual, expected, f"{family} {is_call} {flat}")
        assert actual.product_price is not None
        assert actual.resolved_contract is actual.product_price.resolved_contract
        assert actual.breakevens == pytest.approx(expected.breakevens)
        assert actual.structure_notional is not None


@pytest.mark.parametrize("family", ["1x1.5_spread", "1x2_spread"])
@pytest.mark.parametrize("magnitude", [None, 1, 12])
def test_target_gates_are_preserved(family, magnitude):
    ms, target = _make_ms("USDBRL", 90, magnitude, "base_higher", None)
    definitions = [dict(label="atmf", long_type="atmf"),
                   dict(label="half-sigma", long_type="half_sigma", min_target_z=0.5)]
    new = price_variants(ms, family, target=target, variants_override=definitions)
    old = legacy_entry(ms, family, definitions, target, True, None)
    assert [v.variant_label for v in new] == [v.variant_label for v in old]
    for actual, expected in zip(new, old):
        _assert_parity(actual, expected, family)


def test_recommendation_display_does_not_reconstruct(monkeypatch):
    from agentic.standard_pack import _priced_structure_for
    ms, target = _make_ms("USDBRL", 90, 12, "base_higher", None)
    variant = price_variants(ms, "vanilla", target=target)[0]

    def fail(*args, **kwargs):
        raise AssertionError("display must use the retained price")

    monkeypatch.setattr("analytics.product_pricer.price", fail)
    monkeypatch.setattr("analytics.product_pricer.build_structure", fail)
    retained = _priced_structure_for("vanilla", variant.variant_label, ms, True,
                                      target, None, None, variant=variant)
    assert retained is variant.product_price


@pytest.mark.parametrize("request_text", ["vanilla ATMF", "1x1 ATMF/25 delta", "1x1.5 34/25", "1x2 ATMF/25 delta"])
def test_custom_trade_is_priced_once(request_text, monkeypatch):
    from analytics import product_pricer
    from agentic.price_structure import price_structure
    ms, target = _make_ms("USDBRL", 90, 12, "base_higher", None)
    original = product_pricer.price
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(product_pricer, "price", counted)
    result = price_structure(request_text, ms, is_call=True, target=target)
    assert len(calls) == 1
    assert result.priced_structure is result.variant.product_price


@pytest.mark.parametrize("family,definition", [
    ("vanilla", dict(label="tiny", delta=0.1)),
    ("1x1_spread", dict(label="tiny", long_delta=0.5, short_delta=0.25)),
    ("1x1_spread", dict(label="reversed", long_delta=0.1, short_delta=0.5)),
])
def test_legacy_small_premium_and_reversed_input_semantics(family, definition):
    from analytics.market_state import compute_market_state
    ms = compute_market_state(spot=5, fwd=5, vol=0.0001, T=7 / 365,
                              r_d=0, r_f=0, target=5.1, direction="base_higher")
    new = price_variants(ms, family, target=5.1, variants_override=[definition])[0]
    old = legacy_entry(ms, family, [definition], 5.1, True, None)[0]
    _assert_parity(new, old, family)
    assert not new.is_zero_cost


@pytest.mark.parametrize("direction", ["base_higher", "base_lower"])
@pytest.mark.parametrize("sizing_method", ["fixed_loss", "kelly"])
def test_ranked_pack_and_sizing_match_legacy_entry(direction, sizing_method, monkeypatch):
    from agentic.standard_pack import build_pack
    from analytics.distributions import interpolate_atm_vol
    from config.loader import load_config
    from knowledge_engine.models import TradeView
    from pricing.forwards import rate_context_for_snapshot
    from tests._curves import stated_lognormal
    from tests.test_product_model_parity import _frozen_snapshot

    ccy = _frozen_snapshot().get("USDBRL")
    view = TradeView(pair="USDBRL", direction=direction, direction_conviction="medium",
                     horizon_days=90, magnitude_pct=12, mode="recommend")
    config = load_config()
    options = dict(linear_notional=1_000_000, sizing_method=sizing_method)
    if sizing_method == "kelly":
        rc = rate_context_for_snapshot(ccy, 90 / 365)
        probs, bins = stated_lognormal(rc.forward * (1.08 if direction == "base_higher" else 0.92),
                                       interpolate_atm_vol(ccy, 90), 90 / 365)
        options.update(kelly_probs=probs, kelly_bins=bins)
    new = build_pack(view, ccy, config, **options)
    migrated = [r for r in new.recommended if r.variant.product_price is not None]
    assert migrated
    assert all(r.priced_structure is r.variant.product_price for r in migrated)

    monkeypatch.setattr("analytics.structure_pricer._product_variants", legacy_entry)
    old = build_pack(view, ccy, config, **options)
    assert [(r.structure_id, r.variant.variant_label, r.rank) for r in new.recommended] == [
        (r.structure_id, r.variant.variant_label, r.rank) for r in old.recommended]
    for actual, expected in zip(new.recommended, old.recommended):
        assert actual.score_ccy == pytest.approx(expected.score_ccy, abs=1e-6)
        assert actual.variant.structure_notional == pytest.approx(expected.variant.structure_notional, abs=1e-6)
        assert actual.variant.net_premium_pct == pytest.approx(expected.variant.net_premium_pct, abs=1e-12)
