import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from agentic.price_structure import PricedStructure, PricingUnavailable, price_structure
from agentic.standard_pack import _price_recommended_fallback, build_pack
from analytics.construction_risk import declared_additional_loss, passes_no_tails
from analytics.market_state import compute_market_state
from analytics.structure_pricer import _load_variants, price_variants
from config.loader import load_config
from data.snapshot_loader import load_snapshot
from knowledge_engine.comparator import PMPreferences, build_comparator_inputs
from knowledge_engine.construction_policy import configured_additional_loss, family_has_no_tail_construction
from knowledge_engine.models import StructureSelectionResult, StructureShortlistItem, TradeView
from knowledge_engine.structure_scorer import score_structures


NO_TAILS = "Avoid tail-risky structures"


@pytest.fixture
def market():
    return compute_market_state(
        spot=5.0, fwd=5.2, vol=0.15, T=0.25, r_d=0.12, r_f=0.04,
        target=5.45, direction="base_higher",
    )


def selection():
    return StructureSelectionResult(shortlist=[StructureShortlistItem(
        structure_id="vanilla", display_name="Vanilla", rank=1, rationale="test",
        rule_id="test", sizing_modifier=None, caution=None, optimised_for="test",
    )], rules_fired=[])


@pytest.fixture
def mixed_catalog(tmp_path, monkeypatch):
    catalog = _load_variants()
    catalog["vanilla"] = [
        {"label": "allow", "delta": 0.25, "can_lose_beyond_premium": False},
        {"label": "deny", "delta": 0.35, "can_lose_beyond_premium": True},
        {"label": "unknown", "delta": 0.20},
    ]
    source = tmp_path / "variants.json"
    source.write_text(json.dumps(catalog))
    monkeypatch.setattr("analytics.structure_pricer._VARIANTS_PATH", source)
    return catalog


def test_catalog_has_explicit_flags_and_expected_classifications():
    catalog = _load_variants()
    additional_loss = {"seagull", "1x1.5_spread", "1x2_spread"}
    for family, constructions in catalog.items():
        assert len({item["label"] for item in constructions}) == len(constructions)
        for item in constructions:
            assert type(item["can_lose_beyond_premium"]) is bool
            assert item["can_lose_beyond_premium"] is (family in additional_loss)


@pytest.mark.parametrize("value", [None, "false", "true", 0, 1, [], {}])
def test_only_boolean_false_passes(value):
    assert declared_additional_loss({"can_lose_beyond_premium": value}) is None
    assert not passes_no_tails({"can_lose_beyond_premium": value})
    assert passes_no_tails({"can_lose_beyond_premium": False})


def test_family_gate_uses_config_including_one_and_half_ratio(market):
    ids = {item.structure_id for item in score_structures(market, NO_TAILS).shortlist}
    assert not ids & {"seagull", "1x1.5_spread", "1x2_spread", "risk_reversal"}
    assert {"vanilla", "1x2x1_spread"} <= ids


def test_mixed_family_filters_constructions_not_whole_family(market, mixed_catalog):
    assert family_has_no_tail_construction("vanilla")
    all_variants = price_variants(market, "vanilla", target=5.45, loss_budget=1)
    allowed = price_variants(market, "vanilla", target=5.45, loss_budget=1, exclude_loss_beyond_premium=True)
    assert {item.variant_label for item in all_variants} == {"allow", "deny", "unknown"}
    assert [item.variant_label for item in allowed] == ["allow"]
    assert asdict(allowed[0]) == asdict(all_variants[0])


def test_custom_matching_is_by_terms_not_label(mixed_catalog):
    assert configured_additional_loss("vanilla", {"label": "custom", "delta": 0.25}) is False
    assert configured_additional_loss("vanilla", {"label": "allow", "delta": 0.31}) is None


def test_custom_unknown_remains_priceable_only_without_restriction(market):
    unclassified = price_structure("vanilla 31Δ", market, is_call=True, target=5.45)
    assert isinstance(unclassified, PricedStructure)
    assert unclassified.variant.can_lose_beyond_premium is None
    blocked = price_structure("vanilla 31Δ", market, is_call=True, target=5.45, structure_constraint=NO_TAILS)
    assert isinstance(blocked, PricingUnavailable)
    assert "no approved risk classification" in blocked.detail
    known = price_structure("vanilla 25Δ", market, is_call=True, target=5.45, structure_constraint=NO_TAILS)
    assert isinstance(known, PricedStructure)
    assert known.variant.can_lose_beyond_premium is False
    risky = price_structure("1x1.5 25Δ/10Δ", market, is_call=True, target=5.45, structure_constraint=NO_TAILS)
    assert isinstance(risky, PricingUnavailable)
    assert "can lose more than premium" in risky.detail


def test_comparator_and_fallback_cannot_reintroduce_disallowed_variants(market, mixed_catalog):
    result = build_comparator_inputs(
        market, selection(), target=5.45, is_call=True,
        stop_price=5.1, loss_budget=1, preferences=PMPreferences(structure_constraint=NO_TAILS),
    )
    assert [item.variant_label for item in result.priced_variants_by_structure["vanilla"]] == ["allow"]
    fallback = _price_recommended_fallback(market, selection(), None, True, None, structure_constraint=NO_TAILS)
    assert [item.variant.variant_label for item in fallback] == ["allow"]


def test_trade_view_evaluation_filters_same_mixed_family(market, mixed_catalog):
    from interface.structure_eval import compute_structure_evaluation

    flow = SimpleNamespace(
        market_state=market, selector_result=selection(), target_rr=3,
        structure_constraint=NO_TAILS, view=SimpleNamespace(pair="USDBRL", direction="base_higher"),
    )
    result = compute_structure_evaluation(flow, 5.45)
    assert result is not None
    variants = [item for item in result.variants if item.structure_id != "linear"]
    assert [item.variant_label for item in variants] == ["allow"]


@pytest.mark.parametrize("direction", ["base_higher", "base_lower"])
def test_recommendation_pack_has_only_classified_no_tail_trades(direction):
    snapshot = load_snapshot()
    view = TradeView(
        pair="USDBRL", direction=direction, direction_conviction="medium",
        horizon_days=90, magnitude_pct=6.0, mode="recommend",
    )
    pack = build_pack(view, snapshot.get("USDBRL"), load_config(), structure_constraint=NO_TAILS)
    assert pack.recommended
    assert all(item.variant.can_lose_beyond_premium is False for item in pack.recommended)
