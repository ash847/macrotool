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


def test_defined_loss_filter_is_applied_after_pricing(market):
    ids = {item.structure_id for item in score_structures(market, NO_TAILS).shortlist}
    for family in ids & {"seagull", "1x1.5_spread", "1x2_spread", "risk_reversal"}:
        assert not price_variants(market, family, target=5.45, structure_constraint=NO_TAILS)
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


def test_custom_vanilla_is_classified_from_its_legs_without_a_catalog_entry(market):
    off_menu = price_structure("vanilla 31Δ", market, is_call=True, target=5.45)
    assert isinstance(off_menu, PricedStructure)
    assert off_menu.variant.can_lose_beyond_premium is False   # a long option, at any delta
    blocked = price_structure("vanilla 31Δ", market, is_call=True, target=5.45, structure_constraint=NO_TAILS)
    assert isinstance(blocked, PricedStructure)
    assert blocked.variant.economics.contractual_loss_status == "bounded"
    known = price_structure("vanilla 25Δ", market, is_call=True, target=5.45, structure_constraint=NO_TAILS)
    assert isinstance(known, PricedStructure)
    assert known.variant.economics.contractual_loss_status == "bounded"
    risky = price_structure("1x1.5 25Δ/10Δ", market, is_call=True, target=5.45, structure_constraint=NO_TAILS)
    assert isinstance(risky, PricingUnavailable)
    assert "verified finite maximum loss" in risky.detail


def test_comparator_and_fallback_cannot_reintroduce_disallowed_variants(market, mixed_catalog):
    result = build_comparator_inputs(
        market, selection(), target=5.45, is_call=True,
        stop_price=5.1, loss_budget=1, preferences=PMPreferences(structure_constraint=NO_TAILS),
    )
    assert {item.variant_label for item in result.priced_variants_by_structure["vanilla"]} == {"allow", "deny", "unknown"}
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
    assert {item.variant_label for item in variants} == {"allow", "deny", "unknown"}


@pytest.mark.parametrize("direction", ["base_higher", "base_lower"])
def test_recommendation_pack_has_only_classified_no_tail_trades(direction):
    snapshot = load_snapshot()
    view = TradeView(
        pair="USDBRL", direction=direction, direction_conviction="medium",
        horizon_days=90, magnitude_pct=6.0, mode="recommend",
    )
    pack = build_pack(view, snapshot.get("USDBRL"), load_config(), structure_constraint=NO_TAILS)
    assert pack.recommended
    options = [item for item in pack.recommended if item.structure_id != "linear"]
    assert options
    assert all(item.variant.can_lose_beyond_premium is False for item in options)
    assert not any(item.structure_id == "linear" for item in pack.recommended)
    assert any(item["structure_id"] == "linear" for item in pack.tail_exclusions)


# ---------------------------------------------------------------------------
# Leg-derived classification (strike and delta requests alike), catalog cross-check
# ---------------------------------------------------------------------------

from analytics.construction_risk import LegRisk, leg_risk, reconcile  # noqa: E402
from knowledge_engine.tail_policy import _declared_tails, assign_variant_tails  # noqa: E402

C, P = True, False   # is_call


@pytest.mark.parametrize("terms, expected", [
    ([(1, 5.0, C)], (False, False, False)),                                  # long call
    ([(1, 5.0, P)], (False, False, False)),                                  # long put
    ([(1, 5.0, C), (-1, 5.4, C)], (False, False, False)),                    # call spread
    ([(1, 5.0, P), (-1, 4.8, P)], (False, False, False)),                    # put spread
    ([(1, 5.0, C), (-2, 5.4, C)], (False, True, True)),                      # 1x2 call
    ([(1, 5.0, P), (-1.5, 4.8, P)], (True, False, True)),                    # 1x1.5 put
    ([(1, 5.0, C), (-2, 5.4, C), (1, 5.8, C)], (False, False, False)),       # butterfly
    ([(1, 5.0, P), (-1, 4.8, P), (-0.67, 5.6, C)], (False, True, True)),     # seagull, put view
    ([(1, 5.0, C), (-1, 5.4, C), (-0.4, 4.6, P)], (True, False, True)),      # seagull, call view
    ([(1, 5.0, P), (-1, 4.8, P), (-0.0, 5.6, C)], (False, False, False)),    # seagull, no wing sold
    ([(1, 4.8, P), (-1, 5.0, P)], (False, False, True)),                     # spread the wrong way round
])
def test_leg_risk_reads_tails_and_loss_from_the_legs(terms, expected):
    assert leg_risk(terms) == LegRisk(*expected)


@pytest.mark.parametrize("terms", [[], [(1, 0.0, C)], [(1, float("nan"), C)]])
def test_leg_risk_refuses_invalid_legs(terms):
    assert leg_risk(terms) is None


@pytest.mark.parametrize("declared, derived, expected", [
    (None, True, True), (None, False, False), (True, True, True), (False, False, False),
    (True, False, None), (False, True, None), (True, None, True), (None, None, None),
])
def test_reconcile_uses_legs_and_treats_disagreement_as_unknown(declared, derived, expected):
    assert reconcile(declared, derived) is expected


@pytest.mark.parametrize("pair", sorted(load_snapshot().currencies))
@pytest.mark.parametrize("direction", ["base_higher", "base_lower"])
def test_every_menu_construction_agrees_with_the_catalog(pair, direction):
    """The catalog cross-check: on every pair and direction, each priced menu construction's
    leg-derived classification matches its catalog declaration — so none turns unknown."""
    from analytics.vol_surface import build_vol_surface
    from pricing.forwards import rate_context_for_snapshot

    snap = load_snapshot()
    ccy = snap.get(pair)
    T = 0.25
    rc = rate_context_for_snapshot(ccy, T)
    vol = 0.12   # classification depends on leg geometry, not the vol level
    target = rc.forward * (1.06 if direction == "base_higher" else 0.94)
    ms = compute_market_state(spot=ccy.spot, fwd=rc.forward, vol=vol, T=T, r_d=rc.r_d, r_f=rc.r_f,
                              target=target, direction=direction)
    is_call = target > rc.forward
    try:
        surface = build_vol_surface(ccy)
    except Exception:
        surface = None
    checked = 0
    for family, constructions in _load_variants().items():
        if family in ("european_digital_rko", "rko"):
            continue
        warnings = []
        for variant in price_variants(ms, family, target=target, is_call=is_call, smile=surface,
                                      loss_budget=1.0, warnings=warnings):
            declared = next(c for c in constructions if c["label"] == variant.variant_label)
            seagull_no_wing = family == "seagull" and (variant.wing_ratio or 0) <= 0
            if not seagull_no_wing:
                assert variant.can_lose_beyond_premium is declared["can_lose_beyond_premium"], variant.variant_label
            assign_variant_tails(family, variant, is_call)
            if not seagull_no_wing:
                assert (variant.lower_spot_tail, variant.higher_spot_tail) == _declared_tails(family, is_call), \
                    variant.variant_label
            checked += 1
        assert not [w for w in warnings if "treated as unknown" in w]
    assert checked > 20


def test_strike_requests_are_classified_and_gated_like_menu_ones(market):
    spread = price_structure("1x1 5.25 / 5.40", market, is_call=True, target=5.45)
    assert spread.variant.can_lose_beyond_premium is False
    assert isinstance(price_structure("1x1 5.25 / 5.40", market, is_call=True, target=5.45,
                                      structure_constraint=NO_TAILS), PricedStructure)
    ratio = price_structure("1x2 5.25 / 5.40", market, is_call=True, target=5.45)
    assert ratio.variant.can_lose_beyond_premium is True
    assign_variant_tails("1x2_spread", ratio.variant, True, construction={"long_strike": 5.25, "short_strike": 5.40})
    assert (ratio.variant.lower_spot_tail, ratio.variant.higher_spot_tail) == (False, True)


def test_catalog_disagreement_is_unknown_and_noted(market, mixed_catalog):
    warnings = []
    variants = price_variants(market, "vanilla", target=5.45, loss_budget=1, warnings=warnings)
    deny = next(v for v in variants if v.variant_label == "deny")   # declared True; a long call can't
    assert deny.can_lose_beyond_premium is None
    assert any("'deny'" in w and "treated as unknown" in w for w in warnings)
    unknown = next(v for v in variants if v.variant_label == "unknown")   # no declaration: legs decide
    assert unknown.can_lose_beyond_premium is False
