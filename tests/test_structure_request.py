"""Phase 1 tests — structure-request grammar + parser.

Pure: no LLM, no MarketState, no pricing. Proves the grammar is a *superset* of
the curated structure_variants.json menu (so it breaks nothing) and that every
unsafe request is rejected with a structured reason.
"""

import json
from pathlib import Path

import pytest

from agentic.structure_request import (
    BAD_DELTA,
    BAD_LEG_KIND_FOR_FAMILY,
    BAD_PREMIUM,
    EMPTY_REQUEST,
    FAMILY_DISABLED,
    UNKNOWN_FAMILY,
    WRONG_LEG_COUNT,
    ClarificationNeeded,
    StructureRequest,
    StructureRequestError,
    parse_structure_request,
    to_variant_dict,
)

_VARIANTS_PATH = (
    Path(__file__).parent.parent / "knowledge" / "defaults" / "structure_variants.json"
)


def _dict_no_label(req_str: str) -> dict:
    res = parse_structure_request(req_str)
    assert isinstance(res, StructureRequest), res
    d = to_variant_dict(res)
    d.pop("label", None)
    return d


# ---------------------------------------------------------------------------
# 1. Happy path per family
# ---------------------------------------------------------------------------

def test_vanilla():
    assert _dict_no_label("vanilla 25Δ") == {"delta": 0.25}


def test_1x1_spread():
    assert _dict_no_label("1x1 40Δ vs 20Δ") == {"long_delta": 0.40, "short_delta": 0.20}


def test_ratio_delta_pair():
    assert _dict_no_label("34 vs 25 1x1.5") == {"long_delta": 0.34, "short_delta": 0.25}


def test_seagull():
    assert _dict_no_label("seagull 50Δ/25Δ/25Δ") == {
        "spread_long": 0.50,
        "spread_short": 0.25,
        "wing_delta": 0.25,
    }


def test_european_rko():
    assert _dict_no_label("erko 40Δ/20Δ") == {"long_delta": 0.40, "barrier": 0.20}


def test_digital():
    assert _dict_no_label("digital 10%") == {"target_prem_pct": 0.10}


# ---------------------------------------------------------------------------
# 2. Delta normalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("req", ["vanilla 34Δ", "vanilla 0.34", "vanilla 34 delta"])
def test_delta_normalization(req):
    assert _dict_no_label(req) == {"delta": 0.34}


def test_atmf_long_leg_is_half_delta():
    assert _dict_no_label("vanilla ATMF") == {"delta": 0.50}


# ---------------------------------------------------------------------------
# 3. Ratio-family dual form
# ---------------------------------------------------------------------------

def test_ratio_anchored_atmf():
    assert _dict_no_label("1x1.5 ATMF vs target") == {"long_type": "atmf"}


def test_ratio_anchored_half_sigma():
    assert _dict_no_label("½σ vs target 1x2") == {
        "long_type": "half_sigma",
        "min_target_z": 0.5,
    }


def test_ratio_half_sigma_words():
    assert _dict_no_label("1x1.5 half sigma vs target") == {
        "long_type": "half_sigma",
        "min_target_z": 0.5,
    }


# ---------------------------------------------------------------------------
# 4. Curated guard (digital)
# ---------------------------------------------------------------------------

def test_digital_rejects_delta_leg():
    with pytest.raises(StructureRequestError) as e:
        parse_structure_request("digital 25Δ")
    assert e.value.reason == BAD_LEG_KIND_FOR_FAMILY


# ---------------------------------------------------------------------------
# 5. Disabled families
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("req", ["rko 25%", "european digital rko 10%", "digital rko 10%"])
def test_disabled_families(req):
    with pytest.raises(StructureRequestError) as e:
        parse_structure_request(req)
    assert e.value.reason == FAMILY_DISABLED


# ---------------------------------------------------------------------------
# 6. Rejections
# ---------------------------------------------------------------------------

def test_unrecognized_token_clarifies():
    # An unrecognized word is NOT silently ignored — it triggers clarification.
    res = parse_structure_request("frobnicator 25Δ")
    assert isinstance(res, ClarificationNeeded)
    assert "frobnicator" in res.question


def test_unknown_structure_name_clarifies():
    # An unknown *structure* word (not a family synonym) → clarify, not infer.
    res = parse_structure_request("calendar 25Δ vs 10Δ")
    assert isinstance(res, ClarificationNeeded)
    assert "calendar" in res.question


def test_clean_request_not_flagged_as_unrecognized():
    # Legitimate markers (ATMF, target, vs, Δ) must not be flagged.
    assert isinstance(parse_structure_request("1x1.5 ATMF vs target"), StructureRequest)


def test_unknown_family_guard():
    # UNKNOWN_FAMILY is a defensive guard in to_variant_dict — unreachable via
    # parse, but must fire if a bogus family is ever constructed directly.
    bogus = StructureRequest(family="bogus", legs=(), canonical="bogus")
    with pytest.raises(StructureRequestError) as e:
        to_variant_dict(bogus)
    assert e.value.reason == UNKNOWN_FAMILY


def test_empty_request():
    with pytest.raises(StructureRequestError) as e:
        parse_structure_request("   ")
    assert e.value.reason == EMPTY_REQUEST


def test_wrong_leg_count():
    with pytest.raises(StructureRequestError) as e:
        parse_structure_request("vanilla 25Δ vs 10Δ")
    assert e.value.reason == WRONG_LEG_COUNT


def test_bad_delta():
    with pytest.raises(StructureRequestError) as e:
        parse_structure_request("vanilla 120Δ")
    assert e.value.reason == BAD_DELTA


def test_bad_premium():
    with pytest.raises(StructureRequestError) as e:
        parse_structure_request("digital 0%")
    assert e.value.reason == BAD_PREMIUM


# ---------------------------------------------------------------------------
# 7. Ambiguity
# ---------------------------------------------------------------------------

def test_ambiguous_two_bare_deltas():
    res = parse_structure_request("34 vs 25")
    assert isinstance(res, ClarificationNeeded)


# ---------------------------------------------------------------------------
# 8. Direction words ignored
# ---------------------------------------------------------------------------

def test_direction_words_ignored():
    call = _dict_no_label("vanilla 25Δ call")
    put = _dict_no_label("vanilla 25Δ put")
    assert call == put == {"delta": 0.25}


# ---------------------------------------------------------------------------
# 9. Parity with the curated menu — grammar is a superset
# ---------------------------------------------------------------------------

# A request string for each curated variant; must round-trip to the SAME dict.
_PARITY = {
    "vanilla": [("vanilla ATMF", 0), ("vanilla 25Δ", 1), ("vanilla 15Δ", 2)],
    "1x1_spread": [
        ("1x1 ATMF/25Δ", 0), ("1x1 25Δ/10Δ", 1), ("1x1 25Δ/15Δ", 2),
        ("1x1 40Δ/20Δ", 3), ("1x1 30Δ/10Δ", 4), ("1x1 20Δ/10Δ", 5),
    ],
    "1x1.5_spread": [
        ("1x1.5 ATMF vs target", 0), ("1x1.5 ½σ vs target", 1),
        ("1x1.5 ATMF/25Δ", 2), ("1x1.5 25Δ/10Δ", 3),
    ],
    "european_digital": [
        ("digital 30%", 0), ("digital 20%", 1), ("digital 10%", 2),
    ],
    "european_rko": [
        ("erko ATMF/25Δ", 0), ("erko 25Δ/10Δ", 1), ("erko 40Δ/20Δ", 3),
    ],
    "seagull": [("seagull ATMF/25Δ/25Δ", 0), ("seagull 25Δ/10Δ/25Δ", 1)],
}


def _curated_keys(variant: dict) -> dict:
    """Variant dict with the label and construction-only keys, comparable to ours."""
    return {k: v for k, v in variant.items() if k != "label"}


def test_parity_with_curated_menu():
    with open(_VARIANTS_PATH) as f:
        menu = json.load(f)

    for family, cases in _PARITY.items():
        curated_variants = menu[family]
        for req_str, idx in cases:
            ours = _dict_no_label(req_str)
            theirs = _curated_keys(curated_variants[idx])
            assert ours == theirs, f"{family} '{req_str}': {ours} != {theirs}"
