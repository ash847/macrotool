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

# A request string for each curated construction; each must parse to a construction the
# catalog actually contains. Membership rather than position: these were pinned to list
# indices, so editing the vanilla ladder silently repointed "vanilla 35Δ" at the 40Δ
# entry and the failure read as a parser bug rather than a stale fixture.
_PARITY = {
    "vanilla": ["vanilla ATMF", "vanilla 40Δ", "vanilla 30Δ", "vanilla 25Δ",
                "vanilla 20Δ", "vanilla 15Δ", "vanilla 10Δ"],
    "1x1_spread": ["1x1 ATMF/25Δ", "1x1 25Δ/10Δ", "1x1 25Δ/15Δ",
                   "1x1 40Δ/20Δ", "1x1 30Δ/10Δ", "1x1 20Δ/10Δ"],
    "1x1.5_spread": ["1x1.5 ATMF/25Δ", "1x1.5 25Δ/10Δ", "1x1.5 25Δ/15Δ",
                     "1x1.5 40Δ/20Δ", "1x1.5 30Δ/10Δ", "1x1.5 20Δ/10Δ"],
    "european_digital": ["digital 30%", "digital 20%", "digital 10%"],
    "european_rko": ["erko ATMF/25Δ", "erko 25Δ/10Δ", "erko 25Δ/15Δ",
                     "erko 40Δ/20Δ", "erko 30Δ/10Δ", "erko 20Δ/10Δ"],
    "seagull": ["seagull ATMF/25Δ/25Δ", "seagull 25Δ/10Δ/25Δ", "seagull 25Δ/15Δ/25Δ"],
}


def _curated_keys(variant: dict) -> dict:
    """Compare construction terms, not display labels or catalog metadata.

    `ranked` says whether the catalog offers a construction on the menu, which is not
    part of the construction — an unranked entry must still match the request for it.
    """
    return {k: v for k, v in variant.items()
            if k not in {"label", "can_lose_beyond_premium", "ranked"}}


def test_parity_with_curated_menu():
    """Every curated construction is reachable by a request string, and parses to itself."""
    with open(_VARIANTS_PATH) as f:
        menu = json.load(f)

    for family, requests in _PARITY.items():
        curated = [_curated_keys(v) for v in menu[family]]
        for req_str in requests:
            ours = _dict_no_label(req_str)
            assert ours in curated, f"{family} '{req_str}': {ours} not in {curated}"


def test_every_curated_construction_is_covered():
    """Otherwise a new catalog entry could sit unreachable by any request string."""
    with open(_VARIANTS_PATH) as f:
        menu = json.load(f)

    for family, requests in _PARITY.items():
        parsed = [_dict_no_label(r) for r in requests]
        for variant in menu[family]:
            terms = _curated_keys(variant)
            assert terms in parsed, f"{family} {variant['label']}: no request string covers it"
