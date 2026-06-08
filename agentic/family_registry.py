"""Closed family registry for the structure-request grammar.

This is the *whole* allowed surface the LLM can name. A family token is resolved
here (via synonyms) or the request is rejected. Each family declares its leg
arity and a construction policy:

  - "free"     arbitrary delta legs accepted (strictly more flexible than the
               curated structure_variants.json menu — the pricer handles any
               delta pair / triple).
  - "curated"  only a premium=N% leg accepted (the digital bisects on premium and
               carries the smile-arb guard; arbitrary strikes/barriers are not
               exposed to the LLM).
  - "disabled" mirrors enabled:false in structure_profiles.json → always rejected.

Phase 1 is pure: this module has no dependency on analytics / conversation / any
LLM. It does not price anything; it only validates and shapes a request.
"""

from __future__ import annotations

from dataclasses import dataclass


# Construction policy per family.
POLICY_FREE = "free"
POLICY_CURATED = "curated"
POLICY_DISABLED = "disabled"


@dataclass(frozen=True)
class FamilySpec:
    family: str           # canonical family id (matches structure_profiles.json keys)
    policy: str           # POLICY_FREE | POLICY_CURATED | POLICY_DISABLED
    arity: tuple[int, ...]  # allowed leg counts


# Canonical family table. Disabled families are still listed so they resolve to a
# clean FAMILY_DISABLED rejection rather than UNKNOWN_FAMILY.
FAMILIES: dict[str, FamilySpec] = {
    "vanilla":              FamilySpec("vanilla",              POLICY_FREE,     (1,)),
    "1x1_spread":           FamilySpec("1x1_spread",           POLICY_FREE,     (2,)),
    "1x1.5_spread":         FamilySpec("1x1.5_spread",         POLICY_FREE,     (2,)),
    "1x2_spread":           FamilySpec("1x2_spread",           POLICY_FREE,     (2,)),
    "seagull":              FamilySpec("seagull",              POLICY_FREE,     (3,)),
    "european_rko":         FamilySpec("european_rko",         POLICY_FREE,     (2,)),
    "european_digital":     FamilySpec("european_digital",     POLICY_CURATED,  (1,)),
    "european_digital_rko": FamilySpec("european_digital_rko", POLICY_DISABLED, (1,)),
    "rko":                  FamilySpec("rko",                  POLICY_DISABLED, (1,)),
}


# Synonym → family, in PRIORITY order (first containment wins). Ordering matters:
# more specific multi-word tokens must precede the tokens they contain
# ("digital rko" before "digital" and "rko"; "european rko"/"erko" before "rko").
_SYNONYMS: list[tuple[str, str]] = [
    # disabled, most specific first
    ("european digital rko", "european_digital_rko"),
    ("digital with rko",     "european_digital_rko"),
    ("digital rko",          "european_digital_rko"),
    # digital (binary)
    ("european digital",     "european_digital"),
    ("digital",              "european_digital"),
    ("binary",               "european_digital"),
    # european reverse knock-out (must precede bare "rko")
    ("european reverse knock-out", "european_rko"),
    ("european reverse knockout",  "european_rko"),
    ("european rko",         "european_rko"),
    ("ereko",                "european_rko"),
    ("erko",                 "european_rko"),
    # path-dependent rko (disabled)
    ("reverse knock-out",    "rko"),
    ("reverse knockout",     "rko"),
    ("rko",                  "rko"),
    # seagull
    ("seagull",              "seagull"),
    # ratio spreads — 1x1.5 / 1x2 before 1x1 (1x1 is a substring of 1x1.5)
    ("1x1.5 spread",         "1x1.5_spread"),
    ("1 x 1.5",              "1x1.5_spread"),
    ("1x1.5",                "1x1.5_spread"),
    ("1.5 spread",           "1x1.5_spread"),
    ("1x2 spread",           "1x2_spread"),
    ("1 x 2",                "1x2_spread"),
    ("1x2",                  "1x2_spread"),
    ("1x1 spread",           "1x1_spread"),
    ("1 x 1",                "1x1_spread"),
    ("1x1",                  "1x1_spread"),
    ("call spread",          "1x1_spread"),
    ("put spread",           "1x1_spread"),
    # generic "spread" → 1x1 (least specific, last)
    ("spread",               "1x1_spread"),
    # vanilla
    ("vanilla",              "vanilla"),
    ("plain option",         "vanilla"),
]


def resolve_family(text: str) -> tuple[str | None, str | None]:
    """Find the first family synonym contained in ``text`` (already lowercased).

    Returns ``(family, matched_token)`` or ``(None, None)`` if no family word is
    present (the caller may then try to infer a family from the leg shape).
    """
    for token, family in _SYNONYMS:
        if token in text:
            return family, token
    return None, None
