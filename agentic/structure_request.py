"""Structure-request grammar + parser (Phase 1 — load-bearing safety boundary).

String in, validated spec out. No LLM, no MarketState, no pricing. Everything the
LLM is allowed to say about a structure passes through here and is either resolved
to a known variant shape or rejected with a structured reason.

The LLM names a trade the way a PM says it ("34Δ vs 25Δ 1x1.5", "1x1 5.00 / 4.80",
"digital 10%"). **A bare number is an absolute strike; a delta must be marked**
(``25Δ``, ``25d``, ``25 delta``) — reading a strike as a delta silently prices a
different trade, so the unmarked reading is the literal one. ``describe_leg_inputs``
states the reading back. Explicit PM-requested 1xR ratios are retained. The LLM
never sets direction, signs or sized notionals —
those are supplied downstream (direction from the session view; weights inside the
pricer; delta legs placed on the smile by the pricer's resolver). ``to_variant_dict`` emits exactly the per-variant
dict shape ``analytics.structure_pricer.price_variants`` already consumes.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass
from typing import Literal, Union

from analytics.strike_resolver import MAX_DELTA, MIN_DELTA
from agentic.family_registry import (
    FAMILIES,
    RATIO_TOKEN,
    POLICY_CURATED,
    POLICY_DISABLED,
    POLICY_FREE,
    resolve_family,
)

LegKind = Literal["delta", "strike", "atmf", "sigma", "target", "premium"]

# Rejection reasons (each maps to a clean agent-facing message).
EMPTY_REQUEST = "EMPTY_REQUEST"
UNKNOWN_FAMILY = "UNKNOWN_FAMILY"
FAMILY_DISABLED = "FAMILY_DISABLED"
WRONG_LEG_COUNT = "WRONG_LEG_COUNT"
BAD_DELTA = "BAD_DELTA"
BAD_STRIKE = "BAD_STRIKE"
BAD_PREMIUM = "BAD_PREMIUM"
BAD_LEG_KIND_FOR_FAMILY = "BAD_LEG_KIND_FOR_FAMILY"
MIXED_ANCHOR = "MIXED_ANCHOR"
BAD_LEG = "BAD_LEG"


@dataclass(frozen=True)
class LegRef:
    kind: LegKind
    value: float | None = None  # delta∈(0,1); strike>0; sigma multiple (signed); premium∈(0,1)

    def render(self) -> str:
        if self.kind == "delta":
            return f"{self.value * 100:g}Δ"
        if self.kind == "strike":
            return f"{self.value:g}"
        if self.kind == "atmf":
            return "ATMF"
        if self.kind == "sigma":
            return f"{self.value:+g}σ"
        if self.kind == "target":
            return "target"
        if self.kind == "premium":
            return f"{int(round(self.value * 100))}%"
        return "?"


@dataclass(frozen=True)
class StructureRequest:
    family: str
    legs: tuple[LegRef, ...]
    canonical: str
    short_ratio: float | None = None


@dataclass(frozen=True)
class ClarificationNeeded:
    question: str


class StructureRequestError(ValueError):
    """Structured rejection carrying a machine reason and a human detail."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


ParseResult = Union[StructureRequest, ClarificationNeeded]

# Direction / noise words parsed but NOT used for construction (is_call comes from
# the session view; cross-checked in Phase 2). Stripped here.
_DIRECTION_WORDS = ("call", "put", "long", "short")

_LEG_SEPARATORS = re.compile(r"\s*(?:\bvs\b|/|,)\s*", re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?\d*\.?\d+")
_DELTA_MARK = re.compile(r"\d\s*d\b")   # "25d" / "25 d" (Δ and "delta" normalize to d)

# Alphabetic tokens that are legitimate residue after the family word and
# direction words are removed: leg markers + the "vs" separator. Anything else
# left over (an unknown structure name, a typo, junk) → clarification, never a
# silent guess.
_ALLOWED_ALPHA = frozenset({"d", "atmf", "atm", "sigma", "target", "tgt", "prem", "vs"})
_ALPHA_RUN = re.compile(r"[a-z]+")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_structure_request(text: str) -> ParseResult:
    """Parse a PM-style structure request into a validated ``StructureRequest``.

    Returns ``ClarificationNeeded`` when the request is genuinely ambiguous (e.g.
    two bare deltas with no family word). Raises ``StructureRequestError`` on a
    malformed/ineligible request.
    """
    if text is None or not text.strip():
        raise StructureRequestError(EMPTY_REQUEST, "empty request")

    norm = _normalize(text)

    # 1. Resolve and remove the family token (so its digits don't parse as legs).
    # Parse the complete ratio before synonym matching (1x1 is a substring of 1x1.3).
    ratio_match = RATIO_TOKEN.search(norm)
    ratio = None
    if ratio_match:
        ratio = float(ratio_match.group(1))
        if not math.isfinite(ratio) or ratio < 1:
            raise StructureRequestError(BAD_LEG, "Spread short ratio must be finite and at least 1")
    family, token = resolve_family(norm)
    remainder = norm.replace(token, " ", 1) if token else norm
    if ratio_match:
        remainder = re.sub(r"\bspread\b", " ", remainder)

    # 2. Strip direction / noise words.
    remainder = _strip_direction_words(remainder)

    # 2b. Any unrecognized alphabetic token (unknown structure name, typo, junk)
    #     → clarify rather than silently ignore it and guess from the legs.
    unknown = _unrecognized_tokens(remainder)
    if unknown:
        joined = ", ".join(f"'{t}'" for t in unknown)
        return ClarificationNeeded(
            f"Unrecognized term(s): {joined}. Use a known family (vanilla, 1x1, 1x1.5, "
            f"1x2, seagull, european_rko, digital) and legs (e.g. a strike 5.10, 25Δ, ATMF, "
            f"½σ, target, 10%)."
        )

    # 3. Parse leg references.
    legs = _parse_legs(remainder)

    # 4. Infer family from leg shape if no family word was present.
    if family is None:
        inferred = _infer_family(legs)
        if isinstance(inferred, ClarificationNeeded):
            return inferred
        family = inferred

    spec = FAMILIES.get(family)
    if spec is None:
        raise StructureRequestError(UNKNOWN_FAMILY, f"unknown family '{family}'")
    if spec.policy == POLICY_DISABLED:
        raise StructureRequestError(FAMILY_DISABLED, f"family '{family}' is disabled")

    # 5. Validate leg count.
    if len(legs) not in spec.arity:
        raise StructureRequestError(
            WRONG_LEG_COUNT,
            f"family '{family}' expects {' or '.join(map(str, spec.arity))} leg(s), got {len(legs)}",
        )

    # 6. Family-specific validation happens in to_variant_dict's builders; build
    #    the canonical echo and the request now (builders re-validate leg kinds).
    name = f"1x{ratio} spread" if family == "ratio_spread" else family
    canonical = f"{name} " + "/".join(leg.render() for leg in legs)
    req = StructureRequest(family=family, legs=tuple(legs), canonical=canonical,
                           short_ratio=ratio if family == "ratio_spread" else None)
    # Eagerly validate by building once (raises on bad leg kinds for the family).
    to_variant_dict(req)
    return req


def to_variant_dict(req: StructureRequest) -> dict:
    variant = _to_variant_dict(req)
    # Keep numeric delta placeholders for compatibility, but preserve anchor intent.
    atmf = [i for i, leg in enumerate(req.legs) if leg.kind == "atmf"]
    if atmf and "long_type" not in variant:
        variant["atmf_legs"] = atmf
    return variant


def _to_variant_dict(req: StructureRequest) -> dict:
    """Turn a parsed request into the synthetic per-variant dict the pricer consumes.

    Output shape mirrors structure_variants.json exactly, plus a ``label`` key.
    """
    family = req.family
    legs = req.legs
    label = req.canonical

    if family == "vanilla":
        return {"label": label, **_leg_entry(legs[0], family, "delta", "strike")}

    if family == "1x1_spread":
        return {
            "label": label,
            **_leg_entry(legs[0], family, "long_delta", "long_strike"),
            **_leg_entry(legs[1], family, "short_delta", "short_strike"),
        }

    if family in ("1x1.5_spread", "1x2_spread"):
        return {"label": label, **_ratio_variant(legs, family)}

    if family == "ratio_spread":
        ratio = req.short_ratio
        if ratio is None or not math.isfinite(ratio) or ratio <= 1:
            raise StructureRequestError(BAD_LEG, "Flexible ratio must be finite and greater than 1")
        return {"label": label, "short_ratio": ratio, **_ratio_variant(legs, family)}

    if family == "seagull":
        return {
            "label": label,
            **_leg_entry(legs[0], family, "spread_long", "spread_long_strike"),
            **_leg_entry(legs[1], family, "spread_short", "spread_short_strike"),
            **_leg_entry(legs[2], family, "wing_delta", "wing_strike"),
        }

    if family == "european_rko":
        return {
            "label": label,
            **_leg_entry(legs[0], family, "long_delta", "long_strike"),
            **_leg_entry(legs[1], family, "barrier", "barrier_strike"),
        }

    if family == "european_digital":
        return {"label": label, "target_prem_pct": _require_premium(legs[0], family)}

    # Disabled families never reach here (rejected in parse).
    raise StructureRequestError(UNKNOWN_FAMILY, f"no builder for family '{family}'")


# ---------------------------------------------------------------------------
# Normalization & tokenizing
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    t = text.strip().lower()
    t = t.replace("½", "0.5")          # ½ → 0.5
    t = t.replace("half sigma", "0.5sigma").replace("half-sigma", "0.5sigma")
    t = t.replace("δ", "d").replace("delta", "d")  # δ (lowercased Δ) / "delta" → d marker
    t = t.replace("σ", "sigma")        # σ (lowercased) → sigma
    t = re.sub(r"\s+", " ", t)
    return t


def _strip_direction_words(text: str) -> str:
    for w in _DIRECTION_WORDS:
        text = re.sub(rf"\b{w}\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _unrecognized_tokens(remainder: str) -> list[str]:
    """Alphabetic runs left in the remainder that are not legitimate leg markers.

    Preserves order, de-duplicates. Empty list → the remainder is clean.
    """
    seen: list[str] = []
    for tok in _ALPHA_RUN.findall(remainder):
        if tok not in _ALLOWED_ALPHA and tok not in seen:
            seen.append(tok)
    return seen


def _parse_legs(text: str) -> list[LegRef]:
    chunks = [c.strip() for c in _LEG_SEPARATORS.split(text) if c.strip()]
    return [_parse_leg(c) for c in chunks]


def _parse_leg(chunk: str) -> LegRef:
    c = chunk.strip()

    # premium: contains '%'
    if "%" in c or "prem" in c:
        m = _NUMBER.search(c)
        if not m:
            raise StructureRequestError(BAD_PREMIUM, f"no premium number in '{chunk}'")
        val = float(m.group()) / 100.0
        if not (0.0 < val < 1.0):
            raise StructureRequestError(BAD_PREMIUM, f"premium out of range in '{chunk}'")
        return LegRef("premium", val)

    # target
    if "target" in c or re.fullmatch(r"tgt", c):
        return LegRef("target")

    # atmf
    if "atmf" in c or "atm" in c:
        return LegRef("atmf")

    # sigma
    if "sigma" in c:
        m = _NUMBER.search(c)
        val = float(m.group()) if m else 1.0
        return LegRef("sigma", val)

    m = _NUMBER.search(c)

    # delta: a number explicitly marked Δ / d / delta (normalized to a 'd' marker)
    if m and _DELTA_MARK.search(c):
        raw = float(m.group())
        delta = raw / 100.0 if raw > 1.0 else raw
        if not (MIN_DELTA <= delta <= MAX_DELTA):
            raise StructureRequestError(
                BAD_DELTA,
                f"{raw:g}Δ in '{chunk}' is outside the supported "
                f"{MIN_DELTA * 100:g}Δ–{MAX_DELTA * 100:g}Δ range",
            )
        return LegRef("delta", delta)

    # strike: any other number is an absolute strike level — never guessed as a delta
    if m:
        strike = float(m.group())
        if strike <= 0.0:
            raise StructureRequestError(BAD_STRIKE, f"strike must be positive in '{chunk}'")
        return LegRef("strike", strike)

    raise StructureRequestError(BAD_LEG, f"could not parse leg '{chunk}'")


def _infer_family(legs: list[LegRef]) -> Union[str, ClarificationNeeded]:
    kinds = [leg.kind for leg in legs]
    n = len(legs)
    if n == 1:
        if kinds[0] == "premium":
            return "european_digital"
        if kinds[0] in ("delta", "strike", "atmf"):
            return "vanilla"
        raise StructureRequestError(
            BAD_LEG_KIND_FOR_FAMILY, f"cannot infer a 1-leg family from a {kinds[0]} leg"
        )
    if n == 2:
        # 1x1 / 1x1.5 / 1x2 all share a 2-leg delta shape — genuinely ambiguous.
        return ClarificationNeeded("1x1, 1x1.5 or 1x2 spread?")
    if n == 3:
        return "seagull"
    raise StructureRequestError(
        WRONG_LEG_COUNT, f"no family takes {n} legs without an explicit name"
    )


# ---------------------------------------------------------------------------
# Leg-kind validators (used by the builders)
# ---------------------------------------------------------------------------

def _require_delta(leg: LegRef, family: str) -> float:
    if leg.kind == "atmf":
        return 0.50
    if leg.kind != "delta":
        raise StructureRequestError(
            BAD_LEG_KIND_FOR_FAMILY,
            f"family '{family}' needs a strike, delta or ATMF leg, got {leg.kind}",
        )
    return leg.value


def _leg_entry(leg: LegRef, family: str, delta_key: str, strike_key: str) -> dict:
    """One leg as variant-dict entries: an absolute strike under ``strike_key``, else a
    delta placeholder under ``delta_key`` (ATMF identity is retained separately)."""
    if leg.kind == "strike":
        return {strike_key: leg.value}
    return {delta_key: _require_delta(leg, family)}


def describe_leg_inputs(req: StructureRequest) -> str | None:
    """How each numeric leg was read (strike or delta), for the reply — so the PM sees
    the assumption rather than having to trust it. None when no leg is a number."""
    parts = [
        f"{leg.render()} = {'strike' if leg.kind == 'strike' else 'delta'}"
        for leg in req.legs if leg.kind in ("strike", "delta")
    ]
    if not parts:
        return None
    return ("LEG INPUTS: " + "; ".join(parts)
            + " (numbers are read as strikes unless marked Δ/delta).")


def _require_premium(leg: LegRef, family: str) -> float:
    if leg.kind != "premium":
        raise StructureRequestError(
            BAD_LEG_KIND_FOR_FAMILY,
            f"family '{family}' is curated — only a premium=N% leg is allowed, got {leg.kind}",
        )
    return leg.value


def _ratio_variant(legs: tuple[LegRef, ...], family: str) -> dict:
    """Build a 1x1.5 / 1x2 variant: either a delta-pair or an anchored→target form."""
    a, b = legs

    # Leg-pair form: each leg a strike, delta or ATMF
    pair_kinds = ("delta", "strike", "atmf")
    if a.kind in pair_kinds and b.kind in pair_kinds:
        return {
            **_leg_entry(a, family, "long_delta", "long_strike"),
            **_leg_entry(b, family, "short_delta", "short_strike"),
        }

    # Anchored→target form: long anchor (ATMF or ½σ) vs target.
    if b.kind == "target":
        if a.kind == "atmf":
            return {"long_type": "atmf"}
        if a.kind == "sigma" and abs(a.value) == 0.5:
            return {"long_type": "half_sigma", "min_target_z": 0.5}
        raise StructureRequestError(
            BAD_LEG_KIND_FOR_FAMILY,
            f"family '{family}' anchor leg must be ATMF or ½σ, got {a.render()}",
        )

    raise StructureRequestError(
        MIXED_ANCHOR,
        f"family '{family}' legs must be two strikes/deltas or anchor/target, got {a.kind}/{b.kind}",
    )
