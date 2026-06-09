"""Phase 2 — the ``price_structure`` tool (Tier 2, fine-grained).

Bridges a PM-style structure request to a concrete priced variant, against the
*current* (frozen) market state. The agent supplies only the request string; the
direction (``is_call``), target, loss budget, and vol surface come from the live
session — never from the LLM. This is the Tier-2 tool in the two-tier surface:
it prices one payoff and never rebuilds the market state.

Flow: parse (Phase 1 grammar) → ``to_variant_dict`` → ``price_variants`` with a
single-variant override → structured result. Outcomes are returned as tagged
values (no raw exceptions leak to the agent for the expected cases):

  - ``PricedStructure``    — success; carries the request + the PricedVariant.
  - ``ClarificationNeeded``— request was ambiguous/unrecognized (from the parser).
  - ``PricingUnavailable`` — parsed fine but couldn't be priced for this view
                             (e.g. needs a target, target too close, or dropped by
                             the smile-arbitrage guard).

``StructureRequestError`` (a hard malformed request — bad delta/premium, wrong leg
count, disabled family) is still raised, mirroring the parser; the Phase 3 tool
adapter converts it to a tool-result message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from analytics.market_state import MarketState
from analytics.structure_pricer import PricedVariant, price_variants
from agentic.structure_request import (
    ClarificationNeeded,
    StructureRequest,
    parse_structure_request,
    to_variant_dict,
)

_UNSET = object()


@dataclass(frozen=True)
class PricedStructure:
    request: StructureRequest
    variant: PricedVariant
    warnings: tuple[str, ...] = ()
    # Product-model PricedStructure (analytics.product_model) — legs first-class, for the
    # explicit per-leg render. Metrics/ccy still come from `variant` (sized to loss_budget).
    priced_structure: object | None = None


@dataclass(frozen=True)
class PricingUnavailable:
    request: StructureRequest
    detail: str
    warnings: tuple[str, ...] = ()


PriceStructureResult = Union[PricedStructure, ClarificationNeeded, PricingUnavailable]


def price_structure(
    request_text: str,
    ms: MarketState,
    *,
    is_call: bool,
    target: float | None = None,
    stop_price: float | None = None,
    loss_budget: float | None = None,
    smile: object = _UNSET,
) -> PriceStructureResult:
    """Price a single PM-requested structure against the current market state.

    Args:
        request_text: PM-style request, e.g. "34 vs 25 1x1.5", "digital 10%".
        ms:           the current (frozen) MarketState from the standard pack.
        is_call:      direction, from the session view (base_higher → call). NOT
                      from the LLM.
        target:       target spot level, from the session view (Tier-1 input).
        stop_price:   optional stop level (seagull max-loss sizing).
        loss_budget:  optional base-ccy loss budget → populates the ccy fields.
        smile:        VolSurface; defaults to ``ms.surface`` so entry pricing uses
                      the same surface the pack was built against.
    """
    parsed = parse_structure_request(request_text)
    if isinstance(parsed, ClarificationNeeded):
        return parsed

    variant_dict = to_variant_dict(parsed)
    surface = getattr(ms, "surface", None) if smile is _UNSET else smile

    warnings: list[str] = []
    priced = price_variants(
        ms,
        parsed.family,
        target=target,
        is_call=is_call,
        stop_price=stop_price,
        loss_budget=loss_budget,
        smile=surface,
        warnings=warnings,
        variants_override=[variant_dict],
    )

    if not priced:
        return PricingUnavailable(
            request=parsed,
            detail=_unavailable_detail(parsed.family, target, warnings),
            warnings=tuple(warnings),
        )

    # Product-model structure for the explicit per-leg breakdown (legs first-class).
    # Best-effort; metrics/ccy still come from `variant`. stop_price omitted — only the
    # legs are consumed here, not the product-model max-loss.
    legs_struct = None
    try:
        from analytics.product_pricer import build_structure, price as _price_product
        st = build_structure(parsed.family, variant_dict, is_call)
        if st is not None:
            legs_struct = _price_product(st, ms, target=target, smile=surface)
    except Exception:
        legs_struct = None

    return PricedStructure(
        request=parsed, variant=priced[0], warnings=tuple(warnings),
        priced_structure=legs_struct,
    )


def _unavailable_detail(family: str, target: float | None, warnings: list[str]) -> str:
    if warnings:
        return "; ".join(warnings)
    if target is None and family in ("european_rko", "1x1.5_spread", "1x2_spread"):
        return f"'{family}' needs a target level to be priced for this view"
    return f"'{family}' could not be priced for the current view (target may be too close to the forward)"
