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

import math
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
    linear_notional: float = 100.0,
    smile: object = _UNSET,
    sizing_spec: object = None,
    structure_constraint: str = "No restriction",
) -> PriceStructureResult:
    """Price a single PM-requested structure against the current market state.

    Args:
        request_text: PM-style request, e.g. "34Δ vs 25Δ 1x1.5", "1x1 5.00 / 4.80" (strikes), "digital 10%".
        ms:           the current (frozen) MarketState from the standard pack.
        is_call:      direction, from the session view (base_higher → call). NOT
                      from the LLM.
        target:       target spot level, from the session view (Tier-1 input).
        stop_price:   optional stop level (seagull max-loss sizing).
        loss_budget:  optional base-ccy loss budget → populates the ccy fields.
        linear_notional: the PM's real sizing capital (W) — bounds the notional cap
                      (10×W) applied in ``_size_variant``. Must match the session's
                      bankroll or the cap clips the sized notional to a placeholder
                      value regardless of loss_budget.
        smile:        VolSurface; defaults to ``ms.surface`` so entry pricing uses
                      the same surface the pack was built against.
        sizing_spec:  the pack's SizingSpec (Kelly) so a PM-named structure is sized
                      under the same regime as the recommendations; None → fixed-loss.
    """
    parsed = parse_structure_request(request_text)
    if isinstance(parsed, ClarificationNeeded):
        return parsed

    variant_dict = to_variant_dict(parsed)
    from knowledge_engine.construction_policy import configured_additional_loss

    classification = configured_additional_loss(parsed.family, variant_dict)
    variant_dict["can_lose_beyond_premium"] = classification
    if structure_constraint == "Avoid capped structures" and parsed.family != "vanilla":
        return PricingUnavailable(
            request=parsed,
            detail="Excluded by the active vanilla-only preference.",
        )
    implausible = _implausible_strike(parsed, ms)
    if implausible is not None:
        return PricingUnavailable(request=parsed, detail=implausible)
    surface = getattr(ms, "surface", None) if smile is _UNSET else smile

    warnings: list[str] = []
    priced = price_variants(
        ms,
        parsed.family,
        target=target,
        is_call=is_call,
        stop_price=stop_price,
        loss_budget=loss_budget,
        linear_notional=linear_notional,
        smile=surface,
        warnings=warnings,
        variants_override=[variant_dict],
        sizing_spec=sizing_spec,
        structure_constraint=structure_constraint,
    )

    if not priced:
        return PricingUnavailable(
            request=parsed,
            detail=_unavailable_detail(parsed.family, target, warnings),
            warnings=tuple(warnings),
        )
    reversed_legs = _reversed_spread(parsed.family, priced[0].strikes, is_call)
    if reversed_legs is not None:
        return PricingUnavailable(request=parsed, detail=reversed_legs)

    # Migrated families retain the exact entry product; no second pricing pass.
    # Unmigrated families keep the best-effort leg-display bridge for now.
    legs_struct = getattr(priced[0], "product_price", None)
    try:
        if legs_struct is None:
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


def characterize_against_pack(
    variant,
    family: str,
    ms: MarketState,
    *,
    is_call: bool,
    target: float | None,
    smile: object,
    weights: dict | None,
) -> frozenset:
    """IP-clean qualitative tags for a PM-named structure, scored against the
    *frozen* pack (same scenario grid + same resolved weights). Mirrors the
    pack-build path so an off-menu structure lands in the identical vocabulary.

    Best-effort: returns an empty set if the structure can't be scenario-scored
    (e.g. no target, or no resolved weights on the pack).
    """
    if target is None or not weights:
        return frozenset()
    try:
        from analytics.scenario_generator import generate_scenarios
        from analytics.scenario_pricer import price_scenarios
        from knowledge_engine.scenario_scorer import score_structure
        from knowledge_engine.comparator import summarize_scenario_rows
        from knowledge_engine.structure_attributes import attributes

        trade_inputs = {
            "spot": ms.spot, "forward": ms.fwd, "implied_vol": ms.vol,
            "tenor_years": ms.T, "target": target, "r_d": ms.r_d, "r_f": ms.r_f,
        }
        scenarios = generate_scenarios(trade_inputs)
        rows = price_scenarios(variant, family, scenarios, trade_inputs, is_call, surface=smile)
        pm_score = score_structure(rows, weights)
        return attributes(family, pm_score, summarize_scenario_rows(rows))
    except Exception:
        return frozenset()


# A strike further than this many standard deviations (σ√T) from the forward is not a
# plausible level — most likely a delta written without its Δ. Refuse; never reinterpret.
_MAX_STRIKE_SIGMAS = 5.0
_SPREAD_FAMILIES = ("1x1_spread", "1x1.5_spread", "1x2_spread", "ratio_spread", "seagull")


def _implausible_strike(parsed: StructureRequest, ms: MarketState) -> str | None:
    sd = ms.vol * math.sqrt(ms.T)
    for leg in parsed.legs:
        if leg.kind != "strike" or sd <= 0:
            continue
        n = abs(math.log(leg.value / ms.fwd)) / sd
        if n > _MAX_STRIKE_SIGMAS:
            return (f"Strike {leg.value:g} is {n:.0f} standard deviations from the forward "
                    f"{ms.fwd:.6g}, so it was not priced. Numbers are read as strikes; if "
                    f"{leg.value:g} was meant as a delta, write it as {leg.value:g}Δ.")
    return None


def _reversed_spread(family: str, strikes: list[float], is_call: bool) -> str | None:
    """The long leg must sit nearer the money than the short leg (in the view's
    direction); otherwise the request describes the opposite spread."""
    if family not in _SPREAD_FAMILIES or len(strikes) < 2:
        return None
    k_long, k_short = strikes[0], strikes[1]
    if (k_long < k_short) if is_call else (k_long > k_short):
        return None
    side = "above" if not is_call else "below"
    return (f"For this {'call' if is_call else 'put'} spread the long strike must be {side} "
            f"the short strike, but the request gives long {k_long:.6g} and short "
            f"{k_short:.6g}, so it was not priced. Legs are read in order: long, then short.")


def _unavailable_detail(family: str, target: float | None, warnings: list[str]) -> str:
    if warnings:
        return "; ".join(warnings)
    if target is None and family in ("european_rko", "1x1.5_spread", "1x2_spread", "ratio_spread"):
        return f"'{family}' needs a target level to be priced for this view"
    return f"'{family}' could not be priced for the current view (target may be too close to the forward)"
