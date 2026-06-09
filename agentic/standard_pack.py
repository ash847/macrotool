"""The deterministic "standard pack" — the Tier-1 engine chain, factored out of
``conversation.flow.ConversationFlow._run_engines`` so the agent loop and the
legacy state machine share one implementation (no divergence).

``build_pack`` runs the full chain for a view: market state → structure scorer →
sizing → price distributions. It is pure orchestration over the existing engines;
it computes no numbers itself. The agent's Tier-1 ``run_standard_pack`` tool calls
this; ``flow._run_engines`` delegates its core chain to it too.
"""

from __future__ import annotations

from dataclasses import dataclass

from analytics.market_state import MarketState
from analytics.models import MaturityHistogram, PriceDistribution
from analytics.structure_pricer import PricedVariant, price_variants
from knowledge_engine.models import SizingOutput, StructureSelectionResult, TradeView


def target_from_reference(
    reference: float, direction: str, magnitude_pct: float | None
) -> float | None:
    """Target spot level from a reference (forward or spot) + signed magnitude.

    Lives here (not in ``conversation.flow``) so the shared pack builder can use it
    without importing the flow module; ``flow`` re-exports it for back-compat.
    """
    if magnitude_pct is None:
        return None
    sign = 1 if direction == "base_higher" else -1
    return reference * (1 + sign * magnitude_pct / 100)


@dataclass
class RecommendedStructure:
    """A specific, priced representative construction for a shortlisted family."""

    structure_id: str
    display_name: str
    rank: int
    rationale: str
    variant: PricedVariant


@dataclass
class StandardPack:
    """The deterministic engine output for one view — the agent's Tier-1 result."""

    market_state: MarketState
    selector_result: StructureSelectionResult
    sizing: SizingOutput | None
    flat_distribution: PriceDistribution | None
    smile_distribution: PriceDistribution | None
    maturity_histogram: MaturityHistogram | None
    target: float | None        # target spot (from forward + magnitude), or None
    is_call: bool               # direction → call/put, for downstream Tier-2 pricing
    recommended: list[RecommendedStructure]   # specific priced structures per family


def _price_recommended(
    ms: MarketState, selector_result, target, is_call, surface, cap: int = 6
) -> list[RecommendedStructure]:
    """Price a representative variant for each shortlisted family.

    Picks the first curated variant that prices (the curated menu lists the most
    canonical construction first — e.g. ATMF-vs-target for ratio spreads). Gives
    the agent concrete structures to recommend, not just family names.
    """
    out: list[RecommendedStructure] = []
    for item in selector_result.shortlist[:cap]:
        try:
            variants = price_variants(
                ms, item.structure_id, target=target, is_call=is_call,
                smile=surface, warnings=[],
            )
        except Exception:
            variants = []
        rep = next((v for v in variants if v.net_premium_pct is not None), None)
        if rep is not None:
            out.append(RecommendedStructure(
                structure_id=item.structure_id,
                display_name=item.display_name,
                rank=item.rank,
                rationale=item.rationale,
                variant=rep,
            ))
    return out


def build_pack(
    view: TradeView,
    ccy,                                # CurrencySnapshot
    cfg,                                # ResolvedConfig
    structure_constraint: str = "No restriction",
) -> StandardPack:
    """Run the full deterministic chain for a view. Pure orchestration.

    Mirrors ``flow._run_engines``' core: identical inputs → identical engine calls.
    Distributions are best-effort (non-fatal), matching the flow's contract.
    """
    from pricing.forwards import rate_context_for_snapshot
    from knowledge_engine.loader import load_affinity_scores
    from analytics.distributions import (
        interpolate_atm_vol,
        compute_flat_vol_distribution,
        compute_smile_distribution,
        compute_maturity_histogram,
    )
    from analytics.market_state import compute_market_state
    from knowledge_engine.structure_scorer import score_structures
    from knowledge_engine.sizing_engine import compute_sizing

    T = view.horizon_years
    rate_ctx = rate_context_for_snapshot(ccy, T)
    atm_vol = interpolate_atm_vol(ccy, view.horizon_days)
    target = target_from_reference(rate_ctx.forward, view.direction, view.magnitude_pct)
    carry_regime_cuts = load_affinity_scores()["thresholds"]["carry_regime"]

    # Build the vol surface once; reuse everywhere a vanilla is priced. Falls back
    # to flat ATM vol on a thin/incomplete surface so the pipeline never breaks.
    surface = None
    try:
        from analytics.vol_surface import build_vol_surface
        surface = build_vol_surface(ccy)
    except Exception:
        surface = None

    market_state = compute_market_state(
        spot=rate_ctx.spot,
        fwd=rate_ctx.forward,
        vol=atm_vol,
        T=T,
        r_d=rate_ctx.r_d,
        r_f=rate_ctx.r_f,
        target=target,
        direction=view.direction,
        carry_regime_cuts=carry_regime_cuts,
        surface=surface,
    )

    selector_result = score_structures(
        market_state,
        structure_constraint=structure_constraint,
    )

    top = selector_result.shortlist[0] if selector_result.shortlist else None
    sizing = compute_sizing(view, ccy, top, cfg) if top else None

    flat_distribution = smile_distribution = maturity_histogram = None
    try:
        flat_distribution = compute_flat_vol_distribution(ccy, view.horizon_days)
        smile_distribution = compute_smile_distribution(ccy, view.horizon_days)
        maturity_histogram = compute_maturity_histogram(
            flat_distribution, smile_distribution
        )
    except Exception:
        pass  # distributions are enrichment; never break the pack

    is_call = view.direction == "base_higher"
    recommended = _price_recommended(
        market_state, selector_result, target, is_call, surface
    )

    return StandardPack(
        market_state=market_state,
        selector_result=selector_result,
        sizing=sizing,
        flat_distribution=flat_distribution,
        smile_distribution=smile_distribution,
        maturity_histogram=maturity_histogram,
        target=target,
        is_call=is_call,
        recommended=recommended,
    )
