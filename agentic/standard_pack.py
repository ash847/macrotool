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


def _major_risk(structure_id: str) -> str:
    """The engine's canonical primary-risk text for a structure (from profiles).

    Used so the LLM relays the correct risk instead of inventing payoff geometry.
    """
    try:
        from knowledge_engine.loader import load_structure_profiles
        return load_structure_profiles().get(structure_id, {}).get("major_risk", "") or ""
    except Exception:
        return ""


def _priced_structure_for(family, label, ms, is_call, target, surface, stop_price):
    """The product-model PricedStructure (legs first-class) for a recommended variant,
    looked up by family+label. Used so the renderer can show explicit legs (deltas /
    notionals / call-put) instead of a flat strike list — killing the fabrication class.
    Best-effort: None if it can't be built."""
    try:
        from analytics.product_pricer import build_structure, price
        from analytics.structure_pricer import _load_variants
        vd = next((v for v in _load_variants().get(family, []) if v.get("label") == label), None)
        if vd is None:
            return None
        st = build_structure(family, vd, is_call)
        if st is None:
            return None
        return price(st, ms, target=target, smile=surface, stop_price=stop_price)
    except Exception:
        return None


@dataclass
class RecommendedStructure:
    """A specific, priced representative construction for a shortlisted family."""

    structure_id: str
    display_name: str
    rank: int
    rationale: str
    variant: PricedVariant
    score_ccy: float | None = None   # scenario-weighted P&L the variant was ranked on
    major_risk: str = ""             # engine's canonical primary risk (from profiles)
    priced_structure: object | None = None   # product-model PricedStructure (legs first-class)
    drivers: dict | None = None      # P&L driver split (Carry/Directional/Adverse/Vega), pct
                                     # — kept server-side for derivation; NOT rendered to the LLM
    attributes: frozenset = frozenset()   # IP-clean qualitative tags (the LLM-facing findings)


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
    loss_budget: float | None = None          # base-ccy max-loss budget the trades are sized to
                                              # (= LINEAR_NOTIONAL × stop%, R:R-derived)
    active_context: str | None = None         # fired scenario-weighting context id
    deciding_axis: str | None = None          # IP-clean gloss: what separated #1 from #2
    scenario_weights: dict | None = None      # resolved PM weights (Tier-2 reuses to score a PM structure)


_COMPARATOR_LINEAR_NOTIONAL = 100.0


def _recommend_ranked(
    ms: MarketState, selector_result, target, is_call, surface,
    primary_objective, trade_management, structure_constraint, target_rr,
    user_email=None,
) -> list[RecommendedStructure]:
    """Per shortlisted family, pick the variant the scoring ranks best.

    Uses the comparator's scenario evaluation (pm_score.score_ccy — PM-overlay
    weighted P&L in base ccy, the same metric the Trade View variants table and
    Kelly's Trade-Rec ranking use). Requires a target (the scenario grid is target-
    centred); callers fall back to the first-curated representative without one.
    """
    from knowledge_engine.comparator import PMPreferences, build_comparator_inputs

    fwd = ms.fwd
    move_pct = abs(target - fwd) / fwd if fwd else 0.0
    if move_pct <= 0:
        raise ValueError("no usable move for comparator")
    stop_pct = move_pct / max(target_rr, 1e-6)
    stop_price = fwd * (1 - stop_pct) if is_call else fwd * (1 + stop_pct)
    # Loss budget computed on the fly, identical to the Trade View screen:
    # LINEAR_NOTIONAL × stop% (stop% = move/R:R). Each variant is sized so its max
    # loss equals this; the displayed notionals are on this standard 100-unit basis.
    loss_budget = _COMPARATOR_LINEAR_NOTIONAL * stop_pct

    inputs = build_comparator_inputs(
        ms, selector_result,
        target=target, is_call=is_call,
        stop_price=stop_price, loss_budget=loss_budget,
        linear_notional=_COMPARATOR_LINEAR_NOTIONAL,
        preferences=PMPreferences(
            primary_objective=primary_objective,
            trade_management=trade_management,
            structure_constraint=structure_constraint,
        ),
        smile=surface,
        user_email=user_email,
    )

    from knowledge_engine.scenario_scorer import driver_contribs
    from knowledge_engine.structure_attributes import attributes as _attributes, deciding_axis as _deciding_axis

    out: list[RecommendedStructure] = []
    aggs_by_rank: list[tuple[int, object]] = []   # (rank, aggregates) for the deciding-axis pick
    for item in selector_result.shortlist:
        evals = inputs.variant_evaluations_by_structure.get(item.structure_id) or []
        scored = [e for e in evals if e.pm_score.score_ccy is not None]
        best = (
            max(scored, key=lambda e: e.pm_score.score_ccy) if scored
            else (evals[0] if evals else None)
        )
        if best is not None:
            out.append(RecommendedStructure(
                structure_id=item.structure_id,
                display_name=item.display_name,
                rank=item.rank,
                rationale=item.rationale,
                variant=best.variant,
                score_ccy=best.pm_score.score_ccy,
                major_risk=_major_risk(item.structure_id),
                priced_structure=_priced_structure_for(
                    item.structure_id, best.variant.variant_label, ms, is_call,
                    target, surface, stop_price,
                ),
                drivers=driver_contribs(best.pm_score),
                attributes=_attributes(item.structure_id, best.pm_score, best.aggregates),
            ))
            aggs_by_rank.append((item.rank, best.aggregates))

    # Deciding axis: the single scenario bucket that most separates the top-ranked
    # recommendation from the runner-up (IP-clean gloss; numbers stay server-side).
    deciding = None
    aggs_by_rank.sort(key=lambda x: x[0])
    if len(aggs_by_rank) >= 2:
        deciding = _deciding_axis(aggs_by_rank[0][1], aggs_by_rank[1][1])

    return out, loss_budget, inputs.active_context, deciding, inputs.weights


def _price_recommended_fallback(
    ms: MarketState, selector_result, target, is_call, surface, cap: int = 6
) -> list[RecommendedStructure]:
    """Fallback when no target: first curated variant that prices, per family."""
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
                major_risk=_major_risk(item.structure_id),
                priced_structure=_priced_structure_for(
                    item.structure_id, rep.variant_label, ms, is_call, target, surface, None,
                ),
            ))
    return out


def build_pack(
    view: TradeView,
    ccy,                                # CurrencySnapshot
    cfg,                                # ResolvedConfig
    structure_constraint: str = "No restriction",
    primary_objective: str = "Balanced",
    trade_management: str = "Standard hold",
    target_rr: float = 3.0,
    user_email: str | None = None,
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
    recommended: list[RecommendedStructure] = []
    loss_budget: float | None = None
    active_context: str | None = None
    deciding_axis: str | None = None
    scenario_weights: dict | None = None
    if target is not None:
        try:
            recommended, loss_budget, active_context, deciding_axis, scenario_weights = _recommend_ranked(
                market_state, selector_result, target, is_call, surface,
                primary_objective, trade_management, structure_constraint, target_rr,
                user_email=user_email,
            )
        except Exception:
            recommended, loss_budget, active_context, deciding_axis, scenario_weights = [], None, None, None, None
    if not recommended:
        recommended = _price_recommended_fallback(
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
        loss_budget=loss_budget,
        active_context=active_context,
        deciding_axis=deciding_axis,
        scenario_weights=scenario_weights,
    )
