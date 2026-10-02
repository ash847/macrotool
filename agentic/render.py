"""Labelled text renderers for the agent's tool results.

The agent narrates over these strings — it never sees raw objects. Renderers
label "baseline (from the standard pack)" vs a PM-requested structure so the
agent can cite the right source, and they surface only computed numbers.
"""

from __future__ import annotations

import re

from agentic.price_structure import PricedStructure, PricingUnavailable
from agentic.standard_pack import StandardPack
from agentic.shortlist import render_shortlist, shortlist_reference
from analytics.product_model import AnchorKind
from knowledge_engine.models import TradeView
from knowledge_engine.payoff_risk import payoff_risk_note
from knowledge_engine.scenario_scorer import cell_label

_TOP_N = 5   # recommended structures shown by default; rest surfaced only on request

# The engine rationale carries a trailing "[scores on: <affinity dimensions>]" /
# "[penalised by: ...]" suffix — that is scoring METHODOLOGY (IP). Strip it before the
# LLM sees it; keep only the plain description. The full version stays in the admin UI.
_RATIONALE_METHOD_SUFFIX = re.compile(r"\s*\[(?:scores on|penalised by)[^\]]*\]")


def _clean_rationale(text: str) -> str:
    return _RATIONALE_METHOD_SUFFIX.sub("", text or "").strip()


def _anchor_label(anchor) -> str:
    k = anchor.kind
    if k == AnchorKind.DELTA:
        return f"{round(anchor.value * 100)}Δ"
    if k == AnchorKind.ATMF:
        return "ATMF"
    if k == AnchorKind.HALF_SIGMA:
        return "½σ"
    if k == AnchorKind.TARGET:
        return "target"
    if k == AnchorKind.PREMIUM:
        return f"{round(anchor.value * 100)}% prem"
    return f"K={anchor.value:.4f}"


def _legs_breakdown(ps, base_notional: float | None = None, ccy: str = "") -> list[str]:
    """Explicit per-leg lines from a product-model PricedStructure — each leg's side /
    anchor / right / strike + the ACTUAL sized notional (leg ratio × the structure's sized
    base notional). The leg ratio (1, 1.5, …) is the structure name, not the notional."""
    if ps is None:
        return []
    lines = []
    for index, pl in enumerate(ps.priced_legs):
        side = "long" if pl.notional > 0 else "short"
        right = pl.leg.right.value.capitalize()
        instr = "Digital " if pl.leg.instrument.value == "digital" else ""
        label = "wing" if ps.structure.family == "1x2x1_spread" and index == 2 else _anchor_label(pl.leg.anchor)
        head = f"      {side} {label} {instr}{right} @ {pl.strike:.4f}"
        if label == "wing":
            head += " (geometrically constructed strike; delta not supplied)"
        if base_notional is not None:
            head += f"  · notional ≈{abs(pl.notional) * base_notional:,.0f} {ccy}".rstrip()
        lines.append(head)
    if ps.barrier is not None:
        lines.append(f"      knock-out barrier @ {ps.barrier:.4f}")
    return lines


def _payoff_line(priced_structure, variant, structure_id: str, indent: str = "     ") -> str | None:
    """Render deterministic expiry risk and net-loss thresholds from priced legs."""
    if priced_structure is None or not getattr(priced_structure, "priced_legs", None):
        return None
    is_call = priced_structure.priced_legs[0].leg.right.value == "call"
    return f"{indent}PAYOFF: " + payoff_risk_note(structure_id, variant, priced_structure, is_call)


def _trade_tag(pack, view) -> str:
    exp = pack.expiry.strftime("%d-%b-%y") if getattr(pack, "expiry", None) else f"{view.horizon_days}d"
    return f"{view.pair} {exp}"


def _carry_explanation(ms, view) -> str:
    from knowledge_engine.loader import load_agent_vocabulary

    wording = load_agent_vocabulary()["carry_explanations"]
    base, quote = view.pair[:3], view.pair[3:]
    if ms.fwd == ms.spot:
        explanation = wording["equal"]
    else:
        relation = "below" if ms.fwd < ms.spot else "above"
        action = "buy" if view.direction == "base_higher" else "sell"
        alignment = "WITH the carry" if ms.with_carry else "COUNTER to the carry"
        higher_yield = base if relation == "below" else quote
        explanation = (
            f"This view is {alignment}. {higher_yield} has the higher implied interest rate. "
            + wording[f"{action}_{relation}"].format(base=base, quote=quote)
        )
        exposure = "long" if action == "buy" else "short"
        outcome = "positive" if ms.with_carry else "negative"
        explanation += (
            f" For a {exposure} {view.pair} forward, unchanged expiry spot produces a "
            f"{outcome} payoff before costs. "
            + wording["favourable_interpretation" if ms.with_carry else "adverse_interpretation"]
        )
    return (
        f"{explanation} {view.pair} spot={ms.spot:.4f}, forward={ms.fwd:.4f}; "
        f"horizon={view.horizon_days}d (c={ms.c:+.3f}, regime={ms.carry_regime}). "
        + wording["qualification"]
    )


def render_pack(pack: StandardPack, view: TradeView) -> str:
    """Render the deterministic standard pack as labelled text for the agent."""
    ms = pack.market_state
    direction = "Long" if view.direction == "base_higher" else "Short"
    base_ccy = view.pair[:3]   # ccy1 — all base-ccy amounts below are in this currency
    lines: list[str] = []
    from knowledge_engine.tail_policy import tail_constraint_label, tail_risk_text
    lines.append(f"TAIL CONSTRAINT: {tail_constraint_label(pack.tail_constraint)}; effective: {tail_constraint_label(pack.resolved_tail_constraint)}.")
    for excluded in pack.tail_exclusions:
        lines.append(f"TAIL EXCLUSION: {excluded['structure_id']} — {excluded['variant']}: {excluded['reason']}")

    lines.append(
        f"VIEW: {direction} {view.pair} · {view.horizon_days}d"
        + (f" · target {pack.target:.4f}" if pack.target is not None else " · no target")
    )

    lines.append("\nMARKET CONTEXT (computed):")
    lines.append(f"  spot={ms.spot:.4f}  fwd={ms.fwd:.4f}  atm_vol={ms.vol:.4%}")
    lines.append(f"  CARRY: {_carry_explanation(ms, view)}")
    if ms.atmfsratio is not None:
        lines.append(
            f"  carry-capture payout ratio={ms.atmfsratio:.2f} (payout of the carry-capturing "
            "spread; higher → carry capture is better rewarded. This is NOT a measure of "
            "whether carry helps or hurts your view.)"
        )
    if ms.target_z is not None:
        lines.append(f"  target_z(fwd)={ms.target_z:+.2f}σ  put_call={ms.put_call}")

    # Context guidance — the verbal spec of how this regime is scored (the scenario-
    # weighting lens). Relay when explaining WHY a structure suits the regime; it does
    # not override the engine's ranked pick.
    _ctx_id = getattr(pack, "active_context", None)
    if _ctx_id:
        from knowledge_engine.scenario_weighter import get_context_commentary
        _comm = get_context_commentary(_ctx_id)
        if _comm.get("market_behavior") or _comm.get("trade_guidance"):
            lines.append("\nCONTEXT GUIDANCE — the scoring lens for the current regime. Paraphrase this in your own"
                         " words to explain the fit; do NOT state any internal regime/label name. It explains the"
                         " engine's ranking, it never overrides it:")
            if _comm.get("market_behavior"):
                lines.append(f"  Market behaviour: {_comm['market_behavior']}")
            if _comm.get("trade_guidance"):
                lines.append(f"  Privileges: {_comm['trade_guidance']}")

    if pack.recommended and pack.variants_ranked:
        lines.append(
            "\nRECOMMENDED STRUCTURES (specific, priced — individual variants by "
            "PnL score; use these):"
        )
        cap_note = f"notional capped at 10×W = {10 * pack.linear_notional:,.0f} {base_ccy}"
        if pack.sizing_method == "kelly":
            lines.append(
                f"  SIZING REGIME: KELLY (the PM is operating under Kelly sizing — use ONLY "
                f"this regime's framing). Bankroll W = {pack.linear_notional:,.0f} {base_ccy}, "
                f"fractional-Kelly λ = {pack.kelly_lambda:.2f}. Each variant is sized to λ·f*·W "
                f"from the PM's stated distribution for {_trade_tag(pack, view)}, where f* is "
                f"that structure's full-Kelly fraction (stated per structure below); "
                f"{cap_note}."
            )
        elif pack.loss_budget is not None and getattr(pack, "kelly_fallback", False):
            lines.append(
                f"  SIZING REGIME: FIXED-LOSS — the PM selected KELLY, but has NOT set up a "
                f"distribution for {_trade_tag(pack, view)}, so these trades are sized "
                f"fixed-loss instead. LEAD your reply with this: say the sizes are fixed-loss "
                f"because there is no distribution for this pair/expiry, and ask them to set "
                f"one up (sizing settings above the chat) to size under Kelly. Each variant is "
                f"sized using an internal reference input of {pack.loss_budget:,.0f} {base_ccy} "
                f"(= W × the R:R-derived sizing-reference distance); this is not a contractual "
                f"loss limit or an assumed stop execution. {cap_note}, net-credit fixed at 10×W. Never "
                f"state a Kelly number for this trade."
            )
        elif pack.loss_budget is not None:
            lines.append(
                f"  SIZING REGIME: FIXED-LOSS (the PM is operating under fixed-loss sizing — use "
                f"ONLY this regime's framing). Internal reference input = "
                f"{pack.loss_budget:,.0f} {base_ccy} (= W × the R:R-derived sizing-reference distance). "
                f"The reference is used to calculate spend, not an assumed trade exit. "
                f"Sizing uses the stated proxy, subject to caps; the budget is not a contractual loss limit. "
                f"{cap_note}, net-credit fixed at 10×W."
            )
        top = pack.recommended[:_TOP_N]
        for r in top:
            lines.append(
                f"  {r.rank}. {r.display_name} — {r.variant.variant_label} [{r.structure_id}]"
            )
            lines.append("     " + _variant_summary(r.variant))
            lines.append("     " + tail_risk_text(r.variant))
            lines.extend(_legs_breakdown(r.priced_structure, r.variant.structure_notional, base_ccy))
            payoff = _payoff_line(r.priced_structure, r.variant, r.structure_id)
            if payoff:
                lines.append(payoff)
            ccy = _ccy_summary(r.variant, base_ccy)
            if ccy:
                lines.append("     " + ccy)
            # Qualitative, IP-clean findings — what the scoring *learned* about this
            # structure, with no scores / weights / methodology. The raw driver split
            # (r.drivers) stays server-side; it only DERIVES these tags.
            lines.extend(_findings_lines(r.attributes))
            lines.extend(_cell_driver_lines(r.cell_drivers, total_pct=r.absolute_contribution_total_pct))
            lines.append(f"     — {_clean_rationale(r.rationale)}")
        extra = len(pack.recommended) - len(top)
        if extra > 0:
            lines.append(
                f"  (+{extra} more recommendations retained outside the displayed top five. "
                f"Use inspect_recommendations to look up a named family or rank; do not guess.)"
            )
        if pack.deciding_axis:
            lines.append(
                f"  WHAT SEPARATED THE TOP PICK: {pack.deciding_axis}. "
                "(Use to explain the choice; synthesize the findings into a view — "
                "do not list them mechanically, and state no score.)"
            )
    else:
        # No representative priced (e.g. no target supplied) — fall back to families.
        lines.append("\nSTRUCTURE SHORTLIST (scored families):")
        for s in pack.selector_result.shortlist:
            tag = " (overlay)" if s.is_exotic else ""
            lines.append(f"  {s.rank}. {s.display_name} [{s.structure_id}]{tag} — {_clean_rationale(s.rationale)}")
        if not pack.selector_result.shortlist:
            lines.append("  (no eligible structures for this view)")

    # Only the R:R-derived stop is surfaced here. The conviction-mapped Kelly
    # fraction / adjusted-Kelly / Kelly notional are deliberately NOT rendered:
    # they are heuristic defaults, not the elicited Kelly-criterion number from
    # the dedicated Kelly Sizing screen, and the agent must not quote them.
    if pack.smile_distribution is not None or pack.flat_distribution is not None:
        lines.append("\nDISTRIBUTIONS: available (smile + flat) for scenario context.")

    lines.append(f"\nSHORTLIST REFERENCE: {shortlist_reference(pack, view)}")
    lines.append("PUBLIC TABLE (Python renders this automatically; do not retype it):\n" + render_shortlist(pack, view))
    lines.append("First answer: emit [[MARKET_COMMENTARY]] with 2–3 sentences about the market regime and selection implications, then [[TRADE_NOTES]] with one brief risk/trade-off paragraph. Python inserts the shortlist and ranking tables between these sections, then adds the chat invitation. Do not write tables, headings, invitations or five essays. Use inspect_recommendations for follow-up detail or comparisons.")
    return "\n".join(lines)


def _cell_driver_lines(cell_drivers, indent: str = "     ", total_pct: float | None = None) -> list[str]:
    """Signed shares of all absolute contributions, showing the top/bottom cells."""
    if not cell_drivers:
        return []
    pos, neg = cell_drivers
    from knowledge_engine.loader import load_contribution_display
    from knowledge_engine.scenario_scorer import contribution_share

    minimum = load_contribution_display()["minimum_absolute_total_pct"]

    def contribution_label(cell):
        share = contribution_share(cell.contrib_pct, total_pct, minimum)
        normalized = f"{share:+.1%}" if share is not None else "N/A (absolute contribution total unavailable or near zero)"
        return f"{normalized} share of total absolute contribution; original contribution {cell.contrib_pct:+.2%} of trade notional — {cell_label(cell)}"

    out = []
    if pos:
        out.append(
            f"{indent}top contributors: "
            + "; ".join(contribution_label(c) for c in pos)
        )
    if neg:
        out.append(
            f"{indent}top detractors:   "
            + "; ".join(contribution_label(c) for c in neg)
        )
    return out


def _findings_lines(tags, indent: str = "     ") -> list[str]:
    """Per-structure qualitative findings (edges / caveats) from attribute tags.

    IP-clean by construction: only phrasebook glosses, no numbers or method.
    """
    if not tags:
        return []
    from knowledge_engine.structure_attributes import ATTRIBUTES
    ordered = [t for t in ATTRIBUTES if t in tags]
    edges = [ATTRIBUTES[t].gloss for t in ordered if ATTRIBUTES[t].polarity == "edge"]
    caveats = [ATTRIBUTES[t].gloss for t in ordered if ATTRIBUTES[t].polarity in ("caveat", "neutral")]
    out = []
    if edges:
        out.append(f"{indent}findings — edges:   " + "; ".join(edges))
    if caveats:
        out.append(f"{indent}findings — caveats: " + "; ".join(caveats))
    return out


def _sizing_explanation(v, ccy: str) -> str | None:
    """Render the branch and inputs recorded by the actual sizing computation."""
    trace = getattr(v, "sizing_trace", None)
    if trace is None:
        return None
    parts = [
        f"SIZING AUDIT: status={trace.status}; requested={trace.requested_method}; "
        f"effective={trace.effective_method or 'not sized'}. {trace.reason}",
        f"Reference capital={trace.reference_capital:,.2f} {ccy}; "
        f"notional cap={trace.notional_cap:,.2f} {ccy}",
    ]
    if trace.fallback_reason:
        parts.append(f"Fallback reason: {trace.fallback_reason}")
    if trace.loss_budget is not None:
        parts.append(f"Internal reference input={trace.loss_budget:,.2f} {ccy} (explain only on sizing-method requests)")
    if trace.budget_distance is not None:
        parts.append(
            f"Budget origin: reference capital × {trace.budget_distance:.6%}; "
            f"distance = abs(target/reference - 1) / input R:R; "
            f"target={trace.budget_target:.6f}, reference forward={trace.budget_reference:.6f}, "
            f"input R:R={trace.budget_input_rr:g} (sizing reference, not assumed stop execution)"
        )
    if trace.effective_method == "fixed_loss" and trace.per_unit_loss_proxy is not None:
        parts.append(f"Sizing denominator={trace.per_unit_loss_proxy:.6%} per unit of notional")
    if trace.effective_method == "kelly":
        if trace.bankroll is not None:
            parts.append(f"Kelly bankroll={trace.bankroll:,.2f} {ccy}; λ={trace.kelly_lambda:g}")
        if trace.full_kelly_fraction is not None:
            parts.append(
                f"Kelly: f* = {trace.full_kelly_fraction:.8g}× bankroll notional; "
                f"full-Kelly sizing-proxy exposure={trace.full_kelly_proxy_exposure:.4%} "
                f"(not contractual capital at risk); pre-cap notional = λ × f* × bankroll"
            )
        if trace.distribution_id:
            parts.append(f"Stated distribution={trace.distribution_id}, {trace.distribution_points} points")
    if trace.uncapped_notional is not None:
        parts.append(f"Pre-cap notional={trace.uncapped_notional:,.2f} {ccy}")
    if trace.final_notional is not None:
        parts.append(f"Final notional={trace.final_notional:,.2f} {ccy}; determining rule={trace.binding_constraint}")
    return " | ".join(parts)


def _ccy_summary(v, ccy: str = "base ccy") -> str | None:
    """Scale canonical per-notional meanings using the current sized notional."""
    if v.structure_notional is None:
        return _sizing_explanation(v, ccy)
    notional = v.structure_notional
    parts = [f"sized: notional≈{notional:,.0f} {ccy}",
             f"signed premium≈{v.net_premium_pct * notional:,.0f} {ccy}"]
    economics = getattr(v, "economics", None)
    if economics is not None:
        for label, fraction in (
            ("Loss budget", economics.sizing_loss_pct),
            ("contractual maximum loss", economics.contractual_max_loss_pct),
            ("net P&L at target", economics.target_net_pnl_pct),
        ):
            if fraction is not None:
                parts.append(f"{label}≈{fraction * notional:,.0f} {ccy}")
    elif v.max_loss_ccy is not None:
        parts.append(f"Loss budget≈{v.max_loss_ccy:,.0f} {ccy} (sizing amount, not a guaranteed maximum loss)")
    explanation = _sizing_explanation(v, ccy)
    if explanation:
        parts.append(explanation)
    return "  ".join(parts)


def _variant_summary(v) -> str:
    """One-line strikes + premium + payoff + RR for a PricedVariant."""
    strikes = ", ".join(f"{k:.4f}" for k in v.strikes)
    parts = [f"strikes=[{strikes}]"]
    if v.wing_ratio is not None:
        # Seagull: long 1 / short 1 / wing sold at wing_ratio units (sized to fund
        # the structure to zero cost — NOT 1x1x1).
        parts.append(f"legs=1×1×{v.wing_ratio:g} (long/short/wing)")
    if v.barrier:
        parts.append(f"barrier={v.barrier:.4f}")
    # Sign tag disambiguates the premium: a net debit is premium the PM PAYS; a net credit is
    # premium the PM RECEIVES. Keeps the agent from reading a positive premium as "receiving".
    if v.is_zero_cost or abs(v.net_premium_pct) < 1e-9:
        prem_tag = "zero-cost"
    elif v.net_premium_pct < 0:
        prem_tag = "net credit — PM receives"
    else:
        prem_tag = "net debit — PM pays"
    parts.append(f"premium={v.net_premium_pct:.2%} ({prem_tag})")
    risk = getattr(v, "can_lose_beyond_premium", None)
    risk_label = "yes" if risk is True else "no" if risk is False else "unknown — construction not classified"
    parts.append(f"can lose beyond premium paid: {risk_label} (construction config; not a maximum-loss amount)")
    economics = getattr(v, "economics", None)
    if economics is None:
        parts.append("financial definitions unavailable; do not infer maximum loss or target return")
    else:
        if economics.sizing_loss_pct is not None:
            parts.append(f"internal per-unit sizing input={economics.sizing_loss_pct:.2%} ({economics.sizing_loss_method}; explain only on sizing-method requests)")
        if economics.contractual_max_loss_pct is None:
            parts.append(f"contractual maximum loss: {economics.contractual_loss_status} — {economics.contractual_loss_reason}")
        else:
            parts.append(f"contractual maximum loss={economics.contractual_max_loss_pct:.2%} ({economics.contractual_loss_reason})")
        horizon = f"at {economics.evaluation_days}-day horizon ({economics.valuation_kind})"
        if economics.target_net_pnl_pct is not None:
            parts.append(f"net P&L at target {horizon}={economics.target_net_pnl_pct:.2%}")
        else:
            parts.append(f"net P&L at target {horizon}: unavailable — {economics.target_pnl_reason}")
        if economics.target_return_on_premium is not None:
            parts.append(f"target return on premium={economics.target_return_on_premium:.2f}× (net P&L / premium paid)")
        else:
            parts.append(f"target return on premium: {economics.ratio_reason}")
        parts.append(f"basis: {economics.basis}; {economics.valuation_convention}")
    if v.is_zero_cost:
        parts.append("zero-cost")
    return "  ".join(parts)


def render_priced_structure(ps: PricedStructure, attributes=frozenset(), base_ccy: str = "base ccy") -> str:
    """Render a single PM-requested priced structure (Tier-2 result).

    ``attributes`` are the IP-clean findings computed against the frozen pack so
    a PM-named (off-menu) structure is characterized in the same vocabulary as
    the recommended set — the LLM can then contrast it by diffing the findings.
    """
    v = ps.variant
    lines = [f"PM-REQUESTED STRUCTURE: {ps.request.canonical}", "  " + _variant_summary(v)]
    lines.extend(_legs_breakdown(getattr(ps, "priced_structure", None), v.structure_notional, base_ccy))
    payoff = _payoff_line(getattr(ps, "priced_structure", None), v, ps.request.family, indent="  ")
    if payoff:
        lines.append(payoff)
    ccy = _ccy_summary(v, base_ccy)
    if ccy:
        lines.append("  " + ccy)
    lines.extend(_findings_lines(attributes, indent="  "))
    if ps.warnings:
        lines.append("  warnings: " + "; ".join(ps.warnings))
    return "\n".join(lines)


def render_recommended(rec, base_ccy: str = "base ccy") -> str:
    """Render a recommended (already-priced) structure pulled from the pack."""
    lines = [
        f"RECOMMENDED {rec.display_name} — {rec.variant.variant_label} [{rec.structure_id}]",
        "  " + _variant_summary(rec.variant),
    ]
    from knowledge_engine.tail_policy import tail_risk_text
    lines.append("  " + tail_risk_text(rec.variant))
    lines.extend(_legs_breakdown(getattr(rec, "priced_structure", None), rec.variant.structure_notional, base_ccy))
    payoff = _payoff_line(getattr(rec, "priced_structure", None), rec.variant, rec.structure_id, indent="  ")
    if payoff:
        lines.append(payoff)
    ccy = _ccy_summary(rec.variant, base_ccy)
    if ccy:
        lines.append("  " + ccy)
    lines.extend(_findings_lines(getattr(rec, "attributes", frozenset()), indent="  "))
    lines.extend(_cell_driver_lines(getattr(rec, "cell_drivers", None), indent="  ", total_pct=rec.absolute_contribution_total_pct))
    if rec.major_risk:
        lines.append(f"  risk (engine): {rec.major_risk}")
    lines.append(f"  — {_clean_rationale(rec.rationale)}")
    return "\n".join(lines)


def render_unavailable(u: PricingUnavailable) -> str:
    return f"COULD NOT PRICE '{u.request.canonical}': {u.detail}"
