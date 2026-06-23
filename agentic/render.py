"""Labelled text renderers for the agent's tool results.

The agent narrates over these strings — it never sees raw objects. Renderers
label "baseline (from the standard pack)" vs a PM-requested structure so the
agent can cite the right source, and they surface only computed numbers.
"""

from __future__ import annotations

import re

from agentic.price_structure import PricedStructure, PricingUnavailable
from agentic.standard_pack import StandardPack
from analytics.product_model import AnchorKind
from knowledge_engine.models import TradeView

_TOP_N = 3   # recommended structures shown by default; rest surfaced only on request

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
    for pl in ps.priced_legs:
        side = "long" if pl.notional > 0 else "short"
        right = pl.leg.right.value.capitalize()
        instr = "Digital " if pl.leg.instrument.value == "digital" else ""
        head = f"      {side} {_anchor_label(pl.leg.anchor)} {instr}{right} @ {pl.strike:.4f}"
        if base_notional is not None:
            head += f"  · notional ≈{abs(pl.notional) * base_notional:,.0f} {ccy}".rstrip()
        lines.append(head)
    if ps.barrier is not None:
        lines.append(f"      knock-out barrier @ {ps.barrier:.4f}")
    return lines


def render_pack(pack: StandardPack, view: TradeView) -> str:
    """Render the deterministic standard pack as labelled text for the agent."""
    ms = pack.market_state
    direction = "Long" if view.direction == "base_higher" else "Short"
    base_ccy = view.pair[:3]   # ccy1 — all base-ccy amounts below are in this currency
    lines: list[str] = []

    lines.append(
        f"VIEW: {direction} {view.pair} · {view.horizon_days}d"
        + (f" · target {pack.target:.4f}" if pack.target is not None else " · no target")
    )

    lines.append("\nMARKET CONTEXT (computed):")
    lines.append(f"  spot={ms.spot:.4f}  fwd={ms.fwd:.4f}  atm_vol={ms.vol:.4%}")
    if ms.with_carry:
        carry = (
            "WITH the carry — long the higher-yielding currency, so the forward drift is in "
            "your favour (you effectively sell forward at a premium to spot). Carry SUPPORTS "
            "this view."
        )
    else:
        carry = (
            "COUNTER to the carry — long the lower-yielding currency; the forward drift works "
            "against this view."
        )
    lines.append(f"  CARRY: this view is {carry} (c={ms.c:+.3f}, regime={ms.carry_regime})")
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
            lines.append(f"\nCONTEXT GUIDANCE — scoring lens for '{_ctx_id}' (relay when explaining the fit, not as an override):")
            if _comm.get("market_behavior"):
                lines.append(f"  Market behaviour: {_comm['market_behavior']}")
            if _comm.get("trade_guidance"):
                lines.append(f"  Privileges: {_comm['trade_guidance']}")

    if pack.recommended:
        lines.append(
            "\nRECOMMENDED STRUCTURES (specific, priced — best variant per family by "
            "scenario-weighted P&L; use these):"
        )
        if pack.loss_budget is not None:
            lines.append(
                f"  (each variant sized so its max loss = the loss budget "
                f"{pack.loss_budget:,.2f} {base_ccy}, on a 100-unit linear notional, R:R-derived; "
                f"notional capped at 1,000 units, and net-credit structures fixed at 1,000)"
            )
        top = pack.recommended[:_TOP_N]
        for r in top:
            lines.append(
                f"  {r.rank}. {r.display_name} — {r.variant.variant_label} [{r.structure_id}]"
            )
            lines.append("     " + _variant_summary(r.variant))
            lines.extend(_legs_breakdown(r.priced_structure, r.variant.structure_notional, base_ccy))
            ccy = _ccy_summary(r.variant, base_ccy)
            if ccy:
                lines.append("     " + ccy)
            # Qualitative, IP-clean findings — what the scoring *learned* about this
            # structure, with no scores / weights / methodology. The raw driver split
            # (r.drivers) stays server-side; it only DERIVES these tags.
            lines.extend(_findings_lines(r.attributes))
            # major_risk is intentionally NOT surfaced by default — it's a generic
            # family-level caveat the PM rarely wants unprompted. It stays in the
            # data and is rendered on request via render_recommended (price_structure).
            lines.append(f"     — {_clean_rationale(r.rationale)}")
        extra = len(pack.recommended) - len(top)
        if extra > 0:
            lines.append(
                f"  (+{extra} more structures considered — list them only if the PM asks.)"
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
    if pack.sizing is not None and pack.sizing.stop_level is not None:
        sz = pack.sizing
        sd = f"{sz.stop_distance_pct:.2%}" if sz.stop_distance_pct is not None else "n/a"
        lines.append("\nSIZING (baseline, top structure):")
        lines.append(f"  stop={sz.stop_level:.4f} (dist {sd})")

    if pack.smile_distribution is not None or pack.flat_distribution is not None:
        lines.append("\nDISTRIBUTIONS: available (smile + flat) for scenario context.")

    return "\n".join(lines)


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


def _ccy_summary(v, ccy: str = "base ccy") -> str | None:
    """Notional/premium/max-loss in the pair's base currency, on the standard
    linear-notional basis (sized so max loss = the R:R-derived loss budget — same
    as the Trade View variants table). None when the variant wasn't sized."""
    if v.structure_notional is None:
        return None
    return (
        f"sized: notional≈{v.structure_notional:,.0f} {ccy}  premium≈{v.net_premium_ccy:,.0f} {ccy}  "
        f"max_loss≈{v.max_loss_ccy:,.0f} {ccy} (linear-notional basis)"
    )


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
    parts.append(f"premium={v.net_premium_pct:.2%}")
    if v.payoff_at_target_pct is not None:
        parts.append(f"payoff@target={v.payoff_at_target_pct:.2%}")
    if v.rr_at_target is not None:
        parts.append(f"rr={v.rr_at_target:.2f}")
    parts.append(f"max_loss={v.max_loss_pct:.2%}")
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
    lines.extend(_legs_breakdown(getattr(rec, "priced_structure", None), rec.variant.structure_notional, base_ccy))
    ccy = _ccy_summary(rec.variant, base_ccy)
    if ccy:
        lines.append("  " + ccy)
    lines.extend(_findings_lines(getattr(rec, "attributes", frozenset()), indent="  "))
    if rec.major_risk:
        lines.append(f"  risk (engine): {rec.major_risk}")
    lines.append(f"  — {_clean_rationale(rec.rationale)}")
    return "\n".join(lines)


def render_unavailable(u: PricingUnavailable) -> str:
    return f"COULD NOT PRICE '{u.request.canonical}': {u.detail}"
