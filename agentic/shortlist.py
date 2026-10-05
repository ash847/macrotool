"""Deterministic public shortlist, separate from the agent's detailed context."""

import hashlib
import html
import json
import math
import re

from knowledge_engine import ui_labels as UL
from knowledge_engine.loader import load_agent_vocabulary
from knowledge_engine.tail_policy import tail_constraint_label, tail_risk_text

SHORTLIST_TOKEN = "[[SHORTLIST]]"
MARKET_TOKEN = "[[MARKET_COMMENTARY]]"
NOTES_TOKEN = "[[TRADE_NOTES]]"
INSPECTION_DISPLAYS = ("none", "trade_details", "contributors", "detractors", "drivers", "both", "dashboard")


def inspection_tables(display, ranks, *, layout=None, fields=None, driver_count=None):
    from agentic.dashboard import DEFAULT_FIELDS, FIELD_LABELS, LAYOUTS

    if display == "dashboard":
        layout = "trades_as_columns" if layout is None else layout
        fields = list(DEFAULT_FIELDS) if fields is None else fields
        driver_count = 1 if driver_count is None else driver_count
        if layout not in LAYOUTS:
            raise ValueError("Choose trades_as_columns or trades_as_rows for layout.")
        if (not isinstance(fields, list) or not fields or any(not isinstance(field, str) or field not in FIELD_LABELS for field in fields)
                or len(set(fields)) != len(fields)):
            raise ValueError("Choose distinct supported dashboard fields: " + ", ".join(FIELD_LABELS))
        if type(driver_count) is not int or not 1 <= driver_count <= 3:
            raise ValueError("driver_count must be an integer from 1 to 3 (retained top drivers only).")
        return {"dashboard": {"ranks": list(ranks), "layout": layout, "fields": list(fields), "driver_count": driver_count}}
    if any(value is not None for value in (layout, fields, driver_count)):
        raise ValueError("Use display=dashboard to select fields, layout or driver_count.")
    kinds = {
        "none": (), "trade_details": ("trade_details",),
        "contributors": ("contributors",), "detractors": ("detractors",),
        "drivers": ("contributors", "detractors"),
        "both": ("trade_details", "contributors", "detractors"),
    }
    return {kind: list(ranks) for kind in kinds[display]}


def _cell(value) -> str:
    return html.escape(str(value)).replace("|", "&#124;").replace("\n", " ")


def _money(value, currency) -> str:
    if value is None or not math.isfinite(value):
        return "Unavailable"
    if 0 < abs(value) < 0.01:
        return f"{'−' if value < 0 else ''}<0.01 {currency}"
    return f"{0.0 if value == 0 else value:,.2f} {currency}"


def _terms(recommendation) -> str:
    variant = recommendation.variant
    parts = [_cell(recommendation.display_name), _cell(variant.variant_label)]
    product = recommendation.priced_structure
    if product is not None and product.priced_legs:
        parts.extend(
            f"{'Buy' if leg.notional > 0 else 'Sell'} {abs(leg.notional):g}× "
            f"{_cell(leg.leg.right.value)} {leg.strike:.4f}"
            for leg in product.priced_legs
        )
    elif variant.strikes:
        parts.append("Strikes: " + ", ".join(f"{strike:.4f}" for strike in variant.strikes))
    if variant.barrier is not None:
        parts.append(f"KO {variant.barrier:.4f}")
    parts.append(_cell(tail_risk_text(variant)))
    return "; ".join(parts)


def render_shortlist(pack, view, ranks=None) -> str:
    if ranks is None:
        return render_market_state(pack, view) + "\n\n" + render_trade_tables(pack, view)
    vocabulary = load_agent_vocabulary()
    selected = pack.recommended[:5] if ranks is None else [
        recommendation for recommendation in pack.recommended if recommendation.rank in ranks
    ]
    if not selected:
        return "No priced recommendations are available for this view."
    currency = view.pair[:3]
    rows = [
        f"**{'Top recommendations' if ranks is None else 'Trade comparison'} — {view.pair}**",
        vocabulary["shortlist_scope"],
        "",
        "| Rank | Structure / key terms | Sized notional | Premium | Net P&L at target | Target return on premium | Additional loss beyond premium? |",
        "| --- | --- | ---: | --- | ---: | ---: | --- |",
    ]
    for rec in selected:
        variant = rec.variant
        economics = getattr(variant, "economics", None)
        premium = variant.net_premium_ccy
        if premium is None:
            premium_text = f"{abs(variant.net_premium_pct):.2%} of notional"
        else:
            premium_text = _money(abs(premium), currency)
        if variant.net_premium_pct > 0:
            premium_text = "Pay " + premium_text
        elif variant.net_premium_pct < 0:
            premium_text = "Receive " + premium_text
        else:
            premium_text = "Zero"
        pnl = None
        if economics is not None and economics.target_net_pnl_pct is not None and variant.structure_notional is not None:
            pnl = economics.target_net_pnl_pct * variant.structure_notional
        pnl_text = _money(pnl, currency)
        if economics is not None:
            pnl_text += f"; {economics.evaluation_days}d · {'expiry' if economics.valuation_kind == 'expiry_payoff' else 'MtM'}"
        ratio = "Unavailable"
        if economics is not None:
            if economics.target_return_on_premium is not None:
                ratio = f"{economics.target_return_on_premium:.2f}×"
            elif economics.ratio_status == "not_applicable":
                ratio = vocabulary["no_premium_return"]
            else:
                ratio = "Unavailable — " + _cell(economics.ratio_reason or "not calculated")
        flag = getattr(variant, "can_lose_beyond_premium", None)
        risk = "Yes" if flag is True else "No" if flag is False else "Unknown"
        rows.append(
            f"| {rec.rank} | {_terms(rec)} | {_money(variant.structure_notional, currency)} | "
            f"{premium_text} | {pnl_text} | {ratio} | {risk} |"
        )
    if pack.target is not None:
        rows.append(f"\nTarget spot: {pack.target:.4f}. Net P&L is after entry premium at the horizon shown.")
    rows.append(vocabulary["premium_risk_note"])
    if pack.kelly_fallback:
        rows.insert(1, "**" + vocabulary["kelly_fallback"] + "**")
    return "\n".join(rows)


def render_market_state(pack, view) -> str:
    state = pack.market_state
    L = UL.label
    target_spot = f"{pack.target:.4f}" if pack.target is not None else "—"
    target_z = f"{state.target_z:+.2f}σ" if state.target_z is not None else "—"
    target_z_spot = f"{state.target_z_spot:+.2f}σ" if state.target_z_spot is not None else "—"
    ratio = f"{state.atmfsratio:.2f}x" if state.atmfsratio is not None else "—"
    quotes = pack.market_quotes
    rr = f"{quotes['rr25']:+.2%}" if quotes.get("rr25") is not None else "—"
    fly = f"{quotes['fly25']:+.2%}" if quotes.get("fly25") is not None else "—"
    return "\n".join([
        f"### Market state — {view.pair}",
        f"| {L('spot')} | {L('forward')} | {L('implied_vol')} | {L('horizon')} | Target |",
        "| :---: | :---: | :---: | :---: | :---: |",
        f"| {state.spot:.4f} | {state.fwd:.4f} | {state.vol:.1%} | {view.horizon_days}d | {target_spot} |",
        "",
        f"| {L('carry')} | {L('carry_vs_vol')} | {L('target_distance_spot')} | "
        f"{L('target_distance_fwd')} | {L('carry_payout_ratio')} |",
        "| :---: | :---: | :---: | :---: | :---: |",
        f"| {state.c:+.3f} | {UL.carry_vs_vol_label(state.carry_regime)} | {target_z_spot} | {target_z} | {ratio} |",
        "",
        f"| {L('rate_base', ccy=view.pair[:3])} | {L('rate_quote', ccy=view.pair[3:])} | "
        f"{L('skew')} | {L('smile_curvature')} |",
        "| :---: | :---: | :---: | :---: |",
        f"| {state.r_f:.2%} | {state.r_d:.2%} | {rr} | {fly} |",
    ])


def render_trade_tables(pack, view) -> str:
    rows = ["### Structure Fit", "Ranked by how well each type of structure fits the view, as a percentage of the maximum possible; not a probability of success.",
            "", f"| # | Structure | {UL.label('fit_score')} |", "| --- | --- | ---: |"]
    for rank, family in enumerate(pack.affinity_shortlist, 1):
        rows.append(f"| {rank} | {_cell(family['display_name'])} | {family['fit_pct']:.0f}% |")
    if not pack.affinity_shortlist:
        rows.append("\nNo eligible primary structures.")
    rows.extend(["", "### Ranked packages"])
    if not pack.variants_ranked:
        rows.append("Scenario ranking unavailable; specify a usable target to rank individual variants.")
        return "\n".join(rows)
    rows.extend([load_agent_vocabulary()["shortlist_scope"], ""])
    selected = pack.recommended[:5]
    kelly = any(rec.variant.kelly_fraction is not None for rec in selected)
    rows.append("| Rank | Structure | Variant | Strikes | Notional | Premium |"
                + (f" {UL.label('kelly_risk')} |" if kelly else ""))
    rows.append("| --- | --- | --- | --- | ---: | ---: |" + (" ---: |" if kelly else ""))
    for rec in selected:
        variant = rec.variant
        strikes = " / ".join(f"{strike:.4f}" for strike in variant.strikes) or "—"
        notional = "—" if variant.structure_notional is None else f"{'-' if variant.structure_notional < 0 else ''}{view.pair[:3]} {abs(variant.structure_notional):,.0f}"
        row = f"| {rec.rank} | {_cell(rec.display_name)} | {_cell(variant.variant_label)} | {strikes} | {notional} | {variant.net_premium_pct:+.2%} |"
        if kelly:
            risk = "—" if variant.kelly_fraction is None else f"{variant.kelly_fraction * (variant.max_loss_pct or 0.0):.0%}"
            row += f" {risk} |"
        rows.append(row)
    rows.append("\nPositive premium is paid; negative premium is received. PnL score reflects performance across modelled market outcomes, not a guaranteed return.")
    if pack.resolved_tail_constraint != "none":
        rows.append(f"Active tail constraint: {tail_constraint_label(pack.tail_constraint)}; effective: {tail_constraint_label(pack.resolved_tail_constraint)}.")
        if not selected:
            rows.append("No priced variants satisfy the active tail constraint; none have been substituted.")
    if kelly:
        rows.append(f"{UL.label('kelly_risk')} is the full-Kelly sizing-loss proxy as a share of W, before λ; not contractual maximum loss.")
    if any(rec.structure_id == "linear" for rec in selected):
        rows.append("Linear is the Trade View benchmark with modelled capped scenario losses, not contractual protection.")
    if pack.kelly_fallback:
        rows.insert(0, "**" + load_agent_vocabulary()["kelly_fallback"] + "**\n")
    return "\n".join(rows)


def shortlist_reference(pack, view) -> str:
    payload = {
        "table": render_shortlist(pack, view, [rec.rank for rec in pack.recommended]),
        "market": [pack.market_state.spot, pack.market_state.fwd, pack.market_state.vol,
                   pack.market_state.r_d, pack.market_state.r_f],
        "pair": view.pair, "horizon": view.horizon_days,
        "method": pack.sizing_method, "weights": pack.scenario_weights,
        "tail_constraint": pack.tail_constraint, "effective_tails": pack.resolved_tail_constraint,
        "distributions": [getattr(getattr(rec.variant, "sizing_trace", None), "distribution_id", None)
                          for rec in pack.recommended],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def render_driver_table(pack, view, ranks, kind):
    from knowledge_engine.loader import load_contribution_display
    from knowledge_engine.scenario_scorer import cell_label, contribution_share

    minimum = load_contribution_display()["minimum_absolute_total_pct"]
    selected = [rec for rec in pack.recommended if rec.rank in ranks]
    if not selected:
        return "No retained recommendations match the requested driver table."
    heading = "Top contributors" if kind == "contributors" else "Top detractors"
    sections = [f"### {heading} — {view.pair}"]
    for rec in selected:
        sections.append(f"**Rank {rec.rank} · {_cell(rec.display_name)} · {_cell(rec.variant.variant_label)}**")
        if rec.cell_drivers is None:
            sections.append("Scenario contribution data unavailable for this variant.")
            continue
        cells = rec.cell_drivers[0 if kind == "contributors" else 1]
        if not cells:
            sections.append(f"No {'positive' if kind == 'contributors' else 'negative'} weighted contributions reported.")
            continue
        rows = ["| Scenario | Share of total absolute contribution |",
                "| --- | ---: |"]
        for cell in cells:
            share = contribution_share(cell.contrib_pct, rec.absolute_contribution_total_pct, minimum)
            value = f"{share:+.1%}" if share is not None else "N/A"
            rows.append(f"| {_cell(cell_label(cell))} | {value} |")
        sections.append("\n".join(rows))
    sections.append("Signed shares use all scenario cells for each variant, not just the rows shown. "
                    "They measure relative influence, not probabilities or shares of net profit. "
                    "N/A means the absolute contribution total is unavailable or near zero.")
    return "\n\n".join(sections)


def render_inspection_tables(pack, view, tables):
    if "dashboard" in tables:
        from agentic.dashboard import render_dashboard
        return render_dashboard(pack, view, **tables["dashboard"])
    sections = []
    for kind in ("trade_details", "contributors", "detractors"):
        ranks = tables.get(kind, [])
        if ranks:
            sections.append(render_shortlist(pack, view, ranks) if kind == "trade_details"
                            else render_driver_table(pack, view, ranks, kind))
    return "\n\n".join(sections)


def present_shortlist(text, pack, view, *, automatic=False, ranks=None, tables=None) -> str:
    if tables is None and not automatic and SHORTLIST_TOKEN not in text:
        return text
    if pack is None or view is None:
        if tables is not None:
            return "No priced shortlist is available yet."
        return text.replace(SHORTLIST_TOKEN, "No priced shortlist is available yet.")
    text = re.sub(r"<table\b[^>]*>.*?(?:</table\s*>|$)", "", text, flags=re.IGNORECASE | re.DOTALL)
    lines = text.replace(SHORTLIST_TOKEN, "").splitlines()
    narration_lines = []
    index = 0
    while index < len(lines):
        if index + 1 < len(lines) and "|" in lines[index] and "|" in lines[index + 1] and "-" in lines[index + 1] and not set(lines[index + 1]) - set(" |:-\t"):
            index += 2
            while index < len(lines) and "|" in lines[index]:
                index += 1
        else:
            narration_lines.append(lines[index])
            index += 1
    narration = re.sub(r"```[^\n]*\n\s*```", "", "\n".join(narration_lines)).strip()
    if tables is not None:
        narration = narration.replace(MARKET_TOKEN, "").replace(NOTES_TOKEN, "").strip()
        return "\n\n".join(section for section in (render_inspection_tables(pack, view, tables), narration) if section)
    if ranks is None:
        narration = narration.replace(MARKET_TOKEN, "").strip()
        if NOTES_TOKEN in narration:
            market, _, notes = narration.partition(NOTES_TOKEN)
        else:
            market, _, notes = narration.partition("\n\n")
        sections = [render_market_state(pack, view), market.strip(), render_trade_tables(pack, view),
                    notes.replace(NOTES_TOKEN, "").strip()]
        if pack.variants_ranked:
            sections.append(load_agent_vocabulary()["chat_invitation"])
        return "\n\n".join(section for section in sections if section)
    narration = narration.replace(MARKET_TOKEN, "").replace(NOTES_TOKEN, "").strip()
    return render_shortlist(pack, view, ranks) + ("\n\n" + narration if narration else "")
