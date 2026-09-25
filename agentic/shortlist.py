"""Deterministic public shortlist, separate from the agent's detailed context."""

import hashlib
import html
import json
import math

from knowledge_engine.loader import load_agent_vocabulary

SHORTLIST_TOKEN = "[[SHORTLIST]]"


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
    return "<br>".join(parts)


def render_shortlist(pack, view, ranks=None) -> str:
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
            pnl_text += f"<br>{economics.evaluation_days}d · {'expiry' if economics.valuation_kind == 'expiry_payoff' else 'MtM'}"
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


def shortlist_reference(pack, view) -> str:
    payload = {
        "table": render_shortlist(pack, view, [rec.rank for rec in pack.recommended]),
        "market": [pack.market_state.spot, pack.market_state.fwd, pack.market_state.vol,
                   pack.market_state.r_d, pack.market_state.r_f],
        "pair": view.pair, "horizon": view.horizon_days,
        "method": pack.sizing_method, "weights": pack.scenario_weights,
        "distributions": [getattr(getattr(rec.variant, "sizing_trace", None), "distribution_id", None)
                          for rec in pack.recommended],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def present_shortlist(text, pack, view, *, automatic=False, ranks=None) -> str:
    if not automatic and SHORTLIST_TOKEN not in text:
        return text
    if pack is None or view is None:
        return text.replace(SHORTLIST_TOKEN, "No priced shortlist is available yet.")
    table = render_shortlist(pack, view, ranks)
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
    narration = "\n".join(narration_lines).strip()
    return table + ("\n\n" + narration if narration else "")
