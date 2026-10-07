"""Flexible table presentation over retained engine facts, without repricing."""

import math

from agentic.shortlist import _cell, _money
from knowledge_engine.loader import load_contribution_display
from knowledge_engine.scenario_scorer import cell_label, contribution_share
from knowledge_engine.tail_policy import tail_constraint_label, tail_risk_text

LAYOUTS = ("trades_as_columns", "trades_as_rows")
FIELD_LABELS = {
    "legs": "Strikes / legs",
    "notional": "Sized notional",
    "premium": "Premium (paid / received)",
    "target_pnl": "Net P&L at target",
    "target_return_on_premium": "Target return on premium",
    "loss_budget": "Loss budget",
    "additional_loss_beyond_premium": "Additional loss beyond premium?",
    "directional_tails": "Directional tails",
    "risk_note": "Payoff / risk",
    "top_contributor": "Top contributor",
    "top_detractor": "Top detractor",
}
DEFAULT_FIELDS = tuple(field for field in FIELD_LABELS if field not in ("risk_note", "top_detractor"))


def _scaled(fraction, notional, currency):
    return _money(None if fraction is None or notional is None else fraction * notional, currency)


def _driver(rec, positive, count):
    if rec.cell_drivers is None:
        return "Unavailable — scenario evidence not retained"
    cells = rec.cell_drivers[0 if positive else 1][:count]
    if not cells:
        return "No positive contributions reported" if positive else "No negative contributions reported"
    minimum = load_contribution_display()["minimum_absolute_total_pct"]
    rows = []
    for cell in cells:
        share = contribution_share(cell.contrib_pct, rec.absolute_contribution_total_pct, minimum)
        value = f"{share:+.1%}" if share is not None else "N/A"
        rows.append(f"{_cell(cell_label(cell))}: {value}")
    return "; ".join(rows)


def dashboard_cells(rec, pack, view, driver_count):
    variant = rec.variant
    economics = variant.economics
    currency = view.pair[:3]
    product = rec.priced_structure
    legs = []
    if product is not None and product.priced_legs:
        legs = [f"{'Buy' if leg.notional > 0 else 'Sell'} {abs(leg.notional):g}× "
                f"{_cell(leg.leg.right.value)} {leg.strike:.4f}" for leg in product.priced_legs]
    elif variant.strikes:
        legs = ["Strikes: " + " / ".join(f"{strike:.4f}" for strike in variant.strikes)]
    elif rec.structure_id == "linear":
        legs = ["Linear benchmark; modelled stop is not contractual protection"]
    if variant.barrier is not None:
        legs.append(f"KO barrier {variant.barrier:.4f}")
    premium = "Unavailable"
    fraction = variant.net_premium_pct
    if fraction is not None and math.isfinite(fraction):
        if fraction == 0:
            premium = "Zero (0.00% of notional)"
        else:
            amount = _money(abs(variant.net_premium_ccy), currency) if variant.net_premium_ccy is not None else "Amount unavailable"
            premium = f"{'Pay' if fraction > 0 else 'Receive'} {amount}; {abs(fraction):.2%} of notional"
    pnl = ratio = "Unavailable"
    budget = _money(variant.max_loss_ccy, currency) if rec.structure_id == "linear" else "Unavailable"
    if economics is not None:
        pnl = _scaled(economics.target_net_pnl_pct, variant.structure_notional, currency)
        pnl += f"; {economics.evaluation_days}d · {'expiry' if economics.valuation_kind == 'expiry_payoff' else 'MtM'}"
        if economics.target_return_on_premium is not None:
            ratio = f"{economics.target_return_on_premium:.2f}×"
        elif economics.ratio_status == "not_applicable":
            ratio = _cell((economics.ratio_reason or "N/A").replace("Not applicable", "N/A"))
        else:
            ratio = "Unavailable — " + _cell(economics.ratio_reason or "not calculated")
        budget = _scaled(economics.sizing_loss_pct, variant.structure_notional, currency)
    flag = variant.can_lose_beyond_premium
    return {
        "legs": "; ".join(legs) or "Unavailable — leg details not retained",
        "notional": _money(variant.structure_notional, currency),
        "premium": premium, "target_pnl": pnl, "target_return_on_premium": ratio,
        "loss_budget": budget,
        "additional_loss_beyond_premium": "Yes" if flag is True else "No" if flag is False else "Unknown",
        "directional_tails": _cell(tail_risk_text(variant)),
        "risk_note": _cell(rec.major_risk) if rec.major_risk else "Unavailable — no retained risk note",
        "top_contributor": _driver(rec, True, driver_count),
        "top_detractor": _driver(rec, False, driver_count),
    }


def render_dashboard(pack, view, ranks, layout, fields, driver_count):
    selected = [rec for rec in pack.recommended if rec.rank in ranks]
    if not selected:
        return "No retained recommendations match the requested dashboard."
    titles = [f"#{rec.rank} · {_cell(rec.display_name)} · {_cell(rec.variant.variant_label)}" for rec in selected]
    data = [dashboard_cells(rec, pack, view, driver_count) for rec in selected]
    labels = [FIELD_LABELS[field] + (f"s (up to {driver_count})" if field in ("top_contributor", "top_detractor") and driver_count > 1 else "")
              for field in fields]
    if layout == "trades_as_columns":
        rows = ["| Metric | " + " | ".join(titles) + " |", "| --- | " + " | ".join("---" for _ in selected) + " |"]
        rows.extend("| " + label + " | " + " | ".join(cells[field] for cells in data) + " |" for field, label in zip(fields, labels))
    else:
        rows = ["| Trade | " + " | ".join(labels) + " |", "| --- | " + " | ".join("---" for _ in fields) + " |"]
        rows.extend("| " + title + " | " + " | ".join(cells[field] for field in fields) + " |" for title, cells in zip(titles, data))
    notes = [f"**Trade dashboard — {view.pair}**", "\n".join(rows)]
    if pack.target is not None:
        notes.append(f"Target spot: {pack.target:.4f}. Target P&L is net of entry premium at the horizon shown.")
    notes.append("Loss budget is a sizing amount, not a guaranteed maximum loss. Some structures can lose more.")
    if any(field in fields for field in ("top_contributor", "top_detractor")):
        notes.append("Driver shares use the total absolute contribution across all scenario cells for that variant, not just displayed drivers. They are not probabilities or shares of net profit. N/A means the denominator is unavailable or near zero.")
    if pack.resolved_tail_constraint != "none":
        notes.append("Active tail constraint: " + tail_constraint_label(pack.resolved_tail_constraint) + ".")
    if pack.kelly_fallback:
        notes.append("Fixed-loss sizing used: no stated Kelly distribution for this pair/expiry.")
    return "\n\n".join(notes)
