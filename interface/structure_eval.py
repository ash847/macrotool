"""Structure evaluation rendering for the Trade View page.

Contains shared currency/label helpers, the structure variants block, and the
full scenario-weighted evaluation block with advisor pack preview.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd
import streamlit as st

from conversation.flow import ConversationFlow, target_from_reference


# ---------------------------------------------------------------------------
# Shared constants and formatting helpers
# ---------------------------------------------------------------------------

LINEAR_NOTIONAL = 100.0  # base ccy units; equivalent linear-trade notional

_CCY_SYM = {"USD": "$", "EUR": "€", "GBP": "£"}


def fmt_ccy(amount: float | None, ccy: str) -> str:
    """Format a base-ccy amount. Raises on unknown ccy — fail loud, not silent."""
    if amount is None:
        return "—"
    if ccy not in _CCY_SYM:
        raise ValueError(f"No currency symbol mapping for base ccy {ccy!r}")
    sign = "-" if amount < 0 else ""
    return f"{sign}{ccy} {abs(amount):,.2f}"


def fmt_ccy_label(amount: float | None, ccy: str) -> str:
    return fmt_ccy(amount, ccy)


def _is_call_spread(pv) -> bool:
    return len(pv.strikes) >= 2 and pv.strikes[1] > pv.strikes[0]


def _format_delta_token(token: str) -> str | None:
    cleaned = token.strip().replace("Δ", "D")
    match = re.search(r"(\d+(?:\.\d+)?)\s*D", cleaned, re.IGNORECASE)
    if match:
        return f"{match.group(1)}D"
    if cleaned.upper() == "ATMF":
        return "ATMF"
    return None


def _spread_delta_labels(pv, fallback_first: str | None = None) -> list[str | None]:
    if "/" not in pv.variant_label:
        return [fallback_first, None]
    left, right = pv.variant_label.split("/", 1)
    return [
        _format_delta_token(left) or fallback_first,
        _format_delta_token(right),
    ]


def _strikes_with_deltas(pv, deltas: list[str | None] | None = None) -> list[str]:
    labels: list[str] = []
    for idx, strike in enumerate(pv.strikes):
        strike_text = f"{strike:.4f}"
        delta = deltas[idx] if deltas and idx < len(deltas) else None
        labels.append(f"{strike_text} ({delta})" if delta else strike_text)
    return labels


def _vanilla_option_type(pv) -> str:
    label = pv.variant_label.lower()
    if "put" in label:
        return "put"
    if "call" in label:
        return "call"
    if len(pv.strikes) == 1:
        return "call"
    return "option"


def variant_display_label(structure_id: str, pv) -> str:
    if structure_id == "vanilla":
        return f"Vanilla {_vanilla_option_type(pv)}"

    if structure_id == "1x1_spread" and len(pv.strikes) >= 2:
        spread_type = "Call spread" if _is_call_spread(pv) else "Put spread"
        return f"1x1 {spread_type}"

    if structure_id in {"1x1.5_spread", "1x2_spread"} and len(pv.strikes) >= 2:
        ratio = "1x1.5" if structure_id == "1x1.5_spread" else "1x2"
        direction = "call" if _is_call_spread(pv) else "put"
        return f"{ratio} Ratio {direction} spread"

    if structure_id == "seagull" and len(pv.strikes) >= 3:
        wing_ratio = pv.wing_ratio if pv.wing_ratio is not None else 1.0
        is_call_spread = _is_call_spread(pv)
        spread_type = "Call spread" if is_call_spread else "Put spread"
        wing_type = "put" if is_call_spread else "call"
        return f"Seagull  ·  1x1 {spread_type} + {wing_ratio:.2f}x {wing_type} wing"

    if structure_id == "european_digital":
        return "European digital"

    if structure_id == "european_digital_rko":
        return "European digital RKO"

    if structure_id == "european_rko":
        return "European RKO"

    return pv.variant_label


def variant_label_with_strikes(structure_id: str, pv) -> str:
    strikes = [f"{k:.4f}" for k in pv.strikes]
    label = variant_display_label(structure_id, pv)

    if structure_id == "vanilla" and strikes:
        delta = _format_delta_token(pv.variant_label)
        return f"{label}  ·  Strike: {_strikes_with_deltas(pv, [delta])[0]}"

    if structure_id == "1x1_spread" and len(strikes) >= 2:
        strike_labels = _strikes_with_deltas(pv, _spread_delta_labels(pv, fallback_first="ATMF"))
        return f"{label}  ·  Strikes: {' / '.join(strike_labels)}"

    if structure_id == "seagull" and len(strikes) >= 3:
        deltas = _spread_delta_labels(pv, fallback_first="ATMF")
        wing_match = re.search(r"\+\s*(.+)$", pv.variant_label)
        wing_delta = _format_delta_token(wing_match.group(1)) if wing_match else None
        strike_labels = _strikes_with_deltas(pv, deltas + [wing_delta])
        return f"{label}  ·  Strikes: {' / '.join(strike_labels)}"

    if structure_id in {"1x1.5_spread", "1x2_spread"} and len(strikes) >= 2:
        strike_labels = _strikes_with_deltas(pv, _spread_delta_labels(pv, fallback_first="ATMF"))
        return f"{label}  ·  Strikes: {' / '.join(strike_labels)}"

    if structure_id == "european_rko" and strikes:
        ko = f"{pv.barrier:.4f}" if pv.barrier is not None else "—"
        return f"{label}  ·  Strike: {strikes[0]}  ·  Barrier: {ko}"

    if structure_id == "european_digital" and strikes:
        return f"{label}  ·  Barrier: {strikes[0]}"

    if structure_id == "european_digital_rko" and strikes:
        american_barrier = f"{pv.barrier:.4f}" if pv.barrier is not None else "—"
        return (
            f"{label}  ·  European digital barrier: {strikes[0]}  ·  "
            f"KO barrier: {american_barrier}"
        )

    if strikes:
        return f"{label}  ·  Strikes: {' / '.join(strikes)}"
    return label


def target_price(flow: ConversationFlow) -> float | None:
    if not (flow.view and flow.view.magnitude_pct):
        return None
    if flow.market_state is not None:
        anchor = flow.market_state.fwd
    else:
        anchor = flow.ccy.spot
    return target_from_reference(anchor, flow.view.direction, flow.view.magnitude_pct)


# ---------------------------------------------------------------------------
# Structure variants block (inline pricing table, no scenario weighting)
# ---------------------------------------------------------------------------

def _build_smile(flow: ConversationFlow):
    """Resolve the trade's VolSurface, or None on any failure.

    Prefers the surface the market state was already priced against (so every
    screen shares one surface); otherwise builds one from the ccy snapshot.
    Returns None (→ flat ATM vol fallback) when the pair has no complete vol
    surface, so pricing never crashes on a thin surface — it degrades to the
    legacy flat-vol behaviour.
    """
    ms = getattr(flow, "market_state", None)
    surface = getattr(ms, "surface", None) if ms is not None else None
    if surface is not None:
        return surface
    ccy = getattr(flow, "ccy", None)
    if ccy is None:
        return None
    try:
        from analytics.vol_surface import build_vol_surface

        return build_vol_surface(ccy)
    except Exception:
        return None


def _df_key(prefix: str, suffix: str) -> str | None:
    """Stable, unique st.dataframe key when a prefix is supplied (e.g. the Batch page
    renders these tables once per trade on one page). Empty prefix → None, preserving
    the single-render Trade View behaviour exactly."""
    return f"{prefix}{suffix}" if prefix else None


def _fit_height(prefix: str, n_rows: int) -> int | None:
    """Full pixel height for a dataframe so it shows every row WITHOUT an internal
    scrollbar — only when a key_prefix is set (Batch page). A capped-height table
    traps the mouse wheel and blocks the page from scrolling past it; sizing to the
    content removes that trap. Empty prefix → None → Streamlit's default height
    (Trade View unchanged)."""
    if not prefix:
        return None
    return 38 + 35 * max(n_rows, 1) + 2  # header + rows + border


def _show_df(df, *, key=None, height=None) -> None:
    """st.dataframe wrapper that OMITS height when None. Some Streamlit versions
    reject height=None (must be a positive int / 'content' / 'stretch'), so we only
    pass it when set — Trade View gets the default capped table, Batch gets a full
    content height (no inner scrollbar)."""
    kwargs = {"use_container_width": True, "hide_index": True}
    if key is not None:
        kwargs["key"] = key
    if height is not None:
        kwargs["height"] = height
    st.dataframe(df, **kwargs)


def render_structure_variants(
    flow: ConversationFlow,
    is_call: bool,
    target: float | None,
    stop_price: float | None,
    loss_budget: float | None,
    key_prefix: str = "",
) -> None:
    from analytics.structure_pricer import price_variants as _price_variants

    ms = flow.market_state
    _base_ccy = flow.view.pair[:3]
    _primary_items = flow.selector_result.shortlist
    _smile = _build_smile(flow)

    # Single pricing pass: collect priced variants and any structures whose
    # digital legs hit a smile arbitrage (dropped rather than mis-marked).
    _priced: list = []          # (item, pvs)
    _unpriced: list[str] = []   # display names with one or more dropped variants
    for _item in _primary_items:
        _warns: list[str] = []
        try:
            _pvs = _price_variants(
                ms, _item.structure_id, target=target, is_call=is_call,
                stop_price=stop_price, loss_budget=loss_budget, smile=_smile, warnings=_warns,
            )
        except Exception as _e:
            st.caption(f"DEBUG {_item.structure_id}: error — {_e}")
            continue
        if _warns:
            _unpriced.append(_item.display_name)
        if _pvs:
            _priced.append((_item, _pvs))

    if not _priced and not _unpriced:
        return

    st.subheader("Structure variants")
    st.caption(
        ("Indicative pricing — interpolated smile vol per strike. " if _smile is not None
         else "Indicative pricing — flat ATM vol for all strikes. ")
        + "Premium and payoff as % of spot. "
        "**Payout/$1**: gross payoff at target per $1 of max loss (zero-cost seagull: "
        "loss on short wing at stop price, expiry basis — understates MtM risk before expiry)."
    )

    if _unpriced:
        st.warning(
            "⚠️ **Not priced: " + ", ".join(_unpriced) + ".** "
            "One or more variants fall in a region where the interpolated vol smile implies "
            "a local arbitrage (negative risk-neutral density — typically far-OTM spline "
            "overshoot), so no reliable digital price is available. These are omitted rather "
            "than shown with an unreliable mark."
        )

    for _i, (_item, _pvs) in enumerate(_priced):
        _title = _item.display_name
        with st.expander(_title, expanded=(_i == 0)):
            _rows = []
            _has_barrier = any(pv.barrier is not None for pv in _pvs)
            _has_wing    = any(pv.wing_ratio is not None for pv in _pvs)
            for pv in _pvs:
                _payoff = pv.payoff_at_target_pct
                if _payoff and _payoff > 1e-6 and pv.max_loss_pct > 0:
                    _payout_per_1 = f"{_payoff / pv.max_loss_pct:.1f}×"
                else:
                    _payout_per_1 = "—"
                _prem_cell = (
                    "zero cost" if pv.is_zero_cost
                    else f"{pv.net_premium_pct:.1%}  ({fmt_ccy(pv.net_premium_ccy, _base_ccy)})"
                )
                _payoff_cell = (
                    f"{_payoff:.0%}  ({fmt_ccy(pv.payoff_at_target_ccy, _base_ccy)})"
                    if _payoff is not None else "—"
                )
                r = {
                    "Variant":    variant_display_label(_item.structure_id, pv),
                    "Strikes":    " / ".join(f"{K:.4f}" for K in pv.strikes),
                    "Notional":   fmt_ccy(pv.structure_notional, _base_ccy),
                    "Premium":    _prem_cell,
                    "Break-even": f"{pv.breakeven:.4f}" if pv.breakeven is not None else "—",
                    "Payout at target": _payoff_cell,
                    "Max loss":   f"{pv.max_loss_pct:.1%}  ({fmt_ccy(pv.max_loss_ccy, _base_ccy)})",
                    "Payout/$1":  _payout_per_1,
                }
                if _has_barrier:
                    r["Barrier"] = f"{pv.barrier:.4f}" if pv.barrier is not None else "—"
                if _has_wing:
                    r["Wing ×"] = f"{pv.wing_ratio:.2f}" if pv.wing_ratio is not None else "—"
                _rows.append(r)
            _show_df(
                pd.DataFrame(_rows),
                key=_df_key(key_prefix, f"var_{_item.structure_id}_{_i}"),
                height=_fit_height(key_prefix, len(_rows)),
            )


# ---------------------------------------------------------------------------
# Structure evaluation (scenario-weighted P&L tables)
# ---------------------------------------------------------------------------

# P&L-driver buckets over the scenario-grid columns. Every GRID_COL maps to
# exactly one bucket, so a variant's driver contributions sum back to its
# weighted P&L (see driver_contribs). "Adverse" = the spot-anchored downside
# cells; "Vega" = the vol-shock column.
DRIVER_BUCKETS: dict[str, list[str]] = {
    "Carry":       ["S"],
    "Directional": ["t%→K", "K−½σ", "K", "K+½σ"],
    "Adverse":     ["−½σ", "−1σ"],
    "Vega":        ["Δvol"],
}


def driver_contribs(score) -> dict[str, float]:
    """Decompose a ScoreResult's weighted P&L into driver buckets — the sum of
    per-cell ``contrib_pct`` grouped by grid column. Exhaustive: the bucket
    totals sum back to ``score.score_pct``."""
    by_col: dict[str, float] = {}
    for c in score.cells:
        by_col[c.col] = by_col.get(c.col, 0.0) + c.contrib_pct
    return {
        bucket: sum(by_col.get(col, 0.0) for col in cols)
        for bucket, cols in DRIVER_BUCKETS.items()
    }


@dataclass
class VariantEval:
    """One priced variant scored across the grid under baseline + context weights.

    ``(structure_id, variant_label)`` is a stable identity across trades — the
    delta-based labels in structure_variants.json are pair/tenor-independent — so
    the Batch pivot can align the same variant across different trades.
    """
    structure_id: str
    struct_label: str
    variant_label: str
    pv: object
    rows: list
    score: object        # ScoreResult — context (PM-overlay) weighted
    score_base: object   # ScoreResult — baseline (pre-overlay) weighted

    @property
    def score_pct(self) -> float:
        return self.score.score_pct

    @property
    def score_base_pct(self) -> float:
        return self.score_base.score_pct

    @property
    def delta_pct(self) -> float:
        """Weighting effect: context − baseline. This is what re-tuning moves."""
        return self.score.score_pct - self.score_base.score_pct

    @property
    def drivers(self) -> dict[str, float]:
        return driver_contribs(self.score)


@dataclass
class EvalResult:
    """Pure output of compute_structure_evaluation — everything the render and the
    Batch pivot need, with no Streamlit dependency."""
    ms: object
    is_call: bool
    target: float
    stop: float
    loss_budget: float
    inputs: dict
    scenarios: list
    smile: object
    weighter: object
    weights: dict
    multipliers: dict
    base_weights: dict
    base_fired: object
    overlay_fired: list
    fired_all: list
    active_ctx: str
    base_ccy: str
    structs: list        # legacy [{item, variants:[{pv,rows,score,score_base}], label}]
    variants: list       # flat list[VariantEval], ranked by context score_ccy desc


def compute_structure_evaluation(flow: ConversationFlow, target: float | None) -> "EvalResult | None":
    """Price every shortlist variant across the scenario grid, weight each under
    the baseline and the active PM-overlay context, and rank them. Pure compute —
    no rendering. Returns None when the flow has nothing to evaluate.

    render_structure_evaluation renders from this; the Batch pivot consumes the
    returned EvalResult (its ``variants`` + per-variant ``drivers``) directly.
    """
    if not (flow.market_state and flow.selector_result and flow.selector_result.shortlist and target is not None):
        return None

    from analytics.structure_pricer import PricedVariant as _PricedVariant, price_variants as _pv_fn
    from analytics.scenario_generator import generate_scenarios as _gen_sc
    from analytics.scenario_pricer import price_linear_scenarios as _price_linear_sc, price_scenarios as _price_sc
    from knowledge_engine.scenario_weighter import compute_family_weights as _compute_w
    from knowledge_engine.scenario_scorer import score_structure as _score_struct

    ms = flow.market_state
    is_call = flow.view.direction == "base_higher"
    move = abs(target - ms.fwd) / ms.fwd
    stop_pct = move / flow.target_rr
    stop = ms.fwd * (1 - stop_pct) if is_call else ms.fwd * (1 + stop_pct)
    loss_budget = LINEAR_NOTIONAL * stop_pct

    weighter = _compute_w(
        ms,
        primary_objective=getattr(flow, "primary_objective", "Balanced"),
        trade_management=getattr(flow, "trade_management", "Standard hold"),
    )
    weights = weighter.weights
    multipliers = weighter.multipliers
    base_fired = getattr(weighter, "base_fired", None)
    if base_fired is not None:
        base_weights = {
            cid: base_fired.multipliers[cid] / sum(base_fired.multipliers.values())
            for cid in base_fired.multipliers
        }
    else:
        base_weights = weights

    base_ccy = flow.view.pair[:3]
    inputs = {
        "spot": ms.spot,
        "forward": ms.fwd,
        "implied_vol": ms.vol,
        "tenor_years": ms.T,
        "target": target,
        "r_d": ms.r_d,
        "r_f": ms.r_f,
    }
    scenarios = _gen_sc(inputs)
    smile = _build_smile(flow)

    structs: list = []
    for item in flow.selector_result.shortlist:
        try:
            pvs = _pv_fn(
                ms, item.structure_id,
                target=target, is_call=is_call,
                stop_price=stop, loss_budget=loss_budget,
                smile=smile,
            )
        except Exception:
            continue
        if not pvs:
            continue
        variants = []
        for pv in pvs:
            rows = _price_sc(pv, item.structure_id, scenarios, inputs, is_call, surface=smile)
            variants.append({
                "pv": pv,
                "rows": rows,
                "score": _score_struct(rows, weights),
                "score_base": _score_struct(rows, base_weights),
            })
        if not variants:
            continue
        structs.append({"item": item, "variants": variants, "label": item.display_name})

    # Linear benchmark (delta-1, max-loss capped) — mirrors Trade View.
    linear_item = SimpleNamespace(structure_id="linear", display_name="Linear")
    linear_pv = _PricedVariant(
        variant_label="Delta 1 (max-loss capped)",
        strikes=[], barrier=None, net_premium_pct=0.0, breakeven=None,
        payoff_at_target_pct=None, rr_at_target=None, max_loss_pct=stop_pct,
        wing_ratio=None, is_zero_cost=True, structure_notional=LINEAR_NOTIONAL,
        net_premium_ccy=0.0, payoff_at_target_ccy=None, max_loss_ccy=loss_budget,
    )
    linear_rows = _price_linear_sc(scenarios, inputs, is_call, LINEAR_NOTIONAL, loss_budget)
    structs.append({
        "item": linear_item,
        "variants": [{
            "pv": linear_pv, "rows": linear_rows,
            "score": _score_struct(linear_rows, weights),
            "score_base": _score_struct(linear_rows, base_weights),
        }],
        "label": linear_item.display_name,
    })

    if not structs:
        return None

    # Active-context label (base + any overlays, first-match fallback).
    overlay_fired = getattr(weighter, "overlay_fired", [])
    fired_all = getattr(weighter, "fired", [])
    active_parts: list[str] = []
    if base_fired:
        active_parts.append(base_fired.id.replace("_", " ").title())
    if overlay_fired:
        active_parts.extend(c.id.replace("_", " ").title() for c in overlay_fired)
    if not active_parts and fired_all:
        active_parts.append(fired_all[0].id.replace("_", " ").title())
    active_ctx = " + ".join(active_parts) if active_parts else "Baseline grid"

    flat = [
        VariantEval(
            structure_id=s["item"].structure_id,
            struct_label=s["label"],
            variant_label=v["pv"].variant_label,
            pv=v["pv"], rows=v["rows"], score=v["score"], score_base=v["score_base"],
        )
        for s in structs for v in s["variants"]
    ]
    flat.sort(key=lambda ve: ve.score.score_ccy if ve.score.score_ccy is not None else 0.0, reverse=True)

    return EvalResult(
        ms=ms, is_call=is_call, target=target, stop=stop, loss_budget=loss_budget,
        inputs=inputs, scenarios=scenarios, smile=smile,
        weighter=weighter, weights=weights, multipliers=multipliers, base_weights=base_weights,
        base_fired=base_fired, overlay_fired=overlay_fired, fired_all=fired_all,
        active_ctx=active_ctx, base_ccy=base_ccy, structs=structs, variants=flat,
    )


def render_structure_evaluation(
    flow: ConversationFlow,
    is_admin: bool,
    target: float | None,
    key_prefix: str = "",
) -> None:
    _res = compute_structure_evaluation(flow, target)
    if _res is None:
        return

    from analytics.scenario_generator import (
        GRID_COLS as _SC_GRID_COLS,
        col_label as _sc_col_label,
        valid_grid_rows as _valid_grid_rows,
    )
    from conversation.explanation_context import (
        render_explanation_pack_overview as _render_expl_overview,
        render_structure_comparisons as _render_structure_comparisons,
        render_variant_comparisons as _render_variant_comparisons,
    )
    from knowledge_engine.comparator import (
        VariantEvaluation as _VariantEvaluation,
        build_recommendation_pack as _build_expl_pack,
        summarize_scenario_rows as _summarize_scenario_rows,
    )

    _ev_ms = _res.ms
    _ev_is_call = _res.is_call
    _ev_target = _res.target
    _ev_weighter = _res.weighter
    _ev_weights = _res.weights
    _ev_multipliers = _res.multipliers
    _ev_base_weights = _res.base_weights
    _ev_base = _res.base_ccy
    _ev_structs = _res.structs
    _base_fired = _res.base_fired
    _overlay_fired = _res.overlay_fired
    _fired_all = _res.fired_all
    _active_ctx = _res.active_ctx

    st.session_state["last_scenario_results"] = _ev_structs[-1]["variants"][-1]["rows"]

    st.subheader("Structure Evaluation")
    st.markdown(f"**Active scenario weighting:** {_active_ctx}")

    _carry_lbl = {0: "noisy", 1: "potential", 2: "high"}[_ev_ms.carry_regime]
    _dir_lbl = "with-carry" if _ev_ms.with_carry else "counter-carry"
    _tz_lbl = (
        f"target {abs(_ev_ms.target_z_spot):.2f}σ from spot ({abs(_ev_ms.target_z):.2f}σ from fwd)"
        if _ev_ms.target_z_spot is not None else "no target"
    )
    _tenor_days = int(round(_ev_ms.T * 365))
    _tenor_lbl = f"{_tenor_days}d tenor"
    _vol_lbl = f"vol {_ev_ms.vol:.1%}"
    st.caption(
        f"carry {_carry_lbl} ({_dir_lbl})  ·  {_tz_lbl}  ·  "
        f"{_tenor_lbl}  ·  {_vol_lbl}.  "
        "Scenario MtM as % of entry spot.  P&L vs entry premium."
    )

    _expander_label = f"Selected scenario grid — {_active_ctx}"
    with st.expander(_expander_label, expanded=False):
        _w_rows = []
        for _row in _valid_grid_rows(_ev_ms.T):
            for _col in _SC_GRID_COLS:
                _cid = f"{_row}|{_col}"
                if _cid not in _ev_multipliers:
                    continue
                _w_rows.append({
                    "Row": _row,
                    "Scenario": _sc_col_label(_col),
                    "Multiplier": f"{_ev_multipliers[_cid]:.1f}",
                    "Weight": f"{_ev_weights[_cid]:.1%}",
                })
        _show_df(
            pd.DataFrame(_w_rows),
            key=_df_key(key_prefix, "weights"),
            height=_fit_height(key_prefix, len(_w_rows)),
        )
        if _fired_all:
            _ctx_rows = [{
                "Layer": "Base" if _ctx == _base_fired else "Overlay",
                "Weighting": _ctx.id.replace("_", " "),
                "Reasoning": _ctx.comment,
            } for _ctx in _fired_all]
            _show_df(
                pd.DataFrame(_ctx_rows),
                key=_df_key(key_prefix, "ctx"),
                height=_fit_height(key_prefix, len(_ctx_rows)),
            )
        else:
            st.caption("No context-specific weighting active — the baseline grid applies unchanged.")

    if is_admin:
        try:
            _expl_variants = {
                _s["item"].structure_id: [_v["pv"] for _v in _s["variants"]]
                for _s in _ev_structs
            }
            _expl_scores = {
                _s["item"].structure_id: _s["variants"][0]["score"]
                for _s in _ev_structs
            }
            _expl_variant_evals = {
                _s["item"].structure_id: [
                    _VariantEvaluation(
                        variant=_v["pv"],
                        rows=_v["rows"],
                        base_score=_v["score_base"],
                        pm_score=_v["score"],
                        aggregates=_summarize_scenario_rows(_v["rows"]),
                    )
                    for _v in _s["variants"]
                ]
                for _s in _ev_structs
            }
            _expl_pack = _build_expl_pack(
                _ev_ms,
                flow.selector_result,
                _expl_variants,
                _expl_scores,
                variant_evaluations_by_structure=_expl_variant_evals,
            )
            from conversation.explanation_context import render_explanation_pack as _render_full_pack
            from interface.advisor_chat import build_chat_system_prompt as _build_chat_system
            _full_pack_text = _render_full_pack(_expl_pack)
            with st.expander("Full LLM prompt", expanded=False):
                st.code(_build_chat_system(_full_pack_text), language="text")
            with st.expander("Explanation pack preview", expanded=False):
                st.code(_render_expl_overview(_expl_pack), language="text")
                if _expl_pack.variant_comparisons:
                    with st.expander("Variant comparisons", expanded=False):
                        st.code(
                            _render_variant_comparisons(_expl_pack.variant_comparisons),
                            language="text",
                        )
                if _expl_pack.comparisons:
                    with st.expander("Structure comparisons", expanded=False):
                        st.code(
                            _render_structure_comparisons(list(_expl_pack.comparisons.values())),
                            language="text",
                        )
                if not _expl_pack.variant_comparisons and not _expl_pack.comparisons:
                    st.caption("No comparator sections available for this pack.")
        except Exception as _e:
            with st.expander("Explanation pack preview", expanded=False):
                st.caption(f"Unable to build explanation pack preview: {_e}")

    _all_ranked = sorted(
        [
            {"struct_label": _ev_s["label"], "item": _ev_s["item"], "ev_v": _ev_v}
            for _ev_s in _ev_structs
            for _ev_v in _ev_s["variants"]
        ],
        key=lambda x: x["ev_v"]["score"].score_ccy if x["ev_v"]["score"].score_ccy is not None else 0.0,
        reverse=True,
    )
    st.session_state["kelly_ranked_trade_rec_variants"] = _all_ranked
    for _rank_idx, _ranked_entry in enumerate(_all_ranked):
        _ev_v = _ranked_entry["ev_v"]
        _pv0 = _ev_v["pv"]
        _score = _ev_v["score"]
        _score_base = _ev_v["score_base"]
        _notional_str = (
            fmt_ccy(_pv0.structure_notional, _ev_base)
            if _pv0.structure_notional is not None else None
        )
        _variant_title = variant_label_with_strikes(_ranked_entry["item"].structure_id, _pv0)
        if _notional_str:
            _variant_title += f"  ·  Notional: {_notional_str}"
        elif _pv0.is_zero_cost and _pv0.max_loss_pct < 1e-9:
            _variant_title += "  ·  Notional: unscaled"
        _base_pct = f"{_score_base.score_pct:.2%}"
        _base_ccy_str = (
            f"  ({fmt_ccy_label(_score_base.score_ccy, _ev_base)})"
            if _score_base.score_ccy is not None
            else ("  (unscaled)" if _pv0.structure_notional is None else "")
        )
        _ctx_pct = f"{_score.score_pct:.2%}"
        _ctx_ccy_str = (
            f"  ({fmt_ccy_label(_score.score_ccy, _ev_base)})"
            if _score.score_ccy is not None
            else ("  (unscaled)" if _pv0.structure_notional is None else "")
        )
        _variant_title += (
            f"  ·  Scenario weighted P&L: {_base_pct}{_base_ccy_str}"
            f"  ·  PM overlay weighted P&L: {_ctx_pct}{_ctx_ccy_str}"
        )

        with st.expander(_variant_title, expanded=False):
            _bd_by_cell = {b.scenario_id: b for b in _score.cells}
            _summary_rows = []
            for _row in _valid_grid_rows(_ev_ms.T):
                for _col in _SC_GRID_COLS:
                    _cid = f"{_row}|{_col}"
                    _bd = _bd_by_cell.get(_cid)
                    if _bd is None:
                        continue
                    _summary_rows.append({
                        "Row": _row,
                        "Scenario": _sc_col_label(_col),
                        "P&L": f"{_bd.pnl_pct:+.2%}  ({fmt_ccy(_bd.pnl_ccy, _ev_base)})",
                        "Multiplier": f"{_bd.multiplier:.1f}",
                        "Weight": f"{_bd.normalized_weight:.1%}",
                        "Weighted contrib": (
                            f"{_bd.contrib_pct:+.2%}"
                            + (f"  ({fmt_ccy(_bd.contrib_ccy, _ev_base)})" if _bd.contrib_ccy is not None else "")
                        ),
                    })
            if _summary_rows:
                _show_df(
                    pd.DataFrame(_summary_rows),
                    key=_df_key(key_prefix, f"sum_{_rank_idx}"),
                    height=_fit_height(key_prefix, len(_summary_rows)),
                )

            with st.expander("Scenarios", expanded=False):
                _ev_by_row: dict[str, list] = {}
                for r in _ev_v["rows"]:
                    _ev_by_row.setdefault(r["row"], []).append(r)
                for _row in _valid_grid_rows(_ev_ms.T):
                    if _row not in _ev_by_row:
                        continue
                    _row_rows = sorted(_ev_by_row[_row], key=lambda x: _SC_GRID_COLS.index(x["col"]))
                    # Per-row roll-down forward = the no-move (S) cell's scenario forward,
                    # which decays toward spot as the horizon shrinks (carry reference).
                    _s_cell = next((x for x in _row_rows if x["col"] == "S"), None)
                    _fwd_lbl = f"  ·  fwd {_s_cell['scenario_fwd']:.4f}" if _s_cell else ""
                    st.markdown(f"**{_row}**{_fwd_lbl}")
                    _row_df = pd.DataFrame([{
                        "Scenario":  _sc_col_label(r["col"]),
                        "T%":        f"{r['time_fraction']:.0%}",
                        "Fwd":       f"{r['scenario_fwd']:.4f}",
                        "Spot":      f"{r['scenario_spot']:.4f}",
                        "Vol shift": r["vol_shift"] if isinstance(r["vol_shift"], str) else (f"{r['vol_shift']:+.0%}" if r["vol_shift"] != 0 else "—"),
                        "Vol":       f"{r['scenario_vol']:.1%}",
                        "Price":     f"{r['price_pct']:.2%}  ({fmt_ccy(r['price_ccy'], _ev_base)})",
                        "P&L":       f"{r['pnl_pct']:+.2%}  ({fmt_ccy(r['pnl_ccy'], _ev_base)})",
                    } for r in _row_rows])
                    _show_df(
                        _row_df,
                        key=_df_key(key_prefix, f"row_{_rank_idx}_{_row}"),
                        height=_fit_height(key_prefix, len(_row_df)),
                    )
