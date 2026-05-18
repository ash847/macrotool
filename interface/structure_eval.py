"""Structure evaluation rendering for the Trade View page.

Contains shared currency/label helpers, the structure variants block, and the
full scenario-weighted evaluation block with advisor pack preview.
"""

from __future__ import annotations

import re
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


def variant_label_with_strikes(structure_id: str, pv) -> str:
    strikes = [f"{k:.4f}" for k in pv.strikes]
    label = pv.variant_label

    if structure_id == "vanilla" and strikes:
        return f"{label} {strikes[0]}"

    if structure_id == "1x1_spread" and len(strikes) >= 2:
        parts = label.split("/")
        if len(parts) == 2:
            return f"{parts[0].strip()} {strikes[0]} / {parts[1].strip()} {strikes[1]}"

    if structure_id == "seagull" and len(strikes) >= 3:
        if " + " in label:
            spread_part, wing_part = label.split(" + ", 1)
            spread_bits = spread_part.split("/")
            if len(spread_bits) == 2:
                return (
                    f"{spread_bits[0].strip()} {strikes[0]} / "
                    f"{spread_bits[1].strip()} {strikes[1]} + "
                    f"{wing_part.strip()} {strikes[2]}"
                )

    if structure_id in {"1x1.5_spread", "1x2_spread"} and len(strikes) >= 2:
        if " / " in label:
            left, right = label.split(" / ", 1)
            return f"{left.strip()} {strikes[0]} / {right.strip()} {strikes[1]}"

    if structure_id == "european_rko" and strikes:
        ko = f"{pv.barrier:.4f}" if pv.barrier is not None else "—"
        return f"{label} {strikes[0]}  ·  KO at {ko}"

    if structure_id == "european_digital" and strikes:
        return f"{label} {strikes[0]}"

    if structure_id == "european_digital_rko" and strikes:
        american_barrier = f"{pv.barrier:.4f}" if pv.barrier is not None else "—"
        return (
            f"{label}  ·  European barrier {strikes[0]}  ·  "
            f"American barrier {american_barrier}"
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

def render_structure_variants(
    flow: ConversationFlow,
    is_call: bool,
    target: float | None,
    stop_price: float | None,
    loss_budget: float | None,
) -> None:
    from analytics.structure_pricer import price_variants as _price_variants

    ms = flow.market_state
    _base_ccy = flow.view.pair[:3]
    _primary_items = flow.selector_result.shortlist

    _any_variants = any(
        _price_variants(ms, s.structure_id, target=target, is_call=is_call, stop_price=stop_price, loss_budget=loss_budget)
        for s in _primary_items
    )
    if not _any_variants:
        return

    st.subheader("Structure variants")
    st.caption(
        "Indicative pricing — flat ATM vol for all strikes. "
        "Premium and payoff as % of spot. "
        "**Payout/$1**: gross payoff at target per $1 of max loss (zero-cost seagull: "
        "loss on short wing at stop price, expiry basis — understates MtM risk before expiry)."
    )

    for _i, _item in enumerate(_primary_items):
        try:
            _pvs = _price_variants(ms, _item.structure_id, target=target, is_call=is_call, stop_price=stop_price, loss_budget=loss_budget)
        except Exception as _e:
            st.caption(f"DEBUG {_item.structure_id}: error — {_e}")
            continue
        if not _pvs:
            continue
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
                    "Variant":    pv.variant_label,
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
            st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Structure evaluation (scenario-weighted P&L tables)
# ---------------------------------------------------------------------------

def render_structure_evaluation(
    flow: ConversationFlow,
    is_admin: bool,
    target: float | None,
) -> None:
    if not (flow.market_state and flow.selector_result and flow.selector_result.shortlist and target is not None):
        return

    _ev_ms = flow.market_state
    _ev_is_call = flow.view.direction == "base_higher"
    _ev_target = target
    _ev_move = abs(_ev_target - _ev_ms.fwd) / _ev_ms.fwd
    _ev_stop_pct = _ev_move / flow.target_rr
    _ev_stop = _ev_ms.fwd * (1 - _ev_stop_pct) if _ev_is_call else _ev_ms.fwd * (1 + _ev_stop_pct)
    _ev_loss_budget = LINEAR_NOTIONAL * _ev_stop_pct

    from analytics.structure_pricer import PricedVariant as _PricedVariant, price_variants as _pv_fn
    from analytics.scenario_generator import (
        GRID_COLS as _SC_GRID_COLS,
        generate_scenarios as _gen_sc,
        valid_grid_rows as _valid_grid_rows,
    )
    from analytics.scenario_pricer import price_linear_scenarios as _price_linear_sc, price_scenarios as _price_sc
    from knowledge_engine.scenario_weighter import compute_family_weights as _compute_w
    from knowledge_engine.scenario_scorer  import score_structure       as _score_struct
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

    _ev_weighter = _compute_w(
        _ev_ms,
        primary_objective=getattr(flow, "primary_objective", "Balanced"),
        trade_management=getattr(flow, "trade_management", "Standard hold"),
    )
    _ev_weights  = _ev_weighter.weights
    _ev_multipliers = _ev_weighter.multipliers
    _ev_base_fired = getattr(_ev_weighter, "base_fired", None)
    if _ev_base_fired is not None:
        _ev_base_weights = {
            _cid: _ev_base_fired.multipliers[_cid] / sum(_ev_base_fired.multipliers.values())
            for _cid in _ev_base_fired.multipliers
        }
    else:
        _ev_base_weights = _ev_weights

    _ev_base = flow.view.pair[:3]

    _ev_inputs = {
        "spot": _ev_ms.spot,
        "forward": _ev_ms.fwd,
        "implied_vol": _ev_ms.vol,
        "tenor_years": _ev_ms.T,
        "target": _ev_target,
        "r_d": _ev_ms.r_d,
        "r_f": _ev_ms.r_f,
    }
    _ev_scenarios = _gen_sc(_ev_inputs)

    _ev_structs = []
    for _ev_item in flow.selector_result.shortlist:
        try:
            _ev_pvs = _pv_fn(
                _ev_ms, _ev_item.structure_id,
                target=_ev_target, is_call=_ev_is_call,
                stop_price=_ev_stop, loss_budget=_ev_loss_budget,
            )
        except Exception:
            continue
        if not _ev_pvs:
            continue
        _ev_variants = []
        for _ev_pv in _ev_pvs:
            _ev_rows = _price_sc(
                _ev_pv, _ev_item.structure_id, _ev_scenarios, _ev_inputs, _ev_is_call
            )
            _ev_score = _score_struct(_ev_rows, _ev_weights)
            _ev_score_base = _score_struct(_ev_rows, _ev_base_weights)
            _ev_variants.append({
                "pv": _ev_pv,
                "rows": _ev_rows,
                "score": _ev_score,
                "score_base": _ev_score_base,
            })
        if not _ev_variants:
            continue
        _ev_structs.append({
            "item":     _ev_item,
            "variants": _ev_variants,
            "label":    _ev_item.display_name,
        })

    _linear_item = SimpleNamespace(structure_id="linear", display_name="Linear")
    _linear_pv = _PricedVariant(
        variant_label="Delta 1 (max-loss capped)",
        strikes=[],
        barrier=None,
        net_premium_pct=0.0,
        breakeven=None,
        payoff_at_target_pct=None,
        rr_at_target=None,
        max_loss_pct=_ev_stop_pct,
        wing_ratio=None,
        is_zero_cost=True,
        structure_notional=LINEAR_NOTIONAL,
        net_premium_ccy=0.0,
        payoff_at_target_ccy=None,
        max_loss_ccy=_ev_loss_budget,
    )
    _linear_rows = _price_linear_sc(
        _ev_scenarios,
        _ev_inputs,
        _ev_is_call,
        LINEAR_NOTIONAL,
        _ev_loss_budget,
    )
    _linear_score = _score_struct(_linear_rows, _ev_weights)
    _linear_score_base = _score_struct(_linear_rows, _ev_base_weights)
    _ev_structs.append({
        "item": _linear_item,
        "variants": [{
            "pv": _linear_pv,
            "rows": _linear_rows,
            "score": _linear_score,
            "score_base": _linear_score_base,
        }],
        "label": _linear_item.display_name,
    })

    if not _ev_structs:
        return

    st.session_state["last_scenario_results"] = _ev_structs[-1]["variants"][-1]["rows"]

    st.subheader("Structure Evaluation")

    _base_fired = getattr(_ev_weighter, "base_fired", None)
    _overlay_fired = getattr(_ev_weighter, "overlay_fired", [])
    _fired_all = getattr(_ev_weighter, "fired", [])
    _active_parts = []
    if _base_fired:
        _active_parts.append(_base_fired.id.replace("_", " ").title())
    if _overlay_fired:
        _active_parts.extend(_ctx.id.replace("_", " ").title() for _ctx in _overlay_fired)
    if not _active_parts and _fired_all:
        _active_parts.append(_fired_all[0].id.replace("_", " ").title())
    _active_ctx = " + ".join(_active_parts) if _active_parts else "Baseline grid"
    st.markdown(f"**Active scenario weighting:** {_active_ctx}")

    _carry_lbl = {0: "noisy", 1: "potential", 2: "high"}[_ev_ms.carry_regime]
    _dir_lbl = "with-carry" if _ev_ms.with_carry else "counter-carry"
    _tz_lbl = (
        f"target {abs(_ev_ms.target_z):.2f}σ from forward"
        if _ev_ms.target_z is not None else "no target"
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
                    "Scenario": _col,
                    "Multiplier": f"{_ev_multipliers[_cid]:.1f}",
                    "Weight": f"{_ev_weights[_cid]:.1%}",
                })
        st.dataframe(pd.DataFrame(_w_rows), use_container_width=True, hide_index=True)
        if _fired_all:
            _ctx_rows = [{
                "Layer": "Base" if _ctx == _base_fired else "Overlay",
                "Weighting": _ctx.id.replace("_", " "),
                "Reasoning": _ctx.comment,
            } for _ctx in _fired_all]
            st.dataframe(pd.DataFrame(_ctx_rows), use_container_width=True, hide_index=True)
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
    for _ranked_entry in _all_ranked:
        _ev_v = _ranked_entry["ev_v"]
        _pv0 = _ev_v["pv"]
        _score = _ev_v["score"]
        _score_base = _ev_v["score_base"]
        _notional_str = (
            fmt_ccy(_pv0.structure_notional, _ev_base)
            if _pv0.structure_notional is not None else None
        )
        _variant_title = _ranked_entry["struct_label"]
        _variant_title += "  ·  " + variant_label_with_strikes(_ranked_entry["item"].structure_id, _pv0)
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
                        "Scenario": _col,
                        "P&L": f"{_bd.pnl_pct:+.2%}  ({fmt_ccy(_bd.pnl_ccy, _ev_base)})",
                        "Multiplier": f"{_bd.multiplier:.1f}",
                        "Weight": f"{_bd.normalized_weight:.1%}",
                        "Weighted contrib": (
                            f"{_bd.contrib_pct:+.2%}"
                            + (f"  ({fmt_ccy(_bd.contrib_ccy, _ev_base)})" if _bd.contrib_ccy is not None else "")
                        ),
                    })
            if _summary_rows:
                st.dataframe(pd.DataFrame(_summary_rows), use_container_width=True, hide_index=True)

            with st.expander("Scenarios", expanded=False):
                _ev_by_row: dict[str, list] = {}
                for r in _ev_v["rows"]:
                    _ev_by_row.setdefault(r["row"], []).append(r)
                for _row in _valid_grid_rows(_ev_ms.T):
                    if _row not in _ev_by_row:
                        continue
                    st.markdown(f"**{_row}**")
                    _row_rows = sorted(_ev_by_row[_row], key=lambda x: _SC_GRID_COLS.index(x["col"]))
                    _row_df = pd.DataFrame([{
                        "Scenario":  r["col"],
                        "T%":        f"{r['time_fraction']:.0%}",
                        "Fwd":       f"{r['scenario_fwd']:.4f}",
                        "Spot":      f"{r['scenario_spot']:.4f}",
                        "Vol shift": r["vol_shift"] if isinstance(r["vol_shift"], str) else (f"{r['vol_shift']:+.0%}" if r["vol_shift"] != 0 else "—"),
                        "Vol":       f"{r['scenario_vol']:.1%}",
                        "Price":     f"{r['price_pct']:.2%}  ({fmt_ccy(r['price_ccy'], _ev_base)})",
                        "P&L":       f"{r['pnl_pct']:+.2%}  ({fmt_ccy(r['pnl_ccy'], _ev_base)})",
                    } for r in _row_rows])
                    st.dataframe(_row_df, use_container_width=True, hide_index=True)
