"""Compact tester-surface rendering for the 'Trade view' nav page.

Shows a top-5 table (structure, strikes, notional, premium) followed by a
short streamed LLM commentary explaining the regime and why the ranking makes
sense. Cached per trade parameters so reruns don't re-call the API.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from interface.structure_eval import (
    LINEAR_NOTIONAL,
    _KELLY_RISK_HELP,
    compute_structure_evaluation,
    fmt_ccy,
)

_SYSTEM = (
    "You are a concise EM FX options strategist writing for a fund PM. "
    "Write exactly 2–3 sentences. No bullet points, no headers, no markdown "
    "formatting, no invented numbers. Every claim must follow directly from "
    "the inputs provided."
)

_INSTR = (
    "Write 2–3 sentences: "
    "(1) characterise the regime briefly in plain terms, "
    "(2) explain what it implies for structure selection, "
    "(3) say in one concrete phrase why the top-ranked structure fits better "
    "than the alternatives shown."
)


def _stream_commentary(api_key: str, prompt: str):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text


def _build_prompt(ev, ms, flow) -> str:
    from knowledge_engine.scenario_weighter import get_context_commentary

    base_fired = ev.base_fired
    ctx_id = getattr(base_fired, "id", None)
    comm = get_context_commentary(ctx_id) if ctx_id else {}
    ctx_name = ctx_id.replace("_", " ").title() if ctx_id else "Baseline"

    pair = flow.view.pair
    base_ccy, quote_ccy = pair[:3], pair[3:]
    dir_label = f"{base_ccy} higher vs {quote_ccy}" if ev.is_call else f"{base_ccy} lower vs {quote_ccy}"
    carry_lbl = {0: "noisy (regime 0)", 1: "moderate carry (regime 1)", 2: "high carry (regime 2)"}[ms.carry_regime]
    with_carry_lbl = "with-carry" if ms.with_carry else "counter-carry"
    target_lbl = f"{abs(ms.target_z_spot):.1f}σ from spot" if ms.target_z_spot is not None else "no target"

    top5 = ev.variants[:5]
    struct_lines: list[str] = []
    for i, ve in enumerate(top5, 1):
        pv = ve.pv
        strikes_str = " / ".join(f"{k:.4f}" for k in pv.strikes) if pv.strikes else "—"
        notional_str = f"{base_ccy} {pv.structure_notional:,.0f}" if pv.structure_notional else "—"
        prem_str = f"{pv.net_premium_pct:+.2%}"
        struct_lines.append(
            f"{i}. {ve.struct_label} ({ve.variant_label}) | "
            f"strikes {strikes_str} | notional {notional_str} | prem {prem_str}"
        )

    parts = [
        f"PAIR: {pair} | DIRECTION: {dir_label} | {with_carry_lbl}",
        f"REGIME: {ctx_name}",
    ]
    if comm.get("market_behavior"):
        parts.append(f"MARKET: {comm['market_behavior']}")
    if comm.get("trade_guidance"):
        parts.append(f"GUIDANCE: {comm['trade_guidance']}")
    parts += [
        f"CARRY: c = {ms.c:+.3f} | {carry_lbl} | VOL: {ms.vol:.0%} ATM | TARGET: {target_lbl}",
        "",
        f"TOP {len(top5)} STRUCTURES (scenario-weighted P&L, highest first):",
        *struct_lines,
        "",
        _INSTR,
    ]
    return "\n".join(parts)


def _render_regime_summary(ev, ms, flow, target: float) -> None:
    """The regime text summary (streamed LLM commentary, cached per trade)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return
    cache_key = (
        f"tester_comm_{flow.view.pair}_{flow.view.horizon_days}"
        f"_{flow.view.direction}_{target:.4f}"
    )
    cached = st.session_state.get(cache_key)
    if cached:
        st.markdown(cached)
        return
    prompt = _build_prompt(ev, ms, flow)
    chunks: list[str] = []
    container = st.empty()
    with st.spinner("Reading the regime…"):
        for chunk in _stream_commentary(api_key, prompt):
            chunks.append(chunk)
            container.markdown("".join(chunks))
    if chunks:
        st.session_state[cache_key] = "".join(chunks)


def _render_shortlist(ms, flow) -> None:
    """Top-3 shortlisted structures by affinity score, as names + score expressed as a
    % of the maximum possible affinity score. No strikes — this is the family-level fit
    ranking (distinct from the priced 'Top structures' table below)."""
    from knowledge_engine.structure_scorer import get_scoring_detail, max_possible_score

    detail = get_scoring_detail(
        ms, structure_constraint=getattr(flow, "structure_constraint", "No restriction")
    )
    primaries = [r for r in detail if not r["overlay_only"] and r["eligible"]][:3]
    if not primaries:
        return
    ceiling = max_possible_score() or 1.0

    st.subheader("Shortlisted structures")
    st.caption("Ranked by structure-fit score (affinity), shown as a % of the maximum "
               "possible score. Family-level — strikes are in the table below.")
    rows = []
    for i, r in enumerate(primaries, 1):
        pct = max(0.0, min(100.0, 100.0 * (r["total_score"] or 0.0) / ceiling))
        rows.append({"#": i, "Structure": r["display_name"], "Fit score": f"{pct:.0f}%"})
    st.dataframe(pd.DataFrame(rows).set_index("#"), use_container_width=True)


def _render_priced_table(ev) -> None:
    """The priced 'Top structures' table (with strikes), scenario-weighted P&L order."""
    base_ccy = ev.base_ccy
    rows = []
    for i, ve in enumerate(ev.variants[:5], 1):
        pv = ve.pv
        row = {
            "#": i,
            "Structure": ve.struct_label,
            "Variant": ve.variant_label,
            "Strikes": " / ".join(f"{k:.4f}" for k in pv.strikes) if pv.strikes else "—",
            "Notional": fmt_ccy(pv.structure_notional, base_ccy),
            "Premium": f"{pv.net_premium_pct:+.2%}",
        }
        if getattr(pv, "kelly_fraction", None) is not None:
            row["Kelly risk"] = f"{pv.kelly_fraction * (pv.max_loss_pct or 0.0):.0%}"
        rows.append(row)
    st.subheader("Top structures")
    st.caption("Priced variants with strikes, ordered by scenario-weighted P&L. "
               "Kelly risk (when shown) is full-Kelly capital at risk (pre-λ) as a share of W.")
    _cfg = ({"Kelly risk": st.column_config.Column(help=_KELLY_RISK_HELP)}
            if any("Kelly risk" in r for r in rows) else None)
    st.dataframe(pd.DataFrame(rows).set_index("#"), use_container_width=True, column_config=_cfg)


def render_tester_recommendations(flow, is_call: bool, target: float | None) -> None:
    """Tester Trade View output, in workflow order: regime text summary → affinity
    shortlist (names + % of max score) → priced table with strikes. Market state is
    rendered above this by the app."""
    if target is None:
        return
    ev = compute_structure_evaluation(flow, target)
    if ev is None or not ev.variants:
        return

    _render_regime_summary(ev, ev.ms, flow, target)
    _render_shortlist(ev.ms, flow)
    _render_priced_table(ev)
