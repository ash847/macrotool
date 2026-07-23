"""
Batch page — run several trades through the deterministic engine at once and show
the same analytics as Trade View (ranked Structure variants + full Structure
Evaluation), grouped per trade in a collapsible section.

Input is a JSON file (`interface/batch_trades.json`), NOT an on-screen field. Each
trade is a plain string `"<tenor> <pair> <target-level>"`, e.g. "3m USDBRL 5.60".
Constraints are defaults (No restriction / Balanced / Standard hold), R:R = 3.0.
Direction is inferred from the target level vs the horizon forward. No LLM pack.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import streamlit as st

import pandas as pd

from interface.structure_eval import (
    LINEAR_NOTIONAL,
    sizing_capital,
    DRIVER_BUCKETS,
    target_price,
    compute_structure_evaluation,
    render_structure_variants,
    render_structure_evaluation,
)
from knowledge_engine.models import TradeView
from pricing.forwards import rate_context_for_snapshot

BATCH_FILE = Path(__file__).parent / "batch_trades.json"

_TENOR_RE = re.compile(r"^(\d+(?:\.\d+)?)([mwyMWY])$")


def _tenor_to_days(tok: str) -> int | None:
    m = _TENOR_RE.match(tok)
    if not m:
        return None
    n = float(m.group(1))
    unit = m.group(2).lower()
    if unit == "m":
        return round(n * 365 / 12)
    if unit == "w":
        return round(n * 7)
    if unit == "y":
        return round(n * 365)
    return None


def _clean_trades(trades) -> list[str]:
    if not isinstance(trades, list):
        return []
    return [str(t).strip() for t in trades if str(t).strip()]


def _load_batches() -> tuple[list[tuple[str, list[str]]], str | None]:
    """Load the named batches from the JSON file as an ordered list of (name, trades).

    Canonical format: {"batches": [{"name": "...", "trades": ["3m USDBRL 5.60", ...]}]}.
    Also accepts {"batches": {"name": [...]}} and the legacy single-batch forms
    ({"trades": [...]} or a bare list), wrapped as one batch named "Batch".
    """
    if not BATCH_FILE.exists():
        return [], f"Batch file not found: `{BATCH_FILE}`"
    try:
        data = json.loads(BATCH_FILE.read_text())
    except Exception as e:
        return [], f"Could not parse `{BATCH_FILE.name}`: {e}"

    batches: list[tuple[str, list[str]]] = []
    if isinstance(data, dict) and isinstance(data.get("batches"), list):
        for i, b in enumerate(data["batches"]):
            if isinstance(b, dict):
                name = str(b.get("name") or f"Batch {i + 1}").strip()
                batches.append((name, _clean_trades(b.get("trades", []))))
    elif isinstance(data, dict) and isinstance(data.get("batches"), dict):
        for name, trades in data["batches"].items():
            batches.append((str(name).strip(), _clean_trades(trades)))
    elif isinstance(data, dict) and "trades" in data:          # legacy single batch
        batches.append(("Batch", _clean_trades(data.get("trades", []))))
    elif isinstance(data, list):                                # legacy bare list
        batches.append(("Batch", _clean_trades(data)))
    else:
        return [], 'Batch JSON must define "batches" (list of {name, trades}) or a "trades" list.'

    batches = [(n, t) for (n, t) in batches if t]  # drop empty/unnamed-empty batches
    if not batches:
        return [], "No non-empty batches found in the JSON."
    # De-dup names (later ones get a suffix) so the selectbox keys stay unique.
    seen: dict[str, int] = {}
    out: list[tuple[str, list[str]]] = []
    for name, trades in batches:
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 1
        out.append((name, trades))
    return out, None


def _parse_trade(s: str, snapshot) -> tuple[str, int, float]:
    """Parse '<tenor> <pair> <target>' (any order) → (pair, horizon_days, target)."""
    toks = s.replace(",", " ").split()
    pairs_upper = {k.upper(): k for k in snapshot.currencies.keys()}
    pair = horizon_days = target = None
    for t in toks:
        tu = t.upper()
        if pair is None and tu in pairs_upper:
            pair = pairs_upper[tu]
            continue
        if horizon_days is None and (d := _tenor_to_days(t)) is not None:
            horizon_days = d
            continue
        if target is None:
            try:
                target = float(t)
                continue
            except ValueError:
                pass
    if pair is None:
        raise ValueError("no recognised pair (e.g. USDBRL)")
    if horizon_days is None:
        raise ValueError("no recognised tenor (e.g. 3m / 6m / 1y)")
    if target is None:
        raise ValueError("no target level")
    return pair, horizon_days, target


def _build_and_run(trade_str: str, make_flow, snapshot, user_email: str | None = None):
    """Parse + run one trade. Returns a populated ConversationFlow. Raises on error."""
    pair, horizon_days, target = _parse_trade(trade_str, snapshot)
    ccy = snapshot.get(pair)
    if ccy is None:
        raise ValueError(f"pair {pair} not in snapshot")

    rate_ctx = rate_context_for_snapshot(ccy, horizon_days / 365.0)
    fwd = rate_ctx.forward
    if abs(target / fwd - 1.0) < 1e-9:
        raise ValueError(f"target {target} equals the {horizon_days}d forward {fwd:.4f}")
    direction = "base_higher" if target > fwd else "base_lower"
    magnitude_pct = abs(target / fwd - 1.0) * 100.0

    flow = make_flow()
    flow.view = TradeView(
        pair=pair,
        direction=direction,
        direction_conviction="medium",
        horizon_days=horizon_days,
        magnitude_pct=magnitude_pct,
    )
    flow.ccy = ccy
    flow.structure_constraint = "No restriction"
    flow.primary_objective = "Balanced"
    flow.trade_management = "Standard hold"
    flow.target_rr = 3.0
    flow.user_email = user_email  # reflect the logged-in user's weights profile
    flow._run_engines()
    return flow


def _run_batch(trades: list[str], make_flow, snapshot, user_email: str | None = None) -> list[dict]:
    results: list[dict] = []
    for raw in trades:
        entry: dict = {"title": raw, "flow": None, "error": None}
        try:
            entry["flow"] = _build_and_run(raw, make_flow, snapshot, user_email=user_email)
        except Exception as e:  # one bad trade must not kill the batch
            entry["error"] = str(e)
        results.append(entry)
    return results


def _compute_batch_evals(results: list[dict]) -> list:
    """Price + score every trade ONCE, right after the batch runs. Returned list is
    index-aligned with `results` (None for failed / empty trades). Both the pivot
    and the per-trade detail render from this cache, so Streamlit reruns (e.g.
    toggling a pivot radio) don't re-price the whole batch."""
    evals: list = []
    for r in results:
        flow = r.get("flow")
        if r.get("error") or flow is None:
            evals.append(None)
            continue
        evals.append(compute_structure_evaluation(flow, target_price(flow)))
    return evals


def _build_pivot_rows(results: list[dict], evals: list) -> list[dict]:
    """One row per (trade × variant) across the whole batch, for the cross-trade
    pivot. Each row carries the weighting effect (baseline / context / Δ), the
    P&L-driver decomposition, and a stable variant identity that aligns the same
    variant across trades. Gated-out variants simply don't appear for that trade."""
    rows: list[dict] = []
    for r, res in zip(results, evals):
        if res is None:
            continue
        for rank, ve in enumerate(res.variants, start=1):
            d = ve.drivers
            rows.append({
                "trade": r["title"],
                "context": res.active_ctx,
                # display label (struct · variant) and stable cross-trade key
                "variant": f"{ve.struct_label} · {ve.variant_label}",
                "rank": rank,
                "ctx_pnl": ve.score_pct,
                "base_pnl": ve.score_base_pct,
                "delta": ve.delta_pct,
                "carry": d["Carry"],
                "directional": d["Directional"],
                "adverse": d["Adverse"],
                "vega": d["Vega"],
                "premium": ve.pv.net_premium_pct,
                "payoff": ve.pv.payoff_at_target_pct,
            })
    return rows


# Flat-table column layout: (source key, display label, is_percent-points).
_FLAT_COLS: list[tuple[str, str, bool]] = [
    ("trade", "Trade", False),
    ("variant", "Variant", False),
    ("context", "Context", False),
    ("rank", "Rank", False),
    ("base_pnl", "Base P&L %", True),
    ("ctx_pnl", "Ctx P&L %", True),
    ("delta", "Δ %", True),
    ("carry", "Carry %", True),
    ("directional", "Direction %", True),
    ("adverse", "Adverse %", True),
    ("vega", "Vega %", True),
    ("premium", "Premium %", True),
    ("payoff", "Payoff %", True),
]


def _render_pivot(results: list[dict], evals: list) -> None:
    rows = _build_pivot_rows(results, evals)
    if not rows:
        st.info("No priced variants to pivot across the batch.")
        return
    df = pd.DataFrame(rows)

    # Context-routing tally — verifies first-match weighting selection across the set.
    ctx_counts = df.drop_duplicates("trade")["context"].value_counts()
    st.caption(
        "**Context routing:**  "
        + "   ·   ".join(f"{ctx} ({n})" for ctx, n in ctx_counts.items())
    )
    st.caption(
        "P&L-driver buckets:  "
        + "  ·  ".join(f"**{b}** = {'/'.join(cols)}" for b, cols in DRIVER_BUCKETS.items())
        + ".  Buckets sum to Ctx P&L.  Δ = Ctx − Base (the weighting effect)."
    )

    mode = st.radio(
        "Pivot",
        ["Flat (all variants)", "By variant", "Best per trade"],
        horizontal=True,
        key="batch_pivot_mode",
    )

    if mode == "By variant":
        metric = st.radio(
            "Cell metric",
            ["Rank", "Ctx P&L", "Δ (Ctx−Base)"],
            horizontal=True,
            key="batch_pivot_metric",
        )
        src = {"Rank": "rank", "Ctx P&L": "ctx_pnl", "Δ (Ctx−Base)": "delta"}[metric]
        mat = df.pivot_table(index="variant", columns="trade", values=src, aggfunc="first")

        def _fmt(v: float) -> str:
            if pd.isna(v):
                return "—"  # variant gated out / ineligible for this trade
            return f"{int(v)}" if metric == "Rank" else f"{v * 100:+.2f}"

        st.dataframe(mat.map(_fmt), use_container_width=True)
        st.caption("`—` = variant gated out / ineligible for that trade (not a low rank).")
        return

    src_df = df if mode == "Flat (all variants)" else df[df["rank"] == 1]
    out = pd.DataFrame()
    for key, label, is_pct in _FLAT_COLS:
        col = src_df[key]
        out[label] = (col * 100.0) if is_pct else col
    out = out.sort_values(["Trade", "Rank"]).reset_index(drop=True)

    col_config = {
        label: st.column_config.NumberColumn(label, format="%+.2f")
        for _key, label, is_pct in _FLAT_COLS if is_pct
    }
    st.dataframe(
        out, use_container_width=True, hide_index=True, column_config=col_config
    )


def _render_trade_analytics(flow, is_admin: bool, key_prefix: str, eval_result=None) -> None:
    ms = flow.market_state
    if not (ms and flow.selector_result and flow.selector_result.shortlist):
        st.warning("No eligible structures for this trade.")
        return

    is_call = flow.view.direction == "base_higher"
    target = target_price(flow)
    if target is None:
        st.warning("No target level resolved.")
        return

    # Compact market read (mirrors the top of Trade View).
    pair = flow.view.pair
    tz = f"{ms.target_z:+.2f}σ" if ms.target_z is not None else "—"
    st.caption(
        f"**{pair}** {'Long' if is_call else 'Short'} · {flow.view.horizon_days}d · "
        f"spot {ms.spot:.4f} · fwd {ms.fwd:.4f} · target {target:.4f} · "
        f"target_z(fwd) {tz} · carry regime {ms.carry_regime}"
    )
    if ms.target_z is not None and abs(ms.target_z) < 0.25:
        st.info("Target is < 0.25σ from the forward — a small move to structure around.")

    move_pct = abs(target / ms.fwd - 1.0)
    stop_pct = move_pct / flow.target_rr
    stop_price = ms.fwd * (1 - stop_pct) if is_call else ms.fwd * (1 + stop_pct)
    loss_budget = sizing_capital() * stop_pct

    # Context-weighted scenario P&L per variant (base ccy), keyed by stable identity,
    # so the variants table can show it as a column.
    _spnl_map = None
    if eval_result is not None:
        _spnl_map = {
            (v.structure_id, v.variant_label): v.score.score_ccy
            for v in eval_result.variants
        }

    render_structure_variants(
        flow, is_call, target, stop_price, loss_budget,
        key_prefix=key_prefix, scenario_pnl=_spnl_map,
    )
    render_structure_evaluation(flow, is_admin, target, key_prefix=key_prefix, eval_result=eval_result)


def render(make_flow, snapshot, is_admin: bool, user_email: str | None = None) -> None:
    st.header("Batch")
    batches, load_err = _load_batches()
    if load_err:
        st.error(load_err)
        st.caption(
            "Define batches in `interface/batch_trades.json`, e.g. "
            '`{"batches": [{"name": "USDBRL sweep", "trades": ["3m USDBRL 5.60", ...]}]}`.'
        )
        return

    by_name = dict(batches)
    names = [n for n, _ in batches]
    selected_name = st.selectbox(
        "Batch", names, key="batch_select",
        help="Batches are defined in interface/batch_trades.json. Pick one, then Run.",
    )
    trades = by_name[selected_name]

    st.caption(
        f"**{selected_name}** — {len(trades)} trade(s) · constraints = defaults "
        "(No restriction / Balanced / Standard hold) · R:R = 3.0 · no LLM pack."
    )
    with st.expander(f"Trades in “{selected_name}”", expanded=False):
        for _t in trades:
            st.caption(f"• {_t}")

    run = st.button("Run batch", type="primary")

    # Results are cached per batch name, so switching batches shows that batch's last
    # run (or a prompt to run) and never re-runs on its own. Running (re)computes the
    # selected batch. Nothing runs until the button is pressed.
    cache: dict = st.session_state.setdefault("batch_runs", {})
    if run:
        with st.spinner(f"Running “{selected_name}” — {len(trades)} trade(s)…"):
            results = _run_batch(trades, make_flow, snapshot, user_email=user_email)
            cache[selected_name] = {"results": results, "evals": _compute_batch_evals(results)}

    entry = cache.get(selected_name)
    if entry is None:
        st.info(f"Press **Run batch** to evaluate “{selected_name}”.")
        return

    results = entry["results"]
    evals = entry["evals"]
    if not results:
        st.info("This batch has no trades to run.")
        return

    n_ok = sum(1 for r in results if r["error"] is None)
    st.caption(f"“{selected_name}” — {n_ok}/{len(results)} ran cleanly.")

    st.subheader("Cross-trade pivot")
    _render_pivot(results, evals)

    st.subheader("Per-trade detail")
    for _idx, (r, res) in enumerate(zip(results, evals)):
        label = r["title"] + ("   ❌" if r["error"] else "")
        with st.expander(label, expanded=False):
            if r["error"]:
                st.error(f"Could not run this trade: {r['error']}")
                continue
            _render_trade_analytics(r["flow"], is_admin, key_prefix=f"batch{_idx}_", eval_result=res)
