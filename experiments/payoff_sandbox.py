"""Localhost sandbox for the payoff chart — NOT part of the app.

No login, no Supabase, no LLM. Runs the real deterministic engine (`build_pack`) on a
hard-coded view and draws the interactive expiry payoff chart for whichever recommended
structure you pick, with the engine's own numbers alongside so you can eyeball that the
chart and the table agree.

    .venv/bin/streamlit run experiments/payoff_sandbox.py

Natural-language entry is deliberately absent for now: first get the chart interaction
right on a fixed view, then bolt the typed-view step on (it would feed `VIEW` below).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from agentic.standard_pack import build_pack
from analytics.payoff_curve import curves_for_pack
from config.loader import load_config
from data.snapshot_loader import load_snapshot
from interface.payoff_chart import PLOTLY_CONFIG, payoff_figure
from knowledge_engine.models import TradeView

# The hard-coded starting view. Edit freely.
VIEW = dict(pair="USDBRL", direction="base_higher", horizon_days=91, magnitude_pct=4.0)

st.set_page_config(page_title="Payoff chart sandbox", layout="wide")


@st.cache_resource(show_spinner="Running the engine…")
def _snapshot_and_cfg():
    return load_snapshot(), load_config()


@st.cache_resource(show_spinner="Building the standard pack…")
def _pack(pair: str, direction: str, horizon_days: int, magnitude_pct: float):
    snap, cfg = _snapshot_and_cfg()
    view = TradeView(
        pair=pair, direction=direction, direction_conviction="high",
        horizon_days=horizon_days, magnitude_pct=magnitude_pct, mode="recommend",
    )
    pack = build_pack(view, snap.get(pair), cfg)
    return pack, curves_for_pack(pack, pair)


snap, _ = _snapshot_and_cfg()
pairs = list(snap.currencies.keys())

st.title("Payoff chart sandbox")
st.caption("Fixed view → real engine → interactive expiry payoff. Branch `feature/payoff-chart-sandbox`.")

with st.sidebar:
    st.header("View")
    pair = st.selectbox("Pair", pairs, index=pairs.index(VIEW["pair"]) if VIEW["pair"] in pairs else 0)
    direction = st.radio(
        "Direction (vs base ccy)", ["base_higher", "base_lower"],
        index=0 if VIEW["direction"] == "base_higher" else 1,
    )
    horizon_days = st.number_input("Horizon (days)", 7, 730, VIEW["horizon_days"], step=7)
    magnitude_pct = st.number_input("Target move from forward (%)", 0.5, 30.0, VIEW["magnitude_pct"], step=0.5)

pack, curves = _pack(pair, direction, int(horizon_days), float(magnitude_pct))
ms = pack.market_state

c1, c2, c3, c4 = st.columns(4)
c1.metric("Spot", f"{ms.spot:.4f}")
c2.metric("Forward", f"{ms.fwd:.4f}")
c3.metric("Target", f"{pack.target:.4f}" if pack.target else "—")
c4.metric("ATM vol", f"{ms.vol:.2%}")

if not pack.recommended:
    st.warning("The engine returned no recommended structures for this view.")
    st.stop()


def _label(i: int) -> str:
    r = pack.recommended[i]
    ok = curves[i] is not None and curves[i].ok
    return f"#{r.rank}  {r.display_name} · {r.variant.variant_label}" + ("" if ok else "  (chart unavailable)")


left, right = st.columns([3, 1])
with right:
    idx = st.selectbox("Structure", range(len(pack.recommended)), format_func=_label)
    has_notional = curves[idx] is not None and curves[idx].notional is not None
    units_choice = st.radio(
        "Units", ["% of notional", f"{pair[:3]} at sized notional"], index=0,
        disabled=not has_notional, help="Same curve, rescaled. Notional = the pack's sized notional.",
    )
    units = "pct" if units_choice.startswith("%") else "notional"

rec, curve = pack.recommended[idx], curves[idx]
v = rec.variant

with left:
    fig = payoff_figure(curve, units=units, sigma_sqrt_t=ms.vol * ms.T ** 0.5)
    if fig is None:
        st.error(
            "Chart withheld — it did not reconcile with the engine. "
            + ("; ".join(f"{c.name}: {c.detail}" for c in curve.failures) if curve else "No curve could be built.")
        )
    else:
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        st.caption(
            "Expiry P&L net of premium, base-ccy basis (payoff ÷ terminal spot − premium). "
            "Drag to zoom either axis · scroll to zoom · double-click to reset. "
            f"Sizing regime: {curve.sizing_method.replace('_', ' ')}."
        )

with right:
    st.subheader("Engine numbers")
    st.markdown(
        f"- **Premium**: {v.net_premium_pct:+.3%} of notional "
        f"({'debit' if v.net_premium_pct > 0 else 'credit' if v.net_premium_pct < 0 else 'zero-cost'})\n"
        f"- **Max loss (engine)**: {v.max_loss_pct:.3%}\n"
        f"- **Payoff at target (engine)**: "
        f"{'—' if v.payoff_at_target_pct is None else f'{v.payoff_at_target_pct:.3%}'}\n"
        f"- **Breakeven (engine)**: {'—' if v.breakeven is None else f'{v.breakeven:.4f}'}\n"
        f"- **Sized notional**: {'—' if v.structure_notional is None else f'{v.structure_notional:,.1f} {pair[:3]}'}\n"
        f"- **Strikes**: {', '.join(f'{k:.4f}' for k in v.strikes)}"
        + (f"\n- **Barrier**: {v.barrier:.4f}" if v.barrier is not None else "")
    )

if curve is not None:
    st.divider()
    st.subheader("Chart self-checks")
    for c in curve.checks:
        mark = "✅" if c.ok else ("❌" if c.severity == "fail" else "⚠️")
        st.markdown(f"{mark} **{c.name}** — {c.detail}")
