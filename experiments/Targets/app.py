"""
FX View Structuring Tool — Streamlit UI.
Run: streamlit run experiments/Targets/app.py
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import plotly.graph_objects as go
from calc import (
    horizon_vol,
    make_buckets,
    init_p_from_q,
    make_zone_buckets,
    init_zone_p,
)

st.set_page_config(
    page_title="FX View Structuring",
    page_icon="📐",
    layout="wide",
)

# ── Header ──────────────────────────────────────────────────────────────────
st.title("FX View Structuring")
st.caption(
    "Convert your view into a probability distribution. "
    "The shape tells you which option structure fits."
)

with st.expander("Testing guidelines", expanded=True):
    st.markdown(
        "Does this make it easy for you to describe a view? "
        "Is there some information we are not capturing through this interface?"
    )

st.divider()

# ── Market inputs ────────────────────────────────────────────────────────────
ic = st.columns([0.8, 1.1, 1.1, 1.1, 1.1, 1.1])
pair    = ic[0].text_input("Pair", "EURUSD")
spot    = ic[1].number_input("Spot",    value=1.0350, format="%.4f", step=0.0001, min_value=0.0001)
forward = ic[2].number_input("Forward", value=1.0380, format="%.4f", step=0.0001, min_value=0.0001)
vol_pct = ic[3].number_input("IV (%)",  value=9.0, step=0.1, min_value=0.1, max_value=150.0)
tenor_m = ic[4].number_input("Tenor (months)", value=3, min_value=1, max_value=36, step=1)
target  = ic[5].number_input("Target",  value=1.0500, format="%.4f", step=0.0001, min_value=0.0001)

vol     = vol_pct / 100.0
T       = tenor_m / 12.0
sigma_T = horizon_vol(vol, T)
move_pct = (target / forward - 1) * 100
move_sigma = math.log(target / forward) / sigma_T if sigma_T > 0 else 0.0

if target > forward * 1.002:
    direction = "⬆ Upside"
elif target < forward * 0.998:
    direction = "⬇ Downside"
else:
    direction = "↔ Neutral / Range"

st.caption(
    f"**{pair}** &nbsp;|&nbsp; {direction} &nbsp;|&nbsp; "
    f"Horizon vol: **{sigma_T*100:.2f}%** &nbsp;|&nbsp; "
    f"Move: **{move_pct:+.2f}%** ({move_sigma:+.2f}σ)"
)

st.divider()

# ── Bucket generation ────────────────────────────────────────────────────────
buckets = make_buckets(forward, sigma_T)
n = len(buckets)

# Index of the bucket that contains the target (for bold label)
target_bucket_idx = next(
    (i for i, b in enumerate(buckets)
     if (b["lower"] is None or b["lower"] <= target)
     and (b["upper"] is None or target < b["upper"])),
    -1,
)

# ── Session state: initialise / reset when market inputs change ──────────────
setup_key = f"{forward:.6f}_{target:.6f}_{sigma_T:.8f}"

if st.session_state.get("_setup_key") != setup_key:
    st.session_state["_setup_key"] = setup_key
    for i, v in enumerate(init_p_from_q(buckets)):
        st.session_state[f"p_{i}"] = v

# ── Read current P values (used for chart drawn before sliders) ──────────────
p_default = init_p_from_q(buckets)
p_current = [st.session_state.get(f"p_{i}", p_default[i]) for i in range(n)]

# ── Distribution chart ───────────────────────────────────────────────────────
short_labels = [b["short"] for b in buckets]
q_vals       = p_default          # rounded integers — matches P init so bars are equal on load

# Bold the tick label for the bucket that contains the target
tick_labels = [
    f"<b>{lbl} ◀</b>" if i == target_bucket_idx else lbl
    for i, lbl in enumerate(short_labels)
]

fig = go.Figure()
fig.add_trace(go.Bar(
    name="Market (Q)",
    x=short_labels,
    y=q_vals,
    marker_color="rgba(150,150,150,0.45)",
    marker_line_color="rgba(130,130,130,0.9)",
    marker_line_width=1,
    hovertemplate="%{x}<br>Market: %{y:.1f}%<extra></extra>",
))
fig.add_trace(go.Bar(
    name="Your view (P)",
    x=short_labels,
    y=p_current,
    marker_color="rgba(30,100,220,0.55)",
    marker_line_color="rgba(20,80,200,0.9)",
    marker_line_width=1,
    hovertemplate="%{x}<br>Your view: %{y:.0f}%<extra></extra>",
))
fig.update_layout(
    barmode="group",
    height=280,
    margin=dict(t=10, b=10, l=0, r=0),
    yaxis_title="Probability (%)",
    xaxis_title=None,
    legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    yaxis=dict(gridcolor="rgba(200,200,200,0.4)", range=[0, max(max(q_vals), max(p_current)) * 1.2 + 2]),
    xaxis=dict(tickangle=-20, ticktext=tick_labels, tickvals=short_labels),
    font=dict(size=12),
)
st.plotly_chart(fig, use_container_width=True)

# ── Probability sliders ──────────────────────────────────────────────────────

def _decrement(i: int) -> None:
    st.session_state[f"p_{i}"] = max(0, st.session_state[f"p_{i}"] - 1)

def _increment(i: int) -> None:
    st.session_state[f"p_{i}"] = min(100, st.session_state[f"p_{i}"] + 1)

def _reset_to_market() -> None:
    for idx, val in enumerate(init_p_from_q(buckets)):
        st.session_state[f"p_{idx}"] = val

st.markdown("**Assign probabilities** — 1% increments, must sum to 100%")

slider_cols = st.columns(n)
p_values = []
for i, (col, b) in enumerate(zip(slider_cols, buckets)):
    with col:
        v = st.slider(
            label=b["short"],
            min_value=0,
            max_value=100,
            step=1,
            key=f"p_{i}",
        )
        p_values.append(v)
        btn_l, btn_r = st.columns(2)
        btn_l.button("−", key=f"dec_{i}", on_click=_decrement, args=(i,), use_container_width=True)
        btn_r.button("+", key=f"inc_{i}", on_click=_increment, args=(i,), use_container_width=True)

total = sum(p_values)

sum_col, btn_col, _ = st.columns([2, 2, 6])
with sum_col:
    if total == 100:
        st.success(f"Total: {total}% ✓")
    else:
        st.error(f"Total: {total}% — needs to be 100%")
with btn_col:
    st.button("↺ Reset to market", on_click=_reset_to_market)

st.divider()

# ── Target interpretation ────────────────────────────────────────────────────
st.subheader("Target interpretation")

target_mode = st.radio(
    "My target is like a:",
    ["Threshold", "Zone", "Waypoint"],
    index=1,
    horizontal=True,
    captions=[
        "I care about finishing above/below it",
        "It's the most likely landing area",
        "Directional — overshoot is fine",
    ],
)

# ── Zone sub-distribution ────────────────────────────────────────────────────
if target_mode == "Zone":
    st.divider()

    # Extremity check: buckets 0 and 7 are open-ended tails
    in_tail = target_bucket_idx in (0, len(buckets) - 1)

    if in_tail:
        st.caption("Target is in a tail bucket — zone zoom not available.")
    else:
        parent_bucket = buckets[target_bucket_idx]
        parent_p = p_values[target_bucket_idx]

        zone_buckets = make_zone_buckets(parent_bucket, forward, sigma_T)
        nz = len(zone_buckets)

        # Session state: reset when parent bucket, parent probability, or inputs change
        zone_key = f"z_{target_bucket_idx}_{parent_p}_{forward:.6f}_{sigma_T:.8f}"
        if st.session_state.get("_zone_key") != zone_key:
            st.session_state["_zone_key"] = zone_key
            for i, v in enumerate(init_zone_p(zone_buckets, parent_p)):
                st.session_state[f"z_{i}"] = v

        # Read current zone values for chart
        z_default = init_zone_p(zone_buckets, parent_p)
        z_current = [st.session_state.get(f"z_{i}", z_default[i]) for i in range(nz)]

        # Zone chart
        z_labels = [b["short"] for b in zone_buckets]
        zq_vals  = z_default      # rounded integers — matches P init so bars are equal on load

        # Mark which sub-bucket contains the target
        target_z_idx = next(
            (i for i, b in enumerate(zone_buckets) if b["lower"] <= target < b["upper"]),
            -1,
        )
        z_tick_labels = [
            f"<b>{lbl} ◀</b>" if i == target_z_idx else lbl
            for i, lbl in enumerate(z_labels)
        ]

        zfig = go.Figure()
        zfig.add_trace(go.Bar(
            name="Market (Q)",
            x=z_labels,
            y=zq_vals,
            marker_color="rgba(150,150,150,0.45)",
            marker_line_color="rgba(130,130,130,0.9)",
            marker_line_width=1,
            hovertemplate="%{x}<br>Market: %{y:.2f}%<extra></extra>",
        ))
        zfig.add_trace(go.Bar(
            name="Your view (P)",
            x=z_labels,
            y=z_current,
            marker_color="rgba(30,100,220,0.55)",
            marker_line_color="rgba(20,80,200,0.9)",
            marker_line_width=1,
            hovertemplate="%{x}<br>Your view: %{y:.0f}%<extra></extra>",
        ))
        y_max = max(max(zq_vals), max(z_current), 1) * 1.25 + 0.5
        zfig.update_layout(
            barmode="group",
            height=240,
            margin=dict(t=10, b=10, l=0, r=0),
            yaxis_title="Probability (%)",
            xaxis_title=None,
            legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis=dict(gridcolor="rgba(200,200,200,0.4)", range=[0, y_max]),
            xaxis=dict(tickangle=-20, ticktext=z_tick_labels, tickvals=z_labels),
            font=dict(size=12),
        )

        st.caption(
            f"Zoom: **{parent_bucket['label']}** — "
            f"allocated **{parent_p}%** in the chart above"
        )
        st.plotly_chart(zfig, use_container_width=True)

        # Zone sub-sliders
        def _z_decrement(i: int) -> None:
            st.session_state[f"z_{i}"] = max(0, st.session_state[f"z_{i}"] - 1)

        def _z_increment(i: int) -> None:
            st.session_state[f"z_{i}"] = min(parent_p, st.session_state[f"z_{i}"] + 1)

        def _reset_zone() -> None:
            for idx, val in enumerate(init_zone_p(zone_buckets, parent_p)):
                st.session_state[f"z_{idx}"] = val

        zcols = st.columns(nz)
        z_values = []
        for i, (col, zb) in enumerate(zip(zcols, zone_buckets)):
            with col:
                v = st.slider(
                    label=zb["short"],
                    min_value=0,
                    max_value=max(1, parent_p),
                    step=1,
                    key=f"z_{i}",
                )
                z_values.append(v)
                zb_l, zb_r = st.columns(2)
                zb_l.button("−", key=f"zdec_{i}", on_click=_z_decrement, args=(i,), use_container_width=True)
                zb_r.button("+", key=f"zinc_{i}", on_click=_z_increment, args=(i,), use_container_width=True)

        z_total = sum(z_values)
        zs_col, zb_col, _ = st.columns([2, 2, 6])
        with zs_col:
            if z_total == parent_p:
                st.success(f"Zone total: {z_total}% ✓")
            else:
                st.error(f"Zone total: {z_total}% — must equal {parent_p}%")
        with zb_col:
            st.button("↺ Reset zone", on_click=_reset_zone, key="reset_zone")
