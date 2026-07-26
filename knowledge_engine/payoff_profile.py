"""Engine-authored terminal-payoff geometry for the agent narration.

The agent must never *author* payoff geometry (which side is the tail, where the
breakevens sit, does it pay above or below, is it path-dependent) — it gets the
direction wrong. This module computes those facts from the *actual priced legs* so
the agent relays them verbatim, exactly as it relays the per-leg breakdown and the
numbers.

Design — compute, don't template. Every vanilla-leg family has a piecewise-linear
terminal payoff with breakpoints at the strikes, so the profile is derived EXACTLY
(evaluate at the strike knots + one sentinel each side) rather than sampled — no
numerical error, and the tail direction / peak are premium- and spot-independent.
The binary (digital) and barrier (ERKO / RKO) families take explicit branches keyed
off strikes + barrier + the profile's ``path_dependent`` flag.

Pure module: geometry only. No scores, weights, thresholds, or scenario aggregates —
strictly less sensitive than the ``findings`` tags already shipped. Mirrors the house
pattern in ``structure_attributes.py`` (derive facts from numbers, render a fixed
vocabulary over them).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

_EPS = 1e-9

Tail = Literal["capped", "loss_upside", "loss_downside"]
Nature = Literal["expiry_only", "path_dependent", "binary_expiry"]
Flow = Literal["debit", "credit", "zero_cost"]

# Genuinely path-terminating barriers (checked on the path, not just at expiry). ERKO is
# NOT here — it is expiry-only (see pricing/european_rko.py). Both of these are enabled=false
# and only reachable via an off-menu price_structure request.
_PATH_DEPENDENT = {"rko", "european_digital_rko"}
_BINARY = {"european_digital", "european_digital_rko"}


@dataclass(frozen=True)
class PayoffProfile:
    value_region: str          # where it pays, e.g. "between 5.5694 and 6.1030"
    breakevens: tuple[float, ...]   # structural zero-crossings (already-shown-order numbers)
    max_payoff_where: str      # "at 5.9000", "uncapped on a further move up"
    tail: Tail
    tail_where: str | None     # "on a move above 6.1030"
    product_nature: Nature
    premium_flow: Flow
    pre_expiry_note: str | None = None   # FORK FLEX — reserved for a future pre-expiry / MtM
                                         # one-liner; None today, renderer emits only if set.


# --- leg input: a normalized tuple so the module needs no product_model import -----
# (notional_signed, strike, is_call)
Leg = tuple[float, float, bool]


def _premium_flow(net_premium_pct: float, is_zero_cost: bool) -> Flow:
    if is_zero_cost or abs(net_premium_pct) <= _EPS:
        return "zero_cost"
    return "credit" if net_premium_pct < 0 else "debit"


def _nature(structure_id: str) -> Nature:
    if structure_id in _PATH_DEPENDENT:
        return "path_dependent"
    if structure_id == "european_digital":
        return "binary_expiry"
    return "expiry_only"


def _terminal_value(legs: list[Leg], s: float) -> float:
    """Σ signed_notional · intrinsic(strike, s) — the same primitive the pricer sums
    (analytics/product_pricer._intrinsic). Piecewise-linear in s."""
    total = 0.0
    for notional, strike, is_call in legs:
        intrinsic = max(s - strike, 0.0) if is_call else max(strike - s, 0.0)
        total += notional * intrinsic
    return total


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def _crossings(pts: list[float], vals: list[float]) -> list[float]:
    """Linear-interpolated zero-crossings between consecutive evaluation points.
    Exact for a piecewise-linear payoff whose breakpoints are all in ``pts``."""
    out: list[float] = []
    for i in range(len(pts) - 1):
        a, b = vals[i], vals[i + 1]
        if (a < -_EPS and b > _EPS) or (a > _EPS and b < -_EPS):
            t = a / (a - b)
            out.append(pts[i] + t * (pts[i + 1] - pts[i]))
    return out


def _vanilla_profile(structure_id: str, legs: list[Leg], flow: Flow) -> PayoffProfile:
    knots = sorted({strike for _, strike, _ in legs})
    span = (knots[-1] - knots[0]) or knots[0] * 0.1
    pad = max(span, knots[0] * 0.05)
    lo, hi = knots[0] - 3 * pad, knots[-1] + 3 * pad

    # Dense grid for the positive-region scan; endpoints snap to the exact breakpoints
    # (knots ∪ crossings) so no numerical wobble reaches the rendered numbers.
    N = 800
    grid = [lo + (hi - lo) * i / N for i in range(N + 1)]
    gv = [_terminal_value(legs, s) for s in grid]

    slope_lo = (gv[1] - gv[0]) / (grid[1] - grid[0])
    slope_hi = (gv[-1] - gv[-2]) / (grid[-1] - grid[-2])
    crossings = _crossings(grid, gv)

    # peak of a piecewise-linear payoff is attained at a knot
    knot_vals = [(k, _terminal_value(legs, k)) for k in knots]
    peak_s = max(knot_vals, key=lambda kv: kv[1])[0]

    breakpoints = sorted(set(knots) | {round(c, 6) for c in crossings})

    def _snap(x: float) -> float:
        return min(breakpoints, key=lambda b: abs(b - x)) if breakpoints else x

    # --- tail: where does loss run unbounded? (premium-independent) ---
    tail: Tail = "capped"
    tail_where: str | None = None
    if slope_hi < -_EPS:
        tail = "loss_upside"
        edge = max([c for c in crossings if c >= peak_s - _EPS], default=knots[-1])
        tail_where = f"on a move above {_fmt(_snap(edge))}"
    elif slope_lo > _EPS:
        tail = "loss_downside"
        edge = min([c for c in crossings if c <= peak_s + _EPS], default=knots[0])
        tail_where = f"on a move below {_fmt(_snap(edge))}"

    # --- value region: the contiguous positive run bracketing the peak ---
    pk = min(range(len(grid)), key=lambda i: abs(grid[i] - peak_s))
    left = pk
    while left > 0 and gv[left - 1] > _EPS:
        left -= 1
    right = pk
    while right < len(grid) - 1 and gv[right + 1] > _EPS:
        right += 1
    unbounded_up = right >= len(grid) - 1 and slope_hi > _EPS
    unbounded_down = left <= 0 and slope_lo < -_EPS
    left_edge = _snap(grid[left])
    right_edge = _snap(grid[right])

    if unbounded_up and not unbounded_down:
        value_region = f"above {_fmt(left_edge)}"
    elif unbounded_down and not unbounded_up:
        value_region = f"below {_fmt(right_edge)}"
    elif right_edge - left_edge > _EPS:
        value_region = f"between {_fmt(left_edge)} and {_fmt(right_edge)}"
    else:
        value_region = f"around {_fmt(peak_s)}"

    if unbounded_up:
        max_payoff_where = "uncapped on a further move up"
    elif unbounded_down:
        max_payoff_where = "uncapped on a further move down"
    else:
        max_payoff_where = f"at {_fmt(peak_s)}"

    return PayoffProfile(
        value_region=value_region,
        breakevens=tuple(round(c, 6) for c in crossings),
        max_payoff_where=max_payoff_where,
        tail=tail,
        tail_where=tail_where,
        product_nature=_nature(structure_id),
        premium_flow=flow,
    )


def payoff_profile(
    structure_id: str,
    legs: list[Leg],
    *,
    net_premium_pct: float,
    is_zero_cost: bool,
    is_call: bool,
    strikes: list[float] | None = None,
    barrier: float | None = None,
) -> PayoffProfile | None:
    """Compute the terminal payoff profile for one priced structure.

    ``legs`` are (signed_notional, strike, is_call) tuples for vanilla-leg families.
    Binary / barrier families branch on ``structure_id`` and use ``strikes`` + ``barrier``.
    Returns None if the geometry can't be resolved (renderer then omits the PAYOFF line).
    """
    flow = _premium_flow(net_premium_pct, is_zero_cost)
    strikes = strikes or [k for _, k, _ in legs]
    if not strikes:
        return None
    k = strikes[0]

    # --- binary (digital) ---
    if structure_id in _BINARY:
        side = "above" if is_call else "below"
        return PayoffProfile(
            value_region=f"if spot finishes {side} {_fmt(k)} at expiry",
            breakevens=(),
            max_payoff_where="a fixed base-ccy payout (100% of notional at target)",
            tail="capped",
            tail_where=None,
            product_nature="path_dependent" if structure_id == "european_digital_rko" else "binary_expiry",
            premium_flow=flow,
        )

    # --- European reverse knock-out: vanilla payoff between strike and barrier, zero
    #     beyond it, tested at EXPIRY ONLY (not path-dependent) ---
    if structure_id == "european_rko" and barrier is not None:
        return PayoffProfile(
            value_region=(
                f"between the {_fmt(k)} strike and the {_fmt(barrier)} knock-out "
                f"(pays nothing if spot finishes beyond {_fmt(barrier)})"
            ),
            breakevens=(),
            max_payoff_where=f"near {_fmt(barrier)} (just short of the knock-out)",
            tail="capped",
            tail_where=None,
            product_nature="expiry_only",
            premium_flow=flow,
        )

    # --- path-dependent barrier (rko / edrko — enabled=false; off-menu only) ---
    if structure_id in _PATH_DEPENDENT and barrier is not None:
        side = "above" if is_call else "below"
        return PayoffProfile(
            value_region=f"like a vanilla {side} {_fmt(k)}, subject to the {_fmt(barrier)} barrier",
            breakevens=(),
            max_payoff_where=f"capped near the {_fmt(barrier)} barrier",
            tail="capped",
            tail_where=None,
            product_nature="path_dependent",
            premium_flow=flow,
        )

    # --- vanilla-leg families (piecewise-linear, computed exactly) ---
    if not legs:
        return None
    return _vanilla_profile(structure_id, legs, flow)


# --- rendering -------------------------------------------------------------
_FLOW_PROSE = {
    "debit": "net debit (you pay the premium)",
    "credit": "net credit (you receive the premium)",
    "zero_cost": "zero-cost",
}
_NATURE_PROSE = {
    "expiry_only": "settles on the expiry level only (not path-dependent)",
    "path_dependent": "path-dependent — the barrier can terminate it before expiry",
    "binary_expiry": "binary — pays a fixed amount if it finishes in the money at expiry",
}


def render_payoff(p: PayoffProfile, indent: str = "     ") -> str:
    """One labelled PAYOFF line. Every number here is one already printed elsewhere in
    the pack (strikes / barrier); this only connects them, so no number is minted."""
    parts = [f"pays {p.value_region}", f"best {p.max_payoff_where}", _FLOW_PROSE[p.premium_flow]]
    if p.tail == "capped":
        parts.append("loss is capped (worst case = the max loss shown)")
    elif p.tail_where:
        parts.append(f"turns increasingly negative {p.tail_where} (uncapped)")
    parts.append(_NATURE_PROSE[p.product_nature])
    if p.pre_expiry_note:
        parts.append(p.pre_expiry_note)
    return f"{indent}PAYOFF: " + "; ".join(parts) + "."
