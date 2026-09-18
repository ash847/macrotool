"""Sizing specification + Kelly notional math (pure, engine-layer, no IO).

Two sizing methods feed the single `_size_variant` seam in `structure_pricer`:

  - **fixed_loss** — every variant scaled to the same max loss (`loss_budget`).
  - **kelly** — each variant sized to its growth-optimal bet under the PM's
    distribution: `N = λ · x* · W`, where `x*` maximises `Σ p·ln(1 + x·π)` over
    the per-unit-notional P&L `π`.

The Kelly fraction here uses a **per-notional** return basis (`π = DF·payoff −
net_premium`), not the premium-basis `(payoff−cost)/cost` of
`interface/kelly_v2/kelly.py`. That generalises to spreads / zero-cost / net-credit
structures (no division by premium) and the ruin bound caps tail-risky leverage —
the principled replacement for the deferred "size on scenario worst-case loss" fix.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize_scalar

# Conviction → weight toward the target (vs the forward) for the view-implied seed.
_CONVICTION_WEIGHT = {"low": 0.33, "medium": 0.66, "high": 1.0}

DEFAULT_BANKROLL = 100.0          # nominal W; == interface LINEAR_NOTIONAL for scale continuity
DEFAULT_KELLY_LAMBDA = 0.5        # fractional-Kelly multiplier (full Kelly is fragile)
_F_MAX = 1000.0                   # search ceiling for x = notional/bankroll; real cap applied downstream


@dataclass(frozen=True)
class SizingSpec:
    """How to size variants. Defaults reproduce today's fixed-loss behaviour exactly,
    so every existing caller/test is unchanged."""
    method: Literal["fixed_loss", "kelly"] = "fixed_loss"
    target_rr: float = 3.0
    kelly_lambda: float = DEFAULT_KELLY_LAMBDA
    bankroll: float = DEFAULT_BANKROLL
    # PM distribution over terminal spot (the edge). Stored as plain arrays so the
    # engine layer needs no dependency on the UI's Distribution type.
    kelly_probs: tuple[float, ...] | None = None
    kelly_bins: tuple[float, ...] | None = None

    def has_distribution(self) -> bool:
        return self.kelly_probs is not None and self.kelly_bins is not None


def per_notional_pnl(
    payoff_at_bins: np.ndarray,
    net_premium_pct: float,
    discount_factor: float = 1.0,
) -> np.ndarray:
    """Per-unit-notional P&L (base-ccy fraction) at each distribution bin:
    `π(S) = DF · payoff(S) − net_premium_pct`. `payoff_at_bins` is the terminal
    base-ccy payoff per unit notional (from `analytics.payoffs`)."""
    return discount_factor * np.asarray(payoff_at_bins, dtype=float) - net_premium_pct


def kelly_fraction_per_notional(
    probs,
    pnl_per_notional,
    f_max: float = _F_MAX,
) -> float:
    """`x* = argmax_x Σ p·ln(1 + x·π)`, where `x = notional / bankroll`.

    Returns `x* ≥ 0`. No edge (`E[π] ≤ 0`) → 0. The ruin bound `1 + x·π_min > 0`
    caps leverage when the worst outcome is a loss; with no loss outcome the search
    runs to `f_max` (the downstream notional cap then binds)."""
    probs = np.asarray(probs, dtype=float)
    pnl = np.asarray(pnl_per_notional, dtype=float)
    if probs.size == 0 or pnl.size != probs.size:
        return 0.0
    e_pnl = float(np.dot(probs, pnl))
    if e_pnl <= 0.0:
        return 0.0
    pmin = float(pnl.min())
    upper = min(f_max, 1.0 / (-pmin) - 1e-9) if pmin < 0 else f_max
    if upper <= 0.0:
        return 0.0

    def neg_log_growth(x: float) -> float:
        terms = 1.0 + x * pnl
        if np.any(terms <= 0.0):
            return np.inf
        return -float(np.dot(probs, np.log(terms)))

    res = minimize_scalar(neg_log_growth, bounds=(0.0, upper), method="bounded", options={"xatol": 1e-7})
    return max(0.0, float(res.x))


def kelly_notional(probs, pnl_per_notional, spec: SizingSpec, cap: float) -> float:
    """Comparative Kelly notional: `min(λ · x* · W, cap)`."""
    x_star = kelly_fraction_per_notional(probs, pnl_per_notional)
    return min(spec.kelly_lambda * x_star * spec.bankroll, cap)


def view_implied_distribution(
    spot: float,
    fwd: float,
    vol: float,
    T: float,
    target: float,
    conviction: str = "medium",
    n_bins: int = 41,
    sigma_extent: float = 4.0,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """A real-world (edged) terminal-spot distribution synthesised from the PM's view.

    A lognormal whose log-centre is blended between the forward and the target by
    conviction (`high` → fully at the target), with width `vol·√T`. Because the
    centre sits off the forward, it carries directional edge — so Kelly will size a
    bet (unlike the risk-neutral/market-implied distribution, which has zero edge).
    Used as the low-friction default seed and for the Batch ranking sweep. Returns
    `(probs, bins)` as plain tuples — the shape `SizingSpec` carries."""
    if target <= 0 or fwd <= 0 or spot <= 0:
        raise ValueError("spot/fwd/target must be positive")
    w = _CONVICTION_WEIGHT.get(conviction, 0.66)
    sig = max(vol * math.sqrt(max(T, 1e-9)), 1e-6)
    ln_center = (1.0 - w) * math.log(fwd) + w * math.log(target)
    bins = np.linspace(
        math.exp(ln_center - sigma_extent * sig),
        math.exp(ln_center + sigma_extent * sig),
        n_bins,
    )
    z = (np.log(bins) - ln_center) / sig
    dens = np.exp(-0.5 * z * z) / bins            # lognormal pdf shape
    probs = dens / dens.sum()
    return tuple(float(p) for p in probs), tuple(float(b) for b in bins)
