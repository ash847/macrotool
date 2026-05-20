"""
Kelly fraction for a single option position priced against PM's elicited
distribution. Two solvers:

  * `kelly_continuous` — Thorp closed-form `f* ≈ E[r] / Var[r]`. Fast,
    intuitive, but a Taylor expansion that breaks down when |r| ≈ 1.
  * `kelly_discrete` — 1-D numerical maximisation of `E[log(1 + f·r)]`.
    Canonical answer; respects the lower bound r = -1 (total premium loss).

Both operate on the **Full edge** trade economics. Sized on PM's own
elicited distribution — truncation effects flow through naturally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from elicitation import Distribution
from pricing import PayoffFn


SAFETY_F_MAX: float = 0.999  # keeps log(1 + f·(-1)) finite for total-loss outcomes
MIN_VARIANCE: float = 1e-12


@dataclass(frozen=True)
class KellyReport:
    f_continuous: float        # Thorp closed-form
    f_discrete: float          # numerical max of E[log(1+f·r)]
    f_raw: float               # whichever solver we use as the "Kelly answer"
    f_displayed: float         # after multiplier and cap
    multiplier: float
    position_cap: float
    expected_return: float     # E[r]
    variance: float            # Var[r]
    prob_loss: float           # P(r < 0)
    prob_total_loss: float     # P(r = -1) — option expires worthless
    expected_log_growth: float # E[log(1 + f_displayed · r)] — the thing Kelly maximises


def _returns(
    dist: Distribution,
    payoff: PayoffFn,
    cost: float,
    discount_factor: float,
) -> np.ndarray:
    """Per-unit-capital return at each bin: r = (DF · payoff(S) − cost) / cost."""
    if cost <= 0:
        raise ValueError(f"cost must be positive; got {cost}")
    return (discount_factor * payoff(dist.bins) - cost) / cost


def kelly_continuous(
    dist: Distribution,
    payoff: PayoffFn,
    cost: float,
    discount_factor: float = 1.0,
) -> float:
    """Thorp closed-form approximation: `f* ≈ E[r] / Var[r]`."""
    r = _returns(dist, payoff, cost, discount_factor)
    e_r = float(np.dot(dist.probs, r))
    if e_r <= 0:
        return 0.0
    var_r = float(np.dot(dist.probs, (r - e_r) ** 2))
    if var_r <= MIN_VARIANCE:
        # Deterministic positive return — Kelly would say bet everything.
        # Long-option leverage bound is just under 1; report that.
        return SAFETY_F_MAX
    return max(0.0, e_r / var_r)


def kelly_discrete(
    dist: Distribution,
    payoff: PayoffFn,
    cost: float,
    discount_factor: float = 1.0,
    f_max: float = SAFETY_F_MAX,
) -> float:
    """Numerically maximise E[log(1 + f·r)] over f ∈ [0, f_max]."""
    r = _returns(dist, payoff, cost, discount_factor)
    e_r = float(np.dot(dist.probs, r))
    if e_r <= 0:
        return 0.0

    # If the worst-case return is > -1 (e.g. all outcomes in-the-money), the
    # log term is finite at f = 1 and we can raise the search ceiling.
    r_min = float(r.min())
    upper = min(f_max, SAFETY_F_MAX) if r_min <= -1.0 + 1e-9 else f_max

    def neg_log_growth(f: float) -> float:
        terms = 1.0 + f * r
        # Guard against log(0) when f hits the boundary.
        if np.any(terms <= 0):
            return np.inf
        return -float(np.dot(dist.probs, np.log(terms)))

    result = minimize_scalar(
        neg_log_growth,
        bounds=(0.0, upper),
        method="bounded",
        options={"xatol": 1e-6},
    )
    return max(0.0, float(result.x))


def _expected_log_growth(
    dist: Distribution,
    payoff: PayoffFn,
    cost: float,
    discount_factor: float,
    f: float,
) -> float:
    if f <= 0.0:
        return 0.0
    r = _returns(dist, payoff, cost, discount_factor)
    terms = 1.0 + f * r
    if np.any(terms <= 0):
        return float("-inf")
    return float(np.dot(dist.probs, np.log(terms)))


def compute_kelly(
    dist: Distribution,
    payoff: PayoffFn,
    cost: float,
    discount_factor: float = 1.0,
    multiplier: float = 0.5,
    position_cap: float = 0.20,
) -> KellyReport:
    """Build the full Kelly report. `multiplier` ∈ (0, 1], `position_cap` ∈ (0, 1]."""
    if not (0.0 < multiplier <= 1.0):
        raise ValueError(f"multiplier must lie in (0, 1]; got {multiplier}")
    if not (0.0 < position_cap <= 1.0):
        raise ValueError(f"position_cap must lie in (0, 1]; got {position_cap}")

    f_c = kelly_continuous(dist, payoff, cost, discount_factor)
    f_d = kelly_discrete(dist, payoff, cost, discount_factor)

    # The "Kelly answer" we use for sizing is the discrete (numerical) solver —
    # the continuous one is shown alongside as a diagnostic.
    f_raw = f_d
    f_displayed = min(f_raw * multiplier, position_cap)

    r = _returns(dist, payoff, cost, discount_factor)
    e_r = float(np.dot(dist.probs, r))
    var_r = float(np.dot(dist.probs, (r - e_r) ** 2))
    prob_loss = float(dist.probs[r < 0].sum())
    prob_total_loss = float(dist.probs[r <= -1.0 + 1e-9].sum())

    expected_log_growth = _expected_log_growth(
        dist, payoff, cost, discount_factor, f_displayed
    )

    return KellyReport(
        f_continuous=f_c,
        f_discrete=f_d,
        f_raw=f_raw,
        f_displayed=f_displayed,
        multiplier=multiplier,
        position_cap=position_cap,
        expected_return=e_r,
        variance=var_r,
        prob_loss=prob_loss,
        prob_total_loss=prob_total_loss,
        expected_log_growth=expected_log_growth,
    )
