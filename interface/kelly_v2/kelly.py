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

from .elicitation import Distribution
from .pricing import PayoffFn


SAFETY_F_MAX: float = 0.999  # keeps log(1 + f·(-1)) finite for total-loss outcomes
MIN_VARIANCE: float = 1e-12
# If r_min is more negative than this the structure has material downside beyond
# the distribution's truncated support — Kelly is unreliable and we decline.
UNBOUNDED_LOSS_THRESHOLD: float = -10.0


@dataclass(frozen=True)
class KellyReport:
    f_continuous: float        # Thorp closed-form
    f_discrete: float          # numerical max of E[log(1+f·r)]
    f_raw: float               # whichever solver we use as the "Kelly answer"
    f_displayed: float         # after multiplier
    multiplier: float
    expected_return: float     # E[r]
    variance: float            # Var[r]
    prob_loss: float           # P(r < 0)
    prob_total_loss: float     # P(r = -1) — option expires worthless
    expected_log_growth: float # E[log(1 + f_displayed · r)] — the thing Kelly maximises
    unbounded_loss: bool       # True if r_min << -1 → Kelly declined (all f_* are 0)


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

    # Set the search ceiling so that log(1 + f·r_min) never hits log(0).
    # For long options r_min = -1 exactly → upper = SAFETY_F_MAX (< 1).
    # For structures with r_min < -1 (loss exceeds premium) → upper = 1/|r_min| - ε.
    # For structures with r_min > -1 (all bins ITM) → upper = f_max unconstrained.
    r_min = float(r.min())
    if r_min <= -1.0 + 1e-9:
        upper = min(f_max, 1.0 / (-r_min) - 1e-9)
    else:
        upper = f_max

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


def kelly_growth_curve(
    dist: Distribution,
    payoff: PayoffFn,
    cost: float,
    discount_factor: float = 1.0,
    f_star: float = 0.0,
    n_points: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (f_values, geo_growth, er_curve) for plotting the Kelly growth curve.

    geo_growth: exp(E[log(1+f·r)]) − 1 — geometric (compounding) return per trade.
      This is what Kelly maximises. Peaks at f_star then falls back through zero.
    er_curve:   f · E[r] — expected portfolio return. Linear in f, always above
      geo_growth past small f because it ignores the variance drag.

    The gap between the two lines is the variance drag — the visual reason why
    chasing expected return oversizes and destroys long-run wealth.

    x range: 0 → ~2.5×f_star. Capped at the ruin boundary so log terms stay finite.
    """
    r = _returns(dist, payoff, cost, discount_factor)
    r_min = float(r.min())
    e_r = float(np.dot(dist.probs, r))

    if r_min <= -1.0 + 1e-9:
        f_upper = min(1.0 / (-r_min) - 1e-9, SAFETY_F_MAX)
    else:
        f_upper = SAFETY_F_MAX

    # x range: 0 → 2.5 × f_star so f* sits at 40% of the x-axis, giving
    # equal room to see both the peak and the overbetting penalty.
    if f_star > 0:
        f_max_plot = min(2.5 * f_star, f_upper)
    else:
        f_max_plot = min(0.05, f_upper)

    f_values = np.linspace(0.0, f_max_plot, n_points)
    log_growth = np.array([
        float(np.dot(dist.probs, np.log(np.clip(1.0 + f * r, 1e-300, None))))
        for f in f_values
    ])
    geo_growth = np.exp(log_growth) - 1.0
    er_curve = f_values * e_r
    return f_values, geo_growth, er_curve


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
) -> KellyReport:
    """Build the full Kelly report. `multiplier` ∈ (0, 1]."""
    if not (0.0 < multiplier <= 1.0):
        raise ValueError(f"multiplier must lie in (0, 1]; got {multiplier}")

    r = _returns(dist, payoff, cost, discount_factor)
    r_min = float(r.min())
    unbounded = r_min < UNBOUNDED_LOSS_THRESHOLD

    if unbounded:
        # Decline: structure has potential loss far exceeding premium; Kelly
        # would be driven by the distribution truncation point, not the view.
        e_r = float(np.dot(dist.probs, r))
        var_r = float(np.dot(dist.probs, (r - e_r) ** 2))
        prob_loss = float(dist.probs[r < 0].sum())
        prob_total_loss = float(dist.probs[r <= -1.0 + 1e-9].sum())
        return KellyReport(
            f_continuous=0.0, f_discrete=0.0, f_raw=0.0, f_displayed=0.0,
            multiplier=multiplier,
            expected_return=e_r, variance=var_r,
            prob_loss=prob_loss, prob_total_loss=prob_total_loss,
            expected_log_growth=0.0, unbounded_loss=True,
        )

    f_c = kelly_continuous(dist, payoff, cost, discount_factor)
    f_d = kelly_discrete(dist, payoff, cost, discount_factor)

    # The "Kelly answer" we use for sizing is the discrete (numerical) solver —
    # the continuous one is shown alongside as a diagnostic.
    f_raw = f_d
    f_displayed = f_raw * multiplier

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
        expected_return=e_r,
        variance=var_r,
        prob_loss=prob_loss,
        prob_total_loss=prob_total_loss,
        expected_log_growth=expected_log_growth,
        unbounded_loss=False,
    )
