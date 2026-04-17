"""
Price distribution computations for the analytics repository.

Two distribution variants:

  flat_atm  — GBM with constant ATM vol (symmetric in log-space, no smile)
  smile     — GBM where σ varies by percentile via the delta-pillar surface

GBM formula (vol of log-returns, GBM convention):
  S(t, p) = S₀ × exp((μ − σ_p²/2) × t + σ_p × √t × Φ⁻¹(p))
  where  μ = log(F_T / S₀) / T   (risk-neutral forward drift)
         σ_p is the implied vol at the percentile's corresponding delta

For the smile variant, the call-delta at each percentile is:
  Δ_call = Φ(σ_ATM × √T_horizon − Φ⁻¹(p))
and the surface vol is then interpolated at that delta.  The delta is evaluated
at the horizon tenor and held constant for all intermediate time steps (frozen
smile), which is adequate for a price-fan visualisation.
"""

from __future__ import annotations

import math

from scipy.stats import norm

from data.schema import CurrencySnapshot
from analytics.models import (
    BAND_LABELS,
    BAND_PERCENTILES,
    HistogramBin,
    MaturityHistogram,
    PercentileBand,
    PriceDistribution,
)

# Tenor label → days (Act/365, consistent with pricing/forwards.py)
_TENOR_DAYS: dict[str, int] = {
    "1W": 7, "1M": 30, "2M": 60, "3M": 91, "6M": 182, "1Y": 365,
}

# Delta pillars ordered by ascending call delta (FX forward-delta convention):
#   OTM call 10DC → Δ_call ≈ 0.10
#   OTM put  10DP → Δ_call ≈ 0.90  (put 10D ↔ call 90D by parity)
_PILLARS: list[str] = ["10DC", "25DC", "ATM", "25DP", "10DP"]
_DELTAS: list[float] = [0.10, 0.25, 0.50, 0.75, 0.90]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _interp_vol(ccy: CurrencySnapshot, horizon_days: int, delta_label: str) -> float:
    """
    Interpolate vol at a given delta pillar to an arbitrary horizon.
    Uses total-variance interpolation: σ²(T) × T is linear in T.
    """
    points: list[tuple[float, float]] = []
    for tl, t_days in _TENOR_DAYS.items():
        v = ccy.get_vol(tl, delta_label)  # type: ignore[arg-type]
        if v is not None:
            points.append((t_days / 365.0, v))
    if not points:
        raise ValueError(f"No vol data for delta {delta_label!r} on {ccy.pair}")
    points.sort()

    T = horizon_days / 365.0
    tvars: list[tuple[float, float]] = [(t, v * v * t) for t, v in points]

    if T <= tvars[0][0]:
        return points[0][1]
    if T >= tvars[-1][0]:
        return points[-1][1]

    for i in range(len(tvars) - 1):
        t0, var0 = tvars[i]
        t1, var1 = tvars[i + 1]
        if t0 <= T <= t1:
            w = (T - t0) / (t1 - t0)
            var_interp = var0 + w * (var1 - var0)
            return math.sqrt(var_interp / T)

    return points[-1][1]


def _smile_vol_for_percentile(
    ccy: CurrencySnapshot,
    horizon_days: int,
    sigma_atm: float,
    percentile: float,
) -> float:
    """
    Return smile vol for a given percentile.

    Step 1: map percentile → forward call delta via flat-ATM approximation:
              Δ_call = Φ(σ_ATM × √T − Φ⁻¹(p))
    Step 2: linear interpolation across the five delta pillars at this horizon.
    """
    T = horizon_days / 365.0
    z = norm.ppf(percentile)
    call_delta = norm.cdf(sigma_atm * math.sqrt(T) - z)
    call_delta = max(_DELTAS[0], min(_DELTAS[-1], call_delta))

    pillar_vols = [_interp_vol(ccy, horizon_days, p) for p in _PILLARS]

    for i in range(len(_DELTAS) - 1):
        d0, d1 = _DELTAS[i], _DELTAS[i + 1]
        if d0 <= call_delta <= d1:
            w = (call_delta - d0) / (d1 - d0)
            return pillar_vols[i] + w * (pillar_vols[i + 1] - pillar_vols[i])

    return pillar_vols[-1]


def _gbm_price(S0: float, mu: float, sigma: float, t: float, z: float) -> float:
    """GBM price at time t for a given z = Φ⁻¹(p)."""
    return S0 * math.exp((mu - sigma * sigma / 2.0) * t + sigma * math.sqrt(t) * z)


def _axis_limits(
    S0: float, mu: float, sigma: float, T: float, n_sigma: float = 5.0
) -> tuple[float, float]:
    """Y-axis bounds at ±n_sigma at maturity using flat-ATM vol."""
    lo = S0 * math.exp((mu - sigma * sigma / 2.0) * T - n_sigma * sigma * math.sqrt(T))
    hi = S0 * math.exp((mu - sigma * sigma / 2.0) * T + n_sigma * sigma * math.sqrt(T))
    return lo, hi


def _time_steps(horizon_days: int, n_steps: int) -> tuple[list[int], list[float]]:
    """Evenly-spaced time steps from 0 to horizon_days (inclusive)."""
    step_days = [round(i * horizon_days / n_steps) for i in range(n_steps + 1)]
    step_years = [d / 365.0 for d in step_days]
    return step_days, step_years


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_flat_vol_distribution(
    ccy: CurrencySnapshot,
    horizon_days: int,
    n_steps: int | None = None,
) -> PriceDistribution:
    """
    GBM price distribution with constant ATM vol (no smile).

    Drift anchored to the horizon forward: E[S_T] = F_T (mean = forward).
    """
    from pricing.forwards import interpolate_forward

    T = horizon_days / 365.0
    S0 = ccy.spot
    F_T = interpolate_forward(ccy, T)
    mu = math.log(F_T / S0) / T
    sigma = _interp_vol(ccy, horizon_days, "ATM")

    steps = n_steps if n_steps is not None else max(2, math.ceil(horizon_days / 7))
    step_days, step_years = _time_steps(horizon_days, steps)

    bands: list[PercentileBand] = []
    terminal: list[float] = []

    for pct, label in zip(BAND_PERCENTILES, BAND_LABELS):
        z = norm.ppf(pct)
        prices = [S0 if t == 0.0 else _gbm_price(S0, mu, sigma, t, z) for t in step_years]
        bands.append(PercentileBand(label=label, prices=prices))
        terminal.append(_gbm_price(S0, mu, sigma, T, z))

    axis_lo, axis_hi = _axis_limits(S0, mu, sigma, T)

    return PriceDistribution(
        pair=ccy.pair,
        spot=S0,
        horizon_days=horizon_days,
        vol_type="flat_atm",
        atm_vol=sigma,
        time_steps_days=step_days,
        bands=bands,
        axis_min=axis_lo,
        axis_max=axis_hi,
        terminal_minus3s=terminal[0],
        terminal_minus2s=terminal[1],
        terminal_minus1s=terminal[2],
        terminal_median=terminal[3],
        terminal_plus1s=terminal[4],
        terminal_plus2s=terminal[5],
        terminal_plus3s=terminal[6],
    )


def compute_smile_distribution(
    ccy: CurrencySnapshot,
    horizon_days: int,
    n_steps: int | None = None,
) -> PriceDistribution:
    """
    GBM price distribution where σ varies by percentile using the surface smile.

    Each percentile is mapped to a call delta (using ATM vol as first pass),
    then vol is interpolated from the five delta pillars at the horizon tenor.
    The resulting smile vol is held constant over all time steps (frozen smile).
    """
    from pricing.forwards import interpolate_forward

    T = horizon_days / 365.0
    S0 = ccy.spot
    F_T = interpolate_forward(ccy, T)
    mu = math.log(F_T / S0) / T
    sigma_atm = _interp_vol(ccy, horizon_days, "ATM")

    steps = n_steps if n_steps is not None else max(2, math.ceil(horizon_days / 7))
    step_days, step_years = _time_steps(horizon_days, steps)

    bands: list[PercentileBand] = []
    terminal: list[float] = []
    max_sigma = sigma_atm

    for pct, label in zip(BAND_PERCENTILES, BAND_LABELS):
        z = norm.ppf(pct)
        sigma = _smile_vol_for_percentile(ccy, horizon_days, sigma_atm, pct)
        max_sigma = max(max_sigma, sigma)
        prices = [S0 if t == 0.0 else _gbm_price(S0, mu, sigma, t, z) for t in step_years]
        bands.append(PercentileBand(label=label, prices=prices))
        terminal.append(_gbm_price(S0, mu, sigma, T, z))

    axis_lo, axis_hi = _axis_limits(S0, mu, max_sigma, T)

    return PriceDistribution(
        pair=ccy.pair,
        spot=S0,
        horizon_days=horizon_days,
        vol_type="smile",
        atm_vol=sigma_atm,
        time_steps_days=step_days,
        bands=bands,
        axis_min=axis_lo,
        axis_max=axis_hi,
        terminal_minus3s=terminal[0],
        terminal_minus2s=terminal[1],
        terminal_minus1s=terminal[2],
        terminal_median=terminal[3],
        terminal_plus1s=terminal[4],
        terminal_plus2s=terminal[5],
        terminal_plus3s=terminal[6],
    )


def compute_maturity_histogram(
    flat: PriceDistribution,
    smile: PriceDistribution,
) -> MaturityHistogram:
    """
    Probability of spot landing in each 5%-of-spot bin at maturity.

    Flat probabilities use the exact lognormal CDF.
    Smile probabilities use piecewise-linear interpolation of the smile CDF
    across the seven stored terminal percentile points.
    """
    S0 = flat.spot
    sigma_sqrtT = math.log(flat.terminal_plus1s / flat.terminal_median)
    drift_term = math.log(flat.terminal_median / S0)

    # Bin edges: 5% of spot, covering ±3.5σ
    bin_width = S0 * 0.05
    lo = math.floor(flat.terminal_median * math.exp(-3.5 * sigma_sqrtT) / bin_width) * bin_width
    hi = math.ceil(flat.terminal_median * math.exp(3.5 * sigma_sqrtT) / bin_width) * bin_width

    edges: list[float] = []
    k = lo
    while k <= hi + bin_width * 0.01:
        edges.append(round(k, 6))
        k += bin_width

    n = len(edges) - 1

    # Flat CDF
    def flat_cdf(K: float) -> float:
        return norm.cdf((math.log(K / S0) - drift_term) / sigma_sqrtT)

    # Smile CDF (piecewise linear in log-price space)
    smile_pts = sorted([
        (smile.terminal_minus3s, 0.0013),
        (smile.terminal_minus2s, 0.0228),
        (smile.terminal_minus1s, 0.1587),
        (smile.terminal_median,  0.5000),
        (smile.terminal_plus1s,  0.8413),
        (smile.terminal_plus2s,  0.9772),
        (smile.terminal_plus3s,  0.9987),
    ], key=lambda x: x[0])

    def smile_cdf(K: float) -> float:
        if K <= smile_pts[0][0]:
            return 0.0
        if K >= smile_pts[-1][0]:
            return 1.0
        for i in range(len(smile_pts) - 1):
            k0, p0 = smile_pts[i]
            k1, p1 = smile_pts[i + 1]
            if k0 <= K <= k1:
                w = math.log(K / k0) / math.log(k1 / k0)
                return p0 + w * (p1 - p0)
        return 1.0

    bins: list[HistogramBin] = []
    for i in range(n):
        lo_e, hi_e = edges[i], edges[i + 1]
        bins.append(HistogramBin(
            label=f"{lo_e:.2f}–{hi_e:.2f}",
            lo=lo_e,
            hi=hi_e,
            flat_pct=round((flat_cdf(hi_e) - flat_cdf(lo_e)) * 100, 2),
            smile_pct=round((smile_cdf(hi_e) - smile_cdf(lo_e)) * 100, 2),
        ))

    return MaturityHistogram(
        pair=flat.pair,
        spot=S0,
        horizon_days=flat.horizon_days,
        bins=bins,
    )


def interpolate_atm_vol(ccy: CurrencySnapshot, horizon_days: int) -> float:
    """Interpolate ATM vol to an arbitrary horizon using total-variance interpolation."""
    return _interp_vol(ccy, horizon_days, "ATM")


def interpolate_vol(ccy: CurrencySnapshot, horizon_days: int, delta_label: str) -> float:
    """Interpolate vol at any delta pillar (e.g. '25DC', '25DP', '10DC') to an arbitrary horizon."""
    return _interp_vol(ccy, horizon_days, delta_label)
