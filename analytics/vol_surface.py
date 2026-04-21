"""
Smile interpolation across the vol surface.

Architecture
------------
The market snapshot stores vols at 5 fixed delta pillars × 6 tenors:

  Pillars (call-equivalent delta): 10DC=0.10, 25DC=0.25, ATM≈0.50, 25DP→0.75, 10DP→0.90
  Tenors: 1W, 1M, 2M, 3M, 6M, 1Y

SmileInterpolator builds a per-tenor natural cubic spline in call-equivalent
delta space, then interpolates across time using total-variance (σ²T linear in T)
at each delta query.

Delta convention: forward delta, premium not included (same as black_scholes.py).
  Call delta ∈ (0, 1).  Put delta converted via: call_delta = 1 + put_delta.

vol_at_delta(delta, horizon_days):
  - Accepts call delta (positive) or put delta (negative).
  - Interpolates smile at each bracketing tenor, then total-variance across time.

vol_at_strike(K, F, T, horizon_days):
  - Iterative: start with ATM vol, compute call_delta, look up smile vol,
    recompute delta. Converges in 2–3 steps for typical smile shapes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import norm

from data.schema import CurrencySnapshot

# Pillar labels and their call-equivalent delta values.
# 10DP / 25DP are OTM puts; by put-call parity (forward delta, no premium):
#   call_delta = 1 + put_delta   →   25DP (put Δ = -0.25) → call Δ = 0.75
_PILLARS: list[str]  = ["10DC", "25DC", "ATM", "25DP", "10DP"]
_CALL_DELTAS: list[float] = [0.10,   0.25,   0.50,  0.75,   0.90]

_TENOR_DAYS: dict[str, int] = {
    "1W": 7, "1M": 30, "2M": 60, "3M": 91, "6M": 182, "1Y": 365,
}


@dataclass(frozen=True)
class _TenorSmile:
    """Cubic spline fit for a single tenor."""
    tenor_days: int
    T: float               # tenor_days / 365
    spline: CubicSpline    # x = call_delta, y = vol


class SmileInterpolator:
    """
    Interpolates vol at arbitrary (delta, horizon) from the 5-pillar surface.

    Usage:
        interp = SmileInterpolator(ccy)
        vol = interp.vol_at_delta(-0.25, horizon_days=120)   # 25-delta put at 4M
        vol = interp.vol_at_strike(5.50, F=5.897, horizon_days=120)
    """

    def __init__(self, ccy: CurrencySnapshot) -> None:
        self._ccy = ccy
        self._tenors: list[_TenorSmile] = self._build_tenors()

    def _build_tenors(self) -> list[_TenorSmile]:
        smiles = []
        for label, days in _TENOR_DAYS.items():
            vols = []
            for pillar in _PILLARS:
                v = self._ccy.get_vol(label, pillar)  # type: ignore[arg-type]
                if v is None:
                    break
                vols.append(v)
            if len(vols) == len(_PILLARS):
                spline = CubicSpline(
                    _CALL_DELTAS, vols,
                    bc_type="not-a-knot",
                    extrapolate=True,
                )
                smiles.append(_TenorSmile(days, days / 365.0, spline))
        if not smiles:
            raise ValueError(f"No complete vol surface for {self._ccy.pair}")
        smiles.sort(key=lambda s: s.tenor_days)
        return smiles

    def vol_at_delta(self, delta: float, horizon_days: int) -> float:
        """
        Interpolate vol at a given forward delta and horizon.

        Args:
            delta: call delta ∈ (0,1) or put delta ∈ (-1,0).
            horizon_days: horizon in calendar days.
        Returns:
            Annualised implied vol (e.g. 0.175 = 17.5%).
        """
        call_delta = _to_call_delta(delta)
        call_delta = float(np.clip(call_delta, _CALL_DELTAS[0], _CALL_DELTAS[-1]))
        T = horizon_days / 365.0
        return self._interp_total_variance(call_delta, T)

    def vol_at_strike(
        self,
        K: float,
        F: float,
        horizon_days: int,
        max_iter: int = 10,
        tol: float = 1e-8,
    ) -> float:
        """
        Derive smile vol for a given strike via iteration.

        Algorithm:
          1. Start with ATM vol as initial guess.
          2. Compute call_delta = N(d1) at current vol.
          3. Look up smile vol at that delta.
          4. Repeat until vol converges.
        """
        T = horizon_days / 365.0
        sigma = self.vol_at_delta(0.5, horizon_days)  # ATM seed

        for _ in range(max_iter):
            if T <= 0 or sigma <= 0:
                break
            d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
            call_delta = float(norm.cdf(d1))
            call_delta = float(np.clip(call_delta, _CALL_DELTAS[0], _CALL_DELTAS[-1]))
            sigma_new = self._interp_total_variance(call_delta, T)
            if abs(sigma_new - sigma) < tol:
                return sigma_new
            sigma = sigma_new

        return sigma

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _interp_total_variance(self, call_delta: float, T: float) -> float:
        """Total-variance interpolation across tenors at a fixed call_delta."""
        tenors = self._tenors

        # Evaluate spline at call_delta for each tenor → total variance
        tvars: list[tuple[float, float]] = [
            (s.T, float(s.spline(call_delta)) ** 2 * s.T)
            for s in tenors
        ]

        if T <= tvars[0][0]:
            return float(tenors[0].spline(call_delta))
        if T >= tvars[-1][0]:
            return float(tenors[-1].spline(call_delta))

        for i in range(len(tvars) - 1):
            t0, var0 = tvars[i]
            t1, var1 = tvars[i + 1]
            if t0 <= T <= t1:
                w = (T - t0) / (t1 - t0)
                var_interp = var0 + w * (var1 - var0)
                return math.sqrt(max(var_interp / T, 0.0))

        return float(tenors[-1].spline(call_delta))


def _to_call_delta(delta: float) -> float:
    """Convert put delta (-1, 0) to call-equivalent delta (0, 1). Calls pass through."""
    if delta < 0:
        return 1.0 + delta
    return delta
