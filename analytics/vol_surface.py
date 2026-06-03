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
from typing import Protocol, runtime_checkable

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


# ---------------------------------------------------------------------------
# Pluggable surface interface
# ---------------------------------------------------------------------------

@runtime_checkable
class VolSurface(Protocol):
    """Stable interface every vanilla pricing site depends on.

    Implementations decide HOW the vol is built (cubic-spline smile today, SABR
    later); call sites only ever ask for a vol at a strike or a delta. Swapping
    the build method, the pillar resolution, or the interpolation scheme is a
    change behind this interface, never at the call site.

    ``vol_at_delta`` takes a *signed* delta: a positive call delta in (0,1) or a
    negative put delta in (-1,0), matching ``black_scholes`` delta conventions.
    """

    def vol_at_strike(self, K: float, F: float, horizon_days: int) -> float: ...

    def vol_at_delta(self, delta: float, horizon_days: int) -> float: ...


class FlatSurface:
    """Degenerate surface: returns one ATM vol for every strike and delta.

    This is the flat-smile approximation expressed as a first-class surface, so
    every pricing site can hold a ``VolSurface`` unconditionally instead of
    branching on ``smile is None``. Pricing against a ``FlatSurface(atm_vol)`` is
    byte-for-byte identical to the legacy scalar-vol path.
    """

    __slots__ = ("_v",)

    def __init__(self, atm_vol: float) -> None:
        self._v = float(atm_vol)

    def vol_at_strike(self, K: float, F: float, horizon_days: int) -> float:
        return self._v

    def vol_at_delta(self, delta: float, horizon_days: int) -> float:
        return self._v


def build_vol_surface(
    ccy: CurrencySnapshot,
    *,
    method: str = "cubic_spline",
    **params,
) -> VolSurface:
    """Construct a VolSurface for a pair via the chosen build method.

    ``method`` selects the builder; ``**params`` carry resolution/interpolation
    knobs forwarded to that builder. Add a new method (e.g. ``"sabr"``) by adding
    a branch here and a class that satisfies the ``VolSurface`` protocol — no
    call site changes. Raises ``ValueError`` for an unknown method or, for
    ``cubic_spline``, an incomplete surface.
    """
    if method == "cubic_spline":
        return SmileInterpolator(ccy, **params)
    raise ValueError(f"Unknown vol surface build method: {method!r}")


def smile_skew_spread(
    surface: VolSurface | None,
    K: float,
    F: float,
    horizon_days: int,
) -> float:
    """Sticky-delta skew spread: vol_at_strike(K) minus the ATM vol at horizon.

    Used to re-anchor a scenario's ATM vol level (which carries term structure
    and any vol shock) while applying the surface's skew shape at the strike's
    delta *under the scenario forward* — i.e. sticky-delta. Returns 0.0 for a
    None or flat surface, so the caller's scalar vol is reproduced exactly.
    """
    if surface is None:
        return 0.0
    atm = surface.vol_at_delta(0.50, horizon_days)
    return surface.vol_at_strike(K, F, horizon_days) - atm


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

    def __init__(
        self,
        ccy: CurrencySnapshot,
        *,
        pillars: list[str] | None = None,
        call_deltas: list[float] | None = None,
        tenor_days: dict[str, int] | None = None,
        bc_type: str = "not-a-knot",
        delta_clip: tuple[float, float] | None = None,
    ) -> None:
        self._ccy = ccy
        self._pillars = pillars or list(_PILLARS)
        self._call_deltas = call_deltas or list(_CALL_DELTAS)
        self._tenor_days = tenor_days or dict(_TENOR_DAYS)
        self._bc_type = bc_type
        self._delta_lo, self._delta_hi = delta_clip or (
            self._call_deltas[0], self._call_deltas[-1]
        )
        self._tenors: list[_TenorSmile] = self._build_tenors()

    def _build_tenors(self) -> list[_TenorSmile]:
        smiles = []
        for label, days in self._tenor_days.items():
            vols = []
            for pillar in self._pillars:
                v = self._ccy.get_vol(label, pillar)  # type: ignore[arg-type]
                if v is None:
                    break
                vols.append(v)
            if len(vols) == len(self._pillars):
                spline = CubicSpline(
                    self._call_deltas, vols,
                    bc_type=self._bc_type,
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
        call_delta = float(np.clip(call_delta, self._delta_lo, self._delta_hi))
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
            call_delta = float(np.clip(call_delta, self._delta_lo, self._delta_hi))
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
