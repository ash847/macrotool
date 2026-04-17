"""
Pydantic data models for the analytics repository.

PriceDistribution stores a pre-computed GBM price fan — percentile bands
from t=0 to maturity — for both flat-ATM-vol and smile-adjusted assumptions.

Used by:
  context_builder.py  → text statistics block injected into the LLM system prompt
  interface/charts.py → Plotly distribution fan chart
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

VolType = Literal["flat_atm", "smile"]

# 7 bands at exact ±1σ, ±2σ, ±3σ intervals (norm.ppf of these percentiles)
BAND_PERCENTILES: list[float] = [0.0013, 0.0228, 0.1587, 0.5000, 0.8413, 0.9772, 0.9987]
BAND_LABELS: list[str] = ["-3σ", "-2σ", "-1σ", "Median", "+1σ", "+2σ", "+3σ"]


class HistogramBin(BaseModel):
    label: str        # "5.50–5.79"
    lo: float
    hi: float
    flat_pct: float   # probability (%)
    smile_pct: float  # probability (%)


class MaturityHistogram(BaseModel):
    pair: str
    spot: float
    horizon_days: int
    bins: list[HistogramBin]


class PercentileBand(BaseModel):
    label: str           # e.g. "-3σ", "Median", "+2σ"
    prices: list[float]  # price at each time step (index 0 = spot at t=0)


class PriceDistribution(BaseModel):
    pair: str
    spot: float
    horizon_days: int
    vol_type: VolType
    atm_vol: float               # ATM vol interpolated to horizon tenor (decimal, e.g. 0.175)
    time_steps_days: list[int]   # x-axis: [0, 7, 14, ..., horizon_days]
    bands: list[PercentileBand]  # 7 bands ordered -3σ → +3σ (same order as BAND_LABELS)
    axis_min: float              # ±5σ lower price at T  — chart y-axis lower bound
    axis_max: float              # ±5σ upper price at T  — chart y-axis upper bound
    # Terminal statistics for LLM context block
    terminal_minus3s: float
    terminal_minus2s: float
    terminal_minus1s: float
    terminal_median: float
    terminal_plus1s: float
    terminal_plus2s: float
    terminal_plus3s: float
