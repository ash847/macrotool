"""
Pydantic models for market data.

The vol surface is stored as a delta-tenor grid — standard FX options convention.
Deltas: 10DP, 25DP, ATM, 25DC, 10DC
Tenors: 1W, 1M, 2M, 3M, 6M, 1Y

Vol surface convention: delta labels are always expressed relative to the BASE
currency (ccy1) in the pair. "Call" and "put" without qualification mean a call
or put on ccy1. Examples:
  - USDBRL: 25DC = 25-delta USD call (BRL put); 25DP = 25-delta USD put (BRL call).
    25DC > 25DP means USD calls are more expensive than USD puts — topside USD skew.
  - USDTRY: same convention; 25DC = USD call.
  - EURPLN: 25DC = EUR call (PLN put).

DiscountFactor curves (usd_df_curve, eur_df_curve) are stored per pair on the
same tenor pillars as the vol surface.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


DeltaLabel = Literal["10DP", "25DP", "ATM", "25DC", "10DC"]
TenorLabel = Literal["1W", "1M", "2M", "3M", "6M", "1Y"]
InstrumentType = Literal["NDF", "Deliverable"]


class ForwardPoint(BaseModel):
    tenor: TenorLabel
    points: float        # forward points in pip terms (e.g. 450.0 for USDBRL)
    outright: float      # full outright forward rate


class VolSurfaceNode(BaseModel):
    tenor: TenorLabel
    delta: DeltaLabel
    vol: float           # annualised vol, decimal form (0.182 = 18.2%)

    @field_validator("vol")
    @classmethod
    def vol_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Vol must be positive")
        return v


class DiscountFactor(BaseModel):
    tenor: TenorLabel
    df: float            # discount factor (e.g. 0.9888 for 3M at 4.5%)


class CurrencySnapshot(BaseModel):
    pair: str                             # "USDBRL", "USDTRY", "EURPLN"
    instrument_type: InstrumentType
    spot: float
    as_of: date
    forwards: list[ForwardPoint]
    vol_surface: list[VolSurfaceNode]
    usd_df_curve: list[DiscountFactor] = Field(default_factory=list)
    eur_df_curve: list[DiscountFactor] = Field(default_factory=list)
    gbp_df_curve: list[DiscountFactor] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_required_base_df_curve(self) -> "CurrencySnapshot":
        base_ccy = self.pair[:3]
        required_curve_map = {
            "USD": self.usd_df_curve,
            "EUR": self.eur_df_curve,
            "GBP": self.gbp_df_curve,
        }
        required_curve = required_curve_map.get(base_ccy)
        if required_curve is not None and not required_curve:
            raise ValueError(
                f"{self.pair} requires a {base_ccy.lower()}_df_curve for pricing, but none was provided."
            )
        return self

    def get_forward(self, tenor: TenorLabel) -> ForwardPoint | None:
        return next((f for f in self.forwards if f.tenor == tenor), None)

    def get_vol(self, tenor: TenorLabel, delta: DeltaLabel) -> float | None:
        node = next(
            (n for n in self.vol_surface if n.tenor == tenor and n.delta == delta),
            None,
        )
        return node.vol if node else None

    def get_atm_vol(self, tenor: TenorLabel) -> float | None:
        return self.get_vol(tenor, "ATM")

    def get_usd_df(self, tenor: TenorLabel) -> float | None:
        node = next((d for d in self.usd_df_curve if d.tenor == tenor), None)
        return node.df if node else None

    def get_eur_df(self, tenor: TenorLabel) -> float | None:
        node = next((d for d in self.eur_df_curve if d.tenor == tenor), None)
        return node.df if node else None

    def get_gbp_df(self, tenor: TenorLabel) -> float | None:
        node = next((d for d in self.gbp_df_curve if d.tenor == tenor), None)
        return node.df if node else None


class MarketSnapshot(BaseModel):
    snapshot_date: date
    data_note: str = "Synthetic POC data — not for trading"
    currencies: dict[str, CurrencySnapshot]  # keyed by pair string

    def get(self, pair: str) -> CurrencySnapshot:
        if pair not in self.currencies:
            raise KeyError(f"Pair '{pair}' not in snapshot. Available: {list(self.currencies.keys())}")
        return self.currencies[pair]
