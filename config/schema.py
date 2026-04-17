"""
Pydantic models for the layered config system.

Three layers: base defaults (JSON) → user profile (JSON) → session overrides (in-memory).
The reasoning engine always reads ResolvedConfig, never the raw layers directly.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Sizing config
# ---------------------------------------------------------------------------

class KellyConfig(BaseModel):
    default_fraction: float = 0.5
    high_conviction_fraction: float = 0.75
    low_conviction_fraction: float = 0.25
    max_fraction: float = 1.0
    min_fraction: float = 0.1

    @field_validator("default_fraction", "high_conviction_fraction",
                     "low_conviction_fraction", "max_fraction", "min_fraction")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Kelly fraction must be positive")
        return v


class VolRegimeSizeAdjustment(BaseModel):
    low: float = 1.1
    normal: float = 1.0
    elevated: float = 0.75
    stressed: float = 0.5


class StopConfig(BaseModel):
    method: Literal["vol_derived"] = "vol_derived"
    atr_multiple: float = 2.0


class TrancheSchedule(BaseModel):
    count: int
    weights: list[float]

    @field_validator("weights")
    @classmethod
    def weights_must_sum_to_one(cls, v: list[float]) -> list[float]:
        if abs(sum(v) - 1.0) > 1e-6:
            raise ValueError(f"Tranche weights must sum to 1.0, got {sum(v)}")
        return v


class TrancheEntryConfig(BaseModel):
    low_timing_conviction: TrancheSchedule = TrancheSchedule(
        count=3, weights=[0.4, 0.35, 0.25]
    )
    medium_timing_conviction: TrancheSchedule = TrancheSchedule(
        count=2, weights=[0.6, 0.4]
    )
    high_timing_conviction: TrancheSchedule = TrancheSchedule(
        count=1, weights=[1.0]
    )


class TakeProfitScale(BaseModel):
    at_pct_of_target: float
    reduce_position_by: float


class TakeProfitConfig(BaseModel):
    scale_1: TakeProfitScale = TakeProfitScale(at_pct_of_target=0.5, reduce_position_by=0.33)
    scale_2: TakeProfitScale = TakeProfitScale(at_pct_of_target=0.75, reduce_position_by=0.33)
    runner_note: str = "Trail stop on remainder at entry + 0.5 * move"


class SizingConfig(BaseModel):
    kelly: KellyConfig = Field(default_factory=KellyConfig)
    vol_regime_size_adjustment: VolRegimeSizeAdjustment = Field(
        default_factory=VolRegimeSizeAdjustment
    )
    stop: StopConfig = Field(default_factory=StopConfig)
    tranche_entry: TrancheEntryConfig = Field(default_factory=TrancheEntryConfig)
    take_profit: TakeProfitConfig = Field(default_factory=TakeProfitConfig)


# ---------------------------------------------------------------------------
# Vol regime config
# ---------------------------------------------------------------------------

class RegimeBand(BaseModel):
    min_percentile: float = 0.0
    max_percentile: float = 1.0


class VolRegimeThresholds(BaseModel):
    low: RegimeBand = RegimeBand(min_percentile=0.0, max_percentile=0.30)
    normal: RegimeBand = RegimeBand(min_percentile=0.30, max_percentile=0.65)
    elevated: RegimeBand = RegimeBand(min_percentile=0.65, max_percentile=0.85)
    stressed: RegimeBand = RegimeBand(min_percentile=0.85, max_percentile=1.0)


class SkewBand(BaseModel):
    min_abs_rr: float = 0.0
    max_abs_rr: float = 1.0


class SkewThresholds(BaseModel):
    low: SkewBand = SkewBand(min_abs_rr=0.0, max_abs_rr=0.005)
    moderate: SkewBand = SkewBand(min_abs_rr=0.005, max_abs_rr=0.015)
    high: SkewBand = SkewBand(min_abs_rr=0.015, max_abs_rr=1.0)


class VolRegimeConfig(BaseModel):
    regime_percentile_thresholds: VolRegimeThresholds = Field(
        default_factory=VolRegimeThresholds
    )
    skew_rr_vol_thresholds: SkewThresholds = Field(default_factory=SkewThresholds)
    lookback_window_days: int = 252


# ---------------------------------------------------------------------------
# Structure preferences config
# ---------------------------------------------------------------------------

class StructureConfig(BaseModel):
    excluded_structures: list[str] = Field(default_factory=list)
    preferred_structures: list[str] = Field(default_factory=list)
    max_shortlist_size: int = 3
    always_include_vanilla_baseline: bool = True
    exotic_comparison_available: bool = True
    default_tenor_buffer_days: int = 5


# ---------------------------------------------------------------------------
# Display / output config
# ---------------------------------------------------------------------------

class ScenarioMatrixConfig(BaseModel):
    time_horizons: list[str] = Field(default=["1M", "2M", "3M"])
    spot_range_pct: float = 10.0      # ±10% spot grid
    vol_range_pct: float = 25.0       # ±25% vol perturbation
    spot_steps: int = 7
    vol_steps: int = 3


class DisplayConfig(BaseModel):
    scenario_matrix: ScenarioMatrixConfig = Field(default_factory=ScenarioMatrixConfig)
    show_source_trace: bool = False   # show which config layer each value came from
    decimal_places_spot: int = 4
    decimal_places_vol: int = 1       # vol displayed as pct, e.g. 18.2%
    decimal_places_pnl: int = 0       # P&L in USD rounded to nearest dollar


# ---------------------------------------------------------------------------
# Resolved config — what the reasoning engine reads
# ---------------------------------------------------------------------------

class ResolvedConfig(BaseModel):
    """
    Always constructed via config.resolver.resolve(), never directly.
    Represents the merged result of base defaults + user profile + session overrides.
    """
    sizing: SizingConfig = Field(default_factory=SizingConfig)
    structures: StructureConfig = Field(default_factory=StructureConfig)
    vol_regime: VolRegimeConfig = Field(default_factory=VolRegimeConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    source_trace: dict[str, str] = Field(
        default_factory=dict,
        description="Maps config field paths to their source layer: 'default', 'profile', or 'session'"
    )

    def get_kelly_fraction(self, conviction: str) -> float:
        """Convenience accessor used by the sizing engine."""
        match conviction:
            case "high":
                return self.sizing.kelly.high_conviction_fraction
            case "low":
                return self.sizing.kelly.low_conviction_fraction
            case _:
                return self.sizing.kelly.default_fraction

    def get_vol_size_multiplier(self, regime: str) -> float:
        """Convenience accessor used by the sizing engine."""
        return getattr(self.sizing.vol_regime_size_adjustment, regime, 1.0)

    def get_tranche_schedule(self, timing_conviction: str) -> TrancheSchedule:
        """Convenience accessor used by the sizing engine."""
        match timing_conviction:
            case "high":
                return self.sizing.tranche_entry.high_timing_conviction
            case "low":
                return self.sizing.tranche_entry.low_timing_conviction
            case _:
                return self.sizing.tranche_entry.medium_timing_conviction


# ---------------------------------------------------------------------------
# User profile — persists across sessions
# ---------------------------------------------------------------------------

class UserProfileSizing(BaseModel):
    kelly_fraction: float | None = None
    stop_atr_multiple: float | None = None
    max_tranche_count: int | None = None


class UserProfileStructures(BaseModel):
    excluded_structures: list[str] = Field(default_factory=list)
    preferred_structures: list[str] = Field(default_factory=list)
    exotic_comparisons: bool = True


class UserProfile(BaseModel):
    profile_version: str = "1.0"
    sizing: UserProfileSizing = Field(default_factory=UserProfileSizing)
    structure_preferences: UserProfileStructures = Field(
        default_factory=UserProfileStructures
    )
    vol_regime_thresholds: VolRegimeThresholds | None = None
    display: DisplayConfig = Field(default_factory=DisplayConfig)


# ---------------------------------------------------------------------------
# Session overrides — in-memory, current conversation only
# ---------------------------------------------------------------------------

class SessionOverride(BaseModel):
    field_path: str    # dot-path into ResolvedConfig, e.g. "sizing.kelly.default_fraction"
    value: Any
    scope: Literal["session", "profile"]
    raw_text: str      # what the PM actually said, for audit trail


class SessionOverrides(BaseModel):
    overrides: list[SessionOverride] = Field(default_factory=list)

    def apply(self, field_path: str, value: Any, scope: Literal["session", "profile"], raw_text: str) -> None:
        """Add or replace an override for a given field path."""
        self.overrides = [o for o in self.overrides if o.field_path != field_path]
        self.overrides.append(SessionOverride(
            field_path=field_path,
            value=value,
            scope=scope,
            raw_text=raw_text,
        ))

    def get(self, field_path: str) -> Any | None:
        for override in self.overrides:
            if override.field_path == field_path:
                return override.value
        return None

    def profile_scope_overrides(self) -> list[SessionOverride]:
        return [o for o in self.overrides if o.scope == "profile"]
