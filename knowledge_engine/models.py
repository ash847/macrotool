"""
Input and output models for the knowledge engine.

TradeView is the structured representation of what a PM has told the tool —
extracted by the LLM from natural language and validated here.

All knowledge engine components operate on these typed models, not raw strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ConvictionLevel = Literal["high", "medium", "low"]
Direction       = Literal["base_higher", "base_lower"]
Mode            = Literal["recommend", "critique"]


@dataclass
class TradeView:
    """
    Structured representation of the PM's stated trade view.

    `direction` uses the base currency convention:
      - "base_higher": base currency appreciates vs term (USDBRL goes UP for USD bulls;
                       EURPLN goes UP for EUR bulls)
      - "base_lower":  base currency depreciates vs term

    Budget: provide one of budget_usd, max_loss_usd, or notional_usd.
    The sizing engine will work from whichever is provided.
    """
    pair:                 str
    direction:            Direction
    direction_conviction: ConvictionLevel
    horizon_days:         int
    timing_conviction:    ConvictionLevel = "medium"
    magnitude_pct:        float | None = None   # expected move in % (e.g. 5.0 = 5%)
    budget_usd:           float | None = None   # premium budget in USD/EUR
    max_loss_usd:         float | None = None   # max acceptable loss
    notional_usd:         float | None = None   # explicit notional if PM supplies it
    catalyst:             str   | None = None
    mode:                 Mode                  = "recommend"
    pm_structure_description: str | None = None  # critique mode only

    @property
    def horizon_years(self) -> float:
        return self.horizon_days / 365.0

    @property
    def has_budget_constraint(self) -> bool:
        return self.budget_usd is not None and self.max_loss_usd is None

    @property
    def has_target_level(self) -> bool:
        return self.magnitude_pct is not None


# ---------------------------------------------------------------------------
# Structure selection outputs
# ---------------------------------------------------------------------------

@dataclass
class StructureShortlistItem:
    structure_id:    str
    display_name:    str
    rank:            int
    rationale:       str            # specific to this view, from the rule
    rule_id:         str
    sizing_modifier: str | None     # e.g. "tranche_entry"
    caution:         str | None     # from catalog, if present
    optimised_for:   str            # from catalog
    is_exotic:       bool = False   # exotic structures marked separately


@dataclass
class StructureSelectionResult:
    shortlist:       list[StructureShortlistItem]
    rules_fired:     list[str]        # which rule IDs matched
    skew_context:    str = ""         # reserved for future skew context injection
    vol_regime_note: str = ""         # reserved for future vol regime injection


# ---------------------------------------------------------------------------
# Sizing outputs
# ---------------------------------------------------------------------------

@dataclass
class TakeProfitLevel:
    at_pct_of_target: float
    reduce_by_pct:    float
    target_spot:      float | None    # absolute spot level if magnitude_pct known
    note:             str


@dataclass
class SizingOutput:
    # Kelly calculation
    kelly_fraction:          float
    kelly_conviction_used:   str     # "high" / "medium" / "low"
    kelly_source:            str     # "default" / "profile" / "session"

    # Vol adjustment (reserved for regime-based sizing; 1.0 until regime data is added)
    vol_adjustment:          float
    adjusted_kelly:          float   # kelly_fraction × vol_adjustment

    # Notional
    base_notional_usd:       float | None
    kelly_notional_usd:      float | None   # base × adjusted_kelly
    budget_type:             str            # "from_budget" / "from_max_loss" / "direct" / "unknown"

    # Stop
    stop_level:              float | None
    stop_distance_pct:       float | None
    daily_range_est:         float | None

    # Tranche schedule (populated when sizing_modifier = "tranche_entry")
    tranche_schedule:        list[float] | None   # e.g. [0.4, 0.35, 0.25]
    tranche_count:           int | None

    # Take profits
    take_profit_levels:      list[TakeProfitLevel]

    # Reasoning
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Convention outputs
# ---------------------------------------------------------------------------

@dataclass
class ResolvedConventions:
    pair:              str
    instrument_type:   str        # "NDF" or "Deliverable"

    # Settlement
    settlement_currency: str
    fixing_source:     str | None   # None for deliverable
    fixing_time_local: str | None
    fixing_timezone:   str | None
    fixing_time_london: str | None
    fixing_lag_days:   int | None
    settlement_days:   int

    # Options
    options_cut:       str
    premium_currency:  str
    delta_convention:  str
    smile_convention:  str
    liquid_tenors:     list[str]
    liquid_strikes:    str

    # Context
    risk_notes:        list[str]
    carry_notes:       list[str]
    market_structure:  dict


# ---------------------------------------------------------------------------
# Critique outputs
# ---------------------------------------------------------------------------

@dataclass
class CritiqueOutput:
    verdict:               str                  # "appropriate_for_view" | "suboptimal_but_defensible" | "materially_misaligned"
    pm_structure:          str
    recommended_alternative: str | None

    # Evaluation dimensions
    ev_comparison_note:    str
    scenario_weakness:     str
    execution_notes:       str
    gamma_notes:           str
    hedge_effectiveness:   str

    # Applied sizing (same as recommend mode, but for PM's structure)
    sizing:                SizingOutput | None

    dimension_scores: dict[str, str] = field(default_factory=dict)   # dim_id → "strong" / "acceptable" / "weak"
