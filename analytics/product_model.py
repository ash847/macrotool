"""Product model — `Leg` / `Structure` / `PricedStructure`.

Phase A of the product-model refactor (see PRODUCT_MODEL_PLAN.md). The package is a
first-class object: a `Structure` is a market-independent *definition* (legs, each with
a signed notional and a strike anchor); `price(structure, ms)` produces a
`PricedStructure` (resolved strikes/vols + linearly-aggregated package metrics).

Signed notional: + long / − short; magnitude is the weight. A package value/risk is then
a literal weighted sum `Σ signed_notionalᵢ × unitᵢ` — this subsumes the seagull wing ratio
and the ratio-spread weights (they are just leg notionals).

`PricedStructure` exposes back-compat properties (`strikes`, `variant_label`, …) so it can
stand in for the legacy flat `PricedVariant` during the consumer migration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Instrument(str, Enum):
    VANILLA = "vanilla"
    DIGITAL = "digital"          # binary; strike solved for a premium target


class Right(str, Enum):
    CALL = "call"
    PUT = "put"


class AnchorKind(str, Enum):
    DELTA = "delta"              # value = call/put delta (0,1)
    ATMF = "atmf"                # strike = forward
    HALF_SIGMA = "half_sigma"    # strike = F·e^{±0.5σ√T} toward the view
    TARGET = "target"            # strike = the target spot
    STRIKE = "strike"            # value = absolute strike
    PREMIUM = "premium"          # value = target premium %, strike solved (digital)


@dataclass(frozen=True)
class Anchor:
    kind: AnchorKind
    value: float | None = None


@dataclass(frozen=True)
class Leg:
    """One instrument in a package, market-independent. Strike is resolved at price time."""
    instrument: Instrument
    right: Right
    signed_notional: float       # + long / − short; magnitude = weight
    anchor: Anchor


@dataclass(frozen=True)
class Structure:
    """A package definition. Market-independent, serializable, immutable."""
    family: str                  # e.g. "1x1.5_spread" — display/provenance/dispatch
    legs: tuple[Leg, ...]
    label: str = ""              # variant label, e.g. "25Δ / 15Δ"
    barrier_anchor: Anchor | None = None   # RKO family (definitional)


@dataclass
class PricedLeg:
    leg: Leg
    strike: float
    vol: float
    unit_premium: float          # Black-76 absolute premium of 1 unit (quote-ccy, discounted)


@dataclass
class PricedStructure:
    """Valuation/risk output of price(structure, ms). Metrics in base-ccy %, common basis."""
    structure: Structure
    priced_legs: list[PricedLeg]
    net_premium_pct: float
    payoff_at_target_pct: float | None
    rr_at_target: float | None
    max_loss_pct: float
    breakeven: float | None
    is_zero_cost: bool
    barrier: float | None = None
    wing_ratio: float | None = None      # derived (|wing leg notional|) for display/back-compat
    warnings: list[str] = field(default_factory=list)

    # --- back-compat shim (lets PricedStructure stand in for PricedVariant) ---
    @property
    def strikes(self) -> list[float]:
        return [pl.strike for pl in self.priced_legs]

    @property
    def variant_label(self) -> str:
        return self.structure.label
