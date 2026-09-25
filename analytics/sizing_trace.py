"""Computed sizing provenance; never a second sizing implementation."""

from dataclasses import dataclass
import hashlib
import json
from typing import Literal


@dataclass(frozen=True)
class SizingTrace:
    status: Literal["sized", "zero", "unavailable", "error"]
    effective_method: Literal["fixed_loss", "kelly"] | None
    reason: str
    reference_capital: float
    notional_cap: float
    final_notional: float | None = None
    requested_method: str | None = None
    fallback_reason: str | None = None
    loss_budget: float | None = None
    per_unit_loss_proxy: float | None = None
    uncapped_notional: float | None = None
    binding_constraint: str | None = None
    bankroll: float | None = None
    kelly_lambda: float | None = None
    full_kelly_fraction: float | None = None
    full_kelly_proxy_exposure: float | None = None
    distribution_id: str | None = None
    distribution_points: int | None = None
    budget_distance: float | None = None
    budget_reference: float | None = None
    budget_target: float | None = None
    budget_input_rr: float | None = None
    currency_basis: str = "base currency, full units"


def distribution_fingerprint(probs, bins) -> str:
    payload = json.dumps([[float(value) for value in probs], [float(value) for value in bins]])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
