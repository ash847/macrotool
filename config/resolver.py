"""
Merges the three config layers (base defaults, user profile, session overrides)
into a single ResolvedConfig.

Merge strategy:
  - Later layers win on non-null values.
  - source_trace records which layer each field came from.
  - Dot-path overrides from session layer are applied last via _set_nested().
"""

from __future__ import annotations

import copy
from typing import Any

from config.schema import (
    DisplayConfig,
    KellyConfig,
    ResolvedConfig,
    SessionOverrides,
    SizingConfig,
    StructureConfig,
    UserProfile,
    VolRegimeConfig,
    VolRegimeThresholds,
)


def resolve(
    base: ResolvedConfig,
    user_profile: UserProfile | None,
    session_overrides: SessionOverrides,
) -> ResolvedConfig:
    """Merge base → user_profile → session_overrides into a ResolvedConfig."""
    cfg = base.model_copy(deep=True)
    trace = dict(cfg.source_trace)

    # --- Layer 2: user profile ---
    if user_profile is not None:
        cfg, trace = _apply_user_profile(cfg, user_profile, trace)

    # --- Layer 3: session overrides (dot-path based) ---
    for override in session_overrides.overrides:
        _set_nested(cfg, override.field_path, override.value)
        trace[override.field_path] = "session"

    cfg.source_trace = trace
    return cfg


def _apply_user_profile(
    cfg: ResolvedConfig,
    profile: UserProfile,
    trace: dict,
) -> tuple[ResolvedConfig, dict]:
    """Apply non-null user profile fields to cfg."""

    # Sizing overrides
    s = profile.sizing
    if s.kelly_fraction is not None:
        cfg.sizing.kelly.default_fraction = s.kelly_fraction
        trace["sizing.kelly.default_fraction"] = "profile"
    if s.stop_atr_multiple is not None:
        cfg.sizing.stop.atr_multiple = s.stop_atr_multiple
        trace["sizing.stop.atr_multiple"] = "profile"
    if s.max_tranche_count is not None:
        # Cap all tranche schedules at the user's max
        for attr in ("low_timing_conviction", "medium_timing_conviction", "high_timing_conviction"):
            schedule = getattr(cfg.sizing.tranche_entry, attr)
            if schedule.count > s.max_tranche_count:
                new_count = s.max_tranche_count
                # Re-normalise weights to the first new_count entries
                raw = schedule.weights[:new_count]
                total = sum(raw)
                new_weights = [w / total for w in raw]
                schedule.count = new_count
                schedule.weights = new_weights
        trace["sizing.tranche_entry.*.count"] = "profile"

    # Structure preferences
    sp = profile.structure_preferences
    if sp.excluded_structures:
        cfg.structures.excluded_structures = sp.excluded_structures
        trace["structures.excluded_structures"] = "profile"
    if sp.preferred_structures:
        cfg.structures.preferred_structures = sp.preferred_structures
        trace["structures.preferred_structures"] = "profile"
    if not sp.exotic_comparisons:
        cfg.structures.exotic_comparison_available = False
        trace["structures.exotic_comparison_available"] = "profile"

    # Vol regime thresholds
    if profile.vol_regime_thresholds is not None:
        cfg.vol_regime.regime_percentile_thresholds = profile.vol_regime_thresholds
        trace["vol_regime.regime_percentile_thresholds"] = "profile"

    # Display config
    cfg.display = profile.display
    trace["display"] = "profile"

    return cfg, trace


def _set_nested(obj: Any, path: str, value: Any) -> None:
    """
    Set a value on a nested Pydantic model using a dot-path string.
    e.g. _set_nested(cfg, "sizing.kelly.default_fraction", 0.25)
    """
    parts = path.split(".")
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def explain_source(cfg: ResolvedConfig, field_path: str) -> str:
    """
    Return a human-readable explanation of where a config value came from.
    Used by the conversation layer when the PM asks 'why are you using X?'
    """
    source = cfg.source_trace.get(field_path) or cfg.source_trace.get("*", "default")
    match source:
        case "default":
            return f"the tool default ({field_path})"
        case "profile":
            return f"your saved profile preference ({field_path})"
        case "session":
            return f"your instruction earlier in this conversation ({field_path})"
        case _:
            return source
