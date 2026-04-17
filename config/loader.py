"""
Loads config from the three layers and returns a ResolvedConfig.

Usage:
    from config.loader import load_config
    cfg = load_config()   # base defaults only
    cfg = load_config(user_profile_path="config/user_profile.json")
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from config.schema import (
    DisplayConfig,
    ResolvedConfig,
    SessionOverrides,
    SizingConfig,
    StructureConfig,
    UserProfile,
    VolRegimeConfig,
)
from config.resolver import resolve


_REPO_ROOT = Path(__file__).parent.parent
_DEFAULTS_DIR = _REPO_ROOT / "knowledge" / "defaults"


def _load_json(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    # Strip _description keys — they're for human readers only
    return {k: v for k, v in data.items() if not k.startswith("_")}


def load_base_config() -> ResolvedConfig:
    """Load the base defaults from knowledge/defaults/*.json."""
    sizing_data = _load_json(_DEFAULTS_DIR / "sizing_defaults.json")
    vol_data = _load_json(_DEFAULTS_DIR / "vol_regime_defaults.json")
    structure_data = _load_json(_DEFAULTS_DIR / "structure_defaults.json")

    sizing = _parse_sizing(sizing_data)
    vol_regime = _parse_vol_regime(vol_data)
    structures = _parse_structures(structure_data)

    base = ResolvedConfig(
        sizing=sizing,
        structures=structures,
        vol_regime=vol_regime,
        display=DisplayConfig(),
        source_trace={},
    )
    # Mark everything as coming from 'default'
    base.source_trace = {"*": "default"}
    return base


def load_user_profile(path: str | Path | None = None) -> UserProfile | None:
    """Load user profile JSON if it exists. Returns None if not found."""
    if path is None:
        path = os.environ.get(
            "MACROTOOL_USER_PROFILE_PATH",
            str(_REPO_ROOT / "config" / "user_profile.json"),
        )
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    return UserProfile.model_validate(data)


def load_config(
    user_profile_path: str | Path | None = None,
    session_overrides: SessionOverrides | None = None,
) -> ResolvedConfig:
    """
    Load and merge all config layers.

    Args:
        user_profile_path: Path to user_profile.json. Defaults to env var or config/user_profile.json.
        session_overrides: In-memory overrides from the current conversation.

    Returns:
        ResolvedConfig with source_trace populated.
    """
    base = load_base_config()
    user_profile = load_user_profile(user_profile_path)
    return resolve(base, user_profile, session_overrides or SessionOverrides())


# ---------------------------------------------------------------------------
# Internal parsing helpers — convert raw JSON dicts to Pydantic models
# ---------------------------------------------------------------------------

def _parse_sizing(data: dict) -> SizingConfig:
    from config.schema import (
        KellyConfig, VolRegimeSizeAdjustment, StopConfig,
        TrancheEntryConfig, TrancheSchedule, TakeProfitConfig, TakeProfitScale,
    )
    kelly = KellyConfig(**data["kelly"])
    vol_adj = VolRegimeSizeAdjustment(**data["vol_regime_size_adjustment"])
    stop = StopConfig(**data["stop"])

    te_raw = data["tranche_entry"]
    tranche = TrancheEntryConfig(
        low_timing_conviction=TrancheSchedule(**te_raw["low_timing_conviction"]),
        medium_timing_conviction=TrancheSchedule(**te_raw["medium_timing_conviction"]),
        high_timing_conviction=TrancheSchedule(**te_raw["high_timing_conviction"]),
    )

    tp_raw = data["take_profit"]
    take_profit = TakeProfitConfig(
        scale_1=TakeProfitScale(**tp_raw["scale_1"]),
        scale_2=TakeProfitScale(**tp_raw["scale_2"]),
        runner_note=tp_raw["runner"]["note"],
    )

    return SizingConfig(
        kelly=kelly,
        vol_regime_size_adjustment=vol_adj,
        stop=stop,
        tranche_entry=tranche,
        take_profit=take_profit,
    )


def _parse_vol_regime(data: dict) -> VolRegimeConfig:
    from config.schema import VolRegimeThresholds, RegimeBand, SkewThresholds, SkewBand

    pt = data["regime_percentile_thresholds"]
    thresholds = VolRegimeThresholds(
        low=RegimeBand(**{k.replace("_percentile", ""): v for k, v in pt["low"].items()}
                       if "max_percentile" in pt["low"] else
                       {"max_percentile": pt["low"].get("max_percentile", 0.30)}),
        normal=RegimeBand(**pt["normal"]),
        elevated=RegimeBand(**pt["elevated"]),
        stressed=RegimeBand(
            min_percentile=pt["stressed"].get("min_percentile", 0.85),
            max_percentile=1.0,
        ),
    )

    sk = data["skew_rr_vol_thresholds"]
    skew = SkewThresholds(
        low=SkewBand(**sk["low"]),
        moderate=SkewBand(**sk["moderate"]),
        high=SkewBand(min_abs_rr=sk["high"]["min_abs_rr"], max_abs_rr=1.0),
    )

    return VolRegimeConfig(
        regime_percentile_thresholds=thresholds,
        skew_rr_vol_thresholds=skew,
        lookback_window_days=data.get("lookback_window_days", 252),
    )


def _parse_structures(data: dict) -> StructureConfig:
    return StructureConfig(
        excluded_structures=data.get("excluded_structures", []),
        preferred_structures=data.get("preferred_structures", []),
        max_shortlist_size=data.get("max_shortlist_size", 3),
        always_include_vanilla_baseline=data.get("always_include_vanilla_baseline", True),
        exotic_comparison_available=data.get("exotic_comparison_available", True),
        default_tenor_buffer_days=data.get("default_tenor_buffer_days", 5),
    )
