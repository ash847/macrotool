"""
Loads all knowledge JSON files (facts and defaults) into typed Python objects.

Keeps raw dicts available alongside typed objects — the structure selector
and sizing engine operate on the typed config, but the LLM context builder
may need the raw catalog text directly.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_FACTS_DIR    = _REPO_ROOT / "knowledge" / "facts"
_DEFAULTS_DIR = _REPO_ROOT / "knowledge" / "defaults"


def _load(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Facts (convention files — immutable)
# ---------------------------------------------------------------------------

_SUPPORTED_PAIRS = ["USDBRL", "USDTRY", "EURPLN"]


@lru_cache(maxsize=None)
def load_convention_facts(pair: str) -> dict:
    """Load raw convention facts for a currency pair."""
    if pair not in _SUPPORTED_PAIRS:
        raise ValueError(f"Unsupported pair '{pair}'. Supported: {_SUPPORTED_PAIRS}")
    return _load(_FACTS_DIR / f"{pair}.json")


@lru_cache(maxsize=None)
def load_all_convention_facts() -> dict[str, dict]:
    return {pair: load_convention_facts(pair) for pair in _SUPPORTED_PAIRS}


# ---------------------------------------------------------------------------
# Defaults (judgment layer — mutable via config system)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_structure_defaults() -> dict:
    return _load(_DEFAULTS_DIR / "structure_defaults.json")


@lru_cache(maxsize=None)
def load_sizing_defaults() -> dict:
    return _load(_DEFAULTS_DIR / "sizing_defaults.json")


@lru_cache(maxsize=None)
def load_vol_regime_defaults() -> dict:
    return _load(_DEFAULTS_DIR / "vol_regime_defaults.json")


@lru_cache(maxsize=None)
def load_critique_defaults() -> dict:
    return _load(_DEFAULTS_DIR / "critique_defaults.json")


@lru_cache(maxsize=None)
def load_structure_profiles() -> dict:
    return _load(_DEFAULTS_DIR / "structure_profiles.json")


@lru_cache(maxsize=None)
def load_affinity_scores() -> dict:
    with open(_DEFAULTS_DIR / "affinity_scores.json") as f:
        import json
        return json.load(f)  # keep _-prefixed keys (thresholds, bucket_labels etc.)


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def get_decision_rules() -> list[dict]:
    return load_structure_defaults()["decision_rules"]


def get_structure_catalog() -> dict[str, dict]:
    return load_structure_defaults()["structure_catalog"]


def get_structure_info(structure_id: str) -> dict:
    catalog = get_structure_catalog()
    if structure_id not in catalog:
        raise KeyError(f"Unknown structure '{structure_id}'. Available: {list(catalog)}")
    return catalog[structure_id]


def get_critique_dimensions() -> list[dict]:
    return load_critique_defaults()["evaluation_dimensions"]
