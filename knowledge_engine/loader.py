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

def _supported_pairs() -> list[str]:
    """Every pair with a facts file — the source of truth for which pairs have
    conventions loaded. Adding knowledge/facts/{PAIR}.json is what "expanding the
    set" means here; no separate allowlist to keep in sync."""
    return sorted(p.stem for p in _FACTS_DIR.glob("*.json"))


@lru_cache(maxsize=None)
def load_convention_facts(pair: str) -> dict:
    """Load raw convention facts for a currency pair."""
    supported = _supported_pairs()
    if pair not in supported:
        raise ValueError(f"Unsupported pair '{pair}'. Supported: {supported}")
    return _load(_FACTS_DIR / f"{pair}.json")


@lru_cache(maxsize=None)
def load_all_convention_facts() -> dict[str, dict]:
    return {pair: load_convention_facts(pair) for pair in _supported_pairs()}


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


_affinity_cache: dict | None = None


def load_affinity_scores() -> dict:
    global _affinity_cache
    if _affinity_cache is not None:
        return _affinity_cache
    try:
        from interface.supabase_logger import fetch_config_for_engine
        data = fetch_config_for_engine("affinity_scores")
        if data:
            _affinity_cache = data
            return _affinity_cache
    except Exception:
        pass
    with open(_DEFAULTS_DIR / "affinity_scores.json") as f:
        _affinity_cache = json.load(f)
    return _affinity_cache


def clear_affinity_scores_cache() -> None:
    global _affinity_cache
    _affinity_cache = None


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
