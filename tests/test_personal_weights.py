"""Per-user scenario-weights profile resolution + per-profile cache isolation.

The loader resolves: personal profile (allowlisted user, non-sentinel config) →
global → local JSON. The per-profile cache must not bleed one user's weights to
another (critical on the shared Streamlit Cloud process).
"""

import json

import pytest

import knowledge_engine.scenario_weighter as sw


# A minimal valid scenario_definitions config with a recognisable marker so we can
# tell which profile was loaded.
def _cfg(marker: float) -> dict:
    return {
        "baseline": marker,            # marker rides on `baseline` — easy to assert
        "min_multiplier": 0.1,
        "base_weightings": [],
        "preference_overlays": [],
    }


GLOBAL = _cfg(1.0)
PERSONAL = _cfg(2.0)
SENTINEL = {"_inherit_global": True}


@pytest.fixture
def patched(monkeypatch):
    """Patch the allowlist + Supabase fetch the loader lazy-imports, and reset cache."""
    sw.clear_scenario_weights_cache()

    allow = {"vip@x.com"}
    monkeypatch.setattr(
        "interface.security.can_have_personal_weights",
        lambda email: email in allow,
    )

    store: dict[str, dict] = {}

    def fake_fetch(key):
        return (store.get(key), "supabase" if key in store else "missing")

    monkeypatch.setattr(
        "interface.supabase_logger.fetch_config_for_engine_with_meta", fake_fetch
    )
    monkeypatch.setattr(
        "interface.supabase_logger.personal_weights_key",
        lambda email: f"scenario_definitions::{email}",
    )
    yield store
    sw.clear_scenario_weights_cache()


def test_allowlisted_user_with_personal_config_gets_personal(patched):
    patched["scenario_definitions"] = GLOBAL
    patched["scenario_definitions::vip@x.com"] = PERSONAL
    assert sw.load_scenario_weights_config("vip@x.com")["baseline"] == 2.0


def test_non_allowlisted_user_gets_global(patched):
    patched["scenario_definitions"] = GLOBAL
    patched["scenario_definitions::other@x.com"] = PERSONAL  # exists but not allowlisted
    assert sw.load_scenario_weights_config("other@x.com")["baseline"] == 1.0


def test_allowlisted_user_without_personal_config_falls_back_to_global(patched):
    patched["scenario_definitions"] = GLOBAL
    assert sw.load_scenario_weights_config("vip@x.com")["baseline"] == 1.0


def test_sentinel_reverts_to_global(patched):
    patched["scenario_definitions"] = GLOBAL
    patched["scenario_definitions::vip@x.com"] = SENTINEL
    assert sw.load_scenario_weights_config("vip@x.com")["baseline"] == 1.0


def test_none_user_gets_global(patched):
    patched["scenario_definitions"] = GLOBAL
    assert sw.load_scenario_weights_config(None)["baseline"] == 1.0


def test_cache_isolation_between_profiles(patched):
    """Two users must not share a cache entry — the shared-process correctness fix."""
    patched["scenario_definitions"] = GLOBAL
    patched["scenario_definitions::vip@x.com"] = PERSONAL

    assert sw.load_scenario_weights_config("vip@x.com")["baseline"] == 2.0  # personal
    assert sw.load_scenario_weights_config("other@x.com")["baseline"] == 1.0  # global
    # Re-read the personal one: still personal (not clobbered by the global read).
    assert sw.load_scenario_weights_config("vip@x.com")["baseline"] == 2.0


def test_clear_one_profile_leaves_others(patched):
    patched["scenario_definitions"] = GLOBAL
    patched["scenario_definitions::vip@x.com"] = PERSONAL
    sw.load_scenario_weights_config("vip@x.com")
    sw.load_scenario_weights_config(None)

    sw.clear_scenario_weights_cache("scenario_definitions::vip@x.com")
    # Personal entry re-fetches (now changed); global entry stays cached.
    patched["scenario_definitions::vip@x.com"] = _cfg(3.0)
    patched["scenario_definitions"] = _cfg(9.0)  # would only show if global cache cleared
    assert sw.load_scenario_weights_config("vip@x.com")["baseline"] == 3.0  # re-fetched
    assert sw.load_scenario_weights_config(None)["baseline"] == 1.0  # still cached
