"""Supabase affinity config overlays local defaults (doesn't wipe local-only families)."""
from knowledge_engine.loader import _merge_affinity


def test_remote_wins_but_local_only_families_preserved():
    local = {
        "thresholds": {"carry_regime": [0.25, 0.65]},
        "structures": {
            "1x2_spread": {"target_z_abs": {"near": 2.0}},
            "1x2x1_spread": {"target_z_abs": {"near": 2.01}},   # local-only (new family)
        },
    }
    remote = {
        "thresholds": {"carry_regime": [0.30, 0.70]},           # tuned remotely
        "structures": {
            "1x2_spread": {"target_z_abs": {"near": 1.5}},       # tuned remotely
        },
    }
    merged = _merge_affinity(local, remote)
    # remote tuning wins where present
    assert merged["thresholds"]["carry_regime"] == [0.30, 0.70]
    assert merged["structures"]["1x2_spread"]["target_z_abs"]["near"] == 1.5
    # local-only family survives the overlay
    assert "1x2x1_spread" in merged["structures"]
    assert merged["structures"]["1x2x1_spread"]["target_z_abs"]["near"] == 2.01


def test_no_remote_returns_local_family_set():
    local = {"structures": {"1x2x1_spread": {}, "vanilla": {}}}
    merged = _merge_affinity(local, {"structures": {"vanilla": {"x": 1}}})
    assert set(merged["structures"]) == {"1x2x1_spread", "vanilla"}
