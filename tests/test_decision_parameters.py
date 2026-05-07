from interface.decision_parameters import (
    _local_metadata_differs,
    _merge_metadata,
    _missing_local_structures,
)


def test_missing_local_structures_detects_new_branch_structure():
    loaded = {
        "structures": {
            "vanilla": {},
            "rko": {},
        }
    }
    local = {
        "structures": {
            "vanilla": {},
            "rko": {},
            "european_rko": {},
        }
    }

    assert _missing_local_structures(loaded, local) == ["european_rko"]


def test_local_metadata_differs_detects_branch_schema_change():
    loaded = {
        "_bucket_labels": {"target_z_abs": ["near", "far"]},
        "structures": {"vanilla": {}},
    }
    local = {
        "_bucket_labels": {"target_z_abs": ["near", "moderate", "far"]},
        "structures": {"vanilla": {}},
    }

    assert _local_metadata_differs(loaded, local) is True


def test_merge_metadata_preserves_original_headers_and_working_values():
    original = {
        "_description": "remote header",
        "_bucket_labels": {"target_z_abs": ["near", "far"]},
        "thresholds": {"target_z_abs": [0.5, 1.25, 1.75]},
    }
    working = {
        "thresholds": {"target_z_abs": [0.4, 1.2, 1.8]},
        "structures": {"european_rko": {"gates": {}}},
    }

    out = _merge_metadata(original, working)

    assert out["_description"] == "remote header"
    assert out["_bucket_labels"] == {"target_z_abs": ["near", "far"]}
    assert out["thresholds"] == {"target_z_abs": [0.4, 1.2, 1.8]}
    assert "european_rko" in out["structures"]
