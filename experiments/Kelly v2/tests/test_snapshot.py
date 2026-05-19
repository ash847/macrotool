import json
from pathlib import Path

import numpy as np
import pytest

from baseline import (
    SNAPSHOT_SCHEMA_VERSION,
    load_snapshot,
    save_snapshot,
    synthetic_lognormal_baseline,
)
from elicitation import Distribution
from pricing import forward_of, price_vanilla


FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def test_round_trip_synthetic_baseline(tmp_path):
    dist = synthetic_lognormal_baseline(forward=5.0, sigma=0.10, tenor_years=0.25, n_bins=200)
    path = tmp_path / "snap.json"
    save_snapshot(
        dist, path,
        pair="USDBRL", forward=5.0, tenor_years=0.25, source="synthetic_lognormal_v1",
    )

    loaded, meta = load_snapshot(path)
    assert isinstance(loaded, Distribution)
    assert loaded.n_bins == dist.n_bins
    np.testing.assert_allclose(loaded.bins, dist.bins)
    np.testing.assert_allclose(loaded.probs, dist.probs)

    assert meta["pair"] == "USDBRL"
    assert meta["forward"] == 5.0
    assert meta["tenor_years"] == 0.25
    assert meta["source"] == "synthetic_lognormal_v1"
    assert meta["schema_version"] == SNAPSHOT_SCHEMA_VERSION


def test_committed_fixture_loads_and_prices_sensibly():
    """The committed synthetic_usdbrl_3m.json fixture must load and produce
    sane pricing — ATM call should be within 5 bp of Black-Scholes."""
    path = FIXTURE_DIR / "synthetic_usdbrl_3m.json"
    dist, meta = load_snapshot(path)
    assert meta["pair"] == "USDBRL"
    assert dist.probs.sum() == pytest.approx(1.0, abs=1e-6)

    fwd = meta["forward"]
    # Forward implied by the distribution should be close to the metadata forward.
    implied_fwd = forward_of(dist)
    assert implied_fwd == pytest.approx(fwd, rel=1e-3)

    # Sanity: ATM call price should be positive and well-behaved.
    call = price_vanilla(dist, fwd, is_call=True)
    assert call > 0
    assert call < fwd  # bounded by the forward


def test_rejects_unknown_schema_version(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 999, "bins": [], "probs": []}))
    with pytest.raises(ValueError, match="schema_version"):
        load_snapshot(path)


def test_rejects_misaligned_bins_probs(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "bins": [1.0, 2.0, 3.0],
        "probs": [0.5, 0.5],
    }))
    with pytest.raises(ValueError, match="same shape"):
        load_snapshot(path)


def test_rejects_non_monotonic_bins(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "bins": [1.0, 3.0, 2.0],
        "probs": [1.0 / 3, 1.0 / 3, 1.0 / 3],
    }))
    with pytest.raises(ValueError, match="strictly increasing"):
        load_snapshot(path)


def test_rejects_probs_not_summing_to_one(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "bins": [1.0, 2.0, 3.0],
        "probs": [0.3, 0.3, 0.3],
    }))
    with pytest.raises(ValueError, match="sum to 1"):
        load_snapshot(path)


def test_rejects_negative_probs(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "bins": [1.0, 2.0, 3.0],
        "probs": [0.5, -0.1, 0.6],
    }))
    with pytest.raises(ValueError, match="non-negative"):
        load_snapshot(path)
