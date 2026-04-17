"""
Loads and validates the market snapshot from JSON.

Usage:
    from data.snapshot_loader import load_snapshot
    snapshot = load_snapshot()
    brl = snapshot.get("USDBRL")
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from data.schema import MarketSnapshot

_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_SNAPSHOT_PATH = _REPO_ROOT / "data" / "market_snapshot.json"


def load_snapshot(path: str | Path | None = None) -> MarketSnapshot:
    """
    Load and validate the market snapshot.

    Path resolution order:
      1. path argument
      2. MACROTOOL_SNAPSHOT_PATH environment variable
      3. data/market_snapshot.json (default)
    """
    if path is None:
        path = os.environ.get("MACROTOOL_SNAPSHOT_PATH", str(_DEFAULT_SNAPSHOT_PATH))
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Market snapshot not found at: {p}")

    with open(p) as f:
        raw = json.load(f)

    return MarketSnapshot.model_validate(raw)
