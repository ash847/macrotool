"""
Detects preference-change intents from LLM responses and converts them to SessionOverrides.

The LLM is prompted to emit structured tags when it detects a preference change:
  [PREF_CHANGE: {"field": "sizing.kelly.default_fraction", "value": 0.25, "scope": "session"}]

This module parses those tags out of the LLM response before it is shown to the PM.
"""

from __future__ import annotations

import json
import re
from typing import Any

from config.schema import SessionOverride, SessionOverrides

# The LLM must only use field paths from this allowlist.
# Convention facts are never included — they cannot be overridden.
OVERRIDABLE_FIELDS: dict[str, dict] = {
    "sizing.kelly.default_fraction": {
        "type": float,
        "min": 0.1,
        "max": 1.0,
        "description": "Kelly fraction for default conviction level",
    },
    "sizing.kelly.high_conviction_fraction": {
        "type": float,
        "min": 0.1,
        "max": 1.0,
        "description": "Kelly fraction for high conviction",
    },
    "sizing.kelly.low_conviction_fraction": {
        "type": float,
        "min": 0.1,
        "max": 1.0,
        "description": "Kelly fraction for low conviction",
    },
    "sizing.stop.atr_multiple": {
        "type": float,
        "min": 0.5,
        "max": 5.0,
        "description": "ATR multiple for vol-derived stop placement",
    },
    "structures.excluded_structures": {
        "type": list,
        "description": "Structures the PM does not want recommended",
    },
    "structures.preferred_structures": {
        "type": list,
        "description": "Structures the PM prefers to see first",
    },
    "structures.max_shortlist_size": {
        "type": int,
        "min": 1,
        "max": 5,
        "description": "Maximum number of structures in the recommendation shortlist",
    },
    "structures.exotic_comparison_available": {
        "type": bool,
        "description": "Whether exotic comparisons are offered",
    },
    "display.scenario_matrix.time_horizons": {
        "type": list,
        "description": "Time horizons shown in the scenario matrix",
    },
    "display.scenario_matrix.spot_range_pct": {
        "type": float,
        "min": 1.0,
        "max": 30.0,
        "description": "Spot range as ± percentage in the scenario matrix",
    },
    "vol_regime.regime_percentile_thresholds.elevated.min_percentile": {
        "type": float,
        "min": 0.5,
        "max": 0.95,
        "description": "Percentile threshold above which vol regime is classified as elevated",
    },
}


_TAG_PATTERN = re.compile(
    r'\[PREF_CHANGE:\s*(\{.*?\})\]',
    re.DOTALL,
)


def extract_overrides(llm_response: str) -> tuple[str, list[SessionOverride]]:
    """
    Parse [PREF_CHANGE: {...}] tags from an LLM response.

    Returns:
        (clean_response, overrides) — response with tags stripped, and parsed overrides.
    """
    overrides: list[SessionOverride] = []
    clean = llm_response

    for match in _TAG_PATTERN.finditer(llm_response):
        raw_json = match.group(1)
        try:
            data = json.loads(raw_json)
            override = _parse_override(data, raw_text=match.group(0))
            if override is not None:
                overrides.append(override)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass  # Malformed tag — ignore silently
        clean = clean.replace(match.group(0), "")

    return clean.strip(), overrides


def _parse_override(data: dict, raw_text: str) -> SessionOverride | None:
    field = data.get("field", "")
    if field not in OVERRIDABLE_FIELDS:
        return None

    value = data.get("value")
    scope = data.get("scope", "session")
    if scope not in ("session", "profile"):
        scope = "session"

    # Type coercion and validation
    spec = OVERRIDABLE_FIELDS[field]
    try:
        value = _coerce(value, spec)
    except (TypeError, ValueError):
        return None

    return SessionOverride(
        field_path=field,
        value=value,
        scope=scope,
        raw_text=raw_text,
    )


def _coerce(value: Any, spec: dict) -> Any:
    target_type = spec["type"]
    if target_type == float:
        value = float(value)
        if "min" in spec and value < spec["min"]:
            raise ValueError(f"Value {value} below minimum {spec['min']}")
        if "max" in spec and value > spec["max"]:
            raise ValueError(f"Value {value} above maximum {spec['max']}")
    elif target_type == int:
        value = int(value)
        if "min" in spec and value < spec["min"]:
            raise ValueError
        if "max" in spec and value > spec["max"]:
            raise ValueError
    elif target_type == bool:
        if isinstance(value, str):
            value = value.lower() in ("true", "yes", "1")
        else:
            value = bool(value)
    elif target_type == list:
        if not isinstance(value, list):
            raise TypeError("Expected list")
    return value


def overridable_fields_description() -> str:
    """
    Returns a human-readable list of overridable fields for inclusion in the LLM system prompt.
    """
    lines = ["Overridable configuration fields (use in [PREF_CHANGE: ...] tags):"]
    for field, spec in OVERRIDABLE_FIELDS.items():
        desc = spec["description"]
        type_name = spec["type"].__name__
        bounds = ""
        if "min" in spec or "max" in spec:
            bounds = f" (range: {spec.get('min', '?')}–{spec.get('max', '?')})"
        lines.append(f"  {field} [{type_name}{bounds}]: {desc}")
    return "\n".join(lines)
