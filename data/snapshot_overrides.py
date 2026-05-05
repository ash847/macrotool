"""
Market snapshot override application.

Builds a fresh MarketSnapshot from a base snapshot plus sparse per-pair
overrides stored in Streamlit session state.
"""

from __future__ import annotations

from copy import deepcopy

from data.schema import MarketSnapshot


def apply_overrides(base: MarketSnapshot, overrides: dict) -> MarketSnapshot:
    """
    Return a new MarketSnapshot with sparse overrides merged onto `base`.

    Supported override shape:
      {
        "USDBRL": {
          "forwards": {"1M": 5.82},
          "atm_vols": {"1M": 0.182},
          "risk_reversals": {"1M": {"10": 0.049, "25": 0.026}},
          "butterflies": {"1M": {"10": 0.053, "25": 0.023}},
        }
      }
    """
    raw = deepcopy(base.model_dump(mode="python"))
    currencies = raw["currencies"]

    for pair, pair_override in overrides.items():
        if pair not in currencies:
            raise ValueError(f"Unknown pair override '{pair}'")
        pair_raw = currencies[pair]
        _apply_pair_overrides(pair_raw, pair_override)

    return MarketSnapshot.model_validate(raw)


def _apply_pair_overrides(pair_raw: dict, pair_override: dict) -> None:
    spot = pair_raw["spot"]

    if "forwards" in pair_override:
        scale = _infer_forward_point_scale(pair_raw["forwards"], spot)
        forward_map = {f["tenor"]: f for f in pair_raw["forwards"]}
        for tenor, outright in pair_override["forwards"].items():
            if tenor not in forward_map:
                raise ValueError(f"Unknown forward tenor '{tenor}' for {pair_raw['pair']}")
            point = forward_map[tenor]
            point["outright"] = float(outright)
            point["points"] = round((float(outright) - spot) * scale, 6)

    if any(k in pair_override for k in ("atm_vols", "risk_reversals", "butterflies")):
        vol_map = {(n["tenor"], n["delta"]): n for n in pair_raw["vol_surface"]}
        tenors = sorted({n["tenor"] for n in pair_raw["vol_surface"]})
        delta_pairs = sorted({
            delta[:-2]
            for tenor, delta in vol_map
            if delta.endswith("DC") and (tenor, f"{delta[:-2]}DP") in vol_map
        })

        _validate_vol_override_keys(pair_raw["pair"], tenors, delta_pairs, pair_override)

        for tenor in tenors:
            base_atm = float(vol_map[(tenor, "ATM")]["vol"])
            atm = _override_or_base(pair_override.get("atm_vols", {}), tenor, base_atm)
            vol_map[(tenor, "ATM")]["vol"] = float(atm)

            for delta in delta_pairs:
                call_key = (tenor, f"{delta}DC")
                put_key = (tenor, f"{delta}DP")
                base_call = vol_map[call_key]["vol"]
                base_put = vol_map[put_key]["vol"]
                base_rr = base_call - base_put
                base_bf = 0.5 * (base_call + base_put) - base_atm
                rr = _nested_override_or_base(pair_override.get("risk_reversals", {}), tenor, delta, base_rr)
                bf = _nested_override_or_base(pair_override.get("butterflies", {}), tenor, delta, base_bf)
                vol_map[call_key]["vol"] = float(atm) + float(bf) + float(rr) / 2.0
                vol_map[put_key]["vol"] = float(atm) + float(bf) - float(rr) / 2.0


def _override_or_base(overrides: dict, tenor: str, base_value: float) -> float:
    if tenor in overrides:
        return float(overrides[tenor])
    return float(base_value)


def _nested_override_or_base(overrides: dict, tenor: str, delta: str, base_value: float) -> float:
    if tenor in overrides:
        tenor_values = overrides[tenor]
        if delta in tenor_values:
            return float(tenor_values[delta])
    return float(base_value)


def _infer_forward_point_scale(forwards: list[dict], spot: float) -> float:
    for row in forwards:
        diff = row["outright"] - spot
        if abs(diff) < 1e-12 or abs(row["points"]) < 1e-12:
            continue
        return row["points"] / diff
    raise ValueError("Could not infer forward point scale from snapshot")


def _validate_vol_override_keys(pair: str, tenors: list[str], delta_pairs: list[str], pair_override: dict) -> None:
    for tenor in pair_override.get("atm_vols", {}):
        if tenor not in tenors:
            raise ValueError(f"Unknown ATM tenor '{tenor}' for {pair}")

    for section_name in ("risk_reversals", "butterflies"):
        for tenor, delta_map in pair_override.get(section_name, {}).items():
            if tenor not in tenors:
                raise ValueError(f"Unknown {section_name} tenor '{tenor}' for {pair}")
            for delta in delta_map:
                if delta not in delta_pairs:
                    raise ValueError(f"Unknown {section_name} delta '{delta}' for {pair}")
