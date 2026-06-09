"""Parity harness — the new product pricer must reproduce the legacy
`structure_pricer.price_variants` byte-for-byte on every ported family, across
the supported pairs/directions/tenors, under both flat and smile surfaces.

This is the linchpin of the product-model refactor (PRODUCT_MODEL_PLAN.md): it lets the
internal pricer be replaced with the legacy result as the golden reference. Families not
yet ported are simply skipped (build_structure returns None).
"""

import math

import pytest

from analytics.distributions import interpolate_atm_vol
from analytics.market_state import compute_market_state
from analytics.product_pricer import build_structure, price
from analytics.structure_pricer import _load_variants, price_variants
from agentic.standard_pack import target_from_reference
from data.snapshot_loader import load_snapshot
from pricing.forwards import rate_context_for_snapshot

_SNAP = load_snapshot()
_PORTED = ["vanilla", "1x1_spread", "1x1.5_spread", "1x2_spread", "seagull"]

# (pair, horizon_days, magnitude_pct, direction)
_MARKETS = [
    ("USDBRL", 90, 6.0, "base_higher"),
    ("USDBRL", 90, 6.0, "base_lower"),
    ("USDBRL", 90, 12.0, "base_lower"),   # extended target
    ("EURPLN", 180, 5.0, "base_higher"),
    ("EURPLN", 180, 5.0, "base_lower"),
    ("GBPUSD", 60, 3.0, "base_higher"),
]


def _make_ms(pair, horizon_days, magnitude_pct, direction, surface):
    ccy = _SNAP.get(pair)
    T = horizon_days / 365.0
    rc = rate_context_for_snapshot(ccy, T)
    vol = interpolate_atm_vol(ccy, horizon_days)
    target = target_from_reference(rc.forward, direction, magnitude_pct)
    ms = compute_market_state(
        spot=rc.spot, fwd=rc.forward, vol=vol, T=T, r_d=rc.r_d, r_f=rc.r_f,
        target=target, direction=direction, surface=surface,
    )
    return ms, target


def _surface_for(pair, flat):
    if flat:
        return None
    from analytics.vol_surface import build_vol_surface
    try:
        return build_vol_surface(_SNAP.get(pair))
    except Exception:
        return None


def _close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)


def _assert_parity(new, old, ctx):
    assert len(new.strikes) == len(old.strikes), f"{ctx}: strike count"
    for kn, ko in zip(new.strikes, old.strikes):
        assert _close(kn, ko), f"{ctx}: strike {kn} != {ko}"
    for fld in ("net_premium_pct", "payoff_at_target_pct", "rr_at_target",
                "max_loss_pct", "breakeven", "wing_ratio"):
        assert _close(getattr(new, fld), getattr(old, fld)), (
            f"{ctx}: {fld} {getattr(new, fld)} != {getattr(old, fld)}"
        )
    assert new.is_zero_cost == old.is_zero_cost, f"{ctx}: is_zero_cost"


@pytest.mark.parametrize("flat", [True, False], ids=["flat", "smile"])
@pytest.mark.parametrize("pair,horizon,mag,direction", _MARKETS)
def test_parity_ported_families(pair, horizon, mag, direction, flat):
    surface = _surface_for(pair, flat)
    ms, target = _make_ms(pair, horizon, mag, direction, surface)
    is_call = direction == "base_higher"
    cfg = _load_variants()

    # Stop price (seagull max-loss) — derived as the engine does (R:R default 3.0).
    move_pct = abs(target - ms.fwd) / ms.fwd
    stop_pct = move_pct / 3.0
    stop_price = ms.fwd * (1 - stop_pct) if is_call else ms.fwd * (1 + stop_pct)

    for family in _PORTED:
        legacy = price_variants(
            ms, family, target=target, is_call=is_call, stop_price=stop_price, smile=surface
        )
        by_label = {v.variant_label: v for v in legacy}
        for variant in cfg[family]:
            # Legacy drops variants its eligibility gate rejects (e.g. half_sigma below
            # min_target_z); compare only the variants it actually priced.
            if variant["label"] not in by_label:
                continue
            st = build_structure(family, variant, is_call)
            assert st is not None, f"{family} not built"
            new = price(st, ms, target=target, smile=surface, stop_price=stop_price)
            old = by_label[variant["label"]]
            _assert_parity(new, old, f"{pair} {direction} {family} '{variant['label']}' flat={flat}")
