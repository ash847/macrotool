"""Retained pre-migration entry routines; never used by production dispatch."""

import math

from analytics import structure_pricer as legacy
from analytics.expiry_breakeven import expiry_breakevens, variant_terms


def legacy_entry(ms, family, definitions, target, is_call, smile, *, stop_price=None):
    fn = {"vanilla": legacy._vanilla, "1x1_spread": legacy._spread,
          "1x1.5_spread": legacy._1x1p5, "1x2_spread": legacy._1x2,
          "1x2x1_spread": legacy._1x2x1, "seagull": legacy._seagull}[family]
    vm = legacy._VolModel(ms.vol, smile, ms.fwd, round(ms.T * 365))
    args = (definitions, ms.fwd, ms.vol, ms.T, math.exp(-ms.r_d * ms.T),
            ms.r_d, ms.r_f, ms.spot, ms.vol * math.sqrt(ms.T), is_call, target)
    rows = fn(*args, stop_price, vm) if family == "seagull" else fn(*args, vm)
    for row in rows:
        terms = variant_terms(family, row.strikes, is_call, row.wing_ratio)
        row.breakevens, row.breakeven_has_zero_region = expiry_breakevens(terms, row.net_premium_pct)
        roots = row.breakevens
        row.breakeven = roots[0] if roots is not None and len(roots) == 1 and not row.breakeven_has_zero_region else None
    return rows
