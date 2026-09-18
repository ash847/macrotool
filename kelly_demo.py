"""Kelly vs fixed-loss sizing — CLI comparison (no LLM, no UI).

Prices each shortlisted family's representative variant two ways — equal max loss
(today's default) and Kelly (sized to the growth-optimal bet under a view-implied
distribution) — and prints the notionals side by side, so the comparative effect
is visible without the Streamlit UI.

    .venv/bin/python kelly_demo.py --pair USDTRY --magnitude 6 --lambda 0.5
"""
from __future__ import annotations

import argparse
import math

from agentic.standard_pack import build_pack
from analytics.payoffs import base_ccy_payoff_for_trade_rec
from analytics.sizing import (
    SizingSpec,
    kelly_fraction_per_notional,
    per_notional_pnl,
    view_implied_distribution,
)
from analytics.structure_pricer import price_variants
from config.loader import load_config
from data.snapshot_loader import load_snapshot
from knowledge_engine.models import TradeView


def run(pair, direction, horizon, magnitude, conviction, lam):
    ccy = load_snapshot().get(pair)
    cfg = load_config()
    view = TradeView(pair=pair, direction=direction, direction_conviction=conviction,
                     horizon_days=horizon, magnitude_pct=magnitude, mode="recommend")
    pack = build_pack(view, ccy, cfg)
    ms, target = pack.market_state, pack.target
    is_call = direction == "base_higher"
    loss_budget = pack.loss_budget

    import numpy as np
    probs, bins = view_implied_distribution(ms.spot, ms.fwd, ms.vol, ms.T, target, conviction)
    probs_a, bins_a = np.array(probs), np.array(bins)
    df_f = math.exp(-ms.r_f * ms.T)
    spec = SizingSpec(method="kelly", kelly_lambda=lam, bankroll=100.0,
                      kelly_probs=probs, kelly_bins=bins)

    print(f"\n{pair}  {direction}  {horizon}d  target≈{target:.4f}  "
          f"spot={ms.spot:.4f} fwd={ms.fwd:.4f} vol={ms.vol:.1%}  conviction={conviction}  λ={lam}")
    print(f"{'family':<18}{'fixed N':>12}{'kelly N':>12}{'x* (full)':>12}{'flag':>14}")
    print("-" * 68)
    cap = 1000.0
    for item in pack.selector_result.shortlist:
        fam = item.structure_id
        try:
            fixed = price_variants(ms, fam, target=target, is_call=is_call,
                                   stop_price=None, loss_budget=loss_budget)
            kelly = price_variants(ms, fam, target=target, is_call=is_call,
                                   stop_price=None, sizing_spec=spec)
        except Exception as e:
            print(f"{fam:<18}  error: {e}")
            continue
        if not fixed or not kelly:
            continue
        fn = fixed[0].structure_notional or 0.0
        kn = kelly[0].structure_notional or 0.0
        try:
            payoff = base_ccy_payoff_for_trade_rec(fam, strikes=kelly[0].strikes,
                                                   barrier=kelly[0].barrier, is_call=is_call,
                                                   entry_spot=ms.spot, wing_ratio=kelly[0].wing_ratio)
            pnl = per_notional_pnl(payoff(bins_a), kelly[0].net_premium_pct, df_f)
            xstar = kelly_fraction_per_notional(probs_a, pnl)
        except Exception:
            xstar = float("nan")
        flag = "no edge" if kn <= 0 else ("capped" if kn >= cap - 1e-6 else "")
        print(f"{fam:<18}{fn:>12,.0f}{kn:>12,.0f}{xstar:>12.2f}{flag:>14}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Kelly vs fixed-loss sizing comparison")
    p.add_argument("--pair", default="USDTRY")
    p.add_argument("--direction", default="base_higher", choices=["base_higher", "base_lower"])
    p.add_argument("--horizon", default=91, type=int)
    p.add_argument("--magnitude", default=6.0, type=float)
    p.add_argument("--conviction", default="high", choices=["high", "medium", "low"])
    p.add_argument("--lambda", dest="lam", default=0.5, type=float)
    a = p.parse_args()
    run(a.pair, a.direction, a.horizon, a.magnitude, a.conviction, a.lam)
