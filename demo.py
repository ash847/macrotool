"""
Pipeline demo — runs the full recommend-mode workflow without the LLM.

Shows exactly what the LLM will receive as structured context at each step.
Useful for verifying domain logic before wiring up the conversation layer.

Usage:
    .venv/bin/python demo.py
    .venv/bin/python demo.py --pair USDTRY --direction base_higher --conviction high
    .venv/bin/python demo.py --mode critique --structure risk_reversal
"""

from __future__ import annotations

import argparse
import math
import textwrap

from config.loader import load_config
from config.schema import SessionOverrides
from data.snapshot_loader import load_snapshot
from knowledge_engine.conventions import resolve as resolve_conventions, format_for_context
from knowledge_engine.models import TradeView
from knowledge_engine.structure_scorer import score_structures
from analytics.market_state import compute_market_state
from analytics.distributions import interpolate_atm_vol
from knowledge_engine.sizing_engine import compute_sizing, format_sizing_for_context
from knowledge_engine.critique_engine import evaluate_structure
from pricing.forwards import build_rate_context, tenor_to_years, DEFAULT_SETTLEMENT_RATES
from pricing.black_scholes import call_value, put_value
from pricing.scenario import build_scenario_matrix, ScenarioConfig, format_scenario_table


def run_demo(
    pair: str = "USDBRL",
    direction: str = "base_higher",
    direction_conviction: str = "high",
    timing_conviction: str = "medium",
    horizon_days: int = 91,
    magnitude_pct: float = 5.0,
    budget_usd: float = 200_000,
    catalyst: str = "Federal Reserve rate decision + Lula fiscal announcement",
    mode: str = "recommend",
    pm_structure: str | None = None,
    session_kelly: float | None = None,
) -> None:
    separator = "=" * 72

    print(separator)
    print(f"  MacroTool Pipeline Demo — {pair} | {mode.upper()} mode")
    print(separator)

    # ------------------------------------------------------------------
    # Load market data and config
    # ------------------------------------------------------------------
    snapshot = load_snapshot()
    ccy = snapshot.get(pair)
    cfg = load_config()

    # Apply any session overrides
    if session_kelly is not None:
        session = SessionOverrides()
        session.apply("sizing.kelly.default_fraction", session_kelly, "session",
                      f"demo override: kelly={session_kelly}")
        from config.resolver import resolve
        cfg = resolve(cfg, None, session)
        print(f"\n[SESSION OVERRIDE] Kelly fraction set to {session_kelly}")

    # ------------------------------------------------------------------
    # Build TradeView
    # ------------------------------------------------------------------
    view = TradeView(
        pair                = pair,
        direction           = direction,
        direction_conviction= direction_conviction,
        timing_conviction   = timing_conviction,
        horizon_days        = horizon_days,
        magnitude_pct       = magnitude_pct,
        budget_usd          = budget_usd,
        catalyst            = catalyst,
        mode                = mode,
        pm_structure_description = pm_structure,
    )

    print(f"\n{'─'*72}")
    print("STEP 0 — TRADE VIEW (extracted from PM input)")
    print(f"{'─'*72}")
    print(f"  Pair:               {view.pair}")
    print(f"  Direction:          {view.direction}")
    print(f"  Direction conviction: {view.direction_conviction}")
    print(f"  Timing conviction:  {view.timing_conviction}")
    print(f"  Horizon:            {view.horizon_days} days ({view.horizon_years:.2f}y)")
    print(f"  Target move:        {view.magnitude_pct}%")
    print(f"  Budget:             ${view.budget_usd:,.0f}" if view.budget_usd else "  Budget:             not provided")
    print(f"  Catalyst:           {view.catalyst or 'not stated'}")

    # ------------------------------------------------------------------
    # Step 1 — Conventions (what the LLM will cite)
    # ------------------------------------------------------------------
    print(f"\n{'─'*72}")
    print("STEP 1 — CONVENTIONS (injected into LLM context)")
    print(f"{'─'*72}")
    conventions = resolve_conventions(pair)
    print(format_for_context(conventions))

    # ------------------------------------------------------------------
    # Step 2 — Structure selection
    # ------------------------------------------------------------------
    print(f"\n{'─'*72}")
    print("STEP 2 — STRUCTURE SELECTION (quantitative scorer)")
    print(f"{'─'*72}")
    r_d = DEFAULT_SETTLEMENT_RATES.get(pair, 0.043)
    rate_ctx = build_rate_context(ccy, view.horizon_years, r_d)
    atm_vol = interpolate_atm_vol(ccy, view.horizon_days)
    target = ccy.spot * (1 + (1 if view.direction == "base_higher" else -1) * (view.magnitude_pct or 0) / 100)
    ms = compute_market_state(
        spot=rate_ctx.spot, fwd=rate_ctx.forward, vol=atm_vol,
        T=view.horizon_years, r_d=rate_ctx.r_d, r_f=rate_ctx.r_f,
        target=target if view.magnitude_pct else None,
        direction=view.direction,
    )
    selector_result = score_structures(ms)

    print(f"\nRules fired: {', '.join(selector_result.rules_fired)}")
    print(f"\nShortlist:")
    for item in selector_result.shortlist:
        exotic_tag = " [EXOTIC — comparison only]" if item.is_exotic else ""
        print(f"  {item.rank}. {item.display_name}{exotic_tag}")
        print(f"     Optimised for: {item.optimised_for}")
        print(f"     Rationale: {textwrap.fill(item.rationale, width=66, subsequent_indent='     ')}")
        if item.caution:
            print(f"     ⚠ Caution: {textwrap.fill(item.caution, width=63, subsequent_indent='       ')}")
        if item.sizing_modifier:
            print(f"     Sizing: {item.sizing_modifier}")

    # ------------------------------------------------------------------
    # Step 2b — Scenario matrix for top structure
    # ------------------------------------------------------------------
    if selector_result.shortlist:
        top = selector_result.shortlist[0]
        print(f"\nScenario matrix — {top.display_name}")

        rate_ctx = build_rate_context(
            ccy,
            tenor_to_years("3M"),
            DEFAULT_SETTLEMENT_RATES[pair],
        )
        atm_vol = ccy.get_atm_vol("3M")
        strike = rate_ctx.forward  # ATM forward

        # Build pricer fn based on direction
        if top.structure_id in ("vanilla_call", "call_spread", "spot", "forward"):
            def pricer(spot, T_rem, sigma):
                return call_value(spot, strike, T_rem, sigma, rate_ctx.r_d, rate_ctx.r_f)
        else:
            def pricer(spot, T_rem, sigma):
                return put_value(spot, strike, T_rem, sigma, rate_ctx.r_d, rate_ctx.r_f)

        initial_value = pricer(rate_ctx.spot, rate_ctx.T, atm_vol)

        scenario_cfg = ScenarioConfig(
            spot_range_pct=8,
            spot_steps=5,
            vol_range_pct=25,
            vol_steps=3,
            time_horizons_years=[1/12, 2/12, 3/12],
            tenor_labels=["1M", "2M", "3M"],
        )
        matrix = build_scenario_matrix(pricer, rate_ctx.spot, atm_vol, rate_ctx.T,
                                       initial_value, scenario_cfg)
        notional = 1_000_000  # $1M for illustration
        print(format_scenario_table(matrix, notional_scale=notional,
                                    title=f"P&L ($) — long {top.display_name} on ${notional:,.0f} notional"))

    # ------------------------------------------------------------------
    # Step 3 — Sizing
    # ------------------------------------------------------------------
    print(f"\n{'─'*72}")
    print("STEP 3 — SIZING (pre-computed for LLM narration)")
    print(f"{'─'*72}")
    top_structure = selector_result.shortlist[0]
    sizing = compute_sizing(view, ccy, top_structure, cfg)
    print(format_sizing_for_context(sizing))

    print(f"\n  Kelly fraction:         {sizing.kelly_fraction:.2f} ({sizing.kelly_conviction_used} conviction)")
    print(f"  Adjusted Kelly:         {sizing.adjusted_kelly:.3f}")
    if sizing.base_notional_usd:
        print(f"  Base notional:          ${sizing.base_notional_usd:,.0f}")
        print(f"  Kelly-adjusted notional:${sizing.kelly_notional_usd:,.0f}")
    print(f"  Stop level:             {sizing.stop_level:.4f} ({sizing.stop_distance_pct:.2f}% from spot {ccy.spot:.4f})")
    if sizing.tranche_schedule:
        pct_str = ", ".join(f"{w*100:.0f}%" for w in sizing.tranche_schedule)
        print(f"  Tranche schedule:       {sizing.tranche_count} tranches [{pct_str}]")
    if sizing.take_profit_levels:
        print("  Take profits:")
        for tp in sizing.take_profit_levels:
            spot_str = f" → spot {tp.target_spot:.4f}" if tp.target_spot else ""
            print(f"    At {tp.at_pct_of_target*100:.0f}% of target{spot_str}: {tp.note}")

    # ------------------------------------------------------------------
    # Critique mode
    # ------------------------------------------------------------------
    if mode == "critique" and pm_structure:
        print(f"\n{'─'*72}")
        print(f"CRITIQUE — evaluating '{pm_structure}' against view")
        print(f"{'─'*72}")
        critique = evaluate_structure(view, pm_structure, selector_result, sizing)
        print(f"\nVerdict: {critique.verdict.upper().replace('_', ' ')}")
        print(f"Recommended alternative: {critique.recommended_alternative}")
        print(f"\nEV comparison:\n  {textwrap.fill(critique.ev_comparison_note, 68)}")
        print(f"\nScenario weakness:\n  {textwrap.fill(critique.scenario_weakness, 68)}")
        print(f"\nExecution notes:\n  {textwrap.fill(critique.execution_notes, 68)}")
        print(f"\nGamma profile:\n  {textwrap.fill(critique.gamma_notes, 68)}")
        print(f"\nHedge effectiveness:\n  {textwrap.fill(critique.hedge_effectiveness, 68)}")
        print(f"\nDimension scores:")
        for dim, score in critique.dimension_scores.items():
            print(f"  {dim}: {score}")

    print(f"\n{separator}")
    print("  End of pre-LLM context — above is what the LLM receives")
    print(separator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MacroTool pipeline demo")
    parser.add_argument("--pair", default="USDBRL", choices=["USDBRL", "USDTRY", "EURPLN"])
    parser.add_argument("--direction", default="base_higher", choices=["base_higher", "base_lower"])
    parser.add_argument("--conviction", default="high", choices=["high", "medium", "low"])
    parser.add_argument("--timing", default="medium", choices=["high", "medium", "low"])
    parser.add_argument("--horizon", default=91, type=int)
    parser.add_argument("--magnitude", default=5.0, type=float)
    parser.add_argument("--budget", default=200_000, type=float)
    parser.add_argument("--mode", default="recommend", choices=["recommend", "critique"])
    parser.add_argument("--structure", default=None, help="PM structure (critique mode)")
    parser.add_argument("--kelly", default=None, type=float, help="Override Kelly fraction")
    args = parser.parse_args()

    run_demo(
        pair               = args.pair,
        direction          = args.direction,
        direction_conviction = args.conviction,
        timing_conviction  = args.timing,
        horizon_days       = args.horizon,
        magnitude_pct      = args.magnitude,
        budget_usd         = args.budget,
        mode               = args.mode,
        pm_structure       = args.structure,
        session_kelly      = args.kelly,
    )
