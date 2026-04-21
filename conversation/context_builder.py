"""
Context builder — assembles the system prompt for each flow step.

The system prompt has two parts:
  1. Static base instructions (same for all steps) — cacheable prefix
  2. Dynamic data blocks (pre-computed knowledge engine outputs) — varies per step

The LLM sees structured text blocks, never raw JSON or Python objects.
All quantitative reasoning was done in Python before this layer.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from config.override_detector import overridable_fields_description
from config.schema import ResolvedConfig
from data.schema import CurrencySnapshot
from knowledge_engine.conventions import format_for_context
from knowledge_engine.models import (
    CritiqueOutput,
    SizingOutput,
    StructureSelectionResult,
    TradeView,
)
from knowledge_engine.sizing_engine import format_sizing_for_context
from pricing.scenario import format_scenario_table, build_scenario_matrix, ScenarioConfig
from pricing.forwards import build_rate_context, tenor_to_years, DEFAULT_SETTLEMENT_RATES
from pricing.black_scholes import call_value, put_value
from analytics.models import PriceDistribution, MaturityHistogram

_PROMPTS = Path(__file__).parent / "prompts"


def _load(name: str) -> str:
    return (_PROMPTS / name).read_text()


def _block(title: str, content: str) -> str:
    bar = "═" * 60
    return f"\n{bar}\n[{title}]\n{bar}\n{content}\n"


def _build_base(overridable: str) -> str:
    base = _load("system_base.txt")
    return base.replace("{overridable_fields}", overridable)


def _format_analytics_block(
    flat: PriceDistribution,
    smile: PriceDistribution,
    histogram: MaturityHistogram | None = None,
) -> str:
    """Format terminal distribution statistics for injection into the LLM context."""

    def _stats(dist: PriceDistribution) -> str:
        return (
            f"  \u22123\u03c3: {dist.terminal_minus3s:.4f}  |  "
            f"\u22122\u03c3: {dist.terminal_minus2s:.4f}  |  "
            f"\u22121\u03c3: {dist.terminal_minus1s:.4f}\n"
            f"  Median: {dist.terminal_median:.4f}\n"
            f"  +1\u03c3: {dist.terminal_plus1s:.4f}  |  "
            f"+2\u03c3: {dist.terminal_plus2s:.4f}  |  "
            f"+3\u03c3: {dist.terminal_plus3s:.4f}"
        )

    result = (
        f"Spot: {flat.spot:.4f}  |  Horizon: {flat.horizon_days}d\n\n"
        f"Flat ATM vol ({flat.atm_vol * 100:.1f}%):\n{_stats(flat)}\n\n"
        f"Smile-adjusted:\n{_stats(smile)}"
    )

    if histogram:
        bin_lines = "\n".join(
            f"  {b.label}: flat {b.flat_pct:.1f}%  smile {b.smile_pct:.1f}%"
            for b in histogram.bins
            if b.flat_pct >= 0.5 or b.smile_pct >= 0.5  # skip negligible tails
        )
        result += f"\n\nMaturity probability by price bin:\n{bin_lines}"

    return result


# ---------------------------------------------------------------------------
# Step-specific system prompt builders
# ---------------------------------------------------------------------------

def build_intake_prompt(snapshot=None) -> str:
    """INTAKE step: extract the view. Injects live spots so LLM can convert absolute targets."""
    base = _build_base(overridable_fields_description())
    step = _load("view_extraction.txt")
    spot_block = ""
    if snapshot is not None:
        lines = [f"{pair}: spot {ccy.spot:.4f}" for pair, ccy in snapshot.currencies.items()]
        spot_block = _block("SPOT REFERENCE", "\n".join(lines))
    return base + "\n\n" + step + spot_block


def build_validation_prompt(
    view: TradeView,
    ccy: CurrencySnapshot,
    selector_result: StructureSelectionResult,
    flat_distribution: PriceDistribution | None = None,
    smile_distribution: PriceDistribution | None = None,
    maturity_histogram: MaturityHistogram | None = None,
    target_rr: float | None = None,
) -> str:
    """VALIDATION step: market context injected; LLM narrates it."""
    from knowledge_engine.conventions import resolve as resolve_conventions

    base = _build_base(overridable_fields_description())
    step = _load("view_validation.txt")

    conventions = resolve_conventions(view.pair)
    conv_text = format_for_context(conventions)

    atm_1m = ccy.get_atm_vol("1M")
    atm_3m = ccy.get_atm_vol("3M")
    fwd_3m = ccy.get_forward("3M")

    market_block = _block(
        "MARKET CONTEXT",
        f"Pair: {view.pair} | Spot: {ccy.spot:.4f}\n"
        f"ATM vol: 1M={atm_1m*100:.1f}%  3M={atm_3m*100:.1f}%\n"
        f"3M forward: {fwd_3m.outright:.4f}\n\n"
        + conv_text,
    )

    analytics_block = ""
    if flat_distribution and smile_distribution:
        analytics_block = _block(
            f"PRICE DISTRIBUTION AT MATURITY ({flat_distribution.horizon_days}d)",
            _format_analytics_block(flat_distribution, smile_distribution, maturity_histogram),
        )

    rr_block = ""
    if target_rr is not None:
        rr_block = _block("RISK/REWARD TARGET", f"PM targets risk 1 to make {target_rr:.1f} (i.e. {target_rr:.1f}× return on max loss).")

    return base + "\n\n" + step + market_block + analytics_block + rr_block


def build_structure_rec_prompt(
    view: TradeView,
    ccy: CurrencySnapshot,
    selector_result: StructureSelectionResult,
    target_rr: float | None = None,
) -> str:
    """STRUCTURE_REC step: shortlist + scenario matrix."""
    base = _build_base(overridable_fields_description())
    step = _load("structure_rec.txt")

    atm_1m = ccy.get_atm_vol("1M")
    atm_3m = ccy.get_atm_vol("3M")

    # Skew at the trade horizon (3M if available, else 1M)
    skew_tenor = "3M" if atm_3m else "1M"
    v25dc = ccy.get_vol(skew_tenor, "25DC")
    v25dp = ccy.get_vol(skew_tenor, "25DP")
    atm_h  = ccy.get_atm_vol(skew_tenor)
    if v25dc and v25dp and atm_h:
        rr = v25dc - v25dp
        fly = 0.5 * (v25dc + v25dp) - atm_h
        skew_sign = "topside" if rr > 0 else "downside"
        # For risk reversals: which leg is bought/sold depends on view direction
        # 25DC = call on base ccy (ccy1); 25DP = put on base ccy (ccy1)
        if view.direction == "base_higher":
            buy_leg  = f"25DC ({v25dc*100:.1f}%)"
            sell_leg = f"25DP ({v25dp*100:.1f}%)"
            rr_note  = f"Buying the {'expensive' if rr>0 else 'cheap'} leg (25DC), selling the {'cheap' if rr>0 else 'expensive'} leg (25DP)"
        else:
            buy_leg  = f"25DP ({v25dp*100:.1f}%)"
            sell_leg = f"25DC ({v25dc*100:.1f}%)"
            rr_note  = f"Buying the {'expensive' if rr<0 else 'cheap'} leg (25DP), selling the {'cheap' if rr<0 else 'expensive'} leg (25DC)"
        skew_block = (
            f"25d RR ({skew_tenor}): {rr*100:+.2f}% ({skew_sign} skew — 25DC={v25dc*100:.1f}% / ATM={atm_h*100:.1f}% / 25DP={v25dp*100:.1f}%)\n"
            f"25d Fly ({skew_tenor}): {fly*100:+.2f}%\n"
            f"Risk reversal for this view: buy {buy_leg}, sell {sell_leg}\n"
            f"  → {rr_note}"
        )
    else:
        skew_block = "Skew data unavailable"

    market_block = _block(
        "MARKET CONTEXT",
        f"Pair: {view.pair} | Spot: {ccy.spot:.4f} | Direction: {view.direction} | put_call: {view.direction == 'base_higher' and 'Call' or 'Put'}\n"
        f"ATM vol: 1M={atm_1m*100:.1f}%  3M={atm_3m*100:.1f}%\n"
        + skew_block,
    )

    shortlist_lines = [f"Rules fired: {', '.join(selector_result.rules_fired)}\n"]
    for item in selector_result.shortlist:
        exotic_tag = " [EXOTIC — comparison only]" if item.is_exotic else ""
        shortlist_lines.append(
            f"{item.rank}. {item.display_name}{exotic_tag}\n"
            f"   Optimised for: {item.optimised_for}\n"
            f"   Rationale: {textwrap.fill(item.rationale, 70, subsequent_indent='   ')}"
        )
        if item.caution:
            shortlist_lines.append(
                f"   ⚠ Caution: {textwrap.fill(item.caution, 67, subsequent_indent='     ')}"
            )
        if item.sizing_modifier:
            shortlist_lines.append(f"   Sizing: {item.sizing_modifier}")
        shortlist_lines.append("")

    shortlist_block = _block("STRUCTURE SHORTLIST", "\n".join(shortlist_lines))

    # Scenario matrix for top non-exotic structure
    scenario_block = ""
    top_non_exotic = next((i for i in selector_result.shortlist if not i.is_exotic), None)
    if top_non_exotic and top_non_exotic.structure_id not in ("spot", "forward"):
        try:
            rate_ctx = build_rate_context(
                ccy,
                tenor_to_years("3M"),
                DEFAULT_SETTLEMENT_RATES[view.pair],
            )
            atm_vol = ccy.get_atm_vol("3M")
            strike = rate_ctx.forward

            if top_non_exotic.structure_id in ("vanilla_call", "call_spread", "spot", "forward"):
                def pricer(spot, T_rem, sigma):
                    return call_value(spot, strike, T_rem, sigma, rate_ctx.r_d, rate_ctx.r_f)
            else:
                def pricer(spot, T_rem, sigma):
                    return put_value(spot, strike, T_rem, sigma, rate_ctx.r_d, rate_ctx.r_f)

            initial_value = pricer(rate_ctx.spot, rate_ctx.T, atm_vol)
            cfg = ScenarioConfig(
                spot_range_pct=8, spot_steps=5,
                vol_range_pct=25, vol_steps=3,
                time_horizons_years=[1 / 12, 2 / 12, 3 / 12],
                tenor_labels=["1M", "2M", "3M"],
            )
            matrix = build_scenario_matrix(
                pricer, rate_ctx.spot, atm_vol, rate_ctx.T, initial_value, cfg
            )
            notional = view.notional_usd or 1_000_000
            table = format_scenario_table(
                matrix,
                notional_scale=notional,
                title=f"P&L ($) — long {top_non_exotic.display_name} | ${notional:,.0f} notional",
            )
            scenario_block = _block(
                f"SCENARIO MATRIX — {top_non_exotic.display_name}", table
            )
        except Exception:
            pass  # Scenario matrix is best-effort; don't break the flow

    rr_block = ""
    if target_rr is not None:
        rr_block = _block("RISK/REWARD TARGET", f"PM targets risk 1 to make {target_rr:.1f} (i.e. {target_rr:.1f}× return on max loss). Use this to assess which structures can realistically deliver this R:R at the stated target, given their premium cost and payoff profile.")

    return base + "\n\n" + step + market_block + shortlist_block + scenario_block + rr_block


def build_sizing_prompt(
    view: TradeView,
    ccy: CurrencySnapshot,
    selector_result: StructureSelectionResult,
    sizing: SizingOutput,
) -> str:
    """SIZING step: sizing output injected."""
    base = _build_base(overridable_fields_description())
    step = _load("sizing.txt")

    shortlist_summary = ", ".join(
        f"{i.rank}. {i.display_name}" for i in selector_result.shortlist if not i.is_exotic
    )
    context_block = _block(
        "TRADE CONTEXT",
        f"Pair: {view.pair} | Spot: {ccy.spot:.4f} | "
        f"Direction: {view.direction} | "
        f"Horizon: {view.horizon_days}d\n"
        f"Shortlist: {shortlist_summary}",
    )

    sizing_text = format_sizing_for_context(sizing)
    sizing_block = _block("SIZING OUTPUT", sizing_text)

    return base + "\n\n" + step + context_block + sizing_block


def build_entry_exit_prompt(
    view: TradeView,
    ccy: CurrencySnapshot,
    selector_result: StructureSelectionResult,
    sizing: SizingOutput,
) -> str:
    """ENTRY_EXIT step: all context, produce complete trade memo."""
    base = _build_base(overridable_fields_description())
    step = _load("entry_exit.txt")

    shortlist_lines = []
    for item in selector_result.shortlist:
        if not item.is_exotic:
            shortlist_lines.append(f"{item.rank}. {item.display_name} — {item.rationale}")

    context_block = _block(
        "TRADE CONTEXT",
        f"Pair: {view.pair} | Spot: {ccy.spot:.4f} | "
        f"Direction: {view.direction} | "
        f"Horizon: {view.horizon_days}d | "
        f"Catalyst: {view.catalyst or 'not stated'}\n"
        f"Conviction: direction={view.direction_conviction}, timing={view.timing_conviction}",
    )

    shortlist_block = _block(
        "STRUCTURE SHORTLIST (non-exotic)",
        "\n".join(shortlist_lines),
    )

    sizing_block = _block("SIZING OUTPUT", format_sizing_for_context(sizing))

    return (
        base + "\n\n" + step + context_block + shortlist_block + sizing_block
    )


def build_critique_prompt(
    view: TradeView,
    ccy: CurrencySnapshot,
    selector_result: StructureSelectionResult,
    sizing: SizingOutput,
    critique: CritiqueOutput,
) -> str:
    """CRITIQUE step: full critique output injected."""
    base = _build_base(overridable_fields_description())
    step = _load("critique.txt")

    context_block = _block(
        "TRADE CONTEXT",
        f"Pair: {view.pair} | Spot: {ccy.spot:.4f} | "
        f"PM structure: {critique.pm_structure} | "
        f"Direction: {view.direction} | Horizon: {view.horizon_days}d",
    )

    critique_lines = [
        f"Verdict: {critique.verdict.upper().replace('_', ' ')}",
        f"Recommended alternative: {critique.recommended_alternative or 'none — current structure is appropriate'}",
        "",
        f"EV comparison: {critique.ev_comparison_note}",
        f"Scenario weakness: {critique.scenario_weakness}",
        f"Execution: {critique.execution_notes}",
        f"Gamma profile: {critique.gamma_notes}",
        f"Hedge effectiveness: {critique.hedge_effectiveness}",
        "",
        "Dimension scores:",
    ]
    for dim, score in critique.dimension_scores.items():
        critique_lines.append(f"  {dim}: {score}")

    critique_block = _block("CRITIQUE OUTPUT", "\n".join(critique_lines))
    sizing_block = _block("SIZING OUTPUT (applied to PM structure)", format_sizing_for_context(sizing))

    return base + "\n\n" + step + context_block + critique_block + sizing_block


def build_followup_prompt(
    view: TradeView,
    ccy: CurrencySnapshot,
    selector_result: StructureSelectionResult,
    sizing: SizingOutput,
    critique: CritiqueOutput | None = None,
) -> str:
    """DONE step: full context for follow-up Q&A."""
    base = _build_base(overridable_fields_description())
    followup_instruction = (
        "\nSTEP: FOLLOW-UP\n"
        "The trade recommendation has been presented. Answer any follow-up questions "
        "from the PM using the pre-computed data in the blocks below. "
        "Do not re-narrate everything — answer the specific question concisely.\n"
    )

    shortlist_lines = [f"{i.rank}. {i.display_name}" for i in selector_result.shortlist]
    context_block = _block(
        "FULL TRADE CONTEXT",
        f"Pair: {view.pair} | Spot: {ccy.spot:.4f} | "
        f"Direction: {view.direction} | Horizon: {view.horizon_days}d | "
        f"Mode: {view.mode}\n"
        f"Structures: {', '.join(shortlist_lines)}",
    )

    sizing_block = _block("SIZING OUTPUT", format_sizing_for_context(sizing))

    critique_block = ""
    if critique:
        critique_block = _block(
            "CRITIQUE OUTPUT",
            f"Verdict: {critique.verdict} | PM structure: {critique.pm_structure}",
        )

    return base + followup_instruction + context_block + sizing_block + critique_block
