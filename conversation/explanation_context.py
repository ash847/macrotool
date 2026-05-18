"""
Renderer for comparator packs.

This module deliberately renders a compact, PM-safe explanation artifact. It is
not wired into prompts yet; tests and an admin preview can inspect this text
before any LLM sees it.
"""

from __future__ import annotations

import json
from pathlib import Path

from knowledge_engine.comparator import PairwiseComparison, RecommendationExplanationPack, Reason

_SCENARIO_DEFS_PATH = Path(__file__).parent.parent / "knowledge" / "defaults" / "scenario_definitions.json"


def _load_scenario_column_descriptions() -> dict[str, str]:
    try:
        data = json.loads(_SCENARIO_DEFS_PATH.read_text())
        return data.get("scenario_column_descriptions", {})
    except Exception:
        return {}


def _select_display_variants(ranked_variants: list) -> list:
    """Top 5 variants globally, plus best variant per structure type not already represented."""
    top5 = ranked_variants[:5]
    seen_structures = {rv.structure_id for rv in top5}
    extras = []
    seen_extras = set()
    for rv in ranked_variants[5:]:
        if rv.structure_id not in seen_structures and rv.structure_id not in seen_extras:
            extras.append(rv)
            seen_extras.add(rv.structure_id)
    return top5 + extras


def render_explanation_pack(
    pack: RecommendationExplanationPack,
    *,
    max_comparisons: int | None = None,
) -> str:
    lines: list[str] = render_explanation_pack_overview(pack).splitlines()
    variant_comparisons = pack.variant_comparisons
    if max_comparisons is not None:
        variant_comparisons = variant_comparisons[:max_comparisons]
    variant_text = render_variant_comparisons(variant_comparisons)
    if variant_text:
        lines.extend(["", variant_text])

    comparisons = list(pack.comparisons.values())
    if max_comparisons is not None:
        comparisons = comparisons[:max_comparisons]
    structure_text = render_structure_comparisons(comparisons)
    if structure_text:
        lines.extend(["", structure_text])

    if pack.unavailable_comparisons:
        unavailable = [
            f"- {item.challenger_display_name}: {item.plain}"
            for item in pack.unavailable_comparisons
        ]
        lines.extend(_section("Unavailable comparisons", unavailable))

    lines.extend([
        "Disclosure:",
        "- Explain qualitatively.",
        "- Do not reveal raw weights, thresholds, JSON scores, or scoring formulas.",
        "- Do not invent scores, strikes, barriers, premiums, or scenario P&Ls.",
    ])

    return "\n".join(lines).strip()


def _wing_risk_lines(structure_id: str, rv: "RankedVariant", is_call: bool) -> list[str]:
    strikes = rv.strikes
    barrier = rv.barrier

    if structure_id == "vanilla":
        return []

    if structure_id == "1x1_spread":
        if len(strikes) >= 2:
            return [f"- Short leg at {strikes[1]:.4f}: cap — profit stops here, no additional loss"]
        return []

    if structure_id in ("1x1.5_spread", "1x2_spread"):
        if len(strikes) >= 2:
            direction = "rises" if is_call else "falls"
            return [
                f"- Short leg at {strikes[1]:.4f}: tail — profit peaks at this strike, "
                f"then reverses as spot {direction} beyond it; P&L crosses zero and turns "
                f"increasingly negative on large moves"
            ]
        return []

    if structure_id == "seagull":
        lines = []
        if len(strikes) >= 2:
            lines.append(f"- Short in-direction leg at {strikes[1]:.4f}: cap — profit stops here, no additional loss")
        if len(strikes) >= 3:
            direction = "falls" if is_call else "rises"
            lines.append(f"- Short funding wing at {strikes[2]:.4f}: tail — P&L turns increasingly negative if spot {direction} through this level")
        return lines

    if structure_id in ("rko",):
        if barrier is not None:
            return [f"- Barrier at {barrier:.4f}: knockout — option is knocked out if spot trades through this level at any point; no loss beyond premium paid"]
        return []

    if structure_id in ("european_rko",):
        if barrier is not None:
            return [f"- Expiry knockout at {barrier:.4f}: knockout — option expires worthless if spot finishes beyond this level; no loss beyond premium paid"]
        return []

    if structure_id in ("european_digital", "european_digital_rko", "linear"):
        return []

    return []


def render_explanation_pack_overview(pack: RecommendationExplanationPack) -> str:
    lines: list[str] = [
        "RECOMMENDATION EXPLANATION PACK",
        f"Chosen: {pack.chosen_display_name}",
        f"Recommendation basis: {pack.recommendation_basis}",
        "",
    ]

    # Active scenario weighting context
    if pack.active_context_ids:
        ctx_lines = []
        for ctx_id, ctx_comment in zip(pack.active_context_ids, pack.active_context_comments):
            layer = "base" if ctx_id == pack.active_context_ids[0] else "overlay"
            ctx_lines.append(f"- [{layer}] {ctx_id.replace('_', ' ')}: {ctx_comment}")
        lines.extend(_section("Active scenario weighting", ctx_lines))

    if pack.ranked_variants:
        display_variants = _select_display_variants(pack.ranked_variants)
        ranking_lines = []
        for r in display_variants:
            marker = " [best of type]" if r.scenario_rank > 5 else ""
            ranking_lines.append(
                f"- {r.scenario_rank}. {_variant_name(r)} "
                f"(affinity rank {r.affinity_rank}, PM weighted P&L {_fmt_ccy(r.pm_score_ccy)})"
                f"{marker}"
            )
        lines.extend(_section("Variant ranking (top 5 + best per structure type)", ranking_lines))

    if pack.ranked_variants:
        chosen_rv = pack.ranked_variants[0]
        wing_lines = _wing_risk_lines(chosen_rv.structure_id, chosen_rv, pack.is_call)
        if wing_lines:
            lines.extend(_section("Wing risk (chosen structure)", wing_lines))

    if pack.summary_reasons:
        lines.extend(_section("Summary", _reason_lines(pack.summary_reasons)))

    if pack.construction_reasons:
        construction = [
            f"- {r.plain} ({r.strike_label}: {r.strike_value:.4f}, {r.relation} {r.reference})"
            for r in pack.construction_reasons
        ]
        lines.extend(_section("Construction", construction))

    if pack.risk_reasons:
        lines.extend(_section("Risks", _reason_lines(pack.risk_reasons)))

    # Scenario column definitions — helps LLM interpret the grid columns
    col_descs = _load_scenario_column_descriptions()
    if col_descs:
        col_lines = [f"- {col}: {desc}" for col, desc in col_descs.items()]
        lines.extend(_section("Scenario column definitions", col_lines))

    return "\n".join(lines).strip()


def render_variant_comparisons(comparisons) -> str:
    lines: list[str] = []
    for comparison in comparisons:
        lines.extend(_render_variant_comparison(comparison))
    return "\n".join(lines).strip()


def render_structure_comparisons(comparisons: list[PairwiseComparison]) -> str:
    lines: list[str] = []
    for comparison in comparisons:
        lines.extend(_render_comparison(comparison))
    return "\n".join(lines).strip()


def _render_comparison(comparison: PairwiseComparison) -> list[str]:
    lines = [
        f"Comparison: {comparison.chosen_id} vs {comparison.challenger_id}",
        f"Verdict: {comparison.verdict}; confidence: {comparison.confidence}",
        f"Headline: {comparison.headline}",
    ]
    if comparison.chosen_edges:
        lines.extend(_section("Chosen edges", _reason_lines(comparison.chosen_edges), leading_blank=False))
    if comparison.challenger_edges:
        lines.extend(_section("Challenger edges", _reason_lines(comparison.challenger_edges), leading_blank=False))
    if comparison.caveats:
        lines.extend(_section("Caveats", _reason_lines(comparison.caveats), leading_blank=False))
    if comparison.counterfactuals:
        lines.extend(_section("Counterfactuals", _reason_lines(comparison.counterfactuals), leading_blank=False))
    lines.append("")
    return lines


def _render_variant_comparison(comparison) -> list[str]:
    lines = [
        f"Variant comparison: {_variant_name(comparison.chosen)} vs {_variant_name(comparison.challenger)}",
        f"Verdict: {comparison.verdict}; confidence: {comparison.confidence}",
        f"Headline: {comparison.headline}",
        "",
    ]
    return lines


def _section(title: str, body: list[str], *, leading_blank: bool = True) -> list[str]:
    lines = [""] if leading_blank else []
    lines.append(f"{title}:")
    lines.extend(body)
    return lines


def _reason_lines(reasons: list[Reason]) -> list[str]:
    lines = []
    for reason in reasons:
        suffix = f" [{reason.materiality}]" if reason.materiality == "high" else ""
        line = f"- {reason.plain}{suffix}"
        if reason.detail:
            line += f" {reason.detail}"
        lines.append(line)
    return lines


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def _fmt_ccy(value: float | None) -> str:
    return "n/a" if value is None else f"${value:,.2f}"


def _variant_name(variant) -> str:
    return f"{variant.structure_display_name} - {variant.variant_label}"
