"""
Renderer for comparator packs.

This module deliberately renders a compact, PM-safe explanation artifact. It is
not wired into prompts yet; tests and an admin preview can inspect this text
before any LLM sees it.
"""

from __future__ import annotations

from knowledge_engine.comparator import PairwiseComparison, RecommendationExplanationPack, Reason


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


def render_explanation_pack_overview(pack: RecommendationExplanationPack) -> str:
    lines: list[str] = [
        "RECOMMENDATION EXPLANATION PACK",
        f"Chosen: {pack.chosen_display_name}",
        f"Recommendation basis: {pack.recommendation_basis}",
        "",
    ]

    if pack.ranked_variants:
        ranking_lines = [
            (
                f"- {r.scenario_rank}. {_variant_name(r)} "
                f"(affinity rank {r.affinity_rank}, PM weighted P&L {_fmt_ccy(r.pm_score_ccy)})"
            )
            for r in pack.ranked_variants
        ]
        lines.extend(_section("Variant ranking", ranking_lines))

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
