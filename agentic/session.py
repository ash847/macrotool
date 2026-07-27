"""Agent session state — the shallow conversation/context model.

Holds the current view, the current StandardPack, the Tier-1 cache, the
provider-neutral message history, and the structures priced this session. The
quant lives in Python (build_pack); this object only tracks *which* view is live
and caches packs so identical view inputs never recompute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic.price_structure import PricedStructure
from agentic.standard_pack import StandardPack
from data.schema import MarketSnapshot
from knowledge_engine.models import TradeView


@dataclass
class AgentSession:
    snapshot: MarketSnapshot
    cfg: Any                                    # ResolvedConfig
    structure_constraint: str = "No restriction"
    primary_objective: str = "Balanced"
    trade_management: str = "Standard hold"
    target_rr: float = 3.0          # R:R slider — drives the loss budget on the fly
    linear_notional: float = 100.0  # master sizing capital W; the UI passes the sidebar value
    # Active sizing regime the PM is operating under — the agent is locked to it.
    sizing_method: str = "fixed_loss"       # "fixed_loss" | "kelly"
    kelly_lambda: float = 0.5
    kelly_probs: tuple[float, ...] | None = None   # elicited edge distribution (Kelly only)
    kelly_bins: tuple[float, ...] | None = None

    view: TradeView | None = None
    pack: StandardPack | None = None

    messages: list[dict] = field(default_factory=list)
    priced: list[PricedStructure] = field(default_factory=list)
    _cache: dict[tuple, StandardPack] = field(default_factory=dict)

    def cache_key(self, view: TradeView) -> tuple:
        """Tier-1 cache key — identical view inputs reuse the pack, no recompute."""
        return (
            view.pair,
            view.direction,
            view.horizon_days,
            view.magnitude_pct,
            view.mode,
            self.structure_constraint,
            self.primary_objective,
            self.trade_management,
            self.target_rr,
            self.linear_notional,
            self.sizing_method,
            self.kelly_lambda,
            self.kelly_probs,
            self.kelly_bins,
        )

    def get_cached(self, view: TradeView) -> StandardPack | None:
        return self._cache.get(self.cache_key(view))

    def store(self, view: TradeView, pack: StandardPack) -> None:
        self._cache[self.cache_key(view)] = pack
