"""Seed an AgentSession from an already-computed trade (the Trade View screen).

Task-1 support: the Trade View page runs the deterministic engine for a view and
shows the recommendation. When the PM opens the in-context chat, we want the agent
to start *already knowing* that trade — identical to the Agent tab but pre-loaded,
with no extra API call.

We do that by injecting a synthetic ``run_standard_pack`` turn into the message
history: an assistant ``tool_use`` block followed by the matching ``tool_result``
carrying the rendered pack, then a canned assistant opening line. The model therefore
sees the pack exactly as if it had called the tool itself, so the first real PM turn
narrates over it (routing rule 4) instead of re-running the engine. We also pre-warm
the Tier-1 cache so any re-run of the same view reuses the pack (no recompute, no
divergence).

The synthetic blocks use the Anthropic message shape (dicts), which the only wired
adapter (AnthropicToolLLM) passes straight through. This helper is LLM-free and pure,
so it is unit-testable without an API key.
"""

from __future__ import annotations

from agentic.render import render_pack
from agentic.session import AgentSession
from agentic.standard_pack import StandardPack
from knowledge_engine.models import TradeView

# Fixed id linking the synthetic tool_use to its tool_result. Stable so _agent_tool_trace
# reconstructs the seeded call cleanly.
SEED_TOOL_ID = "seed_tradeview"

DEFAULT_OPENING = (
    "This is the trade from your Trade View screen. Ask me anything about it — the "
    "recommended structures, the sizing, the market read, or an alternative you're weighing."
)


def view_to_pack_args(view: TradeView) -> dict:
    """Reconstruct the ``run_standard_pack`` tool input for a view, so the seeded
    tool_use block mirrors a real call the model could have made."""
    args: dict = {
        "pair": view.pair,
        "horizon_days": view.horizon_days,
        "direction": view.direction,
        "mode": view.mode,
    }
    if view.magnitude_pct is not None:
        args["magnitude_pct"] = view.magnitude_pct
    return args


def seed_session_from_pack(
    session: AgentSession,
    view: TradeView,
    pack: StandardPack,
    opening_line: str = DEFAULT_OPENING,
) -> None:
    """Load ``view``/``pack`` into ``session`` and seed the message history with a
    synthetic run_standard_pack turn + a canned opening. Idempotent-safe to call once
    on a fresh session; the history ends on an assistant turn so the next real user
    message alternates correctly."""
    session.view = view
    session.pack = pack
    session.store(view, pack)

    session.messages.append(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Loading the trade from the Trade View screen."},
                {
                    "type": "tool_use",
                    "id": SEED_TOOL_ID,
                    "name": "run_standard_pack",
                    "input": view_to_pack_args(view),
                },
            ],
        }
    )
    session.messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": SEED_TOOL_ID,
                    "content": render_pack(pack, view),
                    "is_error": False,
                }
            ],
        }
    )
    session.messages.append({"role": "assistant", "content": opening_line})
