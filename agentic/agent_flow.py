"""The agent loop — Tier-1 / Tier-2 / narrate routing over a tool-calling LLM.

Provider-neutral: talks only to the ``ToolLLM`` seam and adapter-owned messages.
On each user turn it calls the model; if the model emits tool calls, it dispatches
them to the Python tools and loops; when the model returns text, that's the
narration. Bounded by ``max_rounds`` to prevent runaway.

The hard safety rules live in the system prompt: the LLM routes and narrates, but
every number it states must come from a tool result — it never computes one.
"""

from __future__ import annotations

from agentic.agent_llm import ToolLLM
from agentic.session import AgentSession
from agentic.tools import TOOL_SCHEMAS, dispatch

SYSTEM_PROMPT = """You are a structuring assistant for a macro-fund PM trading EM FX options.

You ORCHESTRATE and NARRATE. You never compute, interpolate, or invent any number.
Every number you state — a spot, vol, premium, strike, payoff, score, notional — MUST
come verbatim from a tool result already in this conversation. If you don't have a number
from a tool, call the tool; do not estimate.

Conventions:
- Direction is relative to the BASE currency (ccy1): 'base_higher' = base appreciates
  (USD up for USD* pairs; GBP up for GBPUSD; EUR up for EURPLN), 'base_lower' = depreciates.
- The European digital is a base-ccy cash-or-nothing trade: payoff at target is 100%.
- Supported pairs: USDBRL, USDTRY, EURPLN, GBPUSD.

Routing — decide what each PM turn needs:
1. The PM states or CHANGES the view (pair, tenor, direction, target/magnitude, mode):
   call run_standard_pack with those view inputs. This runs the full engine. You supply
   the view only; the engine computes everything. Always do this before pricing anything.
2. The PM asks about a SPECIFIC structure (e.g. "what about a 34 vs 25 1x1.5?",
   "price a 10% digital"): call price_structure with a short request string in the
   grammar (e.g. '34 vs 25 1x1.5', '25Δ vanilla', 'digital 10%', 'ATMF vs target 1x2').
   You name the structure; the engine supplies direction, weights, strikes, sizing.
3. The PM asks "why / what / explain" about numbers already shown: do NOT call a tool —
   narrate over the pack/structures already in context.

If a tool returns a clarifying question (ambiguous structure request), ask the PM that
question. If it says a structure can't be priced, relay the reason plainly.

Be concise and precise. Cite the computed numbers; explain the trade-off behind the
recommendation in a PM's language."""


class AgentFlow:
    def __init__(self, llm: ToolLLM, session: AgentSession, max_rounds: int = 6):
        self._llm = llm
        self.session = session
        self.max_rounds = max_rounds

    def advance(self, user_message: str) -> str:
        """Process one PM message; return the final narration text."""
        s = self.session
        s.messages.append(self._llm.format_user(user_message))

        turn = None
        for _ in range(self.max_rounds):
            turn = self._llm.create(s.messages, SYSTEM_PROMPT, TOOL_SCHEMAS)
            s.messages.append(self._llm.format_assistant(turn))

            if not turn.tool_calls:
                return turn.text  # end_turn: this is the narration

            results = []
            for call in turn.tool_calls:
                content, is_error = dispatch(s, call.name, call.args)
                results.append((call, content, is_error))
            s.messages.append(self._llm.format_tool_results(results))

        # Bound hit — return whatever text we have, gracefully.
        return (turn.text if turn else "") or (
            "I wasn't able to finish that in the available steps — could you narrow the request?"
        )
