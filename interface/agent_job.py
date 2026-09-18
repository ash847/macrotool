"""Run one Agent turn in a background thread, so it completes (and is saved) even if
the PM navigates away mid-turn.

Why: Streamlit stops a script run at its next ``st.*`` call when the user clicks
elsewhere. With the LLM call inline, the reply arrived, the next ``st.markdown``
raised, and the reply was never shown or saved — while the model's hidden history
already contained it. Here the thread owns the turn (``flow.advance`` + saving the
exchange); the page merely waits on it, interruptibly, and picks the result up on
whichever run next renders the Agent page.

No Streamlit calls in this module — the thread has no script context. Anything that
needs ``st`` (error logging, session state) is done by the caller on finalize.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class AgentJob:
    conversation_id: str
    flow: Any                       # AgentFlow — identity is checked on finalize
    prompt: str
    seq: int
    pre_len: int
    started_at: float = field(default_factory=time.monotonic)
    reply: str | None = None
    failed: bool = False
    error: BaseException | None = None
    conversation: Any = None        # Conversation after record_exchange
    store_error: BaseException | None = None
    done: threading.Event = field(default_factory=threading.Event)

    def elapsed(self) -> int:
        return int(time.monotonic() - self.started_at)


def start_agent_job(
    flow,
    prompt: str,
    *,
    conversation,
    seq: int,
    service,
    after: Callable[[AgentJob], None] | None = None,
) -> AgentJob:
    """Start the turn and return immediately. ``after`` runs in the thread once the
    turn is saved (e.g. telemetry); it must not touch Streamlit."""
    job = AgentJob(
        conversation_id=conversation.id, flow=flow, prompt=prompt, seq=seq,
        pre_len=len(flow.session.messages),
    )

    def run() -> None:
        try:
            job.reply = flow.advance(prompt)
        except Exception as e:
            job.error = e
            job.failed = True
            job.reply = f"⚠️ {type(e).__name__}: {e}"
        try:
            job.conversation = service.record_exchange(
                conversation, flow.session, seq=seq, prompt=prompt,
                reply=job.reply, pre_len=job.pre_len, failed=job.failed,
            )
        except Exception as e:
            job.store_error = e
        if after is not None:
            try:
                after(job)
            except Exception:
                pass
        job.done.set()

    threading.Thread(target=run, name=f"agent-turn-{conversation.id[:8]}", daemon=True).start()
    return job
