"""Orchestration over the store: record exchanges, track the active idea, resume and
refresh. UI-free and LLM-free so it is fully testable with ``InMemoryStore``.

Active-idea rule (Phase 1): the active idea is whatever the session's live view is
after an exchange — i.e. the last ``run_standard_pack``. It labels the conversation.
Every idea touched is linked to the conversation (history only, for option C).

Resume: the stored provider messages are replayed verbatim, then the active idea's
pack is rebuilt on the **current** snapshot so Tier-2 pricing works immediately. If
the rebuilt pack differs from the last saved one, the conversation is stale: the UI
shows a banner and ``refresh`` injects a clearly-labelled REFRESHED pack turn before
the model answers anything else.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

from agentic.render import render_pack
from agentic.session import AgentSession
from agentic.tools import dispatch
from workspace.identity import classify_view_change, find_idea
from workspace.models import (
    Conversation,
    IdeaExpired,
    IdeaVersion,
    TradeIdea,
    Turn,
    ViewSpec,
)
from workspace.serialize import messages_from_json, messages_to_json

REFRESH_TOOL_ID_PREFIX = "refresh_"
_CACHED_SUFFIX = "\n\n(reused cached pack — view unchanged)"


def snapshot_fingerprint(snapshot) -> str:
    """Stable identity of the market data a pack was built on."""
    return hashlib.sha256(snapshot.model_dump_json().encode()).hexdigest()[:16]


def session_prefs(session: AgentSession) -> dict:
    """The settings a pack was built under — part of a version's identity. Records
    which stated distribution (if any) sized the active trade."""
    from analytics.sizing import curve_key
    curve = None
    if session.view is not None and session.stated_curve_for(session.view) is not None:
        curve = curve_key(session.view.pair, session.expiry_for(session.view))
    return {
        "structure_constraint": session.structure_constraint,
        "primary_objective": session.primary_objective,
        "trade_management": session.trade_management,
        "target_rr": session.target_rr,
        "linear_notional": session.linear_notional,
        "sizing_method": session.sizing_method,
        "kelly_lambda": session.kelly_lambda,
        "kelly_curve": curve,
    }


def _top_structure(pack) -> str | None:
    rec = pack.recommended[0] if pack and pack.recommended else None
    if rec is None:
        return None
    return f"{rec.display_name} — {rec.variant.variant_label}"


def display_turns(turns: list[Turn]) -> list[tuple[str, str]]:
    """The (role, text) transcript the chat UI renders."""
    out: list[tuple[str, str]] = []
    for t in turns:
        if t.kind == "exchange":
            out.append(("user", t.user_text))
        out.append(("assistant", t.reply_text))
    return out


@dataclass
class ResumeResult:
    conversation: Conversation
    turns: list[Turn]
    session: AgentSession
    active: IdeaVersion | None = None
    stale: bool = False               # rebuilt pack differs from the last saved one
    note: str | None = None           # something the PM should be told (expired, …)


@dataclass
class SettingsOutcome:
    conversation: Conversation
    message: str | None = None        # the labelled note shown in the chat (None if no trade)
    turn: Turn | None = None


@dataclass
class RefreshOutcome:
    conversation: Conversation
    changed: bool
    message: str
    turn: Turn | None = None


@dataclass
class ConversationService:
    store: object                     # ConversationStore
    user_email: str

    # -- conversations ------------------------------------------------------

    def new_conversation(self, surface: str = "agent_tab") -> Conversation:
        """A fresh, unsaved conversation. Persisted on its first exchange, so an
        opened-then-abandoned chat leaves no row."""
        return Conversation(user_email=self.user_email, surface=surface)

    def list_conversations(self, limit: int = 30) -> list[Conversation]:
        return self.store.list_conversations(self.user_email, limit=limit)

    def rename(self, conv: Conversation, title: str) -> Conversation:
        title = title.strip()
        conv = conv.touched(title=title or conv.title, title_custom=bool(title))
        self.store.save_conversation(conv)
        return conv

    def archive(self, conv: Conversation) -> Conversation:
        conv = conv.touched(archived=True)
        self.store.save_conversation(conv)
        return conv

    def active_version(self, conv: Conversation) -> IdeaVersion | None:
        if not conv.active_idea_id:
            return None
        return self.store.latest_version(self.user_email, conv.active_idea_id)

    # -- recording ----------------------------------------------------------

    def record_exchange(
        self,
        conv: Conversation,
        session: AgentSession,
        *,
        seq: int,
        prompt: str,
        reply: str,
        pre_len: int,
        failed: bool = False,
    ) -> Conversation:
        """Persist one PM exchange and update the active idea.

        A failed exchange keeps its display text but stores no provider messages, so
        a resumed history never contains a half-finished tool loop.
        """
        conv, version, link = self._track_active_idea(conv, session)
        turn = Turn(
            conversation_id=conv.id,
            user_email=self.user_email,
            seq=seq,
            user_text=prompt,
            reply_text=reply,
            llm_messages=[] if failed else messages_to_json(session.messages[pre_len:]),
            idea_version_id=version.id if version else None,
            snapshot_date=session.snapshot.snapshot_date,
        )
        self._persist(conv, link, turn)
        return conv

    def _persist(self, conv: Conversation, link_idea_id: str | None, turn: Turn) -> None:
        """Write in foreign-key order: (idea + version already saved) → conversation →
        conversation_ideas link → turn."""
        self.store.save_conversation(conv)
        if link_idea_id:
            self.store.link_idea(self.user_email, conv.id, link_idea_id)
        self.store.append_turn(turn)

    def _track_active_idea(
        self, conv: Conversation, session: AgentSession
    ) -> tuple[Conversation, IdeaVersion | None, str | None]:
        """Save the idea/version the session's live pack represents (if new) and return
        the updated conversation, the version, and the idea id to (re)link — the link
        itself is written by ``_persist`` once the conversation row exists."""
        if session.view is None or session.pack is None:
            return conv, None, None
        snap_date = session.snapshot.snapshot_date
        spec = ViewSpec.from_view(session.view, session.pack.target, snap_date)
        prefs = session_prefs(session)

        ideas = self.store.ideas_for_conversation(self.user_email, conv.id)
        idea = find_idea(ideas, spec)
        latest = self.store.latest_version(self.user_email, idea.id) if idea else None
        change = classify_view_change(latest, spec, prefs)
        fingerprint = snapshot_fingerprint(session.snapshot)
        rendered = render_pack(session.pack, session.view)
        if change == "same" and (
            latest.snapshot_fingerprint != fingerprint or latest.rendered_pack != rendered
        ):
            change = "revision"   # same view, new data or engine output → new evaluation

        if idea is None:
            idea = TradeIdea(user_email=self.user_email, pair=spec.pair,
                             direction=spec.direction)
        if change == "same":
            version = latest
        else:
            version = IdeaVersion(
                idea_id=idea.id,
                user_email=self.user_email,
                version_no=(latest.version_no + 1) if latest else 1,
                spec=spec,
                prefs=prefs,
                snapshot_date=snap_date,
                snapshot_fingerprint=fingerprint,
                rendered_pack=rendered,
                top_structure=_top_structure(session.pack),
            )
            self.store.save_idea(idea)
            self.store.add_version(version)
        link = idea.id if (change != "same" or conv.active_idea_id != idea.id) else None

        title = conv.title if conv.title_custom else spec.label()
        conv = conv.touched(active_idea_id=idea.id, title=title,
                            last_snapshot_date=snap_date)
        return conv, version, link

    # -- resume / refresh ---------------------------------------------------

    def resume(
        self, conversation_id: str, make_session: Callable[[Conversation], AgentSession]
    ) -> ResumeResult | None:
        """Rehydrate a saved conversation into a fresh session (no LLM call).
        ``make_session(conv)`` builds the session from the chat's own settings."""
        conv = self.store.get_conversation(self.user_email, conversation_id)
        if conv is None:
            return None
        turns = self.store.list_turns(self.user_email, conv.id)
        session = make_session(conv)
        session.messages = [m for t in turns for m in messages_from_json(t.llm_messages)]

        result = ResumeResult(conversation=conv, turns=turns, session=session)
        active = self.active_version(conv)
        result.active = active
        if active is None:
            return result
        try:
            content = self._rebuild(session, active.spec)
        except IdeaExpired as e:
            result.note = f"This trade has expired — {e}. Start a new view to continue."
            return result
        except Exception as e:   # engine failure must not block reading the chat
            result.note = f"Couldn't rebuild the trade on current data ({type(e).__name__})."
            return result
        result.stale = content != active.rendered_pack
        if session.view is not None and session.view.direction != active.spec.direction:
            result.note = (
                "The target is now on the other side of the forward, so the engine reads "
                "this as the opposite direction. Restate the view to continue."
            )
        return result

    def apply_settings(
        self,
        conv: Conversation,
        session: AgentSession,
        settings,
        distributions: dict,
        *,
        seq: int,
        persist: bool,
    ) -> SettingsOutcome:
        """Apply the chat's sizing settings / preferences / distributions.

        With an active trade, it is re-run under the new settings and a labelled
        SETTINGS UPDATED pack turn is injected (the model is told it supersedes the
        earlier figures); the note is persisted as a ``settings`` turn. Without one,
        the settings are just stored (``persist`` = the chat already has a row).
        """
        from workspace.settings import curves_from

        settings.apply_to(session)
        session.kelly_curves = curves_from(distributions)
        conv = conv.touched(settings=settings.to_dict(), distributions=dict(distributions))
        active = self.active_version(conv)
        if active is None:
            if persist:
                self.store.save_conversation(conv)
            return SettingsOutcome(conv)

        session._cache.clear()
        snap_date = session.snapshot.snapshot_date
        content = self._rebuild(session, active.spec)
        message = settings_note(session)
        pre_len = len(session.messages)
        inject_pack_turn(
            session, active.spec.pack_args(snap_date), content,
            header=f"SETTINGS UPDATED — {message} Every earlier figure in this "
                   "conversation was sized under the previous settings; quote current "
                   "figures only from this pack.",
            user_note="[The PM changed the chat's sizing settings / preferences — re-run "
                      "the active trade under them.]",
        )
        session.messages.append({"role": "assistant", "content": message})
        conv, version, link = self._track_active_idea(conv, session)
        turn = Turn(
            conversation_id=conv.id, user_email=self.user_email, seq=seq, kind="settings",
            user_text="", reply_text=message,
            llm_messages=messages_to_json(session.messages[pre_len:]),
            idea_version_id=version.id if version else None, snapshot_date=snap_date,
        )
        self._persist(conv, link, turn)
        return SettingsOutcome(conv, message, turn)

    def refresh(
        self, conv: Conversation, session: AgentSession, *, seq: int, force_turn: bool = False
    ) -> RefreshOutcome:
        """Re-run the active idea on the session's (current) snapshot.

        Unchanged output → nothing is added to the transcript. Changed output (or
        ``force_turn``) → a REFRESHED pack turn is injected into the model's history,
        a new idea version is saved, and a refresh turn is persisted.
        """
        active = self.active_version(conv)
        snap_date = session.snapshot.snapshot_date
        if active is None:
            return RefreshOutcome(conv, False, "Nothing to refresh yet — no trade in this chat.")
        session._cache.clear()   # weights/affinity may have changed since the cache filled
        content = self._rebuild(session, active.spec)
        changed = content != active.rendered_pack
        if not changed and not force_turn:
            same_snap = active.snapshot_fingerprint == snapshot_fingerprint(session.snapshot)
            why = ("No new market data since last evaluation"
                   if same_snap else "Market data updated but the trade is unchanged")
            return RefreshOutcome(
                conv, False,
                f"{why} (snapshot {snap_date.isoformat()}) — figures unchanged.",
            )

        label = active.spec.label()
        pre_len = len(session.messages)
        inject_pack_turn(
            session, active.spec.pack_args(snap_date), content,
            header=f"REFRESHED — active trade: {label}, market data as of "
                   f"{snap_date.isoformat()}. Every earlier pack in this conversation (any "
                   "pair) is historical; quote current figures only from this pack.",
            user_note="[Conversation resumed — refresh the active trade to the latest "
                      "market data.]",
        )
        message = (f"Refreshed **{label}** to market data as of {snap_date.isoformat()}. "
                   "Figures above this point are from the earlier evaluation.")
        session.messages.append({"role": "assistant", "content": message})

        conv, version, link = self._track_active_idea(conv, session)
        turn = Turn(
            conversation_id=conv.id, user_email=self.user_email, seq=seq, kind="refresh",
            user_text="", reply_text=message,
            llm_messages=messages_to_json(session.messages[pre_len:]),
            idea_version_id=version.id if version else None, snapshot_date=snap_date,
        )
        self._persist(conv, link, turn)
        return RefreshOutcome(conv, True, message, turn)

    @staticmethod
    def _rebuild(session: AgentSession, spec: ViewSpec) -> str:
        """Rebuild ``spec`` through the real tool path (same validation, caching and
        rendering as a model-issued call). Sets ``session.view``/``session.pack``."""
        args = spec.pack_args(session.snapshot.snapshot_date)
        content, is_error = dispatch(session, "run_standard_pack", args)
        if is_error:
            raise RuntimeError(content)
        return content.removesuffix(_CACHED_SUFFIX)


def settings_note(session: AgentSession) -> str:
    """Plain-English note of how the active trade is now sized (shown in the chat)."""
    view, pack = session.view, session.pack
    tag = ""
    if view is not None:
        exp = pack.expiry.strftime("%d-%b-%y") if pack is not None and pack.expiry else ""
        tag = f"{view.pair} {exp}".strip()
    if session.sizing_method == "kelly":
        if pack is not None and getattr(pack, "kelly_fallback", False):
            return (f"Sizing updated: **Kelly selected, but there is no distribution for "
                    f"{tag}**, so this trade is sized **fixed-loss** (R:R {session.target_rr:g}). "
                    f"Set up a distribution for {tag} to size under Kelly.")
        return (f"Sizing updated: **Kelly λ {session.kelly_lambda:g}** on your stated "
                f"distribution for {tag}. Figures below reflect this.")
    return (f"Sizing updated: **fixed-loss, R:R {session.target_rr:g}**. "
            "Figures below reflect this.")


def inject_pack_turn(session: AgentSession, args: dict, content: str, *, header: str,
                     user_note: str) -> None:
    """Append a synthetic, clearly-labelled run_standard_pack exchange (same message
    shape as ``agentic.seed``) so the model sees the re-run pack as a tool result whose
    ``header`` says it supersedes earlier figures. History stays user/assistant-
    alternating: user note → assistant tool_use → user tool_result (caller appends the
    closing assistant text)."""
    tool_id = f"{REFRESH_TOOL_ID_PREFIX}{len(session.messages)}"
    session.messages.append({"role": "user", "content": user_note})
    session.messages.append({
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Re-running the active trade."},
            {"type": "tool_use", "id": tool_id, "name": "run_standard_pack", "input": args},
        ],
    })
    session.messages.append({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": header + "\n\n" + content,
            "is_error": False,
        }],
    })
