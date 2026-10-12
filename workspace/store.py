"""``ConversationStore`` seam — Supabase in production, in-memory for tests and as the
fallback when Supabase (or the tables) are unavailable.

**Privacy is enforced here, in one place:** every read takes ``user_email`` and
filters on it. Every table carries ``user_email`` (denormalised) so no read ever
relies on a join to scope ownership. The app writes with the service key, which
bypasses RLS, so this filter is the guard.
"""

from __future__ import annotations

from typing import Protocol

from workspace.models import Conversation, IdeaVersion, TradeIdea, Turn, utcnow

T_CONVERSATIONS = "conversations"
T_TURNS = "conversation_turns"
T_IDEAS = "trade_ideas"
T_VERSIONS = "idea_versions"
T_LINKS = "conversation_ideas"


class StoreError(RuntimeError):
    pass


class ConversationStore(Protocol):
    persistent: bool

    def save_conversation(self, conv: Conversation) -> None: ...
    def get_conversation(self, user_email: str, conversation_id: str) -> Conversation | None: ...
    def list_conversations(
        self, user_email: str, include_archived: bool = False, limit: int = 50
    ) -> list[Conversation]: ...
    def append_turn(self, turn: Turn) -> None: ...
    def list_turns(self, user_email: str, conversation_id: str) -> list[Turn]: ...
    def save_idea(self, idea: TradeIdea) -> None: ...
    def get_idea(self, user_email: str, idea_id: str) -> TradeIdea | None: ...
    def ideas_for_conversation(self, user_email: str, conversation_id: str) -> list[TradeIdea]: ...
    def link_idea(self, user_email: str, conversation_id: str, idea_id: str) -> None: ...
    def add_version(self, version: IdeaVersion) -> None: ...
    def latest_version(self, user_email: str, idea_id: str) -> IdeaVersion | None: ...


# ---------------------------------------------------------------------------
# In-memory
# ---------------------------------------------------------------------------

class InMemoryStore:
    """Dict-backed store. Lives for the browser session when used as the fallback."""

    persistent = False
    settings_persist = True

    def __init__(self) -> None:
        self._convs: dict[str, Conversation] = {}
        self._turns: list[Turn] = []
        self._ideas: dict[str, TradeIdea] = {}
        self._versions: list[IdeaVersion] = []
        self._links: dict[tuple[str, str], dict] = {}

    def save_conversation(self, conv: Conversation) -> None:
        self._convs[conv.id] = conv

    def get_conversation(self, user_email, conversation_id):
        c = self._convs.get(conversation_id)
        return c if c is not None and c.user_email == user_email else None

    def list_conversations(self, user_email, include_archived=False, limit=50):
        out = [c for c in self._convs.values()
               if c.user_email == user_email and (include_archived or not c.archived)]
        out.sort(key=lambda c: c.updated_at, reverse=True)
        return out[:limit]

    def _require_conversation(self, conversation_id: str) -> None:
        # Mirror the SQL foreign keys so tests catch write-ordering bugs.
        if conversation_id not in self._convs:
            raise StoreError(f"conversation {conversation_id} not saved yet")

    def append_turn(self, turn: Turn) -> None:
        self._require_conversation(turn.conversation_id)
        self._turns.append(turn)

    def list_turns(self, user_email, conversation_id):
        out = [t for t in self._turns
               if t.conversation_id == conversation_id and t.user_email == user_email]
        return sorted(out, key=lambda t: t.seq)

    def save_idea(self, idea: TradeIdea) -> None:
        self._ideas[idea.id] = idea

    def get_idea(self, user_email, idea_id):
        i = self._ideas.get(idea_id)
        return i if i is not None and i.user_email == user_email else None

    def ideas_for_conversation(self, user_email, conversation_id):
        links = [l for l in self._links.values()
                 if l["conversation_id"] == conversation_id and l["user_email"] == user_email]
        links.sort(key=lambda l: l["last_active_at"], reverse=True)
        return [self._ideas[l["idea_id"]] for l in links if l["idea_id"] in self._ideas]

    def link_idea(self, user_email, conversation_id, idea_id):
        self._require_conversation(conversation_id)
        if idea_id not in self._ideas:
            raise StoreError(f"idea {idea_id} not saved yet")
        self._links[(conversation_id, idea_id)] = {
            "conversation_id": conversation_id, "idea_id": idea_id,
            "user_email": user_email, "last_active_at": utcnow(),
        }

    def add_version(self, version: IdeaVersion) -> None:
        if version.idea_id not in self._ideas:
            raise StoreError(f"idea {version.idea_id} not saved yet")
        self._versions.append(version)

    def latest_version(self, user_email, idea_id):
        vs = [v for v in self._versions if v.idea_id == idea_id and v.user_email == user_email]
        return max(vs, key=lambda v: v.version_no) if vs else None


# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------

class SupabaseStore:
    """Supabase-backed store over the tables in ``db/workspace_schema.sql``.

    Every call is wrapped: a failure raises ``StoreError`` so the UI can degrade
    rather than crash the chat.
    """

    persistent = True

    def __init__(self, client) -> None:
        self._c = client

    def _t(self, name):
        return self._c.table(name)

    def check(self) -> None:
        """Cheap probe that all tables exist (raises StoreError otherwise)."""
        for t in (T_CONVERSATIONS, T_TURNS, T_IDEAS, T_VERSIONS, T_LINKS):
            try:
                self._t(t).select("*").limit(1).execute()
            except Exception as e:
                raise StoreError(f"table {t!r} unavailable: {e}") from e

    def _run(self, what, fn):
        try:
            return fn()
        except StoreError:
            raise
        except Exception as e:
            raise StoreError(f"{what}: {e}") from e

    # Columns added after the first schema release; a DB that hasn't run the ALTER yet
    # degrades to saving without them (chat settings then last the session only).
    _LATE_CONV_COLUMNS = ("settings", "distributions")
    settings_persist = True

    def save_conversation(self, conv):
        row = conv.to_row()
        try:
            self._t(T_CONVERSATIONS).upsert(row).execute()
            return
        except Exception:
            pass
        slim = {k: v for k, v in row.items() if k not in self._LATE_CONV_COLUMNS}
        self._run("save_conversation", lambda: self._t(T_CONVERSATIONS).upsert(slim).execute())
        self.settings_persist = False

    def get_conversation(self, user_email, conversation_id):
        res = self._run("get_conversation", lambda: self._t(T_CONVERSATIONS).select("*")
                        .eq("id", conversation_id).eq("user_email", user_email)
                        .limit(1).execute())
        return Conversation.from_row(res.data[0]) if res.data else None

    def list_conversations(self, user_email, include_archived=False, limit=50):
        def q():
            query = self._t(T_CONVERSATIONS).select("*").eq("user_email", user_email)
            if not include_archived:
                query = query.eq("archived", False)
            return query.order("updated_at", desc=True).limit(limit).execute()
        return [Conversation.from_row(r) for r in self._run("list_conversations", q).data]

    def append_turn(self, turn):
        self._run("append_turn", lambda: self._t(T_TURNS).insert(turn.to_row()).execute())

    def list_turns(self, user_email, conversation_id):
        res = self._run("list_turns", lambda: self._t(T_TURNS).select("*")
                        .eq("conversation_id", conversation_id).eq("user_email", user_email)
                        .order("seq").execute())
        return [Turn.from_row(r) for r in res.data]

    def save_idea(self, idea):
        self._run("save_idea", lambda: self._t(T_IDEAS).upsert(idea.to_row()).execute())

    def get_idea(self, user_email, idea_id):
        res = self._run("get_idea", lambda: self._t(T_IDEAS).select("*")
                        .eq("id", idea_id).eq("user_email", user_email).limit(1).execute())
        return TradeIdea.from_row(res.data[0]) if res.data else None

    def ideas_for_conversation(self, user_email, conversation_id):
        links = self._run("ideas_for_conversation", lambda: self._t(T_LINKS).select("*")
                          .eq("conversation_id", conversation_id)
                          .eq("user_email", user_email)
                          .order("last_active_at", desc=True).execute()).data
        ideas = []
        for l in links:
            idea = self.get_idea(user_email, l["idea_id"])
            if idea is not None:
                ideas.append(idea)
        return ideas

    def link_idea(self, user_email, conversation_id, idea_id):
        row = {"conversation_id": conversation_id, "idea_id": idea_id,
               "user_email": user_email, "last_active_at": utcnow()}
        self._run("link_idea", lambda: self._t(T_LINKS)
                  .upsert(row, on_conflict="conversation_id,idea_id").execute())

    def add_version(self, version):
        self._run("add_version", lambda: self._t(T_VERSIONS).insert(version.to_row()).execute())

    def latest_version(self, user_email, idea_id):
        res = self._run("latest_version", lambda: self._t(T_VERSIONS).select("*")
                        .eq("idea_id", idea_id).eq("user_email", user_email)
                        .order("version_no", desc=True).limit(1).execute())
        return IdeaVersion.from_row(res.data[0]) if res.data else None
