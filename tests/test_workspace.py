"""Saved conversations + active idea (workspace/).

No API: the agent loop runs on FakeToolLLM, persistence on InMemoryStore.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from agentic.agent_flow import AgentFlow, build_system_prompt
from agentic.agent_llm import FakeToolLLM, LLMTurn, ToolCall
from agentic.session import AgentSession
from config.loader import load_config
from data.snapshot_loader import load_snapshot
from knowledge_engine.models import TradeView
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
from workspace.service import ConversationService, display_turns, snapshot_fingerprint
from workspace.store import InMemoryStore

SNAP = load_snapshot()
CFG = load_config()
D0 = SNAP.snapshot_date
ME = "pm@fund.com"


def _spec(**kw) -> ViewSpec:
    base = dict(pair="USDBRL", direction="base_lower", expiry_date=D0 + timedelta(days=90),
                target_level=5.0)
    base.update(kw)
    return ViewSpec(**base)


def _version(spec: ViewSpec, prefs=None, n=1) -> IdeaVersion:
    return IdeaVersion(idea_id="i1", user_email=ME, version_no=n, spec=spec,
                       prefs=prefs or {}, snapshot_date=D0, snapshot_fingerprint="x")


# ---------------------------------------------------------------------------
# ViewSpec
# ---------------------------------------------------------------------------

class TestViewSpec:
    def test_from_view_freezes_absolute_expiry_and_target(self):
        view = TradeView(pair="USDBRL", direction="base_lower", direction_conviction="medium",
                         horizon_days=90, magnitude_pct=3.0)
        spec = ViewSpec.from_view(view, 4.9876543219, D0)
        assert spec.expiry_date == D0 + timedelta(days=90)
        assert spec.target_level == pytest.approx(4.987654, abs=1e-9)

    def test_pack_args_horizon_counts_down_from_snapshot_date(self):
        spec = _spec()
        assert spec.pack_args(D0)["horizon_days"] == 90
        assert spec.pack_args(D0 + timedelta(days=30))["horizon_days"] == 60

    def test_pack_args_uses_target_level_else_direction(self):
        assert "target_level" in _spec().pack_args(D0)
        assert "direction" not in _spec().pack_args(D0)
        directional = _spec(target_level=None).pack_args(D0)
        assert directional["direction"] == "base_lower"
        assert "target_level" not in directional

    def test_expired(self):
        with pytest.raises(IdeaExpired):
            _spec().pack_args(D0 + timedelta(days=90))

    def test_label(self):
        assert _spec().label() == f"USDBRL ↓ 5 · {(D0 + timedelta(days=90)).strftime('%d-%b-%y')}"
        assert _spec(direction="base_higher", target_level=162.35).label().startswith(
            "USDBRL ↑ 162.35")
        assert _spec(target_level=168.0342).label().startswith("USDBRL ↓ 168.03 ")
        assert _spec(target_level=4.35125).label().startswith("USDBRL ↓ 4.3513 ")

    def test_json_round_trip(self):
        spec = _spec()
        assert ViewSpec.from_json(spec.to_json()) == spec


# ---------------------------------------------------------------------------
# Identity rule
# ---------------------------------------------------------------------------

class TestIdentity:
    def test_no_prior_is_new_idea(self):
        assert classify_view_change(None, _spec(), {}) == "new_idea"

    def test_pair_change_is_new_idea(self):
        assert classify_view_change(_version(_spec()), _spec(pair="USDJPY"), {}) == "new_idea"

    def test_direction_flip_is_new_idea(self):
        assert classify_view_change(
            _version(_spec()), _spec(direction="base_higher"), {}) == "new_idea"

    def test_target_or_expiry_or_prefs_change_is_revision(self):
        v = _version(_spec(), {"target_rr": 3.0})
        assert classify_view_change(v, _spec(target_level=4.9), {"target_rr": 3.0}) == "revision"
        assert classify_view_change(
            v, _spec(expiry_date=D0 + timedelta(days=60)), {"target_rr": 3.0}) == "revision"
        assert classify_view_change(v, _spec(), {"target_rr": 2.0}) == "revision"

    def test_identical_is_same(self):
        assert classify_view_change(_version(_spec(), {"a": 1}), _spec(), {"a": 1}) == "same"

    def test_find_idea_by_pair_and_direction(self):
        a = TradeIdea(user_email=ME, pair="USDBRL", direction="base_lower")
        b = TradeIdea(user_email=ME, pair="USDJPY", direction="base_higher")
        assert find_idea([a, b], _spec(pair="USDJPY", direction="base_higher")) is b
        assert find_idea([a, b], _spec(direction="base_higher")) is None


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialize:
    def test_fake_adapter_toolcalls_round_trip(self):
        msgs = [{"role": "assistant", "content": "",
                 "tool_calls": [ToolCall(id="t1", name="run_standard_pack", args={"pair": "X"})]}]
        back = messages_from_json(messages_to_json(msgs))
        assert back == msgs
        assert isinstance(back[0]["tool_calls"][0], ToolCall)

    def test_anthropic_blocks_become_plain_dicts(self):
        from anthropic.types import TextBlock, ToolUseBlock
        msgs = [{"role": "assistant", "content": [
            TextBlock(type="text", text="hi"),
            ToolUseBlock(type="tool_use", id="tu1", name="run_standard_pack", input={"a": 1}),
        ]}]
        data = messages_to_json(msgs)
        assert data[0]["content"][0] == {"type": "text", "text": "hi"}
        assert data[0]["content"][1]["input"] == {"a": 1}
        assert "citations" not in data[0]["content"][0]   # None fields dropped
        import json
        json.dumps(data)   # JSON-safe


# ---------------------------------------------------------------------------
# Store privacy
# ---------------------------------------------------------------------------

class TestStorePrivacy:
    def test_other_users_cannot_read(self):
        store = InMemoryStore()
        conv = Conversation(user_email=ME)
        store.save_conversation(conv)
        store.append_turn(Turn(conversation_id=conv.id, user_email=ME, seq=0,
                               user_text="q", reply_text="a"))
        idea = TradeIdea(user_email=ME, pair="USDBRL", direction="base_lower")
        store.save_idea(idea)
        store.link_idea(ME, conv.id, idea.id)
        store.add_version(IdeaVersion(idea_id=idea.id, user_email=ME, version_no=1,
                                      spec=_spec(), prefs={}, snapshot_date=D0,
                                      snapshot_fingerprint="x"))
        other = "someone@else.com"
        assert store.get_conversation(other, conv.id) is None
        assert store.list_conversations(other) == []
        assert store.list_turns(other, conv.id) == []
        assert store.get_idea(other, idea.id) is None
        assert store.ideas_for_conversation(other, conv.id) == []
        assert store.latest_version(other, idea.id) is None
        assert store.get_conversation(ME, conv.id) is conv

    def test_archived_hidden_from_list(self):
        store = InMemoryStore()
        svc = ConversationService(store, ME)
        conv = svc.new_conversation()
        store.save_conversation(conv)
        svc.archive(conv)
        assert svc.list_conversations() == []


# ---------------------------------------------------------------------------
# Service: active idea + resume + refresh (engine runs for real, LLM is fake)
# ---------------------------------------------------------------------------

def _session(snapshot=SNAP, conv=None) -> AgentSession:
    """Session factory as the UI builds it: from the chat's own settings + curves."""
    from workspace.settings import ChatSettings, curves_from
    s = AgentSession(snapshot=snapshot, cfg=CFG,
                     kelly_curves=curves_from(conv.distributions if conv else None))
    ChatSettings.from_dict(conv.settings if conv else None).apply_to(s)
    return s


def _pack_turn(tid: str, **args) -> list[LLMTurn]:
    return [
        LLMTurn(text="", tool_calls=[ToolCall(id=tid, name="run_standard_pack", args=args)],
                stop_reason="tool_use"),
        LLMTurn(text=f"narration {tid}", tool_calls=[], stop_reason="end_turn"),
    ]


BRL = dict(pair="USDBRL", horizon_days=90, target_level=4.9)
JPY = dict(pair="USDJPY", horizon_days=60, direction="base_higher", magnitude_pct=3.0)


def _chat(svc: ConversationService, turns: list[tuple[str, dict | None]]):
    """Drive a conversation: each (prompt, pack-args-or-None) is one PM exchange."""
    script: list[LLMTurn] = []
    for i, (_, args) in enumerate(turns):
        if args is None:
            script.append(LLMTurn(text=f"answer {i}", tool_calls=[], stop_reason="end_turn"))
        else:
            script += _pack_turn(f"t{i}", **args)
    session = _session()
    flow = AgentFlow(FakeToolLLM(script=script), session)
    conv = svc.new_conversation()
    for i, (prompt, _) in enumerate(turns):
        pre = len(session.messages)
        reply = flow.advance(prompt)
        conv = svc.record_exchange(conv, session, seq=i, prompt=prompt, reply=reply,
                                   pre_len=pre)
    return conv, session


class TestActiveIdea:
    def test_first_pack_creates_idea_and_labels_chat(self):
        svc = ConversationService(InMemoryStore(), ME)
        conv, session = _chat(svc, [("BRL lower to 4.90 3m", BRL)])
        active = svc.active_version(conv)
        assert active.spec.pair == "USDBRL" and active.spec.direction == "base_lower"
        assert active.spec.target_level == pytest.approx(4.9)
        assert active.spec.expiry_date == D0 + timedelta(days=90)
        assert conv.title == active.spec.label()
        assert active.rendered_pack and active.top_structure

    def test_chat_before_any_view_has_no_idea(self):
        svc = ConversationService(InMemoryStore(), ME)
        conv, _ = _chat(svc, [("what can you do?", None)])
        assert conv.active_idea_id is None
        assert conv.title == "New conversation"
        assert svc.store.get_conversation(ME, conv.id) is not None

    def test_active_idea_follows_last_pack_and_history_is_kept(self):
        svc = ConversationService(InMemoryStore(), ME)
        conv, _ = _chat(svc, [("BRL", BRL), ("now JPY", JPY), ("why?", None)])
        assert svc.active_version(conv).spec.pair == "USDJPY"
        assert conv.title.startswith("USDJPY ↑")
        pairs = {i.pair for i in svc.store.ideas_for_conversation(ME, conv.id)}
        assert pairs == {"USDBRL", "USDJPY"}

    def test_returning_to_a_pair_reuses_its_idea(self):
        svc = ConversationService(InMemoryStore(), ME)
        conv, _ = _chat(svc, [("BRL", BRL), ("JPY", JPY), ("back to BRL, 4.80",
                                                          {**BRL, "target_level": 4.8})])
        ideas = svc.store.ideas_for_conversation(ME, conv.id)
        assert len(ideas) == 2
        active = svc.active_version(conv)
        assert active.spec.pair == "USDBRL" and active.version_no == 2

    def test_same_view_again_adds_no_version(self):
        svc = ConversationService(InMemoryStore(), ME)
        conv, _ = _chat(svc, [("BRL", BRL), ("again", BRL)])
        assert svc.active_version(conv).version_no == 1

    def test_custom_title_survives_idea_change(self):
        svc = ConversationService(InMemoryStore(), ME)
        conv, session = _chat(svc, [("BRL", BRL)])
        conv = svc.rename(conv, "My BRL thoughts")
        session.view = None   # force a re-track with a fresh view below
        flow = AgentFlow(FakeToolLLM(script=_pack_turn("x", **JPY)), session)
        pre = len(session.messages)
        reply = flow.advance("jpy")
        conv = svc.record_exchange(conv, session, seq=1, prompt="jpy", reply=reply, pre_len=pre)
        assert conv.title == "My BRL thoughts"

    def test_failed_exchange_stores_no_provider_messages(self):
        svc = ConversationService(InMemoryStore(), ME)
        session = _session()
        conv = svc.new_conversation()
        session.messages.append({"role": "user", "content": "boom"})
        conv = svc.record_exchange(conv, session, seq=0, prompt="boom", reply="⚠️ error",
                                   pre_len=0, failed=True)
        (turn,) = svc.store.list_turns(ME, conv.id)
        assert turn.llm_messages == [] and turn.reply_text == "⚠️ error"


class TestResumeRefresh:
    def _saved(self):
        svc = ConversationService(InMemoryStore(), ME)
        conv, live = _chat(svc, [("BRL", BRL), ("why?", None)])
        return svc, conv, live

    def test_resume_replays_history_and_rebuilds_pack(self):
        svc, conv, live = self._saved()
        res = svc.resume(conv.id, lambda c: _session(SNAP, c))
        assert res.session.messages == messages_from_json(messages_to_json(live.messages))
        assert res.session.pack is not None and res.session.view.pair == "USDBRL"
        assert res.session.view.horizon_days == 90
        assert res.stale is False and res.note is None
        assert display_turns(res.turns) == [
            ("user", "BRL"), ("assistant", "narration t0"),
            ("user", "why?"), ("assistant", "answer 1"),
        ]

    def test_resume_is_private(self):
        svc, conv, _ = self._saved()
        assert ConversationService(svc.store, "other@x.com").resume(conv.id, lambda c: _session(SNAP, c)) is None

    def test_refresh_on_unchanged_data_adds_nothing(self):
        svc, conv, _ = self._saved()
        res = svc.resume(conv.id, lambda c: _session(SNAP, c))
        n = len(res.session.messages)
        out = svc.refresh(res.conversation, res.session, seq=4)
        assert out.changed is False
        assert "No new market data" in out.message and D0.isoformat() in out.message
        assert len(res.session.messages) == n
        assert len(svc.store.list_turns(ME, conv.id)) == 2

    def test_changed_market_data_is_stale_and_refresh_injects_turn(self):
        svc, conv, _ = self._saved()
        moved = SNAP.model_copy(deep=True)
        moved.currencies["USDBRL"].spot *= 1.01
        res = svc.resume(conv.id, lambda c: _session(moved, c))
        assert res.stale is True
        out = svc.refresh(res.conversation, res.session, seq=4)
        assert out.changed is True
        msgs = res.session.messages
        roles = [m["role"] for m in msgs[-4:]]
        assert roles == ["user", "assistant", "user", "assistant"]
        tool_result = msgs[-2]["content"][0]["content"]
        assert tool_result.startswith("REFRESHED — active trade: USDBRL ↓")
        assert svc.active_version(out.conversation).version_no == 2
        turns = svc.store.list_turns(ME, conv.id)
        assert turns[-1].kind == "refresh" and len(turns[-1].llm_messages) == 4
        assert display_turns(turns)[-1][0] == "assistant"

    def test_resumed_conversation_continues_with_refreshed_pack_in_context(self):
        svc, conv, _ = self._saved()
        res = svc.resume(conv.id, lambda c: _session(SNAP, c))
        svc.refresh(res.conversation, res.session, seq=4, force_turn=True)
        llm = FakeToolLLM(script=[LLMTurn(text="ok", tool_calls=[], stop_reason="end_turn")])
        AgentFlow(llm, res.session).advance("and now?")
        seen = llm.seen[0]["messages"]
        assert any("REFRESHED" in str(m.get("content")) for m in seen)
        assert seen[-1] == {"role": "user", "content": "and now?"}

    def test_expired_idea_reports_note(self):
        svc, conv, _ = self._saved()
        later = SNAP.model_copy(update={"snapshot_date": D0 + timedelta(days=120)})
        res = svc.resume(conv.id, lambda c: _session(later, c))
        assert res.note and "expired" in res.note
        assert res.session.pack is None

    def test_fingerprint_changes_with_data(self):
        moved = SNAP.model_copy(deep=True)
        moved.currencies["USDBRL"].spot *= 1.01
        assert snapshot_fingerprint(SNAP) == snapshot_fingerprint(load_snapshot())
        assert snapshot_fingerprint(SNAP) != snapshot_fingerprint(moved)


# ---------------------------------------------------------------------------
# SupabaseStore row mapping + scoping, against a fake PostgREST client
# ---------------------------------------------------------------------------

_PKS = {"conversation_ideas": ("conversation_id", "idea_id")}


class _Res:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, db, table):
        self.db, self.table, self.filters, self.op = db, table, [], "select"
        self._order, self._limit, self.row = None, None, None

    def select(self, *_):
        return self

    def eq(self, col, val):
        self.filters.append((col, val))
        return self

    def order(self, col, desc=False):
        self._order = (col, desc)
        return self

    def limit(self, n):
        self._limit = n
        return self

    def insert(self, row):
        self.op, self.row = "insert", row
        return self

    def upsert(self, row, on_conflict=None):
        self.op, self.row = "upsert", row
        return self

    def execute(self):
        import json
        rows = self.db.setdefault(self.table, [])
        if self.op in ("insert", "upsert"):
            row = json.loads(json.dumps(self.row))   # must be JSON-serialisable
            pk = _PKS.get(self.table, ("id",))
            rows[:] = [r for r in rows if tuple(r[k] for k in pk) != tuple(row[k] for k in pk)]
            rows.append(row)
            return _Res([row])
        self.db["_selects"].append((self.table, dict(self.filters)))
        out = [r for r in rows if all(r.get(c) == v for c, v in self.filters)]
        if self._order:
            out.sort(key=lambda r: r[self._order[0]], reverse=self._order[1])
        return _Res(out[: self._limit] if self._limit else out)


class _FakeClient:
    def __init__(self):
        self.db = {"_selects": []}

    def table(self, name):
        return _Query(self.db, name)


class TestSupabaseStore:
    def test_round_trip_resume_and_every_read_is_user_scoped(self):
        from workspace.store import SupabaseStore
        client = _FakeClient()
        svc = ConversationService(SupabaseStore(client), ME)
        conv, live = _chat(svc, [("BRL", BRL), ("JPY", JPY)])

        res = svc.resume(conv.id, lambda c: _session(SNAP, c))
        assert res.conversation.title.startswith("USDJPY ↑")
        assert res.active.spec.pair == "USDJPY"
        assert res.stale is False
        assert res.session.messages == messages_from_json(messages_to_json(live.messages))
        assert [i.pair for i in svc.store.ideas_for_conversation(ME, conv.id)] == [
            "USDJPY", "USDBRL"]

        selects = client.db["_selects"]
        assert selects and all(f.get("user_email") == ME for _, f in selects)
        assert ConversationService(svc.store, "x@y.com").resume(conv.id, lambda c: _session(SNAP, c)) is None


def test_system_prompt_states_refreshed_rule():
    prompt = build_system_prompt(("USDBRL",))
    assert "REFRESHED" in prompt and "SETTINGS UPDATED" in prompt


# ---------------------------------------------------------------------------
# Per-chat settings + distributions
# ---------------------------------------------------------------------------

class TestChatSettings:
    def _brl_chat(self):
        svc = ConversationService(InMemoryStore(), ME)
        conv, session = _chat(svc, [("BRL", BRL)])
        return svc, conv, session

    def _curve(self, session):
        from tests._curves import stated_lognormal
        ms = session.pack.market_state
        return stated_lognormal(ms.fwd * 0.97, ms.vol, ms.T)   # PM-stated, BRL lower

    def test_settings_round_trip(self):
        from workspace.settings import ChatSettings
        s = ChatSettings(sizing_method="kelly", kelly_lambda=0.25, target_rr=2.0,
                         structure_constraint="Avoid complex structures")
        assert ChatSettings.from_dict(s.to_dict()) == s
        assert ChatSettings.from_dict({"junk": 1}) == ChatSettings()

    def test_kelly_without_distribution_reruns_fixed_loss_and_says_so(self):
        from workspace.settings import ChatSettings
        svc, conv, session = self._brl_chat()
        out = svc.apply_settings(conv, session, ChatSettings(sizing_method="kelly"), {},
                                 seq=1, persist=True)
        assert session.pack.sizing_method == "fixed_loss" and session.pack.kelly_fallback
        assert "no distribution for USDBRL" in out.message and "fixed-loss" in out.message
        assert out.conversation.settings["sizing_method"] == "kelly"
        roles = [m["role"] for m in session.messages[-4:]]
        assert roles == ["user", "assistant", "user", "assistant"]
        tool_result = session.messages[-2]["content"][0]["content"]
        assert tool_result.startswith("SETTINGS UPDATED")
        assert "has NOT set up a distribution" in tool_result
        turns = svc.store.list_turns(ME, conv.id)
        assert turns[-1].kind == "settings" and display_turns(turns)[-1] == ("assistant", out.message)

    def test_distribution_for_the_active_trade_sizes_kelly(self):
        from analytics.sizing import curve_key
        from workspace.settings import ChatSettings, with_distribution
        svc, conv, session = self._brl_chat()
        probs, bins = self._curve(session)
        key = curve_key("USDBRL", session.expiry_for(session.view))
        dists = with_distribution({}, key, probs, bins)
        out = svc.apply_settings(conv, session, ChatSettings(sizing_method="kelly"), dists,
                                 seq=1, persist=True)
        assert session.pack.sizing_method == "kelly" and not session.pack.kelly_fallback
        assert "Kelly λ 0.5" in out.message and "USDBRL" in out.message
        assert key in out.conversation.distributions

    def test_switching_trade_in_a_kelly_chat_falls_back_until_a_distribution_exists(self):
        from analytics.sizing import curve_key
        from workspace.settings import ChatSettings, with_distribution
        svc, conv, session = self._brl_chat()
        probs, bins = self._curve(session)
        key = curve_key("USDBRL", session.expiry_for(session.view))
        conv = svc.apply_settings(conv, session, ChatSettings(sizing_method="kelly"),
                                  with_distribution({}, key, probs, bins),
                                  seq=1, persist=True).conversation
        flow = AgentFlow(FakeToolLLM(script=_pack_turn("j", **JPY) + _pack_turn("b", **BRL)),
                         session)
        flow.advance("now JPY")
        assert session.pack.kelly_fallback                      # no JPY distribution
        flow.advance("back to BRL")
        assert session.pack.sizing_method == "kelly"            # BRL's applies again

    def test_resume_restores_the_chats_settings_and_distributions(self):
        from analytics.sizing import curve_key
        from workspace.settings import ChatSettings, with_distribution
        svc, conv, session = self._brl_chat()
        probs, bins = self._curve(session)
        key = curve_key("USDBRL", session.expiry_for(session.view))
        svc.apply_settings(conv, session, ChatSettings(sizing_method="kelly", kelly_lambda=0.3),
                           with_distribution({}, key, probs, bins), seq=1, persist=True)
        res = svc.resume(conv.id, lambda c: _session(SNAP, c))
        assert res.session.sizing_method == "kelly" and res.session.kelly_lambda == 0.3
        assert res.session.pack.sizing_method == "kelly"
        assert res.stale is False

    def test_settings_before_any_trade_are_stored_without_a_turn(self):
        from workspace.settings import ChatSettings
        svc = ConversationService(InMemoryStore(), ME)
        conv, session = _chat(svc, [("hello", None)])
        out = svc.apply_settings(conv, session, ChatSettings(target_rr=2.0), {},
                                 seq=1, persist=True)
        assert out.message is None
        assert svc.store.get_conversation(ME, conv.id).settings["target_rr"] == 2.0
        assert len(svc.store.list_turns(ME, conv.id)) == 1

    def test_supabase_store_degrades_when_settings_columns_missing(self):
        from workspace.store import SupabaseStore
        client = _FakeClient()
        store = SupabaseStore(client)
        orig = client.table

        def no_late_columns(name):
            q = orig(name)
            real_upsert = q.upsert

            def upsert(row, on_conflict=None):
                if name == "conversations" and "settings" in row:
                    raise RuntimeError("column settings does not exist")
                return real_upsert(row, on_conflict)
            q.upsert = upsert
            return q
        client.table = no_late_columns
        conv = Conversation(user_email=ME, settings={"sizing_method": "kelly"})
        store.save_conversation(conv)
        assert store.settings_persist is False
        assert store.get_conversation(ME, conv.id) is not None
