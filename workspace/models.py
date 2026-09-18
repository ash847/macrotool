"""Pure data objects for saved conversations and trade ideas. No IO.

The persisted view is a ``ViewSpec``: an **absolute target level and expiry date**,
not the engine's ``horizon_days`` / ``magnitude_pct``. A 3m view reopened a month
later is a 2m trade to the same date and level; re-deriving from horizon/% would
silently move both. The clock is always the **snapshot date** (the pricing date),
never wall-clock time, so a static snapshot reproduces the same trade exactly.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _date(v) -> date | None:
    if v is None or isinstance(v, date):
        return v
    return date.fromisoformat(str(v)[:10])


def _round_level(x: float | None) -> float | None:
    return None if x is None else round(float(x), 6)


def format_level(x: float) -> str:
    """Human level at FX-quote precision: 5.6 → '5.6', 4.35125 → '4.3513',
    38.4567 → '38.457', 168.0342 → '168.03'."""
    dp = 2 if abs(x) >= 100 else 3 if abs(x) >= 10 else 4
    return f"{x:,.{dp}f}".rstrip("0").rstrip(".")


class IdeaExpired(ValueError):
    """The idea's expiry is on/before the snapshot date — nothing left to price."""


@dataclass(frozen=True)
class ViewSpec:
    pair: str
    direction: str                    # "base_higher" | "base_lower"
    expiry_date: date
    target_level: float | None = None  # None = pure directional (no target)
    mode: str = "recommend"
    direction_conviction: str = "medium"

    @classmethod
    def from_view(cls, view, target: float | None, snapshot_date: date) -> "ViewSpec":
        """Freeze an engine ``TradeView`` (+ the pack's absolute target) into a spec."""
        return cls(
            pair=view.pair,
            direction=view.direction,
            expiry_date=snapshot_date + timedelta(days=int(view.horizon_days)),
            target_level=_round_level(target),
            mode=view.mode,
            direction_conviction=view.direction_conviction,
        )

    def horizon_days(self, snapshot_date: date) -> int:
        return (self.expiry_date - snapshot_date).days

    def pack_args(self, snapshot_date: date) -> dict:
        """``run_standard_pack`` tool input that reproduces this view on ``snapshot_date``.

        Uses ``target_level`` so the tool re-derives magnitude (and direction) from
        *that day's* forward — the same path as a PM naming a level.
        """
        days = self.horizon_days(snapshot_date)
        if days <= 0:
            raise IdeaExpired(
                f"{self.pair} idea expired {self.expiry_date.isoformat()} "
                f"(snapshot {snapshot_date.isoformat()})"
            )
        args: dict = {
            "pair": self.pair,
            "horizon_days": days,
            "mode": self.mode,
            "direction_conviction": self.direction_conviction,
        }
        if self.target_level is not None:
            args["target_level"] = self.target_level
        else:
            args["direction"] = self.direction
        return args

    def label(self) -> str:
        arrow = "↑" if self.direction == "base_higher" else "↓"
        parts = [f"{self.pair} {arrow}"]
        if self.target_level is not None:
            parts[0] += f" {format_level(self.target_level)}"
        parts.append(self.expiry_date.strftime("%d-%b-%y"))
        return " · ".join(parts)

    def to_json(self) -> dict:
        return {
            "pair": self.pair, "direction": self.direction,
            "expiry_date": self.expiry_date.isoformat(),
            "target_level": self.target_level, "mode": self.mode,
            "direction_conviction": self.direction_conviction,
        }

    @classmethod
    def from_json(cls, d: dict) -> "ViewSpec":
        return cls(
            pair=d["pair"], direction=d["direction"],
            expiry_date=_date(d["expiry_date"]),
            target_level=d.get("target_level"),
            mode=d.get("mode", "recommend"),
            direction_conviction=d.get("direction_conviction", "medium"),
        )


@dataclass
class TradeIdea:
    """A trade thought, identified by pair + direction. Versions carry the details."""
    user_email: str
    pair: str
    direction: str
    id: str = field(default_factory=new_id)
    status: str = "active"
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)

    def to_row(self) -> dict:
        return {
            "id": self.id, "user_email": self.user_email, "pair": self.pair,
            "direction": self.direction, "status": self.status,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, r: dict) -> "TradeIdea":
        return cls(
            id=r["id"], user_email=r["user_email"], pair=r["pair"],
            direction=r["direction"], status=r.get("status", "active"),
            created_at=r.get("created_at") or utcnow(),
            updated_at=r.get("updated_at") or utcnow(),
        )


@dataclass
class IdeaVersion:
    """One evaluation of an idea: the spec, the prefs it ran under, and what came out."""
    idea_id: str
    user_email: str
    version_no: int
    spec: ViewSpec
    prefs: dict
    snapshot_date: date
    snapshot_fingerprint: str
    rendered_pack: str = ""
    top_structure: str | None = None
    id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=utcnow)

    def to_row(self) -> dict:
        return {
            "id": self.id, "idea_id": self.idea_id, "user_email": self.user_email,
            "version_no": self.version_no, "spec": self.spec.to_json(),
            "prefs": self.prefs, "snapshot_date": self.snapshot_date.isoformat(),
            "snapshot_fingerprint": self.snapshot_fingerprint,
            "rendered_pack": self.rendered_pack, "top_structure": self.top_structure,
            "created_at": self.created_at,
        }

    @classmethod
    def from_row(cls, r: dict) -> "IdeaVersion":
        return cls(
            id=r["id"], idea_id=r["idea_id"], user_email=r["user_email"],
            version_no=int(r["version_no"]), spec=ViewSpec.from_json(r["spec"]),
            prefs=r.get("prefs") or {}, snapshot_date=_date(r["snapshot_date"]),
            snapshot_fingerprint=r.get("snapshot_fingerprint") or "",
            rendered_pack=r.get("rendered_pack") or "",
            top_structure=r.get("top_structure"),
            created_at=r.get("created_at") or utcnow(),
        )


@dataclass
class Conversation:
    user_email: str
    surface: str = "agent_tab"
    title: str = "New conversation"
    title_custom: bool = False          # PM renamed it — stop auto-titling
    archived: bool = False
    active_idea_id: str | None = None
    last_snapshot_date: date | None = None
    settings: dict = field(default_factory=dict)        # ChatSettings.to_dict()
    distributions: dict = field(default_factory=dict)   # {curve_key: {"probs", "bins"}}
    id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)

    def touched(self, **changes) -> "Conversation":
        return replace(self, updated_at=utcnow(), **changes)

    def to_row(self) -> dict:
        return {
            "id": self.id, "user_email": self.user_email, "surface": self.surface,
            "title": self.title, "title_custom": self.title_custom,
            "archived": self.archived, "active_idea_id": self.active_idea_id,
            "last_snapshot_date": (
                self.last_snapshot_date.isoformat() if self.last_snapshot_date else None
            ),
            "settings": self.settings, "distributions": self.distributions,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, r: dict) -> "Conversation":
        return cls(
            id=r["id"], user_email=r["user_email"],
            surface=r.get("surface", "agent_tab"),
            title=r.get("title") or "New conversation",
            title_custom=bool(r.get("title_custom")),
            archived=bool(r.get("archived")),
            active_idea_id=r.get("active_idea_id"),
            last_snapshot_date=_date(r.get("last_snapshot_date")),
            settings=r.get("settings") or {},
            distributions=r.get("distributions") or {},
            created_at=r.get("created_at") or utcnow(),
            updated_at=r.get("updated_at") or utcnow(),
        )


@dataclass
class Turn:
    """One PM exchange: what they typed, the reply shown, and the exact provider
    messages appended during it (the replay source for the model's context)."""
    conversation_id: str
    user_email: str
    seq: int
    user_text: str
    reply_text: str
    llm_messages: list = field(default_factory=list)   # JSON-safe (see serialize)
    idea_version_id: str | None = None
    snapshot_date: date | None = None
    kind: str = "exchange"              # "exchange" | "refresh" | "settings"
    id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=utcnow)

    def to_row(self) -> dict:
        return {
            "id": self.id, "conversation_id": self.conversation_id,
            "user_email": self.user_email, "seq": self.seq, "kind": self.kind,
            "user_text": self.user_text, "reply_text": self.reply_text,
            "llm_messages": self.llm_messages, "idea_version_id": self.idea_version_id,
            "snapshot_date": self.snapshot_date.isoformat() if self.snapshot_date else None,
            "created_at": self.created_at,
        }

    @classmethod
    def from_row(cls, r: dict) -> "Turn":
        return cls(
            id=r["id"], conversation_id=r["conversation_id"],
            user_email=r["user_email"], seq=int(r["seq"]),
            kind=r.get("kind") or "exchange",
            user_text=r.get("user_text") or "", reply_text=r.get("reply_text") or "",
            llm_messages=r.get("llm_messages") or [],
            idea_version_id=r.get("idea_version_id"),
            snapshot_date=_date(r.get("snapshot_date")),
            created_at=r.get("created_at") or utcnow(),
        )
