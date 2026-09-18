"""Per-conversation sizing settings and PM preferences.

Each chat carries its own sizing method (fixed-loss / Kelly), λ, R:R and PM
preferences, plus the PM's stated Kelly distributions keyed by pair + expiry
(``analytics.sizing.curve_key``). Only capital W is global (the PM's book size).

A chat set to Kelly sizes a trade under Kelly only if a distribution is stated for
that trade's pair + expiry; otherwise the trade is sized fixed-loss and flagged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

DEFAULTS = {
    "sizing_method": "fixed_loss",
    "kelly_lambda": 0.5,
    "target_rr": 3.0,
    "primary_objective": "Balanced",
    "structure_constraint": "No restriction",
    "trade_management": "Standard hold",
}


@dataclass(frozen=True)
class ChatSettings:
    sizing_method: str = DEFAULTS["sizing_method"]          # "fixed_loss" | "kelly"
    kelly_lambda: float = DEFAULTS["kelly_lambda"]
    target_rr: float = DEFAULTS["target_rr"]
    primary_objective: str = DEFAULTS["primary_objective"]
    structure_constraint: str = DEFAULTS["structure_constraint"]
    trade_management: str = DEFAULTS["trade_management"]

    @classmethod
    def from_dict(cls, d: dict | None) -> "ChatSettings":
        d = d or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)

    def apply_to(self, session) -> None:
        session.sizing_method = self.sizing_method
        session.kelly_lambda = float(self.kelly_lambda)
        session.target_rr = float(self.target_rr)
        session.primary_objective = self.primary_objective
        session.structure_constraint = self.structure_constraint
        session.trade_management = self.trade_management

    def sizing_label(self) -> str:
        if self.sizing_method == "kelly":
            return f"Kelly λ {self.kelly_lambda:g}"
        return f"Fixed-loss R:R {self.target_rr:g}"


def curves_from(distributions: dict | None) -> dict:
    """``{curve_key: (probs, bins)}`` from a conversation's stored distributions
    (``{curve_key: {"probs": [...], "bins": [...]}}``)."""
    out = {}
    for key, d in (distributions or {}).items():
        try:
            out[key] = (tuple(float(p) for p in d["probs"]), tuple(float(b) for b in d["bins"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def with_distribution(distributions: dict | None, key: str, probs, bins) -> dict:
    """A copy of ``distributions`` with ``key`` set (or removed when probs is None)."""
    out = dict(distributions or {})
    if probs is None or bins is None:
        out.pop(key, None)
    else:
        out[key] = {"probs": [float(p) for p in probs], "bins": [float(b) for b in bins]}
    return out
