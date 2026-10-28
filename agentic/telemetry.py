"""Per-call metadata kept separate from provider messages and engine tool results."""

import time


def tracked_call(llm, session, system, tools, exchange_start):
    started = time.perf_counter()
    record = {
        "event": "llm_call",
        "exchange_start": exchange_start,
        "model": getattr(llm, "model", None),
        "input_tokens": None,
        "cache_creation_input_tokens": None,
        "cache_read_input_tokens": None,
        "output_tokens": None,
        "stop_reason": None,
        "is_error": False,
    }
    try:
        turn = llm.create(session.messages, system, tools)
        record.update(turn.metrics or {})
        record["stop_reason"] = turn.stop_reason
        return turn
    except Exception as error:
        record["is_error"] = True
        record["error_type"] = type(error).__name__
        raise
    finally:
        record.setdefault("latency_ms", round((time.perf_counter() - started) * 1000, 2))
        session.llm_calls.append(record)


def exchange_trace(tool_trace, session, exchange_start):
    return list(tool_trace or []) + [
        dict(call) for call in getattr(session, "llm_calls", [])
        if call["exchange_start"] == exchange_start
    ]


def _bget(block, key):
    """Read a field from a content block that may be an SDK object or a dict."""
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)


def tool_trace_from_messages(messages) -> list[dict]:
    """Reconstruct (tool name, args, result, is_error) from a message slice.

    Matches each assistant tool_use block to its tool_result by id. This is the
    ground truth the model received — read the narration against it.
    """
    pending: dict[str, dict] = {}
    trace: list[dict] = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            btype = _bget(b, "type")
            if role == "assistant" and btype == "tool_use":
                bid = _bget(b, "id")
                pending[bid] = {"name": _bget(b, "name"), "args": _bget(b, "input")}
            elif role == "user" and btype == "tool_result":
                tid = _bget(b, "tool_use_id")
                call = pending.pop(tid, {"name": "?", "args": None})
                trace.append({
                    "name": call["name"],
                    "args": call["args"],
                    "result": _bget(b, "content"),
                    "is_error": bool(_bget(b, "is_error")),
                })
    return trace


def view_json(view) -> dict | None:
    """Compact, self-describing view snapshot stored on each chat turn."""
    if view is None:
        return None
    return {
        "pair": view.pair, "direction": view.direction,
        "horizon_days": view.horizon_days, "magnitude_pct": view.magnitude_pct,
        "mode": view.mode,
    }
