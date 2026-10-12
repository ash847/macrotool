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
