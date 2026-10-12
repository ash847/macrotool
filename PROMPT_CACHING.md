# Agent prompt caching and usage telemetry

The Anthropic agent adapter enables five-minute ephemeral caching by default:
an explicit system breakpoint reuses the tools/system prefix across chats, while
top-level automatic caching reuses growing conversation history and tool results.
Original message content, tool schemas and signed thinking blocks are untouched.
Set `cache_enabled=False` when constructing `AnthropicToolLLM` to disable both.
The legacy conversation client is not changed by this caching implementation.

The SDK minimum is 0.86.0, verified to support top-level `cache_control`.
Prefixes must match, meet Anthropic's model-specific minimum length, and be reused
before expiry to hit. Changes to tools/system/history may miss. There is no local
response cache and no skipping of engine calculations or substitution of stale
market data. Caching does not shorten context, guarantee identical sampled output,
or reduce output-token pricing.

Each model call records model, request ID, stop reason, complete-call latency in
milliseconds (not time to first token), cache setting, ordinary input tokens,
cache creation tokens, cache read tokens and output tokens. The three input fields
are separate usage categories; cached reads are not included in `input_tokens`.
Missing usage remains null, not zero. Failed calls record only error type and
elapsed time; SDK-internal retry attempts are not individually reported.

Metrics are kept in `AgentSession.llm_calls`, outside model history. The existing
chat logger appends `event: "llm_call"` records to the assistant row's `tool_trace`
in Supabase `chat_turns`, scoped to that exchange. Tool entries remain unchanged.
No database migration is needed. As with existing chat telemetry, persistence is
best-effort and requires configured Supabase logging; usage is not persisted in
the workspace's separate `conversation_turns` table. Trace consumers should
distinguish `llm_call` events from tool-result records.

Compare cache creation/read/input counts and full-call latency over real sessions
before changing TTL or estimating savings. Cache writes cost more than ordinary
input; do not claim a whole-bill saving from the cached-input discount alone.

Tests: `tests/test_prompt_caching.py`. For three paid live requests, set
`RUN_LIVE_CACHE_TESTS=1` and provide `ANTHROPIC_API_KEY` or `LIVE_SECRETS_FILE`, then
run `tests/test_prompt_caching_live.py -s`. The test checks a cold write, a tool
continuation cache hit and a new-chat system-prefix hit without printing secrets.
