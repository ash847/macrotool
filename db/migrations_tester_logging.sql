-- Tester-rollout logging tables. Run in the Supabase SQL editor (service role).
-- All writes go through the service key from interface/supabase_logger.py.
-- Keep RLS DISABLED on these (like queries/feedback/config_history) since access
-- is mediated by the service key, not row-level policies.

-- 1) Chat transcripts — one row per chat turn (Agent tab + in-context trade chat).
create table if not exists chat_turns (
  id          bigint generated always as identity primary key,
  session_id  uuid,
  chat_id     uuid        not null,
  seq         int         not null,
  surface     text        not null,   -- 'agent_tab' | 'trade_view'
  role        text        not null,   -- 'user' | 'assistant'
  text        text,
  tool_trace  jsonb,                  -- assistant turns: [{name,args,result,is_error}]
  pair        text,
  view_json   jsonb,
  user_email  text,
  created_at  timestamptz not null default now()
);
create index if not exists chat_turns_chat_idx    on chat_turns (chat_id, seq);
create index if not exists chat_turns_session_idx on chat_turns (session_id);
alter table chat_turns disable row level security;

-- 2) Application errors — so failures testers hit are visible remotely
--    (the local logs/session.log is ephemeral on Streamlit Cloud).
create table if not exists app_errors (
  id          bigint generated always as identity primary key,
  session_id  uuid,
  context     text,
  error_type  text,
  message     text,
  traceback   text,
  user_email  text,
  created_at  timestamptz not null default now()
);
create index if not exists app_errors_session_idx on app_errors (session_id);
alter table app_errors disable row level security;

-- 3) One-click reactions — 👍/👎 (+ optional one-tap reason on 👎) on a
--    recommendation or a chat reply.
create table if not exists reactions (
  id           bigint generated always as identity primary key,
  session_id   uuid,
  surface      text,                  -- 'trade_view' | 'agent_tab'
  target_kind  text,                  -- 'recommendation' | 'chat'
  target_ref   text,                  -- trade signature, or chat_id:seq
  rating       text,                  -- 'up' | 'down'
  reason       text,                  -- reason chip on 👎 (null on 👍)
  pair         text,
  view_summary text,
  chat_id      uuid,
  seq          int,
  user_email   text,
  created_at   timestamptz not null default now()
);
create index if not exists reactions_session_idx on reactions (session_id);
alter table reactions disable row level security;

-- 4) OPTIONAL — add session_id to the existing tables so chat/errors/reactions
--    join to the engine runs by session. Safe to skip; the app degrades to
--    logging without session_id if these columns are absent.
alter table queries  add column if not exists session_id uuid;
alter table feedback add column if not exists session_id uuid;
