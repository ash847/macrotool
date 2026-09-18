-- Saved conversations + trade ideas (workspace/). Run once in the Supabase SQL editor.
--
-- The app writes with the SERVICE key (bypasses RLS); per-user scoping is enforced in
-- workspace/store.py, which filters every read on user_email. RLS is enabled with no
-- policies so the anon key can read/write nothing on these tables.
--
-- Phase 1 (option A): conversations.active_idea_id labels each chat. conversation_ideas
-- records every idea a chat touched — unused by the UI today, it is the join that
-- option C (ideas shared across conversations) needs, so C requires no migration.

create table if not exists trade_ideas (
    id          uuid primary key,
    user_email  text not null,
    pair        text not null,
    direction   text not null,          -- base_higher | base_lower
    status      text not null default 'active',
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);
create index if not exists trade_ideas_user_idx on trade_ideas (user_email);

create table if not exists idea_versions (
    id                    uuid primary key,
    idea_id               uuid not null references trade_ideas (id) on delete cascade,
    user_email            text not null,
    version_no            integer not null,
    spec                  jsonb not null,  -- ViewSpec: pair, direction, expiry_date, target_level, mode, conviction
    prefs                 jsonb not null default '{}'::jsonb,
    snapshot_date         date not null,
    snapshot_fingerprint  text not null,
    rendered_pack         text not null default '',
    top_structure         text,
    created_at            timestamptz not null default now(),
    unique (idea_id, version_no)
);
create index if not exists idea_versions_user_idea_idx on idea_versions (user_email, idea_id);

create table if not exists conversations (
    id                  uuid primary key,
    user_email          text not null,
    surface             text not null default 'agent_tab',
    title               text not null default 'New conversation',
    title_custom        boolean not null default false,
    archived            boolean not null default false,
    active_idea_id      uuid references trade_ideas (id) on delete set null,
    last_snapshot_date  date,
    created_at          timestamptz not null default now(),
    updated_at          timestamptz not null default now()
);
create index if not exists conversations_user_updated_idx
    on conversations (user_email, archived, updated_at desc);

create table if not exists conversation_ideas (
    conversation_id  uuid not null references conversations (id) on delete cascade,
    idea_id          uuid not null references trade_ideas (id) on delete cascade,
    user_email       text not null,
    last_active_at   timestamptz not null default now(),
    primary key (conversation_id, idea_id)
);
create index if not exists conversation_ideas_user_idx on conversation_ideas (user_email);

create table if not exists conversation_turns (
    id               uuid primary key,
    conversation_id  uuid not null references conversations (id) on delete cascade,
    user_email       text not null,
    seq              integer not null,
    kind             text not null default 'exchange',   -- exchange | refresh
    user_text        text not null default '',
    reply_text       text not null default '',
    llm_messages     jsonb not null default '[]'::jsonb,  -- exact provider messages (replay source)
    idea_version_id  uuid references idea_versions (id) on delete set null,
    snapshot_date    date,
    created_at       timestamptz not null default now()
);
create index if not exists conversation_turns_conv_idx
    on conversation_turns (user_email, conversation_id, seq);

alter table trade_ideas        enable row level security;
alter table idea_versions      enable row level security;
alter table conversations      enable row level security;
alter table conversation_ideas enable row level security;
alter table conversation_turns enable row level security;
