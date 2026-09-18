-- Per-chat sizing settings + stated Kelly distributions (run once, after workspace_schema.sql).
alter table conversations add column if not exists settings      jsonb not null default '{}'::jsonb;
alter table conversations add column if not exists distributions jsonb not null default '{}'::jsonb;
