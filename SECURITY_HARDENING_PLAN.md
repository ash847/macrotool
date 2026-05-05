# Security Hardening Plan — Streamlit + Supabase

Goal: lock down the public Streamlit app and the Supabase backend so that
(a) only authenticated users can use the tool, (b) only admins can see logs
and edit config, and (c) the database cannot be read or scribbled on by
anyone who finds the URL and the publishable key.

## Threat model (one sentence)

The app runs on Streamlit Community Cloud at a public URL; the realistic
adversary is anyone on the internet who finds that URL — they can hit the
app, hit Supabase REST directly with any leaked key, and burn the team's
Anthropic credits. Internal "curious PM" is a secondary concern.

## Posture decisions (made up front so phases don't conflict)

- **Identity**: Streamlit native `st.login` (≥ 1.42) with Google OIDC.
  Admins are an allowlist of verified Google emails in
  `st.secrets["admin_emails"]`. If the fund has Google Workspace, set the
  Google OAuth consent screen to **Internal** so only `@fund.com` accounts
  can even attempt login.
- **Supabase keys — two-client model**:
  - `SUPABASE_ANON_KEY` — used for user-facing writes (`log_query`,
    `log_feedback`). RLS policies allow insert-only on those tables.
  - `SUPABASE_SERVICE_KEY` — used **only** inside `assert_admin()`-gated
    code paths for reads of `queries` / `config_history` and writes of
    config. Bypasses RLS by design; Python is the gate.
  - Engine config reads (`fetch_config` for the live engine) go through
    the service key from server code only; never returned to the client
    page.
- **RLS policy stance**: deny-by-default on every table. Explicit
  insert-only policies for `queries` and `feedback`. No public select on
  anything. This is the load-bearing security control — everything else is
  defence-in-depth.
- **Anthropic key**: server-only. Remove the sidebar key input. Anyone who
  reaches the app has been authenticated, so the team's credits are no
  longer exposed to the open internet.

---

## Phase 0 — Triage (do today, before any code changes)

1. **Rotate the current `SUPABASE_KEY` in the Supabase dashboard.** Assume
   it is compromised. Update Streamlit secrets with the new value under
   the *old* name for now; renaming happens in Phase 2.
2. **Scan repo + history for leaked secrets**:
   ```bash
   brew install gitleaks
   gitleaks detect --source . --no-banner
   git log -p -- '**/secrets*' '**/.env*' | head -200
   ```
   Anything that surfaces gets rotated immediately at its provider.
3. **Confirm secrets file is gitignored**: `.streamlit/secrets.toml` must
   not be tracked.
4. **Snapshot the Supabase schema** (`pg_dump --schema-only`) so we can
   roll back RLS changes if Phase 3 misfires.

Acceptance: gitleaks clean, old Supabase key revoked, schema snapshot
saved locally.

---

## Phase 1 — Supabase RLS + key split (the actual security fix)

Do this in a **staging Supabase project** first if possible; otherwise do
it on prod during a quiet window with the schema snapshot ready.

1. **Create a service-role key reference**: Supabase already provides
   `service_role` — just plan to use it from server-only code paths.
2. **Write RLS policies** (run in SQL editor):
   ```sql
   -- queries: anon can insert only, nobody can read
   alter table queries enable row level security;
   create policy "anon insert queries" on queries
     for insert to anon with check (true);

   -- feedback: same
   alter table feedback enable row level security;
   create policy "anon insert feedback" on feedback
     for insert to anon with check (true);

   -- config_history: nothing for anon. Service role bypasses RLS.
   alter table config_history enable row level security;
   ```
3. **Revoke broad grants** (if any `GRANT ... TO anon` exist on these
   tables beyond what RLS now permits, revoke them).
4. **Smoke test from outside the app**:
   ```bash
   # With anon key — should succeed
   curl -X POST "$SUPABASE_URL/rest/v1/queries" \
     -H "apikey: $ANON" -H "Authorization: Bearer $ANON" \
     -H "Content-Type: application/json" \
     -d '{"prompt":"test"}'
   # With anon key — should return [] or 401
   curl "$SUPABASE_URL/rest/v1/queries?select=*" \
     -H "apikey: $ANON" -H "Authorization: Bearer $ANON"
   curl "$SUPABASE_URL/rest/v1/config_history?select=*" \
     -H "apikey: $ANON" -H "Authorization: Bearer $ANON"
   ```
5. **Add rate-limit guard** on `queries` insert (Postgres trigger or
   Supabase Edge Function): cap at e.g. 60 inserts per IP per hour.
   Insert-only RLS still permits unlimited spam from anyone with the anon
   key — this stops table-bloat DoS.
6. **Run Supabase Security Advisor**, confirm no "public table" or
   "RLS disabled" warnings remain on `queries`, `feedback`,
   `config_history`.

Acceptance: direct REST reads of any sensitive table return empty/401
with the anon key; the running app still writes query/feedback rows
successfully.

---

## Phase 2 — Refactor `interface/supabase_logger.py`

Split the single client into two, and split the helpers by audience.

```python
# interface/supabase_logger.py (sketch)

_anon_client = None     # for log_query, log_feedback
_service_client = None  # for fetch_*, save_config — admin-gated callers only

def _init() -> None:
    url = os.environ["SUPABASE_URL"]
    anon = os.environ.get("SUPABASE_ANON_KEY")
    svc  = os.environ.get("SUPABASE_SERVICE_KEY")
    ...

# --- user surface (safe to call from any page) ---
def log_query(...): _anon_client.table("queries").insert(...).execute()
def log_feedback(...): _anon_client.table("feedback").insert(...).execute()

# --- engine surface (server-side only, never returned to UI) ---
def fetch_config_for_engine(key: str) -> dict | None:
    # Uses service key. Result feeds the engine; never rendered to the page.
    ...

# --- admin surface (callers MUST pre-check is_admin) ---
def fetch_queries(*, _admin: bool) -> list[dict]:
    if not _admin:
        raise PermissionError("admin only")
    ...
def fetch_config_history(key: str, *, _admin: bool) -> list[dict]: ...
def save_config(key: str, value: dict, *, _admin: bool) -> bool: ...
```

The `_admin: bool` keyword-only argument forces every call site to
declare intent — easy to grep, hard to forget. The real check still lives
at the page level, this is a defence-in-depth backstop.

Rename env vars in `_inject_secrets()` and Streamlit secrets:
- `SUPABASE_KEY` → `SUPABASE_ANON_KEY`
- (new) `SUPABASE_SERVICE_KEY`

Acceptance: every admin helper raises if `_admin=False`. Grep for
`fetch_queries\|fetch_config_history\|save_config` shows every call site
passing `_admin=is_admin_user()`.

---

## Phase 3 — Auth gate (Google OIDC via `st.login`)

1. **Google Cloud setup** (one-off, ~15 min):
   - New project (or reuse existing).
   - APIs & Services → OAuth consent screen → **Internal** if Workspace,
     else **External** + add admin allowlist later.
   - Credentials → Create OAuth client → Web application.
   - Authorised redirect URI:
     `https://<your-app>.streamlit.app/oauth2callback`
   - Save `client_id` / `client_secret`.
2. **Add to `.streamlit/secrets.toml`** (and Streamlit Cloud secrets UI):
   ```toml
   [auth]
   redirect_uri = "https://<your-app>.streamlit.app/oauth2callback"
   cookie_secret = "openssl rand -hex 32 output here"
   client_id = "..."
   client_secret = "..."
   server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

   admin_emails = ["ash@fund.com", "..."]
   ```
3. **Gate the app** at the top of `interface/app.py`, after
   `st.set_page_config` and `_inject_secrets()`:
   ```python
   if not st.user.is_logged_in:
       st.title("MacroTool")
       st.write("Sign in to continue.")
       st.button("Sign in with Google", on_click=st.login)
       st.stop()

   USER_EMAIL = st.user.email
   IS_ADMIN = USER_EMAIL in st.secrets.get("admin_emails", [])
   ```
4. **Sidebar / navigation**: only render Structure Selection, Context
   Rules, and Query Log entries when `IS_ADMIN`. Backend admin helpers
   (Phase 2) re-check.
5. **Sign-out** in the sidebar: `st.button("Sign out", on_click=st.logout)`.
6. **Pin Streamlit version** in `pyproject.toml` to a release that
   supports `st.login` (≥ 1.42).

Acceptance: logged-out visitor sees only the login screen; logged-in
non-admin sees Trade View only; admin sees everything; navigating
directly to an admin page URL while non-admin still fails server-side.

---

## Phase 4 — Anthropic key & cost protection

1. **Remove the sidebar API-key input** in `interface/app.py:166-176`.
   The team key from secrets is the only path. Authentication makes the
   "let users bring their own key" escape hatch unnecessary.
2. **Surface a usage-cap warning**: optional, but cheap — track per-user
   `INTAKE` calls per day in session state and refuse beyond N.
3. **Confirm `ANTHROPIC_API_KEY` is server-side only** — never logged,
   never echoed to the page.

Acceptance: no path in the UI exposes or accepts an Anthropic key from
the user.

---

## Phase 5 — Verification (negative tests, not just happy path)

Run all of the following and record outputs:

1. **Logged-out visitor** hits the app URL → only login screen renders.
2. **Logged-in non-admin** hits each admin page URL directly → server-side
   `assert_admin` blocks; helper raises `PermissionError`.
3. **Anyone with anon key** hits Supabase REST directly:
   - `select` on `queries` / `config_history` → empty or 401.
   - `insert` on `config_history` → denied.
   - `insert` on `queries` → succeeds (expected) but rate-limited after N.
4. **Anyone with old (rotated) key** → all calls 401.
5. **gitleaks + trufflehog** clean against current `HEAD` and full history.
6. **Supabase Security Advisor** shows no warnings on the three tables.
7. **Stale session**: remove a user from `admin_emails`, force a refresh
   → admin nav disappears; admin helpers raise.

---

## Phase 6 — Docs & ops

1. Update `README.md` with the new login flow and admin-allowlist mechanism.
2. Update `CLAUDE.md` "Deployment" section: list the new secrets
   (`SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_KEY`, `[auth]` block,
   `admin_emails`) and document that adding an admin requires editing the
   Streamlit Cloud secrets and restarting the app.
3. Add a short `SECURITY.md` with the threat model, key rotation
   procedure, and the response runbook for "we suspect a key leaked."
4. Set a calendar reminder: rotate `SUPABASE_SERVICE_KEY` every 90 days.

---

## Order of operations (TL;DR)

```
0. Rotate current Supabase key. gitleaks scan.
1. Supabase RLS + revokes (in staging if possible).
2. Refactor supabase_logger.py into anon/service/admin surfaces.
3. Add st.login + admin allowlist + page gating.
4. Remove sidebar Anthropic key input.
5. Verification (including direct-REST negative tests).
6. Docs + rotation reminder.
```

Phase 1 is the only step that actually changes the security posture; 2–4
are how the app keeps working under that posture. Don't reverse the order.
