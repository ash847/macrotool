# Security

## Threat model

MacroTool runs at a public Streamlit Community Cloud URL. The primary threat is any unauthenticated internet user discovering that URL, hitting the app, or calling Supabase REST directly with a leaked publishable key.

## Controls

- Google OIDC via Streamlit `st.login`
- admin allowlist via `admin_emails` in Streamlit secrets
- split Supabase key model:
  - `SUPABASE_SERVICE_KEY` for server-side app writes, engine config reads, and admin-only operations
  - `SUPABASE_ANON_KEY` retained only for direct REST smoke tests and Security Advisor validation
- server-only Anthropic API key
- deny-by-default Supabase RLS on `queries`, `feedback`, `config_history`

## Required secrets

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
SUPABASE_URL = "..."
SUPABASE_ANON_KEY = "..."
SUPABASE_SERVICE_KEY = "..."
admin_emails = ["ash@fund.com"]

[auth]
redirect_uri = "https://<your-app>.streamlit.app/~/+/oauth2callback"
cookie_secret = "<openssl rand -hex 32>"

[auth.google]
client_id = "..."
client_secret = "..."
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
```

## Key rotation

- Rotate `SUPABASE_SERVICE_KEY` every 90 days.
- Rotate `SUPABASE_ANON_KEY` immediately if it is exposed outside the app config.
- Rotate `ANTHROPIC_API_KEY` immediately if it appears in logs, screenshots, or a leaked secrets file.

## Suspected key leak runbook

1. Rotate the exposed key at the provider immediately.
2. Update Streamlit Cloud secrets with the new value.
3. Restart the Streamlit app.
4. Re-run direct REST negative checks against Supabase with the anon key.
5. Run `gitleaks detect --source . --no-banner` and inspect recent history for the leak source.
6. Record the incident and confirm the old key now returns `401`.

## Supabase RLS baseline

Expected posture:

- `queries`: no anon access; service role only
- `feedback`: no anon access; service role only
- `config_history`: no anon access; service role only

The code assumes those policies exist. Without them, the database may remain exposed even if the app itself is gated.
