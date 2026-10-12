# About and contact

Testers have Agent and About pages only; existing Trade view/admin page state is
redirected to Agent. Admins retain their existing pages and also see About. The
admin's View as tester toggle uses the same restricted page list and route guard.

Editorial copy lives in `interface/about_content.json`, including the supplied
team biography and closing aside.
The first two sections adapt the provided introduction, with static-data and
indicative-pricing limitations and separate Kelly/risk-budget sizing descriptions.

The contact form is disabled until the destination and sender account are
configured. No messages are silently discarded or reported as sent while disabled.
The optional email transport uses authenticated encrypted SMTP and plain-text
messages, a fixed server-configured recipient, and the signed-in user's Reply-To.

Configure these in the deployed app's Streamlit secrets, not a committed file:

```toml
[contact_email]
host = "smtp.your-provider.example"
port = 587
security = "starttls"
username = "your-smtp-username"
password = "your-smtp-password"
from_email = "verified-sender@your-domain.example"
to_email = ["ashwath.venkataraman@gmail.com", "vincent_craignou@hotmail.com"]
```

For implicit TLS, use `security = "ssl"` and the provider's port (usually 465).
Confirm the hosting environment permits the provider's SMTP endpoint before
enabling it. If it does not, use the provider's HTTPS email API instead; the UI
is independent of `send_contact` so its transport can be replaced. Do not use an
unverified From address or a user's supplied message as an email header.

Mail service acceptance is reported, not guaranteed inbox delivery. No automatic
retry occurs after a timeout (delivery may be uncertain). Failed submissions keep
their text in the current form, not in durable storage. If email is not desired,
the alternative is a dedicated Supabase contact inbox with admin review; existing
best-effort feedback logging should not be presented as guaranteed delivery.

Limits: 5,000 characters and a 60-second successful-send cooldown per browser
session. This is not a global per-user abuse limit, durable queue, or deduplication
guarantee. Add those before opening anonymous or high-volume access.

The existing footballnews project's Brevo SMTP account is used for contact mail:
`smtp-relay.brevo.com`, port 587, STARTTLS. Its credentials remain outside Git.
The recipient setting accepts one address or a list; both configured addresses
are passed explicitly in the SMTP delivery envelope.

A local live test was accepted by Brevo for both recipients. Inbox delivery and
SMTP connectivity from the deployed host still need confirmation. Copy the
`[contact_email]` block from the local secrets file into the deployed app's
Streamlit secrets; local secrets are not deployed by Git. Preserve existing
authentication and database settings. Never paste credentials into chat or commit
them. Consider a separate Brevo SMTP key for MacroTool to allow independent rotation.
