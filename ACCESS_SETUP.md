# Magic-link beta admission

Before deploying v0.2.47, add top-level `allowed_emails = ["your@email.com"]`
in Streamlit secrets, before any TOML section headings. Include every admin and
tester. Missing, empty, or malformed lists deny everyone; admin status never
bypasses admission. Emails are matched case-insensitively with whitespace trimmed.

Keep `[auth]` and `[auth.auth0]` with the working callback URL and credentials.
Remove `[auth.google]` on deployment; Google is no longer offered. Existing Google
sessions must sign out and use magic link. Admission requires the configured Auth0
issuer, an email-passwordless subject, and a boolean verified-email claim.
Keep Auth0 email signups enabled so new people can verify identity and request
access. Keep Streamlit's outer sharing gate public if outsiders should reach this
request flow. Public hosting does not bypass the app admission check.

Access requests use the existing `[contact_email]` SMTP credentials. They go only
to Ash's Gmail by default, not the contact form's shared recipient list. Optionally
set `access_request_to_email` inside `[contact_email]` to change the recipient.
No visitor-supplied recipient or free-text email body is accepted. Requests do not
grant access or edit secrets. Approve manually by editing `allowed_emails`.

Duplicate protection is shared across sessions in one server process: one attempt
per verified email per 24 hours, and at most 20 attempts per hour globally. Failed
delivery attempts also count. State resets on process restart/cache clear and is
not shared across replicas; this is not a durable public-scale anti-abuse system.
Delivery errors are not exposed to users. SMTP acceptance does not guarantee inbox
delivery. Auth0 magic-link sending has separate provider limits.

Test before release: approved tester and admin, unapproved email and request,
missing/empty whitelist, unverified identity, old Google session, SMTP failure,
duplicate requests. Local MACROTOOL_DEV bypass remains development-only.
