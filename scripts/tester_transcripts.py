"""Rebuild tester chat transcripts from Supabase chat_turns into one local HTML file.

Usage (from the repo root):
  .venv/bin/python -I scripts/tester_transcripts.py [--since 2026-10-13] [--exclude vincent,ash@fidantic.com] [--out file.html]

Reads .streamlit/secrets.toml (service key). Output is gitignored — it holds testers' conversations.
"""
import argparse, collections, html, json, tomllib, urllib.parse, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--since", default="2026-10-13")
ap.add_argument("--exclude", default="vincent,ash@fidantic.com")
ap.add_argument("--out", default="tester_transcripts.html")
a = ap.parse_args()

s = tomllib.load(open(".streamlit/secrets.toml", "rb"))
base, key = s["SUPABASE_URL"].rstrip("/"), s["SUPABASE_SERVICE_KEY"]
H = {"apikey": key, "Authorization": f"Bearer {key}"}


def fetch(table, params):
    rows, start = [], 0
    while True:
        q = urllib.parse.urlencode(params)
        req = urllib.request.Request(f"{base}/rest/v1/{table}?{q}", headers={**H, "Range": f"{start}-{start+999}"})
        r = json.load(urllib.request.urlopen(req))
        rows += r
        if len(r) < 1000:
            return rows
        start += 1000


excl = [x.strip().lower() for x in a.exclude.split(",") if x.strip()]
keep = lambda e: e and not any(x in e.lower() for x in excl)

turns = [r for r in fetch("chat_turns", {"select": "*", "created_at": f"gte.{a.since}", "order": "created_at"})
         if keep(r["user_email"])]
reacts = fetch("reactions", {"select": "*", "created_at": f"gte.{a.since}"})
errors = [r for r in fetch("app_errors", {"select": "*", "created_at": f"gte.{a.since}"}) if keep(r.get("user_email"))]

by_user = collections.defaultdict(lambda: collections.defaultdict(list))
for t in turns:
    by_user[t["user_email"]][t["chat_id"]].append(t)

e = html.escape
out = [f"""<!doctype html><meta charset=utf-8><title>Tester transcripts</title>
<style>
:root{{--bg:#f7f7f5;--fg:#1d1d1b;--mut:#6b6b66;--u:#e3ecfa;--a:#fff;--bd:#ddd;--err:#fbe4e4}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1b1b1a;--fg:#e8e8e3;--mut:#9a9a93;--u:#24324a;--a:#262624;--bd:#3a3a37;--err:#4a2626}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif;max-width:900px;margin:0 auto;padding:16px}}
h2{{margin-top:2.5em;border-bottom:1px solid var(--bd)}} h3{{color:var(--mut);font-size:14px;font-weight:600}}
.m{{border:1px solid var(--bd);border-radius:10px;padding:10px 14px;margin:8px 0;white-space:pre-wrap}}
.user{{background:var(--u);margin-left:15%}} .assistant{{background:var(--a);margin-right:5%}}
.meta{{font-size:12px;color:var(--mut);margin-bottom:4px}} .err{{background:var(--err)}}
details{{font-size:12px;margin-top:8px}} pre{{white-space:pre-wrap;font-size:12px;background:var(--bg);padding:8px;border-radius:6px}}
nav a{{display:inline-block;margin-right:14px}}
</style>
<h1>Tester transcripts</h1><p class=meta>Since {e(a.since)} · excluding {e(', '.join(excl))} · {len(turns)} rows</p><nav>"""]
order = sorted(by_user, key=lambda u: -sum(len(c) for c in by_user[u].values()))
out += [f'<a href="#{e(u)}">{e(u)}</a>' for u in order] + ["</nav>"]

react_by_target = collections.defaultdict(list)
for r in reacts:
    react_by_target[str(r.get("target_id") or r.get("target") or "")].append(r)

for u in order:
    out.append(f'<h2 id="{e(u)}">{e(u)}</h2>')
    for chat_id, rows in sorted(by_user[u].items(), key=lambda kv: kv[1][0]["created_at"]):
        rows.sort(key=lambda r: (r["created_at"], r.get("seq") or 0))
        pairs = sorted({r["pair"] for r in rows if r.get("pair")})
        out.append(f'<h3>{rows[0]["created_at"][:16].replace("T", " ")} UTC · {e(rows[0].get("surface") or "")}'
                   f'{" · " + e(", ".join(pairs)) if pairs else ""} · chat {e(chat_id[:8])}</h3>')
        for r in rows:
            out.append(f'<div class="m {e(r["role"])}"><div class=meta>{e(r["role"])} · {r["created_at"][11:19]}</div>{e(r["text"] or "")}')
            for tc in r.get("tool_trace") or []:
                out.append(f'<details><summary>tool: {e(tc.get("name",""))} {e(json.dumps(tc.get("args", {}))[:200])}</summary>'
                           f'<pre>{e(str(tc.get("result", "")))}</pre></details>')
            out.append("</div>")
    for er in [x for x in errors if x["user_email"] == u]:
        out.append(f'<div class="m err"><div class=meta>error · {er["created_at"][:19]} · {e(er.get("context") or "")}</div>'
                   f'{e(er.get("error_type") or "")}: {e((er.get("message") or "")[:400])}</div>')
    ur = [r for r in reacts if r.get("user_email") == u]
    if ur:
        out.append("<h3>Reactions</h3>" + "".join(
            f'<div class=m><pre>{e(json.dumps({k: v for k, v in r.items() if k not in ("id", "session_id", "user_email")}, indent=1))}</pre></div>'
            for r in ur))

open(a.out, "w").write("\n".join(out))
print(f"wrote {a.out}: {len(by_user)} users, {sum(len(c) for c in by_user.values())} chats")
