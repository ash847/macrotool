"""Side-by-side: the same question answered by plain Claude (no tools) and by MacroTool's agent.

For each idea in compare_ideas.json this asks
  * plain Claude  - the same model as the agent, no system prompt, no tools (what a normal
                    Claude chat does with the question), and
  * MacroTool     - a fresh MacroTool agent conversation, exactly what ask_macrotool returns,
and writes one HTML page with the two answers next to each other.

Uses the real Anthropic API (a few cents per idea). Run:
    .venv/bin/python mcp_server/compare.py [--ideas path.json] [--out report.html]
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mcp_server import server as S  # noqa: E402
from agentic.agent_llm import DEFAULT_MODEL  # noqa: E402


def ask_plain(question: str) -> tuple[str, float]:
    import anthropic
    client = anthropic.Anthropic(api_key=S._secret("ANTHROPIC_API_KEY"))
    t0 = time.time()
    msg = client.messages.create(
        model=S._secret("ANTHROPIC_MODEL") or DEFAULT_MODEL, max_tokens=4000,
        messages=[{"role": "user", "content": question}])
    return "".join(b.text for b in msg.content if b.type == "text"), time.time() - t0


def ask_macrotool(question: str) -> tuple[str, float]:
    t0 = time.time()
    return S._new_flow().advance(question), time.time() - t0


def _safe(fn, q):
    try:
        return fn(q)
    except Exception as e:                      # one failure must not lose the others
        return f"(failed: {type(e).__name__}: {e})", 0.0


PAGE = """<!doctype html><meta charset="utf-8"><title>MacroTool vs plain Claude</title>
<style>
 body{{font:15px/1.5 -apple-system,system-ui,sans-serif;margin:0;background:#f6f4ef;color:#222}}
 header{{padding:20px 28px}} h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#666}}
 section{{margin:0 28px 28px;background:#fff;border-radius:12px;padding:16px 20px;box-shadow:0 1px 3px #0001}}
 .q{{font-weight:600;font-size:16px;margin-bottom:10px}}
 .cols{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
 .col h3{{margin:0 0 6px;font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:#666}}
 .col.mt h3{{color:#0a6b4d}} .body{{overflow-x:auto}}
 table{{border-collapse:collapse;font-size:13px;margin:8px 0}} td,th{{border:1px solid #ddd;padding:3px 8px}}
 pre.raw{{white-space:pre-wrap}}
 @media(max-width:900px){{.cols{{grid-template-columns:1fr}}}}
</style>
<header><h1>MacroTool vs plain Claude</h1>
<div class="sub">{when} &middot; model {model} &middot; plain Claude = same model, no tools, no instructions</div></header>
{sections}
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script>if(window.marked){{document.querySelectorAll('pre.raw').forEach(p=>{{const d=document.createElement('div');
d.innerHTML=marked.parse(p.textContent);p.replaceWith(d)}})}}</script>
"""

SECTION = """<section><div class="q">{n}. {q}</div><div class="cols">
<div class="col"><h3>Plain Claude ({ps:.0f}s, {pc} chars)</h3><div class="body"><pre class="raw">{plain}</pre></div></div>
<div class="col mt"><h3>MacroTool ({ms:.0f}s, {mc} chars)</h3><div class="body"><pre class="raw">{mt}</pre></div></div>
</div></section>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ideas", default=str(HERE / "compare_ideas.json"))
    ap.add_argument("--out", default=str(HERE / "reports" / f"compare_{datetime.now():%Y%m%d_%H%M}.html"))
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    ideas = json.loads(Path(args.ideas).read_text())["ideas"]
    with ThreadPoolExecutor(max_workers=2 * len(ideas)) as pool:
        plain = [pool.submit(_safe, ask_plain, q) for q in ideas]
        mt = [pool.submit(_safe, ask_macrotool, q) for q in ideas]
        plain, mt = [f.result() for f in plain], [f.result() for f in mt]

    sections = "".join(
        SECTION.format(n=i, q=html.escape(q), plain=html.escape(p[0]), mt=html.escape(m[0]),
                       ps=p[1], ms=m[1], pc=len(p[0]), mc=len(m[0]))
        for i, (q, p, m) in enumerate(zip(ideas, plain, mt), 1))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PAGE.format(when=f"{datetime.now():%Y-%m-%d %H:%M}",
                               model=S._secret("ANTHROPIC_MODEL") or DEFAULT_MODEL, sections=sections))
    print(f"wrote {out}")
    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
