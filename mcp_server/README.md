# MacroTool in Claude Desktop (local MCP demo)

Adds one tool, `ask_macrotool`, to Claude Desktop. Claude passes your message to
MacroTool's own agent (same prompt, vocabulary, direction handling and Python-built
tables as the app) and shows its reply as returned. You keep chatting with Claude as
normal; MacroTool answers anything about FX option structuring.

Runs entirely on your machine. No hosting, no login. Not for production use.

Source checkpoint: integrated in v0.2.57 / f276056 on `agentic-workflow`, pushed
28 October 2026. All 13 MCP tests passed using the fake model, including stdio;
this does not verify live Claude Desktop/provider setup. The app's Auth0 gate does
not apply to this local-only process. Never expose it as a public service as-is.

## Quick start

```bash
git clone https://github.com/ash847/macrotool.git ~/macrotool-mcp
cd ~/macrotool-mcp && git checkout agentic-workflow
./install.sh
```

`install.sh` builds the environment, stores your own Anthropic key (hidden prompt), adds MacroTool
to Claude Desktop (while the app is closed), and self-checks. How to evaluate it:
`mcp_server/EVALUATION.md`. The manual steps below are what it automates.

## Set up manually (about 10 minutes)

> **Keep everything outside `~/Documents`, `~/Desktop` and `~/Downloads`** (for example
> `~/macrotool-mcp`, with the key in `~/.config/macrotool/secrets.toml`). Claude Desktop starts
> the server as a background program, and macOS silently refuses it access to those folders
> ("Operation not permitted" on `.venv/pyvenv.cfg` in the log). Granting Claude access does
> not help: macOS checks Python itself. Also edit `claude_desktop_config.json` only while
> Claude Desktop is closed, or the app can overwrite your change.

1. **Get the code** and check out the tested `agentic-workflow` branch (or the
   exact `f276056` checkpoint). Older installer comments reference the original
   demo branch; they do not change which checked-out code the installer uses.
2. **Create the environment** (Python 3.13) from the lockfile, then add the MCP package:
   ```bash
   uv sync --python 3.13
   uv pip install --python .venv/bin/python mcp
   ```
   Use `uv sync` rather than a plain `pip install -e .`: the project doesn't cap its
   dependency versions, and a fresh unpinned install pulls much newer `anthropic` and
   `streamlit` releases than the code was built against (a few tests fail on them). If you
   ever re-run `uv sync`, use `uv sync --inexact` so it keeps `mcp`.
3. **Give it an Anthropic key.** The inner agent calls the Anthropic API. Either put
   `ANTHROPIC_API_KEY = "..."` in `.streamlit/secrets.toml` (git-ignored), or point
   `MACROTOOL_SECRETS` at any secrets.toml that contains it (see the config below).
4. **Print the Claude Desktop config for this machine:**
   ```bash
   .venv/bin/python mcp_server/server.py --print-config
   ```
   Merge the `mcpServers` block into
   `~/Library/Application Support/Claude/claude_desktop_config.json`
   (Claude Desktop → Settings → Developer → Edit Config).
5. **Fully quit and reopen Claude Desktop.** A `macrotool` tool appears under the tools menu.

## Try it

- "I think USDBRL falls to 4.50 in three months. What should I trade?"
- then, in the same chat: "How does the 1x1.5 compare with a straight put?"
- "What does skew mean here?" · "What's the main risk?" · "What if the target were 4.20?"

The first answer takes 20–60 seconds (the agent makes several model calls). Claude may
ask you to approve the tool the first time.

## Notes

- Uses `mcp` 2.x (`MCPServer`). If a future release breaks the import, pin `mcp<3`.
- One conversation per server process; it resets when Claude Desktop restarts. Ask Claude
  to "start a new trade" (or say so) to reset it mid-session.
- Each question costs two model calls: Claude's, and MacroTool's agent.
- Server log: `~/Library/Logs/Claude/mcp-server-macrotool.log`.
- Tests: `.venv/bin/python -m pytest tests/test_mcp_server.py` (no API calls: the agent
  runs on a scripted model; the engine is real).
