#!/usr/bin/env bash
# One-command setup of MacroTool for Claude Desktop (macOS).
#
#   git clone https://github.com/ash847/macrotool.git ~/macrotool-mcp
#   cd ~/macrotool-mcp && git checkout mcp-demo
#   ./install.sh
#
# Does: build the Python environment from the lockfile, store your Anthropic API key
# (readable only by you), add MacroTool to Claude Desktop's config, and run a self-check.
# Re-running is safe.
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$PWD"
SECRETS="${MACROTOOL_SECRETS:-$HOME/.config/macrotool/secrets.toml}"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
die() { printf '\n\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

[ "$(uname)" = "Darwin" ] || die "This installer is for macOS (Claude Desktop)."

# macOS silently blocks the background server from these folders.
case "$REPO" in
  "$HOME/Documents"*|"$HOME/Desktop"*|"$HOME/Downloads"*)
    die "Move this folder out of Documents/Desktop/Downloads first (for example to ~/macrotool-mcp).
macOS blocks Claude Desktop's background programs from reading those folders, so MacroTool would not start." ;;
esac

command -v uv >/dev/null 2>&1 || die "uv is not installed. Install it with:  brew install uv   (or see https://docs.astral.sh/uv/)"

say "1/4  Building the Python environment (about a minute)..."
uv sync --frozen --python 3.13 >/dev/null
uv pip install --python .venv/bin/python mcp >/dev/null
echo "done."

say "2/4  Anthropic API key"
if [ -f "$SECRETS" ] && grep -q '^ANTHROPIC_API_KEY' "$SECRETS"; then
  echo "Using the key already stored in $SECRETS"
else
  echo "MacroTool needs your own Anthropic API key (create one at https://console.anthropic.com)."
  echo "Typing is hidden. It is stored only in $SECRETS (readable by you alone)."
  read -r -s -p "API key: " KEY; echo
  case "$KEY" in sk-ant-*) ;; *) die "That doesn't look like an Anthropic key (should start with sk-ant-)." ;; esac
  mkdir -p "$(dirname "$SECRETS")"; chmod 700 "$(dirname "$SECRETS")"
  ( umask 077; printf 'ANTHROPIC_API_KEY = "%s"\n' "$KEY" > "$SECRETS" )
  chmod 600 "$SECRETS"
  unset KEY
fi

say "3/4  Self-check (starts the server and lists its tool; no API call)"
.venv/bin/python - "$SECRETS" <<'EOF'
import asyncio, sys
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
secrets = sys.argv[1]
async def main():
    p = StdioServerParameters(command=sys.executable, args=["mcp_server/server.py"],
                              env={"MACROTOOL_SECRETS": secrets, "PATH": "/usr/bin:/bin"})
    async with stdio_client(p) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            names = [t.name for t in (await s.list_tools()).tools]
            assert names == ["ask_macrotool"], names
            print("server starts and exposes:", names[0])
asyncio.run(asyncio.wait_for(main(), 90))
EOF

say "4/4  Adding MacroTool to Claude Desktop"
.venv/bin/python mcp_server/install_config.py --secrets "$SECRETS" || {
  echo
  echo "Quit Claude Desktop fully (Cmd+Q) and run ./install.sh again to finish this step."
  exit 1
}

say "Done. Next:"
echo "  1. Quit Claude Desktop fully (Cmd+Q) and reopen it."
echo "  2. Settings -> Developer: 'macrotool' should show as running."
echo "  3. In a new chat try:  I think USDBRL falls to 4.50 in three months. What should I trade?"
echo "  Evaluation guide: mcp_server/EVALUATION.md"
