"""Add (or refresh) the `macrotool` entry in Claude Desktop's claude_desktop_config.json.

Run by install.sh with the project's own Python. Safe by design:
  * refuses while Claude Desktop is running (the app rewrites this file from memory and
    would silently undo the change),
  * keeps every other setting, makes a timestamped backup first, and writes atomically.

    python mcp_server/install_config.py --secrets ~/.config/macrotool/secrets.toml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path("~/Library/Application Support/Claude/claude_desktop_config.json").expanduser()


def claude_desktop_running() -> bool:
    out = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True).stdout
    return any(line.strip() == "/Applications/Claude.app/Contents/MacOS/Claude" for line in out.splitlines())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.environ.get("CLAUDE_CONFIG") or str(DEFAULT_CONFIG))
    ap.add_argument("--secrets", required=True, help="path to the secrets.toml holding ANTHROPIC_API_KEY")
    ap.add_argument("--allow-running", action="store_true", help="skip the Claude-is-running check (tests only)")
    args = ap.parse_args()

    if not args.allow_running and claude_desktop_running():
        print("Claude Desktop is still running. Quit it fully (Cmd+Q), then run this again.\n"
              "(While it runs, the app can overwrite the change.)", file=sys.stderr)
        return 2

    cfg_path = Path(args.config).expanduser()
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text() or "{}")
        except json.JSONDecodeError as e:
            print(f"{cfg_path} isn't valid JSON ({e}). Not touching it.", file=sys.stderr)
            return 3
    else:
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = {}

    entry = {
        "command": str(REPO / ".venv" / "bin" / "python"),
        "args": [str(REPO / "mcp_server" / "server.py")],
        "env": {"MACROTOOL_SECRETS": str(Path(args.secrets).expanduser().resolve())},
    }
    if cfg.get("mcpServers", {}).get("macrotool") == entry:
        print("Claude Desktop is already set up for MacroTool.")
        return 0

    if cfg_path.exists():
        backup = cfg_path.with_name(cfg_path.name + time.strftime(".bak-%Y%m%d-%H%M%S"))
        shutil.copy2(cfg_path, backup)
        print(f"Backed up your config to {backup.name}")
    cfg.setdefault("mcpServers", {})["macrotool"] = entry
    tmp = cfg_path.with_name(cfg_path.name + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=2))
    os.replace(tmp, cfg_path)
    print("Added MacroTool to Claude Desktop.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
