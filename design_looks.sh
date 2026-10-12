#!/usr/bin/env bash
# Run the app locally for design work.
#   ./design_looks.sh start      -> the real app (the committed look) on :8501
#   ./design_looks.sh compare    -> :8501 real app, plus alternative looks from themes/*.toml
#                                   on :8502+ (navy_light, graphite_dark, editorial_light)
#   ./design_looks.sh stop | restart
#
# Local dev only: MACROTOOL_DEV=1 bypasses login (admin, dev@local) and this worktree has
# no Supabase secrets, so nothing here writes to production.
#
# What reloads when: Python / look.css edits hot-reload on save (runOnSave). Theme edits
# (.streamlit/config.toml or themes/*.toml) are read at server start — run `restart`.
# The alternative looks are previewed from temp config folders, so the committed
# .streamlit/config.toml is never touched by them.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
APP="$HERE/interface/app.py"
STREAMLIT="${STREAMLIT_BIN:-/Users/ash/Documents/Coding work/agentic-workflow/.venv/bin/streamlit}"
LOGDIR="${TMPDIR:-/tmp}/macrotool-design-logs"
PORTS=(8501 8502 8503 8504)
VARIANTS=(navy_light graphite_dark editorial_light)
mkdir -p "$LOGDIR"

stop() {
  for port in "${PORTS[@]}"; do
    for pid in $(lsof -ti "tcp:$port" -sTCP:LISTEN 2>/dev/null || true); do
      # Only touch servers running THIS worktree's app.
      if ps -o command= -p "$pid" | grep -qF "$APP"; then kill "$pid" && echo "stopped :$port"; fi
    done
  done
}

common_env() {
  export PYTHONPATH="$HERE" MACROTOOL_DEV=1 MACROTOOL_FORCE_ROLE=admin MACROTOOL_DEV_EMAIL=dev@local
}

launch_real() {
  local port="$1"
  ( cd "$HERE" && nohup "$STREAMLIT" run "$APP" --server.headless true --server.port "$port" \
      --server.runOnSave true --browser.gatherUsageStats false > "$LOGDIR/real.log" 2>&1 & )
  echo "started :$port  (real app — committed look)"
}

launch_variant() {
  local port="$1" theme="$2" dir="$LOGDIR/cfg_$2"
  mkdir -p "$dir/.streamlit"
  { printf '[client]\ntoolbarMode = "minimal"\n\n'; cat "$HERE/themes/${theme}.toml"; } > "$dir/.streamlit/config.toml"
  # Empty MACROTOOL_LOOK_CSS switches the spacing/card CSS layer off for these previews.
  ( cd "$dir" && MACROTOOL_LOOK_CSS="" nohup "$STREAMLIT" run "$APP" --server.headless true \
      --server.port "$port" --server.runOnSave true --browser.gatherUsageStats false \
      > "$LOGDIR/${theme}.log" 2>&1 & )
  echo "started :$port  ($theme)"
}

start() {
  common_env
  launch_real 8501
}

compare() {
  common_env
  launch_real 8501
  local port=8502
  for theme in "${VARIANTS[@]}"; do launch_variant "$port" "$theme"; port=$((port + 1)); done
}

case "${1:-start}" in
  start)   start ;;
  compare) compare ;;
  stop)    stop ;;
  restart) stop; sleep 2; start ;;
  *) echo "usage: $0 {start|compare|stop|restart}"; exit 1 ;;
esac
