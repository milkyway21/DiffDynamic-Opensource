#!/bin/bash
# Start DiffDynamic Web Server (standalone UI + API)
# Usage:
#   bash start_server.sh
#   DD_AUTH_MODE=platform DD_ROOT_PATH=/zh/diffdynamic bash start_server.sh
#
# Production nginx (on aidd.shu.edu.cn), strip prefix so this process still sees /:
#   location /zh/diffdynamic/ {
#     proxy_pass http://127.0.0.1:7860/;
#     proxy_set_header Host $host;
#     proxy_set_header Cookie $http_cookie;
#     proxy_set_header X-Forwarded-Prefix /zh/diffdynamic;
#   }
#
# Env:
#   DD_HOST (default 127.0.0.1)  DD_PORT (default 7860)
#   DD_AUTH_MODE=off|platform
#   DD_ROOT_PATH=/zh/diffdynamic
#   DD_PLATFORM_SESSION_URL  DD_PLATFORM_LOGIN_URL

set -euo pipefail
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

# ── Unset proxy to avoid httpx/FastAPI conflicts ─────────────────────
unset all_proxy ALL_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

# ── Defaults ──────────────────────────────────────────────────────────
PORT="${DD_PORT:-7860}"
HOST="${DD_HOST:-127.0.0.1}"
CONDA_ENV="${DD_CONDA_ENV:-diffdynamic}"
PYTHON_BIN=""

# ── Parse args ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --env) CONDA_ENV="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | head -10
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── Resolve Python (prefer conda env) ────────────────────────────────
for cand in \
  "$HOME/anaconda3/envs/$CONDA_ENV/bin/python3" \
  "$HOME/miniconda3/envs/$CONDA_ENV/bin/python3" \
  "/opt/conda/envs/$CONDA_ENV/bin/python3"; do
  if [[ -x "$cand" ]]; then PYTHON_BIN="$cand"; break; fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v conda &>/dev/null; then
    PYTHON_BIN="$(conda run -n "$CONDA_ENV" which python3 2>/dev/null || true)"
  fi
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "[WARN] Conda env '$CONDA_ENV' not found, falling back to system python3" >&2
  PYTHON_BIN="python3"
fi

echo "=== DiffDynamic Web Server ==="
echo "Project : $SCRIPT_DIR"
echo "Host    : $HOST"
echo "Port    : $PORT"
echo "Python  : $PYTHON_BIN"
echo "CondEnv : $CONDA_ENV"
echo ""
# NOTE: must stay single-process — the in-memory JobScheduler, queue caps
# and rate limiter would diverge across uvicorn workers. Scale via
# max_concurrent_jobs / gpu_ids in server_config.json instead.

# ── Preflight checks ─────────────────────────────────────────────────
if [[ ! -f "server/main.py" ]]; then
  echo "[ERROR] server/main.py not found in $SCRIPT_DIR" >&2
  exit 1
fi

# Ensure uploads/outputs directories exist
mkdir -p uploads outputs data

# ── Protect the run database (never delete/recreate) ─────────────────
DB_FILE="${DD_DB_PATH:-$SCRIPT_DIR/diffdynamic.db}"
if [[ -f "$DB_FILE" ]]; then
  if [[ ! -w "$DB_FILE" ]]; then
    echo "[ERROR] Database file $DB_FILE is not writable. Check permissions." >&2
    exit 1
  fi
  echo "[INFO] Database file present: $DB_FILE $(du -h "$DB_FILE" | cut -f1)"
else
  echo "[INFO] No database found yet — will be created on first run."
fi

# ── Check for stale uvicorn/SQLite locks ─────────────────────────────
STALE_LOCK="${DB_FILE}-journal"
if [[ -f "$STALE_LOCK" && ! -s "$STALE_LOCK" ]]; then
  echo "[WARN] Empty -journal is normal after successful close (recovered via WAL/pragma)."
fi

# ── Check port availability ──────────────────────────────────────────
if command -v ss &>/dev/null; then
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$PORT$"; then
    echo "[WARN] Port $PORT already in use. Set DD_PORT or pass --port." >&2
  fi
fi

# ── Temp-file cleanup (docktmp + intermediate .pt only) ───────────────
# Removes rebuildable scratch older than DD_CLEANUP_AGE_DAYS (default 7).
# Disable with DD_CLEANUP_ON_START=0.
if [[ "${DD_CLEANUP_ON_START:-1}" == "1" && -f "server/cleanup.py" ]]; then
  echo "[INFO] Cleaning stale temp files (docktmp / intermediate .pt)..."
  "$PYTHON_BIN" -m server.cleanup \
    --age-days "${DD_CLEANUP_AGE_DAYS:-7}" \
    || echo "[WARN] Cleanup reported issues (continuing)" 2>&1
fi

# ── Export settings ──────────────────────────────────────────────────
export DD_HOST="$HOST"
export DD_PORT="$PORT"
export DD_AUTH_MODE="${DD_AUTH_MODE:-platform}"
# Extra administrators must be configured explicitly in DD_ADMIN_USERS.
# No account gains administrator rights merely by using a particular name.

# ── Start the server ─────────────────────────────────────────────────
exec "$PYTHON_BIN" -m server.main
