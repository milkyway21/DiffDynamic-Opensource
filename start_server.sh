#!/bin/bash
# Start DiffDynamic Web Server
# Usage: bash start_server.sh [--port 7860] [--gpus "0,1,2,3"]

set -e
cd "$(dirname "$0")"

# Unset proxy to avoid httpx/Gradio conflicts
unset all_proxy ALL_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

PORT="${DD_PORT:-7860}"
HOST="${DD_HOST:-0.0.0.0}"

echo "=== DiffDynamic Web Server ==="
echo "Host: $HOST"
echo "Port: $PORT"
echo ""

exec conda run --no-capture-output -n diffdynamic python3 -m server.main
