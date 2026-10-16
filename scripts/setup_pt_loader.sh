#!/usr/bin/env bash
# Prepare the separate CPU PT loader; never changes the scientific conda env.
set -euo pipefail
task_project_root="$(cd "$(dirname "$0")/.." && pwd)"
task_loader_env="$task_project_root/.cache/dd_pt_sandbox"
task_loader_python="${DD_PT_SETUP_PYTHON:-/usr/bin/python3.12}"
command -v bwrap >/dev/null || { echo "bubblewrap is required" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
uv venv --python "$task_loader_python" "$task_loader_env"
uv pip install --python "$task_loader_env/bin/python" \
    --index-url https://download.pytorch.org/whl/cpu \
    -r "$task_project_root/server/pt-loader-requirements.txt"
echo "Secure CPU loader prepared at $task_loader_env"
