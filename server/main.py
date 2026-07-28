"""Entry point: FastAPI backend serving HTML/JS/CSS frontend.

Usage:
    conda run -n diffdynamic python -m server.main
"""

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

for var in ("all_proxy", "ALL_PROXY", "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(var, None)

import uvicorn
from server.api import app
from server.config import get_config


def main():
    cfg = get_config()
    host = os.environ.get("DD_HOST", cfg.host)
    port = int(os.environ.get("DD_PORT", str(cfg.port)))
    print(f"Starting DiffDynamic server on {host}:{port}")
    print(f"  API docs:  http://{host}:{port}/docs")
    print(f"  Web UI:    http://{host}:{port}/")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
