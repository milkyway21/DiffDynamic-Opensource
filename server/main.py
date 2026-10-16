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
    from server.auth import auth_mode
    if auth_mode() == "off" and host not in ("127.0.0.1", "localhost", "::1"):
        raise RuntimeError("Local off-mode authentication requires a loopback bind")
    print(f"Starting DiffDynamic server on {host}:{port}")
    print(f"  Web UI:    http://{host}:{port}/")
    print(f"  API:       http://{host}:{port}/api/health")
    print(f"  Auth mode: {auth_mode()}")
    print(f"  Public prefix (DD_ROOT_PATH): {os.environ.get('DD_ROOT_PATH', '/') or '/'}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
