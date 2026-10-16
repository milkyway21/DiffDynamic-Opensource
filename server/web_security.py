"""Presentation and browser request boundaries for the web API."""

import json
import os
from urllib.parse import urlsplit

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from server.auth import auth_mode


PRIVATE_FIELDS = frozenset({
    "gpus", "gpu_id", "gpu_ids", "preferred_gpu", "default_gpu_id",
    "log_output", "log_tail", "log_lines", "config_snapshot", "sampling_yml",
    "patched_config", "python_bin", "conda_env", "diffdynamic_root",
    "data_root", "pretrained_models", "host", "port", "progress_detail",
    "log", "logs", "stdout", "stderr", "command", "system_info", "hardware",
    "hostname", "device", "devices", "cpu", "ram", "memory", "sampling_config",
})


def public_payload(value):
    """Keep business results while excluding execution diagnostics recursively."""
    if isinstance(value, list):
        return [public_payload(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in PRIVATE_FIELDS or key.startswith(("gpu_", "cpu_", "cuda_", "memory_", "peak_memory")):
            continue
        if key == "has_log":
            result[key] = False
        elif key == "meta" and isinstance(item, dict):
            safe = {name: value for name, value in item.items()
                    if name in ("method", "scaffold_mode", "num_samples", "num_generated",
                                "data_id", "num_molecules", "atom_mode", "source")}
            result[key] = public_payload(safe)
        elif key in ("error", "error_message") and item:
            result[key] = "Task failed. Contact an administrator with the task ID."
        else:
            result[key] = public_payload(item)
    return result


def _allowed_origins(request):
    explicit = os.environ.get("DD_ALLOWED_ORIGINS", "https://aidd.shu.edu.cn")
    origins = {value.strip().rstrip("/") for value in explicit.split(",") if value.strip()}
    # Local tests/development may use a loopback gateway. Never trust a
    # forwarded host to extend the origin allowlist.
    if request.url.hostname in ("127.0.0.1", "localhost", "::1", "testserver"):
        origins.add(f"{request.url.scheme}://{request.url.netloc}")
    return origins


class WebSecurityMiddleware(BaseHTTPMiddleware):
    """Apply headers even to authentication failures and restrict cookie writes."""

    async def dispatch(self, request, call_next):
        unsafe = request.method in ("POST", "PUT", "PATCH", "DELETE")
        if unsafe and request.url.path.startswith("/api/") and auth_mode() == "platform":
            origin = request.headers.get("origin")
            referer = request.headers.get("referer")
            if not origin and referer:
                parts = urlsplit(referer)
                origin = f"{parts.scheme}://{parts.netloc}"
            browser_cookie = "bianlab_token" in request.cookies
            if (origin and origin.rstrip("/") not in _allowed_origins(request)) or (
                browser_cookie and not origin
            ):
                response = JSONResponse({"detail": "Cross-origin request denied"}, status_code=403)
            else:
                response = await call_next(request)
        else:
            response = await call_next(request)

        if (
            request.url.path.startswith("/api/")
            and not getattr(request.state, "dd_is_admin", False)
            and response.headers.get("content-type", "").startswith("application/json")
            and response.status_code < 400
        ):
            body = b"".join([chunk async for chunk in response.body_iterator])
            headers = dict(response.headers)
            headers.pop("content-length", None)
            response = JSONResponse(public_payload(json.loads(body)),
                                    status_code=response.status_code, headers=headers,
                                    background=response.background)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = "frame-ancestors 'self'; object-src 'none'; base-uri 'self'"
        if request.url.path.startswith("/api/") or request.url.path in ("/", "/app"):
            response.headers["Cache-Control"] = "no-store"
        return response
