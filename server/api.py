"""FastAPI REST endpoints for DiffDynamic web service.

Standalone paper-demo UI + API. Public access is intended to go through
nginx on https://aidd.shu.edu.cn (e.g. /zh/diffdynamic/) with BianLab
session cookies. This process should bind to 127.0.0.1.

Auth:
- DD_AUTH_MODE=platform (default): require a valid BianLab /api/auth/session
- DD_AUTH_MODE=off (explicit local lab): require internal cookie/key on /api/*
"""

# Dependencies below are also live injection points for the functional routers.
# ruff: noqa: F401

import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Dict, Optional, List, Literal, Tuple
from urllib.parse import unquote

import yaml
from fastapi import Depends, FastAPI, HTTPException, Query, Request, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from server.schemas import (
    BatchGenerateRequest,
    CustomGenerateRequest,
    DefaultSamplingRequest,
    EvaluateRequest,
    ExtractRequest,
    GenerateRequest,
    GenerationContextRequest,
    OptimizationRequest,
    PocketEvalRequest,
    RerunRequest,
    ScaffoldCascadeRequest,
    ScaffoldPlacement,
    ScaffoldPreviewRequest,
    ScaffoldRegion,
    ScaffoldRequest,
)
from starlette.middleware.base import BaseHTTPMiddleware

from server.config import get_config, save_config, ServerConfig
from server.database import (
    init_db, create_run, update_run, get_run, list_runs, run_to_dict,
    add_molecules, query_molecules, add_evaluation, get_evaluations,
    log_history, get_history, compare_runs, get_dashboard_stats, count_molecules,
    reconcile_stale_runs, count_runs_today,
)
from server.quota import seconds_until_reset, shanghai_day_bounds
from server.jobs import JobSubmitError, get_scheduler
from server.routes import register_routes
from server.web_security import WebSecurityMiddleware
from server.auth import (
    auth_mode,
    public_root,
    request_public_root,
    platform_login_url,
    platform_login_url_for_root,
    platform_home_url,
    platform_home_url_for_root,
    lookup_platform_identity,
    is_admin_user,
    normalize_owner,
    owner_slug,
    user_upload_root,
)
from server.data_manager import (
    get_config_snapshot, write_config, scan_outputs, scan_eval_dirs,
    load_pt_metadata, get_gpu_info, list_proteins, browse_directory,
)

app = FastAPI(
    title="DiffDynamic",
    description="Paper demo web service for DiffDynamic",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


# ── Security helpers ──────────────────────────────────────────────────────

_INTERNAL_KEY_FILE_ENV = "DD_INTERNAL_API_KEY"
_INTERNAL_KEY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".internal_api_key")
_COOKIE_NAME = "dd_internal"


def _keys_match(provided: str, expected: str) -> bool:
    """Constant-time compare that is safe on Python 3.8 (length mismatch raises)."""
    if not expected:
        return False
    a = (provided or "").encode("utf-8")
    b = expected.encode("utf-8")
    if len(a) != len(b):
        hmac.compare_digest(b, b)
        return False
    return hmac.compare_digest(a, b)


def _ensure_internal_api_key() -> str:
    env_key = (os.environ.get(_INTERNAL_KEY_FILE_ENV) or "").strip()
    if env_key:
        return env_key
    if os.path.isfile(_INTERNAL_KEY_PATH):
        with open(_INTERNAL_KEY_PATH, encoding="utf-8") as f:
            stored = f.read().strip()
        if stored:
            return stored
    key = secrets.token_urlsafe(32)
    fd = os.open(_INTERNAL_KEY_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(key)
    return key


def _get_internal_api_key() -> str:
    return _ensure_internal_api_key()


def _internal_key_matched(request: Request, *, allow_cookie: bool) -> bool:
    """Whether the request presents the internal API key.

    In platform mode the ``dd_internal`` cookie is NOT accepted: historical
    off-mode deployments handed that cookie to every visitor, so trusting it
    there would let a stale cookie + client-supplied ``X-DD-User`` header
    bypass the platform session check entirely. Direct loopback tooling must
    explicitly enable DD_ALLOW_INTERNAL_API before using the key header.
    """
    if auth_mode() == "platform":
        # A public request must always resolve its platform session. Internal
        # scripts opt in explicitly and connect directly over loopback.
        peer = request.client.host if request.client else ""
        if os.environ.get("DD_ALLOW_INTERNAL_API") != "1" or peer not in ("127.0.0.1", "::1"):
            return False
        if request.headers.get("x-forwarded-prefix") or request.headers.get("x-forwarded-for"):
            return False
    key = _ensure_internal_api_key()
    provided = request.headers.get("X-DD-Internal-Key", "")
    if not provided and allow_cookie:
        provided = request.cookies.get(_COOKIE_NAME, "")
    return bool(provided) and _keys_match(provided, key)


def verify_internal_key(request: Request):
    """Keep as a dependency for routes; middleware already authenticated."""
    if getattr(request.state, "dd_user", None):
        return
    if _internal_key_matched(request, allow_cookie=(auth_mode() != "platform")):
        return
    raise HTTPException(status_code=401, detail="Unauthorized")


def _request_username(request: Request) -> str:
    user = getattr(request.state, "dd_user", None)
    if user:
        return str(user)[:64]
    if auth_mode() == "platform":
        return "platform_user"
    return request.headers.get("X-DD-User", "web_ui")[:64]


def _request_is_admin(request: Request) -> bool:
    return bool(getattr(request.state, "dd_is_admin", False))


def require_admin(request: Request):
    if not _request_is_admin(request):
        raise HTTPException(status_code=403, detail="Administrator access required")


def _selected_gpu(value, request):
    if value is not None:
        require_admin(request)
    return value


def _owner_filter(request: Request) -> Optional[str]:
    """None for admin (no filter); otherwise current owner's canonical name."""
    if _request_is_admin(request):
        return None
    if normalize_owner(_request_username(request)) == "legacy":
        raise HTTPException(status_code=403, detail="Legacy records require administrator access")
    return normalize_owner(_request_username(request))


def _set_identity(request: Request, username: str, is_superuser: bool = False) -> None:
    request.state.dd_user = normalize_owner(username)
    request.state.dd_is_admin = is_admin_user(request.state.dd_user, is_superuser)


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        root = request_public_root(request.headers)
        request.state.dd_public_root = root
        cookie_path = root or "/"
        mode = auth_mode()
        request.state.dd_user = None
        request.state.dd_is_admin = False

        key_matched = _internal_key_matched(request, allow_cookie=(mode != "platform"))
        if key_matched:
            user = request.headers.get("X-DD-User", "internal")[:64]
            _set_identity(request, user, is_superuser=False)

        if path.startswith("/api/") and path not in ("/api/health",):
            if not getattr(request.state, "dd_user", None):
                if mode == "platform":
                    try:
                        identity = lookup_platform_identity(
                            cookie_header=request.headers.get("cookie", ""),
                            authorization=request.headers.get("authorization", ""),
                        )
                    except Exception:
                        identity = None
                    if not identity:
                        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
                    _set_identity(request, identity.username, identity.is_superuser)
                else:
                    return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

        if path in ("/", "/app") and mode == "platform" and not getattr(request.state, "dd_user", None):
            try:
                identity = lookup_platform_identity(
                    cookie_header=request.headers.get("cookie", ""),
                    authorization=request.headers.get("authorization", ""),
                )
            except Exception:
                identity = None
            if not identity:
                login = platform_login_url_for_root(root)
                here = (root + path) if root else (str(request.url.path) or "/app")
                sep = "&" if "?" in login else "?"
                return RedirectResponse(
                    url=f"{login}{sep}redirect={here}",
                    status_code=302,
                )
            _set_identity(request, identity.username, identity.is_superuser)

        response = await call_next(request)
        if path in ("/", "/app") and mode != "platform":
            response.set_cookie(
                _COOKIE_NAME,
                _ensure_internal_api_key(),
                httponly=True,
                samesite="lax",
                path=cookie_path,
                max_age=86400,
            )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Referrer-Policy"] = "same-origin"
        return response


app.add_middleware(AuthMiddleware)
app.add_middleware(WebSecurityMiddleware)


# ── Rate limiting & submission caps ───────────────────────────────────────

class _SlidingWindowLimiter:
    """Per-user in-memory sliding window (single-process uvicorn only)."""

    def __init__(self):
        self._hits: Dict[str, deque] = {}
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window: float = 60.0) -> None:
        if limit <= 0:
            return
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] > window:
                hits.popleft()
            if len(hits) >= limit:
                retry = max(1, int(window - (now - hits[0])) + 1)
                raise HTTPException(
                    status_code=429,
                    detail=f"Too many requests, retry in {retry}s",
                    headers={"Retry-After": str(retry)},
                )
            hits.append(now)


_rate_limiter = _SlidingWindowLimiter()


def _rate_limit(kind: str):
    def dependency(request: Request):
        cfg = get_config()
        limit = {"submit": cfg.max_submit_per_min, "upload": cfg.max_upload_per_min}[kind]
        _rate_limiter.check(normalize_owner(_request_username(request)), limit)
        return None
    return dependency


_rate_limit_submit = _rate_limit("submit")
_rate_limit_upload = _rate_limit("upload")


@app.exception_handler(JobSubmitError)
async def _job_submit_error_handler(request: Request, exc: JobSubmitError):
    retry = str(int(getattr(exc, "retry_after", 60) or 60))
    return JSONResponse(
        status_code=getattr(exc, "status_code", 429),
        content={"detail": str(exc)},
        headers={"Retry-After": retry},
    )


def _safe_resolve(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return os.path.realpath(path)


def _is_within(path: str, root: str) -> bool:
    """True if path is root itself or inside root (no prefix bypass)."""
    try:
        return os.path.commonpath([os.path.realpath(path), os.path.realpath(root)]) == os.path.realpath(root)
    except ValueError:
        return False


def _top_level_symlink_roots(base: str) -> list:
    """Real directories reached through first-level symlinks under ``base``.

    The test-set pocket directory (data/crossdocked_pocket10_test_only) is a
    symlink to a tree outside data_root; realpath-based jails would otherwise
    reject every file the file browser picks from it.
    """
    roots = []
    try:
        for name in os.listdir(base):
            p = os.path.join(base, name)
            if os.path.islink(p):
                real = os.path.realpath(p)
                if os.path.isdir(real):
                    roots.append(real)
    except OSError:
        pass
    return roots


def _shared_data_roots() -> list:
    """data_root plus the targets of its top-level symlinked directories."""
    cfg = get_config()
    roots = [cfg.data_root]
    if cfg.data_root and os.path.isdir(cfg.data_root):
        roots.extend(_top_level_symlink_roots(cfg.data_root))
    return [os.path.realpath(r) for r in roots if r]


def allowed_roots() -> list:
    cfg = get_config()
    project_outputs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    sdf_store = os.path.join(project_outputs, "sdf_store")
    roots = [
        cfg.output_root,
        cfg.data_root,
        cfg.upload_dir,
        sdf_store,
        project_outputs,
    ]
    roots.extend(_top_level_symlink_roots(cfg.data_root) if cfg.data_root else [])
    return [os.path.realpath(r) for r in roots if r]


def _decode_user_path(path: Optional[str]) -> Optional[str]:
    """Unquote paths that arrived as %2Fdata%2F... through nginx."""
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    decoded = unquote(text)
    if "%" in decoded:
        decoded = unquote(decoded)
    return decoded or None


def validate_path(path: Optional[str], *, must_exist: bool = True, field: str = "path") -> Optional[str]:
    """Validate a user-supplied path lies inside allowed roots. Raises 403 otherwise."""
    path = _decode_user_path(path)
    if not path:
        return None
    abs_path = os.path.realpath(path)
    if not any(_is_within(abs_path, r) for r in allowed_roots()):
        raise HTTPException(status_code=403, detail=f"Path outside allowed directories: {field}")
    if must_exist and not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail=f"Path not found: {field}")
    return abs_path


def assert_path_acl(abs_path: Optional[str], request: Request, *, field: str = "path") -> Optional[str]:
    """Non-admin may only touch their users/{owner} tree (plus shared data/configs).

    Shared data/configs roots stay readable for pocket/preset files, but `.pt`
    under them is denied: a non-admin may only read .pt checkpoints that they
    generated (outputs/users/{owner}) or uploaded (uploads/users/{owner})."""
    if not abs_path:
        return None
    if _request_is_admin(request):
        return abs_path
    cfg = get_config()
    owner = normalize_owner(_request_username(request))
    slug_root_out = os.path.join(cfg.output_root, "users", owner_slug(owner))
    slug_root_up = user_upload_root(cfg.upload_dir, owner)
    configs_root = os.path.join(cfg.diffdynamic_root, "configs")
    if _is_within(abs_path, configs_root):
        raise HTTPException(status_code=403, detail="Administrator access required")
    if os.path.splitext(abs_path)[1].lower() in (".log", ".out", ".err", ".txt", ".json", ".yml", ".yaml"):
        raise HTTPException(status_code=403, detail="Execution diagnostics are restricted")
    shared = _shared_data_roots()
    if any(_is_within(abs_path, r) for r in shared if r):
        if abs_path.lower().endswith(".pt"):
            raise HTTPException(status_code=403, detail=f"Access denied for path: {field}")
        return abs_path
    if _is_within(abs_path, slug_root_out) or _is_within(abs_path, slug_root_up):
        return abs_path
    raise HTTPException(status_code=403, detail=f"Access denied for path: {field}")


def validate_path_for_user(
    path: Optional[str],
    request: Request,
    *,
    must_exist: bool = True,
    field: str = "path",
) -> Optional[str]:
    abs_path = validate_path(path, must_exist=must_exist, field=field)
    return assert_path_acl(abs_path, request, field=field)


def validate_config_path(path: Optional[str]) -> Optional[str]:
    """Config files must live under configs/ or uploads/."""
    if not path:
        return None
    cfg = get_config()
    abs_path = os.path.realpath(path)
    allowed = [os.path.join(cfg.diffdynamic_root, "configs"), cfg.upload_dir]
    if not any(_is_within(abs_path, r) for r in allowed):
        raise HTTPException(status_code=403, detail="Config path outside allowed directories")
    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="Config file not found")
    return abs_path


def _require_run(run_id: int, request: Request):
    r = get_run(run_id, owner=_owner_filter(request))
    if r is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return r


def _require_job(job_id: str, request: Request) -> dict:
    scheduler = get_scheduler()
    status = scheduler.get_status(job_id, owner=_owner_filter(request))
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status

# Config fields that may be updated at runtime via PUT /api/config
CONFIG_MUTABLE_FIELDS = {
    "max_concurrent_jobs", "job_timeout",
    "max_jobs_per_user", "max_pending_jobs",
    "max_submit_per_min", "max_upload_per_min", "max_upload_bytes",
    "max_generate_per_day", "max_eval_per_day",
    "gpu_ids", "default_gpu_id",
}


@app.on_event("startup")
def startup():
    key = _ensure_internal_api_key()
    print(f"[startup] internal API key ready ({len(key)} chars, file={_INTERNAL_KEY_PATH})")
    init_db()
    n = reconcile_stale_runs(reason="server_restart")
    if n:
        print(f"[startup] reconciled {n} stale run(s) -> failed")
    try:
        from server.molecule_ingest import repair_molecule_catalog
        repaired = repair_molecule_catalog()
        if repaired:
            print(f"[startup] molecule catalog repair: {repaired}")
    except Exception as exc:
        print(f"[startup] molecule catalog repair skipped: {exc}")


# ── Generation ───────────────────────────────────────────────────────────


def _validated_scaffold_params(req, request):
    params = req.dict()
    for field in ('protein_path', 'ligand_path', 'molecule_path', 'adaptive_gates_actives'):
        if params.get(field):
            params[field] = validate_path_for_user(params[field], request, field=field)
    params['config_path'] = validate_config_path(req.config_path)
    if params['protein_path']:
        params['ligand_path'] = params['molecule_path'] or params['ligand_path']
    return params


VALID_GENERATE_MODES = {"dynamic", "prudent", "dynamic.prudent"}
VALID_VINA_MODES = {"auto", "score_only", "dock", "minimize", "all", "none", "off"}


def _clamp(value, lo, hi, default):
    if value is None:
        return default
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


# ── Runs ─────────────────────────────────────────────────────────────────


_EXPORT_COLUMNS = (
    ("id", lambda m: m.id),
    ("smiles", lambda m: m.smiles),
    ("pocket_id", lambda m: m.pocket_id),
    ("molecule_index", lambda m: m.molecule_index),
    ("vina_score", lambda m: m.vina_score),
    ("comprehensive_score", lambda m: m.comprehensive_score),
    ("qed", lambda m: m.qed),
    ("sa", lambda m: m.sa),
    ("logp", lambda m: m.logp),
    ("tpsa", lambda m: m.tpsa),
    ("lipinski_pass", lambda m: m.lipinski_pass),
    ("pains_pass", lambda m: m.pains_pass),
    ("lilly_passed", lambda m: m.lilly_passed),
    ("rdkit_valid", lambda m: m.rdkit_valid),
    ("molecule_stable", lambda m: m.molecule_stable),
    ("n_heavy_atoms", lambda m: m.n_heavy_atoms),
    ("tanimoto", lambda m: m.tanimoto),
    ("sdf_path", lambda m: m.sdf_path),
)


_EXPORT_MAX_ROWS = 200_000


# ── Molecules ────────────────────────────────────────────────────────────


# NOTE: GET /api/browse/path (arbitrary absolute-path browsing) was removed:
# it allowed enumerating any directory on the server. Use /api/browse/files,
# which restricts browsing to whitelisted roots.


# ── Data browsing ────────────────────────────────────────────────────────


def _send_sdf_file(path: Optional[str], request: Request):
    decoded = _decode_user_path(path)
    if not decoded:
        raise HTTPException(status_code=400, detail="Missing SDF path")
    abs_path = validate_path_for_user(decoded, request, field="sdf_path")
    if not abs_path or not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="SDF file not found")
    return FileResponse(abs_path, media_type="chemical/x-mdl-sdfile", filename=os.path.basename(abs_path))


# ── File Upload ────────────────────────────────────────────────────────

ALLOWED_UPLOAD_EXTS = {".pdb", ".sdf", ".mol2", ".mol", ".csv", ".txt", ".xlsx", ".xls", ".pt"}


# ── Static files & index page ──────────────────────────────────────────────

_ui_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui")

app.mount("/static", StaticFiles(directory=os.path.join(_ui_root, "static")), name="static")


def _inject_runtime(html: str, request: Request = None) -> str:
    root = public_root()
    if request is not None:
        root = getattr(request.state, "dd_public_root", None) or request_public_root(request.headers)
    home = platform_home_url_for_root(root)
    if root:
        # Keep absolute app links under the public reverse-proxy prefix
        # (e.g. /zh/diffdynamic), otherwise "进入工作台" hits Next.js 404.
        html = html.replace('href="/static/', f'href="{root}/static/')
        html = html.replace('src="/static/', f'src="{root}/static/')
        html = html.replace('"/static/', f'"{root}/static/')
        html = html.replace('href="/app"', f'href="{root}/app"')
        html = html.replace("href='/app'", f"href='{root}/app'")
        html = html.replace('href="/"', f'href="{root}/"')
        html = html.replace("href='/'", f"href='{root}/'")
    # Workbench "back home" should leave DiffDynamic and return to BianLab.
    html = html.replace("__DD_PLATFORM_HOME__", home)
    inject = (
        "<script>window.DD_ROOT=" + json.dumps(root)
        + ";window.DD_AUTH_MODE=" + json.dumps(auth_mode())
        + ";window.DD_LOGIN_URL=" + json.dumps(platform_login_url_for_root(root))
        + ";window.DD_PLATFORM_HOME=" + json.dumps(home)
        + ";</script>\n"
    )
    if "</head>" in html:
        return html.replace("</head>", inject + "</head>", 1)
    return inject + html




# Keep legacy server.api imports and patches bound to the actual route handlers.
register_routes(app, globals())
