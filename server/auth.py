"""BianLab platform session auth for the standalone DiffDynamic UI.

Production: reverse-proxy this service under https://aidd.shu.edu.cn/zh/diffdynamic
and set DD_AUTH_MODE=platform. Browser cookies are forwarded to the platform
session endpoint; unauthenticated visitors are sent to /zh/signin.

Default: DD_AUTH_MODE=platform. For explicit local lab mode, DD_AUTH_MODE=off.
Local APIs still require the internal cookie
set when loading / or /app, or header X-DD-Internal-Key.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import NamedTuple, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest, urlopen

_SESSION_CACHE = {}
_SESSION_TTL = 30.0

LEGACY_OWNERS = frozenset({"", "web_ui", "anonymous", "system", "auto_chain", "internal", "platform_user"})


class PlatformIdentity(NamedTuple):
    username: str
    is_superuser: bool


def auth_mode() -> str:
    mode = (os.environ.get("DD_AUTH_MODE") or "platform").strip().lower()
    if mode in ("platform", "bianlab", "on"):
        return "platform"
    if mode == "off":
        return "off"
    raise ValueError("DD_AUTH_MODE must be platform or off")


def public_root() -> str:
    return (os.environ.get("DD_ROOT_PATH") or "").rstrip("/")


_ALLOWED_PREFIXES = frozenset({"/zh/diffdynamic", "/en/diffdynamic", "/diffdynamic"})


def normalize_forwarded_prefix(raw: str) -> str:
    """Accept only known DiffDynamic public prefixes from nginx."""
    prefix = (raw or "").strip()
    if not prefix:
        return ""
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    prefix = prefix.rstrip("/")
    return prefix if prefix in _ALLOWED_PREFIXES else ""


def request_public_root(headers) -> str:
    """Prefer X-Forwarded-Prefix so /en and /zh share one backend process."""
    get = getattr(headers, "get", None)
    raw = get("x-forwarded-prefix") if get else ""
    if not raw and get:
        raw = get("X-Forwarded-Prefix") or ""
    forwarded = normalize_forwarded_prefix(raw or "")
    if forwarded:
        return forwarded
    return public_root()


def locale_from_prefix(prefix: str) -> str:
    parts = [p for p in (prefix or "").split("/") if p]
    if parts and parts[0] in ("zh", "en"):
        return parts[0]
    # Bare /diffdynamic is the site x-default; English nav is the go-live target.
    if prefix == "/diffdynamic":
        return "en"
    return "zh"


def platform_session_url() -> str:
    return (
        os.environ.get("DD_PLATFORM_SESSION_URL")
        or "https://aidd.shu.edu.cn/api/auth/session"
    ).strip()


def platform_login_url() -> str:
    return (
        os.environ.get("DD_PLATFORM_LOGIN_URL")
        or "https://aidd.shu.edu.cn/zh/signin"
    ).strip()


def platform_login_url_for_root(root: str) -> str:
    """Swap /zh/signin vs /en/signin to match the public prefix."""
    login = platform_login_url()
    loc = locale_from_prefix(root)
    swapped, n = re.subn(r"/(zh|en)/signin", f"/{loc}/signin", login, count=1)
    if n:
        return swapped
    if login.rstrip("/").endswith("/signin"):
        return login
    return f"https://aidd.shu.edu.cn/{loc}/signin"


def platform_home_url() -> str:
    """BianLab site home (e.g. /zh), for leaving the DiffDynamic workbench."""
    return platform_home_url_for_root(public_root())


def platform_home_url_for_root(root: str) -> str:
    loc = locale_from_prefix(root)
    explicit = (os.environ.get("DD_PLATFORM_HOME_URL") or "").strip().rstrip("/")
    if explicit:
        swapped, n = re.subn(r"/(zh|en)$", f"/{loc}", explicit, count=1)
        return swapped if n else explicit
    login = platform_login_url_for_root(root).rstrip("/")
    if login.endswith("/signin"):
        return login[: -len("/signin")] or "/"
    return f"https://aidd.shu.edu.cn/{loc}"


def legacy_owner_slug(username: str) -> str:
    """Previous naming scheme, used only when reserving historical directories."""
    raw = (username or "anonymous").strip()[:64]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    return slug or "anonymous"


def owner_slug(username: str) -> str:
    """Distinct storage names; historical names are reserved by a DB migration."""
    raw = normalize_owner(username)
    from server.database import existing_storage_slug
    existing = existing_storage_slug(raw)
    if existing:
        return existing
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{legacy_owner_slug(raw)[:24]}-{digest}"


def user_output_root(output_root: str, username: str) -> str:
    """Per-user output tree: outputs/users/{slug}/."""
    return os.path.join(output_root, "users", owner_slug(username))


def user_upload_root(upload_dir: str, username: str) -> str:
    """Per-user upload tree: uploads/users/{slug}/."""
    return os.path.join(upload_dir, "users", owner_slug(username))


def user_result_path(output_root: str, username: str) -> str:
    """Default .pt result directory for sampling jobs."""
    return os.path.join(user_output_root(output_root, username), "pt")


def user_sdf_store(output_root: str, username: str) -> str:
    return os.path.join(user_output_root(output_root, username), "sdf_store")


def admin_user_whitelist() -> set:
    raw = (os.environ.get("DD_ADMIN_USERS") or "").strip()
    if not raw:
        return set()
    return {x.strip() for x in raw.split(",") if x.strip()}


def is_admin_user(username: Optional[str], is_superuser: bool = False) -> bool:
    if is_superuser:
        return True
    if not username:
        return False
    # Match the same canonical identity used by ownership and storage. A
    # differently cased platform username must not inherit another's role.
    return username.strip() in admin_user_whitelist()


def _extract_username(payload) -> Optional[str]:
    if payload is None or payload is False:
        return None
    if isinstance(payload, str):
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("username", "name", "email", "user_name", "account"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip() if len(value.strip()) <= 64 else None
    nested = payload.get("user") or payload.get("data") or payload.get("profile") or payload.get("session")
    if nested is not None and nested is not payload:
        return _extract_username(nested)
    return None


def _extract_is_superuser(payload) -> bool:
    if not isinstance(payload, dict):
        return False
    for key in ("isSuperuser", "is_superuser", "is_admin", "isAdmin"):
        val = payload.get(key)
        if isinstance(val, bool):
            return val
    nested = payload.get("user") or payload.get("data") or payload.get("profile") or payload.get("session")
    if nested is not None and nested is not payload:
        return _extract_is_superuser(nested)
    return False


def normalize_owner(username: Optional[str]) -> str:
    """Canonical owner stored on runs.

    Only a missing/blank name maps to the shared ``legacy`` bucket so a real
    platform username never collapses into someone else's history. Historical
    alias rows are still backfilled to ``legacy`` in ``init_db``.
    """
    name = (username or "").strip()
    if len(name) > 64:
        raise ValueError("Account name exceeds 64 characters")
    if not name:
        return "legacy"
    return name


def lookup_platform_identity(
    cookie_header: str = "", authorization: str = ""
) -> Optional[PlatformIdentity]:
    """Return platform username + isSuperuser if the forwarded session is valid."""
    cache_key = hashlib.sha256(f"{cookie_header}|{authorization}".encode("utf-8")).hexdigest()
    now = time.time()
    hit = _SESSION_CACHE.get(cache_key)
    if hit and now - hit[0] < _SESSION_TTL:
        return hit[1]

    url = platform_session_url()
    headers = {"Accept": "application/json", "User-Agent": "DiffDynamic/1.0"}
    if cookie_header:
        headers["Cookie"] = cookie_header
    if authorization:
        headers["Authorization"] = authorization
    req = UrlRequest(url, headers=headers, method="GET")
    identity: Optional[PlatformIdentity] = None
    try:
        with urlopen(req, timeout=5) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        payload = json.loads(raw) if raw.strip() else None
        username = _extract_username(payload)
        if username:
            identity = PlatformIdentity(username, _extract_is_superuser(payload))
    except HTTPError as exc:
        if exc.code not in (401, 403, 404):
            raise
        identity = None
    except (URLError, json.JSONDecodeError, TimeoutError, ValueError):
        identity = None

    if len(_SESSION_CACHE) > 4096:
        _SESSION_CACHE.clear()
    _SESSION_CACHE[cache_key] = (now, identity)
    return identity


def lookup_platform_user(cookie_header: str = "", authorization: str = "") -> Optional[str]:
    """Backward-compatible: return username only."""
    identity = lookup_platform_identity(cookie_header, authorization)
    return identity.username if identity else None
