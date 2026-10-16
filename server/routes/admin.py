"""Admin HTTP routes; business behavior is compatibility-preserving."""

import json
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from server.dependencies import dependencies as deps

router = APIRouter()

__all__ = [
    "browse_history",
    "read_config",
    "update_config",
    "list_presets",
    "gpu_status",
    "health",
    "whoami",
    "get_readme",
]


@router.get("/api/history")
def browse_history(
    request: Request,
    action: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    _auth: None = Depends(deps.verify_internal_key),
):
    limit = deps._clamp(limit, 1, 500, 100)
    offset = max(0, offset)
    records = deps.get_history(
        action=action, limit=limit, offset=offset, owner=deps._owner_filter(request)
    )
    return [
        {
            "id": h.id,
            "action": h.action,
            "details": json.loads(h.details) if h.details else None,
            "user": h.user,
            "created_at": h.created_at.isoformat() if h.created_at else None,
        }
        for h in records
    ]


@router.get("/api/config")
def read_config(request: Request, _auth: None = Depends(deps.verify_internal_key)):
    deps.require_admin(request)
    cfg = deps.get_config()
    runtime = cfg.to_dict()
    runtime.pop("python_bin", None)
    return {
        "runtime": runtime,
        "sampling_yml": deps.get_config_snapshot(cfg.sampling_config),
    }


@router.put("/api/config")
def update_config(
    data: dict,
    request: Request,
    _auth: None = Depends(deps.verify_internal_key),
):
    if not deps._request_is_admin(request):
        raise HTTPException(
            status_code=403, detail="Only administrators may update config"
        )
    top_unknown = set(data.keys()) - {"runtime", "sampling_yml", "sampling_yml_text"}
    if "sampling_yml" in data or "sampling_yml_text" in data:
        raise HTTPException(
            status_code=403, detail="Sampling configuration is read-only"
        )
    if top_unknown:
        raise HTTPException(
            status_code=400, detail=f"Unknown config keys: {sorted(top_unknown)}"
        )
    if "runtime" in data:
        if not isinstance(data["runtime"], dict):
            raise HTTPException(status_code=400, detail="runtime must be an object")
        incoming = data["runtime"]
        unknown = set(incoming.keys()) - deps.CONFIG_MUTABLE_FIELDS
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Runtime fields not writable: {sorted(unknown)}. Allowed: {sorted(deps.CONFIG_MUTABLE_FIELDS)}",
            )
        current = deps.get_config()
        # Validate every value before changing the live configuration.
        checked = {}
        for name, value in incoming.items():
            if name == "gpu_ids":
                if (
                    not isinstance(value, list)
                    or not value
                    or any(type(item) is not int or item < 0 for item in value)
                ):
                    raise HTTPException(
                        status_code=400,
                        detail="gpu_ids must contain nonnegative integers",
                    )
                checked[name] = value
            elif type(value) is not int or value < 0:
                raise HTTPException(
                    status_code=400, detail=f"{name} must be a nonnegative integer"
                )
            elif (
                name in ("max_concurrent_jobs", "job_timeout", "max_upload_bytes")
                and value == 0
            ):
                raise HTTPException(status_code=400, detail=f"{name} must be positive")
            else:
                checked[name] = value
        for field_name, value in checked.items():
            setattr(current, field_name, value)
        deps.save_config(current)
    deps.log_history(
        "config_updated",
        {"keys": list(data.keys())},
        user=deps._request_username(request),
    )
    return {"status": "ok"}


@router.get("/api/presets")
def list_presets(request: Request, _auth: None = Depends(deps.verify_internal_key)):
    deps.require_admin(request)
    """List available preset YAML configs under configs/run_*.yml."""
    cfg = deps.get_config()
    configs_dir = os.path.join(cfg.diffdynamic_root, "configs")
    presets = []
    if os.path.isdir(configs_dir):
        for fn in sorted(os.listdir(configs_dir)):
            if fn.startswith("run_") and fn.endswith((".yml", ".yaml")):
                fpath = os.path.join(configs_dir, fn)
                presets.append(
                    {"name": fn[4:].rsplit(".", 1)[0], "path": fpath, "filename": fn}
                )
    return {"presets": presets}


@router.get("/api/gpus")
def gpu_status(request: Request, _auth: None = Depends(deps.verify_internal_key)):
    deps.require_admin(request)
    return deps.get_gpu_info()


@router.get("/api/health")
def health(request: Request):
    root = getattr(request.state, "dd_public_root", None) or deps.request_public_root(
        request.headers
    )
    return {
        "status": "ok",
        "auth_mode": deps.auth_mode(),
        "login_url": deps.platform_login_url_for_root(root)
        if deps.auth_mode() == "platform"
        else None,
        "root_path": root or "/",
    }


@router.get("/api/me")
def whoami(request: Request, _auth: None = Depends(deps.verify_internal_key)):
    username = deps._request_username(request)
    owner = deps.normalize_owner(username)
    cfg = deps.get_config()
    scheduler = deps.get_scheduler()
    is_admin = deps._request_is_admin(request)
    gen_limit = int(getattr(cfg, "max_generate_per_day", 0) or 0)
    eval_limit = int(getattr(cfg, "max_eval_per_day", 0) or 0)
    gen_used = 0 if is_admin else deps.count_runs_today(owner, "generate")
    eval_used = 0 if is_admin else deps.count_runs_today(owner, "eval")
    _, reset_at = deps.shanghai_day_bounds()
    return {
        "username": username,
        "auth_mode": deps.auth_mode(),
        "is_admin": is_admin,
        "owner": owner,
        "quota": {
            "active_jobs": scheduler.active_jobs_for_owner(owner),
            "max_jobs_per_user": cfg.max_jobs_per_user,
            "max_upload_bytes": cfg.max_upload_bytes,
            "generate_used": gen_used,
            "generate_limit": 0 if is_admin else gen_limit,
            "eval_used": eval_used,
            "eval_limit": 0 if is_admin else eval_limit,
            "resets_at": reset_at.isoformat(),
            "seconds_until_reset": deps.seconds_until_reset(),
            "unlimited": is_admin,
        },
    }


@router.get("/api/readme")
def get_readme(lang: str = "zh", _auth: None = Depends(deps.verify_internal_key)):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(deps.__file__)))
    ui_root = os.path.join(project_root, "ui")
    lang = lang if lang in ("zh", "en") else "zh"
    overview_path = os.path.join(ui_root, "content", f"overview.{lang}.md")
    readme_path = os.path.join(project_root, "README.md")
    for path in (overview_path, readme_path):
        try:
            with open(path, encoding="utf-8") as f:
                return {"content": f.read()}
        except Exception:
            continue
    return {"content": "# DiffDynamic\n\nOverview not found."}
