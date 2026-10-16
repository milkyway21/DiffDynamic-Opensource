"""Files HTTP routes; business behavior is compatibility-preserving."""

import os
import re
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse

from server.dependencies import dependencies as deps

router = APIRouter()

__all__ = [
    "browse_files",
    "browse_outputs",
    "pt_metadata",
    "download_sdf_query",
    "download_sdf",
    "download_file",
    "browse_proteins",
    "upload_file",
]


@router.get("/api/browse/files")
def browse_files(
    request: Request,
    root: str = Query("outputs"),
    subpath: str = "",
    ext: Optional[str] = None,
    _auth: None = Depends(deps.verify_internal_key),
):
    cfg = deps.get_config()
    roots = {
        "outputs": cfg.output_root,
        "data": cfg.data_root,
        "configs": os.path.join(cfg.diffdynamic_root, "configs"),
        "uploads": cfg.upload_dir,
    }
    if root not in roots:
        raise HTTPException(
            status_code=400, detail=f"root must be one of {list(roots.keys())}"
        )
    # Prevent path escape via subpath
    subpath = subpath.replace("\\", "/").lstrip("/")
    if ".." in subpath.split("/"):
        raise HTTPException(status_code=400, detail="invalid subpath")

    is_admin = deps._request_is_admin(request)
    if root == "configs" and not is_admin:
        raise HTTPException(status_code=403, detail="Administrator access required")
    owner = deps.normalize_owner(deps._request_username(request))
    slug = deps.owner_slug(owner)
    if not is_admin and root in ("outputs", "uploads"):
        prefix = f"users/{slug}"
        if not subpath:
            subpath = prefix
        elif subpath != prefix and not subpath.startswith(prefix + "/"):
            raise HTTPException(
                status_code=403, detail="Access denied to this directory"
            )
        # Ensure the user directory exists so browse is not empty/erroring
        os.makedirs(os.path.join(roots[root], prefix), exist_ok=True)

    extensions = tuple(e.strip() for e in ext.split(",")) if ext else None
    if extensions:
        extensions = tuple(e if e.startswith(".") else f".{e}" for e in extensions)
    extra_roots = deps._top_level_symlink_roots(roots[root]) if root == "data" else None
    entries = deps.browse_directory(
        roots[root], subpath, extensions, extra_roots=extra_roots
    )
    if not is_admin:
        private_exts = (".log", ".out", ".err", ".txt", ".json", ".yaml", ".yml")
        entries = [
            entry
            for entry in entries
            if not str(entry.get("name", "")).lower().endswith(private_exts)
        ]
    # Hide .pt files in shared data/configs roots from non-admins: they may only
    # pick their own generated/uploaded .pt (enforced by assert_path_acl too).
    if not is_admin and root in ("data", "configs"):
        entries = [
            e for e in entries if not str(e.get("name", "")).lower().endswith(".pt")
        ]
    abs_root = os.path.realpath(roots[root])
    current = os.path.realpath(os.path.join(abs_root, subpath)) if subpath else abs_root
    return {
        "root": root,
        "subpath": subpath,
        "current_path": current,
        "dirs": [e for e in entries if e.get("type") == "dir"],
        "files": [e for e in entries if e.get("type") == "file"],
    }


@router.get("/api/outputs")
def browse_outputs(request: Request, _auth: None = Depends(deps.verify_internal_key)):
    cfg = deps.get_config()
    if deps._request_is_admin(request):
        root = cfg.output_root
    else:
        root = os.path.join(
            cfg.output_root,
            "users",
            deps.owner_slug(deps.normalize_owner(deps._request_username(request))),
        )
        os.makedirs(root, exist_ok=True)
    return {
        "results": deps.scan_outputs(root),
        "eval_dirs": deps.scan_eval_dirs(root),
    }


@router.get("/api/pt/{path:path}")
def pt_metadata(
    path: str, request: Request, _auth: None = Depends(deps.verify_internal_key)
):
    abs_path = deps.validate_path_for_user(path, request, field="pt_path")
    return deps.load_pt_metadata(
        abs_path,
        owner=None
        if deps._request_is_admin(request)
        else deps._request_username(request),
    )


@router.get("/api/sdf")
def download_sdf_query(
    request: Request,
    path: Optional[str] = Query(None),
    _auth: None = Depends(deps.verify_internal_key),
):
    """Query-string download so nginx does not swallow %2F in the path."""
    return deps._send_sdf_file(path, request)


@router.get("/api/sdf/{path:path}")
def download_sdf(
    path: str, request: Request, _auth: None = Depends(deps.verify_internal_key)
):
    return deps._send_sdf_file(path, request)


@router.get("/api/file/download")
def download_file(
    path: str, request: Request, _auth: None = Depends(deps.verify_internal_key)
):
    """Generic file download for SDF/PDB from output_root or uploads."""
    abs_path = deps.validate_path_for_user(path, request, field="path")
    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="File not found")
    ext = os.path.splitext(abs_path)[1].lower()
    media = {
        ".sdf": "chemical/x-mdl-sdfile",
        ".pdb": "chemical/x-pdb",
        ".mol2": "chemical/x-mol2",
        ".csv": "text/csv",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".pt": "application/octet-stream",
        ".png": "image/png",
        ".pdf": "application/pdf",
    }.get(ext, "application/octet-stream")
    return FileResponse(abs_path, media_type=media, filename=os.path.basename(abs_path))


@router.get("/api/proteins")
def browse_proteins(_auth: None = Depends(deps.verify_internal_key)):
    cfg = deps.get_config()
    return deps.list_proteins(cfg.data_root)


@router.post("/api/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    _auth: None = Depends(deps.verify_internal_key),
    _rl: None = Depends(deps._rate_limit_upload),
):
    """Upload a protein, ligand, result table, or .pt result.

    Uploaded files always land in the caller's own uploads/users/{owner}/ tree;
    Uploaded PT files are parsed only by the isolated secure loader."""
    cfg = deps.get_config()
    max_upload_bytes = int(cfg.max_upload_bytes or 0) or 52428800
    raw_name = os.path.basename(file.filename or "upload")
    ext = os.path.splitext(raw_name)[1].lower()
    if ext not in deps.ALLOWED_UPLOAD_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(deps.ALLOWED_UPLOAD_EXTS))}",
        )
    owner = deps.normalize_owner(deps._request_username(request))
    upload_root = os.path.join(
        deps.user_upload_root(cfg.upload_dir, owner),
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(upload_root, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}_{re.sub(r'[^A-Za-z0-9._-]', '_', raw_name)[:80]}"
    dest = os.path.join(upload_root, safe_name)
    size = 0
    try:
        with open(dest, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (max {max_upload_bytes // (1024 * 1024)}MB)",
                    )
                f.write(chunk)
    except Exception:
        if os.path.exists(dest):
            os.remove(dest)
        raise
    finally:
        await file.close()
    return {
        "filename": raw_name,
        "stored_path": dest,
        "size": size,
        "content_type": file.content_type,
        "owner": owner,
    }
