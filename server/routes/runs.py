"""Runs HTTP routes; business behavior is compatibility-preserving."""

import json
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response

from server.dependencies import dependencies as deps
from server.schemas import (
    CustomGenerateRequest,
    BatchGenerateRequest,
    EvaluateRequest,
    GenerateRequest,
    RerunRequest,
    ScaffoldRequest,
)

router = APIRouter()

__all__ = [
    "list_all_runs",
    "compare_runs_api",
    "get_run_detail",
    "rerun_run",
    "get_run_log",
    "export_run_molecules",
]


@router.get("/api/runs")
def list_all_runs(
    request: Request,
    run_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    _auth: None = Depends(deps.verify_internal_key),
):
    limit = deps._clamp(limit, 1, 500, 100)
    offset = max(0, offset)
    runs = deps.list_runs(
        run_type=run_type,
        status=status,
        limit=limit,
        offset=offset,
        owner=deps._owner_filter(request),
    )
    return [
        {
            "id": r.id,
            "run_type": r.run_type,
            "status": r.status,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "finished_at": r.finished_at.isoformat() if r.finished_at else None,
            "output_path": r.output_path,
            "progress": r.progress,
            "has_log": bool(r.log_output),
            "owner": getattr(r, "owner", None) or r.triggered_by,
            "triggered_by": r.triggered_by,
        }
        for r in runs
    ]


@router.get("/api/runs/compare")
def compare_runs_api(
    request: Request,
    ids: str = Query(..., description="Comma-separated run IDs"),
    _auth: None = Depends(deps.verify_internal_key),
):
    run_ids = [int(x.strip()) for x in ids.split(",") if x.strip().isdigit()]
    if not 2 <= len(run_ids) <= 10:
        raise HTTPException(status_code=400, detail="Provide 2-10 run IDs")
    return deps.compare_runs(run_ids, owner=deps._owner_filter(request))


@router.get("/api/runs/{run_id}")
def get_run_detail(
    run_id: int, request: Request, _auth: None = Depends(deps.verify_internal_key)
):
    r = deps._require_run(run_id, request)
    d = deps.run_to_dict(r)
    d["log_output"] = r.log_output
    d["evaluations"] = [
        {
            "metric": e.metric_name,
            "value": e.metric_value,
            "details": json.loads(e.details) if e.details else None,
        }
        for e in deps.get_evaluations(run_id)
    ]
    return d


@router.post("/api/runs/{run_id}/rerun")
def rerun_run(
    run_id: int,
    req: RerunRequest,
    request: Request,
    _auth: None = Depends(deps.verify_internal_key),
    _rl: None = Depends(deps._rate_limit_submit),
):
    original = deps._require_run(run_id, request)
    user = deps._request_username(request)
    overrides = dict(req.overrides or {})
    if overrides.get("config_path") or overrides.get("patched_config"):
        raise HTTPException(
            status_code=400, detail="Web tasks use default sampling configuration"
        )
    if original.run_type.startswith("generate_"):
        if overrides.get("mode", "dynamic") != "dynamic":
            raise HTTPException(
                status_code=400, detail="Molecule generation uses Dynamic"
            )
        overrides["mode"] = "dynamic"
    for key in ("gpu_id", "preferred_gpu", "gpus"):
        if overrides.get(key) is not None:
            deps.require_admin(request)
    if not deps._request_is_admin(request):
        # The GPU recorded on a completed run is diagnostic output, not an
        # ordinary user's choice for a new run.
        overrides.update({"gpu_id": None, "preferred_gpu": None, "gpus": None})
    if (
        original.run_type.startswith("scaffold_")
        and original.run_type != "scaffold_cascade"
    ):
        overrides = {
            k: v for k, v in overrides.items() if k in ScaffoldRequest.__fields__
        }
        saved = json.loads(original.parameters or "{}").get("scaffold_request")
        if saved:
            saved = {**saved, "config_path": None}
            try:
                validated = deps._validated_scaffold_params(
                    ScaffoldRequest(**{**saved, **overrides}), request
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            overrides = validated
    # Client overrides must not inject scheduler-internal keys (e.g. quota
    # bypasses) nor paths that skip the per-user ACL applied to the original
    # submit endpoints.
    for key in [k for k in overrides if str(k).startswith("_")]:
        overrides.pop(key, None)
    original_params = json.loads(original.parameters or "{}")
    if original.run_type in {"medchem", "medchem_dock"}:
        from server.medchem_schemas import MedchemRequest
        from server.medchem_service import validate_medchem_request

        overrides = {k: v for k, v in overrides.items() if k in MedchemRequest.__fields__}
        saved = {k: v for k, v in original_params.items() if k in MedchemRequest.__fields__}
        try:
            overrides = validate_medchem_request(MedchemRequest(**{**saved, **overrides}), request, deps)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    merged = {**original_params, **overrides}
    for key in (
        "pt_path",
        "protein_path",
        "ligand_path",
        "molecule_path",
        "protein_root",
    ):
        if merged.get(key):
            overrides[key] = deps.validate_path_for_user(
                str(merged[key]), request, field=key
            )
    try:
        if original.run_type == "generate_custom":
            replay = CustomGenerateRequest(**{**merged, "config_path": None})
            overrides.update(
                {
                    "num_samples": deps._clamp(replay.num_samples, 1, 1000, 5),
                    "pocket_radius": max(4.0, min(replay.pocket_radius, 20.0)),
                }
            )
        elif original.run_type == "generate_batch":
            replay = BatchGenerateRequest(**merged)
            if replay.start_id < 0 or replay.end_id < replay.start_id or replay.end_id - replay.start_id > 200:
                raise ValueError("Invalid batch range")
            overrides.update(start_id=replay.start_id, end_id=replay.end_id,
                             batch_size=deps._clamp(replay.batch_size, 1, 100, 5),
                             medchem=replay.medchem.dict() if replay.medchem else None)
        elif original.run_type.startswith("generate_"):
            # Apply the same public limits to replayed and overridden requests.
            replay = GenerateRequest(
                **{**merged, "mode": "dynamic", "config_path": None}
            )
            overrides.update(
                {
                    "batch_size": deps._clamp(replay.batch_size, 1, 100, 5),
                    "data_id": deps._clamp(replay.data_id, 0, 10_000_000, None),
                    "max_samples": deps._clamp(replay.max_samples, 1, 100, 5),
                    "vina_timeout": deps._clamp(replay.vina_timeout, 5, 600, 20),
                }
            )
        elif original.run_type == "evaluate":
            replay = EvaluateRequest(**merged)
            parts = [
                part.strip().lower()
                for part in replay.vina_modes.split(",")
                if part.strip()
            ]
            if not parts or any(part not in deps.VALID_VINA_MODES for part in parts):
                raise ValueError("Unsupported evaluation mode")
            overrides.update(
                {
                    "max_samples": deps._clamp(replay.max_samples, 1, 100, 5),
                    "vina_timeout": deps._clamp(replay.vina_timeout, 5, 600, 20),
                }
            )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid rerun parameters")
    scheduler = deps.get_scheduler()
    result = scheduler.submit_rerun(
        run_id,
        overrides=overrides or None,
        triggered_by=user,
        skip_daily_quota=deps._request_is_admin(request),
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    deps.log_history(
        "rerun", {"source_run_id": run_id, "new_job": result.get("job_id")}, user=user
    )
    return result


@router.get("/api/runs/{run_id}/log")
def get_run_log(
    run_id: int, request: Request, _auth: None = Depends(deps.verify_internal_key)
):
    deps.require_admin(request)
    r = deps._require_run(run_id, request)
    return {"run_id": run_id, "log": r.log_output or "", "status": r.status}


@router.get("/api/runs/{run_id}/export")
def export_run_molecules(
    run_id: int,
    request: Request,
    format: str = Query("csv", pattern="^(csv|xlsx|sdfzip)$"),
    _auth: None = Depends(deps.verify_internal_key),
):
    """On-demand export of a run's molecules (generated from the DB).

    Replaces the per-eval Excel files that used to be written unconditionally
    to disk: same data, but only materialized when a user asks for it.
    """
    r = deps._require_run(run_id, request)
    mols = deps.query_molecules(
        run_id=run_id, limit=deps._EXPORT_MAX_ROWS, offset=0, owner=None
    )
    if not mols:
        raise HTTPException(
            status_code=404, detail="No molecules recorded for this run"
        )
    base_name = f"run_{run_id}_{r.run_type}_molecules"

    if format == "sdfzip":
        import io
        import zipfile

        buf = io.BytesIO()
        added = 0
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            used_names = set()
            for m in mols:
                path = getattr(m, "sdf_path", None)
                if not path:
                    continue
                try:
                    abs_path = deps.validate_path_for_user(
                        path, request, field="sdf_path"
                    )
                except HTTPException:
                    continue
                if not os.path.isfile(abs_path):
                    continue
                name = os.path.basename(abs_path)
                stem, ext = os.path.splitext(name)
                n = 1
                while name in used_names:
                    name = f"{stem}_{n}{ext or '.sdf'}"
                    n += 1
                used_names.add(name)
                zf.write(abs_path, arcname=name)
                added += 1
        if added == 0:
            raise HTTPException(
                status_code=404, detail="No SDF files available for this run"
            )
        return Response(
            content=buf.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{base_name}.zip"'},
        )

    if format == "csv":
        import csv as csvmod
        import io

        buf = io.StringIO()
        writer = csvmod.writer(buf)
        writer.writerow([name for name, _ in deps._EXPORT_COLUMNS])
        for m in mols:
            writer.writerow([getter(m) for _, getter in deps._EXPORT_COLUMNS])
        return Response(
            content=buf.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{base_name}.csv"'},
        )

    import io

    import pandas as pd

    buf = io.BytesIO()
    frame = pd.DataFrame(
        {name: [getter(m) for m in mols] for name, getter in deps._EXPORT_COLUMNS}
    )
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="molecules", index=False)
    return Response(
        content=buf.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{base_name}.xlsx"'},
    )
