"""Owner-scoped medicinal chemistry tasks and artifact access."""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from server.dependencies import dependencies as deps
from server.medchem_schemas import MedchemRequest
from server.medchem_service import artifact_path, child_reports, read_report, validate_medchem_request

router = APIRouter()
__all__ = ["start_medchem", "get_medchem_report", "get_medchem_artifact"]


@router.post("/api/medchem")
def start_medchem(req: MedchemRequest, request: Request,
                  _auth=Depends(deps.verify_internal_key),
                  _rl=Depends(deps._rate_limit_submit)):
    params = validate_medchem_request(req, request, deps)
    return deps.get_scheduler().submit_medchem(
        **params, triggered_by=deps._request_username(request),
        skip_daily_quota=deps._request_is_admin(request),
    )


@router.get("/api/medchem/runs/{run_id}/report")
def get_medchem_report(run_id: int, request: Request,
                       _auth=Depends(deps.verify_internal_key)):
    run = deps._require_run(run_id, request)
    if run.run_type in {"medchem", "medchem_dock"}:
        return read_report(run)
    children = child_reports(run, owner=deps._owner_filter(request))
    if len(children) == 1:
        return read_report(deps._require_run(children[0]["run_id"], request))
    import json
    detail = json.loads(run.progress_detail or "{}")
    return {"run_id": run_id, "status": "failed" if detail.get("medchem_error") else run.status,
            "error": detail.get("medchem_error"), "records": [],
            "artifacts": [], "reports": children}


@router.get("/api/medchem/runs/{run_id}/artifacts/{artifact_id}")
def get_medchem_artifact(run_id: int, artifact_id: str, request: Request,
                         _auth=Depends(deps.verify_internal_key)):
    run = deps._require_run(run_id, request)
    path, item = artifact_path(run, artifact_id)
    media = {"svg": "image/svg+xml", "sdf": "chemical/x-mdl-sdfile",
             "pdb": "chemical/x-pdb", "pdbqt": "chemical/x-pdbqt",
             "csv": "text/csv", "json": "application/json"}.get(item.get("format"))
    return FileResponse(path, media_type=media or "application/octet-stream",
                        filename=path.name, headers={"X-Content-Type-Options": "nosniff"})
