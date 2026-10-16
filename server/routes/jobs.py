"""Jobs HTTP routes; business behavior is compatibility-preserving."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from server.dependencies import dependencies as deps

router = APIRouter()

__all__ = ["list_jobs", "get_job", "cancel_job"]


@router.get("/api/jobs")
def list_jobs(
    request: Request,
    status: Optional[str] = Query(None),
    _auth: None = Depends(deps.verify_internal_key),
):
    scheduler = deps.get_scheduler()
    return scheduler.list_jobs(status=status, owner=deps._owner_filter(request))


@router.get("/api/jobs/{job_id}")
def get_job(
    job_id: str, request: Request, _auth: None = Depends(deps.verify_internal_key)
):
    return deps._require_job(job_id, request)


@router.delete("/api/jobs/{job_id}")
def cancel_job(
    job_id: str, request: Request, _auth: None = Depends(deps.verify_internal_key)
):
    scheduler = deps.get_scheduler()
    ok = scheduler.cancel(job_id, owner=deps._owner_filter(request))
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found or not cancellable")
    return {"cancelled": job_id}
