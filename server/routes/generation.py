"""Generation HTTP routes; business behavior is compatibility-preserving."""

from fastapi import APIRouter, Depends, HTTPException, Request

from server.dependencies import dependencies as deps
from server.schemas import (
    BatchGenerateRequest,
    CustomGenerateRequest,
    GenerateRequest,
    GenerationContextRequest,
)

router = APIRouter()

__all__ = [
    "generation_context",
    "start_generation",
    "start_batch_generation",
    "start_custom_generation",
    "get_generation_status",
]


@router.post("/api/generate/context")
def generation_context(
    req: GenerationContextRequest,
    request: Request,
    _auth: None = Depends(deps.verify_internal_key),
):
    from server.generation_context import load_generation_context

    params = req.dict()
    for field in ("protein_path", "ligand_path"):
        if params.get(field):
            params[field] = deps.validate_path_for_user(
                params[field], request, field=field
            )
    try:
        return load_generation_context(params)
    except (ValueError, OSError, IndexError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/api/generate")
def start_generation(
    req: GenerateRequest,
    request: Request,
    _auth: None = Depends(deps.verify_internal_key),
    _rl: None = Depends(deps._rate_limit_submit),
):
    if req.mode not in deps.VALID_GENERATE_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"mode must be one of {sorted(deps.VALID_GENERATE_MODES)}",
        )
    user = deps._request_username(request)
    scheduler = deps.get_scheduler()
    config_path = deps.validate_config_path(req.config_path)
    result = scheduler.submit_generation(
        mode=req.mode,
        data_id=deps._clamp(req.data_id, 0, 10_000_000, None),
        use_test_set=req.use_test_set,
        gpu_id=deps._selected_gpu(req.gpu_id, request),
        batch_size=deps._clamp(req.batch_size, 1, 100, 5),
        config_path=config_path,
        auto_evaluate=req.auto_evaluate,
        auto_extract=req.auto_extract,
        remove_fragments=req.remove_fragments,
        max_samples=deps._clamp(req.max_samples, 1, 100, 5),
        vina_timeout=deps._clamp(req.vina_timeout, 5, 600, 20),
        triggered_by=user,
        skip_daily_quota=deps._request_is_admin(request),
        medchem=req.medchem.dict() if req.medchem is not None else None,
    )
    return result


@router.post("/api/generate/batch")
def start_batch_generation(
    req: BatchGenerateRequest,
    request: Request,
    _auth: None = Depends(deps.verify_internal_key),
    _rl: None = Depends(deps._rate_limit_submit),
):
    if req.gpus:
        deps.require_admin(request)
    if req.end_id < req.start_id:
        raise HTTPException(status_code=400, detail="end_id must be >= start_id")
    if req.end_id - req.start_id > 200:
        raise HTTPException(
            status_code=400, detail="batch range too large (max 200 ids)"
        )
    if req.gpus and not all(p.strip().isdigit() for p in req.gpus.split(",")):
        raise HTTPException(
            status_code=400, detail="gpus must be comma-separated integers"
        )
    user = deps._request_username(request)
    scheduler = deps.get_scheduler()
    return scheduler.submit_batch_generation(
        start_id=req.start_id,
        end_id=req.end_id,
        gpus=req.gpus,
        batch_size=deps._clamp(req.batch_size, 1, 100, 5),
        config_path=deps.validate_config_path(req.config_path),
        auto_evaluate=req.auto_evaluate,
        sample_only=req.sample_only,
        triggered_by=user,
        skip_daily_quota=deps._request_is_admin(request),
        medchem=req.medchem.dict() if req.medchem is not None else None,
    )


@router.post("/api/generate/custom")
def start_custom_generation(
    req: CustomGenerateRequest,
    request: Request,
    _auth: None = Depends(deps.verify_internal_key),
    _rl: None = Depends(deps._rate_limit_submit),
):
    """Generate molecules from a custom protein PDB (and optional ligand SDF)."""
    protein_path = deps.validate_path_for_user(
        req.protein_path, request, field="protein_path"
    )
    ligand_path = deps.validate_path_for_user(
        req.ligand_path, request, field="ligand_path"
    )
    user = deps._request_username(request)
    scheduler = deps.get_scheduler()
    result = scheduler.submit_custom_generation(
        protein_path=protein_path,
        ligand_path=ligand_path,
        gpu_id=deps._selected_gpu(req.gpu_id, request),
        pocket_radius=max(4.0, min(req.pocket_radius, 20.0)),
        num_samples=deps._clamp(req.num_samples, 1, 1000, 5),
        config_path=deps.validate_config_path(req.config_path),
        auto_evaluate=req.auto_evaluate,
        auto_extract=req.auto_extract,
        remove_fragments=req.remove_fragments,
        triggered_by=user,
        skip_daily_quota=deps._request_is_admin(request),
        medchem=req.medchem.dict() if req.medchem is not None else None,
    )
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result


@router.get("/api/generate/{job_id}")
def get_generation_status(
    job_id: str, request: Request, _auth: None = Depends(deps.verify_internal_key)
):
    return deps._require_job(job_id, request)
