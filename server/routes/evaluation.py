"""Evaluation HTTP routes; business behavior is compatibility-preserving."""

import json
import os

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from server.dependencies import dependencies as deps
from server.schemas import (
    EvaluateRequest,
    ExtractRequest,
    PocketEvalRequest,
)

router = APIRouter()

__all__ = [
    "start_evaluation",
    "get_evaluation_status",
    "start_extraction",
    "start_pocket_eval",
    "serve_pocket_eval_image",
    "get_pocket_eval_results",
]


@router.post("/api/evaluate")
def start_evaluation(
    req: EvaluateRequest,
    request: Request,
    _auth: None = Depends(deps.verify_internal_key),
    _rl: None = Depends(deps._rate_limit_submit),
):
    pt_path = deps.validate_path_for_user(req.pt_path, request, field="pt_path")
    protein_root = (
        deps.validate_path_for_user(req.protein_root, request, field="protein_root")
        if req.protein_root
        else None
    )
    if not protein_root:
        # 评估子进程强制要求 protein_root 存在；脚本默认的
        # data/crossdocked_v1.1_rmsd1.0 在本机不存在，回落到测试集根。
        protein_root = os.path.join(
            deps.get_config().data_root, "crossdocked_pocket10_test_only"
        )
    if req.vina_modes not in deps.VALID_VINA_MODES:
        # 逗号组合（如 "dock,score_only"）：逐项校验后原样透传给脚本解析
        parts = [p.strip().lower() for p in str(req.vina_modes).split(",") if p.strip()]
        if not parts or any(p not in deps.VALID_VINA_MODES for p in parts):
            raise HTTPException(
                status_code=400,
                detail=f"vina_modes must be one of {sorted(deps.VALID_VINA_MODES)} or comma-separated combination",
            )
    user = deps._request_username(request)
    scheduler = deps.get_scheduler()
    result = scheduler.submit_evaluation(
        pt_path=pt_path,
        protein_root=protein_root,
        vina_modes=req.vina_modes,
        max_samples=deps._clamp(req.max_samples, 1, 100, 5),
        vina_timeout=deps._clamp(req.vina_timeout, 5, 600, 20),
        triggered_by=user,
        skip_daily_quota=deps._request_is_admin(request),
    )
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result


@router.get("/api/evaluate/{job_id}")
def get_evaluation_status(
    job_id: str, request: Request, _auth: None = Depends(deps.verify_internal_key)
):
    return deps._require_job(job_id, request)


@router.post("/api/extract")
def start_extraction(
    req: ExtractRequest,
    request: Request,
    _auth: None = Depends(deps.verify_internal_key),
    _rl: None = Depends(deps._rate_limit_submit),
):
    if req.parent_run_id is not None:
        deps._require_run(req.parent_run_id, request)
    pt_path = deps.validate_path_for_user(req.pt_path, request, field="pt_path")
    protein_root = (
        deps.validate_path_for_user(req.protein_root, request, field="protein_root")
        if req.protein_root
        else None
    )
    user = deps._request_username(request)
    scheduler = deps.get_scheduler()
    result = scheduler.submit_extraction(
        pt_path=pt_path,
        protein_root=protein_root,
        remove_fragments=req.remove_fragments,
        max_samples=req.max_samples,
        vina_timeout=req.vina_timeout,
        parent_run_id=req.parent_run_id,
        triggered_by=user,
        skip_daily_quota=deps._request_is_admin(request),
    )
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result


@router.post("/api/pocket-eval")
def start_pocket_eval(
    req: PocketEvalRequest,
    request: Request,
    _auth: None = Depends(deps.verify_internal_key),
    _rl: None = Depends(deps._rate_limit_submit),
):
    if not req.protein_path or not req.ligand_path:
        raise HTTPException(
            status_code=400, detail="protein_path and ligand_path are required"
        )
    if not req.generate_first and not req.pt_path:
        raise HTTPException(
            status_code=400, detail="pt_path is required when generate_first is false"
        )
    protein_path = deps.validate_path_for_user(
        req.protein_path, request, field="protein_path"
    )
    ligand_path = deps.validate_path_for_user(
        req.ligand_path, request, field="ligand_path"
    )
    pt_path = (
        deps.validate_path_for_user(req.pt_path, request, field="pt_path")
        if req.pt_path
        else None
    )
    user = deps._request_username(request)
    scheduler = deps.get_scheduler()
    result = scheduler.submit_pocket_eval(
        pt_path=pt_path,
        protein_path=protein_path,
        ligand_path=ligand_path,
        generate_first=req.generate_first,
        gpu_id=deps._selected_gpu(req.gpu_id, request),
        batch_size=deps._clamp(req.batch_size, 1, 100, 5),
        pocket_radius=max(4.0, min(req.pocket_radius, 20.0)),
        max_samples=deps._clamp(req.max_samples, 1, 100, 5),
        triggered_by=user,
        skip_daily_quota=deps._request_is_admin(request),
    )
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result


@router.get("/api/pocket-eval/images/{path:path}")
def serve_pocket_eval_image(
    path: str,
    request: Request,
    _auth: None = Depends(deps.verify_internal_key),
):
    cfg = deps.get_config()
    if not path or os.path.isabs(path):
        raise HTTPException(status_code=400, detail="Invalid image path")

    output_root = os.path.realpath(cfg.output_root)
    shared_root = os.path.realpath(
        os.path.join(cfg.diffdynamic_root, "pocket_quality_vis")
    )
    relative_path = path.replace("\\", "/")
    output_path = os.path.realpath(os.path.join(output_root, relative_path))
    if deps._is_within(output_path, output_root) and os.path.isfile(output_path):
        abs_path = deps.validate_path_for_user(output_path, request, field="image_path")
    else:
        # Legacy jobs wrote under the shared visualization directory. Keep that
        # compatibility path admin-only while new jobs use per-user output trees.
        shared_path = os.path.realpath(os.path.join(shared_root, relative_path))
        if not deps._is_within(shared_path, shared_root):
            raise HTTPException(
                status_code=403, detail="Path outside allowed directory"
            )
        if not deps._request_is_admin(request):
            raise HTTPException(
                status_code=403, detail="Shared visualization access denied"
            )
        abs_path = shared_path

    if os.path.splitext(abs_path)[1].lower() != ".png":
        raise HTTPException(status_code=400, detail="Only PNG images are supported")
    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(abs_path, media_type="image/png")


@router.get("/api/pocket-eval/results/{job_id}")
def get_pocket_eval_results(
    job_id: str, request: Request, _auth: None = Depends(deps.verify_internal_key)
):
    scheduler = deps.get_scheduler()
    rec = scheduler.get_job_record(job_id, owner=deps._owner_filter(request))
    if rec is None:
        raise HTTPException(status_code=404, detail="Job not found")
    vis_dir = rec.output_path
    images = []
    scores = dict(rec.scores or {})
    cfg = deps.get_config()
    shared_root = os.path.join(cfg.diffdynamic_root, "pocket_quality_vis")
    if vis_dir and deps._is_within(vis_dir, cfg.output_root):
        dir_name = os.path.relpath(vis_dir, cfg.output_root).replace(os.sep, "/")
    elif vis_dir and deps._is_within(vis_dir, shared_root):
        dir_name = os.path.relpath(vis_dir, shared_root).replace(os.sep, "/")
    else:
        dir_name = os.path.basename(vis_dir) if vis_dir else ""
    if vis_dir and os.path.isdir(vis_dir):
        for f in sorted(os.listdir(vis_dir)):
            if f.endswith(".png") and "_notext" not in f:
                images.append(f)
    if not scores and rec.run_id:
        # Map DB metric names (and legacy aliases) back to UI score_* keys
        metric_to_score = {
            "vina_docking": "score_a",
            "clustering": "score_b",
            "drug_likeness": "score_c",
            "ligand_efficiency": "score_a",  # legacy: LE merged into A
            "diversity": "score_d",
            "completeness": "score_e",
            "pocket_chemistry": "score_f",
            "anchor_richness": "score_g",
            "size_consistency": "score_g",  # legacy size → ignore semantics, map g
            "pocket_volume": "score_h",
            "interaction": "score_g",
            "stability": "score_h",
            "overall_pocket_score": "overall_score",
            "score_a": "score_a",
            "score_b": "score_b",
            "score_c": "score_c",
            "score_d": "score_d",
            "score_e": "score_e",
            "score_f": "score_f",
            "score_g": "score_g",
            "score_h": "score_h",
            "overall_score": "overall_score",
        }
        evals = deps.get_evaluations(rec.run_id)
        for e in evals:
            key = None
            if e.details:
                try:
                    details = (
                        json.loads(e.details)
                        if isinstance(e.details, str)
                        else e.details
                    )
                    if isinstance(details, dict) and details.get("dim"):
                        key = details["dim"]
                except (TypeError, ValueError):
                    pass
            if not key:
                key = metric_to_score.get(e.metric_name)
            if key:
                scores[key] = (
                    e.metric_value if e.metric_value is not None else e.details
                )
            if e.metric_name == "overall_pocket_score" and e.details:
                try:
                    details = (
                        json.loads(e.details)
                        if isinstance(e.details, str)
                        else e.details
                    )
                    if isinstance(details, dict) and details.get("label"):
                        scores["overall_label"] = details["label"]
                except (TypeError, ValueError):
                    pass
    return {
        "job_id": job_id,
        "vis_dir": vis_dir,
        "dir_name": dir_name,
        "images": images,
        "scores": scores,
    }
