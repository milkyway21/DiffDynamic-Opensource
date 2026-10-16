"""Scaffold HTTP routes; business behavior is compatibility-preserving."""

import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from server.dependencies import dependencies as deps
from server.schemas import (
    OptimizationRequest,
    ScaffoldCascadeRequest,
    ScaffoldPreviewRequest,
    ScaffoldRequest,
)

router = APIRouter()

__all__ = [
    "scaffold_context",
    "scaffold_preview",
    "start_optimization",
    "get_optimization_status",
    "start_scaffold",
    "preview_adaptive_gates",
    "get_scaffold_status",
    "start_scaffold_cascade",
    "get_scaffold_cascade_status",
]


@router.post("/api/scaffold/context")
def scaffold_context(
    req: ScaffoldRequest,
    request: Request,
    _auth: None = Depends(deps.verify_internal_key),
):
    from server.scaffold_context import load_context

    params = deps._validated_scaffold_params(req, request)
    try:
        return load_context(params)[3]
    except (ValueError, OSError, IndexError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/api/scaffold/preview")
def scaffold_preview(
    req: ScaffoldPreviewRequest,
    request: Request,
    _auth: None = Depends(deps.verify_internal_key),
):
    from server.scaffold_context import preview_context

    params = deps._validated_scaffold_params(req, request)
    try:
        return preview_context(params)
    except (ValueError, OSError, IndexError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/api/optimization")
def start_optimization(
    req: OptimizationRequest,
    request: Request,
    _auth: None = Depends(deps.verify_internal_key),
    _rl: None = Depends(deps._rate_limit_submit),
):
    protein_path = (
        deps.validate_path_for_user(req.protein_path, request, field="protein_path")
        if req.protein_path
        else None
    )
    ligand_path = (
        deps.validate_path_for_user(req.ligand_path, request, field="ligand_path")
        if req.ligand_path
        else None
    )
    molecule_path = (
        deps.validate_path_for_user(req.molecule_path, request, field="molecule_path")
        if req.molecule_path
        else None
    )
    user = deps._request_username(request)
    scheduler = deps.get_scheduler()
    result = scheduler.submit_optimization(
        data_id=deps._clamp(req.data_id, 0, 10_000_000, None),
        protein_path=protein_path,
        ligand_path=ligand_path,
        molecule_path=molecule_path,
        use_test_set=req.use_test_set,
        gpu_id=deps._selected_gpu(req.gpu_id, request),
        config_path=deps.validate_config_path(req.config_path),
        auto_evaluate=req.auto_evaluate,
        auto_extract=req.auto_extract,
        remove_fragments=req.remove_fragments,
        num_samples=deps._clamp(req.num_samples, 1, 100, 2),
        start_t=deps._clamp(req.start_t, 1, 1000, 16),
        stride=deps._clamp(req.stride, 1, 100, 2),
        step_size=max(0.01, min(req.step_size, 2.0)),
        cycles=deps._clamp(req.cycles, 1, 50, 5),
        schedule=req.schedule,
        after_dynamic=req.after_dynamic,
        min_qed=max(0.0, min(req.min_qed, 1.0)),
        min_sa=max(0.0, min(req.min_sa, 1.0)),
        max_survivors_per_cycle=deps._clamp(req.max_survivors_per_cycle, 1, 100, 10),
        max_survivors_total=deps._clamp(req.max_survivors_total, 1, 2000, 300),
        triggered_by=user,
        skip_daily_quota=deps._request_is_admin(request),
    )
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result


@router.get("/api/optimization/{job_id}")
def get_optimization_status(
    job_id: str, request: Request, _auth: None = Depends(deps.verify_internal_key)
):
    return deps._require_job(job_id, request)


@router.post("/api/scaffold")
def start_scaffold(
    req: ScaffoldRequest,
    request: Request,
    _auth: None = Depends(deps.verify_internal_key),
    _rl: None = Depends(deps._rate_limit_submit),
):
    validated = deps._validated_scaffold_params(req, request)
    # Fail invalid anchors/geometry before creating a job or consuming quota.
    if req.placement.mode == "custom" or req.scaffold_source in (
        "atom_indices",
        "smarts",
        "custom",
    ):
        from server.scaffold_context import load_context, preview_context

        try:
            if req.placement.mode == "custom":
                preview_context(validated)
            else:
                load_context(validated)
        except (ValueError, OSError, IndexError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    # gate_mode 优先；兼容旧客户端只传 adaptive_gates_enable
    gm = (req.gate_mode or "adaptive").strip().lower()
    if gm not in ("adaptive", "manual"):
        raise HTTPException(status_code=400, detail="gate_mode must be adaptive|manual")
    if req.adaptive_gates_enable is None:
        adaptive_enable = gm == "adaptive"
    else:
        adaptive_enable = bool(req.adaptive_gates_enable) if gm == "adaptive" else False
        if gm == "manual":
            adaptive_enable = False

    protein_path = (
        deps.validate_path_for_user(req.protein_path, request, field="protein_path")
        if req.protein_path
        else None
    )
    if req.ligand_path:
        deps.validate_path_for_user(req.ligand_path, request, field="ligand_path")
    molecule_path = (
        deps.validate_path_for_user(req.molecule_path, request, field="molecule_path")
        if req.molecule_path
        else None
    )
    adaptive_gates_actives = (
        deps.validate_path_for_user(
            req.adaptive_gates_actives, request, field="adaptive_gates_actives"
        )
        if req.adaptive_gates_actives
        else None
    )
    # SMARTS/atom-indices 只允许化学语法字符，防止注入下游解析
    if req.scaffold_smarts and not re.fullmatch(
        r"[A-Za-z0-9\[\]\(\)#=\-\+\*\.:\|@%1-9\\/&,$;{}<>!?~^' ]*", req.scaffold_smarts
    ):
        raise HTTPException(
            status_code=400, detail="scaffold_smarts contains invalid characters"
        )
    if req.scaffold_atom_indices and not re.fullmatch(
        r"[0-9,\s\-]+", req.scaffold_atom_indices
    ):
        raise HTTPException(
            status_code=400,
            detail="scaffold_atom_indices must be comma-separated integers",
        )

    user = deps._request_username(request)
    scheduler = deps.get_scheduler()
    result = scheduler.submit_scaffold(
        workflow=req.workflow,
        seed=req.seed,
        placement=req.placement.dict(),
        data_id=deps._clamp(req.data_id, 0, 10_000_000, None),
        protein_path=protein_path,
        ligand_path=validated["ligand_path"],
        molecule_path=molecule_path,
        use_test_set=req.use_test_set,
        gpu_id=deps._selected_gpu(req.gpu_id, request),
        config_path=deps.validate_config_path(req.config_path),
        auto_evaluate=req.auto_evaluate,
        auto_extract=req.auto_extract,
        remove_fragments=req.remove_fragments,
        scaffold_mode=req.scaffold_mode,
        scaffold_source=req.scaffold_source,
        scaffold_atom_indices=req.scaffold_atom_indices,
        scaffold_smarts=req.scaffold_smarts,
        fix_scaffold_pos=req.fix_scaffold_pos,
        fix_scaffold_type=req.fix_scaffold_type,
        after_dynamic=req.after_dynamic,
        scaffold_enable_refine=req.scaffold_enable_refine,
        qed_weight=max(0.0, min(req.qed_weight, 10.0)),
        sa_weight=max(0.0, min(req.sa_weight, 10.0)),
        diversity_weight=max(0.0, min(req.diversity_weight, 10.0)),
        rmsd_penalty_weight=max(0.0, min(req.rmsd_penalty_weight, 10.0)),
        min_qed=max(0.0, min(req.min_qed, 1.0)),
        min_sa=max(0.0, min(req.min_sa, 1.0)),
        grow_num_samples=deps._clamp(req.grow_num_samples, 1, 2000, 200),
        grow_start_t=deps._clamp(req.grow_start_t, 1, 1000, 357),
        grow_stride=deps._clamp(req.grow_stride, 1, 200, 15),
        grow_step_size=max(0.01, min(req.grow_step_size, 2.0)),
        grow_lambda_a=deps._clamp(req.grow_lambda_a, 0, 500, 47),
        grow_lambda_b=deps._clamp(req.grow_lambda_b, 0, 500, 11),
        grow_n_extra_mode=req.grow_n_extra_mode,
        grow_n_extra_fixed=deps._clamp(req.grow_n_extra_fixed, 0, 50, 8),
        grow_n_extra_min=deps._clamp(req.grow_n_extra_min, 0, 50, 3),
        grow_n_extra_max=deps._clamp(req.grow_n_extra_max, 0, 50, 20),
        grow_filter_incomplete=req.grow_filter_incomplete,
        grow_min_qed=max(0.0, min(req.grow_min_qed, 1.0)),
        grow_min_sa=max(0.0, min(req.grow_min_sa, 1.0)),
        evolve_population_size=deps._clamp(req.evolve_population_size, 1, 500, 50),
        evolve_n_generations=deps._clamp(req.evolve_n_generations, 1, 100, 5),
        evolve_children_per_parent=deps._clamp(
            req.evolve_children_per_parent, 1, 100, 10
        ),
        evolve_start_t_high=deps._clamp(req.evolve_start_t_high, 1, 1000, 200),
        evolve_start_t_low=deps._clamp(req.evolve_start_t_low, 1, 1000, 30),
        evolve_stride=deps._clamp(req.evolve_stride, 1, 200, 2),
        evolve_step_size=max(0.01, min(req.evolve_step_size, 2.0)),
        evolve_lambda_a=deps._clamp(req.evolve_lambda_a, 0, 500, 10),
        evolve_lambda_b=deps._clamp(req.evolve_lambda_b, 0, 500, 1),
        diversity_filter_enable=req.diversity_filter_enable,
        max_tanimoto=max(0.0, min(req.max_tanimoto, 1.0)),
        prudent_n_generations=deps._clamp(req.prudent_n_generations, 1, 100, 5),
        prudent_n_chains_per_seed=deps._clamp(req.prudent_n_chains_per_seed, 1, 50, 4),
        prudent_advance_top_k=deps._clamp(req.prudent_advance_top_k, 1, 50, 2),
        prudent_renoise_t=deps._clamp(req.prudent_renoise_t, 1, 2000, 600),
        prudent_qed_weight=max(0.0, min(req.prudent_qed_weight, 10.0)),
        prudent_sa_weight=max(0.0, min(req.prudent_sa_weight, 10.0)),
        prudent_vina_weight=max(0.0, min(req.prudent_vina_weight, 10.0)),
        prudent_lipinski_weight=max(0.0, min(req.prudent_lipinski_weight, 10.0)),
        prudent_lilly_weight=max(0.0, min(req.prudent_lilly_weight, 10.0)),
        prudent_vina_exhaustiveness=deps._clamp(
            req.prudent_vina_exhaustiveness, 1, 32, 8
        ),
        prudent_min_qed_for_docking=max(0.0, min(req.prudent_min_qed_for_docking, 1.0)),
        prudent_min_sa_for_docking=max(0.0, min(req.prudent_min_sa_for_docking, 1.0)),
        adaptive_gates_enable=adaptive_enable,
        adaptive_gates_actives=adaptive_gates_actives,
        gate_mode=gm,
        manual_max_logp=max(
            0.0, min(5.0 if req.manual_max_logp is None else req.manual_max_logp, 20.0)
        ),
        manual_max_molwt=max(50.0, min(req.manual_max_molwt or 850.0, 2000.0)),
        manual_max_heavy_atoms=deps._clamp(req.manual_max_heavy_atoms, 1, 200, 60),
        manual_max_rings=deps._clamp(req.manual_max_rings, 0, 30, 7),
        triggered_by=user,
        skip_daily_quota=deps._request_is_admin(request),
    )
    if "error" in result:
        detail = result["error"]
        code = 400 if "Invalid scaffold_mode" in detail else 503
        raise HTTPException(status_code=code, detail=detail)
    return result


@router.get("/api/adaptive_gates/preview")
def preview_adaptive_gates(
    request: Request,
    ligand_path: str,
    actives_csv: Optional[str] = None,
    _auth: None = Depends(deps.verify_internal_key),
):
    """根据原始参考配体预览自适应门控（任务创建前可调用）。"""
    from pathlib import Path as _P

    from diffdynamic.sampling.gates import compute_adaptive_gates

    ligand_path = deps.validate_path_for_user(ligand_path, request, field="ligand_path")
    actives_csv = (
        deps.validate_path_for_user(actives_csv, request, field="actives_csv")
        if actives_csv
        else None
    )
    gates = compute_adaptive_gates(
        ligand_path,
        actives_csv=actives_csv if actives_csv and _P(actives_csv).is_file() else None,
    )
    return {
        "ligand_path": ligand_path,
        "gates": gates.as_prudent_dict(),
        "ligand": gates.ligand,
        "actives_stats": gates.actives_stats,
        "rationale": gates.rationale,
    }


@router.get("/api/scaffold/{job_id}")
def get_scaffold_status(
    job_id: str, request: Request, _auth: None = Depends(deps.verify_internal_key)
):
    return deps._require_job(job_id, request)


@router.post("/api/scaffold-cascade")
def start_scaffold_cascade(
    req: ScaffoldCascadeRequest,
    request: Request,
    _auth: None = Depends(deps.verify_internal_key),
    _rl: None = Depends(deps._rate_limit_submit),
):
    protein_path = (
        deps.validate_path_for_user(req.protein_path, request, field="protein_path")
        if req.protein_path
        else None
    )
    ligand_path = (
        deps.validate_path_for_user(req.ligand_path, request, field="ligand_path")
        if req.ligand_path
        else None
    )
    user = deps._request_username(request)
    scheduler = deps.get_scheduler()
    result = scheduler.submit_scaffold_cascade(
        data_id=deps._clamp(req.data_id, 0, 10_000_000, None),
        protein_path=protein_path,
        ligand_path=ligand_path,
        gpu_id=deps._selected_gpu(req.gpu_id, request),
        config_path=deps.validate_config_path(req.config_path),
        samples_per_round=deps._clamp(req.samples_per_round, 1, 100, 5),
        rounds=deps._clamp(req.rounds, 1, 50, 2),
        auto_evaluate=req.auto_evaluate,
        auto_extract=req.auto_extract,
        remove_fragments=req.remove_fragments,
        triggered_by=user,
        skip_daily_quota=deps._request_is_admin(request),
    )
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result


@router.get("/api/scaffold-cascade/{job_id}")
def get_scaffold_cascade_status(
    job_id: str, request: Request, _auth: None = Depends(deps.verify_internal_key)
):
    return deps._require_job(job_id, request)
