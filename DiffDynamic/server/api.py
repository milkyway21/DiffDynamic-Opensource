"""FastAPI REST endpoints for DiffDynamic web service.

All operations go through the job scheduler (async) or database (sync reads).
"""

import json
import os
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from server.config import get_config, save_config, ServerConfig
from server.database import (
    init_db, create_run, update_run, get_run, list_runs, run_to_dict,
    add_molecules, query_molecules, add_evaluation, get_evaluations,
    log_history, get_history, compare_runs, get_dashboard_stats, count_molecules,
    reconcile_stale_runs,
)
from server.jobs import get_scheduler
from server.data_manager import (
    get_config_snapshot, write_config, scan_outputs, scan_eval_dirs,
    load_pt_metadata, get_gpu_info, list_proteins, browse_directory,
)

app = FastAPI(title="DiffDynamic", description="Paper demo web service for DiffDynamic")


@app.on_event("startup")
def startup():
    init_db()
    n = reconcile_stale_runs(reason="server_restart")
    if n:
        print(f"[startup] reconciled {n} stale run(s) -> failed")


# ── Request models ───────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    mode: str = "dynamic"
    data_id: Optional[int] = None
    use_test_set: bool = True
    gpu_id: Optional[int] = None
    batch_size: Optional[int] = 5
    config_path: Optional[str] = None
    auto_evaluate: bool = True
    auto_extract: bool = True
    remove_fragments: bool = True
    max_samples: Optional[int] = 5
    vina_timeout: int = 20


class BatchGenerateRequest(BaseModel):
    start_id: int = 0
    end_id: int = 0
    gpus: Optional[str] = None
    batch_size: Optional[int] = 5
    config_path: Optional[str] = None
    auto_evaluate: bool = True
    sample_only: bool = False


class RerunRequest(BaseModel):
    overrides: Optional[dict] = None
    triggered_by: str = "web_ui"


class CustomGenerateRequest(BaseModel):
    protein_path: str
    ligand_path: Optional[str] = None
    gpu_id: Optional[int] = None
    pocket_radius: float = 10.0
    num_samples: int = 5
    config_path: Optional[str] = None
    auto_evaluate: bool = True
    auto_extract: bool = True
    remove_fragments: bool = True


class EvaluateRequest(BaseModel):
    pt_path: str
    protein_root: Optional[str] = None
    vina_modes: str = "auto"
    max_samples: Optional[int] = None
    vina_timeout: int = 20


class ExtractRequest(BaseModel):
    pt_path: str
    protein_root: Optional[str] = None
    remove_fragments: bool = True


class PocketEvalRequest(BaseModel):
    pt_path: Optional[str] = None
    protein_path: str
    ligand_path: str
    generate_first: bool = False
    gpu_id: Optional[int] = None
    batch_size: Optional[int] = 5
    pocket_radius: float = 10.0
    max_samples: Optional[int] = None


class OptimizationRequest(BaseModel):
    data_id: Optional[int] = None
    protein_path: Optional[str] = None
    ligand_path: Optional[str] = None
    molecule_path: Optional[str] = None
    use_test_set: bool = True
    gpu_id: Optional[int] = None
    config_path: Optional[str] = None
    auto_evaluate: bool = False
    auto_extract: bool = False
    remove_fragments: bool = True
    num_samples: int = 2
    start_t: int = 16
    stride: int = 2
    step_size: float = 0.2
    cycles: int = 5
    schedule: str = "linear"
    after_dynamic: bool = False
    min_qed: float = 0.3
    min_sa: float = 0.3
    max_survivors_per_cycle: int = 10
    max_survivors_total: int = 300


class ScaffoldCascadeRequest(BaseModel):
    data_id: Optional[int] = None
    protein_path: Optional[str] = None
    ligand_path: Optional[str] = None
    gpu_id: Optional[int] = None
    config_path: Optional[str] = None
    auto_evaluate: bool = False
    auto_extract: bool = False
    remove_fragments: bool = True
    samples_per_round: int = 5
    rounds: int = 2


class ScaffoldRequest(BaseModel):
    data_id: Optional[int] = None
    protein_path: Optional[str] = None
    ligand_path: Optional[str] = None
    molecule_path: Optional[str] = None
    use_test_set: bool = True
    gpu_id: Optional[int] = None
    config_path: Optional[str] = None
    auto_evaluate: bool = False
    auto_extract: bool = False
    remove_fragments: bool = True
    scaffold_mode: str = "grow"  # grow | evolve | dynamic_locked | prudent
    scaffold_source: str = "auto_murcko"
    scaffold_atom_indices: Optional[str] = None
    scaffold_smarts: Optional[str] = None
    fix_scaffold_pos: bool = True
    fix_scaffold_type: bool = True
    after_dynamic: bool = False
    # TargetDiff baseline 结构修复（start_t=9 → 10 步），默认开启
    scaffold_enable_refine: bool = True
    qed_weight: float = 1.0
    sa_weight: float = 1.0
    diversity_weight: float = 0.5
    rmsd_penalty_weight: float = 0.0
    min_qed: float = 0.15
    min_sa: float = 0.15
    grow_num_samples: int = 200
    grow_start_t: int = 357  # 100口袋最优(贝叶斯Round18)
    grow_stride: int = 15
    grow_step_size: float = 0.3262  # 100口袋最优(贝叶斯Round18)
    grow_lambda_a: int = 47  # 100口袋最优(贝叶斯Round18)
    grow_lambda_b: int = 11  # 100口袋最优(贝叶斯Round18)
    grow_n_extra_mode: str = "pocket_prior"  # 100口袋最优: +7.4%综合分
    grow_n_extra_fixed: int = 8
    grow_n_extra_min: int = 3
    grow_n_extra_max: int = 20
    grow_filter_incomplete: bool = False  # 100口袋最优: False(避免有效率暴跌)
    grow_min_qed: float = 0.2  # 100口袋最优grow阈值
    grow_min_sa: float = 0.2  # 100口袋最优grow阈值
    evolve_population_size: int = 50
    evolve_n_generations: int = 5
    evolve_children_per_parent: int = 10
    evolve_start_t_high: int = 200
    evolve_start_t_low: int = 30
    evolve_stride: int = 2
    evolve_step_size: float = 0.2
    evolve_lambda_a: int = 10
    evolve_lambda_b: int = 1
    diversity_filter_enable: bool = True
    max_tanimoto: float = 0.9
    # Scaffold-Prudent
    prudent_n_generations: int = 5
    prudent_n_chains_per_seed: int = 4
    prudent_advance_top_k: int = 2
    prudent_renoise_t: int = 600
    prudent_qed_weight: float = 0.2
    prudent_sa_weight: float = 0.2
    prudent_vina_weight: float = 0.6
    prudent_lipinski_weight: float = 0.0
    prudent_lilly_weight: float = 0.0
    prudent_vina_exhaustiveness: int = 8
    prudent_min_qed_for_docking: float = 0.15
    prudent_min_sa_for_docking: float = 0.15
    # 门控模式：adaptive=按参考配体自适应；manual=使用下方人工门控数值
    gate_mode: str = "adaptive"  # adaptive | manual
    adaptive_gates_enable: Optional[bool] = None  # 兼容旧字段；None 时由 gate_mode 决定
    adaptive_gates_actives: Optional[str] = None
    # 人工门控（gate_mode=manual 时写入 prudent）
    manual_max_logp: Optional[float] = 5.0
    manual_max_molwt: Optional[float] = 850.0
    manual_max_heavy_atoms: Optional[int] = 60
    manual_max_rings: Optional[int] = 7



# ── Generation ───────────────────────────────────────────────────────────

@app.post("/api/generate")
def start_generation(req: GenerateRequest, request: Request):
    user = request.headers.get("X-DD-User", "web_ui")
    scheduler = get_scheduler()
    result = scheduler.submit_generation(
        mode=req.mode,
        data_id=req.data_id,
        use_test_set=req.use_test_set,
        gpu_id=req.gpu_id,
        batch_size=req.batch_size,
        config_path=req.config_path,
        auto_evaluate=req.auto_evaluate,
        auto_extract=req.auto_extract,
        remove_fragments=req.remove_fragments,
        max_samples=req.max_samples,
        vina_timeout=req.vina_timeout,
        triggered_by=user,
    )
    return result


@app.post("/api/generate/batch")
def start_batch_generation(req: BatchGenerateRequest, request: Request):
    user = request.headers.get("X-DD-User", "web_ui")
    scheduler = get_scheduler()
    return scheduler.submit_batch_generation(
        start_id=req.start_id, end_id=req.end_id, gpus=req.gpus,
        batch_size=req.batch_size, config_path=req.config_path,
        auto_evaluate=req.auto_evaluate, sample_only=req.sample_only,
        triggered_by=user,
    )


@app.post("/api/generate/custom")
def start_custom_generation(req: CustomGenerateRequest):
    """Generate molecules from a custom protein PDB (and optional ligand SDF)."""
    scheduler = get_scheduler()
    result = scheduler.submit_custom_generation(
        protein_path=req.protein_path,
        ligand_path=req.ligand_path,
        gpu_id=req.gpu_id,
        pocket_radius=req.pocket_radius,
        num_samples=req.num_samples,
        config_path=req.config_path,
        auto_evaluate=req.auto_evaluate,
        auto_extract=req.auto_extract,
        remove_fragments=req.remove_fragments,
    )
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result


@app.get("/api/generate/{job_id}")
def get_generation_status(job_id: str):
    scheduler = get_scheduler()
    status = scheduler.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


# ── Evaluation ───────────────────────────────────────────────────────────

@app.post("/api/evaluate")
def start_evaluation(req: EvaluateRequest):
    scheduler = get_scheduler()
    result = scheduler.submit_evaluation(
        pt_path=req.pt_path,
        protein_root=req.protein_root,
        vina_modes=req.vina_modes,
        max_samples=req.max_samples,
        vina_timeout=req.vina_timeout,
    )
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result


@app.get("/api/evaluate/{job_id}")
def get_evaluation_status(job_id: str):
    scheduler = get_scheduler()
    status = scheduler.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


# ── Extraction ───────────────────────────────────────────────────────────

@app.post("/api/extract")
def start_extraction(req: ExtractRequest):
    scheduler = get_scheduler()
    result = scheduler.submit_extraction(pt_path=req.pt_path, protein_root=req.protein_root, remove_fragments=req.remove_fragments)
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result


# ── Pocket Evaluation ────────────────────────────────────────────────────

@app.post("/api/pocket-eval")
def start_pocket_eval(req: PocketEvalRequest):
    if not req.protein_path or not req.ligand_path:
        raise HTTPException(status_code=400, detail="protein_path and ligand_path are required")
    if not req.generate_first and not req.pt_path:
        raise HTTPException(status_code=400, detail="pt_path is required when generate_first is false")
    scheduler = get_scheduler()
    result = scheduler.submit_pocket_eval(
        pt_path=req.pt_path,
        protein_path=req.protein_path,
        ligand_path=req.ligand_path,
        generate_first=req.generate_first,
        gpu_id=req.gpu_id,
        batch_size=req.batch_size,
        pocket_radius=req.pocket_radius,
        max_samples=req.max_samples,
    )
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result


@app.get("/api/pocket-eval/images/{path:path}")
def serve_pocket_eval_image(path: str):
    cfg = get_config()
    vis_root = os.path.realpath(os.path.join(cfg.diffdynamic_root, "pocket_quality_vis"))
    abs_path = os.path.realpath(os.path.join(vis_root, path))
    if not abs_path.startswith(vis_root + os.sep) and abs_path != vis_root:
        raise HTTPException(status_code=403, detail="Path outside allowed directory")
    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(abs_path, media_type="image/png")


@app.get("/api/pocket-eval/results/{job_id}")
def get_pocket_eval_results(job_id: str):
    scheduler = get_scheduler()
    rec = scheduler.get_job_record(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Job not found")
    vis_dir = rec.output_path
    images = []
    scores = dict(rec.scores or {})
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
            "ligand_efficiency": "score_c",
            "drug_likeness": "score_d",
            "completeness": "score_e",
            "diversity": "score_f",
            "size_consistency": "score_g",
            "pocket_volume": "score_h",
            "interaction": "score_g",  # legacy misnomer
            "stability": "score_h",    # legacy misnomer
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
        evals = get_evaluations(rec.run_id)
        for e in evals:
            key = None
            if e.details:
                try:
                    details = json.loads(e.details) if isinstance(e.details, str) else e.details
                    if isinstance(details, dict) and details.get("dim"):
                        key = details["dim"]
                except (TypeError, ValueError):
                    pass
            if not key:
                key = metric_to_score.get(e.metric_name)
            if key:
                scores[key] = e.metric_value if e.metric_value is not None else e.details
            if e.metric_name == "overall_pocket_score" and e.details:
                try:
                    details = json.loads(e.details) if isinstance(e.details, str) else e.details
                    if isinstance(details, dict) and details.get("label"):
                        scores["overall_label"] = details["label"]
                except (TypeError, ValueError):
                    pass
    return {"job_id": job_id, "vis_dir": vis_dir, "dir_name": dir_name, "images": images, "scores": scores}


# ── Optimization ─────────────────────────────────────────────────────────

@app.post("/api/optimization")
def start_optimization(req: OptimizationRequest):
    scheduler = get_scheduler()
    result = scheduler.submit_optimization(
        data_id=req.data_id,
        protein_path=req.protein_path,
        ligand_path=req.ligand_path,
        molecule_path=req.molecule_path,
        use_test_set=req.use_test_set,
        gpu_id=req.gpu_id,
        config_path=req.config_path,
        auto_evaluate=req.auto_evaluate,
        auto_extract=req.auto_extract,
        remove_fragments=req.remove_fragments,
        num_samples=req.num_samples,
        start_t=req.start_t,
        stride=req.stride,
        step_size=req.step_size,
        cycles=req.cycles,
        schedule=req.schedule,
        after_dynamic=req.after_dynamic,
        min_qed=req.min_qed,
        min_sa=req.min_sa,
        max_survivors_per_cycle=req.max_survivors_per_cycle,
        max_survivors_total=req.max_survivors_total,
    )
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result


@app.get("/api/optimization/{job_id}")
def get_optimization_status(job_id: str):
    scheduler = get_scheduler()
    status = scheduler.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


# ── Scaffold ──────────────────────────────────────────────────────────────

@app.post("/api/scaffold")
def start_scaffold(req: ScaffoldRequest):
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

    scheduler = get_scheduler()
    result = scheduler.submit_scaffold(
        data_id=req.data_id,
        protein_path=req.protein_path,
        ligand_path=req.ligand_path,
        molecule_path=req.molecule_path,
        use_test_set=req.use_test_set,
        gpu_id=req.gpu_id,
        config_path=req.config_path,
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
        qed_weight=req.qed_weight,
        sa_weight=req.sa_weight,
        diversity_weight=req.diversity_weight,
        rmsd_penalty_weight=req.rmsd_penalty_weight,
        min_qed=req.min_qed,
        min_sa=req.min_sa,
        grow_num_samples=req.grow_num_samples,
        grow_start_t=req.grow_start_t,
        grow_stride=req.grow_stride,
        grow_step_size=req.grow_step_size,
        grow_lambda_a=req.grow_lambda_a,
        grow_lambda_b=req.grow_lambda_b,
        grow_n_extra_mode=req.grow_n_extra_mode,
        grow_n_extra_fixed=req.grow_n_extra_fixed,
        grow_n_extra_min=req.grow_n_extra_min,
        grow_n_extra_max=req.grow_n_extra_max,
        grow_filter_incomplete=req.grow_filter_incomplete,
        grow_min_qed=req.grow_min_qed,
        grow_min_sa=req.grow_min_sa,
        evolve_population_size=req.evolve_population_size,
        evolve_n_generations=req.evolve_n_generations,
        evolve_children_per_parent=req.evolve_children_per_parent,
        evolve_start_t_high=req.evolve_start_t_high,
        evolve_start_t_low=req.evolve_start_t_low,
        evolve_stride=req.evolve_stride,
        evolve_step_size=req.evolve_step_size,
        evolve_lambda_a=req.evolve_lambda_a,
        evolve_lambda_b=req.evolve_lambda_b,
        diversity_filter_enable=req.diversity_filter_enable,
        max_tanimoto=req.max_tanimoto,
        prudent_n_generations=req.prudent_n_generations,
        prudent_n_chains_per_seed=req.prudent_n_chains_per_seed,
        prudent_advance_top_k=req.prudent_advance_top_k,
        prudent_renoise_t=req.prudent_renoise_t,
        prudent_qed_weight=req.prudent_qed_weight,
        prudent_sa_weight=req.prudent_sa_weight,
        prudent_vina_weight=req.prudent_vina_weight,
        prudent_lipinski_weight=req.prudent_lipinski_weight,
        prudent_lilly_weight=req.prudent_lilly_weight,
        prudent_vina_exhaustiveness=req.prudent_vina_exhaustiveness,
        prudent_min_qed_for_docking=req.prudent_min_qed_for_docking,
        prudent_min_sa_for_docking=req.prudent_min_sa_for_docking,
        adaptive_gates_enable=adaptive_enable,
        adaptive_gates_actives=req.adaptive_gates_actives,
        gate_mode=gm,
        manual_max_logp=req.manual_max_logp,
        manual_max_molwt=req.manual_max_molwt,
        manual_max_heavy_atoms=req.manual_max_heavy_atoms,
        manual_max_rings=req.manual_max_rings,
    )
    if "error" in result:
        detail = result["error"]
        code = 400 if "Invalid scaffold_mode" in detail else 503
        raise HTTPException(status_code=code, detail=detail)
    return result


@app.get("/api/adaptive_gates/preview")
def preview_adaptive_gates(
    ligand_path: str,
    actives_csv: Optional[str] = None,
):
    """根据原始参考配体预览自适应门控（任务创建前可调用）。"""
    from pathlib import Path as _P
    from utils.adaptive_gates import compute_adaptive_gates

    if not _P(ligand_path).is_file():
        raise HTTPException(status_code=400, detail=f"ligand not found: {ligand_path}")
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


@app.get("/api/scaffold/{job_id}")
def get_scaffold_status(job_id: str):
    scheduler = get_scheduler()
    status = scheduler.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


# ── Scaffold Cascade ───────────────────────────────────────────────────────

@app.post("/api/scaffold-cascade")
def start_scaffold_cascade(req: ScaffoldCascadeRequest):
    scheduler = get_scheduler()
    result = scheduler.submit_scaffold_cascade(
        data_id=req.data_id,
        protein_path=req.protein_path,
        ligand_path=req.ligand_path,
        gpu_id=req.gpu_id,
        config_path=req.config_path,
        samples_per_round=req.samples_per_round,
        rounds=req.rounds,
        auto_evaluate=req.auto_evaluate,
        auto_extract=req.auto_extract,
        remove_fragments=req.remove_fragments,
    )
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result


@app.get("/api/scaffold-cascade/{job_id}")
def get_scaffold_cascade_status(job_id: str):
    scheduler = get_scheduler()
    status = scheduler.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


# ── Jobs ─────────────────────────────────────────────────────────────────

@app.get("/api/jobs")
def list_jobs(status: Optional[str] = Query(None)):
    scheduler = get_scheduler()
    return scheduler.list_jobs(status=status)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    scheduler = get_scheduler()
    status = scheduler.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@app.delete("/api/jobs/{job_id}")
def cancel_job(job_id: str):
    scheduler = get_scheduler()
    ok = scheduler.cancel(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found or not cancellable")
    return {"cancelled": job_id}


# ── Runs ─────────────────────────────────────────────────────────────────

@app.get("/api/runs")
def list_all_runs(
    run_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    runs = list_runs(run_type=run_type, status=status, limit=limit, offset=offset)
    return [
        {
            "id": r.id, "run_type": r.run_type, "status": r.status,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "finished_at": r.finished_at.isoformat() if r.finished_at else None,
            "output_path": r.output_path, "progress": r.progress,
            "has_log": bool(r.log_output),
        }
        for r in runs
    ]


@app.get("/api/runs/compare")
def compare_runs_api(ids: str = Query(..., description="Comma-separated run IDs")):
    run_ids = [int(x.strip()) for x in ids.split(",") if x.strip().isdigit()]
    if len(run_ids) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 run IDs")
    return compare_runs(run_ids)


@app.get("/api/runs/{run_id}")
def get_run_detail(run_id: int):
    r = get_run(run_id)
    if r is None:
        raise HTTPException(status_code=404, detail="Run not found")
    d = run_to_dict(r)
    d["log_output"] = r.log_output
    d["evaluations"] = [
        {"metric": e.metric_name, "value": e.metric_value,
         "details": json.loads(e.details) if e.details else None}
        for e in get_evaluations(run_id)
    ]
    return d


@app.post("/api/runs/{run_id}/rerun")
def rerun_run(run_id: int, req: RerunRequest, request: Request):
    user = request.headers.get("X-DD-User", req.triggered_by)
    scheduler = get_scheduler()
    result = scheduler.submit_rerun(run_id, overrides=req.overrides, triggered_by=user)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    log_history("rerun", {"source_run_id": run_id, "new_job": result.get("job_id")}, user=user)
    return result


@app.get("/api/runs/{run_id}/log")
def get_run_log(run_id: int):
    r = get_run(run_id)
    if r is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"run_id": run_id, "log": r.log_output or "", "status": r.status}


# ── Molecules ────────────────────────────────────────────────────────────

@app.get("/api/molecules")
def browse_molecules(
    run_id: Optional[int] = None,
    smiles: Optional[str] = None,
    protein: Optional[str] = None,
    min_vina: Optional[float] = None,
    max_vina: Optional[float] = None,
    lipinski: Optional[bool] = None,
    limit: int = 200,
    offset: int = 0,
):
    min_lip = 4 if lipinski else None
    mols = query_molecules(
        run_id=run_id, smiles_like=smiles, min_vina=min_vina,
        max_vina=max_vina, min_lipinski=min_lip, pocket_id=protein,
        limit=limit, offset=offset,
    )
    total = count_molecules(
        run_id=run_id, smiles_like=smiles, min_vina=min_vina,
        max_vina=max_vina, min_lipinski=min_lip, pocket_id=protein,
    )
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [
            {
                "id": m.id, "run_id": m.run_id, "pocket_id": m.pocket_id,
                "smiles": m.smiles,
                "vina_score": m.vina_score, "qed": m.qed, "sa": m.sa,
                "logp": m.logp, "tpsa": m.tpsa,
                "comprehensive_score": m.comprehensive_score,
                "lipinski_pass": m.lipinski_pass, "pains_pass": m.pains_pass,
                "lilly_passed": m.lilly_passed, "lilly_demerit": m.lilly_demerit,
                "lilly_description": m.lilly_description,
                "conformer_energy": m.conformer_energy,
                "rdkit_valid": m.rdkit_valid,
                "molecule_stable": m.molecule_stable,
                "n_heavy_atoms": m.n_heavy_atoms,
                "tanimoto": m.tanimoto,
                "sdf_path": m.sdf_path,
            }
            for m in mols
        ],
    }


@app.get("/api/stats")
def dashboard_stats():
    cfg = get_config()
    scheduler = get_scheduler()
    stats = get_dashboard_stats()
    stats["gpus"] = get_gpu_info()
    stats["queue_length"] = scheduler._queue_position()
    stats["active_jobs"] = len([j for j in scheduler.list_jobs("running")])
    stats["pending_jobs"] = len([j for j in scheduler.list_jobs("pending")])
    return stats


@app.get("/api/browse/files")
def browse_files(
    root: str = Query("outputs"),
    subpath: str = "",
    ext: Optional[str] = None,
):
    cfg = get_config()
    roots = {
        "outputs": cfg.output_root,
        "data": cfg.data_root,
        "configs": os.path.join(cfg.diffdynamic_root, "configs"),
    }
    if root not in roots:
        raise HTTPException(status_code=400, detail=f"root must be one of {list(roots.keys())}")
    extensions = tuple(e.strip() for e in ext.split(",")) if ext else None
    return browse_directory(roots[root], subpath, extensions)


# ── History ──────────────────────────────────────────────────────────────

@app.get("/api/history")
def browse_history(
    action: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    records = get_history(action=action, limit=limit, offset=offset)
    return [
        {
            "id": h.id, "action": h.action,
            "details": json.loads(h.details) if h.details else None,
            "user": h.user,
            "created_at": h.created_at.isoformat() if h.created_at else None,
        }
        for h in records
    ]


# ── Config ───────────────────────────────────────────────────────────────

@app.get("/api/config")
def read_config():
    cfg = get_config()
    return {
        "runtime": cfg.to_dict(),
        "sampling_yml": get_config_snapshot(cfg.sampling_config),
    }


@app.put("/api/config")
def update_config(data: dict):
    if "runtime" in data:
        cfg = ServerConfig(**{**get_config().to_dict(), **data["runtime"]})
        save_config(cfg)
    if "sampling_yml" in data:
        cfg = get_config()
        write_config(cfg.sampling_config, data["sampling_yml"])
    if "sampling_yml_text" in data:
        cfg = get_config()
        parsed = yaml.safe_load(data["sampling_yml_text"])
        write_config(cfg.sampling_config, parsed)
    log_history("config_updated", {"keys": list(data.keys())})
    return {"status": "ok"}


# ── Data browsing ────────────────────────────────────────────────────────

@app.get("/api/outputs")
def browse_outputs():
    cfg = get_config()
    return {
        "results": scan_outputs(cfg.output_root),
        "eval_dirs": scan_eval_dirs(cfg.output_root),
    }


@app.get("/api/pt/{path:path}")
def pt_metadata(path: str):
    cfg = get_config()
    abs_path = os.path.realpath(path)
    if not abs_path.startswith(os.path.realpath(cfg.output_root)) and \
       not abs_path.startswith(os.path.realpath(cfg.data_root)):
        raise HTTPException(status_code=403, detail="Path outside allowed directories")
    return load_pt_metadata(abs_path)


@app.get("/api/sdf/{path:path}")
def download_sdf(path: str):
    cfg = get_config()
    abs_path = os.path.realpath(path)
    # Allow sdf_store and output_root
    sdf_store = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "sdf_store")
    if not abs_path.startswith(os.path.realpath(sdf_store)) and \
       not abs_path.startswith(os.path.realpath(cfg.output_root)):
        raise HTTPException(status_code=403, detail="Path outside allowed directories")
    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="SDF file not found")
    return FileResponse(abs_path, media_type="chemical/x-mdl-sdfile", filename=os.path.basename(abs_path))


@app.get("/api/proteins")
def browse_proteins():
    cfg = get_config()
    return list_proteins(cfg.data_root)


@app.get("/api/gpus")
def gpu_status():
    return get_gpu_info()


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/readme")
def get_readme(lang: str = "zh"):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


# ── Static files & index page ──────────────────────────────────────────────

_ui_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui")

app.mount("/static", StaticFiles(directory=os.path.join(_ui_root, "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = os.path.join(_ui_root, "templates", "index.html")
    with open(index_path) as f:
        return f.read()
