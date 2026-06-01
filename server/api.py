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
    init_db, create_run, update_run, get_run, list_runs,
    add_molecules, query_molecules, add_evaluation, get_evaluations,
    log_history, get_history,
)
from server.jobs import get_scheduler
from server.data_manager import (
    get_config_snapshot, write_config, scan_outputs, scan_eval_dirs,
    load_pt_metadata, get_gpu_info, list_proteins,
)

app = FastAPI(title="DiffDynamic", description="Paper demo web service for DiffDynamic")


@app.on_event("startup")
def startup():
    init_db()


# ── Request models ───────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    mode: str = "dynamic"
    data_id: Optional[int] = None
    use_test_set: bool = True
    gpu_id: Optional[int] = None
    batch_size: Optional[int] = None
    config_path: Optional[str] = None
    auto_evaluate: bool = False
    auto_extract: bool = False
    remove_fragments: bool = False


class CustomGenerateRequest(BaseModel):
    protein_path: str
    ligand_path: Optional[str] = None
    gpu_id: Optional[int] = None
    pocket_radius: float = 10.0
    num_samples: int = 100
    config_path: Optional[str] = None
    auto_evaluate: bool = True
    auto_extract: bool = True
    remove_fragments: bool = False


class EvaluateRequest(BaseModel):
    pt_path: str
    protein_root: Optional[str] = None
    vina_modes: str = "auto"
    max_samples: Optional[int] = None
    vina_timeout: int = 20


class ExtractRequest(BaseModel):
    pt_path: str
    protein_root: Optional[str] = None
    remove_fragments: bool = False


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
    remove_fragments: bool = False
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
    remove_fragments: bool = False
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
    remove_fragments: bool = False
    scaffold_mode: str = "grow"
    scaffold_source: str = "auto_murcko"
    scaffold_atom_indices: Optional[str] = None
    scaffold_smarts: Optional[str] = None
    fix_scaffold_pos: bool = True
    fix_scaffold_type: bool = True
    after_dynamic: bool = False
    qed_weight: float = 1.0
    sa_weight: float = 1.0
    diversity_weight: float = 0.5
    rmsd_penalty_weight: float = 0.0
    min_qed: float = 0.15
    min_sa: float = 0.15
    grow_num_samples: int = 200
    grow_start_t: int = 450
    grow_stride: int = 15
    grow_step_size: float = 0.33
    grow_lambda_a: int = 40
    grow_lambda_b: int = 5
    grow_n_extra_mode: str = "prior_minus_scaffold"
    grow_n_extra_fixed: int = 8
    grow_n_extra_min: int = 3
    grow_n_extra_max: int = 20
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



# ── Generation ───────────────────────────────────────────────────────────

@app.post("/api/generate")
def start_generation(req: GenerateRequest):
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
    )
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result


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
    rec = scheduler._jobs.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Job not found")
    vis_dir = rec.output_path
    images = []
    scores = {}
    dir_name = os.path.basename(vis_dir) if vis_dir else ""
    if vis_dir and os.path.isdir(vis_dir):
        for f in sorted(os.listdir(vis_dir)):
            if f.endswith(".png") and "_notext" not in f:
                images.append(f)
        # Parse scores from CSV if available
        cfg = get_config()
        csv_path = os.path.join(cfg.diffdynamic_root, "pocket_quality_vis", "evaluation_records.csv")
        if os.path.isfile(csv_path):
            import csv
            try:
                last_row = None
                with open(csv_path, encoding="utf-8") as cf:
                    for row in csv.DictReader(cf):
                        last_row = row
                if last_row:
                    scores = {
                        "score_a": last_row.get("score_a", "N/A"),
                        "score_b": last_row.get("score_b", "N/A"),
                        "score_c": last_row.get("score_c", "N/A"),
                        "score_d": last_row.get("score_d", "N/A"),
                        "score_e": last_row.get("score_e", "N/A"),
                        "score_f": last_row.get("score_f", "N/A"),
                        "score_g": last_row.get("score_g", "N/A"),
                        "score_h": last_row.get("score_h", "N/A"),
                        "overall_score": last_row.get("overall_score", "N/A"),
                        "overall_label": last_row.get("overall_label", "unknown"),
                    }
            except Exception:
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
    )
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result


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


@app.get("/api/runs/{run_id}")
def get_run_detail(run_id: int):
    r = get_run(run_id)
    if r is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "id": r.id, "run_type": r.run_type, "status": r.status,
        "parameters": json.loads(r.parameters) if r.parameters else None,
        "config_snapshot": json.loads(r.config_snapshot) if r.config_snapshot else None,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "started_at": r.started_at.isoformat() if r.started_at else None,
        "finished_at": r.finished_at.isoformat() if r.finished_at else None,
        "output_path": r.output_path, "error_message": r.error_message,
        "progress": r.progress, "progress_detail": r.progress_detail,
        "log_output": r.log_output,
    }


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
    mols = query_molecules(
        run_id=run_id, smiles_like=smiles, min_vina=min_vina,
        max_vina=max_vina, lipinski=lipinski, pocket_id=protein,
        limit=limit, offset=offset,
    )
    return [
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
    ]


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
    sdf_store = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DiffDynamic", "outputs", "sdf_store")
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
def get_readme():
    readme_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")
    try:
        with open(readme_path, encoding="utf-8") as f:
            return {"content": f.read()}
    except Exception:
        return {"content": "# DiffDynamic\n\nREADME.md not found."}


# ── Static files & index page ──────────────────────────────────────────────

_ui_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui")

app.mount("/static", StaticFiles(directory=os.path.join(_ui_root, "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    index_path = os.path.join(_ui_root, "templates", "index.html")
    with open(index_path) as f:
        return f.read()
