"""Async job scheduler for GPU-bound DiffDynamic tasks.

Manages a thread pool, GPU allocation, subprocess lifecycle, and progress tracking.
All long-running operations (generation, evaluation, extraction) go through here.
"""

import copy
import itertools
import json
import logging
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml

from server.config import get_config
from server.database import create_run, update_run, log_history, add_molecules, add_evaluation
from server.molecule_ingest import ingest_molecules_from_eval

logger = logging.getLogger(__name__)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict (mutates base)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


class GPUAllocator:
    """Track which GPUs are busy and assign the next available one."""

    def __init__(self, gpu_ids: List[int]):
        self._available = set(gpu_ids)
        self._lock = threading.Lock()

    def acquire(self, preferred: Optional[int] = None) -> int:
        with self._lock:
            if preferred is not None and preferred in self._available:
                self._available.discard(preferred)
                return preferred
            if self._available:
                gpu = min(self._available)
                self._available.discard(gpu)
                return gpu
            return -1

    def release(self, gpu_id: int):
        with self._lock:
            self._available.add(gpu_id)

    @property
    def free_count(self) -> int:
        with self._lock:
            return len(self._available)


class JobRecord:
    """Internal state for a running or completed job."""

    __slots__ = (
        "job_id", "run_id", "job_type", "status", "gpu_id",
        "process", "future", "started_at", "finished_at",
        "output_path", "error", "progress", "log_lines",
    )

    def __init__(self, job_id: str, run_id: int, job_type: str, gpu_id: int):
        self.job_id = job_id
        self.run_id = run_id
        self.job_type = job_type
        self.status = "running"
        self.gpu_id = gpu_id
        self.process: Optional[subprocess.Popen] = None
        self.future: Optional[Future] = None
        self.started_at = datetime.now(timezone.utc)
        self.finished_at: Optional[datetime] = None
        self.output_path: Optional[str] = None
        self.error: Optional[str] = None
        self.progress: float = 0.0
        self.log_lines: List[str] = []

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "job_type": self.job_type,
            "status": self.status,
            "gpu_id": self.gpu_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "output_path": self.output_path,
            "error": self.error,
            "progress": self.progress,
            "log_tail": self.log_lines[-50:],
        }


class JobScheduler:
    """Thread-pool based job scheduler for DiffDynamic GPU tasks."""

    def __init__(self, max_concurrent: int = 2):
        cfg = get_config()
        self._gpu = GPUAllocator(cfg.gpu_ids)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._counter = itertools.count(1)

    def _next_job_id(self, prefix: str) -> str:
        with self._lock:
            n = next(self._counter)
        return f"{prefix}_{n:04d}_{int(time.time())}"

    # ── Public API ───────────────────────────────────────────────────────

    def submit_generation(
        self,
        mode: str = "dynamic",
        data_id: Optional[int] = 0,
        use_test_set: bool = True,
        gpu_id: Optional[int] = None,
        batch_size: Optional[int] = None,
        config_path: Optional[str] = None,
        auto_evaluate: bool = False,
        auto_extract: bool = False,
        remove_fragments: bool = False,
        triggered_by: str = "web_ui",
    ) -> dict:
        """Submit a generation job (dynamic or prudent). Returns job info."""
        cfg = get_config()
        gpu = self._gpu.acquire(gpu_id)
        if gpu < 0:
            return {"error": "No GPU available"}

        parameters = {
            "mode": mode,
            "data_id": data_id,
            "use_test_set": use_test_set,
            "gpu_id": gpu,
            "batch_size": batch_size,
            "auto_evaluate": auto_evaluate,
            "auto_extract": auto_extract,
            "remove_fragments": remove_fragments,
        }

        run_id = create_run(
            run_type=f"generate_{mode}",
            parameters=parameters,
            config_snapshot=self._read_config_snapshot(config_path or cfg.sampling_config),
            triggered_by=triggered_by,
        )
        log_history("start_generation", {"run_id": run_id, "parameters": parameters})

        job_id = self._next_job_id(f"gen_{mode}")
        rec = JobRecord(job_id, run_id, f"generate_{mode}", gpu)

        future = self._executor.submit(self._run_generation, rec, parameters, config_path)
        rec.future = future

        with self._lock:
            self._jobs[job_id] = rec

        return {"job_id": job_id, "run_id": run_id, "gpu_id": gpu}

    def submit_custom_generation(
        self,
        protein_path: str,
        ligand_path: Optional[str] = None,
        gpu_id: Optional[int] = None,
        pocket_radius: float = 10.0,
        num_samples: int = 100,
        config_path: Optional[str] = None,
        auto_evaluate: bool = True,
        auto_extract: bool = True,
        remove_fragments: bool = False,
        triggered_by: str = "web_ui",
    ) -> dict:
        """Submit a custom pocket generation job."""
        cfg = get_config()
        gpu = self._gpu.acquire(gpu_id)
        if gpu < 0:
            return {"error": "No GPU available"}

        # Extract pocket name from protein filename (e.g., "shoc2" from "shoc2.pdb")
        pocket_name = os.path.splitext(os.path.basename(protein_path))[0]

        parameters = {
            "protein_path": protein_path,
            "ligand_path": ligand_path,
            "gpu_id": gpu,
            "pocket_radius": pocket_radius,
            "num_samples": num_samples,
            "auto_evaluate": auto_evaluate,
            "auto_extract": auto_extract,
            "remove_fragments": remove_fragments,
            "pocket_name": pocket_name,
        }
        run_id = create_run(run_type="generate_custom", parameters=parameters, triggered_by=triggered_by)
        log_history("start_custom_generation", {"run_id": run_id, "protein_path": protein_path})

        job_id = self._next_job_id("gen_custom")
        rec = JobRecord(job_id, run_id, "generate_custom", gpu)
        future = self._executor.submit(self._run_custom_generation, rec, parameters, config_path)
        rec.future = future

        with self._lock:
            self._jobs[job_id] = rec

        return {"job_id": job_id, "run_id": run_id, "gpu_id": gpu}

    def submit_evaluation(
        self,
        pt_path: str,
        protein_root: Optional[str] = None,
        vina_modes: str = "auto",
        max_samples: Optional[int] = None,
        vina_timeout: int = 20,
        pocket_name: Optional[str] = None,
        triggered_by: str = "web_ui",
    ) -> dict:
        """Submit an evaluation job for a .pt file."""
        cfg = get_config()
        gpu = self._gpu.acquire(0)
        if gpu < 0:
            return {"error": "No GPU available"}

        parameters = {
            "pt_path": pt_path,
            "protein_root": protein_root,
            "vina_modes": vina_modes,
            "max_samples": max_samples,
            "vina_timeout": vina_timeout,
            "pocket_name": pocket_name,
        }
        run_id = create_run(run_type="evaluate", parameters=parameters, triggered_by=triggered_by)
        log_history("start_evaluation", {"run_id": run_id, "pt_path": pt_path})

        job_id = self._next_job_id("eval")
        rec = JobRecord(job_id, run_id, "evaluate", gpu)
        future = self._executor.submit(self._run_evaluation, rec, parameters)
        rec.future = future

        with self._lock:
            self._jobs[job_id] = rec

        return {"job_id": job_id, "run_id": run_id}

    def submit_extraction(
        self,
        pt_path: str,
        protein_root: Optional[str] = None,
        eval_dir: Optional[str] = None,
        remove_fragments: bool = False,
        triggered_by: str = "web_ui",
    ) -> dict:
        """Submit a .pt → SDF/Excel extraction job."""
        parameters = {"pt_path": pt_path, "protein_root": protein_root, "eval_dir": eval_dir, "remove_fragments": remove_fragments}
        run_id = create_run(run_type="extract", parameters=parameters, triggered_by=triggered_by)
        log_history("start_extraction", {"run_id": run_id, "pt_path": pt_path})

        job_id = self._next_job_id("extract")
        rec = JobRecord(job_id, run_id, "extract", 0)
        future = self._executor.submit(self._run_extraction, rec, parameters)
        rec.future = future

        with self._lock:
            self._jobs[job_id] = rec

        return {"job_id": job_id, "run_id": run_id}

    def submit_pocket_eval(
        self,
        protein_path: str,
        ligand_path: str,
        pt_path: Optional[str] = None,
        generate_first: bool = False,
        gpu_id: Optional[int] = None,
        batch_size: Optional[int] = 5,
        pocket_radius: float = 10.0,
        max_samples: Optional[int] = None,
        triggered_by: str = "web_ui",
    ) -> dict:
        """Submit a pocket quality evaluation job."""
        cfg = get_config()
        gpu = self._gpu.acquire(gpu_id)
        if gpu < 0:
            return {"error": "No GPU available"}

        pocket_name = os.path.splitext(os.path.basename(protein_path))[0]
        parameters = {
            "pt_path": pt_path,
            "protein_path": protein_path,
            "ligand_path": ligand_path,
            "generate_first": generate_first,
            "gpu_id": gpu,
            "batch_size": batch_size,
            "pocket_radius": pocket_radius,
            "max_samples": max_samples,
            "pocket_name": pocket_name,
        }
        run_id = create_run(run_type="pocket_eval", parameters=parameters, triggered_by=triggered_by)
        log_history("start_pocket_eval", {"run_id": run_id, "protein_path": protein_path})

        job_id = self._next_job_id("pocket_eval")
        rec = JobRecord(job_id, run_id, "pocket_eval", gpu)
        future = self._executor.submit(self._run_pocket_evaluation, rec, parameters)
        rec.future = future

        with self._lock:
            self._jobs[job_id] = rec

        return {"job_id": job_id, "run_id": run_id, "gpu_id": gpu}

    # ── Optimization & Scaffold ──────────────────────────────────────────

    def _write_patched_config(self, base_config_path: str, patches: dict, job_id: str) -> str:
        """Read base YAML, deep-merge patches, write to a temp config file."""
        with open(base_config_path) as f:
            base = yaml.safe_load(f)
        _deep_merge(base, patches)
        cfg = get_config()
        out_dir = os.path.join(cfg.diffdynamic_root, "configs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"run_{job_id}.yml")
        with open(out_path, "w") as f:
            yaml.dump(base, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return out_path

    def submit_optimization(
        self,
        data_id: Optional[int] = None,
        protein_path: Optional[str] = None,
        ligand_path: Optional[str] = None,
        molecule_path: Optional[str] = None,
        use_test_set: bool = True,
        gpu_id: Optional[int] = None,
        config_path: Optional[str] = None,
        auto_evaluate: bool = False,
        auto_extract: bool = False,
        remove_fragments: bool = False,
        # Optimization params
        num_samples: int = 2,
        start_t: int = 16,
        stride: int = 2,
        step_size: float = 0.2,
        cycles: int = 5,
        schedule: str = "linear",
        after_dynamic: bool = False,
        min_qed: float = 0.3,
        min_sa: float = 0.3,
        max_survivors_per_cycle: int = 10,
        max_survivors_total: int = 300,
        triggered_by: str = "web_ui",
    ) -> dict:
        """Submit a molecular optimization job."""
        cfg = get_config()
        gpu = self._gpu.acquire(gpu_id)
        if gpu < 0:
            return {"error": "No GPU available"}

        parameters = {
            "data_id": data_id, "protein_path": protein_path,
            "ligand_path": ligand_path, "molecule_path": molecule_path,
            "use_test_set": use_test_set, "gpu_id": gpu,
            "auto_evaluate": auto_evaluate, "auto_extract": auto_extract,
            "remove_fragments": remove_fragments,
            "num_samples": num_samples, "start_t": start_t,
            "stride": stride, "step_size": step_size,
            "cycles": cycles, "schedule": schedule,
            "after_dynamic": after_dynamic,
            "min_qed": min_qed, "min_sa": min_sa,
            "max_survivors_per_cycle": max_survivors_per_cycle,
            "max_survivors_total": max_survivors_total,
        }

        # Build YAML patches
        opt_patches = {
            "sample": {
                "optimization": {
                    "enable": True,
                    "after_dynamic": after_dynamic,
                    "molecule_path": molecule_path,
                    "num_samples": num_samples,
                    "keep_original": True,
                    "start_t": start_t,
                    "stride": stride,
                    "step_size": step_size,
                    "noise_scale": 0,
                    "schedule": schedule,
                    "cycles": cycles,
                    "use_with_noise": True,
                    "use_adaptive_step": True,
                    "use_time_scale": False,
                    "lambda_coeff_a": 1,
                    "lambda_coeff_b": 1,
                    "selector": {
                        "enable_filter": True,
                        "filter_incomplete": True,
                        "max_survivors_per_cycle": max_survivors_per_cycle,
                        "max_survivors_total": max_survivors_total,
                        "min_qed": min_qed,
                        "min_sa": min_sa,
                        "qed_weight": 1,
                        "sa_weight": 1,
                    },
                },
            },
        }

        base_cfg = self._validate_config_path(config_path, cfg)
        job_id = self._next_job_id("opt")
        patched_config = self._write_patched_config(base_cfg, opt_patches, job_id)
        parameters["patched_config"] = patched_config

        run_id = create_run(run_type="optimization", parameters=parameters,
                           config_snapshot=self._read_config_snapshot(base_cfg),
                           triggered_by=triggered_by)
        log_history("start_optimization", {"run_id": run_id, "data_id": data_id})

        rec = JobRecord(job_id, run_id, "optimization", gpu)
        future = self._executor.submit(self._run_optimization, rec, parameters)
        rec.future = future

        with self._lock:
            self._jobs[job_id] = rec

        return {"job_id": job_id, "run_id": run_id, "gpu_id": gpu}

    def submit_scaffold(
        self,
        data_id: Optional[int] = None,
        protein_path: Optional[str] = None,
        ligand_path: Optional[str] = None,
        molecule_path: Optional[str] = None,
        use_test_set: bool = True,
        gpu_id: Optional[int] = None,
        config_path: Optional[str] = None,
        auto_evaluate: bool = False,
        auto_extract: bool = False,
        remove_fragments: bool = False,
        # Scaffold common
        scaffold_mode: str = "grow",
        scaffold_source: str = "auto_murcko",
        scaffold_atom_indices: Optional[str] = None,
        scaffold_smarts: Optional[str] = None,
        fix_scaffold_pos: bool = True,
        fix_scaffold_type: bool = True,
        after_dynamic: bool = False,
        qed_weight: float = 1.0,
        sa_weight: float = 1.0,
        diversity_weight: float = 0.5,
        rmsd_penalty_weight: float = 0.0,
        min_qed: float = 0.15,
        min_sa: float = 0.15,
        # Grow params
        grow_num_samples: int = 200,
        grow_start_t: int = 450,
        grow_stride: int = 15,
        grow_step_size: float = 0.33,
        grow_lambda_a: int = 40,
        grow_lambda_b: int = 5,
        grow_n_extra_mode: str = "prior_minus_scaffold",
        grow_n_extra_fixed: int = 8,
        grow_n_extra_min: int = 3,
        grow_n_extra_max: int = 20,
        # Evolve params
        evolve_population_size: int = 50,
        evolve_n_generations: int = 5,
        evolve_children_per_parent: int = 10,
        evolve_start_t_high: int = 200,
        evolve_start_t_low: int = 30,
        evolve_stride: int = 2,
        evolve_step_size: float = 0.2,
        evolve_lambda_a: int = 10,
        evolve_lambda_b: int = 1,
        # Diversity filter
        diversity_filter_enable: bool = True,
        max_tanimoto: float = 0.9,
        triggered_by: str = "web_ui",
    ) -> dict:
        """Submit a scaffold-constrained generation job (grow or evolve)."""
        cfg = get_config()
        gpu = self._gpu.acquire(gpu_id)
        if gpu < 0:
            return {"error": "No GPU available"}

        parameters = {
            "data_id": data_id, "protein_path": protein_path,
            "ligand_path": ligand_path, "molecule_path": molecule_path,
            "use_test_set": use_test_set, "gpu_id": gpu,
            "auto_evaluate": auto_evaluate, "auto_extract": auto_extract,
            "remove_fragments": remove_fragments,
            "scaffold_mode": scaffold_mode, "scaffold_source": scaffold_source,
        }

        # Build scaffold config
        scaffold_cfg = {
            "enable": True,
            "mode": scaffold_mode,
            "after_dynamic": after_dynamic,
            "save_dynamic_before_scaffold": True,
            "optimization_style_naming": True,
            "scaffold_source": scaffold_source,
            "scaffold_atom_indices": scaffold_atom_indices,
            "scaffold_smarts": scaffold_smarts,
            "fix_scaffold_pos": fix_scaffold_pos,
            "fix_scaffold_type": fix_scaffold_type,
            "schedule": "lambda",
            "use_with_noise": True,
            "use_adaptive_step": True,
            "use_time_scale": False,
            "qed_weight": qed_weight,
            "sa_weight": sa_weight,
            "diversity_weight": diversity_weight,
            "rmsd_penalty_weight": rmsd_penalty_weight,
            "min_qed": min_qed,
            "min_sa": min_sa,
            "keep_original": True,
            "filter_incomplete": True,
            "diversity_filter": {
                "enable": diversity_filter_enable,
                "max_tanimoto": max_tanimoto,
                "fingerprint_radius": 2,
                "fingerprint_nbits": 2048,
            },
        }

        if scaffold_mode == "grow":
            scaffold_cfg["grow"] = {
                "n_extra_mode": grow_n_extra_mode,
                "n_extra_fixed": grow_n_extra_fixed,
                "n_extra_min": grow_n_extra_min,
                "n_extra_max": grow_n_extra_max,
                "n_extra_min_clamp": 1,
                "n_extra_max_clamp": 45,
                "num_samples": grow_num_samples,
                "start_t": grow_start_t,
                "stride": grow_stride,
                "step_size": grow_step_size,
                "lambda_coeff_a": grow_lambda_a,
                "lambda_coeff_b": grow_lambda_b,
                "min_qed": min_qed,
                "min_sa": min_sa,
                "filter_incomplete": False,
            }
        else:  # evolve
            scaffold_cfg["evolve"] = {
                "population_size": evolve_population_size,
                "n_generations": evolve_n_generations,
                "children_per_parent": evolve_children_per_parent,
                "start_t_high": evolve_start_t_high,
                "start_t_low": evolve_start_t_low,
                "noise_anneal": True,
                "stride": evolve_stride,
                "step_size": evolve_step_size,
                "lambda_coeff_a": evolve_lambda_a,
                "lambda_coeff_b": evolve_lambda_b,
            }

        sc_patches = {"sample": {"scaffold": scaffold_cfg}}

        base_cfg = self._validate_config_path(config_path, cfg)
        job_id = self._next_job_id(f"sc_{scaffold_mode}")
        patched_config = self._write_patched_config(base_cfg, sc_patches, job_id)
        parameters["patched_config"] = patched_config

        run_id = create_run(run_type=f"scaffold_{scaffold_mode}", parameters=parameters,
                           config_snapshot=self._read_config_snapshot(base_cfg),
                           triggered_by=triggered_by)
        log_history("start_scaffold", {"run_id": run_id, "scaffold_mode": scaffold_mode})

        rec = JobRecord(job_id, run_id, f"scaffold_{scaffold_mode}", gpu)
        future = self._executor.submit(self._run_scaffold, rec, parameters)
        rec.future = future

        with self._lock:
            self._jobs[job_id] = rec

        return {"job_id": job_id, "run_id": run_id, "gpu_id": gpu}

    def submit_scaffold_cascade(
        self,
        data_id: Optional[int] = None,
        protein_path: Optional[str] = None,
        ligand_path: Optional[str] = None,
        gpu_id: Optional[int] = None,
        config_path: Optional[str] = None,
        samples_per_round: int = 5,
        rounds: int = 2,
        auto_evaluate: bool = False,
        auto_extract: bool = False,
        remove_fragments: bool = False,
        triggered_by: str = "web_ui",
    ) -> dict:
        """Submit a scaffold cascade pipeline job."""
        cfg = get_config()
        gpu = self._gpu.acquire(gpu_id)
        if gpu < 0:
            return {"error": "No GPU available"}

        parameters = {
            "data_id": data_id, "protein_path": protein_path,
            "ligand_path": ligand_path, "gpu_id": gpu,
            "samples_per_round": samples_per_round, "rounds": rounds,
            "config_path": config_path,
            "auto_evaluate": auto_evaluate, "auto_extract": auto_extract,
            "remove_fragments": remove_fragments,
        }

        run_id = create_run(run_type="scaffold_cascade", parameters=parameters,
                           triggered_by=triggered_by)
        log_history("start_scaffold_cascade", {"run_id": run_id, "rounds": rounds})

        job_id = self._next_job_id("cascade")
        rec = JobRecord(job_id, run_id, "scaffold_cascade", gpu)
        future = self._executor.submit(self._run_scaffold_cascade, rec, parameters)
        rec.future = future

        with self._lock:
            self._jobs[job_id] = rec

        return {"job_id": job_id, "run_id": run_id, "gpu_id": gpu}

    def get_status(self, job_id: str) -> Optional[dict]:
        with self._lock:
            rec = self._jobs.get(job_id)
        return rec.to_dict() if rec else None

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            rec = self._jobs.get(job_id)
        if rec is None or rec.status != "running":
            return False
        if rec.process and rec.process.poll() is None:
            rec.process.terminate()
            try:
                rec.process.wait(timeout=10)
            except Exception:
                rec.process.kill()
                rec.process.wait()
        rec.status = "cancelled"
        rec.finished_at = datetime.now(timezone.utc)
        update_run(rec.run_id, status="cancelled", finished_at=rec.finished_at)
        log_history("cancel_job", {"job_id": job_id})
        self._gpu.release(rec.gpu_id)
        return True

    def list_jobs(self, status: Optional[str] = None) -> List[dict]:
        with self._lock:
            jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return [j.to_dict() for j in sorted(jobs, key=lambda x: x.started_at or datetime.min, reverse=True)]

    # ── Internal runners ─────────────────────────────────────────────────

    def _run_generation(self, rec: JobRecord, params: dict, config_path: Optional[str]):
        cfg = get_config()
        try:
            update_run(rec.run_id, status="running", started_at=datetime.now(timezone.utc))

            # sample_diffusion.py CLI:
            #   config (positional) -i DATA_ID --device cuda:N --batch_size N
            #   --mode {baseline,dynamic,optimization,prudent}
            validated_config = self._validate_config_path(config_path, cfg)
            cmd = [
                cfg.python_bin, cfg.sample_script(),
                validated_config,
                "--device", f"cuda:{params['gpu_id']}",
            ]
            # data_id: use -i for single ID
            data_id = params.get("data_id")
            if data_id is not None:
                cmd += ["-i", str(data_id)]
            elif params.get("use_test_set"):
                cmd += ["--use_test_set"]
            # batch_size: controls molecules per run (not num_samples)
            if params.get("batch_size"):
                cmd += ["--batch_size", str(params["batch_size"])]
            # mode override
            if params.get("mode"):
                cmd += ["--mode", params["mode"]]

            env = os.environ.copy()
            env["PYTHONPATH"] = cfg.diffdynamic_root
            env["PYTHONUNBUFFERED"] = "1"

            rec.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=env, cwd=cfg.diffdynamic_root,
                bufsize=1,
            )

            _sync_counter = 0
            for line in rec.process.stdout:
                line = line.rstrip()
                if self._is_important_log(line):
                    rec.log_lines.append(line)
                self._parse_generation_progress(rec, line)
                _sync_counter += 1
                if _sync_counter % 20 == 0:
                    update_run(rec.run_id, progress=rec.progress,
                               log_output="\n".join(rec.log_lines[-500:]))

            rec.process.wait()
            rec.finished_at = datetime.now(timezone.utc)

            if rec.process.returncode == 0:
                rec.status = "completed"
                update_run(rec.run_id, status="completed", finished_at=rec.finished_at,
                           output_path=rec.output_path, progress=1.0,
                           log_output="\n".join(rec.log_lines[-500:]))
                log_history("generation_completed", {"job_id": rec.job_id, "run_id": rec.run_id})

                # Auto-chain: evaluate and/or extract after generation
                auto_eval = params.get("auto_evaluate", False)
                auto_ext = params.get("auto_extract", False)
                if (auto_eval or auto_ext) and rec.output_path:
                    logger.info(f"Auto-chain: eval={auto_eval}, extract={auto_ext}, pt={rec.output_path}")
                    if auto_eval:
                        eval_result = self.submit_evaluation(
                            pt_path=rec.output_path,
                            max_samples=5,
                            vina_timeout=20,
                            triggered_by="auto_chain",
                        )
                        if "error" not in eval_result:
                            log_history("auto_chain_evaluate", {"parent_job": rec.job_id, "eval_job": eval_result["job_id"]})
                    if auto_ext:
                        ext_result = self.submit_extraction(
                            pt_path=rec.output_path,
                            triggered_by="auto_chain",
                        )
                        if "error" not in ext_result:
                            log_history("auto_chain_extract", {"parent_job": rec.job_id, "extract_job": ext_result["job_id"]})
            else:
                rec.status = "failed"
                rec.error = f"Exit code {rec.process.returncode}"
                update_run(rec.run_id, status="failed", finished_at=rec.finished_at,
                           error_message=rec.error,
                           log_output="\n".join(rec.log_lines[-500:]))
                log_history("generation_failed", {"job_id": rec.job_id, "error": rec.error})

        except Exception as e:
            rec.status = "failed"
            rec.error = str(e)
            rec.finished_at = datetime.now(timezone.utc)
            update_run(rec.run_id, status="failed", finished_at=rec.finished_at, error_message=str(e))
            log_history("generation_failed", {"job_id": rec.job_id, "error": str(e)})
        finally:
            self._gpu.release(rec.gpu_id)

    def _run_custom_generation(self, rec: JobRecord, params: dict, config_path: Optional[str]):
        """Run generation from a custom protein PDB file."""
        cfg = get_config()
        try:
            update_run(rec.run_id, status="running", started_at=datetime.now(timezone.utc))

            # sample_diffusion.py --protein_path X --ligand_path Y --device cuda:N
            validated_config = self._validate_config_path(config_path, cfg)
            cmd = [
                cfg.python_bin, cfg.sample_script(),
                validated_config,
                "--protein_path", params["protein_path"],
                "--device", f"cuda:{params['gpu_id']}",
            ]
            if params.get("ligand_path"):
                cmd += ["--ligand_path", params["ligand_path"]]
            cmd += ["--pocket_radius", str(params.get("pocket_radius", 10.0))]

            env = os.environ.copy()
            env["PYTHONPATH"] = cfg.diffdynamic_root
            env["PYTHONUNBUFFERED"] = "1"

            rec.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=env, cwd=cfg.diffdynamic_root,
                bufsize=1,
            )

            _sync_counter = 0
            for line in rec.process.stdout:
                line = line.rstrip()
                if self._is_important_log(line):
                    rec.log_lines.append(line)
                self._parse_generation_progress(rec, line)
                _sync_counter += 1
                if _sync_counter % 20 == 0:
                    update_run(rec.run_id, progress=rec.progress,
                               log_output="\n".join(rec.log_lines[-500:]))

            rec.process.wait()
            rec.finished_at = datetime.now(timezone.utc)

            if rec.process.returncode == 0:
                rec.status = "completed"
                update_run(rec.run_id, status="completed", finished_at=rec.finished_at,
                           output_path=rec.output_path, progress=1.0,
                           log_output="\n".join(rec.log_lines[-500:]))
                log_history("custom_generation_completed", {"job_id": rec.job_id})
                # Auto-chain
                auto_eval = params.get("auto_evaluate", False)
                auto_ext = params.get("auto_extract", False)
                if (auto_eval or auto_ext) and rec.output_path:
                    if auto_eval:
                        self.submit_evaluation(pt_path=rec.output_path, max_samples=5, vina_timeout=20,
                                               pocket_name=params.get("pocket_name"), triggered_by="auto_chain")
                    if auto_ext:
                        self.submit_extraction(pt_path=rec.output_path, triggered_by="auto_chain")
            else:
                rec.status = "failed"
                rec.error = f"Exit code {rec.process.returncode}"
                update_run(rec.run_id, status="failed", finished_at=rec.finished_at,
                           error_message=rec.error,
                           log_output="\n".join(rec.log_lines[-500:]))

        except Exception as e:
            rec.status = "failed"
            rec.error = str(e)
            rec.finished_at = datetime.now(timezone.utc)
            update_run(rec.run_id, status="failed", finished_at=rec.finished_at, error_message=str(e))
        finally:
            self._gpu.release(rec.gpu_id)

    def _run_evaluation(self, rec: JobRecord, params: dict):
        cfg = get_config()
        try:
            update_run(rec.run_id, status="running", started_at=datetime.now(timezone.utc))

            cmd = [
                cfg.python_bin, cfg.evaluate_script(),
                params["pt_path"],
            ]
            if params.get("protein_root"):
                cmd += ["--protein_root", params["protein_root"]]
            if params.get("max_samples"):
                cmd += ["--max_samples", str(params["max_samples"])]
            if params.get("vina_timeout"):
                cmd += ["--vina-timeout-seconds", str(params["vina_timeout"])]
            if params.get("vina_modes"):
                cmd += ["--vina-modes", str(params["vina_modes"])]

            env = os.environ.copy()
            env["PYTHONPATH"] = cfg.diffdynamic_root
            env["PYTHONUNBUFFERED"] = "1"
            env["ADT_PYTHON"] = cfg.python_bin
            env["VINA_DOCK_TIMEOUT_SEC"] = str(params.get("vina_timeout", 20))

            rec.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=env, cwd=cfg.diffdynamic_root,
                bufsize=1,
            )

            _sync_counter = 0
            for line in rec.process.stdout:
                line = line.rstrip()
                if self._is_important_log(line):
                    rec.log_lines.append(line)
                self._parse_eval_progress(rec, line)
                # Sync logs to DB every 20 lines
                _sync_counter += 1
                if _sync_counter % 20 == 0:
                    update_run(rec.run_id, progress=rec.progress,
                               log_output="\n".join(rec.log_lines[-500:]))

            rec.process.wait()
            rec.finished_at = datetime.now(timezone.utc)

            if rec.process.returncode == 0:
                rec.status = "completed"
                update_run(rec.run_id, status="completed", finished_at=rec.finished_at,
                           output_path=rec.output_path, progress=1.0,
                           log_output="\n".join(rec.log_lines[-500:]))
                log_history("evaluation_completed", {"job_id": rec.job_id})
                # Ingest molecules into DB
                if rec.output_path:
                    try:
                        ingest_molecules_from_eval(rec.run_id, rec.output_path,
                                                   pocket_name=params.get("pocket_name"))
                        log_history("molecules_ingested", {"run_id": rec.run_id, "eval_dir": rec.output_path})
                    except Exception as e:
                        logger.warning(f"Failed to ingest molecules: {e}")
            else:
                rec.status = "failed"
                rec.error = f"Exit code {rec.process.returncode}"
                update_run(rec.run_id, status="failed", finished_at=rec.finished_at,
                           error_message=rec.error,
                           log_output="\n".join(rec.log_lines[-500:]))

        except Exception as e:
            rec.status = "failed"
            rec.error = str(e)
            rec.finished_at = datetime.now(timezone.utc)
            update_run(rec.run_id, status="failed", finished_at=rec.finished_at, error_message=str(e))
        finally:
            self._gpu.release(rec.gpu_id)

    def _run_extraction(self, rec: JobRecord, params: dict):
        cfg = get_config()
        try:
            update_run(rec.run_id, status="running", started_at=datetime.now(timezone.utc))

            cmd = [cfg.python_bin, cfg.extract_script()]
            if params.get("protein_root"):
                cmd += ["--protein_root", params["protein_root"]]
            if params.get("remove_fragments"):
                cmd.append("--remove-fragments")
            cmd.append(params["pt_path"])
            if params.get("eval_dir"):
                cmd += ["--eval_dir", params["eval_dir"]]

            env = os.environ.copy()
            env["PYTHONPATH"] = cfg.diffdynamic_root
            env["PYTHONUNBUFFERED"] = "1"
            env["ADT_PYTHON"] = cfg.python_bin

            rec.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=env, cwd=cfg.diffdynamic_root,
                bufsize=1,
            )

            _sync_counter = 0
            eval_dir = None
            for line in rec.process.stdout:
                line = line.rstrip()
                if self._is_important_log(line):
                    rec.log_lines.append(line)
                # Parse eval output directory
                if "评估输出目录:" in line and not eval_dir:
                    m = re.search(r"评估输出目录:\s*(/\S+)", line)
                    if m:
                        eval_dir = m.group(1)
                _sync_counter += 1
                if _sync_counter % 20 == 0:
                    update_run(rec.run_id, log_output="\n".join(rec.log_lines[-500:]))

            rec.process.wait()
            rec.finished_at = datetime.now(timezone.utc)

            if rec.process.returncode == 0:
                rec.status = "completed"
                update_run(rec.run_id, status="completed", finished_at=rec.finished_at,
                           log_output="\n".join(rec.log_lines[-500:]))
                log_history("extraction_completed", {"job_id": rec.job_id})
                # Ingest molecules into DB
                if eval_dir:
                    try:
                        ingest_molecules_from_eval(rec.run_id, eval_dir)
                        log_history("molecules_ingested", {"run_id": rec.run_id, "eval_dir": eval_dir})
                    except Exception as e:
                        logger.warning(f"Failed to ingest molecules: {e}")
            else:
                rec.status = "failed"
                rec.error = f"Exit code {rec.process.returncode}"
                update_run(rec.run_id, status="failed", finished_at=rec.finished_at,
                           error_message=rec.error,
                           log_output="\n".join(rec.log_lines[-500:]))

        except Exception as e:
            rec.status = "failed"
            rec.error = str(e)
            rec.finished_at = datetime.now(timezone.utc)
            update_run(rec.run_id, status="failed", finished_at=rec.finished_at, error_message=str(e))

    def _run_pocket_evaluation(self, rec: JobRecord, params: dict):
        cfg = get_config()
        try:
            update_run(rec.run_id, status="running", started_at=datetime.now(timezone.utc))

            # If generate_first, run generation first to get a .pt file
            if params.get("generate_first") and not params.get("pt_path"):
                gen_cmd = [
                    cfg.python_bin, cfg.sample_script(),
                    cfg.sampling_config,
                    "--device", f"cuda:{params['gpu_id']}",
                    "--protein_path", params["protein_path"],
                ]
                if params.get("ligand_path"):
                    gen_cmd += ["--ligand_path", params["ligand_path"]]
                gen_cmd += ["--pocket_radius", str(params.get("pocket_radius", 10.0))]
                if params.get("batch_size"):
                    gen_cmd += ["--batch_size", str(params["batch_size"])]

                gen_env = os.environ.copy()
                gen_env["PYTHONPATH"] = cfg.diffdynamic_root
                gen_env["PYTHONUNBUFFERED"] = "1"

                rec.log_lines.append("=== Phase 1: Generating molecules ===")
                gen_proc = subprocess.Popen(
                    gen_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, env=gen_env, cwd=cfg.diffdynamic_root,
                    bufsize=1,
                )
                for line in gen_proc.stdout:
                    line = line.rstrip()
                    if self._is_important_log(line):
                        rec.log_lines.append(line)
                    if ".pt" in line and ("saved" in line.lower() or "Results" in line):
                        m = re.search(r"(/[\w./\-]+\.pt)", line)
                        if m:
                            params["pt_path"] = m.group(1)
                gen_proc.wait()
                if gen_proc.returncode != 0:
                    rec.status = "failed"
                    rec.error = f"Generation failed with exit code {gen_proc.returncode}"
                    rec.finished_at = datetime.now(timezone.utc)
                    update_run(rec.run_id, status="failed", finished_at=rec.finished_at,
                               error_message=rec.error,
                               log_output="\n".join(rec.log_lines[-500:]))
                    return
                rec.log_lines.append("=== Phase 2: Evaluating pocket quality ===")
                rec.progress = 0.3
                update_run(rec.run_id, progress=0.3)

            # Build pocket evaluation command
            pt_path = params.get("pt_path")
            protein_path = params.get("protein_path")
            ligand_path = params.get("ligand_path")

            if pt_path:
                cmd = [
                    cfg.python_bin, cfg.pocket_eval_script(),
                    "--pt_file", pt_path,
                    "--fpocket_protein_pdb", protein_path,
                    "--visualize",
                ]
            else:
                cmd = [
                    cfg.python_bin, cfg.pocket_eval_script(),
                    "--eval_ligands", ligand_path,
                    "--custom_pocket_pdb", protein_path,
                    "--vina_outputs_dir", cfg.output_root,
                    "--fpocket_protein_pdb", protein_path,
                    "--visualize",
                ]

            if params.get("max_samples"):
                cmd += ["--idea_e_expected_n_molecules", str(params["max_samples"])]

            env = os.environ.copy()
            env["PYTHONPATH"] = cfg.diffdynamic_root
            env["PYTHONUNBUFFERED"] = "1"

            rec.log_lines.append(f"Running: {' '.join(cmd)}")
            rec.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=env, cwd=cfg.diffdynamic_root,
                bufsize=1,
            )

            _sync_counter = 0
            for line in rec.process.stdout:
                line = line.rstrip()
                if self._is_important_log(line):
                    rec.log_lines.append(line)
                self._parse_pocket_eval_progress(rec, line)
                _sync_counter += 1
                if _sync_counter % 20 == 0:
                    update_run(rec.run_id, progress=rec.progress,
                               log_output="\n".join(rec.log_lines[-500:]))

            rec.process.wait()
            rec.finished_at = datetime.now(timezone.utc)

            if rec.process.returncode == 0:
                rec.status = "completed"
                rec.progress = 1.0
                update_run(rec.run_id, status="completed", finished_at=rec.finished_at,
                           output_path=rec.output_path, progress=1.0,
                           log_output="\n".join(rec.log_lines[-500:]))
                log_history("pocket_eval_completed", {"job_id": rec.job_id, "vis_dir": rec.output_path})
            else:
                rec.status = "failed"
                rec.error = f"Exit code {rec.process.returncode}"
                update_run(rec.run_id, status="failed", finished_at=rec.finished_at,
                           error_message=rec.error,
                           log_output="\n".join(rec.log_lines[-500:]))

        except Exception as e:
            rec.status = "failed"
            rec.error = str(e)
            rec.finished_at = datetime.now(timezone.utc)
            update_run(rec.run_id, status="failed", finished_at=rec.finished_at, error_message=str(e))
        finally:
            self._gpu.release(rec.gpu_id)

    def _run_optimization(self, rec: JobRecord, params: dict):
        """Run molecular optimization via sample_diffusion.py with patched config."""
        cfg = get_config()
        try:
            update_run(rec.run_id, status="running", started_at=datetime.now(timezone.utc))

            patched_config = params.get("patched_config", cfg.sampling_config)
            cmd = [
                cfg.python_bin, cfg.sample_script(),
                patched_config,
                "--device", f"cuda:{params['gpu_id']}",
            ]

            data_id = params.get("data_id")
            if data_id is not None:
                cmd += ["-i", str(data_id)]
            elif params.get("protein_path"):
                cmd += ["--protein_path", params["protein_path"]]
                if params.get("ligand_path"):
                    cmd += ["--ligand_path", params["ligand_path"]]
            elif params.get("use_test_set"):
                cmd += ["--use_test_set"]

            if params.get("molecule_path"):
                cmd += ["--molecule_path", params["molecule_path"]]

            env = os.environ.copy()
            env["PYTHONPATH"] = cfg.diffdynamic_root
            env["PYTHONUNBUFFERED"] = "1"

            rec.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=env, cwd=cfg.diffdynamic_root,
                bufsize=1,
            )

            _sync_counter = 0
            for line in rec.process.stdout:
                line = line.rstrip()
                if self._is_important_log(line):
                    rec.log_lines.append(line)
                self._parse_generation_progress(rec, line)
                _sync_counter += 1
                if _sync_counter % 20 == 0:
                    update_run(rec.run_id, progress=rec.progress,
                               log_output="\n".join(rec.log_lines[-500:]))

            rec.process.wait()
            rec.finished_at = datetime.now(timezone.utc)

            if rec.process.returncode == 0:
                rec.status = "completed"
                update_run(rec.run_id, status="completed", finished_at=rec.finished_at,
                           output_path=rec.output_path, progress=1.0,
                           log_output="\n".join(rec.log_lines[-500:]))
                log_history("optimization_completed", {"job_id": rec.job_id, "run_id": rec.run_id})
                self._auto_chain(rec, params)
            else:
                rec.status = "failed"
                rec.error = f"Exit code {rec.process.returncode}"
                update_run(rec.run_id, status="failed", finished_at=rec.finished_at,
                           error_message=rec.error,
                           log_output="\n".join(rec.log_lines[-500:]))
                log_history("optimization_failed", {"job_id": rec.job_id, "error": rec.error})

        except Exception as e:
            rec.status = "failed"
            rec.error = str(e)
            rec.finished_at = datetime.now(timezone.utc)
            update_run(rec.run_id, status="failed", finished_at=rec.finished_at, error_message=str(e))
            log_history("optimization_failed", {"job_id": rec.job_id, "error": str(e)})
        finally:
            self._gpu.release(rec.gpu_id)

    def _run_scaffold(self, rec: JobRecord, params: dict):
        """Run scaffold-constrained generation via sample_diffusion.py with patched config."""
        cfg = get_config()
        try:
            update_run(rec.run_id, status="running", started_at=datetime.now(timezone.utc))

            patched_config = params.get("patched_config", cfg.sampling_config)
            cmd = [
                cfg.python_bin, cfg.sample_script(),
                patched_config,
                "--device", f"cuda:{params['gpu_id']}",
            ]

            data_id = params.get("data_id")
            if data_id is not None:
                cmd += ["-i", str(data_id)]
            elif params.get("protein_path"):
                cmd += ["--protein_path", params["protein_path"]]
                if params.get("ligand_path"):
                    cmd += ["--ligand_path", params["ligand_path"]]
            elif params.get("use_test_set"):
                cmd += ["--use_test_set"]

            if params.get("molecule_path"):
                cmd += ["--molecule_path", params["molecule_path"]]

            env = os.environ.copy()
            env["PYTHONPATH"] = cfg.diffdynamic_root
            env["PYTHONUNBUFFERED"] = "1"

            rec.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=env, cwd=cfg.diffdynamic_root,
                bufsize=1,
            )

            _sync_counter = 0
            for line in rec.process.stdout:
                line = line.rstrip()
                if self._is_important_log(line):
                    rec.log_lines.append(line)
                self._parse_generation_progress(rec, line)
                _sync_counter += 1
                if _sync_counter % 20 == 0:
                    update_run(rec.run_id, progress=rec.progress,
                               log_output="\n".join(rec.log_lines[-500:]))

            rec.process.wait()
            rec.finished_at = datetime.now(timezone.utc)

            if rec.process.returncode == 0:
                rec.status = "completed"
                update_run(rec.run_id, status="completed", finished_at=rec.finished_at,
                           output_path=rec.output_path, progress=1.0,
                           log_output="\n".join(rec.log_lines[-500:]))
                log_history("scaffold_completed", {"job_id": rec.job_id, "run_id": rec.run_id})
                self._auto_chain(rec, params)
            else:
                rec.status = "failed"
                rec.error = f"Exit code {rec.process.returncode}"
                update_run(rec.run_id, status="failed", finished_at=rec.finished_at,
                           error_message=rec.error,
                           log_output="\n".join(rec.log_lines[-500:]))
                log_history("scaffold_failed", {"job_id": rec.job_id, "error": rec.error})

        except Exception as e:
            rec.status = "failed"
            rec.error = str(e)
            rec.finished_at = datetime.now(timezone.utc)
            update_run(rec.run_id, status="failed", finished_at=rec.finished_at, error_message=str(e))
            log_history("scaffold_failed", {"job_id": rec.job_id, "error": str(e)})
        finally:
            self._gpu.release(rec.gpu_id)

    def _run_scaffold_cascade(self, rec: JobRecord, params: dict):
        """Run scaffold cascade pipeline (multi-round scaffold grow + merge)."""
        cfg = get_config()
        try:
            update_run(rec.run_id, status="running", started_at=datetime.now(timezone.utc))

            cmd = [
                cfg.python_bin, cfg.scaffold_cascade_script(),
                "--device", f"cuda:{params['gpu_id']}",
                "--samples_per_round", str(params.get("samples_per_round", 5)),
                "--rounds", str(params.get("rounds", 2)),
            ]

            data_id = params.get("data_id")
            if data_id is not None:
                cmd += ["-i", str(data_id)]
            if params.get("protein_path"):
                cmd += ["--protein_path", params["protein_path"]]
            if params.get("ligand_path"):
                cmd += ["--ligand_path", params["ligand_path"]]
            if params.get("config_path"):
                cmd += ["--config", params["config_path"]]

            env = os.environ.copy()
            env["PYTHONPATH"] = cfg.diffdynamic_root
            env["PYTHONUNBUFFERED"] = "1"

            rec.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=env, cwd=cfg.diffdynamic_root,
                bufsize=1,
            )

            _sync_counter = 0
            for line in rec.process.stdout:
                line = line.rstrip()
                if self._is_important_log(line) or "[R" in line or "Cascade" in line or "Merge" in line:
                    rec.log_lines.append(line)
                # Parse progress from round markers
                if "[R" in line and "完成" in line:
                    m = re.search(r"(\d+)/(\d+)\s+有效", line)
                    if m:
                        rec.progress = min(0.95, rec.progress + 0.4)
                        update_run(rec.run_id, progress=rec.progress)
                if "Cascade complete" in line:
                    rec.progress = 0.95
                    update_run(rec.run_id, progress=0.95)
                # Capture output path
                if "Cascade complete" in line and "Output:" in line:
                    m = re.search(r"Output:\s*(/\S+\.pt)", line)
                    if m:
                        rec.output_path = m.group(1)
                _sync_counter += 1
                if _sync_counter % 20 == 0:
                    update_run(rec.run_id, progress=rec.progress,
                               log_output="\n".join(rec.log_lines[-500:]))

            rec.process.wait()
            rec.finished_at = datetime.now(timezone.utc)

            if rec.process.returncode == 0:
                rec.status = "completed"
                update_run(rec.run_id, status="completed", finished_at=rec.finished_at,
                           output_path=rec.output_path, progress=1.0,
                           log_output="\n".join(rec.log_lines[-500:]))
                log_history("scaffold_cascade_completed", {"job_id": rec.job_id, "run_id": rec.run_id})
                self._auto_chain(rec, params)
            else:
                rec.status = "failed"
                rec.error = f"Exit code {rec.process.returncode}"
                update_run(rec.run_id, status="failed", finished_at=rec.finished_at,
                           error_message=rec.error,
                           log_output="\n".join(rec.log_lines[-500:]))
                log_history("scaffold_cascade_failed", {"job_id": rec.job_id, "error": rec.error})

        except Exception as e:
            rec.status = "failed"
            rec.error = str(e)
            rec.finished_at = datetime.now(timezone.utc)
            update_run(rec.run_id, status="failed", finished_at=rec.finished_at, error_message=str(e))
            log_history("scaffold_cascade_failed", {"job_id": rec.job_id, "error": str(e)})
        finally:
            self._gpu.release(rec.gpu_id)

    def _auto_chain(self, rec: JobRecord, params: dict):
        """Run auto-evaluate and auto-extract after successful generation."""
        cfg = get_config()
        auto_eval = params.get("auto_evaluate", False)
        auto_ext = params.get("auto_extract", False)
        if (auto_eval or auto_ext) and rec.output_path:
            logger.info(f"Auto-chain: eval={auto_eval}, extract={auto_ext}, pt={rec.output_path}")
            if auto_eval:
                eval_result = self.submit_evaluation(
                    pt_path=rec.output_path,
                    max_samples=5,
                    vina_timeout=20,
                    triggered_by="auto_chain",
                )
                if "error" not in eval_result:
                    log_history("auto_chain_evaluate", {"parent_job": rec.job_id, "eval_job": eval_result["job_id"]})
            if auto_ext:
                ext_result = self.submit_extraction(
                    pt_path=rec.output_path,
                    protein_root=params.get("protein_root") or os.path.join(cfg.data_root, "crossdocked_pocket10_test_only"),
                    remove_fragments=params.get("remove_fragments", False),
                    triggered_by="auto_chain",
                )
                if "error" not in ext_result:
                    log_history("auto_chain_extract", {"parent_job": rec.job_id, "extract_job": ext_result["job_id"]})

    # ── Progress parsing ─────────────────────────────────────────────────

    @staticmethod
    def _is_important_log(line: str) -> bool:
        """Filter: only keep progress, results, errors, key milestones."""
        if not line or line.strip() == '':
            return False
        # Keep progress bars
        if '%|' in line and 'it/s' in line:
            return True
        # Keep errors and tracebacks
        if any(kw in line for kw in ('❌', 'Error', 'error:', 'ERROR', 'Traceback', 'FileNotFoundError', '失败')):
            return True
        # Keep result lines
        if any(kw in line for kw in ('✅', 'Results saved', 'Sample done', 'Peak Memory', '完成', '成功')):
            return True
        # Keep key milestones
        if any(kw in line for kw in ('加载', 'Loaded', '评估', '重建', '对接', '分子', '采样', '进度')):
            return True
        # Keep progress counters like [8/100]
        if re.search(r'\[\d+/\d+\]', line):
            return True
        return False

    def _parse_generation_progress(self, rec: JobRecord, line: str):
        """Extract progress from sample_diffusion.py stdout."""
        m = re.search(r"(\d+)/(\d+)\s+steps?", line)
        if m:
            current, total = int(m.group(1)), int(m.group(2))
            if total > 0:
                rec.progress = current / total
                update_run(rec.run_id, progress=rec.progress,
                           progress_detail=json.dumps({"step": current, "total": total}))

        if ".pt" in line and ("saved" in line.lower() or "Results" in line):
            m2 = re.search(r"(/[\w./\-]+\.pt)", line)
            if m2:
                rec.output_path = m2.group(1)

    def _parse_eval_progress(self, rec: JobRecord, line: str):
        """Extract progress from evaluation stdout."""
        m = re.search(r"(\d+)/(\d+)", line)
        if m and ("molecule" in line.lower() or "评估分子" in line or "it/s" in line):
            current, total = int(m.group(1)), int(m.group(2))
            if total > 0:
                rec.progress = current / total
                update_run(rec.run_id, progress=rec.progress)
        # Capture eval output directory — look for "评估输出目录:" pattern
        if "评估输出目录" in line:
            m2 = re.search(r"评估输出目录:\s*(.+)", line)
            if m2:
                rec.output_path = m2.group(1).strip()
        # Fallback: capture from "输出目录" lines
        elif "输出目录" in line and "eval_" in line:
            m2 = re.search(r"(/[\w./\-]+eval_[\w./\-]+)", line)
            if m2:
                rec.output_path = m2.group(1)
        # Also capture from "结果已保存至" lines
        if "结果已保存至" in line or "Results saved" in line:
            m2 = re.search(r"(/[\w./\-]+\.pt)", line)
            if m2:
                pt_path = m2.group(1)
                eval_dir = os.path.dirname(pt_path)
                if "eval_" in eval_dir:
                    rec.output_path = eval_dir

    def _parse_pocket_eval_progress(self, rec: JobRecord, line: str):
        """Extract progress from pocket evaluation stdout."""
        # Capture visualization output directory
        if "Visualization output ->" in line or "可视化输出" in line:
            m = re.search(r"(?:->|至)\s*(/.+)", line)
            if m:
                rec.output_path = m.group(1).strip()
        # Capture folder name patterns like pocket_name_timestamp
        if "pocket_quality_vis" in line and "/" in line:
            m = re.search(r"(/[\w./\-]+pocket_quality_vis[\w./\-]+)", line)
            if m:
                rec.output_path = m.group(1).strip()
        # Capture overall score
        if "overall" in line.lower() and ("score" in line.lower() or "分" in line):
            m = re.search(r"(\d+\.\d+)", line)
            if m:
                try:
                    rec.progress = min(0.9, 0.5 + float(m.group(1)) * 0.4)
                except ValueError:
                    pass
        # Phase progress
        if "=== Phase 2" in line:
            rec.progress = 0.4
            update_run(rec.run_id, progress=0.4)

    def _validate_config_path(self, config_path: Optional[str], cfg) -> str:
        """Return validated config path, falling back to default."""
        if not config_path:
            return cfg.sampling_config
        abs_path = os.path.realpath(config_path)
        allowed_dir = os.path.realpath(os.path.join(cfg.diffdynamic_root, "configs"))
        if not abs_path.startswith(allowed_dir):
            raise ValueError(f"config_path must be under {allowed_dir}")
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"Config file not found: {abs_path}")
        return abs_path

    def _read_config_snapshot(self, path: str) -> str:
        try:
            with open(path) as f:
                return f.read()
        except Exception:
            return ""


# Singleton
_scheduler: Optional[JobScheduler] = None


def get_scheduler() -> JobScheduler:
    global _scheduler
    if _scheduler is None:
        cfg = get_config()
        _scheduler = JobScheduler(max_concurrent=cfg.max_concurrent_jobs)
    return _scheduler
