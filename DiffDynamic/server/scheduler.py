"""Async job scheduler: GPU/CPU pools, queue, and submit APIs."""

from __future__ import annotations

import itertools
import json
import logging
import os
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from server.config import get_config
from server.database import create_run, update_run, log_history, get_run
from server.job_helpers import release_gpu_safe, cleanup_temp_config, write_patched_config
from server.runners.base import validate_config_path, read_config_snapshot
from server.runners.generate import run_generation, run_batch_generation, run_custom_generation
from server.runners.evaluate import run_evaluation, run_extraction
from server.runners.pocket import run_pocket_evaluation
from server.runners.scaffold import run_optimization, run_scaffold, run_scaffold_cascade

logger = logging.getLogger(__name__)

VALID_SCAFFOLD_MODES = frozenset({"grow", "evolve", "dynamic_locked", "prudent"})


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
        if gpu_id is None or gpu_id < 0:
            return
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
        "output_path", "error", "progress", "log_lines", "scores",
        "queue_position",
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
        self.scores: Dict[str, str] = {}
        self.queue_position: Optional[int] = None

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
            "scores": self.scores or None,
            "queue_position": self.queue_position,
        }


class JobScheduler:
    """Thread-pool based job scheduler with separate GPU and CPU pools."""

    def __init__(self, max_concurrent: int = 2, max_cpu_workers: int = 2):
        cfg = get_config()
        self._gpu = GPUAllocator(cfg.gpu_ids)
        self._gpu_executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self._cpu_executor = ThreadPoolExecutor(max_workers=max_cpu_workers)
        # Keep alias for smoke tests / backward compat
        self._executor = self._gpu_executor
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._queue_lock = threading.Lock()
        self._pending: deque = deque()
        self._counter = itertools.count(1)

    def _next_job_id(self, prefix: str) -> str:
        with self._lock:
            n = next(self._counter)
        return f"{prefix}_{n:04d}_{int(time.time())}"

    def _queue_position(self) -> int:
        with self._queue_lock:
            return len(self._pending)

    def _position_of(self, job_id: str) -> Optional[int]:
        with self._queue_lock:
            for i, (rec, _) in enumerate(self._pending, start=1):
                if rec.job_id == job_id:
                    return i
        return None

    def _enqueue(self, rec: JobRecord, starter: Callable) -> dict:
        rec.status = "pending"
        with self._queue_lock:
            self._pending.append((rec, starter))
            pos = len(self._pending)
        rec.queue_position = pos
        update_run(rec.run_id, status="pending")
        return {
            "job_id": rec.job_id, "run_id": rec.run_id,
            "status": "queued", "queue_position": pos,
        }

    def _dispatch_queue(self):
        while True:
            with self._queue_lock:
                if not self._pending:
                    return
                rec, starter = self._pending[0]
            preferred = getattr(starter, "_preferred_gpu", None)
            gpu = self._gpu.acquire(preferred)
            if gpu < 0:
                return
            with self._queue_lock:
                if not self._pending or self._pending[0][0] is not rec:
                    self._gpu.release(gpu)
                    continue
                self._pending.popleft()
                # Refresh queue positions for remaining
                for i, (r, _) in enumerate(self._pending, start=1):
                    r.queue_position = i
            rec.gpu_id = gpu
            rec.status = "running"
            rec.queue_position = None
            rec.started_at = datetime.now(timezone.utc)
            future = self._gpu_executor.submit(starter, rec, gpu)
            rec.future = future

    def _acquire_or_queue(self, preferred: Optional[int], rec: JobRecord, starter: Callable) -> dict:
        starter._preferred_gpu = preferred  # type: ignore
        gpu = self._gpu.acquire(preferred)
        if gpu < 0:
            with self._lock:
                self._jobs[rec.job_id] = rec
            return self._enqueue(rec, starter)
        rec.gpu_id = gpu
        future = self._gpu_executor.submit(starter, rec, gpu)
        rec.future = future
        with self._lock:
            self._jobs[rec.job_id] = rec
        return {"job_id": rec.job_id, "run_id": rec.run_id, "gpu_id": gpu, "status": "running"}

    def _submit_cpu(self, rec: JobRecord, fn: Callable, *args) -> dict:
        future = self._cpu_executor.submit(fn, *args)
        rec.future = future
        with self._lock:
            self._jobs[rec.job_id] = rec
        return {"job_id": rec.job_id, "run_id": rec.run_id, "status": "running"}

    def _release_and_dispatch(self, rec: JobRecord):
        release_gpu_safe(self._gpu, rec.gpu_id)
        cleanup_temp_config(rec.job_id)
        self._dispatch_queue()

    def _write_patched_config(self, base_config_path: str, patches: dict, job_id: str) -> str:
        return write_patched_config(base_config_path, patches, job_id)

    def _validate_config_path(self, config_path: Optional[str], cfg=None) -> str:
        return validate_config_path(config_path, cfg)

    def _read_config_snapshot(self, path: str) -> str:
        return read_config_snapshot(path)

    # ── Public submit APIs ───────────────────────────────────────────────

    def submit_generation(
        self,
        mode: str = "dynamic",
        data_id: Optional[int] = 0,
        use_test_set: bool = True,
        gpu_id: Optional[int] = None,
        batch_size: Optional[int] = 5,
        config_path: Optional[str] = None,
        auto_evaluate: bool = True,
        auto_extract: bool = True,
        remove_fragments: bool = True,
        max_samples: Optional[int] = 5,
        vina_timeout: int = 20,
        triggered_by: str = "web_ui",
    ) -> dict:
        cfg = get_config()
        if batch_size is None:
            batch_size = 5
        parameters = {
            "mode": mode, "data_id": data_id, "use_test_set": use_test_set,
            "batch_size": batch_size, "auto_evaluate": auto_evaluate,
            "auto_extract": auto_extract, "remove_fragments": remove_fragments,
            "max_samples": max_samples if max_samples is not None else 5,
            "vina_timeout": vina_timeout, "preferred_gpu": gpu_id,
        }
        run_id = create_run(
            run_type=f"generate_{mode}", parameters=parameters,
            config_snapshot=self._read_config_snapshot(config_path or cfg.sampling_config),
            triggered_by=triggered_by,
        )
        log_history("start_generation", {"run_id": run_id, "mode": mode, "data_id": data_id}, user=triggered_by)
        job_id = self._next_job_id(f"gen_{mode}")
        rec = JobRecord(job_id, run_id, f"generate_{mode}", -1)

        def starter(rec, gpu):
            parameters["gpu_id"] = gpu
            run_generation(self, rec, parameters, config_path)

        return self._acquire_or_queue(gpu_id, rec, starter)

    def submit_batch_generation(
        self,
        start_id: int = 0,
        end_id: int = 0,
        gpus: Optional[str] = None,
        batch_size: Optional[int] = 5,
        config_path: Optional[str] = None,
        auto_evaluate: bool = True,
        sample_only: bool = False,
        triggered_by: str = "web_ui",
    ) -> dict:
        cfg = get_config()
        parameters = {
            "start_id": start_id, "end_id": end_id, "gpus": gpus,
            "batch_size": batch_size, "auto_evaluate": auto_evaluate,
            "sample_only": sample_only, "preferred_gpu": None,
        }
        run_id = create_run(
            run_type="generate_batch", parameters=parameters,
            config_snapshot=self._read_config_snapshot(config_path or cfg.sampling_config),
            triggered_by=triggered_by,
        )
        log_history("start_batch_generation", {"run_id": run_id}, user=triggered_by)
        job_id = self._next_job_id("gen_batch")
        rec = JobRecord(job_id, run_id, "generate_batch", -1)

        def starter(rec, gpu):
            parameters["gpu_id"] = gpu
            run_batch_generation(self, rec, parameters, config_path)

        return self._acquire_or_queue(None, rec, starter)

    def submit_rerun(self, run_id: int, overrides: Optional[dict] = None, triggered_by: str = "web_ui") -> dict:
        r = get_run(run_id)
        if r is None:
            return {"error": f"Run {run_id} not found"}
        params = json.loads(r.parameters) if r.parameters else {}
        if overrides:
            params.update(overrides)
        rt = r.run_type
        if rt.startswith("generate_") and rt != "generate_batch":
            mode = params.get("mode") or rt.replace("generate_", "")
            return self.submit_generation(
                mode=mode, data_id=params.get("data_id"),
                use_test_set=params.get("use_test_set", True),
                gpu_id=params.get("preferred_gpu") or params.get("gpu_id"),
                batch_size=params.get("batch_size", 5),
                auto_evaluate=params.get("auto_evaluate", True),
                auto_extract=params.get("auto_extract", True),
                remove_fragments=params.get("remove_fragments", True),
                max_samples=params.get("max_samples", 5),
                vina_timeout=params.get("vina_timeout", 20),
                triggered_by=triggered_by,
            )
        if rt == "evaluate":
            return self.submit_evaluation(
                pt_path=params["pt_path"], protein_root=params.get("protein_root"),
                max_samples=params.get("max_samples"), vina_timeout=params.get("vina_timeout", 20),
                triggered_by=triggered_by,
            )
        if rt == "extract":
            return self.submit_extraction(
                pt_path=params["pt_path"], protein_root=params.get("protein_root"),
                remove_fragments=params.get("remove_fragments", True),
                triggered_by=triggered_by,
            )
        return {"error": f"Re-run not supported for run_type={rt}"}

    def submit_custom_generation(
        self,
        protein_path: str,
        ligand_path: Optional[str] = None,
        gpu_id: Optional[int] = None,
        pocket_radius: float = 10.0,
        num_samples: int = 5,
        config_path: Optional[str] = None,
        auto_evaluate: bool = True,
        auto_extract: bool = True,
        remove_fragments: bool = True,
        triggered_by: str = "web_ui",
    ) -> dict:
        pocket_name = os.path.splitext(os.path.basename(protein_path))[0]
        parameters = {
            "protein_path": protein_path, "ligand_path": ligand_path,
            "pocket_radius": pocket_radius, "num_samples": num_samples,
            "auto_evaluate": auto_evaluate, "auto_extract": auto_extract,
            "remove_fragments": remove_fragments, "pocket_name": pocket_name,
            "max_samples": 5, "vina_timeout": 20, "preferred_gpu": gpu_id,
        }
        run_id = create_run(run_type="generate_custom", parameters=parameters, triggered_by=triggered_by)
        log_history("start_custom_generation", {"run_id": run_id, "protein_path": protein_path}, user=triggered_by)
        job_id = self._next_job_id("gen_custom")
        rec = JobRecord(job_id, run_id, "generate_custom", -1)

        def starter(rec, gpu):
            parameters["gpu_id"] = gpu
            run_custom_generation(self, rec, parameters, config_path)

        return self._acquire_or_queue(gpu_id, rec, starter)

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
        parameters = {
            "pt_path": pt_path, "protein_root": protein_root,
            "vina_modes": vina_modes, "max_samples": max_samples,
            "vina_timeout": vina_timeout, "pocket_name": pocket_name,
        }
        run_id = create_run(run_type="evaluate", parameters=parameters, triggered_by=triggered_by)
        log_history("start_evaluation", {"run_id": run_id, "pt_path": pt_path}, user=triggered_by)
        job_id = self._next_job_id("eval")
        rec = JobRecord(job_id, run_id, "evaluate", -1)
        return self._submit_cpu(rec, run_evaluation, self, rec, parameters)

    def submit_extraction(
        self,
        pt_path: str,
        protein_root: Optional[str] = None,
        eval_dir: Optional[str] = None,
        remove_fragments: bool = True,
        triggered_by: str = "web_ui",
    ) -> dict:
        parameters = {
            "pt_path": pt_path, "protein_root": protein_root,
            "eval_dir": eval_dir, "remove_fragments": remove_fragments,
        }
        run_id = create_run(run_type="extract", parameters=parameters, triggered_by=triggered_by)
        log_history("start_extraction", {"run_id": run_id, "pt_path": pt_path}, user=triggered_by)
        job_id = self._next_job_id("extract")
        rec = JobRecord(job_id, run_id, "extract", -1)
        return self._submit_cpu(rec, run_extraction, self, rec, parameters)

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
        pocket_name = os.path.splitext(os.path.basename(protein_path))[0]
        parameters = {
            "pt_path": pt_path, "protein_path": protein_path, "ligand_path": ligand_path,
            "generate_first": generate_first, "batch_size": batch_size,
            "pocket_radius": pocket_radius, "max_samples": max_samples,
            "pocket_name": pocket_name, "preferred_gpu": gpu_id,
        }
        run_id = create_run(run_type="pocket_eval", parameters=parameters, triggered_by=triggered_by)
        log_history("start_pocket_eval", {"run_id": run_id, "protein_path": protein_path}, user=triggered_by)
        job_id = self._next_job_id("pocket_eval")
        rec = JobRecord(job_id, run_id, "pocket_eval", -1)

        def starter(rec, gpu):
            parameters["gpu_id"] = gpu
            run_pocket_evaluation(self, rec, parameters)

        return self._acquire_or_queue(gpu_id, rec, starter)

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
        remove_fragments: bool = True,
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
        cfg = get_config()
        parameters = {
            "data_id": data_id, "protein_path": protein_path,
            "ligand_path": ligand_path, "molecule_path": molecule_path,
            "use_test_set": use_test_set,
            "auto_evaluate": auto_evaluate, "auto_extract": auto_extract,
            "remove_fragments": remove_fragments,
            "max_samples": 5, "vina_timeout": 20,
            "num_samples": num_samples, "start_t": start_t,
            "stride": stride, "step_size": step_size,
            "cycles": cycles, "schedule": schedule,
            "after_dynamic": after_dynamic,
            "min_qed": min_qed, "min_sa": min_sa,
            "max_survivors_per_cycle": max_survivors_per_cycle,
            "max_survivors_total": max_survivors_total,
            "preferred_gpu": gpu_id,
        }
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

        run_id = create_run(
            run_type="optimization", parameters=parameters,
            config_snapshot=self._read_config_snapshot(base_cfg),
            triggered_by=triggered_by,
        )
        log_history("start_optimization", {"run_id": run_id, "data_id": data_id}, user=triggered_by)
        rec = JobRecord(job_id, run_id, "optimization", -1)

        def starter(rec, gpu):
            parameters["gpu_id"] = gpu
            run_optimization(self, rec, parameters)

        return self._acquire_or_queue(gpu_id, rec, starter)

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
        remove_fragments: bool = True,
        scaffold_mode: str = "grow",
        scaffold_source: str = "auto_murcko",
        scaffold_skip_refine: bool = True,
        scaffold_enable_refine: bool = True,
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
        grow_num_samples: int = 200,
        grow_start_t: int = 357,
        grow_stride: int = 15,
        grow_step_size: float = 0.3262,
        grow_lambda_a: int = 47,
        grow_lambda_b: int = 11,
        grow_n_extra_mode: str = "prior_minus_scaffold",
        grow_n_extra_fixed: int = 8,
        grow_n_extra_min: int = 3,
        grow_n_extra_max: int = 20,
        grow_n_extra_min_clamp: int = 6,
        grow_n_extra_max_clamp: int = 20,
        grow_filter_incomplete: bool = False,
        grow_min_qed: float = 0.2,
        grow_min_sa: float = 0.2,
        evolve_population_size: int = 50,
        evolve_n_generations: int = 5,
        evolve_children_per_parent: int = 10,
        evolve_start_t_high: int = 200,
        evolve_start_t_low: int = 30,
        evolve_stride: int = 2,
        evolve_step_size: float = 0.2,
        evolve_lambda_a: int = 10,
        evolve_lambda_b: int = 1,
        diversity_filter_enable: bool = True,
        max_tanimoto: float = 0.9,
        # Scaffold-Prudent
        prudent_n_generations: int = 5,
        prudent_n_chains_per_seed: int = 4,
        prudent_advance_top_k: int = 2,
        prudent_renoise_t: int = 600,
        prudent_qed_weight: float = 0.2,
        prudent_sa_weight: float = 0.2,
        prudent_vina_weight: float = 0.6,
        prudent_lipinski_weight: float = 0.0,
        prudent_lilly_weight: float = 0.0,
        prudent_vina_exhaustiveness: int = 8,
        prudent_min_qed_for_docking: float = 0.15,
        prudent_min_sa_for_docking: float = 0.15,
        # 门控：adaptive（默认）或 manual（人工数值）
        adaptive_gates_enable: bool = True,
        adaptive_gates_actives: Optional[str] = None,
        gate_mode: str = "adaptive",
        manual_max_logp: Optional[float] = 5.0,
        manual_max_molwt: Optional[float] = 850.0,
        manual_max_heavy_atoms: Optional[int] = 60,
        manual_max_rings: Optional[int] = 7,
        triggered_by: str = "web_ui",
    ) -> dict:
        if scaffold_mode not in VALID_SCAFFOLD_MODES:
            return {
                "error": f"Invalid scaffold_mode={scaffold_mode!r}; "
                         f"expected one of {sorted(VALID_SCAFFOLD_MODES)}",
            }

        cfg = get_config()
        parameters = {
            "data_id": data_id, "protein_path": protein_path,
            "ligand_path": ligand_path, "molecule_path": molecule_path,
            "use_test_set": use_test_set,
            "auto_evaluate": auto_evaluate, "auto_extract": auto_extract,
            "remove_fragments": remove_fragments,
            "max_samples": 5, "vina_timeout": 20,
            "scaffold_mode": scaffold_mode, "scaffold_source": scaffold_source,
            "preferred_gpu": gpu_id,
            "prudent_n_generations": prudent_n_generations,
            "prudent_n_chains_per_seed": prudent_n_chains_per_seed,
            "prudent_advance_top_k": prudent_advance_top_k,
            "prudent_renoise_t": prudent_renoise_t,
            "adaptive_gates_enable": adaptive_gates_enable,
            "adaptive_gates_actives": adaptive_gates_actives,
            "gate_mode": gate_mode,
            "manual_max_logp": manual_max_logp,
            "manual_max_molwt": manual_max_molwt,
            "manual_max_heavy_atoms": manual_max_heavy_atoms,
            "manual_max_rings": manual_max_rings,
        }

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
            "murcko_sites": {
                "p_active": 0.5,
                "max_per_site": 20,
                "min_per_site": 0,
                "per_site_count_mode": "sequential_random",
                "per_site_add_mode": "fragment_prior",
                "fragment_sizes": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20],
                "fragment_weights": [
                    0.02, 0.02, 0.02, 0.02, 0.03, 0.16, 0.10, 0.09, 0.08,
                    0.12, 0.12, 0.10, 0.05, 0.04, 0.015, 0.01, 0.005,
                ],
                "jitter_std": 1.0,
                "jitter_mode": "gaussian",
                "max_active_sites": "n_removed_sidechains",
                "preserve_zero_allocation_sidechains": True,
                "prefer_murcko_sites": True,
                "save_json": True,
                "overflow_mode": "cap",
                "dedup_dist": 0.5,
            },
            "adaptive_gates": {
                "enable": bool(adaptive_gates_enable),
                "ligand": ligand_path or molecule_path,
                "actives": adaptive_gates_actives,
            },
        }

        grow_block = {
            "seed": 42,
            "n_extra_mode": grow_n_extra_mode,
            "n_extra_fixed": grow_n_extra_fixed,
            "n_extra_min": grow_n_extra_min,
            "n_extra_max": grow_n_extra_max,
            "n_extra_min_clamp": grow_n_extra_min_clamp,
            "n_extra_max_clamp": grow_n_extra_max_clamp,
            "num_samples": grow_num_samples,
            "start_t": grow_start_t,
            "stride": grow_stride,
            "step_size": grow_step_size,
            "lambda_coeff_a": grow_lambda_a,
            "lambda_coeff_b": grow_lambda_b,
            "min_qed": grow_min_qed,
            "min_sa": grow_min_sa,
            "filter_incomplete": grow_filter_incomplete,
        }

        if scaffold_mode == "grow":
            scaffold_cfg["grow"] = grow_block
        elif scaffold_mode == "dynamic_locked":
            scaffold_cfg["fix_scaffold_pos"] = True
            scaffold_cfg["fix_scaffold_type"] = False
            scaffold_cfg["num_samples"] = grow_num_samples
            scaffold_cfg["grow"] = grow_block
        elif scaffold_mode == "prudent":
            # Gen0 uses dynamic_locked-style grow; then iterative renoise/refine
            scaffold_cfg["fix_scaffold_pos"] = True
            scaffold_cfg["fix_scaffold_type"] = False
            scaffold_cfg["num_samples"] = grow_num_samples
            scaffold_cfg["grow"] = grow_block
            scaffold_cfg["prudent"] = {
                "n_generations": prudent_n_generations,
                "n_chains_per_seed": prudent_n_chains_per_seed,
                "advance_top_k": prudent_advance_top_k,
                "renoise_t": prudent_renoise_t,
                # null → 用 scaffold.num_samples（= grow_num_samples / API N）
                "target_final_count": None,
                "max_final_fill_attempts": None,
                "qed_weight": prudent_qed_weight,
                "sa_weight": prudent_sa_weight,
                "vina_weight": prudent_vina_weight,
                "lipinski_weight": prudent_lipinski_weight,
                "lilly_demerit_weight": prudent_lilly_weight,
                "vina_exhaustiveness": prudent_vina_exhaustiveness,
                "min_qed_for_docking": prudent_min_qed_for_docking,
                "min_sa_for_docking": prudent_min_sa_for_docking,
                "advance_top_k_mode": "cap",
                "continue_when_no_advance": True,
                "fallback_keep_top_k": 1,
            }
            # 人工门控：写入固定阈值，并关闭自适应
            if (gate_mode or "adaptive").lower() == "manual" or not adaptive_gates_enable:
                scaffold_cfg["adaptive_gates"]["enable"] = False
                if manual_max_logp is not None:
                    scaffold_cfg["prudent"]["max_logp"] = float(manual_max_logp)
                if manual_max_molwt is not None:
                    scaffold_cfg["prudent"]["max_molwt"] = float(manual_max_molwt)
                if manual_max_heavy_atoms is not None:
                    scaffold_cfg["prudent"]["max_heavy_atoms"] = int(manual_max_heavy_atoms)
                if manual_max_rings is not None:
                    scaffold_cfg["prudent"]["max_rings"] = int(manual_max_rings)
                parameters["manual_gates"] = {
                    "max_logp": manual_max_logp,
                    "max_molwt": manual_max_molwt,
                    "max_heavy_atoms": manual_max_heavy_atoms,
                    "max_rings": manual_max_rings,
                }
            # 自适应：任务创建时按原始参考配体写入门控
            _ref_lig = ligand_path or molecule_path
            if adaptive_gates_enable and (gate_mode or "adaptive").lower() != "manual" and _ref_lig and os.path.isfile(_ref_lig):
                try:
                    from utils.adaptive_gates import (
                        apply_gates_to_prudent_cfg,
                        compute_adaptive_gates,
                    )
                    _gates = compute_adaptive_gates(
                        _ref_lig,
                        actives_csv=adaptive_gates_actives,
                        grow_cfg=grow_block,
                        base_prudent=scaffold_cfg["prudent"],
                    )
                    scaffold_cfg["prudent"] = apply_gates_to_prudent_cfg(
                        scaffold_cfg["prudent"], _gates,
                    )
                    scaffold_cfg["prudent"].pop("_adaptive_gates", None)
                    scaffold_cfg["adaptive_gates"]["ligand"] = _ref_lig
                    scaffold_cfg["adaptive_gates"]["preview"] = _gates.as_prudent_dict()
                    parameters["adaptive_gates"] = _gates.as_prudent_dict()
                except Exception as e:
                    logger.warning("adaptive_gates pre-apply failed: %s", e)
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

        sc_skip_refine = False if scaffold_mode in ("dynamic_locked", "prudent") else scaffold_skip_refine
        sc_patches = {
            "sample": {
                "scaffold": scaffold_cfg,
                "dynamic": {"skip_refine": sc_skip_refine},
                "targetdiff_baseline_refine": {
                    "enable": scaffold_enable_refine,
                    "start_t": 9,
                    "after_scaffold_init_only": False,
                },
            },
        }

        base_cfg = self._validate_config_path(config_path, cfg)
        job_id = self._next_job_id(f"sc_{scaffold_mode}")
        patched_config = self._write_patched_config(base_cfg, sc_patches, job_id)
        parameters["patched_config"] = patched_config

        run_id = create_run(
            run_type=f"scaffold_{scaffold_mode}", parameters=parameters,
            config_snapshot=self._read_config_snapshot(base_cfg),
            triggered_by=triggered_by,
        )
        log_history("start_scaffold", {"run_id": run_id, "scaffold_mode": scaffold_mode}, user=triggered_by)
        rec = JobRecord(job_id, run_id, f"scaffold_{scaffold_mode}", -1)

        def starter(rec, gpu):
            parameters["gpu_id"] = gpu
            run_scaffold(self, rec, parameters)

        return self._acquire_or_queue(gpu_id, rec, starter)

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
        remove_fragments: bool = True,
        triggered_by: str = "web_ui",
    ) -> dict:
        parameters = {
            "data_id": data_id, "protein_path": protein_path,
            "ligand_path": ligand_path,
            "samples_per_round": samples_per_round, "rounds": rounds,
            "config_path": config_path,
            "auto_evaluate": auto_evaluate, "auto_extract": auto_extract,
            "remove_fragments": remove_fragments,
            "max_samples": 5, "vina_timeout": 20,
            "preferred_gpu": gpu_id,
        }
        run_id = create_run(run_type="scaffold_cascade", parameters=parameters, triggered_by=triggered_by)
        log_history("start_scaffold_cascade", {"run_id": run_id, "rounds": rounds}, user=triggered_by)
        job_id = self._next_job_id("cascade")
        rec = JobRecord(job_id, run_id, "scaffold_cascade", -1)

        def starter(rec, gpu):
            parameters["gpu_id"] = gpu
            run_scaffold_cascade(self, rec, parameters)

        return self._acquire_or_queue(gpu_id, rec, starter)

    def get_status(self, job_id: str) -> Optional[dict]:
        with self._lock:
            rec = self._jobs.get(job_id)
        if rec is None:
            return None
        d = rec.to_dict()
        if rec.status == "pending":
            d["queue_position"] = self._position_of(job_id)
        return d

    def get_job_record(self, job_id: str) -> Optional[JobRecord]:
        """Return the in-memory JobRecord (for result aggregation endpoints)."""
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            rec = self._jobs.get(job_id)
        if rec is None:
            return False
        if rec.status == "pending":
            with self._queue_lock:
                self._pending = deque(item for item in self._pending if item[0].job_id != job_id)
                for i, (r, _) in enumerate(self._pending, start=1):
                    r.queue_position = i
            rec.status = "cancelled"
            rec.finished_at = datetime.now(timezone.utc)
            update_run(rec.run_id, status="cancelled", finished_at=rec.finished_at)
            log_history("cancel_job", {"job_id": job_id, "was": "pending"})
            return True
        if rec.status != "running":
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
        self._release_and_dispatch(rec)
        return True

    def list_jobs(self, status: Optional[str] = None) -> List[dict]:
        with self._lock:
            jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        result = []
        for j in sorted(jobs, key=lambda x: x.started_at or datetime.min, reverse=True):
            d = j.to_dict()
            if j.status == "pending":
                d["queue_position"] = self._position_of(j.job_id)
            result.append(d)
        return result


_scheduler: Optional[JobScheduler] = None


def get_scheduler() -> JobScheduler:
    global _scheduler
    if _scheduler is None:
        cfg = get_config()
        _scheduler = JobScheduler(max_concurrent=cfg.max_concurrent_jobs)
    return _scheduler
