"""Async job scheduler: GPU/CPU pools, queue, and submit APIs."""

# Submission implementations resolve these live compatibility bindings.
# ruff: noqa: F401

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
from server.job_configs import (
    build_generation_patches,
    build_optimization_patches,
    build_scaffold_patches,
)
from server.database import create_run, update_run, log_history, get_run, count_runs_today
from server.job_helpers import release_gpu_safe, cleanup_temp_config, write_patched_config
from server.auth import normalize_owner, user_result_path, is_admin_user
from server.quota import (
    auto_eval_blocked_message,
    quota_message,
    seconds_until_reset,
)
from server.runners.base import validate_config_path, read_config_snapshot
from server.runners.generate import run_generation, run_batch_generation, run_custom_generation
from server.runners.evaluate import run_evaluation, run_extraction
from server.runners.pocket import run_pocket_evaluation
from server.runners.scaffold import run_optimization, run_scaffold, run_scaffold_cascade

from server.submissions import EvaluationJobs, GenerationJobs, ScaffoldJobs, MedchemJobs
from server.submissions.dependencies import dependencies

dependencies.bind(globals())

logger = logging.getLogger(__name__)

VALID_SCAFFOLD_MODES = frozenset({"grow", "evolve", "dynamic_locked", "prudent"})


class JobSubmitError(RuntimeError):
    """Submission rejected by per-user quota or queue cap.

    Raised before create_run() writes an orphan run row; the API layer maps
    this to an HTTP 429 response.
    """

    def __init__(self, message: str, status_code: int = 429):
        super().__init__(message)
        self.status_code = status_code


class GPUAllocator:
    """Track which GPUs are busy and assign the next available one."""

    def __init__(self, gpu_ids: List[int], default_gpu_id: Optional[int] = None):
        self._available = set(gpu_ids)
        self._default_gpu_id = default_gpu_id
        self._lock = threading.Lock()

    def acquire(self, preferred: Optional[int] = None) -> int:
        with self._lock:
            if preferred is not None and preferred in self._available:
                self._available.discard(preferred)
                return preferred
            default = self._default_gpu_id
            try:
                cfg_default = getattr(get_config(), "default_gpu_id", None)
                if cfg_default is not None:
                    default = cfg_default
            except Exception:
                pass
            if default is not None and default in self._available:
                self._available.discard(default)
                return default
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
        "queue_position", "owner",
    )

    def __init__(self, job_id: str, run_id: int, job_type: str, gpu_id: int, owner: str = "legacy"):
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
        self.owner = normalize_owner(owner)

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
            "owner": self.owner,
        }


class JobScheduler(GenerationJobs, EvaluationJobs, ScaffoldJobs, MedchemJobs):
    """Thread-pool based job scheduler with separate GPU and CPU pools."""

    def __init__(self, max_concurrent: int = 2, max_cpu_workers: int = 2):
        cfg = get_config()
        self._gpu = GPUAllocator(
            cfg.gpu_ids, default_gpu_id=getattr(cfg, "default_gpu_id", None),
        )
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

    def _owner_result_path(self, triggered_by: str) -> str:
        cfg = get_config()
        path = user_result_path(cfg.output_root, triggered_by)
        os.makedirs(path, exist_ok=True)
        return path

    def _attach_owner_params(
        self,
        parameters: dict,
        triggered_by: str,
        daily_bucket: Optional[str] = None,
        skip_daily_quota: bool = False,
    ) -> str:
        owner = normalize_owner(triggered_by)
        # Enforce quota here — every submit_* passes through before create_run(),
        # so a rejection never leaves an orphan run row behind.
        self._enforce_user_quota(owner)
        skip = skip_daily_quota or is_admin_user(triggered_by) or bool(parameters.get("_skip_daily_quota"))
        if skip:
            parameters["_skip_daily_quota"] = True
        else:
            medchem_dock = (parameters.get("medchem") or {}).get("evaluation", {}).get("mode") == "vina_dock"
            extra_eval = daily_bucket == "generate" and bool(parameters.get("auto_evaluate") or medchem_dock)
            self._enforce_daily_quota(owner, daily_bucket, extra_eval=extra_eval)
        parameters["owner"] = owner
        parameters["result_path"] = self._owner_result_path(owner)
        return owner

    def _enforce_daily_quota(self, owner: str, daily_bucket: Optional[str], extra_eval: bool = False):
        if not daily_bucket:
            return
        cfg = get_config()
        retry = seconds_until_reset()

        def _check(bucket: str):
            limit_attr = "max_eval_per_day" if bucket == "eval" else "max_generate_per_day"
            limit = int(getattr(cfg, limit_attr, 0) or 0)
            if limit <= 0:
                return
            used = count_runs_today(owner, bucket)
            if used >= limit:
                msg = quota_message(bucket, used, limit)
                if bucket == "eval" and extra_eval:
                    msg = auto_eval_blocked_message()
                err = JobSubmitError(msg, status_code=429)
                err.retry_after = retry
                raise err

        _check(daily_bucket)
        if extra_eval:
            _check("eval")

    def active_jobs_for_owner(self, owner: str) -> int:
        """Number of running + pending jobs currently held for owner."""
        with self._lock:
            return sum(
                1 for j in self._jobs.values()
                if getattr(j, "owner", None) == owner and j.status in ("running", "pending")
            )

    def _enforce_user_quota(self, owner: str):
        cfg = get_config()
        limit = int(getattr(cfg, "max_jobs_per_user", 0) or 0)
        if limit <= 0:
            return
        active = self.active_jobs_for_owner(owner)
        if active >= limit:
            raise JobSubmitError(
                f"Too many active jobs for user '{owner}' ({active}/{limit}). "
                "Wait for a job to finish or cancel one.",
            )

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
        cfg = get_config()
        cap = int(getattr(cfg, "max_pending_jobs", 0) or 0)
        rec.status = "pending"
        with self._queue_lock:
            if cap > 0 and len(self._pending) >= cap:
                over_cap = True
                pos = 0
            else:
                over_cap = False
                self._pending.append((rec, starter))
                pos = len(self._pending)
        if over_cap:
            # Defensive re-check: the run row already exists at this point,
            # so mark it failed instead of leaving it pending forever.
            rec.status = "failed"
            rec.error = "Server queue is full"
            rec.finished_at = datetime.now(timezone.utc)
            update_run(
                rec.run_id, status="failed", error_message=rec.error,
                finished_at=rec.finished_at,
            )
            raise JobSubmitError("Server queue is full, please retry later.")
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


    def get_status(self, job_id: str, owner: Optional[str] = None) -> Optional[dict]:
        with self._lock:
            rec = self._jobs.get(job_id)
        if rec is None:
            return None
        if owner and getattr(rec, "owner", None) != owner:
            return None
        d = rec.to_dict()
        if rec.status == "pending":
            d["queue_position"] = self._position_of(job_id)
        return d

    def get_job_record(self, job_id: str, owner: Optional[str] = None) -> Optional[JobRecord]:
        """Return the in-memory JobRecord (for result aggregation endpoints)."""
        with self._lock:
            rec = self._jobs.get(job_id)
        if rec is None:
            return None
        if owner and getattr(rec, "owner", None) != owner:
            return None
        return rec

    def cancel(self, job_id: str, owner: Optional[str] = None) -> bool:
        with self._lock:
            rec = self._jobs.get(job_id)
        if rec is None:
            return False
        if owner and getattr(rec, "owner", None) != owner:
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
        # Set terminal intent before waiting, so a finishing runner cannot win the race.
        rec.status = "cancelled"
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

    def list_jobs(self, status: Optional[str] = None, owner: Optional[str] = None) -> List[dict]:
        with self._lock:
            jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        if owner:
            jobs = [j for j in jobs if getattr(j, "owner", None) == owner]
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
