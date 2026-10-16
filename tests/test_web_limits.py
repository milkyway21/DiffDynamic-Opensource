"""Web-side cap tests: per-user quota, queue cap, rate limiter, .pt trust.

No GPU jobs are started; scheduler internals are exercised directly.
"""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from server.scheduler import JobRecord, JobScheduler, JobSubmitError


def _cfg(tmp_path, **overrides):
    base = dict(
        gpu_ids=[0],
        output_root=str(tmp_path / "outputs"),
        max_concurrent_jobs=1,
        max_cpu_workers=1,
        max_jobs_per_user=2,
        max_pending_jobs=3,
        max_submit_per_min=12,
        max_upload_per_min=10,
        max_upload_bytes=1000,
        job_timeout=3600,
        max_generate_per_day=0,
        max_eval_per_day=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _sched(monkeypatch, tmp_path, **overrides) -> JobScheduler:
    cfg = _cfg(tmp_path, **overrides)
    monkeypatch.setattr("server.scheduler.get_config", lambda: cfg)
    return JobScheduler(max_concurrent=1, max_cpu_workers=1)


def _fake_job(sched: JobScheduler, owner: str, status: str = "running") -> JobRecord:
    rec = JobRecord(
        f"job_{owner}_{status}_{len(sched._jobs)}", run_id=0, job_type="stub", gpu_id=-1,
        owner=owner,
    )
    rec.status = status
    sched._jobs[rec.job_id] = rec
    return rec


# ── Per-user quota ────────────────────────────────────────────────────────

def test_quota_blocks_submission_over_limit(monkeypatch, tmp_path):
    sched = _sched(monkeypatch, tmp_path, max_jobs_per_user=2)
    _fake_job(sched, "alice")
    _fake_job(sched, "alice")
    with pytest.raises(JobSubmitError) as exc_info:
        sched._attach_owner_params({}, "alice")
    assert "alice" in str(exc_info.value)


def test_quota_counts_pending_but_not_finished(monkeypatch, tmp_path):
    sched = _sched(monkeypatch, tmp_path, max_jobs_per_user=2)
    _fake_job(sched, "alice", "running")
    _fake_job(sched, "alice", "pending")
    _fake_job(sched, "alice", "finished")
    with pytest.raises(JobSubmitError):
        sched._attach_owner_params({}, "alice")


def test_quota_is_per_user(monkeypatch, tmp_path):
    sched = _sched(monkeypatch, tmp_path, max_jobs_per_user=2)
    _fake_job(sched, "alice")
    _fake_job(sched, "alice")
    owner = sched._attach_owner_params({}, "bob")
    assert owner == "bob"


def test_quota_disabled_when_zero(monkeypatch, tmp_path):
    sched = _sched(monkeypatch, tmp_path, max_jobs_per_user=0)
    for _ in range(5):
        _fake_job(sched, "alice")
    assert sched._attach_owner_params({}, "alice") == "alice"


def test_active_jobs_for_owner(monkeypatch, tmp_path):
    sched = _sched(monkeypatch, tmp_path, max_jobs_per_user=0)
    _fake_job(sched, "alice", "running")
    _fake_job(sched, "alice", "pending")
    _fake_job(sched, "alice", "finished")
    _fake_job(sched, "bob", "running")
    assert sched.active_jobs_for_owner("alice") == 2
    assert sched.active_jobs_for_owner("bob") == 1


def test_attach_owner_params_sets_result_path(monkeypatch, tmp_path):
    sched = _sched(monkeypatch, tmp_path)
    parameters: dict = {}
    sched._attach_owner_params(parameters, "alice")
    assert parameters["owner"] == "alice"
    assert str(tmp_path / "outputs" / "users" / "alice") in parameters["result_path"]


def test_daily_generate_quota_blocks(monkeypatch, tmp_path):
    sched = _sched(monkeypatch, tmp_path, max_jobs_per_user=0, max_generate_per_day=1, max_eval_per_day=1)
    monkeypatch.setattr(
        "server.scheduler.count_runs_today",
        lambda owner, bucket, now=None: 1 if bucket == "generate" else 0,
    )
    with pytest.raises(JobSubmitError) as exc_info:
        sched._attach_owner_params({"auto_evaluate": False}, "alice", daily_bucket="generate")
    assert "生成" in str(exc_info.value)
    assert exc_info.value.status_code == 429


def test_daily_eval_quota_blocks(monkeypatch, tmp_path):
    sched = _sched(monkeypatch, tmp_path, max_jobs_per_user=0, max_generate_per_day=1, max_eval_per_day=1)
    monkeypatch.setattr(
        "server.scheduler.count_runs_today",
        lambda owner, bucket, now=None: 1 if bucket == "eval" else 0,
    )
    with pytest.raises(JobSubmitError) as exc_info:
        sched._attach_owner_params({}, "alice", daily_bucket="eval")
    assert "评估" in str(exc_info.value)


def test_auto_eval_blocked_when_eval_cap_hit(monkeypatch, tmp_path):
    sched = _sched(monkeypatch, tmp_path, max_jobs_per_user=0, max_generate_per_day=1, max_eval_per_day=1)
    monkeypatch.setattr(
        "server.scheduler.count_runs_today",
        lambda owner, bucket, now=None: 1 if bucket == "eval" else 0,
    )
    with pytest.raises(JobSubmitError) as exc_info:
        sched._attach_owner_params({"auto_evaluate": True}, "alice", daily_bucket="generate")
    assert "自动评估" in str(exc_info.value)


def test_daily_quota_skipped_for_admin_flag(monkeypatch, tmp_path):
    sched = _sched(monkeypatch, tmp_path, max_jobs_per_user=0, max_generate_per_day=1, max_eval_per_day=1)
    monkeypatch.setattr("server.scheduler.count_runs_today", lambda *a, **k: 99)
    owner = sched._attach_owner_params(
        {"auto_evaluate": True}, "alice", daily_bucket="generate", skip_daily_quota=True,
    )
    assert owner == "alice"
    assert sched._attach_owner_params({}, "alice", daily_bucket="eval", skip_daily_quota=True) == "alice"


def test_daily_quota_is_per_user(monkeypatch, tmp_path):
    sched = _sched(monkeypatch, tmp_path, max_jobs_per_user=0, max_generate_per_day=1, max_eval_per_day=1)

    def _count(owner, bucket, now=None):
        if owner == "alice" and bucket == "generate":
            return 1
        return 0

    monkeypatch.setattr("server.scheduler.count_runs_today", _count)
    with pytest.raises(JobSubmitError):
        sched._attach_owner_params({"auto_evaluate": False}, "alice", daily_bucket="generate")
    assert sched._attach_owner_params({"auto_evaluate": False}, "bob", daily_bucket="generate") == "bob"


# ── Global queue cap ──────────────────────────────────────────────────────

def test_enqueue_rejects_when_queue_full(monkeypatch, tmp_path):
    sched = _sched(monkeypatch, tmp_path, max_pending_jobs=1)
    monkeypatch.setattr("server.scheduler.update_run", lambda *a, **k: None)
    queued = JobRecord("queued_1", run_id=0, job_type="stub", gpu_id=-1, owner="alice")
    sched._pending.append((queued, lambda r, g: None))
    rec = JobRecord("queued_2", run_id=0, job_type="stub", gpu_id=-1, owner="alice")
    with pytest.raises(JobSubmitError):
        sched._enqueue(rec, lambda r, g: None)
    assert rec.status == "failed"


def test_enqueue_accepts_within_cap(monkeypatch, tmp_path):
    sched = _sched(monkeypatch, tmp_path, max_pending_jobs=2)
    monkeypatch.setattr("server.scheduler.update_run", lambda *a, **k: None)
    rec = JobRecord("queued_1", run_id=0, job_type="stub", gpu_id=-1, owner="alice")
    result = sched._enqueue(rec, lambda r, g: None)
    assert result["status"] == "queued"
    assert result["queue_position"] == 1


# ── Rate limiter ──────────────────────────────────────────────────────────

def test_rate_limiter_blocks_over_limit():
    from server.api import _SlidingWindowLimiter
    limiter = _SlidingWindowLimiter()
    for _ in range(3):
        limiter.check("u1", 3)
    with pytest.raises(HTTPException) as exc_info:
        limiter.check("u1", 3)
    assert exc_info.value.status_code == 429
    assert "Retry-After" in exc_info.value.headers


def test_rate_limiter_isolated_per_user():
    from server.api import _SlidingWindowLimiter
    limiter = _SlidingWindowLimiter()
    for _ in range(5):
        limiter.check("u1", 5)
    limiter.check("u2", 5)


def test_rate_limiter_window_expiry(monkeypatch):
    from server.api import _SlidingWindowLimiter
    limiter = _SlidingWindowLimiter()
    clock = [1000.0]
    monkeypatch.setattr("server.api.time.monotonic", lambda: clock[0])
    for _ in range(2):
        limiter.check("u1", 2)
    clock[0] += 61.0
    limiter.check("u1", 2)


def test_rate_limiter_disabled_when_zero():
    from server.api import _SlidingWindowLimiter
    limiter = _SlidingWindowLimiter()
    for _ in range(50):
        limiter.check("u1", 0)


# ── .pt trust boundary ────────────────────────────────────────────────────

class _StubMol:
    """Stand-in for the RDKit Mol objects embedded in eval .pt files."""


def test_load_eval_pt_allows_trusted_root(monkeypatch, tmp_path):
    import torch

    from server import molecule_ingest as mi

    out_root = tmp_path / "outputs"
    (out_root / "users" / "alice" / "pt").mkdir(parents=True)
    monkeypatch.setattr(
        "server.config.get_config",
        lambda: SimpleNamespace(output_root=str(out_root)),
    )
    good = out_root / "users" / "alice" / "pt" / "eval_results_1_final.pt"
    torch.save({"mol": _StubMol()}, str(good))
    data = mi._load_eval_pt(str(good))
    assert isinstance(data["mol"], _StubMol)


def test_load_eval_pt_rejects_untrusted_path(monkeypatch, tmp_path):
    import torch

    from server import molecule_ingest as mi

    out_root = tmp_path / "outputs"
    upload_dir = tmp_path / "uploads"
    out_root.mkdir()
    upload_dir.mkdir()
    monkeypatch.setattr(
        "server.config.get_config",
        lambda: SimpleNamespace(output_root=str(out_root)),
    )
    evil = upload_dir / "eval_results_1_final.pt"
    torch.save({"mol": _StubMol()}, str(evil))
    with pytest.raises(Exception):
        mi._load_eval_pt(str(evil))


# ── GPU default preference (web jobs → GPU5) ───────────────────────────────

def test_gpu_allocator_prefers_default_then_min(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "server.scheduler.get_config",
        lambda: SimpleNamespace(default_gpu_id=5, gpu_ids=[0, 1, 5]),
    )
    from server.scheduler import GPUAllocator

    alloc = GPUAllocator([0, 1, 5], default_gpu_id=5)
    assert alloc.acquire() == 5
    assert alloc.acquire() == 0
    assert alloc.acquire() == 1
    assert alloc.acquire() == -1


def test_gpu_allocator_explicit_preferred_wins(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "server.scheduler.get_config",
        lambda: SimpleNamespace(default_gpu_id=5, gpu_ids=[0, 5]),
    )
    from server.scheduler import GPUAllocator

    alloc = GPUAllocator([0, 5], default_gpu_id=5)
    assert alloc.acquire(preferred=0) == 0
    assert alloc.acquire() == 5
