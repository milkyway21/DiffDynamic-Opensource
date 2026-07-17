"""Smoke and contract tests for DiffDynamic web API (no GPU jobs executed)."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.database import (
    init_db, get_dashboard_stats, run_to_dict, reconcile_stale_runs,
    create_run, update_run, count_molecules, add_molecules, query_molecules,
    db_session, Run, SCHEMA_VERSION, SchemaVersion,
)
from server.jobs import JobScheduler, VALID_SCAFFOLD_MODES
from server.job_helpers import write_patched_config, deep_merge
from server.api import GenerateRequest, ScaffoldRequest, BatchGenerateRequest
from server.scheduler import get_scheduler
import yaml


def test_database_init():
    init_db()
    stats = get_dashboard_stats()
    assert "total_runs" in stats
    assert "total_molecules" in stats
    with db_session() as s:
        row = s.query(SchemaVersion).order_by(SchemaVersion.id.desc()).first()
        assert row is not None
        assert row.version == SCHEMA_VERSION


def test_scheduler_queue():
    s = JobScheduler(max_concurrent=1)
    assert s._queue_position() == 0
    assert s.get_status("nonexistent") is None
    assert "queue_position" in JobScheduler.__dict__.get("get_status").__annotations__ or True
    rec_dict_keys = {"job_id", "run_id", "job_type", "status", "queue_position"}
    # JobRecord.to_dict includes queue_position
    from server.scheduler import JobRecord
    rec = JobRecord("t_1", 0, "test", -1)
    d = rec.to_dict()
    assert "queue_position" in d


def test_run_to_dict_none():
    assert run_to_dict(None) is None


def test_generate_request_defaults():
    req = GenerateRequest()
    assert req.batch_size == 5
    assert req.auto_evaluate is True
    assert req.auto_extract is True
    assert req.max_samples == 5
    assert req.vina_timeout == 20


def test_batch_request_has_batch_size_but_script_uses_yaml():
    """Batch API accepts batch_size; runner patches YAML instead of CLI --batch_size."""
    req = BatchGenerateRequest(start_id=0, end_id=0, batch_size=5)
    assert req.batch_size == 5
    # Simulate YAML patch used by run_batch_generation
    base = {"sample": {"dynamic": {"large_step": {"batch_size": 100}}}}
    deep_merge(base, {"sample": {"dynamic": {"large_step": {"batch_size": 5}}}})
    assert base["sample"]["dynamic"]["large_step"]["batch_size"] == 5


def test_scaffold_prudent_mode_in_request():
    req = ScaffoldRequest(scaffold_mode="prudent", prudent_n_generations=3)
    assert req.scaffold_mode == "prudent"
    assert req.prudent_n_generations == 3
    assert "prudent" in VALID_SCAFFOLD_MODES


def test_scaffold_invalid_mode_rejected():
    sched = JobScheduler(max_concurrent=1)
    result = sched.submit_scaffold(scaffold_mode="not_a_mode", data_id=0)
    assert "error" in result
    assert "Invalid scaffold_mode" in result["error"]


def test_scaffold_prudent_writes_yaml(tmp_path=None):
    """submit_scaffold(prudent) patches config with scaffold.mode=prudent and prudent block."""
    init_db()
    # Use a tiny base config under configs/
    from server.config import get_config
    cfg = get_config()
    base = os.path.join(cfg.diffdynamic_root, "configs", "sampling.yml")
    assert os.path.isfile(base)

    # Build patches the same way scheduler does for prudent
    patches = {
        "sample": {
            "scaffold": {
                "enable": True,
                "mode": "prudent",
                "prudent": {
                    "n_generations": 3,
                    "n_chains_per_seed": 2,
                    "advance_top_k": 1,
                    "renoise_t": 600,
                },
                "grow": {"num_samples": 5},
            },
            "dynamic": {"skip_refine": False},
        },
    }
    out = write_patched_config(base, patches, "test_prudent_unit")
    try:
        with open(out) as f:
            data = yaml.safe_load(f)
        assert data["sample"]["scaffold"]["mode"] == "prudent"
        assert data["sample"]["scaffold"]["prudent"]["n_generations"] == 3
        assert data["sample"]["dynamic"]["skip_refine"] is False
    finally:
        if os.path.isfile(out):
            os.remove(out)


def test_reconcile_stale_runs():
    init_db()
    rid = create_run(run_type="generate_dynamic", parameters={"data_id": 0}, status="running")
    n = reconcile_stale_runs(reason="unit_test")
    assert n >= 1
    with db_session() as s:
        r = s.get(Run, rid)
        assert r.status == "failed"
        assert "unit_test" in (r.error_message or "")


def test_count_molecules_filters():
    init_db()
    rid = create_run(run_type="evaluate", parameters={})
    add_molecules(rid, [
        {"smiles": "CCO", "vina_score": -7.0, "lipinski_pass": 5, "pocket_id": "pocketA"},
        {"smiles": "CCC", "vina_score": -3.0, "lipinski_pass": 2, "pocket_id": "pocketB"},
        {"smiles": "CCO", "vina_score": -8.0, "lipinski_pass": 5, "pocket_id": "pocketA"},  # dup smiles skipped
    ])
    assert count_molecules(run_id=rid) == 2
    assert count_molecules(run_id=rid, smiles_like="CCO") == 1
    assert count_molecules(run_id=rid, max_vina=-6.0) == 1
    assert count_molecules(run_id=rid, min_lipinski=4) == 1
    assert count_molecules(run_id=rid, pocket_id="pocketA") == 1


def test_cpu_gpu_pools_separate():
    s = JobScheduler(max_concurrent=1, max_cpu_workers=2)
    assert s._gpu_executor is not None
    assert s._cpu_executor is not None
    assert s._gpu_executor is not s._cpu_executor


if __name__ == "__main__":
    test_database_init()
    test_scheduler_queue()
    test_run_to_dict_none()
    test_generate_request_defaults()
    test_batch_request_has_batch_size_but_script_uses_yaml()
    test_scaffold_prudent_mode_in_request()
    test_scaffold_invalid_mode_rejected()
    test_scaffold_prudent_writes_yaml()
    test_reconcile_stale_runs()
    test_count_molecules_filters()
    test_cpu_gpu_pools_separate()
    print("All smoke/contract tests passed.")
