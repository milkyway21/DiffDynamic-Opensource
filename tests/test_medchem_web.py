"""Medchem request, ownership and orchestration regressions; no GPU jobs."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from starlette.requests import Request

from server import api, scheduler, schemas
from server.medchem_schemas import MedchemEvaluation, MedchemOptions, MedchemRequest
from server.medchem_service import artifact_path, read_report
from server.runners import medchem
from server.submissions.generation import GenerationJobs


def request(user="alice"):
    value = Request({"type": "http", "method": "POST", "path": "/api/medchem", "headers": []})
    value.state.dd_user = user
    value.state.dd_is_admin = False
    return value


def run(**kwargs):
    return SimpleNamespace(id=7, owner="alice", run_type="medchem", status="completed",
                           output_path=None, parameters="{}", progress_detail="{}", **kwargs)


@pytest.mark.parametrize("values", [{}, {"smiles": "CC", "input_path": "x.sdf"},
                                    {"smiles": "C\n" * 101}, {"smiles": "C", "max_records": 1000}])
def test_source_and_record_limits(values):
    with pytest.raises(ValidationError):
        MedchemRequest(**values)


@pytest.mark.parametrize("values", [{"top_k": 11}, {"min_similarity": float("nan")},
                                    {"time_budget_seconds": 121}, {"max_heavy_delta": 11},
                                    {"min_heavy_delta": -11}])
def test_option_limits(values):
    with pytest.raises(ValidationError):
        MedchemOptions(**values)


@pytest.mark.parametrize("values", [{"center_xyz": [0, 0, 0]}, {"cpu": 4},
                                    {"center_xyz": [0, 0, 0], "size_xyz": [20, 41, 20]},
                                    {"center_xyz": [float("inf"), 0, 0], "size_xyz": [20] * 3}])
def test_box_and_cpu_limits(values):
    with pytest.raises(ValidationError):
        MedchemEvaluation(**values)


@pytest.mark.parametrize("schema,extra,handler,method", [
    (schemas.GenerateRequest, {}, api.start_generation, "submit_generation"),
    (schemas.BatchGenerateRequest, {"start_id": 0, "end_id": 1}, api.start_batch_generation, "submit_batch_generation"),
    (schemas.CustomGenerateRequest, {"protein_path": "protein.pdb", "ligand_path": "ligand.sdf"}, api.start_custom_generation, "submit_custom_generation"),
])
def test_all_generation_routes_forward_options_and_off_default(monkeypatch, schema, extra, handler, method):
    calls = []
    fake = SimpleNamespace(**{method: lambda **kw: calls.append(kw) or {"job_id": "test"}})
    monkeypatch.setattr(api, "get_scheduler", lambda: fake)
    monkeypatch.setattr(api, "validate_path_for_user", lambda path, *a, **kw: path)
    assert schema(**extra).medchem is None
    handler(schema(**extra, medchem={"profile": "aggressive"}), request())
    assert calls[-1]["medchem"]["profile"] == "aggressive"
    assert calls[-1]["medchem"]["top_k"] == 3
    handler(schema(**extra), request())
    assert calls[-1]["medchem"] is None


def test_report_and_artifacts_require_run_owner(monkeypatch, tmp_path):
    saved = run()
    saved.output_path = str(tmp_path)
    (tmp_path / "report.json").write_text(json.dumps({"status": "completed", "artifacts": []}))
    monkeypatch.setattr(api, "get_run", lambda run_id, owner=None: saved if owner == "alice" else None)
    assert api.get_medchem_report(7, request())["run_id"] == 7
    for callback, args in ((api.get_medchem_report, (7,)), (api.get_medchem_artifact, (7, "x"))):
        with pytest.raises(HTTPException) as error:
            callback(*args, request("bob"))
        assert error.value.status_code == 404


def test_artifact_ids_cannot_escape_or_follow_symlinks(tmp_path):
    root = tmp_path / "job"
    root.mkdir()
    (tmp_path / "secret").write_text("private")
    (root / "linked").symlink_to(tmp_path / "secret")
    (root / "ok.sdf").write_text("structure")
    saved = run()
    saved.output_path = str(root)
    (root / "report.json").write_text(json.dumps({"artifacts": [
        {"artifact_id": "escape", "path": "../secret"},
        {"artifact_id": "link", "path": "linked"},
        {"artifact_id": "ok", "path": "ok.sdf"},
    ]}))
    for identity in ("escape", "link", "../secret"):
        with pytest.raises(HTTPException):
            artifact_path(saved, identity)
    assert artifact_path(saved, "ok")[0] == root / "ok.sdf"
    assert all("path" not in item for item in read_report(saved)["artifacts"])
    for status in ("failed", "cancelled"):
        saved.status = status
        assert read_report(saved)["status"] == status


def test_submit_routes_cpu_and_evaluation_quota(monkeypatch):
    created, buckets = [], []
    monkeypatch.setattr(scheduler, "create_run", lambda **kw: created.append(kw) or 7)
    fake = SimpleNamespace(
        _attach_owner_params=lambda p, owner, **kw: buckets.append(kw["daily_bucket"]) or owner,
        _next_job_id=lambda prefix: prefix,
        _submit_cpu=lambda rec, *a: {"run_type": rec.job_type},
    )
    from server.submissions.medchem import MedchemJobs
    MedchemJobs.submit_medchem(fake, smiles="CC", triggered_by="alice")
    MedchemJobs.submit_medchem(fake, smiles="CC", options={"evaluation": {"mode": "vina_dock"}}, triggered_by="alice")
    assert buckets == [None, "eval"]
    assert [r["run_type"] for r in created] == ["medchem", "medchem_dock"]


def test_generation_chain_keeps_batch_in_one_cpu_task_and_runs_without_auto_eval(monkeypatch, tmp_path):
    submitted, updates = [], []
    monkeypatch.setattr(medchem, "get_config", lambda: SimpleNamespace(data_root=str(tmp_path)))
    monkeypatch.setattr(medchem, "update_run", lambda *a, **kw: updates.append(kw))
    fake = SimpleNamespace(submit_medchem=lambda **kw: submitted.append(kw) or {"run_id": 8})
    rec = SimpleNamespace(run_id=7, owner="alice", output_path="source.pt")
    assert medchem.chain_medchem(fake, rec, {}) == []
    medchem.chain_medchem(fake, rec, {"medchem": MedchemOptions().dict(), "auto_evaluate": False}, ["a.pt", "b.pt"])
    assert len(submitted) == 1
    assert submitted[0]["source_pts"] == ["a.pt", "b.pt"]
    assert submitted[0]["auto_evaluate"] is False
    assert json.loads(updates[0]["progress_detail"])["medchem_source_pts"] == ["a.pt", "b.pt"]


def test_medchem_replay_preserves_saved_internal_values_only(monkeypatch):
    saved = run()
    saved.parameters = json.dumps({"source_run_id": 4, "source_pts": ["saved.pt"],
                                   "auto_evaluate": True, "options": MedchemOptions().dict()})
    monkeypatch.setattr(scheduler, "get_run", lambda *a: saved)
    calls = []
    fake = SimpleNamespace(submit_medchem=lambda **kw: calls.append(kw))
    GenerationJobs.submit_rerun(fake, 7, {"source_pts": ["injected.pt"], "options": {"profile": "aggressive"}})
    assert calls[-1]["source_pts"] == ["saved.pt"]
    assert calls[-1]["auto_evaluate"] is True
    assert calls[-1]["options"]["profile"] == "aggressive"
    GenerationJobs.submit_rerun(fake, 7, {"source_run_id": 6})
    assert "source_pts" not in calls[-1]


@pytest.mark.parametrize("phase", ["prepare", "ingest"])
def test_cancellation_cannot_launch_or_complete_after_preparation(monkeypatch, tmp_path, phase):
    rec = SimpleNamespace(job_id="j", run_id=7, owner="alice", status="running", output_path=None)
    cfg = SimpleNamespace(python_bin="/usr/bin/python3", diffdynamic_root=str(tmp_path))
    monkeypatch.setattr(medchem, "get_config", lambda: cfg)
    monkeypatch.setattr(medchem, "update_run", lambda *a, **kw: None)
    def prepare(*args):
        if phase == "prepare":
            rec.status = "cancelled"
        return {}
    monkeypatch.setattr(medchem, "_source_request", prepare)
    launched = []
    monkeypatch.setattr(medchem, "run_cmd", lambda *a, **kw: launched.append(1) or 0)
    monkeypatch.setattr(medchem, "mark_completed", lambda *a: pytest.fail("Cancelled job completed"))
    monkeypatch.setattr("server.molecule_ingest.ingest_molecules_from_eval", lambda *a: setattr(rec, "status", "cancelled"))
    medchem.run_medchem(SimpleNamespace(_release_and_dispatch=lambda *a: None), rec, {"result_path": str(tmp_path)})
    assert len(launched) == (0 if phase == "prepare" else 1)
    assert rec.status == "cancelled"


def test_batch_rerun_cannot_bypass_range_limit(monkeypatch):
    saved = run()
    saved.run_type = "generate_batch"
    saved.parameters = json.dumps({"start_id": 0, "end_id": 2})
    monkeypatch.setattr(api, "_require_run", lambda *a: saved)
    with pytest.raises(HTTPException) as error:
        api.rerun_run(7, schemas.RerunRequest(overrides={"end_id": 10000000}), request())
    assert error.value.status_code == 400


def test_historical_batch_requires_complete_manifest(monkeypatch, tmp_path):
    saved = run()
    saved.run_type = "generate_batch"
    saved.output_path = str(tmp_path / "last.pt")
    Path(saved.output_path).touch()
    cfg = SimpleNamespace(data_root=str(tmp_path), output_root=str(tmp_path), upload_dir=str(tmp_path))
    monkeypatch.setattr(medchem, "get_config", lambda: cfg)
    monkeypatch.setattr(medchem, "get_run", lambda *a, **kw: saved)
    monkeypatch.setattr("server.safe_pt.prepare_job_pt", lambda path, owner: path)
    monkeypatch.setattr("server.safe_pt.is_generated_pt", lambda path, owner: True)
    with pytest.raises(ValueError, match="complete pocket manifest"):
        medchem._source_request({"source_run_id": 7}, "alice")
    saved.progress_detail = json.dumps({"medchem_source_pts": ["first.pt", "last.pt"]})
    result = medchem._source_request({"source_run_id": 7}, "alice")
    assert result["source_pts"] == ["first.pt", "last.pt"]
    assert result["_trusted_generated_sources"] is True


def test_enabled_generation_never_runs_legacy_reconstruction_chain(monkeypatch):
    from server.runners.base import auto_chain
    calls = []
    monkeypatch.setattr(medchem, "chain_medchem", lambda *args: calls.append(args))
    fake = SimpleNamespace(submit_evaluation=lambda **kw: pytest.fail("Duplicate evaluation"),
                           submit_extraction=lambda **kw: pytest.fail("Duplicate reconstruction"))
    auto_chain(fake, SimpleNamespace(), {"medchem": MedchemOptions().dict(),
                                       "auto_evaluate": True, "auto_extract": True})
    assert len(calls) == 1
