"""Release security contracts; use synthetic local sessions and isolated storage."""

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from server import api
from server.auth import PlatformIdentity, owner_slug
from server.config import ServerConfig
from server.database import create_run, init_db, update_run
from server.job_helpers import dynamic_web_patches
from server.scheduler import JobRecord


@pytest.fixture
def release_app(tmp_path, monkeypatch):
    cfg = ServerConfig(diffdynamic_root=str(tmp_path),
                       output_root=str(tmp_path / "outputs"),
                       data_root=str(tmp_path / "data"),
                       upload_dir=str(tmp_path / "uploads"),
                       gpu_ids=[0], max_generate_per_day=0, max_eval_per_day=0)
    for name in ("outputs", "data", "uploads", "configs"):
        (tmp_path / name).mkdir()
    (tmp_path / "configs/sampling.yml").write_text("sample: {}\n")
    monkeypatch.setattr("server.config._config", cfg)
    monkeypatch.setattr(api, "_rate_limiter", api._SlidingWindowLimiter())
    monkeypatch.setenv("DD_AUTH_MODE", "platform")
    monkeypatch.setenv("DD_ADMIN_USERS", "whitelisted")
    monkeypatch.delenv("DD_ALLOW_INTERNAL_API", raising=False)
    monkeypatch.setattr(api, "lookup_platform_identity", lambda cookie_header="", authorization="":
                        PlatformIdentity("alice", False) if "bianlab_token=alice" in cookie_header else
                        PlatformIdentity("bob", False) if "bianlab_token=bob" in cookie_header else
                        PlatformIdentity("admin", True) if "bianlab_token=admin" in cookie_header else
                        PlatformIdentity("whitelisted", False) if "bianlab_token=whitelisted" in cookie_header else None)
    init_db()
    rid = create_run("generate_dynamic", parameters={"gpu_id": 0, "batch_size": 5},
                     owner="alice", triggered_by="alice")
    update_run(rid, log_output="PRIVATE_LOG GPU_MEMORY", error_message="PRIVATE_TRACE /local/path")
    rec = JobRecord("local-job", rid, "generate_dynamic", 0, owner="alice")
    rec.log_lines = ["PRIVATE_LOG GPU_MEMORY"]
    rec.error = "PRIVATE_TRACE /local/path"
    scheduler = SimpleNamespace(
        get_status=lambda job_id, owner=None: rec.to_dict() if job_id == rec.job_id and owner in (None, "alice") else None,
        list_jobs=lambda status=None, owner=None: [rec.to_dict()] if owner in (None, "alice") else [],
        _queue_position=lambda: 7,
        active_jobs_for_owner=lambda owner: 1,
    )
    monkeypatch.setattr(api, "get_scheduler", lambda: scheduler)
    with TestClient(api.app) as client:
        client.cookies.set("bianlab_token", "alice")
        yield client, cfg, rid, scheduler


@pytest.mark.parametrize("path", ["/api/gpus", "/api/config", "/api/presets", "/api/browse/files?root=configs"])
def test_admin_interfaces_reject_regular_account(release_app, path):
    client, _, _, _ = release_app
    response = client.get(path)
    assert response.status_code == 403
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Content-Type-Options"] == "nosniff"


@pytest.mark.parametrize("prefix", ["jobs", "generate", "evaluate", "scaffold", "optimization", "scaffold-cascade"])
def test_all_status_interfaces_hide_diagnostics(release_app, prefix):
    client, _, _, _ = release_app
    response = client.get(f"/api/{prefix}/local-job")
    assert response.status_code == 200
    assert "PRIVATE_LOG" not in response.text
    assert "PRIVATE_TRACE" not in response.text
    assert "gpu_id" not in response.json()
    assert "log_tail" not in response.json()


def test_task_details_and_logs_are_role_separated(release_app):
    client, _, rid, _ = release_app
    details = client.get(f"/api/runs/{rid}")
    assert details.status_code == 200
    assert not details.json()["has_log"]
    assert "config_snapshot" not in details.json()
    assert "gpu_id" not in details.json()["parameters"]
    assert "PRIVATE_LOG" not in details.text
    assert client.get(f"/api/runs/{rid}/log").status_code == 403
    client.cookies.set("bianlab_token", "admin")
    assert "PRIVATE_LOG" in client.get(f"/api/runs/{rid}/log").text
    assert client.get(f"/api/jobs/local-job").json()["gpu_id"] == 0


def test_stats_never_queries_hardware_for_regular_account(release_app, monkeypatch):
    client, _, _, _ = release_app
    monkeypatch.setattr(api, "get_gpu_info", lambda: pytest.fail("Hardware was queried"))
    response = client.get("/api/stats")
    assert response.status_code == 200
    assert "gpus" not in response.json()
    assert response.json()["queue_length"] != 7


def test_platform_identity_cannot_be_spoofed(release_app):
    client, _, _, _ = release_app
    response = client.get("/api/me", headers={"X-DD-User": "whitelisted", "X-DD-Internal-Key": api._get_internal_api_key()})
    assert response.json()["username"] == "alice"
    assert response.json()["is_admin"] is False
    client.cookies.set("bianlab_token", "whitelisted")
    assert client.get("/api/me").json()["is_admin"] is True
    client.cookies.clear()
    response = client.get("/api/me", headers={"X-DD-User": "whitelisted", "X-DD-Internal-Key": api._get_internal_api_key()})
    assert response.status_code == 401
    assert "Content-Security-Policy" in response.headers


@pytest.mark.parametrize("forwarded", [{"X-Forwarded-Prefix": "/zh/diffdynamic"},
                                       {"X-Forwarded-For": "203.0.113.10"}])
def test_opted_in_loopback_key_cannot_authenticate_proxy_requests(release_app, monkeypatch, forwarded):
    monkeypatch.setenv("DD_ALLOW_INTERNAL_API", "1")
    with TestClient(api.app, client=("127.0.0.1", 12345)) as client:
        response = client.get("/api/me", headers={**forwarded, "X-DD-User": "whitelisted",
                                                  "X-DD-Internal-Key": api._get_internal_api_key()})
        assert response.status_code == 401


def test_cross_account_access_returns_not_found(release_app):
    client, _, rid, _ = release_app
    client.cookies.set("bianlab_token", "bob")
    assert client.get(f"/api/runs/{rid}").status_code == 404
    assert client.get("/api/jobs/local-job").status_code == 404


@pytest.mark.parametrize("endpoint", ["generate", "generate/batch", "generate/custom", "scaffold", "scaffold/preview", "scaffold/context", "optimization", "scaffold-cascade"])
def test_custom_sampling_config_is_rejected(release_app, endpoint):
    client, _, _, _ = release_app
    response = client.post(f"/api/{endpoint}", json={"config_path": "/any/custom.yml", "protein_path": "protein.pdb"},
                           headers={"Origin": "http://testserver"})
    assert response.status_code == 422


def test_generation_cannot_switch_to_prudent(release_app):
    client, _, _, _ = release_app
    response = client.post("/api/generate", json={"mode": "prudent"}, headers={"Origin": "http://testserver"})
    assert response.status_code == 422
    patches = dynamic_web_patches(5)["sample"]
    assert patches["mode"] == "dynamic"
    assert not patches["dynamic"]["prudent"]["enable"]
    assert not patches["dynamic"]["skip_refine"]
    assert not patches["scaffold"]["enable"]


def test_default_dynamic_yaml_preserves_existing_fast_parameters(release_app):
    import yaml
    from server.job_helpers import write_patched_config
    _, cfg, _, _ = release_app
    source = Path(__file__).resolve().parents[1] / "configs/sampling.yml"
    before = yaml.safe_load(source.read_text())
    patched = Path(write_patched_config(str(source), dynamic_web_patches(5), "local-default"))
    result = yaml.safe_load(patched.read_text())
    assert patched.is_relative_to(Path(cfg.diffdynamic_root) / ".cache") if hasattr(patched, "is_relative_to") else ".cache/web_job_configs/" in str(patched)
    assert result["sample"]["mode"] == "dynamic"
    assert not result["sample"]["dynamic"]["prudent"]["enable"]
    assert result["sample"]["dynamic"]["refine"] == before["sample"]["dynamic"]["refine"]
    assert result["sample"]["targetdiff_baseline_refine"] == before["sample"]["targetdiff_baseline_refine"]
    assert result["sample"]["dynamic"]["large_step"] == {**before["sample"]["dynamic"]["large_step"], "batch_size": 5}


@pytest.mark.parametrize("origin", ["https://other.invalid", "null", None])
def test_cookie_writes_require_allowed_origin(release_app, origin):
    client, _, _, _ = release_app
    response = client.post("/api/generate", json={}, headers={"Origin": origin} if origin else {})
    assert response.status_code == 403


def test_direct_log_file_download_is_restricted(release_app):
    client, cfg, _, _ = release_app
    root = Path(cfg.output_root) / "users" / owner_slug("alice")
    root.mkdir(parents=True)
    for extension in ("log", "txt", "json", "yaml"):
        path = root / f"task.{extension}"
        path.write_text("PRIVATE_LOG")
        assert client.get("/api/file/download", params={"path": str(path)}).status_code == 403


def test_rerun_rejects_sampling_and_cross_account_parent(release_app):
    client, _, rid, _ = release_app
    headers = {"Origin": "http://testserver"}
    for override in ({"config_path": "custom.yml"}, {"patched_config": "custom.yml"}, {"mode": "prudent"}):
        assert client.post(f"/api/runs/{rid}/rerun", json={"overrides": override}, headers=headers).status_code == 400
    client.cookies.set("bianlab_token", "bob")
    assert client.post("/api/extract", json={"pt_path": "unused.pt", "parent_run_id": rid}, headers=headers).status_code == 404


def test_distinct_account_names_have_distinct_storage():
    assert owner_slug("a@b") != owner_slug("a_b")
    assert owner_slug("用户甲") != owner_slug("用户乙")
    assert owner_slug("alice") != owner_slug("Alice")


def test_admin_whitelist_matches_canonical_account_exactly(monkeypatch):
    from server.auth import is_admin_user
    monkeypatch.setenv("DD_ADMIN_USERS", "Alice")
    assert is_admin_user("Alice")
    assert not is_admin_user("alice")


@pytest.mark.parametrize("value", ["true", "false", 1, 0, None])
def test_platform_role_requires_a_boolean(value):
    from server.auth import _extract_is_superuser
    assert _extract_is_superuser({"isSuperuser": value}) is False


@pytest.mark.parametrize("field", ["gpu_id", "preferred_gpu", "gpus"])
def test_rerun_cannot_select_hardware(release_app, field):
    client, _, rid, _ = release_app
    response = client.post(f"/api/runs/{rid}/rerun",
                           json={"overrides": {field: 0}}, headers={"Origin": "http://testserver"})
    assert response.status_code == 403


def test_rerun_uses_default_sampling_and_public_limits(release_app):
    client, _, rid, scheduler = release_app
    calls = []
    scheduler.submit_rerun = lambda *args, **kwargs: calls.append(kwargs) or {"job_id": "replay"}
    response = client.post(f"/api/runs/{rid}/rerun",
                           json={"overrides": {"batch_size": 10000, "vina_timeout": 10000,
                                               "_skip_daily_quota": True}},
                           headers={"Origin": "http://testserver"})
    assert response.status_code == 200
    params = calls[0]["overrides"]
    assert params["batch_size"] == 100
    assert params["vina_timeout"] == 600
    assert params["mode"] == "dynamic"
    assert params["gpu_id"] is None and params["preferred_gpu"] is None
    assert "_skip_daily_quota" not in params
    assert calls[0]["triggered_by"] == "alice"
    assert calls[0]["skip_daily_quota"] is False


def test_custom_rerun_keeps_original_task_type(release_app):
    from server.scheduler import JobScheduler
    _, cfg, _, _ = release_app
    protein = Path(cfg.data_root) / "protein.pdb"
    protein.write_text("synthetic protein")
    rid = create_run("generate_custom", parameters={"protein_path": str(protein), "num_samples": 7},
                     owner="alice", triggered_by="alice")
    calls = []
    scheduler = SimpleNamespace(submit_custom_generation=lambda **kwargs: calls.append(kwargs) or {"job_id": "custom"})
    result = JobScheduler.submit_rerun(scheduler, rid, triggered_by="alice")
    assert result["job_id"] == "custom"
    assert calls[0]["protein_path"] == str(protein)
    assert calls[0]["num_samples"] == 7


@pytest.fixture
def migration_db(tmp_path, monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from server import database
    engine = create_engine(f"sqlite:///{tmp_path / 'migration.db'}", future=True)
    monkeypatch.setattr(database, "_engine", engine)
    monkeypatch.setattr(database, "_SessionFactory", sessionmaker(bind=engine, future=True, expire_on_commit=False))
    database.init_db()
    with database.db_session() as session:
        session.query(database.SchemaVersion).update({"version": 2})
    yield database
    engine.dispose()


def test_account_migration_keeps_history_and_is_idempotent(migration_db, tmp_path):
    database = migration_db
    historical = tmp_path / "outputs/users/a_b/result.pt"
    historical.parent.mkdir(parents=True)
    historical.write_bytes(b"historical result")
    rid = database.create_run("generate_dynamic", parameters={}, owner="a@b", triggered_by="a@b")
    database.update_run(rid, output_path=str(historical))
    database.init_db()
    assert owner_slug("a@b") == "a_b"
    assert owner_slug("a_b") != "a_b"
    database.create_run("generate_dynamic", parameters={}, owner="system", triggered_by="system")
    database.init_db()
    assert database.get_run(rid).output_path == str(historical)
    assert historical.read_bytes() == b"historical result"
    assert database.list_runs(owner="system")
    with database.db_session() as session:
        assert session.query(database.AccountStorage).count() == 1


def test_account_migration_stops_on_historical_directory_collision(migration_db):
    for owner in ("a@b", "a_b"):
        migration_db.create_run("generate_dynamic", parameters={}, owner=owner, triggered_by=owner)
    with pytest.raises(RuntimeError, match="directory collision"):
        migration_db.init_db()
    with migration_db.db_session() as session:
        assert session.query(migration_db.AccountStorage).count() == 0


def test_external_pt_safely_round_trips_legacy_evaluators(tmp_path, monkeypatch):
    from server import safe_pt
    from evaluate_pt_with_correct_reconstruct import _torch_load_layered
    from evaluate_pocket_quality import _torch_load_legacy
    cfg = ServerConfig(diffdynamic_root=str(tmp_path), output_root=str(tmp_path / "outputs"), gpu_ids=[0])
    monkeypatch.setattr("server.config._config", cfg)
    source = tmp_path / "input.pt"
    data = {"pred_ligand_pos": [np.zeros((3, 3), dtype=np.float32)],
            "pred_ligand_v": [torch.zeros(3, dtype=torch.long)],
            "data": SimpleNamespace(ligand_filename="ligand.sdf", protein_filename="protein.pdb")}
    torch.save(data, source)
    # No host deserializer may touch the original input.
    with monkeypatch.context() as local:
        local.setattr(safe_pt.torch, "load", lambda *a, **kw: pytest.fail("Host unpickled external input"))
        canonical = safe_pt.prepare_job_pt(source, "alice")
    monkeypatch.setenv("DD_PT_STRICT_LOAD", "1")
    for loader in (_torch_load_layered, _torch_load_legacy):
        loaded = loader(canonical)
        assert tuple(loaded["pred_ligand_pos"][0].shape) == (3, 3)
        assert loaded["data"].protein_filename == "protein.pdb"


def test_external_pt_rejects_executable_pickle(tmp_path):
    from server.safe_pt import load_untrusted_pt
    marker = tmp_path / "executed"
    class Payload:
        def __reduce__(self):
            return os.system, (f"touch {marker}",)
    source = tmp_path / "malicious.pt"
    torch.save({"data": Payload()}, source)
    with pytest.raises(ValueError, match="secure loader rejected"):
        load_untrusted_pt(source, "alice")
    assert not marker.exists()


def test_external_project_pt_supports_graph_data_and_atom_indices(tmp_path):
    from datasets.pl_data import ProteinLigandData
    from server.safe_pt import load_untrusted_pt
    graph = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict={"pos": torch.zeros((2, 3))},
        ligand_dict={"pos": torch.zeros((2, 3)), "element": torch.tensor([6, 6]),
                     "bond_index": torch.tensor([[0, 1], [1, 0]])})
    graph.protein_filename = "protein.pdb"
    graph.ligand_filename = "ligand.sdf"
    source = tmp_path / "graph.pt"
    torch.save({"data": graph, "pred_ligand_pos": [np.zeros((2, 3))]}, source)
    loaded = load_untrusted_pt(source, "alice")
    assert loaded["data"].ligand_nbh_list == {0: [1], 1: [0]}
    assert loaded["data"].protein_filename == "protein.pdb"
    assert tuple(loaded["data"].ligand_pos.shape) == (2, 3)


def test_external_pt_cannot_reference_host_files(tmp_path):
    from server.safe_pt import load_untrusted_pt
    source = tmp_path / "paths.pt"
    torch.save({"data": SimpleNamespace(protein_filename="/etc/secret.pdb")}, source)
    with pytest.raises(ValueError, match="not owned"):
        load_untrusted_pt(source, "alice")


def test_manifest_rejects_array_allocation_bomb(tmp_path):
    from server.safe_pt import _decode_manifest
    with (tmp_path / "array_0.npy").open("wb") as stream:
        np.lib.format.write_array_header_1_0(stream, {"shape": (2**40,), "fortran_order": False,
                                                     "descr": "<f8"})
    (tmp_path / "manifest.json").write_text(json.dumps({"version": 1, "tree": {"__array__": "array_0.npy"}}))
    with pytest.raises(ValueError, match="exceeds limits"):
        _decode_manifest(tmp_path)


def test_sandbox_cannot_access_host_secrets_or_loopback(tmp_path):
    from server.safe_pt import _sandbox_command
    source = tmp_path / "source.pt"
    source.write_bytes(b"placeholder")
    secret = tmp_path / "secret"
    secret.write_text("PRIVATE")
    command = _sandbox_command(source, tmp_path)
    command[-1:] = ["-c", f"import pathlib,socket; assert not pathlib.Path({str(secret)!r}).exists(); assert not pathlib.Path('/data/ye/DiffDynamic/server/.internal_api_key').exists(); s=socket.socket(); assert s.connect_ex(('127.0.0.1',7860)) != 0; print('isolated')"]
    result = subprocess.run(command, capture_output=True, timeout=15, env={"PATH": "/usr/bin:/bin"})
    assert result.returncode == 0, result.stderr.decode()
    assert b"isolated" in result.stdout
