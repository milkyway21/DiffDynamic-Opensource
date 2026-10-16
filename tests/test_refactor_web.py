"""Frozen web contracts for module extraction; no production service or GPU jobs."""

import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.routing import APIRoute
from starlette.requests import Request

from server import api, config, scheduler, schemas
from server.job_configs import (
    build_generation_patches,
    build_optimization_patches,
    build_scaffold_patches,
)

# Captured from the pre-refactor HTTP surface, including page routes.
EXPECTED_ROUTES = {
    "POST /api/medchem",
    "GET /api/medchem/runs/{run_id}/report",
    "GET /api/medchem/runs/{run_id}/artifacts/{artifact_id}",
    "POST /api/generate/context",
    "POST /api/scaffold/context",
    "POST /api/scaffold/preview",
    "POST /api/generate",
    "POST /api/generate/batch",
    "POST /api/generate/custom",
    "GET /api/generate/{job_id}",
    "POST /api/evaluate",
    "GET /api/evaluate/{job_id}",
    "POST /api/extract",
    "POST /api/pocket-eval",
    "GET /api/pocket-eval/images/{path:path}",
    "GET /api/pocket-eval/results/{job_id}",
    "POST /api/optimization",
    "GET /api/optimization/{job_id}",
    "POST /api/scaffold",
    "GET /api/adaptive_gates/preview",
    "GET /api/scaffold/{job_id}",
    "POST /api/scaffold-cascade",
    "GET /api/scaffold-cascade/{job_id}",
    "GET /api/jobs",
    "GET /api/jobs/{job_id}",
    "DELETE /api/jobs/{job_id}",
    "GET /api/runs",
    "GET /api/runs/compare",
    "GET /api/runs/{run_id}",
    "POST /api/runs/{run_id}/rerun",
    "GET /api/runs/{run_id}/log",
    "GET /api/runs/{run_id}/export",
    "GET /api/molecules",
    "GET /api/stats",
    "GET /api/browse/files",
    "GET /api/history",
    "GET /api/config",
    "PUT /api/config",
    "GET /api/presets",
    "GET /api/outputs",
    "GET /api/pt/{path:path}",
    "GET /api/sdf",
    "GET /api/sdf/{path:path}",
    "GET /api/file/download",
    "GET /api/proteins",
    "GET /api/gpus",
    "GET /api/health",
    "GET /api/me",
    "GET /api/readme",
    "POST /api/upload",
    "GET /",
    "GET /app",
}


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _defaults(method):
    return {
        name: parameter.default
        for name, parameter in inspect.signature(method).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def _scaffold_options(overrides):
    options = _defaults(scheduler.JobScheduler.submit_scaffold)
    options.update(overrides)
    options["placement"] = options["placement"] or {"mode": "auto", "regions": []}
    if options["scaffold_enable_refine"] is None:
        options["scaffold_enable_refine"] = options["workflow"] != "repaint"
    return options


def _request():
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/generate",
            "headers": [],
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
        }
    )
    request.state.dd_user = "alice"
    request.state.dd_is_admin = False
    return request


def test_all_legacy_http_routes_and_handler_exports_are_preserved():
    routes = [route for route in api.app.routes if isinstance(route, APIRoute)]
    actual = {method + " " + route.path for route in routes for method in route.methods}
    assert len(routes) == 52
    assert actual == EXPECTED_ROUTES
    for route in routes:
        assert getattr(api, route.name) is route.endpoint
        assert route.endpoint.__module__.startswith("server.routes.")


def test_request_schemas_can_load_without_http_app_or_scheduler():
    code = (
        "import sys; from server.schemas import ScaffoldRequest; "
        "assert 'server.api' not in sys.modules; "
        "assert 'server.scheduler' not in sys.modules; "
        "assert ScaffoldRequest().workflow == 'legacy'"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert api.ScaffoldRequest is schemas.ScaffoldRequest
    assert api.GenerateRequest is schemas.GenerateRequest


def test_extracted_routes_keep_live_facade_dependency_overrides(monkeypatch):
    calls = []
    fake = SimpleNamespace(
        submit_generation=lambda **kw: calls.append(kw) or {"job_id": "fake"},
        get_status=lambda job_id, owner=None: {"job_id": job_id, "owner": owner},
    )
    monkeypatch.setattr(api, "get_scheduler", lambda: fake)
    result = api.start_generation(api.GenerateRequest(), _request())
    assert result == {"job_id": "fake"}
    assert calls[0]["triggered_by"] == "alice"
    assert calls[0]["mode"] == "dynamic"
    assert api.get_job("fake", _request()) == {"job_id": "fake", "owner": "alice"}
    marker = {"from": "facade override"}
    monkeypatch.setattr(api, "_require_job", lambda *args: marker)
    assert api.get_generation_status("fake", _request()) is marker


def test_extracted_submission_uses_scheduler_facade_without_running_job(monkeypatch):
    cfg = SimpleNamespace(gpu_ids=[0], default_gpu_id=0, sampling_config="fixture.yml")
    calls = []
    monkeypatch.setattr(scheduler, "get_config", lambda: cfg)
    monkeypatch.setattr(scheduler, "create_run", lambda **kw: calls.append(kw) or 123)
    monkeypatch.setattr(scheduler, "log_history", lambda *a, **kw: None)
    monkeypatch.setattr(
        scheduler, "run_generation", lambda *a, **kw: pytest.fail("GPU job started")
    )
    instance = scheduler.JobScheduler(max_concurrent=1)
    monkeypatch.setattr(instance, "_attach_owner_params", lambda *a, **kw: "alice")
    monkeypatch.setattr(instance, "_read_config_snapshot", lambda path: "fixture: true")
    monkeypatch.setattr(
        instance, "_acquire_or_queue", lambda gpu, rec, starter: rec.to_dict()
    )
    result = instance.submit_generation(triggered_by="alice")
    assert result["run_id"] == 123
    assert calls[0]["owner"] == "alice"
    assert calls[0]["config_snapshot"] == "fixture: true"
    instance._gpu_executor.shutdown(wait=True)
    instance._cpu_executor.shutdown(wait=True)


# SHA-256 values cover the entire original patch and extra run metadata, not
# selected keys. They were captured by executing the old construction blocks
# before extraction, without scheduling or model execution.
SCAFFOLD_PATCH_CASES = [
    ({}, "20966b2589b081faf78e696cf5b2227d066260f8232cd4851b7419fbc908dfbc"),
    (
        {"scaffold_mode": "evolve"},
        "7cdd97365fb918e1f6af9a7adba06241b56e52a3d89be70938bfbb56a1b83fde",
    ),
    (
        {"scaffold_mode": "dynamic_locked"},
        "4dde35009ca5cde9bc76985a78d0ed41e79c7a7fbec875dc27a48139702b6037",
    ),
    (
        {"scaffold_mode": "prudent"},
        "885f557aaaaba33990b7f7d4e5d97230217993edb7a946787e634373b47f9b7a",
    ),
    (
        {"workflow": "repaint"},
        "6048182147fb11da642205dc467436e2b9cce207c5afbb01001bfd758aa39bf7",
    ),
    (
        {"workflow": "repaint", "scaffold_mode": "dynamic_locked"},
        "b21e1c1f6c53ec389ad1bf08393cfad0e4125ce2c55ee7a134ed1f48250557c9",
    ),
    (
        {"workflow": "repaint", "scaffold_mode": "prudent"},
        "8c98f9c779b65460c9a02978f7229b16e597c8e47bef4f22b5194a62fb30a212",
    ),
    (
        {"grow_n_extra_mode": "fixed", "grow_n_extra_fixed": 35},
        "05db6a1029324d2c778d20ab39aa079486740f09dfd59b378fd2493b671efdb1",
    ),
    (
        {
            "scaffold_mode": "dynamic_locked",
            "grow_n_extra_mode": "fixed",
            "grow_n_extra_fixed": 50,
        },
        "be173ef5a55f3c2629f5996f10d4812715bdb6cb6d9df246893bd6c5ca14710d",
    ),
    (
        {
            "placement": {
                "mode": "custom",
                "regions": [
                    {
                        "anchor_atom_index": 1,
                        "center_xyz": [1.0, 2.0, 3.0],
                        "atom_count": 3,
                        "method": "gaussian",
                        "sigma_axial": 1.0,
                        "sigma_lateral": 1.0,
                    }
                ],
            }
        },
        "f9d991601f65078bb785f383ff5a416324683f8cacb73c3a919450246a10258c",
    ),
    (
        {
            "scaffold_mode": "prudent",
            "gate_mode": "manual",
            "manual_max_logp": 4.2,
            "manual_max_molwt": 600.0,
            "manual_max_heavy_atoms": 42,
            "manual_max_rings": 5,
        },
        "2b0490a38c816219aff4df3b625179584d68b047d44925734034a55b9447ad1a",
    ),
    (
        {
            "scaffold_mode": "prudent",
            "adaptive_gates_enable": False,
            "manual_max_logp": None,
            "manual_max_molwt": None,
            "manual_max_heavy_atoms": None,
            "manual_max_rings": None,
        },
        "dbf79b3c68375bd3a5b9cfc3848e5eb8641b076974e4d328b030c990ff093ff0",
    ),
]


@pytest.mark.parametrize("overrides, expected", SCAFFOLD_PATCH_CASES)
def test_all_scaffold_patch_values_match_pre_refactor(overrides, expected):
    options = _scaffold_options(overrides)
    before = json.dumps(options, sort_keys=True)
    assert _digest(build_scaffold_patches(options)) == expected
    assert json.dumps(options, sort_keys=True) == before


def test_original_scaffold_submission_defaults_are_unchanged():
    assert _digest(_defaults(scheduler.JobScheduler.submit_scaffold)) == (
        "7b88e936fb910764929d63133247a133a3c12e36a8faa42a4f99b283e26e6b97"
    )


def test_optimization_patch_values_match_pre_refactor():
    options = _defaults(scheduler.JobScheduler.submit_optimization)
    assert _digest(build_optimization_patches(options)) == (
        "41654f4c099d4560b110f242eb67579b0f9707207c68ccc2688e9cc8cbe7c8bf"
    )


def test_generation_patch_keeps_dynamic_and_legacy_prudent_defaults():
    options = _defaults(scheduler.JobScheduler.submit_generation)
    assert build_generation_patches(options) == {
        "sample": {
            "mode": "dynamic",
            "dynamic": {
                "skip_refine": False,
                "prudent": {"enable": False},
                "large_step": {"batch_size": 5},
            },
            "scaffold": {"enable": False},
            "optimization": {"enable": False},
        },
    }
    options["mode"] = "prudent"
    assert build_generation_patches(options)["sample"]["dynamic"]["prudent"] == {
        "enable": True,
        "n_sampling": 10,
        "advance_top_k": 2,
        "max_checkpoint_rounds": 1,
        "refine_batch_size": 250,
        "scoring_workers": 100,
    }


@pytest.mark.parametrize("variable, field", list(config.PATH_ENVIRONMENT.items()))
def test_explicit_environment_path_overrides_saved_json(
    monkeypatch, tmp_path, variable, field
):
    saved = {name: "/saved/" + name for name in config.PATH_ENVIRONMENT.values()}
    saved["gpu_ids"] = [0]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(saved))
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    for key in config.PATH_ENVIRONMENT:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(variable, "/explicit/" + field)
    loaded = config.load_config()
    for name in config.PATH_ENVIRONMENT.values():
        assert getattr(loaded, name) == (
            "/explicit/" + field if name == field else saved[name]
        )
    assert json.loads(path.read_text()) == saved


def test_saved_config_paths_are_retained_without_environment_overrides(
    monkeypatch, tmp_path
):
    saved = {name: "/saved/" + name for name in config.PATH_ENVIRONMENT.values()}
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**saved, "gpu_ids": [0], "unknown_legacy_key": True}))
    monkeypatch.setattr(config, "CONFIG_PATH", str(path))
    for key in config.PATH_ENVIRONMENT:
        monkeypatch.delenv(key, raising=False)
    loaded = config.load_config()
    assert {name: getattr(loaded, name) for name in saved} == saved


def test_pt_loader_uses_current_config_instead_of_import_time_override(
    monkeypatch, tmp_path
):
    from server import safe_pt

    root = tmp_path / "isolated-outputs"
    monkeypatch.setattr(
        config, "get_config", lambda: SimpleNamespace(output_root=str(root))
    )
    assert safe_pt.is_generated_pt(root / "result.pt")
    assert not safe_pt.is_generated_pt(tmp_path / "other" / "result.pt")


def test_readme_route_still_reads_project_content_after_route_relocation():
    expected = Path(api.__file__).parent.parent / "ui/content/overview.zh.md"
    assert api.get_readme("zh")["content"] == expected.read_text()


@pytest.mark.parametrize("legacy_override", [False, True])
def test_scaffold_preview_uses_shared_helpers_and_keeps_legacy_overrides(
    monkeypatch, legacy_override
):
    import builtins

    import torch
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from server import scaffold_context

    molecule = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(molecule, randomSeed=42)
    molecule = Chem.RemoveHs(molecule)
    protein = torch.tensor([[10.0, 10.0, 10.0]])
    monkeypatch.setattr(
        scaffold_context,
        "load_context",
        lambda params: (molecule, list(range(6)), protein, {}),
    )
    if legacy_override:
        import scripts.sample_diffusion as historical

        monkeypatch.setattr(historical, "_sample_n_extra_atoms", lambda *a, **kw: 1)

    original_import = builtins.__import__

    def import_shared_only(name, *args, **kwargs):
        if name in ("scripts.sample_diffusion", "diffdynamic.sampling.legacy"):
            pytest.fail("Preview imported the historical sampling command")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_shared_only)
    params = schemas.ScaffoldRequest(
        grow_n_extra_mode="fixed",
        grow_n_extra_fixed=2,
    ).dict()
    preview = scaffold_context.preview_context(params)
    assert len(preview["positions"]) == (1 if legacy_override else 2)
