"""Focused security and output-contract tests; no GPU jobs are started."""

from types import SimpleNamespace

import pytest
from starlette.requests import Request

from server.api import serve_pocket_eval_image
from server.data_manager import scan_eval_dirs, scan_outputs
from server.runners.base import validate_config_path
from server.auth import owner_slug


def _request(username: str = "alice", is_admin: bool = False) -> Request:
    request = Request({
        "type": "http",
        "method": "GET",
        "path": "/api/pocket-eval/images/test.png",
        "headers": [],
        "query_string": b"",
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
        "scheme": "http",
    })
    request.state.dd_user = username
    request.state.dd_is_admin = is_admin
    return request


def test_validate_config_path_rejects_sibling_prefix(tmp_path):
    root = tmp_path / "DiffDynamic"
    configs = root / "configs"
    sibling = root / "configs_evil"
    configs.mkdir(parents=True)
    sibling.mkdir()
    valid = configs / "run.yml"
    invalid = sibling / "run.yml"
    valid.write_text("sample: {}", encoding="utf-8")
    invalid.write_text("sample: {}", encoding="utf-8")
    cfg = SimpleNamespace(
        diffdynamic_root=str(root),
        sampling_config=str(valid),
    )

    assert validate_config_path(str(valid), cfg) == str(valid.resolve())
    with pytest.raises(ValueError):
        validate_config_path(str(invalid), cfg)


def test_user_output_scanners_include_pt_and_eval_under_pt(tmp_path):
    root = tmp_path / "outputs" / "users" / "alice"
    pt_dir = root / "pt"
    eval_dir = pt_dir / "eval_job-1"
    pt_dir.mkdir(parents=True)
    eval_dir.mkdir()
    (pt_dir / "result_7.pt").write_bytes(b"stub")

    outputs = scan_outputs(str(root))
    evals = scan_eval_dirs(str(root))

    assert [item["filename"] for item in outputs] == ["result_7.pt"]
    assert [item["dirname"] for item in evals] == ["eval_job-1"]


def test_pocket_image_requires_owner_tree(monkeypatch, tmp_path):
    output_root = tmp_path / "outputs"
    user_dir = output_root / "users" / owner_slug("alice") / "pt" / "pocket_eval_job"
    user_dir.mkdir(parents=True)
    image = user_dir / "score.png"
    image.write_bytes(b"png")
    cfg = SimpleNamespace(
        output_root=str(output_root),
        data_root=str(tmp_path / "data"),
        upload_dir=str(tmp_path / "uploads"),
        diffdynamic_root=str(tmp_path / "DiffDynamic"),
    )
    monkeypatch.setattr("server.api.get_config", lambda: cfg)

    response = serve_pocket_eval_image(
        f"users/{owner_slug('alice')}/pt/pocket_eval_job/score.png", _request(),
    )
    assert response.path == str(image)

    with pytest.raises(Exception) as exc_info:
        serve_pocket_eval_image(
            "../secret.png", _request(),
        )
    assert getattr(exc_info.value, "status_code", None) == 403
