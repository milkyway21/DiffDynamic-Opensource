"""PT file access-control tests.

Remote users may only read .pt files they generated (outputs/users/{owner}/)
or uploaded (uploads/users/{owner}/). Shared data/configs roots stay readable
for pocket/preset files but must never expose .pt to non-admins. Uploaded .pt
is loaded strictly (no pickle fallback) by the eval subprocesses.

No GPU jobs are started; API endpoints are exercised via TestClient with an
isolated temp tree and a throwaway SQLite database.
"""

import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

# conftest.py 已将 DD_DB_PATH / DD_AUTH_MODE 指向隔离的临时库
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import api as server_api  # noqa: E402
from server.auth import owner_slug  # noqa: E402


@pytest.fixture()
def fs(tmp_path, monkeypatch):
    """Isolated directory tree + fake config patched into server.api."""
    root = tmp_path
    cfg = SimpleNamespace(
        output_root=str(root / "outputs"),
        data_root=str(root / "data"),
        upload_dir=str(root / "uploads"),
        diffdynamic_root=str(root),
        max_upload_bytes=10 * 1024 * 1024,
        max_submit_per_min=100,
        max_upload_per_min=100,
        max_jobs_per_user=2,
        max_generate_per_day=1,
        max_eval_per_day=1,
    )
    (root / "outputs" / "users" / owner_slug("alice") / "pt").mkdir(parents=True)
    (root / "outputs" / "users" / owner_slug("bob") / "pt").mkdir(parents=True)
    (root / "data" / "crossdocked_pocket10_test_only").mkdir(parents=True)
    (root / "configs").mkdir(parents=True)
    (root / "outputs" / "users" / owner_slug("alice") / "pt" / "result_0_own.pt").write_bytes(b"own")
    (root / "outputs" / "users" / owner_slug("bob") / "pt" / "result_0_bob.pt").write_bytes(b"bob")
    (root / "data" / "split.pt").write_bytes(b"shared-pt")
    (root / "data" / "crossdocked_pocket10_test_only" / "prot.pdb").write_text("ATOM\n")
    (root / "configs" / "run_x.yml").write_text("sample: {}\n")
    (root / "configs" / "weights.pt").write_bytes(b"cfg-pt")
    monkeypatch.setattr(server_api, "get_config", lambda: cfg)
    yield cfg


def _fake_request(user: str, is_admin: bool = False):
    return SimpleNamespace(state=SimpleNamespace(dd_user=user, dd_is_admin=is_admin))


def _client():
    return TestClient(server_api.app)


def _auth_headers(client: TestClient, user: str) -> dict:
    return {
        "X-DD-Internal-Key": server_api._get_internal_api_key(),
        "X-DD-User": user,
    }


# ── assert_path_acl core rules ───────────────────────────────────────────

def test_acl_allows_own_generated_pt(fs):
    own_pt = os.path.join(fs.output_root, "users", owner_slug("alice"), "pt", "result_0_own.pt")
    got = server_api.assert_path_acl(own_pt, _fake_request("alice"), field="pt_path")
    assert got == os.path.realpath(own_pt)


def test_acl_allows_own_upload_pt(fs):
    up = os.path.join(fs.upload_dir, "users", owner_slug("alice"), "u", "x.pt")
    got = server_api.assert_path_acl(up, _fake_request("alice"), field="pt_path")
    assert got == os.path.realpath(up)


def test_acl_denies_other_users_outputs(fs):
    other = os.path.join(fs.output_root, "users", owner_slug("bob"), "pt", "result_0_bob.pt")
    with pytest.raises(HTTPException) as e:
        server_api.assert_path_acl(other, _fake_request("alice"), field="pt_path")
    assert e.value.status_code == 403


def test_acl_denies_shared_data_pt(fs):
    with pytest.raises(HTTPException) as e:
        server_api.assert_path_acl(os.path.join(fs.data_root, "split.pt"), _fake_request("alice"))
    assert e.value.status_code == 403


def test_acl_denies_shared_configs_pt(fs):
    with pytest.raises(HTTPException) as e:
        server_api.assert_path_acl(os.path.join(fs.diffdynamic_root, "configs", "weights.pt"), _fake_request("alice"))
    assert e.value.status_code == 403


def test_acl_allows_shared_data_pdb_for_pockets(fs):
    pdb = os.path.join(fs.data_root, "crossdocked_pocket10_test_only", "prot.pdb")
    assert server_api.assert_path_acl(pdb, _fake_request("alice")) == os.path.realpath(pdb)


def test_acl_admin_unrestricted(fs):
    shared_pt = os.path.join(fs.data_root, "split.pt")
    assert server_api.assert_path_acl(shared_pt, _fake_request("admin", is_admin=True))


# ── Endpoint behaviour via TestClient ────────────────────────────────────

def test_browse_data_hides_pt_from_non_admin(fs):
    with _client() as client:
        top = client.get(
            "/api/browse/files",
            params={"root": "data"},
            headers=_auth_headers(client, "alice"),
        )
        sub = client.get(
            "/api/browse/files",
            params={"root": "data", "subpath": "crossdocked_pocket10_test_only", "ext": ".pdb"},
            headers=_auth_headers(client, "alice"),
        )
    assert top.status_code == 200
    top_names = [f["name"] for f in top.json()["files"]]
    assert not any(f.endswith(".pt") for f in top_names)
    assert sub.status_code == 200
    sub_names = [f["name"] for f in sub.json()["files"]]
    assert "prot.pdb" in sub_names


def test_browse_data_shows_pt_to_admin(fs, monkeypatch):
    monkeypatch.setenv("DD_ADMIN_USERS", "quota_web_admin")
    with _client() as client:
        r = client.get(
            "/api/browse/files",
            params={"root": "data"},
            headers=_auth_headers(client, "quota_web_admin"),
        )
    assert r.status_code == 200
    names = [f["name"] for f in r.json()["files"]]
    assert "split.pt" in names


def test_browse_outputs_jailed_to_own_tree(fs):
    with _client() as client:
        r = client.get(
            "/api/browse/files",
            params={"root": "outputs", "subpath": f"users/{owner_slug('alice')}/pt"},
            headers=_auth_headers(client, "alice"),
        )
    assert r.status_code == 200
    body = r.json()
    assert f"users/{owner_slug('alice')}" in body["subpath"]
    names = [f["name"] for f in body["files"]]
    assert "result_0_own.pt" in names
    assert "result_0_bob.pt" not in names


def test_upload_pt_lands_in_own_tree(fs):
    with _client() as client:
        r = client.post(
            "/api/upload",
            files={"file": ("my_result.pt", b"\x80\x04fake-pickle", "application/octet-stream")},
            headers=_auth_headers(client, "alice"),
        )
    assert r.status_code == 200
    body = r.json()
    assert body["stored_path"].startswith(os.path.join(fs.upload_dir, "users", owner_slug("alice")))
    assert os.path.exists(body["stored_path"])


def test_upload_rejects_unknown_ext(fs):
    with _client() as client:
        r = client.post(
            "/api/upload",
            files={"file": ("evil.sh", b"#!/bin/sh\n", "text/x-sh")},
            headers=_auth_headers(client, "alice"),
        )
    assert r.status_code == 400


def test_download_own_pt_ok_and_shared_pt_denied(fs):
    own_pt = os.path.join(fs.output_root, "users", owner_slug("alice"), "pt", "result_0_own.pt")
    shared_pt = os.path.join(fs.data_root, "split.pt")
    with _client() as client:
        ok = client.get(
            "/api/file/download",
            params={"path": own_pt},
            headers=_auth_headers(client, "alice"),
        )
        denied = client.get(
            "/api/file/download",
            params={"path": shared_pt},
            headers=_auth_headers(client, "alice"),
        )
        other = client.get(
            "/api/file/download",
            params={"path": os.path.join(fs.output_root, "users", owner_slug("bob"), "pt", "result_0_bob.pt")},
            headers=_auth_headers(client, "alice"),
        )
    assert ok.status_code == 200
    assert ok.content == b"own"
    assert denied.status_code == 403
    assert other.status_code == 403


def test_download_traversal_denied(fs):
    with _client() as client:
        r = client.get(
            "/api/file/download",
            params={"path": os.path.join(fs.data_root, "..", "..", "server", "api.py")},
            headers=_auth_headers(client, "alice"),
        )
    assert r.status_code in (400, 403, 404)


# ── Symlinked data roots (test-set dir is a symlink off data_root) ───────

def test_symlinked_data_pdb_allowed_but_pt_denied(fs, tmp_path):
    from server.data_manager import browse_directory
    external = tmp_path / "external_pockets" / "pock1"
    external.mkdir(parents=True)
    (external / "prot.pdb").write_text("ATOM\n")
    (external / "weights.pt").write_bytes(b"pt")
    link = os.path.join(fs.data_root, "linked_pockets")
    os.symlink(str(tmp_path / "external_pockets"), link)
    # 重新构建配置感知的共享根（cfg 已 patch，直接调真实函数）
    pdb_resolved = os.path.realpath(os.path.join(link, "pock1", "prot.pdb"))
    pt_resolved = os.path.realpath(os.path.join(link, "pock1", "weights.pt"))
    assert server_api.assert_path_acl(pdb_resolved, _fake_request("alice")) == pdb_resolved
    with pytest.raises(HTTPException) as e:
        server_api.assert_path_acl(pt_resolved, _fake_request("alice"))
    assert e.value.status_code == 403
    # 文件浏览器能列出软链目标内容
    extras = server_api._top_level_symlink_roots(fs.data_root)
    assert any(x.endswith("external_pockets") for x in extras)
    entries = browse_directory(
        fs.data_root, os.path.join("linked_pockets", "pock1"),
        (".pdb",), extra_roots=extras,
    )
    assert [x["name"] for x in entries] == ["prot.pdb"]


def test_browse_symlinked_data_dir_via_api(fs, tmp_path):
    external = tmp_path / "ext2" / "pock2"
    external.mkdir(parents=True)
    (external / "x.pdb").write_text("ATOM\n")
    os.symlink(str(tmp_path / "ext2"), os.path.join(fs.data_root, "ln2"))
    with _client() as client:
        r = client.get(
            "/api/browse/files",
            params={"root": "data", "subpath": "ln2/pock2", "ext": ".pdb"},
            headers=_auth_headers(client, "alice"),
        )
    assert r.status_code == 200
    assert [f["name"] for f in r.json()["files"]] == ["x.pdb"]


# ── Platform-mode internal-key shortcut ──────────────────────────────────

def test_health_login_url_follows_forwarded_prefix(fs, monkeypatch):
    monkeypatch.setenv("DD_AUTH_MODE", "platform")
    with _client() as client:
        en = client.get("/api/health", headers={"X-Forwarded-Prefix": "/en/diffdynamic"})
        zh = client.get("/api/health", headers={"X-Forwarded-Prefix": "/zh/diffdynamic"})
    assert en.status_code == 200
    assert en.json()["login_url"].startswith("https://aidd.shu.edu.cn/en/signin")
    assert zh.json()["login_url"].startswith("https://aidd.shu.edu.cn/zh/signin")

def test_platform_mode_ignores_stale_internal_cookie(fs, monkeypatch):
    """off 模式时代下发的 dd_internal Cookie 不得绕过平台会话校验。"""
    monkeypatch.setenv("DD_AUTH_MODE", "platform")
    async def _no_session(**kwargs):
        return None
    monkeypatch.setattr(server_api, "lookup_platform_identity", lambda **kw: None)
    with _client() as client:
        r = client.get(
            "/api/me",
            headers={"Cookie": f"dd_internal={server_api._get_internal_api_key()}",
                     "X-DD-User": "attacker"},
        )
    assert r.status_code == 401


def test_platform_mode_allows_opted_in_loopback_internal_api(fs, monkeypatch):
    """Internal credentials only bypass sessions on explicitly enabled loopback."""
    monkeypatch.setenv("DD_AUTH_MODE", "platform")
    monkeypatch.setenv("DD_ALLOW_INTERNAL_API", "1")
    with TestClient(server_api.app, client=("127.0.0.1", 50000)) as client:
        r = client.get(
            "/api/me",
            headers={"X-DD-Internal-Key": server_api._get_internal_api_key(),
                     "X-DD-User": "ops_tester"},
        )
    assert r.status_code == 200
    assert r.json()["username"] == "ops_tester"


def test_off_mode_still_accepts_internal_cookie(fs):
    with _client() as client:
        r = client.get(
            "/api/me",
            headers={"Cookie": f"dd_internal={server_api._get_internal_api_key()}"},
        )
    assert r.status_code == 200
    assert r.json()["username"] == "internal"


# ── Subprocess strict-load gate ──────────────────────────────────────────

def test_strict_load_env_for_generated_pt(fs, monkeypatch):
    from server.runners import base as runner_base
    import server.molecule_ingest as mi
    # 生产环境 output_root 即仓库 outputs/；测试里把信任根对齐到 fixture 树
    monkeypatch.setattr(mi, "OUTPUTS_ROOT", str(fs.output_root))
    own_pt = os.path.join(fs.output_root, "users", owner_slug("alice"), "pt", "result_0_own.pt")
    assert runner_base.strict_load_env_for_pt(own_pt) == {}


def test_strict_load_env_for_uploaded_pt(fs):
    from server.runners.base import strict_load_env_for_pt
    up = os.path.join(fs.upload_dir, "users", owner_slug("alice"), "u", "x.pt")
    assert strict_load_env_for_pt(up) == {"DD_PT_STRICT_LOAD": "1"}
    assert strict_load_env_for_pt(os.path.join(fs.data_root, "split.pt")) == {"DD_PT_STRICT_LOAD": "1"}


def test_torch_load_layered_strict_mode_raises(monkeypatch, tmp_path):
    import importlib
    ev = importlib.import_module("evaluate_pt_with_correct_reconstruct")
    bad = tmp_path / "untrusted.pt"
    bad.write_bytes(b"not a real torch file")
    monkeypatch.setenv("DD_PT_STRICT_LOAD", "1")
    with pytest.raises((ValueError, Exception)):
        ev._torch_load_layered(str(bad))


def test_torch_load_layered_strict_mode_never_pickles(monkeypatch, tmp_path):
    """A crafted pickle payload must raise, never execute, under strict mode."""
    import io
    import pickle
    import importlib

    class Boom:
        def __reduce__(self):
            return os.system, ("echo pwned > /dev/null",)

    ev = importlib.import_module("evaluate_pt_with_correct_reconstruct")
    payload = io.BytesIO()
    pickle.dump({"results": Boom()}, payload)
    bad = tmp_path / "evil.pt"
    bad.write_bytes(payload.getvalue())
    monkeypatch.setenv("DD_PT_STRICT_LOAD", "1")
    with pytest.raises(Exception):
        ev._torch_load_layered(str(bad))
