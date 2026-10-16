"""Shared pytest setup.

默认把所有测试绑定到一次性临时 SQLite 库，避免测试读写生产库
diffdynamic.db（生产库中 molecules 存在历史重复行，会让依赖唯一
索引的用例产生 ON CONFLICT 报错）。显式设置 DD_DB_PATH 时不覆盖。
"""

import os
import tempfile
import pytest

if not os.environ.get("DD_DB_PATH"):
    _tmp_dir = tempfile.mkdtemp(prefix="dd_tests_")
    os.environ["DD_DB_PATH"] = os.path.join(_tmp_dir, "test.db")
os.environ.setdefault("DD_AUTH_MODE", "off")
os.environ.setdefault("DD_CLEANUP_ON_START", "0")


@pytest.fixture(autouse=True)
def local_platform_sessions_only(monkeypatch):
    """Tests must never contact the real platform session service."""
    monkeypatch.setattr("server.api.lookup_platform_identity", lambda **kwargs: None)
