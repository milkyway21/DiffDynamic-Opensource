"""Public prefix + locale login URL helpers (no GPU)."""

from server.auth import (
    locale_from_prefix,
    normalize_forwarded_prefix,
    normalize_owner,
    platform_home_url_for_root,
    platform_login_url_for_root,
    request_public_root,
)


class _H(dict):
    def get(self, key, default=None):
        for k, v in self.items():
            if k.lower() == str(key).lower():
                return v
        return default


def test_normalize_forwarded_prefix_allowlist():
    assert normalize_forwarded_prefix("/en/diffdynamic/") == "/en/diffdynamic"
    assert normalize_forwarded_prefix("/zh/diffdynamic") == "/zh/diffdynamic"
    assert normalize_forwarded_prefix("/diffdynamic/") == "/diffdynamic"
    assert normalize_forwarded_prefix("/evil") == ""
    assert normalize_forwarded_prefix("/zh/tamevs") == ""


def test_request_public_root_prefers_header(monkeypatch):
    monkeypatch.setenv("DD_ROOT_PATH", "/zh/diffdynamic")
    assert request_public_root(_H({"X-Forwarded-Prefix": "/en/diffdynamic/"})) == "/en/diffdynamic"
    assert request_public_root(_H({})) == "/zh/diffdynamic"


def test_login_and_home_follow_locale(monkeypatch):
    monkeypatch.setenv("DD_PLATFORM_LOGIN_URL", "https://aidd.shu.edu.cn/zh/signin")
    monkeypatch.setenv("DD_PLATFORM_HOME_URL", "https://aidd.shu.edu.cn/zh")
    assert platform_login_url_for_root("/en/diffdynamic") == "https://aidd.shu.edu.cn/en/signin"
    assert platform_home_url_for_root("/en/diffdynamic") == "https://aidd.shu.edu.cn/en"
    assert platform_login_url_for_root("/zh/diffdynamic") == "https://aidd.shu.edu.cn/zh/signin"
    assert locale_from_prefix("/diffdynamic") == "en"
    assert platform_login_url_for_root("/diffdynamic") == "https://aidd.shu.edu.cn/en/signin"


def test_normalize_owner_blank_only_legacy():
    assert normalize_owner("") == "legacy"
    assert normalize_owner(None) == "legacy"
    assert normalize_owner("  ") == "legacy"
    assert normalize_owner("alice") == "alice"
    assert normalize_owner("internal") == "internal"
    assert normalize_owner("web_ui") == "web_ui"
    assert normalize_owner("go_live_verify") == "go_live_verify"


def test_admin_whitelist_email(monkeypatch):
    from server.auth import is_admin_user
    monkeypatch.setenv("DD_ADMIN_USERS", "quota_web_admin,release_admin@example.invalid")
    assert is_admin_user("release_admin@example.invalid") is True
    assert is_admin_user("release_admin@EXAMPLE.INVALID") is False
    assert is_admin_user("quota_web_bob") is False
    assert is_admin_user("someone", is_superuser=True) is True
