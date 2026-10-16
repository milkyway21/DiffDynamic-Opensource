"""Daily quota helpers (no GPU)."""
from datetime import datetime, timezone

from server.quota import daily_bucket_for_run_type, naive_utc, shanghai_day_bounds


def test_run_type_buckets():
    assert daily_bucket_for_run_type("generate_dynamic") == "generate"
    assert daily_bucket_for_run_type("generate_batch") == "generate"
    assert daily_bucket_for_run_type("scaffold_grow") == "generate"
    assert daily_bucket_for_run_type("scaffold_cascade") == "generate"
    assert daily_bucket_for_run_type("optimization") == "generate"
    assert daily_bucket_for_run_type("evaluate") == "eval"
    assert daily_bucket_for_run_type("pocket_eval") == "eval"
    assert daily_bucket_for_run_type("extract") is None


def test_shanghai_day_bounds_utc_naive():
    # 2026-09-04 19:41 CST == 2026-09-04 11:41 UTC → day starts 2026-09-03 16:00 UTC
    now = datetime(2026, 9, 4, 11, 41, tzinfo=timezone.utc)
    start, end = shanghai_day_bounds(now)
    assert naive_utc(start) == datetime(2026, 9, 3, 16, 0, 0)
    assert naive_utc(end) == datetime(2026, 9, 4, 16, 0, 0)
