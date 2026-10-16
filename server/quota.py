"""Per-user daily generate/eval caps (Asia/Shanghai calendar day)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence, Tuple

SHANGHAI = timezone(timedelta(hours=8))

EVAL_RUN_TYPES = frozenset({"evaluate", "pocket_eval", "medchem_dock"})


def shanghai_day_bounds(now: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    """UTC-aware [start, end) for the current calendar day in Asia/Shanghai."""
    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    local = now_utc.astimezone(SHANGHAI)
    start_local = local.replace(hour=0, minute=0, second=0, microsecond=0)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def seconds_until_reset(now: Optional[datetime] = None) -> int:
    _, end = shanghai_day_bounds(now)
    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    return max(1, int((end - now_utc).total_seconds()))


def daily_bucket_for_run_type(run_type: str) -> Optional[str]:
    rt = (run_type or "").strip()
    if not rt:
        return None
    if rt.startswith("generate_") or rt.startswith("scaffold_") or rt == "optimization":
        return "generate"
    if rt in EVAL_RUN_TYPES:
        return "eval"
    return None


def run_types_for_bucket(bucket: str) -> Sequence[str]:
    if bucket == "eval":
        return tuple(sorted(EVAL_RUN_TYPES))
    return ()


def is_generate_run_type(run_type: str) -> bool:
    return daily_bucket_for_run_type(run_type) == "generate"


def is_eval_run_type(run_type: str) -> bool:
    return daily_bucket_for_run_type(run_type) == "eval"


def quota_message(bucket: str, used: int, limit: int) -> str:
    reset = "将于北京时间 0:00 重置"
    if bucket == "eval":
        return f"今日评估次数已用完（{used}/{limit}）。{reset}"
    return f"今日生成/骨架次数已用完（{used}/{limit}）。{reset}"


def auto_eval_blocked_message() -> str:
    return "今日评估次数已用完，请取消「自动评估」或明日再试。"
