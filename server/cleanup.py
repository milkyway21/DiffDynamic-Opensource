"""Safe disk hygiene for DiffDynamic: remove rebuildable temp artifacts only.

Deletes (never touches results):
  1. docktmp/tmp_docking_*  directories older than --age-days (Vina scratch)
  2. outputs/**/eval_results_*_intermediate_latest.pt older than --age-days
     (intermediate snapshots; the final eval_results_*_final_*.pt is kept)

CLI:
    python -m server.cleanup --dry-run          # preview only
    python -m server.cleanup --age-days 7       # actually delete
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from typing import Dict

from server.config import get_config

DOCKTMP_PREFIX = "tmp_docking_"
INTERMEDIATE_SUFFIX = "_intermediate_latest.pt"


def _is_older(path: str, cutoff_ts: float) -> bool:
    try:
        return os.stat(path).st_mtime < cutoff_ts
    except OSError:
        return False


def _dir_size(path: str) -> int:
    total = 0
    try:
        for entry in os.scandir(path):
            try:
                if entry.is_symlink():
                    continue
                if entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
                elif entry.is_dir(follow_symlinks=False):
                    total += _dir_size(entry.path)
            except OSError:
                continue
    except OSError:
        pass
    return total


def cleanup_docktmp(age_days: float, dry_run: bool) -> Dict[str, object]:
    """Remove stale Vina scratch dirs under docktmp/."""
    cfg = get_config()
    root = os.path.join(cfg.diffdynamic_root, "docktmp")
    cutoff = time.time() - age_days * 86400
    stats = {"removed": 0, "freed_bytes": 0, "kept": 0, "errors": 0}
    if not os.path.isdir(root):
        return stats
    with os.scandir(root) as it:
        entries = sorted(
            (e for e in it if e.name.startswith(DOCKTMP_PREFIX)),
            key=lambda e: e.name,
        )
    for entry in entries:
        try:
            if not _is_older(entry.path, cutoff):
                stats["kept"] += 1
                continue
            size = _dir_size(entry.path)
            if dry_run:
                stats["removed"] += 1
                stats["freed_bytes"] += size
                continue
            shutil.rmtree(entry.path, ignore_errors=True)
            if not os.path.exists(entry.path):
                stats["removed"] += 1
                stats["freed_bytes"] += size
            else:
                stats["errors"] += 1
        except OSError:
            stats["errors"] += 1
    return stats


def cleanup_intermediate_pt(age_days: float, dry_run: bool) -> Dict[str, object]:
    """Remove stale intermediate eval snapshots under outputs/ (finals kept)."""
    cfg = get_config()
    cutoff = time.time() - age_days * 86400
    stats = {"removed": 0, "freed_bytes": 0, "kept": 0, "errors": 0}
    roots = [cfg.output_root]
    users_root = os.path.join(cfg.output_root, "users")
    if os.path.isdir(users_root):
        roots.append(users_root)
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if not name.endswith(INTERMEDIATE_SUFFIX):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    if not _is_older(path, cutoff):
                        stats["kept"] += 1
                        continue
                    size = os.path.getsize(path)
                    if dry_run:
                        stats["removed"] += 1
                        stats["freed_bytes"] += size
                        continue
                    os.remove(path)
                    stats["removed"] += 1
                    stats["freed_bytes"] += size
                except OSError:
                    stats["errors"] += 1
    return stats


def run_cleanup(age_days: float = 7.0, dry_run: bool = False) -> Dict[str, object]:
    report = {
        "dry_run": dry_run,
        "age_days": age_days,
        "docktmp": cleanup_docktmp(age_days, dry_run),
        "intermediate_pt": cleanup_intermediate_pt(age_days, dry_run),
    }
    return report


def _fmt(nbytes: int) -> str:
    value = float(nbytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}GB"


def main() -> int:
    parser = argparse.ArgumentParser(description="DiffDynamic temp-file cleanup")
    parser.add_argument("--age-days", type=float, default=7.0,
                        help="delete artifacts older than this many days (default 7)")
    parser.add_argument("--dry-run", action="store_true", help="preview without deleting")
    args = parser.parse_args()

    report = run_cleanup(age_days=args.age_days, dry_run=args.dry_run)
    mode = "[dry-run] " if args.dry_run else ""
    for section in ("docktmp", "intermediate_pt"):
        s = report[section]
        print(f"{mode}{section}: removed={s['removed']} kept={s['kept']} "
              f"errors={s['errors']} freed={_fmt(s['freed_bytes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
