#!/usr/bin/env python3
"""Monitor and resume the long-running GSPT1 scaffold campaign.

The campaign itself performs generation, reconstruction, and per-round
similarity audits. This watchdog adds a 30-minute health check, resumes a
dead campaign with the next job seed, and stops after an exact reachable hit
or a strong target-side similarity hit.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(
    "/data/zhang/Ye/DiffDynamic_outputs/hsvpol/"
    "molglue_ikzf2_gspt1/diffdynamic/gspt1_scaffold_repaint_v3"
)
DEFAULT_BASE_SEED = 20280000
DEFAULT_INTERVAL = 1800
SEED_PATTERN = re.compile(r"run_scaffold_seed_(\d+)\.yml$")


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _latest_similarity(root: Path) -> dict:
    summaries = []
    latest = _read_json(root / "latest_summary.json")
    if latest:
        summaries.append(latest)
    for path in root.glob("rounds/*/run_*/audit/summary.json"):
        summary = _read_json(path)
        if summary:
            summaries.append(summary)

    class_rows = []
    for summary in summaries:
        classes = summary.get("classes") or {}
        if classes:
            records = classes.items()
        else:
            records = [(summary.get("class_name", "unknown"), summary)]
        for class_name, value in records:
            class_rows.append({
                "class_name": class_name,
                "best_side_similarity": float(
                    value.get("best_side_similarity", 0.0)
                ),
                "best_full_similarity": float(
                    value.get("best_full_similarity", 0.0)
                ),
                "exact_reachable_count": int(
                    value.get("exact_reachable_count", 0)
                ),
                "generated_unique": int(value.get("generated_unique", 0)),
                "round": summary.get("round"),
            })
    best = max(
        class_rows,
        key=lambda row: (
            row["best_side_similarity"],
            row["best_full_similarity"],
        ),
        default={
            "class_name": None,
            "best_side_similarity": 0.0,
            "best_full_similarity": 0.0,
            "exact_reachable_count": 0,
            "generated_unique": 0,
        },
    )
    return {
        "round": best.get("round"),
        "best": best,
        "classes": class_rows,
        "exact_reachable_count": sum(
            row["exact_reachable_count"] for row in class_rows
        ),
    }


def _max_seen_seed(root: Path) -> int | None:
    maximum = None
    for path in root.rglob("run_scaffold_seed_*.yml"):
        match = SEED_PATTERN.search(path.name)
        if match:
            value = int(match.group(1))
            maximum = value if maximum is None else max(maximum, value)
    return maximum


def _next_start_seed(root: Path, base_seed: int) -> int:
    state = _read_json(root / "state.json") or {}
    if int(state.get("next_job_id", 0)) > 0:
        return int(base_seed)
    maximum = _max_seen_seed(root)
    return max(int(base_seed), int(maximum) + 1 if maximum is not None else int(base_seed))


def _pid_from_file(path: Path) -> int | None:
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return None
    return pid if _alive(pid) else None


def _stop_process_group(pid: int, logger) -> None:
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
        logger(f"sent SIGTERM to campaign process group pgid={pgid}")
    except (ProcessLookupError, PermissionError, OSError) as exc:
        logger(f"failed to stop campaign pid={pid}: {exc}")


def _launch_campaign(root: Path, base_seed: int, hours: float, samples: int, logger) -> int:
    start_seed = _next_start_seed(root, base_seed)
    log_path = root / "campaign_launcher.log"
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":"),
        "SCAFFOLD_RECONSTRUCT_WORKERS": "30",
        "PYTHONUNBUFFERED": "1",
    })
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/gspt1_class_campaign.py"),
        "--run",
        "--resume",
        "--hours",
        str(hours),
        "--samples",
        str(samples),
        "--start-seed",
        str(start_seed),
        "--gpus",
        "3,4,5",
    ]
    log_handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()
    (root / "campaign.pid").write_text(str(process.pid) + "\n", encoding="utf-8")
    logger(
        f"launched campaign pid={process.pid} start_seed={start_seed} "
        f"samples={samples} hours={hours}"
    )
    return process.pid


def run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "watchdog.lock"
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("watchdog already running", file=sys.stderr)
            return 2

        log_path = root / "watchdog.log"

        def logger(message: str) -> None:
            line = f"[{time.strftime('%Y-%m-%d %H:%M:%S %z')}] {message}"
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

        campaign_pid_path = root / "campaign.pid"
        if args.attach_pid:
            campaign_pid_path.write_text(
                str(int(args.attach_pid)) + "\n", encoding="utf-8"
            )

        campaign_pid = _pid_from_file(campaign_pid_path)
        if campaign_pid is None and not _read_json(root / "EXACT_MATCH_FOUND"):
            campaign_pid = _launch_campaign(
                root, args.base_seed, args.hours, args.samples, logger
            )
        elif campaign_pid is not None:
            logger(f"attached to existing campaign pid={campaign_pid}")

        _write_json(root / "watchdog_state.json", {
            "interval_seconds": args.interval_seconds,
            "similarity_threshold": args.similarity_threshold,
            "campaign_pid": campaign_pid,
            "started_at_unix": time.time(),
        })

        while True:
            similarity = _latest_similarity(root)
            state = _read_json(root / "state.json") or {}
            best = similarity["best"]
            logger(
                f"health round={similarity['round']} "
                f"next_job_id={state.get('next_job_id')} "
                f"campaign_alive={bool(campaign_pid and _alive(campaign_pid))} "
                f"best_class={best['class_name']} "
                f"best_side={best['best_side_similarity']:.4f} "
                f"best_full={best['best_full_similarity']:.4f} "
                f"exact_reachable={similarity['exact_reachable_count']}"
            )

            exact_hit = similarity["exact_reachable_count"] > 0
            similar_hit = (
                best["best_side_similarity"] >= args.similarity_threshold
            )
            if _read_json(root / "EXACT_MATCH_FOUND") or exact_hit or similar_hit:
                reason = "exact_reachable" if exact_hit else "side_similarity"
                _write_json(root / "SIMILARITY_TARGET_FOUND.json", {
                    "reason": reason,
                    "threshold": args.similarity_threshold,
                    "similarity": similarity,
                    "found_at_unix": time.time(),
                })
                logger(f"target reached ({reason}); stopping campaign")
                if campaign_pid and _alive(campaign_pid):
                    _stop_process_group(campaign_pid, logger)
                return 0

            if campaign_pid is None or not _alive(campaign_pid):
                campaign_pid = _launch_campaign(
                    root, args.base_seed, args.hours, args.samples, logger
                )
                _write_json(root / "watchdog_state.json", {
                    "interval_seconds": args.interval_seconds,
                    "similarity_threshold": args.similarity_threshold,
                    "campaign_pid": campaign_pid,
                    "last_restart_at_unix": time.time(),
                })

            time.sleep(args.interval_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--attach-pid", type=int, default=None)
    parser.add_argument("--interval-seconds", type=int, default=DEFAULT_INTERVAL)
    parser.add_argument("--similarity-threshold", type=float, default=0.70)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--hours", type=float, default=10.0)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    if args.interval_seconds < 60:
        parser.error("--interval-seconds must be at least 60")
    if not 0.0 < args.similarity_threshold <= 1.0:
        parser.error("--similarity-threshold must be in (0, 1]")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
