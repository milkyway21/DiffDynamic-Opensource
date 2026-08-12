#!/usr/bin/env python3
"""Watch and resume the five-lane strict GSPT1 campaign."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(
    "/data/zhang/Ye/DiffDynamic_outputs/hsvpol/"
    "molglue_ikzf2_gspt1/diffdynamic/gspt1_scaffold_strict_gaussian_v1"
)
DEFAULT_START_SEED = 20290000
DEFAULT_INTERVAL_SECONDS = 1800


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _stop_group(pid: int, logger) -> None:
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
        logger(f"sent SIGTERM to strict campaign pgid={pgid}")
    except (ProcessLookupError, PermissionError, OSError) as exc:
        logger(f"failed to stop strict campaign pid={pid}: {exc}")


def _launch(
    root: Path,
    start_seed: int,
    hours: float,
    samples: int,
    logger,
) -> int:
    root.mkdir(parents=True, exist_ok=True)
    log_handle = (root / "campaign_launcher.log").open("a", encoding="utf-8")
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":"),
        "PYTHONUNBUFFERED": "1",
        "SCAFFOLD_RECONSTRUCT_WORKERS": "18",
    })
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/gspt1_strict_gaussian_campaign.py"),
        "--run",
        "--resume",
        "--hours",
        f"{max(float(hours), 1.0 / 3600.0):.6f}",
        "--samples",
        str(int(samples)),
        "--start-seed",
        str(int(start_seed)),
        "--root",
        str(root),
        "--gpus",
        "1,2,3,4,5",
    ]
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
    (root / "campaign.pid").write_text(
        f"{process.pid}\n", encoding="utf-8"
    )
    logger(
        f"launched strict campaign pid={process.pid} "
        f"hours={hours:.3f} samples={samples}"
    )
    return int(process.pid)


def _summary(root: Path) -> dict:
    latest = _read_json(root / "latest_summary.json")
    total = latest.get("total_similarity") or {}
    lane_rows = latest.get("lanes") or {}
    best_lane = max(
        lane_rows.items(),
        key=lambda item: float(item[1].get("best_side_similarity", 0.0)),
        default=(None, {}),
    )
    return {
        "round": latest.get("round"),
        "next_job_id": (_read_json(root / "state.json").get("next_job_id")),
        "best_lane": best_lane[0],
        "best_side_similarity": float(
            best_lane[1].get("best_side_similarity", 0.0)
        ),
        "best_total_similarity": float(total.get("best_similarity", 0.0)),
        "exact_reachable_count": int(
            latest.get("exact_reachable_count", 0)
        ),
        "total_exact_count": int(total.get("exact_count", 0)),
    }


def run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "watchdog.lock").open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("strict watchdog already running", file=sys.stderr)
            return 2

        log_path = root / "watchdog.log"

        def logger(message: str) -> None:
            line = f"[{time.strftime('%Y-%m-%d %H:%M:%S %z')}] {message}"
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

        started = time.time()
        deadline = started + float(args.hours) * 3600.0
        pid = None
        pid_path = root / "campaign.pid"
        if pid_path.exists():
            try:
                candidate = int(pid_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                candidate = None
            if _alive(candidate):
                pid = candidate

        _write_json(root / "watchdog_state.json", {
            "started_at_unix": started,
            "deadline_unix": deadline,
            "interval_seconds": int(args.interval_seconds),
            "similarity_threshold": float(args.similarity_threshold),
            "gpu_mapping": {"no_f": [1, 2], "f_main": [3, 4], "f_large": [5]},
            "gpu0_unused": True,
        })

        while time.time() < deadline:
            remaining_hours = max((deadline - time.time()) / 3600.0, 0.0)
            if pid is None or not _alive(pid):
                if pid is not None:
                    logger(f"campaign pid={pid} is not alive; restarting")
                pid = _launch(
                    root,
                    args.start_seed,
                    remaining_hours,
                    args.samples,
                    logger,
                )

            summary = _summary(root)
            logger(
                f"health round={summary['round']} "
                f"next_job_id={summary['next_job_id']} "
                f"campaign_alive={_alive(pid)} "
                f"best_lane={summary['best_lane']} "
                f"best_side={summary['best_side_similarity']:.4f} "
                f"best_total={summary['best_total_similarity']:.4f} "
                f"exact={summary['exact_reachable_count'] + summary['total_exact_count']}"
            )
            exact = (
                summary["exact_reachable_count"] > 0
                or summary["total_exact_count"] > 0
                or (root / "EXACT_MATCH_FOUND").exists()
            )
            similar = summary["best_side_similarity"] >= float(
                args.similarity_threshold
            )
            if exact or similar:
                reason = "exact" if exact else "side_similarity"
                _write_json(root / "SIMILARITY_TARGET_FOUND.json", {
                    "reason": reason,
                    "summary": summary,
                    "threshold": float(args.similarity_threshold),
                    "found_at_unix": time.time(),
                })
                logger(f"similarity target reached ({reason}); stopping")
                if pid and _alive(pid):
                    _stop_group(pid, logger)
                return 0

            sleep_for = min(
                float(args.interval_seconds),
                max(deadline - time.time(), 0.0),
            )
            if sleep_for > 0:
                time.sleep(sleep_for)

        if pid and _alive(pid):
            _stop_group(pid, logger)
        logger("strict watchdog deadline reached")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--hours", type=float, default=10.0)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--start-seed", type=int, default=DEFAULT_START_SEED)
    parser.add_argument(
        "--interval-seconds", type=int, default=DEFAULT_INTERVAL_SECONDS
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.70)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.hours <= 0:
        parser.error("--hours must be positive")
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
