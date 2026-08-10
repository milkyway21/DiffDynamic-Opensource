#!/usr/bin/env python3
"""Batch chem-only re-eval of jsdpt3010 with rdkit-structure-repair (NO Vina).

Usage:
  conda activate diffdynamic
  python scripts/batch_jsdpt3010_rdkit_repair_chem.py [--workers 8] [--skip-done]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
PTS = sorted(
    (REPO / "jsdpt3010").glob("result_*.pt"),
    key=lambda p: int(p.name.split("_")[1]),
)
PROTEIN_ROOT = REPO / "data" / "crossdocked_pocket10_test_only"
OUT_ROOT = REPO / "jsdpt3010_rdkit_repair_chem"
PY = sys.executable
EVAL = REPO / "evaluate_pt_with_correct_reconstruct.py"
REPAIR_CFG = "rdkit-structure-repair/configs/conservative.yaml"


def data_id_of(pt: Path) -> str:
    return pt.name.split("_")[1]


def is_done(pt: Path) -> bool:
    d = OUT_ROOT / f"data_{data_id_of(pt)}"
    if not d.is_dir():
        return False
    return any(d.glob("eval_*/eval_results_*_final_*.pt"))


def run_one(pt: Path, max_samples: Optional[int]) -> Tuple[str, int, float, str]:
    out_dir = OUT_ROOT / f"data_{data_id_of(pt)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY,
        str(EVAL),
        str(pt),
        "--protein_root",
        str(PROTEIN_ROOT),
        "--output_dir",
        str(out_dir.resolve()),
        "--vina-modes",
        "none",
        "--rdkit-structure-repair",
        "--rdkit-structure-repair-config",
        REPAIR_CFG,
        "--no-distribution-plots",
        "--atom_mode",
        "add_aromatic",
    ]
    if max_samples is not None:
        cmd.extend(["--max_samples", str(max_samples)])
    t0 = time.time()
    log_path = out_dir / "run.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with open(str(log_path), "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO),
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
        )
    dt = time.time() - t0
    return pt.name, proc.returncode, dt, str(log_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None, help="limit mols per pt (debug)")
    parser.add_argument("--limit-pts", type=int, default=None, help="only first N pts")
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="skip pts that already have eval_results_*_final_*.pt",
    )
    args = parser.parse_args()

    if not PROTEIN_ROOT.is_dir():
        print(f"ERROR: protein_root missing: {PROTEIN_ROOT}", file=sys.stderr)
        return 1
    if not PTS:
        print("ERROR: no result_*.pt under jsdpt3010", file=sys.stderr)
        return 1

    pts = PTS[: args.limit_pts] if args.limit_pts else list(PTS)
    if args.skip_done:
        before = len(pts)
        pts = [p for p in pts if not is_done(p)]
        print(f"skip-done: {before - len(pts)} already finished, {len(pts)} remaining")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    print(f"pts={len(pts)} workers={args.workers} out={OUT_ROOT}")
    print("Vina: DISABLED (--vina-modes none) — NO DOCKING")
    print("RDKit structure repair: ON")

    if not pts:
        print("\nDone: ok=0 fail=0 (nothing to run)")
        return 0

    ok = fail = 0
    t_all = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, pt, args.max_samples): pt for pt in pts}
        for fut in as_completed(futs):
            name, code, dt, log = fut.result()
            if code == 0:
                ok += 1
                print(f"[OK] {name} ({dt:.1f}s) log={log}", flush=True)
            else:
                fail += 1
                print(f"[FAIL] {name} rc={code} ({dt:.1f}s) log={log}", flush=True)

    print(f"\nDone: ok={ok} fail={fail} elapsed={time.time()-t_all:.1f}s", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
