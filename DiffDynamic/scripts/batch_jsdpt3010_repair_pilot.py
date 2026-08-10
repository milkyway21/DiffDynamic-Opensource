#!/usr/bin/env python3
"""Pilot experiment: chem-only re-eval of jsdpt3010 subset.

Arms:
  control    — no RDKit structure repair
  aggressive — aggressive_optimize.yaml repair (legality only, formula preserved)
  medchem    — aggressive medchem optimization (T0-T5); *changes the molecular
               formula*, so this arm is a separate ablation, not a repair
               fidelity result

Usage:
  python scripts/batch_jsdpt3010_repair_pilot.py --arm control --limit-pts 10 --workers 4
  python scripts/batch_jsdpt3010_repair_pilot.py --arm aggressive --limit-pts 10 --workers 4
  python scripts/batch_jsdpt3010_repair_pilot.py --arm medchem --limit-pts 10 --workers 4
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
PY = sys.executable
EVAL = REPO / "evaluate_pt_with_correct_reconstruct.py"
AGGRESSIVE_CFG = "rdkit-structure-repair/configs/aggressive_optimize.yaml"
MEDCHEM_CFG = "rdkit-structure-repair/configs/medchem_optimize.yaml"
ARMS = ("control", "aggressive", "medchem")


def data_id_of(pt: Path) -> str:
    return pt.name.split("_")[1]


def run_one(pt: Path, out_root: Path, arm: str, max_samples: Optional[int]) -> Tuple[str, int, float, str]:
    out_dir = out_root / f"data_{data_id_of(pt)}"
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
        "--no-distribution-plots",
        "--atom_mode",
        "add_aromatic",
    ]
    if arm == "aggressive":
        cmd.extend(
            [
                "--rdkit-structure-repair",
                "--rdkit-structure-repair-config",
                AGGRESSIVE_CFG,
            ]
        )
    elif arm == "medchem":
        cmd.extend(
            [
                "--medchem-optimize",
                "--medchem-optimize-config",
                MEDCHEM_CFG,
                # chem-only pilot: no Vina, so evaluating the pre-optimization
                # molecule too is cheap and gives a paired comparison.
                "--keep-preopt-sdf",
                "evaluated",
            ]
        )
    if max_samples is not None:
        cmd.extend(["--max_samples", str(max_samples)])
    t0 = time.time()
    log_path = out_dir / "run.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with open(str(log_path), "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT, env=env)
    return pt.name, proc.returncode, time.time() - t0, str(log_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=list(ARMS), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-pts", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--data-ids", type=str, default=None, help="comma list e.g. 0,1,2")
    args = parser.parse_args()

    out_root = REPO / f"jsdpt3010_pilot_{args.arm}"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(exist_ok=True)

    if args.data_ids:
        want = {int(x) for x in args.data_ids.split(",")}
        pts = [p for p in PTS if int(data_id_of(p)) in want]
    else:
        pts = PTS[: args.limit_pts]

    print(f"arm={args.arm} pts={len(pts)} workers={args.workers} out={out_root}")
    print("Vina: DISABLED")
    if args.arm == "aggressive":
        print("Repair: ON aggressive_optimize (formula preserved)")
    elif args.arm == "medchem":
        print("Repair: OFF / Medchem optimize: ON medchem_optimize.yaml")
        print("        分子式会改变，本臂是独立 ablation，不能当作 reconstruct 保真度结论")
    else:
        print("Repair: OFF")

    ok = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, pt, out_root, args.arm, args.max_samples): pt for pt in pts}
        for fut in as_completed(futs):
            name, code, dt, log = fut.result()
            if code == 0:
                ok += 1
                print(f"[OK] {name} ({dt:.1f}s)", flush=True)
            else:
                fail += 1
                print(f"[FAIL] {name} rc={code} ({dt:.1f}s) log={log}", flush=True)
    print(f"\nDone: ok={ok} fail={fail} elapsed={time.time()-t0:.1f}s", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
