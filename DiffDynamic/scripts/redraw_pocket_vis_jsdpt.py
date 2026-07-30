#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Redraw pocket-quality PNGs only (reads cached Vina; no re-dock)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluate_pocket_quality import evaluate_pocket_quality  # noqa: E402


def _find_pt(pt_dirs, data_id: int):
    for d in pt_dirs:
        if not d or not Path(d).exists():
            continue
        ms = sorted(Path(d).glob(f"result_{data_id}_*.pt"))
        if ms:
            return ms[-1]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pt-dir",
        type=Path,
        action="append",
        default=None,
        help="Search dir for result_{id}_*.pt (can repeat)",
    )
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument(
        "--out-vis",
        type=Path,
        default=REPO / "pocket_quality_vis" / "jsdpt3010_run8_vis_redesign",
    )
    args = ap.parse_args()
    pt_dirs = args.pt_dir or [
        REPO / "jsdpt3010",
        Path("/data/ye/pt/jsdpt3010"),
    ]
    out_root = args.out_vis.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    for i in range(int(args.start), int(args.end) + 1):
        pt = _find_pt(pt_dirs, i)
        if not pt:
            print(f"[{i}] missing pt", flush=True)
            continue
        vis = out_root / str(i)
        print(f"[{i}] redraw {pt.name} -> {vis}", flush=True)
        evaluate_pocket_quality(
            pt_path=str(pt),
            data_id=i,
            visualize=True,
            vis_dir=str(vis),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
