#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-run pocket quality eval on jsdpt3010 (F bio-chem 10Å + H MC 10Å) with viz."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluate_pocket_quality import append_evaluation_record, evaluate_pocket_quality  # noqa: E402


def _find_pt(pt_dir: Path, data_id: int) -> Optional[Path]:
    ms = sorted(pt_dir.glob(f"result_{data_id}_*.pt"))
    return ms[-1] if ms else None


def _one(args: Tuple) -> Dict[str, Any]:
    data_id, pt_path, out_vis, protein_root, do_viz = args
    out: Dict[str, Any] = {
        "pocket_id": int(data_id),
        "ok": False,
        "overall_score": float("nan"),
        "score_f": float("nan"),
        "score_g": float("nan"),
        "score_h": float("nan"),
        "s_pocket": float("nan"),
        "message": "",
        "pt_path": pt_path,
        "_result": None,
    }
    if not pt_path:
        out["message"] = "missing_pt"
        return out
    vis_dir = Path(out_vis) / str(data_id)
    try:
        result = evaluate_pocket_quality(
            pt_path=pt_path,
            data_id=data_id,
            protein_root=protein_root,
            visualize=bool(do_viz),
            vis_dir=str(vis_dir),
        )
    except Exception as exc:
        out["message"] = f"exc:{exc}"
        return out
    if result.get("error"):
        out["message"] = str(result["error"])
        return out
    out["ok"] = True
    out["overall_score"] = float(result.get("overall_score") or float("nan"))
    sp = result.get("s_pocket")
    out["s_pocket"] = float(sp) if sp is not None else float("nan")
    for letter, key in (("f", "score_f"), ("g", "score_g"), ("h", "score_h")):
        dim = result.get(f"idea_{letter}") or {}
        out[key] = float(dim.get("score")) if dim.get("success") else float("nan")
    out["message"] = "ok"
    out["_result"] = result
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-dir", type=Path, default=REPO / "jsdpt3010")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=99)
    ap.add_argument(
        "--out-vis",
        type=Path,
        default=REPO / "pocket_quality_vis" / "jsdpt3010_run7_fh10a_vis",
    )
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--protein-root", type=Path, default=None)
    ap.add_argument("--visualize", dest="visualize", action="store_true", default=True)
    ap.add_argument("--no-visualize", dest="visualize", action="store_false")
    args = ap.parse_args()

    pt_dir = args.pt_dir.resolve()
    out_vis = args.out_vis.resolve()
    out_vis.mkdir(parents=True, exist_ok=True)
    record_path = out_vis / "evaluation_records.csv"
    if record_path.exists():
        record_path.unlink()

    tasks = []
    for i in range(int(args.start), int(args.end) + 1):
        pt = _find_pt(pt_dir, i)
        tasks.append(
            (
                i,
                str(pt) if pt else "",
                str(out_vis),
                str(args.protein_root.resolve()) if args.protein_root else None,
                bool(args.visualize),
            )
        )

    results: List[Dict[str, Any]] = []
    n_workers = max(1, int(args.workers))
    print(
        f"Re-eval {len(tasks)} pockets workers={n_workers} viz={args.visualize} -> {out_vis}",
        flush=True,
    )

    if n_workers == 1:
        for t in tasks:
            r = _one(t)
            results.append(r)
            print(
                f"  [{r['pocket_id']}] ok={r['ok']} overall={r['overall_score']} "
                f"F={r['score_f']} H={r['score_h']} {r['message']}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_one, t): t[0] for t in tasks}
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                print(
                    f"  [{r['pocket_id']}] ok={r['ok']} overall={r['overall_score']} "
                    f"F={r['score_f']} H={r['score_h']} {r['message']}",
                    flush=True,
                )

    results = sorted(results, key=lambda x: int(x["pocket_id"]))
    for r in results:
        res = r.pop("_result", None)
        if r.get("ok") and res is not None:
            append_evaluation_record(res, record_path)

    audit = pd.DataFrame(results)
    audit_path = REPO / "fig7" / "pocket_eval_run7_fh10a_audit.csv"
    audit.to_csv(audit_path, index=False)
    print(f"Audit -> {audit_path}")
    print(f"ok {int(audit.ok.sum())}/{len(audit)}")
    print(f"Records -> {record_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
