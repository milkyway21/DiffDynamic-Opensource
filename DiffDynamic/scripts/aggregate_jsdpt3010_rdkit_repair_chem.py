#!/usr/bin/env python3
"""Aggregate chem-only eval reports under jsdpt3010_rdkit_repair_chem/."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "jsdpt3010_rdkit_repair_chem"
OUT = ROOT / "aggregate_chem_summary.csv"


def parse_report(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    d = {"report": str(path)}
    m = re.search(r"总样本数:\s*(\d+)", text)
    d["n_total"] = int(m.group(1)) if m else None
    m = re.search(r"重建成功:\s*(\d+)\s*\(([\d.]+)%\)", text)
    if m:
        d["n_recon"] = int(m.group(1))
        d["recon_pct"] = float(m.group(2))
    m = re.search(r"完整分子:\s*(\d+)\s*\(([\d.]+)%\)", text)
    if m:
        d["n_complete"] = int(m.group(1))
        d["complete_pct"] = float(m.group(2))
    for key, pat in [
        ("qed_mean", r"QED.*?Mean:\s*([\d.]+)"),
        ("sa_mean", r"SA.*?Mean:\s*([\d.]+)"),
        ("lipinski_mean", r"Lipinski.*?Mean:\s*([\d.]+)"),
        ("uniqueness", r"唯一性 \(Uniqueness\):\s*([\d.]+)"),
        ("internal_sim", r"内部相似度 \(Internal Similarity\):\s*([\d.]+)"),
    ]:
        m = re.search(pat, text, re.S | re.I)
        d[key] = float(m.group(1)) if m else None
    # data id from path
    m = re.search(r"data_(\d+)", str(path))
    d["data_id"] = int(m.group(1)) if m else None
    return d


def main() -> int:
    reports = sorted(ROOT.glob("data_*/eval_*/eval_report_*.txt"))
    if not reports:
        print(f"No reports under {ROOT}", file=sys.stderr)
        return 1
    rows = [parse_report(p) for p in reports]
    fields = [
        "data_id",
        "n_total",
        "n_recon",
        "recon_pct",
        "n_complete",
        "complete_pct",
        "qed_mean",
        "sa_mean",
        "lipinski_mean",
        "uniqueness",
        "internal_sim",
        "report",
    ]
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: x.get("data_id") or -1):
            w.writerow(r)
    n = len(rows)
    recon = sum(r.get("recon_pct") or 0 for r in rows) / n
    print(f"Wrote {OUT} ({n} pockets); mean recon%={recon:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
