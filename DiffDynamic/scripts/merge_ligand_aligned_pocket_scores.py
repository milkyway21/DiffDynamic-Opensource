#!/usr/bin/env python3
"""Merge ligand-aligned P2Rank/FPocket scores into a new plot CSV.

Each tool independently selects the predicted pocket whose center is closest
to the reference-ligand atoms (the matching step performed by the audit
scripts). A score is plotted only when ``match_dist <= 8 A``; shifted pockets
are written as empty cells. Scores are clipped to ``[0, 1]`` by default, or
kept at their raw values with ``--no-clip``. ``--p2rank-range LOW HIGH`` and
``--fpocket-range LOW HIGH`` apply explicit linear mappings to the respective
raw scores. P2Rank's native probability range is ``0..1``, so its recommended
fixed-range normalization is ``--p2rank-range 0 1`` (an identity mapping).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _row_id(row: dict[str, str], index: int) -> int:
    for key in ("data_id", "pocket_id", "id"):
        value = row.get(key)
        if value not in (None, ""):
            return int(float(value))
    return index


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "ok"}


def _load_aligned_audit(
    path: Path,
    score_column: str,
    threshold: float,
    clip_scores: bool,
    score_range: tuple[float, float] | None = None,
) -> tuple[dict[int, float], list[int], dict[int, float]]:
    header, rows = _read_csv(path)
    required = {"pocket_id", "match_dist", score_column}
    missing = required - set(header)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")

    valid: dict[int, float] = {}
    excluded: list[int] = []
    distances: dict[int, float] = {}
    for index, row in enumerate(rows):
        pocket_id = _row_id(row, index)
        distance = _float(row.get("match_dist"))
        score = _float(row.get(score_column))
        if distance is not None:
            distances[pocket_id] = distance
        usable = (
            _truthy(row.get("ok", "true"))
            and distance is not None
            and distance <= threshold
            and score is not None
        )
        if usable:
            if score_range is not None:
                lo, hi = score_range
                score = (score - lo) / (hi - lo)
                valid[pocket_id] = max(0.0, min(1.0, score))
            elif clip_scores:
                valid[pocket_id] = max(0.0, min(1.0, score))
            else:
                valid[pocket_id] = score
        else:
            excluded.append(pocket_id)
    return valid, excluded, distances


def merge(
    source_csv: Path,
    p2rank_audit: Path,
    fpocket_audit: Path,
    output_csv: Path,
    report_json: Path,
    threshold: float,
    clip_scores: bool,
    p2rank_range: tuple[float, float] | None = None,
    fpocket_range: tuple[float, float] | None = None,
) -> None:
    source_header, source_rows = _read_csv(source_csv)
    for column in ("p2rank", "fpocket"):
        if column not in source_header:
            raise ValueError(f"source CSV lacks {column}: {source_csv}")

    p2rank, p2_excluded, p2_distances = _load_aligned_audit(
        p2rank_audit, "p2rank", threshold, clip_scores, p2rank_range
    )
    fpocket, fp_excluded, fp_distances = _load_aligned_audit(
        fpocket_audit, "score", threshold, clip_scores, fpocket_range
    )

    for index, row in enumerate(source_rows):
        pocket_id = _row_id(row, index)
        row["p2rank"] = "" if pocket_id not in p2rank else f"{p2rank[pocket_id]:.6g}"
        row["fpocket"] = "" if pocket_id not in fpocket else f"{fpocket[pocket_id]:.6g}"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=source_header)
        writer.writeheader()
        writer.writerows(source_rows)

    report = {
        "source_csv": str(source_csv.resolve()),
        "p2rank_audit": str(p2rank_audit.resolve()),
        "fpocket_audit": str(fpocket_audit.resolve()),
        "output_csv": str(output_csv.resolve()),
        "match_dist_max_A": threshold,
        "clip_scores": clip_scores,
        "p2rank_linear_range": list(p2rank_range) if p2rank_range else None,
        "fpocket_linear_range": list(fpocket_range) if fpocket_range else None,
        "p2rank_valid_n": len(p2rank),
        "p2rank_excluded_ids": sorted(set(p2_excluded)),
        "fpocket_valid_n": len(fpocket),
        "fpocket_excluded_ids": sorted(set(fp_excluded)),
        "p2rank_match_dist_A": p2_distances,
        "fpocket_match_dist_A": fp_distances,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_csv}")
    print(
        f"match_dist <= {threshold:g} A: "
        f"p2rank={len(p2rank)} (excluded {len(set(p2_excluded))}), "
        f"fpocket={len(fpocket)} (excluded {len(set(fp_excluded))})"
    )
    print(f"report {report_json}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--p2rank-audit", type=Path, required=True)
    parser.add_argument("--fpocket-audit", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--match-dist-max", type=float, default=8.0)
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Keep raw audit scores instead of clipping to [0, 1]",
    )
    parser.add_argument(
        "--p2rank-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Linearly map raw P2Rank probability LOW..HIGH to 0..1; use 0 1 for native probabilities",
    )
    parser.add_argument(
        "--fpocket-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Linearly map raw FPocket score LOW..HIGH to 0..1",
    )
    args = parser.parse_args()
    if args.match_dist_max <= 0:
        parser.error("--match-dist-max must be positive")
    p2rank_range = None
    if args.p2rank_range is not None:
        low, high = args.p2rank_range
        if not low < high:
            parser.error("--p2rank-range requires LOW < HIGH")
        p2rank_range = (low, high)
    fpocket_range = None
    if args.fpocket_range is not None:
        low, high = args.fpocket_range
        if not low < high:
            parser.error("--fpocket-range requires LOW < HIGH")
        fpocket_range = (low, high)
    merge(
        args.source_csv,
        args.p2rank_audit,
        args.fpocket_audit,
        args.output_csv,
        args.report_json,
        args.match_dist_max,
        not args.no_clip,
        p2rank_range,
        fpocket_range,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
