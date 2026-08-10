#!/usr/bin/env python3
"""Copy a plot CSV and replace Sitemap values when a new score exists.

Rows without a usable new SiteMap score retain the value from the source CSV.
The source plot CSV may identify pockets by ``data_id``/``pocket_id`` or by
row order (the historical ``fig7/pocketeval1.csv`` format).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
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


def _usable_score(value: str | None) -> str | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return value


def merge(source_csv: Path, sitemap_csv: Path, output_csv: Path, column: str) -> None:
    source_header, source_rows = _read(source_csv)
    if "Sitemap" not in source_header:
        raise ValueError(f"source CSV lacks Sitemap column: {source_csv}")
    new_header, new_rows = _read(sitemap_csv)
    if "data_id" not in new_header:
        raise ValueError(f"new SiteMap CSV lacks data_id column: {sitemap_csv}")
    if column not in new_header:
        raise ValueError(f"new SiteMap CSV lacks {column} column: {sitemap_csv}")

    new_by_id = {
        _row_id(row, index): _usable_score(row.get(column))
        for index, row in enumerate(new_rows)
    }
    replaced: list[int] = []
    retained: list[int] = []
    for index, row in enumerate(source_rows):
        pocket_id = _row_id(row, index)
        score = new_by_id.get(pocket_id)
        if score is None:
            retained.append(pocket_id)
        else:
            row["Sitemap"] = score
            replaced.append(pocket_id)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=source_header)
        writer.writeheader()
        writer.writerows(source_rows)
    print(f"wrote {output_csv}")
    print(f"rows={len(source_rows)} replaced={len(replaced)} retained_old={len(retained)}")
    print(f"replaced_ids={replaced}")
    print(f"retained_old_ids={retained}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--new-sitemap-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--new-column", default="legacy_combined_score")
    args = parser.parse_args()
    merge(args.source_csv, args.new_sitemap_csv, args.output_csv, args.new_column)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
