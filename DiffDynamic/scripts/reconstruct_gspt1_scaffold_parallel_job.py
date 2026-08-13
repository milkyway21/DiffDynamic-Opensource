#!/usr/bin/env python3
"""Run the existing full PT evaluator over independent scaffold chunks.

The evaluator intentionally remains unchanged per molecule: RDKit metrics,
fragment handling, SDF writing, and ``--vina-modes none`` are all preserved.
This wrapper only partitions a completed PT file so several evaluator
processes can run concurrently on CPU cores.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = REPO_ROOT / "evaluate_pt_with_correct_reconstruct.py"
_TRUE_VALUES = {"1", "true", "yes", "on"}
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def chunk_ranges(num_items: int, workers: int) -> list[tuple[int, int]]:
    """Return balanced, non-empty half-open ranges."""
    if num_items <= 0:
        return []
    if workers <= 0:
        raise ValueError("workers must be positive")
    n_chunks = min(num_items, workers)
    chunk_size = math.ceil(num_items / n_chunks)
    return [
        (start, min(start + chunk_size, num_items))
        for start in range(0, num_items, chunk_size)
    ]


def _slice_if_per_sample(value: Any, start: int, end: int, num_items: int) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == num_items:
        return value[start:end]
    return value


def make_chunk_payload(
    data: dict[str, Any], start: int, end: int, num_items: int
) -> dict[str, Any]:
    """Keep only evaluator inputs and chunk-aligned metadata.

    Trajectory histories are not consumed by the evaluator and can be tens of
    megabytes per PT, so they are deliberately omitted from chunk files.
    """
    payload = dict(data)
    payload["pred_ligand_pos"] = data["pred_ligand_pos"][start:end]
    payload["pred_ligand_v"] = data["pred_ligand_v"][start:end]
    for key in (
        "pred_ligand_pos_traj",
        "pred_ligand_v_traj",
        "pred_ligand_log_v_traj",
        "time",
    ):
        payload.pop(key, None)

    meta = copy.deepcopy(data.get("meta") or {})
    for key in ("records", "refined_candidates"):
        if key in meta:
            meta[key] = _slice_if_per_sample(
                meta[key], start, end, num_items
            )
    payload["meta"] = meta

    extra_info = copy.deepcopy(data.get("extra_info") or {})
    extra_info["parallel_chunk"] = {"start": start, "end": end}
    payload["extra_info"] = extra_info
    return payload


def _run_chunk(
    *,
    chunk_id: int,
    chunk_pt: Path,
    chunk_output: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    chunk_output.mkdir(parents=True, exist_ok=True)
    log_path = chunk_output / "evaluator.log"
    env = os.environ.copy()
    env.update(
        {
            # Each evaluator keeps its own per-molecule isolation.  Limit
            # native math libraries so one chunk consumes one CPU core.
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            # A malformed generated geometry must not hold an entire
            # scaffold lane for the evaluator's historical 3-hour default.
            # This helper is scaffold-only; de novo evaluation is untouched.
            "EVAL_SINGLE_MOL_TIMEOUT": os.environ.get(
                "SCAFFOLD_EVAL_SINGLE_MOL_TIMEOUT", "300"
            ),
            "EVAL_MOLECULE_ID_SUFFIX": f"chunk{chunk_id:04d}",
            "DIFFDYNAMIC_SKIP_EVAL_RECORDS": "1",
        }
    )
    command = [
        sys.executable,
        str(EVALUATOR),
        str(chunk_pt),
        "--vina-modes",
        "none",
        "--receptor_pdb",
        args.receptor_pdb,
        "--protein_root",
        args.protein_root,
        "--reference_ligand",
        args.reference_ligand,
        "--output_dir",
        str(chunk_output),
        "--enable_isolation",
        "--no-distribution-plots",
        "--save_intermediate_interval",
        "0",
    ]
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "chunk_id": chunk_id,
        "returncode": result.returncode,
        "elapsed_seconds": time.time() - started,
        "chunk_pt": str(chunk_pt),
        "chunk_output": str(chunk_output),
        "log": str(log_path),
    }


def run(args: argparse.Namespace) -> int:
    pt_path = Path(args.pt).resolve()
    output_dir = Path(args.output_dir).resolve()
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError("scaffold parallel reconstruction requires a dict PT")
    positions = data.get("pred_ligand_pos")
    atom_types = data.get("pred_ligand_v")
    if positions is None or atom_types is None:
        raise ValueError("PT must contain pred_ligand_pos and pred_ligand_v")
    if len(positions) != len(atom_types):
        raise ValueError("position/type sample counts differ")

    num_items = len(positions)
    ranges = chunk_ranges(num_items, args.workers)
    run_root = output_dir / ".parallel_chunks" / (
        f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    for chunk_id, (start, end) in enumerate(ranges):
        chunk_root = run_root / f"chunk_{chunk_id:04d}"
        chunk_root.mkdir(parents=True, exist_ok=True)
        chunk_pt = chunk_root / "input.pt"
        chunk_data = make_chunk_payload(data, start, end, num_items)
        torch.save(chunk_data, chunk_pt)
        jobs.append((chunk_id, chunk_pt, chunk_root / "evaluation"))

    manifest = {
        "source_pt": str(pt_path),
        "sample_count": num_items,
        "requested_workers": args.workers,
        "chunk_count": len(jobs),
        "vina_modes": "none",
        "run_root": str(run_root),
    }
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    results = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {
            executor.submit(
                _run_chunk,
                chunk_id=chunk_id,
                chunk_pt=chunk_pt,
                chunk_output=chunk_output,
                args=args,
            ): chunk_id
            for chunk_id, chunk_pt, chunk_output in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=True), flush=True)

    results.sort(key=lambda item: item["chunk_id"])
    manifest["chunks"] = results
    manifest["failed_chunks"] = [
        item for item in results if item["returncode"] != 0
    ]
    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    sdf_count = len(list(output_dir.rglob("*.sdf")))
    print(
        json.dumps(
            {
                "sample_count": num_items,
                "chunk_count": len(jobs),
                "failed_chunks": len(manifest["failed_chunks"]),
                "sdf_count": sdf_count,
                "manifest": str(manifest_path),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    return 0 if not manifest["failed_chunks"] and sdf_count > 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--receptor_pdb", required=True)
    parser.add_argument("--protein_root", required=True)
    parser.add_argument("--reference_ligand", required=True)
    parser.add_argument("--workers", type=int, default=30)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
