#!/usr/bin/env python3
"""Batch repair wrapper around the CLI repair command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from structure_repair.cli import main as cli_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch structure repair")
    parser.add_argument("input")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smiles-column", default="reconstructed_smiles")
    parser.add_argument("--id-column", default="molecule_id")
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    argv = [
        "repair",
        args.input,
        "--output",
        args.output,
        "--smiles-column",
        args.smiles_column,
        "--id-column",
        args.id_column,
        "--num-workers",
        str(args.num_workers),
    ]
    if args.config:
        argv.extend(["--config", args.config])
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
