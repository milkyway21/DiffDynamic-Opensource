#!/usr/bin/env python3
"""Simple repair benchmark over a corruption CSV (M3 scaffold)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from structure_repair import load_config, repair_molecule
from structure_repair.corruption import mol_from_smiles
from structure_repair.io import mol_from_smiles as io_mol_from_smiles
from structure_repair.io import mol_to_canonical_smiles
from rdkit import Chem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="corruption_dataset.csv")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="benchmark_summary.csv")
    args = parser.parse_args()
    config = load_config(args.config)

    rows_out = []
    with open(args.input, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            # Prefer reconstructing corrupted mol from clean + rule externally;
            # here we parse corrupted_smiles with sanitize=False when possible.
            smi = row.get("corrupted_smiles") or ""
            mol = Chem.MolFromSmiles(smi, sanitize=False) if smi else None
            if mol is None:
                continue
            try:
                mol.UpdatePropertyCache(strict=False)
            except Exception:  # noqa: BLE001
                pass
            result = repair_molecule(mol, config=config, molecule_id=row["molecule_id"])
            repaired = (
                mol_to_canonical_smiles(result.repaired_mol)
                if result.repaired_mol is not None
                else ""
            )
            expected = row.get("expected_repaired_smiles") or ""
            rows_out.append(
                {
                    "molecule_id": row["molecule_id"],
                    "status": result.status,
                    "expected": expected,
                    "repaired": repaired,
                    "match": int(repaired == expected) if repaired and expected else 0,
                }
            )

    out = Path(args.output)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["molecule_id", "status", "expected", "repaired", "match"]
        )
        writer.writeheader()
        writer.writerows(rows_out)
    n = len(rows_out)
    hits = sum(r["match"] for r in rows_out)
    print(f"Benchmark: {hits}/{n} exact recoveries; wrote {out}")


if __name__ == "__main__":
    main()
