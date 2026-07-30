#!/usr/bin/env python3
"""Build a small corruption dataset from clean SMILES for regression/benchmarks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rdkit import Chem

from structure_repair.atom_mapping import assign_atom_maps
from structure_repair.corruption import (
    corrupt_aromatic_to_all_single,
    corrupt_c6_to_consecutive_doubles,
    corrupt_nitro_neutral_double,
    mol_from_smiles,
)
from structure_repair.io import mol_to_canonical_smiles


DEFAULT_SEEDS = [
    ("benzene", "c1ccccc1"),
    ("nitrobenzene", "c1ccc(cc1)[N+](=O)[O-]"),
    ("pyridine", "c1ccncc1"),
    ("pyrrole", "c1cc[nH]c1"),
    ("naphthalene", "c1ccc2ccccc2c1"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="tests/fixtures/corruption_dataset.csv")
    args = parser.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, smi in DEFAULT_SEEDS:
        clean = assign_atom_maps(mol_from_smiles(smi))
        clean_smi = mol_to_canonical_smiles(clean)
        ops = []
        if name in {"benzene", "nitrobenzene", "pyridine", "naphthalene"}:
            ops.append(("aromatic_to_all_single", corrupt_aromatic_to_all_single))
        if name == "benzene":
            ops.append(("c6_consecutive_doubles", corrupt_c6_to_consecutive_doubles))
        if name == "nitrobenzene":
            ops.append(("nitro_neutral_double", corrupt_nitro_neutral_double))
        for rule, fn in ops:
            try:
                corrupted = fn(clean)
            except Exception as exc:  # noqa: BLE001
                print(f"skip {name}/{rule}: {exc}")
                continue
            rows.append(
                {
                    "molecule_id": f"{name}__{rule}",
                    "clean_smiles": clean_smi,
                    "corrupted_smiles": Chem.MolToSmiles(corrupted) if corrupted else "",
                    "corruption_rule": rule,
                    "expected_repaired_smiles": clean_smi,
                    "changed_atom_maps": "",
                    "changed_bond_maps": "",
                }
            )

    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "molecule_id",
                "clean_smiles",
                "corrupted_smiles",
                "corruption_rule",
                "expected_repaired_smiles",
                "changed_atom_maps",
                "changed_bond_maps",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
