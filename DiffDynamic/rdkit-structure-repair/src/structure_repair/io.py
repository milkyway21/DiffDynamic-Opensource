"""I/O helpers for molecules and batch files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

from rdkit import Chem

from .atom_mapping import clear_atom_maps_for_smiles


PathLike = Union[str, Path]


def mol_from_smiles(smiles: str, sanitize: bool = False) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        return None
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:  # noqa: BLE001
        pass
    return mol


def mol_to_canonical_smiles(mol: Chem.Mol, clear_maps: bool = True) -> Optional[str]:
    try:
        work = clear_atom_maps_for_smiles(mol) if clear_maps else Chem.Mol(mol)
        try:
            Chem.SanitizeMol(work)
        except Exception:  # noqa: BLE001
            # Fall back to unsanitized SMILES
            return Chem.MolToSmiles(work)
        return Chem.MolToSmiles(work)
    except Exception:  # noqa: BLE001
        return None


def read_sdf(path: PathLike, sanitize: bool = False) -> List[Tuple[str, Chem.Mol]]:
    path = Path(path)
    supplier = Chem.SDMolSupplier(str(path), sanitize=sanitize, removeHs=False)
    results: List[Tuple[str, Chem.Mol]] = []
    for i, mol in enumerate(supplier):
        if mol is None:
            continue
        mid = mol.GetProp("_Name") if mol.HasProp("_Name") and mol.GetProp("_Name") else f"mol_{i}"
        results.append((mid, mol))
    return results


def write_sdf(path: PathLike, molecules: List[Tuple[str, Chem.Mol]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    for mid, mol in molecules:
        if mol is None:
            continue
        m = Chem.Mol(mol)
        m.SetProp("_Name", str(mid))
        writer.write(m)
    writer.close()


def read_csv_smiles(
    path: PathLike,
    smiles_column: str = "smiles",
    id_column: str = "molecule_id",
    sanitize: bool = False,
) -> List[Tuple[str, Chem.Mol]]:
    path = Path(path)
    rows: List[Tuple[str, Chem.Mol]] = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            smi = row.get(smiles_column) or row.get("reconstructed_smiles")
            if not smi:
                continue
            mid = row.get(id_column) or row.get("id") or f"row_{i}"
            mol = mol_from_smiles(smi, sanitize=sanitize)
            if mol is None:
                continue
            rows.append((str(mid), mol))
    return rows


def iter_input_molecules(
    path: PathLike,
    smiles_column: str = "smiles",
    id_column: str = "molecule_id",
    sanitize: bool = False,
) -> List[Tuple[str, Chem.Mol]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".sdf":
        return read_sdf(path, sanitize=sanitize)
    if suffix in (".mol", ".rdkit"):
        mol = Chem.MolFromMolFile(str(path), sanitize=sanitize, removeHs=False)
        if mol is None:
            return []
        mid = mol.GetProp("_Name") if mol.HasProp("_Name") else path.stem
        return [(mid, mol)]
    if suffix == ".smi" or suffix == ".smiles":
        out = []
        with open(path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                smi = parts[0]
                mid = parts[1] if len(parts) > 1 else f"mol_{i}"
                mol = mol_from_smiles(smi, sanitize=sanitize)
                if mol is not None:
                    out.append((mid, mol))
        return out
    if suffix == ".csv":
        return read_csv_smiles(path, smiles_column=smiles_column, id_column=id_column, sanitize=sanitize)
    # try smiles string file
    return read_csv_smiles(path, smiles_column=smiles_column, id_column=id_column, sanitize=sanitize)
