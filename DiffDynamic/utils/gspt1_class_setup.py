"""Trusted class definitions and reproducible inputs for GSPT1 sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem

from utils.gspt1_scaffold_prior import (
    build_class_scaffold_profile,
    iter_sdf_molecules,
    write_scaffold_profile,
)


BASE_SCAFFOLD_SMARTS = "O=C1CCC(N2Cc3ccccc3C2=O)C(=O)N1"
F_SCAFFOLD_SMARTS = "O=C1CCC(N2Cc3c(F)cccc3C2=O)C(=O)N1"
EXCLUDED_REFERENCE_INDICES = (6, 12)


@dataclass(frozen=True)
class Gspt1ClassSpec:
    name: str
    reference_indices: tuple[int, ...]
    scaffold_smarts: str
    ligand_kind: str
    jitter_mode: str
    allowed_n_extra: tuple[int, ...]


CLASS_SPECS = (
    Gspt1ClassSpec(
        name="no_f",
        reference_indices=(0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11),
        scaffold_smarts=BASE_SCAFFOLD_SMARTS,
        ligand_kind="native",
        jitter_mode="pocket_aware_template",
        allowed_n_extra=(13, 14),
    ),
    Gspt1ClassSpec(
        name="f_main",
        reference_indices=(14, 15, 16),
        scaffold_smarts=F_SCAFFOLD_SMARTS,
        ligand_kind="f_core",
        jitter_mode="pocket_aware_template",
        allowed_n_extra=(14, 15, 16),
    ),
    Gspt1ClassSpec(
        name="f_large",
        reference_indices=(13,),
        scaffold_smarts=F_SCAFFOLD_SMARTS,
        ligand_kind="f_core",
        jitter_mode="pocket_aware_growth",
        allowed_n_extra=(24,),
    ),
)


def _first_molecule(path: str | Path) -> Chem.Mol:
    molecule = next(iter_sdf_molecules(path), None)
    if molecule is None:
        raise ValueError(f"cannot parse ligand: {path}")
    return molecule


def build_f_core_native_ligand(
    native_ligand: str | Path,
    output_path: str | Path,
) -> Path:
    """Add the trusted CRBN-ring F substituent to the 5HXB native pose."""
    molecule = _first_molecule(native_ligand)
    pattern = Chem.MolFromSmarts(BASE_SCAFFOLD_SMARTS)
    if pattern is None:
        raise ValueError("invalid base scaffold SMARTS")
    match = molecule.GetSubstructMatch(pattern)
    if not match:
        raise ValueError("base scaffold does not match native ligand")
    anchor_idx = int(match[8])
    anchor_atom = molecule.GetAtomWithIdx(anchor_idx)
    if anchor_atom.GetTotalNumHs() < 1:
        raise ValueError("F-core anchor has no replaceable hydrogen")

    editable = Chem.RWMol(molecule)
    fluorine_idx = int(editable.AddAtom(Chem.Atom(9)))
    editable.AddBond(anchor_idx, fluorine_idx, Chem.BondType.SINGLE)
    conformer = editable.GetConformer()
    anchor_point = conformer.GetAtomPosition(anchor_idx)
    anchor = np.asarray(
        [anchor_point.x, anchor_point.y, anchor_point.z], dtype=np.float64
    )
    neighbours = []
    for neighbour in anchor_atom.GetNeighbors():
        point = conformer.GetAtomPosition(neighbour.GetIdx())
        neighbours.append([point.x, point.y, point.z])
    neighbours = np.asarray(neighbours, dtype=np.float64)
    direction = anchor - neighbours.mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        raise ValueError("cannot determine F-core exit direction")
    fluorine_position = anchor + 1.35 * direction / norm
    conformer.SetAtomPosition(fluorine_idx, fluorine_position.tolist())

    output_molecule = editable.GetMol()
    Chem.SanitizeMol(output_molecule)
    f_pattern = Chem.MolFromSmarts(F_SCAFFOLD_SMARTS)
    if f_pattern is None or not output_molecule.HasSubstructMatch(f_pattern):
        raise ValueError("constructed ligand does not contain the F scaffold")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output_path))
    try:
        writer.write(output_molecule)
    finally:
        writer.close()
    return output_path


def prepare_class_profiles(
    reference_sdf: str | Path,
    native_ligand: str | Path,
    output_dir: str | Path,
) -> tuple[Path, dict[str, Path], dict[str, dict]]:
    """Create the F-core ligand and all trusted class profiles."""
    output_dir = Path(output_dir)
    input_dir = output_dir / "inputs"
    profile_dir = output_dir / "profiles"
    f_ligand = build_f_core_native_ligand(
        native_ligand, input_dir / "5HXB_native_f_core.sdf"
    )
    profile_paths: dict[str, Path] = {}
    profiles: dict[str, dict] = {}
    for spec in CLASS_SPECS:
        ligand = native_ligand if spec.ligand_kind == "native" else f_ligand
        profile = build_class_scaffold_profile(
            reference_sdf,
            ligand,
            spec.scaffold_smarts,
            class_name=spec.name,
            reference_indices=spec.reference_indices,
            excluded_reference_indices=EXCLUDED_REFERENCE_INDICES,
            exploration_floor=0.0,
        )
        if tuple(profile["n_extra_values"]) != spec.allowed_n_extra:
            raise ValueError(
                f"{spec.name} sizes {profile['n_extra_values']} do not match "
                f"{list(spec.allowed_n_extra)}"
            )
        profile_path = profile_dir / f"{spec.name}.json"
        write_scaffold_profile(profile, profile_path)
        profile_paths[spec.name] = profile_path
        profiles[spec.name] = profile
    return f_ligand, profile_paths, profiles


def get_class_spec(name: str) -> Gspt1ClassSpec:
    for spec in CLASS_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(name)
