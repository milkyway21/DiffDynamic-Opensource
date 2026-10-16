"""Canonical heavy-atom indices shared by scaffold UI and sampling."""

from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def load_reference_molecule(path, require_3d=True):
    """Load the first 3D molecule, retaining its original index mapping."""
    suffix = Path(path).suffix.lower()
    if suffix in (".sdf", ".mol"):
        mol = Chem.MolFromMolFile(str(path), sanitize=False, removeHs=False)
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), sanitize=False, removeHs=False)
    else:
        raise ValueError("Reference molecule must be SDF, MOL or MOL2")
    if mol is None:
        raise ValueError("Reference molecule could not be parsed")
    for atom in mol.GetAtoms():
        atom.SetIntProp("_source_atom_index", atom.GetIdx())
    try:
        Chem.SanitizeMol(mol)
    except Chem.rdchem.KekulizeException:
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.AROMATIC:
                bond.SetBondType(Chem.BondType.SINGLE)
        Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    if not mol.GetNumAtoms() or not mol.GetNumConformers():
        raise ValueError("Reference molecule requires a 3D conformer")
    positions = np.asarray(mol.GetConformer().GetPositions())
    if not np.isfinite(positions).all() or (
        require_3d and not mol.GetConformer().Is3D()
    ):
        raise ValueError("Reference molecule requires finite 3D coordinates")
    return mol


def reference_molecule_from_data(data):
    """Recover the exact stored graph and pose when dataset source files moved."""
    rw = Chem.RWMol()
    elements = data.ligand_element.detach().cpu().tolist()
    for index, element in enumerate(elements):
        atom = Chem.Atom(int(element))
        atom.SetIntProp("_source_atom_index", index)
        rw.AddAtom(atom)
    kinds = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }
    edges = data.ligand_bond_index.detach().cpu().numpy()
    types = data.ligand_bond_type.detach().cpu().tolist()
    for (left, right), kind in zip(edges.T, types):
        if int(left) < int(right):
            rw.AddBond(int(left), int(right), kinds[int(kind)])
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    conformer = Chem.Conformer(len(elements))
    conformer.Set3D(True)
    for index, xyz in enumerate(data.ligand_pos.detach().cpu().tolist()):
        conformer.SetAtomPosition(index, xyz)
    mol.AddConformer(conformer)
    return mol


def resolve_scaffold_indices(mol, cfg):
    """Resolve a scaffold without silently replacing an invalid manual mask."""
    source = cfg.get("scaffold_source", "auto_murcko")
    if source == "atom_indices":
        raw = cfg.get("scaffold_atom_indices") or []
        if isinstance(raw, str):
            raw = raw.replace(",", " ").split()
        try:
            indices = sorted(set(int(i) for i in raw))
        except (ValueError, TypeError):
            raise ValueError("Scaffold indices must be zero-based integers")
        if not indices or any(i < 0 or i >= mol.GetNumAtoms() for i in indices):
            raise ValueError("Scaffold indices are empty or out of range")
        return indices
    if source in ("smarts", "custom"):
        smarts = cfg.get("scaffold_smarts")
        pattern = Chem.MolFromSmarts(smarts) if smarts else None
        if pattern is None:
            raise ValueError("Invalid or missing scaffold SMARTS")
        indices = sorted(
            {i for match in mol.GetSubstructMatches(pattern) for i in match}
        )
        if not indices:
            raise ValueError("Scaffold SMARTS does not match the reference molecule")
        return indices
    if source == "none":
        return []
    if source not in ("auto_murcko", "auto_murcko_generic"):
        raise ValueError("Unsupported scaffold source")
    core = MurckoScaffold.GetScaffoldForMol(mol)
    match = mol.GetSubstructMatch(core) if core.GetNumAtoms() else ()
    return list(match) or list(range(mol.GetNumAtoms()))


def custom_attachment_sites(mol, scaffold_indices, regions):
    """Build geometry-only sites in the original world coordinate frame."""
    positions = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    sites = []
    for index, region in enumerate(regions):
        anchor_index = int(region["anchor_atom_index"])
        if anchor_index not in scaffold_indices:
            raise ValueError(f"Region {index + 1}: anchor must belong to the scaffold")
        anchor = positions[anchor_index]
        center = np.asarray(region["center_xyz"], dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError(
                f"Region {index + 1}: center must contain three finite coordinates"
            )
        offset = center - anchor
        offset_norm = float(np.linalg.norm(offset))
        if region["method"] == "corridor" and offset_norm < 1e-6:
            raise ValueError(f"Region {index + 1}: center must differ from its anchor")
        sites.append(
            {
                "site_id": index,
                "site_kind": "custom_region",
                "anchor_scaffold_idx": anchor_index,
                "anchor_pos": anchor.tolist(),
                "centroid_pos": center.tolist(),
                "pocket_center_offset": offset.tolist(),
                "exit_direction": (offset / offset_norm).tolist()
                if offset_norm > 1e-6
                else [1.0, 0.0, 0.0],
                "removed_atom_indices": [],
                "custom_region": dict(region),
            }
        )
    return sites
