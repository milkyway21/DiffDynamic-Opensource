"""Apply catalog transforms and carry parent-atom provenance forward.

Reaction SMARTS products come back without conformers, so every candidate
records ``parent_index_map`` (product atom index -> parent atom index) and
``new_atom_indices``.  ``geometry.py`` uses those to pin the pocket-conditioned
coordinates of surviving atoms and only embed what is genuinely new.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem

from .catalog import TransformSpec

_REACT_ATOM_IDX = "react_atom_idx"
PROP_PARENT_IDX = "_sr_parent_idx"


@dataclass
class TransformCandidate:
    """One product of one transform, before geometry and reward."""

    transform_id: str
    tier: str
    name: str
    rationale: str
    mol: Chem.Mol
    cost: float
    requires_3d: bool
    reward_bonus: float = 0.0
    parent_index_map: Dict[int, int] = field(default_factory=dict)
    new_atom_indices: List[int] = field(default_factory=list)
    delta_heavy: int = 0
    smiles: str = ""

    @property
    def has_new_atoms(self) -> bool:
        return bool(self.new_atom_indices)


def heavy_atom_count(mol: Chem.Mol) -> int:
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def canonical_smiles(mol: Chem.Mol) -> str:
    try:
        copied = Chem.Mol(mol)
        for atom in copied.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(copied)
    except Exception:  # noqa: BLE001
        return ""


def prepare_parent(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Heavy-atom working copy with atom maps cleared and provenance stamped."""
    if mol is None:
        return None
    try:
        working = Chem.RemoveHs(Chem.Mol(mol))
    except Exception:  # noqa: BLE001
        working = Chem.Mol(mol)
    for atom in working.GetAtoms():
        atom.SetAtomMapNum(0)
        atom.SetIntProp(PROP_PARENT_IDX, atom.GetIdx())
    return working


def _resolve_provenance(
    product: Chem.Mol, parent: Chem.Mol
) -> Tuple[Dict[int, int], List[int]]:
    """Map product atoms back to parent atoms via RDKit's react_atom_idx."""
    index_map: Dict[int, int] = {}
    new_atoms: List[int] = []
    n_parent = parent.GetNumAtoms()
    for atom in product.GetAtoms():
        atom.SetAtomMapNum(0)
        parent_idx = None
        if atom.HasProp(_REACT_ATOM_IDX):
            try:
                parent_idx = int(atom.GetProp(_REACT_ATOM_IDX))
            except Exception:  # noqa: BLE001
                parent_idx = None
        if parent_idx is None and atom.HasProp(PROP_PARENT_IDX):
            try:
                parent_idx = int(atom.GetProp(PROP_PARENT_IDX))
            except Exception:  # noqa: BLE001
                parent_idx = None
        if parent_idx is not None and 0 <= parent_idx < n_parent:
            index_map[atom.GetIdx()] = parent_idx
            atom.SetIntProp(PROP_PARENT_IDX, parent_idx)
        else:
            new_atoms.append(atom.GetIdx())
            if atom.HasProp(PROP_PARENT_IDX):
                atom.ClearProp(PROP_PARENT_IDX)
    return index_map, new_atoms


def _finalize_product(product: Chem.Mol) -> Optional[Chem.Mol]:
    try:
        product.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(product)
    except Exception:  # noqa: BLE001
        return None
    if len(Chem.GetMolFrags(product)) != 1:
        return None
    return product


def apply_transform(
    parent: Chem.Mol, spec: TransformSpec, max_products: int = 4
) -> List[TransformCandidate]:
    """Enumerate distinct, sanitizable products of one catalog transform."""
    reaction = spec.reaction
    if reaction is None:
        return []
    try:
        product_sets = reaction.RunReactants((parent,))
    except Exception:  # noqa: BLE001
        return []
    if not product_sets:
        return []

    parent_heavy = heavy_atom_count(parent)
    candidates: List[TransformCandidate] = []
    seen = set()
    for product_set in product_sets:
        if len(candidates) >= max_products:
            break
        if not product_set:
            continue
        product = Chem.RWMol(product_set[0]).GetMol()
        index_map, new_atoms = _resolve_provenance(product, parent)
        finalized = _finalize_product(product)
        if finalized is None:
            continue
        smiles = canonical_smiles(finalized)
        if not smiles or smiles in seen:
            continue
        seen.add(smiles)
        candidates.append(
            TransformCandidate(
                transform_id=spec.transform_id,
                tier=spec.tier,
                name=spec.name,
                rationale=spec.rationale,
                mol=finalized,
                cost=spec.cost,
                requires_3d=bool(spec.requires_3d and new_atoms),
                reward_bonus=spec.reward_bonus,
                parent_index_map=index_map,
                new_atom_indices=new_atoms,
                delta_heavy=heavy_atom_count(finalized) - parent_heavy,
                smiles=smiles,
            )
        )
    return candidates


def enumerate_candidates(
    parent: Chem.Mol,
    specs: Sequence[TransformSpec],
    max_products_per_transform: int = 4,
    applied_counts: Optional[Dict[str, int]] = None,
) -> List[TransformCandidate]:
    """Run every eligible catalog transform against one parent molecule."""
    applied_counts = applied_counts or {}
    out: List[TransformCandidate] = []
    for spec in specs:
        if applied_counts.get(spec.transform_id, 0) >= spec.max_applications:
            continue
        out.extend(
            apply_transform(parent, spec, max_products=max_products_per_transform)
        )
    return out
