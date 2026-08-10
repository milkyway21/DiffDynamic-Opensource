"""Unit tests: the 3D contract.

Molecules reaching this layer carry a pose the diffusion model generated inside
a protein pocket.  Two things must hold no matter what a transform does: atoms
that survive keep their exact coordinates, and nothing new may be placed inside
the protein.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from structure_repair.optimize.geometry import (
    CORE_DISPLACEMENT_TOLERANCE,
    PocketClashChecker,
    embed_new_atoms,
    has_conformer,
    load_pocket_coords,
    max_core_displacement,
    transfer_conformer,
)


def embedded(smiles, seed=0xC0FFEE):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(mol, params) == 0, smiles
    AllChem.MMFFOptimizeMolecule(mol)
    return Chem.RemoveHs(mol)


def identity_map(mol):
    return {i: i for i in range(mol.GetNumAtoms())}


def coords(mol):
    conf = mol.GetConformer()
    return np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=float
    )


# ------------------------------------------------------------------ transfer


def test_transfer_conformer_copies_coordinates_exactly():
    parent = embedded("c1ccccc1CCO")
    candidate = Chem.Mol(parent)
    candidate.RemoveAllConformers()
    assert transfer_conformer(parent, candidate, identity_map(parent))
    assert np.allclose(coords(parent), coords(candidate), atol=1e-9)


def test_transfer_conformer_refuses_when_atoms_are_unmapped():
    parent = embedded("c1ccccc1CCO")
    candidate = Chem.MolFromSmiles("c1ccccc1CCOC")
    partial = {i: i for i in range(parent.GetNumAtoms())}
    assert not transfer_conformer(parent, candidate, partial)


# ------------------------------------------------------------------ embedding


def test_embedding_new_atoms_keeps_the_core_pinned():
    """The pocket-conditioned pose is the whole point; it must not move."""
    parent = embedded("c1ccccc1CCO")
    candidate = Chem.MolFromSmiles("c1ccccc1CCOC")  # O-methylation, one new atom
    parent_map = {i: i for i in range(parent.GetNumAtoms())}

    posed = embed_new_atoms(parent, candidate, parent_map, seed=7)
    assert posed is not None
    assert has_conformer(posed)
    assert posed.GetNumAtoms() == candidate.GetNumAtoms()

    displacement = max_core_displacement(parent, posed, parent_map)
    assert displacement is not None
    assert displacement <= CORE_DISPLACEMENT_TOLERANCE


def test_new_atom_lands_at_a_sane_bond_distance():
    parent = embedded("c1ccccc1CCO")
    candidate = Chem.MolFromSmiles("c1ccccc1CCOC")
    parent_map = {i: i for i in range(parent.GetNumAtoms())}
    posed = embed_new_atoms(parent, candidate, parent_map, seed=7)
    assert posed is not None

    new_indices = [i for i in range(posed.GetNumAtoms()) if i not in parent_map]
    assert len(new_indices) == 1
    new_idx = new_indices[0]
    conf = posed.GetConformer()
    for nbr in posed.GetAtomWithIdx(new_idx).GetNeighbors():
        d = conf.GetAtomPosition(new_idx).Distance(conf.GetAtomPosition(nbr.GetIdx()))
        assert 1.0 < d < 2.0, f"new atom sits {d:.2f} A from its neighbour"


def test_embedding_without_a_parent_conformer_is_refused():
    parent = Chem.MolFromSmiles("c1ccccc1CCO")
    candidate = Chem.MolFromSmiles("c1ccccc1CCOC")
    assert embed_new_atoms(parent, candidate, {i: i for i in range(9)}) is None


def test_embedding_is_time_bounded():
    """A contradictory bounds matrix must not be able to spin forever."""
    parent = embedded("CC(=O)C1C(=O)C(CC2NC3C(O)CC=CC3C2)=CC1C")
    candidate = Chem.MolFromSmiles("CC(=O)C1C(=O)C(CC2NC3C(O)CC=CC3C2)=CC1CF")
    parent_map = {i: i for i in range(parent.GetNumAtoms())}
    started = time.monotonic()
    embed_new_atoms(parent, candidate, parent_map, seed=3, time_budget_s=1.0)
    assert time.monotonic() - started < 20.0


# --------------------------------------------------------------- pocket clash


@pytest.fixture
def pocket_pdb(tmp_path):
    """Three protein carbons sitting on the origin."""
    lines = []
    for i, (x, y, z) in enumerate([(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (0.0, 1.5, 0.0)]):
        lines.append(
            f"ATOM  {i + 1:5d}  CA  ALA A{i + 1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    path = tmp_path / "pocket.pdb"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_pocket_coords_skip_hydrogens(tmp_path):
    path = tmp_path / "withh.pdb"
    path.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  HA  ALA A   1       1.000   0.000   0.000  1.00  0.00           H\n"
        "END\n"
    )
    loaded = load_pocket_coords(str(path))
    assert loaded is not None and len(loaded) == 1


def test_molecule_on_top_of_the_protein_is_a_clash(pocket_pdb):
    checker = PocketClashChecker(str(pocket_pdb), cutoff=2.2)
    assert checker.active
    mol = embedded("CCO")
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):  # drop the ligand onto the protein
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x * 0.1, p.y * 0.1, p.z * 0.1))
    assert checker.has_clash(mol)


def test_molecule_far_away_is_not_a_clash(pocket_pdb):
    checker = PocketClashChecker(str(pocket_pdb), cutoff=2.2)
    mol = embedded("CCO")
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x + 50.0, p.y, p.z))
    assert not checker.has_clash(mol)


def test_only_the_listed_atoms_are_checked(pocket_pdb):
    """New atoms are what the engine screens; the pinned core is a given."""
    checker = PocketClashChecker(str(pocket_pdb), cutoff=2.2)
    mol = embedded("CCO")
    conf = mol.GetConformer()
    conf.SetAtomPosition(0, (0.0, 0.0, 0.0))  # atom 0 buried in the protein
    for i in range(1, mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (50.0 + i, 0.0, 0.0))
    assert checker.has_clash(mol, [0])
    assert not checker.has_clash(mol, [1])


def test_checker_without_a_protein_is_inactive():
    checker = PocketClashChecker(None)
    assert not checker.active
    assert not checker.has_clash(embedded("CCO"))
