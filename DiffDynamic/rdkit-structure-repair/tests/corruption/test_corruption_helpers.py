"""Corruption helpers and small dataset builder smoke tests."""

from __future__ import annotations

from structure_repair.corruption import (
    corrupt_aromatic_to_all_single,
    corrupt_c6_to_consecutive_doubles,
    corrupt_nitro_neutral_double,
    mol_from_smiles,
)
from structure_repair.io import mol_to_canonical_smiles


def test_corrupt_all_single_not_aromatic():
    mol = corrupt_aromatic_to_all_single(mol_from_smiles("c1ccccc1"))
    assert not any(a.GetIsAromatic() for a in mol.GetAtoms())
    assert all(
        b.GetBondType().name == "SINGLE" or not b.GetIsAromatic()
        for b in mol.GetBonds()
    )


def test_corrupt_consecutive_has_doubles():
    mol = corrupt_c6_to_consecutive_doubles(mol_from_smiles("c1ccccc1"))
    doubles = sum(1 for b in mol.GetBonds() if b.GetBondType().name == "DOUBLE")
    assert doubles >= 5


def test_corrupt_nitro_loses_charges():
    mol = corrupt_nitro_neutral_double(mol_from_smiles("c1ccc(cc1)[N+](=O)[O-]"))
    n_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    assert n_atoms
    assert n_atoms[0].GetFormalCharge() == 0
