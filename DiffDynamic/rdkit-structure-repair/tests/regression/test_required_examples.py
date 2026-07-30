"""Regression examples required by the project spec."""

from __future__ import annotations

import pytest
from rdkit import Chem

from structure_repair import load_config, repair_molecule
from structure_repair.corruption import (
    corrupt_aromatic_to_all_single,
    corrupt_c6_to_consecutive_doubles,
    corrupt_nitro_neutral_double,
    mol_from_smiles,
)
from structure_repair.io import mol_to_canonical_smiles
from structure_repair.models import STATUS_AMBIGUOUS, STATUS_REPAIRED, STATUS_UNCHANGED


@pytest.fixture
def config():
    return load_config()


def test_all_single_benzene_recovered(config):
    corrupted = corrupt_aromatic_to_all_single(mol_from_smiles("c1ccccc1"))
    result = repair_molecule(corrupted, config=config, molecule_id="GEN_all_single")
    assert result.status == STATUS_REPAIRED
    assert mol_to_canonical_smiles(result.repaired_mol) == "c1ccccc1"
    assert "C6_AROMATIC_RECOVERY" in result.applied_rule_ids
    assert result.score_after is not None
    assert result.score_after < result.score_before


def test_consecutive_double_c6_recovered(config):
    corrupted = corrupt_c6_to_consecutive_doubles(mol_from_smiles("c1ccccc1"))
    result = repair_molecule(corrupted, config=config, molecule_id="GEN_consec")
    assert result.status == STATUS_REPAIRED
    assert mol_to_canonical_smiles(result.repaired_mol) == "c1ccccc1"
    assert any(
        r in result.applied_rule_ids
        for r in ("C6_AROMATIC_RECOVERY", "CUMULENE_VALIDATION_AND_REPAIR")
    )


def test_legal_allene_unchanged(config):
    mol = Chem.MolFromSmiles("C=C=C")
    result = repair_molecule(mol, config=config, molecule_id="allene")
    assert result.status == STATUS_UNCHANGED
    assert mol_to_canonical_smiles(result.repaired_mol or result.original_mol) == "C=C=C"


def test_nitro_charge_and_bond_joint_repair(config):
    corrupted = corrupt_nitro_neutral_double(mol_from_smiles("c1ccc(cc1)[N+](=O)[O-]"))
    result = repair_molecule(corrupted, config=config, molecule_id="nitro")
    assert result.status == STATUS_REPAIRED
    smi = mol_to_canonical_smiles(result.repaired_mol)
    assert smi is not None
    assert "[N+]" in smi and "[O-]" in smi
    assert "FUNCTIONAL_GROUP_REPAIR" in result.applied_rule_ids or "CHARGE_REPAIR" in result.applied_rule_ids


def test_ambiguous_when_winner_margin_too_small(config):
    """Two carboxylate-like repairs with inflated margin → AMBIGUOUS."""
    rw = Chem.RWMol()
    c0 = rw.AddAtom(Chem.Atom(6))
    c1 = rw.AddAtom(Chem.Atom(6))
    o1 = rw.AddAtom(Chem.Atom(8))
    o2 = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(c0, c1, Chem.BondType.SINGLE)
    rw.AddBond(c1, o1, Chem.BondType.SINGLE)
    rw.AddBond(c1, o2, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)

    cfg = dict(config)
    cfg["repair"] = dict(config.get("repair", {}))
    cfg["repair"]["minimum_winner_margin"] = 5000
    result = repair_molecule(mol, config=cfg, molecule_id="ambig")
    assert result.status == STATUS_AMBIGUOUS
    assert result.reject_reason == "winner_margin_too_small"
    assert result.repaired_mol is None
