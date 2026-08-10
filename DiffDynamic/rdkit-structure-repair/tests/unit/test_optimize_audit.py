"""Unit tests: the audit trail.

``optimize_audit.csv`` is the only record of what the layer did to a batch, so
it has to stay readable and complete even for molecules nothing happened to.
"""

from __future__ import annotations

import csv

from rdkit import Chem
from rdkit.Chem import AllChem

from structure_repair.audit import (
    OPTIMIZE_AUDIT_FIELDS,
    optimize_result_to_audit_dict,
    write_optimize_audit_csv,
)
from structure_repair.models import OPT_STATUS_OPTIMIZED, OPT_STATUS_UNCHANGED
from structure_repair.optimize import optimize_molecule


def flat_diene():
    """A planar cyclohexadiene, i.e. a benzene the bond perception missed."""
    import math

    mol = Chem.AddHs(Chem.MolFromSmiles("C1=CC=CC(C(=O)NC)C1"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xBEEF
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    for i, idx in enumerate(mol.GetRingInfo().AtomRings()[0]):
        angle = 2.0 * math.pi / 6.0 * i
        conf.SetAtomPosition(idx, (1.39 * math.cos(angle), 1.39 * math.sin(angle), 0.0))
    return mol


def optimized_result():
    result = optimize_molecule(
        flat_diene(), molecule_id="7", config={"tiers": ["T0"], "check_pocket_clash": False}
    )
    assert result.status == OPT_STATUS_OPTIMIZED
    return result


def unchanged_result():
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1C(=O)NC"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xBEEF
    AllChem.EmbedMolecule(mol, params)
    result = optimize_molecule(
        Chem.RemoveHs(mol),
        molecule_id="8",
        config={"tiers": ["T0"], "check_pocket_clash": False},
    )
    assert result.status == OPT_STATUS_UNCHANGED
    return result


def test_audit_row_has_every_declared_field():
    row = optimize_result_to_audit_dict(optimized_result())
    assert set(row) == set(OPTIMIZE_AUDIT_FIELDS)


def test_audit_row_records_the_transform_and_its_effect():
    row = optimize_result_to_audit_dict(optimized_result())
    assert row["molecule_id"] == "7"
    assert row["status"] == OPT_STATUS_OPTIMIZED
    assert row["transform_chain"] == "T0_AROMATIZE_C6_CARBOCYCLE"
    assert row["transform_tiers"] == "T0"
    assert int(row["n_transforms"]) == 1
    assert float(row["total_reward"]) > 0
    assert float(row["qed_after"]) > float(row["qed_before"])


def test_audit_smiles_carry_no_explicit_hydrogens():
    row = optimize_result_to_audit_dict(optimized_result())
    for key in ("smiles_before", "smiles_after"):
        assert "[H]" not in row[key], f"{key} should be a heavy-atom SMILES"
    assert row["smiles_before"] != row["smiles_after"]


def test_unchanged_molecules_still_get_a_row():
    """A batch summary that silently drops the untouched majority is useless."""
    row = optimize_result_to_audit_dict(unchanged_result())
    assert row["status"] == OPT_STATUS_UNCHANGED
    assert row["transform_chain"] == ""
    assert int(row["n_transforms"]) == 0
    assert row["smiles_before"] == row["smiles_after"]


def test_csv_round_trips(tmp_path):
    path = tmp_path / "optimize_audit.csv"
    write_optimize_audit_csv(path, [optimized_result(), unchanged_result()])

    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 2
    assert list(rows[0]) == OPTIMIZE_AUDIT_FIELDS
    assert {r["molecule_id"] for r in rows} == {"7", "8"}


def test_rejection_reasons_are_reported(tmp_path):
    """Knowing *why* candidates died is what makes the gates tunable."""
    result = optimize_molecule(
        flat_diene(),
        molecule_id="9",
        config={"tiers": ["T1"], "check_pocket_clash": False, "min_reward_gain": 1e6},
    )
    row = optimize_result_to_audit_dict(result)
    assert row["top_reject_reasons"], "no rejection reasons were recorded"
    assert ":" in row["top_reject_reasons"], "reasons should carry counts"
