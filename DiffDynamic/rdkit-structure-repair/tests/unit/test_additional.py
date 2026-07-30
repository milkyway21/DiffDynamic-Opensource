"""Additional unit tests: valence, FG detection, scoring, identity, C5, audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem

from structure_repair import assign_atom_maps, detect_issues, load_config, repair_molecule, staged_sanitize
from structure_repair.audit import append_audit_jsonl, explain_audit_record, load_audit_jsonl, result_to_audit_dict
from structure_repair.candidate_scorer import compute_edit_cost, score_molecule
from structure_repair.corruption import (
    corrupt_aromatic_to_all_single,
    corrupt_clear_charges,
    mol_from_smiles,
)
from structure_repair.identity_check import compare_atom_elements, compare_heavy_atom_count, identity_ok
from structure_repair.io import mol_to_canonical_smiles
from structure_repair.models import AtomEdit, BondEdit, STATUS_REPAIRED, STATUS_UNCHANGED
from structure_repair.mol_edit import molecule_state_hash
from structure_repair.rules.aromatic_ring_rules import generate_c6_kekule_candidates
from structure_repair.rings import order_ring_atoms
from structure_repair.valence import detect_valence_issues


@pytest.fixture
def config():
    return load_config()


def test_carbon_overvalent_detected():
    rw = Chem.RWMol()
    center = rw.AddAtom(Chem.Atom(6))
    for _ in range(5):
        idx = rw.AddAtom(Chem.Atom(6))
        rw.AddBond(center, idx, Chem.BondType.SINGLE)
    mol = assign_atom_maps(rw.GetMol())
    mol.UpdatePropertyCache(strict=False)
    issues = detect_valence_issues(mol)
    assert any(i.issue_code == "CARBON_OVERVALENT" for i in issues)


def test_oxygen_overvalent_without_charge():
    rw = Chem.RWMol()
    o = rw.AddAtom(Chem.Atom(8))
    for _ in range(3):
        c = rw.AddAtom(Chem.Atom(6))
        rw.AddBond(o, c, Chem.BondType.SINGLE)
    mol = assign_atom_maps(rw.GetMol())
    mol.UpdatePropertyCache(strict=False)
    issues = detect_valence_issues(mol)
    assert any(i.issue_code == "OXYGEN_OVERVALENT" for i in issues)


def test_nitro_issue_codes(config):
    mol = assign_atom_maps(mol_from_smiles("c1ccc(cc1)[N+](=O)[O-]"))
    # Manually break charges
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 7:
            atom.SetFormalCharge(0)
            for bond in atom.GetBonds():
                if bond.GetOtherAtom(atom).GetAtomicNum() == 8:
                    bond.SetBondType(Chem.BondType.DOUBLE)
                    bond.GetOtherAtom(atom).SetFormalCharge(0)
    broken = rw.GetMol()
    broken.UpdatePropertyCache(strict=False)
    codes = {i.issue_code for i in detect_issues(broken, config)}
    assert "NITRO_BOND_ORDER_ERROR" in codes


def test_quat_ammonium_permanent_ion(config):
    mol = assign_atom_maps(mol_from_smiles("C[N+](C)(C)C"))
    issues = detect_issues(mol, config)
    assert any(i.issue_code == "PERMANENT_ION" for i in issues)
    result = repair_molecule(mol, config=config, molecule_id="quat")
    assert result.status == STATUS_UNCHANGED


def test_generate_c6_kekule_from_all_single():
    corrupted = corrupt_aromatic_to_all_single(mol_from_smiles("c1ccccc1"))
    Chem.GetSymmSSSR(corrupted)
    ring = list(corrupted.GetRingInfo().AtomRings()[0])
    ordered = order_ring_atoms(corrupted, ring)
    cands = generate_c6_kekule_candidates(corrupted, ordered)
    assert len(cands) >= 1
    assert mol_to_canonical_smiles(cands[0]) == "c1ccccc1"


def test_identity_preserves_elements(config):
    a = assign_atom_maps(mol_from_smiles("c1ccccc1"))
    b = Chem.Mol(a)
    assert compare_atom_elements(a, b)
    assert compare_heavy_atom_count(a, b)
    ok, errs = identity_ok(a, b, config)
    assert ok and errs == []


def test_edit_cost_weights(config):
    cost = compute_edit_cost(
        [BondEdit(1, 2, "SINGLE", "DOUBLE", "set")],
        [AtomEdit(1, "formal_charge", 0, 1)],
        config,
    )
    assert cost == 2.0


def test_molecule_state_hash_stable():
    m1 = assign_atom_maps(mol_from_smiles("CCO"))
    m2 = assign_atom_maps(mol_from_smiles("CCO"))
    assert molecule_state_hash(m1) == molecule_state_hash(m2)


def test_score_clean_lower_than_corrupted(config):
    clean = mol_from_smiles("c1ccccc1")
    bad = corrupt_aromatic_to_all_single(clean)
    assert score_molecule(clean, config) < score_molecule(bad, config)


def test_pyrrole_unchanged(config):
    result = repair_molecule(mol_from_smiles("c1cc[nH]c1"), config=config, molecule_id="pyrrole")
    assert result.status == STATUS_UNCHANGED


def test_pyridine_unchanged(config):
    result = repair_molecule(mol_from_smiles("c1ccncc1"), config=config, molecule_id="pyridine")
    assert result.status == STATUS_UNCHANGED


def test_audit_jsonl_roundtrip(tmp_path, config):
    result = repair_molecule(mol_from_smiles("c1ccccc1"), config=config, molecule_id="audit1")
    path = tmp_path / "audit.jsonl"
    append_audit_jsonl(path, result)
    rows = load_audit_jsonl(path)
    assert len(rows) == 1
    assert rows[0]["molecule_id"] == "audit1"
    assert rows[0]["status"] == STATUS_UNCHANGED
    text = explain_audit_record(rows[0])
    assert "audit1" in text


def test_result_audit_dict_has_required_fields(config):
    corrupted = corrupt_aromatic_to_all_single(mol_from_smiles("c1ccccc1"))
    result = repair_molecule(corrupted, config=config, molecule_id="GEN_00128")
    d = result_to_audit_dict(result)
    for key in (
        "molecule_id",
        "status",
        "original_smiles",
        "repaired_smiles",
        "issues_before",
        "rules_applied",
        "bond_edits",
        "score_before",
        "score_after",
        "confidence",
    ):
        assert key in d
    assert d["status"] == STATUS_REPAIRED
    assert d["confidence"] > 0


def test_clear_charges_on_quat_gets_repaired_or_rejected(config):
    mol = corrupt_clear_charges(mol_from_smiles("C[N+](C)(C)C"))
    result = repair_molecule(mol, config=config, molecule_id="quat_corrupt")
    # May repair by adding + charge back, or reject if ambiguous — must not invent elements
    if result.status == STATUS_REPAIRED:
        assert compare_atom_elements(result.original_mol, result.repaired_mol)
        smi = mol_to_canonical_smiles(result.repaired_mol)
        assert "[N+]" in smi


def test_fused_aromatic_unchanged(config):
    result = repair_molecule(mol_from_smiles("c1ccc2ccccc2c1"), config=config, molecule_id="naph")
    assert result.status == STATUS_UNCHANGED


def test_ketene_unchanged(config):
    result = repair_molecule(Chem.MolFromSmiles("C=C=O"), config=config, molecule_id="ketene")
    assert result.status == STATUS_UNCHANGED


def test_original_mol_not_mutated(config):
    mol = corrupt_aromatic_to_all_single(mol_from_smiles("c1ccccc1"))
    before = Chem.MolToMolBlock(mol)
    repair_molecule(mol, config=config, molecule_id="immut")
    after = Chem.MolToMolBlock(mol)
    assert before == after
