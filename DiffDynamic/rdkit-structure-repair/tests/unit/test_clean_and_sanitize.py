"""Unit tests: atom maps, staged sanitize, valence, clean molecules unchanged."""

from __future__ import annotations

import pytest
from rdkit import Chem

from structure_repair import assign_atom_maps, detect_issues, load_config, repair_molecule, staged_sanitize
from structure_repair.corruption import mol_from_smiles
from structure_repair.io import mol_to_canonical_smiles
from structure_repair.models import STATUS_UNCHANGED


@pytest.fixture
def config():
    return load_config()


CLEAN_SMILES = [
    ("benzene", "c1ccccc1"),
    ("cyclohexane", "C1CCCCC1"),
    ("cyclohexene", "C1=CCCCC1"),
    ("cyclohexadiene_13", "C1=CC=CCC1"),
    ("cyclohexadiene_14", "C1=CCC=CC1"),
    ("allene", "C=C=C"),
    ("cumulene", "C=C=C=C"),
    ("ketene", "C=C=O"),
    ("carbodiimide", "N=C=N"),
    ("pyridine", "c1ccncc1"),
    ("pyrrole", "c1cc[nH]c1"),
    ("imidazole", "c1c[nH]cn1"),
    ("furan", "c1ccoc1"),
    ("thiophene", "c1ccsc1"),
    ("nitrobenzene", "c1ccc(cc1)[N+](=O)[O-]"),
    ("carboxylate", "CC(=O)[O-]"),
    ("quat_ammonium", "C[N+](C)(C)C"),
    ("zwitterion", "C(C(=O)[O-])[NH3+]"),
    ("sulfone", "CS(=O)(=O)C"),
    ("phosphate", "O=P(O)(O)O"),
    ("naphthalene", "c1ccc2ccccc2c1"),
]


def test_assign_atom_maps_stable():
    mol = mol_from_smiles("c1ccccc1")
    mapped = assign_atom_maps(mol)
    maps = [a.GetAtomMapNum() for a in mapped.GetAtoms()]
    assert maps == list(range(1, 7))
    # Second call does not overwrite
    mapped2 = assign_atom_maps(mapped)
    assert [a.GetAtomMapNum() for a in mapped2.GetAtoms()] == maps


def test_staged_sanitize_benzene_ok():
    mol = mol_from_smiles("c1ccccc1")
    report = staged_sanitize(mol)
    assert report.overall_success
    assert report.failure_types == []


def test_staged_sanitize_detects_bad_valence():
    # Build pentavalent carbon manually
    rw = Chem.RWMol()
    idxs = [rw.AddAtom(Chem.Atom(6)) for _ in range(5)]
    center = rw.AddAtom(Chem.Atom(6))
    for i in idxs:
        rw.AddBond(center, i, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    report = staged_sanitize(mol)
    assert not report.overall_success


@pytest.mark.parametrize("name,smi", CLEAN_SMILES)
def test_clean_molecule_unchanged(name, smi, config):
    mol = mol_from_smiles(smi)
    result = repair_molecule(mol, config=config, molecule_id=name)
    assert result.status == STATUS_UNCHANGED, (
        f"{name}: expected UNCHANGED got {result.status} reason={result.reject_reason} "
        f"issues={[i.issue_code for i in result.issues_before]}"
    )
    before = mol_to_canonical_smiles(mol)
    after = mol_to_canonical_smiles(result.repaired_mol or result.original_mol)
    assert before == after


def test_detect_issues_on_clean_benzene(config):
    mol = assign_atom_maps(mol_from_smiles("c1ccccc1"))
    issues = detect_issues(mol, config)
    actionable = [i for i in issues if i.repairable and i.severity in {"error", "warning"}]
    assert actionable == []
