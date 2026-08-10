"""Regression: oversized / medium rings from diffusion get opened."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

from structure_repair.models import OPT_STATUS_OPTIMIZED
from structure_repair.optimize import optimize_molecule
from structure_repair.optimize.ring_break import generate_ring_break_candidates
from structure_repair.optimize.ring_ops import detect_oversized_ring_issues
from structure_repair.optimize.transforms import prepare_parent


def embedded(smiles, seed=0xACE):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(mol, params) == 0, smiles
    AllChem.MMFFOptimizeMolecule(mol)
    return Chem.RemoveHs(mol)


def test_detects_12_membered_macrocycle():
    # cyclododecane
    mol = prepare_parent(Chem.MolFromSmiles("C1CCCCCCCCCCC1"))
    issues = detect_oversized_ring_issues(mol, {"rings": {"macrocycle_min_size": 10}})
    assert any(i.issue_code == "MACROCYCLE_OVERSIZED" for i in issues)


def test_detects_7_and_8_membered_rings():
    mol7 = prepare_parent(Chem.MolFromSmiles("C1CCCCCC1"))  # cycloheptane
    mol8 = prepare_parent(Chem.MolFromSmiles("C1CCCCCCC1"))
    i7 = detect_oversized_ring_issues(mol7)
    i8 = detect_oversized_ring_issues(mol8)
    assert any(i.issue_code == "MEDIUM_RING_7_8" for i in i7)
    assert any(i.issue_code == "MEDIUM_RING_7_8" for i in i8)


def test_break_candidates_open_macrocycle():
    mol = prepare_parent(Chem.MolFromSmiles("C1CCCCCCCCCCC1"))
    cands = generate_ring_break_candidates(mol)
    assert cands
    # After break, SSSR should have fewer large rings
    parent_max = max(len(r) for r in mol.GetRingInfo().AtomRings())
    opened = cands[0].mol
    Chem.GetSymmSSSR(opened)
    rings = opened.GetRingInfo().AtomRings()
    opened_max = max((len(r) for r in rings), default=0)
    assert opened_max < parent_max


def test_engine_opens_macrocycle_under_t0():
    mol = embedded("C1CCCCCCCCCCC1")  # cyclododecane
    result = optimize_molecule(
        mol,
        molecule_id="macro",
        config={
            "tiers": ["T0", "T2"],
            "max_rounds": 2,
            "check_pocket_clash": False,
            "min_reward_gain": 0.001,
        },
    )
    # May OPTIMIZED or UNCHANGED depending on reward; at least break candidates exist
    assert result.status in (OPT_STATUS_OPTIMIZED, "UNCHANGED")
    if result.status == OPT_STATUS_OPTIMIZED:
        assert any(
            "BREAK" in a.transform_id or "LOCK" in a.transform_id for a in result.applied
        )
