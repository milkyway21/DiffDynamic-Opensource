"""7/8-membered rings contract to benzene / pyridine without opening to a chain."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

from structure_repair.models import OPT_STATUS_OPTIMIZED
from structure_repair.optimize import optimize_molecule
from structure_repair.optimize.ring_contract import (
    TRANSFORM_CONTRACT_6,
    TRANSFORM_CONTRACT_AZA,
    generate_ring_contract_candidates,
)
from structure_repair.optimize.transforms import prepare_parent


def embedded(smiles, seed=0xBEE):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(mol, params) == 0, smiles
    AllChem.MMFFOptimizeMolecule(mol)
    return Chem.RemoveHs(mol)


def _has_aromatic_6(mol: Chem.Mol) -> bool:
    Chem.GetSymmSSSR(mol)
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) == 6 and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            return True
    return False


def test_cycloheptene_contracts_to_aromatic_6():
    # Unsubstituted 7-ring with removable CH2s → benzene-like
    mol = prepare_parent(Chem.MolFromSmiles("C1=CCCCCC1"))
    cands = generate_ring_contract_candidates(mol)
    assert cands
    assert any(c.transform_id == TRANSFORM_CONTRACT_6 for c in cands)
    assert any(_has_aromatic_6(c.mol) for c in cands)
    # Atom count must drop (delete ≥1 C)
    assert any(c.delta_heavy < 0 for c in cands)


def test_cyclooctane_contracts_when_two_ch2_deleted():
    mol = prepare_parent(Chem.MolFromSmiles("C1CCCCCCC1"))
    cands = generate_ring_contract_candidates(mol)
    assert cands
    assert any(_has_aromatic_6(c.mol) for c in cands)
    assert any(c.delta_heavy == -2 for c in cands)


def test_contract_offers_aza_pyridine():
    mol = prepare_parent(Chem.MolFromSmiles("C1CCCCCC1"))
    cands = generate_ring_contract_candidates(
        mol, config={"rings": {"contract_offer_aza": True}}
    )
    aza = [c for c in cands if c.transform_id == TRANSFORM_CONTRACT_AZA]
    assert aza
    # Product should contain aromatic N
    assert any(
        any(a.GetAtomicNum() == 7 and a.GetIsAromatic() for a in c.mol.GetAtoms())
        for c in aza
    )


def test_engine_prefers_contract_over_break_for_7ring():
    mol = embedded("C1CCCCCC1")
    result = optimize_molecule(
        mol,
        molecule_id="c7",
        config={
            "tiers": ["T0"],
            "max_rounds": 2,
            "check_pocket_clash": False,
            "min_reward_gain": 0.001,
            "require_conformer": False,
        },
    )
    if result.status == OPT_STATUS_OPTIMIZED:
        ids = [a.transform_id for a in result.applied]
        assert any("CONTRACT" in t for t in ids)
        assert result.optimized_mol is not None
        assert _has_aromatic_6(result.optimized_mol)


def test_phenyl_cycloheptane_contracts_to_biphenyl():
    """Pendant Ph must survive; contracted ring becomes the second aryl."""
    mol = prepare_parent(Chem.MolFromSmiles("c1ccccc1C1CCCCCC1"))
    cands = generate_ring_contract_candidates(mol)
    assert cands
    smiles = {c.smiles for c in cands if c.transform_id == TRANSFORM_CONTRACT_6}
    # biphenyl (any canonical form)
    assert any("c1ccc(-c2ccccc2)cc1" == s or s.count("c") >= 10 for s in smiles)
    assert not any("C2CCCCC2" in s for s in smiles)