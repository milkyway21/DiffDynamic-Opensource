"""Unit tests: the greedy optimization loop end to end.

Covers the flagship use case (a flat cyclohexadiene becoming benzene), the
guard rails that stop the loop, and the provenance the audit trail depends on.
"""

from __future__ import annotations

import pickle
import time

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from structure_repair.models import (
    OPT_STATUS_OPTIMIZED,
    OPT_STATUS_REJECTED,
    OPT_STATUS_UNCHANGED,
)
from structure_repair.optimize import optimize_molecule


def embedded(smiles, seed=0xBEEF):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(mol, params) == 0, smiles
    AllChem.MMFFOptimizeMolecule(mol)
    return Chem.RemoveHs(mol)


def flatten_ring(mol, ring_atoms):
    """Force one ring into a plane so T0's planarity evidence is satisfied.

    A real diffusion sample arrives nearly flat already; this reproduces that
    geometry deterministically instead of hoping MMFF lands there.
    """
    conf = mol.GetConformer()
    for i, idx in enumerate(ring_atoms):
        angle = 2.0 * 3.14159265 / len(ring_atoms) * i
        import math

        conf.SetAtomPosition(
            idx, (1.39 * math.cos(angle), 1.39 * math.sin(angle), 0.0)
        )
    return mol


def heavy_smiles(mol):
    return Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)))


# Big enough that a single edit does not tank the Morgan similarity, and it
# carries a nitro alert, an ester and an ether chain for the tiers to bite on.
BIG_PROBE = "O=[N+]([O-])c1ccc(CCOC)cc1C(=O)OC"


def config(**overrides):
    cfg = {
        "tiers": ["T0", "T1", "T2", "T3", "T4", "T5"],
        "max_rounds": 3,
        "check_pocket_clash": False,
    }
    cfg.update(overrides)
    return cfg


# ------------------------------------------------------- flagship regression


def test_planar_cyclohexadiene_becomes_benzene():
    """The use case that motivated the layer: two isolated double bonds in a
    flat all-carbon six-ring are an unaromatized benzene, not a real diene."""
    mol = embedded("C1=CC=CC(C(=O)NC)C1")
    ring = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
    assert len(ring) == 6
    flatten_ring(mol, ring)

    result = optimize_molecule(mol, molecule_id="diene", config=config(tiers=["T0"]))

    assert result.status == OPT_STATUS_OPTIMIZED
    assert result.transform_chain == "T0_AROMATIZE_C6_CARBOCYCLE"
    product = result.optimized_mol
    assert "c1ccccc1" in heavy_smiles(product) or product.GetAromaticAtoms()
    aromatic_ring_atoms = sum(
        1 for a in product.GetAtoms() if a.GetIsAromatic() and a.IsInRing()
    )
    assert aromatic_ring_atoms == 6


def test_aromatization_preserves_every_heavy_atom():
    mol = embedded("C1=CC=CC(C(=O)NC)C1")
    flatten_ring(mol, [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()])
    before = Chem.RemoveHs(Chem.Mol(mol)).GetNumAtoms()

    result = optimize_molecule(mol, molecule_id="diene", config=config(tiers=["T0"]))

    assert result.status == OPT_STATUS_OPTIMIZED
    assert Chem.RemoveHs(result.optimized_mol).GetNumAtoms() == before


def test_a_puckered_ring_is_left_alone():
    """Without planarity evidence the diene is a real diene, not a benzene."""
    mol = embedded("C1=CC=CC(C(=O)NC)C1")
    conf = mol.GetConformer()
    for i, idx in enumerate(a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()):
        p = conf.GetAtomPosition(idx)  # pucker it well past the plane tolerance
        conf.SetAtomPosition(idx, (p.x, p.y, p.z + (0.9 if i % 2 else -0.9)))

    result = optimize_molecule(mol, molecule_id="puckered", config=config(tiers=["T0"]))

    assert result.status == OPT_STATUS_UNCHANGED
    assert "no_planarity_evidence" in result.rejected_counts


def test_geometry_decides_between_a_benzene_and_a_real_diene():
    """Same topology, two geometries, two verdicts.

    A ring with two double bonds is genuinely ambiguous, so the planarity
    evidence has to be what separates a misperceived benzene from a real
    1,3-cyclohexadiene.  If this collapses to one answer, T0 is either
    flattening real dienes or missing the case it was built for.
    """
    verdicts = {}
    for label, pucker in (("flat", 0.0), ("puckered", 0.9)):
        mol = embedded("C1=CC=CC(C(=O)NC)C1")
        ring = list(mol.GetRingInfo().AtomRings()[0])
        if pucker:
            conf = mol.GetConformer()
            for i, idx in enumerate(ring):
                p = conf.GetAtomPosition(idx)
                conf.SetAtomPosition(idx, (p.x, p.y, p.z + (pucker if i % 2 else -pucker)))
        else:
            flatten_ring(mol, ring)
        verdicts[label] = optimize_molecule(
            mol, molecule_id=label, config=config(tiers=["T0"])
        ).status

    assert verdicts["flat"] == OPT_STATUS_OPTIMIZED
    assert verdicts["puckered"] == OPT_STATUS_UNCHANGED


def test_benzene_ring_is_already_done():
    mol = embedded("c1ccccc1C(=O)NC")
    result = optimize_molecule(mol, molecule_id="benzamide", config=config(tiers=["T0"]))
    assert result.status == OPT_STATUS_UNCHANGED
    assert result.transform_chain == ""


# ----------------------------------------------------------------- guardrails


def test_molecule_without_a_conformer_is_rejected():
    mol = Chem.MolFromSmiles("c1ccccc1CCO")
    result = optimize_molecule(mol, molecule_id="flat", config=config())
    assert result.status == OPT_STATUS_REJECTED
    assert result.reject_reason == "no_conformer"


def test_null_input_is_rejected():
    result = optimize_molecule(None, molecule_id="none", config=config())
    assert result.status == OPT_STATUS_REJECTED
    assert result.reject_reason == "null_input"


def test_transform_budget_caps_the_chain():
    mol = embedded(BIG_PROBE)
    result = optimize_molecule(
        mol,
        molecule_id="budget",
        config=config(constraints={"max_transforms_per_molecule": 1}),
    )
    assert len(result.applied) <= 1


def test_disabling_every_tier_changes_nothing():
    mol = embedded("O=[N+]([O-])c1ccccc1")
    result = optimize_molecule(mol, molecule_id="off", config=config(tiers=[]))
    assert result.status == OPT_STATUS_UNCHANGED
    assert result.transform_chain == ""


def test_a_high_reward_threshold_blocks_everything():
    mol = embedded(BIG_PROBE)
    permissive = optimize_molecule(mol, molecule_id="loose", config=config())
    assert permissive.status == OPT_STATUS_OPTIMIZED

    result = optimize_molecule(
        mol, molecule_id="picky", config=config(min_reward_gain=1e6)
    )
    assert result.status == OPT_STATUS_UNCHANGED
    assert result.rejected_counts.get("reward_below_threshold", 0) > 0


def test_disabled_transform_ids_are_respected():
    mol = embedded(BIG_PROBE)
    baseline = optimize_molecule(mol, molecule_id="a", config=config())
    assert baseline.status == OPT_STATUS_OPTIMIZED
    banned = [a.transform_id for a in baseline.applied]
    result = optimize_molecule(
        mol, molecule_id="b", config=config(disabled_transform_ids=banned)
    )
    assert all(a.transform_id not in banned for a in result.applied)


def test_engine_respects_its_time_budget():
    mol = embedded(BIG_PROBE)
    started = time.monotonic()
    optimize_molecule(mol, molecule_id="clock", config=config(time_budget_s=0.001))
    assert time.monotonic() - started < 15.0


# ------------------------------------------------------------------ contracts


def test_optimized_molecule_keeps_a_conformer():
    """Downstream docking has no way to recover a lost pose."""
    mol = embedded(BIG_PROBE)
    result = optimize_molecule(mol, molecule_id="conf", config=config())
    assert result.status == OPT_STATUS_OPTIMIZED
    assert result.optimized_mol.GetNumConformers() > 0
    assert result.optimized_mol.GetConformer().Is3D()


def test_result_reports_before_and_after_properties():
    mol = embedded("C1=CC=CC(C(=O)NC)C1")
    flatten_ring(mol, [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()])
    result = optimize_molecule(mol, molecule_id="props", config=config(tiers=["T0"]))
    assert result.status == OPT_STATUS_OPTIMIZED
    for key in ("qed", "sa", "rotb", "alerts", "heavy_atoms"):
        assert key in result.properties_before
        assert key in result.properties_after
    assert result.properties_after["qed"] > result.properties_before["qed"]


def test_reward_breakdown_sums_to_the_total():
    mol = embedded("C1=CC=CC(C(=O)NC)C1")
    flatten_ring(mol, [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()])
    result = optimize_molecule(mol, molecule_id="reward", config=config(tiers=["T0"]))
    assert result.status == OPT_STATUS_OPTIMIZED
    parts = sum(v for k, v in result.reward_breakdown.items() if k != "total")
    assert parts == pytest.approx(result.total_reward, abs=1e-4)


def test_internal_provenance_props_are_cleaned_off():
    mol = embedded("C1=CC=CC(C(=O)NC)C1")
    flatten_ring(mol, [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()])
    result = optimize_molecule(mol, molecule_id="clean", config=config(tiers=["T0"]))
    assert result.status == OPT_STATUS_OPTIMIZED
    for atom in result.optimized_mol.GetAtoms():
        assert not atom.HasProp("_dd_parent_idx")


# -------------------------------------------------------------------- pickle


def test_custom_props_survive_a_pickle_round_trip():
    """``evaluate_single_molecule_isolated`` forks, so provenance rides along
    on the molecule itself and only survives with AllProps pickling."""
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    mol = embedded("c1ccccc1CCO")
    mol.SetProp("_dd_preopt_smiles", "c1ccccc1CCO")
    mol.SetProp("_dd_opt_transforms", "T0_AROMATIZE_C6_CARBOCYCLE")
    mol.SetProp("_dd_opt_reward", "0.60716")

    restored = pickle.loads(pickle.dumps(mol))

    assert restored.GetProp("_dd_preopt_smiles") == "c1ccccc1CCO"
    assert restored.GetProp("_dd_opt_transforms") == "T0_AROMATIZE_C6_CARBOCYCLE"
    assert restored.GetProp("_dd_opt_reward") == "0.60716"
    assert restored.GetNumConformers() == 1


def test_optimize_result_survives_a_pickle_round_trip():
    mol = embedded("C1=CC=CC(C(=O)NC)C1")
    flatten_ring(mol, [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()])
    result = optimize_molecule(mol, molecule_id="pkl", config=config(tiers=["T0"]))
    restored = pickle.loads(pickle.dumps(result))
    assert restored.molecule_id == result.molecule_id
    assert restored.transform_chain == result.transform_chain
    assert restored.total_reward == result.total_reward
