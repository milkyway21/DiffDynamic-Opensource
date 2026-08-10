"""Unit tests: the gates that keep an aggressive optimizer from drifting away.

The repair layer's ``identity_ok`` is deliberately absent here, so these gates
are the only thing standing between "optimized analogue" and "a different
molecule wearing the same molecule_id".
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from structure_repair.optimize.constraints import (
    ConstraintConfig,
    check_candidate,
    murcko_smiles,
    tanimoto,
)

PARENT = "O=C(O)Cc1ccc(OC)cc1"  # 4-methoxyphenylacetic acid


def mol(smiles):
    m = Chem.MolFromSmiles(smiles)
    assert m is not None, smiles
    return m


def identity_map(m):
    return {i: i for i in range(m.GetNumAtoms())}


def test_identical_molecule_passes():
    parent = mol(PARENT)
    assert check_candidate(parent, mol(PARENT), "T1", identity_map(parent)) is None


def test_small_bioisostere_passes():
    parent = mol(PARENT)
    candidate = mol("O=C(O)Cc1ccc(OC(F)F)cc1")  # OMe -> OCHF2
    assert check_candidate(parent, candidate, "T3", identity_map(parent)) is None


# ---------------------------------------------------------------- rejections


def test_unrelated_molecule_is_rejected_by_similarity():
    parent = mol(PARENT)
    candidate = mol("CCCCCCCCCCCCCCCC")
    assert check_candidate(parent, candidate, "T1") == "tanimoto_below_floor"


def test_t0_is_exempt_from_the_similarity_floor():
    """Aromatizing changes the fingerprint a lot but not the heavy-atom graph."""
    parent = mol("C1=CC=CC(C(=O)N)C1")
    candidate = mol("c1ccccc1C(=O)N")
    cfg = ConstraintConfig()
    assert tanimoto(parent, candidate) < cfg.min_tanimoto_to_parent
    assert check_candidate(parent, candidate, "T0", cfg=cfg) is None


def test_growing_too_many_heavy_atoms_is_rejected():
    parent = mol(PARENT)
    candidate = mol("O=C(O)Cc1ccc(OCCCCCCCCCC)cc1")
    assert check_candidate(parent, candidate, "T2") == "heavy_atom_gain_too_large"


def test_losing_too_many_heavy_atoms_is_rejected():
    parent = mol(PARENT)
    candidate = mol("c1ccccc1")
    assert check_candidate(parent, candidate, "T5") == "heavy_atom_loss_too_large"


def test_disallowed_element_is_rejected():
    parent = mol("CCc1ccccc1")
    candidate = mol("CC[Si](C)(C)c1ccccc1")
    reason = check_candidate(parent, candidate, "T1")
    assert reason == "disallowed_element_Si"


def test_fragmented_candidate_is_rejected():
    parent = mol(PARENT)
    candidate = mol("O=C(O)Cc1ccccc1.CO")
    assert check_candidate(parent, candidate, "T1") == "fragmented"


def test_transform_budget_is_enforced():
    parent = mol(PARENT)
    cfg = ConstraintConfig()
    reason = check_candidate(
        parent,
        mol(PARENT),
        "T1",
        n_transforms_applied=cfg.max_transforms_per_molecule,
        cfg=cfg,
    )
    assert reason == "transform_budget_exhausted"


def test_ring_loss_is_rejected_under_ring_systems_mode():
    parent = mol("c1ccc2ccccc2c1CCN")
    candidate = mol("c1ccccc1CCN")
    reason = check_candidate(parent, candidate, "T1", cfg=ConstraintConfig())
    # Loses both a ring and too much mass; either verdict blocks it.
    assert reason in {"ring_count_decreased", "heavy_atom_loss_too_large"}


def test_t2_ring_closure_survives_the_similarity_floor():
    """Closing a ring rewrites most Morgan environments without losing atoms.

    Without the T2 exemption the whole conformational-locking tier would be
    unreachable on anything smaller than ~28 heavy atoms.
    """
    parent = mol("OCCCCc1ccc(C(=O)NCc2ccccc2)cc1")
    candidate = mol("C1CCC(O1)c1ccc(C(=O)NCc2ccccc2)cc1")
    cfg = ConstraintConfig()
    assert tanimoto(parent, candidate) < cfg.min_tanimoto_to_parent
    assert check_candidate(parent, candidate, "T2", cfg=cfg) is None
    # The same product would be rejected if it claimed to be a T1 bioisostere.
    assert check_candidate(parent, candidate, "T1", cfg=cfg) == "tanimoto_below_floor"


# ------------------------------------------------------------- MW behaviour


def test_pushing_a_compliant_parent_over_the_window_is_rejected():
    cfg = ConstraintConfig(mw_window=(150.0, 200.0))
    parent = mol(PARENT)
    candidate = mol("O=C(O)Cc1ccc(OCCCC)cc1")
    assert check_candidate(parent, candidate, "T2", cfg=cfg) == "mw_above_window"


def test_a_parent_already_below_the_window_is_not_penalised_further():
    """Small reconstructed molecules must stay optimizable."""
    cfg = ConstraintConfig(mw_window=(150.0, 600.0))
    parent = mol("C1=CCC=CC1")  # 82 Da, already under the floor
    candidate = mol("c1ccccc1")
    assert check_candidate(parent, candidate, "T0", cfg=cfg) is None


def test_stereo_flip_on_a_surviving_atom_is_rejected():
    parent = mol("C[C@H](N)C(=O)O")
    candidate = mol("C[C@@H](N)C(=O)O")
    reason = check_candidate(
        parent, candidate, "T1", parent_index_map=identity_map(parent)
    )
    assert reason == "stereochemistry_flipped"


def test_murcko_helper_reports_the_ring_system():
    assert murcko_smiles(mol("O=C(O)Cc1ccccc1")) == "c1ccccc1"
    assert murcko_smiles(mol("CCCC")) == ""


@pytest.mark.parametrize(
    "mode,expect_none",
    [("off", True), ("strict", False)],
)
def test_scaffold_mode_switches_strictness(mode, expect_none):
    """An aza-scan keeps the ring count but changes the scaffold SMILES."""
    parent = mol("c1ccccc1CC(=O)N")
    candidate = mol("c1ccncc1CC(=O)N")
    cfg = ConstraintConfig(scaffold_mode=mode, scaffold_exempt_tiers=())
    reason = check_candidate(parent, candidate, "T1", cfg=cfg)
    assert (reason is None) is expect_none
