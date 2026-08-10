"""Unit tests: the T0-T5 transform catalog is well formed and actually fires.

Every entry is data, so the catalog can rot silently: a typo in a SMARTS string
produces a reaction that simply never matches, and a wrong ``delta_heavy`` lets
a transform slip past the size gate.  These tests fail loudly in both cases.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from structure_repair.optimize.catalog import (
    catalog_dir,
    load_catalog,
    load_tier_policy,
    validate_catalog,
)
from structure_repair.optimize.transforms import enumerate_candidates, prepare_parent

TIERS = ("T1", "T2", "T3", "T4", "T5")


def test_catalog_loads_and_validates():
    report = validate_catalog()
    assert report["bad_smarts"] == [], f"unusable SMARTS: {report['bad_smarts']}"
    assert report["duplicate_id"] == [], f"duplicate ids: {report['duplicate_id']}"
    assert report["ok"], "catalog validated nothing at all"


def test_every_tier_file_is_present():
    root = catalog_dir()
    for tier in TIERS:
        matches = list(root.glob(f"{tier.lower()}_*.yaml"))
        assert matches, f"no catalog file for tier {tier}"
    assert (root / "t0_aromatize.yaml").is_file()


def test_t0_policy_has_ring_rules():
    policy = load_tier_policy("T0")
    assert policy.get("c6_carbocycle")
    assert policy.get("c5_heterocycle")


@pytest.mark.parametrize("tier", TIERS)
def test_tier_has_enabled_transforms(tier):
    specs = load_catalog(enabled_tiers=[tier])
    assert specs, f"tier {tier} loaded no transforms"
    for spec in specs:
        assert spec.tier == tier
        assert spec.reaction is not None, f"{spec.transform_id} has an unusable SMARTS"
        assert spec.rationale, f"{spec.transform_id} has no rationale"


def test_empty_tier_list_loads_nothing():
    """``[]`` means no tier, not every tier.

    The engine strips T0 (it is programmatic, not SMARTS) before calling here,
    so a T0-only run arrives with an empty list.  Reading that as "unrestricted"
    would silently run all six tiers in a single-tier ablation.
    """
    assert load_catalog(enabled_tiers=[]) == []
    assert load_catalog(enabled_tiers=None), "None must still mean every tier"


def test_disabled_transform_ids_are_honoured():
    specs = load_catalog(enabled_tiers=["T1"])
    victim = specs[0].transform_id
    remaining = load_catalog(enabled_tiers=["T1"], disabled_transform_ids=[victim])
    assert victim not in {s.transform_id for s in remaining}


# Probe molecules chosen so each transform has exactly the group it targets.
TIER_PROBES = {
    "T1": "CCOC(=O)c1ccccc1",          # ethyl benzoate: ester, aryl H
    "T2": "OCCCCc1ccccc1",             # flexible chain with a hydroxyl
    "T3": "O=[N+]([O-])c1ccccc1",      # nitrobenzene: classic structural alert
    "T4": "Fc1ccc(C(=O)N)cc1",         # aryl fluoride for polar-group swaps
    "T5": "CC(C)(C)c1ccccc1",          # tert-butyl to simplify
}


@pytest.mark.parametrize("tier,smiles", sorted(TIER_PROBES.items()))
def test_tier_produces_candidates_on_a_matching_probe(tier, smiles):
    parent = prepare_parent(Chem.MolFromSmiles(smiles))
    assert parent is not None
    specs = load_catalog(enabled_tiers=[tier])
    candidates = enumerate_candidates(parent, specs, max_products_per_transform=4)
    assert candidates, f"tier {tier} fired nothing on {smiles}"
    for cand in candidates:
        assert cand.tier == tier
        assert cand.mol is not None
        assert cand.smiles


@pytest.mark.parametrize("tier", TIERS)
def test_declared_delta_heavy_matches_reality(tier):
    """A wrong ``delta_heavy`` would silently mis-budget the size gate."""
    specs = load_catalog(enabled_tiers=[tier])
    probe_smiles = list(TIER_PROBES.values()) + [
        "CC(=O)Nc1ccccc1O",
        "NCCCCC(=O)O",
        "COc1ccc(CC(N)=O)cc1",
        "O=C(O)CCc1ccccc1",
        "CSCc1ccccc1",
    ]
    checked = 0
    for smiles in probe_smiles:
        parent = prepare_parent(Chem.MolFromSmiles(smiles))
        if parent is None:
            continue
        parent_heavy = sum(1 for a in parent.GetAtoms() if a.GetAtomicNum() > 1)
        for cand in enumerate_candidates(parent, specs, max_products_per_transform=2):
            cand_heavy = sum(1 for a in cand.mol.GetAtoms() if a.GetAtomicNum() > 1)
            assert cand_heavy - parent_heavy == cand.delta_heavy, (
                f"{cand.transform_id} declares delta_heavy={cand.delta_heavy} "
                f"but changed {parent_heavy} -> {cand_heavy} on {smiles}"
            )
            checked += 1
    assert checked, f"tier {tier} never fired on any probe"
