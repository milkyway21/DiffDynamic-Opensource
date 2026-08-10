"""Tests for the target-fragment-free GSPT1 scaffold prior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from utils.gspt1_scaffold_prior import build_scaffold_profile, load_scaffold_profile
from utils.scaffold_sites import (
    build_extra_atom_positions,
    extract_reference_exit_vector_sites,
    merge_reference_exit_sites,
)


REFERENCE_SDF = Path("/data/ye/sdf/GSPT1.sdf")
NATIVE_SDF = Path(
    "/data/ye/e-drug-lab/Scientist_In_E-Drug-Lab/runs/"
    "molecular_glue_structures_ikzf2_gspt1_20260803_235811/"
    "02_gspt1_or_gstp1_molecular_glue/ligand/5HXB_85C_C_502_native.sdf"
)
SMARTS = "O=C1CCC(N2Cc3ccccc3C2=O)C(=O)N1"


@pytest.mark.skipif(
    not REFERENCE_SDF.exists() or not NATIVE_SDF.exists(),
    reason="GSPT1 reference files are not installed",
)
def test_profile_contains_only_coarse_exit_and_size_priors(tmp_path):
    profile = build_scaffold_profile(REFERENCE_SDF, NATIVE_SDF, SMARTS)
    assert profile["n_reference_records"] == 17
    assert profile["n_scaffold"] == 18
    assert profile["matched_reference_records"] >= 15
    assert profile["exit_site_weights"]
    assert profile["n_extra_values"]
    assert "canonical_smiles" not in str(profile)

    output = tmp_path / "profile.json"
    from utils.gspt1_scaffold_prior import write_scaffold_profile

    write_scaffold_profile(profile, output)
    loaded = load_scaffold_profile(output)
    assert loaded["n_scaffold"] == 18
    assert loaded["exit_site_weights"] == profile["exit_site_weights"]


@pytest.mark.skipif(
    not REFERENCE_SDF.exists() or not NATIVE_SDF.exists(),
    reason="GSPT1 reference files are not installed",
)
def test_profile_adds_multiple_native_geometry_exits():
    from rdkit import Chem

    mol = Chem.SDMolSupplier(str(NATIVE_SDF), removeHs=False, sanitize=False)[0]
    Chem.SanitizeMol(mol)
    pattern = Chem.MolFromSmarts(SMARTS)
    scaffold_indices = sorted(pattern and mol.GetSubstructMatches(pattern)[0])
    conf = mol.GetConformer()
    positions = np.asarray(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
         for i in range(mol.GetNumAtoms())],
        dtype=np.float64,
    )
    profile = build_scaffold_profile(REFERENCE_SDF, NATIVE_SDF, SMARTS)
    sites = extract_reference_exit_vector_sites(
        mol, scaffold_indices, positions, profile, offset=1.5
    )
    assert len(sites) >= 4
    assert all(site["removed_atom_indices"] == [] for site in sites)
    assert all(site["site_selection_weight"] > 0 for site in sites)


def test_weighted_single_selection_keeps_one_exit_and_requested_count():
    sites = [
        {
            "site_id": 0,
            "anchor_pos": [0.0, 0.0, 0.0],
            "centroid_pos": [1.0, 0.0, 0.0],
            "site_selection_weight": 6.0,
        },
        {
            "site_id": 1,
            "anchor_pos": [0.0, 0.0, 0.0],
            "centroid_pos": [0.0, 1.0, 0.0],
            "site_selection_weight": 1.0,
        },
    ]
    cfg = {
        "site_selection_mode": "weighted_single",
        "site_budget_mode": "requested",
        "n_extra_min_clamp": 14,
        "n_extra_max_clamp": 25,
        "max_per_site": 25,
        "jitter_mode": "directional",
        "overflow_mode": "cap",
    }
    selected = []
    for seed in range(300):
        positions, meta = build_extra_atom_positions(
            14, sites, cfg, torch.zeros(3), "cpu",
            rng=np.random.default_rng(seed),
        )
        assert positions.shape == (14, 3)
        assert len(meta["site_allocation"]) == 1
        selected.append(meta["site_allocation"][0]["site_id"])
    assert selected.count(0) > selected.count(1)


def test_virtual_exit_receives_geometry_only_template():
    source = {
        "site_id": 0,
        "anchor_scaffold_idx": 1,
        "anchor_pos": [0.0, 0.0, 0.0],
        "centroid_pos": [1.0, 0.0, 0.0],
        "removed_atom_positions": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        "removed_atom_count": 2,
        "site_kind": "murcko_sidechain",
        "site_selection_weight": 1.0,
    }
    virtual = {
        "site_id": -1,
        "anchor_scaffold_idx": 2,
        "anchor_pos": [10.0, 10.0, 10.0],
        "centroid_pos": [10.0, 11.0, 10.0],
        "removed_atom_positions": [],
        "removed_atom_count": 0,
        "site_kind": "reference_exit_vector",
        "profile_slot": 2,
        "site_selection_weight": 3.0,
    }
    merged = merge_reference_exit_sites([source], [virtual])
    assert len(merged) == 2
    assert merged[1]["removed_atom_count"] == 2
    assert merged[1]["template_source_site_id"] == 0
    assert merged[1]["removed_atom_positions"][0] == [10.0, 11.0, 10.0]
