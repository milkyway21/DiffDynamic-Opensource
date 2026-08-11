"""Tests for trusted, discrete GSPT1 class preparation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from rdkit import Chem

from scripts.gspt1_class_campaign import build_class_config, geometry_preflight
from utils.gspt1_class_setup import (
    CLASS_SPECS,
    F_SCAFFOLD_SMARTS,
    prepare_class_profiles,
)
from utils.gspt1_scaffold_prior import load_scaffold_profile
from utils.scaffold_sites import build_extra_atom_positions


REFERENCE_SDF = Path("/data/ye/sdf/GSPT1.sdf")
NATIVE_SDF = Path(
    "/data/ye/e-drug-lab/Scientist_In_E-Drug-Lab/runs/"
    "molecular_glue_structures_ikzf2_gspt1_20260803_235811/"
    "02_gspt1_or_gstp1_molecular_glue/ligand/"
    "5HXB_85C_C_502_native.sdf"
)
PROTEIN_PDB = Path(
    "/data/ye/e-drug-lab/Scientist_In_E-Drug-Lab/runs/"
    "molecular_glue_structures_ikzf2_gspt1_20260803_235811/"
    "02_gspt1_or_gstp1_molecular_glue/receptor/5HXB_receptor_clean.pdb"
)


def _base_config() -> dict:
    return {
        "sample": {
            "scaffold": {"grow": {}, "murcko_sites": {}},
            "targetdiff_baseline_refine": {},
        }
    }


def test_reference_joint_allocation_uses_only_complete_patterns(tmp_path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({
        "profile_version": 2,
        "profile_kind": "gspt1_class",
        "class_name": "test",
        "n_scaffold": 18,
        "exit_site_weights": {"0": 6, "10": 1},
        "n_extra_values": [14],
        "n_extra_weights": [7],
        "allocation_patterns": [
            {"n_extra": 14, "site_counts": {"0": 14}, "weight": 6},
            {
                "n_extra": 14,
                "site_counts": {"0": 13, "10": 1},
                "weight": 1,
            },
        ],
    }))
    sites = [
        {
            "site_id": 0,
            "profile_slot": 0,
            "anchor_pos": [0.0, 0.0, 0.0],
            "centroid_pos": [1.0, 0.0, 0.0],
        },
        {
            "site_id": 1,
            "profile_slot": 10,
            "anchor_pos": [0.0, 2.0, 0.0],
            "centroid_pos": [1.0, 2.0, 0.0],
        },
    ]
    config = {
        "reference_exit_profile": str(profile_path),
        "site_selection_mode": "reference_joint",
        "per_site_count_mode": "split",
        "jitter_mode": "directional",
        "max_active_sites": 2,
        "max_per_site": 30,
        "overflow_mode": "cap",
    }
    observed = set()
    for seed in range(100):
        positions, meta = build_extra_atom_positions(
            14,
            sites,
            config,
            torch.zeros(3),
            "cpu",
            rng=np.random.default_rng(seed),
        )
        assert positions.shape == (14, 3)
        observed.add(tuple(sorted(
            (record["profile_slot"], record["count"])
            for record in meta["site_allocation"]
        )))
    assert observed == {((0, 14),), ((0, 13), (10, 1))}


def test_pocket_aware_template_pushes_points_out_of_protein():
    sites = [{
        "site_id": 0,
        "profile_slot": 0,
        "anchor_pos": [0.0, 0.0, 0.0],
        "centroid_pos": [1.0, 0.0, 0.0],
        "removed_atom_positions": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    }]
    protein = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    config = {
        "site_selection_mode": "weighted_single",
        "site_budget_mode": "requested",
        "jitter_mode": "pocket_aware_template",
        "jitter_std": 0.0,
        "pocket_min_protein_dist": 1.5,
        "pocket_min_point_dist": 0.8,
        "pocket_max_anchor_dist": 10.0,
        "max_per_site": 10,
    }
    positions, _ = build_extra_atom_positions(
        2,
        sites,
        config,
        torch.zeros(3),
        "cpu",
        rng=np.random.default_rng(3),
        protein_positions=protein,
    )
    distances = torch.linalg.norm(positions - protein[0], dim=1)
    assert float(distances.min()) >= 1.5


def test_class_config_locks_only_original_scaffold():
    spec = CLASS_SPECS[0]
    profile = {
        "n_extra_values": [13, 14],
        "n_extra_weights": [4, 7],
        "allocation_patterns": [
            {"n_extra": 13, "site_counts": {"0": 13}, "weight": 4},
            {"n_extra": 14, "site_counts": {"0": 14}, "weight": 7},
        ],
    }
    config = build_class_config(
        _base_config(), spec, Path("profile.json"), profile,
        seed=123, samples=20,
    )
    scaffold = config["sample"]["scaffold"]
    assert scaffold["fix_scaffold_pos"] is True
    assert scaffold["fix_scaffold_type"] is True
    assert scaffold["grow"]["extra_anchor_strength"] == 0.0
    assert scaffold["grow"]["reference_size_values"] == [13, 14]
    refine = config["sample"]["targetdiff_baseline_refine"]
    assert refine["start_t"] == 19
    assert refine["lock_prefix"] == "types_and_pos"


@pytest.mark.skipif(
    not REFERENCE_SDF.exists() or not NATIVE_SDF.exists(),
    reason="GSPT1 reference files are not installed",
)
def test_trusted_profiles_have_exact_discrete_sizes(tmp_path):
    f_ligand, paths, profiles = prepare_class_profiles(
        REFERENCE_SDF, NATIVE_SDF, tmp_path
    )
    assert f_ligand.exists()
    assert profiles["no_f"]["n_extra_values"] == [13, 14]
    assert profiles["f_main"]["n_extra_values"] == [14, 15, 16]
    assert profiles["f_large"]["n_extra_values"] == [24]
    assert profiles["f_main"]["unsupported_extra_element_counts"] == {
        "Br": 1
    }
    assert profiles["no_f"]["excluded_reference_indices"] == [6, 12]
    assert load_scaffold_profile(paths["f_large"])["allocation_patterns"] == [
        {
            "n_extra": 24,
            "site_counts": {"1": 23, "7": 1},
            "weight": 1.0,
        }
    ]
    supplier = Chem.SDMolSupplier(str(f_ligand), removeHs=False)
    molecule = supplier[0]
    assert molecule is not None
    assert molecule.HasSubstructMatch(Chem.MolFromSmarts(F_SCAFFOLD_SMARTS))


@pytest.mark.skipif(
    not REFERENCE_SDF.exists()
    or not NATIVE_SDF.exists()
    or not PROTEIN_PDB.exists(),
    reason="GSPT1 campaign inputs are not installed",
)
def test_geometry_preflight_is_non_clashing(tmp_path):
    f_ligand, paths, profiles = prepare_class_profiles(
        REFERENCE_SDF, NATIVE_SDF, tmp_path
    )
    for spec in CLASS_SPECS:
        config = build_class_config(
            _base_config(), spec, paths[spec.name], profiles[spec.name],
            seed=123, samples=5,
        )
        config["_profile"] = profiles[spec.name]
        ligand = NATIVE_SDF if spec.ligand_kind == "native" else f_ligand
        result = geometry_preflight(
            spec, ligand, PROTEIN_PDB, config, max_attempts_per_size=128
        )
        assert result["observed_n_extra"] == list(spec.allowed_n_extra)
        assert result["minimum_protein_clearance"] >= 1.5
