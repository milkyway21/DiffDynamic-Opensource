"""Tests for trusted, discrete GSPT1 class preparation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from rdkit import Chem

from models.molopt_score_model import ScorePosNet3D
from scripts.gspt1_class_campaign import (
    _parse_count_values,
    _parse_geometry_modes,
    _restrict_profile_to_counts,
    build_class_config,
    geometry_preflight,
)
from scripts.sample_diffusion import _sample_reference_extra_type_quota
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
    assert scaffold["grow"]["extra_type_anchor_strength"] == 0.2
    assert scaffold["grow"]["start_t"] == 999
    assert scaffold["grow"]["forward_noise_init"] is True
    assert scaffold["grow"]["reference_size_values"] == [13, 14]
    assert scaffold["murcko_sites"]["reference_extra_type_prior_mode"] == (
        "quota_random"
    )
    dynamic_refine = config["sample"]["dynamic"]["refine"]
    assert dynamic_refine["max_grad_fusion_iterations"] == 30
    assert dynamic_refine["preserve_schedule_endpoint"] is True
    refine = config["sample"]["targetdiff_baseline_refine"]
    assert refine["start_t"] == 29
    assert refine["lock_prefix"] == "types_and_pos"


def test_strict_fragment_geometry_is_opt_in_and_scaffold_only():
    spec = CLASS_SPECS[0]
    profile = {
        "n_extra_values": [14],
        "n_extra_weights": [1],
        "allocation_patterns": [{
            "n_extra": 14,
            "site_counts": {"0": 14},
            "weight": 1,
        }],
    }
    config = build_class_config(
        _base_config(), spec, Path("profile.json"), profile,
        seed=123, samples=20, geometry_mode="strict_fragment_gaussian",
    )
    sites = config["sample"]["scaffold"]["murcko_sites"]
    assert sites["strict_fragment_gaussian"] is True
    assert sites["strict_anchor_gaussian"] is False
    assert sites["fragment_prior_profile"] == "profile.json"
    assert config["sample"]["scaffold"]["fix_scaffold_pos"] is True

    default = build_class_config(
        _base_config(), spec, Path("profile.json"), profile,
        seed=123, samples=20,
    )
    default_sites = default["sample"]["scaffold"]["murcko_sites"]
    assert default_sites["strict_fragment_gaussian"] is False


def test_soft_fragment_geometry_keeps_added_types_free():
    spec = CLASS_SPECS[0]
    profile = {
        "n_extra_values": [14],
        "n_extra_weights": [1],
        "allocation_patterns": [{
            "n_extra": 14,
            "site_counts": {"0": 14},
            "weight": 1,
        }],
    }
    config = build_class_config(
        _base_config(), spec, Path("profile.json"), profile,
        seed=123, samples=20, geometry_mode="fragment_gaussian",
    )
    scaffold = config["sample"]["scaffold"]
    sites = scaffold["murcko_sites"]
    assert sites["fragment_gaussian"] is True
    assert sites["strict_fragment_gaussian"] is False
    assert sites["fragment_prior_profile"] == "profile.json"
    assert scaffold["extra_atom_type_mode"] == "diffuse"
    assert scaffold["lock_extra_atom_types"] is False
    assert scaffold["grow"]["extra_type_anchor_strength"] == 0.0


def test_geometry_modes_support_per_class_overrides():
    modes = _parse_geometry_modes(
        "no_f=pocket_aware_template,f_main=strict_fragment_gaussian",
        "strict_anchor_gaussian",
    )
    assert modes == {
        "no_f": "pocket_aware_template",
        "f_main": "strict_fragment_gaussian",
        "f_large": "strict_anchor_gaussian",
    }

    with pytest.raises(ValueError, match="duplicate"):
        _parse_geometry_modes(
            "no_f=pocket_aware_template,no_f=strict_anchor_gaussian",
            "pocket_aware_template",
        )


def test_count_focus_filters_all_size_specific_priors():
    profile = {
        "n_extra_values": [13, 14],
        "n_extra_weights": [4, 7],
        "exit_site_weights": {"0": 11.0, "10": 1.0},
        "allocation_patterns": [
            {"n_extra": 13, "site_counts": {"0": 13}, "weight": 4},
            {
                "n_extra": 14,
                "site_counts": {"0": 13, "10": 1},
                "weight": 1,
            },
        ],
        "extra_type_patterns": [
            {
                "n_extra": 13,
                "class_counts": {"C|0": 2, "N|0": 1},
                "weight": 4,
            },
            {
                "n_extra": 14,
                "class_counts": {"C|0": 3, "O|0": 1},
                "weight": 7,
            },
        ],
        "fragment_patterns": [
            {"n_extra": 13, "site_fragments": {}, "weight": 4},
            {"n_extra": 14, "site_fragments": {}, "weight": 7},
        ],
    }
    focused = _restrict_profile_to_counts(profile, (14,))
    assert focused["n_extra_values"] == [14]
    assert focused["n_extra_weights"] == [7.0]
    assert [item["n_extra"] for item in focused["allocation_patterns"]] == [14]
    assert [item["n_extra"] for item in focused["extra_type_patterns"]] == [14]
    assert [item["n_extra"] for item in focused["fragment_patterns"]] == [14]
    assert focused["reference_extra_element_counts"] == {"C": 21.0, "O": 7.0}
    assert focused["exit_site_weights"] == {"0": 13.0, "10": 1.0}
    assert _parse_count_values("14, 14") == (14,)


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
    assert profiles["f_main"]["n_extra_values"] == [14, 16]
    assert profiles["f_large"]["n_extra_values"] == [24]
    assert profiles["f_main"]["unsupported_extra_element_counts"] == {}
    assert profiles["no_f"]["excluded_reference_indices"] == [6, 12, 15]
    assert profiles["f_main"]["included_reference_indices"] == [14, 16]
    assert load_scaffold_profile(paths["f_large"])["allocation_patterns"] == [
        {
            "n_extra": 24,
            "site_counts": {"1": 23, "7": 1},
            "weight": 1.0,
        }
    ]
    assert load_scaffold_profile(paths["f_large"])["extra_type_patterns"] == [
        {
            "n_extra": 24,
            "class_counts": {
                "C|0": 12,
                "C|1": 6,
                "F|0": 2,
                "N|0": 1,
                "O|0": 3,
            },
            "weight": 1.0,
        }
    ]
    supplier = Chem.SDMolSupplier(str(f_ligand), removeHs=False)
    molecule = supplier[0]
    assert molecule is not None
    assert molecule.HasSubstructMatch(Chem.MolFromSmarts(F_SCAFFOLD_SMARTS))


def test_quota_type_prior_preserves_counts_but_not_reference_order():
    profile = {
        "extra_type_patterns": [{
            "n_extra": 8,
            "class_counts": {"C|0": 3, "C|1": 2, "N|0": 1, "O|0": 2},
            "weight": 1.0,
        }]
    }
    first = _sample_reference_extra_type_quota(
        profile, "add_aromatic", 13, 8, "cpu", np.random.default_rng(2)
    ).argmax(dim=-1)
    second = _sample_reference_extra_type_quota(
        profile, "add_aromatic", 13, 8, "cpu", np.random.default_rng(3)
    ).argmax(dim=-1)
    assert torch.bincount(first, minlength=13).tolist()[:6] == [0, 3, 2, 1, 0, 2]
    assert not torch.equal(first, second)


def test_refine_cap_can_preserve_low_noise_endpoint():
    schedule = list(range(650, -1, -5))
    historical = ScorePosNet3D._truncate_schedule_to_grad_fusion_iterations(
        schedule, 30
    )
    truncated = ScorePosNet3D._truncate_schedule_to_grad_fusion_iterations(
        schedule, 30, preserve_endpoint=True
    )
    assert historical == schedule[:30]
    assert historical[-1] != 0
    assert len(truncated) == 30
    assert truncated[0] == 650
    assert truncated[-1] == 0


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
