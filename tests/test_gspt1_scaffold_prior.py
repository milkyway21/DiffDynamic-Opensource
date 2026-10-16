"""Portable tests for generic scaffold profiles and site allocation."""

from __future__ import annotations

import numpy as np
import torch

from utils.gspt1_scaffold_prior import iter_sdf_molecules
from utils.scaffold_sites import (
    build_extra_atom_positions,
    merge_reference_exit_sites,
)


def test_iter_sdf_molecules_preserves_empty_names_in_multiblock_sdf(tmp_path):
    from rdkit import Chem

    path = tmp_path / "empty_names.sdf"
    writer = Chem.SDWriter(str(path))
    first = Chem.MolFromSmiles("CC")
    first.SetProp("record_id", "first")
    writer.write(first)
    writer.write(Chem.MolFromSmiles("CO"))
    writer.close()

    molecules = list(iter_sdf_molecules(path))

    assert len(molecules) == 2
    assert [Chem.MolToSmiles(molecule) for molecule in molecules] == [
        "CC",
        "CO",
    ]
    assert molecules[0].GetProp("record_id") == "first"


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
