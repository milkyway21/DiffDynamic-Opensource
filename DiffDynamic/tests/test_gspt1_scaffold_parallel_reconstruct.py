from scripts.reconstruct_gspt1_scaffold_parallel_job import (
    chunk_ranges,
    make_chunk_payload,
)


def test_chunk_ranges_are_balanced_and_cover_all_items():
    ranges = chunk_ranges(10, 3)
    assert ranges == [(0, 4), (4, 8), (8, 10)]
    assert [item for start, end in ranges for item in range(start, end)] == list(range(10))


def test_chunk_payload_slices_sample_metadata_and_drops_trajectories():
    data = {
        "data": object(),
        "pred_ligand_pos": list(range(5)),
        "pred_ligand_v": list(range(5)),
        "pred_ligand_pos_traj": list(range(5)),
        "time": list(range(5)),
        "meta": {
            "records": [{"idx": i} for i in range(5)],
            "scaffold_cfg": {"n_scaffold_atoms": 18},
        },
        "extra_info": {"data_id": 0},
    }
    chunk = make_chunk_payload(data, 2, 5, 5)
    assert chunk["pred_ligand_pos"] == [2, 3, 4]
    assert chunk["pred_ligand_v"] == [2, 3, 4]
    assert "pred_ligand_pos_traj" not in chunk
    assert "time" not in chunk
    assert chunk["meta"]["records"] == [{"idx": 2}, {"idx": 3}, {"idx": 4}]
    assert chunk["meta"]["scaffold_cfg"] == {"n_scaffold_atoms": 18}
    assert chunk["extra_info"]["parallel_chunk"] == {"start": 2, "end": 5}
