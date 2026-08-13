import os
from pathlib import Path

from scripts.gspt1_unified_reconstruct import (
    Candidate,
    _is_canonical_molecule_path,
    _audit_molecule_paths,
    discover_candidates,
    job_context,
    migrate_corpus,
    select_candidates,
)


def test_audit_only_accepts_source_level_molecules_directory(tmp_path):
    unified_root = tmp_path / "gspt1_unified"
    canonical = (
        unified_root
        / "reconstructed"
        / "batch"
        / "campaign"
        / "job_0000"
        / "src_0000"
        / "molecules"
        / "molecule.sdf"
    )
    evaluation = (
        unified_root
        / "reconstructed"
        / "batch"
        / "campaign"
        / "job_0000"
        / "src_0000"
        / "chunks"
        / "chunk_0000"
        / "attempt"
        / "evaluation"
        / "molecules"
        / "molecule.sdf"
    )
    canonical.parent.mkdir(parents=True)
    evaluation.parent.mkdir(parents=True)
    canonical.touch()
    evaluation.touch()

    assert _is_canonical_molecule_path(canonical, unified_root)
    assert not _is_canonical_molecule_path(evaluation, unified_root)


def test_audit_manifest_deduplicates_reconstructed_retries(tmp_path):
    unified_root = tmp_path / "gspt1_unified"
    molecules_dir = (
        unified_root
        / "reconstructed"
        / "batch"
        / "campaign"
        / "job_0000"
        / "src_0000"
        / "molecules"
    )
    molecules_dir.mkdir(parents=True)
    old = molecules_dir / "chunk0000_000000_old.sdf"
    new = molecules_dir / "chunk0000_000000_new.sdf"
    old.write_text("old", encoding="utf-8")
    new.write_text("new", encoding="utf-8")
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))

    source_rows = {
        "src_0000": {"molecules_dir": str(molecules_dir)},
    }
    manifest = unified_root / "manifest"
    manifest.mkdir(parents=True)
    (manifest / "chunk_reconstruction_manifest.jsonl").write_text(
        '{"source_id":"src_0000","chunk_id":0,"status":"success",'
        '"molecule_file_count":1}\n',
        encoding="utf-8",
    )

    paths = _audit_molecule_paths(unified_root, source_rows)
    assert paths == [new]


def test_job_context_handles_extract_and_jobs_layout(tmp_path):
    data_root = tmp_path / "outputs"
    extract_path = (
        data_root
        / "batch_a"
        / "rounds"
        / "no_f"
        / "run_0000"
        / "gspt1"
        / "extract"
        / "job_0007"
        / "eval"
        / "eval_results_foo_final_bar.pt"
    )
    extract_path.parent.mkdir(parents=True)
    extract_path.touch()
    context = job_context(extract_path, data_root)
    assert context["job"] == "job_0007"
    assert context["job_key"].endswith("run_0000/job_0007")
    assert context["class_name"] == "no_f"

    jobs_path = (
        data_root
        / "batch_a"
        / "rounds"
        / "no_f"
        / "run_0000"
        / "gspt1"
        / "jobs"
        / "job_0007"
        / "run"
        / "result_custom.pt"
    )
    jobs_path.parent.mkdir(parents=True)
    jobs_path.touch()
    assert job_context(jobs_path, data_root)["job"] == "job_0007"


def test_result_pt_has_priority_over_eval_final(tmp_path):
    data_root = tmp_path / "outputs"
    result = (
        data_root
        / "batch"
        / "gspt1"
        / "jobs"
        / "job_0000"
        / "run"
        / "result_custom.pt"
    )
    fallback = (
        data_root
        / "batch"
        / "gspt1"
        / "extract"
        / "job_0000"
        / "eval"
        / "eval_results_final.pt"
    )
    fallback.parent.mkdir(parents=True)
    result.parent.mkdir(parents=True)
    result.touch()
    fallback.touch()

    candidates = discover_candidates(data_root)
    selected = select_candidates(candidates)
    assert len(selected) == 1
    assert selected[0].source_kind == "result"
    assert selected[0].selection_reason == "preferred_result_pt"


def test_chunk_inputs_are_selected_when_result_is_missing(tmp_path):
    data_root = tmp_path / "outputs"
    for index in range(2):
        path = (
            data_root
            / "batch"
            / "gspt1"
            / "extract"
            / "job_0000"
            / ".parallel_chunks"
            / "run_0000"
            / f"chunk_{index:04d}"
            / "input.pt"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    candidates = discover_candidates(data_root)
    selected = select_candidates(candidates)
    assert len(selected) == 2
    assert all(item.source_kind == "chunk_input" for item in selected)


def test_eval_final_snapshot_is_not_reconstructible(tmp_path):
    data_root = tmp_path / "outputs"
    path = (
        data_root
        / "batch"
        / "gspt1"
        / "extract"
        / "job_0000"
        / "eval"
        / "eval_results_foo_final_bar.pt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()

    candidates = discover_candidates(data_root)
    selected = select_candidates(candidates)
    assert len(candidates) == 1
    assert not selected
    assert candidates[0].selection_reason == "no_reconstructible_pt"


def test_migration_moves_data_and_leaves_symlink_entrypoints(tmp_path):
    data_root = tmp_path / "outputs"
    unified_root = data_root / "gspt1_unified"
    source = (
        data_root
        / "batch"
        / "gspt1"
        / "jobs"
        / "job_0000"
        / "run"
        / "result_custom.pt"
    )
    report = data_root / "batch" / "report.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"pt")
    report.write_text("report", encoding="utf-8")

    candidates = discover_candidates(data_root)
    selected = select_candidates(candidates)
    rows = migrate_corpus(data_root, unified_root, candidates, selected)

    assert len(rows) == 1
    raw_path = Path(rows[0]["raw_path"])
    assert raw_path.read_bytes() == b"pt"
    assert (data_root / "batch").is_symlink()
    assert (unified_root / "legacy" / "batch" / "report.json").exists()
    assert (
        unified_root
        / "legacy"
        / "batch"
        / "gspt1"
        / "jobs"
        / "job_0000"
        / "run"
        / "result_custom.pt"
    ).is_symlink()
