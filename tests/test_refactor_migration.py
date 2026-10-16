"""Exercise migration on disposable trees, never the real research directory."""

import json
import os
import sqlite3
from pathlib import Path

import pytest

from tools.maintenance.inventory import build_plan, collect_references, paths_in
from tools.maintenance.migration import apply_plan, load_plan, rollback_plan, save_plan


def make_plan(root):
    plan = build_plan(root)
    manifest = root / "local/inventory/plan.json"
    save_plan(plan, manifest, tables=True)
    return manifest


def test_nested_database_paths_protect_only_referenced_subtrees(tmp_path):
    for name in ("active", "historical"):
        target = tmp_path / "runs" / name
        target.mkdir(parents=True)
        (target / "result.pt").write_text(name)
        (target / "related.sdf").write_text("keep whole topic")
    with sqlite3.connect(tmp_path / "diffdynamic.db") as connection:
        connection.execute(
            "CREATE TABLE runs (output_path TEXT, parameters TEXT, config_snapshot TEXT)"
        )
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?)",
            (None, json.dumps({"inputs": [{"pt": "runs/active/result.pt"}]}), None),
        )
        connection.execute("CREATE TABLE history (details TEXT)")
        connection.execute(
            "INSERT INTO history VALUES (?)",
            (json.dumps({"config": "configs/recorded.yml"}),),
        )
    references, notes = collect_references(tmp_path)
    assert not notes
    assert str(tmp_path / "runs/active/result.pt") in references
    assert str(tmp_path / "configs/recorded.yml") in references
    plan = build_plan(tmp_path)
    assert [Path(entry["old_path"]).name for entry in plan["entries"]] == ["historical"]


def test_symlinks_move_without_dereference_and_rollback(tmp_path):
    research = tmp_path / "analysis/topic"
    research.mkdir(parents=True)
    (research / "result.txt").write_text("keep")
    (research / "inside").symlink_to("result.txt")
    outside = tmp_path.parent / (tmp_path.name + "-external.txt")
    outside.write_text("external")
    (research / "outside").symlink_to(os.path.relpath(outside, research))
    (tmp_path / "latest").symlink_to("analysis/topic/result.txt")
    manifest = make_plan(tmp_path)
    apply_plan(manifest)
    moved = tmp_path / "local/research/analysis/topic"
    assert (moved / "inside").is_symlink()
    assert (moved / "outside").read_text() == "external"
    assert (tmp_path / "latest").read_text() == "keep"
    assert not research.exists()
    apply_plan(manifest)  # Idempotent while the manifest remains applied.
    rollback_plan(manifest)
    rollback_plan(manifest)
    assert (research / "outside").is_symlink()
    assert (research / "outside").read_text() == "external"
    assert os.readlink(tmp_path / "latest") == "analysis/topic/result.txt"
    assert outside.read_text() == "external"
    outside.unlink()


def test_conflict_never_overwrites_and_has_failure_journal(tmp_path):
    (tmp_path / "temporary.sh").write_text("original")
    manifest = make_plan(tmp_path)
    destination = tmp_path / "local/archive/temporary.sh"
    destination.parent.mkdir(parents=True)
    destination.write_text("do not overwrite")
    with pytest.raises(RuntimeError, match="partially applied"):
        apply_plan(manifest)
    assert destination.read_text() == "do not overwrite"
    assert (tmp_path / "temporary.sh").read_text() == "original"
    journal = load_plan(manifest)
    assert journal["entries"][0]["status"] == "failed"
    assert journal["events"][-1]["status"] == "failed"


def test_source_change_and_rollback_change_are_detected(tmp_path):
    script = tmp_path / "temporary.sh"
    script.write_text("before")
    manifest = make_plan(tmp_path)
    script.write_text("changed")
    with pytest.raises(RuntimeError):
        apply_plan(manifest)
    assert script.read_text() == "changed"
    other = tmp_path / "local/inventory/second.json"
    save_plan(build_plan(tmp_path), other, tables=True)
    apply_plan(other)
    moved = tmp_path / "local/archive/temporary.sh"
    moved.write_text("new user work")
    with pytest.raises(RuntimeError, match="Rollback needs attention"):
        rollback_plan(other)
    assert moved.read_text() == "new user work"
    assert not script.exists()


def test_reference_paths_are_lexical_and_project_bounded(tmp_path):
    found = set(paths_in({"a": ["runs/example/result.pt"]}, tmp_path))
    assert tmp_path / "runs/example/result.pt" in found
    assert not list(paths_in("/outside/project/file.pt", tmp_path))


def test_production_import_dependency_prevents_archival(tmp_path):
    (tmp_path / "src/diffdynamic").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src/diffdynamic/flow.py").write_text(
        "from scripts.historical import run\n"
    )
    (tmp_path / "scripts/historical.py").write_text("def run(): pass\n")
    assert not build_plan(tmp_path)["entries"]


def test_interrupted_atomic_move_can_be_recovered(tmp_path):
    from tools.maintenance.migration import rename_without_replace

    (tmp_path / "temporary.sh").write_text("preserved")
    manifest = make_plan(tmp_path)
    plan = load_plan(manifest)
    entry = plan["entries"][0]
    entry["status"] = "moving"
    save_plan(plan, manifest)
    rename_without_replace(entry["old_path"], entry["new_path"])
    apply_plan(manifest)
    assert load_plan(manifest)["entries"][0]["status"] == "moved"
    rollback_plan(manifest)
    assert (tmp_path / "temporary.sh").read_text() == "preserved"


def test_common_evaluation_outputs_stay_and_appear_in_maps(tmp_path):
    import csv

    (tmp_path / "evalresults record.xlsx").write_text("summary")
    (tmp_path / "model_comparison_data.xlsx").write_text("comparison")
    (tmp_path / "molecules_score_70-80").mkdir()
    (tmp_path / "molecules_score_70-80/example.sdf").write_text("molecule")
    (tmp_path / "batchsummary").mkdir()
    manifest = make_plan(tmp_path)
    assert not load_plan(manifest)["entries"]
    with manifest.with_suffix(".compact.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    kept = {Path(row["old_path"]).name: row for row in rows}
    for name in (
        "evalresults record.xlsx",
        "model_comparison_data.xlsx",
        "molecules_score_70-80",
        "batchsummary",
    ):
        assert kept[name]["status"] == "retained"
        assert kept[name]["new_path"] == kept[name]["old_path"]
        assert kept[name]["github"] == "local-only"


def test_migration_metadata_is_not_a_running_task_config(tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "historical.py").write_text("pass\n")
    manifest = make_plan(tmp_path)
    monkeypatch.setenv("DD_SAMPLING_YML", str(manifest))
    refs, notes = collect_references(tmp_path)
    assert not notes
    assert str(scripts) not in refs
    apply_plan(manifest)
    assert (tmp_path / "local/archive/scripts/historical.py").exists()


def test_publication_manifests_stay_in_source_tree(tmp_path):
    (tmp_path / "publish-manifest.json").write_text(
        json.dumps(
            {
                "include": ["publish-manifest.json", "weights-manifest.json"],
                "exclude": [],
            }
        )
    )
    (tmp_path / "weights-manifest.json").write_text('{"weights": []}')
    plan = build_plan(tmp_path)
    assert not plan["entries"]
    rows = {Path(row["old_path"]).name: row for row in plan["classifications"]}
    assert rows["publish-manifest.json"]["github"] == "required"
    assert rows["weights-manifest.json"]["github"] == "required"


def test_new_active_reference_retains_actual_location_and_allows_retry(tmp_path, monkeypatch):
    import csv

    research = tmp_path / "analysis/topic"
    research.mkdir(parents=True)
    result = research / "result.pt"
    result.write_text("active research")
    manifest = make_plan(tmp_path)
    monkeypatch.setenv("DD_SAMPLING_YML", str(result))
    apply_plan(manifest)
    plan = load_plan(manifest)
    entry = plan["entries"][0]
    assert entry["status"] == "retained"
    assert entry["new_path"] == entry["old_path"] == str(research)
    assert entry["planned_new_path"] == str(tmp_path / "local/research/analysis/topic")
    assert result.read_text() == "active research"
    assert not Path(entry["planned_new_path"]).exists()
    apply_plan(manifest)
    rollback_plan(manifest)
    assert result.read_text() == "active research"
    with manifest.with_suffix(".csv").open(encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if row["status"] == "retained"]
    assert rows
    assert all(row["old_path"] == row["new_path"] for row in rows)
