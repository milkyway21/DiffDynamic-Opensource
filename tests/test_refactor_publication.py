"""Publication boundaries must include the app without leaking runtime data."""

import json
from pathlib import Path

import pytest

from tools.maintenance.publishing import export, source_files
from tools.maintenance.weights import ensure_weights


def project(tmp_path):
    paths = [
        "src/diffdynamic/__init__.py",
        "src/diffdynamic/data/graph.py",
        "server/main.py",
        "ui/templates/index.html",
        "ui/templates/landing.html",
        "ui/static/data/landing/reference.sdf",
        "scripts/sample_diffusion.py",
        "weights-manifest.json",
        "server/server_config.json",
        "server/.internal_api_key",
        "local/private.py",
        "outputs/private.pt",
    ]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    (tmp_path / "publish-manifest.json").write_text(
        json.dumps({"include": ["**/*"], "exclude": []})
    )
    return tmp_path


def test_export_preserves_data_package_and_ui_assets(tmp_path):
    root = project(tmp_path / "source")
    files = source_files(root)
    assert "src/diffdynamic/data/graph.py" in files
    assert "ui/static/data/landing/reference.sdf" in files
    assert "server/server_config.json" not in files
    assert "server/.internal_api_key" not in files
    assert not any(p.startswith(("local/", "outputs/")) for p in files)
    candidate = tmp_path / "candidate"
    result = export(root, candidate)
    assert result["git_mutations"] is False
    assert (candidate / "src/diffdynamic/data/graph.py").is_file()
    assert not (candidate / "source").exists()
    with pytest.raises(FileExistsError):
        export(root, candidate)


def test_export_rejects_source_symlinks(tmp_path):
    root = project(tmp_path / "source")
    private = tmp_path / "outside.py"
    private.write_text("fixture")
    (root / "src/diffdynamic/linked.py").symlink_to(private)
    with pytest.raises(ValueError, match="symlinks"):
        source_files(root)


@pytest.mark.parametrize("name", [".env.local", ".env.production", "private.key",
                                  "private.pem", "sessions.sqlite3"])
def test_frontend_private_files_are_never_exported(tmp_path, name):
    root = project(tmp_path / "source")
    private = root / "ui" / name
    private.write_text("fixture")
    assert "ui/" + name not in source_files(root)


def test_weights_never_replace_a_user_checkpoint(tmp_path):
    path = tmp_path / "pretrained_models/custom.pt"
    path.parent.mkdir()
    path.write_bytes(b"user model")
    (tmp_path / "weights-manifest.json").write_text(
        json.dumps(
            {
                "weights": [
                    {
                        "name": "custom.pt",
                        "path": "pretrained_models/custom.pt",
                        "required": True,
                        "sha256": "different",
                        "url": "https://invalid.example/",
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="different checksum"):
        ensure_weights(tmp_path, download=True)
    assert path.read_bytes() == b"user model"


def test_existing_weight_link_can_be_verified_without_writing(tmp_path):
    import hashlib

    external = tmp_path / "shared.pt"
    external.write_bytes(b"shared checkpoint")
    root = tmp_path / "project"
    root.mkdir()
    (root / "pretrained_models").symlink_to(tmp_path)
    (root / "weights-manifest.json").write_text(json.dumps({"weights": [
        {"name": "shared.pt", "path": "pretrained_models/shared.pt",
         "required": True, "sha256": hashlib.sha256(external.read_bytes()).hexdigest()}
    ]}))
    assert ensure_weights(root)[0]["valid"]


@pytest.mark.parametrize("directory", [
    "ablationstudy", "tmp", "tmp_bench", "pocket_quality_vis", "batchsummary",
    "workspace-private", "molecules_score_80plus", "$upload",
])
def test_research_roots_are_blocked_even_with_broad_manifest(tmp_path, directory):
    root = project(tmp_path / "source")
    private = root / directory / "experiment.json"
    private.parent.mkdir()
    private.write_text("local experiment")
    assert directory + "/experiment.json" not in source_files(root)


def test_real_release_contains_medchem_rules_and_portable_tests():
    root = Path(__file__).resolve().parents[1]
    files = source_files(root)
    catalog = root / "rdkit-structure-repair/src/structure_repair/optimize/catalog"
    rules = {path.relative_to(root).as_posix() for path in catalog.glob("*.yaml")}
    assert len(rules) == 6 and rules <= files.keys()
    for name in ("web", "chemistry_pipeline", "docking"):
        assert "tests/test_medchem_" + name + ".py" in files
    assert not any(path.startswith("configs/ablation/") for path in files)
    fixtures = [path for path in files if path.startswith("tests/fixtures/")]
    assert len(fixtures) == 3
    for path in fixtures:
        content = files[path].read_text()
        assert "/home/" not in content and "/data/ye/" not in content
