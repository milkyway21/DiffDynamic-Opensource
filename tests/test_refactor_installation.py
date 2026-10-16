"""Offline wheel/editable installation contracts using disposable environments."""

import json
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

import _diffdynamic_bootstrap  # noqa: F401
from diffdynamic.io import paths

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_RESOURCES = {
    "configs/sampling.yml",
    "configs/benchmark_reference.yml",
    "ui/templates/index.html",
    "ui/templates/landing.html",
    "ui/content/overview.zh.md",
    "ui/content/overview.en.md",
    "ui/static/js/app.js",
    "ui/static/js/workbench/runtime.js",
    "ui/static/js/workbench/medchem.js",
    "ui/static/css/medchem.css",
    "ui/static/js/landing/app.js",
    "ui/static/libs/three/three.module.js",
    "ui/static/vendor/alpinejs.min.js",
    "ui/static/vendor/3Dmol-min.js",
    "server/pt_sandbox_worker.py",
    "server/pt-loader-requirements.txt",
    "server/server_config.example.json",
    "scripts/sample_diffusion.py",
    "scripts/sample_for_pocket.py",
    "scripts/setup_pt_loader.sh",
    "evaluate_pt_with_correct_reconstruct.py",
    "evaluate_pocket_quality.py",
    "batch_sampleandeval_parallel.py",
    "extract_pt_to_sdf_excel.py",
    "scaffold_cascade_pipeline.py",
    "start_server.sh",
    "rdkit-structure-repair/src/structure_repair/repair_engine.py",
    "rdkit-structure-repair/src/structure_repair/optimize/optimize_engine.py",
    "rdkit-structure-repair/configs/conservative.yaml",
    "rdkit-structure-repair/configs/medchem_optimize.yaml",
    "rdkit-structure-repair/src/structure_repair/optimize/catalog/t0_aromatize.yaml",
    "rdkit-structure-repair/src/structure_repair/optimize/catalog/t1_bioisostere.yaml",
    "rdkit-structure-repair/src/structure_repair/optimize/catalog/t2_conformational_lock.yaml",
    "rdkit-structure-repair/src/structure_repair/optimize/catalog/t3_metabolic_blocking.yaml",
    "rdkit-structure-repair/src/structure_repair/optimize/catalog/t4_property_tuning.yaml",
    "rdkit-structure-repair/src/structure_repair/optimize/catalog/t5_synthetic_accessibility.yaml",
}
COMPATIBILITY_FILES = {
    "_diffdynamic_bootstrap.py",
    "datasets/__init__.py",
    "datasets/pl_data.py",
    "models/molopt_score_model.py",
    "models/property_pred/prop_model.py",
    "utils/data.py",
    "utils/transforms.py",
    "utils/evaluation/scoring_func.py",
}


def _environment(temporary):
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("DD_") and key not in ("PYTHONPATH", "PYTHONHOME")
    }
    environment.update(
        DD_BUILD_BASE=str(temporary / "build"),
        DD_BUILD_METADATA_DIR=str(temporary / "metadata"),
        DD_DB_PATH=str(temporary / "isolated.db"),
        DD_INTERNAL_API_KEY="disposable-installation-test-key",
        DD_AUTH_MODE="off",
        DD_CLEANUP_ON_START="0",
        PYTHONDONTWRITEBYTECODE="1",
        PIP_NO_INDEX="1",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        MPLBACKEND="Agg",
    )
    return environment


def _run(command, cwd, environment):
    result = subprocess.run(
        [str(item) for item in command],
        cwd=str(cwd),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout[-12000:]
    return result.stdout


@pytest.fixture(scope="module")
def distributions(tmp_path_factory):
    temporary = tmp_path_factory.mktemp("diffdynamic-distribution")
    environment = _environment(temporary)
    (temporary / "metadata").mkdir()
    _run(
        [
            sys.executable,
            "setup.py",
            "egg_info",
            "--egg-base",
            temporary / "metadata",
            "sdist",
            "--dist-dir",
            temporary / "dist",
            "bdist_wheel",
            "--dist-dir",
            temporary / "dist",
        ],
        ROOT,
        environment,
    )
    return {
        "temporary": temporary,
        "environment": environment,
        "wheel": next((temporary / "dist").glob("*.whl")),
        "sdist": next((temporary / "dist").glob("*.tar.gz")),
    }


@pytest.fixture(scope="module", params=["wheel", "editable"])
def installed(request, distributions):
    kind = request.param
    temporary = distributions["temporary"] / kind
    temporary.mkdir()
    outside = temporary / "unrelated-working-directory"
    outside.mkdir()
    environment = _environment(temporary)
    virtualenv = temporary / "environment"
    _run(
        [
            sys.executable,
            "-m",
            "venv",
            "--system-site-packages",
            "--without-pip",
            virtualenv,
        ],
        outside,
        environment,
    )
    binary = virtualenv / ("Scripts" if os.name == "nt" else "bin")
    python = binary / ("python.exe" if os.name == "nt" else "python")
    arguments = [
        python,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-index",
        "--no-build-isolation",
    ]
    arguments += (
        ["--editable", ROOT] if kind == "editable" else [distributions["wheel"]]
    )
    _run(arguments, outside, environment)
    output = _run(
        [
            python,
            "-c",
            "from diffdynamic.io.paths import project_root; print(project_root())",
        ],
        outside,
        environment,
    )
    return {
        "kind": kind,
        "python": python,
        "binary": binary,
        "outside": outside,
        "environment": environment,
        "root": Path(output.strip().splitlines()[-1]),
    }


def test_wheel_and_source_archives_include_only_public_code_and_resources(
    distributions,
):
    with zipfile.ZipFile(distributions["wheel"]) as archive:
        wheel_members = set(archive.namelist())
        assert PUBLIC_RESOURCES | COMPATIBILITY_FILES <= wheel_members
        metadata = next(
            name for name in wheel_members if name.endswith(".dist-info/METADATA")
        )
        text = archive.read(metadata).decode()
        assert "Requires-Python: >=3.8" in text
        assert "Requires-Dist:" not in text
    with tarfile.open(distributions["sdist"]) as archive:
        source_members = {
            name.split("/", 1)[1] for name in archive.getnames() if "/" in name
        }
        assert PUBLIC_RESOURCES | COMPATIBILITY_FILES <= source_members
    for members in (wheel_members, source_members):
        assert "server/server_config.json" not in members
        assert "server/.internal_api_key" not in members
        assert "utils/gspt1_class_setup.py" not in members
        assert not any(
            name.startswith(
                (
                    "local/",
                    "outputs/",
                    "uploads/",
                    "pretrained_models/",
                    "runs/",
                    "analysis/",
                    ".cache/",
                )
            )
            or "__pycache__" in name
            or name.endswith((".pyc", ".db", ".pt"))
            for name in members
        )


def test_project_root_honors_explicit_override_from_any_directory(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DD_ROOT", str(tmp_path / "configured-resources"))
    assert paths.project_root() == tmp_path / "configured-resources"
    assert paths.project_path("configs", "sampling.yml") == (
        tmp_path / "configured-resources/configs/sampling.yml"
    )


def test_default_root_detects_the_complete_checkout(monkeypatch, tmp_path):
    monkeypatch.delenv("DD_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert paths.project_root() == ROOT


@pytest.mark.installation
def test_installed_imports_legacy_pickles_and_resource_roots(installed):
    code = r"""
import json, pickle
from pathlib import Path
from diffdynamic.io.paths import project_root
from diffdynamic.data.graph import ProteinLigandData
import diffdynamic.data.graph as canonical_graph
from diffdynamic.models.score_model import ScorePosNet3D
from diffdynamic.data.transforms import FeaturizeLigandAtom
from diffdynamic.data.protein import PDBProtein
import server.config as config
import datasets.pl_data as graph
import models.molopt_score_model as model
import utils.transforms as transforms
import utils.data as protein
assert graph.ProteinLigandData is ProteinLigandData
assert model.ScorePosNet3D is ScorePosNet3D
assert transforms.FeaturizeLigandAtom is FeaturizeLigandAtom
assert protein.PDBProtein is PDBProtein
for module, name, expected in (
    ('datasets.pl_data', 'ProteinLigandData', ProteinLigandData),
    ('models.molopt_score_model', 'ScorePosNet3D', ScorePosNet3D),
    ('utils.transforms', 'FeaturizeLigandAtom', FeaturizeLigandAtom),
    ('utils.data', 'PDBProtein', PDBProtein),
):
    assert pickle.loads(('c' + module + '\n' + name + '\n.').encode()) is expected
root = project_root()
cfg = config.ServerConfig()
assert cfg.diffdynamic_root == str(root)
assert Path(cfg.sampling_config).is_file()
assert Path(cfg.sample_script()).is_file()
assert Path(cfg.evaluate_script()).is_file()
assert Path(cfg.extract_script()).is_file()
assert Path(cfg.pocket_eval_script()).is_file()
assert Path(cfg.scaffold_cascade_script()).is_file()
print(json.dumps({'root': str(root), 'library': str(Path(canonical_graph.__file__).resolve()),
                  'server': str(Path(config.__file__).resolve())}))
"""
    output = _run(
        [installed["python"], "-c", code],
        installed["outside"],
        installed["environment"],
    )
    report = json.loads(output.strip().splitlines()[-1])
    assert Path(report["root"]) == installed["root"]
    if installed["kind"] == "wheel":
        assert installed["root"] != ROOT
        assert installed["root"] in Path(report["server"]).parents
        assert installed["root"] in Path(report["library"]).parents
    else:
        assert installed["root"] == ROOT
        assert ROOT in Path(report["library"]).parents


@pytest.mark.installation
@pytest.mark.parametrize(
    "command", ["sample", "batch", "evaluate", "pocket", "extract", "cascade", "plot"]
)
def test_installed_unified_cli_help_from_unrelated_directory(installed, command):
    output = _run(
        [installed["python"], "-m", "diffdynamic", command, "--help"],
        installed["outside"],
        installed["environment"],
    )
    assert "usage:" in output.lower()


@pytest.mark.installation
def test_installed_console_entry_and_legacy_cli_wrappers(installed):
    entry = installed["binary"] / (
        "diffdynamic.exe" if os.name == "nt" else "diffdynamic"
    )
    assert "Usage: diffdynamic" in _run(
        [entry, "--help"], installed["outside"], installed["environment"]
    )
    for relative in (
        "scripts/sample_diffusion.py",
        "evaluate_pt_with_correct_reconstruct.py",
        "evaluate_pocket_quality.py",
        "batch_sampleandeval_parallel.py",
        "extract_pt_to_sdf_excel.py",
        "scaffold_cascade_pipeline.py",
    ):
        output = _run(
            [installed["python"], installed["root"] / relative, "--help"],
            installed["outside"],
            installed["environment"],
        )
        assert "usage:" in output.lower()


@pytest.mark.installation
def test_installed_web_pages_static_files_and_public_defaults(installed):
    code = r"""
import os, re
from pathlib import Path
from fastapi.testclient import TestClient
from diffdynamic.io.paths import project_root
from server import api, config
root = project_root()
# Verify default deployment behavior without an editable checkout's private
# local runtime config. Wheels do not contain that file.
if (root / 'src/diffdynamic').is_dir():
    config.CONFIG_PATH = str(Path(os.environ['DD_DB_PATH']).with_name('missing-config.json'))
cfg = config.load_config()
assert cfg.diffdynamic_root == str(root)
assert cfg.output_root == str(root / 'outputs')
assert cfg.upload_dir == str(root / 'uploads')
client = TestClient(api.app)  # No lifespan: never starts/migrates a service DB.
for route in ('/', '/app'):
    response = client.get(route)
    assert response.status_code == 200
    for asset in re.findall(r'(?:src|href)="(/static/[^"?]+)', response.text):
        assert (root / 'ui' / asset.lstrip('/')).is_file(), asset
assert client.get('/static/css/style.css').status_code == 200
assert client.get('/static/js/workbench/runtime.js').status_code == 200
assert client.get('/static/css/medchem.css').status_code == 200
assert client.get('/static/js/workbench/medchem.js').status_code == 200
assert (root / 'server/pt_sandbox_worker.py').is_file()
if not (root / 'src/diffdynamic').is_dir():
    assert not (root / 'server/.internal_api_key').exists()
print('installed web resources available')
"""
    output = _run(
        [installed["python"], "-c", code],
        installed["outside"],
        installed["environment"],
    )
    assert "installed web resources available" in output


@pytest.mark.installation
def test_installed_optional_repair_and_medchem_engines_load_from_bundled_resources(
    installed,
):
    code = r"""
from pathlib import Path
from rdkit import Chem
from diffdynamic.io.paths import project_root
from diffdynamic.chemistry.reconstruction.repair import _maybe_apply_rdkit_structure_repair
from diffdynamic.evaluation.molecule import state
from diffdynamic.evaluation.molecule.optimization import _load_medchem_optimizer

root = project_root()
molecule = Chem.MolFromSmiles('CCO')
result = _maybe_apply_rdkit_structure_repair(molecule, {
    'enable': True, 'config': 'rdkit-structure-repair/configs/conservative.yaml',
    'on_reject': 'raise',
})
# A swallowed optional-import failure would return the original object.
assert result is not molecule
assert Chem.MolToSmiles(result) == 'CCO'
import structure_repair
assert root in Path(structure_repair.__file__).resolve().parents
from structure_repair.config import default_config_path, load_optimize_config
assert default_config_path().is_file()
assert load_optimize_config()

state._MEDCHEM_OPTIMIZE_CFG = {
    'enable': True, 'config': 'rdkit-structure-repair/configs/medchem_optimize.yaml',
}
state._MEDCHEM_OPTIMIZE_STATE['loaded'] = False
loaded = _load_medchem_optimizer()
assert callable(loaded['fn'])
assert loaded['fn'].__module__ == 'structure_repair.optimize.optimize_engine'
assert loaded['config']['enable'] is True
from structure_repair.optimize import propose_molecule
from structure_repair.optimize.catalog import catalog_dir, load_catalog
catalog = load_catalog()
assert len(catalog) == 34, len(catalog)
assert root in catalog_dir().resolve().parents
parent = Chem.MolFromSmiles('O=[N+]([O-])c1ccc(CCOC)cc1C(=O)OC')
original_smiles = Chem.MolToSmiles(parent)
proposal = propose_molecule(parent, top_k=1, config={
    'time_budget_s': 5,
    'constraints': {'min_tanimoto_to_parent': 0},
})
assert proposal.status == 'PROPOSED', proposal.to_dict()
assert proposal.candidates, proposal.rejected_counts
assert proposal.candidates[0].geometry['status'] == 'passed'
assert proposal.candidates[0].smiles != original_smiles
assert Chem.MolToSmiles(parent) == original_smiles
print('bundled repair and medchem engines loaded')
"""
    output = _run(
        [installed["python"], "-c", code],
        installed["outside"],
        installed["environment"],
    )
    assert "bundled repair and medchem engines loaded" in output
