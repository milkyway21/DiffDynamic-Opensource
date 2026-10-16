"""Docking snapshots and process cleanup; no real Vina or receptor tools required."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from diffdynamic.chemistry.medchem import worker
from diffdynamic.chemistry.medchem.artifacts import Artifacts, write_json

POSE = (
    "MODEL 1\n"
    "REMARK VINA RESULT: -7.125 0.000 0.000\n"
    "ATOM      1  C   LIG A   1      31.500  -8.500  14.500  1.00  0.00     0.000 C\n"
    "ENDMDL\n"
)


def make_input(tmp_path):
    mol = Chem.MolFromSmiles("CCO")
    AllChem.EmbedMolecule(mol, randomSeed=42)
    path = tmp_path / "input.sdf"
    with Chem.SDWriter(str(path)) as writer:
        writer.write(mol)
    protein = tmp_path / "original_receptor.pdb"
    protein.write_text("HEADER ORIGINAL RECEPTOR\nEND\n")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return {
        "structure_path": str(path), "protein_path": str(protein),
        "box": {"center": [31.5, -8.5, 14.5], "size": [20.0, 24.0, 28.0], "source": "explicit"},
        "evaluation": {"mode": "vina_dock", "exhaustiveness": 8, "timeout_seconds": 17, "seed": 99},
        "scratch": str(scratch), "output": str(scratch / "docking.json"),
    }


def test_dock_passes_fixed_box_and_preserves_real_pose(tmp_path, monkeypatch):
    parameters = make_input(tmp_path)
    observed = {}

    class FakeTask:
        def __init__(self, protein, mol, **kwargs):
            observed.update(protein=protein, smiles=Chem.MolToSmiles(mol), **kwargs)
            assert Path(protein).parent == Path(parameters["scratch"])
            assert Path(protein).read_text() == Path(parameters["protein_path"]).read_text()

        def run(self, **kwargs):
            observed["run"] = kwargs
            return [{"affinity": -7.125, "pose": POSE}]

    monkeypatch.setitem(sys.modules, "diffdynamic.evaluation.docking_vina", types.SimpleNamespace(VinaDockingTask=FakeTask))
    input_path = tmp_path / "dock_request.json"
    write_json(input_path, parameters)
    worker._dock(str(input_path))
    result = json.loads(Path(parameters["output"]).read_text())
    assert observed["center"] == parameters["box"]["center"]
    assert observed["box_size"] == parameters["box"]["size"]
    assert observed["run"] == {"mode": "dock", "exhaustiveness": 8, "n_poses": 1,
                                "seed": 99, "cpu": 1, "timeout_sec": 17}
    assert result["status"] == "completed" and result["affinity"] == -7.125
    assert result["pose"] == POSE and result["box"] == parameters["box"]
    assert not Path(parameters["protein_path"]).with_suffix(".pqr").exists()
    assert not Path(parameters["protein_path"]).with_suffix(".pdbqt").exists()


@pytest.mark.parametrize("rows", [[], [{"affinity": float("nan"), "pose": POSE}],
                                  [{"affinity": -7.1, "pose": ""}]])
def test_invalid_docking_output_never_gets_a_success_status(rows, tmp_path, monkeypatch):
    parameters = make_input(tmp_path)

    class FakeTask:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, **kwargs):
            return rows

    monkeypatch.setitem(sys.modules, "diffdynamic.evaluation.docking_vina", types.SimpleNamespace(VinaDockingTask=FakeTask))
    input_path = tmp_path / "dock_request.json"
    write_json(input_path, parameters)
    worker._dock(str(input_path))
    result = json.loads(Path(parameters["output"]).read_text())
    assert result["status"] == "failed" and "pose" not in result


def test_isolated_dock_exports_pdbqt_verbatim_and_uses_requested_deadline(tmp_path, monkeypatch):
    parameters = make_input(tmp_path)
    artifacts = Artifacts(tmp_path)
    aid = artifacts.add("m00001_primary_sdf", "input.sdf", "Primary", "sdf")
    variant = {"variant_id": "m00001_primary", "structure_artifact_id": aid}
    request = {"protein_path": parameters["protein_path"], "box": parameters["box"],
               "options": {"evaluation": parameters["evaluation"]}}

    def child(args, timeout, log_path):
        assert args[0] == "--dock" and timeout == 17
        payload = json.loads(Path(args[1]).read_text())
        assert payload["box"] == parameters["box"]
        write_json(payload["output"], {"status": "completed", "affinity": -7.125, "pose": POSE})
        return 0

    monkeypatch.setattr(worker, "run_child", child)
    result = worker.isolated_dock(variant, request, artifacts)
    pose_item = next(item for item in artifacts.items if item["artifact_id"] == result["pose_artifact_id"])
    assert pose_item["format"] == "pdbqt"
    assert (tmp_path / pose_item["path"]).read_text() == POSE
    assert "pose" not in result


@pytest.mark.parametrize("return_code,status", [(-9, "timeout"), (2, "failed")])
def test_failed_or_timed_out_dock_has_no_pose_artifact(return_code, status, tmp_path, monkeypatch):
    parameters = make_input(tmp_path)
    artifacts = Artifacts(tmp_path)
    aid = artifacts.add("m00001_primary_sdf", "input.sdf", "Primary", "sdf")
    monkeypatch.setattr(worker, "run_child", lambda *args: return_code)
    result = worker.isolated_dock(
        {"variant_id": "m00001_primary", "structure_artifact_id": aid},
        {"protein_path": parameters["protein_path"], "box": parameters["box"],
         "options": {"evaluation": parameters["evaluation"]}}, artifacts,
    )
    assert result["status"] == status
    assert "pose_artifact_id" not in result
    assert len(artifacts.items) == 1


def group_script(pid_path):
    # Descendant deliberately shares the session/process group of its parent.
    return (
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"Path({str(pid_path)!r}).write_text(str(child.pid))\n"
        "time.sleep(30)\n"
    )


def not_running(pid):
    try:
        # A killed grandchild can briefly await reaping by init; a zombie cannot
        # execute work and does not represent a leaked docking subprocess.
        return Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z"
    except FileNotFoundError:
        return True


def await_not_running(pid):
    deadline = time.monotonic() + 2
    while not not_running(pid) and time.monotonic() < deadline:
        time.sleep(.01)
    assert not_running(pid), "descendant survived process-group termination"


def await_pid(path):
    deadline = time.monotonic() + 3
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert path.exists(), "child failed to launch"
    return int(path.read_text())


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux process groups")
def test_run_child_hard_timeout_kills_its_process_group(tmp_path, monkeypatch):
    pid_path = tmp_path / "grandchild.pid"
    real_popen = subprocess.Popen
    children = []

    def spawn(_args, **kwargs):
        assert kwargs["start_new_session"] is True
        process = real_popen([sys.executable, "-c", group_script(pid_path)], **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(worker.subprocess, "Popen", spawn)
    try:
        started = time.monotonic()
        assert worker.run_child(["--dock", "ignored"], .6, tmp_path / "worker.log") == -9
        assert time.monotonic() - started < 5
        grandchild = await_pid(pid_path)
        assert children[0].poll() is not None
        await_not_running(grandchild)
        assert children[0] not in worker._CHILDREN
    finally:
        for process in children:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux process groups")
def test_cancel_terminates_descendants_and_exits_with_signal_status(tmp_path):
    pid_path = tmp_path / "grandchild.pid"
    process = subprocess.Popen([sys.executable, "-c", group_script(pid_path)], start_new_session=True)
    worker._CHILDREN.add(process)
    try:
        grandchild = await_pid(pid_path)
        with pytest.raises(SystemExit) as exit_info:
            worker._cancel(signal.SIGTERM, None)
        assert exit_info.value.code == 128 + signal.SIGTERM
        assert process.poll() is not None
        await_not_running(grandchild)
    finally:
        worker._CHILDREN.discard(process)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
