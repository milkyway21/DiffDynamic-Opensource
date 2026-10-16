"""Chemistry pipeline contracts without a server, database, or real docking job."""

import copy
import csv
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from diffdynamic.chemistry.medchem.artifacts import Artifacts, write_json
from diffdynamic.chemistry.medchem.inputs import parse_record, read_records
from diffdynamic.chemistry.medchem.processing import process_record


def request_for(smiles=None, **overrides):
    request = {
        "options": {
            "operation": "repair", "profile": "local", "top_k": 3,
            "min_similarity": 0.55, "min_heavy_delta": -3, "max_heavy_delta": 4,
            "time_budget_seconds": 5,
            "evaluation": {
                "mode": "none", "seed": 42, "timeout_seconds": 5,
                "exhaustiveness": 1,
            },
        },
    }
    if smiles is not None:
        request["smiles"] = smiles
    request.update(overrides)
    return request


def artifact_molecule(root, items, variant):
    item = next(a for a in items if a["artifact_id"] == variant["structure_artifact_id"])
    supplier = Chem.SDMolSupplier(str(root / item["path"]), sanitize=False, removeHs=False)
    return next(iter(supplier))


def test_invalid_smiles_record_keeps_its_position_and_raw_input(tmp_path):
    request = request_for("CCO ethanol\nnot_a_smiles broken\nCCN ethylamine")
    records = read_records(request)
    assert [r["index"] for r in records] == [0, 1, 2]
    assert parse_record(records[1]) is None
    result, _ = process_record(records[1], request, tmp_path)
    assert result["status"] == "failed"
    assert result["input"]["raw"] == "not_a_smiles"
    assert result["input"]["index"] == 1
    assert result["original"] is None and result["primary"] is None


def test_empty_interior_sdf_record_is_preserved_as_invalid(tmp_path):
    first = Chem.MolToMolBlock(Chem.MolFromSmiles("CCO"))
    last = Chem.MolToMolBlock(Chem.MolFromSmiles("CCN"))
    source = tmp_path / "records.sdf"
    source.write_text(first + "$$$$\n\n$$$$\n" + last + "$$$$\n")
    records = read_records({"input_path": str(source)})
    assert len(records) == 3
    assert parse_record(records[1]) is None
    assert records[2]["index"] == 2


def test_empty_first_sdf_record_is_preserved_as_invalid(tmp_path):
    block = Chem.MolToMolBlock(Chem.MolFromSmiles("CCO"))
    source = tmp_path / "records.sdf"
    source.write_text("$$$$\n" + block + "$$$$\n")
    records = read_records({"input_path": str(source)})
    assert len(records) == 2
    assert parse_record(records[0]) is None
    assert Chem.MolToSmiles(parse_record(records[1])) == "CCO"


@pytest.mark.parametrize("smiles", ["C1CCOC1", "CN(C)O", "CN=C=O", "CC(O)O"])
def test_legal_structure_is_not_changed_by_pipeline(smiles, tmp_path):
    request = request_for(smiles)
    record = read_records(request)[0]
    result, _ = process_record(record, request, tmp_path)
    assert result["status"] == "unchanged"
    assert result["original"]["smiles"] == result["primary"]["smiles"]
    assert result["changes"] == []
    assert result["original"]["geometry"]["aligned"] is False
    assert result["primary"]["geometry"]["aligned"] is False


def test_invalid_nitro_original_is_not_replaced_by_its_repaired_graph(tmp_path):
    request = request_for("CN(=O)=O")
    result, items = process_record(read_records(request)[0], request, tmp_path)
    assert result["status"] == "repaired"
    original = artifact_molecule(tmp_path, items, result["original"])
    primary = artifact_molecule(tmp_path, items, result["primary"])
    old_n = next(a for a in original.GetAtoms() if a.GetAtomicNum() == 7)
    new_n = next(a for a in primary.GetAtoms() if a.GetAtomicNum() == 7)
    assert old_n.GetFormalCharge() == 0
    assert sum(b.GetBondType() == Chem.BondType.DOUBLE for b in old_n.GetBonds()) == 2
    assert new_n.GetFormalCharge() == 1
    assert result["original"]["geometry"]["status"] != "passed"
    assert result["original"]["evaluation"]["status"] == "skipped"
    assert result["changes"]


def test_source_3d_coordinates_remain_unchanged_in_original_snapshot(tmp_path):
    mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](F)Cl"))
    assert AllChem.EmbedMolecule(mol, randomSeed=12) == 0
    block = Chem.MolToMolBlock(mol)
    source = tmp_path / "input.mol"
    source.write_text(block)
    request = request_for(input_path=str(source))
    record = read_records(request)[0]
    before = parse_record(record)
    result, items = process_record(record, request, tmp_path / "out")
    original = artifact_molecule(tmp_path / "out", items, result["original"])
    np.testing.assert_allclose(original.GetConformer().GetPositions(), before.GetConformer().GetPositions())
    assert result["original"]["geometry"]["coordinate_source"] == "external"
    assert result["original"]["geometry"]["aligned"] is False


def test_explicit_hydrogens_do_not_break_primary_atom_mapping(tmp_path):
    mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](F)Cl"))
    order = [4, 0, 1, 2, 3] + list(range(5, mol.GetNumAtoms()))
    mol = Chem.RenumberAtoms(mol, order)
    AllChem.Compute2DCoords(mol)
    source = tmp_path / "input.mol"
    source.write_text(Chem.MolToMolBlock(mol))
    request = request_for(input_path=str(source))
    result, items = process_record(read_records(request)[0], request, tmp_path / "out")
    primary = artifact_molecule(tmp_path / "out", items, result["primary"])
    assert primary.GetNumAtoms() == mol.GetNumAtoms()
    mapping = result["primary"]["atom_mapping"]
    assert len(mapping) == mol.GetNumAtoms()
    for ci, pi in mapping.items():
        assert primary.GetAtomWithIdx(int(ci)).GetAtomicNum() == mol.GetAtomWithIdx(pi).GetAtomicNum()


def test_candidates_inherit_unaligned_status_after_2d_parent_embedding(tmp_path):
    request = request_for("CCOC(=O)c1ccccc1", input_aligned=True)
    request["options"]["operation"] = "propose"
    result, _ = process_record(read_records(request)[0], request, tmp_path)
    assert result["primary"]["geometry"]["aligned"] is False
    assert result["candidates"], result.get("rejected_counts")
    for candidate in result["candidates"]:
        assert candidate["geometry"]["aligned"] is False
        assert candidate["geometry"]["coordinate_source"] == "generated_conformer"


def test_snapshot_drops_inherited_vina_scores(tmp_path):
    mol = Chem.MolFromSmiles("CCO")
    mol.SetProp("vina_affinity", "-11.5")
    mol.SetProp("docking_score", "-9.5")
    artifacts = Artifacts(tmp_path)
    artifact_id = artifacts.structure(mol, "m00001_primary", 0, "CCO", "primary")
    out = artifact_molecule(tmp_path, artifacts.items, {"structure_artifact_id": artifact_id})
    assert not out.HasProp("vina_affinity")
    assert not out.HasProp("docking_score")
    assert mol.GetProp("vina_affinity") == "-11.5"


def test_all_variants_dock_independently_with_identical_parameters(tmp_path, monkeypatch):
    from diffdynamic.chemistry.medchem import worker

    calls = []

    def dock(variant, request, artifacts):
        calls.append((variant["variant_id"], variant["structure_artifact_id"],
                      copy.deepcopy(request["box"]), copy.deepcopy(request["options"]["evaluation"])))
        return {"status": "completed", "affinity": -float(len(calls)), "box": request["box"]}

    monkeypatch.setattr(worker, "isolated_dock", dock)
    request = request_for("CCOC(=O)c1ccccc1")
    request["options"]["operation"] = "propose"
    request["options"]["evaluation"]["mode"] = "vina_dock"
    request["box"] = {"center": [1.0, 2.0, 3.0], "size": [20.0, 20.0, 20.0], "source": "explicit"}
    result, _ = process_record(read_records(request)[0], request, tmp_path)
    variants = [result["original"], result["primary"], *result["candidates"]]
    assert len(variants) >= 3
    assert len(calls) == len(variants)
    assert len({c[1] for c in calls}) == len(variants)
    assert all(c[2] == request["box"] and c[3] == request["options"]["evaluation"] for c in calls)
    assert [v["evaluation"]["affinity"] for v in variants] == [-float(i + 1) for i in range(len(variants))]


def test_history_uses_primary_snapshots_and_preserves_failed_rows(tmp_path):
    source = tmp_path / "history"
    artifacts = Artifacts(source)
    primary = Chem.MolFromSmiles("[13CH3][C@H](F)Cl")
    artifact_id = artifacts.structure(primary, "old_primary", 4, None, "primary")
    protein = source / "protein.pdb"
    protein.write_text("END\n")
    artifacts.add("protein", "protein.pdb", "Protein", "pdb")
    box = {"center": [1, 2, 3], "size": [20, 20, 20], "source": "explicit"}
    report = {
        "artifacts": artifacts.items, "protein_artifact_id": "protein", "box": box,
        "records": [
            {
                "record_id": "old_valid", "name": "historical structure",
                "input": {"raw": "CCO", "kind": "smiles", "index": 4},
                "primary": {
                    "variant_id": "old_primary", "structure_artifact_id": artifact_id,
                    "geometry": {"aligned": False, "coordinate_source": "generated_conformer"},
                    "evaluation": {"status": "completed", "affinity": -11.5},
                },
            },
            {
                "record_id": "old_failed", "name": "failed reconstruction",
                "input": {
                    "raw": "not_a_smiles", "kind": "smiles", "index": 5,
                    "raw_coordinates": [[1, 2, 3]], "raw_atom_types": [6],
                },
                "primary": None, "error": "Reconstruction failed",
            },
        ],
    }
    report_path = source / "report.json"
    report_path.write_text(json.dumps(report))
    records = read_records({"source_report": str(report_path)})
    assert len(records) == 2
    assert records[0]["source_record_id"] == "old_valid"
    assert records[0]["source_variant_id"] == "old_primary"
    assert records[0]["raw"] != "CCO"
    assert Chem.MolToSmiles(parse_record(records[0])) == "[13CH3][C@H](F)Cl"
    assert records[0]["context"] == {
        "input_aligned": False, "coordinate_source": "generated_conformer",
        "protein_path": str(protein), "box": box,
    }
    assert "evaluation" not in records[0]
    assert "affinity" not in records[0]["raw"]
    failed = records[1]
    assert failed["index"] == 1
    assert failed["raw"] == "not_a_smiles"
    assert failed["raw_coordinates"] == [[1, 2, 3]]
    assert failed["raw_atom_types"] == [6]
    assert failed["source_record_id"] == "old_failed"
    assert failed["source_variant_id"] is None
    assert failed["error"] == "Reconstruction failed"
    assert parse_record(failed) is None


@pytest.mark.parametrize("symlink", [False, True])
def test_history_rejects_artifacts_outside_report_root(tmp_path, symlink):
    source = tmp_path / "history"
    source.mkdir()
    outside = tmp_path / "outside.sdf"
    outside.write_text(Chem.MolToMolBlock(Chem.MolFromSmiles("CCO")))
    if symlink:
        (source / "escape.sdf").symlink_to(outside)
        artifact_path = "escape.sdf"
    else:
        artifact_path = "../outside.sdf"
    report = {
        "artifacts": [{"artifact_id": "primary_sdf", "path": artifact_path}],
        "records": [{"record_id": "old", "primary": {
            "variant_id": "primary", "structure_artifact_id": "primary_sdf",
        }}],
    }
    report_path = source / "report.json"
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="outside its report directory"):
        read_records({"source_report": str(report_path)})


def test_missing_historical_snapshot_is_retained_as_a_failure(tmp_path):
    report = {
        "artifacts": [{"artifact_id": "missing_sdf", "path": "missing.sdf"}],
        "records": [{
            "record_id": "old", "input": {"raw": "CCO", "kind": "smiles"},
            "primary": {"variant_id": "old_primary", "structure_artifact_id": "missing_sdf"},
        }],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    records = read_records({"source_report": str(report_path)})
    assert len(records) == 1
    assert records[0]["raw"] == "CCO"
    assert records[0]["source_variant_id"] == "old_primary"
    assert "snapshot is unavailable" in records[0]["error"]
    assert parse_record(records[0]) is None


def _inline_record_child(args, timeout, log_path):
    """Exercise the batch boundary without launching external work or a server."""
    assert args[0] == "--record"
    payload = json.loads(Path(args[1]).read_text())
    result, artifacts = process_record(
        payload["record"], payload["request"], payload["root"],
    )
    write_json(payload["output"], {"record": result, "artifacts": artifacts})
    return 0


def test_history_keeps_its_protein_and_box_across_repeated_processing(tmp_path, monkeypatch):
    from diffdynamic.chemistry.medchem import worker

    old = tmp_path / "old"
    artifacts = Artifacts(old)
    aid = artifacts.structure(Chem.MolFromSmiles("CCO"), "old_primary", 0, "CCO", "primary")
    (old / "protein.pdb").write_text("END\n")
    artifacts.add("protein", "protein.pdb", "Protein", "pdb")
    box = {"center": [1, 2, 3], "size": [20, 20, 20], "source": "explicit"}
    write_json(old / "report.json", {
        "artifacts": artifacts.items, "protein_artifact_id": "protein", "box": box,
        "records": [{
            "record_id": "old_record", "name": "ethanol",
            "input": {"raw": "CCO", "kind": "smiles"},
            "primary": {"variant_id": "old_primary", "structure_artifact_id": aid,
                        "geometry": {"aligned": False, "coordinate_source": "external"}},
        }],
    })
    monkeypatch.setattr(worker, "run_child", _inline_record_child)
    output = tmp_path / "rerun"
    request = request_for(source_report=str(old / "report.json"))
    report = worker.process_batch(request, output)
    assert report["status"] == "completed"
    repeated = read_records({"source_report": str(output / "report.json")})[0]
    assert repeated["context"]["box"] == box
    protein = Path(repeated["context"]["protein_path"])
    assert output in protein.parents
    assert protein.read_text() == "END\n"
    assert repeated["source_record_id"] == "m00001"
    assert repeated["source_variant_id"] == "m00001_primary"


def test_historical_pocket_prefixes_internal_parent_ids_only(tmp_path, monkeypatch):
    from diffdynamic.chemistry.medchem import worker

    def process_one(request, output):
        record = {
            "record_id": "m00001", "index": 0, "name": "ethanol",
            "kind": "smiles", "raw": "CCO", "source_variant_id": "previous_primary",
        }
        result, items = process_record(record, request, output)
        return {"status": "completed", "records": [result], "artifacts": items}

    monkeypatch.setattr(worker, "process_batch", process_one)
    request = request_for(
        source_reports=[str(tmp_path / "old_pocket_1" / "report.json")],
        source_label="Historical pockets",
    )
    report = worker.process_pockets(request, tmp_path / "out")
    result = report["records"][0]
    assert result["original"]["parent_variant_id"] == "previous_primary"
    assert result["primary"]["parent_variant_id"] == result["original"]["variant_id"]
    assert result["pocket_label"] != "report"


def test_generated_record_errors_preserve_coordinates_and_do_not_abort_batch(tmp_path, monkeypatch):
    loading = ModuleType("diffdynamic.evaluation.molecule.loading")
    reconstruction = ModuleType("diffdynamic.evaluation.molecule.reconstruction")
    positions = [np.asarray([[float(i), 0, 0]]) for i in range(3)]
    atom_types = [np.asarray([6]) for _ in positions]
    protein, ligand = tmp_path / "protein.pdb", tmp_path / "ligand.sdf"
    protein.write_text("END\n")
    ligand.write_text(Chem.MolToMolBlock(Chem.MolFromSmiles("C")))
    data = {
        "pred_ligand_pos": positions, "pred_ligand_v": atom_types,
        "data": {"protein_filename": protein.name, "ligand_filename": ligand.name},
    }
    loading.load_pt_file = lambda path: data
    loading.decompdiff_list_to_targetdiff_dict = lambda data, path: (data, None)

    def reconstruct(pos, types, **kwargs):
        if pos[0, 0] == 0:
            raise ValueError("individual reconstruction failed")
        mol = Chem.MolFromSmiles("C")
        if pos[0, 0] == 2:
            mol.SetBoolProp("serialization_failure", True)
        return mol, None

    reconstruction.reconstruct_molecule = reconstruct
    monkeypatch.setitem(sys.modules, loading.__name__, loading)
    monkeypatch.setitem(sys.modules, reconstruction.__name__, reconstruction)
    writer = Chem.MolToMolBlock

    def write(mol, **kwargs):
        if mol.HasProp("serialization_failure"):
            raise ValueError("individual serialization failed")
        return writer(mol, **kwargs)

    monkeypatch.setattr(Chem, "MolToMolBlock", write)
    request = {"source_pt": "mock.pt", "context_roots": [str(tmp_path)]}
    records = read_records(request)
    assert len(records) == 3
    assert request["protein_path"] == str(protein)
    assert request["reference_ligand_path"] == str(ligand)
    assert request["input_aligned"] is True
    assert parse_record(records[1]) is not None
    for index in (0, 2):
        assert parse_record(records[index]) is None
        assert records[index]["index"] == index
        assert records[index]["raw_coordinates"] == positions[index].tolist()
        assert records[index]["raw_atom_types"] == [6]
    assert "reconstruction failed" in records[0]["error"]
    assert "serialization failed" in records[2]["error"]


@pytest.mark.parametrize("failure", ["embedding", "validation"])
def test_repair_with_failed_final_geometry_keeps_original_primary(tmp_path, monkeypatch, failure):
    from structure_repair.optimize import geometry

    if failure == "embedding":
        monkeypatch.setattr(geometry, "generate_conformer", lambda *args, **kwargs: None)
    else:
        monkeypatch.setattr(geometry, "validate_geometry", lambda *args, **kwargs: {
            "status": "failed", "reason": "test_invalid_geometry",
        })
    request = request_for("CN(=O)=O")
    result, items = process_record(read_records(request)[0], request, tmp_path)
    assert result["status"] == "ambiguous"
    assert result["changes"] == []
    assert result["primary"]["smiles"] == result["original"]["smiles"]
    assert result["attempted_repair"]["smiles"] == "C[N+](=O)[O-]"
    assert result["attempted_repair"]["geometry"]["status"] == "failed"
    original = artifact_molecule(tmp_path, items, result["original"])
    primary = artifact_molecule(tmp_path, items, result["primary"])
    assert Chem.MolToSmiles(primary) == Chem.MolToSmiles(original)
    assert len({item["artifact_id"] for item in items}) == len(items)


def _historical_pocket(root, pocket_number, with_invalid=True):
    artifacts = Artifacts(root)
    smiles = "CCOC(=O)c1ccccc1"
    identity = f"historical_{pocket_number}"
    aid = artifacts.structure(Chem.MolFromSmiles(smiles), identity + "_primary", 0, smiles, "primary")
    protein_text = f"REMARK pocket {pocket_number}\nEND\n"
    (root / "protein.pdb").write_text(protein_text)
    artifacts.add("protein", "protein.pdb", "Protein", "pdb")
    box = {"center": [float(pocket_number), 0, 0], "size": [20, 20, 20], "source": "explicit"}
    row = {
        "record_id": identity, "name": "ester", "status": "unchanged",
        "input": {"raw": smiles, "kind": "smiles", "index": 0},
        "primary": {
            "variant_id": identity + "_primary", "structure_artifact_id": aid,
            "geometry": {"aligned": False, "coordinate_source": "generated_conformer"},
            "evaluation": {"status": "completed", "affinity": -99.0},
        },
    }
    second = {
        "record_id": identity + "_invalid", "name": "invalid", "status": "failed",
        "input": {"raw": "not_a_smiles", "kind": "smiles", "index": 1},
        "error": "Historical parsing failed", "primary": None,
    } if with_invalid else copy.deepcopy(row)
    if not with_invalid:
        second["record_id"] += "_second"
    report = {
        "records": [row, second], "artifacts": artifacts.items,
        "protein_artifact_id": "protein", "box": box,
    }
    write_json(root / "report.json", report)
    return str(root / "report.json"), box, protein_text


def test_two_historical_pockets_keep_lineage_context_scores_and_invalid_rows(tmp_path, monkeypatch):
    from diffdynamic.chemistry.medchem import worker

    pockets = [_historical_pocket(tmp_path / f"historical_pocket_{i}", i) for i in (1, 2)]
    monkeypatch.setattr(worker, "run_child", _inline_record_child)
    docked = []

    def dock(variant, request, artifacts):
        docked.append((variant["variant_id"], copy.deepcopy(request["box"])))
        return {"status": "completed", "affinity": -float(len(docked))}

    monkeypatch.setattr(worker, "isolated_dock", dock)
    monkeypatch.setattr(worker, "isolated_evaluate", lambda variant, request, artifacts, index: {
        "success": True, "mol_idx": index,
    })
    request = request_for(source_reports=[pocket[0] for pocket in pockets], auto_evaluate=True)
    request["options"]["operation"] = "propose"
    request["options"]["evaluation"]["mode"] = "vina_dock"
    output = tmp_path / "rerun"
    report = worker.process_pockets(request, output)
    assert report["status"] == "partial"
    assert report["summary"]["total"] == 4
    assert report["summary"]["valid"] == 2
    assert len(list((output / "reconstructed_molecules").glob("*.sdf"))) == 2
    assert len(report["records"]) == 4
    assert [row["input"]["index"] for row in report["records"]] == [0, 1, 2, 3]
    for number, row in enumerate(report["records"][::2], 1):
        original, primary = row["original"], row["primary"]
        assert row["record_id"].startswith(f"p{number:04d}_")
        assert original["parent_variant_id"] == f"historical_{number}_primary"
        assert primary["parent_variant_id"] == original["variant_id"]
        assert row["box"] == pockets[number - 1][1]
        assert row["candidates"]
        for candidate in row["candidates"]:
            assert candidate["parent_variant_id"] == primary["variant_id"]
        for variant in [original, primary, *row["candidates"]]:
            assert variant["evaluation"]["evaluated_variant_id"] == variant["variant_id"]
            assert variant["evaluation"]["affinity"] != -99.0
            assert variant["geometry"]["aligned"] is False
            snapshot = artifact_molecule(output, report["artifacts"], variant)
            assert snapshot.GetProp("Molecule_ID") == variant["variant_id"]
            assert snapshot.GetIntProp("Molecule_Index") == row["input"]["index"]
            artifact = next(a for a in report["artifacts"] if a["artifact_id"] == variant["protein_artifact_id"])
            assert (output / artifact["path"]).read_text() == pockets[number - 1][2]
        assert primary["main_evaluation"]["evaluated_variant_id"] == primary["variant_id"]
    for row in report["records"][1::2]:
        assert row["status"] == "failed"
        assert row["input"]["raw"] == "not_a_smiles"
        assert row["primary"] is None
    results = worker._primary_results(report)
    assert len(results) == 2
    assert [r["mol_idx"] for r in results] == [0, 2]
    assert all(r["evaluated_variant_id"] == r["molecule_id"] for r in results)
    reread = read_records({"source_report": str(output / "report.json")})
    assert [r["context"]["box"] for r in reread[::2]] == [p[1] for p in pockets]
    assert all(parse_record(r) is None for r in reread[1::2])
    for number, record in enumerate(reread):
        protein_path = Path(record["context"]["protein_path"])
        assert output in protein_path.parents
        assert protein_path.read_text() == pockets[number // 2][2]


def test_cancelled_pocket_batch_retains_completed_and_pending_records(tmp_path, monkeypatch):
    from diffdynamic.chemistry.medchem import worker

    pockets = [_historical_pocket(tmp_path / f"historical_pocket_{i}", i, with_invalid=False)
               for i in (1, 2)]
    calls = 0

    def interrupt_fourth(args, timeout, log_path):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise SystemExit(143)
        return _inline_record_child(args, timeout, log_path)

    monkeypatch.setattr(worker, "run_child", interrupt_fourth)
    output = tmp_path / "cancelled"
    request = request_for(source_reports=[p[0] for p in pockets])
    with pytest.raises(SystemExit) as caught:
        worker.process_pockets(request, output)
    assert caught.value.code == 143
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "cancelled"
    assert len(report["records"]) == 4
    assert sum(row["primary"] is not None for row in report["records"]) == 3
    assert report["records"][-1]["status"] == "pending"
    assert report["summary"]["processed"] == 3
    assert (output / "primary.sdf").is_file()
    assert (output / "audit.json").is_file()
    assert len(list((output / "reconstructed_molecules").glob("*.sdf"))) == 3


@pytest.mark.parametrize("smiles", [
    "NC(=O)c1cccc2nc(C(=O)N3CCNCC3)[nH]c12",
    "[13CH3][C@H](F)Cl",
    "F/C=C/F",
    "C[N+](C)(C)C.[Cl-]",
    "[Na+].O=C([O-])C",
    "[2H]O[2H]",
])
def test_structure_snapshot_roundtrip_preserves_identity_and_coordinates(smiles, tmp_path):
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(molecule, randomSeed=57) == 0
    molecule = Chem.RemoveHs(molecule)
    expected = Chem.MolToSmiles(molecule, isomericSmiles=True)
    positions = molecule.GetConformer().GetPositions().copy()
    artifacts = Artifacts(tmp_path)
    aid = artifacts.structure(molecule, "roundtrip_primary", 0, expected, "primary")
    item = next(a for a in artifacts.items if a["artifact_id"] == aid)
    # Use the default sanitizing reader used by scoring and docking workers.
    output = next(iter(Chem.SDMolSupplier(str(tmp_path / item["path"]), removeHs=False)))
    assert output is not None
    assert Chem.MolToSmiles(output, isomericSmiles=True) == expected
    assert output.GetNumAtoms() == molecule.GetNumAtoms()
    assert [(a.GetAtomicNum(), a.GetIsotope(), a.GetFormalCharge()) for a in output.GetAtoms()] == [
        (a.GetAtomicNum(), a.GetIsotope(), a.GetFormalCharge()) for a in molecule.GetAtoms()
    ]
    np.testing.assert_allclose(output.GetConformer().GetPositions(), positions, rtol=0, atol=5.1e-5)
    np.testing.assert_array_equal(molecule.GetConformer().GetPositions(), positions)


def test_csv_exports_primary_score_and_keeps_invalid_input_name_inert(tmp_path):
    from diffdynamic.chemistry.medchem.artifacts import export_summary

    report = {"records": [
        {"record_id": "m1", "name": "example", "status": "unchanged", "candidates": [],
         "primary": {"variant_id": "m1_primary", "stage": "primary", "properties": {"mw": 46},
                     "evaluation": {"status": "skipped"},
                     "main_evaluation": {"status": "completed", "affinity": -4.2}}},
        {"record_id": "m2", "name": "=HYPERLINK(unsafe)", "status": "failed", "error": "unparseable"},
    ], "artifacts": []}
    export_summary(tmp_path, report)
    with (tmp_path / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["vina"] == "-4.2"
    assert rows[0]["evaluation_status"] == "completed"
    assert rows[1]["name"].startswith("'=")
    assert rows[1]["error"] == "unparseable"
