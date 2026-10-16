"""A single record's chemistry, immutable variants and optional evaluation."""

from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from . import RULE_VERSION, ensure_repair_package
from .artifacts import Artifacts, plain
from .inputs import parse_record

ensure_repair_package()


def usable(mol):
    if mol is None:
        return False
    try:
        sanitized = Chem.Mol(mol)
        Chem.SanitizeMol(sanitized)
        if any(a.GetFormalCharge() != b.GetFormalCharge()
               for a, b in zip(mol.GetAtoms(), sanitized.GetAtoms())):
            return False
        return True
    except Exception:
        return False


def smiles_of(mol):
    try:
        copy = Chem.Mol(mol)
        for atom in copy.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(Chem.RemoveHs(copy) if usable(copy) else copy,
                                isomericSmiles=True)
    except Exception:
        return None


def properties_of(mol):
    from structure_repair.optimize.reward import compute_properties

    if not usable(mol):
        return {}
    props = compute_properties(mol)
    props.update(mw=Descriptors.MolWt(mol), logp=Descriptors.MolLogP(mol),
                 tpsa=Descriptors.TPSA(mol), hbd=Lipinski.NumHDonors(mol),
                 hba=Lipinski.NumHAcceptors(mol), rotb=Lipinski.NumRotatableBonds(mol))
    return plain(props)


def structural_findings(mol, config):
    from structure_repair.atom_mapping import assign_atom_maps
    from structure_repair.detector import detect_issues

    try:
        findings = plain(detect_issues(assign_atom_maps(mol), config))
        for finding in findings:
            finding["category"] = "structural_error" if finding.get("severity") == "error" else "structural_finding"
        return findings
    except Exception as exc:
        return [{"category": "structural_error", "issue_code": "detection_failed",
                 "severity": "error", "description": str(exc), "repairable": False}]


def alert_findings(mol):
    if not usable(mol):
        return []
    from rdkit.Chem import FilterCatalog

    try:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
        catalog = FilterCatalog.FilterCatalog(params)
        return [{"category": "medchem_alert", "severity": "warning",
                 "issue_code": item.GetDescription(),
                 "description": item.GetDescription(),
                 "source": item.GetProp("FilterSet") if "FilterSet" in item.GetPropList() else "RDKit FilterCatalog",
                 "repairable": False} for item in catalog.GetMatches(mol)]
    except Exception as exc:
        return [{"category": "check_error", "issue_code": "alerts_unavailable",
                 "description": str(exc), "severity": "warning", "repairable": False}]


def box_for_request(request):
    evaluation = request["options"]["evaluation"]
    if evaluation.get("center_xyz") is not None:
        return {"center": evaluation["center_xyz"], "size": evaluation["size_xyz"], "source": "explicit"}
    reference = request.get("reference_ligand_path")
    if not reference:
        raise ValueError("Docking requires an explicit box or a reference ligand")
    if Path(reference).suffix.lower() == ".sdf":
        mol = next(iter(Chem.SDMolSupplier(reference, removeHs=False)), None)
    else:
        mol = Chem.MolFromMolFile(reference, removeHs=False)
    from structure_repair.optimize.geometry import has_conformer

    if mol is None or not has_conformer(mol):
        raise ValueError("Reference ligand requires finite 3D coordinates")
    indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    positions = mol.GetConformer().GetPositions()[indices]
    extent = positions.max(0) - positions.min(0)
    dimensions = np.maximum(extent + 10.0, 20.0)
    if (dimensions > 40).any():
        raise ValueError("Reference box exceeds 40 Angstrom; set an explicit box")
    return {"center": ((positions.max(0) + positions.min(0)) / 2).tolist(),
            "size": dimensions.tolist(), "source": "reference_ligand"}


def variant_of(mol, record, stage, artifacts, request, config, suffix="", mapping=None,
               edited_geometry=None):
    from structure_repair.optimize.geometry import generate_conformer, has_conformer, validate_geometry

    mol = Chem.Mol(mol)
    valid = usable(mol)
    if valid:
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    coordinate_source = request.get("coordinate_source", "external")
    aligned = bool(request.get("input_aligned") and has_conformer(mol))
    if valid and not has_conformer(mol):
        generated = generate_conformer(mol, seed=request["options"]["evaluation"]["seed"])
        if generated is not None:
            mol = generated
            coordinate_source = "generated_conformer"
        aligned = False
    geometry = validate_geometry(mol) if valid else {"status": "failed", "reason": "invalid_structure"}
    if edited_geometry:
        checked = geometry
        geometry = {**plain(edited_geometry), **checked}
        if edited_geometry.get("status") == "failed":
            geometry.update(status="failed", reason=edited_geometry.get("reason", "edited_geometry_failed"))
    geometry.update(coordinate_source=coordinate_source, aligned=aligned,
                    pocket_status="not_applicable")
    if aligned and request.get("protein_path") and geometry["status"] == "passed":
        from structure_repair.optimize.geometry import PocketClashChecker, load_pocket_coords

        coords = load_pocket_coords(request["protein_path"])
        if coords is None:
            geometry.update(pocket_status="failed", pocket_reason="protein_coordinates_unavailable")
        else:
            from scipy.spatial import cKDTree

            minimum = float(cKDTree(coords).query(mol.GetConformer().GetPositions())[0].min())
            if minimum > 12.0:
                geometry.update(aligned=False, pocket_status="failed", pocket_reason="input_remote_from_protein")
            else:
                clash = PocketClashChecker(request["protein_path"]).has_clash(mol)
                geometry.update(pocket_status="failed" if clash else "passed",
                                pocket_reason="coarse_distance_clash" if clash else None)
    variant_id = record["record_id"] + "_" + stage + suffix
    variant = {"variant_id": variant_id, "stage": stage, "smiles": smiles_of(mol),
               "record_id": record["record_id"],
               "parent_variant_id": record["record_id"] + ("_primary" if stage == "candidate" else "_original") if stage != "original" else record.get("source_variant_id"),
               "properties": properties_of(mol), "findings": structural_findings(mol, config) + alert_findings(mol),
               "changes": [], "atom_mapping": mapping or {}, "geometry": plain(geometry),
               "structure_format": "sdf", "evaluation": {"status": "skipped", "reason": "docking_not_requested"}}
    try:
        variant["structure_artifact_id"] = artifacts.structure(mol, variant_id, record["index"], variant["smiles"], stage)
        variant["depiction_artifact_id"] = artifacts.depiction(mol, variant_id, geometry.get("affected_atoms", []))
    except Exception as exc:
        variant["render_error"] = str(exc)
    return variant, mol


def process_record(record, request, root):
    from structure_repair import repair_molecule
    from structure_repair.atom_mapping import assign_atom_maps, atom_map_to_idx
    from structure_repair.config import load_config, load_optimize_config
    from structure_repair.optimize import propose_molecule
    from structure_repair.optimize.geometry import affected_atom_indices, embed_new_atoms, has_conformer

    artifacts = Artifacts(root)
    config = load_config()
    options = request["options"]
    result = {"record_id": record["record_id"], "name": record["name"], "status": "failed",
              "original": None, "primary": None, "candidates": [], "findings": [], "changes": [],
              "rule_version": RULE_VERSION, "rejected_counts": {}, "input": record}
    mol = parse_record(record)
    if mol is None:
        result["error"] = record.get("error") or "Input could not be parsed"
        return result, artifacts.items
    result["original_atom_maps"] = {a.GetIdx(): a.GetAtomMapNum() for a in mol.GetAtoms()}
    try:
        result["internal_atom_maps"] = atom_map_to_idx(assign_atom_maps(mol))
    except ValueError as exc:
        result["mapping_error"] = str(exc)
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)
        if usable(mol):
            Chem.SanitizeMol(mol)
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception:
        pass
    result["findings"] = structural_findings(mol, config) + alert_findings(mol)
    primary = Chem.Mol(mol)
    mapping = {i: i for i in range(mol.GetNumAtoms())}
    result["status"] = "unchanged" if usable(mol) else "rejected"
    if options["operation"] != "check":
        repair = repair_molecule(mol, config=config, molecule_id=record["record_id"])
        result["status"] = repair.status.lower()
        result["repair_reason"] = repair.reject_reason
        if repair.status in {"REPAIRED", "UNCHANGED"} and repair.repaired_mol is not None:
            before_maps = atom_map_to_idx(repair.original_mol)
            after_maps = atom_map_to_idx(repair.repaired_mol)
            mapping = {out_idx: before_maps[map_id] for map_id, out_idx in after_maps.items() if map_id in before_maps}
            primary = repair.repaired_mol
            result["changes"] = plain(repair.bond_edits) + plain(repair.atom_edits)
            if repair.status == "REPAIRED" and has_conformer(mol):
                corrected = embed_new_atoms(mol, primary, mapping, seed=options["evaluation"]["seed"])
                if corrected is None:
                    primary = Chem.Mol(mol)
                    result.update(status="ambiguous", repair_reason="Repaired graph failed local geometry validation", changes=[])
                else:
                    primary = corrected
        result["applied_rule_ids"] = repair.applied_rule_ids
    original, _ = variant_of(mol, record, "original", artifacts, request, config)
    changed_atoms = sorted(affected_atom_indices(mol, primary, mapping)) if result["changes"] else []
    main, primary_3d = variant_of(primary, record, "primary", artifacts, request, config, mapping=mapping,
                                 edited_geometry={"affected_atoms": changed_atoms})
    if result["status"] == "repaired" and (
        main["geometry"]["status"] != "passed" or main["geometry"].get("pocket_status") == "failed"
    ):
        result["attempted_repair"] = {"smiles": main["smiles"], "changes": result["changes"],
                                      "geometry": main["geometry"]}
        result.update(status="ambiguous", changes=[],
                      repair_reason="Repaired structure did not pass complete geometry validation; original retained")
        primary = Chem.Mol(mol)
        mapping = {i: i for i in range(mol.GetNumAtoms())}
        discarded = {main.get("structure_artifact_id"), main.get("depiction_artifact_id")}
        artifacts.items = [item for item in artifacts.items if item["artifact_id"] not in discarded]
        main, primary_3d = variant_of(primary, record, "primary", artifacts, request, config, mapping=mapping)
    main["changes"] = result["changes"]
    result.update(original=original, primary=main)
    if options["operation"] == "propose" and usable(primary):
        proposal_config = load_optimize_config()
        proposal_config["time_budget_s"] = options["time_budget_seconds"]
        proposal_config.setdefault("constraints", {}).update(
            min_tanimoto_to_parent=options["min_similarity"],
            min_heavy_atom_delta=options["min_heavy_delta"],
            max_heavy_atom_delta=options["max_heavy_delta"],
        )
        proposal = propose_molecule(primary_3d, profile=options["profile"], top_k=options["top_k"],
                                    config=proposal_config, molecule_id=record["record_id"],
                                    protein_path=request.get("protein_path") if main["geometry"].get("aligned") else None)
        result["rejected_counts"] = proposal.rejected_counts
        result["proposal_reason"] = proposal.reject_reason
        result["proposal_rule_version"] = proposal.rule_version
        candidate_request = {**request,
                             "input_aligned": main["geometry"].get("aligned", False),
                             "coordinate_source": main["geometry"].get("coordinate_source", "external")}
        for i, candidate in enumerate(proposal.candidates):
            variant, _ = variant_of(candidate.mol, record, "candidate", artifacts, candidate_request, config,
                                     suffix=f"_{i + 1}", mapping=candidate.parent_index_map,
                                     edited_geometry=candidate.geometry)
            variant.update(transform_id=candidate.transform_id, name=candidate.name,
                           rationale=candidate.rationale, similarity=candidate.similarity,
                           reward=plain(candidate.reward), delta_heavy_atoms=candidate.delta_heavy_atoms)
            variant["changes"] = [{"transform_id": candidate.transform_id, "description": candidate.rationale}]
            result["candidates"].append(variant)
    if options["evaluation"]["mode"] == "vina_dock":
        from .worker import isolated_dock

        for variant in [result["original"], result["primary"], *result["candidates"]]:
            if variant["geometry"]["status"] != "passed" or not variant.get("structure_artifact_id"):
                variant["evaluation"] = {"status": "skipped", "reason": "structure_or_geometry_invalid"}
                continue
            variant["evaluation"] = isolated_dock(variant, request, artifacts)
    if request.get("auto_evaluate") and usable(primary):
        from .worker import isolated_evaluate

        result["primary_evaluation"] = isolated_evaluate(result["primary"], request, artifacts, record["index"])
        evaluated = result["primary_evaluation"]
        docking = result["primary"]["evaluation"]
        affinity = docking.get("affinity") if docking.get("status") == "completed" else evaluated.get("vina_score_only")
        result["primary"]["main_evaluation"] = {
            "status": "completed" if evaluated.get("success") else evaluated.get("status", "failed"),
            "affinity": affinity, "reason": evaluated.get("error"),
            "mode": "dock" if docking.get("status") == "completed" else "score_only" if affinity is not None else "chemistry",
        }
    for variant in [result["original"], result["primary"], *result["candidates"]]:
        variant["evaluation"]["evaluated_variant_id"] = variant["variant_id"]
        if variant.get("main_evaluation"):
            variant["main_evaluation"]["evaluated_variant_id"] = variant["variant_id"]
    return plain(result), artifacts.items
