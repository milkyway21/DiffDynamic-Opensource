"""Independent, auditable analogue proposals from one unchanged parent."""

from __future__ import annotations

import hashlib
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

from ..models import ProposalCandidate, ProposalResult
from .aromatize import generate_aromatization_candidates
from .catalog import catalog_dir, load_catalog, load_tier_policy
from .constraints import ConstraintConfig, check_candidate, ring_systems_preserved, tanimoto
from .geometry import (
    PocketClashChecker, affected_atom_indices, embed_new_atoms, generate_conformer,
    has_conformer, validate_geometry, valid_mapping,
)
from .reward import compute_properties, compute_reward, structural_alerts, weights_from_config
from .ring_break import TRANSFORM_LOCK_AFTER_OPEN, generate_ring_break_candidates
from .ring_contract import generate_ring_contract_candidates
from .transforms import canonical_smiles, enumerate_candidates, prepare_parent

# Deliberately explicit: new rules never silently enter the local profile.
LOCAL_TRANSFORM_IDS = frozenset({
    "T1_ESTER_TO_AMIDE", "T1_PHENOL_TO_FLUORIDE", "T1_CH2_TO_O",
    "T1_S_TO_SO2", "T1_OME_TO_OCHF2", "T2_ETHYLENE_TO_ETHER",
    "T2_GEM_DIMETHYL", "T3_NITRO_TO_NITRILE",
    "T3_NITRO_TO_TRIFLUOROMETHYL", "T3_NITRO_TO_METHYLSULFONYL",
    "T3_ANILINE_ACYLATION", "T3_ALDEHYDE_TO_NITRILE", "T3_THIOUREA_TO_UREA",
    "T3_ARYL_OME_TO_OCF3", "T3_ARYL_OME_TO_OCHF2",
    "T3_BENZYLIC_CH2_TO_CF2", "T3_BENZYLIC_CH2_TO_CHF",
    "T4_ARYL_HALIDE_TO_SULFONAMIDE", "T4_MAGIC_METHYL",
    "T4_ALIPHATIC_CH_TO_CF", "T4_TRUNCATE_TERMINAL_ETHYL",
    "T4_TRUNCATE_TERMINAL_PROPYL", "T5_TBU_TO_IPR",
    "T5_QUATERNARY_CARBON_DEMETHYL", "T5_GEM_DIFLUORO_TO_CH2",
})


def rule_version() -> str:
    digest = hashlib.sha256()
    digest.update(b"proposal-v1:local-whitelist:strict-parent-constraints:geometry-v2")
    sources = sorted(catalog_dir().glob("*.yaml"))
    sources += sorted(Path(__file__).parent.glob("*.py"))
    sources.append(Path(__file__).parent.parent / "identity_check.py")
    for path in sources:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "medchem-" + digest.hexdigest()[:16]


def _stereoisomers(candidate, parent, limit, seed):
    """Enumerate only newly created, unassigned stereocenters, bounded by top_k."""
    existing = dict(Chem.FindMolChiralCenters(parent, includeUnassigned=True))
    new = {
        i for i, label in Chem.FindMolChiralCenters(candidate.mol, includeUnassigned=True)
        if label == "?" and candidate.parent_index_map.get(i) not in existing
    }
    if not new:
        return [candidate]
    options = StereoEnumerationOptions(onlyUnassigned=True, maxIsomers=limit, rand=seed)
    # Prevent assigning pre-existing unspecified stereocenters as a side effect.
    allowed = []
    original_tags = {a.GetIdx(): a.GetChiralTag() for a in candidate.mol.GetAtoms()}
    original_bonds = {
        b.GetIdx(): (b.GetStereo(), list(b.GetStereoAtoms()), b.GetBondDir())
        for b in candidate.mol.GetBonds()
    }
    seen = set()
    for isomer in EnumerateStereoisomers(candidate.mol, options=options):
        for atom in isomer.GetAtoms():
            if atom.GetIdx() not in new:
                atom.SetChiralTag(original_tags[atom.GetIdx()])
        # Enumeration must not invent a configuration for a pre-existing,
        # unspecified alkene merely because a new tetrahedral center appeared.
        for bond in isomer.GetBonds():
            stereo, refs, direction = original_bonds[bond.GetIdx()]
            if len(refs) == 2:
                bond.SetStereoAtoms(*refs)
            bond.SetStereo(stereo)
            bond.SetBondDir(direction)
        smiles = canonical_smiles(isomer)
        if smiles not in seen:
            seen.add(smiles)
            allowed.append(replace(candidate, mol=isomer, smiles=smiles))
    return allowed


def propose_molecule(
    mol: Optional[Chem.Mol],
    profile: str = "local",
    top_k: int = 3,
    config: Optional[Dict[str, Any]] = None,
    protein_path: Optional[str] = None,
    molecule_id: str = "mol",
    repair_config: Optional[Dict[str, Any]] = None,
) -> ProposalResult:
    """Return up to ``top_k`` independently validated single-operation analogues.

    Input must already have passed legality repair. Missing/2D coordinates are
    embedded internally and never used for protein clash checking. Similarity,
    heavy-atom budgets and stereochemistry are always measured against this
    one parent; a tier cannot override them. Reward ranks design hypotheses and
    is not a prediction of activity, safety or binding affinity. Returned atom
    correspondence points to the exact input molecule indices (including when
    explicit hydrogens preceded heavy atoms in the input).
    """
    if profile not in {"local", "aggressive"}:
        raise ValueError("profile must be local or aggressive")
    if not 1 <= int(top_k) <= 10:
        raise ValueError("top_k must be between 1 and 10")
    config = dict(config or {})
    result = ProposalResult(molecule_id=molecule_id, original_mol=mol, rule_version=rule_version())
    rejected = result.rejected_counts

    def reject(reason):
        rejected[reason] = rejected.get(reason, 0) + 1

    if mol is None:
        result.status, result.reject_reason = "REJECTED", "null_input"
        return result
    original = Chem.Mol(mol)
    for atom in original.GetAtoms():
        atom.SetIntProp("_sr_input_atom_index", atom.GetIdx())
    result.original_mol = original
    try:
        Chem.SanitizeMol(original)
        parent = prepare_parent(original)
        if parent is None or parent.GetNumAtoms() == 0:
            raise ValueError("empty_parent")
        if len(Chem.GetMolFrags(parent)) != 1:
            raise ValueError("multi_fragment_parent_preserved")
    except Exception as exc:
        result.status, result.reject_reason = "REJECTED", str(exc)
        return result
    working_to_input = {a.GetIdx(): a.GetIntProp("_sr_input_atom_index")
                        for a in parent.GetAtoms()}
    result.properties_before = compute_properties(parent)
    result.properties_before["structural_alerts"] = structural_alerts(parent)
    constraints = ConstraintConfig.from_dict(config.get("constraints"))
    constraints.tanimoto_exempt_tiers = ()
    constraints.scaffold_mode = "off"  # explicit profile ring gate below
    constraints.preserve_stereo = True
    seed = int(config.get("random_seed", 42))
    deadline = time.monotonic() + max(0.01, float(config.get("time_budget_s", 30)))
    source = "input_3d" if has_conformer(parent) else "generated"
    if source == "generated":
        try:
            parent = generate_conformer(parent, seed)
        except Exception:
            parent = None
        if parent is None:
            result.status, result.reject_reason = "REJECTED", "parent_embedding_failed"
            return result
    checker = PocketClashChecker(protein_path) if protein_path and source == "input_3d" else None
    if checker is not None and not checker.active:
        result.status, result.reject_reason = "REJECTED", "protein_coordinates_unavailable"
        return result
    tiers = config.get("tiers", ["T0", "T1", "T2", "T3", "T4", "T5"])
    disabled = set(config.get("disabled_transform_ids", []))
    catalog = load_catalog(enabled_tiers=[t for t in tiers if t != "T0"], disabled_transform_ids=disabled)
    if profile == "local":
        catalog = [spec for spec in catalog if spec.transform_id in LOCAL_TRANSFORM_IDS]
    raw = enumerate_candidates(parent, catalog, max_products_per_transform=int(config.get("max_products_per_transform", 4)))
    if profile == "aggressive" and "T0" in tiers:
        raw.extend(generate_aromatization_candidates(parent, load_tier_policy("T0"), config=config, rejected=rejected))
        raw.extend(generate_ring_contract_candidates(parent, config=config, rejected=rejected))
        raw.extend(c for c in generate_ring_break_candidates(parent, config=config, rejected=rejected)
                   if c.transform_id != TRANSFORM_LOCK_AFTER_OPEN)
    seen = {canonical_smiles(parent)}
    accepted = []
    weights = weights_from_config(config)
    for raw_candidate in raw:
        if time.monotonic() >= deadline:
            reject("search_budget_exhausted")
            break
        if raw_candidate.transform_id in disabled:
            continue
        for candidate in _stereoisomers(raw_candidate, parent, int(top_k), seed):
            if time.monotonic() >= deadline:
                reject("search_budget_exhausted")
                break
            smiles = canonical_smiles(candidate.mol)
            if smiles in seen:
                reject("duplicate_structure")
                continue
            if not valid_mapping(parent, candidate.mol, candidate.parent_index_map):
                reject("missing_or_invalid_mapping")
                continue
            reason = check_candidate(parent, candidate.mol, candidate.tier,
                                     parent_index_map=candidate.parent_index_map, cfg=constraints)
            if reason:
                reject(reason)
                continue
            if profile == "local" and not ring_systems_preserved(parent, candidate.mol, candidate.parent_index_map):
                reject("local_ring_system_changed")
                continue
            posed = embed_new_atoms(parent, candidate.mol, candidate.parent_index_map,
                                    seed=seed, attempts=int(config.get("embed_attempts", 3)),
                                    time_budget_s=min(float(config.get("embed_time_budget_s", 3)), max(0.01, deadline - time.monotonic())))
            if posed is None:
                reject("local_geometry_failed")
                continue
            affected = affected_atom_indices(parent, posed, candidate.parent_index_map)
            if checker is not None and checker.has_clash(posed, sorted(affected)):
                reject("pocket_clash")
                continue
            geometry = validate_geometry(posed)
            if geometry["status"] != "passed":
                reject("invalid_geometry")
                continue
            geometry.update(source=source, aligned_to_protein=checker is not None,
                            affected_atoms=sorted(affected),
                            fixed_atoms=sorted(set(candidate.parent_index_map) - affected))
            properties = compute_properties(posed)
            properties["structural_alerts"] = structural_alerts(posed)
            reward = compute_reward(result.properties_before, properties, transform_cost=candidate.cost, weights=weights)
            # No transform bonus: unsupported efficacy claims cannot buy acceptance.
            seen.add(smiles)
            accepted.append(ProposalCandidate(
                mol=posed, transform_id=candidate.transform_id, tier=candidate.tier,
                name=candidate.name, rationale=candidate.rationale,
                parent_index_map={ci: working_to_input[pi] for ci, pi in candidate.parent_index_map.items()},
                properties=properties,
                reward=reward, similarity=tanimoto(parent, posed), geometry=geometry,
                delta_heavy_atoms=posed.GetNumHeavyAtoms() - parent.GetNumHeavyAtoms(), smiles=smiles,
            ))
    accepted.sort(key=lambda c: (-c.reward["total"], -c.similarity, c.transform_id, c.smiles))
    result.candidates = accepted[:int(top_k)]
    result.status = "PROPOSED" if result.candidates else "UNCHANGED"
    return result
