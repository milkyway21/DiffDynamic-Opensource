"""T0 aromatization: recover benzene / azole rings that bond perception missed.

Aromatization changes implicit hydrogen counts, which reaction SMARTS handle
badly, so this tier is programmatic: force aromatic flags on the ring, let
RDKit sanitize, and keep the result only if RDKit itself perceives the ring as
aromatic.  No heavy atoms move, so provenance is the identity map and no
re-embedding is needed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from rdkit import Chem

from ..rings import (
    AROMATIZABLE_C5_HETEROCYCLE,
    AROMATIZABLE_C6_RING,
    detect_aromatization_opportunities,
)
from .catalog import TierPolicy
from .transforms import PROP_PARENT_IDX, TransformCandidate, canonical_smiles

TRANSFORM_ID_C6 = "T0_AROMATIZE_C6_CARBOCYCLE"
TRANSFORM_ID_C5 = "T0_AROMATIZE_C5_HETEROCYCLE"

_DEFAULT_C6 = {
    "enabled": True,
    "min_ring_double_bonds": 0,
    "max_ring_double_bonds": 3,
    # Two ring double bonds are ambiguous between a misperceived benzene and a
    # real 1,3-cyclohexadiene, so geometry decides; three alternating ones are
    # already a Kekule benzene that merely lost its aromatic flags.
    "require_planar_below_double_bonds": 3,
    "max_ring_h_count": 10,
    "cost": 1.0,
}
_DEFAULT_C5 = {
    "enabled": True,
    "min_ring_double_bonds": 0,
    "max_ring_double_bonds": 2,
    "require_planar_below_double_bonds": 2,
    "max_ring_h_count": 8,
    "allowed_heteroatoms": [7, 8, 16],
    "cost": 1.5,
}
_DEFAULT_FUSION = {
    "enabled": True,
    "extra_ring_h_allowance": 2,
    "waive_planarity": True,
}


def _aromatize_ring(
    mol: Chem.Mol,
    ring_idxs: Sequence[int],
    nh_atom_idx: Optional[int] = None,
) -> Optional[Chem.Mol]:
    ring_set = {int(i) for i in ring_idxs}
    rw = Chem.RWMol(Chem.Mol(mol))
    for idx in ring_set:
        atom = rw.GetAtomWithIdx(idx)
        atom.SetIsAromatic(True)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    for bond in rw.GetBonds():
        if bond.GetBeginAtomIdx() in ring_set and bond.GetEndAtomIdx() in ring_set:
            bond.SetBondType(Chem.BondType.AROMATIC)
            bond.SetIsAromatic(True)
    if nh_atom_idx is not None:
        pyrrole_n = rw.GetAtomWithIdx(int(nh_atom_idx))
        pyrrole_n.SetNumExplicitHs(1)
        pyrrole_n.SetNoImplicit(True)

    out = rw.GetMol()
    try:
        out.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(out)
    except Exception:  # noqa: BLE001
        return None
    # RDKit refusing to perceive the ring as aromatic also rules out the
    # antiaromatic 4n-pi cases the policy asks us to reject.
    if not all(out.GetAtomWithIdx(i).GetIsAromatic() for i in ring_set):
        return None
    return out


def _policy_section(policy: TierPolicy, key: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    merged.update(policy.get(key, {}) or {})
    return merged


def _gate(
    evidence: Dict[str, Any],
    section: Dict[str, Any],
    fusion: Dict[str, Any],
) -> Optional[str]:
    """Return a rejection reason, or None when the ring passes the policy."""
    if not section.get("enabled", True):
        return "tier_disabled"

    n_double = int(evidence.get("n_ring_double_bonds", 0))
    if n_double < int(section["min_ring_double_bonds"]):
        return "too_few_ring_double_bonds"
    if n_double > int(section["max_ring_double_bonds"]):
        return "too_many_ring_double_bonds"

    fused = bool(evidence.get("fused_to_aromatic")) and bool(fusion.get("enabled", True))
    h_limit = int(section["max_ring_h_count"])
    if fused:
        h_limit += int(fusion.get("extra_ring_h_allowance", 0))
    if int(evidence.get("ring_h_count", 0)) > h_limit:
        return "ring_too_saturated"

    if n_double < int(section["require_planar_below_double_bonds"]):
        if fused and bool(fusion.get("waive_planarity", True)):
            return None
        if evidence.get("planar") is not True:
            return "no_planarity_evidence"
    return None


def generate_aromatization_candidates(
    parent: Chem.Mol,
    policy: TierPolicy,
    config: Optional[Dict[str, Any]] = None,
    rejected: Optional[Dict[str, int]] = None,
) -> List[TransformCandidate]:
    """Enumerate T0 aromatization candidates for one molecule."""
    c6_cfg = _policy_section(policy, "c6_carbocycle", _DEFAULT_C6)
    c5_cfg = _policy_section(policy, "c5_heterocycle", _DEFAULT_C5)
    fusion = _policy_section(policy, "fused_to_aromatic_relaxation", _DEFAULT_FUSION)

    opportunities = detect_aromatization_opportunities(parent, config or {})
    identity_map = {a.GetIdx(): a.GetIdx() for a in parent.GetAtoms()}

    candidates: List[TransformCandidate] = []
    seen: set = set()
    for issue in opportunities:
        evidence = issue.evidence or {}
        ring_idxs = [int(i) for i in evidence.get("ring_idxs", [])]
        if not ring_idxs:
            continue

        if issue.issue_code == AROMATIZABLE_C6_RING:
            section, transform_id, label = c6_cfg, TRANSFORM_ID_C6, "六元碳环芳构化 → 苯环"
        elif issue.issue_code == AROMATIZABLE_C5_HETEROCYCLE:
            allowed = set(int(z) for z in c5_cfg.get("allowed_heteroatoms", []))
            if allowed and not set(evidence.get("hetero_atomic_nums", [])) <= allowed:
                continue
            section, transform_id, label = c5_cfg, TRANSFORM_ID_C5, "五元杂环芳构化 → 唑/呋喃/噻吩"
        else:
            continue

        reason = _gate(evidence, section, fusion)
        if reason is not None:
            if rejected is not None:
                rejected[reason] = rejected.get(reason, 0) + 1
            continue

        # Pyrrole-type nitrogens need an explicit H before kekulization works,
        # so try the no-H form first and then each divalent ring nitrogen.
        nh_attempts: List[Optional[int]] = [None]
        for idx in ring_idxs:
            atom = parent.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() == 7 and atom.GetDegree() == 2:
                nh_attempts.append(idx)

        produced = False
        for nh_idx in nh_attempts:
            product = _aromatize_ring(parent, ring_idxs, nh_idx)
            if product is None:
                continue
            produced = True
            for atom in product.GetAtoms():
                atom.SetIntProp(PROP_PARENT_IDX, atom.GetIdx())
            smiles = canonical_smiles(product)
            if not smiles or smiles in seen:
                continue
            seen.add(smiles)
            candidates.append(
                TransformCandidate(
                    transform_id=transform_id,
                    tier="T0",
                    name=label,
                    rationale=(
                        "环几何近平面且键级可疑，恢复芳香性可同时改善 QED、"
                        "合成可及性与对接姿态的合理性"
                    ),
                    mol=product,
                    cost=float(section.get("cost", 1.0)),
                    requires_3d=False,
                    parent_index_map=dict(identity_map),
                    new_atom_indices=[],
                    delta_heavy=0,
                    smiles=smiles,
                )
            )
            break

        if not produced and rejected is not None:
            rejected["aromatization_sanitize_failed"] = (
                rejected.get("aromatization_sanitize_failed", 0) + 1
            )
    return candidates
