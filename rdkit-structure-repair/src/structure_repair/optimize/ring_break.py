"""Enumerate explicit ring-opening design hypotheses.

Medium and larger rings can be chemically valid, active motifs. Opening them is
an optional aggressive analogue design, never evidence of a reconstruction error.
Unrelated aromatic systems and atom provenance remain intact.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from rdkit import Chem

from .ring_ops import detect_oversized_ring_issues, pick_ring_bond_to_break
from .transforms import PROP_PARENT_IDX, TransformCandidate, canonical_smiles, heavy_atom_count

TRANSFORM_BREAK_MACRO = "T0_BREAK_MACROCYCLE"
TRANSFORM_BREAK_MEDIUM = "T0_BREAK_MEDIUM_RING"
TRANSFORM_LOCK_AFTER_OPEN = "T2_CHAIN_LOCK_AFTER_OPEN"


def _break_bond(mol: Chem.Mol, i1: int, i2: int) -> Optional[Chem.Mol]:
    source_bond = mol.GetBondBetweenAtoms(int(i1), int(i2))
    if source_bond is None or source_bond.GetIsAromatic():
        return None
    work = Chem.Mol(mol)
    for atom in work.GetAtoms():
        # Provenance is recorded before editing, never inferred from atom count.
        atom.SetIntProp(PROP_PARENT_IDX, atom.GetIdx())
    try:
        Chem.Kekulize(work, clearAromaticFlags=True)
    except Exception:
        return None
    rw = Chem.RWMol(work)
    rw.RemoveBond(int(i1), int(i2))
    for idx in (i1, i2):
        rw.GetAtomWithIdx(int(idx)).SetNoImplicit(False)

    out = rw.GetMol()
    try:
        out.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(out)
    except Exception:
        return None
    if len(Chem.GetMolFrags(out)) != 1:
        return None
    return out


def _stamp_identity_provenance(product: Chem.Mol, parent: Chem.Mol) -> Dict[int, int]:
    """Read explicit edit provenance; equal atom counts prove no correspondence."""
    index_map = {}
    for atom in product.GetAtoms():
        if not atom.HasProp(PROP_PARENT_IDX):
            return {}
        pi = atom.GetIntProp(PROP_PARENT_IDX)
        if not 0 <= pi < parent.GetNumAtoms() or pi in index_map.values():
            return {}
        index_map[atom.GetIdx()] = pi
    return index_map


def _ether_lock_candidates(opened: Chem.Mol) -> List[Chem.Mol]:
    """CH2–CH2 → O–CH2 on aliphatic chain segments after ring opening."""
    from rdkit.Chem import AllChem

    try:
        rxn = AllChem.ReactionFromSmarts(
            "[C:1][CH2:2][CH2:3][C:4]>>[C:1][O:2][CH2:3][C:4]"
        )
        rxn.Initialize()
    except Exception:  # noqa: BLE001
        return []
    out = []
    seen = set()
    try:
        sets = rxn.RunReactants((opened,))
    except Exception:  # noqa: BLE001
        return []
    for product_set in sets[:6]:
        if not product_set:
            continue
        p = Chem.Mol(product_set[0])
        try:
            p.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(p)
        except Exception:  # noqa: BLE001
            continue
        from .transforms import _resolve_provenance
        _resolve_provenance(p, opened)
        smi = canonical_smiles(p)
        if not smi or smi in seen:
            continue
        seen.add(smi)
        out.append(p)
    return out


def generate_ring_break_candidates(
    parent: Chem.Mol,
    config: Optional[Dict[str, Any]] = None,
    rejected: Optional[Dict[str, int]] = None,
) -> List[TransformCandidate]:
    config = config or {}
    issues = detect_oversized_ring_issues(parent, config)
    candidates: List[TransformCandidate] = []
    seen = set()
    parent_heavy = heavy_atom_count(parent)

    for issue in issues:
        ring = issue.evidence.get("ring_idxs") or []
        if not ring:
            continue
        pick = pick_ring_bond_to_break(parent, ring)
        if pick is None:
            if rejected is not None:
                rejected["no_breakable_bond"] = rejected.get("no_breakable_bond", 0) + 1
            continue
        opened = _break_bond(parent, pick[0], pick[1])
        if opened is None:
            if rejected is not None:
                rejected["ring_break_sanitize_failed"] = (
                    rejected.get("ring_break_sanitize_failed", 0) + 1
                )
            continue

        if issue.issue_code == "MACROCYCLE_OVERSIZED":
            tid, name, tier, cost, bonus = (
                TRANSFORM_BREAK_MACRO,
                "大环拆键开环",
                "T0",
                1.0,
                0.25,
            )
        else:
            tid, name, tier, cost, bonus = (
                TRANSFORM_BREAK_MEDIUM,
                "中环(7/8)拆键开环",
                "T0",
                1.5,
                0.20,
            )

        variants = [(tid, name, tier, cost, bonus, opened)]
        # Also offer ether-locked follow-ups as single-step combined candidates
        for locked in _ether_lock_candidates(opened):
            variants.append(
                (
                    TRANSFORM_LOCK_AFTER_OPEN,
                    "开环后链内醚氧替换",
                    "T2",
                    1.5,
                    0.30,
                    locked,
                )
            )

        for v_tid, v_name, v_tier, v_cost, v_bonus, mol in variants:
            smi = canonical_smiles(mol)
            if not smi or smi in seen:
                continue
            seen.add(smi)
            index_map = _stamp_identity_provenance(mol, parent)
            # After break, atom count usually unchanged; ether lock replaces C with O
            new_atoms = [i for i in range(mol.GetNumAtoms()) if i not in index_map]
            candidates.append(
                TransformCandidate(
                    transform_id=v_tid,
                    tier=v_tier,
                    name=v_name,
                    rationale=(
                        f"探索 {len(ring)} 元环拆键开环后的构象变化"
                        + ("，并探索链内醚氧替换" if v_tid == TRANSFORM_LOCK_AFTER_OPEN else "")
                        + "；环尺寸本身不是化学错误，结合与合成需验证。"
                    ),
                    mol=mol,
                    cost=v_cost,
                    requires_3d=False,
                    reward_bonus=v_bonus,
                    parent_index_map=index_map,
                    new_atom_indices=new_atoms,
                    delta_heavy=heavy_atom_count(mol) - parent_heavy,
                    smiles=smi,
                )
            )
    return candidates
