"""Break oversized / medium rings produced by diffusion bond perception.

Diffusion often invents 7–8 membered rings and ≥10-atom macrocycles that are
not chemically intended.  Strategy:

1. ≥10 macrocycles: break the weakest bond → open chain, then optional ether lock.
2. 7/8 medium rings: ``ring_contract.py`` first tries contraction to benzene /
   heteroarenes; this module remains the fallback (break → chain → ether lock).
3. Aromatic 6-rings are left to the T0 aromatize path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from rdkit import Chem

from ..mol_edit import finalize_candidate_mol
from .ring_ops import detect_oversized_ring_issues, pick_ring_bond_to_break
from .transforms import PROP_PARENT_IDX, TransformCandidate, canonical_smiles, heavy_atom_count

TRANSFORM_BREAK_MACRO = "T0_BREAK_MACROCYCLE"
TRANSFORM_BREAK_MEDIUM = "T0_BREAK_MEDIUM_RING"
TRANSFORM_LOCK_AFTER_OPEN = "T2_CHAIN_LOCK_AFTER_OPEN"


def _break_bond(mol: Chem.Mol, i1: int, i2: int) -> Optional[Chem.Mol]:
    rw = Chem.RWMol(Chem.Mol(mol))
    if rw.GetBondBetweenAtoms(int(i1), int(i2)) is None:
        return None
    rw.RemoveBond(int(i1), int(i2))
    # Clear aromatic flags that may be inconsistent after opening
    for atom in rw.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(False)
    for bond in rw.GetBonds():
        bond.SetIsAromatic(False)
        if bond.GetBondType() == Chem.BondType.AROMATIC:
            bond.SetBondType(Chem.BondType.SINGLE)
    out = rw.GetMol()
    try:
        out.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(out)
    except Exception:  # noqa: BLE001
        # Try soft sanitize
        try:
            Chem.SanitizeMol(
                out,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
        except Exception:  # noqa: BLE001
            return None
    if len(Chem.GetMolFrags(out)) != 1:
        return None
    return out


def _stamp_identity_provenance(product: Chem.Mol, parent: Chem.Mol) -> Dict[int, int]:
    """Best-effort: same atom count → identity map; else match by PROP_PARENT_IDX."""
    index_map: Dict[int, int] = {}
    if product.GetNumAtoms() == parent.GetNumAtoms():
        for i in range(product.GetNumAtoms()):
            index_map[i] = i
            product.GetAtomWithIdx(i).SetIntProp(PROP_PARENT_IDX, i)
        return index_map
    for atom in product.GetAtoms():
        if atom.HasProp(PROP_PARENT_IDX):
            try:
                index_map[atom.GetIdx()] = int(atom.GetProp(PROP_PARENT_IDX))
            except Exception:  # noqa: BLE001
                pass
    if not index_map:
        # Fall back: match by element sequence order for heavy atoms
        p_heavies = [a.GetIdx() for a in parent.GetAtoms() if a.GetAtomicNum() > 1]
        c_heavies = [a.GetIdx() for a in product.GetAtoms() if a.GetAtomicNum() > 1]
        for ci, pi in zip(c_heavies, p_heavies):
            if parent.GetAtomWithIdx(pi).GetAtomicNum() == product.GetAtomWithIdx(ci).GetAtomicNum():
                index_map[ci] = pi
                product.GetAtomWithIdx(ci).SetIntProp(PROP_PARENT_IDX, pi)
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
                    "开环后醚锁定柔性链",
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
                        f"扩散重建产生 {len(ring)} 元环；拆键开环"
                        + (" 并用醚桥锁定柔性链" if v_tid == TRANSFORM_LOCK_AFTER_OPEN else "")
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
