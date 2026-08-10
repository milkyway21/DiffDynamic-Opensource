"""Constraint gates that replace ``identity_ok`` on the optimize channel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional, Tuple

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

DEFAULT_ALLOWED_ELEMENTS = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53})
_MORGAN_GEN = None


def _morgan_fingerprint(mol: Chem.Mol):
    global _MORGAN_GEN
    if _MORGAN_GEN is None:
        try:
            from rdkit.Chem import rdFingerprintGenerator

            _MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        except Exception:  # noqa: BLE001
            _MORGAN_GEN = False
    if _MORGAN_GEN:
        return _MORGAN_GEN.GetFingerprint(mol)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def tanimoto(a: Chem.Mol, b: Chem.Mol) -> float:
    try:
        return float(DataStructs.TanimotoSimilarity(_morgan_fingerprint(a), _morgan_fingerprint(b)))
    except Exception:  # noqa: BLE001
        return 0.0


def murcko_smiles(mol: Chem.Mol) -> Optional[str]:
    try:
        sc = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(sc) if sc is not None else None
    except Exception:  # noqa: BLE001
        return None


def _scaffold_atom_indices(mol: Chem.Mol) -> FrozenSet[int]:
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return frozenset()
        match = mol.GetSubstructMatch(scaffold)
        return frozenset(match) if match else frozenset()
    except Exception:  # noqa: BLE001
        return frozenset()


@dataclass
class ConstraintConfig:
    min_tanimoto_to_parent: float = 0.55
    tanimoto_exempt_tiers: Tuple[str, ...] = ("T0", "T2")
    scaffold_mode: str = "ring_systems"
    scaffold_exempt_tiers: Tuple[str, ...] = ("T0", "T2", "T5")
    max_heavy_atom_delta: int = 4
    min_heavy_atom_delta: int = -3
    mw_window: Tuple[float, float] = (150.0, 600.0)
    max_transforms_per_molecule: int = 3
    allowed_elements: FrozenSet[int] = field(default_factory=lambda: DEFAULT_ALLOWED_ELEMENTS)
    preserve_stereo: bool = True
    require_single_fragment: bool = True

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ConstraintConfig":
        data = data or {}
        cfg = cls()
        if "min_tanimoto_to_parent" in data:
            cfg.min_tanimoto_to_parent = float(data["min_tanimoto_to_parent"])
        if "tanimoto_exempt_tiers" in data:
            cfg.tanimoto_exempt_tiers = tuple(str(t).upper() for t in data["tanimoto_exempt_tiers"])
        if "scaffold_mode" in data:
            cfg.scaffold_mode = str(data["scaffold_mode"])
        if "scaffold_exempt_tiers" in data:
            cfg.scaffold_exempt_tiers = tuple(str(t).upper() for t in data["scaffold_exempt_tiers"])
        if "max_heavy_atom_delta" in data:
            cfg.max_heavy_atom_delta = int(data["max_heavy_atom_delta"])
        if "min_heavy_atom_delta" in data:
            cfg.min_heavy_atom_delta = int(data["min_heavy_atom_delta"])
        if "mw_window" in data:
            lo, hi = data["mw_window"]
            cfg.mw_window = (float(lo), float(hi))
        if "max_transforms_per_molecule" in data:
            cfg.max_transforms_per_molecule = int(data["max_transforms_per_molecule"])
        if "allowed_elements" in data:
            cfg.allowed_elements = frozenset(int(z) for z in data["allowed_elements"])
        if "preserve_stereo" in data:
            cfg.preserve_stereo = bool(data["preserve_stereo"])
        return cfg


def _check_stereo(original, candidate, parent_index_map):
    chiral = {
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    }
    for cand_idx, parent_idx in parent_index_map.items():
        if parent_idx >= original.GetNumAtoms() or cand_idx >= candidate.GetNumAtoms():
            continue
        before = original.GetAtomWithIdx(parent_idx).GetChiralTag()
        after = candidate.GetAtomWithIdx(cand_idx).GetChiralTag()
        if before in chiral and after in chiral and before != after:
            return "stereochemistry_flipped"
    return None


def _check_scaffold(original, candidate, tier, parent_index_map, cfg):
    mode = (cfg.scaffold_mode or "off").lower()
    if mode == "off" or tier.upper() in cfg.scaffold_exempt_tiers:
        return None
    if mode == "strict":
        if murcko_smiles(original) != murcko_smiles(candidate):
            return "murcko_scaffold_changed"
        return None
    if mode == "atoms":
        survived = set(parent_index_map.values())
        missing = _scaffold_atom_indices(original) - survived
        if missing:
            return "scaffold_atom_deleted"
        return None
    # ring_systems (default): do not allow ring count to drop
    try:
        before = rdMolDescriptors.CalcNumRings(original)
        after = rdMolDescriptors.CalcNumRings(candidate)
    except Exception:  # noqa: BLE001
        return None
    if after < before:
        return "ring_count_decreased"
    return None


def check_candidate(
    original,
    candidate,
    tier,
    parent_index_map=None,
    n_transforms_applied=0,
    cfg=None,
):
    """Return a rejection reason, or ``None`` when the candidate is allowed."""
    cfg = cfg or ConstraintConfig()
    parent_index_map = parent_index_map or {}
    if candidate is None:
        return "null_candidate"
    if n_transforms_applied >= cfg.max_transforms_per_molecule:
        return "transform_budget_exhausted"
    if cfg.require_single_fragment and len(Chem.GetMolFrags(candidate)) != 1:
        return "fragmented"
    for atom in candidate.GetAtoms():
        if atom.GetAtomicNum() not in cfg.allowed_elements:
            return f"disallowed_element_{atom.GetSymbol()}"

    heavy_before = sum(1 for a in original.GetAtoms() if a.GetAtomicNum() > 1)
    heavy_after = sum(1 for a in candidate.GetAtoms() if a.GetAtomicNum() > 1)
    delta = heavy_after - heavy_before
    if delta > cfg.max_heavy_atom_delta:
        return "heavy_atom_gain_too_large"
    if delta < cfg.min_heavy_atom_delta:
        return "heavy_atom_loss_too_large"

    try:
        mw_before = float(Descriptors.MolWt(original))
        mw_after = float(Descriptors.MolWt(candidate))
    except Exception:  # noqa: BLE001
        mw_before = mw_after = 0.0
    low, high = cfg.mw_window
    if mw_before <= high < mw_after:
        return "mw_above_window"
    if mw_before >= low > mw_after:
        return "mw_below_window"

    if tier.upper() not in cfg.tanimoto_exempt_tiers:
        if tanimoto(original, candidate) < cfg.min_tanimoto_to_parent:
            return "tanimoto_below_floor"

    scaffold_reason = _check_scaffold(original, candidate, tier, parent_index_map, cfg)
    if scaffold_reason is not None:
        return scaffold_reason
    if cfg.preserve_stereo:
        stereo_reason = _check_stereo(original, candidate, parent_index_map)
        if stereo_reason is not None:
            return stereo_reason
    return None
