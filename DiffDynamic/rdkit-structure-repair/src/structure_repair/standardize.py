"""Optional RDKit MolStandardize post-processing (deterministic)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from rdkit import Chem


def apply_standardize(mol: Chem.Mol, config: Dict[str, Any]) -> Optional[Chem.Mol]:
    """Apply configured MolStandardize steps. Returns None on failure."""
    std_cfg = config.get("standardize") or {}
    if not std_cfg.get("enable", False):
        return Chem.Mol(mol)

    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except Exception:  # noqa: BLE001
        return Chem.Mol(mol)

    out = Chem.Mol(mol)
    try:
        if std_cfg.get("cleanup", True):
            out = rdMolStandardize.Cleanup(out)
        if std_cfg.get("largest_fragment", True):
            lfc = rdMolStandardize.LargestFragmentChooser()
            out = lfc.choose(out)
        if std_cfg.get("normalize", True):
            normalizer = rdMolStandardize.Normalizer()
            out = normalizer.normalize(out)
        if std_cfg.get("reionize", False):
            reionizer = rdMolStandardize.Reionizer()
            out = reionizer.reionize(out)
        if std_cfg.get("uncharge", False):
            uncharger = rdMolStandardize.Uncharger()
            out = uncharger.uncharge(out)
        out.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(out)
        Chem.AssignStereochemistry(out, cleanIt=True, force=True)
        return out
    except Exception:  # noqa: BLE001
        return None
