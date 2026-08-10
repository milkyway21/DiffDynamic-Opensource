"""Drug-likeness reward that drives which medchem transforms get accepted."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

_SA_SCORER = None
_FILTER_CATALOG = None

DEFAULT_WEIGHTS = {
    "w_qed": 1.0,
    "w_sa": 0.6,
    "w_rotb": 0.3,
    "w_alert": 0.5,
    "w_tpsa": 0.2,
    "w_le": 0.3,
    "w_cost": 0.02,
}
DEFAULT_TPSA_WINDOW = (40.0, 140.0)
DEFAULT_MIN_REWARD_GAIN = 0.01
_SA_SPAN = 9.0
_ROTB_SPAN = 5.0
_TPSA_SPAN = 40.0


def _load_sa_scorer():
    global _SA_SCORER
    if _SA_SCORER is not None:
        return _SA_SCORER
    try:
        from rdkit.Chem import RDConfig

        contrib = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if contrib not in sys.path:
            sys.path.append(contrib)
        import sascorer  # type: ignore

        _SA_SCORER = sascorer
    except Exception:  # noqa: BLE001
        _SA_SCORER = False
    return _SA_SCORER


def _load_filter_catalog():
    global _FILTER_CATALOG
    if _FILTER_CATALOG is not None:
        return _FILTER_CATALOG
    try:
        from rdkit.Chem import FilterCatalog

        params = FilterCatalog.FilterCatalogParams()
        for name in ("PAINS", "BRENK", "NIH"):
            params.AddCatalog(
                getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs, name)
            )
        _FILTER_CATALOG = FilterCatalog.FilterCatalog(params)
    except Exception:  # noqa: BLE001
        _FILTER_CATALOG = False
    return _FILTER_CATALOG


def synthetic_accessibility(mol: Chem.Mol) -> float:
    """SA score in RDKit convention: 1 (easy) .. 10 (hard)."""
    scorer = _load_sa_scorer()
    if scorer:
        try:
            return float(scorer.calculateScore(mol))
        except Exception:  # noqa: BLE001
            pass
    # Crude fallback: more heavy atoms / rings → harder
    heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    try:
        nrings = rdMolDescriptors.CalcNumRings(mol)
    except Exception:  # noqa: BLE001
        nrings = 0
    return max(1.0, min(10.0, 1.0 + heavy / 10.0 + nrings * 0.5))


def count_structural_alerts(mol: Chem.Mol) -> int:
    catalog = _load_filter_catalog()
    if catalog:
        try:
            return int(catalog.GetNumMatches(mol))
        except Exception:  # noqa: BLE001
            pass
    patterns = [
        "[N+](=O)[O-]",
        "[CX3H1](=O)",
        "C1OC1",
        "C=CC(=O)",
        "[NH2]c",
    ]
    n = 0
    for sm in patterns:
        try:
            patt = Chem.MolFromSmarts(sm)
            if patt and mol.HasSubstructMatch(patt):
                n += 1
        except Exception:  # noqa: BLE001
            continue
    return n


def compute_properties(mol: Chem.Mol) -> Dict[str, float]:
    """Every property the reward reads, computed once per molecule."""
    props: Dict[str, float] = {}
    try:
        props["qed"] = float(QED.qed(mol))
    except Exception:  # noqa: BLE001
        props["qed"] = 0.0
    props["sa"] = synthetic_accessibility(mol)
    try:
        props["rotb"] = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
    except Exception:  # noqa: BLE001
        props["rotb"] = 0.0
    try:
        props["tpsa"] = float(Descriptors.TPSA(mol))
    except Exception:  # noqa: BLE001
        props["tpsa"] = 0.0
    try:
        props["mw"] = float(Descriptors.MolWt(mol))
    except Exception:  # noqa: BLE001
        props["mw"] = 0.0
    try:
        props["logp"] = float(Crippen.MolLogP(mol))
    except Exception:  # noqa: BLE001
        props["logp"] = 0.0
    props["alerts"] = float(count_structural_alerts(mol))
    props["heavy_atoms"] = float(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1))
    return props


def _window_distance(value: float, low: float, high: float) -> float:
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def _ligand_efficiency_gain(before, after, vina_before, vina_after):
    heavy_before = max(before.get("heavy_atoms", 1.0), 1.0)
    heavy_after = max(after.get("heavy_atoms", 1.0), 1.0)
    if vina_before is not None and vina_after is not None:
        return (-vina_after / heavy_after) - (-vina_before / heavy_before)
    qed_before = before.get("qed", 0.0) / heavy_before
    qed_after = after.get("qed", 0.0) / heavy_after
    return (qed_after - qed_before) * heavy_before


def compute_reward(
    before,
    after,
    transform_cost=0.0,
    weights=None,
    tpsa_window=None,
    vina_before=None,
    vina_after=None,
    transform_bonus=0.0,
):
    """Return ``{"total": float, <term>: float, ...}`` for one candidate."""
    w = dict(DEFAULT_WEIGHTS)
    w.update(weights or {})
    low, high = tuple(tpsa_window or DEFAULT_TPSA_WINDOW)
    qed_term = w["w_qed"] * (after.get("qed", 0.0) - before.get("qed", 0.0))
    sa_term = w["w_sa"] * (before.get("sa", 5.0) - after.get("sa", 5.0)) / _SA_SPAN
    rotb_term = w["w_rotb"] * (before.get("rotb", 0.0) - after.get("rotb", 0.0)) / _ROTB_SPAN
    alert_term = w["w_alert"] * (before.get("alerts", 0.0) - after.get("alerts", 0.0))
    tpsa_gain = (
        _window_distance(before.get("tpsa", 0.0), low, high)
        - _window_distance(after.get("tpsa", 0.0), low, high)
    ) / _TPSA_SPAN
    tpsa_term = w["w_tpsa"] * tpsa_gain
    le_term = w["w_le"] * _ligand_efficiency_gain(
        before, after, vina_before, vina_after
    )
    cost_term = -w["w_cost"] * float(transform_cost)
    breakdown = {
        "qed": round(qed_term, 5),
        "sa": round(sa_term, 5),
        "rotb": round(rotb_term, 5),
        "alerts": round(alert_term, 5),
        "tpsa": round(tpsa_term, 5),
        "le": round(le_term, 5),
        "cost": round(cost_term, 5),
        "bonus": round(float(transform_bonus), 5),
    }
    breakdown["total"] = round(sum(breakdown.values()), 5)
    return breakdown


def weights_from_config(config: Dict[str, Any]) -> Dict[str, float]:
    section = (config or {}).get("reward", {}) or {}
    weights = dict(DEFAULT_WEIGHTS)
    for key in DEFAULT_WEIGHTS:
        if key in section:
            weights[key] = float(section[key])
    return weights
