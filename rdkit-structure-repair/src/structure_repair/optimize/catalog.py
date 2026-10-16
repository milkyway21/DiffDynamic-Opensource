"""Load the tiered medchem transform catalog from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml
from rdkit import Chem
from rdkit.Chem import AllChem

PathLike = Union[str, Path]

_CATALOG_DIR = Path(__file__).resolve().parent / "catalog"

# T0 is handled programmatically (Kekule enumeration + planarity evidence)
# rather than by reaction SMARTS, because aromatization needs implicit-H
# reassignment that reaction SMARTS handles poorly.
TIER_ORDER = ("T0", "T1", "T2", "T3", "T4", "T5")


@dataclass
class TransformSpec:
    transform_id: str
    tier: str
    name: str
    rationale: str
    smarts: str = ""
    delta_heavy: int = 0
    requires_3d: bool = True
    cost: float = 1.0
    # Credit for medchem value that QED / SA / alert counts cannot see, e.g.
    # blocking a hydrolysis site.  Zero everywhere by default so the layer
    # stays purely metric-driven unless a user opts in.
    reward_bonus: float = 0.0
    enabled: bool = True
    max_applications: int = 1
    positive_examples: List[str] = field(default_factory=list)
    negative_examples: List[str] = field(default_factory=list)
    _reaction: Optional[AllChem.ChemicalReaction] = field(
        default=None, repr=False, compare=False
    )

    @property
    def reaction(self) -> Optional[AllChem.ChemicalReaction]:
        if self._reaction is None and self.smarts:
            try:
                rxn = AllChem.ReactionFromSmarts(self.smarts)
                rxn.Initialize()
                self._reaction = rxn
            except Exception:  # noqa: BLE001
                return None
        return self._reaction

    def is_valid(self) -> bool:
        if not self.smarts:
            return True  # programmatic transform (T0)
        return self.reaction is not None


@dataclass
class TierPolicy:
    """Declarative policy for the programmatic T0 aromatization tier."""

    tier: str = "T0"
    data: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


def catalog_dir() -> Path:
    return _CATALOG_DIR


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_tier_policy(tier: str = "T0", catalog_path: Optional[PathLike] = None) -> TierPolicy:
    root = Path(catalog_path) if catalog_path else _CATALOG_DIR
    for path in sorted(root.glob("*.yaml")):
        data = _load_yaml(path)
        if str(data.get("tier", "")).upper() == tier.upper():
            return TierPolicy(tier=tier.upper(), data=data.get("policy", {}) or {})
    return TierPolicy(tier=tier.upper(), data={})


def load_catalog(
    enabled_tiers: Optional[Sequence[str]] = None,
    catalog_path: Optional[PathLike] = None,
    disabled_transform_ids: Optional[Sequence[str]] = None,
) -> List[TransformSpec]:
    """Load every SMARTS-driven transform for the requested tiers.

    Invalid reaction SMARTS are dropped rather than raising, so a typo in one
    catalog entry cannot take down a whole batch run.
    """
    root = Path(catalog_path) if catalog_path else _CATALOG_DIR
    # None means "every tier"; an empty sequence means "no tier at all".  The
    # engine passes the tier list minus T0, so a T0-only run arrives here empty
    # and must not be read as an unrestricted load.
    wanted = None if enabled_tiers is None else {t.upper() for t in enabled_tiers}
    blocked = {t for t in (disabled_transform_ids or [])}

    specs: List[TransformSpec] = []
    seen_ids = set()
    for path in sorted(root.glob("*.yaml")):
        data = _load_yaml(path)
        tier = str(data.get("tier", "")).upper()
        if not tier:
            continue
        if wanted is not None and tier not in wanted:
            continue
        for entry in data.get("transforms", []) or []:
            tid = str(entry.get("id", "")).strip()
            if not tid or tid in seen_ids or tid in blocked:
                continue
            if not entry.get("enabled", True):
                continue
            spec = TransformSpec(
                transform_id=tid,
                tier=tier,
                name=str(entry.get("name", tid)),
                rationale=str(entry.get("rationale", "")),
                smarts=str(entry.get("smarts", "")),
                delta_heavy=int(entry.get("delta_heavy", 0)),
                requires_3d=bool(entry.get("requires_3d", True)),
                cost=float(entry.get("cost", 1.0)),
                reward_bonus=float(entry.get("reward_bonus", 0.0)),
                enabled=True,
                max_applications=int(entry.get("max_applications", 1)),
                positive_examples=list(entry.get("positive_examples", [])),
                negative_examples=list(entry.get("negative_examples", [])),
            )
            if not spec.is_valid():
                continue
            seen_ids.add(tid)
            specs.append(spec)

    specs.sort(key=lambda s: (TIER_ORDER.index(s.tier) if s.tier in TIER_ORDER else 99, s.transform_id))
    return specs


def validate_catalog(catalog_path: Optional[PathLike] = None) -> Dict[str, List[str]]:
    """Return ``{"ok": [...], "bad_smarts": [...], "bad_delta_heavy": [...]}``.

    Used by tests and by ``structure-repair optimize --validate-catalog``.
    """
    root = Path(catalog_path) if catalog_path else _CATALOG_DIR
    report: Dict[str, List[str]] = {"ok": [], "bad_smarts": [], "duplicate_id": [],
                                   "bad_delta_heavy": [], "bad_mapping": [],
                                   "bad_examples": []}
    seen = set()
    for path in sorted(root.glob("*.yaml")):
        data = _load_yaml(path)
        tier = str(data.get("tier", "")).upper()
        for entry in data.get("transforms", []) or []:
            tid = str(entry.get("id", "")).strip()
            if not tid:
                continue
            if tid in seen:
                report["duplicate_id"].append(tid)
                continue
            seen.add(tid)
            smarts = str(entry.get("smarts", ""))
            if not smarts:
                report["ok"].append(f"{tier}:{tid}")
                continue
            try:
                rxn = AllChem.ReactionFromSmarts(smarts)
                rxn.Initialize()
                if rxn.GetNumReactantTemplates() < 1 or rxn.GetNumProductTemplates() < 1:
                    raise ValueError("empty template")
                if rxn.GetNumReactantTemplates() != 1 or rxn.GetNumProductTemplates() != 1:
                    raise ValueError("requires one reactant and one product")
                before, after = rxn.GetReactantTemplate(0), rxn.GetProductTemplate(0)
                delta = sum(a.GetAtomicNum() != 1 for a in after.GetAtoms()) - sum(a.GetAtomicNum() != 1 for a in before.GetAtoms())
                if delta != int(entry.get("delta_heavy", 0)):
                    report["bad_delta_heavy"].append(f"{tid}: declared {entry.get('delta_heavy')}, actual {delta}")
                maps_before = [a.GetAtomMapNum() for a in before.GetAtoms() if a.GetAtomMapNum()]
                maps_after = [a.GetAtomMapNum() for a in after.GetAtoms() if a.GetAtomMapNum()]
                if (len(set(maps_before)) != len(maps_before)
                        or len(set(maps_after)) != len(maps_after)
                        or not set(maps_after).issubset(maps_before)):
                    report["bad_mapping"].append(tid)
                from .transforms import apply_transform, prepare_parent
                spec = TransformSpec(transform_id=tid, tier=tier, name=tid, rationale="validation",
                                     smarts=smarts, delta_heavy=int(entry.get("delta_heavy", 0)))
                for key, expected in (("positive_examples", True), ("negative_examples", False)):
                    examples = entry.get(key, [])
                    if not examples:
                        report["bad_examples"].append(f"{tid}: missing {key}")
                    for example in examples:
                        mol = Chem.MolFromSmiles(example)
                        candidates = apply_transform(prepare_parent(mol), spec) if mol else []
                        if bool(candidates) != expected:
                            report["bad_examples"].append(f"{tid}: {key}: {example}")
                        for candidate in candidates:
                            if candidate.delta_heavy != spec.delta_heavy:
                                report["bad_delta_heavy"].append(f"{tid}: {example}")
                report["ok"].append(f"{tier}:{tid}")
            except Exception as exc:  # noqa: BLE001
                report["bad_smarts"].append(f"{tier}:{tid}: {exc}")
    return report
