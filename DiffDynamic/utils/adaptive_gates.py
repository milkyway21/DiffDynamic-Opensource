"""根据参考配体（及可选活性分子库）自适应放宽 Prudent 硬门控。

设计原则：
- 以参考配体性质为锚点；有 actives CSV 时一并纳入分位数/极值
- 生成分子通常比配体更噪、更脂、QED 更低，因此加 generation slack
- 返回可直接写入 sample.scaffold.prudent 的键值，并附带 rationale

典型用法::

    from utils.adaptive_gates import compute_adaptive_gates, apply_gates_to_prudent_cfg
    gates = compute_adaptive_gates(ligand_sdf, actives_csv=...)
    prudent_cfg = apply_gates_to_prudent_cfg(prudent_cfg, gates)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

DEFAULT_MARGINS = {
    # QED/SA：相对参考配体最小值再下探（生成分子通常更差）
    "qed_slack": 0.25,
    "sa_slack": 0.25,
    "min_qed_floor": 0.08,
    "min_sa_floor": 0.15,
    # LogP：相对配体上放（保留侧链 + 生长后常显著升高）
    "logp_slack": 5.0,
    "max_logp_floor": 6.5,
    # 尺寸
    "molwt_slack": 300.0,
    "molwt_scale": 1.40,
    "heavy_slack": 25,
    "rings_slack": 5,
    # 绝对安全上限（防止无限放宽）
    "max_logp_cap": 9.0,
    "max_molwt_cap": 1100.0,
    "max_heavy_cap": 90,
    "max_rings_cap": 12,
}


@dataclass
class LigandProps:
    n_heavy: int
    molwt: float
    logp: float
    qed: float
    sa: float
    rings: int
    smiles: Optional[str] = None
    source: str = ""


@dataclass
class AdaptiveGatesResult:
    min_qed_for_docking: float
    min_sa_for_docking: float
    max_logp: float
    max_molwt: float
    max_heavy_atoms: int
    max_rings: int
    ligand: Dict[str, Any] = field(default_factory=dict)
    actives_stats: Optional[Dict[str, Any]] = None
    margins: Dict[str, Any] = field(default_factory=dict)
    rationale: List[str] = field(default_factory=list)

    def as_prudent_dict(self) -> Dict[str, Any]:
        return {
            "min_qed_for_docking": float(self.min_qed_for_docking),
            "min_sa_for_docking": float(self.min_sa_for_docking),
            "max_logp": float(self.max_logp),
            "max_molwt": float(self.max_molwt),
            "max_heavy_atoms": int(self.max_heavy_atoms),
            "max_rings": int(self.max_rings),
        }

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False, default=str)


def _sa_score(mol) -> float:
    """项目约定：越高越好，(10 - SA)/9。"""
    import os
    import sys

    from rdkit.Chem import RDConfig

    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer  # noqa: E402

    return float((10.0 - sascorer.calculateScore(mol)) / 9.0)


def compute_mol_props(mol, source: str = "") -> LigandProps:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, QED

    mh = Chem.RemoveHs(mol)
    Chem.SanitizeMol(mh)
    return LigandProps(
        n_heavy=int(mh.GetNumHeavyAtoms()),
        molwt=float(Descriptors.MolWt(mh)),
        logp=float(Crippen.MolLogP(mh)),
        qed=float(QED.qed(mh)),
        sa=_sa_score(mh),
        rings=int(Descriptors.RingCount(mh)),
        smiles=Chem.MolToSmiles(mh),
        source=source,
    )


def load_ligand_props(ligand_path: Union[str, Path]) -> LigandProps:
    from rdkit import Chem

    path = Path(ligand_path)
    mol = Chem.SDMolSupplier(str(path), removeHs=False)[0]
    if mol is None:
        raise ValueError(f"cannot read ligand SDF: {path}")
    return compute_mol_props(mol, source=str(path))


def load_actives_props(actives_csv: Union[str, Path]) -> List[LigandProps]:
    """从 active_dock_qed_sa.csv（需含 smiles；qed/sa 可选）读取活性分子性质。"""
    import pandas as pd
    from rdkit import Chem

    df = pd.read_csv(actives_csv)
    if "smiles" not in df.columns:
        raise ValueError(f"{actives_csv} missing smiles column")
    out: List[LigandProps] = []
    for _, r in df.iterrows():
        mol = Chem.MolFromSmiles(str(r["smiles"]))
        if mol is None:
            continue
        props = compute_mol_props(mol, source=str(actives_csv))
        # 优先用表内已有 qed/sa（与对接流水线一致）
        if "qed" in df.columns and pd.notna(r.get("qed")):
            props.qed = float(r["qed"])
        if "sa" in df.columns and pd.notna(r.get("sa")):
            props.sa = float(r["sa"])
        out.append(props)
    return out


def _actives_stats(actives: List[LigandProps]) -> Dict[str, Any]:
    arr = {
        "qed": np.array([a.qed for a in actives], dtype=float),
        "sa": np.array([a.sa for a in actives], dtype=float),
        "logp": np.array([a.logp for a in actives], dtype=float),
        "molwt": np.array([a.molwt for a in actives], dtype=float),
        "heavy": np.array([a.n_heavy for a in actives], dtype=float),
        "rings": np.array([a.rings for a in actives], dtype=float),
    }
    stats = {"n": len(actives)}
    for k, v in arr.items():
        stats[k] = {
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "mean": float(np.mean(v)),
            "p05": float(np.quantile(v, 0.05)),
            "p95": float(np.quantile(v, 0.95)),
        }
    return stats


def compute_adaptive_gates(
    ligand_path: Union[str, Path],
    actives_csv: Optional[Union[str, Path]] = None,
    margins: Optional[Dict[str, Any]] = None,
    grow_cfg: Optional[Dict[str, Any]] = None,
    base_prudent: Optional[Dict[str, Any]] = None,
) -> AdaptiveGatesResult:
    """由配体（+活性库）计算自适应门控。

    Args:
        ligand_path: 参考配体 SDF
        actives_csv: 可选活性分子 CSV
        margins: 覆盖 DEFAULT_MARGINS
        grow_cfg: 若含 ligand_size_max / n_extra_max_clamp，纳入 heavy 上限
        base_prudent: 若提供，最终门控取 max/min 与原配置的并集（只放宽不收紧）
    """
    m = {**DEFAULT_MARGINS, **(margins or {})}
    lig = load_ligand_props(ligand_path)
    actives: List[LigandProps] = []
    astats = None
    if actives_csv and Path(actives_csv).exists():
        actives = load_actives_props(actives_csv)
        if actives:
            astats = _actives_stats(actives)

    rationale: List[str] = []

    # ---- QED / SA：相对配体与活性最小值再下探 ----
    qed_anchor = lig.qed
    sa_anchor = lig.sa
    if astats:
        qed_anchor = min(qed_anchor, astats["qed"]["min"])
        sa_anchor = min(sa_anchor, astats["sa"]["min"])
        rationale.append(
            f"QED/SA anchor=min(ligand, actives_min)=({qed_anchor:.3f},{sa_anchor:.3f})"
        )
    else:
        rationale.append(f"QED/SA anchor=ligand=({qed_anchor:.3f},{sa_anchor:.3f})")

    min_qed = max(float(m["min_qed_floor"]), float(qed_anchor) - float(m["qed_slack"]))
    min_sa = max(float(m["min_sa_floor"]), float(sa_anchor) - float(m["sa_slack"]))

    # ---- LogP：相对配体/活性最大值上放 ----
    logp_anchor = lig.logp
    if astats:
        logp_anchor = max(logp_anchor, astats["logp"]["max"])
    max_logp = max(float(m["max_logp_floor"]), float(logp_anchor) + float(m["logp_slack"]))
    max_logp = min(max_logp, float(m["max_logp_cap"]))
    rationale.append(f"max_logp=min(cap, max(floor, anchor+slack))={max_logp:.2f}")

    # ---- MolWt / heavy / rings ----
    molwt_anchor = lig.molwt
    heavy_anchor = lig.n_heavy
    rings_anchor = lig.rings
    if astats:
        molwt_anchor = max(molwt_anchor, astats["molwt"]["max"])
        heavy_anchor = max(heavy_anchor, astats["heavy"]["max"])
        rings_anchor = max(rings_anchor, astats["rings"]["max"])

    max_molwt = max(
        float(molwt_anchor) + float(m["molwt_slack"]),
        float(molwt_anchor) * float(m["molwt_scale"]),
    )
    max_molwt = min(max_molwt, float(m["max_molwt_cap"]))

    grow_cfg = grow_cfg or {}
    size_max = grow_cfg.get("ligand_size_max")
    n_extra_max = grow_cfg.get("n_extra_max_clamp")
    heavy_candidates = [heavy_anchor + float(m["heavy_slack"])]
    if size_max is not None:
        heavy_candidates.append(float(size_max) + float(m["heavy_slack"]) * 0.5)
    if n_extra_max is not None:
        # 骨架≈配体骨架子集，粗估：heavy_anchor + n_extra_max
        heavy_candidates.append(float(heavy_anchor) + float(n_extra_max))
    max_heavy = int(min(max(heavy_candidates), float(m["max_heavy_cap"])))
    max_rings = int(min(rings_anchor + float(m["rings_slack"]), float(m["max_rings_cap"])))

    rationale.append(
        f"size anchors molwt/heavy/rings="
        f"({molwt_anchor:.0f},{heavy_anchor:.0f},{rings_anchor:.0f}) → "
        f"({max_molwt:.0f},{max_heavy},{max_rings})"
    )

    # ---- 与原配置取并集：只放宽不收紧 ----
    # min_* 取更低；max_* 取更高
    base = base_prudent or {}
    if base.get("min_qed_for_docking") is not None:
        old = float(base["min_qed_for_docking"])
        merged = min(old, min_qed)
        if merged != min_qed:
            rationale.append(f"merge min_qed: adaptive={min_qed:.3f} base={old} → {merged:.3f}")
        min_qed = merged
    if base.get("min_sa_for_docking") is not None:
        old = float(base["min_sa_for_docking"])
        merged = min(old, min_sa)
        if merged != min_sa:
            rationale.append(f"merge min_sa: adaptive={min_sa:.3f} base={old} → {merged:.3f}")
        min_sa = merged
    for key, new_val, caster in (
        ("max_logp", max_logp, float),
        ("max_molwt", max_molwt, float),
        ("max_heavy_atoms", max_heavy, int),
        ("max_rings", max_rings, int),
    ):
        if base.get(key) is not None:
            old = caster(base[key])
            merged = caster(max(old, new_val))
            if merged != new_val:
                rationale.append(f"merge {key}: adaptive={new_val} base={old} → {merged}")
            if key == "max_logp":
                max_logp = merged
            elif key == "max_molwt":
                max_molwt = merged
            elif key == "max_heavy_atoms":
                max_heavy = merged
            else:
                max_rings = merged

    return AdaptiveGatesResult(
        min_qed_for_docking=float(min_qed),
        min_sa_for_docking=float(min_sa),
        max_logp=float(max_logp),
        max_molwt=float(max_molwt),
        max_heavy_atoms=int(max_heavy),
        max_rings=int(max_rings),
        ligand=asdict(lig),
        actives_stats=astats,
        margins=m,
        rationale=rationale,
    )


def apply_gates_to_prudent_cfg(
    prudent_cfg: Dict[str, Any],
    gates: AdaptiveGatesResult,
) -> Dict[str, Any]:
    """返回合并后的 prudent 配置副本。"""
    out = dict(prudent_cfg or {})
    out.update(gates.as_prudent_dict())
    out["_adaptive_gates"] = {
        "applied": True,
        "rationale": list(gates.rationale),
        "ligand": gates.ligand,
        "actives_stats": gates.actives_stats,
    }
    return out


def write_gates_json(gates: AdaptiveGatesResult, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(gates.to_json())
    return path


def write_gates_yaml_snippet(gates: AdaptiveGatesResult, path: Union[str, Path]) -> Path:
    """写出可粘贴进 sampling YAML 的 prudent 门控片段。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = gates.as_prudent_dict()
    lines = ["# auto-generated by utils.adaptive_gates", "prudent:"]
    for k, v in d.items():
        lines.append(f"  {k}: {v}")
    lines.append("# rationale:")
    for r in gates.rationale:
        lines.append(f"# - {r}")
    path.write_text("\n".join(lines) + "\n")
    return path
