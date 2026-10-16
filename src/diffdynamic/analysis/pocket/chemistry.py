"""Chemistry responsibilities extracted from evaluate_pocket_quality.py."""

from diffdynamic.analysis.pocket.state import (
    Chem,
    Descriptors,
    HAS_SKLEARN,
    get_chem,
    is_pains,
    np,
    obey_lipinski,
    passes_veber,
)


def evaluate_idea_d_druglikeness(molecules_with_pos):
    """
    想法D：生成分子的药物相似性指标

    Returns:
        dict: {score, qed_mean, qed_std, qed_list, sa_mean, sa_std, sa_list,
               lipinski_mean, lipinski_list, pains_ratio, veber_ratio, n_valid, quality_label, success}
    """
    if (
        not molecules_with_pos
        or get_chem is None
        or is_pains is None
        or obey_lipinski is None
    ):
        return {
            "score": 0.0,
            "qed_mean": None,
            "sa_mean": None,
            "lipinski_mean": None,
            "pains_ratio": None,
            "veber_ratio": None,
            "qed_list": [],
            "sa_list": [],
            "lipinski_list": [],
            "quality_label": "unknown",
            "success": False,
            "message": "无分子数据或 scoring_func 未安装",
        }

    qed_list, sa_list, lipinski_list = [], [], []
    pains_hits = 0
    veber_passes = 0
    n_valid = 0

    for mol, _ in molecules_with_pos:
        if mol is None:
            continue
        try:
            chem = get_chem(mol)
            q = chem.get("qed")
            s = chem.get("sa")
            lip = chem.get("lipinski")
            if q is not None and not (isinstance(q, float) and np.isnan(q)):
                qed_list.append(float(q))
            if s is not None and not (
                isinstance(s, float) and (np.isnan(s) or np.isinf(s))
            ):
                sa_list.append(min(1.0, max(0.0, float(s))))
            if lip is not None and lip != "N/A":
                lipinski_list.append(int(lip) / 5.0)
            if is_pains(mol):
                pains_hits += 1
            if passes_veber is not None:
                rot_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
                if passes_veber(mol, rot_bonds=rot_bonds):
                    veber_passes += 1
            n_valid += 1
        except Exception:
            pass

    if n_valid == 0:
        return {
            "score": 0.0,
            "qed_mean": None,
            "sa_mean": None,
            "lipinski_mean": None,
            "pains_ratio": None,
            "veber_ratio": None,
            "qed_list": [],
            "sa_list": [],
            "lipinski_list": [],
            "quality_label": "unknown",
            "success": False,
            "message": "无有效化学指标",
        }

    qed_mean = float(np.mean(qed_list)) if qed_list else 0.0
    qed_std = float(np.std(qed_list)) if len(qed_list) > 1 else 0.0
    sa_mean = float(np.mean(sa_list)) if sa_list else 0.0
    sa_std = float(np.std(sa_list)) if len(sa_list) > 1 else 0.0
    lipinski_mean = float(np.mean(lipinski_list)) if lipinski_list else 0.0
    pains_ratio = pains_hits / n_valid
    veber_ratio = veber_passes / n_valid

    # 综合分数：QED(35%) + SA(25%) + Lipinski(15%) + Veber(10%) + PAINS惩罚(15%)
    # PAINS 权重从 10% 提升至 15%（假阳性风险高）；新增 Veber 规则
    weights = (0.35, 0.25, 0.15, 0.10, 0.15)
    assert abs(sum(weights) - 1.0) < 1e-6, (
        f"Druglikeness weights must sum to 1.0, got {sum(weights)}"
    )
    score = (
        weights[0] * min(1.0, qed_mean)
        + weights[1] * min(1.0, sa_mean)
        + weights[2] * lipinski_mean
        + weights[3] * veber_ratio
        + weights[4] * (1.0 - pains_ratio)
    )
    # PAINS 命中率 >5% 时软惩罚，提高差集区分度
    if pains_ratio > 0.05:
        score *= 0.85
    score = float(np.clip(score, 0.0, 1.0))

    if score >= 0.6:
        label = "high"
    elif score >= 0.3:
        label = "medium"
    else:
        label = "low"

    return {
        "score": score,
        "qed_mean": qed_mean,
        "qed_std": qed_std,
        "qed_list": qed_list,
        "sa_mean": sa_mean,
        "sa_std": sa_std,
        "sa_list": sa_list,
        "lipinski_mean": lipinski_mean,
        "lipinski_list": lipinski_list,
        "pains_ratio": pains_ratio,
        "veber_ratio": veber_ratio,
        "n_valid": n_valid,
        "quality_label": label,
        "success": True,
        "message": f"QED={qed_mean:.2f}, SA={sa_mean:.2f}, Lipinski={lipinski_mean:.2f}, Veber={veber_ratio * 100:.1f}%, PAINS={pains_ratio * 100:.1f}%",
    }


def _count_complete_molecules_smiles_no_dot(molecules_with_pos):
    """
    完整分子：RDKit ``MolToSmiles`` 中不含 ``'.'``（无多片段/盐桥式断点）。

    Returns:
        int: 满足条件的分子数（``mol is None`` 或无法转 SMILES 的条目不计入）。
    """
    if not molecules_with_pos or Chem is None:
        return 0
    n = 0
    for mol, _ in molecules_with_pos:
        if mol is None:
            continue
        try:
            smi = Chem.MolToSmiles(mol)
            if "." not in smi:
                n += 1
        except Exception:
            continue
    return n


def _idea_e_score_from_rate(rate_for_score):
    """Map yield rate in [0, 1] to quality score (tightened: ≥0.98 full)."""
    r = float(rate_for_score)
    if r >= 0.98:
        return 1.0, "high"
    if r >= 0.90:
        return 0.70 + (r - 0.90) / 0.08 * 0.30, "high"
    if r >= 0.75:
        return 0.40 + (r - 0.75) / 0.15 * 0.30, "medium"
    return (r / 0.75) * 0.40, "low"


def evaluate_idea_e_reconstruction(
    pt_path, molecules_with_pos, expected_n_molecules=None
):
    """
    想法E：完整分子比例（非「重建槽位填满」）。

    分子：``MolToSmiles(mol)`` 中**不含** ``'.'`` 的条目计为完整单组分分子。

    若 ``expected_n_molecules`` 为正整数，分母为该应生成分子数：
    ``reconstruct_rate = n_complete / expected_n_molecules``（字段名沿用 ``reconstruct_rate``，可大于 1）。

    未指定时：有 .pt 则分母为 ``len(pred_ligand_pos)``；无 .pt 则跳过。

    Returns:
        dict: score, reconstruct_rate, n_success（=完整分子数）, n_loaded_molecules, n_total,
              quality_label, success, denominator_source, message
    """
    from diffdynamic.analysis.pocket.inputs import load_pt_file

    n_loaded = len(molecules_with_pos) if molecules_with_pos else 0
    n_complete = _count_complete_molecules_smiles_no_dot(molecules_with_pos)
    n_success = n_complete

    exp = None
    if expected_n_molecules is not None:
        try:
            exp = int(expected_n_molecules)
        except (TypeError, ValueError):
            exp = None

    if exp is not None and exp > 0:
        n_total = exp
        rate = n_success / n_total
        score_eff = min(1.0, rate)
        score, label = _idea_e_score_from_rate(score_eff)
        return {
            "score": min(1.0, score),
            "reconstruct_rate": rate,
            "n_success": n_success,
            "n_loaded_molecules": n_loaded,
            "n_total": n_total,
            "quality_label": label,
            "success": True,
            "denominator_source": "expected_n_molecules",
            "message": (
                f"完整分子比例={rate * 100:.1f}% ({n_success}/{n_total}，分子=SMILES无断点; "
                f"已载入构象 {n_loaded}；分母=应生成分子数)"
            ),
        }

    if pt_path is None:
        return {
            "score": 0.0,
            "reconstruct_rate": None,
            "n_success": n_success,
            "n_loaded_molecules": n_loaded,
            "n_total": n_loaded,
            "quality_label": "unknown",
            "success": False,
            "denominator_source": None,
            "message": "外部配体或无 .pt，且未指定 --idea_e_expected_n_molecules，跳过想法E（不参与综合加权）",
        }
    data = load_pt_file(pt_path) if pt_path else None
    if data is None:
        return {
            "score": 0.0,
            "reconstruct_rate": 0.0,
            "n_success": 0,
            "n_loaded_molecules": 0,
            "n_total": 0,
            "quality_label": "unknown",
            "success": False,
            "denominator_source": None,
            "message": "无法加载 .pt",
        }

    pred_pos = data.get("pred_ligand_pos", [])
    n_total = len(pred_pos) if pred_pos else 0

    if n_total == 0:
        return {
            "score": 0.0,
            "reconstruct_rate": 0.0,
            "n_success": 0,
            "n_loaded_molecules": n_loaded,
            "n_total": 0,
            "quality_label": "unknown",
            "success": False,
            "denominator_source": None,
            "message": "无生成分子（.pt 中 pred_ligand_pos 为空；可改用 --idea_e_expected_n_molecules 指定应生成数）",
        }

    rate = n_success / n_total
    score, label = _idea_e_score_from_rate(rate)

    return {
        "score": min(1.0, score),
        "reconstruct_rate": rate,
        "n_success": n_success,
        "n_loaded_molecules": n_loaded,
        "n_total": n_total,
        "quality_label": label,
        "success": True,
        "denominator_source": "pt_pred_ligand_pos",
        "message": (
            f"完整分子比例={rate * 100:.1f}% ({n_success}/{n_total}，分子=SMILES无断点; "
            f"已载入构象 {n_loaded}；分母=.pt pred_ligand_pos)"
        ),
    }


def evaluate_idea_f_uniqueness(molecules_with_pos):
    """
    想法F：分子唯一性（score 仅由 unique_ratio 决定）。

    满分门槛 unique_full_at=0.95（约 100 个中 ≥95 个不同 SMILES）。
    分段：≥0.95→1.0；[0.70,0.95)→0.70–1.0；[0.40,0.70)→0.30–0.70；<0.40→[0,0.30]。
    ``fingerprint_dissimilarity`` 为参考指标（不进总分）；``tanimoto_diversity`` 为别名。
    """
    from diffdynamic.analysis.pocket.clustering import _compute_morgan_fingerprints

    unique_full_at = 0.95
    if not molecules_with_pos or Chem is None:
        return {
            "score": 0.0,
            "unique_ratio": 0.0,
            "n_unique": 0,
            "n_total": 0,
            "unique_full_at": unique_full_at,
            "fingerprint_dissimilarity": None,
            "tanimoto_diversity": None,
            "fps_matrix": None,
            "quality_label": "unknown",
            "success": False,
            "message": "无分子数据",
        }

    smiles_set = set()
    n_complete = 0
    for mol, _ in molecules_with_pos:
        if mol is None:
            continue
        try:
            smi = Chem.MolToSmiles(mol)
            if smi and "." not in smi:
                smiles_set.add(smi)
                n_complete += 1
        except Exception:
            pass

    if n_complete == 0:
        return {
            "score": 0.0,
            "unique_ratio": 0.0,
            "n_unique": 0,
            "n_total": 0,
            "unique_full_at": unique_full_at,
            "fingerprint_dissimilarity": None,
            "tanimoto_diversity": None,
            "fps_matrix": None,
            "quality_label": "unknown",
            "success": False,
            "message": "无有效 SMILES",
        }

    n_unique = len(smiles_set)
    unique_ratio = n_unique / n_complete

    # Tanimoto/余弦相异度（采样，避免 O(N^2)；不进总分）
    fingerprint_dissimilarity = None
    fps_matrix, _ = _compute_morgan_fingerprints(molecules_with_pos)
    if fps_matrix is not None and len(fps_matrix) >= 2 and HAS_SKLEARN:
        n_fps = len(fps_matrix)
        sample_size = min(n_fps, 200)
        idx = (
            np.random.choice(n_fps, sample_size, replace=False)
            if n_fps > sample_size
            else np.arange(n_fps)
        )
        fps_sample = fps_matrix[idx]
        # 余弦相似度 → 相异度（历史字段曾误称 Tanimoto）
        sim_matrix = (fps_sample @ fps_sample.T) / (
            np.outer(
                np.linalg.norm(fps_sample, axis=1), np.linalg.norm(fps_sample, axis=1)
            )
            + 1e-8
        )
        upper_tri = sim_matrix[np.triu_indices(len(fps_sample), k=1)]
        fingerprint_dissimilarity = float(1.0 - np.mean(upper_tri))

    # ≥95% unique 满分；中等唯一性线性过渡；低唯一性惩罚 mode-collapse
    r = unique_ratio
    if r >= unique_full_at:
        score = 1.0
    elif r >= 0.70:
        score = 0.70 + (r - 0.70) / 0.25 * 0.30  # 0.70→0.70, 0.95→1.0
    elif r >= 0.40:
        score = 0.30 + (r - 0.40) / 0.30 * 0.40  # 0.40→0.30, 0.70→0.70
    else:
        score = r / 0.40 * 0.30  # 0→0, 0.40→0.30

    if score >= 0.6:
        label = "high"
    elif score >= 0.3:
        label = "medium"
    else:
        label = "low"

    return {
        "score": min(1.0, float(score)),
        "unique_ratio": unique_ratio,
        "n_unique": n_unique,
        "n_total": n_complete,
        "unique_full_at": unique_full_at,
        "fingerprint_dissimilarity": fingerprint_dissimilarity,
        "tanimoto_diversity": fingerprint_dissimilarity,  # legacy alias
        "fps_matrix": fps_matrix,
        "quality_label": label,
        "success": True,
        "message": (
            f"唯一性={unique_ratio * 100:.1f}% ({n_unique}/{n_complete}), "
            f"满分线≥{unique_full_at * 100:.0f}%, "
            f"指纹余弦相异度={fingerprint_dissimilarity}"
        ),
    }


def evaluate_idea_g_size_consistency(molecules_with_pos):
    """
    想法G：分子尺寸（原子数、分子量）的一致性

    Returns:
        dict: {score, n_atoms_mean, n_atoms_std, mw_mean, mw_std, mw_cv,
               n_atoms_list, mw_list, quality_label, success}
    """
    if not molecules_with_pos or Chem is None:
        return {
            "score": 0.0,
            "n_atoms_mean": None,
            "mw_mean": None,
            "mw_cv": None,
            "n_atoms_list": [],
            "mw_list": [],
            "quality_label": "unknown",
            "success": False,
            "message": "无分子数据",
        }

    if Descriptors is None:
        return {
            "score": 0.5,
            "n_atoms_mean": None,
            "mw_mean": None,
            "mw_cv": None,
            "n_atoms_list": [],
            "mw_list": [],
            "quality_label": "medium",
            "success": False,
            "message": "无法计算分子量",
        }

    n_atoms_list, mw_list = [], []
    for mol, _ in molecules_with_pos:
        if mol is None:
            continue
        try:
            n_atoms_list.append(mol.GetNumAtoms())
            mw_list.append(Descriptors.ExactMolWt(mol))
        except Exception:
            pass

    if len(mw_list) < 2:
        return {
            "score": 0.5,
            "n_atoms_mean": float(np.mean(n_atoms_list)) if n_atoms_list else None,
            "mw_mean": float(np.mean(mw_list)) if mw_list else None,
            "mw_cv": None,
            "n_atoms_list": n_atoms_list,
            "mw_list": mw_list,
            "quality_label": "medium",
            "success": True,
            "message": "样本过少",
        }

    n_atoms_mean = float(np.mean(n_atoms_list))
    n_atoms_std = float(np.std(n_atoms_list))
    mw_mean = float(np.mean(mw_list))
    mw_std = float(np.std(mw_list))
    mw_cv = mw_std / mw_mean if mw_mean > 0 else 0.0

    # CV 阈值：收紧以提高区分度
    if mw_cv < 0.15:
        score = 1.0
        label = "high"
    elif mw_cv < 0.25:
        score = 0.8
        label = "high"
    elif mw_cv < 0.4:
        score = 0.5
        label = "medium"
    else:
        score = max(0.0, 0.5 - (mw_cv - 0.4) * 0.8)
        label = "low"

    return {
        "score": min(1.0, score),
        "n_atoms_mean": n_atoms_mean,
        "n_atoms_std": n_atoms_std,
        "mw_mean": mw_mean,
        "mw_std": mw_std,
        "mw_cv": mw_cv,
        "n_atoms_list": n_atoms_list,
        "mw_list": mw_list,
        "quality_label": label,
        "success": True,
        "message": f"分子量均值={mw_mean:.0f}, CV={mw_cv:.2f}",
    }
