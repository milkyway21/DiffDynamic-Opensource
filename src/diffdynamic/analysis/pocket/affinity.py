"""Affinity responsibilities extracted from evaluate_pocket_quality.py."""

from diffdynamic.analysis.pocket.state import IDEA_C_LE_SCORE_HIGH, IDEA_C_LE_SCORE_LOW

from diffdynamic.analysis.pocket.state import (
    Chem,
    Path,
    np,
    os,
    pd,
    re,
    torch,
)


def _read_vina_scores_from_excel(excel_path):
    """
    从 complete_molecules_*.xlsx 读取 Vina 分数。
    优先使用 Vina_Dock_亲和力，其次 Vina_Minimize_亲和力，最后 Vina_ScoreOnly_亲和力。
    返回合并后的有效分数列表。
    """
    if pd is None:
        return []
    try:
        xl = pd.ExcelFile(excel_path)
        all_scores = []
        for sheet_name in xl.sheet_names:
            if "_统计信息" in sheet_name:
                continue
            df = pd.read_excel(xl, sheet_name=sheet_name)
            for _, row in df.iterrows():
                v = None
                for col in [
                    "Vina_Dock_亲和力",
                    "Vina_Minimize_亲和力",
                    "Vina_ScoreOnly_亲和力",
                ]:
                    if col not in df.columns:
                        continue
                    val = row[col]
                    if (
                        val is None
                        or val == "N/A"
                        or (isinstance(val, float) and np.isnan(val))
                    ):
                        continue
                    try:
                        v = float(val)
                        break
                    except (ValueError, TypeError):
                        continue
                if v is not None:
                    all_scores.append(v)
        return all_scores
    except Exception:
        return []


def evaluate_idea_a_vina(
    pt_path=None,
    data_id=None,
    wait_timeout=300,
    outputs_dir=None,
    pocket_id_override=None,
):
    """
    想法A：基于 Vina 对接分数评估口袋质量

    Vina 分数来源（按优先级）：
    1. complete_molecules_*.xlsx（Excel 中的 Vina_Dock_亲和力 等列）
    2. eval_results_*.pt（兼容旧格式）

    Args:
        pt_path: 与生成结果同目录的 .pt（用于默认 outputs_dir）；可与 outputs_dir 二选一。
        outputs_dir: 含 eval_* 子目录的文件夹（自定义评估时传入 batch 的 outputs）。
        pocket_id_override: 匹配 eval_{id}_* 的口袋 id（默认 custom 或从 result_* 文件名解析）。

    Returns:
        dict: {score, vina_mean, vina_median, vina_std, vina_best, vina_worst,
               vina_pct_good, num_scores, quality_label, success, message}
    """
    from diffdynamic.analysis.pocket.inputs import _torch_load_legacy

    if outputs_dir is not None:
        outputs_dir = Path(outputs_dir).resolve()
    elif pt_path is not None:
        pt_path = Path(pt_path).resolve()
        outputs_dir = pt_path.parent
    else:
        return {
            "score": 0.0,
            "vina_mean": None,
            "vina_median": None,
            "vina_std": None,
            "num_scores": 0,
            "quality_label": "unknown",
            "success": False,
            "vina_scores": [],
            "message": "未指定 pt_path 或 outputs_dir，无法定位 Vina 评估输出",
        }

    if pocket_id_override is not None:
        pocket_id = str(pocket_id_override)
    elif pt_path is not None:
        pt_stem = Path(pt_path).stem
        if pt_stem.startswith("result_"):
            parts = pt_stem.split("_")
            pocket_id = parts[1] if len(parts) >= 3 else str(data_id or "unknown")
        else:
            pocket_id = str(data_id or "unknown")
    else:
        pocket_id = str(data_id if data_id is not None else "custom")

    # 优先匹配 eval_{pocket_id}_*；batch 常用 eval_<日期>_<时间>_{pocket_id}_gfquadratic_*（须用 eval_*_{id}_*，勿全量 glob eval_*）
    eval_dirs = list(outputs_dir.glob(f"eval_{pocket_id}_*"))
    if not eval_dirs:
        eval_dirs = list(outputs_dir.glob(f"eval_*_{pocket_id}_*"))
    if not eval_dirs:
        eval_dirs = [
            d
            for d in outputs_dir.glob("eval_*")
            if d.is_dir() and f"_{pocket_id}_" in d.name
        ]
    if not eval_dirs:
        eval_dirs = [d for d in outputs_dir.glob("eval_*") if d.is_dir()]

    if not eval_dirs:
        return {
            "score": 0.0,
            "vina_mean": None,
            "vina_median": None,
            "vina_std": None,
            "num_scores": 0,
            "quality_label": "unknown",
            "success": False,
            "vina_scores": [],
            "message": f"未找到评估目录 eval_{pocket_id}_* 或 eval_*_{pocket_id}_*",
        }

    timestamp_pattern = r"_\d{8}_\d{6}"
    new_dirs = [d for d in eval_dirs if re.search(timestamp_pattern, d.name)]
    eval_dir = max(new_dirs if new_dirs else eval_dirs, key=lambda x: x.stat().st_mtime)

    vina_scores = []

    # 1. 优先从 complete_molecules_*.xlsx 读取
    excel_files = list(eval_dir.glob("complete_molecules_*.xlsx")) or list(
        eval_dir.glob("**/complete_molecules_*.xlsx")
    )
    if excel_files:
        latest_excel = max(excel_files, key=os.path.getmtime)
        vina_scores = _read_vina_scores_from_excel(latest_excel)

    # 2. 若 Excel 无有效分数，回退到 eval_results_*.pt
    if not vina_scores and torch is not None:
        eval_files = list(eval_dir.glob("eval_results_*.pt")) or list(
            eval_dir.glob("**/eval_results_*.pt")
        )
        if eval_files:
            try:
                eval_data = _torch_load_legacy(
                    max(eval_files, key=os.path.getmtime), map_location="cpu"
                )
                stats = eval_data.get("statistics", {})
                vina_scores = (
                    stats.get("vina_dock_scores")
                    or stats.get("vina_minimize_scores")
                    or stats.get("vina_score_only_scores")
                    or stats.get("vina_scores")
                    or []
                )
            except Exception:
                pass

    if not vina_scores:
        return {
            "score": 0.0,
            "vina_mean": None,
            "vina_median": None,
            "vina_std": None,
            "num_scores": 0,
            "quality_label": "unknown",
            "success": False,
            "vina_scores": [],
            "message": "未找到 Vina 分数（complete_molecules_*.xlsx 或 eval_results_*.pt 中均无有效数据）",
        }

    try:
        vina_arr = np.array(vina_scores, dtype=np.float64)
        vina_mean = float(np.mean(vina_arr))
        vina_median = float(np.median(vina_arr))
        vina_std = float(np.std(vina_arr)) if len(vina_arr) > 1 else 0.0
        vina_best = float(np.min(vina_arr))
        vina_worst = float(np.max(vina_arr))
        vina_pct_good = float(np.mean(vina_arr <= -7.0))
        n = len(vina_arr)

        # 过滤正值（docking 失败通常返回正值，如 +50 kcal/mol）
        vina_arr_filtered = vina_arr[vina_arr < 0]
        if len(vina_arr_filtered) == 0:
            return {
                "score": 0.0,
                "vina_mean": None,
                "vina_median": None,
                "vina_std": None,
                "num_scores": 0,
                "quality_label": "unknown",
                "success": False,
                "vina_scores": vina_arr.tolist(),
                "message": "所有 Vina 对接均返回正值（可能全部失败）",
            }
        # 使用过滤后的数据重新计算统计量
        vina_mean = float(np.mean(vina_arr_filtered))
        vina_median = float(np.median(vina_arr_filtered))
        vina_std = (
            float(np.std(vina_arr_filtered)) if len(vina_arr_filtered) > 1 else 0.0
        )
        vina_best = float(np.min(vina_arr_filtered))
        vina_worst = float(np.max(vina_arr_filtered))
        vina_pct_good = float(np.mean(vina_arr_filtered <= -7.0))
        n = len(vina_arr_filtered)
        n_bad = len(vina_arr) - n

        # 综合评分：均值(50%) + 最佳分数(30%) + 命中率(20%)
        # 均值反映整体结合质量，最佳分数奖励发现强结合分子，命中率衡量口袋可药性
        mean_component = float(np.clip((-vina_mean - 6) / 6, 0.0, 1.0))
        best_component = float(np.clip((-vina_best - 6) / 6, 0.0, 1.0))
        hit_component = float(vina_pct_good)
        score = 0.5 * mean_component + 0.3 * best_component + 0.2 * hit_component
        score = float(np.clip(score, 0.0, 1.0))

        if score >= 0.5:
            label = "high"
        elif score >= 0.2:
            label = "medium"
        else:
            label = "low"

        return {
            "score": score,
            "vina_mean": vina_mean,
            "vina_median": vina_median,
            "vina_std": vina_std,
            "vina_best": vina_best,
            "vina_worst": vina_worst,
            "vina_pct_good": vina_pct_good,
            "num_scores": n,
            "num_bad_scores": n_bad,
            # 保留与分子下标对齐的完整列表（含失败正值）；想法 C 会跳过 vina>=0
            "vina_scores": vina_arr.tolist(),
            "vina_scores_valid": vina_arr_filtered.tolist(),
            "mean_component": mean_component,
            "best_component": best_component,
            "hit_component": hit_component,
            "quality_label": label,
            "success": True,
            "message": f"Vina 平均={vina_mean:.2f}, 最佳={vina_best:.2f}, 命中率={vina_pct_good * 100:.1f}%, 综合={score:.3f} (均值{mean_component:.2f}*0.5+最佳{best_component:.2f}*0.3+命中{hit_component:.2f}*0.2)",
        }
    except Exception as e:
        return {
            "score": 0.0,
            "vina_mean": None,
            "vina_median": None,
            "vina_std": None,
            "num_scores": 0,
            "quality_label": "unknown",
            "success": False,
            "vina_scores": [],
            "message": str(e),
        }


def _n_heavy_atoms(mol):
    """非氢重原子数。"""
    if mol is None or Chem is None:
        return 0
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def evaluate_idea_c_ligand_efficiency(
    molecules_with_pos,
    vina_scores,
    le_low_for_score=IDEA_C_LE_SCORE_LOW,
    le_high_for_score=IDEA_C_LE_SCORE_HIGH,
):
    """
    配体效率：对每个分子 LE_i = -ΔG_i / N_heavy,i。

    ΔG 采用与想法 A 相同的 Vina 亲和力（kcal/mol，负值表示有利结合），故 LE = -vina_score / N_heavy。

    假设 ``vina_scores[i]`` 与 ``molecules_with_pos[i]`` 一一对应（与 batch 导出顺序一致）；
    仅对 mol 非空、N_heavy > 0 且 ``vina < 0``（对接成功）的条目计算；跳过失败对接以免负 LE。

    Returns:
        dict: score, le_mean, le_median, le_std, le_list, n_valid, n_heavy_per_mol,
              quality_label, success, message, …
    """
    empty = {
        "score": 0.0,
        "le_mean": None,
        "le_median": None,
        "le_std": None,
        "le_list": [],
        "n_heavy_per_mol": [],
        "n_valid": 0,
        "n_molecules": 0,
        "n_vina_scores": 0,
        "n_aligned": 0,
        "n_skipped_failed_vina": 0,
        "quality_label": "unknown",
        "success": False,
        "message": "",
    }

    if not molecules_with_pos or Chem is None:
        empty["message"] = "无分子数据或 RDKit 未安装"
        return empty

    if not vina_scores:
        empty["message"] = "无 Vina 分数（想法A 失败或缺 eval 输出），无法计算 LE"
        return empty

    v_arr = np.asarray(vina_scores, dtype=np.float64).ravel()
    n_mol = len(molecules_with_pos)
    n_v = len(v_arr)
    n_aligned = min(n_mol, n_v)

    le_list = []
    n_heavy_list = []
    n_skipped_failed_vina = 0
    for i in range(n_aligned):
        mol, _ = molecules_with_pos[i]
        if mol is None:
            continue
        nh = _n_heavy_atoms(mol)
        if nh <= 0:
            continue
        dg = float(v_arr[i])
        # 失败对接常返回正值；跳过以保持下标对齐且不产生负 LE
        if not np.isfinite(dg) or dg >= 0.0:
            n_skipped_failed_vina += 1
            continue
        le = -dg / nh
        if not np.isfinite(le):
            continue
        le_list.append(float(le))
        n_heavy_list.append(nh)

    n_valid = len(le_list)
    if n_valid == 0:
        empty["n_molecules"] = n_mol
        empty["n_vina_scores"] = n_v
        empty["n_aligned"] = n_aligned
        empty["n_skipped_failed_vina"] = n_skipped_failed_vina
        empty["message"] = (
            "无有效 (分子, Vina) 对可计算 LE"
            f"（跳过失败对接 {n_skipped_failed_vina}；检查 mol 与重原子数）"
        )
        return empty

    arr = np.array(le_list, dtype=np.float64)
    le_mean = float(np.mean(arr))
    le_median = float(np.median(arr))
    le_std = float(np.std(arr)) if n_valid > 1 else 0.0

    span = le_high_for_score - le_low_for_score
    if span <= 0:
        raw = 0.0
    else:
        raw = (le_mean - le_low_for_score) / span
    score = float(np.clip(raw, 0.0, 1.0))

    if score >= 0.6:
        label = "high"
    elif score >= 0.3:
        label = "medium"
    else:
        label = "low"

    msg_parts = [
        f"LE_mean={le_mean:.4f} kcal·mol⁻¹·重原子⁻¹ (n={n_valid})",
        f"LE_median={le_median:.4f}",
    ]
    if n_skipped_failed_vina:
        msg_parts.append(f"跳过失败对接 {n_skipped_failed_vina}")
    if n_aligned < n_mol or n_aligned < n_v:
        msg_parts.append(f"对齐: 前 {n_aligned} 条 (分子 {n_mol}, Vina {n_v})")
    msg = "; ".join(msg_parts)

    return {
        "score": score,
        "le_mean": le_mean,
        "le_median": le_median,
        "le_std": le_std,
        "le_list": le_list,
        "n_heavy_per_mol": n_heavy_list,
        "n_valid": n_valid,
        "n_molecules": n_mol,
        "n_vina_scores": n_v,
        "n_aligned": n_aligned,
        "n_skipped_failed_vina": n_skipped_failed_vina,
        "le_low_for_score": le_low_for_score,
        "le_high_for_score": le_high_for_score,
        "quality_label": label,
        "success": True,
        "message": msg,
    }
