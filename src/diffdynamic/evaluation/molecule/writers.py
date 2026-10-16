"""Writers responsibilities extracted from evaluate_pt_with_correct_reconstruct.py."""

from diffdynamic.evaluation.molecule.state import (
    Path,
    _OPENPYXL_MISSING_MSG,
    _PANDAS_MISSING_MSG,
    datetime,
    json,
    np,
    os,
    pd,
    time,
    torch,
    traceback,
)
from diffdynamic.evaluation.molecule import state as _state


class _xlsx_file_lock:
    """跨进程 xlsx 读-改-写互斥锁（fcntl）。

    背景：evalresults record.xlsx / sampling_history.xlsx 被并发评估进程
    同时"读取→追加→覆盖写"，曾造成记录重复累积与 zip 撕裂损坏。
    """

    def __init__(self, path, timeout=120.0):
        self.lock_path = Path(str(path) + ".lock")
        self.timeout = timeout
        self._fd = None

    def __enter__(self):
        import fcntl

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = open(self.lock_path, "w")
        deadline = time.time() + self.timeout
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                if time.time() > deadline:
                    self._fd.close()
                    raise TimeoutError(f"获取 {self.lock_path} 超时")
                time.sleep(0.2)

    def __exit__(self, *exc):
        import fcntl

        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            if self._fd is not None:
                self._fd.close()
                self._fd = None
        return False


def _atomic_excel_write(df, path):
    """写入临时文件后 os.replace，保证任何时刻磁盘上的 xlsx 都是完整的。"""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    df.to_excel(tmp, index=False, engine="openpyxl")
    os.replace(tmp, path)


def _json_float(v):
    """Cast to a JSON-safe float (NaN/inf → None)."""
    try:
        f = float(v)
        return f if (f == f and abs(f) != float("inf")) else None
    except (TypeError, ValueError):
        return None


def _mol_json_row(idx, r):
    """JSON-safe per-molecule summary row (scalars only; RDKit mols excluded)."""
    if not isinstance(r, dict):
        return {"idx": idx}
    row = {"idx": idx, "success": bool(r.get("success"))}
    smi = r.get("smiles")
    row["smiles"] = smi if isinstance(smi, str) else None
    for key in ("vina_dock", "vina_score_only", "vina_minimize"):
        v = r.get(key)
        val = None
        if isinstance(v, dict):
            val = _json_float(v.get("affinity"))
        elif isinstance(v, (int, float)):
            val = _json_float(v)
        elif isinstance(v, list) and v:
            first = v[0]
            if isinstance(first, dict):
                val = _json_float(first.get("affinity"))
            else:
                val = _json_float(first)
        row[key] = val
    chem = r.get("chem")
    if isinstance(chem, dict):
        row["qed"] = _json_float(chem.get("qed"))
        row["sa"] = _json_float(chem.get("sa"))
    row["comprehensive_score"] = _json_float(r.get("comprehensive_score"))
    return row


def _write_eval_summary_sidecar(eval_output_dir, output_data):
    """Write summary.json next to the final .pt for fast UI/server reads.

    The web backend reads this sidecar instead of torch.load-ing the (often
    very large) result file.
    """
    try:
        stats = output_data.get("statistics") or {}
        vina = [
            float(x)
            for x in (stats.get("vina_score_only_scores") or [])
            if x is not None
        ]
        if not vina:
            vina = [
                float(x) for x in (stats.get("vina_dock_scores") or []) if x is not None
            ]
        summary = {
            "num_molecules": output_data.get("n_eval_success"),
            "n_reconstruct_success": output_data.get("n_reconstruct_success"),
            "n_eval_success": output_data.get("n_eval_success"),
            "n_complete": output_data.get("n_complete"),
            "mean_vina": round(sum(vina) / len(vina), 2) if vina else None,
            "best_vina": round(min(vina), 2) if vina else None,
            "saved_at": output_data.get("saved_at"),
        }
        (Path(eval_output_dir) / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass


def _remove_intermediate_files(eval_output_dir, eval_timestamp):
    """Delete intermediate snapshots after the final result is written."""
    for suffix in ("_intermediate_latest.json", "_intermediate_latest.pt"):
        path = Path(eval_output_dir) / f"eval_results_{eval_timestamp}{suffix}"
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


def save_intermediate_results(
    eval_output_dir,
    eval_timestamp,
    output_data,
    pt_path,
    ligand_filename,
    protein_root,
    atom_mode,
    exhaustiveness,
    num_samples,
    n_reconstruct_success,
    n_eval_success,
    n_complete,
    data_id,
    results,
    vina_dock_scores,
    vina_score_only_scores,
    vina_minimize_scores,
    qed_values,
    sa_values,
    atom_type_jsd_values,
    bond_length_jsd_all,
    pair_length_jsd_all,
    n_reconstruct_fail,
    failure_stats,
    reconstruct_failures,
    is_final=False,
):
    """
    保存中间或最终评估结果

    中间结果写为轻量 JSON（仅标量汇总行）——避免每 N 个分子对完整 results
    做一次全量深拷贝+torch序列化；最终结果仍写完整 .pt，并附 summary.json
    sidecar 供 Web 端快速读取，随后清理中间文件。

    Args:
        is_final: 是否为最终保存（True）还是中间保存（False）
    """
    if not eval_output_dir:
        return False

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not is_final:
            # ── 轻量中间快照（JSON，覆盖写） ──────────────────────────
            result_file = (
                eval_output_dir
                / f"eval_results_{eval_timestamp}_intermediate_latest.json"
            )
            payload = {
                "pt_file": str(pt_path),
                "ligand_filename": ligand_filename,
                "data_id": data_id,
                "reconstruction_method": _state._RECONSTRUCTION_METHOD,
                "num_samples": num_samples,
                "n_reconstruct_success": n_reconstruct_success,
                "n_eval_success": n_eval_success,
                "n_complete": n_complete,
                "n_reconstruct_fail": n_reconstruct_fail,
                "saved_at": timestamp,
                "is_intermediate": True,
                "molecules": [_mol_json_row(i, r) for i, r in enumerate(results)],
                "statistics": {
                    "vina_dock_scores": [float(x) for x in (vina_dock_scores or [])],
                    "vina_score_only_scores": [
                        float(x) for x in (vina_score_only_scores or [])
                    ],
                    "vina_minimize_scores": [
                        float(x) for x in (vina_minimize_scores or [])
                    ],
                    "qed_mean": float(np.mean(qed_values)) if qed_values else None,
                    "sa_mean": float(np.mean(sa_values)) if sa_values else None,
                    "atom_type_jsd_mean": float(np.mean(atom_type_jsd_values))
                    if atom_type_jsd_values
                    else None,
                },
                "reconstruct_statistics": {
                    "n_reconstruct_fail": n_reconstruct_fail,
                    "reconstruct_success_rate": n_reconstruct_success / num_samples
                    if num_samples > 0
                    else 0,
                    "failure_stats": failure_stats,
                },
            }
            result_file.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            print(f"\n💾 中间进度已保存(JSON): {result_file}")
            return True

        # ── 最终保存（完整 .pt + summary sidecar） ─────────────────────
        result_file = (
            eval_output_dir / f"eval_results_{eval_timestamp}_final_{timestamp}.pt"
        )
        final_data = (
            output_data
            if output_data is not None
            else {
                "results": results,
                "statistics": {
                    "vina_dock_scores": vina_dock_scores,
                    "vina_score_only_scores": vina_score_only_scores,
                    "vina_minimize_scores": vina_minimize_scores,
                    "qed_mean": np.mean(qed_values) if qed_values else None,
                    "sa_mean": np.mean(sa_values) if sa_values else None,
                    "atom_type_jsd_mean": np.mean(atom_type_jsd_values)
                    if atom_type_jsd_values
                    else None,
                },
            }
        )
        final_data.setdefault("saved_at", timestamp)
        torch.save(final_data, result_file)
        _write_eval_summary_sidecar(eval_output_dir, final_data)
        _remove_intermediate_files(eval_output_dir, eval_timestamp)
        print(f"\n✅ 结果已保存至: {result_file}")
        return True
    except Exception as e:
        print(f"\n⚠️  保存{'最终' if is_final else '中间'}结果失败: {e}")
        traceback.print_exc()
        return False


def _build_evaluation_record(
    result, pt_file, ligand_filename, atom_mode, exhaustiveness
):
    """从单个评估结果构建Excel记录行（与evaluation_results格式一致）"""
    vina_dock_affinity = "N/A"
    vina_dock_rmsd_lb = "N/A"
    vina_dock_rmsd_ub = "N/A"
    if result.get("vina_dock") and len(result["vina_dock"]) > 0:
        vina_dock_result = result["vina_dock"][0]
        vina_dock_affinity = vina_dock_result["affinity"]
        vina_dock_rmsd_lb = vina_dock_result.get("rmsd_lb", "N/A")
        vina_dock_rmsd_ub = vina_dock_result.get("rmsd_ub", "N/A")
    vina_score_only_affinity = result.get("vina_score_only", "N/A")
    vina_minimize_affinity = result.get("vina_minimize", "N/A")
    smiles = result.get("smiles", "N/A")
    mol_idx = result.get("mol_idx", "N/A")
    molecule_id = result.get("molecule_id", "N/A")
    chem = result.get("chem", {})
    qed = chem.get("qed", "N/A") if chem else "N/A"
    sa = chem.get("sa", "N/A") if chem else "N/A"
    atom_type_jsd = result.get("atom_type_jsd", "N/A")
    stability = result.get("stability", {})
    molecule_stable = stability.get("molecule_stable", "N/A") if stability else "N/A"
    nr_stable_bonds = stability.get("nr_stable_bonds", "N/A") if stability else "N/A"
    n_atoms_stability = stability.get("n_atoms", "N/A") if stability else "N/A"
    basic_info = result.get("basic_info", {})
    n_atoms = basic_info.get("n_atoms", "N/A") if basic_info else "N/A"
    n_bonds = basic_info.get("n_bonds", "N/A") if basic_info else "N/A"
    n_rings = basic_info.get("n_rings", "N/A") if basic_info else "N/A"
    weight = basic_info.get("weight", "N/A") if basic_info else "N/A"
    logp = result.get("logp", "N/A")
    lipinski = result.get("lipinski", "N/A")
    pains = result.get("pains", "N/A")
    tpsa = result.get("tpsa", "N/A")
    rdkit_valid = result.get("rdkit_valid", "N/A")
    tanimoto_sim = result.get("tanimoto_sim", "N/A")
    rdkit_rmsd = result.get("rdkit_rmsd", {})
    rmsd_max = rdkit_rmsd.get("max", "N/A") if rdkit_rmsd else "N/A"
    rmsd_min = rdkit_rmsd.get("min", "N/A") if rdkit_rmsd else "N/A"
    rmsd_median = rdkit_rmsd.get("median", "N/A") if rdkit_rmsd else "N/A"
    conformer_energy = result.get("conformer_energy", "N/A")
    lilly_medchem_passed_raw = result.get("lilly_medchem_passed", None)
    lilly_medchem_demerit_raw = result.get("lilly_medchem_demerit", None)
    lilly_medchem_description_raw = result.get("lilly_medchem_description", None)
    lilly_medchem_passed = (
        "N/A" if lilly_medchem_passed_raw is None else lilly_medchem_passed_raw
    )
    lilly_medchem_demerit = (
        "N/A" if lilly_medchem_demerit_raw is None else lilly_medchem_demerit_raw
    )
    lilly_medchem_description = (
        "N/A"
        if lilly_medchem_description_raw is None
        else lilly_medchem_description_raw
    )
    record = {
        "分子ID": mol_idx,
        "分子身份证": molecule_id,
        "SMILES": smiles,
        "Vina_Dock_亲和力": vina_dock_affinity,
        "Vina_Dock_RMSD下界": vina_dock_rmsd_lb,
        "Vina_Dock_RMSD上界": vina_dock_rmsd_ub,
        "Vina_ScoreOnly_亲和力": vina_score_only_affinity,
        "Vina_Minimize_亲和力": vina_minimize_affinity,
        "QED评分": qed,
        "SA评分": sa,
        "原子类型分布JSD": atom_type_jsd,
        "分子稳定性": molecule_stable,
        "稳定原子数": nr_stable_bonds,
        "总原子数(稳定性)": n_atoms_stability,
        "原子数": n_atoms,
        "键数": n_bonds,
        "环数": n_rings,
        "分子量": weight,
        "logP": logp,
        "Lipinski规则得分": lipinski,
        "PAINS检测": pains,
        "TPSA": tpsa,
        "RDKit验证": rdkit_valid,
        "Tanimoto相似度": tanimoto_sim,
        "RDKit_RMSD_最大": rmsd_max,
        "RDKit_RMSD_最小": rmsd_min,
        "RDKit_RMSD_中位数": rmsd_median,
        "构象能量": conformer_energy,
        "Lilly_Medchem_通过": lilly_medchem_passed,
        "Lilly_Medchem_扣分": lilly_medchem_demerit,
        "Lilly_Medchem_描述": lilly_medchem_description,
        "原始PT文件": os.path.basename(pt_file),
        "配体文件": os.path.basename(ligand_filename) if ligand_filename else "N/A",
        "原子编码模式": atom_mode,
        "对接强度": exhaustiveness,
        "评估时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    bond_length_jsd = result.get("bond_length_jsd", {})
    if bond_length_jsd:
        for key, value in bond_length_jsd.items():
            record[f"键长JSD_{key}"] = value if value is not None else "N/A"
    else:
        for key in [
            "JSD_6-6|1",
            "JSD_6-6|2",
            "JSD_6-6|4",
            "JSD_6-7|1",
            "JSD_6-7|2",
            "JSD_6-7|4",
            "JSD_6-8|1",
            "JSD_6-8|2",
        ]:
            if f"键长JSD_{key}" not in record:
                record[f"键长JSD_{key}"] = "N/A"
    pair_length_jsd = result.get("pair_length_jsd", {})
    if pair_length_jsd:
        for key, value in pair_length_jsd.items():
            record[f"原子对距离JSD_{key}"] = value if value is not None else "N/A"
    else:
        for key in ["JSD_CC_2A", "JSD_All_12A"]:
            if f"原子对距离JSD_{key}" not in record:
                record[f"原子对距离JSD_{key}"] = "N/A"
    _add_medchem_optimize_columns(record, result)
    return record


def _add_medchem_optimize_columns(record, result):
    """Append medchem provenance columns; all N/A when the layer is off."""
    opt = result.get("medchem_optimize") or {}
    record["优化前SMILES"] = opt.get("preopt_smiles", "N/A") or "N/A"
    record["应用变换"] = opt.get("transform_chain", "N/A") or "N/A"
    record["奖励增益"] = opt.get("total_reward", "N/A")

    preopt = result.get("preopt_eval") or {}
    if preopt:
        for column, key in (
            ("优化前QED评分", "qed"),
            ("优化前SA评分", "sa"),
            ("优化前Vina_Dock_亲和力", "vina_dock"),
            ("优化前Vina_ScoreOnly_亲和力", "vina_score_only"),
            ("优化前Vina_Minimize_亲和力", "vina_minimize"),
            ("优化前Lipinski规则得分", "lipinski"),
        ):
            value = preopt.get(key)
            record[column] = "N/A" if value is None else value


def record_evaluation_results_to_excel(
    results,
    output_dir,
    pt_file,
    ligand_filename,
    atom_mode,
    exhaustiveness,
    timestamp=None,
    num_samples=None,
    n_reconstruct_success=None,
    n_reconstruct_fail=None,
    n_eval_success=None,
    failure_stats=None,
    meta_records=None,
    pt_mode=None,
    sampling_meta=None,
    ref_vina_score=None,
    vina_modes=None,
):
    """
    将评估结果记录到Excel表格中（与原来的dock_generated_molecules.py格式一致）

    dynamic_then_optimization 模式下，默认仅记录 dynamic 原分子（is_original=True），用于与优化对比。
    若 meta 中没有任何 is_original=True（全部为优化变体），则退化为记录全部对接成功的分子，避免 Excel 空表。

    Args:
        meta_records: 元数据（含 is_original），用于筛选 dynamic 阶段分子
        pt_mode: 见上文
    """
    if pd is None:
        print(f"⚠️  {_PANDAS_MISSING_MSG}")
        return False

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    excel_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.xlsx")
    # vina_modes 为空集（none）时，不要求 Vina 结果，仅凭重建成功即可记录
    _vina_optional = vina_modes is not None and len(frozenset(vina_modes)) == 0
    only_dynamic = pt_mode == "dynamic_then_optimization" and meta_records
    # dynamic_then_scaffold + save_dynamic_before_scaffold：只写「dynamic 原始」行（is_dynamic_raw）
    sm = sampling_meta if isinstance(sampling_meta, dict) else {}
    excel_only_raw = (
        pt_mode == "dynamic_then_scaffold"
        and meta_records
        and sm.get("excel_only_dynamic_raw", False)
    )
    if excel_only_raw:
        only_dynamic = True
    # 默认只写「dynamic 原分子」便于与优化对比；若 meta 中全是优化变体（is_original=False），
    # 则没有任何一行会通过筛选，Excel 会空表——此时退化为写入全部对接成功的分子。
    if only_dynamic and meta_records:
        n_meta = len(meta_records)
        has_any_original = any(
            meta_records[i].get("is_original", True) if i < n_meta else True
            for i in range(n_meta)
        )
        if excel_only_raw:
            has_any_original = any(
                meta_records[i].get("is_dynamic_raw", False) if i < n_meta else False
                for i in range(n_meta)
            )
        if not has_any_original:
            only_dynamic = False

    def _is_dynamic_idx(idx):
        if not only_dynamic or idx >= len(meta_records):
            return True
        if excel_only_raw:
            return meta_records[idx].get("is_dynamic_raw", False)
        return meta_records[idx].get("is_original", True)

    # 准备数据（只记录成功重建的分子，避免数据质量下降）
    records = []
    n_dynamic_total = sum(1 for i in range(len(results)) if _is_dynamic_idx(i))
    n_reconstruct_dynamic = sum(
        1
        for i in range(len(results))
        if _is_dynamic_idx(i) and results[i].get("mol") is not None
    )
    n_complete_dynamic = sum(
        1
        for i in range(len(results))
        if _is_dynamic_idx(i)
        and results[i].get("smiles")
        and "." not in (results[i].get("smiles") or "")
    )

    for idx, result in enumerate(results):
        # dynamic_then_optimization 模式下仅记录 dynamic 阶段分子（不含优化后的）
        if not _is_dynamic_idx(idx):
            continue
        # 跳过重建失败的分子（mol为None）
        if result.get("mol") is None:
            continue
        # vina_modes=none 时跳过 Vina 校验，仅记录重建成功的分子
        if _vina_optional:
            record = _build_evaluation_record(
                result, pt_file, ligand_filename, atom_mode, exhaustiveness
            )
            if idx < 3:
                lilly_raw = result.get("lilly_medchem_passed", None)
                print(
                    f"    [Excel记录调试] 分子{idx + 1}: lilly_medchem_passed={lilly_raw} -> {record.get('Lilly_Medchem_通过')}"
                )
            records.append(record)
            continue
        # 只记录对接成功的分子（至少有一种vina模式成功）
        if not result.get("success"):
            continue
        # 检查是否至少有一种vina模式成功
        has_vina_result = (
            (result.get("vina_dock") and len(result["vina_dock"]) > 0)
            or (result.get("vina_score_only") is not None)
            or (result.get("vina_minimize") is not None)
        )
        if not has_vina_result:
            continue

        record = _build_evaluation_record(
            result, pt_file, ligand_filename, atom_mode, exhaustiveness
        )
        if idx < 3:
            lilly_raw = result.get("lilly_medchem_passed", None)
            print(
                f"    [Excel记录调试] 分子{idx + 1}: lilly_medchem_passed={lilly_raw} -> {record.get('Lilly_Medchem_通过')}"
            )
        records.append(record)

    if not records:
        print("⚠️  没有有效的评估结果，跳过Excel记录")
        return False

    # 创建DataFrame并保存到Excel
    df = pd.DataFrame(records)
    # 去重兜底：同一评估流程中同一分子（按分子ID）只保留一条记录
    if "分子ID" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["分子ID"], keep="first").reset_index(drop=True)
        if len(df) < before:
            print(f"  🧹 evaluation_results 去除重复分子记录 {before - len(df)} 条")

    # 按dock模式亲和力排序（从小到大，更小的亲和力表示更好的结合）
    if "Vina_Dock_亲和力" in df.columns:
        # 将 'N/A' 替换为 NaN 以便排序
        df_sorted = df.copy()
        df_sorted["Vina_Dock_亲和力"] = df_sorted["Vina_Dock_亲和力"].replace(
            "N/A", np.nan
        )
        df = df_sorted.sort_values("Vina_Dock_亲和力", na_position="last")

    # 筛选正常分子：Dock 亲和力为结合能时 ≤0 为优；若 Dock 已有有效负值，不因 ScoreOnly 量纲不同而误筛
    def is_valid_molecule(row):
        """判断分子是否正常（优先以 Dock；Dock 缺失时再参考 ScoreOnly / Minimize）"""
        vina_dock = row.get("Vina_Dock_亲和力", "N/A")
        vina_score = row.get("Vina_ScoreOnly_亲和力", "N/A")
        vina_min = row.get("Vina_Minimize_亲和力", "N/A")
        try:
            if vina_dock not in ("N/A", None) and not pd.isna(vina_dock):
                fd = float(vina_dock)
                if fd > 0:
                    return False
                # Dock 已成功且非正，视为对接正常
                return True
            if vina_score not in ("N/A", None) and not pd.isna(vina_score):
                if float(vina_score) > 0:
                    return False
            if vina_min not in ("N/A", None) and not pd.isna(vina_min):
                if float(vina_min) > 0:
                    return False
        except (ValueError, TypeError):
            pass
        return True

    # 筛选正常分子
    df_valid = df[df.apply(is_valid_molecule, axis=1)].copy()

    try:
        # 保存到Excel
        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="评估结果", index=False)

            # 添加正常分子表（剔除vina三个参数大于0的异常分子）
            df_valid.to_excel(writer, sheet_name="正常分子", index=False)

            # 添加统计信息工作表
            if len(records) > 0:
                # 过滤掉N/A值进行统计
                vina_dock_affinity = (
                    df["Vina_Dock_亲和力"].replace("N/A", np.nan).astype(float).dropna()
                )
                vina_score_only_affinity = (
                    df["Vina_ScoreOnly_亲和力"]
                    .replace("N/A", np.nan)
                    .astype(float)
                    .dropna()
                )
                vina_minimize_affinity = (
                    df["Vina_Minimize_亲和力"]
                    .replace("N/A", np.nan)
                    .astype(float)
                    .dropna()
                )
                qed_values = df["QED评分"].replace("N/A", np.nan).astype(float).dropna()
                sa_values = df["SA评分"].replace("N/A", np.nan).astype(float).dropna()

                # 构建统计信息（仅 dynamic 时使用 dynamic 子集统计）
                _n_total = (
                    n_dynamic_total
                    if only_dynamic
                    else (num_samples if num_samples is not None else len(results))
                )
                _n_reconstruct = (
                    n_reconstruct_dynamic
                    if only_dynamic
                    else (
                        n_reconstruct_success
                        if n_reconstruct_success is not None
                        else sum(1 for r in results if r.get("mol") is not None)
                    )
                )
                _n_eval = len(records)
                _n_complete = (
                    n_complete_dynamic
                    if only_dynamic
                    else sum(
                        1
                        for r in results
                        if r.get("smiles") and "." not in r.get("smiles", "")
                    )
                )
                _n_fail = (
                    (n_reconstruct_fail if n_reconstruct_fail is not None else "N/A")
                    if not only_dynamic
                    else (
                        _n_total - _n_reconstruct
                        if _n_total and _n_reconstruct is not None
                        else "N/A"
                    )
                )
                stats_items = [
                    "总样本数",
                    "重建成功",
                    "重建失败",
                    "重建成功百分比(%)",
                    "评估成功",
                    "对接成功百分比(%)",
                    "完整分子",
                ]
                stats_values = [
                    _n_total,
                    _n_reconstruct,
                    _n_fail,
                    (_n_reconstruct / _n_total * 100)
                    if (_n_total and _n_reconstruct is not None and _n_total > 0)
                    else "N/A",
                    _n_eval,
                    (_n_eval / _n_total * 100)
                    if (_n_total and _n_total > 0)
                    else "N/A",
                    _n_complete,
                ]

                # 添加失败原因统计
                if failure_stats:
                    stats_items.extend(
                        ["MolReconsError数量", "reconstruct返回None数量"]
                    )
                    stats_values.extend(
                        [
                            failure_stats.get("MolReconsError", 0),
                            failure_stats.get("reconstruct_returned_none", 0),
                        ]
                    )
                    # 添加其他错误类型
                    for error_type, count in failure_stats.get(
                        "other_errors", {}
                    ).items():
                        stats_items.append(f"{error_type}数量")
                        stats_values.append(count)

                # 添加Vina Dock模式统计
                if len(vina_dock_affinity) > 0:
                    stats_items.extend(
                        [
                            "Vina_Dock_最佳亲和力",
                            "Vina_Dock_平均亲和力",
                            "Vina_Dock_中位数亲和力",
                            "Vina_Dock_最差亲和力",
                            "Vina_Dock_亲和力标准差",
                        ]
                    )
                    stats_values.extend(
                        [
                            vina_dock_affinity.min(),
                            vina_dock_affinity.mean(),
                            vina_dock_affinity.median(),
                            vina_dock_affinity.max(),
                            vina_dock_affinity.std(),
                        ]
                    )

                # 添加Vina Score_Only模式统计
                if len(vina_score_only_affinity) > 0:
                    stats_items.extend(
                        [
                            "Vina_ScoreOnly_最佳亲和力",
                            "Vina_ScoreOnly_平均亲和力",
                            "Vina_ScoreOnly_中位数亲和力",
                            "Vina_ScoreOnly_最差亲和力",
                            "Vina_ScoreOnly_亲和力标准差",
                        ]
                    )
                    stats_values.extend(
                        [
                            vina_score_only_affinity.min(),
                            vina_score_only_affinity.mean(),
                            vina_score_only_affinity.median(),
                            vina_score_only_affinity.max(),
                            vina_score_only_affinity.std(),
                        ]
                    )

                # 添加Vina Minimize模式统计
                if len(vina_minimize_affinity) > 0:
                    stats_items.extend(
                        [
                            "Vina_Minimize_最佳亲和力",
                            "Vina_Minimize_平均亲和力",
                            "Vina_Minimize_中位数亲和力",
                            "Vina_Minimize_最差亲和力",
                            "Vina_Minimize_亲和力标准差",
                        ]
                    )
                    stats_values.extend(
                        [
                            vina_minimize_affinity.min(),
                            vina_minimize_affinity.mean(),
                            vina_minimize_affinity.median(),
                            vina_minimize_affinity.max(),
                            vina_minimize_affinity.std(),
                        ]
                    )

                # 添加参考配体对比统计
                if ref_vina_score is not None:
                    stats_items.extend(
                        ["参考配体_Vina_ScoreOnly", "优于参考配体比例_ScoreOnly(%)"]
                    )
                    if len(vina_score_only_affinity) > 0:
                        n_better = sum(
                            1 for s in vina_score_only_affinity if s < ref_vina_score
                        )
                        pct = n_better / len(vina_score_only_affinity) * 100
                        stats_values.extend([ref_vina_score, pct])
                    else:
                        stats_values.extend([ref_vina_score, "N/A"])

                    if len(vina_dock_affinity) > 0:
                        n_better_d = sum(
                            1 for s in vina_dock_affinity if s < ref_vina_score
                        )
                        pct_d = n_better_d / len(vina_dock_affinity) * 100
                        stats_items.extend(["优于参考配体比例_Dock(%)"])
                        stats_values.extend([pct_d])

                    if len(vina_minimize_affinity) > 0:
                        n_better_m = sum(
                            1 for s in vina_minimize_affinity if s < ref_vina_score
                        )
                        pct_m = n_better_m / len(vina_minimize_affinity) * 100
                        stats_items.extend(["优于参考配体比例_Minimize(%)"])
                        stats_values.extend([pct_m])

                # 添加化学指标统计
                stats_items.extend(["平均QED", "平均SA"])
                stats_values.extend(
                    [
                        qed_values.mean() if len(qed_values) > 0 else "N/A",
                        sa_values.mean() if len(sa_values) > 0 else "N/A",
                    ]
                )

                # 添加分子结构指标统计
                atom_type_jsd_values_excel = (
                    df["原子类型分布JSD"].replace("N/A", np.nan).astype(float).dropna()
                )
                if len(atom_type_jsd_values_excel) > 0:
                    stats_items.extend(
                        [
                            "原子类型分布JSD_均值",
                            "原子类型分布JSD_中位数",
                            "原子类型分布JSD_标准差",
                        ]
                    )
                    stats_values.extend(
                        [
                            atom_type_jsd_values_excel.mean(),
                            atom_type_jsd_values_excel.median(),
                            atom_type_jsd_values_excel.std(),
                        ]
                    )

                # 添加键长分布JSD统计（按键类型）
                bond_length_columns = [
                    col for col in df.columns if col.startswith("键长JSD_")
                ]
                for col in sorted(bond_length_columns):
                    bond_jsd_values = (
                        df[col].replace("N/A", np.nan).astype(float).dropna()
                    )
                    if len(bond_jsd_values) > 0:
                        stats_items.append(f"{col}_均值")
                        stats_values.append(bond_jsd_values.mean())

                # 添加原子对距离分布JSD统计（按类型）
                pair_length_columns = [
                    col for col in df.columns if col.startswith("原子对距离JSD_")
                ]
                for col in sorted(pair_length_columns):
                    pair_jsd_values = (
                        df[col].replace("N/A", np.nan).astype(float).dropna()
                    )
                    if len(pair_jsd_values) > 0:
                        stats_items.append(f"{col}_均值")
                        stats_values.append(pair_jsd_values.mean())

                stats = {"统计项目": stats_items, "数值": stats_values}
                stats_df = pd.DataFrame(stats)
                stats_df.to_excel(writer, sheet_name="统计信息", index=False)

        print(f"✅ 评估结果已保存至Excel: {excel_file}")
        if only_dynamic:
            print(f"   - 仅记录 dynamic 阶段分子（不含优化后），用于与优化效果对比")
        print(f"   - 总记录数: {len(records)}")
        if len(vina_dock_affinity) > 0:
            print(f"   - Vina_Dock最佳亲和力: {vina_dock_affinity.min():.3f} kcal/mol")
            print(f"   - Vina_Dock平均亲和力: {vina_dock_affinity.mean():.3f} kcal/mol")
        if len(vina_score_only_affinity) > 0:
            print(
                f"   - Vina_ScoreOnly平均亲和力: {vina_score_only_affinity.mean():.3f} kcal/mol"
            )
        if len(vina_minimize_affinity) > 0:
            print(
                f"   - Vina_Minimize平均亲和力: {vina_minimize_affinity.mean():.3f} kcal/mol"
            )
        return True

    except ImportError as e:
        error_msg = _OPENPYXL_MISSING_MSG or (
            'Missing optional dependency "openpyxl". '
            "Install it with: pip install openpyxl or conda install openpyxl"
        )
        print(f"⚠️  无法写入Excel: {error_msg}")
        return False
    except Exception as exc:
        print(f"⚠️  写入Excel失败: {exc}")
        traceback.print_exc()
        return False


def _is_valid_molecule_row(row):
    """判断分子是否正常（vina三个参数都不大于0），与 evaluation_results 一致"""
    vina_dock = row.get("Vina_Dock_亲和力", "N/A")
    vina_score = row.get("Vina_ScoreOnly_亲和力", "N/A")
    vina_min = row.get("Vina_Minimize_亲和力", "N/A")
    try:
        if vina_dock not in ("N/A", None) and not pd.isna(vina_dock):
            if float(vina_dock) > 0:
                return False
        if vina_score not in ("N/A", None) and not pd.isna(vina_score):
            if float(vina_score) > 0:
                return False
        if vina_min not in ("N/A", None) and not pd.isna(vina_min):
            if float(vina_min) > 0:
                return False
    except (ValueError, TypeError):
        pass
    return True


def _build_stats_sheet_for_df(
    df, n_total_samples=None, n_reconstruct_success=None, n_complete=None
):
    """为 DataFrame 构建统计信息表，与 evaluation_results 的统计信息格式一致"""
    if df is None or df.empty:
        return None
    stats_items = ["完整分子数"]
    stats_values = [len(df)]
    if n_total_samples is not None:
        stats_items.insert(0, "总样本数")
        stats_values.insert(0, n_total_samples)
    if n_reconstruct_success is not None:
        stats_items.extend(["重建成功", "重建成功百分比(%)"])
        stats_values.extend(
            [
                n_reconstruct_success,
                (n_reconstruct_success / n_total_samples * 100)
                if n_total_samples is not None and n_total_samples > 0
                else "N/A",
            ]
        )
    if n_complete is not None and n_total_samples is not None and n_total_samples > 0:
        stats_items.append("完整分子占比(%)")
        stats_values.append(n_complete / n_total_samples * 100)
    vina_dock_affinity = (
        df["Vina_Dock_亲和力"].replace("N/A", np.nan).astype(float).dropna()
    )
    vina_score_only_affinity = (
        df["Vina_ScoreOnly_亲和力"].replace("N/A", np.nan).astype(float).dropna()
    )
    vina_minimize_affinity = (
        df["Vina_Minimize_亲和力"].replace("N/A", np.nan).astype(float).dropna()
    )
    qed_values = df["QED评分"].replace("N/A", np.nan).astype(float).dropna()
    sa_values = df["SA评分"].replace("N/A", np.nan).astype(float).dropna()
    if len(vina_dock_affinity) > 0:
        stats_items.extend(
            [
                "Vina_Dock_最佳亲和力",
                "Vina_Dock_平均亲和力",
                "Vina_Dock_中位数亲和力",
                "Vina_Dock_最差亲和力",
                "Vina_Dock_亲和力标准差",
            ]
        )
        stats_values.extend(
            [
                vina_dock_affinity.min(),
                vina_dock_affinity.mean(),
                vina_dock_affinity.median(),
                vina_dock_affinity.max(),
                vina_dock_affinity.std(),
            ]
        )
    if len(vina_score_only_affinity) > 0:
        stats_items.extend(
            [
                "Vina_ScoreOnly_最佳亲和力",
                "Vina_ScoreOnly_平均亲和力",
                "Vina_ScoreOnly_中位数亲和力",
                "Vina_ScoreOnly_最差亲和力",
                "Vina_ScoreOnly_亲和力标准差",
            ]
        )
        stats_values.extend(
            [
                vina_score_only_affinity.min(),
                vina_score_only_affinity.mean(),
                vina_score_only_affinity.median(),
                vina_score_only_affinity.max(),
                vina_score_only_affinity.std(),
            ]
        )
    if len(vina_minimize_affinity) > 0:
        stats_items.extend(
            [
                "Vina_Minimize_最佳亲和力",
                "Vina_Minimize_平均亲和力",
                "Vina_Minimize_中位数亲和力",
                "Vina_Minimize_最差亲和力",
                "Vina_Minimize_亲和力标准差",
            ]
        )
        stats_values.extend(
            [
                vina_minimize_affinity.min(),
                vina_minimize_affinity.mean(),
                vina_minimize_affinity.median(),
                vina_minimize_affinity.max(),
                vina_minimize_affinity.std(),
            ]
        )
    stats_items.extend(["平均QED", "平均SA"])
    stats_values.extend(
        [
            qed_values.mean() if len(qed_values) > 0 else "N/A",
            sa_values.mean() if len(sa_values) > 0 else "N/A",
        ]
    )
    atom_type_jsd_values = (
        df["原子类型分布JSD"].replace("N/A", np.nan).astype(float).dropna()
    )
    if len(atom_type_jsd_values) > 0:
        stats_items.extend(
            ["原子类型分布JSD_均值", "原子类型分布JSD_中位数", "原子类型分布JSD_标准差"]
        )
        stats_values.extend(
            [
                atom_type_jsd_values.mean(),
                atom_type_jsd_values.median(),
                atom_type_jsd_values.std(),
            ]
        )
    bond_length_columns = [col for col in df.columns if col.startswith("键长JSD_")]
    for col in sorted(bond_length_columns):
        bond_jsd_values = df[col].replace("N/A", np.nan).astype(float).dropna()
        if len(bond_jsd_values) > 0:
            stats_items.append(f"{col}_均值")
            stats_values.append(bond_jsd_values.mean())
    pair_length_columns = [
        col for col in df.columns if col.startswith("原子对距离JSD_")
    ]
    for col in sorted(pair_length_columns):
        pair_jsd_values = df[col].replace("N/A", np.nan).astype(float).dropna()
        if len(pair_jsd_values) > 0:
            stats_items.append(f"{col}_均值")
            stats_values.append(pair_jsd_values.mean())
    return pd.DataFrame({"统计项目": stats_items, "数值": stats_values})


def record_complete_molecules_to_excel(
    results,
    meta_records,
    output_dir,
    pt_file,
    ligand_filename,
    atom_mode,
    exhaustiveness,
    timestamp=None,
    num_samples=None,
    n_reconstruct_success=None,
    n_complete=None,
):
    """
    将完整分子（SMILES 不含 '.' 的非片段分子）记录到专用 Excel，结构与 evaluation_results 一致：
    - 初次生成完整分子 / 优化后完整分子：与 评估结果 相同的列
    - 初次生成_正常分子 / 优化后_正常分子：与 正常分子 相同的筛选（vina 参数不大于 0）
    - 初次生成_统计信息 / 优化后_统计信息：与 统计信息 相同的统计项
    """
    if pd is None:
        return False
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_file = os.path.join(output_dir, f"complete_molecules_{timestamp}.xlsx")
    records_original = []
    records_optimized = []
    for idx, result in enumerate(results):
        if result.get("mol") is None or not result.get("success"):
            continue
        has_vina = (
            (result.get("vina_dock") and len(result["vina_dock"]) > 0)
            or (result.get("vina_score_only") is not None)
            or (result.get("vina_minimize") is not None)
        )
        if not has_vina:
            continue
        smiles = result.get("smiles") or ""
        if not smiles or "." in smiles:
            continue
        record = _build_evaluation_record(
            result, pt_file, ligand_filename, atom_mode, exhaustiveness
        )
        # 分类依据：优先用 molecule_id（优化变体含 _opN_from_），其次用 meta_records[mol_idx]
        molecule_id = result.get("molecule_id") or ""
        mol_idx = result.get("mol_idx", idx)
        if "_op" in molecule_id and "_from_" in molecule_id:
            is_original = False
        elif meta_records and mol_idx < len(meta_records):
            is_original = meta_records[mol_idx].get("is_original", True)
        else:
            is_original = True
        if is_original:
            records_original.append(record)
        else:
            records_optimized.append(record)
    if not records_original and not records_optimized:
        return False
    try:
        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            for sheet_prefix, records in [
                ("初次生成完整分子", records_original),
                ("优化后完整分子", records_optimized),
            ]:
                if not records:
                    continue
                df = pd.DataFrame(records)
                # 去重兜底：同一分子（按分子ID）只保留一条记录
                if "分子ID" in df.columns:
                    before = len(df)
                    df = df.drop_duplicates(
                        subset=["分子ID"], keep="first"
                    ).reset_index(drop=True)
                    if len(df) < before:
                        print(
                            f"  🧹 {sheet_prefix} 去除重复分子记录 {before - len(df)} 条"
                        )
                if "Vina_Dock_亲和力" in df.columns:
                    df = df.copy()
                    df["Vina_Dock_亲和力"] = df["Vina_Dock_亲和力"].replace(
                        "N/A", np.nan
                    )
                    df = df.sort_values("Vina_Dock_亲和力", na_position="last")
                df.to_excel(writer, sheet_name=sheet_prefix, index=False)
                df_valid = df[df.apply(_is_valid_molecule_row, axis=1)].copy()
                df_valid.to_excel(
                    writer, sheet_name=f"{sheet_prefix}_正常分子", index=False
                )
                stats_df = _build_stats_sheet_for_df(
                    df,
                    n_total_samples=num_samples,
                    n_reconstruct_success=n_reconstruct_success,
                    n_complete=len(records),
                )
                if stats_df is not None:
                    stats_df.to_excel(
                        writer, sheet_name=f"{sheet_prefix}_统计信息", index=False
                    )
        print(f"✅ 完整分子已保存至: {excel_file}")
        print(f"   - 初次生成完整分子: {len(records_original)}")
        print(f"   - 优化后完整分子: {len(records_optimized)}")
        return True
    except Exception as exc:
        print(f"⚠️  写入完整分子Excel失败: {exc}")
        traceback.print_exc()
        return False
