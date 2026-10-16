"""History responsibilities extracted from evaluate_pt_with_correct_reconstruct.py."""

from diffdynamic.evaluation.molecule.state import (
    Path,
    REPO_ROOT,
    glob,
    np,
    os,
    pd,
    re,
    traceback,
)
from diffdynamic.evaluation.molecule import state as _state


def _format_data_id_for_excel(data_id):
    """
    格式化data_id以便正确记录到Excel

    Args:
        data_id: 数据ID（可能是int、str或其他类型）

    Returns:
        int: 如果是有效的整数（0-99），返回整数
        str: 如果是字符串格式的数字，返回字符串
        str: 'N/A' 如果为None或无效
    """
    if data_id is None:
        return "N/A"

    # 尝试转换为整数
    try:
        if isinstance(data_id, (int, float)):
            data_id_int = int(data_id)
            if 0 <= data_id_int <= 99:
                return data_id_int
        elif isinstance(data_id, str):
            # 尝试从字符串中提取数字
            match = re.search(r"\d+", str(data_id))
            if match:
                data_id_int = int(match.group())
                if 0 <= data_id_int <= 99:
                    return data_id_int
            # 如果字符串本身就是数字
            if data_id.strip().isdigit():
                data_id_int = int(data_id.strip())
                if 0 <= data_id_int <= 99:
                    return data_id_int
    except (ValueError, TypeError):
        pass

    # 如果无法转换为有效整数，返回字符串形式
    return str(data_id)


def update_sampling_history(
    pt_file,
    statistics,
    output_dir=None,
    num_samples=None,
    n_reconstruct_success=None,
    n_eval_success=None,
):
    """
    更新 sampling_history.xlsx 中的评测信息（读-改-写全程持锁+原子写，
    防止并发评估覆盖丢失与文件撕裂损坏）

    Args:
        pt_file: .pt文件路径（评估的.pt文件）
        statistics: 统计信息字典（包含Vina评分、QED、SA等）
        output_dir: 输出目录（用于记录eval_output_dir）
        num_samples: 总样本数（用于计算成功率）
        n_reconstruct_success: 重建成功数（用于计算重建成功率）
        n_eval_success: 评估成功数（用于计算对接成功率）

    Returns:
        bool: 是否成功更新
    """
    from diffdynamic.evaluation.molecule.loading import _torch_load_layered
    from diffdynamic.evaluation.molecule.writers import (
        _atomic_excel_write,
        _xlsx_file_lock,
    )

    if pd is None:
        return False

    try:
        from diffdynamic.io.sampling_recorder import DEFAULT_RECORD_PATH

        sampling_history_path = Path(DEFAULT_RECORD_PATH)

        if not sampling_history_path.exists():
            print(f"  ⚠️  sampling_history.xlsx 不存在，跳过更新")
            return False

        # 从.pt文件加载，获取result_file路径（采样时保存的.pt文件路径）
        try:
            data = _torch_load_layered(pt_file)
            result_file_path = None
            if (
                isinstance(data, dict)
                and "extra_info" in data
                and "result_file" in data["extra_info"]
            ):
                result_file_path = data["extra_info"]["result_file"]
            if not result_file_path:
                result_file_path = str(Path(pt_file).resolve())
        except Exception as e:
            print(f"  ⚠️  无法从.pt文件读取信息: {e}")
            result_file_path = str(Path(pt_file).resolve())

        if not result_file_path:
            print(f"  ⚠️  无法确定result_file路径，跳过更新")
            return False

        # 读-改-写全程持锁，防止并发覆盖导致记录丢失/文件撕裂
        with _xlsx_file_lock(sampling_history_path):
            try:
                df = pd.read_excel(sampling_history_path, engine="openpyxl")
            except Exception as e:
                print(f"  ⚠️  无法读取sampling_history.xlsx: {e}")
                return False

            # 标准化路径用于匹配（转换为绝对路径并标准化）
            result_file_path_abs = str(Path(result_file_path).resolve())

            # 查找匹配的记录（通过result_file列匹配）
            if "result_file" not in df.columns:
                print(f"  ⚠️  sampling_history.xlsx 中没有result_file列，跳过更新")
                return False

            mask = (
                df["result_file"].apply(
                    lambda x: (
                        str(Path(str(x)).resolve()) if pd.notna(x) and str(x) else ""
                    )
                )
                == result_file_path_abs
            )

            matching_rows = df[mask]

            if len(matching_rows) == 0:
                print(
                    f"  ⚠️  未找到匹配的采样记录（result_file: {result_file_path_abs}）"
                )
                print(f"     提示：请确保评估的.pt文件是采样时保存的文件")
                return False

            if len(matching_rows) > 1:
                print(f"  ⚠️  找到多条匹配记录，将更新最后一条")
                matching_rows = matching_rows.iloc[[-1]]

            # 准备更新的评测信息
            update_data = {}

            vina_dock_scores = statistics.get("vina_dock_scores", [])
            if vina_dock_scores:
                update_data["eval_最佳亲和力"] = np.min(vina_dock_scores)
                update_data["eval_平均亲和力"] = np.mean(vina_dock_scores)
                update_data["eval_中位数亲和力"] = np.median(vina_dock_scores)

            if (
                num_samples is not None
                and n_reconstruct_success is not None
                and num_samples > 0
            ):
                update_data["eval_重建成功率(%)"] = (
                    n_reconstruct_success / num_samples * 100
                )
            elif num_samples is not None and n_reconstruct_success is not None:
                update_data["eval_重建成功率(%)"] = 0.0

            if (
                num_samples is not None
                and n_eval_success is not None
                and num_samples > 0
            ):
                update_data["eval_对接成功率(%)"] = n_eval_success / num_samples * 100
            elif num_samples is not None and n_eval_success is not None:
                update_data["eval_对接成功率(%)"] = 0.0

            qed_values = statistics.get("qed_values", [])
            if qed_values:
                update_data["eval_QED_均值"] = np.mean(qed_values)
                update_data["eval_QED_中位数"] = np.median(qed_values)

            sa_values = statistics.get("sa_values", [])
            if sa_values:
                update_data["eval_SA_均值"] = np.mean(sa_values)
                update_data["eval_SA_中位数"] = np.median(sa_values)

            if output_dir:
                update_data["eval_output_dir"] = str(Path(output_dir).resolve())
                update_data["sdf_dir"] = str(
                    Path(output_dir).resolve() / "reconstructed_molecules"
                )
                update_data["eval_method"] = "evaluate_pt_with_correct_reconstruct"

            # 更新匹配的记录
            for idx in matching_rows.index:
                for key, value in update_data.items():
                    df.at[idx, key] = value

            # 原子写回（临时文件+替换）
            try:
                _atomic_excel_write(df, sampling_history_path)
                print(f"  ✅ 已更新 sampling_history.xlsx 中的评测信息")
                print(f"     匹配记录数: {len(matching_rows)}")
                print(f"     更新的字段: {list(update_data.keys())}")
                return True
            except Exception as e:
                print(f"  ❌ 保存sampling_history.xlsx失败: {e}")
                traceback.print_exc()
                return False

    except Exception as e:
        print(f"  ⚠️  更新sampling_history.xlsx失败: {e}")
        traceback.print_exc()
        return False


def record_evaluation_to_global_log(
    pt_file,
    ligand_filename,
    protein_root,
    atom_mode,
    exhaustiveness,
    start_time,
    end_time,
    num_samples,
    n_reconstruct_success,
    n_eval_success,
    n_complete,
    statistics,
    output_dir=None,
    data_id=None,
):
    """
    记录评估结果到全局记录文件 evalresults record.xlsx

    Args:
        pt_file: .pt文件路径
        ligand_filename: 配体文件名
        protein_root: 蛋白根目录
        atom_mode: 原子编码模式
        exhaustiveness: 对接强度
        start_time: 评估开始时间（datetime对象）
        end_time: 评估结束时间（datetime对象）
        num_samples: 总样本数
        n_reconstruct_success: 重建成功数
        n_eval_success: 评估成功数
        n_complete: 完整分子数
        statistics: 统计信息字典
        output_dir: 输出目录（用于确定记录文件位置）
        data_id: 数据ID（idx），从extra_info中提取
    """
    from diffdynamic.evaluation.molecule.writers import (
        _atomic_excel_write,
        _xlsx_file_lock,
    )

    if pd is None:
        return False

    try:
        # 确定记录文件路径（在项目根目录，即脚本所在目录）
        record_file = REPO_ROOT / "evalresults record.xlsx"

        # 准备记录数据
        record = {
            "评估开始时间": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "评估结束时间": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "评估耗时(秒)": (end_time - start_time).total_seconds(),
            "PT文件": os.path.basename(pt_file) if pt_file else "N/A",
            "配体文件": os.path.basename(ligand_filename) if ligand_filename else "N/A",
            "蛋白根目录": str(protein_root),
            "原子编码模式": atom_mode,
            "对接强度": exhaustiveness,
            "数据ID": _format_data_id_for_excel(data_id),  # 格式化data_id为整数或字符串
            "总样本数": num_samples,
            "重建成功数": n_reconstruct_success,
            "重建成功率(%)": (n_reconstruct_success / num_samples * 100)
            if num_samples > 0
            else 0,
            "评估成功数": n_eval_success,
            "评估成功率(%)": (n_eval_success / num_samples * 100)
            if num_samples > 0
            else 0,
            "完整分子数": n_complete,
            "完整分子率(%)": (n_complete / num_samples * 100) if num_samples > 0 else 0,
        }

        # 添加Vina评分统计
        vina_dock_scores = statistics.get("vina_dock_scores", [])
        if vina_dock_scores:
            record["Vina_Dock_均值"] = np.mean(vina_dock_scores)
            record["Vina_Dock_中位数"] = np.median(vina_dock_scores)
            record["Vina_Dock_最小值"] = np.min(vina_dock_scores)
            record["Vina_Dock_最大值"] = np.max(vina_dock_scores)
        else:
            record["Vina_Dock_均值"] = "N/A"
            record["Vina_Dock_中位数"] = "N/A"
            record["Vina_Dock_最小值"] = "N/A"
            record["Vina_Dock_最大值"] = "N/A"

        vina_score_only_scores = statistics.get("vina_score_only_scores", [])
        if vina_score_only_scores:
            record["Vina_ScoreOnly_均值"] = np.mean(vina_score_only_scores)
            record["Vina_ScoreOnly_中位数"] = np.median(vina_score_only_scores)
        else:
            record["Vina_ScoreOnly_均值"] = "N/A"
            record["Vina_ScoreOnly_中位数"] = "N/A"

        vina_minimize_scores = statistics.get("vina_minimize_scores", [])
        if vina_minimize_scores:
            record["Vina_Minimize_均值"] = np.mean(vina_minimize_scores)
            record["Vina_Minimize_中位数"] = np.median(vina_minimize_scores)
        else:
            record["Vina_Minimize_均值"] = "N/A"
            record["Vina_Minimize_中位数"] = "N/A"

        # 添加化学指标统计
        qed_values = statistics.get("qed_values", [])
        if qed_values:
            record["QED_均值"] = np.mean(qed_values)
            record["QED_中位数"] = np.median(qed_values)
        else:
            record["QED_均值"] = "N/A"
            record["QED_中位数"] = "N/A"

        sa_values = statistics.get("sa_values", [])
        if sa_values:
            record["SA_均值"] = np.mean(sa_values)
            record["SA_中位数"] = np.median(sa_values)
        else:
            record["SA_均值"] = "N/A"
            record["SA_中位数"] = "N/A"

        # 添加分子结构指标统计
        atom_type_jsd_values = statistics.get("atom_type_jsd_values", [])
        if atom_type_jsd_values:
            record["原子类型分布JSD_均值"] = np.mean(atom_type_jsd_values)
            record["原子类型分布JSD_中位数"] = np.median(atom_type_jsd_values)
        else:
            record["原子类型分布JSD_均值"] = "N/A"
            record["原子类型分布JSD_中位数"] = "N/A"

        # 添加键长分布JSD统计（按键类型）
        bond_length_jsd_all = statistics.get("bond_length_jsd_all", {})
        for key, values in sorted(bond_length_jsd_all.items()):
            if values:
                record[f"键长JSD_{key}_均值"] = np.mean(values)
            else:
                record[f"键长JSD_{key}_均值"] = "N/A"

        # 添加原子对距离分布JSD统计（按类型）
        pair_length_jsd_all = statistics.get("pair_length_jsd_all", {})
        for key, values in sorted(pair_length_jsd_all.items()):
            if values:
                record[f"原子对距离JSD_{key}_均值"] = np.mean(values)
            else:
                record[f"原子对距离JSD_{key}_均值"] = "N/A"

        # 添加稳定性评估统计
        molecule_stable_values = statistics.get("molecule_stable_values", [])
        if molecule_stable_values:
            record["分子稳定性_比例"] = np.mean(
                [int(x) for x in molecule_stable_values]
            )
        else:
            record["分子稳定性_比例"] = "N/A"

        nr_stable_bonds_values = statistics.get("nr_stable_bonds_values", [])
        n_atoms_stability_values = statistics.get("n_atoms_stability_values", [])
        if nr_stable_bonds_values and n_atoms_stability_values:
            total_stable = sum(nr_stable_bonds_values)
            total_atoms = sum(n_atoms_stability_values)
            record["原子稳定性_比例"] = (
                total_stable / total_atoms if total_atoms > 0 else "N/A"
            )
        else:
            record["原子稳定性_比例"] = "N/A"

        # 添加基础结构信息统计
        n_atoms_values = statistics.get("n_atoms_values", [])
        if n_atoms_values:
            record["原子数_均值"] = np.mean(n_atoms_values)
            record["原子数_中位数"] = np.median(n_atoms_values)
        else:
            record["原子数_均值"] = "N/A"
            record["原子数_中位数"] = "N/A"

        n_bonds_values = statistics.get("n_bonds_values", [])
        if n_bonds_values:
            record["键数_均值"] = np.mean(n_bonds_values)
        else:
            record["键数_均值"] = "N/A"

        n_rings_values = statistics.get("n_rings_values", [])
        if n_rings_values:
            record["环数_均值"] = np.mean(n_rings_values)
        else:
            record["环数_均值"] = "N/A"

        weight_values = statistics.get("weight_values", [])
        if weight_values:
            record["分子量_均值"] = np.mean(weight_values)
            record["分子量_中位数"] = np.median(weight_values)
        else:
            record["分子量_均值"] = "N/A"
            record["分子量_中位数"] = "N/A"

        # 添加化学属性统计
        logp_values = statistics.get("logp_values", [])
        if logp_values:
            record["logP_均值"] = np.mean(logp_values)
            record["logP_中位数"] = np.median(logp_values)
        else:
            record["logP_均值"] = "N/A"
            record["logP_中位数"] = "N/A"

        tpsa_values = statistics.get("tpsa_values", [])
        if tpsa_values:
            record["TPSA_均值"] = np.mean(tpsa_values)
            record["TPSA_中位数"] = np.median(tpsa_values)
        else:
            record["TPSA_均值"] = "N/A"
            record["TPSA_中位数"] = "N/A"

        lipinski_values = statistics.get("lipinski_values", [])
        if lipinski_values:
            record["Lipinski规则_均值"] = np.mean(lipinski_values)
            record["Lipinski规则_中位数"] = np.median(lipinski_values)
        else:
            record["Lipinski规则_均值"] = "N/A"
            record["Lipinski规则_中位数"] = "N/A"

        pains_values = statistics.get("pains_values", [])
        if pains_values:
            record["PAINS检测_比例"] = np.mean([int(x) for x in pains_values])
        else:
            record["PAINS检测_比例"] = "N/A"

        # 添加RDKit验证统计
        rdkit_valid_values = statistics.get("rdkit_valid_values", [])
        if rdkit_valid_values:
            record["RDKit验证_通过率"] = np.mean([int(x) for x in rdkit_valid_values])
        else:
            record["RDKit验证_通过率"] = "N/A"

        # 添加唯一性和内部相似度统计
        uniqueness = statistics.get("uniqueness", None)
        if uniqueness is not None:
            record["唯一性"] = uniqueness
        else:
            record["唯一性"] = "N/A"

        internal_similarity = statistics.get("internal_similarity", None)
        if internal_similarity is not None:
            record["内部相似度"] = internal_similarity
        else:
            record["内部相似度"] = "N/A"

        # 添加相似度统计
        tanimoto_sim_values = statistics.get("tanimoto_sim_values", [])
        if tanimoto_sim_values:
            record["Tanimoto相似度_均值"] = np.mean(tanimoto_sim_values)
            record["Tanimoto相似度_中位数"] = np.median(tanimoto_sim_values)
        else:
            record["Tanimoto相似度_均值"] = "N/A"
            record["Tanimoto相似度_中位数"] = "N/A"

        # 添加构象评估统计
        rdkit_rmsd_max_values = statistics.get("rdkit_rmsd_max_values", [])
        if rdkit_rmsd_max_values:
            record["RDKit_RMSD_最大_均值"] = np.mean(rdkit_rmsd_max_values)
        else:
            record["RDKit_RMSD_最大_均值"] = "N/A"

        rdkit_rmsd_median_values = statistics.get("rdkit_rmsd_median_values", [])
        if rdkit_rmsd_median_values:
            record["RDKit_RMSD_中位数_均值"] = np.mean(rdkit_rmsd_median_values)
        else:
            record["RDKit_RMSD_中位数_均值"] = "N/A"

        conformer_energy_values = statistics.get("conformer_energy_values", [])
        if conformer_energy_values:
            record["构象能量_均值"] = np.mean(conformer_energy_values)
            record["构象能量_中位数"] = np.median(conformer_energy_values)
        else:
            record["构象能量_均值"] = "N/A"
            record["构象能量_中位数"] = "N/A"

        # 读取或创建Excel文件（持锁读-改-写，防止并发覆盖与记录重复堆积）
        with _xlsx_file_lock(record_file):
            if record_file.exists():
                # 如果文件存在，读取现有数据
                try:
                    # 尝试读取Excel文件，处理可能的索引列问题
                    df_existing = pd.read_excel(
                        record_file, engine="openpyxl", index_col=None
                    )
                    # 如果第一列是未命名的索引列，删除它
                    if len(df_existing.columns) > 0 and df_existing.columns[
                        0
                    ].startswith("Unnamed:"):
                        df_existing = df_existing.drop(df_existing.columns[0], axis=1)
                    # 打印调试信息
                    if len(df_existing) > 0:
                        print(f"  📊 读取到 {len(df_existing)} 条现有记录")
                except Exception as e:
                    print(f"  ⚠️  读取现有Excel文件失败: {e}")
                    print(f"  📝 将创建新文件")
                    traceback.print_exc()
                    df_existing = pd.DataFrame()
            else:
                df_existing = pd.DataFrame()
                print(f"  📝 Excel文件不存在，将创建新文件: {record_file}")

            # 创建新记录DataFrame
            df_new = pd.DataFrame([record])

            # 合并数据
            if len(df_existing) > 0:
                # 确保列顺序一致（新列追加到末尾）
                all_columns = list(df_existing.columns) + [
                    col for col in df_new.columns if col not in df_existing.columns
                ]
                df_existing = df_existing.reindex(columns=all_columns, fill_value="N/A")
                df_new = df_new.reindex(columns=all_columns, fill_value="N/A")
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_combined = df_new
                print(f"  📊 创建新记录，共 {len(df_combined)} 条")

            # 去重：同一评估结果（除时间列外全列相同）重复登记时只保留最后一条。
            # 注意保留同 PT 不同参数/结果的合法历史行。
            # 归一化两类往返差异：
            #   1) openpyxl 写 30.0 读回 30（int/float）；
            #   2) pd.read_excel 把磁盘上的 'N/A' 读成 NaN，与内存新行字面 'N/A'
            #      永不相等（曾导致重复记录无法即时清除、逐轮堆积）。
            def _norm_for_dedup(df_part):
                out = df_part.copy()
                for c in out.columns:
                    s = out[c]
                    if pd.api.types.is_numeric_dtype(s):
                        out[c] = s.astype("float64")
                    else:
                        s2 = s.replace(
                            {"N/A": np.nan, "n/a": np.nan, "NA": np.nan, "": np.nan}
                        )
                        num = pd.to_numeric(s2, errors="coerce")
                        valid = s2.notna()
                        if valid.any() and (num.notna() == valid).all():
                            out[c] = num.astype("float64").where(valid)
                        else:
                            out[c] = s2
                return out

            time_cols = {"评估开始时间", "评估结束时间", "评估耗时(秒)"}
            dedup_cols = [c for c in df_combined.columns if c not in time_cols]
            before = len(df_combined)
            dup_mask = _norm_for_dedup(df_combined)[dedup_cols].duplicated(keep="last")
            removed = int(dup_mask.sum())
            if removed > 0:
                df_combined = df_combined[~dup_mask].reset_index(drop=True)
                print(f"  🧹 去除完全相同的重复记录 {removed} 条")
            print(f"  📊 合并后共有 {len(df_combined)} 条记录")

            # 原子写回（临时文件+替换，防止进程中途挂掉撕裂 zip）
            try:
                _atomic_excel_write(df_combined, record_file)
                print(f"\n✅ 评估记录已追加至: {record_file}")
                return True
            except Exception as e:
                print(f"  ❌ 保存Excel文件失败: {e}")
                traceback.print_exc()
                return False

    except Exception as e:
        print(f"⚠️  记录评估结果到全局日志失败: {e}")
        traceback.print_exc()
        return False


def parse_params_from_batchsummary_filename(filename):
    """
    从 batchsummary 文件名中提取参数

    文件名格式示例：
    batch_evaluation_summary_20251226_103804_gfquadratic_1_0_tl700_lslambda_80p0_20p0_lsstep_0p5_lsnoise_0p0_rflambda_30p0_5p0_rfstep_0p3_rfnoise_0p08.xlsx
    batch_evaluation_summary_20251224_161650_gfquadratic_1_0_tl750_lslambda_40p0_20p0_rflambda_20p0_2p0.xlsx

    Args:
        filename: 文件名（可以是完整路径或文件名）

    Returns:
        dict: 参数字典，包含所有提取的参数
    """
    # 提取文件名（去除路径）
    if isinstance(filename, Path):
        filename = filename.name
    elif "/" in filename or "\\" in filename:
        filename = os.path.basename(filename)

    # 移除扩展名
    filename = filename.replace(".xlsx", "")

    params = {}

    # 解析梯度融合参数：gfquadratic_1_0
    gf_match = re.search(r"gf(\w+)_(\d+)_(\d+)", filename)
    if gf_match:
        params["grad_fusion_mode"] = gf_match.group(1)
        params["grad_fusion_start"] = int(gf_match.group(2))
        params["grad_fusion_end"] = int(gf_match.group(3))

    # 解析时间边界：tl700
    tl_match = re.search(r"tl(\d+)", filename)
    if tl_match:
        params["time_boundary"] = int(tl_match.group(1))

    # 解析大步探索阶段参数
    # 1. Lambda调度模式：lslambda_80p0_20p0
    ls_lambda_match = re.search(r"lslambda_([\dpm]+)_([\dpm]+)", filename)
    if ls_lambda_match:
        params["large_step_schedule"] = "lambda"
        params["large_step_lambda_a"] = ls_lambda_match.group(1).replace("p", ".")
        params["large_step_lambda_b"] = ls_lambda_match.group(2).replace("p", ".")

    # 2. Linear调度模式：lslinear_20_20
    ls_linear_match = re.search(r"lslinear_(\d+)_(\d+)", filename)
    if ls_linear_match:
        params["large_step_schedule"] = "linear"
        params["large_step_linear_lower"] = int(ls_linear_match.group(1))
        params["large_step_linear_upper"] = int(ls_linear_match.group(2))

    # 3. Fixed调度模式：lsfixed_25
    ls_fixed_match = re.search(r"lsfixed_(\d+)", filename)
    if ls_fixed_match:
        params["large_step_schedule"] = "fixed"
        params["large_step_stride"] = int(ls_fixed_match.group(1))

    # 4. 步长和噪声参数（可选）：lsstep_0p5_lsnoise_0p0
    ls_step_match = re.search(r"lsstep_([\dpm]+)", filename)
    if ls_step_match:
        params["large_step_step_size"] = ls_step_match.group(1).replace("p", ".")

    ls_noise_match = re.search(r"lsnoise_([\dpm]+)", filename)
    if ls_noise_match:
        params["large_step_noise_scale"] = ls_noise_match.group(1).replace("p", ".")

    # 解析精炼阶段参数
    # 1. Lambda调度模式：rflambda_30p0_5p0
    rf_lambda_match = re.search(r"rflambda_([\dpm]+)_([\dpm]+)", filename)
    if rf_lambda_match:
        params["refine_schedule"] = "lambda"
        params["refine_lambda_a"] = rf_lambda_match.group(1).replace("p", ".")
        params["refine_lambda_b"] = rf_lambda_match.group(2).replace("p", ".")

    # 2. Linear调度模式：rflinear_5_20
    rf_linear_match = re.search(r"rflinear_(\d+)_(\d+)", filename)
    if rf_linear_match:
        params["refine_schedule"] = "linear"
        params["refine_linear_lower"] = int(rf_linear_match.group(1))
        params["refine_linear_upper"] = int(rf_linear_match.group(2))

    # 3. Fixed调度模式：rffixed_10
    rf_fixed_match = re.search(r"rffixed_(\d+)", filename)
    if rf_fixed_match:
        params["refine_schedule"] = "fixed"
        params["refine_stride"] = int(rf_fixed_match.group(1))

    # 4. 步长和噪声参数（可选）：rfstep_0p3_rfnoise_0p08
    rf_step_match = re.search(r"rfstep_([\dpm]+)", filename)
    if rf_step_match:
        params["refine_step_size"] = rf_step_match.group(1).replace("p", ".")

    rf_noise_match = re.search(r"rfnoise_([\dpm]+)", filename)
    if rf_noise_match:
        params["refine_noise_scale"] = rf_noise_match.group(1).replace("p", ".")

    return params


def find_latest_batchsummary_file(output_dir=None):
    """
    查找最新的 batchsummary 文件

    Args:
        output_dir: 输出目录（可选，用于推断 batchsummary 目录）

    Returns:
        Path对象或None
    """
    batchsummary_dir = REPO_ROOT / "batchsummary"

    if not batchsummary_dir.exists():
        return None

    # 查找所有 batch_evaluation_summary_*.csv / *.xlsx 文件（以最新修改时间为准）
    files = glob.glob(str(batchsummary_dir / "batch_evaluation_summary_*.csv"))
    files.extend(glob.glob(str(batchsummary_dir / "batch_evaluation_summary_*.xlsx")))

    if not files:
        return None

    # 按修改时间排序，返回最新的
    files.sort(key=os.path.getmtime, reverse=True)
    return Path(files[0])


def update_bestchoice_excel_with_params(output_dir=None):
    """
    从最新的 batchsummary 文件中读取参数，并填写到 evaall bestchoice.xlsx
    只填写一次，避免重复填写

    Args:
        output_dir: 输出目录（可选）
    """
    if pd is None:
        print("  ⚠️  pandas未安装，跳过更新 evaall bestchoice.xlsx")
        return

    try:
        # 查找最新的 batchsummary 文件
        batchsummary_file = find_latest_batchsummary_file(output_dir)
        if batchsummary_file is None:
            print("  ⚠️  未找到 batchsummary 文件，跳过更新 evaall bestchoice.xlsx")
            return

        print(f"  📝 找到 batchsummary 文件: {batchsummary_file.name}")

        # 从文件名提取参数
        params = parse_params_from_batchsummary_filename(batchsummary_file.name)
        if not params:
            print("  ⚠️  未能从文件名中提取参数，跳过更新")
            return

        print(f"  📊 提取的参数: {params}")

        # 从 batchsummary 读取统计信息
        stats_dict = {}
        try:
            if batchsummary_file.suffix.lower() == ".csv":
                stats_path = batchsummary_file.with_name(
                    batchsummary_file.stem + "_统计信息.csv"
                )
                if stats_path.exists():
                    stats_df = pd.read_csv(stats_path, encoding="utf-8-sig")
                    if (
                        not stats_df.empty
                        and "统计项目" in stats_df.columns
                        and "数值" in stats_df.columns
                    ):
                        stats_dict = dict(zip(stats_df["统计项目"], stats_df["数值"]))
                        print(f"  📊 读取到统计信息: {len(stats_dict)} 项")
            else:
                stats_df = pd.read_excel(
                    batchsummary_file, sheet_name="统计信息", engine="openpyxl"
                )
                if (
                    not stats_df.empty
                    and "统计项目" in stats_df.columns
                    and "数值" in stats_df.columns
                ):
                    stats_dict = dict(zip(stats_df["统计项目"], stats_df["数值"]))
                    print(f"  📊 读取到统计信息: {len(stats_dict)} 项")
        except Exception as e:
            print(f"  ⚠️  读取统计信息失败: {e}")

        # 从 batchsummary 读取配置参数（获取步数和取模步长）
        config_params_dict = {}
        total_steps = ""
        actual_length = ""
        try:
            if batchsummary_file.suffix.lower() == ".csv":
                config_path_csv = batchsummary_file.with_name(
                    batchsummary_file.stem + "_配置参数.csv"
                )
                if config_path_csv.exists():
                    config_df = pd.read_csv(config_path_csv, encoding="utf-8-sig")
                    if (
                        not config_df.empty
                        and "参数路径" in config_df.columns
                        and "参数值" in config_df.columns
                    ):
                        config_params_dict = dict(
                            zip(config_df["参数路径"], config_df["参数值"])
                        )
                        total_steps = config_params_dict.get("计算.跳步总次数", "")
                        actual_length = config_params_dict.get("计算.实际长度", "")
                        print(
                            f"  📊 读取到配置参数: 步数={total_steps}, 取模步长={actual_length}"
                        )
            else:
                config_df = pd.read_excel(
                    batchsummary_file, sheet_name="配置参数", engine="openpyxl"
                )
                if (
                    not config_df.empty
                    and "参数路径" in config_df.columns
                    and "参数值" in config_df.columns
                ):
                    config_params_dict = dict(
                        zip(config_df["参数路径"], config_df["参数值"])
                    )
                    total_steps = config_params_dict.get("计算.跳步总次数", "")
                    actual_length = config_params_dict.get("计算.实际长度", "")
                    print(
                        f"  📊 读取到配置参数: 步数={total_steps}, 取模步长={actual_length}"
                    )
        except Exception as e:
            print(f"  ⚠️  读取配置参数失败: {e}")

        # 创建参数签名用于检查是否已存在
        param_signature = (
            params.get("grad_fusion_mode", ""),
            params.get("grad_fusion_start", ""),
            params.get("grad_fusion_end", ""),
            params.get("time_boundary", ""),
            params.get("large_step_schedule", ""),
            params.get("large_step_lambda_a", "")
            or params.get("large_step_linear_lower", "")
            or params.get("large_step_stride", ""),
            params.get("large_step_lambda_b", "")
            or params.get("large_step_linear_upper", ""),
            params.get("large_step_step_size", ""),
            params.get("large_step_noise_scale", ""),
            params.get("refine_schedule", ""),
            params.get("refine_lambda_a", "")
            or params.get("refine_linear_lower", "")
            or params.get("refine_stride", ""),
            params.get("refine_lambda_b", "") or params.get("refine_linear_upper", ""),
            params.get("refine_step_size", ""),
            params.get("refine_noise_scale", ""),
        )

        # 目标文件路径
        bestchoice_file = REPO_ROOT / "evaall bestchoice.xlsx"

        # 计算下降速率
        grad_fusion_start = params.get("grad_fusion_start")
        grad_fusion_end = params.get("grad_fusion_end")
        time_boundary = params.get("time_boundary")
        descent_rate = ""
        if (
            grad_fusion_start is not None
            and grad_fusion_end is not None
            and time_boundary
        ):
            try:
                start_val = (
                    float(grad_fusion_start)
                    if isinstance(grad_fusion_start, str)
                    else grad_fusion_start
                )
                end_val = (
                    float(grad_fusion_end)
                    if isinstance(grad_fusion_end, str)
                    else grad_fusion_end
                )
                tl_val = (
                    float(time_boundary)
                    if isinstance(time_boundary, str)
                    else time_boundary
                )
                if tl_val > 0:
                    descent_rate = (start_val - end_val) / tl_val
            except (ValueError, TypeError):
                pass

        # 准备新行数据（按照用户要求的格式）
        new_row_data = {
            "文件名": batchsummary_file.name,
            "权重策略": params.get("grad_fusion_mode", ""),
            "下降速率": descent_rate if descent_rate != "" else "",
            "开始权重": grad_fusion_start if grad_fusion_start is not None else "",
            "结束权重": grad_fusion_end if grad_fusion_end is not None else "",
            "时间长度 (TL)": time_boundary if time_boundary is not None else "",
            "LSstepsize": params.get("large_step_step_size", ""),
            "LSnosie": params.get("large_step_noise_scale", ""),
            "LSLambda1": params.get("large_step_lambda_a", ""),
            "LSLambda2": params.get("large_step_lambda_b", ""),
            "RFstepsize": params.get("refine_step_size", ""),
            "RFnosie": params.get("refine_noise_scale", ""),
            "RFLambda1": params.get("refine_lambda_a", ""),
            "RFLambda2": params.get("refine_lambda_b", ""),
            "步数": total_steps if total_steps != "" else "",
            "取模步长": actual_length if actual_length != "" else "",
            "可重建率 (%)": stats_dict.get(
                "重建成功百分比(%)", stats_dict.get("重建成功率(%)", "")
            ),
            "对接成功率 (%)": stats_dict.get(
                "对接成功百分比(%)", stats_dict.get("评估成功率(%)", "")
            ),
            "Vina_Dock 亲和力": stats_dict.get(
                "Vina_Dock_平均亲和力", stats_dict.get("Vina_Dock_最佳亲和力", "")
            ),
            "Vina_ScoreOnly": stats_dict.get(
                "Vina_ScoreOnly_平均亲和力",
                stats_dict.get("Vina_ScoreOnly_最佳亲和力", ""),
            ),
            "Vina_Minimize": stats_dict.get(
                "Vina_Minimize_平均亲和力",
                stats_dict.get("Vina_Minimize_最佳亲和力", ""),
            ),
            "QED 评分（均值）": stats_dict.get("平均QED", ""),
        }

        # 转换数值类型
        numeric_fields = [
            "下降速率",
            "开始权重",
            "结束权重",
            "时间长度 (TL)",
            "LSstepsize",
            "LSnosie",
            "LSLambda1",
            "LSLambda2",
            "RFstepsize",
            "RFnosie",
            "RFLambda1",
            "RFLambda2",
            "步数",
            "取模步长",
            "可重建率 (%)",
            "对接成功率 (%)",
            "Vina_Dock 亲和力",
            "Vina_ScoreOnly",
            "Vina_Minimize",
            "QED 评分（均值）",
        ]
        for field in numeric_fields:
            if field in new_row_data and new_row_data[field] != "":
                try:
                    val = new_row_data[field]
                    if isinstance(val, str):
                        val = val.replace("%", "").strip()
                        if val:
                            new_row_data[field] = float(val)
                    elif isinstance(val, (int, float)):
                        new_row_data[field] = float(val)
                except (ValueError, TypeError):
                    pass

        # 定义列的顺序（按照用户要求的格式）
        column_order = [
            "文件名",
            "权重策略",
            "下降速率",
            "开始权重",
            "结束权重",
            "时间长度 (TL)",
            "LSstepsize",
            "LSnosie",
            "LSLambda1",
            "LSLambda2",
            "RFstepsize",
            "RFnosie",
            "RFLambda1",
            "RFLambda2",
            "步数",
            "取模步长",
            "可重建率 (%)",
            "对接成功率 (%)",
            "Vina_Dock 亲和力",
            "Vina_ScoreOnly",
            "Vina_Minimize",
            "QED 评分（均值）",
        ]

        # 尝试读取目标文件
        try:
            # 先尝试用 pandas 读取（更可靠）
            if bestchoice_file.exists():
                df = pd.read_excel(bestchoice_file, engine="openpyxl")
            else:
                df = pd.DataFrame()

            # 检查是否已存在相同的文件名（避免重复填写）
            if len(df) > 0 and "文件名" in df.columns:
                if batchsummary_file.name in df["文件名"].values:
                    print(f"  ✅ 文件名已存在，跳过填写")
                    return

            # 创建新行 DataFrame
            new_row_df = pd.DataFrame([new_row_data])

            # 合并数据
            if len(df) > 0:
                # 确保列顺序一致：先使用标准列顺序，然后添加其他列
                all_columns = column_order.copy()
                # 添加现有文件中存在但不在标准列中的列
                for col in df.columns:
                    if col not in all_columns:
                        all_columns.append(col)
                # 添加新行中存在但不在标准列中的列
                for col in new_row_df.columns:
                    if col not in all_columns:
                        all_columns.append(col)

                df = df.reindex(columns=all_columns, fill_value="")
                new_row_df = new_row_df.reindex(columns=all_columns, fill_value="")
                df = pd.concat([df, new_row_df], ignore_index=True)
            else:
                df = new_row_df

            # 重新排列列顺序，确保标准列在前
            final_columns = column_order.copy()
            for col in df.columns:
                if col not in final_columns:
                    final_columns.append(col)
            df = df.reindex(columns=final_columns)

            # 保存文件
            df.to_excel(bestchoice_file, index=False, engine="openpyxl")
            print(f"  ✅ 已更新 evaall bestchoice.xlsx，数据填写在第 {len(df)} 行")

        except Exception as e:
            # 如果 pandas 读取失败，尝试用 openpyxl
            print(f"  ⚠️  pandas 读取失败，尝试使用 openpyxl: {e}")
            try:
                from openpyxl import load_workbook, Workbook

                # 尝试读取或创建文件
                if bestchoice_file.exists():
                    wb = load_workbook(bestchoice_file, read_only=False)
                    ws = wb.active
                else:
                    wb = Workbook()
                    ws = wb.active
                    # 创建标题行
                    for col_idx, col_name in enumerate(column_order, start=1):
                        ws.cell(row=1, column=col_idx).value = col_name

                # 检查是否已存在相同的文件名
                header_row = 1
                filename_col = None
                for col_idx in range(1, ws.max_column + 1):
                    cell_value = ws.cell(row=header_row, column=col_idx).value
                    if cell_value and str(cell_value).strip() == "文件名":
                        filename_col = col_idx
                        break

                if filename_col:
                    for row_idx in range(header_row + 1, ws.max_row + 1):
                        cell_value = ws.cell(row=row_idx, column=filename_col).value
                        if (
                            cell_value
                            and str(cell_value).strip() == batchsummary_file.name
                        ):
                            print(f"  ✅ 文件名已存在（第 {row_idx} 行），跳过填写")
                            wb.close()
                            return

                # 找到第一个空行或最后一行
                current_row = header_row + 1
                while current_row <= ws.max_row:
                    first_cell = ws.cell(row=current_row, column=1).value
                    if first_cell is None or str(first_cell).strip() == "":
                        break
                    current_row += 1

                if current_row > ws.max_row:
                    current_row = ws.max_row + 1

                # 查找或创建列
                col_mapping = {}
                for col_idx in range(1, ws.max_column + 1):
                    cell_value = ws.cell(row=header_row, column=col_idx).value
                    if cell_value:
                        col_mapping[str(cell_value).strip()] = col_idx

                # 填写数据
                next_col = ws.max_column + 1
                for col_name in column_order:
                    if col_name in col_mapping:
                        col_idx = col_mapping[col_name]
                    else:
                        # 创建新列
                        ws.cell(row=header_row, column=next_col).value = col_name
                        col_idx = next_col
                        next_col += 1

                    value = new_row_data.get(col_name, "")
                    if value != "":
                        ws.cell(row=current_row, column=col_idx).value = value

                # 保存文件
                wb.save(bestchoice_file)
                print(
                    f"  ✅ 已更新 evaall bestchoice.xlsx（使用 openpyxl），数据填写在第 {current_row} 行"
                )
                wb.close()

            except Exception as e2:
                print(f"  ❌ openpyxl 更新也失败: {e2}")
                traceback.print_exc()

    except Exception as e:
        print(f"  ⚠️  更新 evaall bestchoice.xlsx 时出错: {e}")
        traceback.print_exc()
