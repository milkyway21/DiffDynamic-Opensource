"""Reports responsibilities extracted from batch_sampleandeval_parallel.py."""

from diffdynamic.batch.state import (
    OUTPUT_DIR,
    POCKET_TIME_EXCEL,
    Path,
    datetime,
    np,
    os,
    pd,
    time,
    timedelta,
    timezone,
    traceback,
)
from diffdynamic.batch import state as _state


def collect_all_evaluation_results(results, batch_start_time, extra_eval_roots=None):
    """
    从评估结果 Excel 中收集对接成功的分子数据。

    优先仅扫描 **本批次** ``process_evaluation_task`` 返回的 ``eval_output_dir``，
    避免误把 ``outputs/eval_*`` 下其它时间生成的 ``evaluation_results_*.xlsx`` 合并进来。
    若无可用目录信息（兼容旧调用），再回退为扫描 ``eval_*`` + mtime 过滤。

    Args:
        results: 本批次 ``all_results``，含 (data_id, success, ..., eval_output_dir)。
        extra_eval_roots: 额外扫描的父目录列表（例如 MolForm 的 third_party/MolForm/outputs），
            用于单独评估模式时 .pt 不在仓库默认 outputs/ 下的情况。

    Returns:
        tuple: (molecule_records, summary_stats, pocket_stats_list)
            - molecule_records: 所有分子的记录列表
            - summary_stats: 汇总统计信息
            - pocket_stats_list: 每个口袋的统计信息列表（用于加权均值计算）
    """
    from diffdynamic.batch.results import (
        _data_id_from_eval_dir_name,
        _eval_output_dirs_from_batch_results,
    )

    if pd is None:
        print("⚠️  pandas未安装，无法读取Excel文件")
        return [], {}, []

    molecule_records = []
    total_num_samples = 0
    total_n_reconstruct_success = 0
    total_n_eval_success = 0
    pocket_stats_list = []  # 存储每个口袋的统计信息

    batch_start_datetime = datetime.fromtimestamp(batch_start_time)

    scoped_eval_dirs = _eval_output_dirs_from_batch_results(results)
    use_scoped_only = len(scoped_eval_dirs) > 0

    if use_scoped_only:
        eval_dirs = []
        for p in scoped_eval_dirs:
            if p.is_dir():
                eval_dirs.append(p)
            else:
                print(f"⚠️  本批次评估目录不存在或已删除，跳过: {p}")
        if extra_eval_roots:
            for root in extra_eval_roots:
                root = Path(root)
                if not root.is_dir():
                    continue
                for p in scoped_eval_dirs:
                    if p.is_dir():
                        continue
                    alt = root / p.name
                    if alt.is_dir():
                        eval_dirs.append(alt.resolve())
        eval_dirs = list({p.resolve(): p for p in eval_dirs}.values())
        print(
            f"本批次限定 {len(eval_dirs)} 个评估输出目录（不扫全量 outputs/eval_*），读取其中 evaluation_results_*.xlsx"
        )
    else:
        # 兼容：无 eval_output_dir 时沿用旧逻辑（易混入同时间段其它评估，仅作回退）
        eval_dirs = list(OUTPUT_DIR.glob("eval_*"))
        if extra_eval_roots:
            for root in extra_eval_roots:
                root = Path(root)
                if root.is_dir():
                    eval_dirs.extend(root.glob("eval_*"))
        eval_dirs = list({p.resolve(): p for p in eval_dirs}.values())
        if not eval_dirs:
            print(f"⚠️  未找到任何评估目录（在 {OUTPUT_DIR} 等处查找 eval_*）")
            return [], {}, []
        print(
            f"⚠️  未从批次结果解析到 eval_output_dir，回退为扫描 {len(eval_dirs)} 个 eval_* 目录（可能混入历史评估）"
        )

    if not eval_dirs:
        print(f"⚠️  未找到任何可读取的评估目录")
        return [], {}, []

    print(f"将处理 {len(eval_dirs)} 个评估目录中的 Excel...")

    for eval_dir in eval_dirs:
        if not eval_dir.is_dir():
            continue

        # 查找该目录下的所有evaluation_results_*.xlsx文件
        excel_files = list(eval_dir.glob("evaluation_results_*.xlsx"))

        if not excel_files:
            continue

        # 本批次限定目录：直接采用该目录下 xlsx（避免 mtime 与脚本启动先后导致漏读）
        if use_scoped_only:
            recent_excel_files = excel_files
        else:
            recent_excel_files = [
                f
                for f in excel_files
                if datetime.fromtimestamp(f.stat().st_mtime) >= batch_start_datetime
            ]

        if not recent_excel_files:
            continue

        # 选择最新的Excel文件（如果有多个）
        latest_excel_file = max(recent_excel_files, key=os.path.getmtime)

        try:
            # 读取Excel文件的"评估结果"工作表
            df = pd.read_excel(
                latest_excel_file, sheet_name="评估结果", engine="openpyxl"
            )

            if df.empty:
                continue

            # 将DataFrame转换为记录列表
            for _, row in df.iterrows():
                record = row.to_dict()

                # 处理NaN值，将NaN转换为None（后续处理会统一处理None和'N/A'）
                for key, value in record.items():
                    if pd.isna(value):
                        record[key] = None

                # 确保数据ID字段存在（从目录名或文件名中提取）
                data_id = record.get("数据ID")
                if data_id is None or (isinstance(data_id, float) and pd.isna(data_id)):
                    parsed = _data_id_from_eval_dir_name(eval_dir.name)
                    if parsed is not None:
                        record["数据ID"] = parsed

                molecule_records.append(record)

            # 获取数据ID
            data_id = _data_id_from_eval_dir_name(eval_dir.name)

            # 读取统计信息工作表以获取统计数据
            pocket_stats = {"数据ID": data_id, "评估成功数": len(df)}
            try:
                stats_df = pd.read_excel(
                    latest_excel_file, sheet_name="统计信息", engine="openpyxl"
                )
                if (
                    not stats_df.empty
                    and "统计项目" in stats_df.columns
                    and "数值" in stats_df.columns
                ):
                    stats_dict = dict(zip(stats_df["统计项目"], stats_df["数值"]))

                    # 提取统计数据
                    if "总样本数" in stats_dict:
                        try:
                            total_num_samples += int(stats_dict["总样本数"])
                        except (ValueError, TypeError):
                            pass

                    if "重建成功" in stats_dict:
                        try:
                            total_n_reconstruct_success += int(stats_dict["重建成功"])
                        except (ValueError, TypeError):
                            pass

                    if "评估成功" in stats_dict:
                        try:
                            eval_success = int(stats_dict["评估成功"])
                            total_n_eval_success += eval_success
                            pocket_stats["评估成功数"] = eval_success
                        except (ValueError, TypeError):
                            pass

                    # 提取各个参数的均值（用于加权平均计算）
                    # 查找所有包含"平均"的统计项目
                    for stat_name, stat_value in stats_dict.items():
                        if "平均" in stat_name and stat_value not in ("N/A", None, ""):
                            try:
                                # 尝试转换为数值
                                if isinstance(stat_value, str):
                                    # 尝试转换为浮点数
                                    stat_value = float(stat_value)
                                pocket_stats[stat_name] = stat_value
                            except (ValueError, TypeError):
                                pass

                    # 也查找一些常见的统计指标（即使没有"平均"字样）
                    for stat_name in [
                        "Vina_Dock_最佳亲和力",
                        "Vina_Dock_最差亲和力",
                        "Vina_ScoreOnly_最佳亲和力",
                        "Vina_Minimize_最佳亲和力",
                    ]:
                        if stat_name in stats_dict:
                            try:
                                val = stats_dict[stat_name]
                                if val not in ("N/A", None, "") and not pd.isna(val):
                                    if isinstance(val, str):
                                        val = float(val)
                                    pocket_stats[stat_name] = val
                            except (ValueError, TypeError):
                                pass
            except Exception as e:
                # 如果读取统计信息失败，不影响主流程
                pass

            # 如果没有从统计信息中获取到评估成功数，使用分子记录数
            if pocket_stats["评估成功数"] == 0:
                pocket_stats["评估成功数"] = len(df)

            # 添加到口袋统计列表
            if pocket_stats["评估成功数"] > 0:
                pocket_stats_list.append(pocket_stats)

            print(f"  ✅ 已读取: {latest_excel_file.name} ({len(df)} 条记录)")

        except Exception as e:
            print(f"⚠️  读取Excel文件失败 {latest_excel_file}: {e}")
            continue

    # 计算汇总统计信息
    # 计算百分比
    reconstruct_success_rate = (
        (total_n_reconstruct_success / total_num_samples * 100)
        if total_num_samples > 0
        else 0.0
    )
    docking_success_rate = (
        (len(molecule_records) / total_num_samples * 100)
        if total_num_samples > 0
        else 0.0
    )

    # 计算有效分子数量（剔除vinadock>0、vinascore>0或vinamin>0的异常数据）
    # 注意：不删除任何能对接的分子，只是统计有效分子数量
    valid_molecule_count = 0
    for r in molecule_records:
        vina_dock = r.get("Vina_Dock_亲和力", "N/A")
        vina_score = r.get("Vina_ScoreOnly_亲和力", "N/A")
        vina_min = r.get("Vina_Minimize_亲和力", "N/A")

        # 检查是否为异常数据（vinadock>0、vinascore>0或vinamin>0）
        is_abnormal = False
        try:
            if vina_dock not in ("N/A", None) and not pd.isna(vina_dock):
                if float(vina_dock) > 0:
                    is_abnormal = True
            if vina_score not in ("N/A", None) and not pd.isna(vina_score):
                if float(vina_score) > 0:
                    is_abnormal = True
            if vina_min not in ("N/A", None) and not pd.isna(vina_min):
                if float(vina_min) > 0:
                    is_abnormal = True
        except (ValueError, TypeError):
            pass

        # 如果不是异常数据，则计入有效分子
        if not is_abnormal:
            valid_molecule_count += 1

    # 计算有效分子比例（有效分子数量 / 应生成分子数）
    valid_molecule_ratio = (
        (valid_molecule_count / total_num_samples * 100)
        if total_num_samples > 0
        else 0.0
    )

    summary_stats = {
        "batch启动时间": datetime.fromtimestamp(batch_start_time).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "应生成分子数": total_num_samples,
        "可重建分子数": total_n_reconstruct_success,
        "重建成功百分比(%)": f"{reconstruct_success_rate:.2f}",
        "对接成功分子数": len(molecule_records),
        "对接成功百分比(%)": f"{docking_success_rate:.2f}",
        "有效分子数量": valid_molecule_count,
        "有效分子比例(%)": f"{valid_molecule_ratio:.2f}",
    }

    if molecule_records:
        # 计算平均得分
        vina_dock_scores = []
        vina_score_only_scores = []
        vina_minimize_scores = []
        qed_values = []
        sa_values = []

        for r in molecule_records:
            # 处理Vina_Dock_亲和力
            vina_dock = r.get("Vina_Dock_亲和力", "N/A")
            if vina_dock not in ("N/A", None) and not pd.isna(vina_dock):
                try:
                    vina_dock_scores.append(float(vina_dock))
                except (ValueError, TypeError):
                    pass

            # 处理Vina_ScoreOnly_亲和力
            vina_score_only = r.get("Vina_ScoreOnly_亲和力", "N/A")
            if vina_score_only not in ("N/A", None) and not pd.isna(vina_score_only):
                try:
                    vina_score_only_scores.append(float(vina_score_only))
                except (ValueError, TypeError):
                    pass

            # 处理Vina_Minimize_亲和力
            vina_minimize = r.get("Vina_Minimize_亲和力", "N/A")
            if vina_minimize not in ("N/A", None) and not pd.isna(vina_minimize):
                try:
                    vina_minimize_scores.append(float(vina_minimize))
                except (ValueError, TypeError):
                    pass

            # 处理QED评分
            qed = r.get("QED评分", "N/A")
            if qed not in ("N/A", None) and not pd.isna(qed):
                try:
                    qed_values.append(float(qed))
                except (ValueError, TypeError):
                    pass

            # 处理SA评分
            sa = r.get("SA评分", "N/A")
            if sa not in ("N/A", None) and not pd.isna(sa):
                try:
                    sa_values.append(float(sa))
                except (ValueError, TypeError):
                    pass

        if vina_dock_scores:
            summary_stats["Vina_Dock_平均亲和力"] = np.mean(vina_dock_scores)
        if vina_score_only_scores:
            summary_stats["Vina_ScoreOnly_平均亲和力"] = np.mean(vina_score_only_scores)
        if vina_minimize_scores:
            summary_stats["Vina_Minimize_平均亲和力"] = np.mean(vina_minimize_scores)
        if qed_values:
            summary_stats["QED平均评分"] = np.mean(qed_values)
        if sa_values:
            summary_stats["SA平均评分"] = np.mean(sa_values)

    return molecule_records, summary_stats, pocket_stats_list


def calculate_weighted_means(pocket_stats_list):
    """
    计算加权均值

    Args:
        pocket_stats_list: 每个口袋的统计信息列表

    Returns:
        list: 加权均值计算结果列表
    """
    if not pocket_stats_list or len(pocket_stats_list) == 0:
        return []

    try:
        # 转换为DataFrame
        df = pd.DataFrame(pocket_stats_list)

        # 计算总成功对接数
        total_success = df["评估成功数"].sum()
        if total_success == 0:
            return []

        # 找到所有包含"平均"的列（这些是需要计算加权均值的参数）
        valid_mean_columns = [
            col
            for col in df.columns
            if "平均" in col and col != "评估成功数" and col != "数据ID"
        ]

        if not valid_mean_columns:
            return []

        # 存储计算结果
        results = []

        for col in valid_mean_columns:
            # 创建临时数据框，只包含非空值的行
            temp_df = df[["评估成功数", col]].dropna()

            # 计算简单平均值（所有口袋的平均值，不管是否有数据）
            simple_mean = df[col].mean()

            if len(temp_df) == 0:
                real_mean = np.nan
                weighted_sum = 0.0
                used_pockets = 0
            else:
                # 计算加权和：评估成功数 × 参数均值
                temp_df["加权和"] = temp_df["评估成功数"] * temp_df[col]

                # 计算真实均值（加权平均）
                weighted_sum = temp_df["加权和"].sum()
                used_pockets = len(temp_df)
                real_mean = weighted_sum / total_success

            # 计算差异百分比
            if not pd.isna(simple_mean) and not pd.isna(real_mean) and simple_mean != 0:
                diff_percent = (real_mean - simple_mean) / simple_mean * 100
            else:
                diff_percent = np.nan

            results.append(
                {
                    "参数名称": col,
                    "参与计算的口袋数": used_pockets,
                    "加权和": weighted_sum,
                    "简单平均值": simple_mean if not pd.isna(simple_mean) else np.nan,
                    "真实均值（加权平均）": real_mean
                    if not pd.isna(real_mean)
                    else np.nan,
                    "差异百分比": diff_percent if not pd.isna(diff_percent) else np.nan,
                }
            )

        return results

    except Exception as e:
        print(f"⚠️  计算加权均值失败: {e}")
        traceback.print_exc()
        return []
