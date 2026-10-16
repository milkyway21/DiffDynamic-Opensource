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


def record_pocket_generation_time(
    data_id, gpu_id, generation_time, status="成功", pt_file=None
):
    """
    记录口袋生成时间和GPU信息到Excel文件

    Args:
        data_id: 口袋ID
        gpu_id: 使用的GPU ID
        generation_time: 生成时间（秒）
        status: 生成状态（成功/失败）
        pt_file: 生成的PT文件路径
    """
    if pd is None:
        return False

    try:
        # 使用本地时区（CST，UTC+8）生成时间戳
        cst = timezone(timedelta(hours=8))
        timestamp_str = datetime.now(cst).strftime("%Y-%m-%d %H:%M:%S")

        # 准备数据
        new_row = {
            "Pocket_ID": f"pocket_{data_id}",
            "Data_ID": data_id,
            "GPU_ID": gpu_id,
            "Generation_Time(s)": round(generation_time, 2),
            "Status": status,
            "Timestamp": timestamp_str,
            "PT_File": str(pt_file) if pt_file else "",
        }

        # 定义工作表名称
        sheet_name = "DiffDynamic_Generation_Time"

        # 检查Excel文件是否存在
        if POCKET_TIME_EXCEL.exists():
            try:
                # 读取所有工作表
                xl = pd.ExcelFile(str(POCKET_TIME_EXCEL), engine="openpyxl")
                sheet_names = xl.sheet_names

                if sheet_name in sheet_names:
                    # 读取现有工作表
                    df = pd.read_excel(
                        str(POCKET_TIME_EXCEL), sheet_name=sheet_name, engine="openpyxl"
                    )
                else:
                    # 工作表不存在，创建新的DataFrame
                    df = pd.DataFrame(
                        columns=[
                            "Pocket_ID",
                            "Data_ID",
                            "GPU_ID",
                            "Generation_Time(s)",
                            "Status",
                            "Timestamp",
                            "PT_File",
                        ]
                    )

                # 读取其他工作表以便后续写入
                other_sheets = {}
                for sn in sheet_names:
                    if sn != sheet_name:
                        other_sheets[sn] = pd.read_excel(
                            str(POCKET_TIME_EXCEL), sheet_name=sn, engine="openpyxl"
                        )

            except Exception as e:
                print(f"[记录] ⚠️  读取Excel失败，创建新工作表: {e}")
                df = pd.DataFrame(
                    columns=[
                        "Pocket_ID",
                        "Data_ID",
                        "GPU_ID",
                        "Generation_Time(s)",
                        "Status",
                        "Timestamp",
                        "PT_File",
                    ]
                )
                other_sheets = {}
        else:
            # 文件不存在，创建新的DataFrame
            df = pd.DataFrame(
                columns=[
                    "Pocket_ID",
                    "Data_ID",
                    "GPU_ID",
                    "Generation_Time(s)",
                    "Status",
                    "Timestamp",
                    "PT_File",
                ]
            )
            other_sheets = {}

        # 检查是否已存在相同的Pocket_ID记录
        existing_idx = df[df["Pocket_ID"] == f"pocket_{data_id}"].index
        if len(existing_idx) > 0:
            # 更新现有记录
            df.loc[existing_idx[0]] = new_row
            action = "更新"
        else:
            # 添加新记录
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            action = "添加"

        print(
            f"[记录] {action}口袋 {data_id} 的生成时间记录: {generation_time:.2f}s (GPU {gpu_id}, 状态: {status})"
        )

        # 保存到Excel（保留其他工作表）
        try:
            # 创建工作表字典
            sheets_to_write = {sheet_name: df}
            sheets_to_write.update(other_sheets)

            # 写入所有工作表
            with pd.ExcelWriter(str(POCKET_TIME_EXCEL), engine="openpyxl") as writer:
                for sn, s_df in sheets_to_write.items():
                    s_df.to_excel(writer, sheet_name=sn, index=False)

            return True
        except Exception as e:
            print(f"[记录] ⚠️  写入Excel失败: {e}")
            return False

    except Exception as e:
        print(f"[记录] ⚠️  记录口袋生成时间失败 (data_id={data_id}): {e}")
        import traceback

        traceback.print_exc()
        return False


def append_to_excel(
    excel_file,
    timestamp,
    execution_time,
    data_id,
    pt_file,
    vina_mean,
    vina_median,
    num_scores,
    status,
    message,
):
    """
    将每个口袋的评估结果追加到 CSV（与批次主表 ``batch_evaluation_summary_*.csv`` 同目录，进程安全）。
    使用临时文件 + 原子替换；fcntl 文件锁跨进程同步。
    主批次分子表由 ``save_molecules_to_excel`` 在任务结束后写入，此处不修改主表。
    """
    if pd is None:
        return False

    import fcntl  # 用于文件锁

    # 确保 excel_file 是 Path 对象
    if not isinstance(excel_file, Path):
        excel_file = Path(excel_file)

    csv_records = excel_file.with_name(excel_file.stem + "_评估记录.csv")
    csv_stats = excel_file.with_name(excel_file.stem + "_评估统计.csv")

    # 创建锁文件路径（与同 stem 的评估 CSV 共用）
    lock_file = excel_file.with_name(excel_file.stem + "_评估记录.csv.lock")

    # 使用文件锁确保进程安全
    lock_acquired = False
    lock_fd = None
    try:
        # 打开或创建锁文件
        lock_fd = open(str(lock_file), "w")
        # 获取独占锁（非阻塞），最多等待10秒
        for attempt in range(100):  # 100 * 0.1s = 10s
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_acquired = True
                break
            except (IOError, OSError):
                time.sleep(0.1)

        if not lock_acquired:
            print(f"[评估CSV] ⚠️  获取文件锁超时 (data_id={data_id})")
            return False

        try:
            new_row = {
                "执行时间": timestamp,
                "执行耗时(秒)": execution_time,
                "数据ID": data_id,
                "PT文件": os.path.basename(str(pt_file)) if pt_file else "",
                "Vina平均得分": vina_mean if vina_mean is not None else "",
                "Vina中位数得分": vina_median if vina_median is not None else "",
                "得分数量": num_scores if num_scores else 0,
                "状态": status,
                "备注": message,
            }

            # 定义必需的列
            required_columns = [
                "执行时间",
                "执行耗时(秒)",
                "数据ID",
                "PT文件",
                "Vina平均得分",
                "Vina中位数得分",
                "得分数量",
                "状态",
                "备注",
                "累计均值",
            ]

            df = pd.DataFrame()
            cumulative_mean = vina_mean

            if csv_records.exists():
                try:
                    df = pd.read_csv(csv_records, encoding="utf-8-sig")
                    if df.empty or not all(
                        col in df.columns for col in required_columns[:-1]
                    ):
                        print("⚠️  警告: 评估记录 CSV 列不匹配，重新创建表。")
                        df = pd.DataFrame()
                    else:
                        if "累计均值" not in df.columns:
                            df["累计均值"] = ""
                        if "状态" in df.columns:
                            successful_rows = df[df["状态"] == "成功"]
                            if len(successful_rows) > 0:
                                all_means = (
                                    successful_rows["Vina平均得分"].dropna().tolist()
                                )
                                if vina_mean is not None:
                                    all_means.append(vina_mean)
                                cumulative_mean = (
                                    np.mean(all_means) if all_means else None
                                )
                except Exception as e:
                    print(
                        f"⚠️  警告: 读取评估记录 CSV 失败 {csv_records}: {e}，将新建。"
                    )
                    df = pd.DataFrame()

            new_row["累计均值"] = cumulative_mean if cumulative_mean is not None else ""

            # 确保所有列都存在
            for col in required_columns:
                if col not in new_row:
                    new_row[col] = ""

            new_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_df], ignore_index=True)

            excel_file.parent.mkdir(parents=True, exist_ok=True)
            temp_records = csv_records.with_suffix(".csv.tmp")
            temp_stats = csv_stats.with_suffix(".csv.tmp")

            try:
                df.to_csv(temp_records, index=False, encoding="utf-8-sig")
                temp_records.replace(csv_records)

                if len(df) > 0 and "状态" in df.columns:
                    successful_df = df[df["状态"] == "成功"]
                    if len(successful_df) > 0:
                        stats = {
                            "统计项目": [
                                "总评估次数",
                                "成功次数",
                                "失败次数",
                                "当前累计均值",
                                "当前累计中位数",
                                "最佳得分",
                                "最差得分",
                            ],
                            "数值": [
                                len(df),
                                len(successful_df),
                                len(df) - len(successful_df),
                                successful_df["Vina平均得分"].mean()
                                if len(successful_df) > 0
                                else "",
                                successful_df["Vina平均得分"].median()
                                if len(successful_df) > 0
                                else "",
                                successful_df["Vina平均得分"].min()
                                if len(successful_df) > 0
                                else "",
                                successful_df["Vina平均得分"].max()
                                if len(successful_df) > 0
                                else "",
                            ],
                        }
                        stats_df = pd.DataFrame(stats)
                        stats_df.to_csv(temp_stats, index=False, encoding="utf-8-sig")
                        temp_stats.replace(csv_stats)

            except Exception as e:
                for t in (temp_records, temp_stats):
                    if t.exists():
                        try:
                            t.unlink()
                        except Exception:
                            pass
                raise e

            return True

        except Exception as e:
            print(f"⚠️  写入评估记录 CSV 失败 (data_id={data_id}): {e}")
            # 不打印完整traceback，避免输出过多
            return False

    finally:
        # 释放文件锁
        if lock_acquired and lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
                # 尝试删除锁文件
                try:
                    lock_file.unlink(missing_ok=True)
                except:
                    pass
            except:
                pass


def append_to_merged_summary(excel_file, summary_stats, config_file=None):
    """
    将当前批次数据写入汇总 CSV：``batchsummary/merged_summary.csv``（按 ``文件名`` upsert 批次行）、
    ``merged_valid_molecules.csv``（与旧 xlsx 第二表相同：仅**当前批次**「正常分子」快照）。
    使用文件锁与临时文件原子替换，避免反复读写整本 xlsx。

    Args:
        excel_file: 刚保存的批次汇总主 CSV 路径（同主名侧车：`…_统计信息.csv` 等）；若为旧版 .xlsx 仍可读多 sheet
        summary_stats: 汇总统计信息
        config_file: 配置文件路径（用于提取配置参数）
    """
    from diffdynamic.batch.configuration import (
        load_config_to_dataframe,
        parse_filename_params,
    )

    if pd is None:
        return

    import fcntl

    try:
        excel_file = Path(excel_file)
        if not excel_file.exists():
            return

        batchsummary_dir = excel_file.parent
        merged_batches_csv = batchsummary_dir / "merged_summary.csv"
        merged_valid_csv = batchsummary_dir / "merged_valid_molecules.csv"
        merge_lock = batchsummary_dir / "merged_summary.csv.lock"

        # 定义列的顺序
        columns_order = [
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
            "有效分子比例 (%)",
            "Vina_Dock 亲和力",
            "Vina_ScoreOnly",
            "Vina_Minimize",
            "QED 评分（均值）",
            "SA 评分（均值）",
        ]

        # 从文件名解析参数
        filename_params = parse_filename_params(excel_file.name)

        # 从批次汇总读取统计数据（CSV 侧车或旧版 xlsx）
        try:
            if excel_file.suffix.lower() == ".csv":
                stats_side = excel_file.with_name(excel_file.stem + "_统计信息.csv")
                if not stats_side.exists():
                    stats_dict = {}
                else:
                    df_stats = pd.read_csv(stats_side, encoding="utf-8-sig")
                    stats_dict = {}
                    for _, row in df_stats.iterrows():
                        key = str(row["统计项目"])
                        value = row["数值"]
                        if isinstance(value, str):
                            try:
                                value = float(value)
                            except ValueError:
                                pass
                        stats_dict[key] = value
            else:
                df_stats = pd.read_excel(
                    excel_file, sheet_name="统计信息", engine="openpyxl"
                )
                stats_dict = {}
                for _, row in df_stats.iterrows():
                    key = str(row["统计项目"])
                    value = row["数值"]
                    if isinstance(value, str):
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    stats_dict[key] = value
        except Exception:
            stats_dict = {}

        # 从配置参数读取（CSV 侧车或旧版 xlsx）
        config_dict = {}
        try:
            if excel_file.suffix.lower() == ".csv":
                cfg_side = excel_file.with_name(excel_file.stem + "_配置参数.csv")
                if cfg_side.exists():
                    df_config = pd.read_csv(cfg_side, encoding="utf-8-sig")
                    for _, row in df_config.iterrows():
                        key = str(row["参数路径"])
                        value = row["参数值"]
                        if pd.isna(value):
                            continue
                        config_dict[key] = value
            else:
                df_config = pd.read_excel(
                    excel_file, sheet_name="配置参数", engine="openpyxl"
                )
                for _, row in df_config.iterrows():
                    key = str(row["参数路径"])
                    value = row["参数值"]
                    if pd.isna(value):
                        continue
                    config_dict[key] = value
        except Exception:
            pass
        if not config_dict and config_file:
            try:
                config_df = load_config_to_dataframe(config_file)
                if config_df is not None and len(config_df) > 0:
                    for _, row in config_df.iterrows():
                        key = str(row["参数路径"])
                        value = row["参数值"]
                        if pd.isna(value):
                            continue
                        config_dict[key] = value
            except Exception:
                pass

        # 构建新行数据
        row_data = {}

        # 从文件名参数获取（优先）
        row_data.update(filename_params)

        # 从配置参数中获取（如果文件名中没有）
        if "下降速率" not in row_data:
            power = config_dict.get("model.grad_fusion_lambda.power", None)
            if power is not None:
                row_data["下降速率"] = float(power)

        if "步数" not in row_data:
            steps = config_dict.get("计算.跳步总次数", None)
            if steps is not None:
                row_data["步数"] = int(steps)

        if "取模步长" not in row_data:
            mod_step = config_dict.get("计算.实际长度", None)
            if mod_step is not None:
                row_data["取模步长"] = float(mod_step)

        # 从配置参数补充缺失的参数
        if "LSstepsize" not in row_data:
            ls_step = config_dict.get("sample.dynamic.large_step.step_size", None)
            if ls_step is not None:
                row_data["LSstepsize"] = float(ls_step)

        if "LSnosie" not in row_data:
            ls_noise = config_dict.get("sample.dynamic.large_step.noise_scale", None)
            if ls_noise is not None:
                row_data["LSnosie"] = float(ls_noise)

        if "LSLambda1" not in row_data:
            ls_lambda_a = config_dict.get(
                "sample.dynamic.large_step.lambda_coeff_a", None
            )
            if ls_lambda_a is not None:
                row_data["LSLambda1"] = float(ls_lambda_a)

        if "LSLambda2" not in row_data:
            ls_lambda_b = config_dict.get(
                "sample.dynamic.large_step.lambda_coeff_b", None
            )
            if ls_lambda_b is not None:
                row_data["LSLambda2"] = float(ls_lambda_b)

        if "RFstepsize" not in row_data:
            rf_step = config_dict.get("sample.dynamic.refine.step_size", None)
            if rf_step is not None:
                row_data["RFstepsize"] = float(rf_step)

        if "RFnosie" not in row_data:
            rf_noise = config_dict.get("sample.dynamic.refine.noise_scale", None)
            if rf_noise is not None:
                row_data["RFnosie"] = float(rf_noise)

        if "RFLambda1" not in row_data:
            rf_lambda_a = config_dict.get("sample.dynamic.refine.lambda_coeff_a", None)
            if rf_lambda_a is not None:
                row_data["RFLambda1"] = float(rf_lambda_a)

        if "RFLambda2" not in row_data:
            rf_lambda_b = config_dict.get("sample.dynamic.refine.lambda_coeff_b", None)
            if rf_lambda_b is not None:
                row_data["RFLambda2"] = float(rf_lambda_b)

        if "时间长度 (TL)" not in row_data:
            time_boundary = config_dict.get("sample.dynamic.time_boundary", None)
            if time_boundary is not None:
                row_data["时间长度 (TL)"] = int(time_boundary)

        if "权重策略" not in row_data:
            mode = config_dict.get("model.grad_fusion_lambda.mode", None)
            if mode is not None:
                row_data["权重策略"] = str(mode)

        if "开始权重" not in row_data:
            start = config_dict.get("model.grad_fusion_lambda.start", None)
            if start is not None:
                row_data["开始权重"] = float(start)

        if "结束权重" not in row_data:
            end = config_dict.get("model.grad_fusion_lambda.end", None)
            if end is not None:
                row_data["结束权重"] = float(end)

        # 从统计信息中提取（安全转换）
        def safe_float(value, default=0.0):
            try:
                if isinstance(value, str):
                    return float(value)
                return float(value) if value is not None else default
            except (ValueError, TypeError):
                return default

        row_data["可重建率 (%)"] = safe_float(stats_dict.get("重建成功百分比(%)", 0))
        row_data["对接成功率 (%)"] = safe_float(stats_dict.get("对接成功百分比(%)", 0))
        row_data["有效分子比例 (%)"] = safe_float(stats_dict.get("有效分子比例(%)", 0))
        row_data["Vina_Dock 亲和力"] = safe_float(
            stats_dict.get("Vina_Dock_平均亲和力", 0)
        )
        row_data["Vina_ScoreOnly"] = safe_float(
            stats_dict.get("Vina_ScoreOnly_平均亲和力", 0)
        )
        row_data["Vina_Minimize"] = safe_float(
            stats_dict.get("Vina_Minimize_平均亲和力", 0)
        )
        row_data["QED 评分（均值）"] = safe_float(stats_dict.get("QED平均评分", 0))
        row_data["SA 评分（均值）"] = safe_float(stats_dict.get("SA平均评分", 0))

        # 确保所有列都存在
        new_row = {}
        for col in columns_order:
            new_row[col] = row_data.get(col, None)
        # 添加文件名列
        new_row["文件名"] = excel_file.name
        all_columns = columns_order + ["文件名"]

        lock_fd = None
        lock_ok = False
        try:
            lock_fd = open(str(merge_lock), "w")
            for _ in range(100):
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    lock_ok = True
                    break
                except (IOError, OSError):
                    time.sleep(0.1)
            if not lock_ok:
                print("⚠️  获取汇总 CSV 锁超时，跳过 merged_summary 更新")
                return

            if merged_batches_csv.exists():
                try:
                    df_existing = pd.read_csv(merged_batches_csv, encoding="utf-8-sig")
                    if "文件名" in df_existing.columns:
                        df_existing = df_existing[
                            df_existing["文件名"] != excel_file.name
                        ]
                    df_new = pd.DataFrame([new_row])
                    for col in all_columns:
                        if col not in df_existing.columns:
                            df_existing[col] = None
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                except Exception as e:
                    print(f"⚠️  读取 merged_summary.csv 失败，将重建: {e}")
                    df_combined = pd.DataFrame([new_row])
            else:
                df_combined = pd.DataFrame([new_row])

            for col in all_columns:
                if col not in df_combined.columns:
                    df_combined[col] = None
            df_combined = df_combined.reindex(columns=all_columns)

            df_all_valid = pd.DataFrame()
            df_valid_batch = None
            try:
                if excel_file.suffix.lower() == ".csv":
                    v_side = excel_file.with_name(excel_file.stem + "_正常分子.csv")
                    if v_side.exists():
                        df_valid_batch = pd.read_csv(v_side, encoding="utf-8-sig")
                else:
                    df_valid_batch = pd.read_excel(
                        excel_file, sheet_name="正常分子", engine="openpyxl"
                    )
            except Exception:
                df_valid_batch = None

            n_valid_appended = 0
            if df_valid_batch is not None and len(df_valid_batch) > 0:
                df_all_valid = df_valid_batch.copy()
                df_all_valid["来源文件"] = excel_file.name
                n_valid_appended = len(df_all_valid)
                if "Vina_Dock_亲和力" in df_all_valid.columns:
                    df_all_valid["Vina_Dock_亲和力_temp"] = df_all_valid[
                        "Vina_Dock_亲和力"
                    ].replace("N/A", np.nan)
                    df_all_valid = df_all_valid.sort_values(
                        "Vina_Dock_亲和力_temp", na_position="last"
                    )
                    df_all_valid = df_all_valid.drop(columns=["Vina_Dock_亲和力_temp"])

            tmp_batches = batchsummary_dir / "merged_summary.csv.tmp"
            tmp_valid = batchsummary_dir / "merged_valid_molecules.csv.tmp"

            df_combined.to_csv(tmp_batches, index=False, encoding="utf-8-sig")
            tmp_batches.replace(merged_batches_csv)

            if len(df_all_valid) > 0:
                df_all_valid.to_csv(tmp_valid, index=False, encoding="utf-8-sig")
                tmp_valid.replace(merged_valid_csv)
            elif merged_valid_csv.exists():
                merged_valid_csv.unlink(missing_ok=True)

            print(
                f"✅ 已更新汇总 CSV: {merged_batches_csv.name}"
                + (
                    f"，已写入本批正常分子 {n_valid_appended} 条 → {merged_valid_csv.name}"
                    if n_valid_appended
                    else ""
                )
            )

        finally:
            if lock_fd is not None:
                try:
                    if lock_ok:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                except Exception:
                    pass
                try:
                    merge_lock.unlink(missing_ok=True)
                except Exception:
                    pass

    except Exception as e:
        print(f"⚠️  追加到汇总 CSV 失败: {e}")
        traceback.print_exc()


def save_molecules_to_excel(
    excel_file,
    molecule_records,
    summary_stats,
    batch_start_time,
    pocket_stats_list=None,
    config_file=None,
    append_merged_summary=True,
):
    """
    将本批次分子与统计写入 CSV：主表为 ``batch_evaluation_summary_….csv``（分子评估数据），
    侧车文件为同主名的 ``…_正常分子.csv``、``…_统计信息.csv``、``…_加权均值统计.csv``（可选）、``…_配置参数.csv``（可选）。

    Args:
        excel_file: 批次主表路径；若以 ``.xlsx`` 传入则自动改为同主名 ``.csv``
        molecule_records: 分子记录列表
        summary_stats: 汇总统计信息
        batch_start_time: 批次启动时间
        pocket_stats_list: 每个口袋的统计信息列表（用于加权均值计算）
        config_file: 配置文件路径（用于保存配置参数）
        append_merged_summary: 是否在保存后更新 ``batchsummary/merged_summary.csv`` 与 ``merged_valid_molecules.csv``（重跑合并脚本时可关）
    """
    from diffdynamic.batch.configuration import load_config_to_dataframe
    from diffdynamic.batch.results import normalize_batch_summary_path

    if pd is None:
        print("⚠️  pandas未安装，无法保存批次 CSV")
        return False

    def _atomic_df_to_csv(df: pd.DataFrame, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.parent / (dest.name + ".tmp")
        df.to_csv(tmp, index=False, encoding="utf-8-sig")
        tmp.replace(dest)

    try:
        main_csv = normalize_batch_summary_path(excel_file)
        stem = main_csv.stem

        valid_csv = main_csv.with_name(f"{stem}_正常分子.csv")
        stats_csv = main_csv.with_name(f"{stem}_统计信息.csv")
        weighted_csv = main_csv.with_name(f"{stem}_加权均值统计.csv")
        config_csv = main_csv.with_name(f"{stem}_配置参数.csv")

        if molecule_records:
            df_molecules = pd.DataFrame(molecule_records)
            if "Vina_Dock_亲和力" in df_molecules.columns:
                df_molecules["Vina_Dock_亲和力_temp"] = df_molecules[
                    "Vina_Dock_亲和力"
                ].replace("N/A", np.nan)
                df_molecules = df_molecules.sort_values(
                    "Vina_Dock_亲和力_temp", na_position="last"
                )
                df_molecules = df_molecules.drop(columns=["Vina_Dock_亲和力_temp"])
            _atomic_df_to_csv(df_molecules, main_csv)

            def is_valid_molecule(row):
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

            df_valid = df_molecules[
                df_molecules.apply(is_valid_molecule, axis=1)
            ].copy()
            _atomic_df_to_csv(df_valid, valid_csv)
        else:
            _atomic_df_to_csv(pd.DataFrame(), main_csv)
            _atomic_df_to_csv(pd.DataFrame(), valid_csv)

        stats_items = []
        stats_values = []
        for key, value in summary_stats.items():
            stats_items.append(key)
            if isinstance(value, float):
                stats_values.append(f"{value:.3f}")
            else:
                stats_values.append(str(value))
        stats_df = pd.DataFrame({"统计项目": stats_items, "数值": stats_values})
        _atomic_df_to_csv(stats_df, stats_csv)

        wrote_weighted = False
        if pocket_stats_list and len(pocket_stats_list) > 0:
            weighted_mean_results = calculate_weighted_means(pocket_stats_list)
            if weighted_mean_results is not None and len(weighted_mean_results) > 0:
                weighted_mean_df = pd.DataFrame(weighted_mean_results)
                _atomic_df_to_csv(weighted_mean_df, weighted_csv)
                wrote_weighted = True
        if not wrote_weighted and weighted_csv.exists():
            weighted_csv.unlink(missing_ok=True)

        wrote_config = False
        if config_file:
            config_df = load_config_to_dataframe(config_file)
            if config_df is not None and len(config_df) > 0:
                _atomic_df_to_csv(config_df, config_csv)
                wrote_config = True
        if not wrote_config and config_csv.exists():
            config_csv.unlink(missing_ok=True)

        if append_merged_summary:
            try:
                append_to_merged_summary(main_csv, summary_stats, config_file)
            except Exception as e:
                print(f"⚠️  追加到汇总 CSV 失败: {e}")

        return True

    except Exception as e:
        print(f"⚠️  保存批次汇总 CSV 失败: {e}")
        traceback.print_exc()
        return False


from .summary import calculate_weighted_means, collect_all_evaluation_results
