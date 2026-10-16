"""Evaluation responsibilities extracted from batch_sampleandeval_parallel.py."""

from diffdynamic.batch.state import (
    EVAL_SCRIPT,
    Path,
    datetime,
    os,
    subprocess,
    sys,
    time,
    timedelta,
    timezone,
    traceback,
)
from diffdynamic.batch import state as _state


def run_single_evaluation(
    pt_file,
    protein_root,
    data_id,
    atom_mode="add_aromatic",
    exhaustiveness=8,
    cores_per_task=1,
    save_intermediate_interval=0,
    eval_show_output=False,
    eval_save_subprocess_log=False,
    force_mmff_minimize=False,
    mmff_max_iters=None,
    vina_modes=None,
    medchem_optimize=False,
    medchem_optimize_config=None,
    keep_preopt_sdf=None,
):
    """
    执行单个评估任务

    Args:
        pt_file: .pt文件路径
        protein_root: 蛋白质数据根目录
        data_id: 数据ID
        atom_mode: 原子模式
        exhaustiveness: Vina对接强度
        cores_per_task: 每个任务分配的CPU核心数，用于限制子进程的OMP_NUM_THREADS等（默认1表示不限制）
        save_intermediate_interval: 每处理多少个分子保存一次中间结果；0=禁用（批量模式默认0以减少磁盘占用）
        eval_show_output: True 时子进程 stdout/stderr 直接打到终端（多任务并行会交错）
        eval_save_subprocess_log: True 时将子进程完整输出写入 eval_output_dir/evaluate_subprocess.log
        force_mmff_minimize: True 时向 evaluate 传入 --force-mmff-minimize（对接前 MMFF）
        mmff_max_iters: 非 None 时传入 --mmff-max-iters
        vina_modes: 传给 evaluate 的 ``--vina-modes``；None 表示不传参（evaluate 默认 dock+score_only+minimize）
        medchem_optimize: True 时传入 --medchem-optimize，启用激进 medchem 结构优化（会改分子式，须作为独立 ablation 臂）
        medchem_optimize_config: 非 None 时传入 --medchem-optimize-config
        keep_preopt_sdf: 非 None 时传入 --keep-preopt-sdf（none / structure_only / evaluated）

    Returns:
        tuple: (success, message, eval_output_dir)
    """
    from diffdynamic.batch.results import (
        _resolve_final_eval_dir_from_evaluate_subprocess,
    )

    print(f"[评估] 开始评估: {Path(pt_file).name} (data_id={data_id})")
    t0 = time.time()

    pt_path = Path(pt_file)
    # 与 evaluate_pt_with_correct_reconstruct 一致：--output_dir 为 .pt 所在目录（父目录），
    # 由 evaluate 内部 generate_eval_dir_name 只生成单层 eval_时间戳_data_id_配置... 目录，避免 outputs 下出现空壳 eval_*。
    outputs_dir = pt_path.parent.resolve()

    # 使用本地时区（CST，UTC+8）生成时间戳（仅用于子进程日志文件名）
    cst = timezone(timedelta(hours=8))
    eval_timestamp = datetime.now(cst).strftime("%Y%m%d_%H%M%S")

    # 构建评估命令
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        str(pt_file),
        "--protein_root",
        str(protein_root),
        "--output_dir",
        str(outputs_dir),
        "--atom_mode",
        atom_mode,
        "--exhaustiveness",
        str(exhaustiveness),
        "--save_intermediate_interval",
        str(save_intermediate_interval),
    ]
    if force_mmff_minimize:
        cmd.append("--force-mmff-minimize")
    if mmff_max_iters is not None:
        cmd.extend(["--mmff-max-iters", str(mmff_max_iters)])
    if vina_modes is not None and str(vina_modes).strip():
        cmd.extend(["--vina-modes", str(vina_modes).strip()])
    if medchem_optimize:
        cmd.append("--medchem-optimize")
        if medchem_optimize_config:
            cmd.extend(["--medchem-optimize-config", str(medchem_optimize_config)])
        if keep_preopt_sdf:
            cmd.extend(["--keep-preopt-sdf", str(keep_preopt_sdf)])
    if eval_show_output:
        print(f"[评估] [data_id={data_id}] 命令: {' '.join(cmd)}")

    # 限制每个子进程的CPU核心数（Vina/NumPy等使用OpenMP/MKL）
    env = os.environ.copy()
    if cores_per_task >= 1:
        cores_str = str(cores_per_task)
        env["OMP_NUM_THREADS"] = cores_str
        env["MKL_NUM_THREADS"] = cores_str
        env["OPENBLAS_NUM_THREADS"] = cores_str
        env["NUMEXPR_NUM_THREADS"] = cores_str
        env["VECLIB_MAXIMUM_THREADS"] = cores_str
    # 单路径单 .pt 评估：与 evaluate_pt_with_correct_reconstruct 中单分子默认上限一致（可被 EVAL_SINGLE_MOL_TIMEOUT 覆盖）
    if "EVAL_SINGLE_MOL_TIMEOUT" not in env:
        env["EVAL_SINGLE_MOL_TIMEOUT"] = "10800"

    run_kw = dict(check=True, timeout=21600, env=env)
    # 整段 evaluate 子进程上限（6 小时）；单次 Vina 时限在 evaluate 脚本内单独控制

    try:
        captured_stdout = ""
        captured_stderr = ""
        if eval_show_output:
            # 直接继承终端，便于看到 tqdm / 逐分子进度（多进程并行时输出会交错）
            subprocess.run(cmd, **run_kw)
        elif eval_save_subprocess_log:
            log_path = (
                outputs_dir / f"evaluate_subprocess_{data_id}_{eval_timestamp}.log"
            )
            with open(log_path, "w", encoding="utf-8") as logf:
                cp = subprocess.run(
                    cmd, stdout=logf, stderr=subprocess.STDOUT, text=True, **run_kw
                )
            print(f"[评估] [data_id={data_id}] 子进程日志: {log_path}")
            try:
                captured_stdout = log_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                captured_stdout = ""
        else:
            # 默认：捕获输出，避免多路并行时终端与 [评估] 进度行混杂
            cp = subprocess.run(cmd, capture_output=True, text=True, **run_kw)
            captured_stdout = cp.stdout or ""
            captured_stderr = cp.stderr or ""

        final_eval_dir = _resolve_final_eval_dir_from_evaluate_subprocess(
            captured_stdout + "\n" + captured_stderr, outputs_dir, data_id
        )
        if not final_eval_dir:
            print(f"[评估] ⚠️ 无法解析评估输出目录，请检查 {outputs_dir} 下 eval_*")

        elapsed = time.time() - t0
        print(f"[评估] ✅ 评估成功 (data_id={data_id})  耗时 {elapsed:.1f}s")
        return True, "评估成功", final_eval_dir

    except subprocess.TimeoutExpired as e:
        error_msg = f"评估超时 (超过6小时)"
        if e.stdout:
            error_msg += f"\nstdout: {e.stdout[-500:]}"
        if e.stderr:
            error_msg += f"\nstderr: {e.stderr[-500:]}"
        print(f"[评估] ❌ 评估超时 (data_id={data_id})")
        print(f"[评估] {error_msg}")
        return False, error_msg, None
    except subprocess.CalledProcessError as e:
        # 组合 stdout/stderr（capture_output=True 时在此获取）
        error_parts = []
        if e.stdout:
            error_parts.append(f"stdout: {e.stdout}")
        if e.stderr:
            error_parts.append(f"stderr: {e.stderr}")
        if not error_parts:
            error_parts.append(str(e))

        error_msg = "\n".join(error_parts)
        # 显示完整错误信息（不截断）
        print(f"[评估] ❌ 评估失败 (data_id={data_id}):")
        print(f"[评估] {error_msg}")
        # 返回时保留完整错误信息，但限制长度避免过长
        return False, f"评估失败: {error_msg[:1000]}", None
    except Exception as e:
        error_msg = f"评估出错: {str(e)}"
        print(f"[评估] ❌ {error_msg} (data_id={data_id})")
        traceback.print_exc()
        return False, error_msg, None


def process_evaluation_task(args_tuple):
    """
    处理单个评估任务（仅评估，不采样）
    这个函数会在独立的进程中运行，可以使用多个进程并行

    Args:
        args_tuple: (data_id, pt_file, protein_root, atom_mode, exhaustiveness, excel_file,
            batch_start_time, excel_lock, cores_per_task, save_intermediate_interval,
            eval_show_output, eval_save_subprocess_log[, force_mmff_minimize, mmff_max_iters[, vina_modes[, medchem_opts]]]）；
            兼容10 / 12 / 14 / 15 / 16 元组（缺省则 force_mmff=False、mmff_max_iters=None、vina_modes=None、medchem_opts={}）。
            medchem_opts 是个 dict，新开关都往里加，免得每加一个参数就多一条元组长度分支。

    Returns:
        tuple: (data_id, success, message, log_file, pt_file, eval_output_dir)
    """
    from diffdynamic.batch.reports import append_to_excel
    from diffdynamic.batch.results import read_evaluation_results

    medchem_opts = (
        args_tuple[15]
        if len(args_tuple) >= 16 and isinstance(args_tuple[15], dict)
        else {}
    )
    if len(args_tuple) >= 15:
        (
            data_id,
            pt_file,
            protein_root,
            atom_mode,
            exhaustiveness,
            excel_file,
            batch_start_time,
            excel_lock,
            cores_per_task,
            save_intermediate_interval,
            eval_show_output,
            eval_save_subprocess_log,
            force_mmff_minimize,
            mmff_max_iters,
            vina_modes,
        ) = args_tuple[:15]
    elif len(args_tuple) >= 14:
        (
            data_id,
            pt_file,
            protein_root,
            atom_mode,
            exhaustiveness,
            excel_file,
            batch_start_time,
            excel_lock,
            cores_per_task,
            save_intermediate_interval,
            eval_show_output,
            eval_save_subprocess_log,
            force_mmff_minimize,
            mmff_max_iters,
        ) = args_tuple[:14]
        vina_modes = None
    elif len(args_tuple) >= 12:
        (
            data_id,
            pt_file,
            protein_root,
            atom_mode,
            exhaustiveness,
            excel_file,
            batch_start_time,
            excel_lock,
            cores_per_task,
            save_intermediate_interval,
            eval_show_output,
            eval_save_subprocess_log,
        ) = args_tuple[:12]
        force_mmff_minimize = False
        mmff_max_iters = None
        vina_modes = None
    elif len(args_tuple) == 10:
        (
            data_id,
            pt_file,
            protein_root,
            atom_mode,
            exhaustiveness,
            excel_file,
            batch_start_time,
            excel_lock,
            cores_per_task,
            save_intermediate_interval,
        ) = args_tuple
        eval_show_output = False
        eval_save_subprocess_log = False
        force_mmff_minimize = False
        mmff_max_iters = None
        vina_modes = None
    else:
        raise ValueError(
            f"process_evaluation_task: 元组长度应为 10、12、14、15 或 16，实际 {len(args_tuple)}"
        )

    # 设置全局锁（与 excel_file 同 stem 的评估记录 CSV 写入）
    pass  # process state is held in state.py
    _state.excel_write_lock = excel_lock

    task_start_time = time.time()

    try:
        if not pt_file:
            return (data_id, False, "无PT文件", None, None, None)

        # 执行评估
        eval_success, eval_msg, eval_output_dir = run_single_evaluation(
            pt_file,
            protein_root,
            data_id,
            atom_mode,
            exhaustiveness,
            cores_per_task=cores_per_task,
            save_intermediate_interval=save_intermediate_interval,
            eval_show_output=eval_show_output,
            eval_save_subprocess_log=eval_save_subprocess_log,
            force_mmff_minimize=force_mmff_minimize,
            mmff_max_iters=mmff_max_iters,
            vina_modes=vina_modes,
            medchem_optimize=medchem_opts.get("medchem_optimize", False),
            medchem_optimize_config=medchem_opts.get("medchem_optimize_config"),
            keep_preopt_sdf=medchem_opts.get("keep_preopt_sdf"),
        )

        task_time = time.time() - task_start_time
        # 使用本地时区（CST，UTC+8）生成时间戳
        cst = timezone(timedelta(hours=8))
        timestamp_str = datetime.now(cst).strftime("%Y-%m-%d %H:%M:%S")

        if eval_success:
            # 等待评估结果文件生成
            time.sleep(2)
            eval_success_read, vina_mean, vina_median, num_scores, eval_message, _ = (
                read_evaluation_results(pt_file, data_id, wait_timeout=60)
            )

            if eval_success_read:
                if excel_file:
                    append_to_excel(
                        excel_file,
                        timestamp_str,
                        task_time,
                        data_id,
                        pt_file,
                        vina_mean,
                        vina_median,
                        num_scores,
                        "成功",
                        eval_message,
                    )
            else:
                if excel_file:
                    append_to_excel(
                        excel_file,
                        timestamp_str,
                        task_time,
                        data_id,
                        pt_file,
                        None,
                        None,
                        0,
                        "部分成功",
                        f"评估完成但读取结果失败: {eval_message}",
                    )

            return (
                data_id,
                True,
                eval_msg,
                None,
                pt_file,
                str(eval_output_dir) if eval_output_dir else None,
            )
        else:
            if excel_file:
                append_to_excel(
                    excel_file,
                    timestamp_str,
                    task_time,
                    data_id,
                    pt_file,
                    None,
                    None,
                    0,
                    "失败",
                    eval_msg,
                )

            return (data_id, False, eval_msg, None, pt_file, None)

    except Exception as e:
        error_msg = f"评估任务异常 (data_id={data_id}): {str(e)}"
        print(f"[评估] ❌ {error_msg}")
        traceback.print_exc()
        return (data_id, False, error_msg, None, pt_file, None)


def _dispatch_eval_task(task_with_type):
    """分发任务到对应的处理函数（用于 Pool.map 的统一入口）。"""
    from diffdynamic.batch.benchmark import (
        process_decompdiff_pt_evaluation_task,
        process_sdf_evaluation_task,
    )

    task_type, task_args = task_with_type
    if task_type == "targetdiff":
        return process_evaluation_task(task_args)
    elif task_type == "decompdiff":
        return process_decompdiff_pt_evaluation_task(task_args)
    elif task_type == "sdf":
        return process_sdf_evaluation_task(task_args)
    else:
        raise ValueError(f"未知任务类型: {task_type}")
