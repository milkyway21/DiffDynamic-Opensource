"""Sampling responsibilities extracted from batch_sampleandeval_parallel.py."""

from diffdynamic.batch.state import (
    CONFIG,
    OUTPUT_DIR,
    Path,
    REPO_ROOT,
    SCRIPT,
    os,
    subprocess,
    sys,
    time,
    torch,
    traceback,
    yaml,
)
from diffdynamic.batch import state as _state


def run_single_sample(
    data_id, config_file, gpu_id, max_retries=3, retry_delay=5, protein_root=None
):
    """
    执行单个采样任务（带GPU指定和重试机制）

    Args:
        data_id: 数据ID
        config_file: 配置文件路径
        gpu_id: GPU ID
        max_retries: 最大重试次数（默认3次）
        retry_delay: 重试延迟（秒，默认5秒）

    Returns:
        tuple: (success, pt_file_path, message)
    """
    from diffdynamic.batch.configuration import sample_subprocess_timeout_seconds
    from diffdynamic.batch.results import (
        _parse_results_saved_pt_path,
        find_latest_result_file,
    )

    if config_file is None:
        config_file = CONFIG

    print(
        f"[GPU {gpu_id}] 开始采样 data_id={data_id} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})"
    )

    # 构建采样命令（指定GPU为cuda:0，因为CUDA_VISIBLE_DEVICES已经在process_sampling_task中设置）
    # 在这个进程中，cuda:0 对应物理GPU gpu_id
    cmd = [
        sys.executable,
        str(SCRIPT),
        str(config_file),
        "--data_id",
        str(data_id),
        "--device",
        "cuda:0",  # 使用cuda:0，因为CUDA_VISIBLE_DEVICES已经在进程级别设置了
    ]
    if protein_root:
        cmd.extend(["--protein_root", str(protein_root)])
    result_path = os.environ.get("DD_RESULT_PATH")
    if result_path:
        os.makedirs(result_path, exist_ok=True)
        cmd.extend(["--result_path", str(result_path)])

    # 重试机制
    last_error = None
    for attempt in range(max_retries):
        if attempt > 0:
            print(
                f"[GPU {gpu_id}] 重试采样 data_id={data_id} (尝试 {attempt + 1}/{max_retries})"
            )
            time.sleep(retry_delay)  # 等待一段时间再重试

        try:
            t_before = time.time()
            # 执行采样
            # 使用subprocess.run捕获输出，确保能获取完整错误信息
            # 注意：如果进程被系统杀死（如OOM），可能无法捕获完整输出
            # 环境变量已经在进程级别设置，子进程会自动继承
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,  # 捕获输出以避免混乱
                text=True,
                timeout=sample_subprocess_timeout_seconds(config_file),
                encoding="utf-8",
                errors="replace",  # 处理编码错误，避免因编码问题导致输出截断
                cwd=str(
                    REPO_ROOT
                ),  # 保证 ./pretrained_models、./outputs 等相对路径可解析
                # 不需要显式传递 env：进程已设置 CUDA_VISIBLE_DEVICES，子进程会继承
            )

            # 等待一小段时间，确保文件已保存
            time.sleep(1)

            combined_log = (result.stdout or "") + "\n" + (result.stderr or "")
            pt_parsed = _parse_results_saved_pt_path(combined_log)
            if pt_parsed is not None and pt_parsed.is_file():
                pt_file = pt_parsed.resolve()
                print(f"[GPU {gpu_id}] ✅ 采样成功: {pt_file}")
                if result.stdout:
                    for line in result.stdout.splitlines()[-40:]:
                        if any(
                            k in line
                            for k in (
                                "large_step",
                                "candidate",
                                "pos_list",
                                "prudent",
                                "refined",
                                "warning",
                                "error",
                                "无候选",
                            )
                        ):
                            print(f"[GPU {gpu_id}] LOG: {line}")
                return True, str(pt_file), "采样成功"

            pt_file = find_latest_result_file(data_id, min_mtime=t_before - 3.0)
            if pt_file is None:
                pt_file = find_latest_result_file(data_id)

            if pt_file and pt_file.exists():
                print(f"[GPU {gpu_id}] ✅ 采样成功: {pt_file}")
                # 额外打印采样内部关键日志（large_step 候选数、prudent 迭代等），便于诊断 0 样本问题
                if result.stdout:
                    for line in result.stdout.splitlines()[-40:]:
                        if any(
                            k in line
                            for k in (
                                "large_step",
                                "candidate",
                                "pos_list",
                                "prudent",
                                "refined",
                                "warning",
                                "error",
                                "无候选",
                            )
                        ):
                            print(f"[GPU {gpu_id}] LOG: {line}")
                return True, str(pt_file), "采样成功"
            else:
                print(f"[GPU {gpu_id}] ⚠️  采样完成但未找到结果文件 (data_id={data_id})")
                # 输出stdout和stderr帮助调试
                if result.stdout:
                    print(
                        f"[GPU {gpu_id}] stdout: {result.stdout[-500:]}"
                    )  # 最后500字符
                if result.stderr:
                    print(f"[GPU {gpu_id}] stderr: {result.stderr[-500:]}")
                return False, None, "未找到结果文件"

        except subprocess.TimeoutExpired as e:
            error_msg = f"采样超时 (超过 {int(sample_subprocess_timeout_seconds(config_file))} 秒；可用 SAMPLE_SUBPROCESS_TIMEOUT 调大)"
            if e.stdout:
                error_msg += f"\nstdout (最后500字符): {e.stdout[-500:]}"
            if e.stderr:
                error_msg += f"\nstderr (最后500字符): {e.stderr[-500:]}"
            print(f"[GPU {gpu_id}] ❌ 采样超时 (data_id={data_id})")
            print(f"[GPU {gpu_id}] {error_msg}")
            # 保存完整超时日志
            error_log_file = (
                OUTPUT_DIR / f"sampling_timeout_{data_id}_{int(time.time())}.log"
            )
            try:
                with open(error_log_file, "w", encoding="utf-8") as f:
                    f.write(f"采样超时 (data_id={data_id}, GPU={gpu_id})\n")
                    f.write(f"命令: {' '.join(cmd)}\n")
                    if e.stdout:
                        f.write(f"\n完整stdout:\n{e.stdout}\n")
                    if e.stderr:
                        f.write(f"\n完整stderr:\n{e.stderr}\n")
                print(f"[GPU {gpu_id}] 完整超时日志已保存到: {error_log_file}")
            except Exception as log_err:
                print(f"[GPU {gpu_id}] 保存超时日志失败: {log_err}")
            last_error = error_msg
            # 超时不重试
            return False, None, error_msg
        except subprocess.CalledProcessError as e:
            # 组合stdout和stderr获取完整错误信息
            error_parts = []
            if e.stdout:
                error_parts.append(f"stdout: {e.stdout}")
            if e.stderr:
                error_parts.append(f"stderr: {e.stderr}")
            if not error_parts:
                error_parts.append(str(e))

            error_msg = "\n".join(error_parts)
            last_error = error_msg

            # 检查是否是CUDA初始化错误（可重试的错误）
            is_cuda_error = False
            if e.stderr and (
                "CUBLAS_STATUS_NOT_INITIALIZED" in e.stderr
                or "CUDA error" in e.stderr
                or "cublasCreate" in e.stderr
            ):
                is_cuda_error = True
                print(f"[GPU {gpu_id}] ⚠️  检测到CUDA初始化错误，将重试...")

            # 始终保存完整错误日志到文件（即使很短）
            error_log_file = (
                OUTPUT_DIR / f"sampling_error_{data_id}_{int(time.time())}.log"
            )
            try:
                with open(error_log_file, "w", encoding="utf-8") as f:
                    f.write(
                        f"采样失败 (data_id={data_id}, GPU={gpu_id}, 尝试 {attempt + 1}/{max_retries})\n"
                    )
                    f.write(f"命令: {' '.join(cmd)}\n")
                    f.write(f"返回码: {e.returncode}\n")
                    f.write(
                        f"\n完整stdout ({len(e.stdout) if e.stdout else 0} 字符):\n"
                    )
                    f.write(f"{e.stdout if e.stdout else '(无)'}\n")
                    f.write(
                        f"\n完整stderr ({len(e.stderr) if e.stderr else 0} 字符):\n"
                    )
                    f.write(f"{e.stderr if e.stderr else '(无)'}\n")
                    f.write(f"\n组合错误信息:\n{error_msg}\n")
                if attempt == 0:  # 只在第一次失败时打印
                    print(
                        f"[GPU {gpu_id}] ❌ 采样失败 (data_id={data_id}, 返回码={e.returncode})"
                    )
                    print(f"[GPU {gpu_id}] 完整错误日志已保存到: {error_log_file}")
            except Exception as log_err:
                print(f"[GPU {gpu_id}] ⚠️  保存错误日志失败: {log_err}")

            # 显示错误摘要（前500字符）
            if attempt == 0:  # 只在第一次失败时显示详细错误
                error_summary = error_msg[:500] if len(error_msg) > 500 else error_msg
                if len(error_msg) > 500:
                    error_summary += (
                        f"\n... (共{len(error_msg)}字符，完整信息已保存到日志文件)"
                    )
                print(f"[GPU {gpu_id}] 错误摘要:")
                for line in error_summary.split("\n")[:20]:  # 最多显示20行
                    if line.strip():
                        print(f"[GPU {gpu_id}]   {line}")

                # 诊断常见问题
                if not e.stdout and not e.stderr:
                    print(f"[GPU {gpu_id}] ⚠️  警告: 没有捕获到任何输出，可能原因:")
                    print(f"[GPU {gpu_id}]   - 进程被系统杀死（OOM killer）")
                    print(f"[GPU {gpu_id}]   - GPU内存不足")
                    print(f"[GPU {gpu_id}]   - 进程启动失败")
                elif e.returncode == -9 or e.returncode == 137:
                    print(
                        f"[GPU {gpu_id}] ⚠️  警告: 进程被信号9（SIGKILL）杀死，通常是OOM killer"
                    )
                elif e.returncode == -11 or e.returncode == 139:
                    print(
                        f"[GPU {gpu_id}] ⚠️  警告: 进程段错误（SIGSEGV），可能是内存访问错误"
                    )

            # 如果是CUDA错误且还有重试机会，继续重试
            if is_cuda_error and attempt < max_retries - 1:
                continue
            else:
                # 重试次数用完或不是可重试的错误，返回失败
                return (
                    False,
                    None,
                    f"采样失败 (返回码={e.returncode}): {error_msg[:1000]}",
                )
        except Exception as e:
            error_msg = f"采样出错: {str(e)}"
            last_error = error_msg
            print(f"[GPU {gpu_id}] ❌ {error_msg} (data_id={data_id})")
            traceback.print_exc()
            # 如果是最后一次尝试，返回失败
            if attempt >= max_retries - 1:
                return False, None, error_msg
            # 否则继续重试
            continue

    # 所有重试都失败
    return (
        False,
        None,
        f"采样失败（已重试{max_retries}次）: {last_error[:1000] if last_error else '未知错误'}",
    )


def run_single_sample_custom(
    protein_path,
    config_file,
    gpu_id,
    ligand_path=None,
    use_dataset_for_pocket=False,
    pocket_radius=10.0,
    max_retries=3,
    retry_delay=5,
):
    """
    执行自定义蛋白/配体的单个采样任务（带GPU指定和重试机制）

    Args:
        protein_path: 蛋白口袋 PDB 路径
        config_file: 配置文件路径
        gpu_id: GPU ID
        ligand_path: 参考配体 SDF 路径（可选）
        use_dataset_for_pocket: 当蛋白在数据集内时，使用 index.pkl 中的配体
        pocket_radius: 口袋裁剪半径（Å），提供配体时自动裁剪蛋白为口袋
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）

    Returns:
        tuple: (success, pt_file_path, message)
    """
    from diffdynamic.batch.configuration import sample_subprocess_timeout_seconds
    from diffdynamic.batch.results import (
        _parse_results_saved_pt_path,
        find_latest_result_file,
    )

    if config_file is None:
        config_file = CONFIG

    pocket_id = "custom"
    print(f"[GPU {gpu_id}] 开始采样（自定义文件）: {protein_path}")

    cmd = [
        sys.executable,
        str(SCRIPT),
        str(config_file),
        "--protein_path",
        str(protein_path),
        "--device",
        "cuda:0",
    ]
    if ligand_path:
        cmd.extend(["--ligand_path", str(ligand_path)])
    if use_dataset_for_pocket:
        cmd.append("--use_dataset_for_pocket")
    cmd.extend(["--pocket_radius", str(pocket_radius)])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sample_timeout = sample_subprocess_timeout_seconds(config_file)

    last_error = None
    for attempt in range(max_retries):
        if attempt > 0:
            print(
                f"[GPU {gpu_id}] 重试采样（自定义文件）(尝试 {attempt + 1}/{max_retries})"
            )
            time.sleep(retry_delay)

        try:
            t_before = time.time()
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=sample_timeout,
                encoding="utf-8",
                errors="replace",
                env=env,
                cwd=str(REPO_ROOT),
            )
            time.sleep(1)
            combined_log = (result.stdout or "") + "\n" + (result.stderr or "")
            pt_parsed = _parse_results_saved_pt_path(combined_log)
            if pt_parsed is not None and pt_parsed.is_file():
                pt_file = pt_parsed.resolve()
                print(f"[GPU {gpu_id}] ✅ 采样成功（日志解析）: {pt_file}")
                return True, str(pt_file), "采样成功"
            # 回退：按修改时间取最新，但限制为「本次子进程启动之后」写入的 .pt，避免评估到上一次运行的旧文件
            min_mtime = t_before - 2.0
            pt_file = find_latest_result_file(pocket_id, min_mtime=min_mtime)
            if pt_file is None:
                pt_file = find_latest_result_file(pocket_id)
                if pt_file is not None:
                    print(
                        f"[GPU {gpu_id}] ⚠️  警告: 未在日志中解析到 Results saved to，且 outputs 下无本次运行时间戳之后的 "
                        f"result_{pocket_id}_*.pt；若继续将使用可能过旧的文件: {pt_file}"
                    )

            if pt_file and pt_file.exists():
                print(f"[GPU {gpu_id}] ✅ 采样成功: {pt_file}")
                return True, str(pt_file), "采样成功"
            else:
                print(f"[GPU {gpu_id}] ⚠️  采样完成但未找到结果文件")
                if result.stdout:
                    print(f"[GPU {gpu_id}] stdout: {result.stdout[-500:]}")
                if result.stderr:
                    print(f"[GPU {gpu_id}] stderr: {result.stderr[-500:]}")
                return False, None, "未找到结果文件"

        except subprocess.TimeoutExpired as e:
            error_msg = f"采样超时 (超过 {int(sample_timeout)} 秒；可用环境变量 SAMPLE_SUBPROCESS_TIMEOUT 调大)"
            if e.stdout:
                error_msg += f"\nstdout: {e.stdout[-500:]}"
            if e.stderr:
                error_msg += f"\nstderr: {e.stderr[-500:]}"
            print(f"[GPU {gpu_id}] ❌ 采样超时")
            last_error = error_msg
            return False, None, error_msg
        except subprocess.CalledProcessError as e:
            error_parts = []
            if e.stdout:
                error_parts.append(f"stdout: {e.stdout}")
            if e.stderr:
                error_parts.append(f"stderr: {e.stderr}")
            if not error_parts:
                error_parts.append(str(e))
            last_error = "\n".join(error_parts)
            print(f"[GPU {gpu_id}] ❌ 采样失败: {last_error[-500:]}")
            continue

    return (
        False,
        None,
        f"采样失败（已重试{max_retries}次）: {last_error[:1000] if last_error else '未知错误'}",
    )


def _pt_dict_already_baseline_refined(result_dict):
    """判断 .pt 字典是否已含 TargetDiff baseline refine轨迹（避免重复修复）。"""
    if not isinstance(result_dict, dict):
        return False
    meta = result_dict.get("meta")
    if not isinstance(meta, dict):
        return False
    if meta.get("baseline_refine_time_indices") is not None:
        return True
    return meta.get("posthoc_targetdiff_baseline_refine") is not None


def run_posthoc_baseline_refine(pt_file, config_file, gpu_id, protein_root):
    """
    对单个 .pt 执行与 sampling.yml 中 ``sample.targetdiff_baseline_refine`` 一致的扩散修复。
    调用方须已设置 ``CUDA_VISIBLE_DEVICES``（与采样子进程一致时用物理 GPU id）。

    Returns:
        str | None: 成功时为新的 .pt 路径（或已精炼则原路径）；失败时返回原始 ``pt_file`` 字符串。
    """
    pt_path = Path(pt_file)
    if not pt_path.is_file():
        print(f"[GPU {gpu_id}] 警告: baseline refine 跳过，文件不存在 {pt_path}")
        return str(pt_file) if pt_file else None
    if torch is None:
        print(f"[GPU {gpu_id}] 警告: baseline refine 跳过，未安装 torch")
        return str(pt_path)
    try:
        try:
            blob = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(str(pt_path), map_location="cpu")
    except Exception as e:
        print(f"[GPU {gpu_id}] 警告: baseline refine 读取 .pt 失败，使用原文件: {e}")
        return str(pt_path)
    if _pt_dict_already_baseline_refined(blob):
        print(f"[GPU {gpu_id}] 已含 baseline_refine，跳过: {pt_path.name}")
        return str(pt_path)

    cfg_path = Path(config_file)
    if yaml is not None and cfg_path.is_file():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f)
            ref = (y or {}).get("sample", {}).get("targetdiff_baseline_refine", {})
            if not ref.get("enable", False):
                print(
                    f"[GPU {gpu_id}] 警告: sampling.yml 中 targetdiff_baseline_refine.enable 为 false，"
                    f"跳过 post-hoc 修复（见 sample.targetdiff_baseline_refine）"
                )
                return str(pt_path)
        except Exception as e:
            print(f"[GPU {gpu_id}] 警告: 读取 YAML 检查 baseline_refine 失败: {e}")

    try:
        import diffdynamic.io.config as misc
        from refine_saved_pt_baseline_then_eval import load_stack, refine_one_pt
    except Exception as e:
        print(f"[GPU {gpu_id}] 警告: baseline refine 导入失败: {e}")
        return str(pt_path)

    logger = misc.get_logger("batch_baseline_refine")
    pr = str(protein_root).strip() if protein_root else None
    try:
        config, model, _ckpt, dataset_root = load_stack(
            cfg_path.resolve(), "cuda:0", logger
        )
        outp = refine_one_pt(
            pt_path, pt_path.parent, model, config, dataset_root, pr, "cuda:0", logger
        )
        if outp is not None:
            print(f"[GPU {gpu_id}] ✅ baseline refine: {pt_path.name} -> {outp.name}")
            return str(outp)
    except SystemExit as e:
        print(f"[GPU {gpu_id}] 警告: baseline refine 异常退出: {e}")
    except Exception as e:
        traceback.print_exc()
        print(f"[GPU {gpu_id}] 警告: baseline refine 失败，使用原 .pt: {e}")
    return str(pt_path)


def process_baseline_refine_standalone_task(args_tuple):
    """
    独立进程内对单个 .pt 做 baseline refine（设置 CUDA_VISIBLE_DEVICES 后调用 ``run_posthoc_baseline_refine``）。
    args_tuple: (pt_path_str, config_path_str, protein_root_str, gpu_id)
    """
    pt_path_str, config_path_str, protein_root_str, gpu_id = args_tuple
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if torch is not None:
        try:
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
        except Exception:
            pass
    return run_posthoc_baseline_refine(
        pt_path_str, config_path_str, gpu_id, protein_root_str
    )


def process_sampling_task(args_tuple):
    """
    处理单个采样任务（仅采样，不评估）
    这个函数会在独立的进程中运行，限制为4个进程（每个GPU一个）
    记录每个口袋的生成时间和使用的GPU

    Args:
        args_tuple: (data_id, gpu_id, config_file, skip_existing, protein_root[, molecular_repair])
            molecular_repair: none | mmff | targetdiff_baseline_refine（仅后者在本阶段执行，mmff 在评估阶段）

    Returns:
        tuple: (data_id, success, pt_file, message)
    """
    from diffdynamic.batch.reports import record_pocket_generation_time
    from diffdynamic.batch.resources import (
        check_cuda_available_in_subprocess,
        check_gpu_memory_available,
    )
    from diffdynamic.batch.results import find_latest_result_file

    # 记录开始时间
    task_start_time = time.time()

    if len(args_tuple) >= 6:
        data_id, gpu_id, config_file, skip_existing, protein_root, molecular_repair = (
            args_tuple[:6]
        )
        protein_root = str(protein_root).strip() or None
        molecular_repair = (molecular_repair or "none").strip().lower()
    elif len(args_tuple) >= 5:
        data_id, gpu_id, config_file, skip_existing, protein_root = args_tuple[:5]
        protein_root = str(protein_root).strip() or None
        molecular_repair = "none"
    else:
        data_id, gpu_id, config_file, skip_existing = args_tuple
        protein_root = None
        molecular_repair = "none"

    # 在进程开始时立即设置CUDA_VISIBLE_DEVICES
    # 这必须在任何CUDA初始化之前完成
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 关键修复：如果torch已经在主进程中导入，需要清除CUDA缓存并重新初始化
    # 这是因为子进程可能继承了主进程的CUDA上下文状态
    if torch is not None:
        try:
            # 清除可能存在的CUDA缓存
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
            # 尝试重置CUDA状态（如果可能）
            if hasattr(torch.cuda, "_lazy_init"):
                # 清除lazy init状态，强制重新初始化
                torch.cuda._lazy_init()
        except Exception as e:
            # 如果清除失败，继续执行（可能CUDA还未初始化）
            pass

    try:
        # 检查GPU内存是否足够（在设置CUDA_VISIBLE_DEVICES之前检查物理GPU）
        has_memory, free_mb, total_mb, mem_msg = check_gpu_memory_available(
            gpu_id, min_free_memory_mb=5000
        )
        if has_memory is False:
            error_msg = (
                f"GPU {gpu_id} 内存不足 (data_id={data_id})\n"
                f"   {mem_msg}\n"
                f"   需要至少 5000MB 可用内存，但只有 {free_mb}MB\n"
                f"   建议: 等待其他任务完成或清理GPU内存"
            )
            print(f"[GPU {gpu_id}] ⚠️  {error_msg}")
            # 记录失败状态
            generation_time = time.time() - task_start_time
            record_pocket_generation_time(
                data_id,
                gpu_id,
                generation_time,
                status="失败-GPU内存不足",
                pt_file=None,
            )
            return (data_id, False, None, error_msg)
        elif has_memory is True:
            print(f"[GPU {gpu_id}] ✅ {mem_msg}")

        # 验证 CUDA 是否可用（在设置 CUDA_VISIBLE_DEVICES 后）
        cuda_available, num_gpus = check_cuda_available_in_subprocess()
        if not cuda_available:
            error_msg = (
                f"CUDA不可用 (GPU {gpu_id}, CUDA_VISIBLE_DEVICES={gpu_id})\n"
                f"   可能原因:\n"
                f"   1. Docker容器未正确配置GPU支持\n"
                f"   2. NVIDIA驱动未安装或版本不兼容\n"
                f"   3. PyTorch未正确编译CUDA支持\n"
                f"   4. CUDA运行时库未正确安装\n"
                f"   请检查: nvidia-smi 是否可用，以及容器是否正确配置了 --gpus all"
            )
            print(f"[GPU {gpu_id}] ❌ {error_msg}")
            # 记录失败状态
            generation_time = time.time() - task_start_time
            record_pocket_generation_time(
                data_id, gpu_id, generation_time, status="失败-CUDA不可用", pt_file=None
            )
            return (data_id, False, None, error_msg)

        # 检查是否已存在
        if skip_existing:
            pt_file = find_latest_result_file(data_id)
            if pt_file and pt_file.exists():
                print(
                    f"[GPU {gpu_id}] ⏭️  跳过已存在的文件: {pt_file} (data_id={data_id})"
                )
                # 跳过已存在的文件不记录时间（因为没有实际生成）
                pt_file = str(pt_file)
                if molecular_repair == "targetdiff_baseline_refine":
                    refined = run_posthoc_baseline_refine(
                        pt_file, config_file, gpu_id, protein_root
                    )
                    if refined:
                        pt_file = refined
                return (data_id, True, pt_file, "文件已存在")

        # 执行采样（带重试机制）
        sample_success, pt_file, sample_msg = run_single_sample(
            data_id,
            config_file,
            gpu_id,
            max_retries=3,
            retry_delay=5,
            protein_root=protein_root,
        )
        if (
            sample_success
            and pt_file
            and molecular_repair == "targetdiff_baseline_refine"
        ):
            refined = run_posthoc_baseline_refine(
                pt_file, config_file, gpu_id, protein_root
            )
            if refined:
                pt_file = refined

        # 计算生成时间
        generation_time = time.time() - task_start_time

        if not sample_success or pt_file is None:
            # 记录失败状态
            record_pocket_generation_time(
                data_id, gpu_id, generation_time, status="失败-采样失败", pt_file=None
            )
            return (data_id, False, None, sample_msg)

        # 记录成功状态
        record_pocket_generation_time(
            data_id, gpu_id, generation_time, status="成功", pt_file=pt_file
        )

        return (data_id, True, pt_file, sample_msg)

    except Exception as e:
        # 计算生成时间（即使异常也记录）
        generation_time = time.time() - task_start_time
        error_msg = f"采样任务异常 (data_id={data_id}): {str(e)}"
        print(f"[GPU {gpu_id}] ❌ {error_msg}")
        traceback.print_exc()
        # 记录异常状态
        record_pocket_generation_time(
            data_id, gpu_id, generation_time, status="失败-异常", pt_file=None
        )
        return (data_id, False, None, error_msg)
