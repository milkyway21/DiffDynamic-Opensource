"""Resources responsibilities extracted from batch_sampleandeval_parallel.py."""

from diffdynamic.batch.state import (
    os,
    signal,
    subprocess,
    sys,
    time,
    torch,
)
from diffdynamic.batch import state as _state


def parse_gpu_ids(gpu_str):
    """
    解析GPU ID字符串，支持多种格式：
    - "0,1,2,3" -> [0, 1, 2, 3]
    - "0-3" -> [0, 1, 2, 3]
    - "0,2-4,6" -> [0, 2, 3, 4, 6]
    - "all" -> 自动检测所有可用GPU

    Args:
        gpu_str: GPU ID字符串

    Returns:
        list: GPU ID列表
    """
    if gpu_str.lower() == "all":
        # 自动检测所有可用GPU
        if torch is not None and torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            return list(range(num_gpus))
        else:
            print("⚠️  警告: CUDA不可用，无法自动检测GPU")
            return []

    gpu_ids = []
    parts = gpu_str.split(",")

    for part in parts:
        part = part.strip()
        if "-" in part:
            # 范围格式，如 "0-3"
            start, end = part.split("-")
            try:
                start_id = int(start.strip())
                end_id = int(end.strip())
                gpu_ids.extend(range(start_id, end_id + 1))
            except ValueError:
                raise ValueError(f"无效的GPU范围格式: {part}")
        else:
            # 单个ID
            try:
                gpu_ids.append(int(part))
            except ValueError:
                raise ValueError(f"无效的GPU ID: {part}")

    # 去重并排序
    gpu_ids = sorted(list(set(gpu_ids)))
    return gpu_ids


def get_available_gpus():
    """
    获取所有可用的GPU ID列表

    Returns:
        list: 可用GPU ID列表
    """
    if torch is not None and torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def coerce_gpu_ids_to_cuda_visible(gpu_ids):
    """将 `--gpus` 解析得到的 ID 约束到当前 PyTorch/CUDA **可见视图**内的合法索引 ``0 … n-1``。

    典型场景（Docker）：宿主或文档里习惯性称「GPU 5」，但容器内仅映射了 n 张卡，合法索引实为
    ``0…n-1``。误写 ``--gpus \"5\"`` 且恰好 ``torch.cuda.device_count()==5`` 时，旧代码会设置
    ``CUDA_VISIBLE_DEVICES=5``，子进程见不到任何 GPU，导致 ``cuda:0`` 初始化失败。

    规则：
    - ``g < n``：保留；
    - ``g == n``：视为「恰好 n 张卡却写成 n」的常见笔误，压到 ``n-1`` 并打印醒目的说明；
    - ``g > n``：无法自动推断→报错退出；
    - 多条目时对 ``== n`` 的映射会与已有 ``n-1`` 自动去重。

    若主进程检测不到 CUDA，则原样返回（交由子进程与环境变量兜底，保持旧兼容）。
    """
    if gpu_ids is None:
        return gpu_ids
    if torch is None or not torch.cuda.is_available():
        return list(gpu_ids)
    n = int(torch.cuda.device_count())
    if n <= 0:
        return list(gpu_ids)
    uniq_in = sorted({int(g) for g in gpu_ids})
    bad_neg = [g for g in uniq_in if g < 0]
    if bad_neg:
        print(f"❌ 错误: 非法 GPU ID（负值）: {bad_neg}")
        sys.exit(1)

    out_seen = []
    for g in uniq_in:
        if g < n:
            if g not in out_seen:
                out_seen.append(g)
        elif g == n:
            tgt = n - 1
            if tgt not in out_seen:
                print(
                    f"⚠️  GPU ID {g} 在当前 CUDA 视图非法（仅存 0～{n - 1}）；"
                    f"按「恰好 n 张卡却写成 n」处理，改用 {tgt}。"
                    f"若需使用宿主上更大 PCIe 序号之卡，请调整 Docker `--gpus` / NVIDIA_VISIBLE_DEVICES。"
                )
                out_seen.append(tgt)
        else:
            print(
                f"❌ 错误: GPU ID {g} 超出可见范围（当前仅 0～{n - 1}）。"
                " 请修改 --gpus 或为容器挂载该设备。"
            )
            sys.exit(1)
    return out_seen


def _preflight_import_torch_scatter():
    """evaluate 依赖 datasets → torch_scatter；缺失时返回 False。"""
    try:
        import torch_scatter  # noqa: F401

        return True
    except ImportError:
        return False


def check_gpu_memory_available(gpu_id, min_free_memory_mb=5000):
    """
    检查GPU是否有足够的可用内存

    Args:
        gpu_id: GPU ID
        min_free_memory_mb: 最小可用内存（MB），默认5GB

    Returns:
        tuple: (has_enough_memory, free_memory_mb, total_memory_mb, message)
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
                f"--id={gpu_id}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            if len(parts) >= 2:
                free_memory_mb = int(parts[0].strip())
                total_memory_mb = int(parts[1].strip())
                has_enough = free_memory_mb >= min_free_memory_mb
                message = f"GPU {gpu_id}: 可用内存 {free_memory_mb}MB / 总计 {total_memory_mb}MB"
                return has_enough, free_memory_mb, total_memory_mb, message
    except Exception as e:
        pass

    return None, None, None, f"无法检查GPU {gpu_id}的内存状态"


def check_cuda_available_in_subprocess():
    """
    在子进程中检查 CUDA 是否可用
    在设置 CUDA_VISIBLE_DEVICES 后调用此函数

    Returns:
        tuple: (is_available, num_gpus)
    """
    # 方法1：使用 nvidia-smi 检测 GPU（不依赖 PyTorch，更可靠）
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            # nvidia-smi 可用，说明 GPU 驱动正常
            num_gpus = len(result.stdout.strip().split("\n"))
            if num_gpus > 0:
                return True, num_gpus
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    # 方法2：检查 PyTorch CUDA（在设置 CUDA_VISIBLE_DEVICES 后应该能检测到）
    try:
        # 如果 torch 已经在主进程中导入，子进程会继承
        # 但 CUDA 的检测应该会使用新的 CUDA_VISIBLE_DEVICES 环境变量
        if torch is not None:
            # 尝试重新初始化 CUDA（如果可能）
            try:
                # 清除 CUDA 缓存（如果已初始化）
                if hasattr(torch.cuda, "_lazy_init"):
                    torch.cuda._lazy_init()
            except Exception:
                pass

            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                return True, num_gpus
    except Exception as e:
        # 如果检测失败，记录但不抛出异常
        pass

    return False, 0


def cleanup_zombie_processes(interactive=True):
    """
    清理残留的采样和评估相关进程

    Args:
        interactive: 是否交互式询问用户（默认True）

    Returns:
        int: 清理的进程数量
    """
    # 定义要清理的进程模式
    PATTERNS = [
        "batch_sampleandeval_parallel.py",
        "sample_diffusion.py",
        "evaluate_pt_with_correct_reconstruct.py",
    ]

    # 定义要保护的进程模式（薛定谔相关）
    PROTECTED_PATTERNS = ["schrodinger", "gdesmond", "glide", "jmonitor"]

    # 收集所有相关进程PID
    all_pids = []
    for pattern in PATTERNS:
        try:
            pids = subprocess.run(
                ["pgrep", "-f", pattern], capture_output=True, text=True, timeout=5
            )
            if pids.returncode == 0 and pids.stdout.strip():
                for pid_str in pids.stdout.strip().split("\n"):
                    try:
                        pid = int(pid_str.strip())
                        if pid > 0:
                            all_pids.append(pid)
                    except ValueError:
                        continue
        except Exception:
            continue

    # 去重
    all_pids = sorted(list(set(all_pids)))

    if not all_pids:
        return 0

    # 过滤掉受保护的进程和当前进程
    filtered_pids = []
    current_pid = os.getpid()

    for pid in all_pids:
        if pid == current_pid:
            continue

        try:
            # 获取进程命令
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "cmd="],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip():
                cmd = result.stdout.strip()

                # 检查是否受保护
                is_protected = False
                for protected in PROTECTED_PATTERNS:
                    if protected.lower() in cmd.lower():
                        is_protected = True
                        break

                # 检查是否是我们要清理的进程
                if not is_protected:
                    for pattern in PATTERNS:
                        if pattern in cmd:
                            filtered_pids.append((pid, cmd))
                            break
        except Exception:
            continue

    if not filtered_pids:
        return 0

    # 显示找到的进程
    print(f"\n{'=' * 60}")
    print(f"检测到 {len(filtered_pids)} 个残留进程")
    print(f"{'=' * 60}")
    for pid, cmd in filtered_pids[:10]:  # 只显示前10个
        print(f"  PID {pid}: {cmd[:80]}")
    if len(filtered_pids) > 10:
        print(f"  ... 还有 {len(filtered_pids) - 10} 个进程")
    print(f"{'=' * 60}")

    # 询问用户是否清理（仅在交互模式下）
    if interactive:
        try:
            response = input(f"\n是否清理这些残留进程? (y/n，默认y): ").strip().lower()
            if response and response not in ["y", "yes"]:
                print("已跳过清理")
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\n已跳过清理")
            return 0
    else:
        # 非交互模式：直接清理，不询问
        print(f"\n自动清理 {len(filtered_pids)} 个残留进程...")

    # 清理进程
    killed = 0
    failed = 0
    need_root = []

    print(f"\n开始清理...")
    for pid, cmd in filtered_pids:
        try:
            # 先尝试优雅终止
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.2)

            # 检查进程是否还在运行
            try:
                os.kill(pid, 0)  # 检查进程是否存在
                # 如果还在运行，强制终止
                os.kill(pid, signal.SIGKILL)
                time.sleep(0.2)
            except ProcessLookupError:
                # 进程已不存在
                pass

            # 再次检查
            try:
                os.kill(pid, 0)
                # 进程还在运行，可能需要root权限
                need_root.append(pid)
                failed += 1
            except ProcessLookupError:
                killed += 1
        except PermissionError:
            # 需要root权限
            need_root.append(pid)
            failed += 1
        except Exception as e:
            # 其他错误，尝试使用kill命令
            need_root.append(pid)
            failed += 1

    # 如果有需要root权限的进程，尝试使用kill命令（可能需要sudo）
    if need_root:
        print(f"\n检测到 {len(need_root)} 个进程需要root权限，尝试使用kill命令清理...")
        for pid in need_root:
            try:
                # 尝试使用kill命令（如果当前用户有权限）
                result = subprocess.run(
                    ["kill", "-9", str(pid)], capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    time.sleep(0.2)
                    # 检查进程是否已终止
                    try:
                        os.kill(pid, 0)
                        # 进程还在，标记为失败
                    except ProcessLookupError:
                        # 进程已终止
                        killed += 1
                        failed -= 1
                        need_root.remove(pid)
            except Exception:
                pass

        # 如果还有需要root权限的进程，提示用户
        if need_root:
            print(f"⚠️  仍有 {len(need_root)} 个进程需要root权限才能清理")
            print(
                f"   这些进程的PID: {need_root[:10]}{'...' if len(need_root) > 10 else ''}"
            )
            print(f"   建议:")
            print(f"   1. 使用root用户运行脚本")
            print(
                f"   2. 或手动清理: sudo kill -9 {' '.join(map(str, need_root[:10]))}"
            )
            if len(need_root) > 10:
                print(f"   3. 或使用清理脚本: sudo bash cleanup_sampling_as_root.sh")

    if killed > 0:
        print(f"✓ 成功清理 {killed} 个进程")
        time.sleep(1)  # 等待资源释放
    if failed > 0:
        print(f"⚠️  {failed} 个进程清理失败（可能需要root权限）")

    return killed
