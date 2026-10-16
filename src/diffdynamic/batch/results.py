"""Results responsibilities extracted from batch_sampleandeval_parallel.py."""

from diffdynamic.batch.state import (
    OUTPUT_DIR,
    Path,
    glob,
    np,
    os,
    re,
    time,
    torch,
)
from diffdynamic.batch import state as _state


def find_latest_result_file(data_id, output_dir=None, min_mtime=None):
    """
    查找指定data_id最新生成的.pt文件

    Args:
        data_id: 数据ID
        output_dir: 输出目录（默认：outputs）
        min_mtime: 若给定，只考虑修改时间 >= 该值（Unix 时间戳）的文件，避免误用本次运行前生成的旧 .pt

    Returns:
        Path对象或None
    """
    if output_dir is None:
        env_rp = os.environ.get("DD_RESULT_PATH")
        output_dir = Path(env_rp) if env_rp else OUTPUT_DIR

    # 查找所有匹配的.pt文件（格式：result_{data_id}_*.pt）
    pattern = str(output_dir / f"result_{data_id}_*.pt")
    pt_files = glob.glob(pattern)

    if min_mtime is not None:
        pt_files = [p for p in pt_files if os.path.getmtime(p) >= min_mtime]

    if not pt_files:
        return None

    # 按修改时间排序，返回最新的
    pt_files.sort(key=os.path.getmtime, reverse=True)

    # 返回最新的文件
    return Path(pt_files[0]) if pt_files else None


def _parse_results_saved_pt_path(subprocess_text):
    """从 sample_diffusion 的 stdout/stderr 中解析 `Results saved to: <path>`。"""
    if not subprocess_text:
        return None
    m = re.search(r"Results saved to:\s*(.+?)(?:\r?\n|$)", subprocess_text)
    if not m:
        return None
    p = m.group(1).strip()
    return Path(p) if p else None


def extract_data_id_from_pt_filename(pt_path):
    """
    从 .pt 文件名提取 data_id（口袋编号）
    格式：result_{data_id}.pt、result_{data_id}_{timestamp}.pt、result_{data_id}_it{N}_{timestamp}.pt
    （Prudent）
    另：若路径为 docs/<data_id>/.../sample.pt（如 IPDiff sample_for_pocket），用上级数字目录作为 id。

    Returns:
        str: data_id，若无法解析则返回 'unknown'
    """
    path = Path(pt_path).resolve()
    stem = path.stem
    if stem.startswith("result_"):
        parts = stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            return parts[1]
    # docs/10/IPDiff/sample.pt → 10
    try:
        if path.parent.parent.name.isdigit():
            return path.parent.parent.name
    except (IndexError, AttributeError):
        pass
    return "unknown"


def read_evaluation_results(pt_file_path, data_id, wait_timeout=300):
    """
    读取评估结果文件中的统计数据

    Args:
        pt_file_path: 采样结果.pt文件路径
        data_id: 数据ID
        wait_timeout: 等待评估结果的最大时间（秒）

    Returns:
        tuple: (success, vina_mean, vina_median, num_scores, message, eval_output_dir)
    """
    if torch is None or np is None:
        return (False, None, None, 0, "torch或numpy未安装", None)

    pt_file_path = Path(pt_file_path).resolve()
    outputs_dir = pt_file_path.parent

    # 从.pt文件名提取口袋编号
    pt_filename = pt_file_path.stem
    if pt_filename.startswith("result_"):
        parts = pt_filename.split("_")
        if len(parts) >= 3:
            pocket_id = parts[1]
        else:
            pocket_id = str(data_id)
    else:
        pocket_id = str(data_id)

    # 查找评估目录
    eval_dirs = list(outputs_dir.glob(f"eval_{pocket_id}_*"))
    if not eval_dirs:
        eval_dirs = list(outputs_dir.glob("eval_*"))

    if not eval_dirs:
        return (
            False,
            None,
            None,
            0,
            f"未找到评估输出目录（在 {outputs_dir} 中，查找模式: eval_{pocket_id}_*）",
            None,
        )

    # 优先选择带时间戳的新格式目录
    timestamp_pattern = r"_\d{8}_\d{6}_"
    new_format_dirs = [d for d in eval_dirs if re.search(timestamp_pattern, d.name)]
    old_format_dirs = [d for d in eval_dirs if d not in new_format_dirs]

    if new_format_dirs:
        new_format_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        eval_output_dir = new_format_dirs[0]
    elif old_format_dirs:
        old_format_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        eval_output_dir = old_format_dirs[0]
    else:
        eval_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        eval_output_dir = eval_dirs[0]

    if not eval_output_dir.exists():
        return (False, None, None, 0, f"评估输出目录不存在: {eval_output_dir}", None)

    # 等待评估结果文件生成
    start_wait = time.time()
    eval_result_files = []
    while time.time() - start_wait < wait_timeout:
        eval_result_files = list(eval_output_dir.glob("eval_results_*.pt"))
        if eval_result_files:
            break
        time.sleep(2)

    if not eval_result_files:
        all_files = list(eval_output_dir.glob("*"))
        file_list = ", ".join([f.name for f in all_files[:10]])
        if len(all_files) > 10:
            file_list += f" ... (共{len(all_files)}个文件)"
        return (
            False,
            None,
            None,
            0,
            f"等待{wait_timeout}秒后仍未找到评估结果文件 (eval_results_*.pt)\n"
            f"   评估目录: {eval_output_dir}\n"
            f"   目录中的文件: {file_list if all_files else '空目录'}",
            str(eval_output_dir),
        )

    try:
        latest_eval_file = max(eval_result_files, key=os.path.getmtime)
        eval_data = torch.load(latest_eval_file, map_location="cpu")

        statistics = eval_data.get("statistics", {})
        vina_dock_scores = statistics.get("vina_dock_scores", [])
        vina_score_only_scores = statistics.get("vina_score_only_scores", [])
        vina_minimize_scores = statistics.get("vina_minimize_scores", [])
        vina_scores = statistics.get("vina_scores", [])

        if vina_dock_scores:
            vina_scores = vina_dock_scores
        elif vina_minimize_scores:
            vina_scores = vina_minimize_scores
        elif vina_score_only_scores:
            vina_scores = vina_score_only_scores

        n_reconstruct_success = eval_data.get("n_reconstruct_success", 0)
        n_eval_success = eval_data.get("n_eval_success", 0)

        if not vina_scores:
            diagnostic_msg = f"评估结果中无vina得分"
            if n_reconstruct_success > 0 and n_eval_success == 0:
                diagnostic_msg += (
                    f" (重建成功{n_reconstruct_success}个，但对接全部失败)"
                )
            elif n_reconstruct_success == 0:
                diagnostic_msg += f" (重建失败，重建成功数: {n_reconstruct_success})"
            return (False, None, None, 0, diagnostic_msg, str(eval_output_dir))

        vina_mean = float(np.mean(vina_scores))
        vina_median = float(np.median(vina_scores))
        num_scores = len(vina_scores)

        return (
            True,
            vina_mean,
            vina_median,
            num_scores,
            f"成功读取评估结果，得分数量: {num_scores}",
            str(eval_output_dir),
        )

    except Exception as e:
        return (
            False,
            None,
            None,
            0,
            f"读取评估结果异常: {str(e)}",
            str(eval_output_dir),
        )


def _resolve_final_eval_dir_from_evaluate_subprocess(text, outputs_dir: Path, data_id):
    """
    解析 evaluate_pt_with_correct_reconstruct 打印的「评估输出目录」，
    或在无法解析时选取 outputs_dir 下最新的 eval_*（与 generate_eval_dir_name 一致的单层目录）。
    """
    if text:
        m = re.search(r"评估输出目录:\s*(.+?)\s*$", text, re.MULTILINE)
        if m:
            p = Path(m.group(1).strip())
            if p.is_dir():
                return str(p.resolve())
    # 新格式 eval_YYYYMMDD_HHMMSS_<data_id>_...
    ts_pat = re.compile(r"^eval_\d{8}_\d{6}_" + re.escape(str(data_id)) + r"(?:_|$)")
    candidates = [
        p for p in outputs_dir.iterdir() if p.is_dir() and ts_pat.search(p.name)
    ]
    if not candidates:
        candidates = [p for p in outputs_dir.glob("eval_*") if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(candidates[0].resolve())


def _eval_output_dirs_from_batch_results(results):
    """从本批次 ``process_evaluation_task`` 返回值中解析评估输出目录（去重保序）。"""
    seen = set()
    out = []
    for r in results or []:
        if len(r) < 6 or not r[1]:
            continue
        ev = r[5]
        if not ev:
            continue
        p = Path(ev).resolve()
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _data_id_from_eval_dir_name(dir_name: str):
    """
    从 eval 目录名解析口袋/数据 ID。

    新格式（与 ``generate_eval_dir_name`` 一致）: ``eval_YYYYMMDD_HHMMSS_<data_id>_...``
    旧格式: ``eval_<data_id>_...``（第二段为整数 ID）
    """
    if not dir_name.startswith("eval_"):
        return None
    m = re.match(r"^eval_\d{8}_\d{6}_(\d+)(?:_|$)", dir_name)
    if m:
        return int(m.group(1))
    parts = dir_name.split("_")
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            return parts[1]
    return None


def normalize_batch_summary_path(path) -> Path:
    """批次主汇总表统一为 ``.csv``；若传入 ``.xlsx`` 则改为同主名的 ``.csv``。"""
    p = Path(path)
    if p.suffix.lower() == ".xlsx":
        return p.with_suffix(".csv")
    return p


def resolve_eval_only_excel_path(excel_arg, batchsummary_dir, default_excel_path):
    """
    仅评估模式（--pt_dir / --pt_file）下解析批次汇总主表输出路径。

    相对路径统一落在仓库 batchsummary/ 下，便于与 merged_summary.csv 同目录管理；
    绝对路径保留用户指定位置。路径中的 ``.xlsx`` 会规范为 ``.csv``。
    """
    if excel_arg is None or str(excel_arg).strip() == "":
        return normalize_batch_summary_path(default_excel_path)
    p = Path(excel_arg)
    if p.is_absolute():
        return normalize_batch_summary_path(p)
    return normalize_batch_summary_path((Path(batchsummary_dir) / p).resolve())
