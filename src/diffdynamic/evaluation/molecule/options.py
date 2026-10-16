"""Options responsibilities extracted from evaluate_pt_with_correct_reconstruct.py."""

from diffdynamic.evaluation.molecule.state import (
    Path,
    _ALL_VINA_MODES,
    _DEFAULT_VINA_MODES,
    csv,
    datetime,
    os,
    pickle,
)
from diffdynamic.evaluation.molecule import state as _state


def parse_vina_modes_arg(s):
    """解析 ``--vina-modes`` 逗号分隔串，返回 frozenset。

    ``none`` / ``off`` / ``skip`` / ``no_vina``：不运行 AutoDock Vina；Prudent .pt 由评测主流程从
    ``meta.refined_candidates[i].prudent_composite_detail`` 注入 ``vina_affinity`` 到 ``vina_score_only``。
    """
    if s is None:
        return _DEFAULT_VINA_MODES
    text = str(s).strip()
    if not text:
        return _DEFAULT_VINA_MODES
    low = text.lower()
    if low in ("none", "off", "skip", "no_vina"):
        return frozenset()
    # Web API 侧的别名：all/auto → 全部三种模式；auto 也可精确写成 full
    if low in ("auto", "all", "full"):
        return _ALL_VINA_MODES
    allowed = {"dock", "score_only", "minimize"}
    out = []
    for part in text.split(","):
        p = part.strip().lower().replace("-", "_")
        if not p:
            continue
        if p in ("none", "off", "skip", "no_vina"):
            raise ValueError("none/off/skip 不能与 dock / score_only / minimize 并列")
        if p == "scoreonly":
            p = "score_only"
        if p not in allowed:
            raise ValueError(
                f"无效的 Vina 模式 {part!r}，允许: dock, score_only, minimize, none（单独使用）"
            )
        if p not in out:
            out.append(p)
    if not out:
        raise ValueError(
            "至少选择一种 Vina 模式（dock / score_only / minimize），或使用 none 关闭 Vina"
        )
    return frozenset(out)


def _resolve_eval_parallel_workers(explicit):
    """对接阶段并行线程数：>1 时多个分子可同时跑对接，避免单分子占满等待时间。"""
    if explicit is not None:
        return max(1, int(explicit))
    env = os.environ.get("EVAL_PARALLEL_WORKERS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    cpu = os.cpu_count() or 4
    return max(2, min(8, cpu))


def _crossdocked_index_lookup(index_path, data_id):
    """从 CrossDocked 风格 index.pkl 取第 data_id 条的 (protein_rel, ligand_rel)，路径相对数据集根目录。"""
    try:
        ip = Path(index_path).expanduser()
        if not ip.is_file():
            return None
        with open(ip, "rb") as f:
            index = pickle.load(f)
        if not isinstance(index, list):
            return None
        if data_id < 0 or data_id >= len(index):
            return None
        item = index[data_id]
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            return str(item[0]), str(item[1])
        if isinstance(item, dict):
            p = (
                item.get("protein_filename")
                or item.get("protein")
                or item.get("protein_file")
            )
            l = (
                item.get("ligand_filename")
                or item.get("ligand")
                or item.get("ligand_file")
            )
            if p and l:
                return str(p), str(l)
    except Exception:
        return None
    return None


def _infer_data_id_from_docs_layout(pt_path):
    """若路径形如 .../docs/<数字>/.../file.pt，则返回该整数（与 batch_sampleandeval 的目录约定一致）。"""
    try:
        parts = Path(pt_path).resolve().parts
        for i, p in enumerate(parts):
            if p == "docs" and i + 1 < len(parts) and parts[i + 1].isdigit():
                return int(parts[i + 1])
    except Exception:
        pass
    return None


def _benchmark_ligands_csv_lookup(csv_path, data_id):
    """
    从 docs/100ligands.csv 按测试集 data_id 取 (ligand_rel, protein_rel)。
    与 IPDiff/TargetDiff 的 test_set[data_id] 一致；切勿与全量 index.pkl 的行号混用。
    """
    try:
        cp = Path(csv_path).expanduser()
        if not cp.is_file():
            return None
        with open(cp, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return None
            want = int(data_id)
            for row in reader:
                rid = row.get("data_id")
                if rid is None or str(rid).strip() == "":
                    continue
                try:
                    if int(str(rid).strip()) != want:
                        continue
                except ValueError:
                    continue
                lig = row.get("ligand_rel") or row.get("ligand")
                prot = row.get("protein_rel") or row.get("protein")
                if lig:
                    return str(lig).strip(), (str(prot).strip() if prot else None)
    except Exception:
        return None
    return None


def generate_eval_dir_name(data_id, config, timestamp=None):
    """生成包含所有配置参数的评估目录名（照抄自 batch_sample_all.py）

    格式: eval_{timestamp}_{data_id}_{grad_fusion_mode}_{start}_{end}_{time_lower}_{large_step_params}_{refine_params}

    Args:
        data_id: 数据ID
        config: 配置对象（可以是dict或EasyDict）
        timestamp: 时间戳（可选，格式: YYYYMMDD_HHMMSS）

    Returns:
        str: 目录名
    """

    # 支持dict和EasyDict
    def safe_get(obj, path, default):
        """安全获取嵌套属性"""
        keys = path.split(".")
        current = obj
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key, {})
            else:
                current = getattr(current, key, {})
            if current == {}:
                return default
        return current if current != {} else default

    # ✅ 时间戳放在最前面（在eval_之后，data_id之前）
    if timestamp:
        parts = [f"eval_{timestamp}_{data_id}"]
    else:
        # 如果没有提供时间戳，使用当前时间
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [f"eval_{timestamp}_{data_id}"]

    # Grad Fusion Lambda 参数
    grad_fusion_cfg = safe_get(config, "model.grad_fusion_lambda", None)
    if isinstance(grad_fusion_cfg, dict):
        mode = str(grad_fusion_cfg.get("mode", "none"))
        start = str(grad_fusion_cfg.get("start", 0))
        end = str(grad_fusion_cfg.get("end", 0))
        parts.append(f"gf{mode}_{start}_{end}")
    else:
        parts.append("gfnone_0_0")

    # Dynamic 采样参数
    dynamic_cfg = safe_get(config, "sample.dynamic", {})
    large_step_cfg = (
        dynamic_cfg.get("large_step", {}) if isinstance(dynamic_cfg, dict) else {}
    )
    refine_cfg = dynamic_cfg.get("refine", {}) if isinstance(dynamic_cfg, dict) else {}

    # 获取 time_boundary（支持向后兼容）
    def get_time_boundary_from_cfg(cfg, default=750):
        """获取阶段边界时间步，支持向后兼容"""
        if not isinstance(cfg, dict):
            return default
        # 优先读取统一的 time_boundary
        if "time_boundary" in cfg:
            return cfg.get("time_boundary", default)
        # 向后兼容：从 large_step.time_lower 获取
        large_step = cfg.get("large_step", {})
        if isinstance(large_step, dict) and "time_lower" in large_step:
            return large_step.get("time_lower", default)
        # 向后兼容：从 refine.time_upper 获取
        refine = cfg.get("refine", {})
        if isinstance(refine, dict) and "time_upper" in refine:
            return refine.get("time_upper", default)
        return default

    time_boundary = get_time_boundary_from_cfg(dynamic_cfg, 750)
    parts.append(f"tl{time_boundary}")

    # Large Step 参数
    large_schedule = (
        str(large_step_cfg.get("schedule", "none"))
        if isinstance(large_step_cfg, dict)
        else "none"
    )
    if large_schedule == "lambda":
        large_a = str(large_step_cfg.get("lambda_coeff_a", 0))
        large_b = str(large_step_cfg.get("lambda_coeff_b", 0))
        parts.append(f"ls{large_schedule}_{large_a}_{large_b}")
    elif large_schedule == "linear":
        large_upper = str(large_step_cfg.get("linear_step_upper", 0))
        large_lower = str(large_step_cfg.get("linear_step_lower", 0))
        parts.append(f"ls{large_schedule}_{large_upper}_{large_lower}")
    else:
        parts.append(f"ls{large_schedule}")

    # Refine 参数
    refine_schedule = (
        str(refine_cfg.get("schedule", "none"))
        if isinstance(refine_cfg, dict)
        else "none"
    )
    if refine_schedule == "lambda":
        refine_a = str(refine_cfg.get("lambda_coeff_a", 0))
        refine_b = str(refine_cfg.get("lambda_coeff_b", 0))
        parts.append(f"rf{refine_schedule}_{refine_a}_{refine_b}")
    elif refine_schedule == "linear":
        refine_upper = str(refine_cfg.get("linear_step_upper", 0))
        refine_lower = str(refine_cfg.get("linear_step_lower", 0))
        parts.append(f"rf{refine_schedule}_{refine_upper}_{refine_lower}")
    else:
        parts.append(f"rf{refine_schedule}")

    dir_name = "_".join(parts)
    # 替换可能不适合文件名的字符
    # 将浮点数中的点号替换为p（如1.0 -> 1p0），负号替换为m（如-1 -> m1）
    dir_name = dir_name.replace(".", "p").replace("-", "m")
    return dir_name
