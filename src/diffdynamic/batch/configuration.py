"""Configuration responsibilities extracted from batch_sampleandeval_parallel.py."""

from diffdynamic.batch.state import (
    Path,
    os,
    pd,
    re,
    yaml,
)
from diffdynamic.batch import state as _state


def sample_subprocess_timeout_seconds(config_file):
    """
    采样子进程超时（秒）。
    optimization.enable 或 scaffold.enable 时默认 8 小时（scaffold evolve 多代进化耗时长），
    否则 1 小时。环境变量 SAMPLE_SUBPROCESS_TIMEOUT 始终优先（若可解析为浮点数）。
    """
    try:
        env = os.environ.get("SAMPLE_SUBPROCESS_TIMEOUT")
        if env is not None and str(env).strip() != "":
            return float(env)
    except ValueError:
        pass
    default_long = 28800.0
    default_short = 3600.0
    if yaml is None or not config_file:
        return default_short
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        sample_cfg = cfg.get("sample", {}) if cfg else {}
        if sample_cfg.get("optimization", {}).get("enable"):
            return default_long
        if sample_cfg.get("scaffold", {}).get("enable"):
            return default_long
    except Exception:
        pass
    return default_short


def format_float_for_filename(value):
    """
    将浮点数格式化为文件名格式，用p代替小数点
    例如：80.0 -> 80p0, 10.5 -> 10p5, 80 -> 80p0
    """
    if isinstance(value, (int, float)):
        # 转换为浮点数以确保统一格式
        float_value = float(value)
        # 转换为字符串，用p代替小数点
        str_value = str(float_value)
        return str_value.replace(".", "p")
    return str(value)


def generate_config_params_string(config_file):
    """
    从sampling.yml配置文件中提取核心参数并生成参数字符串
    格式根据调度模式动态调整：
    - 梯度融合：gf{mode}_{start}_{end}
    - 时间边界：tl{time_boundary}
    - 大步探索阶段（根据schedule模式）：
      * lambda模式：lslambda_{ls_a}_{ls_b}_lsstep_{step_size}_lsnoise_{noise_scale}
      * linear模式：lslinear_{lower}_{upper}_lsstep_{step_size}_lsnoise_{noise_scale}
      * fixed模式：lsfixed_{stride}_lsstep_{step_size}_lsnoise_{noise_scale}
      * 其他模式：ls{schedule}_{stride}_lsstep_{step_size}_lsnoise_{noise_scale}
    - 精炼阶段（根据schedule模式）：
      * lambda模式：rflambda_{rf_a}_{rf_b}_rfstep_{step_size}_rfnoise_{noise_scale}
      * linear模式：rflinear_{lower}_{upper}_rfstep_{step_size}_rfnoise_{noise_scale}
      * fixed模式：rffixed_{stride}_rfstep_{step_size}_rfnoise_{noise_scale}
      * 其他模式：rf{schedule}_{stride}_rfstep_{step_size}_rfnoise_{noise_scale}

    例如：
    - lambda模式：gfquadratic_1_0_tl750_lslambda_80p0_20p0_lsstep_0p6_lsnoise_0p0_rflambda_10p0_5p0_rfstep_0p4_rfnoise_0p05
    - linear模式：gfquadratic_1_0_tl750_lslinear_20_20_lsstep_0p6_lsnoise_0p0_rflinear_5_20_rfstep_0p4_rfnoise_0p05
    - fixed模式：gfquadratic_1_0_tl750_lsfixed_25_lsstep_0p6_lsnoise_0p0_rffixed_10_rfstep_0p4_rfnoise_0p05

    Args:
        config_file: 配置文件路径

    Returns:
        str: 参数字符串，如果读取失败则返回空字符串
    """
    if yaml is None:
        return ""

    try:
        config_path = Path(config_file)
        if not config_path.exists():
            return ""

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config:
            return ""

        params = []

        # 梯度融合参数
        if "model" in config and "grad_fusion_lambda" in config["model"]:
            gf_config = config["model"]["grad_fusion_lambda"]
            mode = gf_config.get("mode", "unknown")
            start = gf_config.get("start", 0)
            end = gf_config.get("end", 0)
            params.append(f"gf{mode}_{start}_{end}")

        # 时间边界
        if "sample" in config and "dynamic" in config["sample"]:
            dynamic_config = config["sample"]["dynamic"]
            time_boundary = dynamic_config.get("time_boundary", 0)
            params.append(f"tl{time_boundary}")

            # 大步探索阶段参数（根据schedule模式动态调整）
            if "large_step" in dynamic_config:
                ls_config = dynamic_config["large_step"]
                schedule = ls_config.get("schedule", "unknown")

                # 获取step_size和noise_scale参数
                ls_step_size = ls_config.get("step_size", 1.0)
                ls_noise_scale = ls_config.get("noise_scale", 0.0)
                ls_step_size_str = format_float_for_filename(ls_step_size)
                ls_noise_scale_str = format_float_for_filename(ls_noise_scale)

                if schedule == "lambda":
                    # Lambda调度模式
                    ls_a = ls_config.get("lambda_coeff_a", 0.0)
                    ls_b = ls_config.get("lambda_coeff_b", 0.0)
                    ls_a_str = format_float_for_filename(ls_a)
                    ls_b_str = format_float_for_filename(ls_b)
                    params.append(
                        f"lslambda_{ls_a_str}_{ls_b_str}_lsstep_{ls_step_size_str}_lsnoise_{ls_noise_scale_str}"
                    )
                elif schedule == "linear":
                    # 线性调度模式
                    ls_lower = ls_config.get("linear_step_lower", 0)
                    ls_upper = ls_config.get("linear_step_upper", 0)
                    params.append(
                        f"lslinear_{ls_lower}_{ls_upper}_lsstep_{ls_step_size_str}_lsnoise_{ls_noise_scale_str}"
                    )
                elif schedule == "fixed":
                    # 固定步长模式
                    ls_stride = ls_config.get("stride", 0)
                    params.append(
                        f"lsfixed_{ls_stride}_lsstep_{ls_step_size_str}_lsnoise_{ls_noise_scale_str}"
                    )
                else:
                    # 其他模式，使用stride作为标识
                    ls_stride = ls_config.get("stride", 0)
                    schedule_safe = str(schedule).replace("-", "_").replace(" ", "_")
                    params.append(
                        f"ls{schedule_safe}_{ls_stride}_lsstep_{ls_step_size_str}_lsnoise_{ls_noise_scale_str}"
                    )

            # 精炼阶段参数（根据schedule模式动态调整）
            if "refine" in dynamic_config:
                rf_config = dynamic_config["refine"]
                schedule = rf_config.get("schedule", "unknown")

                # 获取step_size和noise_scale参数
                rf_step_size = rf_config.get("step_size", 0.2)
                rf_noise_scale = rf_config.get("noise_scale", 0.05)
                rf_step_size_str = format_float_for_filename(rf_step_size)
                rf_noise_scale_str = format_float_for_filename(rf_noise_scale)

                if schedule == "lambda":
                    # Lambda调度模式
                    rf_a = rf_config.get("lambda_coeff_a", 0.0)
                    rf_b = rf_config.get("lambda_coeff_b", 0.0)
                    rf_a_str = format_float_for_filename(rf_a)
                    rf_b_str = format_float_for_filename(rf_b)
                    params.append(
                        f"rflambda_{rf_a_str}_{rf_b_str}_rfstep_{rf_step_size_str}_rfnoise_{rf_noise_scale_str}"
                    )
                elif schedule == "linear":
                    # 线性调度模式
                    rf_lower = rf_config.get("linear_step_lower", 0)
                    rf_upper = rf_config.get("linear_step_upper", 0)
                    params.append(
                        f"rflinear_{rf_lower}_{rf_upper}_rfstep_{rf_step_size_str}_rfnoise_{rf_noise_scale_str}"
                    )
                elif schedule == "fixed":
                    # 固定步长模式
                    rf_stride = rf_config.get("stride", 0)
                    params.append(
                        f"rffixed_{rf_stride}_rfstep_{rf_step_size_str}_rfnoise_{rf_noise_scale_str}"
                    )
                else:
                    # 其他模式，使用stride作为标识
                    rf_stride = rf_config.get("stride", 0)
                    schedule_safe = str(schedule).replace("-", "_").replace(" ", "_")
                    params.append(
                        f"rf{schedule_safe}_{rf_stride}_rfstep_{rf_step_size_str}_rfnoise_{rf_noise_scale_str}"
                    )

        return "_".join(params)

    except Exception as e:
        print(f"⚠️  读取配置文件参数失败: {e}")
        return ""


def _yaml_prudent_enabled(config_path):
    """sampling.yml 是否启用 Prudent（mode=prudent 或 dynamic.prudent.enable）。"""
    if yaml is None or config_path is None:
        return False
    p = Path(config_path)
    if not p.is_file():
        return False
    try:
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not cfg:
            return False
        sample_cfg = cfg.get("sample", {}) or {}
        if sample_cfg.get("mode") == "prudent":
            return True
        dyn = sample_cfg.get("dynamic", {}) or {}
        pr = dyn.get("prudent", {}) or {}
        return bool(pr.get("enable"))
    except Exception:
        return False


def _pt_prudent_hint(pt_path):
    """从 .pt 内容判断是否 Prudent（meta.prudent 或 mode=prudent）。"""
    try:
        try:
            import torch
        except ImportError:
            return False
        p = Path(pt_path)
        if not p.is_file():
            return False
        try:
            obj = torch.load(str(p), map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(str(p), map_location="cpu")
        if not isinstance(obj, dict):
            return False
        if obj.get("mode") == "prudent":
            return True
        meta = obj.get("meta")
        return bool(meta is not None and isinstance(meta, dict) and meta.get("prudent"))
    except Exception:
        return False


def resolve_eval_vina_modes(cli_value, *, config_path=None, pt_path=None):
    """得出传给 evaluate_pt 的 --vina-modes 字符串。

    ``auto``：若 .pt 显式 Prudent 或 YAML 启用 Prudent → ``none``；否则不传参（沿用 evaluate 默认三种 Vina）。
    ``default`` / ``full``：不传参。
    ``none`` / ``off`` / …：``none``。
    其余：原样传给 evaluate（如 ``score_only``、``dock,minimize``）。
    """
    raw = cli_value if cli_value is not None else "auto"
    if raw is None or str(raw).strip() == "":
        raw = "auto"
    v = str(raw).strip().lower()
    if v == "auto":
        if pt_path is not None:
            pt_p = Path(pt_path)
            if pt_p.is_file() and _pt_prudent_hint(pt_p):
                return "none"
        if config_path is not None and _yaml_prudent_enabled(config_path):
            return "none"
        return None
    if v in ("default", "full", "all", "triple"):
        return None
    if v in ("none", "off", "skip", "no_vina"):
        return "none"
    return str(raw).strip()


def medchem_opts_from_args(args):
    """Collect the --medchem-* passthrough switches into one dict.

    Empty when the layer is off, so evaluate keeps its own defaults and the
    command line stays identical to historical runs.
    """
    if not getattr(args, "medchem_optimize", False):
        return {}
    return {
        "medchem_optimize": True,
        "medchem_optimize_config": getattr(args, "medchem_optimize_config", None),
        "keep_preopt_sdf": getattr(args, "keep_preopt_sdf", None),
    }


def flatten_config(config, parent_key="", sep="."):
    """
    将嵌套的配置字典扁平化为键值对列表

    Args:
        config: 配置字典
        parent_key: 父键（用于构建完整的键路径）
        sep: 分隔符

    Returns:
        list: [(键路径, 值), ...] 的列表
    """
    items = []
    for key, value in config.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if value is None:
            # None值转换为字符串
            items.append((new_key, "null"))
        elif isinstance(value, dict):
            # 递归处理嵌套字典
            items.extend(flatten_config(value, new_key, sep=sep))
        elif isinstance(value, list):
            # 列表转换为字符串
            items.append((new_key, str(value)))
        else:
            # 普通值
            items.append((new_key, value))

    return items


def build_lambda_schedule(start_t, end_t, coeff_a, coeff_b, num_timesteps):
    """构建 lambda 调度序列（从show_sampling_steps.py复制）"""
    start_t = int(max(0, min(start_t, num_timesteps - 1)))
    end_t = int(max(0, min(end_t, num_timesteps - 1)))
    if start_t == end_t:
        return [start_t], []

    decreasing = start_t > end_t
    step_sign = -1 if decreasing else 1
    t = start_t
    indices = []
    step_sizes = []

    while True:
        indices.append(t)
        if t == end_t:
            break
        lambda_t = float(t) / float(num_timesteps)
        lambda_t = max(min(lambda_t, 1.0), 0.0)
        n = coeff_a * lambda_t + coeff_b
        step = max(1, int(round(n)))
        step_sizes.append(step)
        t_next = t + step_sign * step
        if decreasing and t_next < end_t:
            t_next = end_t
        if not decreasing and t_next > end_t:
            t_next = end_t
        if t_next == t:
            t_next = t + step_sign
        t = int(max(0, min(t_next, num_timesteps - 1)))

    return indices, step_sizes


def build_linear_schedule(start_t, end_t, step_upper, step_lower, num_timesteps):
    """构建线性调度序列（从show_sampling_steps.py复制）"""
    if start_t == end_t:
        return [start_t], []

    decreasing = start_t > end_t
    step_sign = -1 if decreasing else 1
    t = start_t
    indices = []
    step_sizes = []

    initial_range = abs(start_t - end_t)

    while True:
        indices.append(t)
        if t == end_t:
            break

        remaining_range = abs(t - end_t)
        if initial_range > 0:
            progress = float(remaining_range) / float(initial_range)
        else:
            progress = 0.0
        progress = max(0.0, min(1.0, progress))

        coeff_a = step_upper - step_lower
        coeff_b = step_lower
        n = coeff_a * progress + coeff_b
        step = max(1, int(round(n)))
        step_sizes.append(step)

        t_next = t + step_sign * step
        if decreasing and t_next < end_t:
            t_next = end_t
        if not decreasing and t_next > end_t:
            t_next = end_t
        if t_next == t:
            t_next = t + step_sign

        t = int(max(0, min(t_next, num_timesteps - 1)))

    return indices, step_sizes


def build_fixed_schedule(start_t, end_t, stride, num_timesteps):
    """构建固定步长调度序列（从show_sampling_steps.py复制）"""
    start_t = int(max(0, min(start_t, num_timesteps - 1)))
    end_t = int(max(0, min(end_t, num_timesteps - 1)))
    if start_t == end_t:
        return [start_t], []

    decreasing = start_t > end_t
    step_sign = -1 if decreasing else 1
    stride = max(1, int(stride))

    indices = []
    step_sizes = []
    t = start_t

    while True:
        indices.append(t)
        if t == end_t:
            break

        t_next = t + step_sign * stride
        step_sizes.append(stride)

        if decreasing and t_next < end_t:
            t_next = end_t
        if not decreasing and t_next > end_t:
            t_next = end_t

        if t_next == t:
            t_next = t + step_sign

        t = int(max(0, min(t_next, num_timesteps - 1)))

    return indices, step_sizes


def calculate_total_sampling_steps(config):
    """
    计算总采样步数（跳步总次数）和实际长度

    Args:
        config: 配置字典

    Returns:
        tuple: (总采样步数, 实际长度)，如果计算失败则返回(None, None)
    """
    try:
        sample_cfg = config.get("sample", {})
        if not isinstance(sample_cfg, dict):
            return None, None

        num_timesteps = sample_cfg.get("num_steps", 1000)
        dynamic_cfg = sample_cfg.get("dynamic", {})
        if not isinstance(dynamic_cfg, dict):
            return None, None

        large_step_cfg = dynamic_cfg.get("large_step", {})
        if not isinstance(large_step_cfg, dict):
            large_step_cfg = {}

        refine_cfg = dynamic_cfg.get("refine", {})
        if not isinstance(refine_cfg, dict):
            refine_cfg = {}

        # 获取 time_boundary
        time_boundary = dynamic_cfg.get("time_boundary", 600)

        # Large Step 阶段
        large_step_schedule = large_step_cfg.get("schedule", "lambda")
        large_step_time_lower = time_boundary
        large_step_time_upper = num_timesteps - 1
        large_step_size = large_step_cfg.get("step_size", 1.0)  # 获取step_size

        # Refine 阶段
        refine_schedule = refine_cfg.get("schedule", "lambda")
        refine_time_upper = time_boundary
        refine_time_lower = refine_cfg.get("time_lower", 0)
        refine_step_size = refine_cfg.get("step_size", 0.2)  # 获取step_size

        # 计算 Large Step 阶段时间步
        if large_step_schedule == "lambda":
            lambda_coeff_a = large_step_cfg.get("lambda_coeff_a", 80.0)
            lambda_coeff_b = large_step_cfg.get("lambda_coeff_b", 20.0)
            large_step_indices, _ = build_lambda_schedule(
                large_step_time_upper,
                large_step_time_lower,
                lambda_coeff_a,
                lambda_coeff_b,
                num_timesteps,
            )
        elif large_step_schedule == "linear":
            step_upper = large_step_cfg.get("linear_step_upper", 100.0)
            step_lower = large_step_cfg.get("linear_step_lower", 20.0)
            large_step_indices, _ = build_linear_schedule(
                large_step_time_upper,
                large_step_time_lower,
                step_upper,
                step_lower,
                num_timesteps,
            )
        else:
            stride = large_step_cfg.get("stride", 15)
            large_step_indices, _ = build_fixed_schedule(
                large_step_time_upper, large_step_time_lower, stride, num_timesteps
            )

        # 计算 Refine 阶段时间步
        if refine_schedule == "lambda":
            lambda_coeff_a = refine_cfg.get("lambda_coeff_a", 40.0)
            lambda_coeff_b = refine_cfg.get("lambda_coeff_b", 5.0)
            refine_indices, _ = build_lambda_schedule(
                refine_time_upper,
                refine_time_lower,
                lambda_coeff_a,
                lambda_coeff_b,
                num_timesteps,
            )
        elif refine_schedule == "linear":
            step_upper = refine_cfg.get("linear_step_upper", 40.0)
            step_lower = refine_cfg.get("linear_step_lower", 5.0)
            refine_indices, _ = build_linear_schedule(
                refine_time_upper,
                refine_time_lower,
                step_upper,
                step_lower,
                num_timesteps,
            )
        else:
            stride = refine_cfg.get("stride", 8)
            refine_indices, _ = build_fixed_schedule(
                refine_time_upper, refine_time_lower, stride, num_timesteps
            )

        # 计算实际长度：ls的步数 × ls的step_size + rf的步数 × rf的step_size
        # 步数是指每个阶段的采样点数量（即indices列表的长度）
        large_step_num_steps = len(large_step_indices) if large_step_indices else 0
        refine_num_steps = len(refine_indices) if refine_indices else 0

        # 计算实际长度
        actual_length = (
            large_step_num_steps * large_step_size + refine_num_steps * refine_step_size
        )

        # 合并时间步序列（去掉重复的连接点）
        if large_step_indices and refine_indices:
            if large_step_indices[-1] == refine_indices[0]:
                all_indices = large_step_indices + refine_indices[1:]
            else:
                all_indices = large_step_indices + refine_indices
        elif large_step_indices:
            all_indices = large_step_indices
        elif refine_indices:
            all_indices = refine_indices
        else:
            return None, None

        total_steps = len(all_indices)

        return total_steps, actual_length

    except Exception as e:
        print(f"⚠️  计算总采样步数和实际长度失败: {e}")
        return None, None


def load_config_to_dataframe(config_file):
    """
    将配置文件加载并转换为DataFrame

    Args:
        config_file: 配置文件路径

    Returns:
        pd.DataFrame: 包含参数路径和值的DataFrame，如果失败则返回None
    """
    if yaml is None:
        return None

    try:
        config_path = Path(config_file)
        if not config_path.exists():
            return None

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config:
            return None

        # 扁平化配置
        flat_config = flatten_config(config)

        # 计算总采样步数和实际长度
        total_steps, actual_length = calculate_total_sampling_steps(config)
        if total_steps is not None:
            flat_config.append(("计算.跳步总次数", total_steps))
        if actual_length is not None:
            flat_config.append(("计算.实际长度", actual_length))

        # 转换为DataFrame
        df = pd.DataFrame(flat_config, columns=["参数路径", "参数值"])

        # 将值转换为字符串（便于在Excel中显示）
        df["参数值"] = df["参数值"].astype(str)

        return df

    except Exception as e:
        print(f"⚠️  读取配置文件失败 {config_file}: {e}")
        return None


def parse_filename_params(filename):
    """
    从文件名解析参数

    示例文件名: batch_evaluation_summary_20260105_002424_gfquadratic_1_0_tl800_lslambda_60p0_20p0_lsstep_0p6_lsnoise_0p0_rflambda_10p0_5p0_rfstep_0p25_rfnoise_0p05.csv

    返回参数字典
    """
    params = {}
    filename_str = str(filename) if isinstance(filename, Path) else filename

    # 提取权重策略 (gfquadratic -> quadratic)
    gf_match = re.search(r"gf(\w+)", filename_str)
    if gf_match:
        params["权重策略"] = gf_match.group(1)

    # 提取开始权重和结束权重 (gfquadratic_1_0 -> 开始权重=1, 结束权重=0)
    weight_match = re.search(r"gf\w+_(\d+)_(\d+)", filename_str)
    if weight_match:
        params["开始权重"] = float(weight_match.group(1))
        params["结束权重"] = float(weight_match.group(2))

    # 提取时间长度 (tl800 -> 800)
    tl_match = re.search(r"tl(\d+)", filename_str)
    if tl_match:
        params["时间长度 (TL)"] = int(tl_match.group(1))

    # 提取LS Lambda值 (lslambda_60p0_20p0 -> LSLambda1=60.0, LSLambda2=20.0)
    ls_lambda_match = re.search(r"lslambda_(\d+p\d+)_(\d+p\d+)", filename_str)
    if ls_lambda_match:
        params["LSLambda1"] = float(ls_lambda_match.group(1).replace("p", "."))
        params["LSLambda2"] = float(ls_lambda_match.group(2).replace("p", "."))

    # 提取LS step size (lsstep_0p6 -> 0.6)
    ls_step_match = re.search(r"lsstep_(\d+p\d+)", filename_str)
    if ls_step_match:
        params["LSstepsize"] = float(ls_step_match.group(1).replace("p", "."))

    # 提取LS noise (lsnoise_0p0 -> 0.0)
    ls_noise_match = re.search(r"lsnoise_(\d+p\d+)", filename_str)
    if ls_noise_match:
        params["LSnosie"] = float(ls_noise_match.group(1).replace("p", "."))

    # 提取RF Lambda值 (rflambda_10p0_5p0 -> RFLambda1=10.0, RFLambda2=5.0)
    rf_lambda_match = re.search(r"rflambda_(\d+p\d+)_(\d+p\d+)", filename_str)
    if rf_lambda_match:
        params["RFLambda1"] = float(rf_lambda_match.group(1).replace("p", "."))
        params["RFLambda2"] = float(rf_lambda_match.group(2).replace("p", "."))

    # 提取RF step size (rfstep_0p25 -> 0.25)
    rf_step_match = re.search(r"rfstep_(\d+p\d+)", filename_str)
    if rf_step_match:
        params["RFstepsize"] = float(rf_step_match.group(1).replace("p", "."))

    # 提取RF noise (rfnoise_0p05 -> 0.05)
    rf_noise_match = re.search(r"rfnoise_(\d+p\d+)", filename_str)
    if rf_noise_match:
        params["RFnosie"] = float(rf_noise_match.group(1).replace("p", "."))

    return params
