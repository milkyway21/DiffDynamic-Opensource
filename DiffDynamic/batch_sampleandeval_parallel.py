#!/usr/bin/env python3
"""
批量采样和评估脚本：并行执行采样和评估
python3 batch_sampleandeval_parallel.py --start 0 --end 99 --gpus "3,4,5" \
  --num_cpu_cores 30 --cores_per_task 10 
功能：
1. 对每个 data_id 执行采样：python3 scripts/sample_diffusion.py configs/sampling.yml --data_id {i} --device cuda:{gpu_id}
   （dynamic 走 unified 还是 legacy 仅由配置文件 sample.dynamic.method 与模型决定，本脚本不覆盖。）
2. 找到生成的文件：outputs/result_{data_id}_[it{闭环轮数}]_*{timestamp}.pt（Prudent 时含 ``it{N}``＝ ``prudent.max_checkpoint_rounds``）
3. （可选）分子修复 --molecular-repair：mmff=对接前 RDKit MMFF；targetdiff_baseline_refine=按 sampling.yml 中 sample.targetdiff_baseline_refine 对 .pt 再扩散修复（与第 294–297 行配置一致）。
4. 执行评估：evaluate_pt_with_correct_reconstruct.py；``--eval-vina-modes auto``（默认）时若为 Prudent（yaml 或 .pt 内含 meta.prudent）则 ``--vina-modes none``，仅用 .pt 中 Prudent 阶段 Vina；非 Prudent 则与历史一致跑 dock/score_only/minimize。

并行配置：
- GPU: 使用GPU 2、3、4、5（4个GPU并行）
- CPU: 使用64个核心（通过multiprocessing限制）

使用方法：
    # 基本用法（0到99，使用默认GPU 2,3,4,5）
    python3 batch_sampleandeval_parallel.py
    cd /workspace/third_party/DecompDiff

export PYTHONPATH=/workspace/third_party/DecompDiff

cd /workspace/third_party/DecompDiff
export PYTHONPATH=/workspace/third_party/DecompDiff

python3 scripts/evaluate_mol_from_meta_full.py \
  /workspace/third_party/DecompDiff/outputs_decompdiff_bench_100p/sampling_drift_010_3dzh_A_rec_3u4i_cvr_lig_tt_docked_0_pocket/result.pt \
  --protein_root /workspace/data/crossdocked_v1.1_rmsd1.0_pocket10 \
  --docking_mode vina_score \
  --exhaustiveness 8 \
  --num_workers 1 \
  --result_path /workspace/docs/10/decompdiff_eval_manual
    # 指定范围
    python3 batch_sampleandeval_parallel.py --start 10 --end 10
    
    # 指定使用的GPU（多种格式）
    python3 batch_sampleandeval_parallel.py --gpus "0,1,2,3"
    python3 batch_sampleandeval_parallel.py --gpus "0-3"
    python3 batch_sampleandeval_parallel.py --gpus "0,2-4,6"
    python3 batch_sampleandeval_parallel.py --gpus "all"  # 使用所有可用GPU
    python3 batch_sampleandeval_parallel.py \
  --collect-benchmark-pocket 10 \
  --protein_root /workspace/data/crossdocked_v1.1_rmsd1.0_pocket10 \
  --methods DecompDiff \
  --num_cpu_cores 10 \
  --cores_per_task 10
    # 指定CPU核心数
    python3 batch_sampleandeval_parallel.py --num_cpu_cores 32

    # 单任务占满 CPU（1 路评估，适合极重对接/调试）5" --num_cpu_cores 40 --cores_per_task 10 --auto-cleanup

    # 指定每任务核心数（20 核、每任务 4 核 -> 最多 5 个并行评估）
    python3 batch_sampleandeval_parallel.py --num_cpu_cores 20 --cores_per_task 4

    # 推荐并行：4 GPU 同时采样；评估阶段最多 40//10=4 个进程并行（每进程 OMP/MKL 约 10 线程）
    python3 batch_sampleandeval_parallel.py --start 0 --end 99 \
      --gpus "1,2,5" --num_cpu_cores 40 --cores_per_task 10  --auto-cleanup
    python3 batch_sampleandeval_parallel.py \
  --data_ids "10" \
  --gpus "0" \
  --num_cpu_cores 20
    # 组合使用
    python3 batch_sampleandeval_parallel.py --start 0 --end 0 --gpus "2" --num_cpu_cores 40 --cores_per_task 40
    python3 batch_sampleandeval_parallel.py \
  --data_ids "10" \
  --gpus "0" \
  --num_cpu_cores 20
    # 指定蛋白质数据根目录
    python3 batch_sampleandeval_parallel.py --protein_root /path/to/protein/data
    
    # 自定义蛋白文件（方案B）：仅蛋白或蛋白+配体
    python3 batch_sampleandeval_parallel.py --protein_path /path/to/your/pocket.pdb
    python3 batch_sampleandeval_parallel.py --protein_path /path/to/pocket.pdb --ligand_path /path/to/ligand.sdf
    python3 batch_sampleandeval_parallel.py --protein_path /path/to/pocket.pdb --use_dataset_for_pocket
    # 仓库挂载到容器 /workspace 时（本地则把路径换成你的仓库目录，如 .../DiffDynamic/shoc2/）
    python3 batch_sampleandeval_parallel.py \
  --protein_path /workspace/shoc2/shoc2.pdb \
  --ligand_path /workspace/shoc2/shoc2ligand.sdf \
  --protein_root /workspace/shoc2 \
  --gpus "0" \
  --num_cpu_cores 20
    # 只生成模式（不执行评估）
    python3 batch_sampleandeval_parallel.py --start 42 --end 42 --gpus "2" --sample-only
    /workspace/outputs/result_0_20260411_115940.pt
    # 若 --protein_root 不存在会自动尝试常见路径；custom .pt 的蛋白路径通常已嵌入文件
    python3 batch_sampleandeval_parallel.py --pt_file outputs/result_0_20260411_115940.pt --protein_root /workspace/data --num_cpu_cores 20
    python3 batch_sampleandeval_parallel.py --pt_file "file1.pt,file2.pt" --protein_root /workspace/data --num_cpu_cores 20 --excel_file my_results.csv
    python3 batch_sampleandeval_parallel.py --pt_dir dd0414base/ddrepair \
  --protein_root /workspace/data --num_cpu_cores 20 --eval-show-output

    # 扫描目录评估 .pt（不采样，仅评估；默认只匹配 result_*.pt，见 --pt-dir-glob）
    # 结果写入仓库 batchsummary/：批次 CSV 主表（分子评估数据）与侧车 CSV（正常分子、统计信息、配置等），以及 merged_summary.csv / merged_valid_molecules.csv
    python3 batch_sampleandeval_parallel.py --pt_dir dd0414base/ddrepair --protein_root /workspace/data --num_cpu_cores 20
    # 该目录下任意 .pt（顶层非递归）：--pt-dir-glob '*.pt'
    python3 batch_sampleandeval_parallel.py --pt_dir /path/to/folder --pt-dir-glob '*.pt' --protein_root /workspace/data --num_cpu_cores 20
    # 需要看 evaluate 详细过程：加 --eval-save-log（写 eval_*/evaluate_subprocess.log）或 --eval-show-output（实时打印，多任务会交错）
    # --excel_file 为相对路径时同样写入 batchsummary/（非仅当前目录）
    python3 batch_sampleandeval_parallel.py --pt_dir outputs/ --protein_root /workspace/data --excel_file dir_evaluation.csv

    # 按测试口袋编号：各基线 .pt 复制到 docs/<编号>/ 下并对每个 .pt 启动 evaluate（需蛋白根目录）
    # 并行度与批量评估相同：min(任务数, num_cpu_cores // cores_per_task)
    python3 batch_sampleandeval_parallel.py --collect-benchmark-pocket 42 --protein_root /workspace/data \
      --num_cpu_cores 20 --cores_per_task 4

    # 只处理特定方法（节省计算资源，跳过 DecompDiff 和 DiffSBDD）
    python3 batch_sampleandeval_parallel.py --collect-benchmark-pocket 42 \
      --protein_root /workspace/data --methods TargetDiff,JSDPT3010,MolForm,IPDiff,GlintDM

    # 只对比核心方法（最快组合）
    python3 batch_sampleandeval_parallel.py --collect-benchmark-pocket 42 \
      --protein_root /workspace/data --methods TargetDiff,JSDPT3010 --num_cpu_cores 10
"""

import os
import sys
import subprocess
import argparse
import time
import pickle
from datetime import datetime, timezone, timedelta
from pathlib import Path
import glob
import shutil
import threading
import traceback
import re
import signal
from multiprocessing import Pool, Manager, cpu_count
from functools import partial

try:
    import pandas as pd
except ImportError:
    pd = None
    print("⚠️  警告: pandas未安装，无法记录Excel。运行: pip install pandas openpyxl")
else:
    try:
        import openpyxl
    except ImportError:
        print("⚠️  警告: openpyxl未安装，无法写入Excel。运行: pip install openpyxl")

try:
    import torch
except ImportError:
    torch = None
    print("⚠️  警告: torch未安装，采样/评估子进程可能不可用。运行: pip install torch")
try:
    import numpy as np
except ImportError:
    np = None
    print("⚠️  警告: numpy未安装，可能影响功能。运行: pip install numpy")

try:
    import yaml
except ImportError:
    yaml = None
    print("⚠️  警告: yaml未安装，无法读取配置文件参数。运行: pip install pyyaml")

# 并行配置（默认值，可通过命令行参数覆盖）
DEFAULT_GPU_IDS = [2, 3, 4, 5]  # 默认使用的GPU ID列表
DEFAULT_NUM_CPU_CORES = 64  # 默认使用的CPU核心数

def record_pocket_generation_time(data_id, gpu_id, generation_time, status='成功', pt_file=None):
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
        timestamp_str = datetime.now(cst).strftime('%Y-%m-%d %H:%M:%S')
        
        # 准备数据
        new_row = {
            'Pocket_ID': f'pocket_{data_id}',
            'Data_ID': data_id,
            'GPU_ID': gpu_id,
            'Generation_Time(s)': round(generation_time, 2),
            'Status': status,
            'Timestamp': timestamp_str,
            'PT_File': str(pt_file) if pt_file else ''
        }
        
        # 定义工作表名称
        sheet_name = 'DiffDynamic_Generation_Time'
        
        # 检查Excel文件是否存在
        if POCKET_TIME_EXCEL.exists():
            try:
                # 读取所有工作表
                xl = pd.ExcelFile(str(POCKET_TIME_EXCEL), engine='openpyxl')
                sheet_names = xl.sheet_names
                
                if sheet_name in sheet_names:
                    # 读取现有工作表
                    df = pd.read_excel(str(POCKET_TIME_EXCEL), sheet_name=sheet_name, engine='openpyxl')
                else:
                    # 工作表不存在，创建新的DataFrame
                    df = pd.DataFrame(columns=['Pocket_ID', 'Data_ID', 'GPU_ID', 'Generation_Time(s)', 
                                              'Status', 'Timestamp', 'PT_File'])
                
                # 读取其他工作表以便后续写入
                other_sheets = {}
                for sn in sheet_names:
                    if sn != sheet_name:
                        other_sheets[sn] = pd.read_excel(str(POCKET_TIME_EXCEL), sheet_name=sn, engine='openpyxl')
                
            except Exception as e:
                print(f"[记录] ⚠️  读取Excel失败，创建新工作表: {e}")
                df = pd.DataFrame(columns=['Pocket_ID', 'Data_ID', 'GPU_ID', 'Generation_Time(s)', 
                                          'Status', 'Timestamp', 'PT_File'])
                other_sheets = {}
        else:
            # 文件不存在，创建新的DataFrame
            df = pd.DataFrame(columns=['Pocket_ID', 'Data_ID', 'GPU_ID', 'Generation_Time(s)', 
                                      'Status', 'Timestamp', 'PT_File'])
            other_sheets = {}
        
        # 检查是否已存在相同的Pocket_ID记录
        existing_idx = df[df['Pocket_ID'] == f'pocket_{data_id}'].index
        if len(existing_idx) > 0:
            # 更新现有记录
            df.loc[existing_idx[0]] = new_row
            action = "更新"
        else:
            # 添加新记录
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            action = "添加"
        
        print(f"[记录] {action}口袋 {data_id} 的生成时间记录: {generation_time:.2f}s (GPU {gpu_id}, 状态: {status})")
        
        # 保存到Excel（保留其他工作表）
        try:
            # 创建工作表字典
            sheets_to_write = {sheet_name: df}
            sheets_to_write.update(other_sheets)
            
            # 写入所有工作表
            with pd.ExcelWriter(str(POCKET_TIME_EXCEL), engine='openpyxl') as writer:
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
    if gpu_str.lower() == 'all':
        # 自动检测所有可用GPU
        if torch is not None and torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            return list(range(num_gpus))
        else:
            print("⚠️  警告: CUDA不可用，无法自动检测GPU")
            return []
    
    gpu_ids = []
    parts = gpu_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # 范围格式，如 "0-3"
            start, end = part.split('-')
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
        print(f'❌ 错误: 非法 GPU ID（负值）: {bad_neg}')
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
                    f'⚠️  GPU ID {g} 在当前 CUDA 视图非法（仅存 0～{n-1}）；'
                    f'按「恰好 n 张卡却写成 n」处理，改用 {tgt}。'
                    f'若需使用宿主上更大 PCIe 序号之卡，请调整 Docker `--gpus` / NVIDIA_VISIBLE_DEVICES。'
                )
                out_seen.append(tgt)
        else:
            print(
                f'❌ 错误: GPU ID {g} 超出可见范围（当前仅 0～{n-1}）。'
                ' 请修改 --gpus 或为容器挂载该设备。'
            )
            sys.exit(1)
    return out_seen


# Excel写入锁（将在main函数中通过Manager创建，用于进程间共享）
excel_write_lock = None

# 项目根目录
REPO_ROOT = Path(__file__).parent
SCRIPT = REPO_ROOT / 'scripts' / 'sample_diffusion.py'
CONFIG = REPO_ROOT / 'configs' / 'sampling.yml'
EVAL_SCRIPT = REPO_ROOT / 'evaluate_pt_with_correct_reconstruct.py'
JSD_EVAL_SCRIPT = REPO_ROOT / 'eval_jsd_only_100pockets.py'
OUTPUT_DIR = REPO_ROOT / 'outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

_env_pocket_xlsx = os.environ.get('POCKET_TIME_EXCEL')
POCKET_TIME_EXCEL = (
    Path(_env_pocket_xlsx).expanduser()
    if _env_pocket_xlsx
    else REPO_ROOT / 'model_comparison_data.xlsx'
)


def sample_subprocess_timeout_seconds(config_file):
    """
    采样子进程超时（秒）。
    optimization.enable 或 scaffold.enable 时默认 8 小时（scaffold evolve 多代进化耗时长），
    否则 1 小时。环境变量 SAMPLE_SUBPROCESS_TIMEOUT 始终优先（若可解析为浮点数）。
    """
    try:
        env = os.environ.get('SAMPLE_SUBPROCESS_TIMEOUT')
        if env is not None and str(env).strip() != '':
            return float(env)
    except ValueError:
        pass
    default_long = 28800.0
    default_short = 3600.0
    if yaml is None or not config_file:
        return default_short
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        sample_cfg = cfg.get('sample', {}) if cfg else {}
        if sample_cfg.get('optimization', {}).get('enable'):
            return default_long
        if sample_cfg.get('scaffold', {}).get('enable'):
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
        return str_value.replace('.', 'p')
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
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if not config:
            return ""
        
        params = []
        
        # 梯度融合参数
        if 'model' in config and 'grad_fusion_lambda' in config['model']:
            gf_config = config['model']['grad_fusion_lambda']
            mode = gf_config.get('mode', 'unknown')
            start = gf_config.get('start', 0)
            end = gf_config.get('end', 0)
            params.append(f"gf{mode}_{start}_{end}")
        
        # 时间边界
        if 'sample' in config and 'dynamic' in config['sample']:
            dynamic_config = config['sample']['dynamic']
            time_boundary = dynamic_config.get('time_boundary', 0)
            params.append(f"tl{time_boundary}")
            
            # 大步探索阶段参数（根据schedule模式动态调整）
            if 'large_step' in dynamic_config:
                ls_config = dynamic_config['large_step']
                schedule = ls_config.get('schedule', 'unknown')
                
                # 获取step_size和noise_scale参数
                ls_step_size = ls_config.get('step_size', 1.0)
                ls_noise_scale = ls_config.get('noise_scale', 0.0)
                ls_step_size_str = format_float_for_filename(ls_step_size)
                ls_noise_scale_str = format_float_for_filename(ls_noise_scale)
                
                if schedule == 'lambda':
                    # Lambda调度模式
                    ls_a = ls_config.get('lambda_coeff_a', 0.0)
                    ls_b = ls_config.get('lambda_coeff_b', 0.0)
                    ls_a_str = format_float_for_filename(ls_a)
                    ls_b_str = format_float_for_filename(ls_b)
                    params.append(f"lslambda_{ls_a_str}_{ls_b_str}_lsstep_{ls_step_size_str}_lsnoise_{ls_noise_scale_str}")
                elif schedule == 'linear':
                    # 线性调度模式
                    ls_lower = ls_config.get('linear_step_lower', 0)
                    ls_upper = ls_config.get('linear_step_upper', 0)
                    params.append(f"lslinear_{ls_lower}_{ls_upper}_lsstep_{ls_step_size_str}_lsnoise_{ls_noise_scale_str}")
                elif schedule == 'fixed':
                    # 固定步长模式
                    ls_stride = ls_config.get('stride', 0)
                    params.append(f"lsfixed_{ls_stride}_lsstep_{ls_step_size_str}_lsnoise_{ls_noise_scale_str}")
                else:
                    # 其他模式，使用stride作为标识
                    ls_stride = ls_config.get('stride', 0)
                    schedule_safe = str(schedule).replace('-', '_').replace(' ', '_')
                    params.append(f"ls{schedule_safe}_{ls_stride}_lsstep_{ls_step_size_str}_lsnoise_{ls_noise_scale_str}")
            
            # 精炼阶段参数（根据schedule模式动态调整）
            if 'refine' in dynamic_config:
                rf_config = dynamic_config['refine']
                schedule = rf_config.get('schedule', 'unknown')
                
                # 获取step_size和noise_scale参数
                rf_step_size = rf_config.get('step_size', 0.2)
                rf_noise_scale = rf_config.get('noise_scale', 0.05)
                rf_step_size_str = format_float_for_filename(rf_step_size)
                rf_noise_scale_str = format_float_for_filename(rf_noise_scale)
                
                if schedule == 'lambda':
                    # Lambda调度模式
                    rf_a = rf_config.get('lambda_coeff_a', 0.0)
                    rf_b = rf_config.get('lambda_coeff_b', 0.0)
                    rf_a_str = format_float_for_filename(rf_a)
                    rf_b_str = format_float_for_filename(rf_b)
                    params.append(f"rflambda_{rf_a_str}_{rf_b_str}_rfstep_{rf_step_size_str}_rfnoise_{rf_noise_scale_str}")
                elif schedule == 'linear':
                    # 线性调度模式
                    rf_lower = rf_config.get('linear_step_lower', 0)
                    rf_upper = rf_config.get('linear_step_upper', 0)
                    params.append(f"rflinear_{rf_lower}_{rf_upper}_rfstep_{rf_step_size_str}_rfnoise_{rf_noise_scale_str}")
                elif schedule == 'fixed':
                    # 固定步长模式
                    rf_stride = rf_config.get('stride', 0)
                    params.append(f"rffixed_{rf_stride}_rfstep_{rf_step_size_str}_rfnoise_{rf_noise_scale_str}")
                else:
                    # 其他模式，使用stride作为标识
                    rf_stride = rf_config.get('stride', 0)
                    schedule_safe = str(schedule).replace('-', '_').replace(' ', '_')
                    params.append(f"rf{schedule_safe}_{rf_stride}_rfstep_{rf_step_size_str}_rfnoise_{rf_noise_scale_str}")
        
        return "_".join(params)
        
    except Exception as e:
        print(f"⚠️  读取配置文件参数失败: {e}")
        return ""


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
        output_dir = OUTPUT_DIR
    
    # 查找所有匹配的.pt文件（格式：result_{data_id}_*.pt）
    pattern = str(output_dir / f'result_{data_id}_*.pt')
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
    m = re.search(r'Results saved to:\s*(.+?)(?:\r?\n|$)', subprocess_text)
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
    if stem.startswith('result_'):
        parts = stem.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            return parts[1]
    # docs/10/IPDiff/sample.pt → 10
    try:
        if path.parent.parent.name.isdigit():
            return path.parent.parent.name
    except (IndexError, AttributeError):
        pass
    return 'unknown'


def _yaml_prudent_enabled(config_path):
    """sampling.yml 是否启用 Prudent（mode=prudent 或 dynamic.prudent.enable）。"""
    if yaml is None or config_path is None:
        return False
    p = Path(config_path)
    if not p.is_file():
        return False
    try:
        with open(p, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        if not cfg:
            return False
        sample_cfg = cfg.get('sample', {}) or {}
        if sample_cfg.get('mode') == 'prudent':
            return True
        dyn = sample_cfg.get('dynamic', {}) or {}
        pr = dyn.get('prudent', {}) or {}
        return bool(pr.get('enable'))
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
            obj = torch.load(str(p), map_location='cpu', weights_only=False)
        except TypeError:
            obj = torch.load(str(p), map_location='cpu')
        if not isinstance(obj, dict):
            return False
        if obj.get('mode') == 'prudent':
            return True
        meta = obj.get('meta')
        return bool(meta is not None and isinstance(meta, dict) and meta.get('prudent'))
    except Exception:
        return False


def resolve_eval_vina_modes(cli_value, *, config_path=None, pt_path=None):
    """得出传给 evaluate_pt 的 --vina-modes 字符串。

    ``auto``：若 .pt 显式 Prudent 或 YAML 启用 Prudent → ``none``；否则不传参（沿用 evaluate 默认三种 Vina）。
    ``default`` / ``full``：不传参。
    ``none`` / ``off`` / …：``none``。
    其余：原样传给 evaluate（如 ``score_only``、``dock,minimize``）。
    """
    raw = (cli_value if cli_value is not None else 'auto')
    if raw is None or str(raw).strip() == '':
        raw = 'auto'
    v = str(raw).strip().lower()
    if v == 'auto':
        if pt_path is not None:
            pt_p = Path(pt_path)
            if pt_p.is_file() and _pt_prudent_hint(pt_p):
                return 'none'
        if config_path is not None and _yaml_prudent_enabled(config_path):
            return 'none'
        return None
    if v in ('default', 'full', 'all', 'triple'):
        return None
    if v in ('none', 'off', 'skip', 'no_vina'):
        return 'none'
    return str(raw).strip()


def _pick_result_pt_standard_layout(base_dir: Path, pocket_id: int):
    """在目录顶层查找 result_{id}.pt 或最新的 result_{id}_*.pt。"""
    if not base_dir.is_dir():
        return None
    pid = str(int(pocket_id))
    exact = base_dir / f'result_{pid}.pt'
    if exact.is_file():
        return exact
    candidates = list(base_dir.glob(f'result_{pid}_*.pt'))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _pick_glintdm_bench_pt(base_dir: Path, pocket_id: int):
    """GlintDM 基准命名：..._<pocket_id>.pt"""
    if not base_dir.is_dir():
        return None
    suffix = f'_{int(pocket_id)}.pt'
    for p in base_dir.glob('*.pt'):
        if p.is_file() and p.name.endswith(suffix):
            return p
    return None


def _pick_decompdiff_bench_pt(base_dir: Path, pocket_id: int):
    """DecompDiff：sampling_drift_{idx:03d}_*/result.pt"""
    if not base_dir.is_dir():
        return None
    pat = f'sampling_drift_{int(pocket_id):03d}_*'
    dirs = sorted([d for d in base_dir.glob(pat) if d.is_dir()])
    if not dirs:
        return None
    cand = dirs[0] / 'result.pt'
    return cand if cand.is_file() else None


def _resolve_protein_root_for_eval_fallback(protein_root_arg):
    """
    解析 --protein_root：若路径不存在则按单独评估模式尝试常见数据目录。
    未传入时亦尝试自动选用首个存在的回退路径。
    """
    fallback_paths = [
        REPO_ROOT / 'data' / 'crossdocked_v1.1_rmsd1.0_pocket10',
        REPO_ROOT / 'data' / 'crossdocked_v1.1_rmsd1.0',
        Path('/workspace/data/crossdocked_v1.1_rmsd1.0_pocket10'),
        Path('/workspace/data'),
        REPO_ROOT,
    ]
    if protein_root_arg:
        pr = Path(protein_root_arg).expanduser()
        if pr.exists():
            return pr
        orig = pr
        for fb in fallback_paths:
            if fb.exists():
                print(f"💡 --protein_root {orig} 不存在，使用回退路径: {fb}")
                return fb
        return None
    for fb in fallback_paths:
        if fb.exists():
            print(f"💡 未指定 --protein_root，自动使用: {fb}")
            return fb
    return None


def _preflight_import_torch_scatter():
    """evaluate 依赖 datasets → torch_scatter；缺失时返回 False。"""
    try:
        import torch_scatter  # noqa: F401
        return True
    except ImportError:
        return False


def default_benchmark_collect_backend_dirs():
    """各基线/对比方法 outputs 根目录（相对本仓库）。"""
    return [
        ('TargetDiff', REPO_ROOT / 'targetdiff-main' / 'outputs'),
        ('JSDPT3010', REPO_ROOT / 'jsdpt3010'),
        ('DiffSBDD', REPO_ROOT / 'third_party' / 'DiffSBDD' / 'outputs_diffsbdd_bench_100p'),
        ('DecompDiff', REPO_ROOT / 'third_party' / 'DecompDiff' / 'outputs_decompdiff_bench_100p'),
        ('MolForm', REPO_ROOT / 'third_party' / 'MolForm' / 'outputs_molform_bench_100p'),
        ('IPDiff', REPO_ROOT / 'third_party' / 'IPDiff' / 'outputs_ipdiff_bench_100p'),
        ('GlintDM', REPO_ROOT / 'glintdm' / 'outputs_glintdm_bench_100p'),
    ]


def _validate_pt_compatible(pt_path):
    """
    检查 .pt 文件格式，返回格式类型和状态。
    支持格式:
    - 'targetdiff': dict 含 'pred_ligand_pos', 'pred_ligand_v'（与主仓库兼容）
    - 'decompdiff': list 含 'pred_pos', 'pred_v', 'ligand_filename'
    Returns: (format_type, info_msg)  format_type: 'targetdiff'|'decompdiff'|None
    """
    try:
        if torch is None:
            return None, "torch 未安装"
        data = torch.load(str(pt_path), map_location='cpu')

        # TargetDiff / JSDPT 格式
        if isinstance(data, dict):
            required_keys = ['pred_ligand_pos', 'pred_ligand_v', 'data']
            if all(k in data for k in required_keys):
                return 'targetdiff', "TargetDiff/JSDPT 格式"
            return None, f"字段缺失: {data.keys()}"

        # DecompDiff 格式: list 且元素含 pred_pos/pred_v
        if isinstance(data, list) and len(data) > 0:
            sample = data[0]
            if isinstance(sample, dict) and 'pred_pos' in sample and 'pred_v' in sample:
                return 'decompdiff', "DecompDiff 格式"
            return None, f"未知 list 格式: {list(sample.keys()) if isinstance(sample, dict) else type(sample)}"

        return None, f"未知根类型: {type(data).__name__}"
    except Exception as e:
        return None, f"加载失败: {e}"


def _get_protein_path_from_index(pocket_id, protein_root):
    """从 CrossDocked index.pkl 获取指定 data_id 的蛋白口袋 PDB 路径。"""
    try:
        index_path = protein_root / "index.pkl"
        if not index_path.exists():
            return None
        with open(index_path, "rb") as f:
            index = pickle.load(f)
        if not isinstance(index, list) or pocket_id >= len(index):
            return None
        entry = index[pocket_id]
        protein_rel = entry[0] if isinstance(entry, (list, tuple)) else entry.get("protein")
        protein_abs = protein_root / protein_rel
        return protein_abs if protein_abs.exists() else None
    except Exception:
        return None


def process_sdf_evaluation_task(args_tuple):
    """
    处理单个 SDF 文件的对接评估任务（用于 DiffSBDD 等 SDF-only 基线）。

    Args:
        args_tuple: (data_id, sdf_file, protein_root, atom_mode, exhaustiveness,
            batch_start_time, excel_lock, cores_per_task[, eval_show_output])

    Returns:
        tuple: (data_id, success, message, sdf_file, eval_output_dir)
    """
    if len(args_tuple) >= 9:
        (data_id, sdf_file, protein_root, atom_mode, exhaustiveness,
         batch_start_time, excel_lock, cores_per_task, eval_show_output) = args_tuple[:9]
    else:
        (data_id, sdf_file, protein_root, atom_mode, exhaustiveness,
         batch_start_time, excel_lock, cores_per_task) = args_tuple[:8]
        eval_show_output = False

    from multiprocessing import current_process
    import subprocess
    import time

    task_start_time = time.time()
    sdf_path = Path(sdf_file)

    # 从 index.pkl 查找蛋白路径
    protein_path = _get_protein_path_from_index(int(data_id), Path(protein_root))
    if protein_path is None:
        return (data_id, False, f"无法从 index.pkl 找到 data_id={data_id} 的蛋白路径", str(sdf_path), None)

    # 评估输出目录
    cst = timezone(timedelta(hours=8))
    eval_timestamp = datetime.now(cst).strftime('%Y%m%d_%H%M%S')
    eval_output_dir = sdf_path.parent / f'eval_{data_id}_{eval_timestamp}'
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # 限制 CPU 核心数
    env = os.environ.copy()
    if cores_per_task >= 1:
        cores_str = str(cores_per_task)
        env['OMP_NUM_THREADS'] = cores_str
        env['MKL_NUM_THREADS'] = cores_str
        env['OPENBLAS_NUM_THREADS'] = cores_str
        env['NUMEXPR_NUM_THREADS'] = cores_str
        env['VECLIB_MAXIMUM_THREADS'] = cores_str

    try:
        print(f"[SDF评估] 开始对接: {sdf_path.name} (data_id={data_id}) 蛋白: {protein_path.name}")

        # 使用 smina 进行对接（直接命令行调用）
        # 首先将蛋白 PDB 转换为 PDBQT（如果还没有）
        protein_pdbqt = eval_output_dir / f"{protein_path.stem}.pdbqt"
        if not protein_pdbqt.exists():
            # Open Babel 2.x: -O 与输出文件名之间不能有空格
            prep_cmd = f"obabel {protein_path} -O{protein_pdbqt} 2>/dev/null || true"
            subprocess.run(prep_cmd, shell=True, env=env, timeout=300)

        # 获取 SDF 中的分子数量并逐个对接
        from rdkit import Chem
        suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
        scores = []

        for i, mol in enumerate(suppl):
            if mol is None:
                continue
            # 计算分子中心
            conf = mol.GetConformer()
            cx = sum(conf.GetAtomPosition(j).x for j in range(mol.GetNumAtoms())) / mol.GetNumAtoms()
            cy = sum(conf.GetAtomPosition(j).y for j in range(mol.GetNumAtoms())) / mol.GetNumAtoms()
            cz = sum(conf.GetAtomPosition(j).z for j in range(mol.GetNumAtoms())) / mol.GetNumAtoms()

            # 单个分子的 SDF 和输出
            mol_sdf = eval_output_dir / f"mol_{i}.sdf"
            mol_pdbqt = eval_output_dir / f"mol_{i}.pdbqt"
            out_sdf = eval_output_dir / f"mol_{i}_out.sdf"

            # 保存单个分子
            writer = Chem.SDWriter(str(mol_sdf))
            writer.write(mol)
            writer.close()

            # 转换为 PDBQT（-O 与输出路径无空格）
            obabel_cmd = f"obabel {mol_sdf} -O{mol_pdbqt} 2>/dev/null"
            subprocess.run(obabel_cmd, shell=True, env=env, timeout=60)

            if not mol_pdbqt.exists():
                continue

            # 运行 smina 对接
            smina_bin = shutil.which('smina.static') or shutil.which('smina') or 'smina'
            smina_cmd = [
                smina_bin,
                '-r', str(protein_pdbqt) if protein_pdbqt.exists() else str(protein_path),
                '-l', str(mol_pdbqt),
                '--center_x', str(cx),
                '--center_y', str(cy),
                '--center_z', str(cz),
                '--size_x', '20',
                '--size_y', '20',
                '--size_z', '20',
                '--exhaustiveness', str(exhaustiveness),
                '--score_only',  # 只打分，不输出构象，更快
            ]

            try:
                smina_result = subprocess.run(smina_cmd, capture_output=True, text=True, env=env, timeout=300)
                # 解析 Affinity
                for line in smina_result.stdout.split('\n'):
                    if 'Affinity:' in line:
                        try:
                            score = float(line.split()[1])
                            scores.append(score)
                            break
                        except (ValueError, IndexError):
                            pass
            except Exception:
                pass

        # 保存结果到文件
        if scores:
            import numpy as np
            results_summary = {
                'scores': scores,
                'mean': np.mean(scores),
                'median': np.median(scores),
                'min': np.min(scores),
                'num_molecules': len(scores)
            }
            import json
            with open(eval_output_dir / 'smina_scores.json', 'w') as f:
                json.dump(results_summary, f, indent=2)

        elapsed = time.time() - task_start_time
        print(f"[SDF评估] ✅ 成功 (data_id={data_id}) 耗时 {elapsed:.1f}s, 成功对接 {len(scores)} 个分子")
        return (data_id, True, f"SDF 对接成功 ({len(scores)} 个分子)", str(sdf_path), str(eval_output_dir))
    except Exception as e:
        error_msg = f"SDF 评估异常: {str(e)}"
        print(f"[SDF评估] ❌ 异常 (data_id={data_id}): {error_msg}")
        import traceback
        traceback.print_exc()
        return (data_id, False, error_msg, str(sdf_path), None)


def process_decompdiff_pt_evaluation_task(args_tuple):
    """
    处理 DecompDiff 格式 .pt 文件的评估任务。
    调用 DecompDiff 的 evaluate_mol_from_meta_full.py 脚本进行评估。

    Args:
        args_tuple: (data_id, pt_file, protein_root, atom_mode, exhaustiveness,
            batch_start_time, excel_lock, cores_per_task[, eval_show_output])

    Returns:
        tuple: (data_id, success, message, pt_file, eval_output_dir)
    """
    if len(args_tuple) >= 9:
        (data_id, pt_file, protein_root, atom_mode, exhaustiveness,
         batch_start_time, excel_lock, cores_per_task, eval_show_output) = args_tuple[:9]
    else:
        (data_id, pt_file, protein_root, atom_mode, exhaustiveness,
         batch_start_time, excel_lock, cores_per_task) = args_tuple[:8]
        eval_show_output = False

    import subprocess
    import time

    task_start_time = time.time()
    pt_path = Path(pt_file)

    # 评估输出目录
    cst = timezone(timedelta(hours=8))
    eval_timestamp = datetime.now(cst).strftime('%Y%m%d_%H%M%S')
    eval_output_dir = pt_path.parent / f'eval_{data_id}_{eval_timestamp}'
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # DecompDiff 评估脚本路径（使用绝对路径）
    decompdiff_dir = REPO_ROOT / 'third_party' / 'DecompDiff'
    decompdiff_dir_abs = decompdiff_dir.resolve()
    decompdiff_eval_script = decompdiff_dir_abs / 'scripts' / 'evaluate_mol_from_meta_full.py'

    if not decompdiff_eval_script.is_file():
        return (data_id, False, f"DecompDiff 评估脚本不存在: {decompdiff_eval_script}", str(pt_path), None)

    # 限制 CPU 核心数
    env = os.environ.copy()
    if cores_per_task >= 1:
        cores_str = str(cores_per_task)
        env['OMP_NUM_THREADS'] = cores_str
        env['MKL_NUM_THREADS'] = cores_str
        env['OPENBLAS_NUM_THREADS'] = cores_str
        env['NUMEXPR_NUM_THREADS'] = cores_str
        env['VECLIB_MAXIMUM_THREADS'] = cores_str

    # 仅保留 DecompDiff 根目录，避免继承父进程的 PYTHONPATH（主仓库 /workspace 的 utils 会遮蔽 DecompDiff）
    env['PYTHONPATH'] = str(decompdiff_dir_abs)

    # 构建命令：使用 -c 执行，在代码中设置 sys.path 和 sys.argv，然后用 runpy 运行脚本
    pt_path_abs = Path(pt_path).resolve()
    protein_root_abs = Path(protein_root).resolve()
    eval_output_dir_abs = Path(eval_output_dir).resolve()
    script_path_abs = decompdiff_eval_script.resolve()
    repo_root_abs = REPO_ROOT.resolve()

    # 从 sys.path 去掉主仓库根，防止 import utils 命中 DiffDynamic 的 utils 包
    dd_s = str(decompdiff_dir_abs)
    repo_s = str(repo_root_abs)
    script_s = str(script_path_abs)
    pt_s = str(pt_path_abs)
    pr_s = str(protein_root_abs)
    out_s = str(eval_output_dir_abs)
    exh_s = str(exhaustiveness)

    setup_code = (
        "import sys, runpy; "
        f"DD={repr(dd_s)}; REPO={repr(repo_s)}; "
        "sys.path=[DD]+[p for p in sys.path if p and p != DD and p != REPO]; "
        f"sys.argv=[{repr(script_s)}, {repr(pt_s)}, '--protein_root', {repr(pr_s)}, "
        "'--docking_mode', 'vina_score', "
        f"'--exhaustiveness', {repr(exh_s)}, "
        "'--num_workers', '1', "
        f"'--result_path', {repr(out_s)}, '--aggregate_meta', 'False']; "
        f"runpy.run_path({repr(script_s)}, run_name='__main__')"
    )

    cmd = [sys.executable, '-c', setup_code]

    try:
        print(f"[DecompDiff评估] 开始评估: {pt_path.name} (data_id={data_id})")
        print(f"[DecompDiff评估] 工作目录: {decompdiff_dir_abs}")
        if eval_show_output:
            result = subprocess.run(cmd, check=True, env=env, cwd=str(decompdiff_dir_abs), timeout=21600)
        else:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env, cwd=str(decompdiff_dir_abs), timeout=21600)

        elapsed = time.time() - task_start_time
        print(f"[DecompDiff评估] ✅ 成功 (data_id={data_id}) 耗时 {elapsed:.1f}s")
        return (data_id, True, "DecompDiff 评估成功", str(pt_path), str(eval_output_dir))
    except subprocess.CalledProcessError as e:
        error_msg = f"DecompDiff 评估失败: {e.stderr[-500:] if e.stderr else str(e)}"
        print(f"[DecompDiff评估] ❌ 失败 (data_id={data_id}): {error_msg}")
        return (data_id, False, error_msg, str(pt_path), None)
    except Exception as e:
        error_msg = f"DecompDiff 评估异常: {str(e)}"
        print(f"[DecompDiff评估] ❌ 异常 (data_id={data_id}): {error_msg}")
        return (data_id, False, error_msg, str(pt_path), None)


def collect_benchmark_pocket_and_eval(
    pocket_id,
    protein_root,
    docs_parent=None,
    atom_mode='add_aromatic',
    exhaustiveness=8,
    num_cpu_cores=64,
    cores_per_task=1,
    eval_show_output=False,
    eval_save_log=False,
    force_mmff_minimize=False,
    mmff_max_iters=None,
    include_methods=None,
):
    """
    将各基线 outputs 中该测试口袋的 .pt 复制到 docs/<口袋编号>/<方法>/pt/（或 SDF 到 sdf/），
    再对每个已复制的 .pt/.sdf 调用本仓库 evaluate 流程（在对应目录下生成 eval_* 及 Excel）。

    Args:
        include_methods: 要处理的方法列表，如 ['TargetDiff', 'JSDPT3010', 'DiffSBDD']。
                        None 表示处理所有方法。

    Returns:
        目标根目录 Path
    """
    if not EVAL_SCRIPT.is_file():
        print(f"❌ 错误: 评估脚本不存在: {EVAL_SCRIPT}")
        sys.exit(1)

    if not _preflight_import_torch_scatter():
        print(
            "❌ 错误: 当前 Python 环境未安装 `torch_scatter`，evaluate 子进程会在 import 阶段失败。\n"
            "   修复（容器内需已激活 conda 环境，如 diffdynamic）：\n"
            "   • 推荐：bash scripts/repair_pyg_for_current_torch.sh\n"
            "     （按当前 torch 版本从 https://data.pyg.org/whl/ 安装匹配的 pyg-lib / torch-scatter 等）\n"
            "   • 或见仓库 初始化.md 中「torch_scatter / PyG」与 pip 安装说明。\n"
            "   安装成功后请重新运行本命令。"
        )
        sys.exit(1)

    docs_parent = docs_parent or (REPO_ROOT / 'docs')
    dest_root = (docs_parent / str(int(pocket_id))).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    protein_root = Path(protein_root).resolve()

    backends = default_benchmark_collect_backend_dirs()

    # 方法筛选
    if include_methods is not None:
        allowed = set(include_methods)
        backends = [(lbl, pth) for lbl, pth in backends if lbl in allowed]
        if not backends:
            print(f"❌ 错误: --methods 指定了无效的方法。可用: {[_lb for _lb, _ in default_benchmark_collect_backend_dirs()]}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f'基准口袋对比：data_id = {int(pocket_id)}（复制 .pt/.sdf → docs 下并启动评估）')
    if include_methods:
        print(f'本次选择的方法: {include_methods}')
    print(f'目标目录: {dest_root}')
    print(f'protein_root: {protein_root}')
    print(f"{'='*60}\n")

    targetdiff_eval_jobs = []  # (label, dest_pt) 用于 TargetDiff/JSDPT 格式
    decompdiff_eval_jobs = []  # (label, dest_pt) 用于 DecompDiff 格式
    sdf_eval_jobs = []         # (label, dest_sdf) 用于 SDF 评估

    for label, root in backends:
        # 先尝试找 .pt
        pt_dst_dir = dest_root / label / 'pt'
        if label == 'DecompDiff':
            src_pt = _pick_decompdiff_bench_pt(root, pocket_id)
        elif label == 'GlintDM':
            src_pt = _pick_glintdm_bench_pt(root, pocket_id)
        else:
            src_pt = _pick_result_pt_standard_layout(root, pocket_id)

        if src_pt is not None:
            pt_dst_dir.mkdir(parents=True, exist_ok=True)
            dest_pt = pt_dst_dir / src_pt.name
            shutil.copy2(src_pt, dest_pt)
            # 验证 .pt 格式
            fmt, info = _validate_pt_compatible(dest_pt)
            if fmt == 'targetdiff':
                print(f'  [{label}] 已复制 .pt (TargetDiff格式): {dest_pt.name} <- {src_pt}')
                targetdiff_eval_jobs.append((label, dest_pt))
            elif fmt == 'decompdiff':
                print(f'  [{label}] 已复制 .pt (DecompDiff格式): {dest_pt.name} <- {src_pt}')
                decompdiff_eval_jobs.append((label, dest_pt))
            else:
                print(f'  [{label}] 已复制 .pt (未知格式，跳过): {dest_pt.name} <- {src_pt}')
                print(f'           原因: {info}')
            continue

        # 无 .pt 时，对 DiffSBDD 尝试找 SDF
        if label == 'DiffSBDD' and root.is_dir():
            sdf_candidates = list(root.glob(f'diffsbdd_pocket_{int(pocket_id)}.sdf'))
            if sdf_candidates:
                sdf_dst_dir = dest_root / label / 'sdf'
                sdf_dst_dir.mkdir(parents=True, exist_ok=True)
                src_sdf = sdf_candidates[0]
                dest_sdf = sdf_dst_dir / src_sdf.name
                shutil.copy2(src_sdf, dest_sdf)
                print(f'  [{label}] 已复制 SDF: {dest_sdf.name} <- {src_sdf}')
                sdf_eval_jobs.append((label, dest_sdf))
                continue

        print(f'  [{label}] 跳过：未找到 .pt 或 SDF（{root}）')

    total_jobs = len(targetdiff_eval_jobs) + len(decompdiff_eval_jobs) + len(sdf_eval_jobs)
    if total_jobs == 0:
        print('\n❌ 未找到任何可评估的 .pt 或 SDF，退出\n')
        sys.exit(1)

    print(f"\n{'='*60}")
    print(
        f'开始并行评估：{len(targetdiff_eval_jobs)} 个 TargetDiff + '
        f'{len(decompdiff_eval_jobs)} 个 DecompDiff + '
        f'{len(sdf_eval_jobs)} 个 SDF = 共 {total_jobs} 路'
    )
    max_parallel = max(1, int(num_cpu_cores) // max(1, int(cores_per_task)))
    workers = min(max_parallel, total_jobs)
    print(
        f'CPU 并行: num_cpu_cores={num_cpu_cores}, cores_per_task={cores_per_task} '
        f'→ 理论最多 {max_parallel} 路并行，实际启动 {workers} 路'
    )
    print(f"{'='*60}\n")

    manager = Manager()
    excel_lock = manager.Lock()
    batch_start_time = time.time()
    pid_int = int(pocket_id)

    # 构建 TargetDiff 格式 .pt 评估任务
    targetdiff_task_tuples = [
        (
            pid_int,
            str(dest_pt),
            str(protein_root),
            atom_mode,
            exhaustiveness,
            None,
            batch_start_time,
            excel_lock,
            max(1, int(cores_per_task)),
            0,
            eval_show_output,
            eval_save_log,
            force_mmff_minimize,
            mmff_max_iters,
        )
        for _label, dest_pt in targetdiff_eval_jobs
    ]

    # 构建 DecompDiff 格式 .pt 评估任务
    decompdiff_task_tuples = [
        (
            pid_int,
            str(dest_pt),
            str(protein_root),
            atom_mode,
            exhaustiveness,
            batch_start_time,
            excel_lock,
            max(1, int(cores_per_task)),
            eval_show_output,
        )
        for _label, dest_pt in decompdiff_eval_jobs
    ]

    # 构建 SDF 评估任务
    sdf_task_tuples = [
        (
            pid_int,
            str(dest_sdf),
            str(protein_root),
            atom_mode,
            exhaustiveness,
            batch_start_time,
            excel_lock,
            max(1, int(cores_per_task)),
            eval_show_output,
        )
        for _label, dest_sdf in sdf_eval_jobs
    ]

    # 合并任务并标记类型以便后续拆分结果
    all_tasks = (
        [("targetdiff", t) for t in targetdiff_task_tuples] +
        [("decompdiff", t) for t in decompdiff_task_tuples] +
        [("sdf", t) for t in sdf_task_tuples]
    )

    pool = None
    raw_results = []
    try:
        pool = Pool(processes=workers)
        # 使用统一 map，按类型分发到不同处理函数
        raw_results = pool.map(_dispatch_eval_task, all_tasks)
    finally:
        if pool:
            pool.close()
            pool.join()

    # 拆分结果
    targetdiff_results = [r for r, t in zip(raw_results, all_tasks) if t[0] == "targetdiff"]
    decompdiff_results = [r for r, t in zip(raw_results, all_tasks) if t[0] == "decompdiff"]
    sdf_results = [r for r, t in zip(raw_results, all_tasks) if t[0] == "sdf"]

    # 汇总输出
    ok = 0
    print(f"\n{'='*60}")
    print('各方法评估结果')
    print(f"{'='*60}")

    for (label, _dest_pt), row in zip(targetdiff_eval_jobs, targetdiff_results):
        success = bool(row[1]) if row and len(row) > 1 else False
        msg = (row[2] if row and len(row) > 2 else '') or ''
        if success:
            ok += 1
            print(f'  [{label}] ✅ 成功 (TargetDiff格式)')
        else:
            short = (msg[:400] + '…') if len(msg) > 400 else msg
            print(f'  [{label}] ❌ 失败 (TargetDiff格式): {short}')

    for (label, _dest_pt), row in zip(decompdiff_eval_jobs, decompdiff_results):
        success = bool(row[1]) if row and len(row) > 1 else False
        msg = (row[2] if row and len(row) > 2 else '') or ''
        if success:
            ok += 1
            print(f'  [{label}] ✅ 成功 (DecompDiff格式)')
        else:
            short = (msg[:400] + '…') if len(msg) > 400 else msg
            print(f'  [{label}] ❌ 失败 (DecompDiff格式): {short}')

    for (label, _dest_sdf), row in zip(sdf_eval_jobs, sdf_results):
        success = bool(row[1]) if row and len(row) > 1 else False
        msg = (row[2] if row and len(row) > 2 else '') or ''
        if success:
            ok += 1
            print(f'  [{label}] ✅ 成功 (SDF)')
        else:
            short = (msg[:400] + '…') if len(msg) > 400 else msg
            print(f'  [{label}] ❌ 失败 (SDF): {short}')

    print(f"{'='*60}\n")

    if ok == len(raw_results):
        print(
            f'✅ 全部 {ok}/{len(raw_results)} 路评估成功；输出在 {dest_root} 各方法子目录下的 eval_*。\n'
        )
    else:
        print(
            f'⚠️  评估结束：成功 {ok}/{len(raw_results)}，失败 {len(raw_results) - ok}。\n'
            f'   结果已保存在 {dest_root}；请根据上方失败信息排查。\n'
        )
        sys.exit(1)

    return dest_root


def _dispatch_eval_task(task_with_type):
    """分发任务到对应的处理函数（用于 Pool.map 的统一入口）。"""
    task_type, task_args = task_with_type
    if task_type == "targetdiff":
        return process_evaluation_task(task_args)
    elif task_type == "decompdiff":
        return process_decompdiff_pt_evaluation_task(task_args)
    elif task_type == "sdf":
        return process_sdf_evaluation_task(task_args)
    else:
        raise ValueError(f"未知任务类型: {task_type}")


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
    if pt_filename.startswith('result_'):
        parts = pt_filename.split('_')
        if len(parts) >= 3:
            pocket_id = parts[1]
        else:
            pocket_id = str(data_id)
    else:
        pocket_id = str(data_id)
    
    # 查找评估目录
    eval_dirs = list(outputs_dir.glob(f'eval_{pocket_id}_*'))
    if not eval_dirs:
        eval_dirs = list(outputs_dir.glob('eval_*'))
    
    if not eval_dirs:
        return (False, None, None, 0, f"未找到评估输出目录（在 {outputs_dir} 中，查找模式: eval_{pocket_id}_*）", None)
    
    # 优先选择带时间戳的新格式目录
    timestamp_pattern = r'_\d{8}_\d{6}_'
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
        eval_result_files = list(eval_output_dir.glob('eval_results_*.pt'))
        if eval_result_files:
            break
        time.sleep(2)
    
    if not eval_result_files:
        all_files = list(eval_output_dir.glob('*'))
        file_list = ', '.join([f.name for f in all_files[:10]])
        if len(all_files) > 10:
            file_list += f' ... (共{len(all_files)}个文件)'
        return (False, None, None, 0, 
                f"等待{wait_timeout}秒后仍未找到评估结果文件 (eval_results_*.pt)\n"
                f"   评估目录: {eval_output_dir}\n"
                f"   目录中的文件: {file_list if all_files else '空目录'}", 
                str(eval_output_dir))
    
    try:
        latest_eval_file = max(eval_result_files, key=os.path.getmtime)
        eval_data = torch.load(latest_eval_file, map_location='cpu')
        
        statistics = eval_data.get('statistics', {})
        vina_dock_scores = statistics.get('vina_dock_scores', [])
        vina_score_only_scores = statistics.get('vina_score_only_scores', [])
        vina_minimize_scores = statistics.get('vina_minimize_scores', [])
        vina_scores = statistics.get('vina_scores', [])
        
        if vina_dock_scores:
            vina_scores = vina_dock_scores
        elif vina_minimize_scores:
            vina_scores = vina_minimize_scores
        elif vina_score_only_scores:
            vina_scores = vina_score_only_scores
        
        n_reconstruct_success = eval_data.get('n_reconstruct_success', 0)
        n_eval_success = eval_data.get('n_eval_success', 0)
        
        if not vina_scores:
            diagnostic_msg = f"评估结果中无vina得分"
            if n_reconstruct_success > 0 and n_eval_success == 0:
                diagnostic_msg += f" (重建成功{n_reconstruct_success}个，但对接全部失败)"
            elif n_reconstruct_success == 0:
                diagnostic_msg += f" (重建失败，重建成功数: {n_reconstruct_success})"
            return (False, None, None, 0, diagnostic_msg, str(eval_output_dir))
        
        vina_mean = float(np.mean(vina_scores))
        vina_median = float(np.median(vina_scores))
        num_scores = len(vina_scores)
        
        return (True, vina_mean, vina_median, num_scores, 
                f"成功读取评估结果，得分数量: {num_scores}", str(eval_output_dir))
        
    except Exception as e:
        return (False, None, None, 0, f"读取评估结果异常: {str(e)}", str(eval_output_dir))


def run_single_sample(data_id, config_file, gpu_id, max_retries=3, retry_delay=5, protein_root=None):
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
    if config_file is None:
        config_file = CONFIG
    
    print(f"[GPU {gpu_id}] 开始采样 data_id={data_id} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')})")
    
    # 构建采样命令（指定GPU为cuda:0，因为CUDA_VISIBLE_DEVICES已经在process_sampling_task中设置）
    # 在这个进程中，cuda:0 对应物理GPU gpu_id
    cmd = [
        sys.executable,
        str(SCRIPT),
        str(config_file),
        '--data_id', str(data_id),
        '--device', 'cuda:0'  # 使用cuda:0，因为CUDA_VISIBLE_DEVICES已经在进程级别设置了
    ]
    if protein_root:
        cmd.extend(['--protein_root', str(protein_root)])
    
    # 重试机制
    last_error = None
    for attempt in range(max_retries):
        if attempt > 0:
            print(f"[GPU {gpu_id}] 重试采样 data_id={data_id} (尝试 {attempt + 1}/{max_retries})")
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
                encoding='utf-8',
                errors='replace',  # 处理编码错误，避免因编码问题导致输出截断
                cwd=str(REPO_ROOT),  # 保证 ./pretrained_models、./outputs 等相对路径可解析
                # 不需要显式传递 env：进程已设置 CUDA_VISIBLE_DEVICES，子进程会继承
            )
            
            # 等待一小段时间，确保文件已保存
            time.sleep(1)
            
            combined_log = (result.stdout or '') + '\n' + (result.stderr or '')
            pt_parsed = _parse_results_saved_pt_path(combined_log)
            if pt_parsed is not None and pt_parsed.is_file():
                pt_file = pt_parsed.resolve()
                print(f"[GPU {gpu_id}] ✅ 采样成功: {pt_file}")
                if result.stdout:
                    for line in result.stdout.splitlines()[-40:]:
                        if any(k in line for k in ('large_step', 'candidate', 'pos_list', 'prudent', 'refined', 'warning', 'error', '无候选')):
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
                        if any(k in line for k in ('large_step', 'candidate', 'pos_list', 'prudent', 'refined', 'warning', 'error', '无候选')):
                            print(f"[GPU {gpu_id}] LOG: {line}")
                return True, str(pt_file), "采样成功"
            else:
                print(f"[GPU {gpu_id}] ⚠️  采样完成但未找到结果文件 (data_id={data_id})")
                # 输出stdout和stderr帮助调试
                if result.stdout:
                    print(f"[GPU {gpu_id}] stdout: {result.stdout[-500:]}")  # 最后500字符
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
            error_log_file = OUTPUT_DIR / f"sampling_timeout_{data_id}_{int(time.time())}.log"
            try:
                with open(error_log_file, 'w', encoding='utf-8') as f:
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
            if e.stderr and ('CUBLAS_STATUS_NOT_INITIALIZED' in e.stderr or 
                           'CUDA error' in e.stderr or
                           'cublasCreate' in e.stderr):
                is_cuda_error = True
                print(f"[GPU {gpu_id}] ⚠️  检测到CUDA初始化错误，将重试...")
            
            # 始终保存完整错误日志到文件（即使很短）
            error_log_file = OUTPUT_DIR / f"sampling_error_{data_id}_{int(time.time())}.log"
            try:
                with open(error_log_file, 'w', encoding='utf-8') as f:
                    f.write(f"采样失败 (data_id={data_id}, GPU={gpu_id}, 尝试 {attempt + 1}/{max_retries})\n")
                    f.write(f"命令: {' '.join(cmd)}\n")
                    f.write(f"返回码: {e.returncode}\n")
                    f.write(f"\n完整stdout ({len(e.stdout) if e.stdout else 0} 字符):\n")
                    f.write(f"{e.stdout if e.stdout else '(无)'}\n")
                    f.write(f"\n完整stderr ({len(e.stderr) if e.stderr else 0} 字符):\n")
                    f.write(f"{e.stderr if e.stderr else '(无)'}\n")
                    f.write(f"\n组合错误信息:\n{error_msg}\n")
                if attempt == 0:  # 只在第一次失败时打印
                    print(f"[GPU {gpu_id}] ❌ 采样失败 (data_id={data_id}, 返回码={e.returncode})")
                    print(f"[GPU {gpu_id}] 完整错误日志已保存到: {error_log_file}")
            except Exception as log_err:
                print(f"[GPU {gpu_id}] ⚠️  保存错误日志失败: {log_err}")
            
            # 显示错误摘要（前500字符）
            if attempt == 0:  # 只在第一次失败时显示详细错误
                error_summary = error_msg[:500] if len(error_msg) > 500 else error_msg
                if len(error_msg) > 500:
                    error_summary += f"\n... (共{len(error_msg)}字符，完整信息已保存到日志文件)"
                print(f"[GPU {gpu_id}] 错误摘要:")
                for line in error_summary.split('\n')[:20]:  # 最多显示20行
                    if line.strip():
                        print(f"[GPU {gpu_id}]   {line}")
                
                # 诊断常见问题
                if not e.stdout and not e.stderr:
                    print(f"[GPU {gpu_id}] ⚠️  警告: 没有捕获到任何输出，可能原因:")
                    print(f"[GPU {gpu_id}]   - 进程被系统杀死（OOM killer）")
                    print(f"[GPU {gpu_id}]   - GPU内存不足")
                    print(f"[GPU {gpu_id}]   - 进程启动失败")
                elif e.returncode == -9 or e.returncode == 137:
                    print(f"[GPU {gpu_id}] ⚠️  警告: 进程被信号9（SIGKILL）杀死，通常是OOM killer")
                elif e.returncode == -11 or e.returncode == 139:
                    print(f"[GPU {gpu_id}] ⚠️  警告: 进程段错误（SIGSEGV），可能是内存访问错误")
            
            # 如果是CUDA错误且还有重试机会，继续重试
            if is_cuda_error and attempt < max_retries - 1:
                continue
            else:
                # 重试次数用完或不是可重试的错误，返回失败
                return False, None, f"采样失败 (返回码={e.returncode}): {error_msg[:1000]}"
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
    return False, None, f"采样失败（已重试{max_retries}次）: {last_error[:1000] if last_error else '未知错误'}"


def run_single_sample_custom(protein_path, config_file, gpu_id, ligand_path=None, use_dataset_for_pocket=False, pocket_radius=10.0, max_retries=3, retry_delay=5):
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
    if config_file is None:
        config_file = CONFIG
    
    pocket_id = 'custom'
    print(f"[GPU {gpu_id}] 开始采样（自定义文件）: {protein_path}")
    
    cmd = [
        sys.executable,
        str(SCRIPT),
        str(config_file),
        '--protein_path', str(protein_path),
        '--device', 'cuda:0'
    ]
    if ligand_path:
        cmd.extend(['--ligand_path', str(ligand_path)])
    if use_dataset_for_pocket:
        cmd.append('--use_dataset_for_pocket')
    cmd.extend(['--pocket_radius', str(pocket_radius)])
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    sample_timeout = sample_subprocess_timeout_seconds(config_file)
    
    last_error = None
    for attempt in range(max_retries):
        if attempt > 0:
            print(f"[GPU {gpu_id}] 重试采样（自定义文件）(尝试 {attempt + 1}/{max_retries})")
            time.sleep(retry_delay)
        
        try:
            t_before = time.time()
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=sample_timeout,
                encoding='utf-8',
                errors='replace',
                env=env,
                cwd=str(REPO_ROOT),
            )
            time.sleep(1)
            combined_log = (result.stdout or '') + '\n' + (result.stderr or '')
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
    
    return False, None, f"采样失败（已重试{max_retries}次）: {last_error[:1000] if last_error else '未知错误'}"


def _resolve_final_eval_dir_from_evaluate_subprocess(text, outputs_dir: Path, data_id):
    """
    解析 evaluate_pt_with_correct_reconstruct 打印的「评估输出目录」，
    或在无法解析时选取 outputs_dir 下最新的 eval_*（与 generate_eval_dir_name 一致的单层目录）。
    """
    if text:
        m = re.search(r'评估输出目录:\s*(.+?)\s*$', text, re.MULTILINE)
        if m:
            p = Path(m.group(1).strip())
            if p.is_dir():
                return str(p.resolve())
    # 新格式 eval_YYYYMMDD_HHMMSS_<data_id>_...
    ts_pat = re.compile(
        r'^eval_\d{8}_\d{6}_' + re.escape(str(data_id)) + r'(?:_|$)'
    )
    candidates = [p for p in outputs_dir.iterdir() if p.is_dir() and ts_pat.search(p.name)]
    if not candidates:
        candidates = [p for p in outputs_dir.glob('eval_*') if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(candidates[0].resolve())


def run_single_evaluation(
    pt_file, protein_root, data_id, atom_mode='add_aromatic', exhaustiveness=8,
    cores_per_task=1, save_intermediate_interval=0,
    eval_show_output=False, eval_save_subprocess_log=False,
    force_mmff_minimize=False, mmff_max_iters=None,
    vina_modes=None,
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
    
    Returns:
        tuple: (success, message, eval_output_dir)
    """
    print(f"[评估] 开始评估: {Path(pt_file).name} (data_id={data_id})")
    t0 = time.time()

    pt_path = Path(pt_file)
    # 与 evaluate_pt_with_correct_reconstruct 一致：--output_dir 为 .pt 所在目录（父目录），
    # 由 evaluate 内部 generate_eval_dir_name 只生成单层 eval_时间戳_data_id_配置... 目录，避免 outputs 下出现空壳 eval_*。
    outputs_dir = pt_path.parent.resolve()

    # 使用本地时区（CST，UTC+8）生成时间戳（仅用于子进程日志文件名）
    cst = timezone(timedelta(hours=8))
    eval_timestamp = datetime.now(cst).strftime('%Y%m%d_%H%M%S')

    # 构建评估命令
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        str(pt_file),
        '--protein_root', str(protein_root),
        '--output_dir', str(outputs_dir),
        '--atom_mode', atom_mode,
        '--exhaustiveness', str(exhaustiveness),
        '--save_intermediate_interval', str(save_intermediate_interval)
    ]
    if force_mmff_minimize:
        cmd.append('--force-mmff-minimize')
    if mmff_max_iters is not None:
        cmd.extend(['--mmff-max-iters', str(mmff_max_iters)])
    if vina_modes is not None and str(vina_modes).strip():
        cmd.extend(['--vina-modes', str(vina_modes).strip()])
    if eval_show_output:
        print(f"[评估] [data_id={data_id}] 命令: {' '.join(cmd)}")
    
    # 限制每个子进程的CPU核心数（Vina/NumPy等使用OpenMP/MKL）
    env = os.environ.copy()
    if cores_per_task >= 1:
        cores_str = str(cores_per_task)
        env['OMP_NUM_THREADS'] = cores_str
        env['MKL_NUM_THREADS'] = cores_str
        env['OPENBLAS_NUM_THREADS'] = cores_str
        env['NUMEXPR_NUM_THREADS'] = cores_str
        env['VECLIB_MAXIMUM_THREADS'] = cores_str
    # 单路径单 .pt 评估：与 evaluate_pt_with_correct_reconstruct 中单分子默认上限一致（可被 EVAL_SINGLE_MOL_TIMEOUT 覆盖）
    if 'EVAL_SINGLE_MOL_TIMEOUT' not in env:
        env['EVAL_SINGLE_MOL_TIMEOUT'] = '10800'

    run_kw = dict(check=True, timeout=21600, env=env)
    # 整段 evaluate 子进程上限（6 小时）；单次 Vina 时限在 evaluate 脚本内单独控制

    try:
        captured_stdout = ''
        captured_stderr = ''
        if eval_show_output:
            # 直接继承终端，便于看到 tqdm / 逐分子进度（多进程并行时输出会交错）
            subprocess.run(cmd, **run_kw)
        elif eval_save_subprocess_log:
            log_path = outputs_dir / f'evaluate_subprocess_{data_id}_{eval_timestamp}.log'
            with open(log_path, 'w', encoding='utf-8') as logf:
                cp = subprocess.run(
                    cmd, stdout=logf, stderr=subprocess.STDOUT, text=True, **run_kw
                )
            print(f"[评估] [data_id={data_id}] 子进程日志: {log_path}")
            try:
                captured_stdout = log_path.read_text(encoding='utf-8', errors='replace')
            except OSError:
                captured_stdout = ''
        else:
            # 默认：捕获输出，避免多路并行时终端与 [评估] 进度行混杂
            cp = subprocess.run(cmd, capture_output=True, text=True, **run_kw)
            captured_stdout = cp.stdout or ''
            captured_stderr = cp.stderr or ''

        final_eval_dir = _resolve_final_eval_dir_from_evaluate_subprocess(
            captured_stdout + '\n' + captured_stderr, outputs_dir, data_id
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
            ['nvidia-smi', '--query-gpu=memory.free,memory.total', 
             '--format=csv,noheader,nounits', f'--id={gpu_id}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(',')
            if len(parts) >= 2:
                free_memory_mb = int(parts[0].strip())
                total_memory_mb = int(parts[1].strip())
                has_enough = free_memory_mb >= min_free_memory_mb
                message = f"GPU {gpu_id}: 可用内存 {free_memory_mb}MB / 总计 {total_memory_mb}MB"
                return has_enough, free_memory_mb, total_memory_mb, message
    except Exception as e:
        pass
    
    return None, None, None, f"无法检查GPU {gpu_id}的内存状态"


def _pt_dict_already_baseline_refined(result_dict):
    """判断 .pt 字典是否已含 TargetDiff baseline refine轨迹（避免重复修复）。"""
    if not isinstance(result_dict, dict):
        return False
    meta = result_dict.get('meta')
    if not isinstance(meta, dict):
        return False
    if meta.get('baseline_refine_time_indices') is not None:
        return True
    return meta.get('posthoc_targetdiff_baseline_refine') is not None


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
            blob = torch.load(str(pt_path), map_location='cpu', weights_only=False)
        except TypeError:
            blob = torch.load(str(pt_path), map_location='cpu')
    except Exception as e:
        print(f"[GPU {gpu_id}] 警告: baseline refine 读取 .pt 失败，使用原文件: {e}")
        return str(pt_path)
    if _pt_dict_already_baseline_refined(blob):
        print(f"[GPU {gpu_id}] 已含 baseline_refine，跳过: {pt_path.name}")
        return str(pt_path)

    cfg_path = Path(config_file)
    if yaml is not None and cfg_path.is_file():
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                y = yaml.safe_load(f)
            ref = (y or {}).get('sample', {}).get('targetdiff_baseline_refine', {})
            if not ref.get('enable', False):
                print(
                    f"[GPU {gpu_id}] 警告: sampling.yml 中 targetdiff_baseline_refine.enable 为 false，"
                    f"跳过 post-hoc 修复（见 sample.targetdiff_baseline_refine）"
                )
                return str(pt_path)
        except Exception as e:
            print(f"[GPU {gpu_id}] 警告: 读取 YAML 检查 baseline_refine 失败: {e}")

    try:
        import utils.misc as misc
        from refine_saved_pt_baseline_then_eval import load_stack, refine_one_pt
    except Exception as e:
        print(f"[GPU {gpu_id}] 警告: baseline refine 导入失败: {e}")
        return str(pt_path)

    logger = misc.get_logger('batch_baseline_refine')
    pr = str(protein_root).strip() if protein_root else None
    try:
        config, model, _ckpt, dataset_root = load_stack(cfg_path.resolve(), 'cuda:0', logger)
        outp = refine_one_pt(
            pt_path, pt_path.parent, model, config, dataset_root, pr, 'cuda:0', logger
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
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    if torch is not None:
        try:
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        except Exception:
            pass
    return run_posthoc_baseline_refine(pt_path_str, config_path_str, gpu_id, protein_root_str)


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
            ['nvidia-smi', '--list-gpus'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            # nvidia-smi 可用，说明 GPU 驱动正常
            num_gpus = len(result.stdout.strip().split('\n'))
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
                if hasattr(torch.cuda, '_lazy_init'):
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
    # 记录开始时间
    task_start_time = time.time()
    
    if len(args_tuple) >= 6:
        data_id, gpu_id, config_file, skip_existing, protein_root, molecular_repair = args_tuple[:6]
        protein_root = str(protein_root).strip() or None
        molecular_repair = (molecular_repair or 'none').strip().lower()
    elif len(args_tuple) >= 5:
        data_id, gpu_id, config_file, skip_existing, protein_root = args_tuple[:5]
        protein_root = str(protein_root).strip() or None
        molecular_repair = 'none'
    else:
        data_id, gpu_id, config_file, skip_existing = args_tuple
        protein_root = None
        molecular_repair = 'none'
    
    # 在进程开始时立即设置CUDA_VISIBLE_DEVICES
    # 这必须在任何CUDA初始化之前完成
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 关键修复：如果torch已经在主进程中导入，需要清除CUDA缓存并重新初始化
    # 这是因为子进程可能继承了主进程的CUDA上下文状态
    if torch is not None:
        try:
            # 清除可能存在的CUDA缓存
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            # 尝试重置CUDA状态（如果可能）
            if hasattr(torch.cuda, '_lazy_init'):
                # 清除lazy init状态，强制重新初始化
                torch.cuda._lazy_init()
        except Exception as e:
            # 如果清除失败，继续执行（可能CUDA还未初始化）
            pass
    
    try:
        # 检查GPU内存是否足够（在设置CUDA_VISIBLE_DEVICES之前检查物理GPU）
        has_memory, free_mb, total_mb, mem_msg = check_gpu_memory_available(gpu_id, min_free_memory_mb=5000)
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
            record_pocket_generation_time(data_id, gpu_id, generation_time, status='失败-GPU内存不足', pt_file=None)
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
            record_pocket_generation_time(data_id, gpu_id, generation_time, status='失败-CUDA不可用', pt_file=None)
            return (data_id, False, None, error_msg)
        
        # 检查是否已存在
        if skip_existing:
            pt_file = find_latest_result_file(data_id)
            if pt_file and pt_file.exists():
                print(f"[GPU {gpu_id}] ⏭️  跳过已存在的文件: {pt_file} (data_id={data_id})")
                # 跳过已存在的文件不记录时间（因为没有实际生成）
                pt_file = str(pt_file)
                if molecular_repair == 'targetdiff_baseline_refine':
                    refined = run_posthoc_baseline_refine(pt_file, config_file, gpu_id, protein_root)
                    if refined:
                        pt_file = refined
                return (data_id, True, pt_file, "文件已存在")
        
        # 执行采样（带重试机制）
        sample_success, pt_file, sample_msg = run_single_sample(
            data_id, config_file, gpu_id, max_retries=3, retry_delay=5, protein_root=protein_root
        )
        if sample_success and pt_file and molecular_repair == 'targetdiff_baseline_refine':
            refined = run_posthoc_baseline_refine(pt_file, config_file, gpu_id, protein_root)
            if refined:
                pt_file = refined
        
        # 计算生成时间
        generation_time = time.time() - task_start_time
        
        if not sample_success or pt_file is None:
            # 记录失败状态
            record_pocket_generation_time(data_id, gpu_id, generation_time, status='失败-采样失败', pt_file=None)
            return (data_id, False, None, sample_msg)
        
        # 记录成功状态
        record_pocket_generation_time(data_id, gpu_id, generation_time, status='成功', pt_file=pt_file)
        
        return (data_id, True, pt_file, sample_msg)
            
    except Exception as e:
        # 计算生成时间（即使异常也记录）
        generation_time = time.time() - task_start_time
        error_msg = f"采样任务异常 (data_id={data_id}): {str(e)}"
        print(f"[GPU {gpu_id}] ❌ {error_msg}")
        traceback.print_exc()
        # 记录异常状态
        record_pocket_generation_time(data_id, gpu_id, generation_time, status='失败-异常', pt_file=None)
        return (data_id, False, None, error_msg)


def process_evaluation_task(args_tuple):
    """
    处理单个评估任务（仅评估，不采样）
    这个函数会在独立的进程中运行，可以使用多个进程并行
    
    Args:
        args_tuple: (data_id, pt_file, protein_root, atom_mode, exhaustiveness, excel_file,
            batch_start_time, excel_lock, cores_per_task, save_intermediate_interval,
            eval_show_output, eval_save_subprocess_log[, force_mmff_minimize, mmff_max_iters[, vina_modes]]）；
            兼容10 / 12 / 14 / 15 元组（缺省则 force_mmff=False、mmff_max_iters=None、vina_modes=None）
    
    Returns:
        tuple: (data_id, success, message, log_file, pt_file, eval_output_dir)
    """
    if len(args_tuple) >= 15:
        (data_id, pt_file, protein_root, atom_mode, exhaustiveness,
         excel_file, batch_start_time, excel_lock, cores_per_task, save_intermediate_interval,
         eval_show_output, eval_save_subprocess_log, force_mmff_minimize, mmff_max_iters,
         vina_modes) = args_tuple[:15]
    elif len(args_tuple) >= 14:
        (data_id, pt_file, protein_root, atom_mode, exhaustiveness,
         excel_file, batch_start_time, excel_lock, cores_per_task, save_intermediate_interval,
         eval_show_output, eval_save_subprocess_log, force_mmff_minimize, mmff_max_iters) = args_tuple[:14]
        vina_modes = None
    elif len(args_tuple) >= 12:
        (data_id, pt_file, protein_root, atom_mode, exhaustiveness,
         excel_file, batch_start_time, excel_lock, cores_per_task, save_intermediate_interval,
         eval_show_output, eval_save_subprocess_log) = args_tuple[:12]
        force_mmff_minimize = False
        mmff_max_iters = None
        vina_modes = None
    elif len(args_tuple) == 10:
        (data_id, pt_file, protein_root, atom_mode, exhaustiveness,
         excel_file, batch_start_time, excel_lock, cores_per_task, save_intermediate_interval) = args_tuple
        eval_show_output = False
        eval_save_subprocess_log = False
        force_mmff_minimize = False
        mmff_max_iters = None
        vina_modes = None
    else:
        raise ValueError(f"process_evaluation_task: 元组长度应为 10、12、14 或 15，实际 {len(args_tuple)}")
    
    # 设置全局锁（与 excel_file 同 stem 的评估记录 CSV 写入）
    global excel_write_lock
    excel_write_lock = excel_lock
    
    task_start_time = time.time()
    
    try:
        if not pt_file:
            return (data_id, False, "无PT文件", None, None, None)
        
        # 执行评估
        eval_success, eval_msg, eval_output_dir = run_single_evaluation(
            pt_file, protein_root, data_id, atom_mode, exhaustiveness, cores_per_task=cores_per_task,
            save_intermediate_interval=save_intermediate_interval,
            eval_show_output=eval_show_output,
            eval_save_subprocess_log=eval_save_subprocess_log,
            force_mmff_minimize=force_mmff_minimize,
            mmff_max_iters=mmff_max_iters,
            vina_modes=vina_modes,
        )
        
        task_time = time.time() - task_start_time
        # 使用本地时区（CST，UTC+8）生成时间戳
        cst = timezone(timedelta(hours=8))
        timestamp_str = datetime.now(cst).strftime('%Y-%m-%d %H:%M:%S')
        
        if eval_success:
            # 等待评估结果文件生成
            time.sleep(2)
            eval_success_read, vina_mean, vina_median, num_scores, eval_message, _ = read_evaluation_results(
                pt_file, data_id, wait_timeout=60
            )
            
            if eval_success_read:
                if excel_file:
                    append_to_excel(
                        excel_file, timestamp_str, task_time, data_id, pt_file,
                        vina_mean, vina_median, num_scores, '成功', eval_message
                    )
            else:
                if excel_file:
                    append_to_excel(
                        excel_file, timestamp_str, task_time, data_id, pt_file,
                        None, None, 0, '部分成功', f"评估完成但读取结果失败: {eval_message}"
                    )
            
            return (data_id, True, eval_msg, None, pt_file, str(eval_output_dir) if eval_output_dir else None)
        else:
            if excel_file:
                append_to_excel(
                    excel_file, timestamp_str, task_time, data_id, pt_file,
                    None, None, 0, '失败', eval_msg
                )
            
            return (data_id, False, eval_msg, None, pt_file, None)
            
    except Exception as e:
        error_msg = f"评估任务异常 (data_id={data_id}): {str(e)}"
        print(f"[评估] ❌ {error_msg}")
        traceback.print_exc()
        return (data_id, False, error_msg, None, pt_file, None)


def append_to_excel(excel_file, timestamp, execution_time, data_id, pt_file, vina_mean, vina_median, 
                    num_scores, status, message):
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

    csv_records = excel_file.with_name(excel_file.stem + '_评估记录.csv')
    csv_stats = excel_file.with_name(excel_file.stem + '_评估统计.csv')
    
    # 创建锁文件路径（与同 stem 的评估 CSV 共用）
    lock_file = excel_file.with_name(excel_file.stem + '_评估记录.csv.lock')
    
    # 使用文件锁确保进程安全
    lock_acquired = False
    lock_fd = None
    try:
        # 打开或创建锁文件
        lock_fd = open(str(lock_file), 'w')
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
                '执行时间': timestamp,
                '执行耗时(秒)': execution_time,
                '数据ID': data_id,
                'PT文件': os.path.basename(str(pt_file)) if pt_file else '',
                'Vina平均得分': vina_mean if vina_mean is not None else '',
                'Vina中位数得分': vina_median if vina_median is not None else '',
                '得分数量': num_scores if num_scores else 0,
                '状态': status,
                '备注': message
            }
            
            # 定义必需的列
            required_columns = ['执行时间', '执行耗时(秒)', '数据ID', 'PT文件', 
                              'Vina平均得分', 'Vina中位数得分', '得分数量', '状态', '备注', '累计均值']
            
            df = pd.DataFrame()
            cumulative_mean = vina_mean
            
            if csv_records.exists():
                try:
                    df = pd.read_csv(csv_records, encoding='utf-8-sig')
                    if df.empty or not all(col in df.columns for col in required_columns[:-1]):
                        print('⚠️  警告: 评估记录 CSV 列不匹配，重新创建表。')
                        df = pd.DataFrame()
                    else:
                        if '累计均值' not in df.columns:
                            df['累计均值'] = ''
                        if '状态' in df.columns:
                            successful_rows = df[df['状态'] == '成功']
                            if len(successful_rows) > 0:
                                all_means = successful_rows['Vina平均得分'].dropna().tolist()
                                if vina_mean is not None:
                                    all_means.append(vina_mean)
                                cumulative_mean = np.mean(all_means) if all_means else None
                except Exception as e:
                    print(f'⚠️  警告: 读取评估记录 CSV 失败 {csv_records}: {e}，将新建。')
                    df = pd.DataFrame()
            
            new_row['累计均值'] = cumulative_mean if cumulative_mean is not None else ''
            
            # 确保所有列都存在
            for col in required_columns:
                if col not in new_row:
                    new_row[col] = ''
            
            new_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_df], ignore_index=True)
            
            excel_file.parent.mkdir(parents=True, exist_ok=True)
            temp_records = csv_records.with_suffix('.csv.tmp')
            temp_stats = csv_stats.with_suffix('.csv.tmp')
            
            try:
                df.to_csv(temp_records, index=False, encoding='utf-8-sig')
                temp_records.replace(csv_records)
                
                if len(df) > 0 and '状态' in df.columns:
                    successful_df = df[df['状态'] == '成功']
                    if len(successful_df) > 0:
                        stats = {
                            '统计项目': [
                                '总评估次数',
                                '成功次数',
                                '失败次数',
                                '当前累计均值',
                                '当前累计中位数',
                                '最佳得分',
                                '最差得分'
                            ],
                            '数值': [
                                len(df),
                                len(successful_df),
                                len(df) - len(successful_df),
                                successful_df['Vina平均得分'].mean() if len(successful_df) > 0 else '',
                                successful_df['Vina平均得分'].median() if len(successful_df) > 0 else '',
                                successful_df['Vina平均得分'].min() if len(successful_df) > 0 else '',
                                successful_df['Vina平均得分'].max() if len(successful_df) > 0 else ''
                            ]
                        }
                        stats_df = pd.DataFrame(stats)
                        stats_df.to_csv(temp_stats, index=False, encoding='utf-8-sig')
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
    if not dir_name.startswith('eval_'):
        return None
    m = re.match(r'^eval_\d{8}_\d{6}_(\d+)(?:_|$)', dir_name)
    if m:
        return int(m.group(1))
    parts = dir_name.split('_')
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            return parts[1]
    return None


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
    if pd is None:
        print('⚠️  pandas未安装，无法读取Excel文件')
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
        eval_dirs = list(OUTPUT_DIR.glob('eval_*'))
        if extra_eval_roots:
            for root in extra_eval_roots:
                root = Path(root)
                if root.is_dir():
                    eval_dirs.extend(root.glob('eval_*'))
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
        excel_files = list(eval_dir.glob('evaluation_results_*.xlsx'))
        
        if not excel_files:
            continue
        
        # 本批次限定目录：直接采用该目录下 xlsx（避免 mtime 与脚本启动先后导致漏读）
        if use_scoped_only:
            recent_excel_files = excel_files
        else:
            recent_excel_files = [
                f for f in excel_files
                if datetime.fromtimestamp(f.stat().st_mtime) >= batch_start_datetime
            ]
        
        if not recent_excel_files:
            continue
        
        # 选择最新的Excel文件（如果有多个）
        latest_excel_file = max(recent_excel_files, key=os.path.getmtime)
        
        try:
            # 读取Excel文件的"评估结果"工作表
            df = pd.read_excel(latest_excel_file, sheet_name='评估结果', engine='openpyxl')
            
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
                data_id = record.get('数据ID')
                if data_id is None or (isinstance(data_id, float) and pd.isna(data_id)):
                    parsed = _data_id_from_eval_dir_name(eval_dir.name)
                    if parsed is not None:
                        record['数据ID'] = parsed
                
                molecule_records.append(record)
            
            # 获取数据ID
            data_id = _data_id_from_eval_dir_name(eval_dir.name)
            
            # 读取统计信息工作表以获取统计数据
            pocket_stats = {'数据ID': data_id, '评估成功数': len(df)}
            try:
                stats_df = pd.read_excel(latest_excel_file, sheet_name='统计信息', engine='openpyxl')
                if not stats_df.empty and '统计项目' in stats_df.columns and '数值' in stats_df.columns:
                    stats_dict = dict(zip(stats_df['统计项目'], stats_df['数值']))
                    
                    # 提取统计数据
                    if '总样本数' in stats_dict:
                        try:
                            total_num_samples += int(stats_dict['总样本数'])
                        except (ValueError, TypeError):
                            pass
                    
                    if '重建成功' in stats_dict:
                        try:
                            total_n_reconstruct_success += int(stats_dict['重建成功'])
                        except (ValueError, TypeError):
                            pass
                    
                    if '评估成功' in stats_dict:
                        try:
                            eval_success = int(stats_dict['评估成功'])
                            total_n_eval_success += eval_success
                            pocket_stats['评估成功数'] = eval_success
                        except (ValueError, TypeError):
                            pass
                    
                    # 提取各个参数的均值（用于加权平均计算）
                    # 查找所有包含"平均"的统计项目
                    for stat_name, stat_value in stats_dict.items():
                        if '平均' in stat_name and stat_value not in ('N/A', None, ''):
                            try:
                                # 尝试转换为数值
                                if isinstance(stat_value, str):
                                    # 尝试转换为浮点数
                                    stat_value = float(stat_value)
                                pocket_stats[stat_name] = stat_value
                            except (ValueError, TypeError):
                                pass
                    
                    # 也查找一些常见的统计指标（即使没有"平均"字样）
                    for stat_name in ['Vina_Dock_最佳亲和力', 'Vina_Dock_最差亲和力', 
                                     'Vina_ScoreOnly_最佳亲和力', 'Vina_Minimize_最佳亲和力']:
                        if stat_name in stats_dict:
                            try:
                                val = stats_dict[stat_name]
                                if val not in ('N/A', None, '') and not pd.isna(val):
                                    if isinstance(val, str):
                                        val = float(val)
                                    pocket_stats[stat_name] = val
                            except (ValueError, TypeError):
                                pass
            except Exception as e:
                # 如果读取统计信息失败，不影响主流程
                pass
            
            # 如果没有从统计信息中获取到评估成功数，使用分子记录数
            if pocket_stats['评估成功数'] == 0:
                pocket_stats['评估成功数'] = len(df)
            
            # 添加到口袋统计列表
            if pocket_stats['评估成功数'] > 0:
                pocket_stats_list.append(pocket_stats)
            
            print(f"  ✅ 已读取: {latest_excel_file.name} ({len(df)} 条记录)")
            
        except Exception as e:
            print(f"⚠️  读取Excel文件失败 {latest_excel_file}: {e}")
            continue
    
    # 计算汇总统计信息
    # 计算百分比
    reconstruct_success_rate = (total_n_reconstruct_success / total_num_samples * 100) if total_num_samples > 0 else 0.0
    docking_success_rate = (len(molecule_records) / total_num_samples * 100) if total_num_samples > 0 else 0.0
    
    # 计算有效分子数量（剔除vinadock>0、vinascore>0或vinamin>0的异常数据）
    # 注意：不删除任何能对接的分子，只是统计有效分子数量
    valid_molecule_count = 0
    for r in molecule_records:
        vina_dock = r.get('Vina_Dock_亲和力', 'N/A')
        vina_score = r.get('Vina_ScoreOnly_亲和力', 'N/A')
        vina_min = r.get('Vina_Minimize_亲和力', 'N/A')
        
        # 检查是否为异常数据（vinadock>0、vinascore>0或vinamin>0）
        is_abnormal = False
        try:
            if vina_dock not in ('N/A', None) and not pd.isna(vina_dock):
                if float(vina_dock) > 0:
                    is_abnormal = True
            if vina_score not in ('N/A', None) and not pd.isna(vina_score):
                if float(vina_score) > 0:
                    is_abnormal = True
            if vina_min not in ('N/A', None) and not pd.isna(vina_min):
                if float(vina_min) > 0:
                    is_abnormal = True
        except (ValueError, TypeError):
            pass
        
        # 如果不是异常数据，则计入有效分子
        if not is_abnormal:
            valid_molecule_count += 1
    
    # 计算有效分子比例（有效分子数量 / 应生成分子数）
    valid_molecule_ratio = (valid_molecule_count / total_num_samples * 100) if total_num_samples > 0 else 0.0
    
    summary_stats = {
        'batch启动时间': datetime.fromtimestamp(batch_start_time).strftime('%Y-%m-%d %H:%M:%S'),
        '应生成分子数': total_num_samples,
        '可重建分子数': total_n_reconstruct_success,
        '重建成功百分比(%)': f"{reconstruct_success_rate:.2f}",
        '对接成功分子数': len(molecule_records),
        '对接成功百分比(%)': f"{docking_success_rate:.2f}",
        '有效分子数量': valid_molecule_count,
        '有效分子比例(%)': f"{valid_molecule_ratio:.2f}",
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
            vina_dock = r.get('Vina_Dock_亲和力', 'N/A')
            if vina_dock not in ('N/A', None) and not pd.isna(vina_dock):
                try:
                    vina_dock_scores.append(float(vina_dock))
                except (ValueError, TypeError):
                    pass
            
            # 处理Vina_ScoreOnly_亲和力
            vina_score_only = r.get('Vina_ScoreOnly_亲和力', 'N/A')
            if vina_score_only not in ('N/A', None) and not pd.isna(vina_score_only):
                try:
                    vina_score_only_scores.append(float(vina_score_only))
                except (ValueError, TypeError):
                    pass
            
            # 处理Vina_Minimize_亲和力
            vina_minimize = r.get('Vina_Minimize_亲和力', 'N/A')
            if vina_minimize not in ('N/A', None) and not pd.isna(vina_minimize):
                try:
                    vina_minimize_scores.append(float(vina_minimize))
                except (ValueError, TypeError):
                    pass
            
            # 处理QED评分
            qed = r.get('QED评分', 'N/A')
            if qed not in ('N/A', None) and not pd.isna(qed):
                try:
                    qed_values.append(float(qed))
                except (ValueError, TypeError):
                    pass
            
            # 处理SA评分
            sa = r.get('SA评分', 'N/A')
            if sa not in ('N/A', None) and not pd.isna(sa):
                try:
                    sa_values.append(float(sa))
                except (ValueError, TypeError):
                    pass
        
        if vina_dock_scores:
            summary_stats['Vina_Dock_平均亲和力'] = np.mean(vina_dock_scores)
        if vina_score_only_scores:
            summary_stats['Vina_ScoreOnly_平均亲和力'] = np.mean(vina_score_only_scores)
        if vina_minimize_scores:
            summary_stats['Vina_Minimize_平均亲和力'] = np.mean(vina_minimize_scores)
        if qed_values:
            summary_stats['QED平均评分'] = np.mean(qed_values)
        if sa_values:
            summary_stats['SA平均评分'] = np.mean(sa_values)
    
    return molecule_records, summary_stats, pocket_stats_list


def flatten_config(config, parent_key='', sep='.'):
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
            items.append((new_key, 'null'))
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
        sample_cfg = config.get('sample', {})
        if not isinstance(sample_cfg, dict):
            return None, None
        
        num_timesteps = sample_cfg.get('num_steps', 1000)
        dynamic_cfg = sample_cfg.get('dynamic', {})
        if not isinstance(dynamic_cfg, dict):
            return None, None
        
        large_step_cfg = dynamic_cfg.get('large_step', {})
        if not isinstance(large_step_cfg, dict):
            large_step_cfg = {}
        
        refine_cfg = dynamic_cfg.get('refine', {})
        if not isinstance(refine_cfg, dict):
            refine_cfg = {}
        
        # 获取 time_boundary
        time_boundary = dynamic_cfg.get('time_boundary', 600)
        
        # Large Step 阶段
        large_step_schedule = large_step_cfg.get('schedule', 'lambda')
        large_step_time_lower = time_boundary
        large_step_time_upper = num_timesteps - 1
        large_step_size = large_step_cfg.get('step_size', 1.0)  # 获取step_size
        
        # Refine 阶段
        refine_schedule = refine_cfg.get('schedule', 'lambda')
        refine_time_upper = time_boundary
        refine_time_lower = refine_cfg.get('time_lower', 0)
        refine_step_size = refine_cfg.get('step_size', 0.2)  # 获取step_size
        
        # 计算 Large Step 阶段时间步
        if large_step_schedule == 'lambda':
            lambda_coeff_a = large_step_cfg.get('lambda_coeff_a', 80.0)
            lambda_coeff_b = large_step_cfg.get('lambda_coeff_b', 20.0)
            large_step_indices, _ = build_lambda_schedule(
                large_step_time_upper, large_step_time_lower,
                lambda_coeff_a, lambda_coeff_b, num_timesteps
            )
        elif large_step_schedule == 'linear':
            step_upper = large_step_cfg.get('linear_step_upper', 100.0)
            step_lower = large_step_cfg.get('linear_step_lower', 20.0)
            large_step_indices, _ = build_linear_schedule(
                large_step_time_upper, large_step_time_lower,
                step_upper, step_lower, num_timesteps
            )
        else:
            stride = large_step_cfg.get('stride', 15)
            large_step_indices, _ = build_fixed_schedule(
                large_step_time_upper, large_step_time_lower,
                stride, num_timesteps
            )
        
        # 计算 Refine 阶段时间步
        if refine_schedule == 'lambda':
            lambda_coeff_a = refine_cfg.get('lambda_coeff_a', 40.0)
            lambda_coeff_b = refine_cfg.get('lambda_coeff_b', 5.0)
            refine_indices, _ = build_lambda_schedule(
                refine_time_upper, refine_time_lower,
                lambda_coeff_a, lambda_coeff_b, num_timesteps
            )
        elif refine_schedule == 'linear':
            step_upper = refine_cfg.get('linear_step_upper', 40.0)
            step_lower = refine_cfg.get('linear_step_lower', 5.0)
            refine_indices, _ = build_linear_schedule(
                refine_time_upper, refine_time_lower,
                step_upper, step_lower, num_timesteps
            )
        else:
            stride = refine_cfg.get('stride', 8)
            refine_indices, _ = build_fixed_schedule(
                refine_time_upper, refine_time_lower,
                stride, num_timesteps
            )
        
        # 计算实际长度：ls的步数 × ls的step_size + rf的步数 × rf的step_size
        # 步数是指每个阶段的采样点数量（即indices列表的长度）
        large_step_num_steps = len(large_step_indices) if large_step_indices else 0
        refine_num_steps = len(refine_indices) if refine_indices else 0
        
        # 计算实际长度
        actual_length = large_step_num_steps * large_step_size + refine_num_steps * refine_step_size
        
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
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if not config:
            return None
        
        # 扁平化配置
        flat_config = flatten_config(config)
        
        # 计算总采样步数和实际长度
        total_steps, actual_length = calculate_total_sampling_steps(config)
        if total_steps is not None:
            flat_config.append(('计算.跳步总次数', total_steps))
        if actual_length is not None:
            flat_config.append(('计算.实际长度', actual_length))
        
        # 转换为DataFrame
        df = pd.DataFrame(flat_config, columns=['参数路径', '参数值'])
        
        # 将值转换为字符串（便于在Excel中显示）
        df['参数值'] = df['参数值'].astype(str)
        
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
    gf_match = re.search(r'gf(\w+)', filename_str)
    if gf_match:
        params['权重策略'] = gf_match.group(1)
    
    # 提取开始权重和结束权重 (gfquadratic_1_0 -> 开始权重=1, 结束权重=0)
    weight_match = re.search(r'gf\w+_(\d+)_(\d+)', filename_str)
    if weight_match:
        params['开始权重'] = float(weight_match.group(1))
        params['结束权重'] = float(weight_match.group(2))
    
    # 提取时间长度 (tl800 -> 800)
    tl_match = re.search(r'tl(\d+)', filename_str)
    if tl_match:
        params['时间长度 (TL)'] = int(tl_match.group(1))
    
    # 提取LS Lambda值 (lslambda_60p0_20p0 -> LSLambda1=60.0, LSLambda2=20.0)
    ls_lambda_match = re.search(r'lslambda_(\d+p\d+)_(\d+p\d+)', filename_str)
    if ls_lambda_match:
        params['LSLambda1'] = float(ls_lambda_match.group(1).replace('p', '.'))
        params['LSLambda2'] = float(ls_lambda_match.group(2).replace('p', '.'))
    
    # 提取LS step size (lsstep_0p6 -> 0.6)
    ls_step_match = re.search(r'lsstep_(\d+p\d+)', filename_str)
    if ls_step_match:
        params['LSstepsize'] = float(ls_step_match.group(1).replace('p', '.'))
    
    # 提取LS noise (lsnoise_0p0 -> 0.0)
    ls_noise_match = re.search(r'lsnoise_(\d+p\d+)', filename_str)
    if ls_noise_match:
        params['LSnosie'] = float(ls_noise_match.group(1).replace('p', '.'))
    
    # 提取RF Lambda值 (rflambda_10p0_5p0 -> RFLambda1=10.0, RFLambda2=5.0)
    rf_lambda_match = re.search(r'rflambda_(\d+p\d+)_(\d+p\d+)', filename_str)
    if rf_lambda_match:
        params['RFLambda1'] = float(rf_lambda_match.group(1).replace('p', '.'))
        params['RFLambda2'] = float(rf_lambda_match.group(2).replace('p', '.'))
    
    # 提取RF step size (rfstep_0p25 -> 0.25)
    rf_step_match = re.search(r'rfstep_(\d+p\d+)', filename_str)
    if rf_step_match:
        params['RFstepsize'] = float(rf_step_match.group(1).replace('p', '.'))
    
    # 提取RF noise (rfnoise_0p05 -> 0.05)
    rf_noise_match = re.search(r'rfnoise_(\d+p\d+)', filename_str)
    if rf_noise_match:
        params['RFnosie'] = float(rf_noise_match.group(1).replace('p', '.'))
    
    return params


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
    if pd is None:
        return
    
    import fcntl
    
    try:
        excel_file = Path(excel_file)
        if not excel_file.exists():
            return
        
        batchsummary_dir = excel_file.parent
        merged_batches_csv = batchsummary_dir / 'merged_summary.csv'
        merged_valid_csv = batchsummary_dir / 'merged_valid_molecules.csv'
        merge_lock = batchsummary_dir / 'merged_summary.csv.lock'
        
        # 定义列的顺序
        columns_order = [
            '权重策略', '下降速率', '开始权重', '结束权重', '时间长度 (TL)',
            'LSstepsize', 'LSnosie', 'LSLambda1', 'LSLambda2',
            'RFstepsize', 'RFnosie', 'RFLambda1', 'RFLambda2',
            '步数', '取模步长', '可重建率 (%)', '对接成功率 (%)',
            '有效分子比例 (%)',
            'Vina_Dock 亲和力', 'Vina_ScoreOnly', 'Vina_Minimize',
            'QED 评分（均值）', 'SA 评分（均值）'
        ]
        
        # 从文件名解析参数
        filename_params = parse_filename_params(excel_file.name)
        
        # 从批次汇总读取统计数据（CSV 侧车或旧版 xlsx）
        try:
            if excel_file.suffix.lower() == '.csv':
                stats_side = excel_file.with_name(excel_file.stem + '_统计信息.csv')
                if not stats_side.exists():
                    stats_dict = {}
                else:
                    df_stats = pd.read_csv(stats_side, encoding='utf-8-sig')
                    stats_dict = {}
                    for _, row in df_stats.iterrows():
                        key = str(row['统计项目'])
                        value = row['数值']
                        if isinstance(value, str):
                            try:
                                value = float(value)
                            except ValueError:
                                pass
                        stats_dict[key] = value
            else:
                df_stats = pd.read_excel(excel_file, sheet_name='统计信息', engine='openpyxl')
                stats_dict = {}
                for _, row in df_stats.iterrows():
                    key = str(row['统计项目'])
                    value = row['数值']
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
            if excel_file.suffix.lower() == '.csv':
                cfg_side = excel_file.with_name(excel_file.stem + '_配置参数.csv')
                if cfg_side.exists():
                    df_config = pd.read_csv(cfg_side, encoding='utf-8-sig')
                    for _, row in df_config.iterrows():
                        key = str(row['参数路径'])
                        value = row['参数值']
                        if pd.isna(value):
                            continue
                        config_dict[key] = value
            else:
                df_config = pd.read_excel(excel_file, sheet_name='配置参数', engine='openpyxl')
                for _, row in df_config.iterrows():
                    key = str(row['参数路径'])
                    value = row['参数值']
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
                        key = str(row['参数路径'])
                        value = row['参数值']
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
        if '下降速率' not in row_data:
            power = config_dict.get('model.grad_fusion_lambda.power', None)
            if power is not None:
                row_data['下降速率'] = float(power)
        
        if '步数' not in row_data:
            steps = config_dict.get('计算.跳步总次数', None)
            if steps is not None:
                row_data['步数'] = int(steps)
        
        if '取模步长' not in row_data:
            mod_step = config_dict.get('计算.实际长度', None)
            if mod_step is not None:
                row_data['取模步长'] = float(mod_step)
        
        # 从配置参数补充缺失的参数
        if 'LSstepsize' not in row_data:
            ls_step = config_dict.get('sample.dynamic.large_step.step_size', None)
            if ls_step is not None:
                row_data['LSstepsize'] = float(ls_step)
        
        if 'LSnosie' not in row_data:
            ls_noise = config_dict.get('sample.dynamic.large_step.noise_scale', None)
            if ls_noise is not None:
                row_data['LSnosie'] = float(ls_noise)
        
        if 'LSLambda1' not in row_data:
            ls_lambda_a = config_dict.get('sample.dynamic.large_step.lambda_coeff_a', None)
            if ls_lambda_a is not None:
                row_data['LSLambda1'] = float(ls_lambda_a)
        
        if 'LSLambda2' not in row_data:
            ls_lambda_b = config_dict.get('sample.dynamic.large_step.lambda_coeff_b', None)
            if ls_lambda_b is not None:
                row_data['LSLambda2'] = float(ls_lambda_b)
        
        if 'RFstepsize' not in row_data:
            rf_step = config_dict.get('sample.dynamic.refine.step_size', None)
            if rf_step is not None:
                row_data['RFstepsize'] = float(rf_step)
        
        if 'RFnosie' not in row_data:
            rf_noise = config_dict.get('sample.dynamic.refine.noise_scale', None)
            if rf_noise is not None:
                row_data['RFnosie'] = float(rf_noise)
        
        if 'RFLambda1' not in row_data:
            rf_lambda_a = config_dict.get('sample.dynamic.refine.lambda_coeff_a', None)
            if rf_lambda_a is not None:
                row_data['RFLambda1'] = float(rf_lambda_a)
        
        if 'RFLambda2' not in row_data:
            rf_lambda_b = config_dict.get('sample.dynamic.refine.lambda_coeff_b', None)
            if rf_lambda_b is not None:
                row_data['RFLambda2'] = float(rf_lambda_b)
        
        if '时间长度 (TL)' not in row_data:
            time_boundary = config_dict.get('sample.dynamic.time_boundary', None)
            if time_boundary is not None:
                row_data['时间长度 (TL)'] = int(time_boundary)
        
        if '权重策略' not in row_data:
            mode = config_dict.get('model.grad_fusion_lambda.mode', None)
            if mode is not None:
                row_data['权重策略'] = str(mode)
        
        if '开始权重' not in row_data:
            start = config_dict.get('model.grad_fusion_lambda.start', None)
            if start is not None:
                row_data['开始权重'] = float(start)
        
        if '结束权重' not in row_data:
            end = config_dict.get('model.grad_fusion_lambda.end', None)
            if end is not None:
                row_data['结束权重'] = float(end)
        
        # 从统计信息中提取（安全转换）
        def safe_float(value, default=0.0):
            try:
                if isinstance(value, str):
                    return float(value)
                return float(value) if value is not None else default
            except (ValueError, TypeError):
                return default
        
        row_data['可重建率 (%)'] = safe_float(stats_dict.get('重建成功百分比(%)', 0))
        row_data['对接成功率 (%)'] = safe_float(stats_dict.get('对接成功百分比(%)', 0))
        row_data['有效分子比例 (%)'] = safe_float(stats_dict.get('有效分子比例(%)', 0))
        row_data['Vina_Dock 亲和力'] = safe_float(stats_dict.get('Vina_Dock_平均亲和力', 0))
        row_data['Vina_ScoreOnly'] = safe_float(stats_dict.get('Vina_ScoreOnly_平均亲和力', 0))
        row_data['Vina_Minimize'] = safe_float(stats_dict.get('Vina_Minimize_平均亲和力', 0))
        row_data['QED 评分（均值）'] = safe_float(stats_dict.get('QED平均评分', 0))
        row_data['SA 评分（均值）'] = safe_float(stats_dict.get('SA平均评分', 0))
        
        # 确保所有列都存在
        new_row = {}
        for col in columns_order:
            new_row[col] = row_data.get(col, None)
        # 添加文件名列
        new_row['文件名'] = excel_file.name
        all_columns = columns_order + ['文件名']
        
        lock_fd = None
        lock_ok = False
        try:
            lock_fd = open(str(merge_lock), 'w')
            for _ in range(100):
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    lock_ok = True
                    break
                except (IOError, OSError):
                    time.sleep(0.1)
            if not lock_ok:
                print('⚠️  获取汇总 CSV 锁超时，跳过 merged_summary 更新')
                return
            
            if merged_batches_csv.exists():
                try:
                    df_existing = pd.read_csv(merged_batches_csv, encoding='utf-8-sig')
                    if '文件名' in df_existing.columns:
                        df_existing = df_existing[df_existing['文件名'] != excel_file.name]
                    df_new = pd.DataFrame([new_row])
                    for col in all_columns:
                        if col not in df_existing.columns:
                            df_existing[col] = None
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                except Exception as e:
                    print(f'⚠️  读取 merged_summary.csv 失败，将重建: {e}')
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
                if excel_file.suffix.lower() == '.csv':
                    v_side = excel_file.with_name(excel_file.stem + '_正常分子.csv')
                    if v_side.exists():
                        df_valid_batch = pd.read_csv(v_side, encoding='utf-8-sig')
                else:
                    df_valid_batch = pd.read_excel(
                        excel_file, sheet_name='正常分子', engine='openpyxl'
                    )
            except Exception:
                df_valid_batch = None
            
            n_valid_appended = 0
            if df_valid_batch is not None and len(df_valid_batch) > 0:
                df_all_valid = df_valid_batch.copy()
                df_all_valid['来源文件'] = excel_file.name
                n_valid_appended = len(df_all_valid)
                if 'Vina_Dock_亲和力' in df_all_valid.columns:
                    df_all_valid['Vina_Dock_亲和力_temp'] = df_all_valid['Vina_Dock_亲和力'].replace(
                        'N/A', np.nan
                    )
                    df_all_valid = df_all_valid.sort_values(
                        'Vina_Dock_亲和力_temp', na_position='last'
                    )
                    df_all_valid = df_all_valid.drop(columns=['Vina_Dock_亲和力_temp'])
            
            tmp_batches = batchsummary_dir / 'merged_summary.csv.tmp'
            tmp_valid = batchsummary_dir / 'merged_valid_molecules.csv.tmp'
            
            df_combined.to_csv(tmp_batches, index=False, encoding='utf-8-sig')
            tmp_batches.replace(merged_batches_csv)
            
            if len(df_all_valid) > 0:
                df_all_valid.to_csv(tmp_valid, index=False, encoding='utf-8-sig')
                tmp_valid.replace(merged_valid_csv)
            elif merged_valid_csv.exists():
                merged_valid_csv.unlink(missing_ok=True)
            
            print(
                f'✅ 已更新汇总 CSV: {merged_batches_csv.name}'
                + (f'，已写入本批正常分子 {n_valid_appended} 条 → {merged_valid_csv.name}'
                   if n_valid_appended else '')
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
        print(f'⚠️  追加到汇总 CSV 失败: {e}')
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
    if pd is None:
        print('⚠️  pandas未安装，无法保存批次 CSV')
        return False

    def _atomic_df_to_csv(df: pd.DataFrame, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.parent / (dest.name + '.tmp')
        df.to_csv(tmp, index=False, encoding='utf-8-sig')
        tmp.replace(dest)

    try:
        main_csv = normalize_batch_summary_path(excel_file)
        stem = main_csv.stem

        valid_csv = main_csv.with_name(f'{stem}_正常分子.csv')
        stats_csv = main_csv.with_name(f'{stem}_统计信息.csv')
        weighted_csv = main_csv.with_name(f'{stem}_加权均值统计.csv')
        config_csv = main_csv.with_name(f'{stem}_配置参数.csv')

        if molecule_records:
            df_molecules = pd.DataFrame(molecule_records)
            if 'Vina_Dock_亲和力' in df_molecules.columns:
                df_molecules['Vina_Dock_亲和力_temp'] = df_molecules['Vina_Dock_亲和力'].replace(
                    'N/A', np.nan
                )
                df_molecules = df_molecules.sort_values(
                    'Vina_Dock_亲和力_temp', na_position='last'
                )
                df_molecules = df_molecules.drop(columns=['Vina_Dock_亲和力_temp'])
            _atomic_df_to_csv(df_molecules, main_csv)

            def is_valid_molecule(row):
                vina_dock = row.get('Vina_Dock_亲和力', 'N/A')
                vina_score = row.get('Vina_ScoreOnly_亲和力', 'N/A')
                vina_min = row.get('Vina_Minimize_亲和力', 'N/A')
                try:
                    if vina_dock not in ('N/A', None) and not pd.isna(vina_dock):
                        if float(vina_dock) > 0:
                            return False
                    if vina_score not in ('N/A', None) and not pd.isna(vina_score):
                        if float(vina_score) > 0:
                            return False
                    if vina_min not in ('N/A', None) and not pd.isna(vina_min):
                        if float(vina_min) > 0:
                            return False
                except (ValueError, TypeError):
                    pass
                return True

            df_valid = df_molecules[df_molecules.apply(is_valid_molecule, axis=1)].copy()
            _atomic_df_to_csv(df_valid, valid_csv)
        else:
            _atomic_df_to_csv(pd.DataFrame(), main_csv)
            _atomic_df_to_csv(pd.DataFrame(), valid_csv)

        stats_items = []
        stats_values = []
        for key, value in summary_stats.items():
            stats_items.append(key)
            if isinstance(value, float):
                stats_values.append(f'{value:.3f}')
            else:
                stats_values.append(str(value))
        stats_df = pd.DataFrame({'统计项目': stats_items, '数值': stats_values})
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
                print(f'⚠️  追加到汇总 CSV 失败: {e}')

        return True

    except Exception as e:
        print(f'⚠️  保存批次汇总 CSV 失败: {e}')
        traceback.print_exc()
        return False


def normalize_batch_summary_path(path) -> Path:
    """批次主汇总表统一为 ``.csv``；若传入 ``.xlsx`` 则改为同主名的 ``.csv``。"""
    p = Path(path)
    if p.suffix.lower() == '.xlsx':
        return p.with_suffix('.csv')
    return p


def resolve_eval_only_excel_path(excel_arg, batchsummary_dir, default_excel_path):
    """
    仅评估模式（--pt_dir / --pt_file）下解析批次汇总主表输出路径。

    相对路径统一落在仓库 batchsummary/ 下，便于与 merged_summary.csv 同目录管理；
    绝对路径保留用户指定位置。路径中的 ``.xlsx`` 会规范为 ``.csv``。
    """
    if excel_arg is None or str(excel_arg).strip() == '':
        return normalize_batch_summary_path(default_excel_path)
    p = Path(excel_arg)
    if p.is_absolute():
        return normalize_batch_summary_path(p)
    return normalize_batch_summary_path((Path(batchsummary_dir) / p).resolve())


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
        "evaluate_pt_with_correct_reconstruct.py"
    ]
    
    # 定义要保护的进程模式（薛定谔相关）
    PROTECTED_PATTERNS = [
        "schrodinger",
        "gdesmond",
        "glide",
        "jmonitor"
    ]
    
    # 收集所有相关进程PID
    all_pids = []
    for pattern in PATTERNS:
        try:
            pids = subprocess.run(
                ['pgrep', '-f', pattern],
                capture_output=True,
                text=True,
                timeout=5
            )
            if pids.returncode == 0 and pids.stdout.strip():
                for pid_str in pids.stdout.strip().split('\n'):
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
                ['ps', '-p', str(pid), '-o', 'cmd='],
                capture_output=True,
                text=True,
                timeout=2
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
    print(f"\n{'='*60}")
    print(f"检测到 {len(filtered_pids)} 个残留进程")
    print(f"{'='*60}")
    for pid, cmd in filtered_pids[:10]:  # 只显示前10个
        print(f"  PID {pid}: {cmd[:80]}")
    if len(filtered_pids) > 10:
        print(f"  ... 还有 {len(filtered_pids) - 10} 个进程")
    print(f"{'='*60}")
    
    # 询问用户是否清理（仅在交互模式下）
    if interactive:
        try:
            response = input(f"\n是否清理这些残留进程? (y/n，默认y): ").strip().lower()
            if response and response not in ['y', 'yes']:
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
                    ['kill', '-9', str(pid)],
                    capture_output=True,
                    text=True,
                    timeout=2
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
            print(f"   这些进程的PID: {need_root[:10]}{'...' if len(need_root) > 10 else ''}")
            print(f"   建议:")
            print(f"   1. 使用root用户运行脚本")
            print(f"   2. 或手动清理: sudo kill -9 {' '.join(map(str, need_root[:10]))}")
            if len(need_root) > 10:
                print(f"   3. 或使用清理脚本: sudo bash cleanup_sampling_as_root.sh")
    
    if killed > 0:
        print(f"✓ 成功清理 {killed} 个进程")
        time.sleep(1)  # 等待资源释放
    if failed > 0:
        print(f"⚠️  {failed} 个进程清理失败（可能需要root权限）")
    
    return killed


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
        total_success = df['评估成功数'].sum()
        if total_success == 0:
            return []
        
        # 找到所有包含"平均"的列（这些是需要计算加权均值的参数）
        valid_mean_columns = [col for col in df.columns 
                            if '平均' in col and col != '评估成功数' and col != '数据ID']
        
        if not valid_mean_columns:
            return []
        
        # 存储计算结果
        results = []
        
        for col in valid_mean_columns:
            # 创建临时数据框，只包含非空值的行
            temp_df = df[['评估成功数', col]].dropna()
            
            # 计算简单平均值（所有口袋的平均值，不管是否有数据）
            simple_mean = df[col].mean()
            
            if len(temp_df) == 0:
                real_mean = np.nan
                weighted_sum = 0.0
                used_pockets = 0
            else:
                # 计算加权和：评估成功数 × 参数均值
                temp_df['加权和'] = temp_df['评估成功数'] * temp_df[col]
                
                # 计算真实均值（加权平均）
                weighted_sum = temp_df['加权和'].sum()
                used_pockets = len(temp_df)
                real_mean = weighted_sum / total_success
            
            # 计算差异百分比
            if not pd.isna(simple_mean) and not pd.isna(real_mean) and simple_mean != 0:
                diff_percent = ((real_mean - simple_mean) / simple_mean * 100)
            else:
                diff_percent = np.nan
            
            results.append({
                '参数名称': col,
                '参与计算的口袋数': used_pockets,
                '加权和': weighted_sum,
                '简单平均值': simple_mean if not pd.isna(simple_mean) else np.nan,
                '真实均值（加权平均）': real_mean if not pd.isna(real_mean) else np.nan,
                '差异百分比': diff_percent if not pd.isna(diff_percent) else np.nan
            })
        
        return results
        
    except Exception as e:
        print(f"⚠️  计算加权均值失败: {e}")
        traceback.print_exc()
        return []


def main():
    parser = argparse.ArgumentParser(description='批量采样和评估脚本（并行执行）')
    
    parser.add_argument('--start', type=int, default=0,
                       help='起始 data_id（默认: 0）')
    parser.add_argument('--end', type=int, default=99,
                       help='结束 data_id（默认: 99）')
    parser.add_argument('--data_ids', type=str, default=None,
                       help='指定 data_id 列表，逗号分隔（如 "5,10,20,22"），优先于 --start/--end')
    parser.add_argument('--protein_path', type=str, default=None,
                       help='自定义蛋白口袋 PDB 路径（与 --start/--end 二选一，用于单次自定义采样）')
    parser.add_argument('--ligand_path', type=str, default=None,
                       help='自定义参考配体 SDF 路径（可选，仅与 --protein_path 配合）')
    parser.add_argument('--use_dataset_for_pocket', action='store_true',
                       help='当蛋白在数据集内时，使用 index.pkl 中的配体（与测试集一致，可避免生成分子不完整）')
    parser.add_argument('--pocket_radius', type=float, default=10.0,
                       help='口袋裁剪半径（Å），提供配体时自动裁剪蛋白为口袋（默认: 10.0）')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（默认: configs/sampling.yml）')
    
    default_protein_root = os.environ.get('PROTEIN_ROOT', None)
    if default_protein_root is None:
        possible_paths = [
            REPO_ROOT / 'data' / 'crossdocked_v1.1_rmsd1.0_pocket10',
            Path('/mnt/e/DiffDynamic/data/crossdocked_v1.1_rmsd1.0_pocket10'),
            REPO_ROOT / 'data' / 'crossdocked_v1.1_rmsd1.0',
        ]
        for path in possible_paths:
            if path.exists():
                default_protein_root = str(path)
                break
    
    parser.add_argument('--protein_root', type=str, default=default_protein_root,
                       help=f'蛋白质数据根目录（默认: {default_protein_root if default_protein_root else "未找到，请指定"}）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='评估输出目录（已废弃，评估结果自动保存在outputs目录下）')
    parser.add_argument('--atom_mode', type=str, default='add_aromatic',
                       help='原子模式（默认: add_aromatic）')
    parser.add_argument('--exhaustiveness', type=int, default=8,
                       help='AutoDock Vina对接强度（默认: 8）')
    parser.add_argument('--skip_existing', action='store_true',
                       help='跳过已存在的.pt文件（不重新采样）')
    parser.add_argument('--excel_file', type=str, default=None,
                       help='批次汇总主 CSV 路径（默认: batchsummary/batch_evaluation_summary_{timestamp}.csv）。'
                            '可写 .xlsx 后缀，运行时会被视为同主名 .csv。'
                            '仅评估模式 --pt_dir/--pt_file：相对路径落入仓库 batchsummary/')
    parser.add_argument('--sample-only', action='store_true',
                       help='只生成模式：只执行采样，不执行评估（默认: False）')
    parser.add_argument('--pt_file', type=str, default=None,
                       help='单独评估模式：指定 .pt 文件路径，仅执行评估不采样。支持多个文件逗号分隔')
    parser.add_argument('--pt_dir', type=str, default=None,
                       help='扫描目录模式：指定目录路径，按 --pt-dir-glob 匹配该目录顶层 .pt 并评估（默认仅 result_*.pt）')
    parser.add_argument(
        '--pt-dir-glob',
        type=str,
        default='result_*.pt',
        help='与 --pt_dir 连用：顶层 glob 模式（默认 result_*.pt）。评估目录内全部 pt 可用 "*.pt"',
    )
    parser.add_argument('--auto-cleanup', action='store_true',
                       help='自动清理残留进程（默认: False）')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='不清理残留进程（默认: True，已废弃，保留兼容）')
    
    # GPU配置参数
    parser.add_argument('--gpus', type=str, default=None,
                       help='指定使用的GPU ID，支持多种格式：\n'
                            '  - "0,1,2,3" 指定多个GPU\n'
                            '  - "0-3" 指定GPU范围\n'
                            '  - "0,2-4,6" 混合格式\n'
                            '  - "all" 使用所有可用GPU\n'
                            f'  （默认: {",".join(map(str, DEFAULT_GPU_IDS))}）')
    parser.add_argument('--num_cpu_cores', type=int, default=None,
                       help=f'用于评估的CPU核心数（默认: {DEFAULT_NUM_CPU_CORES}）')
    parser.add_argument('--cores_per_task', type=int, default=1,
                       help='每个评估任务分配的CPU核心数。并行任务数 = num_cpu_cores // cores_per_task。'
                            '设为>1时，每个子进程会限制OMP_NUM_THREADS等，避免多任务竞争CPU（默认: 1）')
    parser.add_argument('--save_intermediate_interval', type=int, default=0,
                       help='每处理多少个分子保存一次中间结果；0=禁用（默认，批量模式减少磁盘占用）')
    parser.add_argument('--eval-show-output', action='store_true',
                       help='评估子进程把 stdout/stderr 直接打到当前终端（可看到 tqdm/逐分子进度；多任务并行时输出会交错）。')
    parser.add_argument('--eval-save-log', action='store_true',
                       help='将每个评估子进程的完整输出写入对应 eval_* 目录下的 evaluate_subprocess.log（默认不保存）。')
    parser.add_argument(
        '--eval-vina-modes',
        type=str,
        default='auto',
        dest='eval_vina_modes',
        help='传给 evaluate_pt 的 --vina-modes。auto（默认）：sampling.yml 为 Prudent 或 .pt 内含 '
             'meta.prudent 时为 none（不重跑 AutoDock Vina，沿用 .pt 中 Prudent 对接分并做理化性质）；'
             '否则不传参（与历史一致：dock+score_only+minimize）。可显式写 none、score_only、'
             'dock,minimize 等。',
    )
    parser.add_argument(
        '--molecular-repair',
        type=str,
        choices=['none', 'mmff', 'targetdiff_baseline_refine'],
        default='none',
        help='分子修复: none；mmff=评估前强制 RDKit MMFF（--force-mmff-minimize）；'
             'targetdiff_baseline_refine=按 sampling.yml 中 sample.targetdiff_baseline_refine 对 .pt 做扩散修复',
    )
    parser.add_argument(
        '--mmff-max-iters',
        type=int,
        default=None,
        help='与 --molecular-repair mmff 联用时传给 evaluate 的 MMFF 最大迭代次数（不设则与 evaluate/yaml 默认一致）',
    )
    parser.add_argument(
        '--collect-benchmark-pocket',
        type=int,
        default=None,
        metavar='DATA_ID',
        help='跨基线对比：从各预置 outputs 复制该口袋的 .pt 到 docs/<DATA_ID>/<方法>/pt/，并对每个 .pt '
             '启动 evaluate_pt_with_correct_reconstruct（在 pt 目录下生成 eval_* 与表格）；需有效 --protein_root '
             '（未指定时尝试 data/、/workspace/data 等回退路径）。运行前会检查 torch_scatter 是否已安装。'
             '并行评估与批量模式相同，请传 --num_cpu_cores 与 --cores_per_task（不设则 num_cpu_cores 用默认值）。',
    )
    parser.add_argument(
        '--methods',
        type=str,
        default=None,
        help='指定要处理的方法列表，逗号分隔。可用: TargetDiff,JSDPT3010,DiffSBDD,DecompDiff,MolForm,IPDiff,GlintDM。'
             '默认处理所有方法。示例: --methods TargetDiff,JSDPT3010 只处理这两个方法，节省计算资源。'
             '注意: DiffSBDD(SDF) 和 DecompDiff 评估较慢，如不需要可用此参数跳过。',
    )
    
    args = parser.parse_args()
    
    if args.config is None:
        args.config = CONFIG
    else:
        args.config = Path(args.config)
    
    if getattr(args, 'collect_benchmark_pocket', None) is not None:
        pid = int(args.collect_benchmark_pocket)
        if pid < 0:
            print('❌ 错误: --collect-benchmark-pocket 须为非负整数')
            sys.exit(1)
        pr = _resolve_protein_root_for_eval_fallback(getattr(args, 'protein_root', None))
        if pr is None:
            print(
                '❌ 错误: 无法解析蛋白质数据目录。请指定 --protein_root 为 CrossDocked 口袋数据根目录 '
                '（例如 /workspace/data/crossdocked_v1.1_rmsd1.0_pocket10）。'
            )
            sys.exit(1)
        if getattr(args, 'molecular_repair', 'none') == 'targetdiff_baseline_refine':
            print(
                '⚠️  提示: --collect-benchmark-pocket 当前不对各基线 .pt 做 targetdiff_baseline_refine，'
                '将按原 .pt 直接评估（与 --molecular-repair none 等效）。'
            )
        num_cpu_collect = (
            args.num_cpu_cores if args.num_cpu_cores is not None else DEFAULT_NUM_CPU_CORES
        )
        cpt_collect = getattr(args, 'cores_per_task', 1) or 1

        # 解析方法列表
        include_methods = None
        if args.methods:
            include_methods = [m.strip() for m in args.methods.split(',') if m.strip()]

        collect_benchmark_pocket_and_eval(
            pid,
            pr,
            docs_parent=REPO_ROOT / 'docs',
            atom_mode=args.atom_mode,
            exhaustiveness=args.exhaustiveness,
            num_cpu_cores=num_cpu_collect,
            cores_per_task=cpt_collect,
            eval_show_output=getattr(args, 'eval_show_output', False),
            eval_save_log=getattr(args, 'eval_save_log', False),
            force_mmff_minimize=getattr(args, 'molecular_repair', 'none') == 'mmff',
            mmff_max_iters=args.mmff_max_iters if getattr(args, 'molecular_repair', 'none') == 'mmff' else None,
            include_methods=include_methods,
        )
        sys.exit(0)
    
    # 自定义文件模式：提前设置 protein_root（在验证前）
    use_custom_mode = args.protein_path is not None
    use_eval_only_mode = args.pt_file is not None or args.pt_dir is not None
    if use_custom_mode:
        args.protein_path = Path(args.protein_path)
        if not args.protein_path.exists():
            print(f"❌ 错误: 蛋白文件不存在: {args.protein_path}")
            sys.exit(1)
        if args.ligand_path:
            args.ligand_path = Path(args.ligand_path)
            if not args.ligand_path.exists():
                print(f"❌ 错误: 配体文件不存在: {args.ligand_path}")
                sys.exit(1)
        if args.protein_root is None:
            args.protein_root = args.protein_path.parent
            print(f"💡 自定义模式：未指定 --protein_root，使用蛋白所在目录: {args.protein_root}")
    
    # 单独评估模式与自定义模式互斥
    if use_eval_only_mode and use_custom_mode:
        print(f"❌ 错误: --pt_file 与 --protein_path 不能同时指定")
        sys.exit(1)
    
    # 只在非只生成模式下要求protein_root（采样+评估、自定义模式、单独评估模式均需要）
    if not args.sample_only or use_eval_only_mode:
        if args.protein_root is None:
            print(f"❌ 错误: 未指定蛋白质数据根目录（--protein_root）")
            print(f"   请使用 --protein_root 参数指定蛋白质数据目录")
            sys.exit(1)
        
        args.protein_root = Path(args.protein_root)
        
        if not args.protein_root.exists():
            # 单独评估模式：尝试常见数据目录作为回退（custom .pt 的蛋白路径可能已嵌入文件）
            if use_eval_only_mode:
                orig_path = args.protein_root
                fallback_paths = [
                    REPO_ROOT / 'data' / 'crossdocked_v1.1_rmsd1.0_pocket10',
                    REPO_ROOT / 'data' / 'crossdocked_v1.1_rmsd1.0',
                    Path('/workspace/data/crossdocked_v1.1_rmsd1.0_pocket10'),
                    Path('/workspace/data'),
                    REPO_ROOT,
                ]
                for fb in fallback_paths:
                    if fb.exists():
                        args.protein_root = fb
                        print(f"💡 单独评估模式：{orig_path} 不存在，使用回退路径: {args.protein_root}")
                        break
                else:
                    args.protein_root = REPO_ROOT  # 最后回退到项目根目录（custom .pt 的蛋白可能已嵌入）
                    print(f"💡 单独评估模式：{orig_path} 不存在，使用项目根目录: {args.protein_root}")
                    print(f"   （custom .pt 的蛋白路径通常已嵌入文件，评估脚本会优先使用）")
            else:
                print(f"❌ 错误: 蛋白质数据根目录不存在: {args.protein_root}")
                sys.exit(1)
        
        if not EVAL_SCRIPT.exists():
            print(f"❌ 错误: 评估脚本不存在: {EVAL_SCRIPT}")
            sys.exit(1)
    elif args.protein_root:
        # 如果只生成模式下提供了protein_root，也验证一下
        args.protein_root = Path(args.protein_root)
    
    if not use_eval_only_mode and not args.config.exists():
        print(f"❌ 错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    if not use_eval_only_mode and not SCRIPT.exists():
        print(f"❌ 错误: 采样脚本不存在: {SCRIPT}")
        sys.exit(1)
    
    # 解析 data_id 列表（--data_ids 优先于 --start/--end）
    if args.data_ids and not use_custom_mode:
        try:
            data_id_list = [int(x.strip()) for x in args.data_ids.split(',') if x.strip()]
            if not data_id_list:
                print(f"❌ 错误: --data_ids 格式无效，请提供逗号分隔的整数列表（如 5,10,20）")
                sys.exit(1)
        except ValueError:
            print(f"❌ 错误: --data_ids 格式无效，请提供逗号分隔的整数列表（如 5,10,20）")
            sys.exit(1)
    else:
        data_id_list = list(range(args.start, args.end + 1)) if not use_custom_mode else []
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 自动清理残留进程（默认关闭，需显式启用 --auto-cleanup）
    if args.auto_cleanup:
        # --auto-cleanup: 自动清理残留进程（不询问，直接清理）
        interactive = False  # 不询问，直接清理
        cleaned_count = cleanup_zombie_processes(interactive=interactive)
        if cleaned_count > 0:
            print(f"✓ 已清理 {cleaned_count} 个残留进程，等待资源释放...")
            time.sleep(2)  # 等待资源释放
            print()
    else:
        # 默认不清理，仅打印提示
        print("提示: 检测到 --auto-cleanup 未启用，跳过残留进程清理")
        print("      如需清理残留进程，请添加 --auto-cleanup 参数")
        print()
    
    # 解析GPU配置
    if args.gpus:
        try:
            gpu_ids = parse_gpu_ids(args.gpus)
            if not gpu_ids:
                # 如果指定了"all"但检测不到GPU，尝试使用nvidia-smi检测
                if args.gpus.lower() == 'all':
                    try:
                        result = subprocess.run(
                            ['nvidia-smi', '--list-gpus'],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.returncode == 0 and result.stdout.strip():
                            num_gpus = len(result.stdout.strip().split('\n'))
                            gpu_ids = list(range(num_gpus))
                            print(f"⚠️  警告: PyTorch检测不到CUDA，但nvidia-smi检测到 {num_gpus} 个GPU")
                            print(f"   将使用GPU IDs: {gpu_ids}")
                            print(f"   子进程在设置CUDA_VISIBLE_DEVICES后可能会检测到CUDA")
                        else:
                            print(f"⚠️  警告: 未找到可用的GPU（PyTorch和nvidia-smi都检测不到）")
                            print(f"   将继续运行，但任务可能会失败")
                            print(f"   如果所有任务都失败，请检查Docker容器的GPU配置")
                            # 使用默认GPU列表，让程序继续运行
                            gpu_ids = DEFAULT_GPU_IDS
                    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                        print(f"⚠️  警告: 未找到可用的GPU")
                        print(f"   将继续运行，但任务可能会失败")
                        print(f"   如果所有任务都失败，请检查Docker容器的GPU配置")
                        # 使用默认GPU列表，让程序继续运行
                        gpu_ids = DEFAULT_GPU_IDS
                else:
                    print(f"❌ 错误: 未找到可用的GPU")
                    print(f"   指定的GPU: {args.gpus}")
                    sys.exit(1)
        except ValueError as e:
            print(f"❌ 错误: GPU ID格式无效: {e}")
            print(f"   支持的格式: '0,1,2,3', '0-3', '0,2-4,6', 'all'")
            sys.exit(1)
    else:
        gpu_ids = DEFAULT_GPU_IDS
    
    # 验证GPU是否可用
    # 注意：主进程可能检测不到CUDA（因为环境变量或驱动问题），
    # 但子进程在设置CUDA_VISIBLE_DEVICES后可能可以检测到
    # 所以这里只做警告，不阻止运行
    cuda_available_in_main = False
    if torch is not None and torch.cuda.is_available():
        cuda_available_in_main = True
        available_gpus = get_available_gpus()
        invalid_gpus = [gpu_id for gpu_id in gpu_ids if gpu_id not in available_gpus]
        if invalid_gpus:
            print(f"⚠️  警告: 以下GPU在主进程中不可用: {invalid_gpus}")
            print(f"   主进程检测到的可用GPU: {available_gpus}")
            print(f"   将继续尝试使用指定的GPU（子进程可能会检测到）")
    else:
        # 尝试使用 nvidia-smi 作为备用检测方法
        try:
            result = subprocess.run(
                ['nvidia-smi', '--list-gpus'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                num_gpus_detected = len(result.stdout.strip().split('\n'))
                print(f"⚠️  警告: PyTorch检测不到CUDA，但nvidia-smi检测到 {num_gpus_detected} 个GPU")
                print(f"   将继续运行，子进程在设置CUDA_VISIBLE_DEVICES后可能会检测到CUDA")
                print(f"   如果所有任务都失败，请检查:")
                print(f"   1. Docker容器是否正确配置了GPU支持 (--gpus all)")
                print(f"   2. PyTorch是否正确安装了CUDA版本")
                print(f"   3. CUDA运行时库是否正确安装")
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            print("⚠️  警告: CUDA不可用（PyTorch和nvidia-smi都检测不到）")
            print("   将继续运行，但任务可能会失败")
            print("   如果所有任务都失败，请检查:")
            print("   1. Docker容器是否正确配置了GPU支持 (--gpus all)")
            print("   2. NVIDIA驱动是否正确安装")
            print("   3. PyTorch是否正确安装了CUDA版本")
    
    num_gpus = len(gpu_ids)
    num_cpu_cores = args.num_cpu_cores if args.num_cpu_cores is not None else DEFAULT_NUM_CPU_CORES
    
    # 确保batchsummary目录存在
    BATCHSUMMARY_DIR = REPO_ROOT / 'batchsummary'
    BATCHSUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    
    batch_start_time = time.time()
    
    # ========== 单独评估 .pt 文件模式 ==========
    if use_eval_only_mode:
        pt_files = []
        
        # 1. 处理 --pt_dir 目录扫描
        if args.pt_dir is not None:
            pt_dir = Path(args.pt_dir)
            if not pt_dir.is_absolute():
                pt_dir = (REPO_ROOT / pt_dir).resolve()
            if not pt_dir.exists():
                print(f"❌ 错误: 目录不存在: {pt_dir}")
                sys.exit(1)
            if not pt_dir.is_dir():
                print(f"❌ 错误: 指定路径不是目录: {pt_dir}")
                sys.exit(1)
            
            # 扫描目录顶层，按 glob 匹配 .pt（默认 result_*.pt）
            _glob_pat = (args.pt_dir_glob or 'result_*.pt').strip() or 'result_*.pt'
            pt_files_from_dir = list(pt_dir.glob(_glob_pat))
            if not pt_files_from_dir:
                print(f"⚠️ 警告: 目录 {pt_dir} 中未找到匹配 {_glob_pat!r} 的文件")
            else:
                pt_files.extend(sorted(pt_files_from_dir))
                print(f"📁 从目录扫描到 {len(pt_files_from_dir)} 个 .pt 文件")
        
        # 2. 处理 --pt_file 文件列表
        if args.pt_file is not None:
            pt_paths_raw = [p.strip() for p in args.pt_file.split(',') if p.strip()]
            for p in pt_paths_raw:
                path = Path(p)
                if not path.is_absolute():
                    path = (REPO_ROOT / p).resolve()
                if not path.exists():
                    print(f"❌ 错误: .pt 文件不存在: {path}")
                    sys.exit(1)
                pt_files.append(path)
        
        # 去重并验证
        pt_files = list(dict.fromkeys(pt_files))  # 保持顺序去重
        if not pt_files:
            print("❌ 错误: 未找到任何有效的 .pt 文件用于评估")
            sys.exit(1)

        if args.molecular_repair == 'targetdiff_baseline_refine':
            cfg_eval = Path(args.config) if args.config else CONFIG
            if not cfg_eval.is_file():
                print(f"错误: targetdiff_baseline_refine 需要有效的 --config: {cfg_eval}")
                sys.exit(1)
            refine_gpu = gpu_ids[0]
            print(
                f"\n单独评估：对 {len(pt_files)} 个 .pt 执行 targetdiff_baseline_refine "
                f"（GPU {refine_gpu}，与 sampling.yml 中 sample.targetdiff_baseline_refine 一致）\n"
            )
            _rpool = None
            try:
                _rpool = Pool(processes=1)
                _ref_tasks = [
                    (str(p), str(cfg_eval), str(args.protein_root), refine_gpu) for p in pt_files
                ]
                _refined_paths = _rpool.map(process_baseline_refine_standalone_task, _ref_tasks)
                pt_files = [Path(x) for x in _refined_paths]
            finally:
                if _rpool:
                    _rpool.close()
                    _rpool.join()
        
        cst = timezone(timedelta(hours=8))
        timestamp = datetime.now(cst).strftime('%Y%m%d_%H%M%S')
        config_params = generate_config_params_string(args.config) if args.config else ''
        default_excel = (
            BATCHSUMMARY_DIR / f'batch_evaluation_summary_{timestamp}_{config_params}.csv' if config_params
            else BATCHSUMMARY_DIR / f'batch_evaluation_summary_{timestamp}.csv'
        )
        excel_file = resolve_eval_only_excel_path(args.excel_file, BATCHSUMMARY_DIR, default_excel)
        if args.excel_file and not Path(args.excel_file).is_absolute():
            print(f"📊 批次汇总相对路径已解析到 batchsummary: {excel_file}")
        
        print(f"\n{'='*60}")
        print(f"单独评估模式：仅评估 .pt 文件（不采样）")
        print(f"{'='*60}")
        print(f".pt 文件数: {len(pt_files)}")
        for pf in pt_files:
            print(f"  - {pf.name}")
        print(f"蛋白质数据根目录: {args.protein_root}")
        print(f"原子模式: {args.atom_mode}")
        print(f"对接强度: {args.exhaustiveness}")
        print(f"分子修复: {args.molecular_repair}")
        print(f"{'='*60}\n")
        
        cores_per_task = getattr(args, 'cores_per_task', 1)
        _eval_force_mmff = args.molecular_repair == 'mmff'
        _eval_mmff_iters = args.mmff_max_iters if _eval_force_mmff else None
        max_parallel = max(1, num_cpu_cores // max(1, cores_per_task))
        eval_workers = min(max_parallel, len(pt_files))
        print(f"并行评估: {eval_workers} 个进程（CPU 核心 {num_cpu_cores}，每任务 {cores_per_task} 核）\n")
        print(
            f'评估 Vina：--eval-vina-modes={args.eval_vina_modes!r} '
            f'（auto 时对每个 .pt 单独判断是否为 Prudent）\n'
        )
        if getattr(args, 'eval_show_output', False):
            print("提示: 已启用 --eval-show-output，子进程输出直接打印（多任务并行时可能交错）。")
        if getattr(args, 'eval_save_log', False):
            print("提示: 已启用 --eval-save-log，完整日志写入各 eval_* 目录下的 evaluate_subprocess.log。")
        if not getattr(args, 'eval_show_output', False) and not getattr(args, 'eval_save_log', False):
            print("提示: 默认仅打印 [评估] 起止行；要看 evaluate 详细过程请加 --eval-save-log或 --eval-show-output。\n")

        manager_eval = Manager()
        excel_lock_eval = manager_eval.Lock()
        evaluation_tasks = []
        for pt_path in pt_files:
            data_id = extract_data_id_from_pt_filename(pt_path)
            _vm_pt = resolve_eval_vina_modes(
                args.eval_vina_modes,
                config_path=args.config,
                pt_path=pt_path,
            )
            evaluation_tasks.append((
                data_id, str(pt_path), str(args.protein_root),
                args.atom_mode, args.exhaustiveness,
                str(excel_file), batch_start_time, excel_lock_eval, cores_per_task,
                args.save_intermediate_interval,
                getattr(args, 'eval_show_output', False),
                getattr(args, 'eval_save_log', False),
                _eval_force_mmff,
                _eval_mmff_iters,
                _vm_pt,
            ))

        all_results = []
        if evaluation_tasks:
            eval_pool = None
            try:
                eval_pool = Pool(processes=eval_workers)
                all_results = eval_pool.map(process_evaluation_task, evaluation_tasks)
            finally:
                if eval_pool:
                    eval_pool.close()
                    eval_pool.join()
        
        molform_out_parents = list({Path(p).resolve().parent for p in pt_files})
        molecule_records, summary_stats, pocket_stats_list = collect_all_evaluation_results(
            all_results, batch_start_time, extra_eval_roots=molform_out_parents
        )
        
        if excel_file and pd is not None:
            save_molecules_to_excel(excel_file, molecule_records, summary_stats, batch_start_time, pocket_stats_list, config_file=str(args.config))
            print(f"✅ 结果已保存到: {excel_file}")
        
        print(f"\n{'='*60}")
        print(f"单独评估模式处理完成")
        print(f"{'='*60}")
        sys.exit(0)
    
    # ========== 自定义文件模式：单次采样+评估 ==========
    if use_custom_mode:
        pocket_id = 'custom'
        gpu_id = gpu_ids[0]
        
        cst = timezone(timedelta(hours=8))
        timestamp = datetime.now(cst).strftime('%Y%m%d_%H%M%S')
        config_params = generate_config_params_string(args.config)
        excel_file = normalize_batch_summary_path(Path(args.excel_file)) if args.excel_file else (
            BATCHSUMMARY_DIR / f'batch_evaluation_summary_{timestamp}_{config_params}.csv' if config_params
            else BATCHSUMMARY_DIR / f'batch_evaluation_summary_{timestamp}.csv'
        )
        
        print(f"\n{'='*60}")
        print(f"自定义文件模式：单次采样+评估")
        print(f"{'='*60}")
        # 检测是否已进入优化阶段
        try:
            config_path = Path(args.config) if args.config else CONFIG
            if config_path.exists() and yaml is not None:
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f)
                if cfg and cfg.get('sample', {}).get('optimization', {}).get('enable', False):
                    after_dyn = cfg.get('sample', {}).get('optimization', {}).get('after_dynamic', False)
                    print(f"✅ 已进入优化阶段 (optimization.enable=true)" + (" [dynamic后再优化]" if after_dyn else ""))
        except Exception:
            pass
        print(f"蛋白文件: {args.protein_path}")
        print(f"配体文件: {args.ligand_path or '无（仅蛋白）'}")
        print(f"口袋裁剪半径: {args.pocket_radius}Å" + (" (需提供配体)" if not args.ligand_path else ""))
        print(f"使用GPU: {gpu_id}")
        print(f"{'='*60}\n")
        
        if args.skip_existing:
            pt_file = find_latest_result_file(pocket_id)
            if pt_file and pt_file.exists():
                print(f"⏭️  跳过已存在的文件: {pt_file}")
                sample_success, pt_file_str, sample_msg = True, str(pt_file), "文件已存在"
            else:
                sample_success, pt_file_str, sample_msg = run_single_sample_custom(
                    args.protein_path, args.config, gpu_id,
                    ligand_path=args.ligand_path,
                    use_dataset_for_pocket=args.use_dataset_for_pocket,
                    pocket_radius=getattr(args, 'pocket_radius', 10.0)
                )
        else:
            sample_success, pt_file_str, sample_msg = run_single_sample_custom(
                args.protein_path, args.config, gpu_id,
                ligand_path=args.ligand_path,
                use_dataset_for_pocket=args.use_dataset_for_pocket,
                pocket_radius=getattr(args, 'pocket_radius', 10.0)
            )
        
        if not sample_success or pt_file_str is None:
            print(f"采样失败: {sample_msg}")
            sys.exit(1)
        
        if args.molecular_repair == 'targetdiff_baseline_refine':
            _crp = None
            try:
                _crp = Pool(processes=1)
                pt_file_str = _crp.apply(
                    process_baseline_refine_standalone_task,
                    ((pt_file_str, str(args.config), str(args.protein_root), gpu_id),),
                )
            finally:
                if _crp:
                    _crp.close()
                    _crp.join()
        
        if args.sample_only:
            print(f"\n自定义模式采样完成: {pt_file_str}")
            sys.exit(0)
        
        cores_per_task = getattr(args, 'cores_per_task', 1)
        _vm_custom = resolve_eval_vina_modes(
            getattr(args, 'eval_vina_modes', 'auto'),
            config_path=args.config,
            pt_path=Path(pt_file_str),
        )
        eval_success, eval_msg, eval_output_dir = run_single_evaluation(
            pt_file_str, args.protein_root, pocket_id,
            args.atom_mode, args.exhaustiveness, cores_per_task=cores_per_task,
            save_intermediate_interval=args.save_intermediate_interval,
            eval_show_output=getattr(args, 'eval_show_output', False),
            eval_save_subprocess_log=getattr(args, 'eval_save_log', False),
            force_mmff_minimize=args.molecular_repair == 'mmff',
            mmff_max_iters=args.mmff_max_iters if args.molecular_repair == 'mmff' else None,
            vina_modes=_vm_custom,
        )
        all_results = [(pocket_id, sample_success, sample_msg, None, pt_file_str, eval_output_dir)]
        molecule_records, summary_stats, pocket_stats_list = collect_all_evaluation_results(all_results, batch_start_time)
        
        if excel_file and pd is not None:
            save_molecules_to_excel(excel_file, molecule_records, summary_stats, batch_start_time, pocket_stats_list, config_file=str(args.config))
            print(f"✅ 结果已保存到: {excel_file}")
        
        print(f"\n{'='*60}")
        print(f"自定义模式处理完成")
        print(f"{'='*60}")
        sys.exit(0)
    # 使用本地时区（CST，UTC+8）生成时间戳
    cst = timezone(timedelta(hours=8))
    timestamp = datetime.now(cst).strftime('%Y%m%d_%H%M%S')
    
    # 生成配置参数字符串
    config_params = generate_config_params_string(args.config)
    
    if args.excel_file:
        excel_file = normalize_batch_summary_path(Path(args.excel_file))
    else:
        # 原有文件名格式
        base_filename = f'batch_evaluation_summary_{timestamp}'
        # 如果成功提取了配置参数，则添加到文件名中
        if config_params:
            excel_file = BATCHSUMMARY_DIR / f'{base_filename}_{config_params}.csv'
        else:
            excel_file = BATCHSUMMARY_DIR / f'{base_filename}.csv'
    
    # 打印配置信息
    print(f"\n{'='*60}")
    if args.sample_only:
        print(f"批量采样配置（只生成模式）")
    else:
        print(f"批量采样和评估配置（并行版本）")
    print(f"{'='*60}")
    print(f"数据ID: {data_id_list}")
    print(f"并行配置:")
    print(f"  - GPU数量: {num_gpus} (GPU IDs: {gpu_ids})")
    if not args.sample_only:
        cores_per_task = getattr(args, 'cores_per_task', 1)
        print(f"  - CPU核心数: {num_cpu_cores}")
        print(f"  - 每任务核心数: {cores_per_task} (并行任务数={num_cpu_cores // max(1, cores_per_task)})")
    print(f"配置文件: {args.config}")
    if not args.sample_only:
        print(f"蛋白质数据根目录: {args.protein_root}")
        print(f"评估结果保存位置: {OUTPUT_DIR}")
        print(f"原子模式: {args.atom_mode}")
        print(f"对接强度: {args.exhaustiveness}")
    else:
        print(f"输出目录: {OUTPUT_DIR}")
    print(f"跳过已存在: {args.skip_existing}")
    print(f"分子修复: {args.molecular_repair}")
    if not args.sample_only:
        print(f"批次汇总文件: {excel_file}")
    print(f"{'='*60}\n")
    
    # 创建Manager用于进程间共享的锁
    manager = Manager()
    excel_lock = manager.Lock()
    
    total = len(data_id_list)
    start_time = time.time()
    
    # ========== 第一阶段：采样（4个GPU并行） ==========
    print(f"\n{'='*60}")
    print(f"第一阶段：并行采样")
    print(f"{'='*60}")
    print(f"数据ID: {data_id_list} (共 {total} 个)")
    print(f"使用 {num_gpus} 个GPU并行采样 (GPU IDs: {gpu_ids})")
    print(f"GPU列表详情: {gpu_ids}")
    print(f"{'='*60}\n")
    
    # 准备采样任务列表
    # 按GPU分组任务，确保每个GPU的任务均匀分布
    tasks_by_gpu = {gpu_id: [] for gpu_id in gpu_ids}
    for i, data_id in enumerate(data_id_list):
        # 轮询分配GPU（确保正确轮询）
        gpu_id = gpu_ids[i % num_gpus]
        tasks_by_gpu[gpu_id].append((
            data_id,
            gpu_id,
            str(args.config),
            args.skip_existing,
            str(args.protein_root) if getattr(args, 'protein_root', None) else '',
            getattr(args, 'molecular_repair', 'none') or 'none',
        ))
    
    # 交错排列任务，确保所有GPU的任务均匀分布
    # 这样pool.map()处理时，前几个任务会分配给不同的GPU
    sampling_tasks = []
    max_tasks_per_gpu = max(len(tasks) for tasks in tasks_by_gpu.values())
    for i in range(max_tasks_per_gpu):
        for gpu_id in gpu_ids:
            if i < len(tasks_by_gpu[gpu_id]):
                sampling_tasks.append(tasks_by_gpu[gpu_id][i])
    
    # 验证GPU分配（调试信息）
    gpu_distribution = {}
    for data_id, gpu_id, *_ in sampling_tasks:
        if gpu_id not in gpu_distribution:
            gpu_distribution[gpu_id] = []
        gpu_distribution[gpu_id].append(data_id)
    
    print(f"GPU分配验证:")
    for gpu_id in sorted(gpu_distribution.keys()):
        count = len(gpu_distribution[gpu_id])
        print(f"  GPU {gpu_id}: {count} 个任务 (data_ids: {gpu_distribution[gpu_id][:10]}{'...' if count > 10 else ''})")
    print(f"任务列表前10个: {[(t[0], t[1]) for t in sampling_tasks[:10]]}")
    
    # 验证前N个任务是否分配给所有GPU（N=GPU数量）
    if len(sampling_tasks) >= num_gpus:
        first_n_tasks = [(t[0], t[1]) for t in sampling_tasks[:num_gpus]]
        first_n_gpus = [gpu_id for _, gpu_id in first_n_tasks]
        unique_gpus_in_first_n = set(first_n_gpus)
        print(f"前{num_gpus}个任务的GPU分配: {first_n_gpus}")
        print(f"前{num_gpus}个任务覆盖的GPU: {sorted(unique_gpus_in_first_n)} (期望: {sorted(gpu_ids)})")
        if len(unique_gpus_in_first_n) < num_gpus:
            print(f"⚠️  警告: 前{num_gpus}个任务没有覆盖所有GPU！")
            missing_gpus = set(gpu_ids) - unique_gpus_in_first_n
            print(f"   缺失的GPU: {sorted(missing_gpus)}")
    print()
    
    # 检测是否已进入优化阶段（config 中 optimization.enable=true）
    try:
        config_path = Path(args.config) if args.config else CONFIG
        if config_path.exists() and yaml is not None:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            if cfg and cfg.get('sample', {}).get('optimization', {}).get('enable', False):
                after_dyn = cfg.get('sample', {}).get('optimization', {}).get('after_dynamic', False)
                print(f"✅ 已进入优化阶段 (optimization.enable=true)" + (" [dynamic后再优化]" if after_dyn else ""))
                print()
    except Exception:
        pass
    
    sampling_start_time = time.time()
    
    # 采样阶段：限制为GPU数量的进程（每个GPU一个）
    # 确保进程数等于GPU数量，以便充分利用所有GPU
    pool_processes = num_gpus
    print(f"创建进程池: {pool_processes} 个工作进程 (对应 {num_gpus} 个GPU)")
    
    # 使用按GPU分组的方式，确保每个GPU都有独立的工作进程
    # 方法：为每个GPU创建一个独立的进程池，每个进程池只处理该GPU的任务
    all_sampling_results = []
    
    try:
        # 为每个GPU创建独立的进程池
        pools = {}
        async_results = {}
        
        for gpu_id in gpu_ids:
            # 获取该GPU的所有任务
            gpu_tasks = [task for task in sampling_tasks if task[1] == gpu_id]
            if not gpu_tasks:
                continue
            
            print(f"GPU {gpu_id}: {len(gpu_tasks)} 个任务，创建独立进程池...")
            # 为每个GPU创建一个工作进程（因为每个GPU只需要一个进程）
            pool = Pool(processes=1)
            pools[gpu_id] = pool
            # 异步执行该GPU的所有任务
            async_results[gpu_id] = pool.map_async(process_sampling_task, gpu_tasks)
        
        # 等待所有GPU的任务完成
        print(f"等待所有GPU任务完成...")
        for gpu_id in gpu_ids:
            if gpu_id in async_results:
                try:
                    gpu_results = async_results[gpu_id].get()
                    all_sampling_results.extend(gpu_results)
                    print(f"GPU {gpu_id} 完成: {len(gpu_results)} 个任务")
                except Exception as e:
                    print(f"GPU {gpu_id} 执行出错: {e}")
        
        # 按data_id排序结果，保持原始顺序
        sampling_results = sorted(all_sampling_results, key=lambda x: x[0])
    except KeyboardInterrupt:
        print("\n⚠️  收到中断信号，正在清理采样进程...")
        for gpu_id, pool in pools.items():
            if pool:
                pool.terminate()  # 强制终止所有工作进程
                pool.join()       # 等待进程完全退出
        raise
    except Exception as e:
        print(f"\n⚠️  采样阶段发生异常: {e}")
        for gpu_id, pool in pools.items():
            if pool:
                pool.terminate()
                pool.join()
        raise
    finally:
        # 清理所有进程池
        for gpu_id, pool in pools.items():
            if pool:
                pool.close()  # 关闭进程池，不再接受新任务
                pool.join()   # 等待所有工作进程完成
    
    sampling_time = time.time() - sampling_start_time
    
    # 统计采样结果
    sampling_success = [r for r in sampling_results if r[1]]
    sampling_fail = [r for r in sampling_results if not r[1]]
    
    print(f"\n{'='*60}")
    print(f"采样阶段完成")
    print(f"{'='*60}")
    print(f"成功: {len(sampling_success)}")
    print(f"失败: {len(sampling_fail)}")
    print(f"耗时: {sampling_time:.2f} 秒 ({sampling_time/60:.2f} 分钟)")
    print(f"{'='*60}\n")
    
    # 如果启用只生成模式，跳过评估阶段
    if args.sample_only:
        print(f"\n{'='*60}")
        print(f"只生成模式：跳过评估阶段")
        print(f"{'='*60}")
        print(f"采样成功的任务数: {len(sampling_success)}")
        print(f"生成的pt文件位置: {OUTPUT_DIR}")
        print(f"{'='*60}\n")
        
        # 只记录采样结果
        all_results = []
        for r in sampling_success:
            data_id, success, pt_file, msg = r
            all_results.append((data_id, True, "采样成功", None, pt_file, None))
        for r in sampling_fail:
            data_id, success, pt_file, msg = r
            all_results.append((data_id, False, msg, None, pt_file, None))
        
        evaluation_time = 0
    else:
        # ========== 第二阶段：评估（64个CPU核心并行） ==========
        print(f"\n{'='*60}")
        print(f"第二阶段：并行评估")
        print(f"{'='*60}")
        print(f"待评估任务数: {len(sampling_success)}")
        cores_per_task = getattr(args, 'cores_per_task', 1)
        max_parallel = max(1, num_cpu_cores // cores_per_task)
        print(f"使用最多 {max_parallel} 个并行任务（每任务 {cores_per_task} 核，共 {num_cpu_cores} 核）")
        print(f"{'='*60}\n")

        _vm_batch = resolve_eval_vina_modes(
            getattr(args, 'eval_vina_modes', 'auto'),
            config_path=args.config,
            pt_path=None,
        )
        _ev_src = getattr(args, 'eval_vina_modes', 'auto')
        if _vm_batch == 'none':
            print(
                '提示: 当前 sampling 为 Prudent（或等价）→ 评估使用 --vina-modes none，'
                '不重跑 AutoDock Vina，仅用 .pt 中 Prudent 对接分并完成重建与理化。\n'
            )
        elif _vm_batch is None and isinstance(_ev_src, str) and _ev_src.strip().lower() == 'auto':
            print('提示: 非 Prudent 采样 → 评估使用 evaluate 默认三种 Vina 模式。\n')
        
        # 准备评估任务列表（只评估采样成功的）
        cores_per_task = getattr(args, 'cores_per_task', 1)
        _batch_eval_force_mmff = args.molecular_repair == 'mmff'
        _batch_eval_mmff_iters = args.mmff_max_iters if _batch_eval_force_mmff else None
        evaluation_tasks = []
        for r in sampling_success:
            data_id, success, pt_file, msg = r
            evaluation_tasks.append((
                data_id, pt_file, str(args.protein_root),
                args.atom_mode, args.exhaustiveness,
                str(excel_file), batch_start_time, excel_lock, cores_per_task,
                args.save_intermediate_interval,
                getattr(args, 'eval_show_output', False),
                getattr(args, 'eval_save_log', False),
                _batch_eval_force_mmff,
                _batch_eval_mmff_iters,
                _vm_batch,
            ))
        
        # 对于采样失败的任务，也记录到结果中
        all_results = []
        for r in sampling_fail:
            data_id, success, pt_file, msg = r
            all_results.append((data_id, False, msg, None, pt_file, None))
        
        evaluation_time = 0
        if evaluation_tasks:
            evaluation_start_time = time.time()
            
            # 评估阶段：使用指定数量的CPU核心并行
            # 每个任务分配 cores_per_task 个核心，并行任务数 = num_cpu_cores // cores_per_task
            cores_per_task = getattr(args, 'cores_per_task', 1)
            max_parallel = max(1, num_cpu_cores // cores_per_task)
            max_eval_workers = min(max_parallel, len(evaluation_tasks))
            
            eval_pool = None
            try:
                eval_pool = Pool(processes=max_eval_workers)
                evaluation_results = eval_pool.map(process_evaluation_task, evaluation_tasks)
            except KeyboardInterrupt:
                print("\n⚠️  收到中断信号，正在清理评估进程...")
                if eval_pool:
                    eval_pool.terminate()
                    eval_pool.join()
                raise
            except Exception as e:
                print(f"\n⚠️  评估阶段发生异常: {e}")
                if eval_pool:
                    eval_pool.terminate()
                    eval_pool.join()
                raise
            finally:
                if eval_pool:
                    eval_pool.close()
                    eval_pool.join()
            
            evaluation_time = time.time() - evaluation_start_time
            
            # 合并评估结果
            all_results.extend(evaluation_results)
            
            print(f"\n{'='*60}")
            print(f"评估阶段完成")
            print(f"{'='*60}")
            print(f"成功: {sum(1 for r in evaluation_results if r[1])}")
            print(f"失败: {sum(1 for r in evaluation_results if not r[1])}")
            print(f"耗时: {evaluation_time:.2f} 秒 ({evaluation_time/60:.2f} 分钟)")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"评估阶段：无任务（所有采样均失败）")
            print(f"{'='*60}\n")
    
    # JSD 评估：对本次批量生成的所有 .pt 文件计算 JSD（无 MMFF 优化）
    pt_files_this_batch = [str(r[2]) for r in sampling_success if r[2]]  # r[2] = pt_file
    if pt_files_this_batch and JSD_EVAL_SCRIPT.exists():
        print(f"\n{'='*60}")
        print(f"JSD 评估（无对接、无 MMFF）")
        print(f"{'='*60}")
        print(f"本次生成的 .pt 文件数: {len(pt_files_this_batch)}")
        try:
            # 写入临时列表文件，避免命令行过长
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write('\n'.join(pt_files_this_batch))
                pt_list_file = f.name
            try:
                jsd_cmd = [
                    sys.executable,
                    str(JSD_EVAL_SCRIPT),
                    '--pt_list', pt_list_file,
                    '--no_mmff',
                    '--output_dir', str(REPO_ROOT / 'evaljsd'),
                    '--quiet',
                ]
                jsd_proc = subprocess.run(jsd_cmd, cwd=str(REPO_ROOT), timeout=86400)
                if jsd_proc.returncode == 0:
                    print(f"✅ JSD 评估完成，结果保存在 evaljsd/ 下")
                else:
                    print(f"⚠️  JSD 评估退出码: {jsd_proc.returncode}")
            finally:
                try:
                    os.unlink(pt_list_file)
                except OSError:
                    pass
        except subprocess.TimeoutExpired:
            print(f"⚠️  JSD 评估超时")
        except Exception as e:
            print(f"⚠️  JSD 评估异常: {e}")
        print(f"{'='*60}\n")
    
    # 统计最终结果
    success_count = sum(1 for r in all_results if r[1])
    fail_count = sum(1 for r in all_results if not r[1])
    skip_count = 0
    
    # 批量保存所有结果到批次 CSV（只生成模式下跳过）
    if excel_file and not args.sample_only:
        print(f"\n{'='*70}")
        print(f"收集并保存评估结果到批次 CSV...")
        print(f"{'='*70}")
        
        molecule_records, summary_stats, pocket_stats_list = collect_all_evaluation_results(all_results, batch_start_time)
        
        print(f"收集到的分子记录数: {len(molecule_records)}")
        print(f"收集到的口袋统计信息数: {len(pocket_stats_list)}")
        
        if save_molecules_to_excel(excel_file, molecule_records, summary_stats, batch_start_time, pocket_stats_list, config_file=str(args.config)):
            print(f"✅ 成功保存 {len(molecule_records)} 个对接成功分子到 {excel_file}")
            print(f"   统计信息:")
            print(f"     - 应生成分子数: {summary_stats.get('应生成分子数', 0)}")
            print(f"     - 可重建分子数: {summary_stats.get('可重建分子数', 0)}")
            print(f"     - 对接成功分子数: {summary_stats.get('对接成功分子数', 0)}")
            if 'Vina_Dock_平均亲和力' in summary_stats:
                print(f"     - Vina_Dock_平均亲和力: {summary_stats['Vina_Dock_平均亲和力']:.3f} kcal/mol")
            
            # 批量评估完成后，从 batchsummary 文件读取参数并填写到 evaall bestchoice.xlsx
            try:
                # 导入更新函数（sys已在文件顶部导入）
                eval_script_path = REPO_ROOT / 'evaluate_pt_with_correct_reconstruct.py'
                if eval_script_path.exists():
                    # 动态导入模块
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("evaluate_pt_module", eval_script_path)
                    evaluate_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(evaluate_module)
                    
                    # 调用更新函数
                    if hasattr(evaluate_module, 'update_bestchoice_excel_with_params'):
                        evaluate_module.update_bestchoice_excel_with_params(output_dir=None)
            except Exception as e:
                print(f"  ⚠️  更新 evaall bestchoice.xlsx 失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️  批次 CSV 保存失败")
        print(f"{'='*70}\n")
    
    # 打印总结
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    if args.sample_only:
        print(f"批量采样完成（只生成模式）")
    else:
        print(f"批量处理完成（并行版本）")
    print(f"{'='*60}")
    print(f"总计: {total}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
    print(f"  - 采样耗时: {sampling_time:.2f} 秒 ({sampling_time/60:.2f} 分钟)")
    if evaluation_time > 0:
        print(f"  - 评估耗时: {evaluation_time:.2f} 秒 ({evaluation_time/60:.2f} 分钟)")
    print(f"平均每个任务: {elapsed_time/total:.2f} 秒")
    if args.sample_only:
        print(f"📁 生成的pt文件保存在: {OUTPUT_DIR}")
        print(f"   成功生成 {len(sampling_success)} 个pt文件")
    elif excel_file:
        print(f"📊 详细记录已保存至: {excel_file}（及同主名侧车 CSV）")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
