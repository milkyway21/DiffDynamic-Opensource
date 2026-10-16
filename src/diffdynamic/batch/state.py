from diffdynamic.io.paths import project_root

"""Shared imports, constants and process configuration for diffdynamic.batch."""

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


DEFAULT_GPU_IDS = [2, 3, 4, 5]  # 默认使用的GPU ID列表


DEFAULT_NUM_CPU_CORES = 64  # 默认使用的CPU核心数


excel_write_lock = None


REPO_ROOT = project_root()


SCRIPT = REPO_ROOT / "scripts" / "sample_diffusion.py"


CONFIG = REPO_ROOT / "configs" / "sampling.yml"


EVAL_SCRIPT = REPO_ROOT / "evaluate_pt_with_correct_reconstruct.py"


JSD_EVAL_SCRIPT = REPO_ROOT / "eval_jsd_only_100pockets.py"


OUTPUT_DIR = REPO_ROOT / "outputs"


OUTPUT_DIR.mkdir(exist_ok=True)


_env_pocket_xlsx = os.environ.get("POCKET_TIME_EXCEL")


POCKET_TIME_EXCEL = (
    Path(_env_pocket_xlsx).expanduser()
    if _env_pocket_xlsx
    else REPO_ROOT / "model_comparison_data.xlsx"
)
