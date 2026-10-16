from diffdynamic.io.paths import project_root

"""Shared imports, constants and process configuration for diffdynamic.evaluation.molecule."""

import os


import sys


import argparse


import torch


import numpy as np


from pathlib import Path


from datetime import datetime


from tqdm import tqdm


import traceback


import signal


import atexit


import multiprocessing


import pickle


import tempfile


import subprocess


import json


import time


import glob


import re


import shutil


import csv


from diffdynamic.io.identity import extract_protein_id, generate_molecule_id


from types import SimpleNamespace


from concurrent.futures import ThreadPoolExecutor, as_completed


try:
    import pandas as pd
except ImportError:
    pd = None
    _PANDAS_MISSING_MSG = "pandas is required to record evaluation results to Excel. Run `pip install pandas` to enable this feature."
else:
    _PANDAS_MISSING_MSG = ""


try:
    import openpyxl
except ImportError:
    _OPENPYXL_MISSING_MSG = "openpyxl is required to write Excel files. Run `pip install openpyxl` to enable this feature."
else:
    _OPENPYXL_MISSING_MSG = ""


REPO_ROOT = project_root()


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_RDKIT_STRUCTURE_REPAIR_CFG = None


_RECONSTRUCTION_METHOD = "current"


_TRUE_VALUES = {"1", "true", "yes", "on"}


_MEDCHEM_OPTIMIZE_CFG = None


_MEDCHEM_OPTIMIZE_STATE = {
    "loaded": False,
    "fn": None,
    "config": None,
    "repair_config": {},
}


DOCK_TMP_DIR = Path(REPO_ROOT) / "docktmp"


DOCK_TMP_DIR.mkdir(exist_ok=True)


_DEFAULT_VINA_MODES = frozenset(("score_only",))


_ALL_VINA_MODES = frozenset(("dock", "score_only", "minimize"))


DEFAULT_SINGLE_MOL_EVAL_TIMEOUT_SEC = 10800  # 3 小时（原 2h=7200）


try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
except ImportError as e:
    print(f"错误: 无法导入RDKit: {e}")
    print("请安装: conda install -c conda-forge rdkit")
    sys.exit(1)


import diffdynamic.chemistry.reconstruct as reconstruct


import diffdynamic.chemistry.reconstruct_targetdiff_legacy as reconstruct_targetdiff_legacy


import diffdynamic.data.transforms as trans


from diffdynamic.evaluation import scoring_func


from diffdynamic.evaluation.docking_vina import (
    VinaDockingTask,
    resolve_receptor_pdb_for_docking,
)


from diffdynamic.io import config as misc


from diffdynamic.evaluation import eval_atom_type


from diffdynamic.evaluation import eval_bond_length


from diffdynamic.evaluation import analyze


from diffdynamic.evaluation import similarity


from diffdynamic.evaluation.lilly_medchem_rules import evaluate_lilly_medchem_rules


from collections import Counter


_TRUSTED_SAFE_GLOBALS = None
