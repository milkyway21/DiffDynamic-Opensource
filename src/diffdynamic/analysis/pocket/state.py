from diffdynamic.io.paths import project_root

"""Shared imports, constants and process configuration for diffdynamic.analysis.pocket."""

import os


import sys


import argparse


import subprocess


import glob


import re


import csv


import shutil


import tempfile


from pathlib import Path


from datetime import datetime


from multiprocessing import Pool, cpu_count


from functools import partial


REPO_ROOT = project_root()


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import numpy as np


try:
    import torch
except ImportError:
    torch = None
    print("⚠️  警告: torch未安装，部分功能可能受限")


try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, DataStructs
    from rdkit.Chem import rdMolDescriptors
except ImportError:
    Chem = None
    AllChem = None
    Descriptors = None
    DataStructs = None
    rdMolDescriptors = None
    print("⚠️  警告: RDKit未安装，请运行: conda install -c conda-forge rdkit")


try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.metrics import silhouette_score, pairwise_distances
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    DBSCAN = None
    KMeans = None
    silhouette_score = None
    pairwise_distances = None
    PCA = None
    TSNE = None
    StandardScaler = None
    HAS_SKLEARN = False
    print(
        "⚠️  警告: scikit-learn未安装，聚类/降维分析将受限。运行: pip install scikit-learn"
    )


try:
    import umap as umap_module

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    umap_module = None


try:
    import pandas as pd
except ImportError:
    pd = None


try:
    import matplotlib

    matplotlib.use("Agg")  # 非交互式后端，支持无头运行
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize, ListedColormap
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    import matplotlib.patheffects as pe

    HAS_MATPLOTLIB = True
    # Use default sans-serif for English labels (avoids font/encoding issues)
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    gridspec = None
    print("⚠️  警告: matplotlib未安装，可视化功能不可用。运行: pip install matplotlib")


try:
    from scipy.stats import gaussian_kde
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial import ConvexHull

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    gaussian_kde = None
    ConvexHull = None


try:
    import diffdynamic.chemistry.reconstruct as reconstruct
    import diffdynamic.data.transforms as trans
    from diffdynamic.evaluation.scoring_func import (
        get_molecule_force_field,
        get_conformer_energies,
        get_chem,
        is_pains,
        obey_lipinski,
        passes_veber,
    )
except ImportError as e:
    print(f"⚠️  导入项目模块失败: {e}")
    reconstruct = None
    trans = None
    get_molecule_force_field = None
    get_conformer_energies = None
    get_chem = None
    is_pains = None
    obey_lipinski = None
    passes_veber = None


BATCH_SCRIPT = REPO_ROOT / "batch_sampleandeval_parallel.py"


OUTPUT_DIR = REPO_ROOT / "outputs"


DEFAULT_NUM_CPU_CORES = 64


VIS_ROOT = REPO_ROOT / "pocket_quality_vis"


EVAL_RECORDS_CSV = "evaluation_records.csv"


EVAL_RECORD_FIELDS = (
    "pocket_id",
    "timestamp",
    "n_molecules",
    "score_a",
    "score_b",
    "score_c",
    "score_d",
    "score_e",
    "score_f",
    "score_g",
    "score_h",
    "s_pocket",
    "s_compatibility",
    "s_ligand",
    "overall_score",
    "overall_label",
    "pt_path",
    "vis_dir",
)


GEO_EPS = 1e-3


LAYER_WEIGHTS_LIGAND = {"A": 0.45, "B": 0.25, "C": 0.30}


LAYER_WEIGHTS_COMPAT = {"D": 0.45, "E": 0.55}


LAYER_WEIGHTS_POCKET = {"F": 0.20, "G": 0.55, "H": 0.25}


_CPK_COLORS = {
    1: "#E0E0E0",  # H  - 浅灰
    6: "#606060",  # C  - 深灰
    7: "#3050F8",  # N  - 蓝
    8: "#FF2010",  # O  - 红
    9: "#90E050",  # F  - 浅绿
    15: "#FF8000",  # P  - 橙
    16: "#FFFF30",  # S  - 黄
    17: "#1FF01F",  # Cl - 绿
    35: "#A62929",  # Br - 棕红
    53: "#940094",  # I  - 紫
}


_ELEMENT_NAMES = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}


IDEA_C_LE_SCORE_LOW = 0.20


IDEA_C_LE_SCORE_HIGH = 0.42


_ANCHOR_NEG = {"ASP", "GLU"}


_ANCHOR_POS = {"LYS", "ARG", "HIP"}


_ANCHOR_ARO = {"PHE", "TYR", "TRP", "HIS", "HID", "HIE", "HIP"}


_ANCHOR_HYDRO = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "PRO", "TRP", "TYR"}


_POCKET_POLAR = {
    "ASP",
    "GLU",
    "LYS",
    "ARG",
    "HIS",
    "HID",
    "HIE",
    "HIP",
    "SER",
    "THR",
    "ASN",
    "GLN",
    "TYR",
    "CYS",
}


IDEA_H_FPOCKET_PROTEIN_FALLBACK_OPTIMAL_MIN = 300.0


IDEA_H_FPOCKET_PROTEIN_FALLBACK_OPTIMAL_MAX = 2200.0


IDEA_H_FPOCKET_PROTEIN_FALLBACK_ZERO_BELOW = 80.0


IDEA_H_FPOCKET_PROTEIN_FALLBACK_ZERO_ABOVE = 4500.0


IDEA_H_VIZ_REF_OPT_MIN = 350.0


IDEA_H_VIZ_REF_OPT_MAX = 750.0


IDEA_H_VIZ_REF_ZERO_BELOW = 80.0


IDEA_H_VIZ_REF_ZERO_ABOVE = 2000.0


IDEA_H_VIZ_X_MAX_DEFAULT = 1500.0


IDEA_H_VIZ_SUBOPT_ORANGE = "#FFD4A8"


IDEA_H_VIZ_SUBOPT_ALPHA = 0.20


_NOTEXT_SUFFIX = "_notext"
