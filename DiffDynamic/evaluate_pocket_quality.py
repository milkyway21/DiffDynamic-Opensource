#!/usr/bin/env python3
"""
蛋白质口袋质量评估脚本（增强版 v2）

基于扩散模型（DiffSBDD/DiffDynamic）的分子生成输出，综合八个评估维度对蛋白质口袋（结合位点）进行质量评估，
并提供完整的可视化分析套件。

说明：本评分为「生成结果驱动的口袋可药性综合分」，不是与生成无关的纯口袋物理评分。
  A/C/D/H（尤其 FPocket）科学支撑较强；E/G 偏生成质量；B/F 偏采样行为代理。

评估维度：
  想法A：Vina 对接分数与口袋质量
  想法B：原子分布聚类（结合模式收敛性）—— DBSCAN + KMeans；可选按原子子集做 DBSCAN 密度评分
    做法说明（--idea_b_atom_subset）：
      all — 全原子参与 DBSCAN。
      hetero_heavy — 非 H、非 C 的重原子（极性/药效相关，弱化碳骨架主导；**默认**）。
      edge_heavy — 重原子中，非氢重邻居数 ≤ edge_max_heavy_neighbors（默认 2）的原子，
        用图论近似「暴露在外的边缘」。
      hetero_edge — 上两者交集：非碳且偏边缘的重原子，最接近「除碳以外、尤其分子边缘」。
      combined — 综合分：(1−w)×全原子 DBSCAN 分 + w×hetero_edge DBSCAN 分，默认 w=0.5
        （--idea_b_combined_weight）。hetero_edge 点数太少时自动退回只用全原子，并在 message 里说明。
      KMeans 仍只对整条分子的质心，不受子集影响（与原先一致）。
    命令示例：
      python evaluate_pocket_quality.py --pt_file outputs/result_0_xxx.pt \\
        --idea_b_atom_subset hetero_edge --visualize
      python evaluate_pocket_quality.py --pt_file outputs/result_0_xxx.pt \\
        --idea_b_atom_subset combined --idea_b_combined_weight 0.5 --idea_b_edge_max_neighbors 2
  想法C：配体效率 (Ligand Efficiency, LE)
    $LE = -\\Delta G / N_{heavy}$，$\\Delta G$ 取 Vina 亲和力 (kcal/mol，负值有利)，$N_{heavy}$ 为非氢原子数。
    默认假设 Vina 分数与 `molecules_with_pos` 中分子**下标顺序**一致（与 complete_molecules Excel / eval 导出一致）。
    失败对接（vina≥0）不参与 LE。将 LE 均值线性映射到 0–1（参考区间约 0.18–0.45）。
  想法D：药物相似性 (QED/SA/Lipinski/PAINS/Veber)
  想法E：完整分子比例（SMILES 不含 '.' 的单组分分子数 / 应生成或 .pt 槽位数）——生成质量门控
  想法F：分子唯一性（unique_ratio 进分；指纹余弦相异度为参考，不进总分）。
    满分门槛：unique_ratio ≥0.95（约 100 个中 ≥95 个不同 SMILES）；分段映射见 evaluate_idea_f_uniqueness。
  想法G：分子尺寸一致性（MW CV）——生成质量相关
  想法H：口袋体积（几何空腔）。优先 **配体定位 + 蛋白排除** MC（POVME 风格）：
    质心均值作球心、默认 **10 Å** 球内采样；点须不与蛋白重原子 vdW(+1.4Å 探针)重叠，
    且距任一生成配体重原子 ≤3.0 Å。满分带 **350–750** Å³（归零 80 / 2000）。
    无受体 PDB 时回退为旧版「球内配体 vdW 占据」MC（非真实口袋体积，仅作 fallback）。
    Legacy FPocket 体积见 ``evaluate_idea_h_pocket_size_legacy_fpocket``（非默认）。

默认权重（和=1.00）：A0.28 / B0.15 / C0.15 / D0.15 / E0.05 / F0.08 / G0.07 / H0.07
  （F 以 ≥95% unique 为满分，可区分；区分力主要靠 A/C/F）

质量标签阈值（按维）：
  A：high≥0.5 / medium≥0.2
  H：high≥0.8 / medium≥0.5
  其余：high≥0.6 / medium≥0.3
  overall：high≥0.65 / medium≥0.3

可视化模块：
  - 原子分布聚类图（PCA/t-SNE降维 + DBSCAN/KMeans，6子图）
  - Vina + LE 合并图（A_vina_le.png：左/右均为 box+jitter strip+colorbar，等宽；含 _notext）
  - 药物相似性多维散点图（QED vs SA，属性雷达图）
  - 分子指纹相似性热图（Morgan fingerprint；参考多样性用余弦相异度）
  - 分子尺寸分布图（分子量 + 原子数双直方图）
  - 综合雷达图（8 维质量指标，含 LE）

使用方法：
    # 使用已有 .pt 文件评估 + 生成可视化图
    python3 evaluate_pocket_quality.py --pt_file outputs/result_custom_20260321_034536.pt --visualize
    # 想法 H 使用 FPocket 分别算蛋白口袋与配体侧口袋体积（需已安装 fpocket，且在 .pt 同目录下用临时副本运行，避免并行冲突）
    python evaluate_pocket_quality.py --pt_file outputs/result_custom_20260319_001000.pt  --fpocket_protein_pdb shoc2/shoc2.pdb --visualize
    # 仅想法 H 的配体 FPocket 使用外部构象（A–G 仍用 .pt 内分子）
    python3 evaluate_pocket_quality.py --pt_file outputs/result_custom_20260319_001000.pt \\
  --fpocket_protein_pdb shoc2/shoc2.pdb --idea_h_ligand_path shoc2/shoc2ligand.sdf \\
  --idea_e_expected_n_molecules 400 --visualize
    python3 evaluate_pocket_quality.py --pt_file outputs/result_custom_20260319_001000.pt \\
      --fpocket_protein_pdb shoc2/shoc2.pdb --idea_h_ligand_path shoc2/shoc2ligand.sdf --visualize
    # 想法 E：.pt 中 pred_ligand_pos 条数不可靠时，用手动应生成数作分母（成功率=解析到的分子数/N）
    python3 evaluate_pocket_quality.py --pt_file outputs/result_custom_20260319_001000.pt \\
      --idea_e_expected_n_molecules 1000 --visualize
    # 输出目录结构result_custom_20260319_001000.pt：pocket_quality_vis/蛋白质编号_时间戳/A_vina_le.png, B_clustering_dbscan.png, ...
    # 口袋评估记录表：pocket_quality_vis/evaluation_records.csv（每次评估追加一行，含 vis_dir）

    # 批量评估并可视化（支持 CPU 并行，与 batch_sampleandeval_parallel 一致）
    python evaluate_pocket_quality.py --run_batch --start 0 --end 9 --gpus "0" --visualize
    python evaluate_pocket_quality.py --run_batch --start 0 --end 9 --gpus "0" --num_cpu_cores 20 --cores_per_task 10 --visualize

    # 自定义配体文件/目录评估（需 batch 已写出 eval_* 与 complete_molecules_*.xlsx，或指定 --vina_outputs_dir）
    python evaluate_pocket_quality.py --eval_ligands poses.sdf --vina_outputs_dir outputs --visualize
    # 已有 batch 的 outputs（含 eval_custom_* 与 complete_molecules_*.xlsx）
    python evaluate_pocket_quality.py --eval_ligands ./my_poses.sdf \\
      --vina_outputs_dir ./outputs --vina_pocket_id custom --visualize
    # 示例：shoc2 自定义口袋 + 配体（PDB 与 SDF 为两条独立路径，勿拼接）
    python3 evaluate_pocket_quality.py --eval_ligands shoc2/shoc2ligand.sdf \\
      --custom_pocket_pdb shoc2/shoc2.pdb --vina_outputs_dir outputs --visualize
"""

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

REPO_ROOT = Path(__file__).parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

try:
    import torch
except ImportError:
    torch = None
    print("⚠️  警告: torch未安装，部分功能可能受限")


def _torch_load_legacy(path, map_location="cpu"):
    """加载项目内任意 .pt（含自定义类 pickle）。PyTorch 2.6+ 默认 weights_only=True 会失败。"""
    if torch is None:
        raise RuntimeError("需要 torch")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


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
    print("⚠️  警告: scikit-learn未安装，聚类/降维分析将受限。运行: pip install scikit-learn")

# UMAP（可选，降维效果更佳）
try:
    import umap as umap_module
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    umap_module = None

# matplotlib（可视化）
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端，支持无头运行
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

# scipy（KDE 密度估计、凸包体积，可选）
try:
    from scipy.stats import gaussian_kde
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial import ConvexHull
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    gaussian_kde = None
    ConvexHull = None

# 项目模块
try:
    import utils.reconstruct as reconstruct
    import utils.transforms as trans
    from utils.evaluation.scoring_func import (
        get_molecule_force_field, get_conformer_energies,
        get_chem, is_pains, obey_lipinski, passes_veber,
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

BATCH_SCRIPT = REPO_ROOT / 'batch_sampleandeval_parallel.py'
OUTPUT_DIR = REPO_ROOT / 'outputs'

# CPU 并行配置（与 batch_sampleandeval_parallel.py 保持一致）
DEFAULT_NUM_CPU_CORES = 64

# 可视化输出根目录：主文件夹下统一管理，每次实验用时间戳子目录
VIS_ROOT = REPO_ROOT / 'pocket_quality_vis'

# 口袋评估记录表文件名
EVAL_RECORDS_CSV = 'evaluation_records.csv'
EVAL_RECORD_FIELDS = (
    'pocket_id', 'timestamp', 'n_molecules',
    'score_a', 'score_b', 'score_c', 'score_d',
    'score_e', 'score_f', 'score_g', 'score_h',
    's_pocket', 's_compatibility', 's_ligand',
    'overall_score', 'overall_label', 'pt_path', 'vis_dir',
)

# 三层几何平均：层内加权；层间 (Sp*Sc*Sl)^(1/3)
GEO_EPS = 1e-3
LAYER_WEIGHTS_LIGAND = {'A': 0.45, 'B': 0.25, 'C': 0.30}
LAYER_WEIGHTS_COMPAT = {'D': 0.45, 'E': 0.55}
LAYER_WEIGHTS_POCKET = {'F': 0.20, 'G': 0.55, 'H': 0.25}


def _weighted_geo_mean(scores, weights, eps=GEO_EPS):
    """Weighted geometric mean; skip non-finite / missing; renormalize weights."""
    vals, wts = [], []
    for s, w in zip(scores, weights):
        if s is None or w is None or w <= 0:
            continue
        try:
            sf = float(s)
            wf = float(w)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(sf) or not np.isfinite(wf) or wf <= 0:
            continue
        vals.append(max(sf, eps))
        wts.append(wf)
    if not vals:
        return float(eps)
    wts = np.asarray(wts, dtype=np.float64)
    wts = wts / wts.sum()
    vals = np.asarray(vals, dtype=np.float64)
    return float(np.exp(np.sum(wts * np.log(vals))))


def _equal_geo_mean(scores, eps=GEO_EPS):
    """Equal-weight geometric mean (cube root of product when len==3)."""
    vals = []
    for s in scores:
        if s is None:
            continue
        try:
            sf = float(s)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(sf):
            continue
        vals.append(max(sf, eps))
    if not vals:
        return float(eps)
    return float(np.exp(np.mean(np.log(np.asarray(vals, dtype=np.float64)))))


def append_evaluation_record(result, record_path, timestamp=None):
    """
    Append one evaluation record to the pocket evaluation log (CSV).

    Columns: pocket_id, timestamp, n_molecules, score_a..h, overall_score,
    overall_label, pt_path, vis_dir
    """
    record_path = Path(record_path)
    record_path.parent.mkdir(parents=True, exist_ok=True)

    pocket_id = result.get('data_id')
    if pocket_id is None:
        pt_stem = Path(result.get('pt_path', '')).stem
        if pt_stem.startswith('result_') and '_' in pt_stem:
            pocket_id = pt_stem.split('_')[1]
        else:
            pocket_id = pt_stem
    pocket_id = str(pocket_id)

    ts = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')

    def _score(k):
        r = result.get(k, {})
        return f"{r.get('score', 0):.4f}" if r.get('success') else 'N/A'

    row = {
        'pocket_id': pocket_id,
        'timestamp': ts,
        'n_molecules': result.get('n_molecules', 0),
        'score_a': _score('idea_a'),
        'score_b': _score('idea_b'),
        'score_c': _score('idea_c'),
        'score_d': _score('idea_d'),
        'score_e': _score('idea_e'),
        'score_f': _score('idea_f'),
        'score_g': _score('idea_g'),
        'score_h': _score('idea_h'),
        's_pocket': f"{result.get('s_pocket', 0):.4f}" if result.get('s_pocket') is not None else '',
        's_compatibility': (
            f"{result.get('s_compatibility', 0):.4f}"
            if result.get('s_compatibility') is not None else ''
        ),
        's_ligand': f"{result.get('s_ligand', 0):.4f}" if result.get('s_ligand') is not None else '',
        'overall_score': f"{result.get('overall_score', 0):.4f}",
        'overall_label': result.get('overall_label', 'unknown'),
        'pt_path': result.get('pt_path') or result.get('ligand_path') or '',
        'vis_dir': result.get('vis_dir') or '',
    }

    file_exists = record_path.exists()
    if file_exists:
        with open(record_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            old_fields = list(reader.fieldnames or [])
            old_rows = list(reader)
        if 'vis_dir' not in old_fields:
            with open(record_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=EVAL_RECORD_FIELDS)
                writer.writeheader()
                for old in old_rows:
                    writer.writerow({k: old.get(k, '') for k in EVAL_RECORD_FIELDS})
                writer.writerow({k: row.get(k, '') for k in EVAL_RECORD_FIELDS})
            return

    with open(record_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_RECORD_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, '') for k in EVAL_RECORD_FIELDS})


# =============================================================================
# 分子加载与重建
# =============================================================================

def _to_numpy(data):
    """将 torch tensor 或 numpy 转为 numpy"""
    if torch is not None and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.array(data)


def load_pt_file(pt_path):
    """加载 .pt 文件"""
    try:
        data = _torch_load_legacy(pt_path, map_location="cpu")
        return data
    except Exception as e:
        print(f"❌ 加载 .pt 文件失败: {pt_path}: {e}")
        return None


def reconstruct_molecule_from_pt(pos, v, atom_mode='add_aromatic'):
    """从 .pt 中的坐标和原子类型重建 RDKit 分子"""
    if reconstruct is None or trans is None:
        return None
    try:
        pos_array = _to_numpy(pos)
        if pos_array.ndim == 1:
            pos_array = pos_array.reshape(-1, 3)

        v_tensor = torch.tensor(v) if not isinstance(v, torch.Tensor) else v.detach().cpu()
        if v_tensor.dim() > 1:
            v_tensor = v_tensor.argmax(dim=-1)

        atom_numbers = trans.get_atomic_number_from_index(v_tensor, mode=atom_mode)
        aromatic_flags = trans.is_aromatic_from_index(v_tensor, mode=atom_mode)

        mol = reconstruct.reconstruct_from_generated(
            pos_array.tolist(),
            atom_numbers,
            aromatic_flags,
            basic_mode=(atom_mode == 'basic')
        )
        return mol
    except Exception:
        return None


def load_molecules_from_pt(pt_path, atom_mode='add_aromatic', max_mols=None):
    """
    从 .pt 文件加载并重建所有分子

    Returns:
        list: [(mol, pos_array), ...]
    """
    data = load_pt_file(pt_path)
    if data is None:
        return []

    pred_pos = data.get('pred_ligand_pos', [])
    pred_v = data.get('pred_ligand_v', [])

    if not pred_pos or not pred_v:
        return []

    results = []
    for i, (pos, v) in enumerate(zip(pred_pos, pred_v)):
        if max_mols and len(results) >= max_mols:
            break
        mol = reconstruct_molecule_from_pt(pos, v, atom_mode)
        pos_arr = _to_numpy(pos)
        if pos_arr.ndim == 1:
            pos_arr = pos_arr.reshape(-1, 3)
        if mol is not None:
            results.append((mol, pos_arr))

    return results


def _mol_and_pos_from_rdkit_mol(mol):
    """从 RDKit 分子取首构象坐标；若无构象则尝试 Embed。"""
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    if mol.GetNumConformers() < 1 and AllChem is not None:
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
        except Exception:
            return None
    if mol.GetNumConformers() < 1:
        return None
    conf = mol.GetConformer()
    pos = np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32)
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        pos[i, 0], pos[i, 1], pos[i, 2] = p.x, p.y, p.z
    return mol, pos


def load_molecules_from_ligand_paths(ligand_path, max_mols=None):
    """
    从自定义配体文件或目录加载 (mol, pos)，用于无 .pt 的口袋质量评估。

    支持：单个/多个 .sdf、.mol；目录下所有 *.sdf / *.mol。

    Returns:
        list[(mol, pos_array), ...]
    """
    if Chem is None:
        return []
    lp = Path(ligand_path)
    files = []
    if lp.is_dir():
        files = sorted(lp.glob('*.sdf')) + sorted(lp.glob('*.mol'))
    elif lp.is_file():
        files = [lp]
    else:
        return []

    out = []
    for fp in files:
        suf = fp.suffix.lower()
        if suf == '.sdf':
            suppl = Chem.SDMolSupplier(str(fp), sanitize=False, removeHs=False)
            for mol in suppl:
                got = _mol_and_pos_from_rdkit_mol(mol)
                if got:
                    out.append(got)
                    if max_mols and len(out) >= max_mols:
                        return out
        elif suf == '.mol':
            mol = Chem.MolFromMolFile(str(fp), sanitize=False, removeHs=False)
            got = _mol_and_pos_from_rdkit_mol(mol)
            if got:
                out.append(got)
                if max_mols and len(out) >= max_mols:
                    return out
    return out


def resolve_vina_outputs_dir(vina_outputs_dir=None, pt_path=None,
                            ligand_path=None, custom_pocket_pdb=None):
    """
    为想法 A 定位含 eval_* 子目录的 outputs 根目录（与 batch 输出一致）。
    显式 vina_outputs_dir 优先；否则从 .pt 同目录、配体/口袋路径旁推测。
    """
    if vina_outputs_dir:
        p = Path(vina_outputs_dir)
        if p.is_dir():
            return p.resolve()
        return None
    if pt_path:
        pp = Path(pt_path).resolve()
        if pp.is_file():
            return pp.parent
        return None
    candidates = []
    if custom_pocket_pdb:
        base = Path(custom_pocket_pdb).resolve().parent
        candidates.extend([base / 'outputs', base])
    if ligand_path:
        lp = Path(ligand_path).resolve()
        base = lp.parent if lp.is_file() else lp
        candidates.extend([base / 'outputs', base])
    seen = set()
    for c in candidates:
        c = c.resolve()
        if c in seen or not c.is_dir():
            continue
        seen.add(c)
        if list(c.glob('eval_*')):
            return c
    for c in candidates:
        c = c.resolve()
        if c.is_dir():
            return c
    return None


# =============================================================================
# 原子数据收集（供聚类可视化使用）
# =============================================================================

# CPK 颜色方案（原子序数 -> 十六进制颜色）
_CPK_COLORS = {
    1:  '#E0E0E0',   # H  - 浅灰
    6:  '#606060',   # C  - 深灰
    7:  '#3050F8',   # N  - 蓝
    8:  '#FF2010',   # O  - 红
    9:  '#90E050',   # F  - 浅绿
    15: '#FF8000',   # P  - 橙
    16: '#FFFF30',   # S  - 黄
    17: '#1FF01F',   # Cl - 绿
    35: '#A62929',   # Br - 棕红
    53: '#940094',   # I  - 紫
}
_ELEMENT_NAMES = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}


def _collect_atom_data(molecules_with_pos):
    """
    收集所有分子的原子坐标、元素信息和分子质心

    Returns:
        dict:
            'coords'       : np.ndarray (N_total, 3) — 所有原子坐标
            'mol_idx'      : np.ndarray (N_total,)   — 原子所属分子索引
            'atom_nums'    : np.ndarray (N_total,)   — 原子序数
            'centroids'    : np.ndarray (n_mols, 3)  — 每分子质心
            'n_atoms_each' : list[int]               — 每分子原子数
            'n_mols'       : int
    """
    all_coords, mol_indices, atom_nums_list, centroids = [], [], [], []
    n_atoms_each = []

    for mol_i, (mol, pos) in enumerate(molecules_with_pos):
        if pos is None or len(pos) == 0:
            continue
        pos_arr = np.array(pos, dtype=np.float32)
        if pos_arr.ndim == 1:
            pos_arr = pos_arr.reshape(-1, 3)

        n = len(pos_arr)
        all_coords.append(pos_arr)
        mol_indices.extend([mol_i] * n)
        n_atoms_each.append(n)
        centroids.append(pos_arr.mean(axis=0))

        # 必须与 pos 行数 n 一一对应；mol.GetAtoms() 数量常与 n 不一致（H/重建差异等），
        # 否则 idea H 等处 coords 与 radii 长度不齐会广播失败。
        if mol is not None and Chem is not None:
            n_mol = mol.GetNumAtoms()
            if n_mol == n:
                for atom in mol.GetAtoms():
                    atom_nums_list.append(atom.GetAtomicNum())
            elif n_mol > n:
                for i in range(n):
                    atom_nums_list.append(mol.GetAtomWithIdx(i).GetAtomicNum())
            else:
                for atom in mol.GetAtoms():
                    atom_nums_list.append(atom.GetAtomicNum())
                atom_nums_list.extend([6] * (n - n_mol))
        else:
            atom_nums_list.extend([6] * n)  # 默认碳

    if not all_coords:
        return None

    return {
        'coords':       np.vstack(all_coords),
        'mol_idx':      np.array(mol_indices, dtype=np.int32),
        'atom_nums':    np.array(atom_nums_list, dtype=np.int32),
        'centroids':    np.array(centroids, dtype=np.float32),
        'n_atoms_each': n_atoms_each,
        'n_mols':       len(centroids),
    }


def _build_idea_b_coord_mask(molecules_with_pos, subset_mode, edge_max_heavy_neighbors=2):
    """
    与 _collect_atom_data 相同的分子遍历顺序，生成与拼接后 coords 对齐的布尔掩码。

    subset_mode:
        hetero_heavy — 非氢非碳重原子（药效/极性位点）
        edge_heavy   — 重原子且非氢邻居数 <= edge_max_heavy_neighbors（近似分子表面/边缘）
        hetero_edge  — 同时满足 hetero 与 edge（非碳边缘重原子，默认推荐用于结合模式）
    """
    if subset_mode not in ('hetero_heavy', 'edge_heavy', 'hetero_edge'):
        return None
    masks = []
    max_nn = int(edge_max_heavy_neighbors)
    for mol, pos in molecules_with_pos:
        if pos is None or len(pos) == 0:
            continue
        pos_arr = np.asarray(pos, dtype=np.float32)
        if pos_arr.ndim == 1:
            pos_arr = pos_arr.reshape(-1, 3)
        n = len(pos_arr)
        if mol is None or Chem is None or mol.GetNumAtoms() != n:
            masks.append(np.ones(n, dtype=bool))
            continue
        m = np.zeros(n, dtype=bool)
        for i, atom in enumerate(mol.GetAtoms()):
            z = atom.GetAtomicNum()
            hn = sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() > 1)
            if subset_mode == 'hetero_heavy':
                m[i] = z > 1 and z != 6
            elif subset_mode == 'edge_heavy':
                m[i] = z > 1 and hn <= max_nn
            else:  # hetero_edge
                m[i] = z > 1 and z != 6 and hn <= max_nn
        masks.append(m)
    if not masks:
        return None
    return np.concatenate(masks)


def _idea_b_cluster_score_from_n_clusters(n_clusters, max_clusters_for_high,
                                           silhouette=None, compactness=None):
    if n_clusters <= 1:
        base = 1.0
    elif n_clusters <= max_clusters_for_high:
        base = max(0.0, 1.0 - (n_clusters - 1) / max_clusters_for_high)
    else:
        base = max(0.0, 0.5 - (n_clusters - max_clusters_for_high) * 0.1)
    # 轮廓系数修正（多簇时）
    if silhouette is not None and np.isfinite(silhouette):
        if silhouette < 0.1:
            base *= 0.6
        elif silhouette < 0.2:
            base *= 0.8
        elif silhouette > 0.3:
            base = min(1.0, base * 1.1)
    # 紧凑度修正（单簇或轮廓系数不可用时）：avg_dist_to_centroid
    # 越小越紧凑，说明结合模式越收敛
    if compactness is not None and np.isfinite(compactness):
        if compactness < 2.0:
            base *= 1.0   # 非常紧凑，不扣分
        elif compactness < 3.0:
            base *= 0.9
        elif compactness < 4.0:
            base *= 0.75
        elif compactness < 5.0:
            base *= 0.6
        else:
            base *= 0.4   # 非常分散，大幅扣分
    return base


def _dbscan_metrics_idea_b(X, eps, min_samples, max_clusters_for_high):
    """对 3D 坐标矩阵 X 运行 DBSCAN，返回簇数、噪声、轮廓系数与想法 B 密度收敛分。"""
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    silhouette = None
    compactness = None
    unique_labels = set(labels) - {-1}
    if len(unique_labels) >= 2 and len(X) > len(unique_labels):
        try:
            n_sample = min(len(X), 5000)
            idx = (
                np.random.choice(len(X), n_sample, replace=False)
                if len(X) > n_sample else np.arange(len(X))
            )
            mask_valid = labels[idx] != -1
            if mask_valid.sum() > len(unique_labels):
                silhouette = float(
                    silhouette_score(X[idx][mask_valid], labels[idx][mask_valid])
                )
        except Exception:
            pass
    # 紧凑度：per-cluster 平均点到质心距离（Å），仅多簇时有意义
    # 单簇不做 compactness 惩罚（大口袋中分子天然分散，不代表聚类差）
    if n_clusters >= 2:
        compactness_vals = []
        for lbl in unique_labels:
            mask = labels == lbl
            if mask.sum() > 1:
                X_cluster = X[mask]
                centroid = X_cluster.mean(axis=0)
                dists = np.linalg.norm(X_cluster - centroid, axis=1)
                compactness_vals.append(float(np.mean(dists)))
        if compactness_vals:
            compactness = float(np.mean(compactness_vals))
    score = _idea_b_cluster_score_from_n_clusters(n_clusters, max_clusters_for_high, silhouette, compactness)
    return {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': silhouette,
        'compactness': compactness,
        'dbscan_labels': labels,
        'score': score,
    }


def _compute_morgan_fingerprints(molecules_with_pos, radius=2, n_bits=1024):
    """
    计算 Morgan 指纹矩阵（用于多样性分析）

    Returns:
        fps_matrix: np.ndarray (n_valid, n_bits) 或 None
        valid_mols: list of mol
    """
    if Chem is None or rdMolDescriptors is None:
        return None, []

    fps, valid_mols = [], []
    for mol, _ in molecules_with_pos:
        if mol is None:
            continue
        try:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros(n_bits, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
            valid_mols.append(mol)
        except Exception:
            pass

    if not fps:
        return None, []
    return np.vstack(fps).astype(np.float32), valid_mols


# =============================================================================
# 想法A：Vina 对接分数与口袋质量
# =============================================================================

def _read_vina_scores_from_excel(excel_path):
    """
    从 complete_molecules_*.xlsx 读取 Vina 分数。
    优先使用 Vina_Dock_亲和力，其次 Vina_Minimize_亲和力，最后 Vina_ScoreOnly_亲和力。
    返回合并后的有效分数列表。
    """
    if pd is None:
        return []
    try:
        xl = pd.ExcelFile(excel_path)
        all_scores = []
        for sheet_name in xl.sheet_names:
            if '_统计信息' in sheet_name:
                continue
            df = pd.read_excel(xl, sheet_name=sheet_name)
            for _, row in df.iterrows():
                v = None
                for col in ['Vina_Dock_亲和力', 'Vina_Minimize_亲和力', 'Vina_ScoreOnly_亲和力']:
                    if col not in df.columns:
                        continue
                    val = row[col]
                    if val is None or val == 'N/A' or (isinstance(val, float) and np.isnan(val)):
                        continue
                    try:
                        v = float(val)
                        break
                    except (ValueError, TypeError):
                        continue
                if v is not None:
                    all_scores.append(v)
        return all_scores
    except Exception:
        return []


def evaluate_idea_a_vina(pt_path=None, data_id=None, wait_timeout=300,
                         outputs_dir=None, pocket_id_override=None):
    """
    想法A：基于 Vina 对接分数评估口袋质量

    Vina 分数来源（按优先级）：
    1. complete_molecules_*.xlsx（Excel 中的 Vina_Dock_亲和力 等列）
    2. eval_results_*.pt（兼容旧格式）

    Args:
        pt_path: 与生成结果同目录的 .pt（用于默认 outputs_dir）；可与 outputs_dir 二选一。
        outputs_dir: 含 eval_* 子目录的文件夹（自定义评估时传入 batch 的 outputs）。
        pocket_id_override: 匹配 eval_{id}_* 的口袋 id（默认 custom 或从 result_* 文件名解析）。

    Returns:
        dict: {score, vina_mean, vina_median, vina_std, vina_best, vina_worst,
               vina_pct_good, num_scores, quality_label, success, message}
    """
    if outputs_dir is not None:
        outputs_dir = Path(outputs_dir).resolve()
    elif pt_path is not None:
        pt_path = Path(pt_path).resolve()
        outputs_dir = pt_path.parent
    else:
        return {
            'score': 0.0, 'vina_mean': None, 'vina_median': None, 'vina_std': None,
            'num_scores': 0, 'quality_label': 'unknown', 'success': False,
            'vina_scores': [],
            'message': '未指定 pt_path 或 outputs_dir，无法定位 Vina 评估输出',
        }

    if pocket_id_override is not None:
        pocket_id = str(pocket_id_override)
    elif pt_path is not None:
        pt_stem = Path(pt_path).stem
        if pt_stem.startswith('result_'):
            parts = pt_stem.split('_')
            pocket_id = parts[1] if len(parts) >= 3 else str(data_id or 'unknown')
        else:
            pocket_id = str(data_id or 'unknown')
    else:
        pocket_id = str(data_id if data_id is not None else 'custom')

    # 优先匹配 eval_{pocket_id}_*；batch 常用 eval_<日期>_<时间>_{pocket_id}_gfquadratic_*（须用 eval_*_{id}_*，勿全量 glob eval_*）
    eval_dirs = list(outputs_dir.glob(f'eval_{pocket_id}_*'))
    if not eval_dirs:
        eval_dirs = list(outputs_dir.glob(f'eval_*_{pocket_id}_*'))
    if not eval_dirs:
        eval_dirs = [d for d in outputs_dir.glob('eval_*') if d.is_dir() and f'_{pocket_id}_' in d.name]
    if not eval_dirs:
        eval_dirs = [d for d in outputs_dir.glob('eval_*') if d.is_dir()]

    if not eval_dirs:
        return {
            'score': 0.0, 'vina_mean': None, 'vina_median': None, 'vina_std': None,
            'num_scores': 0, 'quality_label': 'unknown', 'success': False,
            'vina_scores': [],
            'message': f'未找到评估目录 eval_{pocket_id}_* 或 eval_*_{pocket_id}_*'
        }

    timestamp_pattern = r'_\d{8}_\d{6}'
    new_dirs = [d for d in eval_dirs if re.search(timestamp_pattern, d.name)]
    eval_dir = max(new_dirs if new_dirs else eval_dirs, key=lambda x: x.stat().st_mtime)

    vina_scores = []

    # 1. 优先从 complete_molecules_*.xlsx 读取
    excel_files = list(eval_dir.glob('complete_molecules_*.xlsx')) or list(eval_dir.glob('**/complete_molecules_*.xlsx'))
    if excel_files:
        latest_excel = max(excel_files, key=os.path.getmtime)
        vina_scores = _read_vina_scores_from_excel(latest_excel)

    # 2. 若 Excel 无有效分数，回退到 eval_results_*.pt
    if not vina_scores and torch is not None:
        eval_files = list(eval_dir.glob('eval_results_*.pt')) or list(eval_dir.glob('**/eval_results_*.pt'))
        if eval_files:
            try:
                eval_data = _torch_load_legacy(
                    max(eval_files, key=os.path.getmtime), map_location="cpu"
                )
                stats = eval_data.get('statistics', {})
                vina_scores = (stats.get('vina_dock_scores') or
                              stats.get('vina_minimize_scores') or
                              stats.get('vina_score_only_scores') or
                              stats.get('vina_scores') or [])
            except Exception:
                pass

    if not vina_scores:
        return {
            'score': 0.0, 'vina_mean': None, 'vina_median': None, 'vina_std': None,
            'num_scores': 0, 'quality_label': 'unknown', 'success': False,
            'vina_scores': [],
            'message': '未找到 Vina 分数（complete_molecules_*.xlsx 或 eval_results_*.pt 中均无有效数据）'
        }

    try:
        vina_arr = np.array(vina_scores, dtype=np.float64)
        vina_mean   = float(np.mean(vina_arr))
        vina_median = float(np.median(vina_arr))
        vina_std    = float(np.std(vina_arr)) if len(vina_arr) > 1 else 0.0
        vina_best   = float(np.min(vina_arr))
        vina_worst  = float(np.max(vina_arr))
        vina_pct_good = float(np.mean(vina_arr <= -7.0))
        n = len(vina_arr)

        # 过滤正值（docking 失败通常返回正值，如 +50 kcal/mol）
        vina_arr_filtered = vina_arr[vina_arr < 0]
        if len(vina_arr_filtered) == 0:
            return {
                'score': 0.0, 'vina_mean': None, 'vina_median': None, 'vina_std': None,
                'num_scores': 0, 'quality_label': 'unknown', 'success': False,
                'vina_scores': vina_arr.tolist(),
                'message': '所有 Vina 对接均返回正值（可能全部失败）'
            }
        # 使用过滤后的数据重新计算统计量
        vina_mean   = float(np.mean(vina_arr_filtered))
        vina_median = float(np.median(vina_arr_filtered))
        vina_std    = float(np.std(vina_arr_filtered)) if len(vina_arr_filtered) > 1 else 0.0
        vina_best   = float(np.min(vina_arr_filtered))
        vina_worst  = float(np.max(vina_arr_filtered))
        vina_pct_good = float(np.mean(vina_arr_filtered <= -7.0))
        n = len(vina_arr_filtered)
        n_bad = len(vina_arr) - n

        # 综合评分：均值(50%) + 最佳分数(30%) + 命中率(20%)
        # 均值反映整体结合质量，最佳分数奖励发现强结合分子，命中率衡量口袋可药性
        mean_component = float(np.clip((-vina_mean - 6) / 6, 0.0, 1.0))
        best_component = float(np.clip((-vina_best - 6) / 6, 0.0, 1.0))
        hit_component = float(vina_pct_good)
        score = 0.5 * mean_component + 0.3 * best_component + 0.2 * hit_component
        score = float(np.clip(score, 0.0, 1.0))

        if score >= 0.5:
            label = 'high'
        elif score >= 0.2:
            label = 'medium'
        else:
            label = 'low'

        return {
            'score': score,
            'vina_mean': vina_mean,
            'vina_median': vina_median,
            'vina_std': vina_std,
            'vina_best': vina_best,
            'vina_worst': vina_worst,
            'vina_pct_good': vina_pct_good,
            'num_scores': n,
            'num_bad_scores': n_bad,
            # 保留与分子下标对齐的完整列表（含失败正值）；想法 C 会跳过 vina>=0
            'vina_scores': vina_arr.tolist(),
            'vina_scores_valid': vina_arr_filtered.tolist(),
            'mean_component': mean_component,
            'best_component': best_component,
            'hit_component': hit_component,
            'quality_label': label,
            'success': True,
            'message': f'Vina 平均={vina_mean:.2f}, 最佳={vina_best:.2f}, 命中率={vina_pct_good*100:.1f}%, 综合={score:.3f} (均值{mean_component:.2f}*0.5+最佳{best_component:.2f}*0.3+命中{hit_component:.2f}*0.2)'
        }
    except Exception as e:
        return {
            'score': 0.0, 'vina_mean': None, 'vina_median': None, 'vina_std': None,
            'num_scores': 0, 'quality_label': 'unknown', 'success': False,
            'vina_scores': [],
            'message': str(e)
        }


# =============================================================================
# 想法B：原子分布聚类（增强版：DBSCAN + KMeans 双重聚类 + 质心分析）
# =============================================================================

def evaluate_idea_b_clustering(
    molecules_with_pos,
    eps=1.5,
    min_samples=3,
    max_clusters_for_high=5,
    n_kmeans=5,
    atom_coord_subset='hetero_heavy',
    edge_max_heavy_neighbors=2,
    combined_focus_weight=0.5,
):
    """
    想法B：对生成分子的原子坐标做 DBSCAN（密度/簇数）并结合质心 KMeans。

    atom_coord_subset（DBSCAN 所用 3D 点集）：
        all          — 全部原子
        hetero_heavy — 非氢、非碳重原子（默认，聚焦药效团相关原子）
        edge_heavy   — 重原子且非氢重邻居数 <= edge_max_heavy_neighbors（近似分子边缘）
        hetero_edge  — 非碳边缘重原子（极性/表面位点，常用于结合模式）
        combined     — (1-w)*全原子 + w*hetero_edge 加权分，w=combined_focus_weight

    eps=1.5 A（默认）比 2.0 A 更好区分不同结合模式（2.0 A 约为 C-C 键长，过于粗糙）。

    KMeans 始终对整条分子的质心，不受子集影响。
    """
    if not molecules_with_pos or not HAS_SKLEARN:
        return {
            'score': 0.0, 'n_clusters': 0, 'n_noise': 0, 'silhouette': None,
            'centroid_kmeans_labels': None, 'kmeans_inertia': None,
            'quality_label': 'unknown', 'success': False,
            'atom_data': None,
            'message': '无分子数据或 sklearn 未安装'
        }

    valid_subsets = ('all', 'hetero_heavy', 'edge_heavy', 'hetero_edge', 'combined')
    if atom_coord_subset not in valid_subsets:
        atom_coord_subset = 'all'

    atom_data = _collect_atom_data(molecules_with_pos)
    if atom_data is None:
        return {
            'score': 0.0, 'n_clusters': 0, 'n_noise': 0, 'silhouette': None,
            'centroid_kmeans_labels': None, 'kmeans_inertia': None,
            'quality_label': 'unknown', 'success': False,
            'atom_data': None,
            'message': '无有效坐标'
        }

    X_all = atom_data['coords']
    centroids = atom_data['centroids']
    n_total = len(X_all)

    def _quality_label_from_score(sc):
        if sc >= 0.6:
            return 'high'
        if sc >= 0.3:
            return 'medium'
        return 'low'

    def _kmeans_block():
        kmeans_labels, kmeans_inertia = None, None
        if len(centroids) >= n_kmeans and KMeans is not None:
            k = min(n_kmeans, len(centroids))
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_labels = km.fit_predict(centroids).tolist()
            kmeans_inertia = float(km.inertia_)
        return kmeans_labels, kmeans_inertia

    if n_total < min_samples * 2:
        kmeans_labels, kmeans_inertia = _kmeans_block()
        return {
            'score': 0.5, 'n_clusters': 1, 'n_noise': 0, 'silhouette': None,
            'centroid_kmeans_labels': kmeans_labels,
            'kmeans_inertia': kmeans_inertia,
            'quality_label': 'medium', 'success': True,
            'atom_data': atom_data,
            'atom_coord_subset': atom_coord_subset,
            'n_atoms_dbscan': n_total,
            'n_atoms_total': n_total,
            'coords_focus': None,
            'score_all_atoms': None,
            'score_focus': None,
            'n_clusters_focus': None,
            'n_noise_focus': None,
            'silhouette_focus': None,
            'dbscan_eps': eps,
            'dbscan_min_samples': min_samples,
            'message': '样本点过少，无法可靠聚类',
        }

    try:
        np.random.seed(42)
        kmeans_labels, kmeans_inertia = _kmeans_block()

        def _run_on_subset(X_sub, _label_tag):
            if len(X_sub) < min_samples * 2:
                return None
            return _dbscan_metrics_idea_b(
                X_sub, eps, min_samples, max_clusters_for_high
            )

        if atom_coord_subset == 'all':
            m_all = _run_on_subset(X_all, 'all')
            if m_all is None:
                raise RuntimeError('DBSCAN 子集为空')
            n_clusters = m_all['n_clusters']
            n_noise = m_all['n_noise']
            silhouette = m_all['silhouette']
            score = m_all['score']
            db_labels = m_all['dbscan_labels']
            out_focus = None
            score_all_atoms = score
            score_focus = None
            n_cf = n_noise_focus = sil_f = None
            message = (
                f'簇数={n_clusters}, 噪声点={n_noise}, 轮廓系数={silhouette}'
            )
        elif atom_coord_subset == 'combined':
            m_all = _run_on_subset(X_all, 'all')
            if m_all is None:
                raise RuntimeError('DBSCAN 子集为空')
            mask_fe = _build_idea_b_coord_mask(
                molecules_with_pos, 'hetero_edge', edge_max_heavy_neighbors
            )
            if mask_fe is None or mask_fe.sum() < min_samples * 2:
                n_clusters = m_all['n_clusters']
                n_noise = m_all['n_noise']
                silhouette = m_all['silhouette']
                score = m_all['score']
                db_labels = m_all['dbscan_labels']
                score_all_atoms = m_all['score']
                score_focus = None
                n_cf = n_noise_focus = sil_f = None
                out_focus = None
                note = '（hetero_edge 点数不足，仅用全原子）'
            else:
                X_fe = X_all[mask_fe]
                m_fe = _run_on_subset(X_fe, 'hetero_edge')
                w = float(np.clip(combined_focus_weight, 0.0, 1.0))
                score_all_atoms = m_all['score']
                score_focus = m_fe['score'] if m_fe else None
                if m_fe is None:
                    score = m_all['score']
                    n_clusters = m_all['n_clusters']
                    n_noise = m_all['n_noise']
                    silhouette = m_all['silhouette']
                    db_labels = m_all['dbscan_labels']
                    n_cf = n_noise_focus = sil_f = None
                    out_focus = None
                    note = '（hetero_edge 子集聚类不可用，仅用全原子）'
                else:
                    score = (1.0 - w) * m_all['score'] + w * m_fe['score']
                    n_clusters = m_all['n_clusters']
                    n_noise = m_all['n_noise']
                    silhouette = m_all['silhouette']
                    db_labels = m_all['dbscan_labels']
                    n_cf = m_fe['n_clusters']
                    n_noise_focus = m_fe['n_noise']
                    sil_f = m_fe['silhouette']
                    out_focus = np.asarray(X_fe, dtype=np.float64)
                    note = f'（combined w={w:.2f} 全原子+hetero_edge）'
            message = (
                f'簇数={n_clusters}, 噪声={n_noise}, 轮廓={silhouette}; '
                f'focus簇={n_cf}, focus噪声={n_noise_focus}{note}'
            )
        else:
            mask = _build_idea_b_coord_mask(
                molecules_with_pos, atom_coord_subset, edge_max_heavy_neighbors
            )
            if mask is None or mask.sum() < min_samples * 2:
                return {
                    'score': 0.5, 'n_clusters': 1, 'n_noise': 0, 'silhouette': None,
                    'centroid_kmeans_labels': kmeans_labels,
                    'kmeans_inertia': kmeans_inertia,
                    'quality_label': 'medium', 'success': True,
                    'atom_data': atom_data,
                    'atom_coord_subset': atom_coord_subset,
                    'n_atoms_dbscan': int(mask.sum()) if mask is not None else 0,
                    'n_atoms_total': n_total,
                    'coords_focus': None,
                    'score_all_atoms': None,
                    'score_focus': None,
                    'n_clusters_focus': None,
                    'n_noise_focus': None,
                    'silhouette_focus': None,
                    'dbscan_eps': eps,
                    'dbscan_min_samples': min_samples,
                    'message': f'子集 {atom_coord_subset} 点数不足，无法可靠聚类',
                }
            X_sub = X_all[mask]
            m_sub = _run_on_subset(X_sub, atom_coord_subset)
            if m_sub is None:
                raise RuntimeError('DBSCAN 失败')
            n_clusters = m_sub['n_clusters']
            n_noise = m_sub['n_noise']
            silhouette = m_sub['silhouette']
            score = m_sub['score']
            db_labels = m_sub['dbscan_labels']
            out_focus = np.asarray(X_sub, dtype=np.float64)
            score_all_atoms = None
            score_focus = None
            n_cf = n_noise_focus = sil_f = None
            message = (
                f'[{atom_coord_subset}] 簇数={n_clusters}, 噪声={n_noise}, '
                f'轮廓={silhouette}, DBSCAN点数={len(X_sub)}/{n_total}'
            )

        label = _quality_label_from_score(score)

        if atom_coord_subset in ('all', 'combined'):
            n_atoms_dbscan = n_total
        else:
            n_atoms_dbscan = int(mask.sum())

        ret = {
            'score': score,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette': silhouette,
            'dbscan_labels': db_labels.tolist(),
            'centroid_kmeans_labels': kmeans_labels,
            'kmeans_inertia': kmeans_inertia,
            'quality_label': label,
            'success': True,
            'atom_data': atom_data,
            'atom_coord_subset': atom_coord_subset,
            'n_atoms_dbscan': n_atoms_dbscan,
            'n_atoms_total': n_total,
            'coords_focus': out_focus,
            'dbscan_eps': eps,
            'dbscan_min_samples': min_samples,
            'message': message,
        }

        if atom_coord_subset == 'combined':
            ret['score_all_atoms'] = score_all_atoms
            ret['score_focus'] = score_focus
            ret['n_clusters_focus'] = n_cf
            ret['n_noise_focus'] = n_noise_focus
            ret['silhouette_focus'] = sil_f
            ret['combined_focus_weight'] = float(np.clip(combined_focus_weight, 0.0, 1.0))
        else:
            ret['score_all_atoms'] = score if atom_coord_subset == 'all' else None
            ret['score_focus'] = score if atom_coord_subset != 'all' else None
            ret['n_clusters_focus'] = n_cf
            ret['n_noise_focus'] = n_noise_focus
            ret['silhouette_focus'] = sil_f

        return ret
    except Exception as e:
        return {
            'score': 0.0, 'n_clusters': 0, 'n_noise': 0, 'silhouette': None,
            'centroid_kmeans_labels': None, 'kmeans_inertia': None,
            'quality_label': 'unknown', 'success': False,
            'atom_data': atom_data,
            'atom_coord_subset': atom_coord_subset,
            'message': str(e)
        }


# =============================================================================
# 想法C：配体效率 LE = (-ΔG) / N_heavy（Vina 近似 ΔG，kcal/mol）
# =============================================================================

# 将 LE 均值映射到 [0,1] 的参考区间（kcal·mol⁻¹·重原子⁻¹）
# 典型药物样分子 LE 范围 0.2-0.4 (Hopkins et al., 2014)，下限从 0.12 收紧至 0.18
IDEA_C_LE_SCORE_LOW = 0.20
IDEA_C_LE_SCORE_HIGH = 0.42


def _n_heavy_atoms(mol):
    """非氢重原子数。"""
    if mol is None or Chem is None:
        return 0
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def evaluate_idea_c_ligand_efficiency(
    molecules_with_pos,
    vina_scores,
    le_low_for_score=IDEA_C_LE_SCORE_LOW,
    le_high_for_score=IDEA_C_LE_SCORE_HIGH,
):
    """
    配体效率：对每个分子 LE_i = -ΔG_i / N_heavy,i。

    ΔG 采用与想法 A 相同的 Vina 亲和力（kcal/mol，负值表示有利结合），故 LE = -vina_score / N_heavy。

    假设 ``vina_scores[i]`` 与 ``molecules_with_pos[i]`` 一一对应（与 batch 导出顺序一致）；
    仅对 mol 非空、N_heavy > 0 且 ``vina < 0``（对接成功）的条目计算；跳过失败对接以免负 LE。

    Returns:
        dict: score, le_mean, le_median, le_std, le_list, n_valid, n_heavy_per_mol,
              quality_label, success, message, …
    """
    empty = {
        'score': 0.0,
        'le_mean': None,
        'le_median': None,
        'le_std': None,
        'le_list': [],
        'n_heavy_per_mol': [],
        'n_valid': 0,
        'n_molecules': 0,
        'n_vina_scores': 0,
        'n_aligned': 0,
        'n_skipped_failed_vina': 0,
        'quality_label': 'unknown',
        'success': False,
        'message': '',
    }

    if not molecules_with_pos or Chem is None:
        empty['message'] = '无分子数据或 RDKit 未安装'
        return empty

    if not vina_scores:
        empty['message'] = '无 Vina 分数（想法A 失败或缺 eval 输出），无法计算 LE'
        return empty

    v_arr = np.asarray(vina_scores, dtype=np.float64).ravel()
    n_mol = len(molecules_with_pos)
    n_v = len(v_arr)
    n_aligned = min(n_mol, n_v)

    le_list = []
    n_heavy_list = []
    n_skipped_failed_vina = 0
    for i in range(n_aligned):
        mol, _ = molecules_with_pos[i]
        if mol is None:
            continue
        nh = _n_heavy_atoms(mol)
        if nh <= 0:
            continue
        dg = float(v_arr[i])
        # 失败对接常返回正值；跳过以保持下标对齐且不产生负 LE
        if not np.isfinite(dg) or dg >= 0.0:
            n_skipped_failed_vina += 1
            continue
        le = -dg / nh
        if not np.isfinite(le):
            continue
        le_list.append(float(le))
        n_heavy_list.append(nh)

    n_valid = len(le_list)
    if n_valid == 0:
        empty['n_molecules'] = n_mol
        empty['n_vina_scores'] = n_v
        empty['n_aligned'] = n_aligned
        empty['n_skipped_failed_vina'] = n_skipped_failed_vina
        empty['message'] = (
            '无有效 (分子, Vina) 对可计算 LE'
            f'（跳过失败对接 {n_skipped_failed_vina}；检查 mol 与重原子数）'
        )
        return empty

    arr = np.array(le_list, dtype=np.float64)
    le_mean = float(np.mean(arr))
    le_median = float(np.median(arr))
    le_std = float(np.std(arr)) if n_valid > 1 else 0.0

    span = le_high_for_score - le_low_for_score
    if span <= 0:
        raw = 0.0
    else:
        raw = (le_mean - le_low_for_score) / span
    score = float(np.clip(raw, 0.0, 1.0))

    if score >= 0.6:
        label = 'high'
    elif score >= 0.3:
        label = 'medium'
    else:
        label = 'low'

    msg_parts = [
        f'LE_mean={le_mean:.4f} kcal·mol⁻¹·重原子⁻¹ (n={n_valid})',
        f'LE_median={le_median:.4f}',
    ]
    if n_skipped_failed_vina:
        msg_parts.append(f'跳过失败对接 {n_skipped_failed_vina}')
    if n_aligned < n_mol or n_aligned < n_v:
        msg_parts.append(f'对齐: 前 {n_aligned} 条 (分子 {n_mol}, Vina {n_v})')
    msg = '; '.join(msg_parts)

    return {
        'score': score,
        'le_mean': le_mean,
        'le_median': le_median,
        'le_std': le_std,
        'le_list': le_list,
        'n_heavy_per_mol': n_heavy_list,
        'n_valid': n_valid,
        'n_molecules': n_mol,
        'n_vina_scores': n_v,
        'n_aligned': n_aligned,
        'n_skipped_failed_vina': n_skipped_failed_vina,
        'le_low_for_score': le_low_for_score,
        'le_high_for_score': le_high_for_score,
        'quality_label': label,
        'success': True,
        'message': msg,
    }


# =============================================================================
# 想法D：药物相似性 (QED, SA, Lipinski, PAINS)
# =============================================================================

def evaluate_idea_d_druglikeness(molecules_with_pos):
    """
    想法D：生成分子的药物相似性指标

    Returns:
        dict: {score, qed_mean, qed_std, qed_list, sa_mean, sa_std, sa_list,
               lipinski_mean, lipinski_list, pains_ratio, veber_ratio, n_valid, quality_label, success}
    """
    if not molecules_with_pos or get_chem is None or is_pains is None or obey_lipinski is None:
        return {
            'score': 0.0, 'qed_mean': None, 'sa_mean': None,
            'lipinski_mean': None, 'pains_ratio': None, 'veber_ratio': None,
            'qed_list': [], 'sa_list': [], 'lipinski_list': [],
            'quality_label': 'unknown', 'success': False,
            'message': '无分子数据或 scoring_func 未安装'
        }

    qed_list, sa_list, lipinski_list = [], [], []
    pains_hits = 0
    veber_passes = 0
    n_valid = 0

    for mol, _ in molecules_with_pos:
        if mol is None:
            continue
        try:
            chem = get_chem(mol)
            q = chem.get('qed')
            s = chem.get('sa')
            lip = chem.get('lipinski')
            if q is not None and not (isinstance(q, float) and np.isnan(q)):
                qed_list.append(float(q))
            if s is not None and not (isinstance(s, float) and (np.isnan(s) or np.isinf(s))):
                sa_list.append(min(1.0, max(0.0, float(s))))
            if lip is not None and lip != 'N/A':
                lipinski_list.append(int(lip) / 5.0)
            if is_pains(mol):
                pains_hits += 1
            if passes_veber is not None:
                rot_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
                if passes_veber(mol, rot_bonds=rot_bonds):
                    veber_passes += 1
            n_valid += 1
        except Exception:
            pass

    if n_valid == 0:
        return {
            'score': 0.0, 'qed_mean': None, 'sa_mean': None,
            'lipinski_mean': None, 'pains_ratio': None, 'veber_ratio': None,
            'qed_list': [], 'sa_list': [], 'lipinski_list': [],
            'quality_label': 'unknown', 'success': False,
            'message': '无有效化学指标'
        }

    qed_mean = float(np.mean(qed_list)) if qed_list else 0.0
    qed_std  = float(np.std(qed_list)) if len(qed_list) > 1 else 0.0
    sa_mean  = float(np.mean(sa_list)) if sa_list else 0.0
    sa_std   = float(np.std(sa_list)) if len(sa_list) > 1 else 0.0
    lipinski_mean = float(np.mean(lipinski_list)) if lipinski_list else 0.0
    pains_ratio = pains_hits / n_valid
    veber_ratio = veber_passes / n_valid

    # 综合分数：QED(35%) + SA(25%) + Lipinski(15%) + Veber(10%) + PAINS惩罚(15%)
    # PAINS 权重从 10% 提升至 15%（假阳性风险高）；新增 Veber 规则
    weights = (0.35, 0.25, 0.15, 0.10, 0.15)
    assert abs(sum(weights) - 1.0) < 1e-6, f"Druglikeness weights must sum to 1.0, got {sum(weights)}"
    score = (
        weights[0] * min(1.0, qed_mean) +
        weights[1] * min(1.0, sa_mean) +
        weights[2] * lipinski_mean +
        weights[3] * veber_ratio +
        weights[4] * (1.0 - pains_ratio)
    )
    # PAINS 命中率 >5% 时软惩罚，提高差集区分度
    if pains_ratio > 0.05:
        score *= 0.85
    score = float(np.clip(score, 0.0, 1.0))

    if score >= 0.6:
        label = 'high'
    elif score >= 0.3:
        label = 'medium'
    else:
        label = 'low'

    return {
        'score': score,
        'qed_mean': qed_mean, 'qed_std': qed_std, 'qed_list': qed_list,
        'sa_mean': sa_mean,   'sa_std': sa_std,   'sa_list': sa_list,
        'lipinski_mean': lipinski_mean, 'lipinski_list': lipinski_list,
        'pains_ratio': pains_ratio,
        'veber_ratio': veber_ratio,
        'n_valid': n_valid,
        'quality_label': label,
        'success': True,
        'message': f'QED={qed_mean:.2f}, SA={sa_mean:.2f}, Lipinski={lipinski_mean:.2f}, Veber={veber_ratio*100:.1f}%, PAINS={pains_ratio*100:.1f}%'
    }


# =============================================================================
# 想法E：完整分子比例（SMILES 无 '.'）
# =============================================================================

def _count_complete_molecules_smiles_no_dot(molecules_with_pos):
    """
    完整分子：RDKit ``MolToSmiles`` 中不含 ``'.'``（无多片段/盐桥式断点）。

    Returns:
        int: 满足条件的分子数（``mol is None`` 或无法转 SMILES 的条目不计入）。
    """
    if not molecules_with_pos or Chem is None:
        return 0
    n = 0
    for mol, _ in molecules_with_pos:
        if mol is None:
            continue
        try:
            smi = Chem.MolToSmiles(mol)
            if '.' not in smi:
                n += 1
        except Exception:
            continue
    return n


def _idea_e_score_from_rate(rate_for_score):
    """Map yield rate in [0, 1] to quality score (tightened: ≥0.98 full)."""
    r = float(rate_for_score)
    if r >= 0.98:
        return 1.0, 'high'
    if r >= 0.90:
        return 0.70 + (r - 0.90) / 0.08 * 0.30, 'high'
    if r >= 0.75:
        return 0.40 + (r - 0.75) / 0.15 * 0.30, 'medium'
    return (r / 0.75) * 0.40, 'low'


def evaluate_idea_e_reconstruction(pt_path, molecules_with_pos, expected_n_molecules=None):
    """
    想法E：完整分子比例（非「重建槽位填满」）。

    分子：``MolToSmiles(mol)`` 中**不含** ``'.'`` 的条目计为完整单组分分子。

    若 ``expected_n_molecules`` 为正整数，分母为该应生成分子数：
    ``reconstruct_rate = n_complete / expected_n_molecules``（字段名沿用 ``reconstruct_rate``，可大于 1）。

    未指定时：有 .pt 则分母为 ``len(pred_ligand_pos)``；无 .pt 则跳过。

    Returns:
        dict: score, reconstruct_rate, n_success（=完整分子数）, n_loaded_molecules, n_total,
              quality_label, success, denominator_source, message
    """
    n_loaded = len(molecules_with_pos) if molecules_with_pos else 0
    n_complete = _count_complete_molecules_smiles_no_dot(molecules_with_pos)
    n_success = n_complete

    exp = None
    if expected_n_molecules is not None:
        try:
            exp = int(expected_n_molecules)
        except (TypeError, ValueError):
            exp = None

    if exp is not None and exp > 0:
        n_total = exp
        rate = n_success / n_total
        score_eff = min(1.0, rate)
        score, label = _idea_e_score_from_rate(score_eff)
        return {
            'score': min(1.0, score),
            'reconstruct_rate': rate,
            'n_success': n_success,
            'n_loaded_molecules': n_loaded,
            'n_total': n_total,
            'quality_label': label,
            'success': True,
            'denominator_source': 'expected_n_molecules',
            'message': (
                f'完整分子比例={rate*100:.1f}% ({n_success}/{n_total}，分子=SMILES无断点; '
                f'已载入构象 {n_loaded}；分母=应生成分子数)'
            ),
        }

    if pt_path is None:
        return {
            'score': 0.0,
            'reconstruct_rate': None,
            'n_success': n_success,
            'n_loaded_molecules': n_loaded,
            'n_total': n_loaded,
            'quality_label': 'unknown',
            'success': False,
            'denominator_source': None,
            'message': '外部配体或无 .pt，且未指定 --idea_e_expected_n_molecules，跳过想法E（不参与综合加权）',
        }
    data = load_pt_file(pt_path) if pt_path else None
    if data is None:
        return {
            'score': 0.0,
            'reconstruct_rate': 0.0,
            'n_success': 0,
            'n_loaded_molecules': 0,
            'n_total': 0,
            'quality_label': 'unknown',
            'success': False,
            'denominator_source': None,
            'message': '无法加载 .pt',
        }

    pred_pos = data.get('pred_ligand_pos', [])
    n_total = len(pred_pos) if pred_pos else 0

    if n_total == 0:
        return {
            'score': 0.0,
            'reconstruct_rate': 0.0,
            'n_success': 0,
            'n_loaded_molecules': n_loaded,
            'n_total': 0,
            'quality_label': 'unknown',
            'success': False,
            'denominator_source': None,
            'message': '无生成分子（.pt 中 pred_ligand_pos 为空；可改用 --idea_e_expected_n_molecules 指定应生成数）',
        }

    rate = n_success / n_total
    score, label = _idea_e_score_from_rate(rate)

    return {
        'score': min(1.0, score),
        'reconstruct_rate': rate,
        'n_success': n_success,
        'n_loaded_molecules': n_loaded,
        'n_total': n_total,
        'quality_label': label,
        'success': True,
        'denominator_source': 'pt_pred_ligand_pos',
        'message': (
            f'完整分子比例={rate*100:.1f}% ({n_success}/{n_total}，分子=SMILES无断点; '
            f'已载入构象 {n_loaded}；分母=.pt pred_ligand_pos)'
        ),
    }


# =============================================================================
# 想法F：分子唯一性与多样性（增强：Tanimoto 多样性）
# =============================================================================

def evaluate_idea_f_uniqueness(molecules_with_pos):
    """
    想法F：分子唯一性（score 仅由 unique_ratio 决定）。

    满分门槛 unique_full_at=0.95（约 100 个中 ≥95 个不同 SMILES）。
    分段：≥0.95→1.0；[0.70,0.95)→0.70–1.0；[0.40,0.70)→0.30–0.70；<0.40→[0,0.30]。
    ``fingerprint_dissimilarity`` 为参考指标（不进总分）；``tanimoto_diversity`` 为别名。
    """
    unique_full_at = 0.95
    if not molecules_with_pos or Chem is None:
        return {
            'score': 0.0, 'unique_ratio': 0.0, 'n_unique': 0, 'n_total': 0,
            'unique_full_at': unique_full_at,
            'fingerprint_dissimilarity': None, 'tanimoto_diversity': None,
            'fps_matrix': None,
            'quality_label': 'unknown', 'success': False, 'message': '无分子数据'
        }

    smiles_set = set()
    n_complete = 0
    for mol, _ in molecules_with_pos:
        if mol is None:
            continue
        try:
            smi = Chem.MolToSmiles(mol)
            if smi and '.' not in smi:
                smiles_set.add(smi)
                n_complete += 1
        except Exception:
            pass

    if n_complete == 0:
        return {
            'score': 0.0, 'unique_ratio': 0.0, 'n_unique': 0, 'n_total': 0,
            'unique_full_at': unique_full_at,
            'fingerprint_dissimilarity': None, 'tanimoto_diversity': None,
            'fps_matrix': None,
            'quality_label': 'unknown', 'success': False, 'message': '无有效 SMILES'
        }

    n_unique = len(smiles_set)
    unique_ratio = n_unique / n_complete

    # Tanimoto/余弦相异度（采样，避免 O(N^2)；不进总分）
    fingerprint_dissimilarity = None
    fps_matrix, _ = _compute_morgan_fingerprints(molecules_with_pos)
    if fps_matrix is not None and len(fps_matrix) >= 2 and HAS_SKLEARN:
        n_fps = len(fps_matrix)
        sample_size = min(n_fps, 200)
        idx = np.random.choice(n_fps, sample_size, replace=False) if n_fps > sample_size else np.arange(n_fps)
        fps_sample = fps_matrix[idx]
        # 余弦相似度 → 相异度（历史字段曾误称 Tanimoto）
        sim_matrix = (fps_sample @ fps_sample.T) / (
            np.outer(np.linalg.norm(fps_sample, axis=1), np.linalg.norm(fps_sample, axis=1)) + 1e-8
        )
        upper_tri = sim_matrix[np.triu_indices(len(fps_sample), k=1)]
        fingerprint_dissimilarity = float(1.0 - np.mean(upper_tri))

    # ≥95% unique 满分；中等唯一性线性过渡；低唯一性惩罚 mode-collapse
    r = unique_ratio
    if r >= unique_full_at:
        score = 1.0
    elif r >= 0.70:
        score = 0.70 + (r - 0.70) / 0.25 * 0.30  # 0.70→0.70, 0.95→1.0
    elif r >= 0.40:
        score = 0.30 + (r - 0.40) / 0.30 * 0.40  # 0.40→0.30, 0.70→0.70
    else:
        score = r / 0.40 * 0.30  # 0→0, 0.40→0.30

    if score >= 0.6:
        label = 'high'
    elif score >= 0.3:
        label = 'medium'
    else:
        label = 'low'

    return {
        'score': min(1.0, float(score)),
        'unique_ratio': unique_ratio,
        'n_unique': n_unique,
        'n_total': n_complete,
        'unique_full_at': unique_full_at,
        'fingerprint_dissimilarity': fingerprint_dissimilarity,
        'tanimoto_diversity': fingerprint_dissimilarity,  # legacy alias
        'fps_matrix': fps_matrix,
        'quality_label': label,
        'success': True,
        'message': (
            f'唯一性={unique_ratio*100:.1f}% ({n_unique}/{n_complete}), '
            f'满分线≥{unique_full_at*100:.0f}%, '
            f'指纹余弦相异度={fingerprint_dissimilarity}'
        ),
    }


def _merge_affinity_le(idea_vina, idea_le, w_v=0.55, w_le=0.45):
    """Merge Vina (idea A raw) and LE into slot-A Affinity+LE score."""
    idea_vina = idea_vina or {}
    idea_le = idea_le or {}
    v_ok = bool(idea_vina.get('success'))
    le_ok = bool(idea_le.get('success'))
    out = dict(idea_vina)
    out['le_component'] = idea_le
    out['vina_component_score'] = idea_vina.get('score') if v_ok else None
    out['le_component_score'] = idea_le.get('score') if le_ok else None
    if v_ok and le_ok:
        score = w_v * float(idea_vina['score']) + w_le * float(idea_le['score'])
        out['score'] = float(min(1.0, score))
        out['success'] = True
        out['message'] = (
            f"Aff+LE={out['score']:.3f} "
            f"(vina={idea_vina['score']:.3f}×{w_v}+LE={idea_le['score']:.3f}×{w_le})"
        )
    elif v_ok:
        out['score'] = float(idea_vina['score'])
        out['success'] = True
        out['message'] = f"仅 Vina={out['score']:.3f}（无有效 LE）"
    elif le_ok:
        out = {
            'score': float(idea_le['score']),
            'success': True,
            'le_component': idea_le,
            'vina_scores': idea_vina.get('vina_scores') or [],
            'vina_component_score': None,
            'le_component_score': idea_le.get('score'),
            'message': f"仅 LE={idea_le['score']:.3f}（无有效 Vina）",
        }
    else:
        out['score'] = 0.0
        out['success'] = False
        out['message'] = 'Vina 与 LE 均失败'
    s = float(out.get('score') or 0.0)
    if out.get('success'):
        out['quality_label'] = 'high' if s >= 0.5 else ('medium' if s >= 0.2 else 'low')
    else:
        out['quality_label'] = 'unknown'
    return out


def _fpocket_score_map(score_raw):
    """Map FPocket Score (~0–40+) to [0,1]."""
    if score_raw is None or not np.isfinite(float(score_raw)):
        return None
    return float(np.clip(float(score_raw) / 40.0, 0.0, 1.0))


def _fpocket_hydrophobicity_midpeak(h_raw):
    """
    Hydrophobicity: mid-range full score; extremes penalized.
    FPocket hydrophobicity often roughly in [-2, 2] or wider — normalize via tanh then mid-peak.
    """
    if h_raw is None or not np.isfinite(float(h_raw)):
        return None
    # map to ~[0,1] via sigmoid-ish on raw
    x = 0.5 * (np.tanh(float(h_raw) / 2.0) + 1.0)
    if 0.30 <= x <= 0.70:
        return 1.0
    if x < 0.30:
        return float(0.40 + (x / 0.30) * 0.60)  # 0→0.4, 0.3→1.0
    # x > 0.70
    return float(1.0 - (x - 0.70) / 0.30 * 0.60)  # 0.7→1, 1→0.4


def _fpocket_druggability_map(d_raw):
    if d_raw is None or not np.isfinite(float(d_raw)):
        return None
    d = float(np.clip(float(d_raw), 0.0, 1.0))
    if d < 0.2:
        return float(d / 0.2 * 0.35)  # <0.2 → ≤0.35
    return d


def evaluate_idea_f_pocket_chemistry(
    fpocket_protein_pdb=None,
    molecules_with_pos=None,
    centroid_radius=10.0,
    fpocket_cmd='fpocket',
    fpocket_pocket_index=1,
    fpocket_timeout=600,
):
    """
    想法F（口袋化学）：配体邻域残基的生物学代理分（疏水 / 极性 / 封闭度）。

    不调用 FPocket。``fpocket_protein_pdb`` 参数名保留以兼容旧调用，实为受体 PDB。
    默认邻域半径 **10 Å**。无蛋白 PDB、无配体质心、或邻域无残基 → success=False。
    """
    empty = {
        'score': 0.0, 'success': False, 'quality_label': 'unknown',
        'hydro_fraction': None, 'polar_fraction': None, 'enclosure_score': None,
        'n_residues': 0, 'n_heavy_atoms': 0,
        'message': '未指定蛋白 PDB 或配体坐标不可用',
        # legacy keys kept empty for old viz/CSV consumers
        'druggability': None, 'fpocket_score': None, 'hydrophobicity_score': None,
        'druggability_mapped': None, 'fpocket_score_mapped': None,
        'hydrophobicity_mapped': None,
    }
    if not fpocket_protein_pdb or not str(fpocket_protein_pdb).strip():
        return empty
    prot = Path(fpocket_protein_pdb).expanduser().resolve()
    if not prot.is_file():
        empty['message'] = f'蛋白 PDB 不存在: {prot}'
        return empty
    center = _ligand_centroid_from_molecules(molecules_with_pos)
    if center is None:
        empty['message'] = '无法从分子估计口袋中心'
        return empty

    data, err = _collect_pocket_residue_composition(prot, center, radius=float(centroid_radius))
    if data is None:
        empty['message'] = err or '残基组成收集失败'
        return empty
    n_res = int(data.get('n_residues') or 0)
    if n_res <= 0:
        empty['message'] = (
            f'口袋邻域（{float(centroid_radius):g}Å）内无蛋白残基；'
            '请确认 PDB 与配体坐标同一参考系'
        )
        return empty

    n_hydro = int(data.get('n_hydrophobic') or 0)
    n_polar = int(data.get('n_polar') or 0)
    n_heavy = int(data.get('n_heavy_atoms') or 0)
    hydro_frac = float(n_hydro) / float(n_res)
    polar_frac = float(n_polar) / float(n_res)
    # enclosure: saturate around ~350 heavy atoms in 10Å shell
    enc_raw = float(np.clip(n_heavy / 350.0, 0.0, 1.0))
    hydro_m = _fraction_midpeak_01(hydro_frac)
    polar_m = _fraction_midpeak_01(polar_frac)
    enc_m = _fraction_midpeak_01(enc_raw)

    parts = [hydro_m, polar_m, enc_m]
    score = float(_weighted_geo_mean(parts, [1.0, 1.0, 1.0]))
    label = 'high' if score >= 0.6 else ('medium' if score >= 0.3 else 'low')
    return {
        'score': score,
        'success': True,
        'quality_label': label,
        'hydro_fraction': hydro_frac,
        'polar_fraction': polar_frac,
        'enclosure_score': enc_m,
        'enclosure_raw': enc_raw,
        'n_residues': n_res,
        'n_hydrophobic': n_hydro,
        'n_polar': n_polar,
        'n_heavy_atoms': n_heavy,
        'centroid_radius': float(centroid_radius),
        'ligand_centroid': [float(center[0]), float(center[1]), float(center[2])],
        'druggability': None,
        'fpocket_score': None,
        'hydrophobicity_score': None,
        'druggability_mapped': hydro_m,
        'fpocket_score_mapped': polar_m,
        'hydrophobicity_mapped': enc_m,
        'message': (
            f'bio_chem={score:.3f} (hydro={hydro_m:.3f}/{hydro_frac:.2f}, '
            f'polar={polar_m:.3f}/{polar_frac:.2f}, enc={enc_m:.3f}, '
            f'n_res={n_res}, n_heavy={n_heavy}, r={float(centroid_radius):g}Å)'
        ),
    }


# Residue classification for P4 anchors (protonation convention: HIS≈HID donor/acceptor)
_ANCHOR_NEG = {'ASP', 'GLU'}
_ANCHOR_POS = {'LYS', 'ARG', 'HIP'}
_ANCHOR_ARO = {'PHE', 'TYR', 'TRP', 'HIS', 'HID', 'HIE', 'HIP'}
_ANCHOR_HYDRO = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'PRO', 'TRP', 'TYR'}
_POCKET_POLAR = {
    'ASP', 'GLU', 'LYS', 'ARG', 'HIS', 'HID', 'HIE', 'HIP',
    'SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS',
}


def _fraction_midpeak_01(x):
    """Map fraction in [0,1] with mid-range peak (0.30–0.70 → 1.0)."""
    if x is None or not np.isfinite(float(x)):
        return 0.0
    x = float(np.clip(float(x), 0.0, 1.0))
    if 0.30 <= x <= 0.70:
        return 1.0
    if x < 0.30:
        return float(0.40 + (x / 0.30) * 0.60)
    return float(1.0 - (x - 0.70) / 0.30 * 0.60)


def _collect_pocket_residue_composition(protein_pdb, center, radius=10.0):
    """
    Count residues / hydrophobic / polar / heavy atoms within radius of ligand center.
    Default radius 10 Å. Returns (dict, err_msg).
    """
    try:
        from Bio.PDB import PDBParser, is_aa
    except ImportError:
        return None, 'Bio.PDB 未安装'

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('prot', str(protein_pdb))
    except Exception as exc:
        return None, f'PDB 解析失败: {exc}'

    center = np.asarray(center, dtype=np.float64).reshape(3)
    n_res = 0
    n_hydro = 0
    n_polar = 0
    n_heavy = 0
    r = float(radius)

    for model in structure:
        for chain in model:
            for res in chain:
                if not is_aa(res, standard=True):
                    resname = res.get_resname().strip().upper()
                    if resname not in ('HID', 'HIE', 'HIP'):
                        continue
                else:
                    resname = res.get_resname().strip().upper()
                try:
                    if 'CA' in res:
                        ca = np.asarray(res['CA'].coord, dtype=np.float64)
                    else:
                        coords = [a.coord for a in res.get_atoms()]
                        if not coords:
                            continue
                        ca = np.mean(coords, axis=0)
                except Exception:
                    continue
                if np.linalg.norm(ca - center) > r:
                    continue
                n_res += 1
                if resname in _ANCHOR_HYDRO:
                    n_hydro += 1
                if resname in _POCKET_POLAR:
                    n_polar += 1
                for atom in res.get_atoms():
                    el = atom.element.strip().upper() if atom.element else ''
                    if el and el != 'H':
                        try:
                            pos = np.asarray(atom.coord, dtype=np.float64)
                        except Exception:
                            continue
                        if np.linalg.norm(pos - center) <= r:
                            n_heavy += 1

    return {
        'n_residues': n_res,
        'n_hydrophobic': n_hydro,
        'n_polar': n_polar,
        'n_heavy_atoms': n_heavy,
    }, None


def _ligand_centroid_from_molecules(molecules_with_pos):
    pts = []
    for mol, pos in molecules_with_pos or []:
        if pos is None:
            continue
        try:
            arr = np.asarray(pos, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] >= 3 and len(arr) > 0:
                pts.append(arr[:, :3].mean(axis=0))
        except Exception:
            pass
    if not pts:
        return None
    return np.mean(np.vstack(pts), axis=0)


def _collect_pocket_anchor_sites(protein_pdb, center, radius=12.0):
    """
    Collect anchor site coordinates + category counts from protein PDB
    within radius of center. Uses Bio.PDB.
    Returns dict with counts and list of (xyz, category).
    """
    try:
        from Bio.PDB import PDBParser, is_aa
    except ImportError:
        return None, 'Bio.PDB 未安装'

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('prot', str(protein_pdb))
    except Exception as exc:
        return None, f'PDB 解析失败: {exc}'

    center = np.asarray(center, dtype=np.float64).reshape(3)
    sites = []  # (xyz, cat) cat in donor,acceptor,aromatic,ionic,hydrophobic
    n_res_in_pocket = 0

    for model in structure:
        for chain in model:
            for res in chain:
                if not is_aa(res, standard=True):
                    # still allow HIS variants
                    resname = res.get_resname().strip().upper()
                    if resname not in ('HID', 'HIE', 'HIP'):
                        continue
                else:
                    resname = res.get_resname().strip().upper()
                try:
                    if 'CA' in res:
                        ca = res['CA'].coord
                    else:
                        coords = [a.coord for a in res.get_atoms()]
                        if not coords:
                            continue
                        ca = np.mean(coords, axis=0)
                except Exception:
                    continue
                ca = np.asarray(ca, dtype=np.float64)
                if np.linalg.norm(ca - center) > radius:
                    continue
                n_res_in_pocket += 1

                # backbone donor/acceptor
                if 'N' in res:
                    sites.append((np.asarray(res['N'].coord, dtype=np.float64), 'donor'))
                if 'O' in res:
                    sites.append((np.asarray(res['O'].coord, dtype=np.float64), 'acceptor'))

                if resname in _ANCHOR_NEG:
                    for aname in ('OD1', 'OD2', 'OE1', 'OE2'):
                        if aname in res:
                            sites.append((np.asarray(res[aname].coord, dtype=np.float64), 'ionic'))
                            sites.append((np.asarray(res[aname].coord, dtype=np.float64), 'acceptor'))
                if resname in _ANCHOR_POS or resname == 'LYS':
                    for aname in ('NZ', 'NH1', 'NH2', 'NE'):
                        if aname in res:
                            sites.append((np.asarray(res[aname].coord, dtype=np.float64), 'ionic'))
                            sites.append((np.asarray(res[aname].coord, dtype=np.float64), 'donor'))
                if resname in _ANCHOR_ARO:
                    ring_atoms = []
                    for a in res.get_atoms():
                        if a.element.strip().upper() == 'C' and a.name.strip() not in ('CA', 'C', 'CB'):
                            ring_atoms.append(a.coord)
                    if ring_atoms:
                        sites.append((np.mean(ring_atoms, axis=0), 'aromatic'))
                # sidechain H-bond for Ser/Thr/Tyr/Asn/Gln/Trp
                if resname in ('SER', 'THR', 'TYR'):
                    for aname in ('OG', 'OG1', 'OH'):
                        if aname in res:
                            sites.append((np.asarray(res[aname].coord, dtype=np.float64), 'donor'))
                            sites.append((np.asarray(res[aname].coord, dtype=np.float64), 'acceptor'))
                if resname in ('ASN', 'GLN', 'TRP'):
                    for aname in ('ND2', 'NE1', 'NE2'):
                        if aname in res:
                            sites.append((np.asarray(res[aname].coord, dtype=np.float64), 'donor'))
                    for aname in ('OD1', 'OE1'):
                        if aname in res:
                            sites.append((np.asarray(res[aname].coord, dtype=np.float64), 'acceptor'))
                if resname in _ANCHOR_HYDRO:
                    if 'CB' in res:
                        sites.append((np.asarray(res['CB'].coord, dtype=np.float64), 'hydrophobic'))
                    else:
                        sites.append((ca, 'hydrophobic'))

    return {
        'sites': sites,
        'n_pocket_residues': n_res_in_pocket,
    }, None


def evaluate_idea_g_anchor_richness(fpocket_protein_pdb=None, molecules_with_pos=None,
                                    centroid_radius=12.0):
    """
    想法G（P4）：相互作用锚点丰富度 = 0.35 richness + 0.35 coverage + 0.30 balance。
    仅用蛋白原子类型；需要蛋白 PDB + 配体质心（或失败）。
    """
    empty = {
        'score': 0.0, 'success': False, 'quality_label': 'unknown',
        's_richness': None, 's_coverage': None, 's_balance': None,
        'n_eff': 0, 'n_cap': None, 'n_clusters': 0,
        'message': '需要蛋白 PDB 与配体坐标以估计口袋锚点',
    }
    if not fpocket_protein_pdb or not str(fpocket_protein_pdb).strip():
        return empty
    prot = Path(fpocket_protein_pdb).expanduser().resolve()
    if not prot.is_file():
        empty['message'] = f'蛋白 PDB 不存在: {prot}'
        return empty
    center = _ligand_centroid_from_molecules(molecules_with_pos)
    if center is None:
        empty['message'] = '无法从分子估计口袋中心'
        return empty

    data, err = _collect_pocket_anchor_sites(prot, center, radius=centroid_radius)
    if err or not data:
        empty['message'] = err or '锚点收集失败'
        return empty

    sites = data['sites']
    n_res = int(data['n_pocket_residues'])
    if n_res <= 0 or not sites:
        empty['message'] = (
            f'口袋邻域（{centroid_radius}Å）内无蛋白残基/锚点；请确认 PDB 与配体坐标同一参考系'
        )
        return empty
    n_res = max(1, n_res)
    cats = {'donor': 0, 'acceptor': 0, 'aromatic': 0, 'ionic': 0, 'hydrophobic': 0}
    coords = []
    for xyz, cat in sites:
        if cat in cats:
            cats[cat] += 1
        if cat in ('donor', 'acceptor', 'aromatic', 'ionic'):
            coords.append(xyz)

    n_eff = cats['donor'] + cats['acceptor'] + cats['aromatic'] + cats['ionic']
    n_cap = float(np.clip(0.8 * n_res, 12, 40))
    s_rich = float(min(1.0, n_eff / n_cap)) if n_cap > 0 else 0.0

    # Coverage via DBSCAN on polar/ionic/aromatic anchors
    s_cov = 0.30
    n_clusters = 0
    if len(coords) >= 2 and DBSCAN is not None:
        X = np.asarray(coords, dtype=np.float64)
        labels = DBSCAN(eps=3.5, min_samples=2).fit(X).labels_
        uniq = [l for l in set(labels) if l >= 0]
        n_clusters = len(uniq)
        if n_clusters < 2:
            s_cov = 0.45
            # tighten if single tiny cluster
            if n_clusters == 1:
                mask = labels == uniq[0]
                rad = float(np.max(np.linalg.norm(X[mask] - X[mask].mean(axis=0), axis=1)))
                if rad < 2.0:
                    s_cov = 0.30
        else:
            cents = [X[labels == lab].mean(axis=0) for lab in uniq]
            dists = []
            for i in range(len(cents)):
                for j in range(i + 1, len(cents)):
                    dists.append(float(np.linalg.norm(cents[i] - cents[j])))
            dmin = min(dists) if dists else 0.0
            if 4.0 <= dmin <= 14.0:
                s_cov = 1.0
            elif dmin < 4.0:
                s_cov = 0.55
            else:
                s_cov = 0.70
    elif len(coords) < 2:
        s_cov = 0.25

    # Balance: Shannon entropy over 4 interaction classes (ignore hydrophobic for balance core)
    counts = np.array([
        cats['donor'], cats['acceptor'], cats['aromatic'], cats['ionic']
    ], dtype=np.float64)
    if counts.sum() <= 0:
        s_bal = 0.0
    else:
        p = counts / counts.sum()
        p = p[p > 0]
        ent = float(-np.sum(p * np.log(p)) / np.log(4.0))
        s_bal = ent
        if ent < 0.5:
            s_bal = min(s_bal, 0.5)

    score = 0.35 * s_rich + 0.35 * s_cov + 0.30 * s_bal
    label = 'high' if score >= 0.6 else ('medium' if score >= 0.3 else 'low')
    # Keep a capped list of sites for 2D viz (xyz + category)
    site_xyz = []
    site_cat = []
    for xyz, cat in sites[:800]:
        site_xyz.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
        site_cat.append(cat)
    return {
        'score': float(min(1.0, score)),
        'success': True,
        'quality_label': label,
        's_richness': s_rich,
        's_coverage': s_cov,
        's_balance': s_bal,
        'n_eff': int(n_eff),
        'n_cap': n_cap,
        'n_clusters': n_clusters,
        'n_pocket_residues': n_res,
        'anchor_counts': cats,
        'ligand_centroid': [float(center[0]), float(center[1]), float(center[2])],
        'centroid_radius': float(centroid_radius),
        'site_xyz': site_xyz,
        'site_categories': site_cat,
        'message': (
            f"anchor={score:.3f} rich={s_rich:.2f} cov={s_cov:.2f} bal={s_bal:.2f} "
            f"Neff={n_eff}/{n_cap:.0f} clusters={n_clusters}"
        ),
    }


# =============================================================================
# 想法G（旧）：分子尺寸一致性 — 仅辅图/诊断，不进三层总分
# =============================================================================

def evaluate_idea_g_size_consistency(molecules_with_pos):
    """
    想法G：分子尺寸（原子数、分子量）的一致性

    Returns:
        dict: {score, n_atoms_mean, n_atoms_std, mw_mean, mw_std, mw_cv,
               n_atoms_list, mw_list, quality_label, success}
    """
    if not molecules_with_pos or Chem is None:
        return {
            'score': 0.0, 'n_atoms_mean': None, 'mw_mean': None, 'mw_cv': None,
            'n_atoms_list': [], 'mw_list': [],
            'quality_label': 'unknown', 'success': False, 'message': '无分子数据'
        }

    if Descriptors is None:
        return {
            'score': 0.5, 'n_atoms_mean': None, 'mw_mean': None, 'mw_cv': None,
            'n_atoms_list': [], 'mw_list': [],
            'quality_label': 'medium', 'success': False, 'message': '无法计算分子量'
        }

    n_atoms_list, mw_list = [], []
    for mol, _ in molecules_with_pos:
        if mol is None:
            continue
        try:
            n_atoms_list.append(mol.GetNumAtoms())
            mw_list.append(Descriptors.ExactMolWt(mol))
        except Exception:
            pass

    if len(mw_list) < 2:
        return {
            'score': 0.5,
            'n_atoms_mean': float(np.mean(n_atoms_list)) if n_atoms_list else None,
            'mw_mean': float(np.mean(mw_list)) if mw_list else None,
            'mw_cv': None,
            'n_atoms_list': n_atoms_list, 'mw_list': mw_list,
            'quality_label': 'medium', 'success': True, 'message': '样本过少'
        }

    n_atoms_mean = float(np.mean(n_atoms_list))
    n_atoms_std  = float(np.std(n_atoms_list))
    mw_mean = float(np.mean(mw_list))
    mw_std  = float(np.std(mw_list))
    mw_cv   = mw_std / mw_mean if mw_mean > 0 else 0.0

    # CV 阈值：收紧以提高区分度
    if mw_cv < 0.15:
        score = 1.0
        label = 'high'
    elif mw_cv < 0.25:
        score = 0.8
        label = 'high'
    elif mw_cv < 0.4:
        score = 0.5
        label = 'medium'
    else:
        score = max(0.0, 0.5 - (mw_cv - 0.4) * 0.8)
        label = 'low'

    return {
        'score': min(1.0, score),
        'n_atoms_mean': n_atoms_mean, 'n_atoms_std': n_atoms_std,
        'mw_mean': mw_mean, 'mw_std': mw_std, 'mw_cv': mw_cv,
        'n_atoms_list': n_atoms_list, 'mw_list': mw_list,
        'quality_label': label,
        'success': True,
        'message': f'分子量均值={mw_mean:.0f}, CV={mw_cv:.2f}'
    }


# =============================================================================
# 想法H：口袋体积（MC 占据 或可选 FPocket 蛋白/配体）
# =============================================================================

# 配体 FPocket 成功时，想法 H 的分数按配体 Volume 与 MC 同档区间（optimal_min/max, zero_below/above）计算。
# 仅当配体 FPocket 不可用、但蛋白 FPocket 成功时，用蛋白 Volume 与下列宽区间回退评分。
IDEA_H_FPOCKET_PROTEIN_FALLBACK_OPTIMAL_MIN = 300.0
IDEA_H_FPOCKET_PROTEIN_FALLBACK_OPTIMAL_MAX = 2200.0
IDEA_H_FPOCKET_PROTEIN_FALLBACK_ZERO_BELOW = 80.0
IDEA_H_FPOCKET_PROTEIN_FALLBACK_ZERO_ABOVE = 4500.0

# 可视化：参照刻度（350–750 满分，80 / 2000 归零尾）
IDEA_H_VIZ_REF_OPT_MIN = 350.0
IDEA_H_VIZ_REF_OPT_MAX = 750.0
IDEA_H_VIZ_REF_ZERO_BELOW = 80.0
IDEA_H_VIZ_REF_ZERO_ABOVE = 2000.0
# H volume figure: fixed axis cap (extend only if scored volume exceeds this)
IDEA_H_VIZ_X_MAX_DEFAULT = 1500.0
# Light orange for sub-optimal reference bands (between zero tails and full-score band)
IDEA_H_VIZ_SUBOPT_ORANGE = '#FFD4A8'
IDEA_H_VIZ_SUBOPT_ALPHA = 0.20


def _vdw_radius_from_atomic_num(z):
    z = int(z)
    return {
        1: 1.10, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
        15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98,
    }.get(z, 1.80)


def _idea_h_score_from_volume(volume_ang3, optimal_min, optimal_max, zero_below, zero_above):
    """想法 H 体积分段线性评分（MC 与 FPocket 使用不同 optimal/zero 参数）。"""
    zb, za = float(zero_below), float(zero_above)
    v = float(volume_ang3)
    if v < zb or v >= za:
        score = 0.0
    elif v < optimal_min:
        denom = optimal_min - zb
        score = (v - zb) / denom if denom > 0 else 0.0
    elif v <= optimal_max:
        score = 1.0
    else:
        denom = za - optimal_max
        score = max(0.0, 1.0 - (v - optimal_max) / denom) if denom > 0 else 0.0
    if score >= 0.8:
        label = 'high'
    elif score >= 0.5:
        label = 'medium'
    else:
        label = 'low'
    return min(1.0, score), label


def _parse_fpocket_info_txt(text):
    """解析 FPocket 的 *_info.txt 中各 Pocket 块（Volume、Score、Druggability 等）。"""
    pockets = []
    rx_hdr = re.compile(r'^Pocket\s+(\d+)\s*:\s*$', re.MULTILINE)

    def _grab(block, pat):
        mm = re.search(pat, block, re.MULTILINE | re.IGNORECASE)
        if not mm:
            return None
        try:
            return float(mm.group(1))
        except (TypeError, ValueError):
            return None

    for m in rx_hdr.finditer(text):
        start = m.end()
        m2 = rx_hdr.search(text, start)
        end = m2.start() if m2 else len(text)
        block = text[start:end]
        pid = int(m.group(1))
        pockets.append({
            'pocket_id': pid,
            'score': _grab(block, r'^\s*Score\s*:\s*([0-9.eE+-]+)\s*$'),
            'druggability': _grab(block, r'^\s*Druggability\s+Score\s*:\s*([0-9.eE+-]+)\s*$'),
            'n_alpha_spheres': _grab(block, r'^\s*Number\s+of\s+Alpha\s+Spheres\s*:\s*([0-9.eE+-]+)\s*$'),
            'volume': _grab(block, r'^\s*Volume\s*:\s*([0-9.eE+-]+)\s*$'),
            'hydrophobicity_score': _grab(block, r'^\s*Hydrophobicity\s+score\s*:\s*([0-9.eE+-]+)\s*$'),
        })
    return pockets


def resolve_receptor_pdb_from_pt(pt_path, protein_root=None):
    """
    从 .pt 的 protein_filename / ligand_filename 解析本地匹配受体 PDB。
    常见容器路径 ``/workspace/data/crossdocked_...`` 会映射到仓库
    ``data/crossdocked_pocket10_test_only``。
    """
    pt_path = Path(pt_path)
    if not pt_path.is_file():
        return None
    try:
        raw = load_pt_file(pt_path)
    except Exception:
        return None
    data = raw.get('data') if isinstance(raw, dict) else None
    protein_fn = getattr(data, 'protein_filename', None) if data is not None else None
    ligand_fn = getattr(data, 'ligand_filename', None) if data is not None else None
    if isinstance(data, dict):
        protein_fn = protein_fn or data.get('protein_filename')
        ligand_fn = ligand_fn or data.get('ligand_filename')
    if not protein_fn:
        return None

    pf = Path(str(protein_fn))
    if pf.is_file():
        return str(pf.resolve())

    search_roots = []
    if protein_root:
        search_roots.append(Path(protein_root))
    search_roots.extend([
        REPO_ROOT / 'data' / 'crossdocked_pocket10_test_only',
        REPO_ROOT / 'data' / 'crossdocked_v1.1_rmsd1.0_pocket10',
        REPO_ROOT / 'data',
    ])
    for extra in (Path('/data/ye/protein-ligand'), Path('/data/ye')):
        if extra.is_dir():
            search_roots.append(extra)

    rel_bits = []
    if ligand_fn:
        lig_parent = Path(str(ligand_fn)).parent
        if str(lig_parent) not in ('.', ''):
            rel_bits.append(lig_parent / pf.name)
    if len(pf.parts) >= 2:
        rel_bits.append(Path(*pf.parts[-2:]))
    rel_bits.append(Path(pf.name))

    seen = set()
    for root in search_roots:
        if not root or not Path(root).exists():
            continue
        root = Path(root)
        for rel in rel_bits:
            cand = (root / rel).resolve()
            key = str(cand)
            if key in seen:
                continue
            seen.add(key)
            if cand.is_file():
                return str(cand)
    return None


def resolve_fpocket_executable(fpocket_cmd='fpocket'):
    """
    解析 fpocket 可执行文件路径：--fpocket_cmd、FPOCKET_CMD 环境变量、PATH、CONDA_PREFIX/bin。
    找不到则返回 None。
    """
    def _try(s):
        if not s or not str(s).strip():
            return None
        s = str(s).strip()
        p = Path(s)
        if p.is_file() and os.access(p, os.X_OK):
            return str(p.resolve())
        w = shutil.which(s)
        return w

    cmd = (fpocket_cmd or 'fpocket').strip() or 'fpocket'
    x = _try(cmd)
    if x:
        return x
    x = _try(os.environ.get('FPOCKET_CMD', ''))
    if x:
        return x
    conda = os.environ.get('CONDA_PREFIX', '').strip()
    if conda:
        for name in ('fpocket', 'FPocket'):
            cp = Path(conda) / 'bin' / name
            if cp.is_file() and os.access(cp, os.X_OK):
                return str(cp.resolve())
    for name in ('fpocket', 'FPocket'):
        w = shutil.which(name)
        if w:
            return w
    return None


def run_fpocket_on_pdb(pdb_path, fpocket_cmd='fpocket', timeout=600):
    """
    在临时目录中复制 PDB 并运行 fpocket，避免多进程评估时共用同一 ``*_out`` 目录互相覆盖。

    Returns:
        (ok, message, pockets_list)
    """
    pdb_path = Path(pdb_path).resolve()
    if not pdb_path.is_file():
        return False, f'PDB 不存在: {pdb_path}', []
    exe = resolve_fpocket_executable(fpocket_cmd)
    if not exe:
        return (
            False,
            '未找到 fpocket 可执行文件。可选: conda install -c conda-forge fpocket；'
            '或 export FPOCKET_CMD=/绝对路径/fpocket；或 --fpocket_cmd 指向可执行文件',
            [],
        )
    stem = pdb_path.stem
    with tempfile.TemporaryDirectory(prefix='fpocket_') as td:
        td = Path(td)
        work_pdb = td / f'{stem}.pdb'
        shutil.copy2(pdb_path, work_pdb)
        cmd = [exe, '-f', str(work_pdb)]
        try:
            r = subprocess.run(cmd, cwd=str(td), capture_output=True, text=True, timeout=int(timeout))
        except FileNotFoundError:
            return False, f'无法执行（路径失效）: {exe}', []
        except subprocess.TimeoutExpired:
            return False, f'fpocket 超时 ({timeout}s)', []
        if r.returncode != 0:
            err = (r.stderr or r.stdout or '')[:800]
            return False, f'fpocket 退出码 {r.returncode}: {err}', []
        out_dir = td / f'{stem}_out'
        info_path = out_dir / f'{stem}_info.txt'
        if not info_path.is_file():
            infos = sorted(out_dir.glob('*_info.txt')) if out_dir.is_dir() else []
            info_path = infos[0] if infos else None
        if info_path is None or not info_path.is_file():
            return False, '未找到 FPocket 输出的 *_info.txt', []
        text = info_path.read_text(encoding='utf-8', errors='replace')
        pockets = _parse_fpocket_info_txt(text)
    return True, 'ok', pockets


def _fpocket_pick_pocket(pockets, pocket_index_1based):
    """pocket_index_1based：与 info.txt 中 Pocket N 的 N 一致（默认 1）。"""
    if not pockets:
        return None
    want = int(pocket_index_1based)
    for p in pockets:
        if p.get('pocket_id') == want:
            return p
    idx = want - 1
    if 0 <= idx < len(pockets):
        return pockets[idx]
    return pockets[0]


def _write_ligands_multimodel_pdb(molecules_with_pos, out_path, max_models=50):
    """
    将多条 (mol, pos) 写成**单结构 PDB**（仅 ATOM + CRYST1 + END），供 FPocket 读取。

    FPocket 对输入常按「蛋白」解析：会忽略/剥离 HETATM，且对 MODEL/ENDMDL 与块内 END 的兼容性差，
    易出现 “contains no atoms”。此处将 HETATM 改为 ATOM、去掉 RDKit 块中的 END/CONECT，
    每条构象用不同残基序号拼成一条链，避免空文件。
    """
    if Chem is None:
        return False, 'RDKit 未安装，无法写配体 PDB'
    try:
        from rdkit.Geometry import Point3D
    except ImportError:
        return False, 'RDKit.Geometry 不可用'

    all_lines = []
    coords_box = []
    global_serial = 1
    n_mols_written = 0

    for mi, (mol, pos) in enumerate(molecules_with_pos):
        if n_mols_written >= max_models:
            break
        if mol is None or pos is None:
            continue
        pos_arr = np.asarray(pos, dtype=np.float64)
        if pos_arr.ndim == 1:
            pos_arr = pos_arr.reshape(-1, 3)
        n_atom = mol.GetNumAtoms()
        if pos_arr.shape[0] != n_atom:
            continue
        mol_h = Chem.Mol(mol)
        conf = Chem.Conformer(mol_h.GetNumAtoms())
        for i in range(mol_h.GetNumAtoms()):
            x, y, z = pos_arr[i]
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        mol_h.RemoveAllConformers()
        mol_h.AddConformer(conf)
        block = Chem.MolToPDBBlock(mol_h)
        resseq = n_mols_written + 1
        for line in block.splitlines():
            if len(line) < 54:
                continue
            rec = line[:6]
            if rec.startswith('HETATM'):
                line = 'ATOM  ' + line[6:]
            elif not rec.startswith('ATOM'):
                continue
            # PDB 列 23–26：残基序号（1-based 列号）；每条构象一个残基，避免重叠
            new_line = line[:22] + f'{resseq:4d}' + line[26:]
            new_line = new_line[:6] + f'{global_serial:5d}' + new_line[11:]
            all_lines.append(new_line)
            try:
                coords_box.append((
                    float(new_line[30:38]), float(new_line[38:46]), float(new_line[46:54]),
                ))
            except ValueError:
                pass
            global_serial += 1
        n_mols_written += 1

    if not all_lines:
        return False, '无有效 (mol,pos) 可写入配体 PDB（或 RDKit 未输出 ATOM/HETATM 行）'

    if coords_box:
        xs = [c[0] for c in coords_box]
        ys = [c[1] for c in coords_box]
        zs = [c[2] for c in coords_box]
        margin = 50.0
        a = max(xs) - min(xs) + 2.0 * margin
        b = max(ys) - min(ys) + 2.0 * margin
        c = max(zs) - min(zs) + 2.0 * margin
        a, b, c = max(a, 80.0), max(b, 80.0), max(c, 80.0)
    else:
        a = b = c = 100.0
    cryst1 = f'CRYST1{a:9.3f}{b:9.3f}{c:9.3f}  90.00  90.00  90.00 P 1           1\n'
    text = cryst1 + '\n'.join(all_lines) + '\nEND\n'

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(text, encoding='utf-8')
    return True, ''


def _sample_uniform_in_ball(center, radius, n, rng):
    """n 个均匀分布于球内的 3D 点（Marsaglia 方向 + r∝u^(1/3)）。"""
    center = np.asarray(center, dtype=np.float64).reshape(1, 3)
    u = rng.uniform(0.0, 1.0, size=n)
    v = rng.normal(size=(n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    v = v / norms
    rad = radius * (u ** (1.0 / 3.0))
    return center + rad[:, None] * v


def _load_protein_heavy_coords(protein_pdb):
    """
    Load protein heavy-atom coordinates + atomic numbers from a PDB file.

    Prefers Bio.PDB; falls back to fixed-column PDB parsing.
    Returns (coords (N,3), atom_nums (N,), err_msg_or_None).
    """
    _SYM_Z = {
        'C': 6, 'N': 7, 'O': 8, 'S': 16, 'P': 15, 'SE': 34,
        'MG': 12, 'ZN': 30, 'CA': 20, 'FE': 26, 'CL': 17, 'NA': 11,
        'F': 9, 'BR': 35, 'I': 53, 'K': 19, 'MN': 25, 'CU': 29,
    }
    prot = Path(protein_pdb).expanduser().resolve()
    if not prot.is_file():
        return None, None, f'蛋白 PDB 不存在: {prot}'

    coords_list = []
    z_list = []

    try:
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('prot', str(prot))
        for atom in structure.get_atoms():
            el = (atom.element or '').strip().upper()
            if not el or el in ('H', 'D'):
                continue
            try:
                pos = np.asarray(atom.coord, dtype=np.float64)
            except Exception:
                continue
            coords_list.append(pos)
            z_list.append(int(_SYM_Z.get(el, 6)))
    except Exception:
        coords_list = []
        z_list = []

    if not coords_list:
        try:
            text = prot.read_text(encoding='utf-8', errors='replace')
        except Exception as exc:
            return None, None, f'无法读取蛋白 PDB: {exc}'
        for line in text.splitlines():
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            if len(line) < 54:
                continue
            el_col = line[76:78].strip().upper() if len(line) >= 78 else ''
            name = line[12:16].strip().upper()
            if el_col in ('H', 'D'):
                continue
            if not el_col and name.startswith('H'):
                continue
            el = el_col or ''.join(c for c in name if c.isalpha())[:2].upper()
            if el in ('H', 'D'):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                zc = float(line[46:54])
            except ValueError:
                continue
            coords_list.append(np.array([x, y, zc], dtype=np.float64))
            z_list.append(int(_SYM_Z.get(el, 6)))

    if not coords_list:
        return None, None, f'蛋白 PDB 无重原子坐标: {prot}'

    coords = np.vstack(coords_list).astype(np.float64)
    atom_nums = np.asarray(z_list, dtype=np.int32)
    return coords, atom_nums, None


def _evaluate_idea_h_mc_only(molecules_with_pos, optimal_min=350, optimal_max=750,
                             zero_below=80.0, zero_above=2000.0,
                             centroid_radius=10.0, n_mc_samples=20000,
                             as_fallback=False):
    """
    想法H 回退：质心均值 + 球内配体 vdW 占据 × 球体积。

    注意：这不是真实口袋空腔体积（忽略蛋白、随配体堆叠膨胀），仅在无受体 PDB 时使用。
    """
    if not molecules_with_pos:
        return {
            'score': 0.0, 'volume_ang3': None, 'volume_method': None,
            'quality_label': 'unknown', 'success': False, 'message': '无分子数据'
        }

    atom_data = _collect_atom_data(molecules_with_pos)
    if atom_data is None or len(atom_data['centroids']) < 1:
        return {
            'score': 0.0, 'volume_ang3': None, 'volume_method': None,
            'quality_label': 'unknown', 'success': False, 'message': '无分子质心'
        }

    centroids = np.asarray(atom_data['centroids'], dtype=np.float64)
    ref_center = centroids.mean(axis=0)
    n_centroids = len(centroids)
    r = float(centroid_radius)
    v_ball = (4.0 / 3.0) * np.pi * (r ** 3)

    coords = np.asarray(atom_data['coords'], dtype=np.float64)
    atom_nums = np.asarray(atom_data['atom_nums'], dtype=np.int32)
    if len(coords) < 1:
        return {
            'score': 0.0, 'volume_ang3': None, 'volume_method': None,
            'quality_label': 'unknown', 'success': False, 'message': '无原子坐标'
        }

    radii = np.array([_vdw_radius_from_atomic_num(z) for z in atom_nums], dtype=np.float64)

    rng = np.random.default_rng(42)
    samples = _sample_uniform_in_ball(ref_center, r, n_mc_samples, rng)

    chunk = 2048
    n_hit = 0
    for s in range(0, n_mc_samples, chunk):
        blk = samples[s : s + chunk]
        d = np.linalg.norm(blk[:, None, :] - coords[None, :, :], axis=2) - radii[None, :]
        n_hit += int((d < 0.0).any(axis=1).sum())

    frac = n_hit / float(n_mc_samples)
    volume_ang3 = float(frac * v_ball)
    volume_method = (
        'mean_centroid_10A_ligand_mc_fallback' if as_fallback
        else 'mean_centroid_10A_ligand_mc'
    )

    zb, za = float(zero_below), float(zero_above)
    score, label = _idea_h_score_from_volume(volume_ang3, optimal_min, optimal_max, zb, za)

    return {
        'score': score,
        'volume_ang3': volume_ang3,
        'volume_method': volume_method,
        'quality_label': label,
        'success': True,
        'optimal_vol_min': optimal_min,
        'optimal_vol_max': optimal_max,
        'vol_score_zero_below': zb,
        'vol_score_zero_above': za,
        'centroid_radius': r,
        'n_centroids': n_centroids,
        'reference_centroid': ref_center.tolist(),
        'single_sphere_ref_volume_ang3': float(v_ball),
        'ligand_occupancy_fraction': float(frac),
        'cavity_fraction': None,
        'n_protein_heavy': None,
        'protein_probe_ang': None,
        'ligand_proximity_ang': None,
        'fpocket_protein_pdb': None,
        'fpocket_protein_volume_ang3': None,
        'fpocket_ligand_volume_ang3': None,
        'fpocket_protein_pocket_meta': None,
        'fpocket_ligand_pocket_meta': None,
        'fpocket_ligand_success': None,
        'fpocket_skipped': False,
        'fpocket_skip_reason': None,
        'fpocket_resolved_executable': None,
        'volume_score_band': 'mc_centroid_10A_fallback' if as_fallback else 'mc_centroid_10A',
        'message': (
            f'{"[fallback] " if as_fallback else ""}'
            f'配体占据体积≈{volume_ang3:.0f} Å³ (质心均值 {r:.0f}Å 球, n分子={n_centroids}, '
            f'占据率={frac*100:.1f}%; 非蛋白空腔)'
        ),
    }


def _evaluate_idea_h_protein_cavity_mc(
    molecules_with_pos,
    protein_pdb,
    optimal_min=350,
    optimal_max=750,
    zero_below=80.0,
    zero_above=2000.0,
    centroid_radius=10.0,
    n_mc_samples=20000,
    protein_probe_ang=1.4,
    ligand_proximity_ang=3.0,
):
    """
    POVME-style cavity: points in ligand-centered sphere that
      (1) do not clash with protein heavy atoms (vdW + probe), and
      (2) lie within ligand_proximity_ang of any generated ligand heavy atom.
    """
    if not molecules_with_pos:
        return {
            'score': 0.0, 'volume_ang3': None, 'volume_method': None,
            'quality_label': 'unknown', 'success': False, 'message': '无分子数据',
        }

    atom_data = _collect_atom_data(molecules_with_pos)
    if atom_data is None or len(atom_data['centroids']) < 1:
        return {
            'score': 0.0, 'volume_ang3': None, 'volume_method': None,
            'quality_label': 'unknown', 'success': False, 'message': '无分子质心',
        }

    prot_coords, prot_z, err = _load_protein_heavy_coords(protein_pdb)
    if err or prot_coords is None or len(prot_coords) < 1:
        return None  # caller falls back

    centroids = np.asarray(atom_data['centroids'], dtype=np.float64)
    ref_center = centroids.mean(axis=0)
    n_centroids = len(centroids)
    r = float(centroid_radius)
    v_ball = (4.0 / 3.0) * np.pi * (r ** 3)
    probe = float(protein_probe_ang)
    prox = float(ligand_proximity_ang)

    lig_coords = np.asarray(atom_data['coords'], dtype=np.float64)
    lig_z = np.asarray(atom_data['atom_nums'], dtype=np.int32)
    heavy_mask = lig_z > 1
    if heavy_mask.any():
        lig_heavy = lig_coords[heavy_mask]
    else:
        lig_heavy = lig_coords
    if len(lig_heavy) < 1:
        return {
            'score': 0.0, 'volume_ang3': None, 'volume_method': None,
            'quality_label': 'unknown', 'success': False, 'message': '无配体重原子',
        }

    prot_radii = np.array(
        [_vdw_radius_from_atomic_num(int(z)) + probe for z in prot_z],
        dtype=np.float64,
    )

    # Restrict protein atoms to those near the sphere (speed)
    d_c = np.linalg.norm(prot_coords - ref_center.reshape(1, 3), axis=1)
    keep_p = d_c <= (r + float(prot_radii.max()) + 0.5)
    if keep_p.any():
        prot_coords = prot_coords[keep_p]
        prot_radii = prot_radii[keep_p]

    rng = np.random.default_rng(42)
    samples = _sample_uniform_in_ball(ref_center, r, int(n_mc_samples), rng)

    chunk = 1024
    n_cav = 0
    n_mc = int(n_mc_samples)
    for s in range(0, n_mc, chunk):
        blk = samples[s : s + chunk]
        # near ligand cloud
        d_lig = np.linalg.norm(blk[:, None, :] - lig_heavy[None, :, :], axis=2)
        near_lig = (d_lig <= prox).any(axis=1)
        if not near_lig.any():
            continue
        cand = blk[near_lig]
        # not inside protein
        d_prot = np.linalg.norm(cand[:, None, :] - prot_coords[None, :, :], axis=2)
        clash = (d_prot < prot_radii[None, :]).any(axis=1)
        n_cav += int((~clash).sum())

    frac = n_cav / float(n_mc)
    volume_ang3 = float(frac * v_ball)
    volume_method = f'ligand_protein_cavity_mc_{r:g}A'

    zb, za = float(zero_below), float(zero_above)
    score, label = _idea_h_score_from_volume(volume_ang3, optimal_min, optimal_max, zb, za)
    prot_path = str(Path(protein_pdb).expanduser().resolve())

    return {
        'score': score,
        'volume_ang3': volume_ang3,
        'volume_method': volume_method,
        'quality_label': label,
        'success': True,
        'optimal_vol_min': optimal_min,
        'optimal_vol_max': optimal_max,
        'vol_score_zero_below': zb,
        'vol_score_zero_above': za,
        'centroid_radius': r,
        'n_centroids': n_centroids,
        'reference_centroid': ref_center.tolist(),
        'single_sphere_ref_volume_ang3': float(v_ball),
        'ligand_occupancy_fraction': None,
        'cavity_fraction': float(frac),
        'n_protein_heavy': int(len(prot_coords)),
        'protein_probe_ang': probe,
        'ligand_proximity_ang': prox,
        'fpocket_protein_pdb': prot_path,
        'fpocket_protein_volume_ang3': None,
        'fpocket_ligand_volume_ang3': None,
        'fpocket_protein_pocket_meta': None,
        'fpocket_ligand_pocket_meta': None,
        'fpocket_ligand_success': None,
        'fpocket_skipped': False,
        'fpocket_skip_reason': None,
        'fpocket_resolved_executable': None,
        'volume_score_band': 'ligand_protein_cavity_10A',
        'message': (
            f'口袋空腔≈{volume_ang3:.0f} Å³ (prot+lig MC, r={r:.0f}Å, probe={probe:.1f}Å, '
            f'lig≤{prox:.1f}Å, n分子={n_centroids}, 空腔率={frac*100:.1f}%, '
            f'蛋白重原子={len(prot_coords)}, pdb={Path(prot_path).name})'
        ),
    }


def evaluate_idea_h_pocket_size(molecules_with_pos, optimal_min=350, optimal_max=750,
                                 zero_below=80.0, zero_above=2000.0,
                                 centroid_radius=10.0, n_mc_samples=20000,
                                 fpocket_protein_pdb=None, fpocket_cmd='fpocket',
                                 fpocket_pocket_index=1, fpocket_max_ligand_models=50,
                                 fpocket_timeout=600,
                                 fpocket_optimal_min=None, fpocket_optimal_max=None,
                                 fpocket_zero_below=None, fpocket_zero_above=None):
    """
    想法H：口袋体积 —— 优先「配体定位 + 蛋白排除」空腔 MC（POVME 风格）。

    - 有受体 PDB（``fpocket_protein_pdb``）：10 Å 球内、非蛋白重叠且靠近配体云的空腔体积。
    - 无 PDB / 解析失败：回退配体占据 MC（标注 fallback）。
    - 满分带默认 350–750 Å³。FPocket 体积不作为默认（见 legacy 函数）。
    """
    zb, za = float(zero_below), float(zero_above)
    r = float(centroid_radius) if centroid_radius is not None else 10.0
    pdb_arg = fpocket_protein_pdb
    if pdb_arg and str(pdb_arg).strip():
        cav = _evaluate_idea_h_protein_cavity_mc(
            molecules_with_pos,
            protein_pdb=pdb_arg,
            optimal_min=optimal_min,
            optimal_max=optimal_max,
            zero_below=zb,
            zero_above=za,
            centroid_radius=r,
            n_mc_samples=n_mc_samples,
        )
        if cav is not None and cav.get('success'):
            return cav
        # fall through to ligand-only MC
    return _evaluate_idea_h_mc_only(
        molecules_with_pos, optimal_min, optimal_max, zb, za,
        r, n_mc_samples, as_fallback=bool(pdb_arg),
    )


def evaluate_idea_h_pocket_size_legacy_fpocket(molecules_with_pos, optimal_min=350, optimal_max=750,
                                 zero_below=80.0, zero_above=2000.0,
                                 centroid_radius=10.0, n_mc_samples=20000,
                                 fpocket_protein_pdb=None, fpocket_cmd='fpocket',
                                 fpocket_pocket_index=1, fpocket_max_ligand_models=50,
                                 fpocket_timeout=600,
                                 fpocket_optimal_min=None, fpocket_optimal_max=None,
                                 fpocket_zero_below=None, fpocket_zero_above=None):
    """
    旧版 H：可选 FPocket Volume（已弃用主路径；保留供对照）。

    - 未指定 ``fpocket_protein_pdb``：MC（默认 10 Å）。
    - 指定 ``fpocket_protein_pdb`` 且蛋白 FPocket 成功：优先配体 FPocket Volume。
    """
    zb, za = float(zero_below), float(zero_above)
    pdb_arg = fpocket_protein_pdb
    if pdb_arg is None or str(pdb_arg).strip() == '':
        return _evaluate_idea_h_mc_only(
            molecules_with_pos, optimal_min, optimal_max, zb, za,
            centroid_radius, n_mc_samples,
        )

    prot_path = Path(pdb_arg).expanduser().resolve()
    if not prot_path.is_file():
        return {
            'score': 0.0, 'volume_ang3': None, 'volume_method': 'fpocket_failed',
            'quality_label': 'unknown', 'success': False,
            'message': f'FPocket 蛋白 PDB 不存在: {prot_path}',
            'fpocket_protein_pdb': str(prot_path),
            'fpocket_protein_volume_ang3': None,
            'fpocket_ligand_volume_ang3': None,
            'fpocket_protein_pocket_meta': None,
            'fpocket_ligand_pocket_meta': None,
            'fpocket_ligand_success': False,
            'optimal_vol_min': optimal_min,
            'optimal_vol_max': optimal_max,
            'vol_score_zero_below': zb,
            'vol_score_zero_above': za,
        }

    exe0 = resolve_fpocket_executable(fpocket_cmd)
    if exe0 is None:
        mc = _evaluate_idea_h_mc_only(
            molecules_with_pos, optimal_min, optimal_max, zb, za,
            centroid_radius, n_mc_samples,
        )
        mc['fpocket_protein_pdb'] = str(prot_path)
        mc['fpocket_skipped'] = True
        mc['fpocket_skip_reason'] = 'executable_not_found'
        mc['fpocket_resolved_executable'] = None
        hint = (
            'conda install -c conda-forge fpocket；或 export FPOCKET_CMD=/绝对路径/fpocket；'
            '或 --fpocket_cmd 指向可执行文件'
        )
        mc['message'] = f'[FPocket 不可用，已改用 MC 体积评分] {mc["message"]}（{hint}）'
        return mc

    ok_p, msg_p, pockets_p = run_fpocket_on_pdb(prot_path, fpocket_cmd=fpocket_cmd, timeout=fpocket_timeout)
    if not ok_p:
        return {
            'score': 0.0, 'volume_ang3': None, 'volume_method': 'fpocket_failed',
            'quality_label': 'unknown', 'success': False,
            'message': f'蛋白 FPocket 失败: {msg_p}',
            'fpocket_protein_pdb': str(prot_path),
            'fpocket_protein_volume_ang3': None,
            'fpocket_ligand_volume_ang3': None,
            'fpocket_protein_pocket_meta': None,
            'fpocket_ligand_pocket_meta': None,
            'fpocket_ligand_success': False,
            'optimal_vol_min': optimal_min,
            'optimal_vol_max': optimal_max,
            'vol_score_zero_below': zb,
            'vol_score_zero_above': za,
        }

    pick_p = _fpocket_pick_pocket(pockets_p, fpocket_pocket_index)
    vol_p = pick_p.get('volume') if pick_p else None
    if vol_p is None:
        return {
            'score': 0.0, 'volume_ang3': None, 'volume_method': 'fpocket_failed',
            'quality_label': 'unknown', 'success': False,
            'message': f'蛋白 FPocket 未解析到 Pocket {fpocket_pocket_index} 的 Volume 字段',
            'fpocket_protein_pdb': str(prot_path),
            'fpocket_protein_volume_ang3': None,
            'fpocket_ligand_volume_ang3': None,
            'fpocket_protein_pocket_meta': pick_p,
            'fpocket_ligand_pocket_meta': None,
            'fpocket_ligand_success': False,
            'optimal_vol_min': optimal_min,
            'optimal_vol_max': optimal_max,
            'vol_score_zero_below': zb,
            'vol_score_zero_above': za,
        }

    vol_protein = float(vol_p)
    f_omin = float(
        fpocket_optimal_min if fpocket_optimal_min is not None else IDEA_H_FPOCKET_PROTEIN_FALLBACK_OPTIMAL_MIN
    )
    f_omax = float(
        fpocket_optimal_max if fpocket_optimal_max is not None else IDEA_H_FPOCKET_PROTEIN_FALLBACK_OPTIMAL_MAX
    )
    f_zb = float(
        fpocket_zero_below if fpocket_zero_below is not None else IDEA_H_FPOCKET_PROTEIN_FALLBACK_ZERO_BELOW
    )
    f_za = float(
        fpocket_zero_above if fpocket_zero_above is not None else IDEA_H_FPOCKET_PROTEIN_FALLBACK_ZERO_ABOVE
    )

    lig_vol = None
    lig_meta = None
    lig_ok = False
    lig_msg = ''
    if molecules_with_pos:
        with tempfile.TemporaryDirectory(prefix='fpocket_lig_') as ltd:
            lig_pdb = Path(ltd) / 'ligands.pdb'
            w_ok, w_msg = _write_ligands_multimodel_pdb(
                molecules_with_pos, lig_pdb, max_models=int(fpocket_max_ligand_models),
            )
            if w_ok:
                ok_l, msg_l, pockets_l = run_fpocket_on_pdb(
                    lig_pdb, fpocket_cmd=fpocket_cmd, timeout=fpocket_timeout,
                )
                if ok_l:
                    if not pockets_l:
                        lig_msg = '配体 FPocket 未检测到任何口袋（小分子上常见，体积仅供参考）'
                    else:
                        pick_l = _fpocket_pick_pocket(pockets_l, fpocket_pocket_index)
                        if pick_l and pick_l.get('volume') is not None:
                            lig_vol = float(pick_l['volume'])
                            lig_meta = pick_l
                            lig_ok = True
                        else:
                            lig_msg = f'配体 FPocket 无有效 Volume（Pocket {fpocket_pocket_index}）'
                else:
                    lig_msg = f'配体 FPocket 失败: {msg_l}'
            else:
                lig_msg = w_msg

    if lig_ok and lig_vol is not None:
        vol_scored = float(lig_vol)
        score, label = _idea_h_score_from_volume(vol_scored, optimal_min, optimal_max, zb, za)
        omin_out, omax_out = float(optimal_min), float(optimal_max)
        zb_out, za_out = zb, za
        volume_score_band = 'fpocket_ligand_mc_band'
        vol_method = (
            f'fpocket_ligand_pocket{int(lig_meta.get("pocket_id", fpocket_pocket_index))}_mc_band'
        )
    else:
        vol_scored = vol_protein
        score, label = _idea_h_score_from_volume(vol_scored, f_omin, f_omax, f_zb, f_za)
        omin_out, omax_out = f_omin, f_omax
        zb_out, za_out = f_zb, f_za
        volume_score_band = 'fpocket_protein_fallback'
        vol_method = f'fpocket_protein_pocket{int(pick_p.get("pocket_id", fpocket_pocket_index))}_fallback'

    parts = [
        f'蛋白 FPocket 口袋#{int(pick_p.get("pocket_id", fpocket_pocket_index))} '
        f'Volume≈{vol_protein:.0f} Å³（参考）',
    ]
    if lig_ok and lig_vol is not None:
        parts.append(
            f'配体 FPocket Volume≈{lig_vol:.0f} Å³（用于 H 分，区间 {optimal_min:.0f}–{optimal_max:.0f} Å³ 满分，'
            f'<{zb:.0f} 或 ≥{za:.0f} 为 0）'
        )
    elif lig_msg:
        parts.append(f'配体 FPocket: {lig_msg}（H 分已按蛋白 FPocket 体积 + 宽区间回退）')
    else:
        parts.append('配体侧 FPocket 未计算或未得体积；H 分按蛋白 FPocket 体积 + 宽区间回退')

    ad_h = _collect_atom_data(molecules_with_pos) if molecules_with_pos else None
    n_centroids = len(ad_h['centroids']) if ad_h and ad_h.get('centroids') is not None else None

    return {
        'score': score,
        'volume_ang3': vol_scored,
        'volume_method': vol_method,
        'quality_label': label,
        'success': True,
        'optimal_vol_min': omin_out,
        'optimal_vol_max': omax_out,
        'vol_score_zero_below': zb_out,
        'vol_score_zero_above': za_out,
        'volume_score_band': volume_score_band,
        'n_centroids': n_centroids,
        'reference_centroid': None,
        'single_sphere_ref_volume_ang3': None,
        'ligand_occupancy_fraction': None,
        'fpocket_protein_pdb': str(prot_path),
        'fpocket_protein_volume_ang3': vol_protein,
        'fpocket_ligand_volume_ang3': lig_vol,
        'fpocket_protein_pocket_meta': pick_p,
        'fpocket_ligand_pocket_meta': lig_meta,
        'fpocket_ligand_success': lig_ok,
        'fpocket_skipped': False,
        'fpocket_skip_reason': None,
        'fpocket_resolved_executable': exe0,
        'message': '；'.join(parts),
    }


# =============================================================================
# 可视化模块：完整制图套件
# =============================================================================

_NOTEXT_SUFFIX = '_notext'


def _strip_figure_text(fig):
    """
    去掉图中全部可读文字（轴标题、刻度数字、图例、colorbar 标签、
    散点/柱状图上的标注与饼图百分比等），仅保留几何与颜色，便于在排版软件中自行配字。
    """
    if not HAS_MATPLOTLIB:
        return
    from matplotlib.projections.polar import PolarAxes

    supt = getattr(fig, '_suptitle', None)
    if supt is not None:
        try:
            supt.set_visible(False)
        except Exception:
            try:
                supt.set_text('')
            except Exception:
                pass

    for lg in list(getattr(fig, 'legends', []) or []):
        try:
            lg.remove()
        except Exception:
            pass

    for ax in fig.axes:
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
        leg = ax.get_legend()
        if leg is not None:
            try:
                leg.remove()
            except Exception:
                pass
        ax.tick_params(
            axis='both',
            which='both',
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )
        if isinstance(ax, PolarAxes):
            try:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            except Exception:
                pass
        for txt in list(ax.texts):
            try:
                txt.remove()
            except Exception:
                try:
                    txt.set_visible(False)
                except Exception:
                    pass


def _safe_savefig(fig, path, dpi=180, save_notext=True):
    """
    安全保存图像，创建父目录。

    在保存带完整标注的原图后，默认再保存一份仅保留图形元素、无文字的同分辨率图：
    文件名在原 stem 后追加 ``_notext``（如 ``A_vina_hist_notext.png``），便于科研排版自行控制字体。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {path}")

    ext = path.suffix.lower()
    if save_notext and ext in ('.png', '.pdf', '.svg', '.jpg', '.jpeg', '.tif', '.tiff', '.webp'):
        notext_path = path.parent / f"{path.stem}{_NOTEXT_SUFFIX}{path.suffix}"
        # 去字后再 tight_layout 会破坏「主图 + 窄 colorbar」等布局；先冻结各轴位置再恢复
        pos_bounds = [tuple(ax.get_position().bounds) for ax in fig.axes]
        _strip_figure_text(fig)
        if len(fig.axes) == len(pos_bounds):
            for ax, b in zip(fig.axes, pos_bounds):
                ax.set_position(b)
        try:
            fig.tight_layout()
        except Exception:
            pass
        if len(fig.axes) == len(pos_bounds):
            for ax, b in zip(fig.axes, pos_bounds):
                ax.set_position(b)
        fig.savefig(notext_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {notext_path}")

    plt.close(fig)
    return str(path)


def _viz_title(title_prefix, core):
    """Journal-style title: 'Pocket 0 · Vina' or just 'Vina'."""
    p = str(title_prefix).strip() if title_prefix is not None else ''
    if not p:
        return core
    if p.isdigit() or (p.startswith('pocket') is False and p.replace('_', '').isalnum() and len(p) <= 8):
        # numeric / short id → Pocket N
        if p.isdigit():
            return f'Pocket {p} · {core}'
        return f'{p} · {core}'
    return f'{p} · {core}'


def visualize_clustering_2d(molecules_with_pos, idea_b_result=None,
                             output_dir=None, title_prefix='',
                             n_kmeans=5, use_tsne=False):
    """
    Idea B · Mode — visualizations aligned with scoring.

    Primary figure (B_clustering_dbscan.png):
      Uses the **same 3D point set + DBSCAN labels** as ``evaluate_idea_b_clustering``
      (default subset=hetero_heavy, eps=1.5), projected to 2D with PCA for display only.
      Does **not** re-run DBSCAN in 2D on all atoms.

    Auxiliary figures: molecule-index / centroid KMeans / CPK / raw XY
    (not used for Mode score).
    """
    if not HAS_MATPLOTLIB:
        print("⚠️ visualize_clustering_2d needs matplotlib")
        return {}
    if not HAS_SKLEARN:
        print("⚠️ visualize_clustering_2d needs scikit-learn")
        return {}

    try:
        from utils.pocket_vis_nature import ACCENT, FONT, ICEBLUE_EDGE, ICEBLUE_FILL, NEUTRAL, apply_nature_axes
    except ImportError:
        ACCENT, ICEBLUE_FILL, ICEBLUE_EDGE, NEUTRAL = '#F56E1A', '#C5DDF0', '#5A7FA0', '#4A4A4A'
        FONT = {'title': 11, 'label': 9, 'tick': 8, 'annot': 7}

        def apply_nature_axes(ax, grid_y=True, grid_x=False):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    atom_data = (idea_b_result.get('atom_data') if idea_b_result else None) \
                or _collect_atom_data(molecules_with_pos)
    if atom_data is None or len(atom_data['coords']) < 5:
        print("⚠️ Insufficient atom coordinates for clustering plot")
        return {}

    coords_all = atom_data['coords']
    mol_idx_all = atom_data['mol_idx']
    atom_nums_all = atom_data['atom_nums']
    centroids = atom_data['centroids']
    n_mols = atom_data['n_mols']

    # ---- Mode-scored point set (must match evaluate_idea_b_clustering) ----
    subset = 'hetero_heavy'
    eps_scored = 1.5
    min_s_scored = 3
    X_mode = None
    labels_mode = None
    score_b = None
    n_clusters_b = None
    n_noise_b = None

    if idea_b_result and idea_b_result.get('success'):
        subset = str(idea_b_result.get('atom_coord_subset') or subset)
        eps_scored = float(idea_b_result.get('dbscan_eps', eps_scored))
        min_s_scored = int(idea_b_result.get('dbscan_min_samples', min_s_scored))
        score_b = idea_b_result.get('score')
        n_clusters_b = idea_b_result.get('n_clusters')
        n_noise_b = idea_b_result.get('n_noise')
        Xf = idea_b_result.get('coords_focus')
        labs = idea_b_result.get('dbscan_labels')
        if Xf is not None and labs is not None:
            X_mode = np.asarray(Xf, dtype=np.float64)
            labels_mode = np.asarray(labs, dtype=int)
            if len(X_mode) != len(labels_mode):
                X_mode, labels_mode = None, None
        if X_mode is None and subset == 'all' and labs is not None:
            X_mode = np.asarray(coords_all, dtype=np.float64)
            labels_mode = np.asarray(labs, dtype=int)
            if len(X_mode) != len(labels_mode):
                X_mode, labels_mode = None, None

    # Fallback: rebuild subset + run same 3D DBSCAN as scoring (never 2D all-atom)
    if X_mode is None and DBSCAN is not None:
        if subset == 'all':
            X_mode = np.asarray(coords_all, dtype=np.float64)
        else:
            mask = _build_idea_b_coord_mask(molecules_with_pos, subset, 2)
            if mask is not None and mask.sum() >= 5:
                X_mode = np.asarray(coords_all, dtype=np.float64)[mask]
            else:
                X_mode = np.asarray(coords_all, dtype=np.float64)
        clustering = DBSCAN(eps=eps_scored, min_samples=min_s_scored).fit(X_mode)
        labels_mode = clustering.labels_
        n_clusters_b = len(set(labels_mode)) - (1 if -1 in labels_mode else 0)
        n_noise_b = int(np.sum(labels_mode == -1))

    output_dir = Path(output_dir) if output_dir else None
    if not output_dir:
        return {}
    saved = {}

    def _save(name, fig):
        p = output_dir / name
        _safe_savefig(fig, p)
        saved[name.replace('.png', '')] = str(p)

    # ---- Fig 1 (PRIMARY): Mode DBSCAN labels → PCA 2D (display only) ----
    fig1, ax1 = plt.subplots(figsize=(6.2, 5.0))
    if X_mode is not None and len(X_mode) >= 3 and labels_mode is not None:
        pca_m = PCA(n_components=2, random_state=42)
        xy = pca_m.fit_transform(X_mode)
        var = pca_m.explained_variance_ratio_
        unique_lbls = sorted(set(labels_mode.tolist()))
        n_colors = max(len(unique_lbls), 2)
        palette = plt.cm.get_cmap('tab10', n_colors)
        for li, lbl in enumerate(unique_lbls):
            mask = labels_mode == lbl
            color = '#B0B0B0' if lbl == -1 else palette(li % n_colors)
            alpha = 0.35 if lbl == -1 else 0.70
            s = 10 if lbl == -1 else 14
            lname = f'noise ({mask.sum()})' if lbl == -1 else f'mode {lbl} ({mask.sum()})'
            ax1.scatter(
                xy[mask, 0], xy[mask, 1], c=[color], s=s, alpha=alpha,
                label=lname, rasterized=True, edgecolors='none', zorder=3,
            )
        apply_nature_axes(ax1, grid_y=True, grid_x=True)
        ax1.set_xlabel(f'PC1 ({var[0]*100:.1f}%)', fontsize=FONT['label'])
        ax1.set_ylabel(f'PC2 ({var[1]*100:.1f}%)', fontsize=FONT['label'])
        sc_txt = f'{float(score_b):.3f}' if score_b is not None else 'n/a'
        ax1.set_title(
            _viz_title(title_prefix, 'B · Mode (3D DBSCAN→PCA)'),
            fontsize=FONT['title'], pad=8, color=NEUTRAL,
        )
        ax1.text(
            0.02, 0.98,
            f'subset={subset}  eps={eps_scored:g}  min_s={min_s_scored}\n'
            f'clusters={n_clusters_b}  noise={n_noise_b}  B={sc_txt}',
            transform=ax1.transAxes, ha='left', va='top',
            fontsize=FONT['annot'], color='#555555',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#DDDDDD', alpha=0.92),
        )
        if len(unique_lbls) <= 12:
            ax1.legend(fontsize=FONT['annot'], loc='lower right', framealpha=0.9, markerscale=1.4)
    else:
        ax1.text(0.5, 0.5, 'Mode DBSCAN unavailable', ha='center', va='center',
                 transform=ax1.transAxes, color='#888888')
    plt.tight_layout()
    _save('B_clustering_dbscan.png', fig1)

    # Drop obsolete focus duplicate (main plot is already the scored subset)
    for obsolete in ('B_clustering_dbscan_focus.png', 'B_clustering_dbscan_focus_notext.png'):
        op = output_dir / obsolete
        if op.exists():
            try:
                op.unlink()
            except OSError:
                pass

    # ---- Aux: all-atom PCA colored by molecule index (NOT scored) ----
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords_all)
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(coords_scaled)
    pca_var = pca.explained_variance_ratio_
    centroids_scaled = scaler.transform(centroids)
    centroids_2d = pca.transform(centroids_scaled)
    ax_xlabel = f'PC1 ({pca_var[0]*100:.1f}%)'
    ax_ylabel = f'PC2 ({pca_var[1]*100:.1f}%)'

    fig2, ax2 = plt.subplots(figsize=(6.2, 5.0))
    sc2 = ax2.scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=mol_idx_all, cmap='viridis', s=5, alpha=0.35, rasterized=True,
        vmin=0, vmax=max(n_mols - 1, 1),
    )
    plt.colorbar(sc2, ax=ax2, label='Molecule index', shrink=0.85)
    ax2.scatter(
        centroids_2d[:, 0], centroids_2d[:, 1],
        c=ACCENT, s=42, marker='*', zorder=6, alpha=0.9,
        label=f'Centroids (n={len(centroids_2d)})',
        edgecolors='white', linewidths=0.4,
    )
    apply_nature_axes(ax2, grid_y=True, grid_x=True)
    ax2.legend(fontsize=FONT['annot'], loc='upper right')
    ax2.set_xlabel(ax_xlabel, fontsize=FONT['label'])
    ax2.set_ylabel(ax_ylabel, fontsize=FONT['label'])
    ax2.set_title(
        _viz_title(title_prefix, 'B aux · all atoms by mol (not scored)'),
        fontsize=FONT['title'], pad=8, color=NEUTRAL,
    )
    plt.tight_layout()
    _save('B_clustering_molindex.png', fig2)

    # ---- Aux: soft density of Mode points ----
    fig3, ax3 = plt.subplots(figsize=(6.2, 5.0))
    if X_mode is not None and len(X_mode) >= 8:
        pca_d = PCA(n_components=2, random_state=42)
        xy_d = pca_d.fit_transform(X_mode)
        try:
            import seaborn as sns
            sns.kdeplot(
                x=xy_d[:, 0], y=xy_d[:, 1], ax=ax3,
                levels=6, thresh=0.1, fill=True, cmap='Blues', alpha=0.85,
            )
            sns.kdeplot(
                x=xy_d[:, 0], y=xy_d[:, 1], ax=ax3,
                levels=1, thresh=0.12, fill=False, color=ICEBLUE_EDGE, linewidths=1.0,
            )
        except Exception:
            ax3.scatter(xy_d[:, 0], xy_d[:, 1], s=8, c=ICEBLUE_FILL, alpha=0.5)
        apply_nature_axes(ax3, grid_y=True, grid_x=True)
        ax3.set_xlabel('PC1', fontsize=FONT['label'])
        ax3.set_ylabel('PC2', fontsize=FONT['label'])
        ax3.set_title(
            _viz_title(title_prefix, f'B aux · {subset} density'),
            fontsize=FONT['title'], pad=8, color=NEUTRAL,
        )
    else:
        ax3.text(0.5, 0.5, 'Density unavailable', ha='center', va='center',
                 transform=ax3.transAxes)
    plt.tight_layout()
    _save('B_clustering_kde.png', fig3)

    # ---- Aux: centroid KMeans (stored from scoring; not Mode primary score) ----
    fig4, ax4 = plt.subplots(figsize=(6.2, 5.0))
    kmeans_labels = None
    if idea_b_result and idea_b_result.get('centroid_kmeans_labels'):
        kmeans_labels = np.array(idea_b_result['centroid_kmeans_labels'])
    elif KMeans is not None and len(centroids_2d) >= n_kmeans:
        k = min(n_kmeans, len(centroids_2d))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_labels = km.fit_predict(centroids_2d)
    if kmeans_labels is not None:
        n_km = len(set(kmeans_labels))
        pal_km = plt.cm.get_cmap('Set2', max(n_km, 2))
        for ki in range(n_km):
            mk = kmeans_labels == ki
            ax4.scatter(
                centroids_2d[mk, 0], centroids_2d[mk, 1],
                c=[pal_km(ki)], s=70, marker='o',
                label=f'cent {ki} ({mk.sum()})',
                edgecolors='white', linewidths=0.7, zorder=4,
            )
        ax4.legend(fontsize=FONT['annot'], loc='best', ncol=2 if n_km > 5 else 1)
    else:
        ax4.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c=ICEBLUE_EDGE, s=50)
    apply_nature_axes(ax4, grid_y=True, grid_x=True)
    ax4.set_xlabel('PC1', fontsize=FONT['label'])
    ax4.set_ylabel('PC2', fontsize=FONT['label'])
    ax4.set_title(
        _viz_title(title_prefix, 'B aux · mol centroids KMeans'),
        fontsize=FONT['title'], pad=8, color=NEUTRAL,
    )
    plt.tight_layout()
    _save('B_clustering_kmeans.png', fig4)

    # ---- Aux: CPK on Mode atoms when possible ----
    fig5, ax5 = plt.subplots(figsize=(6.2, 5.0))
    if X_mode is not None and subset != 'all':
        mask = _build_idea_b_coord_mask(molecules_with_pos, subset, 2)
        if mask is not None and mask.sum() == len(X_mode):
            atom_nums_plot = atom_nums_all[mask]
            pca_c = PCA(n_components=2, random_state=42)
            xy_c = pca_c.fit_transform(X_mode)
        else:
            atom_nums_plot = atom_nums_all
            xy_c = coords_2d
    else:
        atom_nums_plot = atom_nums_all
        xy_c = coords_2d
    unique_atypes = sorted(set(np.asarray(atom_nums_plot).tolist()))
    for anum in unique_atypes:
        mask_a = np.asarray(atom_nums_plot) == anum
        color_a = _CPK_COLORS.get(int(anum), '#888888')
        name_a = _ELEMENT_NAMES.get(int(anum), f'Z={anum}')
        ax5.scatter(
            xy_c[mask_a, 0], xy_c[mask_a, 1],
            c=color_a, s=10, alpha=0.65,
            label=f'{name_a} ({mask_a.sum()})', rasterized=True,
        )
    apply_nature_axes(ax5, grid_y=True, grid_x=True)
    ax5.set_xlabel('PC1', fontsize=FONT['label'])
    ax5.set_ylabel('PC2', fontsize=FONT['label'])
    ax5.set_title(
        _viz_title(title_prefix, f'B aux · CPK ({subset})'),
        fontsize=FONT['title'], pad=8, color=NEUTRAL,
    )
    ax5.legend(
        fontsize=FONT['annot'], loc='best', markerscale=1.6, framealpha=0.92,
        ncol=2 if len(unique_atypes) > 6 else 1,
    )
    plt.tight_layout()
    _save('B_clustering_cpk.png', fig5)

    # ---- Aux: raw XY of Mode coords ----
    fig6, ax6 = plt.subplots(figsize=(6.2, 5.0))
    if X_mode is not None:
        sc6 = ax6.scatter(
            X_mode[:, 0], X_mode[:, 1],
            c=labels_mode if labels_mode is not None else 'steelblue',
            cmap='tab10', s=8, alpha=0.55, rasterized=True,
        )
        if labels_mode is not None:
            plt.colorbar(sc6, ax=ax6, label='DBSCAN label', shrink=0.85)
        ax6.scatter(
            centroids[:, 0], centroids[:, 1],
            c=ACCENT, s=28, marker='D', alpha=0.85, zorder=5,
            label='Mol centroids', edgecolors='white', linewidths=0.4,
        )
        ax6.legend(fontsize=FONT['annot'])
    apply_nature_axes(ax6, grid_y=True, grid_x=True)
    ax6.set_xlabel('X (Å)', fontsize=FONT['label'])
    ax6.set_ylabel('Y (Å)', fontsize=FONT['label'])
    ax6.set_title(
        _viz_title(title_prefix, f'B aux · {subset} XY (scored labels)'),
        fontsize=FONT['title'], pad=8, color=NEUTRAL,
    )
    plt.tight_layout()
    _save('B_clustering_xy.png', fig6)

    _ = (use_tsne, ICEBLUE_FILL)
    return saved


def visualize_vina_le_combined(
    idea_a_result=None,
    idea_le_result=None,
    output_dir=None,
    title_prefix='',
):
    """
    Idea A: Vina + LE as two narrow box+strip columns on one frame.

    Twin y-axes only (no colorbars): left = Vina, right = LE.
    Points keep RdYlGn coloring; scales are read from the axes.
    """
    if not HAS_MATPLOTLIB:
        return {}
    output_dir = Path(output_dir) if output_dir else None
    if not output_dir:
        return {}

    try:
        from utils.pocket_vis_nature import (
            ACCENT, FONT, ICEBLUE_EDGE, ICEBLUE_FILL, NEUTRAL, apply_nature_axes,
        )
    except ImportError:
        ACCENT, ICEBLUE_FILL, ICEBLUE_EDGE, NEUTRAL = '#F56E1A', '#C5DDF0', '#5A7FA0', '#4A4A4A'
        FONT = {'title': 11, 'label': 9, 'tick': 8, 'annot': 7}

        def apply_nature_axes(ax, grid_y=True, grid_x=False):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if grid_y:
                ax.grid(True, axis='y', alpha=0.3, zorder=0)

    # ---- prepare Vina ----
    vina_ok = False
    vina = np.array([], dtype=np.float64)
    if idea_a_result and idea_a_result.get('success'):
        raw = idea_a_result.get('vina_scores_valid') or idea_a_result.get('vina_scores') or []
        vina = np.asarray(raw, dtype=np.float64)
        vina = vina[np.isfinite(vina)]
        vina = vina[vina < 0]
        vina_ok = len(vina) > 0

    # ---- prepare LE ----
    le_src = idea_le_result or {}
    if not le_src.get('success') and idea_a_result:
        le_src = idea_a_result.get('le_component') or {}
    le_ok = False
    le = np.array([], dtype=np.float64)
    lo = float(IDEA_C_LE_SCORE_LOW)
    hi = float(IDEA_C_LE_SCORE_HIGH)
    if le_src and le_src.get('success'):
        le = np.asarray(le_src.get('le_list') or [], dtype=np.float64)
        le = le[np.isfinite(le)]
        lo = float(le_src.get('le_low_for_score', lo))
        hi = float(le_src.get('le_high_for_score', hi))
        le_ok = len(le) > 0

    if not vina_ok and not le_ok:
        return {}

    def _box_strip(ax, scores, *, x, cmap, vmin, vmax, seed, facecolor):
        rng = np.random.default_rng(seed)
        ax.boxplot(
            scores,
            vert=True,
            patch_artist=True,
            widths=0.28,
            positions=[x],
            manage_ticks=False,
            boxprops=dict(facecolor=facecolor, edgecolor=ICEBLUE_EDGE, alpha=0.85),
            medianprops=dict(color='red', lw=2.0),
            whiskerprops=dict(color='#888888', lw=1.0),
            capprops=dict(color='#888888', lw=1.0),
            flierprops=dict(
                marker='o', markersize=3.5, markerfacecolor='none',
                markeredgecolor='#555555', markeredgewidth=0.7,
            ),
        )
        jitter = rng.uniform(-0.10, 0.10, len(scores))
        # colored points; no colorbar — twin y-axes carry the numeric scales
        ax.scatter(
            x + jitter, scores, c=scores, cmap=cmap, s=18, alpha=0.65,
            zorder=3, vmin=vmin, vmax=vmax, edgecolors='none',
        )

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax_r = ax.twinx()
    xticks, xlabels = [], []
    x_v, x_l = 0.0, 1.0

    if vina_ok:
        mean_v = float(np.mean(vina))
        pct_hit = float(np.mean(vina <= -7.0) * 100)
        _box_strip(
            ax, vina, x=x_v, cmap='RdYlGn_r',
            vmin=float(vina.min()), vmax=float(vina.max()),
            seed=42, facecolor=ICEBLUE_FILL,
        )
        ax.hlines(-7.0, x_v - 0.38, x_v + 0.38, colors='red',
                  linestyles='--', lw=1.15, alpha=0.85, zorder=5)
        y_pad = max(0.5, float(np.ptp(vina)) * 0.08 + 0.1)
        ax.set_ylim(float(vina.min()) - y_pad, float(vina.max()) + y_pad)
        ax.set_ylabel('Vina (kcal/mol)', fontsize=FONT['label'], color=ICEBLUE_EDGE)
        ax.tick_params(axis='y', labelcolor=ICEBLUE_EDGE, labelsize=FONT['tick'])
        xticks.append(x_v)
        xlabels.append('Vina')
        ax.text(
            0.02, 0.98,
            f'Vina mean={mean_v:.2f}  n={len(vina)}  P(≤−7)={pct_hit:.0f}%',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=FONT['annot'], color='#666666',
        )
    else:
        ax.set_yticks([])
        ax.set_ylabel('')

    if le_ok:
        mean_le = float(np.mean(le))
        _box_strip(
            ax_r, le, x=x_l, cmap='RdYlGn',
            vmin=lo, vmax=hi, seed=43, facecolor='#FBC9A6',
        )
        ax_r.hlines(lo, x_l - 0.38, x_l + 0.38, colors='red',
                    linestyles='--', lw=1.0, alpha=0.8, zorder=5)
        ax_r.hlines(hi, x_l - 0.38, x_l + 0.38, colors='green',
                    linestyles='--', lw=1.0, alpha=0.8, zorder=5)
        y_pad = max(0.02, float(np.ptp(le)) * 0.08 + 0.01)
        ax_r.set_ylim(float(le.min()) - y_pad, float(le.max()) + y_pad)
        ax_r.set_ylabel('LE (kcal·mol⁻¹·heavy⁻¹)', fontsize=FONT['label'], color=ACCENT)
        ax_r.tick_params(axis='y', labelcolor=ACCENT, labelsize=FONT['tick'])
        xticks.append(x_l)
        xlabels.append('LE')
        ax.text(
            0.98, 0.98,
            f'LE mean={mean_le:.3f}  n={len(le)}  map {lo:.2f}–{hi:.2f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=FONT['annot'], color='#666666',
        )
    else:
        ax_r.set_yticks([])
        ax_r.set_ylabel('')

    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=FONT['label'])

    # Left axis: Nature style but keep room for twin right axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(ICEBLUE_EDGE)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_color('#888888')
    ax.spines['bottom'].set_linewidth(0.8)
    ax.yaxis.grid(True, color='#E8E8E8', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Right axis: LE numeric scale only (no colorbar)
    ax_r.spines['top'].set_visible(False)
    ax_r.spines['left'].set_visible(False)
    ax_r.spines['bottom'].set_visible(False)
    ax_r.spines['right'].set_visible(True)
    ax_r.spines['right'].set_color(ACCENT)
    ax_r.spines['right'].set_linewidth(1.0)

    ax.set_title(
        _viz_title(title_prefix, 'A · Vina & LE'),
        fontsize=FONT['title'], pad=8, color=NEUTRAL,
    )
    fig.tight_layout()
    path = output_dir / 'A_vina_le.png'
    _safe_savefig(fig, path, dpi=180)

    for obsolete in (
        'A_vina_hist.png', 'A_vina_hist_notext.png',
        'A_vina_box.png', 'A_vina_box_notext.png',
        'A_vina_cdf.png', 'A_vina_cdf_notext.png',
        'A_le_hist.png', 'A_le_hist_notext.png',
        'A_le_box.png', 'A_le_box_notext.png',
        'A_le_cdf.png', 'A_le_cdf_notext.png',
    ):
        op = output_dir / obsolete
        if op.exists():
            try:
                op.unlink()
            except OSError:
                pass

    return {'A_vina_le': str(path)}


def visualize_vina_distribution(idea_a_result, output_dir=None, title_prefix='', idea_le_result=None):
    """Backward-compatible wrapper → combined Vina+LE figure."""
    return visualize_vina_le_combined(
        idea_a_result=idea_a_result,
        idea_le_result=idea_le_result,
        output_dir=output_dir,
        title_prefix=title_prefix,
    )


def visualize_le_distribution(idea_c_result, output_dir=None, title_prefix='', idea_a_result=None):
    """Backward-compatible wrapper → combined Vina+LE figure."""
    return visualize_vina_le_combined(
        idea_a_result=idea_a_result,
        idea_le_result=idea_c_result,
        output_dir=output_dir,
        title_prefix=title_prefix,
    )


def visualize_druglikeness(idea_d_result, output_dir=None, title_prefix='', vina_scores=None):
    """
    Idea D - Druglikeness multi-dimensional visualization (5 separate figures)

    Subplots: QED vs SA scatter (color=vinadock) / QED dist / SA dist / Lipinski / radar
    """
    if not HAS_MATPLOTLIB:
        return {}
    if not idea_d_result.get('success'):
        return {}
    qed_list = np.array(idea_d_result.get('qed_list', []))
    sa_list  = np.array(idea_d_result.get('sa_list',  []))
    lip_list = np.array(idea_d_result.get('lipinski_list', []))
    if len(qed_list) == 0:
        return {}
    output_dir = Path(output_dir) if output_dir else None
    if not output_dir:
        return {}
    saved = {}
    vina_arr = None
    if vina_scores and len(vina_scores) > 0:
        try:
            vina_arr = np.asarray(vina_scores, dtype=np.float64)
        except (TypeError, ValueError):
            vina_arr = None

    # ---- Fig 1: QED vs SA scatter (color=vinadock) ----
    n_pts = min(len(qed_list), len(sa_list))
    try:
        from utils.pocket_vis_nature import ACCENT, FONT, ICEBLUE_EDGE, ICEBLUE_FILL, apply_nature_axes
        from matplotlib.colors import LinearSegmentedColormap
        _cmap = LinearSegmentedColormap.from_list(
            'ice_accent_vina', [ACCENT, '#FBC9A6', ICEBLUE_FILL, ICEBLUE_EDGE],
        )
    except ImportError:
        ACCENT, ICEBLUE_FILL, ICEBLUE_EDGE = '#F56E1A', '#C5DDF0', '#5A7FA0'
        FONT = {'title': 11, 'label': 9, 'tick': 8, 'annot': 7}
        _cmap = 'cividis_r'

        def apply_nature_axes(ax, grid_y=True, grid_x=False):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    use_vina = (
        vina_arr is not None
        and len(vina_arr) > 0
        and np.isfinite(vina_arr[:n_pts]).any()
    )
    if use_vina:
        c_vals = np.asarray(vina_arr[:n_pts], dtype=np.float64)
        # if shorter than n_pts, trim scatter to vina length
        n_use = min(n_pts, len(vina_arr))
        c_vals = np.asarray(vina_arr[:n_use], dtype=np.float64)
        q_use = qed_list[:n_use]
        s_use = sa_list[:n_use]
        finite = np.isfinite(c_vals)
        if finite.any():
            c_label = 'Vina (kcal/mol)'
            c_vmin = float(np.nanmin(c_vals))
            c_vmax = float(np.nanmax(c_vals))
            c_cmap = _cmap
        else:
            use_vina = False
    if not use_vina:
        n_use = n_pts
        q_use = qed_list[:n_use]
        s_use = sa_list[:n_use]
        c_vals = lip_list[:n_use] if len(lip_list) >= n_use else np.ones(n_use)
        c_label = 'Lipinski (0–1)'
        c_vmin, c_vmax = 0, 1
        c_cmap = 'Blues'
    fig1, ax1 = plt.subplots(figsize=(5.8, 4.6))
    sc = ax1.scatter(q_use, s_use,
                     c=c_vals, cmap=c_cmap,
                     s=36, alpha=0.75, edgecolors='white', linewidths=0.3,
                     vmin=c_vmin, vmax=c_vmax, zorder=3)
    cb = fig1.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
    cb.set_label(c_label, fontsize=FONT['annot'])
    cb.ax.tick_params(labelsize=FONT['tick'])
    ax1.axvline(0.5, color='#BBBBBB', linestyle=':', lw=1.0, alpha=0.8)
    ax1.axhline(0.5, color='#BBBBBB', linestyle=':', lw=1.0, alpha=0.8)
    from matplotlib.patches import Rectangle
    rect = Rectangle((0.5, 0.5), 0.5, 0.5, linewidth=0, edgecolor='none',
                      facecolor='#A8C5A0', alpha=0.12, zorder=1)
    ax1.add_patch(rect)
    qed_m = idea_d_result.get('qed_mean', 0)
    sa_m = idea_d_result.get('sa_mean', 0)
    ax1.scatter([qed_m], [sa_m], c=ACCENT, s=90, marker='*', zorder=10,
                edgecolors='white', linewidths=0.5)
    apply_nature_axes(ax1)
    ax1.set_xlabel('QED', fontsize=FONT['label'])
    ax1.set_ylabel('SA', fontsize=FONT['label'])
    ax1.set_xlim(0, 1.02)
    ax1.set_ylim(0, 1.02)
    ax1.set_title(_viz_title(title_prefix, 'C · QED vs SA'), fontsize=FONT['title'], pad=8)
    plt.tight_layout()
    _safe_savefig(fig1, output_dir / 'C_druglikeness_qed_vs_sa.png', dpi=180)
    saved['C_druglikeness_qed_vs_sa'] = str(output_dir / 'C_druglikeness_qed_vs_sa.png')

    # ---- Fig 2: QED distribution ----
    fig2, ax2 = plt.subplots(figsize=(5.0, 3.8))
    ax2.hist(qed_list, bins=20, color=ICEBLUE_FILL, edgecolor='white',
             alpha=0.95, density=True, zorder=3)
    if HAS_SCIPY and gaussian_kde is not None and len(qed_list) > 3:
        kde = gaussian_kde(qed_list, bw_method=0.25)
        xq = np.linspace(0, 1, 200)
        ax2.plot(xq, kde(xq), color=ICEBLUE_EDGE, lw=1.6)
    ax2.axvline(float(np.mean(qed_list)), color=ACCENT, lw=1.2, ls='--')
    apply_nature_axes(ax2)
    ax2.set_xlabel('QED', fontsize=FONT['label'])
    ax2.set_ylabel('Density', fontsize=FONT['label'])
    ax2.set_title(_viz_title(title_prefix, 'C · QED'), fontsize=FONT['title'], pad=8)
    plt.tight_layout()
    _safe_savefig(fig2, output_dir / 'C_druglikeness_qed_dist.png', dpi=180)
    saved['C_druglikeness_qed_dist'] = str(output_dir / 'C_druglikeness_qed_dist.png')

    # ---- Fig 3: SA distribution ----
    fig3, ax3 = plt.subplots(figsize=(5.0, 3.8))
    ax3.hist(sa_list, bins=20, color=ICEBLUE_FILL, edgecolor='white',
             alpha=0.95, density=True, zorder=3)
    if HAS_SCIPY and gaussian_kde is not None and len(sa_list) > 3:
        kde = gaussian_kde(sa_list, bw_method=0.25)
        xs = np.linspace(0, 1, 200)
        ax3.plot(xs, kde(xs), color=ICEBLUE_EDGE, lw=1.6)
    ax3.axvline(float(np.mean(sa_list)), color=ACCENT, lw=1.2, ls='--')
    apply_nature_axes(ax3)
    ax3.set_xlabel('SA', fontsize=FONT['label'])
    ax3.set_ylabel('Density', fontsize=FONT['label'])
    ax3.set_title(_viz_title(title_prefix, 'C · SA'), fontsize=FONT['title'], pad=8)
    plt.tight_layout()
    _safe_savefig(fig3, output_dir / 'C_druglikeness_sa_dist.png', dpi=180)
    saved['C_druglikeness_sa_dist'] = str(output_dir / 'C_druglikeness_sa_dist.png')

    # ---- Fig 4: Lipinski compliance ----
    fig4, ax4 = plt.subplots(figsize=(5.0, 3.8))
    if len(lip_list) > 0:
        ax4.hist(lip_list * 5, bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                 color=ICEBLUE_EDGE, edgecolor='white', alpha=0.9, zorder=3)
        ax4.set_xlabel('Lipinski rules passed', fontsize=FONT['label'])
        ax4.set_ylabel('Count', fontsize=FONT['label'])
        ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax4.axvline(4, color=ACCENT, lw=1.2, linestyle='--')
        apply_nature_axes(ax4)
        ax4.set_title(_viz_title(title_prefix, 'C · Lipinski'), fontsize=FONT['title'], pad=8)
    else:
        ax4.text(0.5, 0.5, 'No Lipinski data', ha='center', va='center',
                 transform=ax4.transAxes, fontsize=10)
    plt.tight_layout()
    _safe_savefig(fig4, output_dir / 'C_druglikeness_lipinski.png', dpi=180)
    saved['C_druglikeness_lipinski'] = str(output_dir / 'C_druglikeness_lipinski.png')

    # ---- Fig 5: 5-axis druglikeness radar (pentagon); overall only in caption ----
    fig5 = plt.figure(figsize=(5.6, 5.4))
    ax5 = fig5.add_subplot(111, polar=True)
    categories = ['QED', 'SA', 'Lipinski', 'Veber', 'PAINS-free']
    qed_m   = float(idea_d_result.get('qed_mean', 0) or 0)
    sa_m    = float(idea_d_result.get('sa_mean',  0) or 0)
    lip_m   = float(idea_d_result.get('lipinski_mean', 0) or 0)
    veber_r = float(idea_d_result.get('veber_ratio', 0) or 0)
    pains_r = float(idea_d_result.get('pains_ratio', 0) or 0)
    overall = float(idea_d_result.get('score', 0) or 0)
    values = [
        float(min(1.0, qed_m)),
        float(min(1.0, sa_m)),
        float(np.clip(lip_m, 0.0, 1.0)),
        float(np.clip(veber_r, 0.0, 1.0)),
        float(np.clip(1.0 - pains_r, 0.0, 1.0)),
    ]
    N = 5
    assert len(categories) == N and len(values) == N
    # equal 72° spacing → closed pentagon
    angles = (np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)).tolist()
    angles_c = angles + angles[:1]
    values_c = values + values[:1]

    ax5.set_facecolor('#FAFBFC')
    # clear default polar spokes so only our 5 axes remain
    ax5.set_xticks([])
    ax5.plot(
        angles_c, values_c, 'o-', lw=1.9, color=ICEBLUE_EDGE,
        markersize=6, markerfacecolor=ICEBLUE_FILL, markeredgecolor=ICEBLUE_EDGE,
        zorder=3,
    )
    ax5.fill(angles_c, values_c, alpha=0.28, color=ICEBLUE_FILL, zorder=2)
    ax5.set_thetagrids(np.degrees(angles), categories, fontsize=FONT['tick'])
    ax5.set_ylim(0, 1.0)
    ax5.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax5.set_yticklabels([])
    ax5.spines['polar'].set_color('#B0B0B0')
    ax5.grid(True, color='#E8E8E8', linewidth=0.7)
    ax5.set_title(
        _viz_title(title_prefix, 'C · Druglikeness'),
        fontsize=FONT['title'], pad=16,
    )
    # keep caption inside figure so bbox_inches='tight' does not crop it
    fig5.text(
        0.5, 0.035,
        f'C overall = {overall:.3f}',
        ha='center', va='bottom',
        fontsize=FONT['annot'] + 1, color='#555555',
    )
    fig5.subplots_adjust(top=0.86, bottom=0.12, left=0.10, right=0.90)
    path = output_dir / 'C_druglikeness_radar.png'
    _safe_savefig(fig5, path, dpi=180)
    saved['C_druglikeness_radar'] = str(path)
    return saved


def visualize_fingerprint_diversity(molecules_with_pos, idea_f_result=None,
                                     output_dir=None, title_prefix='',
                                     color_by_qed=None, max_mols=150):
    """
    Idea F - Chemical space diversity (Morgan FP) (4 separate figures)

    Subplots: PCA / t-SNE / Tanimoto heatmap / uniqueness
    """
    if not HAS_MATPLOTLIB or not HAS_SKLEARN:
        return {}
    fps_matrix = None
    if idea_f_result and idea_f_result.get('fps_matrix') is not None:
        fps_matrix = idea_f_result['fps_matrix']
    if fps_matrix is None:
        fps_matrix, _ = _compute_morgan_fingerprints(molecules_with_pos)
    if fps_matrix is None or len(fps_matrix) < 4:
        return {}
    output_dir = Path(output_dir) if output_dir else None
    if not output_dir:
        return {}
    saved = {}

    n_fps = len(fps_matrix)
    if n_fps > max_mols:
        idx_sub = np.random.choice(n_fps, max_mols, replace=False)
        fps_sub = fps_matrix[idx_sub]
    else:
        idx_sub = np.arange(n_fps)
        fps_sub = fps_matrix

    qed_colors = None
    if color_by_qed is not None and len(color_by_qed) == n_fps:
        qed_colors = np.array(color_by_qed)[idx_sub]
    c_vals = qed_colors if qed_colors is not None else np.arange(len(fps_sub))

    # ---- Fig 1: PCA ----
    pca_fp = PCA(n_components=2, random_state=42)
    fps_2d_pca = pca_fp.fit_transform(fps_sub)
    var_ratio = pca_fp.explained_variance_ratio_
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    sc1 = ax1.scatter(fps_2d_pca[:, 0], fps_2d_pca[:, 1],
                      c=c_vals, cmap='plasma', s=30, alpha=0.7,
                      edgecolors='none')
    plt.colorbar(sc1, ax=ax1, label='QED' if qed_colors is not None else 'Molecule index', shrink=0.85)
    ax1.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)', fontsize=9)
    ax1.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)', fontsize=9)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    _safe_savefig(fig1, output_dir / 'F_fingerprint_pca.png')
    saved['F_fingerprint_pca'] = str(output_dir / 'F_fingerprint_pca.png')

    # ---- Fig 2: t-SNE ----
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    if TSNE is not None and len(fps_sub) >= 4:
        perp = min(30, len(fps_sub) - 1)
        try:
            tsne_fp = TSNE(n_components=2, perplexity=perp, random_state=42,
                           n_jobs=1, max_iter=500)
        except TypeError:
            tsne_fp = TSNE(n_components=2, perplexity=perp, random_state=42,
                           n_jobs=1, n_iter=500)
        fps_2d_tsne = tsne_fp.fit_transform(fps_sub)
        sc2 = ax2.scatter(fps_2d_tsne[:, 0], fps_2d_tsne[:, 1],
                          c=c_vals, cmap='plasma', s=30, alpha=0.7, edgecolors='none')
        plt.colorbar(sc2, ax=ax2, label='QED' if qed_colors is not None else 'Molecule index', shrink=0.85)
        ax2.set_xlabel('t-SNE 1', fontsize=9)
        ax2.set_ylabel('t-SNE 2', fontsize=9)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 't-SNE unavailable', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=10)
    plt.tight_layout()
    _safe_savefig(fig2, output_dir / 'F_fingerprint_tsne.png')
    saved['F_fingerprint_tsne'] = str(output_dir / 'F_fingerprint_tsne.png')

    # ---- Fig 3: Tanimoto heatmap ----
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    n_heat = min(len(fps_sub), 60)
    fps_heat = fps_sub[:n_heat]
    norms = np.linalg.norm(fps_heat, axis=1, keepdims=True) + 1e-8
    fps_normed = fps_heat / norms
    sim_mat = fps_normed @ fps_normed.T
    np.fill_diagonal(sim_mat, 1.0)
    im = ax3.imshow(sim_mat, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax3, label='Cosine similarity', shrink=0.85)
    ax3.set_xlabel('Molecule index', fontsize=9)
    ax3.set_ylabel('Molecule index', fontsize=9)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    _safe_savefig(fig3, output_dir / 'F_fingerprint_heatmap.png')
    saved['F_fingerprint_heatmap'] = str(output_dir / 'F_fingerprint_heatmap.png')

    # ---- Fig 4: Uniqueness ----
    fig4, ax4 = plt.subplots(figsize=(5, 5))
    if idea_f_result and idea_f_result.get('success'):
        stats_labels = ['Unique', 'Duplicate']
        n_unique = idea_f_result.get('n_unique', 0)
        n_total  = idea_f_result.get('n_total', n_unique)
        unique_ratio = idea_f_result.get('unique_ratio', n_unique / max(n_total, 1))
        full_at = float(idea_f_result.get('unique_full_at', 0.95))
        stats_vals = [n_unique, max(0, n_total - n_unique)]
        colors_pie = ['#4CAF50', '#FF5722']
        wedges, texts, autotexts = ax4.pie(
            stats_vals, labels=stats_labels, colors=colors_pie,
            autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=1.5),
        )
        for at in autotexts:
            at.set_fontsize(10)
        ax4.set_title(
            f'unique_ratio={unique_ratio*100:.1f}% ({n_unique}/{n_total})\n'
            f'full score @ ≥{full_at*100:.0f}% unique',
            fontsize=9,
        )
    else:
        ax4.text(0.5, 0.5, 'No uniqueness data', ha='center', va='center',
                 transform=ax4.transAxes, fontsize=10)
    plt.tight_layout()
    _safe_savefig(fig4, output_dir / 'F_fingerprint_uniqueness.png')
    saved['F_fingerprint_uniqueness'] = str(output_dir / 'F_fingerprint_uniqueness.png')
    return saved


def visualize_size_distribution(idea_g_result, output_dir=None, title_prefix=''):
    """
    Auxiliary — molecular size distribution (not scored as G; anchors are G)

    Subplots: MW hist / atom count hist / MW vs atom count scatter
    """
    if not HAS_MATPLOTLIB:
        return {}
    if not idea_g_result.get('success'):
        return {}
    mw_list     = np.array(idea_g_result.get('mw_list', []))
    natoms_list = np.array(idea_g_result.get('n_atoms_list', []))
    if len(mw_list) < 2:
        return {}
    output_dir = Path(output_dir) if output_dir else None
    if not output_dir:
        return {}
    saved = {}
    mw_mean = float(np.mean(mw_list))
    mw_cv   = idea_g_result.get('mw_cv', 0) or 0

    # ---- Fig 1: MW distribution ----
    fig1, ax = plt.subplots(figsize=(6, 4))
    ax.hist(mw_list, bins=25, color='cornflowerblue', edgecolor='white',
            alpha=0.8, density=True)
    if HAS_SCIPY and gaussian_kde is not None and len(mw_list) > 3:
        kde = gaussian_kde(mw_list, bw_method='scott')
        xm = np.linspace(mw_list.min() * 0.9, mw_list.max() * 1.1, 300)
        ax.plot(xm, kde(xm), 'r-', lw=2, label='KDE')
    ax.axvline(mw_mean, color='orange', lw=1.5, linestyle='--',
               label=f'Mean={mw_mean:.0f}')
    ax.axvline(500, color='green', lw=1.2, linestyle=':', label='Lipinski MW<=500')
    ax.set_xlabel('Molecular weight (Da)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title(_viz_title(title_prefix, 'Aux · MW'), fontsize=10, pad=6)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _safe_savefig(fig1, output_dir / 'aux_size_mw.png')
    saved['aux_size_mw'] = str(output_dir / 'aux_size_mw.png')

    # ---- Fig 2: Atom count distribution ----
    fig2, ax = plt.subplots(figsize=(6, 4))
    if len(natoms_list) > 0:
        bins_a = range(int(natoms_list.min()), int(natoms_list.max()) + 2)
        ax.hist(natoms_list, bins=bins_a, color='mediumorchid', edgecolor='white',
                alpha=0.8)
        ax.axvline(float(np.mean(natoms_list)), color='orange', lw=1.5, linestyle='--',
                   label=f'Mean={np.mean(natoms_list):.1f}')
        ax.set_xlabel('Atom count (incl. H)', fontsize=9)
        ax.set_ylabel('Molecule count', fontsize=9)
        ax.legend(fontsize=7)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No atom count data', ha='center', va='center',
                transform=ax.transAxes, fontsize=10)
    plt.tight_layout()
    _safe_savefig(fig2, output_dir / 'aux_size_atoms.png')
    saved['aux_size_atoms'] = str(output_dir / 'aux_size_atoms.png')

    # ---- Fig 3: MW vs atom count ----
    fig3, ax = plt.subplots(figsize=(6, 4))
    mw_arr = np.array(mw_list)
    if len(natoms_list) >= len(mw_arr):
        natoms_aligned = natoms_list[:len(mw_arr)]
    else:
        natoms_aligned = natoms_list
        mw_arr = mw_arr[:len(natoms_list)]
    sc = ax.scatter(natoms_aligned, mw_arr,
                    c=mw_arr, cmap='coolwarm', s=25, alpha=0.6,
                    edgecolors='none')
    plt.colorbar(sc, ax=ax, label='MW (Da)', shrink=0.85)
    if len(natoms_aligned) > 2:
        z = np.polyfit(natoms_aligned, mw_arr, 1)
        p = np.poly1d(z)
        x_line = np.linspace(natoms_aligned.min(), natoms_aligned.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', lw=1.5, label=f'Linear trend (slope={z[0]:.1f})')
        ax.legend(fontsize=7)
    ax.set_xlabel('Atom count', fontsize=9)
    ax.set_ylabel('Molecular weight (Da)', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _safe_savefig(fig3, output_dir / 'aux_size_mw_vs_atoms.png')
    saved['aux_size_mw_vs_atoms'] = str(output_dir / 'aux_size_mw_vs_atoms.png')
    return saved


def visualize_pocket_volume(idea_h_result, output_dir=None, title_prefix=''):
    """
    Idea H — pocket cavity volume (ligand-centered, protein-excluded MC when available).
    Full-score band 350–750 Å³; Nature iceblue / accent styling.
    """
    if not HAS_MATPLOTLIB:
        return {}
    if not idea_h_result.get('success'):
        return {}
    vol = idea_h_result.get('volume_ang3')
    if vol is None:
        return {}
    output_dir = Path(output_dir) if output_dir else None
    if not output_dir:
        return {}
    try:
        from utils.pocket_vis_nature import (
            ACCENT, BAND_HIGH, BAND_MED, FONT, ICEBLUE_EDGE, ICEBLUE_FILL,
            apply_nature_axes,
        )
    except ImportError:
        ACCENT, ICEBLUE_FILL, ICEBLUE_EDGE = '#F56E1A', '#C5DDF0', '#5A7FA0'
        BAND_HIGH, BAND_MED = '#A8C5A0', '#F0D9A8'
        FONT = {'title': 11, 'label': 9, 'tick': 8, 'annot': 7}

        def apply_nature_axes(ax, grid_y=True, grid_x=False):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    score = float(idea_h_result.get('score') or 0)
    vol = float(vol)
    rmin = float(idea_h_result.get('optimal_vol_min', IDEA_H_VIZ_REF_OPT_MIN))
    rmax = float(idea_h_result.get('optimal_vol_max', IDEA_H_VIZ_REF_OPT_MAX))
    rzb = float(idea_h_result.get('vol_score_zero_below', IDEA_H_VIZ_REF_ZERO_BELOW))
    rza = float(idea_h_result.get('vol_score_zero_above', IDEA_H_VIZ_REF_ZERO_ABOVE))
    radius = float(idea_h_result.get('centroid_radius', 10.0) or 10.0)
    method = str(idea_h_result.get('volume_method') or '')
    if 'cavity' in method or 'protein' in method:
        method_tag = 'cav·prot+lig'
    elif 'fallback' in method:
        method_tag = 'MC·lig-only(fallback)'
    else:
        method_tag = 'MC·lig-only'

    x_min = 0.0
    x_cap = float(IDEA_H_VIZ_X_MAX_DEFAULT)
    x_max = x_cap if vol <= x_cap else float(vol) * 1.06

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    # zero tails
    if rzb > 0:
        ax.axvspan(x_min, min(rzb, x_max), color='#E8C4C0', alpha=0.35, zorder=1)
    if rza < x_max:
        ax.axvspan(max(rza, x_min), x_max, color='#E8C4C0', alpha=0.35, zorder=1)
    # sub-optimal
    if rzb < rmin:
        ax.axvspan(max(x_min, rzb), min(rmin, x_max), color=BAND_MED, alpha=0.45, zorder=1)
    if rmax < rza:
        ax.axvspan(max(x_min, rmax), min(rza, x_max), color=BAND_MED, alpha=0.45, zorder=1)
    # full-score band
    ax.axvspan(rmin, rmax, color=BAND_HIGH, alpha=0.55, zorder=2)
    ax.axvline(rmin, color='#6B8F71', ls=':', lw=1.0, alpha=0.8, zorder=3)
    ax.axvline(rmax, color='#6B8F71', ls=':', lw=1.0, alpha=0.8, zorder=3)

    ax.barh(0, vol, left=0, height=0.34, color=ICEBLUE_FILL,
            edgecolor=ICEBLUE_EDGE, linewidth=1.0, zorder=4)
    ax.axvline(vol, color=ACCENT, ls='-', lw=1.6, alpha=0.95, zorder=5)
    ax.plot([vol], [0], marker='o', markersize=7, color=ACCENT,
            markeredgecolor='white', markeredgewidth=0.8, zorder=6)

    apply_nature_axes(ax, grid_y=False, grid_x=True)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([])
    ax.set_xlabel('Volume (Å³)', fontsize=FONT['label'])
    ttl = _viz_title(title_prefix, 'H · Pocket volume')
    ax.set_title(ttl, fontsize=FONT['title'], pad=8)
    ax.text(
        0.02, 0.96,
        f'{method_tag} · r={radius:g} Å  ·  band {rmin:.0f}–{rmax:.0f}  ·  H={score:.3f}',
        transform=ax.transAxes, ha='left', va='top',
        fontsize=FONT['annot'], color='#666666',
    )
    ax.text(
        vol, 0.28, f'V={vol:.0f}',
        ha='center', va='bottom', fontsize=FONT['label'], color=ACCENT,
    )
    plt.tight_layout()
    path = output_dir / 'H_pocket_volume.png'
    _safe_savefig(fig, path, dpi=180)
    return {'H_pocket_volume': str(path)}


def resolve_ligand_sdf_from_pt(pt_path, protein_root=None):
    """Resolve CrossDocked-style native ligand SDF path from .pt ligand_filename."""
    pt_path = Path(pt_path)
    if not pt_path.is_file():
        return None
    try:
        raw = load_pt_file(pt_path)
    except Exception:
        return None
    data = raw.get('data') if isinstance(raw, dict) else None
    ligand_fn = getattr(data, 'ligand_filename', None) if data is not None else None
    if isinstance(data, dict):
        ligand_fn = ligand_fn or data.get('ligand_filename')
    if not ligand_fn:
        return None
    lf = Path(str(ligand_fn))
    if lf.is_file():
        return str(lf.resolve())

    search_roots = []
    if protein_root:
        search_roots.append(Path(protein_root))
    search_roots.extend([
        REPO_ROOT / 'data' / 'crossdocked_pocket10_test_only',
        REPO_ROOT / 'data' / 'crossdocked_v1.1_rmsd1.0_pocket10',
        REPO_ROOT / 'data',
    ])
    rel_bits = []
    if len(lf.parts) >= 2:
        rel_bits.append(Path(*lf.parts[-2:]))
    rel_bits.append(Path(lf.name))
    for root in search_roots:
        if not root or not Path(root).exists():
            continue
        for rel in rel_bits:
            cand = Path(root) / rel
            if cand.is_file():
                return str(cand.resolve())
        hits = list(Path(root).rglob(lf.name))
        if hits:
            return str(hits[0].resolve())
    return None


def _mol_atom_type_counts(mol):
    """Count heavy atoms by element label (excl. H). Used for chemspace series stats."""
    if mol is None:
        return {}
    try:
        zs = [int(a.GetAtomicNum()) for a in mol.GetAtoms() if int(a.GetAtomicNum()) > 1]
    except Exception:
        return {}
    z_to_lab = {
        6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P',
        16: 'S', 17: 'Cl', 35: 'Br', 53: 'I',
    }
    out = {}
    for z in zs:
        lab = z_to_lab.get(z, 'Other')
        out[lab] = out.get(lab, 0) + 1
    return out


def visualize_chemspace_umap(
    molecules_with_pos,
    idea_a_result=None,
    output_dir=None,
    title_prefix='',
    native_molecules=None,  # kept for API compat; intentionally unused
):
    """
    Fig.2a contour grammar (J Cheminform 10.1186/s13321-026-01230-5):

    Morgan2048 → UMAP; **one outermost density ring per atom type** (thin line,
    no nested levels). Density is weighted by **atom counts across all molecules**
    (legend n = total atoms of that type, not molecule presence). Sparse types
    drawn as open circles. No Native overlay.
    """
    if not HAS_MATPLOTLIB:
        return {}
    output_dir = Path(output_dir) if output_dir else None
    if not output_dir:
        return {}
    try:
        from utils.pocket_vis_nature import FONT, NEUTRAL, embed_2d, morgan_fps_from_molecules
    except ImportError:
        return {}

    from matplotlib.lines import Line2D

    # Element colors: chemistry / PyMOL-CPK common (C green as requested)
    # Ref: CPK + PyMOL defaults (O red, N blue, S yellow, P orange; C→green)
    ATOM_COLORS = {
        'C': '#33AA33',   # green (common chem ball-stick / user request)
        'N': '#0000FF',   # blue (PyMOL / CPK)
        'O': '#FF0000',   # red (PyMOL / CPK)
        'F': '#7CFC00',   # light green (halogen)
        'P': '#FF8C00',   # orange (PyMOL / CPK)
        'S': '#E6C200',   # yellow (PyMOL / CPK)
        'Cl': '#00AA00',  # green (CPK chlorine)
        'Br': '#A52A2A',  # dark red-brown
        'I': '#9400D3',   # purple
        'Other': '#808080',
    }
    ATOM_ORDER = ('C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'Other')
    MIN_KDE = 8
    LINE_W = 0.50  # thin single outer ring only

    gen_fps, gen_keep = morgan_fps_from_molecules(molecules_with_pos, n_bits=2048, radius=2)
    if gen_fps is None or len(gen_keep) < 8:
        return {}

    xy_gen, method = embed_2d(gen_fps)

    # per-molecule atom counts → series weighted by total atoms (not molecule presence)
    # groups[lab] = list of (umap_row_index, atom_count_in_that_mol)
    groups = {lab: [] for lab in ATOM_ORDER}
    for i, idx in enumerate(gen_keep):
        mol = molecules_with_pos[idx][0] if idx < len(molecules_with_pos) else None
        counts = _mol_atom_type_counts(mol)
        for lab, cnt in counts.items():
            if cnt <= 0:
                continue
            if lab in groups:
                groups[lab].append((i, int(cnt)))
            else:
                groups.setdefault('Other', []).append((i, int(cnt)))

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    legend_handles = []

    def _draw_small_rings(xy, color, zorder=4):
        """Even n=1: open circle ('小圈子'), not a filled dot."""
        ax.scatter(
            xy[:, 0], xy[:, 1],
            s=55, facecolors='none', edgecolors=color,
            linewidths=LINE_W, zorder=zorder,
        )
        return 'ring'

    def _draw_outer_ring(xy, weights, color, zorder=2):
        """Single lowest-density contour; KDE weighted by atom counts."""
        n_atoms = int(np.sum(weights)) if weights is not None else len(xy)
        if n_atoms < MIN_KDE or len(xy) < 1:
            return _draw_small_rings(xy, color, zorder=zorder + 2)
        w = np.asarray(weights, dtype=np.float64)
        # Expand to atom-count points + tiny jitter so mass (not just support) shapes the ring.
        # Shared molecule support (C/O≈all mols) otherwise yields near-identical outer edges.
        reps = np.maximum(1, np.round(w).astype(int))
        if int(reps.sum()) > 8000:
            reps = np.maximum(1, np.round(reps * (8000.0 / reps.sum())).astype(int))
        xy_exp = np.repeat(xy, reps, axis=0)
        rng = np.random.RandomState(42 + int(zorder) + int(n_atoms) % 97)
        span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-6)
        jitter = 0.012 * span
        xy_exp = xy_exp + rng.normal(0.0, jitter, size=xy_exp.shape)
        if HAS_SCIPY and gaussian_kde is not None:
            try:
                # lower bw → sharper, less soft outline
                kde = gaussian_kde(xy_exp.T, bw_method=0.16)
                xmin, xmax = float(xy_exp[:, 0].min()), float(xy_exp[:, 0].max())
                ymin, ymax = float(xy_exp[:, 1].min()), float(xy_exp[:, 1].max())
                pad_x = 0.28 * (xmax - xmin + 1e-6)
                pad_y = 0.28 * (ymax - ymin + 1e-6)
                xx, yy = np.meshgrid(
                    np.linspace(xmin - pad_x, xmax + pad_x, 180),
                    np.linspace(ymin - pad_y, ymax + pad_y, 180),
                )
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                # slightly higher than pure outer fringe → weight differences show more
                level = float(zz.max() * 0.22)
                ax.contour(
                    xx, yy, zz, levels=[level], colors=[color],
                    linewidths=LINE_W, zorder=zorder,
                )
                return 'contour'
            except Exception:
                pass
        try:
            import seaborn as sns
            sns.kdeplot(
                x=xy_exp[:, 0], y=xy_exp[:, 1], ax=ax,
                levels=1, thresh=0.22, fill=False,
                color=color, linewidths=LINE_W, bw_adjust=0.35,
                zorder=zorder,
            )
            return 'contour'
        except Exception:
            return _draw_small_rings(xy, color, zorder=zorder + 2)

    for zi, lab in enumerate(ATOM_ORDER):
        entries = groups.get(lab) or []
        if not entries:
            continue
        color = ATOM_COLORS.get(lab, '#A0A0A0')
        row_idx = np.asarray([e[0] for e in entries], dtype=int)
        weights = np.asarray([e[1] for e in entries], dtype=np.float64)
        xy = xy_gen[row_idx]
        n_atoms = int(weights.sum())
        style = _draw_outer_ring(xy, weights, color, zorder=2 + zi)
        if style == 'contour':
            legend_handles.append(
                Line2D([0], [0], color=color, lw=LINE_W + 0.3, label=f'{lab} (n={n_atoms})'),
            )
        else:
            legend_handles.append(
                Line2D([0], [0], marker='o', color=color, markerfacecolor='none',
                       markeredgecolor=color, markersize=6, linestyle='None',
                       label=f'{lab} (n={n_atoms})'),
            )

    # Frame + grid like Fig.2a
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color('#444444')
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(axis='both', labelsize=FONT['tick'], colors=NEUTRAL, length=3.5, width=0.7)
    ax.grid(True, color='#D0D0D0', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    pad_x = 0.18 * (xy_gen[:, 0].max() - xy_gen[:, 0].min() + 1e-6)
    pad_y = 0.18 * (xy_gen[:, 1].max() - xy_gen[:, 1].min() + 1e-6)
    ax.set_xlim(xy_gen[:, 0].min() - pad_x, xy_gen[:, 0].max() + pad_x)
    ax.set_ylim(xy_gen[:, 1].min() - pad_y, xy_gen[:, 1].max() + pad_y)

    ax.set_xlabel(f'{method} 1', fontsize=FONT['label'], color=NEUTRAL)
    ax.set_ylabel(f'{method} 2', fontsize=FONT['label'], color=NEUTRAL)
    ax.set_title(
        _viz_title(title_prefix, 'Chemical space'),
        fontsize=FONT['title'], color=NEUTRAL, pad=10,
    )

    if legend_handles:
        ax.legend(
            handles=legend_handles, loc='lower left', fontsize=FONT['annot'],
            frameon=True, fancybox=False, edgecolor='#CCCCCC', framealpha=0.95,
        )

    _ = (idea_a_result, native_molecules)

    plt.tight_layout()
    path = output_dir / 'chemspace_umap.png'
    _safe_savefig(fig, path, dpi=180, save_notext=True)
    notext = output_dir / 'chemspace_umap_notext.png'
    out = {'chemspace_umap': str(path)}
    if notext.exists():
        out['chemspace_umap_notext'] = str(notext)
    return out


def visualize_pocket_chemistry(idea_f_result, output_dir=None, title_prefix=''):
    """
    Idea F - ligand-neighborhood bio chemistry (hydrophobic / polar / enclosure), 10 Å.
    """
    if not HAS_MATPLOTLIB:
        return {}
    if not idea_f_result or not idea_f_result.get('success'):
        return {}
    output_dir = Path(output_dir) if output_dir else None
    if not output_dir:
        return {}
    try:
        from utils.pocket_vis_nature import (
            ACCENT,
            FONT,
            ICEBLUE_EDGE,
            ICEBLUE_FILL,
            apply_nature_axes,
        )
    except ImportError:
        ACCENT, ICEBLUE_FILL, ICEBLUE_EDGE = '#F56E1A', '#C5DDF0', '#5A7FA0'
        FONT = {'title': 11, 'label': 9, 'tick': 8, 'annot': 7}

        def apply_nature_axes(ax, grid_y=True, grid_x=False):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    labels = ['Hydrophobic', 'Polar', 'Enclosure']
    # Show raw bio fractions as primary (readable when mapped saturates at 1)
    raws = [
        idea_f_result.get('hydro_fraction'),
        idea_f_result.get('polar_fraction'),
        idea_f_result.get('enclosure_raw'),
    ]
    mapped = [
        idea_f_result.get('druggability_mapped'),
        idea_f_result.get('fpocket_score_mapped'),
        idea_f_result.get('hydrophobicity_mapped'),
    ]
    vals = [float(v) if v is not None and np.isfinite(v) else 0.0 for v in raws]
    mapped_f = [float(v) if v is not None and np.isfinite(v) else 0.0 for v in mapped]
    overall = float(idea_f_result.get('score') or 0.0)
    radius = idea_f_result.get('centroid_radius', 10.0)

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    colors = [ICEBLUE_EDGE, '#7BA3C4', ACCENT]
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, edgecolor='white', width=0.62, zorder=3)
    for bar, raw, mp in zip(bars, vals, mapped_f):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            raw + 0.045,
            f'{raw:.2f}  (→{mp:.2f})',
            ha='center',
            va='bottom',
            fontsize=FONT['annot'],
            color='#4A4A4A',
        )
    apply_nature_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT['label'])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel('Raw fraction', fontsize=FONT['label'])
    ax.set_title(_viz_title(title_prefix, 'F · Pocket chemistry'), fontsize=FONT['title'], pad=8)
    ax.text(
        0.98,
        0.98,
        f'r={float(radius):g} Å  ·  F={overall:.3f}',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=FONT['annot'],
        color='#666666',
    )
    plt.tight_layout()
    _safe_savefig(fig, output_dir / 'F_pocket_chemistry.png', dpi=180)
    return {'F_pocket_chemistry': str(output_dir / 'F_pocket_chemistry.png')}


def visualize_pocket_chemistry_legacy_fpocket_bars(idea_f_result, output_dir=None, title_prefix=''):
    """Deprecated stub — redirects to ligand-neighborhood chemistry plot."""
    return visualize_pocket_chemistry(idea_f_result, output_dir=output_dir, title_prefix=title_prefix)


def visualize_anchor_richness(idea_g_result, output_dir=None, title_prefix=''):
    """Idea G — P4 interaction anchors (Nature style)."""
    if not HAS_MATPLOTLIB:
        return {}
    if not idea_g_result or not idea_g_result.get('success'):
        return {}
    output_dir = Path(output_dir) if output_dir else None
    if not output_dir:
        return {}
    try:
        from utils.pocket_vis_nature import (
            ACCENT, FONT, ICEBLUE_EDGE, ICEBLUE_FILL, apply_nature_axes,
        )
    except ImportError:
        ACCENT, ICEBLUE_FILL, ICEBLUE_EDGE = '#F56E1A', '#C5DDF0', '#5A7FA0'
        FONT = {'title': 11, 'label': 9, 'tick': 8, 'annot': 7}

        def apply_nature_axes(ax, grid_y=True, grid_x=False):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    saved = {}
    counts = idea_g_result.get('anchor_counts') or {}
    cat_order = ['donor', 'acceptor', 'aromatic', 'ionic', 'hydrophobic']
    cat_vals = [int(counts.get(c, 0) or 0) for c in cat_order]
    cat_colors = [ICEBLUE_EDGE, '#7BA3C4', ACCENT, '#4A4A4A', ICEBLUE_FILL]

    fig1, ax1 = plt.subplots(figsize=(5.8, 3.8))
    x = np.arange(len(cat_order))
    bars = ax1.bar(x, cat_vals, color=cat_colors, edgecolor='white', width=0.62, zorder=3)
    for bar, val in zip(bars, cat_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, val + 0.4, str(val),
            ha='center', va='bottom', fontsize=FONT['annot'], color='#4A4A4A',
        )
    apply_nature_axes(ax1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cat_order, fontsize=FONT['tick'])
    ax1.set_ylabel('Anchor count', fontsize=FONT['label'])
    n_eff = idea_g_result.get('n_eff')
    n_cap = idea_g_result.get('n_cap')
    ttl = _viz_title(title_prefix, 'G · Anchors')
    ax1.set_title(ttl, fontsize=FONT['title'], pad=8)
    ax1.text(
        0.98, 0.98, f'Neff={n_eff}/{n_cap:.0f}' if n_cap else f'Neff={n_eff}',
        transform=ax1.transAxes, ha='right', va='top',
        fontsize=FONT['annot'], color='#666666',
    )
    plt.tight_layout()
    _safe_savefig(fig1, output_dir / 'G_anchor_categories.png', dpi=180)
    saved['G_anchor_categories'] = str(output_dir / 'G_anchor_categories.png')

    sub_labels = ['Richness', 'Coverage', 'Balance', 'Overall']
    sub_vals = [
        float(idea_g_result.get('s_richness') or 0.0),
        float(idea_g_result.get('s_coverage') or 0.0),
        float(idea_g_result.get('s_balance') or 0.0),
        float(idea_g_result.get('score') or 0.0),
    ]
    fig2, ax2 = plt.subplots(figsize=(5.4, 3.6))
    y = np.arange(len(sub_labels))
    colors2 = [ICEBLUE_FILL, '#7BA3C4', ICEBLUE_EDGE, ACCENT]
    bars2 = ax2.barh(y, sub_vals, color=colors2, edgecolor='white', height=0.55, zorder=3)
    for bar, val in zip(bars2, sub_vals):
        ax2.text(
            val + 0.015, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=FONT['annot'], color='#4A4A4A',
        )
    apply_nature_axes(ax2, grid_y=False, grid_x=True)
    ax2.set_yticks(y)
    ax2.set_yticklabels(sub_labels, fontsize=FONT['label'])
    ax2.set_xlim(0, 1.18)
    ax2.set_xlabel('Score', fontsize=FONT['label'])
    ax2.set_title(
        _viz_title(title_prefix, 'G · Anchor scores'),
        fontsize=FONT['title'], pad=8,
    )
    plt.tight_layout()
    _safe_savefig(fig2, output_dir / 'G_anchor_scores.png', dpi=180)
    saved['G_anchor_scores'] = str(output_dir / 'G_anchor_scores.png')

    xyz = idea_g_result.get('site_xyz') or []
    cats = idea_g_result.get('site_categories') or []
    if len(xyz) >= 2 and len(cats) == len(xyz):
        arr = np.asarray(xyz, dtype=np.float64)
        fig3, ax3 = plt.subplots(figsize=(5.4, 5.0))
        # High-contrast categorical colors (types must be separable at a glance)
        cat_style = {
            'donor':       {'c': '#2E6EB5', 'marker': 'o', 'z': 3},   # blue
            'acceptor':    {'c': '#D94A3D', 'marker': 's', 'z': 3},   # red
            'aromatic':    {'c': '#E6A817', 'marker': 'D', 'z': 5},   # gold
            'ionic':       {'c': '#7B2D8E', 'marker': '^', 'z': 4},   # purple
            'hydrophobic': {'c': '#2A9D6E', 'marker': 'o', 'z': 2},   # green
        }
        # draw rarer / more important types last so they sit on top
        draw_order = ['hydrophobic', 'donor', 'acceptor', 'ionic', 'aromatic']
        for cat in draw_order:
            mask = [c == cat for c in cats]
            n_cat = sum(1 for m in mask if m)
            if n_cat < 1:
                continue
            pts = arr[np.asarray(mask)]
            st = cat_style[cat]
            ax3.scatter(
                pts[:, 0], pts[:, 1],
                s=36 if cat != 'aromatic' else 48,
                alpha=0.88,
                c=st['c'],
                marker=st['marker'],
                label=f'{cat} ({n_cat})',
                edgecolors='white',
                linewidths=0.35,
                zorder=st['z'],
            )
        # Pocket neighborhood ring only (no centroid star)
        cent = idea_g_result.get('ligand_centroid')
        if cent is not None and len(cent) >= 2:
            rad = float(idea_g_result.get('centroid_radius') or 12.0)
            circ = plt.Circle(
                (cent[0], cent[1]), rad, fill=False, ls='--',
                color='#9A9A9A', lw=1.1, alpha=0.75, zorder=1,
            )
            ax3.add_patch(circ)
        apply_nature_axes(ax3, grid_y=False, grid_x=False)
        ax3.set_aspect('equal', adjustable='datalim')
        ax3.set_xlabel('X (Å)', fontsize=FONT['label'])
        ax3.set_ylabel('Y (Å)', fontsize=FONT['label'])
        ax3.set_title(
            _viz_title(title_prefix, 'G · Sites XY'),
            fontsize=FONT['title'], pad=8,
        )
        ax3.legend(
            fontsize=FONT['annot'], loc='best', frameon=True,
            fancybox=False, edgecolor='#DDDDDD', framealpha=0.92,
            markerscale=1.15,
        )
        plt.tight_layout()
        _safe_savefig(fig3, output_dir / 'G_anchor_xy.png', dpi=180)
        saved['G_anchor_xy'] = str(output_dir / 'G_anchor_xy.png')

    return saved


def visualize_layer_summary(result, output_dir=None, title_prefix=''):
    """Three-layer scores Sp / Sc / Sl + overall (Nature style)."""
    if not HAS_MATPLOTLIB:
        return {}
    output_dir = Path(output_dir) if output_dir else None
    if not output_dir:
        return {}
    try:
        from utils.pocket_vis_nature import ACCENT, FONT, ICEBLUE_EDGE, ICEBLUE_FILL, apply_nature_axes
    except ImportError:
        ACCENT, ICEBLUE_FILL, ICEBLUE_EDGE = '#F56E1A', '#C5DDF0', '#5A7FA0'
        FONT = {'title': 11, 'label': 9, 'tick': 8, 'annot': 7}

        def apply_nature_axes(ax, grid_y=True, grid_x=False):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    labels = ['Sp', 'Sc', 'Sl', 'Overall']
    raw = [
        result.get('s_pocket'),
        result.get('s_compatibility'),
        result.get('s_ligand'),
        result.get('overall_score'),
    ]
    vals_f = [float(v) if v is not None else 0.0 for v in raw]
    present = [v is not None for v in raw]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    colors = [ICEBLUE_EDGE, '#7BA3C4', ICEBLUE_FILL, ACCENT]
    x = np.arange(len(labels))
    bars = ax.bar(x, vals_f, color=colors, edgecolor='white', width=0.62, zorder=3)
    ax.axhline(0.70, color='#6B8F71', ls=':', lw=1.0, alpha=0.8, zorder=2)
    ax.axhline(0.45, color='#C4A35A', ls=':', lw=1.0, alpha=0.8, zorder=2)
    for bar, val, ok in zip(bars, vals_f, present):
        y_txt = min(val + 0.03, 1.08)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_txt,
            f'{val:.3f}' if ok else 'n/a',
            ha='center',
            va='bottom',
            fontsize=FONT['annot'],
            color='#4A4A4A',
        )
    apply_nature_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT['label'])
    ax.set_ylim(0, 1.22)
    ax.set_ylabel('Score', fontsize=FONT['label'])
    ax.set_title(_viz_title(title_prefix, 'Layer summary'), fontsize=FONT['title'], pad=8)
    ax.text(
        0.02, 0.98, 'Sp=pocket  Sc=compat  Sl=ligand',
        transform=ax.transAxes, ha='left', va='top',
        fontsize=FONT['annot'], color='#999999',
    )
    plt.tight_layout()
    _safe_savefig(fig, output_dir / 'summary_layers.png', dpi=180)
    return {'summary_layers': str(output_dir / 'summary_layers.png')}


def visualize_uniqueness_completeness(
    idea_d_result=None,
    idea_e_result=None,
    output_dir=None,
    title_prefix='',
):
    """
    D + E combined — H-style score-band gauge, single iceblue tone.

    Per-row bands match scoring breakpoints:
      D Unique: low <0.40 / mid 0.40–0.70 / high 0.70–0.95 / full ≥0.95
      E Complete: low <0.75 / mid 0.75–0.90 / high 0.90–0.98 / full ≥0.98
    """
    if not HAS_MATPLOTLIB:
        return {}
    output_dir = Path(output_dir) if output_dir else None
    if not output_dir:
        return {}

    d_ok = bool(idea_d_result and idea_d_result.get('success'))
    e_ok = bool(idea_e_result and idea_e_result.get('success'))
    if not d_ok and not e_ok:
        return {}

    try:
        from utils.pocket_vis_nature import FONT, ICEBLUE_EDGE, ICEBLUE_FILL, NEUTRAL, apply_nature_axes
    except ImportError:
        ICEBLUE_FILL, ICEBLUE_EDGE, NEUTRAL = '#C5DDF0', '#5A7FA0', '#4A4A4A'
        FONT = {'title': 11, 'label': 9, 'tick': 8, 'annot': 7}

        def apply_nature_axes(ax, grid_y=True, grid_x=False):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    BAND_LOW = '#EEF4F8'
    BAND_MID = '#D5E6F2'
    BAND_HIGH = '#B7D4E8'
    BAND_FULL = '#8FB8D4'
    BAR_FACE = ICEBLUE_FILL
    BAR_EDGE = ICEBLUE_EDGE
    MARK = '#3F6F8F'
    x_max = 1.05

    rows = []
    if d_ok:
        n_u = int(idea_d_result.get('n_unique') or 0)
        n_t = int(idea_d_result.get('n_total') or n_u)
        ratio = float(idea_d_result.get('unique_ratio') or (n_u / max(n_t, 1)))
        score = float(idea_d_result.get('score') or 0.0)
        full_at = float(idea_d_result.get('unique_full_at') or 0.95)
        rows.append({
            'label': 'D · Unique',
            'ratio': float(np.clip(ratio, 0.0, 1.0)),
            'score': score,
            'annot': f'{n_u}/{n_t}  D={score:.3f}',
            'full_at': full_at,
            'breaks': (0.0, 0.40, 0.70, 0.95, 1.0),
            'band_note': 'D full≥0.95',
        })
    if e_ok:
        n_ok = int(idea_e_result.get('n_success') or 0)
        n_tot = int(idea_e_result.get('n_total') or n_ok)
        rate = idea_e_result.get('reconstruct_rate')
        if rate is None:
            rate = n_ok / max(n_tot, 1)
        rate = float(rate)
        score = float(idea_e_result.get('score') or 0.0)
        rows.append({
            'label': 'E · Complete',
            'ratio': float(np.clip(rate, 0.0, 1.0)),
            'score': score,
            'annot': f'{n_ok}/{n_tot}  E={score:.3f}',
            'full_at': 0.98,
            'breaks': (0.0, 0.75, 0.90, 0.98, 1.0),
            'band_note': 'E full≥0.98',
        })

    n_rows = len(rows)
    fig_h = 2.2 + 1.15 * n_rows
    fig, ax = plt.subplots(figsize=(6.0, fig_h))
    band_colors = (BAND_LOW, BAND_MID, BAND_HIGH, BAND_FULL)

    y_positions = list(range(n_rows - 1, -1, -1))
    for y, row in zip(y_positions, rows):
        b0, b1, b2, b3, b4 = row['breaks']
        for (xa, xb), col in zip(((b0, b1), (b1, b2), (b2, b3), (b3, b4)), band_colors):
            ax.fill_between([xa, xb], y - 0.48, y + 0.48, color=col, alpha=0.95, zorder=0, linewidth=0)
        ax.plot(
            [row['full_at'], row['full_at']], [y - 0.48, y + 0.48],
            color=BAR_EDGE, ls=':', lw=1.0, alpha=0.75, zorder=1,
        )
        r = row['ratio']
        ax.barh(
            y, r, left=0, height=0.42,
            color=BAR_FACE, edgecolor=BAR_EDGE, linewidth=1.0, zorder=3,
        )
        ax.plot([r, r], [y - 0.28, y + 0.28], color=MARK, lw=1.6, zorder=5)
        ax.plot(
            [r], [y], marker='o', markersize=7, color=MARK,
            markeredgecolor='white', markeredgewidth=0.8, zorder=6,
        )
        ax.text(
            0.01, y + 0.30, row['annot'],
            ha='left', va='bottom', fontsize=FONT['annot'], color='#666666',
        )
        ax.text(
            min(max(r, 0.08), 0.92), y, f'{r:.2f}',
            ha='center', va='center', fontsize=FONT['annot'], color=MARK,
            fontweight='bold',
        )

    apply_nature_axes(ax, grid_y=False, grid_x=True)
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(-0.70, n_rows - 0.15)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r['label'] for r in rows], fontsize=FONT['label'])
    ax.set_xlabel('Ratio', fontsize=FONT['label'])
    ax.set_title(
        _viz_title(title_prefix, 'D/E · Uniqueness & Completeness'),
        fontsize=FONT['title'], pad=8,
    )
    ax.text(
        0.02, 0.02,
        'bands  D: <0.40 / 0.40–0.70 / 0.70–0.95 / ≥0.95   ·   E: <0.75 / 0.75–0.90 / 0.90–0.98 / ≥0.98',
        transform=ax.transAxes, fontsize=FONT['annot'], color='#888888',
        va='bottom', ha='left',
    )

    plt.tight_layout()
    path = output_dir / 'DE_uniqueness_completeness.png'
    _safe_savefig(fig, path, dpi=180)
    return {'DE_uniqueness_completeness': str(path)}


def visualize_uniqueness(idea_d_result, output_dir=None, title_prefix='', idea_e_result=None):
    """Backward-compatible wrapper → combined D/E figure."""
    return visualize_uniqueness_completeness(
        idea_d_result=idea_d_result,
        idea_e_result=idea_e_result,
        output_dir=output_dir,
        title_prefix=title_prefix,
    )


def visualize_completeness(idea_e_result, output_dir=None, title_prefix='', idea_d_result=None):
    """Backward-compatible wrapper → combined D/E figure (no-op if D already drawn)."""
    # Prefer single combined call from generate_all_visualizations.
    if idea_d_result is not None:
        return visualize_uniqueness_completeness(
            idea_d_result=idea_d_result,
            idea_e_result=idea_e_result,
            output_dir=output_dir,
            title_prefix=title_prefix,
        )
    return visualize_uniqueness_completeness(
        idea_d_result=None,
        idea_e_result=idea_e_result,
        output_dir=output_dir,
        title_prefix=title_prefix,
    )


def visualize_radar_summary(result, output_dir=None, title_prefix=''):
    """Summary radar — 8-axis A–H (summary_radar8.png) + horizontal bars."""
    if not HAS_MATPLOTLIB:
        return {}
    idea_keys = ['idea_a', 'idea_b', 'idea_c', 'idea_d', 'idea_e', 'idea_f', 'idea_g', 'idea_h']
    idea_labels = ['A Aff', 'B Mode', 'C Drug', 'D Uniq', 'E Comp', 'F Chem', 'G Anch', 'H Vol']
    scores = []
    for k in idea_keys:
        r = result.get(k, {})
        scores.append(r.get('score', 0.0) if r.get('success') else 0.0)
    overall = float(result.get('overall_score') or 0.0)
    output_dir = Path(output_dir) if output_dir else None
    if not output_dir:
        return {}
    try:
        from utils.pocket_vis_nature import ACCENT, FONT, ICEBLUE_EDGE, ICEBLUE_FILL, apply_nature_axes
    except ImportError:
        ACCENT, ICEBLUE_FILL, ICEBLUE_EDGE = '#F56E1A', '#C5DDF0', '#5A7FA0'
        FONT = {'title': 11, 'label': 9, 'tick': 8, 'annot': 7}

        def apply_nature_axes(ax, grid_y=True, grid_x=False):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    saved = {}

    # ---- Fig 1: Radar ----
    fig1, ax_radar = plt.subplots(figsize=(5.6, 5.2), subplot_kw=dict(polar=True))
    N = len(idea_labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals = scores + scores[:1]
    ax_radar.set_facecolor('#FAFBFC')
    ax_radar.fill(angles, [0.70] * (N + 1), color='#A8C5A0', alpha=0.12)
    ax_radar.fill(angles, [0.45] * (N + 1), color='#F0D9A8', alpha=0.10)
    ax_radar.plot(angles, vals, 'o-', lw=1.8, color=ICEBLUE_EDGE, markersize=5,
                  markerfacecolor=ICEBLUE_FILL, markeredgecolor=ICEBLUE_EDGE, zorder=5)
    ax_radar.fill(angles, vals, alpha=0.22, color=ICEBLUE_FILL)
    ax_radar.set_thetagrids(np.degrees(angles[:-1]), idea_labels, fontsize=FONT['label'])
    ax_radar.set_ylim(0, 1)
    ax_radar.spines['polar'].set_color('#B0B0B0')
    ax_radar.spines['polar'].set_linewidth(0.8)
    ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_radar.set_yticklabels([])  # avoid collision with value labels
    ax_radar.grid(True, color='#E8E8E8', linewidth=0.7)
    for angle, val in zip(angles[:-1], scores):
        r_txt = val - 0.12 if val >= 0.92 else val + 0.10
        r_txt = float(np.clip(r_txt, 0.08, 0.98))
        ax_radar.annotate(
            f'{val:.2f}', xy=(angle, val),
            xytext=(angle, r_txt),
            fontsize=FONT['annot'], ha='center', va='center', color='#4A4A4A',
        )
    ax_radar.set_title(_viz_title(title_prefix, 'A–H radar'), fontsize=FONT['title'], pad=14, color='#4A4A4A')
    plt.tight_layout()
    _safe_savefig(fig1, output_dir / 'summary_radar8.png', dpi=180)
    saved['summary_radar8'] = str(output_dir / 'summary_radar8.png')

    # ---- Fig 2: Horizontal A–H score bars (beautified) ----
    fig2, ax_bar = plt.subplots(figsize=(6.0, 4.6))
    # Per-dimension palette (ligand / compatibility / pocket layers)
    DIM_COLORS = {
        'A Aff': '#3D7A9C',   # ligand — deep iceblue
        'B Mode': '#5A9E8F',  # teal
        'C Drug': '#7EB3C9',  # light iceblue
        'D Uniq': '#6B9E78',  # compatibility — soft green
        'E Comp': '#8FB89A',
        'F Chem': '#D4894A',  # pocket — warm amber (not purple)
        'G Anch': '#E0A060',
        'H Vol':  '#C4A06A',
    }
    # Score-aware lighten: low scores fade toward neutral
    colors_bar = []
    for lab, s in zip(idea_labels, scores):
        base = DIM_COLORS.get(lab, ICEBLUE_EDGE)
        # blend with light gray when score is low
        fade = float(np.clip(s, 0.0, 1.0))
        r = int(base[1:3], 16)
        g = int(base[3:5], 16)
        b = int(base[5:7], 16)
        nr = int(r * fade + 200 * (1 - fade))
        ng = int(g * fade + 200 * (1 - fade))
        nb = int(b * fade + 200 * (1 - fade))
        colors_bar.append(f'#{nr:02X}{ng:02X}{nb:02X}')

    y = np.arange(len(idea_labels))
    # soft track + score bars (A at top via invert_yaxis)
    ax_bar.barh(
        y, [1.0] * len(scores),
        color='#F0F2F4', edgecolor='none', height=0.68, zorder=1,
    )
    bars = ax_bar.barh(
        y, scores, color=colors_bar,
        edgecolor='white', linewidth=0.8, height=0.68, zorder=3,
    )
    _ = bars

    ax_bar.axvline(0.70, color='#6B8F71', ls=':', lw=1.1, alpha=0.80, zorder=2)
    ax_bar.axvline(0.45, color='#C4A35A', ls=':', lw=1.1, alpha=0.80, zorder=2)
    ax_bar.axvline(overall, color=ACCENT, ls='-', lw=1.6, alpha=0.95, zorder=4)
    ax_bar.plot(
        [overall], [len(idea_labels) - 0.45], marker='D', markersize=6,
        color=ACCENT, markeredgecolor='white', markeredgewidth=0.6, zorder=5,
        clip_on=False,
    )

    for yi, val in zip(y, scores):
        txt_x = val + 0.02 if val < 0.88 else val - 0.02
        ha = 'left' if val < 0.88 else 'right'
        tc = '#4A4A4A' if val < 0.88 else 'white'
        ax_bar.text(
            txt_x, yi, f'{val:.2f}', va='center', ha=ha,
            fontsize=FONT['annot'], color=tc, fontweight='medium', zorder=6,
        )

    apply_nature_axes(ax_bar, grid_y=False, grid_x=True)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(idea_labels, fontsize=FONT['label'])
    ax_bar.invert_yaxis()  # A Aff on top
    ax_bar.set_xlim(0, 1.12)
    ax_bar.set_ylim(len(idea_labels) - 0.15, -0.85)
    ax_bar.set_xlabel('Score', fontsize=FONT['label'])
    ax_bar.set_title(
        _viz_title(title_prefix, 'A–H scores'),
        fontsize=FONT['title'], pad=8,
    )
    ax_bar.text(
        0.98, 0.02, f'overall = {overall:.3f}',
        transform=ax_bar.transAxes, ha='right', va='bottom',
        fontsize=FONT['annot'], color=ACCENT, fontweight='bold',
    )
    # compact layer legend
    ax_bar.text(
        0.02, 0.02,
        'ligand A–C  ·  compatibility D–E  ·  pocket F–H',
        transform=ax_bar.transAxes, ha='left', va='bottom',
        fontsize=FONT['annot'], color='#888888',
    )
    plt.tight_layout()
    _safe_savefig(fig2, output_dir / 'summary_bar.png', dpi=180)
    saved['summary_bar'] = str(output_dir / 'summary_bar.png')
    return saved


def generate_all_visualizations(result, molecules_with_pos, output_dir,
                                 title_prefix='', use_tsne=False):
    """
    一键生成所有可视化图像

    Args:
        result            : evaluate_pocket_quality() 的返回值
        molecules_with_pos: [(mol, pos_array), ...]
        output_dir        : 输出目录
        title_prefix      : 图标题前缀（通常为口袋 ID）
        use_tsne          : 是否在聚类图中使用 t-SNE

    Returns:
        dict: {图名: 文件路径}
    """
    if not HAS_MATPLOTLIB:
        print("⚠️ matplotlib 未安装，跳过可视化")
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    print(f"\n{'='*60}")
    print(f"Generating visualizations -> {output_dir}")
    print('='*60)

    # 0. Chemical space UMAP (Fig.2a contours, colored by primary atom type; no Native)
    r = visualize_chemspace_umap(
        molecules_with_pos,
        idea_a_result=result.get('idea_a', {}),
        output_dir=output_dir,
        title_prefix=title_prefix,
    )
    saved.update(r)

    # 1. Atom clustering (Idea B)
    idea_b = result.get('idea_b', {})
    r = visualize_clustering_2d(
        molecules_with_pos,
        idea_b_result=idea_b if idea_b.get('success') else None,
        output_dir=output_dir,
        title_prefix=title_prefix,
        n_kmeans=5,
        use_tsne=use_tsne,
    )
    saved.update(r)

    # 2. Vina + LE on one figure (Idea A components)
    r = visualize_vina_le_combined(
        idea_a_result=result.get('idea_a', {}),
        idea_le_result=result.get('idea_le') or (result.get('idea_a') or {}).get('le_component') or {},
        output_dir=output_dir,
        title_prefix=title_prefix,
    )
    saved.update(r)

    # 3. Druglikeness (Idea C slot; keep D_* filenames for continuity)
    idea_a = result.get('idea_a', {})
    vina_scores = idea_a.get('vina_scores') if idea_a.get('success') else None
    r = visualize_druglikeness(
        result.get('idea_c', {}),
        output_dir=output_dir,
        title_prefix=title_prefix,
        vina_scores=vina_scores,
    )
    saved.update(r)

    # 4. Uniqueness + Completeness (D/E combined, H-style bands, one tone)
    r = visualize_uniqueness_completeness(
        idea_d_result=result.get('idea_d', {}),
        idea_e_result=result.get('idea_e', {}),
        output_dir=output_dir,
        title_prefix=title_prefix,
    )
    saved.update(r)

    # 4b. Pocket chemistry F
    r = visualize_pocket_chemistry(
        result.get('idea_f', {}),
        output_dir=output_dir,
        title_prefix=title_prefix,
    )
    saved.update(r)

    # 5. Size distribution — auxiliary only (not scored; G = anchors)
    try:
        size_aux = evaluate_idea_g_size_consistency(molecules_with_pos)
        r = visualize_size_distribution(
            size_aux,
            output_dir=output_dir,
            title_prefix=title_prefix,
        )
        saved.update(r)
    except Exception:
        pass

    # 5b. Anchor richness G
    r = visualize_anchor_richness(
        result.get('idea_g', {}),
        output_dir=output_dir,
        title_prefix=title_prefix,
    )
    saved.update(r)

    # 6. Pocket volume (Idea H)
    r = visualize_pocket_volume(
        result.get('idea_h', {}),
        output_dir=output_dir,
        title_prefix=title_prefix,
    )
    saved.update(r)

    # 7. Summary radar / bar
    r = visualize_radar_summary(
        result,
        output_dir=output_dir,
        title_prefix=title_prefix,
    )
    saved.update(r)

    # 7b. three-layer summary
    r = visualize_layer_summary(
        result,
        output_dir=output_dir,
        title_prefix=title_prefix,
    )
    saved.update(r)

    # Drop obsolete filenames from earlier naming schemes
    for obsolete in (
        'C_le_hist.png', 'C_le_hist_notext.png',
        'C_le_box.png', 'C_le_box_notext.png',
        'C_le_cdf.png', 'C_le_cdf_notext.png',
        'G_size_mw.png', 'G_size_mw_notext.png',
        'G_size_atoms.png', 'G_size_atoms_notext.png',
        'G_size_mw_vs_atoms.png', 'G_size_mw_vs_atoms_notext.png',
        'D_uniqueness.png', 'D_uniqueness_notext.png',
        'E_completeness.png', 'E_completeness_notext.png',
    ):
        op = output_dir / obsolete
        if op.exists():
            try:
                op.unlink()
            except OSError:
                pass

    print(f"\nGenerated {len(saved)} images:")
    for name, path in saved.items():
        print(f"  [{name}] {path}")
    return saved


# =============================================================================
# 综合评估
# =============================================================================

def evaluate_pocket_quality(
    pt_path=None,
    ligand_path=None,
    custom_pocket_pdb=None,
    vina_outputs_dir=None,
    vina_pocket_id=None,
    protein_root=None,
    data_id=None,
    atom_mode='add_aromatic',
    weight_a=0.45, weight_b=0.25, weight_c=0.30,
    weight_d=0.45, weight_e=0.55, weight_f=0.20, weight_g=0.55,
    weight_h=0.25,
    idea_b_atom_coord_subset='hetero_heavy',
    idea_b_eps=1.5,
    idea_b_edge_max_heavy_neighbors=2,
    idea_b_combined_focus_weight=0.5,
    visualize=False, vis_dir=None, use_tsne=False,
    fpocket_protein_pdb=None,
    fpocket_cmd='fpocket',
    fpocket_pocket_index=1,
    fpocket_max_ligand_models=50,
    fpocket_timeout=600,
    fpocket_optimal_min=None,
    fpocket_optimal_max=None,
    fpocket_zero_below=None,
    fpocket_zero_above=None,
    idea_h_ligand_path=None,
    idea_e_expected_n_molecules=None,
):
    """
    综合评估口袋质量（八维三层）。

    字母槽：A Aff+LE / B Mode / C Drug / D Unique / E Complete / F Chem / G Anchor / H Geom。
    层：ligand=ABC，compatibility=DE，pocket=FGH。
    层内加权几何平均；层间 S_overall=(Sp·Sc·Sl)^(1/3)。

    weight_* 为层内相对权重（默认见 LAYER_WEIGHTS_*）；失败维剔除后重归一。
    """
    pp = Path(pt_path) if pt_path else None
    lp = Path(ligand_path) if ligand_path else None

    if pp is not None and pp.exists() and pp.is_file():
        molecules = load_molecules_from_pt(pp, atom_mode=atom_mode)
        eval_tag = data_id
        evaluation_source = 'pt'
    elif lp is not None and lp.exists():
        molecules = load_molecules_from_ligand_paths(lp)
        eval_tag = data_id if data_id is not None else 'custom'
        evaluation_source = 'ligand_file'
        pp = None
    else:
        hint = pt_path or ligand_path
        return {
            'error': f'未找到有效的 .pt 或配体路径: {hint}',
            'overall_score': 0.0, 'overall_label': 'unknown',
        }

    if not molecules:
        return {
            'error': '未能从 .pt 或配体文件解析出带 3D 构象的分子',
            'overall_score': 0.0, 'overall_label': 'unknown',
        }

    molecules_for_h = molecules
    idea_h_ligand_path_resolved = None
    idea_h_ligand_override_used = False
    if idea_h_ligand_path and str(idea_h_ligand_path).strip():
        ihp = Path(idea_h_ligand_path).expanduser().resolve()
        if ihp.exists():
            alt_h = load_molecules_from_ligand_paths(
                ihp, max_mols=int(fpocket_max_ligand_models) if fpocket_max_ligand_models else None,
            )
            if alt_h:
                molecules_for_h = alt_h
                idea_h_ligand_path_resolved = str(ihp)
                idea_h_ligand_override_used = True

    vod = resolve_vina_outputs_dir(
        vina_outputs_dir=vina_outputs_dir,
        pt_path=str(pp) if pp is not None else None,
        ligand_path=str(lp) if lp is not None else None,
        custom_pocket_pdb=custom_pocket_pdb,
    )
    if vod is None and pp is not None and pp.is_file():
        vod = pp.parent

    # Prefer explicit fpocket pdb; else custom pocket pdb; else resolve from .pt
    pocket_pdb = fpocket_protein_pdb or custom_pocket_pdb
    if not pocket_pdb and pp is not None:
        pocket_pdb = resolve_receptor_pdb_from_pt(pp, protein_root=protein_root)
        if pocket_pdb:
            print(f'Auto-resolved receptor PDB from .pt -> {pocket_pdb}', flush=True)

    idea_a_vina = evaluate_idea_a_vina(
        pt_path=str(pp) if pp is not None else None,
        data_id=eval_tag,
        outputs_dir=str(vod) if vod is not None else None,
        pocket_id_override=vina_pocket_id,
    )
    idea_b = evaluate_idea_b_clustering(
        molecules,
        eps=idea_b_eps,
        atom_coord_subset=idea_b_atom_coord_subset,
        edge_max_heavy_neighbors=idea_b_edge_max_heavy_neighbors,
        combined_focus_weight=idea_b_combined_focus_weight,
    )
    vina_for_le = idea_a_vina.get('vina_scores') if idea_a_vina.get('success') else []
    idea_le = evaluate_idea_c_ligand_efficiency(molecules, vina_for_le)
    idea_a = _merge_affinity_le(idea_a_vina, idea_le)
    idea_c = evaluate_idea_d_druglikeness(molecules)  # slot C = druglikeness
    idea_d = evaluate_idea_f_uniqueness(molecules)    # slot D = uniqueness
    idea_e = evaluate_idea_e_reconstruction(
        str(pp) if pp is not None else None,
        molecules,
        expected_n_molecules=idea_e_expected_n_molecules,
    )
    idea_f = evaluate_idea_f_pocket_chemistry(
        fpocket_protein_pdb=pocket_pdb,
        molecules_with_pos=molecules,
        centroid_radius=10.0,
    )
    idea_g = evaluate_idea_g_anchor_richness(
        fpocket_protein_pdb=pocket_pdb,
        molecules_with_pos=molecules,
    )
    idea_h = evaluate_idea_h_pocket_size(
        molecules_for_h,
        optimal_min=350, optimal_max=750,
        centroid_radius=10.0,
        fpocket_protein_pdb=pocket_pdb,
    )
    if evaluation_source == 'pt' and idea_h_ligand_path and str(idea_h_ligand_path).strip():
        idea_h = dict(idea_h)
        if idea_h_ligand_override_used:
            idea_h['ligand_fpocket_pose_source'] = 'idea_h_ligand_path'
            idea_h['ligand_fpocket_pose_path'] = idea_h_ligand_path_resolved
        else:
            idea_h['ligand_fpocket_pose_source'] = 'pt'
            idea_h['ligand_fpocket_pose_path'] = None
            _ihp = Path(idea_h_ligand_path).expanduser().resolve()
            if not _ihp.exists():
                idea_h['ligand_fpocket_override_note'] = (
                    f'--idea_h_ligand_path 不存在: {idea_h_ligand_path}'
                )
            else:
                idea_h['ligand_fpocket_override_note'] = (
                    '已指定 --idea_h_ligand_path 但未解析到带 3D 构象的分子，H 配体 FPocket 仍用 .pt 分子'
                )

    # Aux LE result for LE plots (slot C is druglikeness)
    idea_le_for_viz = idea_le

    dim_map = {
        'A': idea_a, 'B': idea_b, 'C': idea_c, 'D': idea_d,
        'E': idea_e, 'F': idea_f, 'G': idea_g, 'H': idea_h,
    }
    w_map = {
        'A': weight_a, 'B': weight_b, 'C': weight_c, 'D': weight_d,
        'E': weight_e, 'F': weight_f, 'G': weight_g, 'H': weight_h,
    }

    def _layer_score(keys, default_w):
        scores, weights = [], []
        for k in keys:
            res = dim_map[k]
            if res.get('success'):
                scores.append(res.get('score', 0.0))
                weights.append(w_map.get(k, default_w.get(k, 1.0)))
        if not scores:
            return float(GEO_EPS), False
        return _weighted_geo_mean(scores, weights), True

    s_ligand, ok_l = _layer_score(['A', 'B', 'C'], LAYER_WEIGHTS_LIGAND)
    s_compat, ok_c = _layer_score(['D', 'E'], LAYER_WEIGHTS_COMPAT)
    s_pocket, ok_p = _layer_score(['F', 'G', 'H'], LAYER_WEIGHTS_POCKET)

    layer_scores = []
    if ok_p:
        layer_scores.append(s_pocket)
    if ok_c:
        layer_scores.append(s_compat)
    if ok_l:
        layer_scores.append(s_ligand)
    overall_score = _equal_geo_mean(layer_scores) if layer_scores else 0.0

    if overall_score >= 0.70:
        overall_label = 'high'
    elif overall_score >= 0.45:
        overall_label = 'medium'
    else:
        overall_label = 'low'

    cpp = Path(custom_pocket_pdb).resolve() if custom_pocket_pdb else None

    result = {
        'pt_path': str(pp.resolve()) if pp is not None else None,
        'ligand_path': str(lp.resolve()) if lp is not None else None,
        'custom_pocket_pdb': str(cpp) if cpp is not None else None,
        'evaluation_source': evaluation_source,
        'vina_outputs_dir': str(vod) if vod is not None else None,
        'data_id': eval_tag,
        'n_molecules': len(molecules),
        'idea_a': idea_a,
        'idea_b': idea_b,
        'idea_c': idea_c,
        'idea_d': idea_d,
        'idea_e': idea_e,
        'idea_f': idea_f,
        'idea_g': idea_g,
        'idea_h': idea_h,
        'idea_le': idea_le_for_viz,
        's_pocket': s_pocket if ok_p else None,
        's_compatibility': s_compat if ok_c else None,
        's_ligand': s_ligand if ok_l else None,
        'overall_score': overall_score,
        'overall_label': overall_label,
        'weights': w_map,
        'layer_weights': {
            'ligand': dict(LAYER_WEIGHTS_LIGAND),
            'compatibility': dict(LAYER_WEIGHTS_COMPAT),
            'pocket': dict(LAYER_WEIGHTS_POCKET),
        },
        'visualizations': {},
        'idea_h_ligand_path': idea_h_ligand_path_resolved,
        'idea_h_ligand_override_used': idea_h_ligand_override_used,
        'idea_e_expected_n_molecules': idea_e_expected_n_molecules,
        'vis_dir': str(Path(vis_dir).resolve()) if vis_dir else None,
        'fpocket_protein_pdb': str(Path(pocket_pdb).resolve()) if pocket_pdb else None,
    }

    if visualize and len(molecules) > 0 and vis_dir:
        if data_id is not None:
            prefix = str(data_id)
        elif pp is not None:
            prefix = pp.stem
        else:
            prefix = lp.stem if lp is not None else 'custom'
        vis_paths = generate_all_visualizations(
            result, molecules, vis_dir,
            title_prefix=prefix,
            use_tsne=use_tsne,
        )
        result['visualizations'] = vis_paths
        print(f"Visualization dir -> {result['vis_dir']}", flush=True)

    return result


# =============================================================================
# 调用 batch 脚本
# =============================================================================

def run_batch_sampleandeval(start=1, end=99, gpus='0', num_cpu_cores=None, cores_per_task=1,
                             protein_path=None, ligand_path=None, protein_root=None,
                             sample_only=False):
    """调用 batch_sampleandeval_parallel.py 执行采样和评估"""
    if num_cpu_cores is None:
        num_cpu_cores = min(DEFAULT_NUM_CPU_CORES, cpu_count())
    if not BATCH_SCRIPT.exists():
        return False, [], f'脚本不存在: {BATCH_SCRIPT}'

    cmd = [sys.executable, str(BATCH_SCRIPT)]

    if protein_path:
        cmd.extend(['--protein_path', str(protein_path)])
        if ligand_path:
            cmd.extend(['--ligand_path', str(ligand_path)])
    else:
        cmd.extend(['--start', str(start), '--end', str(end)])

    cmd.extend([
        '--gpus', gpus,
        '--num_cpu_cores', str(num_cpu_cores),
        '--cores_per_task', str(cores_per_task),
    ])
    if protein_root:
        cmd.extend(['--protein_root', str(protein_root)])
    if sample_only:
        cmd.append('--sample-only')

    print(f'执行: {" ".join(cmd)}')
    try:
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    except subprocess.CalledProcessError as e:
        return False, [], f'batch 脚本执行失败: {e}'

    if protein_path:
        pattern = str(OUTPUT_DIR / 'result_custom_*.pt')
    else:
        pattern = str(OUTPUT_DIR / 'result_*_*.pt')

    pt_files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return True, pt_files, '完成'


def _eval_single_pt_task(args):
    """Parallel evaluation worker for Pool.map (receives single tuple)"""
    (pt, protein_root, data_id, atom_mode, weight_a, weight_b, weight_c, weight_d,
     weight_e, weight_f, weight_g, weight_h, visualize, vis_dir, use_tsne,
     idea_b_atom_coord_subset, idea_b_edge_max_heavy_neighbors,
     idea_b_combined_focus_weight, vina_outputs_dir, vina_pocket_id,
     custom_pocket_pdb, fpocket_protein_pdb, fpocket_cmd, fpocket_pocket_index,
     fpocket_max_ligand_models, fpocket_timeout,
     fpocket_optimal_min, fpocket_optimal_max, fpocket_zero_below, fpocket_zero_above,
     idea_h_ligand_path, idea_e_expected_n_molecules) = args
    return evaluate_pocket_quality(
        pt_path=pt, protein_root=protein_root, data_id=data_id, atom_mode=atom_mode,
        weight_a=weight_a, weight_b=weight_b, weight_c=weight_c, weight_d=weight_d,
        weight_e=weight_e, weight_f=weight_f, weight_g=weight_g, weight_h=weight_h,
        idea_b_atom_coord_subset=idea_b_atom_coord_subset,
        idea_b_edge_max_heavy_neighbors=idea_b_edge_max_heavy_neighbors,
        idea_b_combined_focus_weight=idea_b_combined_focus_weight,
        vina_outputs_dir=vina_outputs_dir,
        vina_pocket_id=vina_pocket_id,
        custom_pocket_pdb=custom_pocket_pdb,
        visualize=visualize, vis_dir=vis_dir, use_tsne=use_tsne,
        fpocket_protein_pdb=fpocket_protein_pdb,
        fpocket_cmd=fpocket_cmd,
        fpocket_pocket_index=fpocket_pocket_index,
        fpocket_max_ligand_models=fpocket_max_ligand_models,
        fpocket_timeout=fpocket_timeout,
        fpocket_optimal_min=fpocket_optimal_min,
        fpocket_optimal_max=fpocket_optimal_max,
        fpocket_zero_below=fpocket_zero_below,
        fpocket_zero_above=fpocket_zero_above,
        idea_h_ligand_path=idea_h_ligand_path,
        idea_e_expected_n_molecules=idea_e_expected_n_molecules,
    )


def find_pt_files_for_range(start, end, output_dir=None):
    """根据 data_id 范围查找 .pt 文件"""
    output_dir = output_dir or OUTPUT_DIR
    found = []
    for i in range(start, end + 1):
        pattern = str(output_dir / f'result_{i}_*.pt')
        matches = glob.glob(pattern)
        if matches:
            found.append(max(matches, key=os.path.getmtime))
    return found


# =============================================================================
# 主入口与打印
# =============================================================================

def print_evaluation_report(result):
    """打印评估报告（含所有维度及丰富指标）"""
    print('\n' + '=' * 70)
    print('蛋白质口袋质量评估报告')
    print('=' * 70)
    if result.get('error'):
        print(f"错误: {result['error']}")
        print('=' * 70 + '\n')
        return
    src = result.get('pt_path') or result.get('ligand_path')
    print(f"数据源: {src or 'N/A'} ({result.get('evaluation_source', 'unknown')})")
    if result.get('custom_pocket_pdb'):
        print(f"自定义口袋 PDB: {result.get('custom_pocket_pdb')}")
    if result.get('vina_outputs_dir'):
        print(f"Vina 输出目录: {result.get('vina_outputs_dir')}")
    print(f"分子数: {result.get('n_molecules', 0)}")
    print()

    a = result.get('idea_a', {})
    print('【想法A】亲和力 + 配体效率（Aff+LE）')
    print(f"  状态: {'成功' if a.get('success') else '失败'}")
    if a.get('success'):
        if a.get('vina_mean') is not None:
            print(f"  Vina 平均/中位数/标准差: {a.get('vina_mean'):.2f} / "
                  f"{a.get('vina_median'):.2f} / {a.get('vina_std'):.2f} kcal/mol")
            print(f"  Vina 最佳/最差: {a.get('vina_best'):.2f} / {a.get('vina_worst'):.2f} kcal/mol")
            print(f"  亲和力良好比例(Vina≤-7): {a.get('vina_pct_good', 0)*100:.1f}%")
            print(f"  有效分数数量: {a.get('num_scores')}")
        le = a.get('le_component') or result.get('idea_le') or {}
        if le.get('success') and le.get('le_mean') is not None:
            print(
                f"  LE 均值/中位数: {le.get('le_mean'):.4f} / {le.get('le_median'):.4f} "
                f"kcal·mol⁻¹·重原子⁻¹"
            )
        print(f"  质量分数: {a.get('score'):.3f} ({a.get('quality_label')})")
        if a.get('message'):
            print(f"  摘要: {a.get('message')}")
    else:
        print(f"  备注: {a.get('message', '')}")
    print()

    b = result.get('idea_b', {})
    print('【想法B】原子分布聚类（结合模式收敛性）')
    print(f"  状态: {'成功' if b.get('success') else '失败'}")
    if b.get('success'):
        sub = b.get('atom_coord_subset', 'all')
        print(f"  DBSCAN 坐标子集: {sub}（全原子质心 KMeans 不变）")
        nad = b.get('n_atoms_dbscan')
        nt = b.get('n_atoms_total')
        if nad is not None and nt is not None:
            print(f"  DBSCAN 用原子数: {nad} / 总原子 {nt}")
        print(f"  DBSCAN 簇数 / 噪声点数: {b.get('n_clusters')} / {b.get('n_noise', 'N/A')}")
        sil = b.get('silhouette')
        print(f"  轮廓系数: {f'{sil:.4f}' if sil is not None else 'N/A'}")
        if sub == 'combined':
            sa = b.get('score_all_atoms')
            sf = b.get('score_focus')
            w = b.get('combined_focus_weight')
            print(
                f"  分项: 全原子分={sa if sa is not None else 'N/A'}, "
                f"hetero_edge分={sf if sf is not None else 'N/A'}, w={w}"
            )
            print(
                f"  hetero_edge 簇/噪声: {b.get('n_clusters_focus')} / "
                f"{b.get('n_noise_focus', 'N/A')}"
            )
        ki = b.get('kmeans_inertia')
        print(f"  质心 KMeans 惯性: {f'{ki:.2f}' if ki is not None else 'N/A'}")
        print(f"  质量分数: {b.get('score'):.3f} ({b.get('quality_label')})")
    else:
        print(f"  备注: {b.get('message', '')}")
    print()

    c = result.get('idea_c', {})
    print('【想法C】药物相似性 (QED/SA/Lipinski/PAINS)')
    print(f"  状态: {'成功' if c.get('success') else '失败'}")
    if c.get('success'):
        print(f"  QED 均值±标准差: {c.get('qed_mean', 0):.3f} ± {c.get('qed_std', 0):.3f}")
        print(f"  SA  均值±标准差: {c.get('sa_mean', 0):.3f} ± {c.get('sa_std', 0):.3f}")
        print(f"  Lipinski 合规(0-1): {c.get('lipinski_mean', 0):.2f}")
        print(f"  PAINS 命中率: {c.get('pains_ratio', 0)*100:.1f}%")
        print(f"  质量分数: {c.get('score'):.3f} ({c.get('quality_label')})")
    else:
        print(f"  备注: {c.get('message', '')}")
    print()

    d = result.get('idea_d', {})
    print('【想法D】分子唯一性（进分）与指纹相异度（参考）')
    print(f"  状态: {'成功' if d.get('success') else '失败'}")
    if d.get('success'):
        print(f"  唯一分子/有效分子: {d.get('n_unique')}/{d.get('n_total')}")
        print(f"  唯一性比例: {d.get('unique_ratio', 0)*100:.1f}%")
        fd = d.get('fingerprint_dissimilarity', d.get('tanimoto_diversity'))
        print(f"  指纹余弦相异度(参考,不进分): {f'{fd:.4f}' if fd is not None else 'N/A'}")
        print(f"  质量分数: {d.get('score'):.3f} ({d.get('quality_label')})")
    else:
        print(f"  备注: {d.get('message', '')}")
    print()

    e = result.get('idea_e', {})
    print("【想法E】完整分子比例（SMILES 无 '.' 断点）")
    print(f"  状态: {'成功' if e.get('success') else '失败'}")
    if e.get('success'):
        ds = e.get('denominator_source')
        if ds == 'expected_n_molecules':
            print('  分母来源: 用户指定应生成分子数 (--idea_e_expected_n_molecules)')
        elif ds == 'pt_pred_ligand_pos':
            print('  分母来源: .pt 中 pred_ligand_pos 条数')
        nl = e.get('n_loaded_molecules')
        if nl is not None:
            print(f"  已载入 3D 构象数: {nl}")
        print(f"  完整分子数/分母: {e.get('n_success')}/{e.get('n_total')}")
        print(f"  完整分子比例: {e.get('reconstruct_rate', 0)*100:.1f}%")
        print(f"  质量分数: {e.get('score'):.3f} ({e.get('quality_label')})")
    else:
        print(f"  备注: {e.get('message', '')}")
    print()

    f = result.get('idea_f', {})
    print('【想法F】口袋化学（配体邻域疏水/极性/封闭度，不调用 FPocket）')
    print(f"  状态: {'成功' if f.get('success') else '失败'}")
    if f.get('success'):
        print(
            f"  邻域半径: {f.get('centroid_radius')} Å；"
            f"残基数={f.get('n_residues')}；重原子={f.get('n_heavy_atoms')}"
        )
        print(
            f"  比例: hydro={f.get('hydro_fraction')}, polar={f.get('polar_fraction')}, "
            f"enc_raw={f.get('enclosure_raw')}"
        )
        print(
            f"  mapped: hydro={f.get('druggability_mapped')}, "
            f"polar={f.get('fpocket_score_mapped')}, enc={f.get('hydrophobicity_mapped')}"
        )
        print(f"  质量分数: {f.get('score'):.3f} ({f.get('quality_label')})")
        if f.get('message'):
            print(f"  摘要: {f.get('message')}")
    else:
        print(f"  备注: {f.get('message', '')}")
    print()

    g = result.get('idea_g', {})
    print('【想法G】相互作用锚点丰富度（P4）')
    print(f"  状态: {'成功' if g.get('success') else '失败'}")
    if g.get('success'):
        print(
            f"  richness/coverage/balance: "
            f"{g.get('s_richness'):.3f} / {g.get('s_coverage'):.3f} / {g.get('s_balance'):.3f}"
        )
        print(
            f"  Neff={g.get('n_eff')} / cap={g.get('n_cap'):.0f}, "
            f"clusters={g.get('n_clusters')}, pocket_res={g.get('n_pocket_residues')}"
        )
        ac = g.get('anchor_counts') or {}
        if ac:
            print(f"  counts: {ac}")
        print(f"  质量分数: {g.get('score'):.3f} ({g.get('quality_label')})")
        if g.get('message'):
            print(f"  摘要: {g.get('message')}")
    else:
        print(f"  备注: {g.get('message', '')}")
    print()

    h = result.get('idea_h', {})
    print('【想法H】口袋几何体积（配体质心 10Å 球内 MC 占据，满分带 350–750 Å³）')
    print(f"  状态: {'成功' if h.get('success') else '失败'}")
    if h.get('success'):
        vol = h.get('volume_ang3')
        if vol is not None:
            print(f"  用于评分的体积: {vol:.0f} Å³ ({h.get('volume_method', '')})")
        print(f"  质量分数: {h.get('score'):.3f} ({h.get('quality_label')})")
        if h.get('message'):
            print(f"  摘要: {h.get('message')}")
    else:
        print(f"  备注: {h.get('message', '')}")
    print()

    print('【综合评估】三层几何平均')
    sp, sc, sl = result.get('s_pocket'), result.get('s_compatibility'), result.get('s_ligand')
    print(
        f"  S_pocket={f'{sp:.3f}' if sp is not None else 'n/a'}  "
        f"S_compat={f'{sc:.3f}' if sc is not None else 'n/a'}  "
        f"S_ligand={f'{sl:.3f}' if sl is not None else 'n/a'}"
    )
    print(f"  综合质量分数: {result.get('overall_score', 0):.3f}")
    print(f"  质量等级: {result.get('overall_label', 'unknown').upper()}")

    vis = result.get('visualizations', {})
    if vis:
        print(f"\n【可视化图像】")
        for name, path in vis.items():
            print(f"  [{name}] {path}")

    print('=' * 70 + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='蛋白质口袋质量评估（基于扩散模型生成，含可视化）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 直接评估 .pt 文件并生成可视化
  python evaluate_pocket_quality.py --pt_file outputs/result_5.pt --visualize

  # 指定可视化输出目录
  python evaluate_pocket_quality.py --pt_file outputs/result_5.pt --visualize --vis_dir ./my_vis

  # 批量评估并可视化
  python evaluate_pocket_quality.py --run_batch --start 1 --end 10 --gpus "0" --visualize

  # 单口袋（先生成再评估）
  python evaluate_pocket_quality.py --pocket_pdb data/pocket.pdb --visualize

  # 自定义口袋+配体：仅评估已有 SDF/MOL（不依赖 .pt；想法E 跳过）
  python evaluate_pocket_quality.py --eval_ligands my/poses.sdf --vina_outputs_dir outputs --visualize
  python evaluate_pocket_quality.py --eval_ligands shoc2/8v1tligand.sdf \\
    --custom_pocket_pdb shoc2/8v1t.pdb --vina_outputs_dir outputs --visualize

  # 想法 H：FPocket 蛋白口袋参与评分；配体合并 PDB 再跑 FPocket 仅报告
  python evaluate_pocket_quality.py --pt_file outputs/result_5.pt \\
    --fpocket_protein_pdb data/target.pdb --visualize
        """
    )

    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--pocket_pdb', type=str, help='口袋 PDB 文件路径（将调用 batch 脚本生成）')
    g.add_argument('--pt_file', type=str, help='已有 .pt 文件路径（跳过生成，直接评估）')
    g.add_argument(
        '--pt_dir',
        type=str,
        metavar='DIR',
        help='已有 .pt 所在目录：与 --start/--end 联用，按 result_{i}_*.pt 批量评估（不重新生成）；'
             '未指定 --vina_outputs_dir 时默认用该目录定位 eval_*',
    )
    g.add_argument('--run_batch', action='store_true',
                   help='运行 batch_sampleandeval_parallel 后评估指定范围')
    g.add_argument(
        '--eval_ligands',
        type=str,
        metavar='PATH',
        help='自定义配体 .sdf/.mol 或目录（仅评估，不读 .pt；需 batch 已生成 eval_* 或指定 --vina_outputs_dir）',
    )

    parser.add_argument('--protein_root', type=str, default=None)
    parser.add_argument(
        '--ligand_path',
        type=str,
        default=None,
        help='与 --pocket_pdb 联用：参考配体 SDF，传给 batch_sampleandeval_parallel',
    )
    parser.add_argument(
        '--custom_pocket_pdb',
        type=str,
        default=None,
        help='与 --eval_ligands 联用：自定义口袋 PDB，用于推测 Vina outputs 位置（可选）',
    )
    parser.add_argument(
        '--vina_outputs_dir',
        type=str,
        default=None,
        help='含 eval_* 子目录的文件夹（一般为 batch 的 outputs）；自定义评估时建议显式指定',
    )
    parser.add_argument(
        '--vina_pocket_id',
        type=str,
        default=None,
        help='匹配 eval_{id}_* 的口袋 id（默认：custom 或由 --pt_file 文件名推断）',
    )

    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=99)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--num_cpu_cores', type=int, default=None,
                        help=f'CPU 核心数（默认: min({DEFAULT_NUM_CPU_CORES}, cpu_count())）')
    parser.add_argument('--cores_per_task', type=int, default=1,
                        help='每任务 CPU 核心数，并行数 = num_cpu_cores // cores_per_task（默认: 1）')

    parser.add_argument('--atom_mode', type=str, default='add_aromatic')
    parser.add_argument('--weight_a', type=float, default=0.45,
                        help='层内 A Aff+LE 权重（默认 0.45，ligand 层）')
    parser.add_argument('--weight_b', type=float, default=0.25,
                        help='层内 B Mode 权重（默认 0.25）')
    parser.add_argument('--weight_c', type=float, default=0.30,
                        help='层内 C Drug 权重（默认 0.30）')
    parser.add_argument('--weight_d', type=float, default=0.45,
                        help='层内 D Unique 权重（默认 0.45，compat 层）')
    parser.add_argument('--weight_e', type=float, default=0.55,
                        help='层内 E Complete 权重（默认 0.55）')
    parser.add_argument(
        '--idea_e_expected_n_molecules',
        type=int,
        default=None,
        metavar='N',
        help=(
            "想法E：应生成分子数（正整数）。比例=完整分子数/N，完整分子=SMILES 不含 '.'；"
            "指定后不再用 .pt 里 pred_ligand_pos 长度作分母。"
            "仅 --eval_ligands 时也可凭此启用想法E"
        ),
    )
    parser.add_argument('--weight_f', type=float, default=0.20,
                        help='层内 F Chem 权重（默认 0.20，pocket 层）')
    parser.add_argument('--weight_g', type=float, default=0.55,
                        help='层内 G Anchor 权重（默认 0.55）')
    parser.add_argument('--weight_h', type=float, default=0.25,
                        help='层内 H Geom 权重（默认 0.25）')
    parser.add_argument(
        '--fpocket_protein_pdb',
        type=str,
        default=None,
        help='蛋白 PDB（F/G/H）；未指定时从 .pt 的 protein_filename 自动解析；'
             '需已安装 fpocket（F/H）',
    )
    parser.add_argument(
        '--fpocket_cmd',
        type=str,
        default='fpocket',
        help='FPocket 可执行文件；也可用环境变量 FPOCKET_CMD（默认先找命令行再 PATH/conda）',
    )
    parser.add_argument(
        '--fpocket_pocket_index',
        type=int,
        default=1,
        help='采用 info.txt 中 Pocket N 的编号 N（默认 1，一般为打分最高位点）',
    )
    parser.add_argument(
        '--fpocket_max_ligand_models',
        type=int,
        default=50,
        help='配体侧 FPocket：合并 PDB 中最多写入的构象数（默认 50）',
    )
    parser.add_argument(
        '--fpocket_timeout',
        type=int,
        default=600,
        help='单次 fpocket 子进程超时秒数（默认 600）',
    )
    parser.add_argument(
        '--idea_h_fpocket_vol_min',
        type=float,
        default=None,
        help='配体 FPocket 失败时，蛋白体积回退评分：满分下限 Å³（默认 300）',
    )
    parser.add_argument(
        '--idea_h_fpocket_vol_max',
        type=float,
        default=None,
        help='配体 FPocket 失败时，蛋白体积回退评分：满分上限 Å³（默认 2200）',
    )
    parser.add_argument(
        '--idea_h_fpocket_vol_zero_below',
        type=float,
        default=None,
        help='蛋白体积回退评分：低于该值 Å³ 为 0 分（默认 80）',
    )
    parser.add_argument(
        '--idea_h_fpocket_vol_zero_above',
        type=float,
        default=None,
        help='蛋白体积回退评分：≥该值 Å³ 为 0 分（默认 4500）',
    )
    parser.add_argument(
        '--idea_h_ligand_path',
        type=str,
        default=None,
        metavar='PATH',
        help='与 --pt_file 联用：仅想法 H 配体侧 FPocket 使用该 .sdf/.mol 或目录（A–G 仍用 .pt）；'
             '也可与 --eval_ligands 联用以覆盖 H 的配体构象来源',
    )
    parser.add_argument(
        '--idea_b_atom_subset',
        type=str,
        default='hetero_heavy',
        choices=('all', 'hetero_heavy', 'edge_heavy', 'hetero_edge', 'combined'),
        help='想法B：DBSCAN 用的坐标子集（默认 hetero_heavy）。hetero_heavy=非H非C；'
             'edge_heavy=重邻居≤阈值的表面重原子；hetero_edge=二者交；'
             'combined=(1-w)*全原子+w*hetero_edge（见下）',
    )
    parser.add_argument(
        '--idea_b_edge_max_neighbors',
        type=int,
        default=2,
        help='edge_heavy / hetero_edge：非氢重邻居数上限（默认 2）',
    )
    parser.add_argument(
        '--idea_b_combined_weight',
        type=float,
        default=0.5,
        help='atom_subset=combined 时 hetero_edge 分项权重 w（默认 0.5）',
    )

    # 可视化参数
    parser.add_argument('--visualize', action='store_true',
                        help='生成所有可视化图像（需要 matplotlib）')
    parser.add_argument('--vis_dir', type=str, default=None,
                        help='可视化根目录（默认: pocket_quality_vis/，其下自动创建 蛋白质编号_时间戳/）')
    parser.add_argument('--use_tsne', action='store_true',
                        help='聚类图中使用 t-SNE 替代 PCA（数据量大时较慢）')

    args = parser.parse_args()

    if args.idea_e_expected_n_molecules is not None and args.idea_e_expected_n_molecules <= 0:
        print('❌ --idea_e_expected_n_molecules 须为正整数', file=sys.stderr)
        sys.exit(2)

    if args.eval_ligands:
        el = Path(args.eval_ligands)
        if not el.exists():
            print(f'❌ 配体路径不存在: {el}')
            sys.exit(1)

        base_root = Path(args.vis_dir) if args.vis_dir else VIS_ROOT
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        vis_dir = None
        if args.visualize:
            vis_dir = base_root / f"custom_ligands_{timestamp}"
            vis_dir.mkdir(parents=True, exist_ok=True)
            print(f"Visualization output -> {base_root}")

        cpp = str(Path(args.custom_pocket_pdb).resolve()) if args.custom_pocket_pdb else None
        result = evaluate_pocket_quality(
            pt_path=None,
            ligand_path=str(el.resolve()),
            custom_pocket_pdb=cpp,
            vina_outputs_dir=args.vina_outputs_dir,
            vina_pocket_id=args.vina_pocket_id,
            protein_root=args.protein_root,
            data_id=args.vina_pocket_id or 'custom',
            atom_mode=args.atom_mode,
            weight_a=args.weight_a,
            weight_b=args.weight_b,
            weight_c=args.weight_c,
            weight_d=args.weight_d,
            weight_e=args.weight_e,
            weight_f=args.weight_f,
            weight_g=args.weight_g,
            weight_h=args.weight_h,
            idea_b_atom_coord_subset=args.idea_b_atom_subset,
            idea_b_edge_max_heavy_neighbors=args.idea_b_edge_max_neighbors,
            idea_b_combined_focus_weight=args.idea_b_combined_weight,
            visualize=args.visualize,
            vis_dir=str(vis_dir) if vis_dir else None,
            use_tsne=args.use_tsne,
            fpocket_protein_pdb=args.fpocket_protein_pdb,
            fpocket_cmd=args.fpocket_cmd,
            fpocket_pocket_index=args.fpocket_pocket_index,
            fpocket_max_ligand_models=args.fpocket_max_ligand_models,
            fpocket_timeout=args.fpocket_timeout,
            fpocket_optimal_min=args.idea_h_fpocket_vol_min,
            fpocket_optimal_max=args.idea_h_fpocket_vol_max,
            fpocket_zero_below=args.idea_h_fpocket_vol_zero_below,
            fpocket_zero_above=args.idea_h_fpocket_vol_zero_above,
            idea_h_ligand_path=args.idea_h_ligand_path,
            idea_e_expected_n_molecules=args.idea_e_expected_n_molecules,
        )
        if result.get('error'):
            print(f"❌ {result['error']}")
            sys.exit(1)
        print_evaluation_report(result)
        record_path = base_root / EVAL_RECORDS_CSV
        append_evaluation_record(result, record_path, timestamp=timestamp)
        print(f"\nEvaluation records -> {record_path}")
        sys.exit(0)

    pt_files_to_eval = []

    if args.run_batch:
        ok, pt_files, msg = run_batch_sampleandeval(
            start=args.start, end=args.end,
            gpus=args.gpus,
            num_cpu_cores=args.num_cpu_cores,
            cores_per_task=args.cores_per_task,
            protein_root=args.protein_root,
        )
        if not ok:
            print(f'❌ {msg}')
            sys.exit(1)
        pt_files_to_eval = find_pt_files_for_range(args.start, args.end)
        if not pt_files_to_eval:
            pt_files_to_eval = pt_files

    elif args.pt_dir:
        pt_dir = Path(args.pt_dir).resolve()
        if not pt_dir.is_dir():
            print(f'❌ --pt_dir 不是目录或不存在: {pt_dir}')
            sys.exit(1)
        if args.vina_outputs_dir is None:
            args.vina_outputs_dir = str(pt_dir)
            print(f'--vina_outputs_dir 未指定，默认使用 --pt_dir: {pt_dir}')
        pt_files_to_eval = find_pt_files_for_range(args.start, args.end, output_dir=pt_dir)
        if not pt_files_to_eval:
            print(f'❌ 在 {pt_dir} 未找到 result_{{{args.start}..{args.end}}}_*.pt')
            sys.exit(1)
        print(f'从 --pt_dir 收集到 {len(pt_files_to_eval)} 个 .pt（data_id {args.start}..{args.end}）')

    elif args.pocket_pdb:
        protein_path = Path(args.pocket_pdb)
        if not protein_path.exists():
            print(f'❌ 口袋文件不存在: {protein_path}')
            sys.exit(1)

        ok, pt_files, msg = run_batch_sampleandeval(
            start=0, end=0,
            gpus=args.gpus,
            num_cpu_cores=args.num_cpu_cores,
            cores_per_task=args.cores_per_task,
            protein_path=protein_path,
            ligand_path=Path(args.ligand_path) if args.ligand_path else None,
            protein_root=args.protein_root,
        )
        if not ok:
            print(f'❌ {msg}')
            sys.exit(1)
        pt_files_to_eval = pt_files

    else:
        pt_path = Path(args.pt_file)
        if not pt_path.exists():
            print(f'❌ .pt 文件不存在: {pt_path}')
            sys.exit(1)
        pt_files_to_eval = [str(pt_path)]
        # 单文件时若未指定 vina_outputs_dir，用 .pt 所在目录
        if args.vina_outputs_dir is None:
            args.vina_outputs_dir = str(pt_path.parent.resolve())

    if not pt_files_to_eval:
        print('❌ 未找到可评估的 .pt 文件')
        sys.exit(1)

    # CPU 并行配置（与 batch_sampleandeval_parallel 一致）
    num_cpu_cores = args.num_cpu_cores if args.num_cpu_cores is not None else min(DEFAULT_NUM_CPU_CORES, cpu_count())
    cores_per_task = max(1, args.cores_per_task)
    max_parallel = max(1, num_cpu_cores // cores_per_task)

    print(f'将评估 {len(pt_files_to_eval)} 个 .pt 文件')
    print(f'CPU 并行: {num_cpu_cores} 核, 每任务 {cores_per_task} 核, 最多 {min(max_parallel, len(pt_files_to_eval))} 个并行任务')

    # 可视化：主文件夹 pocket_quality_vis/，子文件夹命名为 蛋白质编号_时间戳（如 3dzh_20260309_110357）
    base_root = Path(args.vis_dir) if args.vis_dir else VIS_ROOT
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 构建任务列表
    tasks = []
    for pt in pt_files_to_eval:
        stem = Path(pt).stem
        data_id = None
        if stem.startswith('result_'):
            parts = stem.split('_')
            if len(parts) >= 3:
                try:
                    data_id = int(parts[1])
                except ValueError:
                    data_id = parts[1]

        vis_dir = None
        if args.visualize:
            pocket_name = str(data_id) if data_id is not None else stem
            folder_name = f"{pocket_name}_{timestamp}"
            vis_dir = base_root / folder_name
            vis_dir.mkdir(parents=True, exist_ok=True)
            if len(tasks) == 0:
                print(f"Visualization output -> {base_root}")

        tasks.append((
            pt, args.protein_root, data_id, args.atom_mode,
            args.weight_a, args.weight_b, args.weight_c, args.weight_d,
            args.weight_e, args.weight_f, args.weight_g, args.weight_h,
            args.visualize, vis_dir, args.use_tsne,
            args.idea_b_atom_subset,
            args.idea_b_edge_max_neighbors,
            args.idea_b_combined_weight,
            args.vina_outputs_dir,
            args.vina_pocket_id,
            args.custom_pocket_pdb,
            args.fpocket_protein_pdb,
            args.fpocket_cmd,
            args.fpocket_pocket_index,
            args.fpocket_max_ligand_models,
            args.fpocket_timeout,
            args.idea_h_fpocket_vol_min,
            args.idea_h_fpocket_vol_max,
            args.idea_h_fpocket_vol_zero_below,
            args.idea_h_fpocket_vol_zero_above,
            args.idea_h_ligand_path,
            args.idea_e_expected_n_molecules,
        ))

    # 并行或串行执行
    if len(tasks) > 1 and max_parallel > 1:
        n_workers = min(max_parallel, len(tasks))
        try:
            with Pool(processes=n_workers) as pool:
                results = pool.map(_eval_single_pt_task, tasks)
        except KeyboardInterrupt:
            print('\n⚠️  收到中断信号，已停止')
            sys.exit(1)
    else:
        results = [_eval_single_pt_task(t) for t in tasks]

    for result in results:
        try:
            print_evaluation_report(result)
        except Exception as exc:
            print(f'⚠️  print_evaluation_report failed: {exc}', flush=True)
        record_path = base_root / EVAL_RECORDS_CSV
        try:
            append_evaluation_record(result, record_path, timestamp=timestamp)
        except PermissionError as e:
            print(f"⚠️  未写入 {record_path}: {e}", flush=True)
    if results:
        print(f"\nEvaluation records -> {base_root / EVAL_RECORDS_CSV}")


if __name__ == '__main__':
    main()
