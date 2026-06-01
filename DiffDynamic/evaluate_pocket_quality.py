#!/usr/bin/env python3
"""
蛋白质口袋质量评估脚本（增强版 v2）

基于扩散模型（DiffSBDD/DiffDynamic）的分子生成输出，综合八个评估维度对蛋白质口袋（结合位点）进行质量评估，
并提供完整的可视化分析套件。

评估维度：
  想法A：Vina 对接分数与口袋质量
  想法B：原子分布聚类（结合模式收敛性）—— DBSCAN + KMeans；可选按原子子集做 DBSCAN 密度评分
    做法说明（--idea_b_atom_subset）：
      all — 全原子参与 DBSCAN（默认）。
      hetero_heavy — 非 H、非 C 的重原子（极性/药效相关，弱化碳骨架主导）。
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
    将 LE 均值线性映射到 0–1 质量分（默认约 0.12–0.45 kcal·mol⁻¹·重原子⁻¹ 为弱→强参考区间，可调常量）。
  想法D：药物相似性 (QED/SA/Lipinski/PAINS)
  想法E：完整分子比例（SMILES 不含 '.' 的单组分分子数 / 应生成或 .pt 槽位数）
  想法F：分子唯一性与多样性（含指纹 Tanimoto 多样性）
  想法G：分子尺寸一致性
  想法H：口袋体积。默认 **MC**：质心均值 10 Å 球内配体占据，满分区间 400–600 Å³。
    指定 ``--fpocket_protein_pdb`` 且 FPocket 成功：**配体 FPocket Volume 按 400–600（及 100/900）参与 H 分**；
    蛋白 FPocket Volume 作参考；仅当配体侧 FPocket 失败时用蛋白体积 + 宽区间回退评分。

可视化模块：
  - 原子分布聚类图（PCA/t-SNE降维 + DBSCAN/KMeans，6子图）
  - Vina 分数分布图（直方图 + 箱线图 + 小提琴图）
  - 配体效率 LE（每分子：C_le_hist / C_le_box 箱线+抖动散点 / C_le_cdf；与 A 同布局，含 _notext）
  - 药物相似性多维散点图（QED vs SA，属性雷达图）
  - 分子指纹相似性热图（Morgan fingerprint Tanimoto 矩阵）
  - 分子尺寸分布图（分子量 + 原子数双直方图）
  - 综合雷达图（8 维质量指标，含 LE）

使用方法：
    # 使用已有 .pt 文件评估 + 生成可视化图
    python3 evaluate_pocket_quality.py --pt_file outputs/result_custom_20260321_034536.pt --visualize
    # 想法 H 使用 FPocket 分别算蛋白口袋与配体侧口袋体积（需已安装 fpocket，且在 .pt 同目录下用临时副本运行，避免并行冲突）
    python evaluate_pocket_quality.py --pt_file outputs/result_custom_20260319_001000.pt  --fpocket_protein_pdb shoc2/shoc2.pdb --visualize
    # 仅想法 H 的配体 FPocket 使用外部构象（A–G 仍用 .pt 内分子）
    python3 evaluate_pocket_quality.py --pt_file outputs/result_custom_20260319_001000.pt \
  --fpocket_protein_pdb shoc2/shoc2.pdb --idea_h_ligand_path shoc2/shoc2ligand.sdf \
  --idea_e_expected_n_molecules 400 --visualize
    python3 evaluate_pocket_quality.py --pt_file outputs/result_custom_20260319_001000.pt \
      --fpocket_protein_pdb shoc2/shoc2.pdb --idea_h_ligand_path shoc2/shoc2ligand.sdf --visualize
    # 想法 E：.pt 中 pred_ligand_pos 条数不可靠时，用手动应生成数作分母（成功率=解析到的分子数/N）
    python3 evaluate_pocket_quality.py --pt_file outputs/result_custom_20260319_001000.pt \
      --idea_e_expected_n_molecules 1000 --visualize
    # 输出目录结构result_custom_20260319_001000.pt：pocket_quality_vis/蛋白质编号_时间戳/A_vina_hist.png, B_clustering_dbscan.png, ...
    # 口袋评估记录表：pocket_quality_vis/evaluation_records.csv（每次评估追加一行）

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
        get_chem, is_pains, obey_lipinski,
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

BATCH_SCRIPT = REPO_ROOT / 'batch_sampleandeval_parallel.py'
OUTPUT_DIR = REPO_ROOT / 'outputs'

# CPU 并行配置（与 batch_sampleandeval_parallel.py 保持一致）
DEFAULT_NUM_CPU_CORES = 64

# 可视化输出根目录：主文件夹下统一管理，每次实验用时间戳子目录
VIS_ROOT = REPO_ROOT / 'pocket_quality_vis'

# 口袋评估记录表文件名
EVAL_RECORDS_CSV = 'evaluation_records.csv'


def append_evaluation_record(result, record_path, timestamp=None):
    """
    Append one evaluation record to the pocket evaluation log (CSV).

    Columns: pocket_id, timestamp, n_molecules, score_a..h, overall_score, overall_label, pt_path
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
        'overall_score': f"{result.get('overall_score', 0):.4f}",
        'overall_label': result.get('overall_label', 'unknown'),
        'pt_path': result.get('pt_path') or result.get('ligand_path') or '',
    }

    file_exists = record_path.exists()
    with open(record_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


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


def _idea_b_cluster_score_from_n_clusters(n_clusters, max_clusters_for_high):
    if n_clusters <= 1:
        return 1.0
    if n_clusters <= max_clusters_for_high:
        return max(0.0, 1.0 - (n_clusters - 1) / max_clusters_for_high)
    return max(0.0, 0.5 - (n_clusters - max_clusters_for_high) * 0.1)


def _dbscan_metrics_idea_b(X, eps, min_samples, max_clusters_for_high):
    """对 3D 坐标矩阵 X 运行 DBSCAN，返回簇数、噪声、轮廓系数与想法 B 密度收敛分。"""
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    silhouette = None
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
    score = _idea_b_cluster_score_from_n_clusters(n_clusters, max_clusters_for_high)
    return {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': silhouette,
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

        # 质量映射：-12 kcal/mol → 1.0，-6 kcal/mol → 0.0
        raw_score = (-vina_mean - 6) / 6
        score = float(np.clip(raw_score, 0.0, 1.0))

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
            'vina_scores': vina_arr.tolist(),
            'quality_label': label,
            'success': True,
            'message': f'Vina 平均={vina_mean:.2f} kcal/mol, 最佳={vina_best:.2f}, 良好比例={vina_pct_good*100:.1f}%'
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
    eps=2.0,
    min_samples=3,
    max_clusters_for_high=5,
    n_kmeans=5,
    atom_coord_subset='all',
    edge_max_heavy_neighbors=2,
    combined_focus_weight=0.5,
):
    """
    想法B：对生成分子的原子坐标做 DBSCAN（密度/簇数）并结合质心 KMeans。

    atom_coord_subset（DBSCAN 所用 3D 点集）：
        all          — 全部原子（默认，与旧版一致）
        hetero_heavy — 非氢、非碳重原子
        edge_heavy   — 重原子且非氢重邻居数 <= edge_max_heavy_neighbors（近似分子边缘）
        hetero_edge  — 非碳边缘重原子（极性/表面位点，常用于结合模式）
        combined     — (1-w)*全原子 + w*hetero_edge 加权分，w=combined_focus_weight

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

# 将 LE 均值映射到 [0,1] 的参考区间（kcal·mol⁻¹·重原子⁻¹），可按项目口径调整
IDEA_C_LE_SCORE_LOW = 0.12
IDEA_C_LE_SCORE_HIGH = 0.45


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
    仅对 mol 非空且 N_heavy > 0 的条目计算；有效样本数不足时失败。

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
    for i in range(n_aligned):
        mol, _ = molecules_with_pos[i]
        if mol is None:
            continue
        nh = _n_heavy_atoms(mol)
        if nh <= 0:
            continue
        dg = float(v_arr[i])
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
        empty['message'] = '无有效 (分子, Vina) 对可计算 LE（检查 mol 与重原子数）'
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
               lipinski_mean, lipinski_list, pains_ratio, n_valid, quality_label, success}
    """
    if not molecules_with_pos or get_chem is None or is_pains is None or obey_lipinski is None:
        return {
            'score': 0.0, 'qed_mean': None, 'sa_mean': None,
            'lipinski_mean': None, 'pains_ratio': None,
            'qed_list': [], 'sa_list': [], 'lipinski_list': [],
            'quality_label': 'unknown', 'success': False,
            'message': '无分子数据或 scoring_func 未安装'
        }

    qed_list, sa_list, lipinski_list = [], [], []
    pains_hits = 0
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
            n_valid += 1
        except Exception:
            pass

    if n_valid == 0:
        return {
            'score': 0.0, 'qed_mean': None, 'sa_mean': None,
            'lipinski_mean': None, 'pains_ratio': None,
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

    # 综合分数：QED(40%) + SA(30%) + Lipinski(20%) + PAINS惩罚(10%)
    score = (
        0.4 * min(1.0, qed_mean) +
        0.3 * min(1.0, sa_mean) +
        0.2 * lipinski_mean +
        0.1 * (1.0 - pains_ratio)
    )
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
        'n_valid': n_valid,
        'quality_label': label,
        'success': True,
        'message': f'QED={qed_mean:.2f}, SA={sa_mean:.2f}, Lipinski={lipinski_mean:.2f}, PAINS={pains_ratio*100:.1f}%'
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
    """Map yield rate in [0, 1] to quality score and label (same thresholds as legacy E)."""
    if rate_for_score >= 0.9:
        return 1.0, 'high'
    if rate_for_score >= 0.7:
        return 0.5 + (rate_for_score - 0.7) / 0.2 * 0.5, 'medium'
    return rate_for_score / 0.7 * 0.5, 'low'


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
    想法F：分子唯一性/多样性

    新增：计算 Morgan 指纹的平均 Tanimoto 多样性（1 - 相似度）

    Returns:
        dict: {score, unique_ratio, n_unique, n_total,
               tanimoto_diversity, quality_label, success}
    """
    if not molecules_with_pos or Chem is None:
        return {
            'score': 0.0, 'unique_ratio': 0.0, 'n_unique': 0, 'n_total': 0,
            'tanimoto_diversity': None,
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
            'tanimoto_diversity': None,
            'quality_label': 'unknown', 'success': False, 'message': '无有效 SMILES'
        }

    n_unique = len(smiles_set)
    unique_ratio = n_unique / n_complete

    # Tanimoto 多样性（采样计算，避免 O(N^2) 全量）
    tanimoto_diversity = None
    fps_matrix, _ = _compute_morgan_fingerprints(molecules_with_pos)
    if fps_matrix is not None and len(fps_matrix) >= 2 and HAS_SKLEARN:
        n_fps = len(fps_matrix)
        sample_size = min(n_fps, 200)
        idx = np.random.choice(n_fps, sample_size, replace=False) if n_fps > sample_size else np.arange(n_fps)
        fps_sample = fps_matrix[idx]
        # 近似 Tanimoto：用余弦距离
        sim_matrix = (fps_sample @ fps_sample.T) / (
            np.outer(np.linalg.norm(fps_sample, axis=1), np.linalg.norm(fps_sample, axis=1)) + 1e-8
        )
        upper_tri = sim_matrix[np.triu_indices(len(fps_sample), k=1)]
        tanimoto_diversity = float(1.0 - np.mean(upper_tri))

    # 理想唯一性区间 0.3-0.8
    if 0.3 <= unique_ratio <= 0.8:
        score = 1.0
        label = 'high'
    elif 0.15 <= unique_ratio < 0.3 or 0.8 < unique_ratio <= 0.95:
        score = 0.6
        label = 'medium'
    else:
        score = max(0.0, 0.4 - abs(unique_ratio - 0.5) * 0.5)
        label = 'low'

    return {
        'score': min(1.0, score),
        'unique_ratio': unique_ratio,
        'n_unique': n_unique,
        'n_total': n_complete,
        'tanimoto_diversity': tanimoto_diversity,
        'fps_matrix': fps_matrix,
        'quality_label': label,
        'success': True,
        'message': f'唯一性={unique_ratio*100:.1f}% ({n_unique}/{n_complete}), Tanimoto多样性={tanimoto_diversity}'
    }


# =============================================================================
# 想法G：分子尺寸一致性
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

    if mw_cv < 0.2:
        score = 1.0
        label = 'high'
    elif mw_cv < 0.3:
        score = 0.8
        label = 'high'
    elif mw_cv < 0.5:
        score = 0.5
        label = 'medium'
    else:
        score = max(0.0, 0.5 - (mw_cv - 0.5))
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

# 可视化：与文档一致的「配体 / M.C.」参照刻度（400–600 满分，100 / 900 归零尾）
IDEA_H_VIZ_REF_OPT_MIN = 400.0
IDEA_H_VIZ_REF_OPT_MAX = 600.0
IDEA_H_VIZ_REF_ZERO_BELOW = 100.0
IDEA_H_VIZ_REF_ZERO_ABOVE = 900.0
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


def _evaluate_idea_h_mc_only(molecules_with_pos, optimal_min=400, optimal_max=600,
                             zero_below=100.0, zero_above=900.0,
                             centroid_radius=10.0, n_mc_samples=20000):
    """
    想法H（MC）：质心均值点 + 球内 MC，配体 vdW 占据比例 × 球体积。
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
    volume_method = 'mean_centroid_10A_ligand_mc'

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
        'n_centroids': n_centroids,
        'reference_centroid': ref_center.tolist(),
        'single_sphere_ref_volume_ang3': float(v_ball),
        'ligand_occupancy_fraction': float(frac),
        'fpocket_protein_pdb': None,
        'fpocket_protein_volume_ang3': None,
        'fpocket_ligand_volume_ang3': None,
        'fpocket_protein_pocket_meta': None,
        'fpocket_ligand_pocket_meta': None,
        'fpocket_ligand_success': None,
        'fpocket_skipped': False,
        'fpocket_skip_reason': None,
        'fpocket_resolved_executable': None,
        'volume_score_band': 'mc_centroid_10A',
        'message': (
            f'口袋体积≈{volume_ang3:.0f} Å³ (质心均值点 {r:.0f}Å 球内配体占据, n分子={n_centroids}, '
            f'占据率={frac*100:.1f}%, 球体积上限≈{v_ball:.0f} Å³)'
        ),
    }


def evaluate_idea_h_pocket_size(molecules_with_pos, optimal_min=400, optimal_max=600,
                                 zero_below=100.0, zero_above=900.0,
                                 centroid_radius=10.0, n_mc_samples=20000,
                                 fpocket_protein_pdb=None, fpocket_cmd='fpocket',
                                 fpocket_pocket_index=1, fpocket_max_ligand_models=50,
                                 fpocket_timeout=600,
                                 fpocket_optimal_min=None, fpocket_optimal_max=None,
                                 fpocket_zero_below=None, fpocket_zero_above=None):
    """
    想法H：口袋体积。

    - 未指定 ``fpocket_protein_pdb``：沿用 **MC**（质心均值 10 Å 球内配体 vdW 占据），
      满分区间 ``optimal_min``–``optimal_max``（默认 400–600 Å³）。
    - 指定 ``fpocket_protein_pdb`` 且蛋白 FPocket 成功：**优先用配体 FPocket Volume** 按与 MC 相同的
      ``optimal_min``–``optimal_max``（默认 400–600 Å³ 满分，100/900 归零尾）打分；配体 FPocket 失败时
      才用蛋白 Volume + 宽区间回退（``IDEA_H_FPOCKET_PROTEIN_FALLBACK_*``，可由 ``fpocket_optimal_*`` 覆盖）。
    - 找不到 fpocket 可执行文件时 **自动退回 MC**（``fpocket_skipped=True``）。
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


def _safe_savefig(fig, path, dpi=150, save_notext=True):
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


def visualize_clustering_2d(molecules_with_pos, idea_b_result=None,
                             output_dir=None, title_prefix='',
                             n_kmeans=5, use_tsne=False):
    """
    Idea B - Atom distribution clustering (6 figures + optional DBSCAN focus)

    Subplots: DBSCAN / molecule index / KDE / KMeans / CPK / raw X-Y;
    若评估使用非全原子子集且返回 coords_focus，额外保存 B_clustering_dbscan_focus.png
    """
    if not HAS_MATPLOTLIB:
        print("⚠️ visualize_clustering_2d needs matplotlib")
        return {}
    if not HAS_SKLEARN:
        print("⚠️ visualize_clustering_2d needs scikit-learn")
        return {}

    atom_data = (idea_b_result.get('atom_data') if idea_b_result else None) \
                or _collect_atom_data(molecules_with_pos)
    if atom_data is None or len(atom_data['coords']) < 5:
        print("⚠️ Insufficient atom coordinates for clustering plot")
        return {}

    coords    = atom_data['coords']
    mol_idx   = atom_data['mol_idx']
    atom_nums = atom_data['atom_nums']
    centroids = atom_data['centroids']
    n_mols    = atom_data['n_mols']

    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    if use_tsne and len(coords) <= 10000 and TSNE is not None:
        reducer = TSNE(n_components=2, perplexity=min(30, len(coords)-1),
                       random_state=42, n_jobs=-1 if len(coords) > 1000 else 1)
        coords_2d = reducer.fit_transform(coords_scaled)
        ax_xlabel = 't-SNE 1'
        ax_ylabel = 't-SNE 2'
        pca_var   = [None, None]
        pca_approx = PCA(n_components=2)
        pca_approx.fit(coords_scaled)
        centroids_scaled = scaler.transform(centroids)
        centroids_2d = pca_approx.transform(centroids_scaled)
    else:
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(coords_scaled)
        pca_var   = pca.explained_variance_ratio_
        centroids_scaled = scaler.transform(centroids)
        centroids_2d = pca.transform(centroids_scaled)
        ax_xlabel = f'PC1 ({pca_var[0]*100:.1f}%)'
        ax_ylabel = f'PC2 ({pca_var[1]*100:.1f}%)'

    dbscan_labels = np.zeros(len(coords_2d), dtype=int)
    if DBSCAN is not None:
        db = DBSCAN(eps=0.8, min_samples=5).fit(coords_2d)
        dbscan_labels = db.labels_

    kmeans_labels = None
    if idea_b_result and idea_b_result.get('centroid_kmeans_labels'):
        kmeans_labels = np.array(idea_b_result['centroid_kmeans_labels'])
    elif KMeans is not None and len(centroids_2d) >= n_kmeans:
        k = min(n_kmeans, len(centroids_2d))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_labels = km.fit_predict(centroids_2d)

    output_dir = Path(output_dir) if output_dir else None
    if not output_dir:
        return {}
    saved = {}
    n_valid_clusters = len(set(dbscan_labels) - {-1})
    unique_lbls = sorted(set(dbscan_labels))
    n_colors = max(len(unique_lbls), 2)
    palette  = plt.cm.get_cmap('tab20', n_colors)

    def _save(name, fig):
        p = output_dir / name
        _safe_savefig(fig, p)
        saved[name.replace('.png', '')] = str(p)

    # ---- Fig 1: DBSCAN ----
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    for li, lbl in enumerate(unique_lbls):
        mask = dbscan_labels == lbl
        color = '#AAAAAA' if lbl == -1 else palette(li % n_colors)
        alpha = 0.25 if lbl == -1 else 0.55
        s     = 4   if lbl == -1 else 6
        lname = f'Noise ({mask.sum()})' if lbl == -1 else f'Cluster {lbl} ({mask.sum()})'
        ax1.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                    c=[color], s=s, alpha=alpha, label=lname, rasterized=True)
    ax1.set_xlabel(ax_xlabel, fontsize=8)
    ax1.set_ylabel(ax_ylabel, fontsize=8)
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    if len(unique_lbls) <= 12:
        ax1.legend(fontsize=6, loc='best', markerscale=2,
                   framealpha=0.7, ncol=2 if len(unique_lbls) > 8 else 1)
    plt.tight_layout()
    _save('B_clustering_dbscan.png', fig1)

    # ---- Fig 2: Molecule index ----
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    sc2 = ax2.scatter(coords_2d[:, 0], coords_2d[:, 1],
                      c=mol_idx, cmap='viridis', s=5, alpha=0.35, rasterized=True,
                      vmin=0, vmax=n_mols - 1)
    plt.colorbar(sc2, ax=ax2, label='Molecule index', shrink=0.85)
    ax2.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                c='red', s=50, marker='*', zorder=6, alpha=0.9,
                label=f'Centroids (n={len(centroids_2d)})',
                edgecolors='white', linewidths=0.4)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.set_xlabel(ax_xlabel, fontsize=8)
    ax2.set_ylabel(ax_ylabel, fontsize=8)
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    _save('B_clustering_molindex.png', fig2)

    # ---- Fig 3: KDE ----
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.set_facecolor('#0d0d1a')
    try:
        x1, x2 = coords_2d[:, 0].min(), coords_2d[:, 0].max()
        y1, y2 = coords_2d[:, 1].min(), coords_2d[:, 1].max()
        n_kde_sample = min(len(coords_2d), 8000)
        idx_kde = np.random.choice(len(coords_2d), n_kde_sample, replace=False) \
                  if len(coords_2d) > n_kde_sample else np.arange(len(coords_2d))
        if HAS_SCIPY and gaussian_kde is not None:
            kde = gaussian_kde(coords_2d[idx_kde].T, bw_method=0.12)
            xx, yy = np.mgrid[x1:x2:80j, y1:y2:80j]
            z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            ax3.contourf(xx, yy, z, levels=20, cmap='inferno', alpha=0.9)
            ax3.contour(xx, yy, z, levels=8, colors='white', alpha=0.25, linewidths=0.5)
        else:
            hb = ax3.hexbin(coords_2d[idx_kde, 0], coords_2d[idx_kde, 1],
                             gridsize=35, cmap='inferno', mincnt=1)
            plt.colorbar(hb, ax=ax3, shrink=0.85, label='Atom count')
        ax3.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                    c='cyan', s=30, marker='^', zorder=5, alpha=0.85,
                    label='Centroids', edgecolors='white', linewidths=0.4)
        ax3.legend(fontsize=8, loc='upper right')
    except Exception as exc:
        ax3.text(0.5, 0.5, f'Density plot failed\n{exc}', ha='center', va='center',
                 transform=ax3.transAxes, color='white', fontsize=9)
    ax3.set_xlabel(ax_xlabel, fontsize=8, color='white')
    ax3.set_ylabel(ax_ylabel, fontsize=8, color='white')
    ax3.tick_params(colors='white')
    plt.tight_layout()
    _save('B_clustering_kde.png', fig3)

    # ---- Fig 4: KMeans ----
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    if kmeans_labels is not None:
        n_km = len(set(kmeans_labels))
        pal_km = plt.cm.get_cmap('Set1', n_km)
        for ki in range(n_km):
            mk = kmeans_labels == ki
            ax4.scatter(centroids_2d[mk, 0], centroids_2d[mk, 1],
                        c=[pal_km(ki)], s=80, marker='o',
                        label=f'Mode {ki} ({mk.sum()})',
                        edgecolors='white', linewidths=0.8, zorder=4)
        ax4.legend(fontsize=7, loc='best', ncol=2 if n_km > 5 else 1)
    else:
        ax4.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                    c=range(len(centroids_2d)), cmap='Spectral', s=70,
                    edgecolors='gray', linewidths=0.5)
    ax4.set_xlabel('PC1', fontsize=8)
    ax4.set_ylabel('PC2', fontsize=8)
    ax4.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    _save('B_clustering_kmeans.png', fig4)

    # ---- Fig 5: CPK（白底，与其它聚类子图一致）----
    fig5, ax5 = plt.subplots(figsize=(7, 5))
    fig5.patch.set_facecolor('white')
    ax5.set_facecolor('white')
    unique_atypes = sorted(set(atom_nums.tolist()))
    for anum in unique_atypes:
        mask_a = atom_nums == anum
        color_a = _CPK_COLORS.get(anum, '#888888')
        name_a  = _ELEMENT_NAMES.get(anum, f'Z={anum}')
        ax5.scatter(coords_2d[mask_a, 0], coords_2d[mask_a, 1],
                    c=color_a, s=7, alpha=0.55,
                    label=f'{name_a} ({mask_a.sum()})', rasterized=True)
    ax5.set_xlabel(ax_xlabel, fontsize=8, color='black')
    ax5.set_ylabel(ax_ylabel, fontsize=8, color='black')
    ax5.tick_params(colors='black')
    ax5.grid(True, alpha=0.25, linewidth=0.5, color='0.75')
    ax5.legend(fontsize=6, loc='best', markerscale=2,
               framealpha=0.92, facecolor='white', edgecolor='0.7',
               labelcolor='black',
               ncol=2 if len(unique_atypes) > 6 else 1)
    plt.tight_layout()
    _save('B_clustering_cpk.png', fig5)

    # ---- Fig 6: Raw X-Y ----
    fig6, ax6 = plt.subplots(figsize=(7, 5))
    sc6 = ax6.scatter(coords[:, 0], coords[:, 1],
                      c=mol_idx, cmap='plasma', s=4, alpha=0.2,
                      rasterized=True, vmin=0, vmax=n_mols - 1)
    plt.colorbar(sc6, ax=ax6, label='Molecule index', shrink=0.85)
    ax6.scatter(centroids[:, 0], centroids[:, 1],
                c='orange', s=30, marker='D', alpha=0.8, zorder=5,
                label='Centroids', edgecolors='white', linewidths=0.4)
    ax6.legend(fontsize=8)
    ax6.set_xlabel('X (Angstrom)', fontsize=8)
    ax6.set_ylabel('Y (Angstrom)', fontsize=8)
    ax6.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    _save('B_clustering_xy.png', fig6)

    # ---- Fig 7 (optional): 与评估一致的 3D DBSCAN，仅非碳边缘重原子等子集 ----
    if idea_b_result and DBSCAN is not None:
        Xf = idea_b_result.get('coords_focus')
        if Xf is not None:
            Xf = np.asarray(Xf, dtype=np.float64)
            if len(Xf) >= 5:
                eps3 = float(idea_b_result.get('dbscan_eps', 2.0))
                ms3 = int(idea_b_result.get('dbscan_min_samples', 3))
                sub_name = str(idea_b_result.get('atom_coord_subset', ''))
                db3 = DBSCAN(eps=eps3, min_samples=ms3).fit(Xf)
                lbl3 = db3.labels_
                pca_f = PCA(n_components=2, random_state=42)
                Xf_2d = pca_f.fit_transform(Xf)
                fig7, ax7 = plt.subplots(figsize=(7, 5))
                uq = sorted(set(lbl3.tolist()))
                n_c7 = max(len(uq), 2)
                pal7 = plt.cm.get_cmap('tab20', n_c7)
                for j, lb in enumerate(uq):
                    mk = lbl3 == lb
                    c = '#AAAAAA' if lb == -1 else pal7(j % n_c7)
                    ax7.scatter(
                        Xf_2d[mk, 0], Xf_2d[mk, 1], c=[c], s=8, alpha=0.55, rasterized=True,
                        label=('noise' if lb == -1 else f'c{lb}') + f' ({mk.sum()})',
                    )
                ax7.set_title(
                    f'DBSCAN 3D→PCA2D | subset={sub_name} | eps={eps3}, min_s={ms3}',
                    fontsize=9,
                )
                ax7.set_xlabel('PC1', fontsize=8)
                ax7.set_ylabel('PC2', fontsize=8)
                ax7.grid(True, alpha=0.3, linewidth=0.5)
                if len(uq) <= 14:
                    ax7.legend(fontsize=6, loc='best', ncol=2, framealpha=0.7)
                plt.tight_layout()
                _save('B_clustering_dbscan_focus.png', fig7)

    return saved


def visualize_vina_distribution(idea_a_result, output_dir=None, title_prefix=''):
    """
    Idea A - Vina score distribution (3 separate figures)

    Subplots: hist+KDE / box+scatter / CDF
    """
    if not HAS_MATPLOTLIB:
        return {}
    vina_scores = idea_a_result.get('vina_scores', [])
    if not vina_scores or not idea_a_result.get('success'):
        return {}
    scores = np.array(vina_scores, dtype=np.float64)
    output_dir = Path(output_dir) if output_dir else None
    if not output_dir:
        return {}
    saved = {}

    # ---- Fig 1: Histogram + KDE ----
    fig1, ax = plt.subplots(figsize=(6, 4))
    bins = max(10, min(50, len(scores) // 3))
    n, bin_edges, patches = ax.hist(scores, bins=bins, color='steelblue',
                                     edgecolor='white', alpha=0.75, density=True, label='Density')
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        norm_val = np.clip((-left_edge - 6) / 6, 0, 1)
        patch.set_facecolor(plt.cm.RdYlGn(norm_val))
    if HAS_SCIPY and gaussian_kde is not None and len(scores) > 3:
        kde = gaussian_kde(scores, bw_method=0.3)
        x_line = np.linspace(scores.min() - 0.5, scores.max() + 0.5, 300)
        ax.plot(x_line, kde(x_line), 'k-', lw=2, label='KDE')
    ax.axvline(-7.0, color='red', linestyle='--', lw=1.5, label='Threshold -7 kcal/mol')
    ax.axvline(float(np.mean(scores)), color='orange', linestyle='-', lw=1.5,
               label=f'Mean {np.mean(scores):.2f}')
    ax.set_xlabel('Vina score (kcal/mol)', fontsize=9)
    ax.set_ylabel('Probability density', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p1 = output_dir / 'A_vina_hist.png'
    _safe_savefig(fig1, p1)
    saved['A_vina_hist'] = str(p1)

    # ---- Fig 2: Box + strip scatter（点阵）；colorbar 用 axes_grid1 与主图解耦，避免去字/tight 挤坏）
    fig2, ax = plt.subplots(figsize=(5.8, 4.2))
    rng = np.random.default_rng(42)
    ax.boxplot(
        scores,
        vert=True,
        patch_artist=True,
        widths=0.35,
        positions=[1.0],
        boxprops=dict(facecolor='lightsteelblue', alpha=0.7),
        medianprops=dict(color='red', lw=2),
        whiskerprops=dict(color='gray'),
        capprops=dict(color='gray'),
    )
    jitter = rng.uniform(-0.14, 0.14, len(scores))
    sc = ax.scatter(
        1.0 + jitter,
        scores,
        c=scores,
        cmap='RdYlGn_r',
        s=22,
        alpha=0.65,
        zorder=3,
        vmin=scores.min(),
        vmax=scores.max(),
        edgecolors='none',
    )
    try:
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3.8%', pad=0.12)
        cb = fig2.colorbar(sc, cax=cax)
        cb.set_label('Vina (kcal/mol)', fontsize=8)
        cb.ax.tick_params(labelsize=7)
    except Exception:
        plt.colorbar(sc, ax=ax, label='Vina score (kcal/mol)', shrink=0.82, pad=0.02)
    ax.axhline(-7.0, color='red', linestyle='--', lw=1.2, label='Threshold -7')
    ax.set_xticks([1.0])
    ax.set_xticklabels(['Generated mols'], fontsize=8)
    ax.set_xlim(0.55, 1.45)
    y_pad = max(0.6, float(np.ptp(scores)) * 0.08 + 0.1)
    ax.set_ylim(float(scores.min()) - y_pad, float(scores.max()) + y_pad)
    ax.set_ylabel('Vina score (kcal/mol)', fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    p2 = output_dir / 'A_vina_box.png'
    _safe_savefig(fig2, p2)
    saved['A_vina_box'] = str(p2)

    # ---- Fig 3: CDF ----
    fig3, ax = plt.subplots(figsize=(6, 4))
    sorted_scores = np.sort(scores)
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax.plot(sorted_scores, cdf, 'b-', lw=2, label='CDF')
    ax.fill_between(sorted_scores, 0, cdf, alpha=0.2, color='steelblue')
    ax.axvline(-7.0, color='red', linestyle='--', lw=1.5,
               label=f'P(<=-7) = {np.mean(scores <= -7)*100:.1f}%')
    ax.axvline(-9.0, color='orange', linestyle=':', lw=1.2, label='-9 kcal/mol')
    ax.set_xlabel('Vina score (kcal/mol)', fontsize=9)
    ax.set_ylabel('Cumulative probability', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    p3 = output_dir / 'A_vina_cdf.png'
    _safe_savefig(fig3, p3)
    saved['A_vina_cdf'] = str(p3)
    return saved


def visualize_le_distribution(idea_c_result, output_dir=None, title_prefix=''):
    """
    Idea C - Per-molecule ligand efficiency LE distribution (mirrors Idea A layout).

    Figures: histogram + KDE / box + jittered strip scatter / CDF.
    Each file is saved with _safe_savefig (text + _notext variants).
    """
    if not HAS_MATPLOTLIB:
        return {}
    le_list = idea_c_result.get('le_list') or []
    if not le_list or not idea_c_result.get('success'):
        return {}
    scores = np.array(le_list, dtype=np.float64)
    lo = float(idea_c_result.get('le_low_for_score', IDEA_C_LE_SCORE_LOW))
    hi = float(idea_c_result.get('le_high_for_score', IDEA_C_LE_SCORE_HIGH))
    output_dir = Path(output_dir) if output_dir else None
    if not output_dir:
        return {}
    saved = {}
    le_unit = 'LE (kcal/mol per heavy atom)'

    # ---- Fig 1: Histogram + KDE ----
    fig1, ax = plt.subplots(figsize=(6, 4))
    bins = max(10, min(50, len(scores) // 3))
    n, bin_edges, patches = ax.hist(
        scores, bins=bins, color='steelblue',
        edgecolor='white', alpha=0.75, density=True, label='Density',
    )
    span = max(hi - lo, 1e-9)
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        norm_val = np.clip((float(left_edge) - lo) / span, 0, 1)
        patch.set_facecolor(plt.cm.RdYlGn(norm_val))
    if HAS_SCIPY and gaussian_kde is not None and len(scores) > 3 and float(np.ptp(scores)) > 1e-12:
        try:
            kde = gaussian_kde(scores, bw_method=0.2)
            pad = max(0.02, float(np.ptp(scores)) * 0.08)
            x_line = np.linspace(scores.min() - pad, scores.max() + pad, 300)
            ax.plot(x_line, kde(x_line), 'k-', lw=2, label='KDE')
        except (np.linalg.LinAlgError, ValueError):
            pass
    ax.axvline(lo, color='red', linestyle='--', lw=1.5, label=f'Score map low {lo:.3f}')
    ax.axvline(hi, color='green', linestyle='--', lw=1.5, label=f'Score map high {hi:.3f}')
    ax.axvline(float(np.mean(scores)), color='orange', linestyle='-', lw=1.5,
               label=f'Mean {np.mean(scores):.4f}')
    ax.set_xlabel(le_unit, fontsize=9)
    ax.set_ylabel('Probability density', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p1 = output_dir / 'C_le_hist.png'
    _safe_savefig(fig1, p1)
    saved['C_le_hist'] = str(p1)

    # ---- Fig 2: Box + strip scatter (same style as A_vina_box) ----
    fig2, ax = plt.subplots(figsize=(5.8, 4.2))
    rng = np.random.default_rng(43)
    ax.boxplot(
        scores,
        vert=True,
        patch_artist=True,
        widths=0.35,
        positions=[1.0],
        boxprops=dict(facecolor='lightsteelblue', alpha=0.7),
        medianprops=dict(color='red', lw=2),
        whiskerprops=dict(color='gray'),
        capprops=dict(color='gray'),
    )
    jitter = rng.uniform(-0.14, 0.14, len(scores))
    sc = ax.scatter(
        1.0 + jitter,
        scores,
        c=scores,
        cmap='RdYlGn',
        s=22,
        alpha=0.65,
        zorder=3,
        vmin=lo,
        vmax=hi,
        edgecolors='none',
    )
    try:
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3.8%', pad=0.12)
        cb = fig2.colorbar(sc, cax=cax)
        cb.set_label(le_unit, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    except Exception:
        plt.colorbar(sc, ax=ax, label=le_unit, shrink=0.82, pad=0.02)
    ax.axhline(lo, color='red', linestyle='--', lw=1.2, label=f'Map low {lo:.3f}')
    ax.axhline(hi, color='green', linestyle='--', lw=1.2, label=f'Map high {hi:.3f}')
    ax.set_xticks([1.0])
    ax.set_xticklabels(['Generated mols'], fontsize=8)
    ax.set_xlim(0.55, 1.45)
    y_pad = max(0.02, float(np.ptp(scores)) * 0.08 + 0.01)
    ax.set_ylim(float(scores.min()) - y_pad, float(scores.max()) + y_pad)
    ax.set_ylabel(le_unit, fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    p2 = output_dir / 'C_le_box.png'
    _safe_savefig(fig2, p2)
    saved['C_le_box'] = str(p2)

    # ---- Fig 3: CDF ----
    fig3, ax = plt.subplots(figsize=(6, 4))
    sorted_scores = np.sort(scores)
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax.plot(sorted_scores, cdf, 'b-', lw=2, label='CDF')
    ax.fill_between(sorted_scores, 0, cdf, alpha=0.2, color='steelblue')
    ax.axvline(lo, color='red', linestyle='--', lw=1.5,
               label=f'P(LE≤low) = {np.mean(scores <= lo)*100:.1f}%')
    ax.axvline(hi, color='green', linestyle='--', lw=1.5,
               label=f'P(LE≥high) = {np.mean(scores >= hi)*100:.1f}%')
    ax.axvline(float(np.median(scores)), color='orange', linestyle=':', lw=1.2,
               label=f'Median {np.median(scores):.4f}')
    ax.set_xlabel(le_unit, fontsize=9)
    ax.set_ylabel('Cumulative probability', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    p3 = output_dir / 'C_le_cdf.png'
    _safe_savefig(fig3, p3)
    saved['C_le_cdf'] = str(p3)
    return saved


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
    vina_arr = np.array(vina_scores, dtype=np.float64) if vina_scores and len(vina_scores) > 0 else None

    # ---- Fig 1: QED vs SA scatter (color=vinadock) ----
    n_pts = min(len(qed_list), len(sa_list))
    # Color by vinadock when available, else fallback to Lipinski
    if vina_arr is not None and len(vina_arr) >= n_pts:
        c_vals = vina_arr[:n_pts]
        c_label = 'Vina score (kcal/mol)'
        c_vmin, c_vmax = c_vals.min(), c_vals.max()
        c_cmap = 'RdYlGn_r'  # more negative (better) -> green
    else:
        c_vals = lip_list[:n_pts] if len(lip_list) >= n_pts else np.ones(n_pts)
        c_label = 'Lipinski compliance (0-1)'
        c_vmin, c_vmax = 0, 1
        c_cmap = 'RdYlGn'
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    sc = ax1.scatter(qed_list[:n_pts], sa_list[:n_pts],
                     c=c_vals, cmap=c_cmap,
                     s=40, alpha=0.7, edgecolors='gray', linewidths=0.3,
                     vmin=c_vmin, vmax=c_vmax)
    plt.colorbar(sc, ax=ax1, label=c_label, shrink=0.85)
    ax1.axvline(0.5, color='gray', linestyle='--', lw=1, alpha=0.6, label='QED=0.5')
    ax1.axhline(0.5, color='gray', linestyle='--', lw=1, alpha=0.6, label='SA=0.5')
    from matplotlib.patches import Rectangle
    rect = Rectangle((0.5, 0.5), 0.5, 0.5, linewidth=1.5, edgecolor='green',
                      facecolor='green', alpha=0.07, label='Good zone (QED>0.5, SA>0.5)')
    ax1.add_patch(rect)
    qed_m = idea_d_result.get('qed_mean', 0)
    sa_m = idea_d_result.get('sa_mean', 0)
    ax1.scatter([qed_m], [sa_m], c='red', s=150, marker='*', zorder=10,
                label=f'Mean ({qed_m:.2f}, {sa_m:.2f})')
    ax1.set_xlabel('QED (druglikeness, higher better)', fontsize=9)
    ax1.set_ylabel('SA (synthetic accessibility, higher better)', fontsize=9)
    ax1.legend(fontsize=7, loc='upper left')
    ax1.set_xlim(0, 1.02)
    ax1.set_ylim(0, 1.02)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    _safe_savefig(fig1, output_dir / 'D_druglikeness_qed_vs_sa.png')
    saved['D_druglikeness_qed_vs_sa'] = str(output_dir / 'D_druglikeness_qed_vs_sa.png')

    # ---- Fig 2: QED distribution ----
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.hist(qed_list, bins=20, color='steelblue', edgecolor='white',
             alpha=0.8, density=True)
    if HAS_SCIPY and gaussian_kde is not None and len(qed_list) > 3:
        kde = gaussian_kde(qed_list, bw_method=0.25)
        xq = np.linspace(0, 1, 200)
        ax2.plot(xq, kde(xq), 'r-', lw=2)
    ax2.axvline(float(np.mean(qed_list)), color='orange', lw=1.5,
                linestyle='--', label=f'Mean={np.mean(qed_list):.3f}')
    ax2.set_xlabel('QED', fontsize=9)
    ax2.set_ylabel('Density', fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    _safe_savefig(fig2, output_dir / 'D_druglikeness_qed_dist.png')
    saved['D_druglikeness_qed_dist'] = str(output_dir / 'D_druglikeness_qed_dist.png')

    # ---- Fig 3: SA distribution ----
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    ax3.hist(sa_list, bins=20, color='mediumseagreen', edgecolor='white',
             alpha=0.8, density=True)
    if HAS_SCIPY and gaussian_kde is not None and len(sa_list) > 3:
        kde = gaussian_kde(sa_list, bw_method=0.25)
        xs = np.linspace(0, 1, 200)
        ax3.plot(xs, kde(xs), 'r-', lw=2)
    ax3.axvline(float(np.mean(sa_list)), color='orange', lw=1.5,
                linestyle='--', label=f'Mean={np.mean(sa_list):.3f}')
    ax3.set_xlabel('SA (synthetic accessibility)', fontsize=9)
    ax3.set_ylabel('Density', fontsize=9)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    _safe_savefig(fig3, output_dir / 'D_druglikeness_sa_dist.png')
    saved['D_druglikeness_sa_dist'] = str(output_dir / 'D_druglikeness_sa_dist.png')

    # ---- Fig 4: Lipinski compliance ----
    fig4, ax4 = plt.subplots(figsize=(5, 4))
    if len(lip_list) > 0:
        ax4.hist(lip_list * 5, bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                 color='coral', edgecolor='white', alpha=0.85)
        ax4.set_xlabel('Lipinski rules passed', fontsize=9)
        ax4.set_ylabel('Molecule count', fontsize=9)
        ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.axvline(4, color='green', lw=1.5, linestyle='--', label='>=4 compliant')
        ax4.legend(fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'No Lipinski data', ha='center', va='center',
                 transform=ax4.transAxes, fontsize=10)
    plt.tight_layout()
    _safe_savefig(fig4, output_dir / 'D_druglikeness_lipinski.png')
    saved['D_druglikeness_lipinski'] = str(output_dir / 'D_druglikeness_lipinski.png')

    # ---- Fig 5: Drug property radar ----
    fig5 = plt.figure(figsize=(6, 5))
    ax5 = fig5.add_subplot(111, polar=True)
    categories = ['QED', 'SA', 'Lipinski\ncompliant', 'PAINS\nfree', 'Overall']
    qed_m   = idea_d_result.get('qed_mean', 0) or 0
    sa_m    = idea_d_result.get('sa_mean',  0) or 0
    lip_m   = idea_d_result.get('lipinski_mean', 0) or 0
    pains_r = idea_d_result.get('pains_ratio', 0) or 0
    overall = idea_d_result.get('score', 0) or 0
    values  = [qed_m, sa_m, lip_m, 1.0 - pains_r, overall]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]
    ax5.set_facecolor('#f8f9fa')
    ax5.plot(angles, values, 'o-', lw=2, color='steelblue')
    ax5.fill(angles, values, alpha=0.25, color='steelblue')
    ax5.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=8)
    ax5.set_ylim(0, 1)
    ax5.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax5.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=6)
    ax5.grid(True, alpha=0.4)
    plt.tight_layout()
    _safe_savefig(fig5, output_dir / 'D_druglikeness_radar.png')
    saved['D_druglikeness_radar'] = str(output_dir / 'D_druglikeness_radar.png')
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
        stats_vals = [n_unique, n_total - n_unique]
        colors_pie = ['#4CAF50', '#FF5722']
        wedges, texts, autotexts = ax4.pie(
            stats_vals, labels=stats_labels, colors=colors_pie,
            autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=1.5),
        )
        for at in autotexts:
            at.set_fontsize(10)
    else:
        ax4.text(0.5, 0.5, 'No uniqueness data', ha='center', va='center',
                 transform=ax4.transAxes, fontsize=10)
    plt.tight_layout()
    _safe_savefig(fig4, output_dir / 'F_fingerprint_uniqueness.png')
    saved['F_fingerprint_uniqueness'] = str(output_dir / 'F_fingerprint_uniqueness.png')
    return saved


def visualize_size_distribution(idea_g_result, output_dir=None, title_prefix=''):
    """
    Idea G - Molecular size distribution (3 separate figures)

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
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _safe_savefig(fig1, output_dir / 'G_size_mw.png')
    saved['G_size_mw'] = str(output_dir / 'G_size_mw.png')

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
    _safe_savefig(fig2, output_dir / 'G_size_atoms.png')
    saved['G_size_atoms'] = str(output_dir / 'G_size_atoms.png')

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
    _safe_savefig(fig3, output_dir / 'G_size_mw_vs_atoms.png')
    saved['G_size_mw_vs_atoms'] = str(output_dir / 'G_size_mw_vs_atoms.png')
    return saved


def visualize_pocket_volume(idea_h_result, output_dir=None, title_prefix=''):
    """
    Idea H - Pocket volume visualization

    X-axis default limit 1500 Å³ (extends slightly if scored volume exceeds it). Background: 0-100 and
    900-1500 use the same red zero-reference tint; 100-400 and 600-900 use light orange; 400-600 stays
    green full-score. Protein FPocket fallback highlights use the same light orange (not a large
    saturated block).
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
    saved = {}
    score = idea_h_result.get('score', 0)
    vol = float(vol)

    rmin, rmax = IDEA_H_VIZ_REF_OPT_MIN, IDEA_H_VIZ_REF_OPT_MAX
    rzb, rza = IDEA_H_VIZ_REF_ZERO_BELOW, IDEA_H_VIZ_REF_ZERO_ABOVE
    band = idea_h_result.get('volume_score_band') or ''
    prot_fb = band == 'fpocket_protein_fallback'

    # 背景区：0–100 与 900–1500 为零分参照（同色系）；100–400、600–900 为淡橙；400–600 满分绿。
    # 横轴默认上限 1500，避免右侧珊瑚色块随体积无限拉长。
    vmin, vmax, zb, za = rmin, rmax, rzb, rza

    fig, ax = plt.subplots(figsize=(7, 4))
    x_min = 0.0
    x_cap = float(IDEA_H_VIZ_X_MAX_DEFAULT)
    x_max = x_cap
    if vol > x_cap:
        x_max = float(vol) * 1.06

    red_alpha = 0.11
    if rzb > 0:
        ax.axvspan(x_min, min(rzb, x_max), alpha=red_alpha, color='red',
                   label=f'Zero score: V < {rzb:.0f} (reference)')
    # 100–400、600–900：淡橙（参照区间）
    if rzb < vmin:
        ax.axvspan(
            max(x_min, rzb), min(vmin, x_max),
            alpha=IDEA_H_VIZ_SUBOPT_ALPHA, color=IDEA_H_VIZ_SUBOPT_ORANGE,
            label='Sub-optimal (reference)',
        )
    if vmax < rza:
        ax.axvspan(
            max(x_min, vmax), min(rza, x_max),
            alpha=IDEA_H_VIZ_SUBOPT_ALPHA, color=IDEA_H_VIZ_SUBOPT_ORANGE,
        )
    ax.axvspan(vmin, vmax, alpha=0.22, color='green',
               label=f'Full-score band {vmin:.0f}-{vmax:.0f} Å³ (ligand / M.C.)')
    if rza < x_max:
        ax.axvspan(rza, x_max, alpha=red_alpha, color='red',
                   label=f'Zero score: V ≥ {rza:.0f} (reference)')
    ax.axvline(vmin, color='green', linestyle='--', lw=1.1, alpha=0.65)
    ax.axvline(vmax, color='green', linestyle='--', lw=1.1, alpha=0.65)

    if prot_fb:
        act_lo = float(idea_h_result.get('optimal_vol_min', IDEA_H_FPOCKET_PROTEIN_FALLBACK_OPTIMAL_MIN))
        act_hi = float(idea_h_result.get('optimal_vol_max', IDEA_H_FPOCKET_PROTEIN_FALLBACK_OPTIMAL_MAX))
        a_zb = float(idea_h_result.get('vol_score_zero_below', IDEA_H_FPOCKET_PROTEIN_FALLBACK_ZERO_BELOW))
        a_za = float(idea_h_result.get('vol_score_zero_above', IDEA_H_FPOCKET_PROTEIN_FALLBACK_ZERO_ABOVE))
        s0, s1 = min(act_lo, act_hi), max(act_lo, act_hi)
        span_lo = max(x_min, min(s0, x_max))
        span_hi = min(max(s1, span_lo), x_max)
        if span_hi > span_lo:
            ax.axvspan(
                span_lo, span_hi,
                alpha=0.12, color=IDEA_H_VIZ_SUBOPT_ORANGE,
                label=f'H full-score band (protein) {act_lo:.0f}-{act_hi:.0f}',
            )
        note = (
            f'Axis reference: ligand/M.C. 400-600 / tails 100-900; '
            f'scoring used protein volume, band [{act_lo:.0f}, {act_hi:.0f}], '
            f'zero if V<{a_zb:.0f} or V≥{a_za:.0f}'
        )
        ax.text(0.02, 0.88, note, transform=ax.transAxes, fontsize=7, va='top', color='#444444')

    ax.barh(0, vol, left=0, height=0.38, color='steelblue', alpha=0.86,
            label=f'Scoring volume V = {vol:.0f} Å³')
    ax.axvline(vol, color='navy', linestyle='-', lw=1.8, alpha=0.88)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.52, 0.52)
    ax.set_xlabel('Volume (Å³)', fontsize=9)
    ax.set_yticks([])
    ax.legend(fontsize=6.5, loc='upper right')
    ax.grid(True, alpha=0.28, axis='x')
    ax.text(0.02, 0.98, f'Score (H): {score:.3f}', transform=ax.transAxes,
            fontsize=10, va='top', fontweight='bold')
    if title_prefix:
        ax.set_title(f'{title_prefix} pocket volume', fontsize=10)
    plt.tight_layout()
    _safe_savefig(fig, output_dir / 'H_pocket_volume.png')
    saved['H_pocket_volume'] = str(output_dir / 'H_pocket_volume.png')
    return saved


def visualize_radar_summary(result, output_dir=None, title_prefix=''):
    """
    Summary radar - 8-axis quality spider chart (A–H, including C ligand efficiency LE).

    Subplots: radar / bar chart
    """
    if not HAS_MATPLOTLIB:
        return {}
    idea_keys = ['idea_a', 'idea_b', 'idea_c', 'idea_d', 'idea_e', 'idea_f', 'idea_g', 'idea_h']
    idea_labels = ['A Vina', 'B Cluster', 'C LE', 'D Drug', 'E Recon', 'F Divers', 'G Size', 'H Pocket']
    scores = []
    for k in idea_keys:
        r = result.get(k, {})
        scores.append(r.get('score', 0.0) if r.get('success') else 0.0)
    overall = result.get('overall_score', 0.0)
    output_dir = Path(output_dir) if output_dir else None
    if not output_dir:
        return {}
    saved = {}

    # ---- Fig 1: Radar ----
    fig1, ax_radar = plt.subplots(figsize=(7, 6), subplot_kw=dict(polar=True))
    N = len(idea_labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals = scores + scores[:1]
    ax_radar.set_facecolor('#f4f4f8')
    for r_thresh, col, alpha in [(0.6, 'green', 0.07), (0.3, 'orange', 0.07)]:
        ax_radar.fill(angles, [r_thresh] * (N + 1), color=col, alpha=alpha)
    ax_radar.plot(angles, vals, 'o-', lw=2.2, color='steelblue', zorder=5)
    ax_radar.fill(angles, vals, alpha=0.30, color='steelblue')
    ax_radar.set_thetagrids(np.degrees(angles[:-1]), idea_labels, fontsize=9)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=6, color='gray')
    ax_radar.grid(True, alpha=0.4)
    for angle, val, label in zip(angles[:-1], scores, idea_labels):
        ax_radar.annotate(f'{val:.2f}',
                          xy=(angle, val),
                          xytext=(angle, val + 0.08),
                          fontsize=7, ha='center', va='bottom', color='navy',
                          fontweight='bold')
    plt.tight_layout()
    _safe_savefig(fig1, output_dir / 'summary_radar.png')
    saved['summary_radar'] = str(output_dir / 'summary_radar.png')

    # ---- Fig 2: Bar chart ----
    fig2, ax_bar = plt.subplots(figsize=(7, 5))
    colors_bar = []
    for s in scores:
        if s >= 0.6:
            colors_bar.append('#4CAF50')
        elif s >= 0.3:
            colors_bar.append('#FF9800')
        else:
            colors_bar.append('#F44336')
    bars = ax_bar.barh(idea_labels, scores, color=colors_bar,
                       edgecolor='white', height=0.6)
    ax_bar.axvline(0.6, color='green', linestyle='--', lw=1.2, alpha=0.7, label='High threshold 0.6')
    ax_bar.axvline(0.3, color='orange', linestyle='--', lw=1.2, alpha=0.7, label='Med threshold 0.3')
    ax_bar.axvline(overall, color='navy', linestyle='-', lw=2.0, alpha=0.9,
                   label=f'Overall {overall:.3f}')
    for bar, val in zip(bars, scores):
        ax_bar.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    ax_bar.set_xlim(0, 1.15)
    ax_bar.set_xlabel('Quality score (0-1)', fontsize=9)
    ax_bar.legend(fontsize=7, loc='lower right')
    ax_bar.grid(True, alpha=0.3, axis='x')
    ax_bar.axvspan(0.6, 1.15, alpha=0.05, color='green')
    ax_bar.axvspan(0.3, 0.6,  alpha=0.05, color='orange')
    ax_bar.axvspan(0.0, 0.3,  alpha=0.05, color='red')
    plt.tight_layout()
    _safe_savefig(fig2, output_dir / 'summary_bar.png')
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

    # 2. Vina distribution (Idea A)
    r = visualize_vina_distribution(
        result.get('idea_a', {}),
        output_dir=output_dir,
        title_prefix=title_prefix,
    )
    saved.update(r)

    # 2b. Ligand efficiency LE per molecule (Idea C), same layout as Vina (hist / box+scatter / CDF)
    r = visualize_le_distribution(
        result.get('idea_c', {}),
        output_dir=output_dir,
        title_prefix=title_prefix,
    )
    saved.update(r)

    # 3. Druglikeness (Idea D) - pass vina_scores for QED vs vinadock y-axis
    idea_a = result.get('idea_a', {})
    vina_scores = idea_a.get('vina_scores') if idea_a.get('success') else None
    r = visualize_druglikeness(
        result.get('idea_d', {}),
        output_dir=output_dir,
        title_prefix=title_prefix,
        vina_scores=vina_scores,
    )
    saved.update(r)

    # 4. Fingerprint diversity (Idea F)
    idea_d = result.get('idea_d', {})
    qed_list = idea_d.get('qed_list') if idea_d.get('success') else None
    r = visualize_fingerprint_diversity(
        molecules_with_pos,
        idea_f_result=result.get('idea_f', {}),
        output_dir=output_dir,
        title_prefix=title_prefix,
        color_by_qed=qed_list,
    )
    saved.update(r)

    # 5. Size distribution (Idea G)
    r = visualize_size_distribution(
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

    # 7. Summary radar
    r = visualize_radar_summary(
        result,
        output_dir=output_dir,
        title_prefix=title_prefix,
    )
    saved.update(r)

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
    weight_a=0.25, weight_b=0.15, weight_c=0.15,
    weight_d=0.15, weight_e=0.10, weight_f=0.10, weight_g=0.10,
    weight_h=0.08,
    idea_b_atom_coord_subset='all',
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
    综合评估口袋质量（八维度加权 A/B/C/D/E/F/G/H；C 为配体效率 LE，依赖 Vina 与分子顺序对齐）

    分子来源（二选一）：
        pt_path     — 扩散模型输出的 .pt（含 pred_ligand_pos / pred_ligand_v）
        ligand_path — 自定义 .sdf / .mol 或含此类文件的目录（3D 构象；无 .pt 时想法 E 跳过）

    想法 A（Vina）需在 outputs 下存在 eval_* 与 complete_molecules_*.xlsx（或 eval_results_*.pt），
    可通过 vina_outputs_dir / custom_pocket_pdb / 配体路径旁自动推测。

    Args:
        idea_b_atom_coord_subset : 想法B 的 DBSCAN 点集（all / hetero_heavy / edge_heavy /
            hetero_edge / combined），默认 all；combined 为全原子与 hetero_edge 加权。
        idea_b_edge_max_heavy_neighbors : edge_heavy / hetero_edge 的非氢重邻居上限。
        idea_b_combined_focus_weight : combined 模式下 hetero_edge 分项权重 w。
        vina_pocket_id : 匹配 eval_{id}_* 的目录名片段；默认由 data_id 或 'custom' 决定。
        fpocket_protein_pdb : 若指定，想法 H 用 FPocket 算蛋白口袋体积，并对生成配体合并 PDB 再跑 FPocket（报告用）。
        idea_h_ligand_path : 可选。数据源为 .pt 时，仅**想法 H 配体侧 FPocket** 改用该路径下的 .sdf/.mol（或目录），
            A–G 仍用 .pt 重建分子；未解析到构象时自动退回 .pt 配体列表。
        fpocket_cmd / fpocket_pocket_index / fpocket_max_ligand_models / fpocket_timeout ：FPocket 调用参数。
        fpocket_optimal_min 等：仅**配体 FPocket 失败**时，蛋白体积回退评分用的宽区间（默认
            IDEA_H_FPOCKET_PROTEIN_FALLBACK_*）；配体成功时 H 分始终用 optimal_min/max（MC 档）。
        visualize : 是否生成可视化图（需要 matplotlib）
        vis_dir   : 可视化图输出目录（由 main 构建为 pocket_quality_vis/蛋白质编号_时间戳/）
        idea_e_expected_n_molecules : 可选正整数。想法 E 分母：应生成分子数；分子为 **SMILES 不含 '.'**
            的完整单组分分子数 / 该值。指定后不再用 .pt 的 ``pred_ligand_pos`` 长度；仅配体文件评估时也可启用想法 E。

    Returns:
        dict: 包含各维度详细指标、综合质量分数及可视化路径
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

    idea_a = evaluate_idea_a_vina(
        pt_path=str(pp) if pp is not None else None,
        data_id=eval_tag,
        outputs_dir=str(vod) if vod is not None else None,
        pocket_id_override=vina_pocket_id,
    )
    idea_b = evaluate_idea_b_clustering(
        molecules,
        atom_coord_subset=idea_b_atom_coord_subset,
        edge_max_heavy_neighbors=idea_b_edge_max_heavy_neighbors,
        combined_focus_weight=idea_b_combined_focus_weight,
    )
    vina_for_le = idea_a.get('vina_scores') if idea_a.get('success') else []
    idea_c = evaluate_idea_c_ligand_efficiency(molecules, vina_for_le)
    idea_d = evaluate_idea_d_druglikeness(molecules)
    idea_e = evaluate_idea_e_reconstruction(
        str(pp) if pp is not None else None,
        molecules,
        expected_n_molecules=idea_e_expected_n_molecules,
    )
    idea_f = evaluate_idea_f_uniqueness(molecules)
    idea_g = evaluate_idea_g_size_consistency(molecules)
    idea_h = evaluate_idea_h_pocket_size(
        molecules_for_h,
        fpocket_protein_pdb=fpocket_protein_pdb,
        fpocket_cmd=fpocket_cmd,
        fpocket_pocket_index=fpocket_pocket_index,
        fpocket_max_ligand_models=fpocket_max_ligand_models,
        fpocket_timeout=fpocket_timeout,
        fpocket_optimal_min=fpocket_optimal_min,
        fpocket_optimal_max=fpocket_optimal_max,
        fpocket_zero_below=fpocket_zero_below,
        fpocket_zero_above=fpocket_zero_above,
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

    ideas = {
        'A': (idea_a, weight_a), 'B': (idea_b, weight_b), 'C': (idea_c, weight_c),
        'D': (idea_d, weight_d), 'E': (idea_e, weight_e), 'F': (idea_f, weight_f),
        'G': (idea_g, weight_g), 'H': (idea_h, weight_h),
    }
    total_weight = 0.0
    weighted_sum = 0.0
    for k, (res, w) in ideas.items():
        if res.get('success') and w > 0:
            total_weight += w
            weighted_sum += w * res.get('score', 0.0)

    overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    if overall_score >= 0.6:
        overall_label = 'high'
    elif overall_score >= 0.3:
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
        'overall_score': overall_score,
        'overall_label': overall_label,
        'weights': {k: w for k, (_, w) in ideas.items()},
        'visualizations': {},
        'idea_h_ligand_path': idea_h_ligand_path_resolved,
        'idea_h_ligand_override_used': idea_h_ligand_override_used,
        'idea_e_expected_n_molecules': idea_e_expected_n_molecules,
    }

    # 可视化：vis_dir 由调用方传入（已为 蛋白质编号_时间戳 结构）
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
    print('【想法A】Vina 对接分数与口袋质量')
    print(f"  状态: {'成功' if a.get('success') else '失败'}")
    if a.get('success'):
        print(f"  Vina 平均/中位数/标准差: {a.get('vina_mean'):.2f} / "
              f"{a.get('vina_median'):.2f} / {a.get('vina_std'):.2f} kcal/mol")
        print(f"  Vina 最佳/最差: {a.get('vina_best'):.2f} / {a.get('vina_worst'):.2f} kcal/mol")
        print(f"  亲和力良好比例(Vina≤-7): {a.get('vina_pct_good', 0)*100:.1f}%")
        print(f"  有效分数数量: {a.get('num_scores')}")
        print(f"  质量分数: {a.get('score'):.3f} ({a.get('quality_label')})")
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
    print('【想法C】配体效率 LE = -ΔG / N_heavy（ΔG 为 Vina 亲和力 kcal/mol）')
    print(f"  状态: {'成功' if c.get('success') else '失败'}")
    if c.get('success'):
        lo = c.get('le_low_for_score', IDEA_C_LE_SCORE_LOW)
        hi = c.get('le_high_for_score', IDEA_C_LE_SCORE_HIGH)
        print(
            f"  LE 均值/中位数±std: {c.get('le_mean'):.4f} / {c.get('le_median'):.4f} "
            f"± {c.get('le_std', 0):.4f} kcal·mol⁻¹·重原子⁻¹"
        )
        print(f"  有效分子数: {c.get('n_valid')} (对齐长度 {c.get('n_aligned')})")
        print(f"  质量分映射参考区间: [{lo}, {hi}]（LE 均值线性映射到 0–1）")
        print(f"  质量分数: {c.get('score'):.3f} ({c.get('quality_label')})")
    else:
        print(f"  备注: {c.get('message', '')}")
    print()

    d = result.get('idea_d', {})
    print('【想法D】药物相似性 (QED/SA/Lipinski/PAINS)')
    print(f"  状态: {'成功' if d.get('success') else '失败'}")
    if d.get('success'):
        print(f"  QED 均值±标准差: {d.get('qed_mean', 0):.3f} ± {d.get('qed_std', 0):.3f}")
        print(f"  SA  均值±标准差: {d.get('sa_mean', 0):.3f} ± {d.get('sa_std', 0):.3f}")
        print(f"  Lipinski 合规(0-1): {d.get('lipinski_mean', 0):.2f}")
        print(f"  PAINS 命中率: {d.get('pains_ratio', 0)*100:.1f}%")
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
    print('【想法F】分子唯一性与多样性')
    print(f"  状态: {'成功' if f.get('success') else '失败'}")
    if f.get('success'):
        print(f"  唯一分子/有效分子: {f.get('n_unique')}/{f.get('n_total')}")
        print(f"  唯一性比例: {f.get('unique_ratio', 0)*100:.1f}%")
        td = f.get('tanimoto_diversity')
        print(f"  Tanimoto 多样性: {f'{td:.4f}' if td is not None else 'N/A'}")
        print(f"  质量分数: {f.get('score'):.3f} ({f.get('quality_label')})")
    else:
        print(f"  备注: {f.get('message', '')}")
    print()

    g = result.get('idea_g', {})
    print('【想法G】分子尺寸一致性')
    print(f"  状态: {'成功' if g.get('success') else '失败'}")
    if g.get('success'):
        print(f"  原子数 均值±标准差: {g.get('n_atoms_mean', 0):.1f} ± {g.get('n_atoms_std', 0):.1f}")
        print(f"  分子量 均值: {g.get('mw_mean', 0):.0f}, 变异系数(CV): {g.get('mw_cv', 0):.2f}")
        print(f"  质量分数: {g.get('score'):.3f} ({g.get('quality_label')})")
    else:
        print(f"  备注: {g.get('message', '')}")
    print()

    h = result.get('idea_h', {})
    fp_pdb = h.get('fpocket_protein_pdb')
    if fp_pdb:
        if h.get('fpocket_skipped'):
            print('【想法H】口袋体积（已指定蛋白 PDB，但 FPocket 不可用 → 已退回 MC）')
        else:
            print('【想法H】口袋体积（FPocket：配体体积按 MC 区间计分；蛋白体积参考）')
    if result.get('idea_h_ligand_path'):
        print(f"  配体 FPocket 构象来源: 外部文件（仅 H）: {result.get('idea_h_ligand_path')}")
    elif h.get('ligand_fpocket_override_note'):
        print(f"  备注: {h.get('ligand_fpocket_override_note')}")
    else:
        print('【想法H】口袋体积（质心均值 10Å 球内占据 / MC）')
    print(f"  状态: {'成功' if h.get('success') else '失败'}")
    if h.get('success'):
        vol = h.get('volume_ang3')
        if vol is not None:
            print(f"  用于评分的体积: {vol:.0f} Å³ ({h.get('volume_method', '')})")
        if fp_pdb:
            print(f"  蛋白 PDB: {fp_pdb}")
            if h.get('fpocket_skipped'):
                print('  说明: 未找到 fpocket 可执行文件，体积按 MC 计算（见 message）')
            elif h.get('fpocket_resolved_executable'):
                print(f"  FPocket 可执行文件: {h.get('fpocket_resolved_executable')}")
            if fp_pdb and not h.get('fpocket_skipped'):
                pm = h.get('fpocket_protein_pocket_meta') or {}
                if pm.get('score') is not None:
                    ds = pm.get('druggability')
                    if ds is not None:
                        print(f"  蛋白口袋 FPocket Score: {pm['score']:.3f}；Druggability: {ds:.3f}")
                    else:
                        print(f"  蛋白口袋 FPocket Score: {pm['score']:.3f}")
                lv = h.get('fpocket_ligand_volume_ang3')
                vp = h.get('fpocket_protein_volume_ang3')
                if vp is not None:
                    print(f"  蛋白 FPocket Volume: {vp:.0f} Å³（参考）")
                if h.get('volume_score_band') == 'fpocket_protein_fallback':
                    if lv is not None:
                        print(f"  配体 FPocket Volume: {lv:.0f} Å³（未参与 H 分）")
                    elif not h.get('fpocket_ligand_success'):
                        print('  配体 FPocket: 未得到体积（H 分已按蛋白体积回退，见 message）')
        if (not fp_pdb) or h.get('fpocket_skipped'):
            nc = h.get('n_centroids')
            vsref = h.get('single_sphere_ref_volume_ang3')
            frac = h.get('ligand_occupancy_fraction')
            if nc is not None and vsref is not None:
                print(f"  分子数: {nc}；10Å 球体积上限≈{vsref:.0f} Å³", end='')
                if frac is not None:
                    print(f"；MC 占据率={frac*100:.1f}%")
                else:
                    print()
        if h.get('message'):
            print(f"  摘要: {h.get('message')}")
        omin = h.get('optimal_vol_min', 400)
        omax = h.get('optimal_vol_max', 600)
        zb = h.get('vol_score_zero_below', 100)
        za = h.get('vol_score_zero_above', 900)
        print(
            f"  满分区间: {omin:.0f}–{omax:.0f} Å³；"
            f"其外按线性扣分；<{zb:.0f} 或 ≥{za:.0f} Å³ 为 0 分"
        )
        print(f"  质量分数: {h.get('score'):.3f} ({h.get('quality_label')})")
    else:
        print(f"  备注: {h.get('message', '')}")
    print()

    print('【综合评估】')
    w = result.get('weights', {})
    active = [k for k in 'ABCDEFGH' if w.get(k, 0) > 0]
    print(f"  参与加权: {', '.join(active)}")
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
    parser.add_argument('--weight_a', type=float, default=0.25)
    parser.add_argument('--weight_b', type=float, default=0.15)
    parser.add_argument('--weight_c', type=float, default=0.15,
                        help='想法C 配体效率 LE 权重（默认 0.15）')
    parser.add_argument('--weight_d', type=float, default=0.15)
    parser.add_argument('--weight_e', type=float, default=0.10)
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
    parser.add_argument('--weight_f', type=float, default=0.10)
    parser.add_argument('--weight_g', type=float, default=0.10)
    parser.add_argument('--weight_h', type=float, default=0.08,
                        help='想法H 口袋体积权重（默认0.08）')
    parser.add_argument(
        '--fpocket_protein_pdb',
        type=str,
        default=None,
        help='想法H：蛋白 PDB 路径，指定则用 FPocket 解析 *_info.txt 中口袋 Volume（Å³），'
             '并对生成配体合并 PDB 再跑 FPocket（报告）；需系统已安装 fpocket',
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
        default='all',
        choices=('all', 'hetero_heavy', 'edge_heavy', 'hetero_edge', 'combined'),
        help='想法B：DBSCAN 用的坐标子集。hetero_heavy=非H非C；edge_heavy=重邻居≤阈值的表面重原子；'
             'hetero_edge=二者交；combined=(1-w)*全原子+w*hetero_edge（见下）',
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
        print_evaluation_report(result)
        record_path = base_root / EVAL_RECORDS_CSV
        try:
            append_evaluation_record(result, record_path, timestamp=timestamp)
        except PermissionError as e:
            print(f"⚠️  未写入 {record_path}: {e}", flush=True)
    if results:
        print(f"\nEvaluation records -> {base_root / EVAL_RECORDS_CSV}")


if __name__ == '__main__':
    main()
