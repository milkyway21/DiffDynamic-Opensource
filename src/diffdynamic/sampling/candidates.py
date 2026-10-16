"""Sampling candidates responsibilities for ligand sampling."""

from __future__ import annotations

from rdkit import Chem
from diffdynamic.evaluation import scoring_func
import numpy as np
import torch
import diffdynamic.chemistry.reconstruct as reconstruct
import diffdynamic.data.transforms as trans


def evaluate_candidate(
    pos_array, v_array, ligand_atom_mode, selector_cfg,
    scaffold_bonds=None, n_scaffold=None,
):  # 评估候选分子的化学指标。
    """根据采样结果评估分子质量，计算化学指标并返回评分。

    Args:
        pos_array: 形状为 `[num_atoms, 3]` 的坐标数组。
        v_array: 原子类型索引数组。
        ligand_atom_mode: 配体原子编码模式（与训练配置保持一致）。
        selector_cfg: 筛选配置，包含权重及阈值。
        scaffold_bonds: 可选参考骨架键表（0-based，前缀原子）。
        n_scaffold: 骨架原子数（与 scaffold_bonds 配套）。

    Returns:
        dict: 包含化学指标、SMILES、筛选状态及综合评分。
    """
    metrics = {}  # 初始化化学指标字典。
    smiles = None  # 初始 SMILES。
    status = 'ok'  # 状态标记。
    score_value = float('inf')  # 评分值（越小越好）。

    try:
        v_tensor = torch.tensor(v_array, dtype=torch.long)  # 将类别数组转为张量。
        atom_numbers = trans.get_atomic_number_from_index(v_tensor, mode=ligand_atom_mode)  # 获取原子序号。
        aromatic_flags = trans.is_aromatic_from_index(v_tensor, mode=ligand_atom_mode)  # 获取芳香标记。
        basic_mode = str(ligand_atom_mode) == 'basic'
        mol = reconstruct.reconstruct_from_generated(
            pos_array, atom_numbers, aromatic_flags,
            basic_mode=basic_mode,
            scaffold_bonds=scaffold_bonds,
            n_scaffold=n_scaffold,
        )  # 重建分子。
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)  # 转换为 SMILES。
            # 分子不完整：SMILES 含 '.' 表示多个不连通片段，直接过滤
            if selector_cfg.get('filter_incomplete', True) and smiles and '.' in smiles:
                status = 'incomplete'
                metrics = scoring_func.get_chem(mol)  # 仍计算指标供日志
                score_value = float('inf')
            else:
                metrics = scoring_func.get_chem(mol)  # 计算化学指标（QED/SA）。
                qed = metrics.get('qed', float('nan'))  # 提取 QED。
                sa = metrics.get('sa', float('inf'))  # 提取 SA。
                qed_weight = selector_cfg.get('qed_weight', 1.0)  # QED 权重。
                sa_weight = selector_cfg.get('sa_weight', 1.0)  # SA 权重。
                if np.isnan(qed) or np.isnan(sa):  # 若指标非法，则标记无效。
                    score_value = float('inf')
                    status = 'metric_nan'
                else:
                    # SA和QED都越大越好，所以score = -(sa_weight * sa + qed_weight * qed)，score越小越好
                    score_value = -(sa_weight * sa + qed_weight * qed)  # 根据权重计算综合分数。
        else:
            status = 'mol_none'  # 重建返回空。
            metrics = {'qed': float('nan'), 'sa': float('inf')}
    except reconstruct.MolReconsError:
        status = 'reconstruct_failed'  # 重建抛出特定错误。
        metrics = {'qed': float('nan'), 'sa': float('inf')}
        score_value = float('inf')
    except Exception as exc:
        status = f'error:{exc.__class__.__name__}'  # 捕获其他异常。
        metrics = {'qed': float('nan'), 'sa': float('inf')}
        score_value = float('inf')

    min_qed = selector_cfg.get('min_qed')  # QED 下限。
    # SA越大越好，所以使用min_sa作为下限；为了向后兼容，也支持max_sa（但会转换为min_sa的语义）
    min_sa = selector_cfg.get('min_sa')
    max_sa = selector_cfg.get('max_sa')  # 向后兼容：如果配置了max_sa但没有min_sa，则使用max_sa作为min_sa
    if min_sa is None and max_sa is not None:
        min_sa = max_sa  # 向后兼容：将max_sa当作min_sa使用
    if metrics:
        qed = metrics.get('qed', float('nan'))
        sa = metrics.get('sa', float('inf'))
        if min_qed is not None and not np.isnan(qed) and qed < min_qed:  # 不满足 QED 下限则过滤。
            status = 'filtered_qed'
            score_value = float('inf')
        if min_sa is not None and not np.isnan(sa) and sa < min_sa:  # 不满足 SA 下限则过滤（SA越大越好）。
            status = 'filtered_sa'
            score_value = float('inf')

    return {
        'metrics': metrics,
        'smiles': smiles,
        'status': status,
        'score': score_value
    }  # 返回评估结果。


def _selector_metrics_or_skip(pos_array, v_array, ligand_atom_mode, selector_cfg, enable_selection=True):
    """selector 关闭时跳过 RDKit/QED/SA（不影响坐标）；开启时走 ``evaluate_candidate``。"""
    if not enable_selection:
        return {
            'metrics': {},
            'smiles': None,
            'status': 'skipped',
            'score': float('inf'),
        }
    return evaluate_candidate(pos_array, v_array, ligand_atom_mode, selector_cfg)


def select_top_candidates(candidates, top_n):  # 根据评分选出 top-N 候选。
    """按照综合得分排序候选分子，返回排名靠前的候选集合。

    Args:
        candidates: 由 `evaluate_candidate` 生成的候选列表。
        top_n: 需要保留的候选数量，若小于等于 0 则返回空列表。

    Returns:
        list[dict]: 经过排序与补充后的候选列表（长度不超过 `top_n`）。
    """
    if top_n <= 0:
        return []
    valid = [c for c in candidates if np.isfinite(c['score'])]  # 过滤有限分数的候选。
    if not valid:
        valid = candidates  # 若全部无效，则退回所有候选。
    sorted_candidates = sorted(valid, key=lambda x: x['score'])  # 按分数升序排序。
    if len(sorted_candidates) >= top_n:
        return sorted_candidates[:top_n]  # 返回前 top-N。
    # 补充剩余候选：如果有效候选不足 top_n，则从无效候选中选择补充。
    sorted_ids = {id(c) for c in sorted_candidates}  # 使用对象 id 避免 numpy 比较歧义。
    remaining = [c for c in candidates if id(c) not in sorted_ids]  # 找出未排序的候选。
    sorted_remaining = sorted(remaining, key=lambda x: x['score'])
    combined = sorted_candidates + sorted_remaining  # 合并两部分。
    return combined[:min(len(combined), top_n)]  # 返回最多 top-N 个候选。
