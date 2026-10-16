"""Sampling scaffold fitness responsibilities for ligand sampling."""

from __future__ import annotations

from rdkit import Chem
from diffdynamic.evaluation import scoring_func
import numpy as np
import diffdynamic.chemistry.reconstruct as reconstruct


def _compute_scaffold_fitness(
    pos_np: np.ndarray,
    v_np: np.ndarray,
    ligand_atom_mode: str,
    scaffold_cfg: dict,
    ref_pos_np: np.ndarray,
    scaffold_mask_np: np.ndarray,
    existing_mols_smiles: list,
) -> dict:
    """
    多目标适应度评分（越高越好）。

    fitness = qed_w * QED + sa_w * SA
              + div_w * diversity_bonus
              - rmsd_w * scaffold_RMSD

    多样性奖励：基于 Morgan 指纹，计算当前分子与已选种群的平均 Tanimoto 距离。
    种群多样性越高（越不像已有成员），奖励越大；防止种群塌缩。

    Args:
        pos_np: 当前分子坐标 [N,3]。
        v_np: 当前分子原子类型 [N]。
        ligand_atom_mode: 原子编码模式。
        scaffold_cfg: scaffold_optimization 配置。
        ref_pos_np: 参考分子坐标（用于 RMSD）。
        scaffold_mask_np: 骨架掩码 [N]，1=骨架原子。
        existing_mols_smiles: 当前种群中已有分子的 SMILES 列表（多样性对比用）。

    Returns:
        dict with keys: fitness, qed, sa, scaffold_rmsd, diversity_bonus, smiles, status
    """
    from rdkit.Chem import Descriptors, QED
    from rdkit.Chem import AllChem
    from rdkit import DataStructs

    result = {
        'fitness': -float('inf'),
        'qed': 0.0,
        'sa': 0.0,
        'scaffold_rmsd': float('inf'),
        'diversity_bonus': 0.0,
        'smiles': None,
        'status': 'failed',
    }

    try:
        mol = reconstruct.reconstruct_from_type_indices(
            pos_np, v_np, ligand_atom_mode=ligand_atom_mode,
        )
    except Exception:
        mol = None

    if mol is None:
        result['status'] = 'reconstruct_failed'
        return result

    smiles = Chem.MolToSmiles(mol)
    result['smiles'] = smiles

    # 完整性检查（无断键）
    if scaffold_cfg.get('filter_incomplete', True):
        frags = smiles.split('.')
        if len(frags) > 1:
            result['status'] = 'incomplete'
            return result

    # QED/SA
    try:
        qed_val = QED.qed(mol)
    except Exception:
        qed_val = 0.0
    try:
        sa_val = scoring_func.compute_sa_score(mol)
    except Exception:
        sa_val = 0.0

    result['qed'] = float(qed_val)
    result['sa'] = float(sa_val)

    # QED/SA 阈值过滤
    min_qed = scaffold_cfg.get('min_qed', 0.0)
    min_sa = scaffold_cfg.get('min_sa', 0.0)
    if qed_val < min_qed:
        result['status'] = 'filtered_qed'
        return result
    if sa_val < min_sa:
        result['status'] = 'filtered_sa'
        return result

    # 骨架 RMSD（仅统计骨架原子）
    scaffold_mask_flat = scaffold_mask_np.astype(bool)
    if scaffold_mask_flat.any() and pos_np.shape[0] == ref_pos_np.shape[0]:
        rmsd = float(np.sqrt(np.mean((pos_np[scaffold_mask_flat] - ref_pos_np[scaffold_mask_flat]) ** 2)))
    else:
        rmsd = float(np.sqrt(np.mean((pos_np - ref_pos_np) ** 2)))
    result['scaffold_rmsd'] = rmsd

    # 多样性奖励：与种群现有成员的 Tanimoto 距离均值
    div_bonus = 0.0
    div_filter_cfg = scaffold_cfg.get('diversity_filter', {})
    fp_radius = int(div_filter_cfg.get('fingerprint_radius', 2))
    fp_nbits = int(div_filter_cfg.get('fingerprint_nbits', 2048))
    if existing_mols_smiles:
        try:
            fp_cur = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_nbits)
            distances = []
            for smi in existing_mols_smiles:
                ref_m = Chem.MolFromSmiles(smi)
                if ref_m is not None:
                    fp_ref = AllChem.GetMorganFingerprintAsBitVect(ref_m, fp_radius, nBits=fp_nbits)
                    sim = DataStructs.TanimotoSimilarity(fp_cur, fp_ref)
                    distances.append(1.0 - sim)
            if distances:
                div_bonus = float(np.mean(distances))
        except Exception:
            div_bonus = 0.0
    result['diversity_bonus'] = div_bonus

    # 综合适应度
    qed_w = float(scaffold_cfg.get('qed_weight', 1.0))
    sa_w = float(scaffold_cfg.get('sa_weight', 1.0))
    div_w = float(scaffold_cfg.get('diversity_weight', 0.3))
    rmsd_w = float(scaffold_cfg.get('rmsd_penalty_weight', 0.0))
    fitness = qed_w * qed_val + sa_w * sa_val + div_w * div_bonus - rmsd_w * rmsd
    result['fitness'] = fitness
    result['status'] = 'ok'
    return result


def _diversity_filter_population(
    candidates: list,
    population_size: int,
    scaffold_cfg: dict,
) -> list:
    """
    多样性感知种群选择：贪心选取前 population_size 个，保证 Tanimoto 相似度低于阈值。

    算法：
    1. 按 fitness 降序排列所有候选
    2. 贪心地加入新成员：若与已选任一成员 Tanimoto 相似度 > max_tanimoto，跳过
    3. 若贪心未选满 population_size（常见于 evolve 子代 SMILES 极相似），再按 fitness 序补位，
       保证在「有足够 status=ok 候选」时尽量达到 population_size，避免种群塌缩为 1 条。

    Args:
        candidates: 候选字典列表，每个含 'fitness'、'smiles' 等键。
        population_size: 目标种群大小。
        scaffold_cfg: scaffold_optimization 配置。

    Returns:
        selected: 经多样性筛选后的种群列表，按 fitness 降序。
    """
    from rdkit.Chem import AllChem
    from rdkit import DataStructs

    div_filter_cfg = scaffold_cfg.get('diversity_filter', {})
    enable_div = div_filter_cfg.get('enable', True)
    max_tanimoto = float(div_filter_cfg.get('max_tanimoto', 0.85))
    fp_radius = int(div_filter_cfg.get('fingerprint_radius', 2))
    fp_nbits = int(div_filter_cfg.get('fingerprint_nbits', 2048))

    ok_candidates = [c for c in candidates if c.get('status') == 'ok']
    ok_candidates.sort(key=lambda x: x.get('fitness', -float('inf')), reverse=True)

    if not enable_div:
        return ok_candidates[:population_size]

    selected = []
    selected_fps = []
    selected_smiles = set()
    for cand in ok_candidates:
        if len(selected) >= population_size:
            break
        smi = cand.get('smiles')
        if smi is None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_nbits)
        except Exception:
            continue
        too_similar = any(
            DataStructs.TanimotoSimilarity(fp, fp_sel) > max_tanimoto
            for fp_sel in selected_fps
        )
        if not too_similar:
            selected.append(cand)
            selected_fps.append(fp)
            selected_smiles.add(smi)

    # 补位阶段 1：按不同 SMILES 补入（化学多样性）
    if len(selected) < population_size:
        for cand in ok_candidates:
            if len(selected) >= population_size:
                break
            smi = cand.get('smiles')
            if smi is None or smi in selected_smiles:
                continue
            selected.append(cand)
            selected_smiles.add(smi)

    # 补位阶段 2：SMILES 相同但 3D 坐标不同的也保留（3D 构象多样性）
    # 当 start_t 较低时子代 SMILES 常完全一致，但坐标仍有差异，保留这些 3D 变体
    if len(selected) < population_size:
        for cand in ok_candidates:
            if len(selected) >= population_size:
                break
            if cand in selected:
                continue
            pos_cand = cand.get('pos')
            if pos_cand is None:
                continue
            too_close_3d = False
            for sel in selected:
                pos_sel = sel.get('pos')
                if pos_sel is not None and pos_cand.shape == pos_sel.shape:
                    rmsd = float(np.sqrt(np.mean((pos_cand - pos_sel) ** 2)))
                    if rmsd < 0.3:
                        too_close_3d = True
                        break
            if not too_close_3d:
                selected.append(cand)

    return selected
