"""Sampling scaffold inputs responsibilities for ligand sampling."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from rdkit import Chem
from diffdynamic.evaluation import atom_num
from diffdynamic.chemistry.scaffold_prior import extend_scaffold_to_first_amine
from diffdynamic.chemistry.scaffold_inputs import load_reference_molecule
from diffdynamic.chemistry.scaffold_inputs import reference_molecule_from_data
from diffdynamic.chemistry.scaffold_inputs import resolve_scaffold_indices
import numpy as np
import os
import torch


def _detect_murcko_scaffold_indices(mol) -> list:
    """
    使用 Murcko 分解自动识别分子主骨架原子索引。

    Murcko 骨架定义（Bemis-Murcko, 1996）：
    - 保留所有环系（ring systems）
    - 保留连接各环系的链接骨架（linker atoms）
    - 移除所有侧链（side chains / R-groups）

    算法：
    1. 计算 Murcko 骨架分子
    2. 用子结构匹配（GetSubstructMatches）将骨架原子映射回原始分子索引

    Args:
        mol: RDKit Mol 对象（需已添加氢原子以外的完整分子）。

    Returns:
        list[int]: 骨架原子的原始 0-based 索引列表。
                   若失败（无环等）返回空列表。
    """
    logger = None
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
        
        # 确保分子有 RingInfo
        try:
            ri = mol.GetRingInfo()
            if ri is None or ri.NumRings() < 0:
                Chem.GetSSSR(mol)
        except:
            pass
            
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return []
        matches = mol.GetSubstructMatches(scaffold)
        if not matches:
            return []
        return list(matches[0])
    except Exception as e:
        if logger and 'logger' in locals():
            logger.warning(f'[_detect_murcko_scaffold_indices] 失败: {e}')
        return []


def _detect_generic_murcko_scaffold_indices(mol) -> list:
    """
    使用「通用 Murcko 骨架」（Generic Murcko Scaffold）识别骨架原子。

    通用骨架将所有原子替换为碳，所有键替换为单键，仅保留拓扑结构，
    对杂环分子（含 N/O/S）也能提供稳定的骨架识别。
    与 _detect_murcko_scaffold_indices 配合使用：优先尝试普通 Murcko，
    若原子数 0 再尝试通用骨架。

    Args:
        mol: RDKit Mol 对象。

    Returns:
        list[int]: 骨架原子的原始 0-based 索引列表。
    """
    logger = None
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
        
        # 确保分子有 RingInfo
        try:
            ri = mol.GetRingInfo()
            if ri is None or ri.NumRings() < 0:
                Chem.GetSSSR(mol)
        except:
            pass
            
        generic = MurckoScaffold.MakeScaffoldGeneric(
            MurckoScaffold.GetScaffoldForMol(mol)
        )
        if generic is None or generic.GetNumAtoms() == 0:
            return []
        matches = mol.GetSubstructMatches(generic)
        if not matches:
            return []
        return list(matches[0])
    except Exception as e:
        if logger and 'logger' in locals():
            logger.warning(f'[_detect_generic_murcko_scaffold_indices] 失败: {e}')
        return []


def _detect_ring_based_scaffold_indices(mol, logger=None) -> list:
    """
    基于环检测的简化骨架提取算法（MurckoScaffold 的替代方案）。
    不依赖 MurckoScaffold，直接检测环结构作为骨架。
    
    算法：
    1. 找到所有环原子（使用 SSSR - Smallest Set of Smallest Rings）
    2. 添加连接环的链接原子（linker atoms）
    3. 保留环原子和链接原子作为骨架，移除侧链原子
    
    Args:
        mol: RDKit Mol 对象。
        logger: 可选的日志记录器。
    
    Returns:
        list[int]: 骨架原子的原始 0-based 索引列表。失败时返回空列表。
    """
    try:
        # 确保 RingInfo 已初始化
        try:
            ri = mol.GetRingInfo()
            if ri is None or ri.NumRings() < 0:
                Chem.GetSSSR(mol)
                ri = mol.GetRingInfo()
        except Exception as ring_err:
            if logger:
                logger.warning(f'[_detect_ring_based] RingInfo 初始化失败: {ring_err}')
            return []
        
        if not ri or ri.NumRings() < 0:
            if logger:
                logger.warning('[_detect_ring_based] RingInfo 未初始化')
            return []
        
        # 获取所有环原子
        ring_atoms = set()
        atom_rings = ri.AtomRings()
        
        if not atom_rings:
            if logger:
                logger.info('[_detect_ring_based] 分子中没有环结构')
            return []
        
        for ring in atom_rings:
            for atom_idx in ring:
                ring_atoms.add(atom_idx)
        
        if not ring_atoms:
            return []
        
        if logger:
            logger.info(f'[_detect_ring_based] 检测到 {len(atom_rings)} 个环，{len(ring_atoms)} 个环原子')
        
        # 找到连接环的链接原子（linker atoms）
        # 链接原子定义：连接两个不同环的路径上的原子
        linker_atoms = set()
        
        for atom_idx in range(mol.GetNumAtoms()):
            if atom_idx in ring_atoms:
                continue
            
            atom = mol.GetAtomWithIdx(atom_idx)
            neighbors = atom.GetNeighbors()
            
            # 检查该原子的邻居属于哪些环
            neighbor_rings = set()
            for neighbor in neighbors:
                neighbor_idx = neighbor.GetIdx()
                # 检查邻居是否在环中
                for ring_idx, ring in enumerate(atom_rings):
                    if neighbor_idx in ring:
                        neighbor_rings.add(ring_idx)
            
            # 如果邻居属于 2 个或更多不同的环，则该原子是链接原子
            if len(neighbor_rings) >= 2:
                linker_atoms.add(atom_idx)
        
        if logger and linker_atoms:
            logger.info(f'[_detect_ring_based] 检测到 {len(linker_atoms)} 个链接原子')
        
        # 组合环原子和链接原子作为骨架
        scaffold_atoms = ring_atoms.union(linker_atoms)
        
        # 确保至少保留了一些原子
        if not scaffold_atoms:
            if logger:
                logger.warning('[_detect_ring_based] 未找到任何骨架原子')
            return []
        
        if logger:
            logger.info(f'[_detect_ring_based] 最终骨架: {len(scaffold_atoms)} 个原子（{len(ring_atoms)} 环 + {len(linker_atoms)} 链接）')
        
        return sorted(list(scaffold_atoms))
        
    except Exception as e:
        if logger:
            logger.warning(f'[_detect_ring_based] 骨架提取失败: {e}')
        return []


def _extract_scaffold_bond_pairs(ref_mol, scaffold_indices, logger=None):
    """从参考分子提取骨架内部键，映射到前缀下标 0..n_scaffold-1。

    Returns:
        list of (i_new, j_new, order, aromatic) 或空列表。
    """
    if ref_mol is None or not scaffold_indices:
        return []
    try:
        index_map = {int(old): new for new, old in enumerate(scaffold_indices)}
        scaffold_set = set(index_map.keys())
        bonds = []
        for bond in ref_mol.GetBonds():
            bi = bond.GetBeginAtomIdx()
            bj = bond.GetEndAtomIdx()
            if bi not in scaffold_set or bj not in scaffold_set:
                continue
            i_new = index_map[bi]
            j_new = index_map[bj]
            aromatic = bool(
                bond.GetIsAromatic() or bond.GetBondType() == Chem.BondType.AROMATIC
            )
            if aromatic:
                order = 5
            else:
                order = int(round(bond.GetBondTypeAsDouble()))
                order = max(1, min(3, order))
            bonds.append((i_new, j_new, order, aromatic))
        return bonds
    except Exception as e:
        if logger:
            logger.warning(f'[_extract_scaffold_bond_pairs] 失败: {e}')
        return []


def _extract_scaffold_submol(ref_mol, scaffold_indices, logger=None):
    """
    从参考分子中提取骨架原子的子结构并创建新的 RDKit 分子。
    
    Args:
        ref_mol: 原始参考分子 (RDKit Mol 对象)
        scaffold_indices: 骨架原子索引列表 (0-based)
        logger: 可选的日志记录器
    
    Returns:
        Chem.Mol: 包含骨架原子和内部键的新分子，失败时返回 None
    """
    if ref_mol is None or not scaffold_indices:
        return None
    
    try:
        # 创建可编辑分子
        scaffold_indices_set = set(scaffold_indices)
        
        # 使用 RDKit 的 GetSubmol 方法提取子结构（保留键连接信息）
        # 注意：GetSubmol 会自动包含连接骨架原子的键
        atom_map = {}  # 原始索引 -> 新索引
        emol = Chem.EditableMol(Chem.Mol())
        
        # 添加原子
        for old_idx in scaffold_indices:
            if old_idx < ref_mol.GetNumAtoms():
                atom = ref_mol.GetAtomWithIdx(old_idx)
                new_idx = emol.AddAtom(atom)
                atom_map[old_idx] = new_idx
        
        # 添加骨架原子之间的键（只保留两端都在骨架中的键）
        for bond in ref_mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            
            # 只保留骨架原子之间的键
            if begin_idx in scaffold_indices_set and end_idx in scaffold_indices_set:
                begin_new = atom_map[begin_idx]
                end_new = atom_map[end_idx]
                bond_type = bond.GetBondType()
                emol.AddBond(begin_new, end_new, bond_type)
        
        # 创建最终分子
        scaffold_mol = emol.GetMol()
        
        # 设置 3D 坐标（如果有）
        if ref_mol.GetNumConformers() > 0:
            conf = ref_mol.GetConformer()
            new_conf = Chem.Conformer(len(scaffold_indices))
            
            for old_idx, new_idx in atom_map.items():
                pos = conf.GetAtomPosition(old_idx)
                new_conf.SetAtomPosition(new_idx, pos)
            
            scaffold_mol.AddConformer(new_conf)
        
        # 尝试计算 2D 坐标（用于可视化）
        try:
            Chem.Compute2DCoords(scaffold_mol)
        except Exception:
            pass
        
        return scaffold_mol
        
    except Exception as e:
        if logger:
            logger.warning(f'[_extract_scaffold_submol] 提取骨架失败: {e}')
        return None


def _resolve_scaffold_mask(
    scaffold_cfg: dict,
    ref_ligand_pos,
    ref_ligand_v,
    ligand_atom_mode: str,
    num_atoms: int,
    device,
) -> 'torch.Tensor':
    """
    根据配置解析骨架掩码（[num_atoms] float32 张量，1=骨架原子，0=非骨架原子）。

    支持六种定义方式（Murcko 与自定义并列入口）：
    - atom_indices  ：直接指定 0-based 原子索引列表
    - smarts / custom：通过 SMARTS 子结构匹配（custom 为 smarts 别名）
    - auto_murcko   ：Bemis-Murcko 分解自动识别主骨架（环系 + 链接体）
    - auto_murcko_generic：通用 Murcko 骨架（全碳骨架，对杂环更鲁棒）
    - none          ：不约束任何原子（返回全零掩码）

    Args:
        scaffold_cfg: scaffold_optimization / scaffold_grow 配置字典。
        ref_ligand_pos: 参考配体坐标 [N,3] Tensor。
        ref_ligand_v: 参考配体原子类型 [N] Tensor。
        ligand_atom_mode: 原子编码模式（用于重建 RDKit Mol）。
        num_atoms: 原子总数。
        device: 计算设备。

    Returns:
        scaffold_mask: float32 Tensor [num_atoms]，1=骨架，0=非骨架。
    """
    logger = None
    source = str(scaffold_cfg.get('scaffold_source', 'none') or 'none').strip().lower()
    if source == 'custom':
        source = 'smarts'
    mask_np = np.zeros(num_atoms, dtype=np.float32)

    def _try_get_mol():
        """从坐标和原子类型创建基本 RDKit 分子（使用距离阈值构建键，保留环结构）"""
        try:
            pos_np = ref_ligand_pos.detach().cpu().numpy() if hasattr(ref_ligand_pos, 'detach') else np.array(ref_ligand_pos)
            v_np = ref_ligand_v.detach().cpu().numpy() if hasattr(ref_ligand_v, 'detach') else np.array(ref_ligand_v)

            # 直接创建基本分子，避免使用可能有bug的重建函数
            mol = Chem.RWMol()
            atomic_nums = []

            # 典型共价键长度上界 (Å)，超过此距离不建键
            COVALENT_RADIUS = {6: 0.77, 7: 0.75, 8: 0.73, 9: 0.71, 15: 1.10, 16: 1.02, 17: 0.99, 35: 1.14, 53: 1.33}
            BOND_DIST_FACTOR = 1.3  # 容差系数

            for i in range(len(v_np)):
                atom_type = int(v_np[i])
                if atom_type == 0: atomic_num = 6
                elif atom_type == 1: atomic_num = 7
                elif atom_type == 2: atomic_num = 8
                elif atom_type == 3: atomic_num = 9
                elif atom_type == 4: atomic_num = 15
                elif atom_type == 5: atomic_num = 16
                elif atom_type == 6: atomic_num = 17
                elif atom_type == 7: atomic_num = 35
                elif atom_type == 8: atomic_num = 53
                else: atomic_num = 6

                atomic_nums.append(atomic_num)
                atom = Chem.Atom(atomic_num)
                mol.AddAtom(atom)

            # 基于 3D 距离构建键（保留环结构）
            n = len(atomic_nums)
            # 最大允许价态
            MAX_VALENCE = {6: 4, 7: 3, 8: 2, 9: 1, 15: 5, 16: 6, 17: 1, 35: 1, 53: 1}
            bond_pairs = []
            for i in range(n):
                ri = COVALENT_RADIUS.get(atomic_nums[i], 0.77)
                for j in range(i + 1, n):
                    rj = COVALENT_RADIUS.get(atomic_nums[j], 0.77)
                    threshold = (ri + rj) * BOND_DIST_FACTOR
                    dx = pos_np[i][0] - pos_np[j][0]
                    dy = pos_np[i][1] - pos_np[j][1]
                    dz = pos_np[i][2] - pos_np[j][2]
                    dist = (dx*dx + dy*dy + dz*dz) ** 0.5
                    if dist < threshold:
                        bond_pairs.append((i, j, dist))
            # 按距离排序，优先添加短键
            bond_pairs.sort(key=lambda x: x[2])
            atom_bond_count = [0] * n
            for i, j, dist in bond_pairs:
                max_i = MAX_VALENCE.get(atomic_nums[i], 4)
                max_j = MAX_VALENCE.get(atomic_nums[j], 4)
                if atom_bond_count[i] < max_i and atom_bond_count[j] < max_j:
                    try:
                        mol.AddBond(i, j, Chem.BondType.SINGLE)
                        atom_bond_count[i] += 1
                        atom_bond_count[j] += 1
                    except:
                        pass

            final_mol = mol.GetMol()

            # 设置 3D 坐标
            conf = Chem.Conformer(n)
            for i, (x, y, z) in enumerate(pos_np):
                conf.SetAtomPosition(i, (float(x), float(y), float(z)))
            final_mol.AddConformer(conf)

            if logger:
                logger.info(f'[_try_get_mol] 成功创建分子: {final_mol.GetNumAtoms()} 原子, {final_mol.GetNumBonds()} 键')
            return final_mol
            
        except Exception as e:
            if logger:
                logger.warning(f'[_try_get_mol] 创建基本分子失败: {e}')
            return None

    if source == 'none':
        pass

    elif source == 'atom_indices':
        indices = scaffold_cfg.get('scaffold_atom_indices') or []
        for idx in indices:
            if 0 <= int(idx) < num_atoms:
                mask_np[int(idx)] = 1.0

    elif source == 'smarts':
        smarts = scaffold_cfg.get('scaffold_smarts')
        if smarts is not None:
            try:
                pattern = Chem.MolFromSmarts(smarts)
                mol = _try_get_mol()
                if mol is not None and pattern is not None:
                    matches = mol.GetSubstructMatches(pattern)
                    for match in matches:
                        for atom_idx in match:
                            if 0 <= atom_idx < num_atoms:
                                mask_np[atom_idx] = 1.0
            except Exception:
                pass

    elif source in ('auto_murcko', 'auto_murcko_generic'):
        try:
            mol = _try_get_mol()
            if mol is not None:
                if logger:
                    logger.info(f'[ScaffoldMask] 重建分子成功: {mol.GetNumAtoms()} 个原子')

                # 优先使用 Murcko 骨架（Bemis-Murcko 分解，保留环系+链接体，移除侧链）
                indices = _detect_murcko_scaffold_indices(mol)
                if logger:
                    logger.info(f'[ScaffoldMask] Murcko 骨架: {len(indices)} 个骨架原子')

                # Murcko 失败时回退到环检测法
                if not indices:
                    if source == 'auto_murcko':
                        indices = _detect_ring_based_scaffold_indices(mol, logger)
                        if logger:
                            logger.info(f'[ScaffoldMask] 环检测回退: {len(indices)} 个骨架原子')
                        if not indices:
                            indices = _detect_generic_murcko_scaffold_indices(mol)
                            if logger:
                                logger.info(f'[ScaffoldMask] 通用 Murcko 回退: {len(indices)} 个骨架原子')
                    else:
                        indices = _detect_generic_murcko_scaffold_indices(mol)
                        if logger:
                            logger.info(f'[ScaffoldMask] 通用 Murcko: {len(indices)} 个骨架原子')

                for atom_idx in indices:
                    if 0 <= atom_idx < num_atoms:
                        mask_np[atom_idx] = 1.0
            else:
                if logger:
                    logger.warning('[ScaffoldMask] 重建分子失败，无法提取骨架')
        except Exception as e:
            if logger:
                logger.warning(f'[ScaffoldMask] 骨架提取异常: {e}')
            pass

    return torch.tensor(mask_np, dtype=torch.float32, device=device)


def _get_scaffold_mode_cfg(config) -> 'tuple[dict, str]':
    """
    从统一 scaffold 配置节中读取并展平当前 mode 的参数。

    统一结构：
        scaffold:
          enable: ...
          mode: evolve | grow
          <shared params>        # 顶层共用参数
          evolve: <evolve-only>  # 进化专属，会覆盖顶层同名键
          grow:   <grow-only>    # 生长专属，会覆盖顶层同名键

    返回：(merged_cfg, mode_str)
    merged_cfg 是将 scaffold.<mode> 子节覆盖进顶层后的合并字典，
    使各功能函数可以直接用 cfg.get('key') 获取值而无需关心 mode。
    """
    base = dict(config.sample.get('scaffold', {}))
    mode = base.get('mode', 'evolve')
    sub = base.get(mode)
    if sub is not None:
        merged = {**base, **dict(sub)}
    else:
        merged = base
    return merged, mode


def _sample_n_extra_atoms(
    grow_cfg: dict,
    protein_pos,
    n_scaffold_atoms: int,
    model_num_timesteps: int,
    sample_idx: int = 0,
    retry: int = 0,
) -> int:
    """
    根据策略决定在骨架之外新生成的原子数（N_extra）。

    策略（grow_cfg.n_extra_mode）：
    - prior_minus_scaffold：从口袋尺寸先验采样总原子数，减去 n_scaffold_atoms
      → 新分子大小与口袋匹配，骨架大小已占用一部分配额
    - pocket_prior：完全由口袋先验决定（不减去骨架原子数）
      → 适合"在骨架基础上大幅扩展"场景
    - fixed：固定数量（n_extra_fixed）
    - range：在 [n_extra_min, n_extra_max] 内均匀随机采样
    - ligand_size_normal_minus_scaffold：从已知配体重原子数 Normal(μ,σ) 离散采样总原子数，再减骨架

    结果均被 clamp 到 [n_extra_min_clamp, n_extra_max_clamp]。

    Args:
        grow_cfg: scaffold_grow 配置字典。
        protein_pos: 蛋白口袋原子坐标 Tensor [N_protein, 3]（用于先验采样）。
        n_scaffold_atoms: 骨架原子数。
        model_num_timesteps: 模型扩散步数（未使用，预留）。
        sample_idx: 样本序号（ligand_size 模式可复现；与 murcko 位点 RNG 错开 +1）。
        retry: 同槽位补足重试次数（0=首次，与历史一致）。

    Returns:
        n_extra: 要新生成的原子数（>= 1）。
    """
    from diffdynamic.sampling.scaffold.placement import _murcko_rng_offset

    mode = grow_cfg.get('n_extra_mode', 'pocket_prior')
    min_clamp = int(grow_cfg.get('n_extra_min_clamp', 1))
    max_clamp = int(grow_cfg.get('n_extra_max_clamp', 40))

    if mode in ('prior_minus_scaffold', 'pocket_prior'):
        pos_np = protein_pos.detach().cpu().numpy() if hasattr(protein_pos, 'detach') else np.array(protein_pos)
        pocket_size = atom_num.get_space_size(pos_np)
        if np.isnan(pocket_size) or np.isinf(pocket_size) or pocket_size <= 0:
            pocket_size = 30.0
        # pocket prior 本身无固定 seed；补足重试时用偏移扰动采样
        if int(retry) > 0:
            _rng = np.random.default_rng(
                int(grow_cfg.get('seed', 42)) + _murcko_rng_offset(sample_idx, retry) + 17
            )
            # 重复采样取第 retry 次风格的扰动：对 pocket_size 加小噪声再采样
            pocket_size = float(pocket_size) * float(_rng.uniform(0.92, 1.08))
        count_rng = (np.random.default_rng(
            int(grow_cfg.get('seed', 42)) + _murcko_rng_offset(sample_idx, retry) + 17
        ) if grow_cfg.get('web_input_contract') else None)
        total = int(atom_num.sample_atom_num(pocket_size, rng=count_rng)
                    if count_rng is not None else atom_num.sample_atom_num(pocket_size))
        if mode == 'prior_minus_scaffold':
            n_extra = total - n_scaffold_atoms
        else:
            n_extra = total

    elif mode == 'fixed':
        n_extra = int(grow_cfg.get('n_extra_fixed', 8))

    elif mode == 'range':
        lo = int(grow_cfg.get('n_extra_min', 3))
        hi = int(grow_cfg.get('n_extra_max', 20))
        if int(retry) > 0 or grow_cfg.get('web_input_contract'):
            _rng = np.random.default_rng(
                int(grow_cfg.get('seed', 42)) + _murcko_rng_offset(sample_idx, retry) + 19
            )
            n_extra = int(_rng.integers(max(lo, 1), max(hi, lo) + 1))
        else:
            n_extra = int(np.random.randint(max(lo, 1), max(hi, lo) + 1))

    elif mode in (
        'ligand_size_normal_minus_scaffold',
        'ligand_normal_minus_scaffold',
        'active_size_normal_minus_scaffold',
    ):
        # 从已知配体重原子数的正态离散分布采样总原子数，再减骨架。
        # 独立 seed 流：base + offset + 1，避免消耗 murcko 位点 RNG。
        mu = float(grow_cfg.get('ligand_size_mean', grow_cfg.get('ligand_atom_mean', 35)))
        sigma = float(grow_cfg.get('ligand_size_std', grow_cfg.get('ligand_atom_std', 4)))
        size_min = int(grow_cfg.get('ligand_size_min', max(1, int(round(mu - 3 * sigma)))))
        size_max = int(grow_cfg.get('ligand_size_max', int(round(mu + 3 * sigma))))
        base = int(grow_cfg.get('seed', 42))
        size_rng = np.random.default_rng(base + _murcko_rng_offset(sample_idx, retry) + 1)
        if sigma <= 0:
            total = int(round(mu))
        else:
            total = int(round(float(size_rng.normal(mu, sigma))))
        total = int(np.clip(total, size_min, size_max))
        n_extra = total - int(n_scaffold_atoms)

    elif mode in ('reference_size_prior', 'reference_n_extra_prior'):
        values = [int(v) for v in grow_cfg.get('reference_size_values', [])]
        weights = [float(v) for v in grow_cfg.get('reference_size_weights', [])]
        if not values or len(values) != len(weights) or sum(weights) <= 0:
            n_extra = int(grow_cfg.get('n_extra_fixed', 8))
        else:
            size_rng = np.random.default_rng(
                int(grow_cfg.get('seed', 42))
                + _murcko_rng_offset(sample_idx, retry)
                + 23
            )
            n_extra = int(size_rng.choice(values, p=np.asarray(weights) / sum(weights)))

    else:
        n_extra = int(grow_cfg.get('n_extra_fixed', 8))

    return int(np.clip(n_extra, min_clamp, max_clamp))


def _sample_reference_exact_record(
    grow_cfg: dict,
    sample_idx: int,
    retry: int = 0,
) -> dict:
    """Select one scaffold reference count/type record for a sample slot.

    This mode is intentionally opt-in and is consumed only by the
    ``scaffold.dynamic_locked`` path.  The selected record contains no graph
    or coordinates; it couples the exact heavy-atom count and exchangeable
    element/aromatic multisets to fixed-scaffold exit slots.
    """
    from diffdynamic.sampling.scaffold.placement import _murcko_rng_offset

    records = [
        dict(record)
        for record in (grow_cfg.get('reference_exact_records') or [])
    ]
    if not records:
        raise ValueError(
            'reference_exact_record mode requires reference_exact_records'
        )
    weights = np.asarray(
        [max(float(record.get('weight', 1.0)), 0.0) for record in records],
        dtype=np.float64,
    )
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        raise ValueError(
            'reference_exact_record mode has no positive finite record weights'
        )
    rng = np.random.default_rng(
        int(grow_cfg.get('seed', 42))
        + _murcko_rng_offset(sample_idx, retry)
        + 29
    )
    selected = int(rng.choice(len(records), p=weights / weights.sum()))
    record = records[selected]
    n_extra = int(record.get('n_extra', 0))
    type_counts = record.get('type_counts') or {}
    if n_extra <= 0 or sum(int(value) for value in type_counts.values()) != n_extra:
        raise ValueError(
            f'invalid selected exact reference record: {record}'
        )
    site_counts = record.get('site_counts') or {}
    if site_counts and sum(int(value) for value in site_counts.values()) != n_extra:
        raise ValueError(
            f'invalid selected exact reference site allocation: {record}'
        )
    site_type_counts = record.get('site_type_counts') or {}
    if site_type_counts:
        aggregate_types: dict[str, int] = {}
        for raw_slot, raw_counts in site_type_counts.items():
            slot = str(int(raw_slot))
            if slot not in {str(int(key)) for key in site_counts}:
                raise ValueError(
                    f'exact reference site type slot is absent: {record}'
                )
            if sum(int(value) for value in raw_counts.values()) != int(
                site_counts.get(slot, site_counts.get(int(slot), 0))
            ):
                raise ValueError(
                    f'exact reference site type count disagrees: {record}'
                )
            for key, value in raw_counts.items():
                aggregate_types[str(key)] = (
                    aggregate_types.get(str(key), 0) + int(value)
                )
        if aggregate_types != {
            str(key): int(value) for key, value in type_counts.items()
        }:
            raise ValueError(
            f'exact reference total/site type multisets disagree: {record}'
            )
    attachment_type_counts = record.get('site_attachment_type_counts') or {}
    for raw_slot, raw_counts in attachment_type_counts.items():
        slot = str(int(raw_slot))
        slot_counts = site_type_counts.get(slot)
        if slot_counts is None:
            raise ValueError(
                f'exact reference attachment type slot is absent: {record}'
            )
        for key, value in raw_counts.items():
            if int(value) > int(slot_counts.get(str(key), 0)):
                raise ValueError(
                    'exact reference attachment type count exceeds its slot '
                    f'type count: {record}'
                )
    site_centers = record.get('site_centers_world') or {}
    for raw_slot, raw_center in site_centers.items():
        center = np.asarray(raw_center, dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError(
                f'invalid exact reference site centroid: {record}'
            )
    record['_selection_index'] = selected
    return record


def _exact_record_type_hint_keys(
    record: dict,
    rng: np.random.Generator,
    site_allocation: list[dict] | None = None,
) -> list[str]:
    """Expand exact type multisets in the selected scaffold-slot order."""
    site_type_counts = record.get('site_type_counts') or {}
    if site_type_counts:
        if not site_allocation:
            raise ValueError(
                'site-specific exact types require a realized site allocation'
            )
        keys: list[str] = []
        observed_slots: list[str] = []
        for allocation in site_allocation:
            slot = str(int(allocation['profile_slot']))
            observed_slots.append(slot)
            counts = site_type_counts.get(slot)
            if not counts:
                raise ValueError(
                    f'exact reference record has no types for slot {slot}: '
                    f'{record}'
                )
            slot_keys: list[str] = []
            for raw_key, raw_count in sorted(counts.items()):
                slot_keys.extend([str(raw_key)] * int(raw_count))
            if len(slot_keys) != int(allocation['count']):
                raise ValueError(
                    f'exact reference slot {slot} type count disagrees with '
                    f'placement: {record}'
                )
            # Put the atom classes observed directly at the scaffold exit at
            # the front of the slot's exchangeable list.  The first corridor
            # positions are then occupied by chemically attachable head
            # classes without importing the reference bond graph.
            attachment_counts = {
                str(key): int(value)
                for key, value in (
                    (record.get('site_attachment_type_counts') or {}).get(slot)
                    or {}
                ).items()
            }
            slot_counter = Counter(slot_keys)
            head_keys: list[str] = []
            for key, count in sorted(attachment_counts.items()):
                take = min(int(count), int(slot_counter.get(key, 0)))
                head_keys.extend([key] * take)
                slot_counter[key] -= take
            remainder: list[str] = []
            for key, count in sorted(slot_counter.items()):
                remainder.extend([str(key)] * int(count))
            rng.shuffle(head_keys)
            rng.shuffle(remainder)
            keys.extend(head_keys + remainder)
        if set(observed_slots) != {str(int(slot)) for slot in site_type_counts}:
            raise ValueError(
                f'exact reference placement omitted a typed slot: {record}'
            )
        if len(keys) != int(record.get('n_extra', 0)):
            raise ValueError(
                f'exact site type multisets do not match n_extra: {record}'
            )
        return keys

    keys: list[str] = []
    for raw_key, raw_count in sorted(
        (record.get('type_counts') or {}).items()
    ):
        count = int(raw_count)
        if count <= 0:
            continue
        keys.extend([str(raw_key)] * count)
    rng.shuffle(keys)
    if len(keys) != int(record.get('n_extra', 0)):
        raise ValueError(
            'exact reference type multiset does not match n_extra: '
            f'{record}'
        )
    return keys


def _apply_exact_record_site_centers(
    attachment_sites: list[dict],
    record: dict | None,
) -> list[dict]:
    """Return exit sites with record-level coarse Gaussian means applied.

    Only one centroid per fixed scaffold exit is consumed.  The returned
    sites contain no target-side atom coordinates or graph data, and added
    coordinates remain unmasked after this initialization.
    """
    if not record:
        return attachment_sites
    raw_centers = record.get('site_centers_world') or {}
    fallback = record.get('side_centroid_world')
    if not raw_centers and fallback is None:
        return attachment_sites
    fallback_array = (
        np.asarray(fallback, dtype=np.float64)
        if fallback is not None else None
    )
    updated: list[dict] = []
    for source_site in attachment_sites or []:
        site = dict(source_site)
        slot = site.get('profile_slot')
        center = raw_centers.get(str(int(slot))) if slot is not None else None
        if center is None:
            center_array = fallback_array
        else:
            center_array = np.asarray(center, dtype=np.float64)
        if center_array is None or center_array.shape != (3,):
            updated.append(site)
            continue
        anchor = np.asarray(site.get('anchor_pos'), dtype=np.float64)
        offset = center_array - anchor
        distance = float(np.linalg.norm(offset))
        if distance <= 1e-8 or not np.isfinite(offset).all():
            updated.append(site)
            continue
        site['pocket_center_offset'] = offset.tolist()
        site['pocket_center_distance'] = distance
        site['pocket_center_source'] = (
            'exact_record_scaffold_aligned_coarse_centroid'
        )
        updated.append(site)
    return updated


def _pin_place_cfg_to_n_extra(place_cfg: dict, n_extra: int, grow_cfg: dict) -> dict:
    """当 n_extra 已由 ligand-size 正态等精确采样时，把位点 budget clamp 钉死为该值。

    sequential_random / random_per_site 默认会在 [min,max] 内再随机 budget，
    不钉死则无法落实「总原子数 = 配体正态采样」的意图。
    """
    mode = str((grow_cfg or {}).get('n_extra_mode', '')).strip().lower()
    if mode not in (
        'ligand_size_normal_minus_scaffold',
        'ligand_normal_minus_scaffold',
        'active_size_normal_minus_scaffold',
        'fixed',
    ):
        return place_cfg
    cfg = dict(place_cfg)
    n = max(1, int(n_extra))
    cfg['n_extra_min_clamp'] = n
    cfg['n_extra_max_clamp'] = n
    return cfg


def _indices_from_custom_scaffold(ref_mol, sc_cfg, logger=None):
    """自定义骨架：smarts/custom（SMARTS）或 atom_indices；与 Murcko 并列。"""
    source = str(sc_cfg.get('scaffold_source', '') or '').strip().lower()
    if source == 'custom':
        source = 'smarts'
    indices = []

    if source == 'atom_indices':
        raw = sc_cfg.get('scaffold_atom_indices') or []
        n = ref_mol.GetNumAtoms() if ref_mol is not None else 0
        for idx in raw:
            i = int(idx)
            if 0 <= i < n:
                indices.append(i)
        indices = sorted(set(indices))
        if logger:
            logger.info(f'[ScaffoldExtract] atom_indices 骨架: {len(indices)} 个原子')
        return indices

    if source == 'smarts':
        smarts = sc_cfg.get('scaffold_smarts')
        if not smarts or ref_mol is None:
            if logger:
                logger.warning('[ScaffoldExtract] smarts/custom 缺少 scaffold_smarts 或分子')
            return []
        try:
            pattern = Chem.MolFromSmarts(str(smarts))
            if pattern is None:
                if logger:
                    logger.warning(f'[ScaffoldExtract] 无效 SMARTS: {smarts}')
                return []
            matches = ref_mol.GetSubstructMatches(pattern)
            for match in matches:
                indices.extend(int(a) for a in match)
            indices = sorted(set(indices))
            if logger:
                logger.info(
                    f'[ScaffoldExtract] smarts/custom 命中 {len(matches)} 处，'
                    f'骨架原子 {len(indices)} | SMARTS={smarts}'
                )
        except Exception as e:
            if logger:
                logger.warning(f'[ScaffoldExtract] smarts/custom 匹配失败: {e}')
            return []
        return indices

    return []


def _extract_scaffold_from_reference(data, sc_cfg, ligand_atom_mode,
                                     ref_ligand_pos, ref_ligand_v, log_ref_v,
                                     n_scaffold_ref, device, config=None, logger=None):
    """从参考配体提取骨架原子索引（共享逻辑，供 scaffold_grow / scaffold_dynamic_locked 复用）。

    流程：优先从原始配体文件加载 RDKit Mol →
    Murcko 与自定义（smarts/custom/atom_indices）并列解析；
    失败则按来源回退（自定义走 _resolve_scaffold_mask；Murcko 可坐标重建）；
    若仍为空，强制使用全部原子作为骨架。

    Returns:
        (scaffold_indices, ref_mol, ref_ligand_name, n_scaffold)
    """
    if sc_cfg.get('web_input_contract'):
        ref_mol = reference_molecule_from_data(data)
        indices = resolve_scaffold_indices(ref_mol, sc_cfg)
        name = Path(getattr(data, 'ligand_filename', 'reference')).stem
        return indices, ref_mol, name, len(indices)
    # ---- 解析骨架掩码 ---------------------------------------------------------
    scaffold_indices = []
    ref_mol = None
    ref_ligand_name = None
    scaffold_source = str(sc_cfg.get('scaffold_source', 'auto_murcko') or 'auto_murcko').strip().lower()
    is_custom = scaffold_source in ('smarts', 'custom', 'atom_indices')

    # 尝试从数据中获取原始参考分子的 RDKit Mol 对象
    if hasattr(data, 'ligand_filename') and data.ligand_filename:
        ref_ligand_path = data.ligand_filename
        ref_ligand_name = Path(ref_ligand_path).stem

        if logger:
            logger.info(f'[ScaffoldExtract] 尝试加载配体文件: {ref_ligand_path}')
            logger.info(f'[ScaffoldExtract] 文件存在性检查: {os.path.exists(ref_ligand_path)}')

        # 如果文件不存在，尝试从数据集根目录解析
        if not os.path.exists(ref_ligand_path):
            dataset_root = None
            try:
                if config is not None:
                    dataset_root = config.data.path
            except Exception:
                pass
            if not dataset_root:
                try:
                    if config is not None and hasattr(config, 'sample'):
                        dataset_root = getattr(config.sample, 'data', {}).get('path', None)
                except Exception:
                    pass

            # 如果路径是相对路径，尝试在不同位置查找
            possible_paths = [
                Path(ref_ligand_path),
            ]
            if dataset_root:
                possible_paths.append(Path(dataset_root) / ref_ligand_path)
            possible_paths += [
                Path('./data/crossdocked_pocket10_test_only') / ref_ligand_path,
                Path('/workspace/data/crossdocked_v1.1_rmsd1.0') / ref_ligand_path,
                Path('/workspace/data/crossdocked_v1.1_rmsd1.0_pocket10') / ref_ligand_path,
            ]

            for try_path in possible_paths:
                if try_path.exists():
                    ref_ligand_path = str(try_path)
                    if logger:
                        logger.info(f'[ScaffoldExtract] 找到配体文件: {ref_ligand_path}')
                    break

        if os.path.exists(ref_ligand_path):
            try:
                ref_mol = load_reference_molecule(ref_ligand_path)
                if ref_mol is not None:
                    if sc_cfg.get('web_input_contract'):
                        scaffold_indices = resolve_scaffold_indices(ref_mol, sc_cfg)
                        return scaffold_indices, ref_mol, ref_ligand_name, len(scaffold_indices)
                    try:
                        Chem.SanitizeMol(ref_mol)
                        ri = ref_mol.GetRingInfo()
                        if ri is None or ri.NumRings() < 0:
                            Chem.GetSSSR(ref_mol)
                        if logger:
                            logger.info(f'[ScaffoldExtract] 分子环信息初始化成功: {ref_mol.GetNumAtoms()} 原子, {ref_mol.GetRingInfo().NumRings()} 个环')
                    except Exception as e:
                        if logger:
                            logger.warning(f'[ScaffoldExtract] 初始化 RingInfo 失败: {e}')

                    if is_custom:
                        scaffold_indices = _indices_from_custom_scaffold(ref_mol, sc_cfg, logger=logger)
                        if logger and scaffold_indices:
                            logger.info(
                                f'[ScaffoldExtract] 从原始配体文件自定义骨架: '
                                f'{len(scaffold_indices)} 个原子 (source={scaffold_source})'
                            )
                        elif logger:
                            logger.warning('[ScaffoldExtract] 自定义骨架匹配返回空列表')
                    elif scaffold_source in ('auto_murcko', 'auto_murcko_generic'):
                        if logger:
                            logger.info('[ScaffoldExtract] 尝试 Murcko 骨架提取...')
                        indices = _detect_murcko_scaffold_indices(ref_mol)
                        if logger:
                            logger.info(f'[ScaffoldExtract] Murcko 检测到 {len(indices)} 个骨架原子')

                        if not indices:
                            if logger:
                                logger.info('[ScaffoldExtract] Murcko 失败，尝试环检测骨架提取...')
                            indices = _detect_ring_based_scaffold_indices(ref_mol, logger)
                            if logger:
                                logger.info(f'[ScaffoldExtract] 环检测到 {len(indices)} 个骨架原子')

                            if not indices and scaffold_source == 'auto_murcko':
                                indices = _detect_generic_murcko_scaffold_indices(ref_mol)
                                if logger:
                                    logger.info(f'[ScaffoldExtract] 通用 Murcko 检测到 {len(indices)} 个骨架原子')

                        scaffold_indices = indices
                        if logger and indices:
                            logger.info(f'[ScaffoldExtract] 从原始配体文件提取骨架: {len(indices)} 个原子')
                        elif logger:
                            logger.warning('[ScaffoldExtract] 所有骨架提取方法均返回空列表')
            except Exception as e:
                if sc_cfg.get('web_input_contract'):
                    raise
                if logger:
                    logger.warning(f'[ScaffoldExtract] 从文件提取骨架失败: {e}')

    # 如果无法从文件获取骨架，回退
    if not scaffold_indices:
        if is_custom:
            if logger:
                logger.info('[ScaffoldExtract] 自定义骨架失败，回退到 _resolve_scaffold_mask...')
            scaffold_mask = _resolve_scaffold_mask(
                sc_cfg, ref_ligand_pos, ref_ligand_v, ligand_atom_mode, n_scaffold_ref, device
            )
            scaffold_mask_np = scaffold_mask.cpu().numpy()
            scaffold_indices = np.where(scaffold_mask_np > 0.5)[0].tolist()
        else:
            if logger:
                logger.info('[ScaffoldExtract] 从文件提取骨架失败，尝试从坐标重建...')

            from diffdynamic.chemistry import reconstruct
            try:
                pos_np = ref_ligand_pos.detach().cpu().numpy() if hasattr(ref_ligand_pos, 'detach') else np.array(ref_ligand_pos)
                v_np = ref_ligand_v.detach().cpu().numpy() if hasattr(ref_ligand_v, 'detach') else np.array(ref_ligand_v)

                reconstructed_mol = reconstruct.reconstruct_from_type_indices(
                    pos_np, v_np, ligand_atom_mode=ligand_atom_mode,
                )

                if reconstructed_mol is not None and logger:
                    logger.info(f'[ScaffoldExtract] 坐标重建分子成功: {reconstructed_mol.GetNumAtoms()} 个原子')

                if reconstructed_mol is not None:
                    if logger:
                        logger.info('[ScaffoldExtract] 尝试对重建分子使用 Murcko 骨架提取...')
                    indices = _detect_murcko_scaffold_indices(reconstructed_mol)
                    if logger:
                        logger.info(f'[ScaffoldExtract] Murcko 检测到 {len(indices)} 个骨架原子')

                    if not indices:
                        indices = _detect_ring_based_scaffold_indices(reconstructed_mol, logger)
                        if logger:
                            logger.info(f'[ScaffoldExtract] 环检测到 {len(indices)} 个骨架原子')

                    if indices:
                        scaffold_indices = indices
                        if logger:
                            logger.info(f'[ScaffoldExtract] 从重建分子提取骨架: {len(indices)} 个原子')
                    else:
                        if logger:
                            logger.info('[ScaffoldExtract] 骨架提取失败，回退到 _resolve_scaffold_mask...')
                        scaffold_mask = _resolve_scaffold_mask(
                            sc_cfg, ref_ligand_pos, ref_ligand_v, ligand_atom_mode, n_scaffold_ref, device
                        )
                        scaffold_mask_np = scaffold_mask.cpu().numpy()
                        scaffold_indices = np.where(scaffold_mask_np > 0.5)[0].tolist()
                else:
                    if logger:
                        logger.warning('[ScaffoldExtract] 坐标重建分子失败，使用 _resolve_scaffold_mask...')
                    scaffold_mask = _resolve_scaffold_mask(
                        sc_cfg, ref_ligand_pos, ref_ligand_v, ligand_atom_mode, n_scaffold_ref, device
                    )
                    scaffold_mask_np = scaffold_mask.cpu().numpy()
                    scaffold_indices = np.where(scaffold_mask_np > 0.5)[0].tolist()
            except Exception as e:
                if logger:
                    logger.warning(f'[ScaffoldExtract] 坐标重建过程失败: {e}')
                scaffold_mask = _resolve_scaffold_mask(
                    sc_cfg, ref_ligand_pos, ref_ligand_v, ligand_atom_mode, n_scaffold_ref, device
                )
                scaffold_mask_np = scaffold_mask.cpu().numpy()
                scaffold_indices = np.where(scaffold_mask_np > 0.5)[0].tolist()

    n_scaffold = len(scaffold_indices)

    if (
        n_scaffold > 0
        and bool(sc_cfg.get('preserve_to_amine', False))
    ):
        if ref_mol is None:
            raise RuntimeError(
                '[ScaffoldExtract] preserve_to_amine requires the original '
                'reference molecule to be available'
            )
        scaffold_indices, preserved_info = extend_scaffold_to_first_amine(
            ref_mol, scaffold_indices
        )
        if preserved_info is None:
            raise RuntimeError(
                '[ScaffoldExtract] preserve_to_amine could not find a '
                'non-aromatic N outside the scaffold'
            )
        n_scaffold = len(scaffold_indices)
        if logger:
            logger.info(
                '[ScaffoldExtract] preserve_to_amine: fixed prefix +%d '
                'atoms (%s), scaffold=%d'
                % (
                    len(preserved_info['path_atom_indices']),
                    '-'.join(preserved_info['path_symbols']),
                    n_scaffold,
                )
            )

    if n_scaffold == 0:
        if logger:
            logger.warning('[ScaffoldExtract] 未找到骨架原子，将使用所有原子作为骨架（强制固定整个分子）')
        scaffold_indices = list(range(n_scaffold_ref))
        n_scaffold = n_scaffold_ref
        if logger:
            logger.info(f'[ScaffoldExtract] 强制使用所有 {n_scaffold} 个原子作为骨架')

    if logger:
        logger.info(
            f'[ScaffoldExtract] 骨架原子: {n_scaffold}/{n_scaffold_ref} '
            f'(来源: {sc_cfg.get("scaffold_source", "auto_murcko")})'
        )

    return scaffold_indices, ref_mol, ref_ligand_name, n_scaffold
