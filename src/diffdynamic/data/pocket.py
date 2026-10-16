"""Data pocket responsibilities for ligand sampling."""

from __future__ import annotations

from diffdynamic.data.graph import ProteinLigandData
from diffdynamic.data.graph import torchify_dict
from pathlib import Path
from diffdynamic.data.protein import PDBProtein
from diffdynamic.data.protein import parse_sdf_file
import numpy as np
import os


def resolve_and_set_absolute_protein_path(data, dataset_root=None, protein_root=None, logger=None):
    """将 `data.protein_filename` 解析为磁盘上存在的绝对 PDB 路径。

    与 `evaluate_pt_with_correct_reconstruct` / `VinaDockingTask.from_generated_mol` 一致：
    1) 已是可访问文件则规范为绝对路径；
    2) 相对路径则相对训练数据根目录 ``dataset_root``（即检查点里 ``data.path``）拼接；
    3) 仍失败则按 CrossDocked 惯例，用 ``protein_root`` + ``dirname(ligand_filename)`` +
       ``basename(ligand_filename)[:10] + '.pdb'`` 推断受体（同 docking_vina.from_generated_mol）；
    4) 再尝试在各 root 下直接拼接 ``protein_filename`` 相对路径。

    Args:
        data: ProteinLigandData
        dataset_root: 数据集原始目录（含 index / 口袋 PDB 的父目录）
        protein_root: 与评估时 ``--protein_root`` 相同，用于按配体路径推断受体
        logger: 可选
    """
    from diffdynamic.sampling.common import _log

    if data is None:
        return
    pf = getattr(data, 'protein_filename', None)
    lig_fn = getattr(data, 'ligand_filename', None)
    env_root = os.environ.get('PROTEIN_ROOT', '').strip() or None

    def _iter_roots():
        for r in (protein_root, env_root, dataset_root):
            if not r:
                continue
            try:
                yield str(Path(r).expanduser().resolve())
            except Exception:
                yield str(r)

    if pf:
        ps = str(pf)
        if os.path.isfile(ps):
            data.protein_filename = str(Path(ps).resolve())
            return
        if dataset_root:
            try:
                joined = str((Path(dataset_root).expanduser().resolve() / ps))
            except Exception:
                joined = os.path.normpath(os.path.join(str(dataset_root), ps))
            if os.path.isfile(joined):
                data.protein_filename = str(Path(joined).resolve())
                if logger:
                    logger.info(f'[Protein path] 相对路径已解析（相对 data.path）: {data.protein_filename}')
                return

    if lig_fn:
        lig_s = str(lig_fn)
        sub = os.path.join(os.path.dirname(lig_s), os.path.basename(lig_s)[:10] + '.pdb')
        for root in _iter_roots():
            alt = os.path.normpath(os.path.join(root, sub))
            if os.path.isfile(alt):
                data.protein_filename = str(Path(alt).resolve())
                if logger:
                    logger.info(f'[Protein path] 按配体+protein_root 推断受体: {data.protein_filename}')
                return

    if pf and not os.path.isabs(str(pf)):
        for root in _iter_roots():
            alt = os.path.normpath(os.path.join(root, str(pf)))
            if os.path.isfile(alt):
                data.protein_filename = str(Path(alt).resolve())
                if logger:
                    logger.info(f'[Protein path] 在数据根下找到: {data.protein_filename}')
                return

    if logger:
        _log(
            logger,
            'warning',
            f'[Protein path] 未解析到有效 PDB（protein_filename={pf!r}, ligand_filename={lig_fn!r}）。'
            f' Prudent/Vina 需传 --protein_root、环境变量 PROTEIN_ROOT，'
            f'或 sampling.yml 中 sample.dynamic.prudent.protein_root（与评估一致）。',
        )


def load_custom_pocket_data(protein_path, ligand_path=None, transform=None,
                            pocket_radius=10.0, logger=None):
    """
    从自定义蛋白/配体文件加载数据，用于生成分子。
    
    当提供配体时，自动将蛋白裁剪为配体周围 pocket_radius Å 的口袋，
    以匹配训练数据（crossdocked_v1.1_rmsd1.0_pocket10）的口袋尺度，
    避免因全蛋白输入导致生成碎片化分子。
    
    Args:
        protein_path: 蛋白 PDB 文件路径（必需，可以是全蛋白或预裁剪口袋）
        ligand_path: 参考配体 SDF/MOL2 文件路径（可选，用于自动裁剪口袋和 ref 模式原子数）
        transform: 特征转换管道（可选）
        pocket_radius: 口袋裁剪半径（Å），仅在提供配体时生效（默认: 10.0，匹配训练数据）
        logger: 日志记录器（可选）
    
    Returns:
        ProteinLigandData: 转换后的图数据对象
    """
    from diffdynamic.sampling.common import _log

    protein_path = Path(protein_path).resolve()
    if not protein_path.exists():
        raise FileNotFoundError(f'蛋白文件不存在: {protein_path}')
    
    protein_obj = PDBProtein(str(protein_path))
    full_atom_count = len(protein_obj.element)
    
    # 解析配体
    if ligand_path is not None:
        ligand_path = Path(ligand_path).resolve()
        if not ligand_path.exists():
            raise FileNotFoundError(f'配体文件不存在: {ligand_path}')
        ligand_dict = parse_sdf_file(str(ligand_path))
    else:
        ligand_dict = {
            'element': np.array([], dtype=np.int64),
            'pos': np.empty((0, 3), dtype=np.float32),
            'atom_feature': np.empty((0, 8), dtype=np.int64),
            'bond_index': np.empty((2, 0), dtype=np.int64),
            'bond_type': np.array([], dtype=np.int64),
        }
    
    # 自动口袋裁剪：当提供配体且 pocket_radius > 0 时，
    # 提取配体周围 pocket_radius Å 内的残基作为口袋
    if ligand_path is not None and pocket_radius > 0:
        selected_residues = protein_obj.query_residues_ligand(
            ligand_dict, radius=pocket_radius
        )
        if selected_residues:
            atom_indices = sorted(set(
                idx for res in selected_residues for idx in res['atoms']
            ))
            full_dict = protein_obj.to_dict_atom()
            pocket_dict = {
                'element': full_dict['element'][atom_indices],
                'molecule_name': full_dict.get('molecule_name', 'POCKET'),
                'pos': full_dict['pos'][atom_indices],
                'is_backbone': full_dict['is_backbone'][atom_indices],
                'atom_name': [full_dict['atom_name'][i] for i in atom_indices],
                'atom_to_aa_type': full_dict['atom_to_aa_type'][atom_indices],
            }
            pocket_atom_count = len(atom_indices)
            _log(logger, 'info',
                 f'自动裁剪口袋: {full_atom_count} → {pocket_atom_count} 原子 '
                 f'(配体周围 {pocket_radius}Å, {len(selected_residues)} 个残基)')
        else:
            pocket_dict = protein_obj.to_dict_atom()
            _log(logger, 'warning',
                 f'配体 {pocket_radius}Å 内未找到残基，使用完整蛋白 ({full_atom_count} 原子)。'
                 f'请检查蛋白和配体的坐标系是否一致。')
    else:
        pocket_dict = protein_obj.to_dict_atom()
        if full_atom_count > 500:
            _log(logger, 'warning',
                 f'蛋白原子数较多 ({full_atom_count})，可能不是裁剪过的口袋。'
                 f'建议提供参考配体 (--ligand_path) 以自动裁剪为结合位点口袋，'
                 f'或手动提取结合位点附近 10Å 的口袋 PDB。'
                 f'全蛋白输入可能导致生成碎片化分子。')
        elif logger:
            logger.info(f'蛋白原子数: {full_atom_count} (未提供配体，跳过口袋裁剪)')
    
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict=torchify_dict(ligand_dict),
    )
    data.protein_filename = str(protein_path)
    data.ligand_filename = str(ligand_path) if ligand_path is not None else 'N/A'
    assert data.protein_pos.size(0) > 0, '蛋白口袋为空'
    
    if transform is not None:
        data = transform(data)
    return data
