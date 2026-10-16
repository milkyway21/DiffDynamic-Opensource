"""Sampling pipelines scaffold_optimization responsibilities for ligand sampling."""

from __future__ import annotations

from pathlib import Path
from torch_geometric.transforms import Compose
from diffdynamic.data.protein import parse_sdf_file
import torch
import diffdynamic.data.transforms as trans


def run_scaffold_optimization(args, model, data, config, ligand_atom_mode, ligand_featurizer, logger, opt_cfg):
    """Run the scaffold optimization pipeline and return its result bundle."""
    from diffdynamic.sampling.refinement import apply_targetdiff_baseline_refine_to_sampling_result
    from diffdynamic.sampling.scaffold.evolve import scaffold_optimize_molecule

    config.sample.setdefault('scaffold', {})
    mol_path = getattr(args, 'molecule_path', None) or opt_cfg.get('molecule_path', None)
    if mol_path is not None:
        mol_path = Path(mol_path).resolve()
        if not mol_path.exists():
            raise FileNotFoundError(f'待优化分子文件不存在: {mol_path}')
        logger.info(f'[ScaffoldOptimization] 从独立文件加载待优化分子: {mol_path}')
        mol_dict = parse_sdf_file(str(mol_path))
        data.ligand_filename = str(mol_path)
        data.ligand_element = torch.tensor(mol_dict['element'], dtype=torch.long)
        data.ligand_pos = torch.tensor(mol_dict['pos'], dtype=torch.float32)
        data.ligand_bond_index = torch.tensor(mol_dict['bond_index'], dtype=torch.long)
        data.ligand_bond_type = torch.tensor(mol_dict['bond_type'], dtype=torch.long)
        mol_transform = Compose([ligand_featurizer, trans.FeaturizeLigandBond()])
        data = mol_transform(data)
    else:
        # 未指定 molecule_path，使用当前 data 中的配体（来自测试集或 ligand_path）
        use_test_set = getattr(args, 'use_test_set', False)
        data_id = getattr(args, 'data_id', None)
        if use_test_set and data_id is not None:
            logger.info(f'[ScaffoldOptimization] 使用测试集配体作为优化起点: data_id={data_id}')
        elif hasattr(data, 'ligand_filename') and data.ligand_filename:
            logger.info(f'[ScaffoldOptimization] 使用已加载配体作为优化起点: {data.ligand_filename}')
        else:
            logger.info('[ScaffoldOptimization] 使用已加载配体作为优化起点')

    scaff_out = scaffold_optimize_molecule(
        model=model,
        data=data,
        config=config,
        ligand_atom_mode=ligand_atom_mode,
        device=args.device,
        logger=logger,
    )
    result = {
        'data': data,
        'pred_ligand_pos': scaff_out['pos_list'],
        'pred_ligand_v': scaff_out['v_list'],
        'pred_ligand_pos_traj': scaff_out['pos_traj'],
        'pred_ligand_v_traj': scaff_out['v_traj'],
        'pred_ligand_log_v_traj': scaff_out['log_v_traj'],
        'time': scaff_out['time_list'],
        'meta': scaff_out['meta'],
        'mode': 'scaffold_optimization',
    }
    apply_targetdiff_baseline_refine_to_sampling_result(
        model, data, result, config, device=args.device, logger=logger
    )

    return result
