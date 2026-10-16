"""Sampling trajectory responsibilities for ligand sampling."""

from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from diffdynamic.io.paths import project_root
from pathlib import Path
from rdkit import Chem
from diffdynamic.io.identity import extract_protein_id
from diffdynamic.io.identity import generate_molecule_id
import numpy as np
import torch
import diffdynamic.chemistry.reconstruct as reconstruct
import diffdynamic.data.transforms as trans

REPO_ROOT = project_root()


def _coerce_time_indices_seq(seq):
    """将 meta 中的时间索引转为 list[int]（支持 list / numpy / torch）。"""
    if seq is None:
        return None
    if isinstance(seq, torch.Tensor):
        return [int(x) for x in seq.detach().cpu().flatten().tolist()]
    if isinstance(seq, np.ndarray):
        return [int(x) for x in seq.flatten().tolist()]
    return [int(x) for x in list(seq)]


def _get_time_indices_for_trajectory(result, config, n_steps_hint=None, sample_idx=0):
    """从 result 和 config 中提取与 pos_traj 对应的时间步序列。

    dynamic 下须合并 large_step 与 refine 两段索引；任一段缺失时仍合并另一段，避免 large（高 t）标签整体丢失。
    使用 records[sample_idx] 的 model_meta，避免多样本时错用第一条记录的调度。
    """
    from diffdynamic.sampling.configuration import get_time_boundary

    meta = result.get('meta', {})
    mode = result.get('mode', 'baseline')
    num_timesteps = getattr(config.sample, 'num_steps', 1000) or 1000

    if mode in ('dynamic', 'prudent'):
        base_ti = None
        # 1) unified：model_meta.large_step_time_indices + refine_time_indices（可仅一段有值）
        records = meta.get('records', [])
        if records:
            rec = records[sample_idx] if sample_idx < len(records) else records[-1]
            model_meta = rec.get('model_meta') or {}
            large_ti = _coerce_time_indices_seq(model_meta.get('large_step_time_indices'))
            refine_ti = _coerce_time_indices_seq(model_meta.get('refine_time_indices'))
            merged = []
            if large_ti is not None:
                merged.extend(large_ti)
            if refine_ti is not None:
                merged.extend(refine_ti)
            if merged:
                base_ti = merged
        # 2) legacy：refined_candidates[].time_indices（通常仅 refine）
        if base_ti is None:
            refined = meta.get('refined_candidates', [])
            if refined:
                rc = refined[sample_idx] if sample_idx < len(refined) else refined[-1]
                ti = rc.get('time_indices')
                if ti is not None:
                    base_ti = _coerce_time_indices_seq(ti)
        # 3) 回退：按配置生成近似序列（可能与实际轨迹步数不匹配）
        if base_ti is None:
            dynamic_cfg = config.sample.get('dynamic', {})
            time_boundary = get_time_boundary(dynamic_cfg, 750)
            base_ti = list(range(num_timesteps - 1, time_boundary - 1, -1)) + list(range(time_boundary, -1, -1))
        # 4) baseline_refine 追加
        baseline_refine_ti = meta.get('baseline_refine_time_indices')
        if baseline_refine_ti is not None:
            br = _coerce_time_indices_seq(baseline_refine_ti)
            if br is not None:
                return list(base_ti) + br
        return base_ti

    if mode in ('optimization', 'dynamic_then_optimization'):
        opt_ti = meta.get('optimization_time_indices')
        if opt_ti is not None:
            base_ti = _coerce_time_indices_seq(opt_ti)
        else:
            opt_cfg = config.sample.get('optimization', {})
            start_t = int(opt_cfg.get('start_t', 300))
            n_steps = n_steps_hint or max(10, start_t)
            base_ti = np.linspace(start_t, 0, n_steps, dtype=int).tolist()
        br_ti = meta.get('baseline_refine_time_indices')
        if br_ti is not None:
            br = _coerce_time_indices_seq(br_ti)
            if br is not None:
                return base_ti + br
        return base_ti

    n_steps = n_steps_hint or num_timesteps
    return list(range(num_timesteps - 1, num_timesteps - n_steps - 1, -1))


def _step_traj_as_int_list(atom_numbers):
    """供 reconstruct 使用：get_atomic_number_from_index 已返回 list，不能调用 .tolist()。"""
    if isinstance(atom_numbers, (list, tuple)):
        return [int(x) for x in atom_numbers]
    if isinstance(atom_numbers, np.ndarray):
        return atom_numbers.astype(np.int64).flatten().tolist()
    if isinstance(atom_numbers, torch.Tensor):
        return atom_numbers.detach().cpu().flatten().long().tolist()
    return [int(x) for x in atom_numbers.tolist()]


def _step_traj_as_bool_list(aromatic_flags):
    """供 reconstruct 使用的芳香标记列表。"""
    if aromatic_flags is None:
        return None
    if isinstance(aromatic_flags, (list, tuple)):
        return [bool(x) for x in aromatic_flags]
    if isinstance(aromatic_flags, np.ndarray):
        return aromatic_flags.astype(bool).flatten().tolist()
    if isinstance(aromatic_flags, torch.Tensor):
        return aromatic_flags.detach().cpu().flatten().bool().tolist()
    return [bool(x) for x in aromatic_flags.tolist()]


def _align_time_indices_to_n_frames(time_indices, n_frames):
    """将 t 标签序列截断或填充到与 pos_traj 帧数一致。"""
    ti = list(time_indices) if time_indices is not None else []
    if n_frames <= 0:
        return []
    if len(ti) == n_frames:
        return ti
    if len(ti) > n_frames:
        # 标签多于帧时取尾部：兼容「仅保存 refine 段」而 meta 仍含 large+refine 的旧结果，避免 t 与帧错位
        return ti[-n_frames:]
    pad = ti[-1] if ti else 0
    return ti + [pad] * (n_frames - len(ti))


def _merge_step_sdfs_for_pymol(mol_dir, out_filename='trajectory_pymol.sdf', logger=None):
    """将单步 SDF 按文件名排序合并为单文件多记录 SDF，供 PyMOL 中 load 后以 state 切换播放。

    不修改各 step_*.sdf；合并文件与单步文件同目录。
    """
    files = sorted(mol_dir.glob('step_*_t*.sdf'))
    if not files:
        return False
    out_path = mol_dir / out_filename
    try:
        writer = Chem.SDWriter(str(out_path))
        try:
            for fp in files:
                suppl = Chem.SDMolSupplier(str(fp), sanitize=False, removeHs=False)
                mol = next(iter(suppl), None)
                if mol is None:
                    if logger:
                        logger.warning('[StepTrajectory] 合并跳过无法读取: %s', fp.name)
                    continue
                mol.SetProp('_Name', fp.stem)
                writer.write(mol)
        finally:
            writer.close()
    except Exception as e:
        if logger:
            logger.warning('[StepTrajectory] 合并 PyMOL 多帧 SDF 失败: %s', e)
        return False
    if logger:
        logger.info('[StepTrajectory] PyMOL 多帧轨迹: %s (%d 帧)', out_path, len(files))
    return True


def save_step_trajectory_sdf(result, result_path, config, ligand_atom_mode, pocket_id=None, timestamp=None, logger=None):
    """将采样轨迹按步保存为 SDF 文件。

    每个分子一个子文件夹，按分子身份证命名：蛋白质ID_生成时间_分子评分_序号（如 1A4K_20240101_123456_88p89_001）。
    格式与 utils.molecule_id.generate_molecule_id 一致。
    t > bond_threshold 时仅保存原子位置（无键），t <= bond_threshold 时成键并保存完整 SDF。
    merge_pymol_sdf 为 true 时，每个分子子目录下额外写入单文件多记录 SDF（默认 trajectory_pymol.sdf），便于 PyMOL 加载后按 state 播放。

    Args:
        result: 采样结果字典，含 pred_ligand_pos_traj, pred_ligand_v_traj, meta, data
        result_path: 主结果目录（轨迹写入路径由 output_dir 决定：相对路径相对于仓库根目录）
        config: 采样配置
        ligand_atom_mode: 配体原子编码模式
        pocket_id: 口袋/数据编号（data_id 或 'custom'），protein_id 提取失败时用作回退
        timestamp: 生成时间（YYYYMMDD_HHMMSS 字符串或 datetime），用于分子身份证
        logger: 可选日志
    """
    cfg = config.sample.get('step_trajectory_save', {})
    if not cfg.get('enable', False):
        return

    output_dir = cfg.get('output_dir', './diffway')
    bond_threshold = int(cfg.get('bond_threshold', 300))
    merge_pymol_sdf = cfg.get('merge_pymol_sdf', True)
    merge_sdf_filename = str(cfg.get('merge_sdf_filename', 'trajectory_pymol.sdf'))

    _out = Path(output_dir)
    root = _out if _out.is_absolute() else (REPO_ROOT / _out)
    root.mkdir(parents=True, exist_ok=True)

    pos_traj_list = result.get('pred_ligand_pos_traj', [])
    v_traj_list = result.get('pred_ligand_v_traj', [])

    if not pos_traj_list:
        if logger:
            logger.warning('[StepTrajectory] 无轨迹数据，跳过跳步留存')
        return

    n_samples = len(pos_traj_list)

    # 提取蛋白质 ID（口袋）和生成时间，用于分子身份证
    data = result.get('data')
    protein_filename = getattr(data, 'protein_filename', None) if data is not None else None
    ligand_filename = getattr(data, 'ligand_filename', None) if data is not None else None
    protein_id = extract_protein_id(ligand_filename=ligand_filename, protein_filename=protein_filename)
    if protein_id == 'UNKNOWN' and pocket_id is not None:
        protein_id = f'P{pocket_id}'  # 回退：用口袋编号
    generation_time = timestamp if timestamp else datetime.now(timezone(timedelta(hours=8)))
    meta = result.get('meta', {})
    meta_records = meta.get('records') or meta.get('refined_candidates', [])

    for sample_idx in range(n_samples):
        pos_traj = pos_traj_list[sample_idx]
        v_traj = v_traj_list[sample_idx] if sample_idx < len(v_traj_list) else None

        if not pos_traj:
            continue

        # 从 meta.records 获取 QED/SA 计算初步评分（0-100），无则用 0
        rec = meta_records[sample_idx] if sample_idx < len(meta_records) else {}
        metrics = rec.get('metrics', {})
        qed = rec.get('qed') if 'qed' in rec else metrics.get('qed')
        sa = rec.get('sa') if 'sa' in rec else metrics.get('sa')
        if qed is not None and sa is not None and not (np.isnan(qed) or np.isnan(sa)):
            score = (float(qed) + float(sa)) / 2.0 * 100.0
        else:
            score = 0.0

        mol_id_base = generate_molecule_id(protein_id, generation_time, score)
        mol_id = f'{mol_id_base}_{sample_idx + 1:03d}'  # 加序号保证唯一
        mol_dir = root / mol_id
        mol_dir.mkdir(parents=True, exist_ok=True)

        time_indices = _get_time_indices_for_trajectory(
            result, config, n_steps_hint=len(pos_traj), sample_idx=sample_idx
        )
        if (
            logger
            and time_indices is not None
            and len(time_indices) != len(pos_traj)
        ):
            logger.warning(
                '[StepTrajectory] 样本 %d：time_indices 长度 (%d) 与轨迹帧数 (%d) 不一致，'
                '已截断或末项 t 填充；文件名中 t 可能与采样步不完全一致。',
                sample_idx,
                len(time_indices),
                len(pos_traj),
            )

        ti_for_mol = _align_time_indices_to_n_frames(time_indices, len(pos_traj))

        for step_idx, pos in enumerate(pos_traj):
            pos_array = np.asarray(pos, dtype=np.float64)
            if pos_array.ndim == 3:
                pos_array = pos_array[0]
            if pos_array.size == 0:
                continue

            t = ti_for_mol[step_idx] if step_idx < len(ti_for_mol) else step_idx

            if v_traj is not None and step_idx < len(v_traj):
                v_array = np.asarray(v_traj[step_idx])
                if v_array.ndim == 2:
                    v_array = v_array[0] if v_array.shape[0] == 1 else v_array
                v_tensor = torch.tensor(v_array, dtype=torch.long)
                atom_numbers = trans.get_atomic_number_from_index(v_tensor, mode=ligand_atom_mode)
                aromatic_flags = trans.is_aromatic_from_index(v_tensor, mode=ligand_atom_mode)
            else:
                atom_numbers = np.ones(pos_array.shape[0], dtype=np.int64) * 6
                aromatic_flags = None

            out_name = mol_dir / f'step_{step_idx:04d}_t{t}.sdf'

            try:
                if t <= bond_threshold:
                    aromatic = _step_traj_as_bool_list(aromatic_flags)
                    mol = reconstruct.reconstruct_from_generated(
                        pos_array.tolist(),
                        _step_traj_as_int_list(atom_numbers),
                        aromatic=aromatic,
                        basic_mode=(aromatic is None)
                    )
                    if mol is not None:
                        Chem.MolToMolFile(mol, str(out_name))
                else:
                    reconstruct.save_positions_only_to_sdf(
                        pos_array, atom_numbers, str(out_name)
                    )
            except reconstruct.MolReconsError:
                reconstruct.save_positions_only_to_sdf(pos_array, atom_numbers, str(out_name))
            except Exception as e:
                if logger:
                    logger.warning(
                        f'[StepTrajectory] 保存 step {step_idx} t={t} 成键失败，回退为仅坐标 SDF: {e}'
                    )
                try:
                    reconstruct.save_positions_only_to_sdf(pos_array, atom_numbers, str(out_name))
                except Exception as e2:
                    if logger:
                        logger.warning(f'[StepTrajectory] 回退保存仍失败 step {step_idx} t={t}: {e2}')

        if merge_pymol_sdf:
            _merge_step_sdfs_for_pymol(mol_dir, out_filename=merge_sdf_filename, logger=logger)

    if logger:
        logger.info(f'[StepTrajectory] 跳步留存已保存至: {root} (共 {n_samples} 个分子)')
