"""Sampling scaffold locked responsibilities for ligand sampling."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from diffdynamic.data.graph import FOLLOW_BATCH
from diffdynamic.models.probability import center_pos
from diffdynamic.models.probability import ensure_log_ligand
from diffdynamic.models.probability import index_to_log_onehot
from diffdynamic.models.probability import log_sample_categorical
from pathlib import Path
from rdkit import Chem
from torch_geometric.data import Batch
from torch_scatter import scatter_mean
from diffdynamic.chemistry.scaffold_prior import load_scaffold_profile
from diffdynamic.chemistry.scaffold_sites import build_extra_atom_positions
from diffdynamic.chemistry.scaffold_sites import cap_n_extra_for_sites
from diffdynamic.chemistry.scaffold_sites import get_murcko_sites_cfg
from diffdynamic.chemistry.scaffold_sites import load_or_extract_attachment_sites
from diffdynamic.chemistry.scaffold_sites import uses_site_budget_placement
import numpy as np
import time
import torch
import torch.nn.functional as F


def scaffold_dynamic_locked_molecule(
    model,
    data,
    config,
    ligand_atom_mode: str,
    device: str = 'cuda:0',
    logger=None,
    output_dir: str = None,
) -> dict:
    """骨架锁定动态模式：跑完整 dynamic 的 large_step + refine 两阶段。

    与 scaffold_grow 的区别：
    - grow：单条反扩散循环（start_t≈357→0），同时锁定骨架位置和类型。
    - dynamic_locked：复用 dynamic 的 large_step（999→~500）+ refine（time_boundary→0）两阶段，
      通过 fix_scaffold_pos/fix_scaffold_type 控制骨架位置和类型；额外原子从 attachment site 初始化。

    流水线：
        1. 提取骨架（复用 _extract_scaffold_from_reference）。
        2. 对每个样本在 CPU 上：
           a. 采样 N_extra（复用 _sample_n_extra_atoms，pocket_prior 等）。
           b. 在 scaffold attachment/exit-vector 方向初始化新原子，不复制参考配体侧链。
           c. 构建 repaint_cfg：骨架硬锁定，新原子可选软锚定。
        3. 按 dynamic.large_step.batch_size 拼批跑 large_step + refine（可变 n_extra
           可同批；CUDA OOM 自动减半）。batch_size=1 时与原逐样本路径等价。
        4. 后处理、骨架 RMSD、指标记录。
    """
    from diffdynamic.sampling.candidates import evaluate_candidate
    from diffdynamic.sampling.common import MixedRepaintConfig
    from diffdynamic.sampling.common import concat_repaint_cfgs
    from diffdynamic.sampling.common import extra_pos_anchor_mask
    from diffdynamic.sampling.common import pack_repeated_protein
    from diffdynamic.sampling.common import run_gpu_chunks_with_oom_backoff
    from diffdynamic.sampling.common import safe_repeat_interleave
    from diffdynamic.sampling.common import scaffold_dl_gpu_batch_size
    from diffdynamic.sampling.common import split_tensor_by_counts
    from diffdynamic.sampling.configuration import _dynamic_subcfg_grad_cap_kwargs
    from diffdynamic.sampling.configuration import get_time_boundary
    from diffdynamic.sampling.optimization import _forward_diffuse_molecule
    from diffdynamic.sampling.scaffold.inputs import _apply_exact_record_site_centers
    from diffdynamic.sampling.scaffold.inputs import _exact_record_type_hint_keys
    from diffdynamic.sampling.scaffold.inputs import _extract_scaffold_from_reference
    from diffdynamic.sampling.scaffold.inputs import _extract_scaffold_submol
    from diffdynamic.sampling.scaffold.inputs import _get_scaffold_mode_cfg
    from diffdynamic.sampling.scaffold.inputs import _pin_place_cfg_to_n_extra
    from diffdynamic.sampling.scaffold.inputs import _sample_n_extra_atoms
    from diffdynamic.sampling.scaffold.inputs import _sample_reference_exact_record
    from diffdynamic.sampling.scaffold.placement import _murcko_placement_cfg
    from diffdynamic.sampling.scaffold.placement import _murcko_sample_rng
    from diffdynamic.sampling.scaffold.placement import _reference_extra_type_log_prior
    from diffdynamic.sampling.scaffold.placement import _resolve_init_preserved_atoms
    from diffdynamic.sampling.scaffold.placement import _sample_reference_extra_type_quota
    from diffdynamic.sampling.scaffold.placement import _select_side_chain_context_anchor_indices

    sc_cfg, _ = _get_scaffold_mode_cfg(config)   # dynamic_locked 无子节时 merged=顶层 scaffold
    side_chain_only = bool(sc_cfg.get('side_chain_only', False))
    grow_cfg = config.sample.get('scaffold', {}).get('grow', {})
    center_pos_mode = config.sample.get('center_pos_mode', 'protein')
    pos_only = config.sample.get('pos_only', False)

    dynamic_cfg = config.sample.get('dynamic', {})
    large_cfg = dynamic_cfg.get('large_step', {})
    refine_cfg = dict(dynamic_cfg.get('refine', {}))
    time_boundary = get_time_boundary(dynamic_cfg, 750)
    if 'time_upper' not in refine_cfg:
        refine_cfg['time_upper'] = time_boundary
    tbr_cfg = config.sample.get('targetdiff_baseline_refine', {})
    baseline_init_only = bool(
        tbr_cfg.get('enable', False) and tbr_cfg.get('after_scaffold_init_only', False)
    )
    # 仅跑 DiffDynamic refine（跳过 large_step；与 baseline_init_only / prudent 互斥）
    refine_only = bool(sc_cfg.get('refine_only', False)) and not baseline_init_only
    # 本模式契约：默认跑完整 large_step + refine；仅 baseline_init_only 时跳过
    skip_refine = False

    if not hasattr(data, 'ligand_pos') or data.ligand_pos is None or data.ligand_pos.size(0) == 0:
        raise ValueError(
            '骨架锁定动态模式需要参考配体（含 3D 坐标 SDF）以提取骨架，请通过 --ligand_path 指定。'
        )

    ref_ligand_pos = data.ligand_pos.to(device)
    ref_ligand_v = data.ligand_atom_feature_full.to(device)
    n_scaffold_ref = ref_ligand_pos.size(0)
    log_ref_v = index_to_log_onehot(ref_ligand_v, model.num_classes)

    ref_pos_np = ref_ligand_pos.detach().cpu().numpy().astype(np.float64)
    ref_v_np = np.argmax(log_ref_v.detach().cpu().numpy(), axis=-1).astype(np.int64)

    # ---- 提取骨架（复用共享逻辑）-------------------------------------
    scaffold_indices, ref_mol, ref_ligand_name, n_scaffold = _extract_scaffold_from_reference(
        data, sc_cfg, ligand_atom_mode, ref_ligand_pos, ref_ligand_v, log_ref_v,
        n_scaffold_ref, device, config=config, logger=logger,
    )

    fix_scaffold_pos = bool(sc_cfg.get('fix_scaffold_pos', True))
    fix_scaffold_type = bool(sc_cfg.get('fix_scaffold_type', False))   # 本模式默认 False

    if logger:
        _tbr_steps = int(tbr_cfg.get('start_t', 9)) + 1 if baseline_init_only else None
        logger.info(
            f'[ScaffoldDynamicLocked] 骨架原子: {n_scaffold}/{n_scaffold_ref} | '
            f'model_input={"interface_anchor_plus_side_chain" if side_chain_only else "full_scaffold_plus_side_chain"} | '
            f'锁定位置: {fix_scaffold_pos} | 锁定类型: {fix_scaffold_type} | '
            f'skip_refine: {skip_refine} | time_boundary: {time_boundary}'
            + (f' | baseline_init_only: TargetDiff { _tbr_steps }步' if baseline_init_only else '')
            + (' | refine_only: 跳过 large_step' if refine_only else '')
        )

    # ---- 保存骨架 SDF 结构（复用 grow 的保存逻辑）----------------------
    try:
        if ref_mol is None and hasattr(data, 'ligand_mol') and data.ligand_mol is not None:
            ref_mol = data.ligand_mol
            if hasattr(data, 'ligand_filename') and data.ligand_filename:
                ref_ligand_name = Path(data.ligand_filename).stem
        if ref_mol is not None and n_scaffold > 0 and not side_chain_only:
            scaffold_mol = _extract_scaffold_submol(ref_mol, scaffold_indices, logger)
            if scaffold_mol is not None:
                if output_dir is not None:
                    base_output_dir = Path(output_dir)
                else:
                    base_output_dir = Path(config.sample.get('output_dir', './outputs'))
                scaffold_output_dir = base_output_dir / 'scaffold'
                scaffold_output_dir.mkdir(parents=True, exist_ok=True)
                if ref_ligand_name:
                    scaffold_filename = f'{ref_ligand_name}_scaffold_dynamic_locked.sdf'
                else:
                    cst = timezone(timedelta(hours=8))
                    timestamp = datetime.now(cst).strftime('%Y%m%d_%H%M%S')
                    scaffold_filename = f'scaffold_dl_{n_scaffold}atoms_{timestamp}.sdf'
                scaffold_sdf_path = scaffold_output_dir / scaffold_filename
                writer = Chem.SDWriter(str(scaffold_sdf_path))
                writer.write(scaffold_mol)
                writer.close()
                if logger:
                    logger.info(f'[ScaffoldDynamicLocked] 骨架结构已保存: {scaffold_sdf_path}')
    except Exception as e:
        if logger:
            logger.warning(f'[ScaffoldDynamicLocked] 保存骨架 SDF 失败: {e}')

    if output_dir is not None:
        _dl_base_output_dir = Path(output_dir)
    else:
        _dl_base_output_dir = Path(config.sample.get('output_dir', './outputs'))

    _murcko_sites_cfg = get_murcko_sites_cfg(sc_cfg)
    _attachment_sites = load_or_extract_attachment_sites(
        ref_mol, scaffold_indices,
        sc_cfg.get('scaffold_source', 'auto_murcko'),
        ref_pos_np, _dl_base_output_dir, ref_ligand_name,
        _murcko_sites_cfg, logger=logger,
    )
    if side_chain_only and not _attachment_sites:
        raise ValueError(
            'side_chain_only requires at least one precise scaffold exit site'
        )

    # ---- 生成参数 ------------------------------------------------------------
    num_samples = int(sc_cfg.get('num_samples', grow_cfg.get('num_samples', 200)))

    pos_list, v_list = [], []
    pos_traj_list, v_traj_list, log_v_traj_list = [], [], []
    time_list = []
    meta_records = []
    total_time = 0.0

    selector_cfg = {
        'min_qed': grow_cfg.get('min_qed', 0.2),
        'min_sa': grow_cfg.get('min_sa', 0.2),
        'filter_incomplete': grow_cfg.get('filter_incomplete', False),
        'qed_weight': 1.0,
        'sa_weight': 1.0,
    }

    log_mode_is_log_prob = (getattr(model, 'ligand_v_input', 'onehot') == 'log_prob')

    extra_anchor_strength = float(grow_cfg.get('extra_anchor_strength', 0.0))
    extra_anchor_atoms = int(grow_cfg.get('extra_anchor_atoms', 0) or 0)
    extra_atom_type_mode = str(
        sc_cfg.get('extra_atom_type_mode', 'auto')
    ).strip().lower()
    extra_types_explicitly_diffused = extra_atom_type_mode in {
        'diffuse',
        'unlocked',
        'free',
        'model',
    }
    lock_extra_atom_types = bool(
        sc_cfg.get('lock_extra_atom_types', False)
    )
    allow_soft_extra_anchor = bool(
        sc_cfg.get('allow_soft_extra_anchor', False)
    )
    if lock_extra_atom_types and not allow_soft_extra_anchor:
        # This scaffold contract intentionally leaves added coordinates to the
        # normal DiffDynamic reverse process. Only their element classes are
        # repainted from the exact x0 type condition.
        extra_anchor_strength = 0.0
    extra_type_anchor_strength = float(
        grow_cfg.get('extra_type_anchor_strength', 0.0)
    )
    strict_anchor_gaussian = bool(
        _murcko_sites_cfg.get('strict_anchor_gaussian', False)
    )
    strict_fragment_gaussian = bool(
        _murcko_sites_cfg.get('strict_fragment_gaussian', False)
    )
    fragment_gaussian = bool(
        _murcko_sites_cfg.get('fragment_gaussian', False)
    )
    pocket_center_gaussian = bool(
        _murcko_sites_cfg.get('pocket_center_gaussian', False)
    )
    interface_corridor_gaussian = bool(
        _murcko_sites_cfg.get('interface_corridor_gaussian', False)
    )
    if fragment_gaussian:
        # Soft fragment geometry is scaffold-only.  The initial x0 type quota
        # may seed q(x_t|x_0), but no added element is RePaint-locked.
        extra_atom_type_mode = 'diffuse'
        extra_types_explicitly_diffused = True
    if extra_types_explicitly_diffused:
        # Type-free scaffold ablation: keep the exact atom count, but let the
        # normal categorical reverse process predict every added element.
        # Coordinates remain governed by the ordinary DiffDynamic process.
        extra_type_anchor_strength = 0.0
    elif (
        lock_extra_atom_types
        or strict_anchor_gaussian
        or strict_fragment_gaussian
        or pocket_center_gaussian
        or interface_corridor_gaussian
    ):
        # The profile is an exchangeable element multiset.  RePaint keeps its
        # forward-noised state fixed at every reverse step; it is not a soft
        # class preference and must not be replaced by the model prediction.
        extra_type_anchor_strength = 1.0
    extra_type_anchor_strength = min(max(extra_type_anchor_strength, 0.0), 1.0)
    extra_types_locked = bool(
        not extra_types_explicitly_diffused
        and (
            lock_extra_atom_types
            or strict_anchor_gaussian
            or strict_fragment_gaussian
            or pocket_center_gaussian
            or interface_corridor_gaussian
        )
    )
    forward_noise_init = bool(grow_cfg.get('forward_noise_init', False))
    forward_noise_init_mode = str(
        grow_cfg.get('forward_noise_init_mode', 'full_q')
    ).strip().lower()
    if forward_noise_init_mode not in {
        'full_q', 'scaffold_only', 'scaffold_context', 'none'
    }:
        raise ValueError(
            'scaffold.grow.forward_noise_init_mode must be one of '
            "'full_q', 'scaffold_only', 'scaffold_context', or 'none'"
        )
    extra_type_context_mode = str(
        grow_cfg.get('extra_type_context_mode', 'q')
    ).strip().lower()
    if extra_type_context_mode not in {'q', 'clean'}:
        raise ValueError(
            'scaffold.grow.extra_type_context_mode must be one of '
            "'q' or 'clean'"
        )
    diffusion_start_t = int(np.clip(
        grow_cfg.get('start_t', model.num_timesteps - 1),
        1,
        model.num_timesteps - 1,
    ))
    extra_type_profile = None
    if (
        _murcko_sites_cfg.get('reference_extra_type_prior_mode') == 'quota_random'
        and _murcko_sites_cfg.get('reference_exit_profile')
    ):
        try:
            extra_type_profile = load_scaffold_profile(
                _murcko_sites_cfg['reference_exit_profile']
            )
        except Exception as exc:
            if logger:
                logger.warning(f'[ScaffoldTypes] quota prior unavailable: {exc}')
    extra_type_log_prior = _reference_extra_type_log_prior(
        _murcko_sites_cfg,
        ligand_atom_mode,
        model.num_classes,
        device,
        logger=logger,
    )

    def _extra_log_v_init(
        count: int,
        rng: np.random.Generator,
        exact_record: dict | None = None,
    ):
        if count <= 0:
            return torch.zeros(0, model.num_classes, device=device)
        if exact_record is not None:
            exact_type_keys = None
            if strict_fragment_gaussian or fragment_gaussian:
                fragment_hints = _site_place_meta.get(
                    'fragment_type_hints'
                ) or []
                expected_counts = Counter(
                    str(key)
                    for key, value in (
                        exact_record.get('type_counts') or {}
                    ).items()
                    for _ in range(int(value))
                )
                if Counter(str(key) for key in fragment_hints) == expected_counts:
                    exact_type_keys = [str(key) for key in fragment_hints]
            if exact_type_keys is None:
                exact_type_keys = _exact_record_type_hint_keys(
                    exact_record,
                    rng,
                    _site_place_meta.get('site_allocation'),
                )
            quota_log_v = _sample_reference_extra_type_quota(
                {},
                ligand_atom_mode,
                model.num_classes,
                count,
                device,
                rng,
                type_hint_keys=exact_type_keys,
            )
            if quota_log_v is None:
                raise ValueError(
                    'exact scaffold reference record contains an unsupported '
                    f'element/aromatic class: {exact_record}'
                )
            return quota_log_v
        type_hint_keys = None
        if strict_fragment_gaussian:
            type_hint_keys = _site_place_meta.get('fragment_type_hints')
            if not type_hint_keys:
                raise ValueError(
                    'strict_fragment_gaussian produced no fragment type hints'
                )
        if extra_type_profile is not None:
            quota_log_v = _sample_reference_extra_type_quota(
                extra_type_profile,
                ligand_atom_mode,
                model.num_classes,
                count,
                device,
                rng,
                type_hint_keys=type_hint_keys,
            )
            if quota_log_v is not None:
                return quota_log_v
            if (
                strict_anchor_gaussian
                or strict_fragment_gaussian
                or interface_corridor_gaussian
            ):
                raise ValueError(
                    'strict scaffold Gaussian mode requires an exact supported '
                    f'element quota for n_extra={count}'
                )
        if extra_type_log_prior is not None:
            return extra_type_log_prior.unsqueeze(0).expand(count, -1)
        return F.log_softmax(
            torch.zeros(count, model.num_classes, device=device), dim=-1
        )

    def _emit_locked_sample(prep, final_pos_t, log_v_out, dt):
        n_extra = int(prep['n_extra'])
        n_locked = int(prep['n_locked'])
        n_total = int(prep['n_total'])
        n_preserved = int(prep['n_preserved'])
        model_n_scaffold = int(prep['model_n_scaffold'])
        model_context_indices = prep['model_context_indices']
        exact_record = prep['exact_record']
        extra_log_v = prep['extra_log_v']
        zero_site_ids = prep['zero_site_ids']
        preserved_indices = prep['preserved_indices']
        site_place_meta = prep['site_place_meta'] or {}

        pos_np = final_pos_t.detach().cpu().numpy().astype(np.float64)
        v_np = log_v_out.argmax(dim=-1).detach().cpu().numpy().astype(np.int64)

        # Strict mode restores the exact exchangeable type quota at x_0.  This
        # guards the final tensor against any model-side class drift while
        # leaving normal scaffold and de novo post-processing unchanged.
        if n_extra > 0:
            if extra_types_locked:
                v_np[n_locked:] = extra_log_v.argmax(dim=-1).detach().cpu().numpy()
            else:
                # 保留 F/Cl；仅清除不应作为新增重原子节点的 H 类别。
                extra_v = v_np[n_locked:]
                to_carbon = (extra_v == 0)    # H
                if to_carbon.any():
                    extra_v[to_carbon] = 1    # → 非芳香 C
                    v_np[n_locked:] = extra_v

        # 骨架 RMSD（验证位置锁定效果）
        if model_n_scaffold > 0:
            scaffold_pos_ref_world = ref_ligand_pos[
                model_context_indices
            ].detach().cpu().numpy()
            scaffold_pos_gen = pos_np[:model_n_scaffold]
            scaffold_rmsd = float(np.sqrt(np.mean((scaffold_pos_gen - scaffold_pos_ref_world) ** 2)))
        else:
            scaffold_rmsd = 0.0

        metric_info = evaluate_candidate(pos_np, v_np, ligand_atom_mode, selector_cfg)

        pos_list.append(pos_np)
        v_list.append(v_np)
        pos_traj_list.append([])
        v_traj_list.append([])
        log_v_traj_list.append([])
        time_list.append(dt)
        meta_records.append({
            'method': (
                'scaffold_side_chain_only'
                if side_chain_only else 'scaffold_dynamic_locked'
            ),
            'ligand_num_atoms': n_total,
            # n_scaffold is the locked prefix in the emitted tensor.  The
            # complete extracted scaffold size is retained separately so
            # downstream tooling does not mistake the interface context for
            # the full CRBN scaffold.
            'n_scaffold': model_n_scaffold,
            'reference_scaffold_atom_count': n_scaffold,
            'side_chain_only': side_chain_only,
            'context_anchor_count': model_n_scaffold,
            'context_anchor_reference_indices': [
                int(index) for index in model_context_indices
            ],
            'n_preserved': n_preserved,
            'n_extra': n_extra,
            'exact_reference_index': (
                int(exact_record['reference_index'])
                if exact_record is not None else None
            ),
            'exact_reference_name': (
                str(exact_record.get('reference_name', ''))
                if exact_record is not None else None
            ),
            'exact_reference_type_counts': (
                dict(exact_record.get('type_counts') or {})
                if exact_record is not None else None
            ),
            'n_locked': n_locked,
            'zero_allocation_site_ids': zero_site_ids,
            'preserved_atom_indices': preserved_indices,
            'scaffold_rmsd': scaffold_rmsd,
            'rmsd_from_ref': float(np.sqrt(np.mean(
                (pos_np[:n_scaffold_ref] - ref_pos_np) ** 2
            ))) if (not side_chain_only and n_scaffold_ref == n_total)
            else float('nan'),
            'time': dt,
            'smiles': metric_info.get('smiles'),
            'qed': metric_info.get('metrics', {}).get('qed'),
            'sa': metric_info.get('metrics', {}).get('sa'),
            'status': metric_info.get('status'),
            'is_original': False,
            'initial_extra_positions': prep.get('initial_extra_positions'),
            'extra_placement': site_place_meta.get('placement') if site_place_meta else 'unknown',
            'site_allocation': site_place_meta.get('site_allocation') if site_place_meta else None,
            'strict_anchor_gaussian': strict_anchor_gaussian,
            'strict_fragment_gaussian': strict_fragment_gaussian,
            'fragment_gaussian': fragment_gaussian,
            'pocket_center_gaussian': pocket_center_gaussian,
            'interface_corridor_gaussian': interface_corridor_gaussian,
            'strict_extra_type_locked': extra_types_locked,
            'extra_atom_types_locked': extra_types_locked,
            'extra_atom_type_mode': extra_atom_type_mode,
            'extra_position_mask': (
                extra_anchor_strength
                if allow_soft_extra_anchor
                else 0.0 if lock_extra_atom_types else extra_anchor_strength
            ),
            'allow_soft_extra_anchor': allow_soft_extra_anchor,
            'forward_noise_init_mode': forward_noise_init_mode,
            'extra_type_context_mode': extra_type_context_mode,
            'fragment_layout': (
                site_place_meta.get('fragment_layout')
                if strict_fragment_gaussian or fragment_gaussian else None
            ),
            'gpu_reverse_batch_size': int(prep.get('gpu_reverse_batch_size') or 1),
        })

    def _run_locked_reverse_chunk(chunk):
        n_mol = len(chunk)
        counts = [int(item['n_total']) for item in chunk]
        t0 = time.time()
        if baseline_init_only:
            if logger and chunk[0]['sample_idx'] == 0:
                logger.info(
                    f'[ScaffoldDynamicLocked] baseline_init_only: 跳过 large_step/refine，'
                    f'将由 TargetDiff baseline 做 {int(tbr_cfg.get("start_t", 9)) + 1} 步修复'
                )
            dt = (time.time() - t0) / max(n_mol, 1)
            outs = []
            for item in chunk:
                init_v = item['init_ligand_v_input']
                if log_mode_is_log_prob or (
                    isinstance(init_v, torch.Tensor) and init_v.dim() == 2
                ):
                    log_v_out = init_v
                else:
                    log_v_out = index_to_log_onehot(init_v, model.num_classes)
                outs.append((item, item['init_ligand_pos'], log_v_out, dt))
            return outs

        try:
            packed_repaint = concat_repaint_cfgs(
                [item['repaint_cfg'] for item in chunk]
            )
        except MixedRepaintConfig:
            if n_mol == 1:
                raise
            merged = []
            for item in chunk:
                merged.extend(_run_locked_reverse_chunk([item]))
            return merged

        protein_pos_batch, protein_v_batch, batch_protein = pack_repeated_protein(
            data, n_mol, device,
        )
        batch_ligand = safe_repeat_interleave(
            torch.arange(n_mol, dtype=torch.long, device=device),
            counts,
            device=device,
        )
        if refine_only:
            init_pos = torch.cat([item['init_ligand_pos'] for item in chunk], dim=0)
            init_v = torch.cat(
                [item['init_ligand_v_input'] for item in chunk], dim=0
            )
        else:
            init_pos = torch.cat([item['large_init_pos'] for item in chunk], dim=0)
            init_v = torch.cat([item['large_init_v'] for item in chunk], dim=0)

        with torch.no_grad():
            if refine_only:
                if logger and chunk[0]['sample_idx'] == 0:
                    logger.info(
                        f'[ScaffoldDynamicLocked] refine_only: 跳过 large_step，'
                        f'仅 DiffDynamic refine (t={refine_cfg.get("time_upper", time_boundary)}→'
                        f'{refine_cfg.get("time_lower", 0)})'
                    )
                res_refine = model.sample_diffusion_refinement(
                    protein_pos=protein_pos_batch,
                    protein_v=protein_v_batch,
                    batch_protein=batch_protein,
                    init_ligand_pos=init_pos,
                    init_ligand_v=init_v,
                    batch_ligand=batch_ligand,
                    center_pos_mode=center_pos_mode,
                    pos_only=pos_only,
                    step_stride=refine_cfg.get('stride'),
                    step_size=refine_cfg.get('step_size'),
                    add_noise=refine_cfg.get('noise_scale'),
                    pos_clip=refine_cfg.get('pos_clip'),
                    v_clip=refine_cfg.get('v_clip'),
                    time_upper=refine_cfg.get('time_upper', time_boundary),
                    time_lower=refine_cfg.get('time_lower', 0),
                    num_cycles=refine_cfg.get('cycles', 1),
                    log_ligand_input_mode='log_prob' if log_mode_is_log_prob else 'auto',
                    repaint_cfg=packed_repaint,
                    **_dynamic_subcfg_grad_cap_kwargs(refine_cfg),
                )
                final_pos_t = res_refine['pos']
                log_v_out = res_refine['log_v']
            else:
                res_large = model.sample_diffusion_large_step(
                    protein_pos=protein_pos_batch,
                    protein_v=protein_v_batch,
                    batch_protein=batch_protein,
                    init_ligand_pos=init_pos,
                    init_ligand_v=init_v,
                    batch_ligand=batch_ligand,
                    num_steps=large_cfg.get('num_steps'),
                    center_pos_mode=center_pos_mode,
                    pos_only=pos_only,
                    step_stride=large_cfg.get('stride'),
                    step_size=large_cfg.get('step_size'),
                    add_noise=large_cfg.get('noise_scale'),
                    pos_clip=large_cfg.get('pos_clip'),
                    v_clip=large_cfg.get('v_clip'),
                    log_ligand_input_mode='log_prob' if log_mode_is_log_prob else 'auto',
                    repaint_cfg=packed_repaint,
                    **_dynamic_subcfg_grad_cap_kwargs(large_cfg),
                )
                if not skip_refine:
                    refine_init_pos = res_large['pos']
                    if log_mode_is_log_prob:
                        refine_init_v = res_large['log_v']
                    else:
                        refine_init_v = res_large['v']
                    res_refine = model.sample_diffusion_refinement(
                        protein_pos=protein_pos_batch,
                        protein_v=protein_v_batch,
                        batch_protein=batch_protein,
                        init_ligand_pos=refine_init_pos,
                        init_ligand_v=refine_init_v,
                        batch_ligand=batch_ligand,
                        center_pos_mode=center_pos_mode,
                        pos_only=pos_only,
                        step_stride=refine_cfg.get('stride'),
                        step_size=refine_cfg.get('step_size'),
                        add_noise=refine_cfg.get('noise_scale'),
                        pos_clip=refine_cfg.get('pos_clip'),
                        v_clip=refine_cfg.get('v_clip'),
                        time_upper=refine_cfg.get('time_upper', time_boundary),
                        time_lower=refine_cfg.get('time_lower', 0),
                        num_cycles=refine_cfg.get('cycles', 1),
                        log_ligand_input_mode='log_prob' if log_mode_is_log_prob else 'auto',
                        repaint_cfg=packed_repaint,
                        **_dynamic_subcfg_grad_cap_kwargs(refine_cfg),
                    )
                    final_pos_t = res_refine['pos']
                    log_v_out = res_refine['log_v']
                else:
                    final_pos_t = res_large['pos']
                    log_v_out = res_large['log_v']

        dt = (time.time() - t0) / max(n_mol, 1)
        pos_parts = split_tensor_by_counts(final_pos_t, counts)
        v_parts = split_tensor_by_counts(log_v_out, counts)
        return [
            (chunk[i], pos_parts[i], v_parts[i], dt)
            for i in range(n_mol)
        ]

    # ---- CPU 逐样本投放；GPU 按 large_step.batch_size 拼批反向 -----------
    prepared = []
    for sample_idx in range(num_samples):
        exact_record = None
        if str(grow_cfg.get('n_extra_mode', '')).strip().lower() == (
            'reference_exact_record'
        ):
            exact_record = _sample_reference_exact_record(
                grow_cfg, sample_idx, retry=0
            )
            n_extra = int(exact_record['n_extra'])
        else:
            n_extra = _sample_n_extra_atoms(
                grow_cfg,
                data.protein_pos,
                n_scaffold,
                model.num_timesteps,
                sample_idx=sample_idx,
            )
        _place_cfg = _pin_place_cfg_to_n_extra(
            _murcko_placement_cfg(grow_cfg, _murcko_sites_cfg), n_extra, grow_cfg,
        )
        sample_attachment_sites = _attachment_sites
        if exact_record is not None and exact_record.get('site_counts'):
            _place_cfg = dict(_place_cfg)
            _place_cfg['reference_exact_site_counts'] = {
                str(int(slot)): int(count)
                for slot, count in exact_record['site_counts'].items()
            }
            sample_attachment_sites = _apply_exact_record_site_centers(
                _attachment_sites,
                exact_record,
            )
            if exact_record.get('interface_center_world'):
                _place_cfg['interface_center_world'] = list(
                    exact_record['interface_center_world']
                )
            if exact_record.get('site_type_counts'):
                _place_cfg['reference_exact_site_type_counts'] = {
                    str(slot): {
                        str(key): int(value)
                        for key, value in counts.items()
                    }
                    for slot, counts in exact_record[
                        'site_type_counts'
                    ].items()
                }
            if exact_record.get('site_attachment_type_counts'):
                _place_cfg['reference_exact_site_attachment_type_counts'] = {
                    str(slot): {
                        str(key): int(value)
                        for key, value in counts.items()
                    }
                    for slot, counts in exact_record[
                        'site_attachment_type_counts'
                    ].items()
                }
            if exact_record.get('site_centers_world'):
                _place_cfg['reference_exact_site_centers_world'] = {
                    str(slot): list(center)
                    for slot, center in exact_record[
                        'site_centers_world'
                    ].items()
                }
        if not uses_site_budget_placement(_place_cfg):
            capped_n_extra = cap_n_extra_for_sites(
                n_extra, sample_attachment_sites, _murcko_sites_cfg
            )
            if exact_record is not None and capped_n_extra != n_extra:
                raise ValueError(
                    'exact scaffold reference count exceeds scaffold-site '
                    f'capacity: n_extra={n_extra}, capacity={capped_n_extra}, '
                    f'record={exact_record}'
                )
            n_extra = capped_n_extra
        n_total = n_scaffold + n_extra

        if logger and (sample_idx == 0 or sample_idx % 10 == 0):
            logger.info(
                f'[ScaffoldDynamicLocked] 样本 {sample_idx + 1}/{num_samples} | '
                f'骨架原子: {n_scaffold} + 新原子: {n_extra} = 总计: {n_total}'
            )

        # ---- 构建单样本批次 ---------------------------------------------
        batch = Batch.from_data_list([data.clone()], follow_batch=FOLLOW_BATCH).to(device)
        batch_protein = batch.protein_element_batch

        # 蛋白中心（= center_pos('protein') 对 batch_size=1 的 offset）
        protein_centroid = scatter_mean(batch.protein_pos, batch_protein, dim=0)[0]  # [3]

        # ---- 初始化：从 scaffold attachment/exit-vector 放置新原子；不保留旧侧链 --
        # build_extra_atom_positions 使用锚点、去除片段位置和 anchor→centroid 方向，
        # 避免把整段新片段撒在远离成键锚点的侧链质心球内。
        sample_rng = _murcko_sample_rng(grow_cfg, sample_idx)
        extra_pos_world, _site_place_meta = build_extra_atom_positions(
            n_extra, sample_attachment_sites, _place_cfg,
            protein_centroid, device, fallback_noise_scale=1.0, logger=logger,
            rng=sample_rng,
            protein_positions=batch.protein_pos,
        )
        n_extra = int(extra_pos_world.size(0))
        if exact_record is not None and n_extra != int(exact_record['n_extra']):
            raise ValueError(
                'exact scaffold reference count changed during placement: '
                f"requested={exact_record['n_extra']} placed={n_extra}"
            )
        if side_chain_only:
            model_context_indices = _select_side_chain_context_anchor_indices(
                sample_attachment_sites,
                _site_place_meta,
                scaffold_indices,
                logger=logger if sample_idx == 0 else None,
            )
        else:
            model_context_indices = list(scaffold_indices)
        model_n_scaffold = len(model_context_indices)
        (
            preserved_pos, preserved_log_v, preserved_indices, zero_site_ids,
            n_preserved, n_locked, n_total,
        ) = _resolve_init_preserved_atoms(
            sample_attachment_sites, _site_place_meta, _place_cfg,
            ref_ligand_pos, log_ref_v, model_n_scaffold, n_extra,
            logger=logger, log_tag='[ScaffoldDynamicLocked]', sample_idx=sample_idx,
        )

        # world-frame init: noise everywhere, then overwrite locked + extra
        init_ligand_pos = protein_centroid.unsqueeze(0).expand(n_total, -1) + \
                          torch.randn(n_total, 3, device=device)
        if model_n_scaffold > 0 and fix_scaffold_pos:
            init_ligand_pos[:model_n_scaffold] = ref_ligand_pos[
                model_context_indices
            ]
        if n_preserved > 0 and fix_scaffold_pos:
            init_ligand_pos[model_n_scaffold:n_locked] = preserved_pos
        if n_extra > 0:
            init_ligand_pos[n_locked:] = extra_pos_world
        batch_ligand = torch.zeros(n_total, dtype=torch.long, device=device)

        extra_log_v = _extra_log_v_init(
            n_extra,
            sample_rng,
            exact_record=exact_record,
        )
        if fix_scaffold_type and n_locked > 0:
            scaffold_log_part = (
                log_ref_v[model_context_indices] if model_n_scaffold > 0
                else torch.zeros(0, model.num_classes, device=device)
            )
            init_ligand_v_input = torch.cat(
                [scaffold_log_part, preserved_log_v, extra_log_v], dim=0
            )
        else:
            uniform_logits = torch.zeros(n_total, model.num_classes, device=device)
            if log_mode_is_log_prob:
                init_ligand_v_input = F.log_softmax(uniform_logits, dim=-1)
            else:
                init_ligand_v_input = log_sample_categorical(uniform_logits)

        # ---- 构建 repaint_cfg：锁定 CRBN scaffold，新增坐标不回拉 -----------
        repaint_cfg = None
        type_mask_local = None
        if n_locked > 0:
            scaffold_pos_orig = (
                ref_ligand_pos[model_context_indices]
                if model_n_scaffold > 0
                else torch.zeros(0, 3, device=device)
            )
            locked_pos_world = torch.cat([scaffold_pos_orig, preserved_pos], dim=0)
            locked_pos_centered = locked_pos_world - protein_centroid.unsqueeze(0)
            if n_extra > 0:
                extra_pos_centered = extra_pos_world - protein_centroid.unsqueeze(0)
            else:
                extra_pos_centered = torch.zeros(0, 3, device=device)
            x0_pos_ref = torch.cat([locked_pos_centered, extra_pos_centered], dim=0)

            scaffold_log_v = (
                log_ref_v[model_context_indices] if model_n_scaffold > 0
                else torch.zeros(0, model.num_classes, device=device)
            )
            x0_log_v_ref = torch.cat(
                [scaffold_log_v, preserved_log_v, extra_log_v], dim=0
            )

            pos_mask_local = extra_pos_anchor_mask(
                n_total,
                n_locked,
                n_extra,
                extra_anchor_strength,
                extra_anchor_atoms,
                device=device,
                fix_scaffold_pos=fix_scaffold_pos,
                distal_strength=float(
                    grow_cfg.get('extra_anchor_distal_strength', 0.0) or 0.0
                ),
            )

            type_mask_local = None
            if fix_scaffold_type or extra_type_anchor_strength > 0.0:
                type_mask_local = torch.zeros(n_total, device=device)
                if fix_scaffold_type:
                    type_mask_local[:n_locked] = 1.0
                if extra_type_anchor_strength > 0.0 and n_extra > 0:
                    type_mask_local[n_locked:] = extra_type_anchor_strength

            repaint_cfg = {
                'x0_pos': x0_pos_ref,
                'x0_log_v': x0_log_v_ref,
                'pos_mask': pos_mask_local,
                'type_mask': type_mask_local,
                'use_mean_for_discrete': True,
                '_use_dual_mask': True,
                '_type_context_mode': extra_type_context_mode,
                '_clean_type_context': (
                    extra_type_context_mode == 'clean'
                ),
            }

        large_init_pos = init_ligand_pos
        large_init_v = init_ligand_v_input
        if (
            forward_noise_init
            and forward_noise_init_mode != 'none'
            and not baseline_init_only
            and not refine_only
        ):
            _, init_pos_centered, init_offset = center_pos(
                batch.protein_pos,
                init_ligand_pos,
                batch_protein,
                batch_ligand,
                mode=center_pos_mode,
            )
            noised_pos, noised_log_v = _forward_diffuse_molecule(
                model,
                init_pos_centered,
                ensure_log_ligand(
                    init_ligand_v_input, model.num_classes, mode='auto'
                ),
                batch_ligand,
                diffusion_start_t,
                device,
            )
            large_init_pos = noised_pos + init_offset[batch_ligand]
            large_init_v = noised_log_v
            if (
                extra_type_context_mode == 'clean'
                and type_mask_local is not None
            ):
                # Keep only the explicitly type-masked classes visible at the
                # first model call.  Added coordinates are still q-noised and
                # position-unmasked; this is a type-context ablation only.
                clean_type_mask = type_mask_local.view(-1, 1).bool()
                large_init_v = torch.where(
                    clean_type_mask,
                    init_ligand_v_input,
                    large_init_v,
                )
            if forward_noise_init_mode == 'scaffold_only' and n_extra > 0:
                # Experimental scaffold-only initialization: retain the
                # scaffold-derived extra-atom Gaussian cloud while still
                # entering the normal reverse schedule at t=999.  Added
                # coordinates remain unmasked after initialization; this is
                # not a position constraint or a target-side pose injection.
                large_init_pos[n_locked:] = init_ligand_pos[n_locked:]
                large_init_v[n_locked:] = init_ligand_v_input[n_locked:]
            elif forward_noise_init_mode == 'scaffold_context' and n_locked > 0:
                # Conditional scaffold context: keep only the fixed scaffold
                # clean at the t=999 model input so the network can see its
                # geometry immediately.  Added atoms still enter as ordinary
                # q(x_999 | x_0) noise and remain position-unmasked.
                large_init_pos[:n_locked] = init_ligand_pos[:n_locked]
                large_init_v[:n_locked] = init_ligand_v_input[:n_locked]
            if logger and sample_idx == 0:
                logger.info(
                    f'[ScaffoldDynamicLocked] q(x_t|x_0) initialization at '
                    f't={diffusion_start_t}; mode={forward_noise_init_mode}; '
                    'locked scaffold context is reintroduced by RePaint'
                )

        prepared.append({
            'sample_idx': sample_idx,
            'n_extra': n_extra,
            'n_locked': n_locked,
            'n_total': n_total,
            'n_preserved': n_preserved,
            'model_n_scaffold': model_n_scaffold,
            'model_context_indices': list(model_context_indices),
            'exact_record': exact_record,
            'extra_log_v': extra_log_v,
            'init_ligand_pos': init_ligand_pos,
            'init_ligand_v_input': init_ligand_v_input,
            'large_init_pos': large_init_pos,
            'large_init_v': large_init_v,
            'repaint_cfg': repaint_cfg,
            'zero_site_ids': zero_site_ids,
            'preserved_indices': preserved_indices,
            'initial_extra_positions': extra_pos_world.detach().cpu().tolist(),
            'site_place_meta': _site_place_meta,
        })

    gpu_batch = scaffold_dl_gpu_batch_size(config)
    if logger:
        logger.info(
            f'[ScaffoldDynamicLocked] reverse GPU batch_size={gpu_batch} '
            f'for {len(prepared)} samples'
        )
    used_batch = gpu_batch
    for used_batch, _chunk, results in run_gpu_chunks_with_oom_backoff(
        prepared,
        gpu_batch,
        _run_locked_reverse_chunk,
        logger=logger,
    ):
        for item, final_pos_t, log_v_out, dt in results:
            total_time += dt
            item['gpu_reverse_batch_size'] = int(used_batch)
            _emit_locked_sample(item, final_pos_t, log_v_out, dt)
    if logger and prepared:
        logger.info(
            f'[ScaffoldDynamicLocked] reverse used GPU batch_size={used_batch}'
        )

    # ---- 首位插入提取后的骨架（保持下游提取/评估格式兼容）--------------
    # Side-chain-only output deliberately has no full-scaffold sentinel entry;
    # its first atoms are the fixed interface context anchors.
    if n_scaffold > 0 and not side_chain_only:
        scaffold_pos_np = ref_pos_np[scaffold_indices]
        scaffold_v_np = ref_v_np[scaffold_indices]
        scaffold_metric = evaluate_candidate(scaffold_pos_np, scaffold_v_np, ligand_atom_mode, selector_cfg)
        pos_list.insert(0, scaffold_pos_np)
        v_list.insert(0, scaffold_v_np)
        pos_traj_list.insert(0, [])
        v_traj_list.insert(0, [])
        log_v_traj_list.insert(0, [])
        time_list.insert(0, 0.0)
        meta_records.insert(0, {
            'method': 'scaffold_dynamic_locked',
            'ligand_num_atoms': n_scaffold,
            'n_scaffold': n_scaffold,
            'n_extra': 0,
            'scaffold_rmsd': 0.0,
            'rmsd_from_ref': 0.0,
            'time': 0.0,
            'smiles': scaffold_metric.get('smiles'),
            'qed': scaffold_metric.get('metrics', {}).get('qed'),
            'sa': scaffold_metric.get('metrics', {}).get('sa'),
            'status': scaffold_metric.get('status'),
            'is_original': True,
            'is_extracted_scaffold': True,
        })

    if logger:
        valid = sum(1 for m in meta_records if m.get('status') == 'ok' and not m.get('is_original'))
        avg_scaffold_rmsd = np.mean([
            m['scaffold_rmsd'] for m in meta_records
            if not m.get('is_original') and np.isfinite(m.get('scaffold_rmsd', float('nan')))
        ]) if valid > 0 else 0.0
        logger.info(
            f'[ScaffoldDynamicLocked] 完成: {valid}/{num_samples} 有效分子 | '
            f'平均骨架 RMSD={avg_scaffold_rmsd:.3f} Å | 总耗时 {total_time:.2f}s'
        )

    return {
        'pos_list': pos_list,
        'v_list': v_list,
        'pos_traj': pos_traj_list,
        'v_traj': v_traj_list,
        'log_v_traj': log_v_traj_list,
        'time_list': time_list,
        'meta': {
            'method': (
                'scaffold_side_chain_only'
                if side_chain_only else 'scaffold_dynamic_locked'
            ),
            'records': meta_records,
            'scaffold_cfg': {
                'scaffold_source': sc_cfg.get('scaffold_source'),
                'n_scaffold_atoms': n_scaffold,
                'scaffold_indices': scaffold_indices,
                'side_chain_only': side_chain_only,
                'side_chain_context_mode': (
                    'active_exit_anchor_atoms'
                    if side_chain_only else None
                ),
                'fix_scaffold_pos': fix_scaffold_pos,
                'fix_scaffold_type': fix_scaffold_type,
                'lock_extra_atom_types': lock_extra_atom_types,
                'extra_atom_type_mode': extra_atom_type_mode,
                'extra_position_mask': (
                    extra_anchor_strength
                    if allow_soft_extra_anchor
                    else 0.0 if lock_extra_atom_types else extra_anchor_strength
                ),
                'allow_soft_extra_anchor': allow_soft_extra_anchor,
                'extra_type_mask': extra_type_anchor_strength,
                'allow_forced_attachment_bonds': bool(
                    sc_cfg.get('allow_forced_attachment_bonds', False)
                ),
                'scaffold_reconstruction_covalent_factor': float(
                    sc_cfg.get('scaffold_reconstruction_covalent_factor', 1.3)
                ),
                'time_boundary': time_boundary,
                'skip_refine': skip_refine,
                'attachment_sites': _attachment_sites,
                'murcko_sites_cfg': _murcko_sites_cfg,
            },
            'total_time': total_time,
        },
    }
