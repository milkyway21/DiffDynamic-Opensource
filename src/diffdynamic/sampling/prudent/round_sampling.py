from __future__ import annotations

"""Explicit, disjoint Prudent rounds with independent parent-path selection."""

import time
from types import SimpleNamespace

import numpy as np
import torch
from rdkit import Chem


def split_round_schedule(model, dynamic_cfg):
    """Build the original lambda schedule once and consume its capped prefix."""
    prudent = dynamic_cfg['prudent']
    refine = dynamic_cfg['refine']
    steps = [int(x) for x in prudent['round_grad_fusion_steps']]
    top_k = [int(x) for x in prudent['round_advance_top_k']]
    budget = int(refine['max_grad_fusion_iterations'])
    if (not steps or len(steps) != len(top_k) or min(steps + top_k) < 1
            or sum(steps) != budget):
        raise ValueError('Round lengths/top-k must match and sum to refine budget')
    if int(refine.get('cycles', 1)) != 1 or refine.get('schedule') != 'lambda':
        raise ValueError('Fixed Prudent rounds require lambda schedule, cycles=1')
    if refine.get('use_time_scale', False):
        raise ValueError('Fixed rounds require use_time_scale=false')
    n_sampling = int(prudent['n_sampling'])
    if max(top_k) > n_sampling:
        raise ValueError('Round top-k exceeds children per parent')
    full = model._build_lambda_schedule(
        start_t=int(dynamic_cfg.get('time_boundary', 650)),
        end_t=int(refine.get('time_lower', 0)),
        coeff_a=refine.get('lambda_coeff_a', 60),
        coeff_b=refine.get('lambda_coeff_b', 5),
    )
    schedule = [int(t) for t in full[:budget]]
    if len(schedule) != budget or len(set(schedule)) != budget:
        raise ValueError('Original lambda schedule cannot provide requested budget')
    offsets = np.cumsum([0] + steps)
    slices = [schedule[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    return schedule, slices, top_k


def _finite(value):
    try:
        return float(value) if np.isfinite(float(value)) else None
    except (ValueError, TypeError):
        return None


def _sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def run_fixed_rounds(model, data, config, ligand_atom_mode, ls_out,
                     total_candidates, *, helpers, device='cuda:0', logger=None,
                     skip_targetdiff_baseline_refine=False):
    """Gate actual intermediate states; each survivor owns its next family."""
    h = SimpleNamespace(**helpers)
    dynamic = config.sample['dynamic']
    prudent = dynamic['prudent']
    refine = dynamic['refine']
    schedule, segments, top_k = split_round_schedule(model, dynamic)
    n_sampling = int(prudent['n_sampling'])
    center_mode = config.sample.get('center_pos_mode', 'protein')
    pos_only = config.sample.get('pos_only', False)
    parents = []
    for root, candidate in enumerate(total_candidates):
        parent = h._prudent_copy_candidate(candidate)
        parent['lineage'] = {'root_seed': root, 'candidate_id': f's{root}',
                             'parent_id': None}
        parents.append(parent)
    round_rows, lineage_rows, shortages = [], [], []
    run_start = time.perf_counter()
    for rnd, (segment, keep) in enumerate(zip(segments, top_k)):
        children, families = [], {}
        for family, parent in enumerate(parents):
            families[family] = []
            for child_idx in range(n_sampling):
                child = h._prudent_copy_candidate(parent)
                child['lineage'] = {
                    'root_seed': parent['lineage']['root_seed'],
                    'parent_id': parent['lineage']['candidate_id'],
                    'candidate_id': f"{parent['lineage']['candidate_id']}/r{rnd+1}c{child_idx}",
                    'round': rnd + 1, 'family': family, 'child_index': child_idx,
                }
                children.append(child)
                families[family].append(child)
        _sync(device)
        started = time.perf_counter()
        updated = h._prudent_refine_batch(
            model, data, children, dynamic, refine, center_mode, pos_only, device,
            time_upper=segment[0], time_lower=segment[-1],
            max_grad_cap=len(segment), logger=logger,
            explicit_time_indices=segment, continuation=rnd > 0,
        )
        _sync(device)
        gpu_s = time.perf_counter() - started
        for old, new in zip(children, updated):
            old.update(new)
        started = time.perf_counter()
        scored = h._prudent_score_all_chains(
            families, chk=rnd, n_rounds=len(segments), prudent_cfg=prudent,
            protein_path=getattr(data, 'protein_filename', None),
            lig_fn=getattr(data, 'ligand_filename', None),
            vina_protein_root=prudent.get('protein_root'),
            ligand_atom_mode=ligand_atom_mode,
            size_gates=h._prudent_parse_size_gates(prudent),
            min_q_dock=float(prudent.get('min_qed_for_docking', .2)),
            min_s_dock=float(prudent.get('min_sa_for_docking', .2)),
            scoring_workers=int(prudent.get('scoring_workers', 10)),
            logger=logger, log_chains=prudent.get('log_per_chain_scores', False),
            use_process_pool=False,
        )
        parents = []
        for family, scores in scored.items():
            selected, _, _ = h._prudent_select_survivors(
                scores, seed_idx=family, rnd=rnd, n_rounds=len(segments),
                advance_top_k=keep, prudent_cfg=prudent,
                vina_advance_thr=prudent.get('max_vina_affinity_for_advance'),
                ligand_atom_mode=ligand_atom_mode, logger=logger,
            )
            parents.extend(selected)
            if len(selected) < keep:
                shortages.append({'round': rnd + 1, 'family': family,
                                  'requested': keep, 'actual': len(selected),
                                  'reason': 'existing_gate_fallback_smiles_dedup'})
            selected_ids = {id(c) for c in selected}
            for rank, score in enumerate(sorted(scores, key=lambda x: x[0],
                                                 reverse=True), 1):
                comp, detail, candidate, _, mol, *_ = score
                try:
                    smiles = Chem.MolToSmiles(mol, canonical=True) if mol else None
                except Exception:
                    smiles = None
                annotation = candidate.get('prudent_round_analysis', {})
                eligible = bool(annotation.get('eligible_for_advance'))
                selected_flag = id(candidate) in selected_ids
                lineage_rows.append({
                    **candidate['lineage'], 'schedule_start': segment[0],
                    'schedule_end': segment[-1],
                    'cumulative_refine_steps': sum(map(len, segments[:rnd+1])),
                    'smiles': smiles, 'reconstructed': bool(mol),
                    'complete': bool(smiles and '.' not in smiles),
                    'rank_in_family': rank, 'selected': selected_flag,
                    'eligible': eligible, 'fallback': selected_flag and not eligible,
                    'qed': _finite(detail.get('qed')), 'sa': _finite(detail.get('sa')),
                    'vina_affinity': _finite(detail.get('vina_affinity')),
                    'composite': _finite(comp),
                    'reason': detail.get('reason', detail.get('reconstruct_error')),
                })
        scoring_s = time.perf_counter() - started
        round_rows.append({'round': rnd + 1, 'time_indices': segment,
                           'n_parents': len(families), 'n_candidates': len(children),
                           'n_selected': len(parents), 'advance_top_k': keep,
                           'gpu_s': gpu_s, 'scoring_s': scoring_s})
        if logger:
            logger.info('[PrudentRounds] round=%s steps=%s parents=%s candidates=%s selected=%s',
                        rnd + 1, segment, len(families), len(children), len(parents))

    records = []
    for candidate in parents:
        record = dict(candidate)
        record['source_index'] = candidate['lineage']['root_seed']
        record['prudent_composite_detail'] = candidate.get('prudent_score_detail', {})
        record['gate_score_state'] = 'pre_final_TBR10'
        records.append(record)
    positions = [c['pos'] for c in records]
    types = [c['v'] for c in records]
    pos_traj = [c.get('pos_traj', []) for c in records]
    v_traj = [c.get('v_traj', []) for c in records]
    _sync(device)
    started = time.perf_counter()
    baseline_indices = None
    if not skip_targetdiff_baseline_refine:
        result = h.apply_targetdiff_baseline_refinement(
            model, data, positions, types, config, device=device, logger=logger)
        if len(result) == 5:
            positions, types, p_add, v_add, baseline_indices = result
            pos_traj = [list(p) + list(a) for p, a in zip(pos_traj, p_add)]
            v_traj = [list(v) + list(a) for v, a in zip(v_traj, v_add)]
        else:
            positions, types = result
    _sync(device)
    baseline_s = time.perf_counter() - started
    for record, pos, v in zip(records, positions, types):
        record['pos'], record['v'] = pos, v
        record['final_state'] = 'post_final_TBR10'
    ls_wall = float(ls_out['meta']['prudent_large_step_wall_s'])
    complete_s = ls_wall + time.perf_counter() - run_start
    gpu_s = sum(r['gpu_s'] for r in round_rows)
    cpu_s = sum(r['scoring_s'] for r in round_rows)
    timing = {'large_step_s': ls_wall, 'gpu_s': gpu_s, 'cpu_s': cpu_s,
              'baseline_s': baseline_s, 'total_s': complete_s,
              'n_mol': len(positions), 'target_n_mol': 100,
              'normalized_100_s': complete_s * 100 / len(positions) if positions else None,
              'status': 'ok' if len(positions) == 100 else 'shortfall',
              'rounds': round_rows, 'shortage_reasons': shortages}
    meta = dict(ls_out.get('meta', {}))
    meta.update({'refined_candidates': records, 'timing': timing,
                 'baseline_refine_time_indices': baseline_indices,
                 'time_records': {'large_step': [ls_wall],
                                  'refine': [r['gpu_s'] for r in round_rows],
                                  'scoring': [r['scoring_s'] for r in round_rows],
                                  'baseline': [baseline_s]}})
    meta['prudent'] = {
        'mode': 'fixed_direct_intermediate_rounds', 'n_sampling': n_sampling,
        'round_grad_fusion_steps': [len(s) for s in segments],
        'round_advance_top_k': top_k, 'refine_time_indices': schedule,
        'max_grad_fusion_iterations': len(schedule),
        'n_rounds': len(segments), 'weights': {
            k: prudent.get(k + '_weight', 0) for k in ('qed', 'sa', 'vina', 'lipinski', 'lilly_demerit')},
        'candidate_lineage': lineage_rows, 'shortages': shortages,
        'dedup_scope': 'within each parent family, existing strict SMILES rule',
        'final_tbr_calls': int(baseline_indices is not None),
    }
    if logger:
        logger.info('[Timing] gpu=%.1fs cpu=%.1fs baseline=%.1fs total=%.1fs n_mol=%s',
                    gpu_s, cpu_s, baseline_s, complete_s, len(positions))
    return {'pos_list': positions, 'v_list': types, 'pos_traj': pos_traj,
            'v_traj': v_traj, 'log_v_traj': [c.get('log_v_traj', []) for c in records],
            'time_list': [ls_wall] + [r['gpu_s'] for r in round_rows], 'meta': meta}
