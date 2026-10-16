"""Sampling prudent scoring responsibilities for ligand sampling."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from diffdynamic.evaluation import scoring_func
import numpy as np
import os
import threading


def _prudent_score_one_candidate(ci, c, *, seed_idx, rnd, prudent_cfg, protein_path, lig_fn,
                                 vina_protein_root, ligand_atom_mode, size_gates, min_q_dock,
                                 min_s_dock, scaffold_bonds=None, n_scaffold=None, logger=None,
                                 log_chains=True, n_rounds=1):
    """单条链打分：重建 → QED/SA/LogP →（达标则）Vina → composite。

    可在工作线程中调用（受体制备已内部加锁，score_only 并行）。
    返回 ``(comp, detail, c, ci, mol, qed, sa, logp)``。
    """
    from diffdynamic.sampling.prudent.reconstruction import _prudent_reconstruct_mol

    bonds = c.get('scaffold_bond_pairs', scaffold_bonds)
    n_sc = c.get('n_scaffold', n_scaffold)
    mol = _prudent_reconstruct_mol(
        c['pos'], c['v'], ligand_atom_mode,
        logger=logger,
        context=f's{seed_idx}r{rnd}c{ci}',
        scaffold_bonds=bonds,
        n_scaffold=n_sc,
    )
    qed, sa, logp, chem_err = _prudent_chem_scores(mol, logger=logger)

    gate_ok, gate_reasons = _prudent_size_gate_ok(
        mol, qed, sa, logp, prudent_cfg, gates=size_gates,
    )
    if not gate_ok:
        comp = float('-inf')
        reject_reason = gate_reasons[0] if gate_reasons else 'qed_sa_rejected'
        if mol is None:
            reject_reason = 'mol_reconstruct_failed_or_none'
        detail = {
            'qed': qed, 'sa': sa, 'logp': logp,
            'vina_affinity': float('nan'), 'vina_norm': 0.0,
            'vina_skipped': True, 'reconstruct_error': chem_err,
            'reason': reject_reason,
        }
        if logger and log_chains:
            qed_s = f'{qed:.4f}' if np.isfinite(qed) else 'nan'
            sa_s = f'{sa:.4f}' if np.isfinite(sa) else 'nan'
            logp_s = f'{logp:.2f}' if np.isfinite(logp) else 'nan'
            gate_hint = ', '.join(gate_reasons) if gate_reasons else reject_reason
            logger.info(
                f'[Prudent][score] seed={seed_idx} r={rnd + 1}/{n_rounds} chain={ci} | '
                f'QED={qed_s} SA={sa_s} LogP={logp_s} | REJECTED ({gate_hint}), '
                f'composite=-inf'
            )
        return comp, detail, c, ci, mol, qed, sa, logp

    comp, detail = _prudent_composite_score(
        mol, prudent_cfg, protein_path, lig_fn,
        logger=logger, protein_root=vina_protein_root,
        prefetched_chem=(qed, sa, logp, chem_err),
        skip_vina=False,
    )
    if logger and log_chains:
        aff = detail.get('vina_affinity')
        aff_s = (
            f'{float(aff):.3f}'
            if aff is not None and np.isfinite(float(aff))
            else 'nan'
        )
        _lip_n = detail.get('lipinski', 0)
        try:
            _lip_n = int(_lip_n) if _lip_n is not None and _lip_n != 'N/A' else 0
        except (TypeError, ValueError):
            _lip_n = 0
        _l_dem = detail.get('lilly_demerit', 0)
        try:
            _l_dem = int(_l_dem or 0)
        except (TypeError, ValueError):
            _l_dem = 0
        _l_pass = bool(detail.get('lilly_passed', False))
        logger.info(
            f'[Prudent][score] seed={seed_idx} r={rnd + 1}/{n_rounds} chain={ci} | '
            f'QED={detail.get("qed", 0.0):.4f} SA={detail.get("sa", 0.0):.4f} '
            f'Vina_kcal={aff_s} vina_norm={detail.get("vina_norm", 0.0):.4f} '
            f'lipinski_n={_lip_n} lipinski_frac={detail.get("lipinski_frac", 0.0):.4f} '
            f'lilly_demerit={_l_dem} lilly_passed={1 if _l_pass else 0} '
            f'lilly_norm={detail.get("lilly_norm", 0.0):.4f} '
            f'composite={comp:.4f}'
        )
    return comp, detail, c, ci, mol, qed, sa, logp


def _score_chain_picklable(args):
    """进程池可 pickle 的单链打分包装。

    ``args``: dict 含 ci/c/各参数（不含 logger，进程池子进程无法共享 logger）。
    返回 ``(comp, detail, c, ci, mol_ok, qed, sa, logp)``，mol 以 bool 形式返回（RDKit Mol 不可 pickle）。
    """
    ci = args['ci']
    c = args['c']
    seed_idx = args['seed_idx']
    rnd = args['rnd']
    n_rounds = args['n_rounds']
    prudent_cfg = args['prudent_cfg']
    protein_path = args['protein_path']
    lig_fn = args['lig_fn']
    vina_protein_root = args['vina_protein_root']
    ligand_atom_mode = args['ligand_atom_mode']
    size_gates = args['size_gates']
    min_q_dock = args['min_q_dock']
    min_s_dock = args['min_s_dock']
    scaffold_bonds = args.get('scaffold_bonds')
    n_scaffold = args.get('n_scaffold')
    log_chains = args.get('log_chains', True)
    comp, detail, c2, ci2, mol, qed, sa, logp = _prudent_score_one_candidate(
        ci, c,
        seed_idx=seed_idx, rnd=rnd, prudent_cfg=prudent_cfg,
        protein_path=protein_path, lig_fn=lig_fn,
        vina_protein_root=vina_protein_root,
        ligand_atom_mode=ligand_atom_mode,
        size_gates=size_gates, min_q_dock=min_q_dock, min_s_dock=min_s_dock,
        scaffold_bonds=scaffold_bonds, n_scaffold=n_scaffold,
        logger=None, log_chains=log_chains, n_rounds=n_rounds,
    )
    return (comp, detail, c2, ci2, mol is not None, qed, sa, logp)


def _prudent_dedup_survivors_smiles(scored, advance_top_k, ligand_atom_mode='add_aromatic'):
    """10 选 2：取 composite 最高的前 k 条，且任意两条 canonical SMILES 不同。

    返回 (survivors, dropped_count)。SMILES 无法解析（None/空）的链视为可保留但优先排在末尾。
    """
    from diffdynamic.sampling.prudent.reconstruction import _prudent_canonical_smiles

    scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)
    survivors = []
    seen = set()
    dropped = 0
    for t in scored_sorted:
        if len(survivors) >= advance_top_k:
            break
        comp, detail, c, ci, *_ = t
        smi = None
        try:
            smi = _prudent_canonical_smiles(
                None, pos=c['pos'], v=c['v'], ligand_atom_mode=ligand_atom_mode,
            )
        except Exception:
            smi = None
        if smi is None:
            # 无法解析 SMILES：能保留则保留（占一个槽位），避免死循环
            survivors.append(t)
            continue
        if smi in seen:
            dropped += 1
            continue
        seen.add(smi)
        survivors.append(t)
    return survivors, dropped


def _prudent_hard_pick_top_k(scored, advance_top_k, dedup_survivors, ligand_atom_mode='add_aromatic'):
    """不满足晋级条件时的硬保底：取 composite 最高的前 k 条（SMILES 去重），不淘汰任何链。

    返回 (survivors, dropped_count)。这样每个 seed 至少产出 k 个分子（pool 有 k 条时）。
    """
    from diffdynamic.sampling.prudent.reconstruction import _prudent_canonical_smiles

    scored_sorted = sorted(scored, key=lambda x: x[0], reverse=True)
    if not dedup_survivors:
        return scored_sorted[:advance_top_k], 0
    survivors = []
    seen = set()
    dropped = 0
    for t in scored_sorted:
        if len(survivors) >= advance_top_k:
            break
        comp, detail, c, ci, *_ = t
        smi = None
        try:
            smi = _prudent_canonical_smiles(
                None, pos=c['pos'], v=c['v'], ligand_atom_mode=ligand_atom_mode,
            )
        except Exception:
            smi = None
        if smi is None:
            survivors.append(t)
            continue
        if smi in seen:
            dropped += 1
            continue
        seen.add(smi)
        survivors.append(t)
    return survivors, dropped


def _prudent_score_one_round(
    pool,
    *,
    seed_idx,
    rnd,
    n_rounds,
    advance_top_k,
    prudent_cfg,
    protein_path,
    lig_fn,
    vina_protein_root,
    ligand_atom_mode,
    vina_advance_thr,
    logger,
    scaffold_bonds=None,
    n_scaffold=None,
    parallel=True,
):
    """一轮 GPU refine 之后：QED/SA/LogP →（条件）Vina → 晋级筛选。仅 CPU，可在工作线程中调用。

    ``parallel=True`` 时用 ``scoring_workers`` 线程池并行池内各链；调用方若已在外层线程池
    （async 分支按 seed 并行）应传 ``parallel=False`` 避免线程池嵌套导致 GIL 竞争。
    """
    min_q_dock = float(prudent_cfg.get('min_qed_for_docking', 0.3))
    min_s_dock = float(prudent_cfg.get('min_sa_for_docking', 0.3))
    size_gates = _prudent_parse_size_gates(prudent_cfg)
    log_chains = prudent_cfg.get('log_per_chain_scores', True)

    n_workers = int(prudent_cfg.get('scoring_workers', 0) or 0)
    n_workers = max(1, n_workers)
    if parallel and n_workers > 1 and len(pool) > 1:
        scored = []
        with ThreadPoolExecutor(max_workers=min(n_workers, len(pool))) as executor:
            futs = {
                executor.submit(
                    _prudent_score_one_candidate,
                    ci, c,
                    seed_idx=seed_idx, rnd=rnd, prudent_cfg=prudent_cfg,
                    protein_path=protein_path, lig_fn=lig_fn,
                    vina_protein_root=vina_protein_root,
                    ligand_atom_mode=ligand_atom_mode,
                    size_gates=size_gates, min_q_dock=min_q_dock, min_s_dock=min_s_dock,
                    scaffold_bonds=scaffold_bonds, n_scaffold=n_scaffold,
                    logger=logger, log_chains=log_chains, n_rounds=n_rounds,
                ): ci
                for ci, c in enumerate(pool)
            }
            for fut in futs:
                scored.append(fut.result())
    else:
        scored = []
        for ci, c in enumerate(pool):
            scored.append(_prudent_score_one_candidate(
                ci, c,
                seed_idx=seed_idx, rnd=rnd, prudent_cfg=prudent_cfg,
                protein_path=protein_path, lig_fn=lig_fn,
                vina_protein_root=vina_protein_root,
                ligand_atom_mode=ligand_atom_mode,
                size_gates=size_gates, min_q_dock=min_q_dock, min_s_dock=min_s_dock,
                scaffold_bonds=scaffold_bonds, n_scaffold=n_scaffold,
                logger=logger, log_chains=log_chains, n_rounds=n_rounds,
            ))

    # 按 composite 排序（-inf 的会被排在最后，自然淘汰）
    scored.sort(key=lambda x: x[0], reverse=True)

    if logger:
        top_msg = scored[0][0] if scored else float('nan')
        logger.info(
            f'[Prudent] seed={seed_idx} round={rnd + 1}/{n_rounds} | '
            f'candidates={len(scored)} | best_composite={top_msg:.4f}'
        )

    filtered = []

    def _annotate_pool(survivor_id_set):
        for t in scored:
            comp, detail, c, ci, mol, qed, sa, logp = t
            el = _prudent_eligible_for_advance(
                mol, qed, sa, detail,
                prudent_cfg, vina_advance_thr,
            )
            aff = detail.get('vina_affinity')
            try:
                aff_f = float(aff) if aff is not None and np.isfinite(float(aff)) else None
            except (TypeError, ValueError):
                aff_f = None
            c['prudent_round_analysis'] = {
                'seed_idx': int(seed_idx),
                'round': int(rnd),
                'chain_idx': int(ci),
                'composite': float(comp) if np.isfinite(float(comp)) else None,
                'qed': detail.get('qed'),
                'sa': detail.get('sa'),
                'vina_affinity': aff_f,
                'vina_norm': detail.get('vina_norm'),
                'eligible_for_advance': bool(el),
                'survivor_topk': bool(id(c) in survivor_id_set),
            }

    for t in scored:
        comp, detail, c, ci, mol, qed, sa, logp = t
        if _prudent_eligible_for_advance(
            mol, qed, sa, detail, prudent_cfg, vina_advance_thr,
        ):
            filtered.append(t)
    if logger:
        n_elim = len(scored) - len(filtered)
        if n_elim:
            logger.info(
                f'[Prudent] seed={seed_idx} r={rnd + 1}/{n_rounds} | '
                f'晋级筛选（QED/SA 门槛 + Vina 阈值 + vina_weight 时须已对接）'
                f'→ {len(filtered)}/{len(scored)} 条可进 advance_top_k，{n_elim} 条淘汰'
            )
    _atk_mode = str(prudent_cfg.get('advance_top_k_mode', 'cap')).strip().lower()
    if _atk_mode == 'cap' and prudent_cfg.get('advance_top_k_strict') is True:
        _atk_mode = 'strict'
    dedup_survivors = bool(prudent_cfg.get('dedup_survivors', False))

    # 晋级候选（QED/SA/Vina 全过门槛）
    survivors = filtered[:advance_top_k]

    if len(survivors) < advance_top_k:
        if not filtered:
            if logger:
                logger.warning(
                    f'[Prudent] seed={seed_idx} r={rnd + 1}/{n_rounds} | '
                    f'无分子满足完整晋级条件（min_qed/min_sa、max_vina_affinity_for_advance、'
                    f'vina_weight 对应须对接等）'
                )
        if scored:
            # 硬保底：不满足晋级也取 composite 最高的前 k 条（SMILES 去重）
            hard, _hard_dropped = _prudent_hard_pick_top_k(
                scored, advance_top_k, dedup_survivors, ligand_atom_mode=ligand_atom_mode,
            )
            # 用硬保底结果补足到 k 条
            seen_ids = {id(t[2]) for t in survivors}
            for t in hard:
                if len(survivors) >= advance_top_k:
                    break
                if id(t[2]) in seen_ids:
                    continue
                seen_ids.add(id(t[2]))
                survivors.append(t)
            if logger:
                logger.info(
                    f'[Prudent] seed={seed_idx} r={rnd + 1}/{n_rounds} | '
                    f'晋级候选不足 {advance_top_k} 条，硬保底选中 {len(survivors)} 条'
                    f'（含 {len(survivors) - min(advance_top_k, len(filtered))} 条未过门槛的 top 链）'
                )

    if _atk_mode in ('strict', 'require_full_k', 'full_k') and len(filtered) < advance_top_k and not survivors:
        if logger:
            logger.warning(
                f'[Prudent] seed={seed_idx} r={rnd + 1}/{n_rounds} | '
                f'advance_top_k_mode=strict：满足晋级条件 {len(filtered)} 条 < '
                f'advance_top_k={advance_top_k}，且无任何可硬保底候选，该 seed 后续轮次跳过'
            )
        _annotate_pool(set())
        return [], [], True

    if dedup_survivors:
        # 最终去重：确保任意两条 survivor SMILES 不同（硬保底已去重，这里兜底）
        survivors, _final_dropped = _prudent_hard_pick_top_k(
            survivors, advance_top_k, True, ligand_atom_mode=ligand_atom_mode,
        )
        if logger and _final_dropped:
            logger.info(
                f'[Prudent] seed={seed_idx} r={rnd + 1}/{n_rounds} | '
                f'dedup_survivors：跳过 {_final_dropped} 条 SMILES 重复候选，选中 {len(survivors)} 条'
            )

    pool_out = [t[2] for t in survivors]
    survivor_ids = {id(c) for c in pool_out}
    for t in survivors:
        _, det, c, _ci, *_ = t
        c['prudent_score_detail'] = det

    _annotate_pool(survivor_ids)

    pending = []
    if vina_advance_thr is not None:
        for rank, t in enumerate(survivors):
            comp, det, c, ci, *_ = t
            aff = det.get('vina_affinity')
            try:
                aff_f = float(aff) if aff is not None and np.isfinite(float(aff)) else None
            except (TypeError, ValueError):
                aff_f = None
            pending.append({
                'seed_idx': seed_idx,
                'round': rnd + 1,
                'n_rounds': n_rounds,
                'chain_idx': ci,
                'rank_in_round': rank,
                'vina_affinity': aff_f,
                'composite': float(comp) if np.isfinite(float(comp)) else None,
                'qed': det.get('qed'),
                'sa': det.get('sa'),
                'max_vina_affinity_for_advance': vina_advance_thr,
            })
    return pool_out, pending, False


def _prudent_score_all_chains(pools_by_seed, *, chk, n_rounds, prudent_cfg, protein_path,
                              lig_fn, vina_protein_root, ligand_atom_mode, size_gates,
                              min_q_dock, min_s_dock, scoring_workers, scaffold_bonds=None,
                              n_scaffold=None, logger=None, log_chains=True,
                              use_process_pool=False):
    """对多 seed 的 pool 全链并行打分。

    ``pools_by_seed``: {seed_idx: [cand, ...]}。返回 {seed_idx: [(comp, detail, c, ci, mol, qed, sa, logp), ...]}
    该函数只打分不筛选；筛选由调用方按 seed 执行（避免跨 seed 边界破坏 survivor 语义）。

    ``use_process_pool=True`` 时用 ``ProcessPoolExecutor``（绕开 GIL，RDKit/Vina 可真正多核并行）；
    返回元组第 5 项为 ``mol_ok: bool`` 而非 RDKit Mol（Mol 不可 pickle）。
    """
    # OmegaConf DictConfig 不可 pickle，转普通 dict 供进程池使用（线程池也无害）
    if not isinstance(prudent_cfg, dict):
        try:
            prudent_cfg = dict(prudent_cfg)
        except Exception:
            prudent_cfg = {k: prudent_cfg[k] for k in prudent_cfg.keys()}
    if not isinstance(size_gates, dict):
        try:
            size_gates = dict(size_gates)
        except Exception:
            size_gates = {k: size_gates[k] for k in size_gates.keys()}
    total_items = []  # (seed_idx, chain_idx, cand)
    for seed_idx, pool in pools_by_seed.items():
        for ci, c in enumerate(pool):
            total_items.append((seed_idx, ci, c))
    results = {}
    n_workers = max(1, int(scoring_workers or 0))
    if n_workers > 1 and len(total_items) > 1:
        if use_process_pool:
            from concurrent.futures import ProcessPoolExecutor
            args_list = []
            for (seed_idx, ci, c) in total_items:
                args_list.append({
                    'ci': ci, 'c': c, 'seed_idx': seed_idx, 'rnd': chk,
                    'n_rounds': n_rounds, 'prudent_cfg': prudent_cfg,
                    'protein_path': protein_path, 'lig_fn': lig_fn,
                    'vina_protein_root': vina_protein_root,
                    'ligand_atom_mode': ligand_atom_mode,
                    'size_gates': size_gates, 'min_q_dock': min_q_dock,
                    'min_s_dock': min_s_dock,
                    'scaffold_bonds': scaffold_bonds, 'n_scaffold': n_scaffold,
                    'log_chains': False,
                })
            # 进程池 worker 上限 24：fork 继承父进程内存（含 CUDA context ~9GB），
            # 24 worker × ~10GB ≈ 240GB，接近机器内存上限，避免 OOM
            _proc_workers = min(n_workers, len(total_items), 24)
            with ProcessPoolExecutor(max_workers=_proc_workers) as executor:
                out_rows = list(executor.map(_score_chain_picklable, args_list))
            for (seed_idx, ci, c), scored_row in zip(total_items, out_rows):
                results.setdefault(seed_idx, []).append(scored_row)
        else:
            with ThreadPoolExecutor(max_workers=min(n_workers, len(total_items))) as executor:
                futs = {}
                for (seed_idx, ci, c) in total_items:
                    futs[executor.submit(
                        _prudent_score_one_candidate,
                        ci, c,
                        seed_idx=seed_idx, rnd=chk, prudent_cfg=prudent_cfg,
                        protein_path=protein_path, lig_fn=lig_fn,
                        vina_protein_root=vina_protein_root,
                        ligand_atom_mode=ligand_atom_mode,
                        size_gates=size_gates, min_q_dock=min_q_dock, min_s_dock=min_s_dock,
                        scaffold_bonds=scaffold_bonds, n_scaffold=n_scaffold,
                        logger=logger, log_chains=log_chains, n_rounds=n_rounds,
                    )] = (seed_idx, ci, c)
                for fut in futs:
                    seed_idx, ci, c = futs[fut]
                    scored_row = fut.result()
                    results.setdefault(seed_idx, []).append(scored_row)
    else:
        for (seed_idx, ci, c) in total_items:
            scored_row = _prudent_score_one_candidate(
                ci, c,
                seed_idx=seed_idx, rnd=chk, prudent_cfg=prudent_cfg,
                protein_path=protein_path, lig_fn=lig_fn,
                vina_protein_root=vina_protein_root,
                ligand_atom_mode=ligand_atom_mode,
                size_gates=size_gates, min_q_dock=min_q_dock, min_s_dock=min_s_dock,
                scaffold_bonds=scaffold_bonds, n_scaffold=n_scaffold,
                logger=logger, log_chains=log_chains, n_rounds=n_rounds,
            )
            results.setdefault(seed_idx, []).append(scored_row)
    return results


def _prudent_select_survivors(scored, *, seed_idx, rnd, n_rounds, advance_top_k,
                              prudent_cfg, vina_advance_thr, ligand_atom_mode, logger=None):
    """按 composite 排序 + 晋级筛选 + 硬保底 + SMILES 去重，选出 survivor。

    返回 (pool_out, pending, broke)，与 _prudent_score_one_round 尾部一致。
    """
    scored = sorted(scored, key=lambda x: x[0], reverse=True)
    if logger:
        top_msg = scored[0][0] if scored else float('nan')
        logger.info(
            f'[Prudent] seed={seed_idx} round={rnd + 1}/{n_rounds} | '
            f'candidates={len(scored)} | best_composite={top_msg:.4f}'
        )

    def _annotate_pool(survivor_id_set):
        for t in scored:
            comp, detail, c, ci, mol_or_ok, qed, sa, logp = t
            el = _prudent_eligible_for_advance(
                mol_or_ok if hasattr(mol_or_ok, 'GetNumAtoms') else (None if not mol_or_ok else True),
                qed, sa, detail, prudent_cfg, vina_advance_thr,
            )
            aff = detail.get('vina_affinity')
            try:
                aff_f = float(aff) if aff is not None and np.isfinite(float(aff)) else None
            except (TypeError, ValueError):
                aff_f = None
            c['prudent_round_analysis'] = {
                'seed_idx': int(seed_idx),
                'round': int(rnd),
                'chain_idx': int(ci),
                'composite': float(comp) if np.isfinite(float(comp)) else None,
                'qed': detail.get('qed'),
                'sa': detail.get('sa'),
                'vina_affinity': aff_f,
                'vina_norm': detail.get('vina_norm'),
                'eligible_for_advance': bool(el),
                'survivor_topk': bool(id(c) in survivor_id_set),
            }

    filtered = []
    for t in scored:
        comp, detail, c, ci, mol_or_ok, qed, sa, logp = t
        if _prudent_eligible_for_advance(
            mol_or_ok if hasattr(mol_or_ok, 'GetNumAtoms') else (None if not mol_or_ok else True),
            qed, sa, detail, prudent_cfg, vina_advance_thr,
        ):
            filtered.append(t)
    if logger:
        n_elim = len(scored) - len(filtered)
        if n_elim:
            logger.info(
                f'[Prudent] seed={seed_idx} r={rnd + 1}/{n_rounds} | '
                f'晋级筛选（QED/SA 门槛 + Vina 阈值 + vina_weight 时须已对接）'
                f'→ {len(filtered)}/{len(scored)} 条可进 advance_top_k，{n_elim} 条淘汰'
            )

    _atk_mode = str(prudent_cfg.get('advance_top_k_mode', 'cap')).strip().lower()
    if _atk_mode == 'cap' and prudent_cfg.get('advance_top_k_strict') is True:
        _atk_mode = 'strict'
    dedup_survivors = bool(prudent_cfg.get('dedup_survivors', False))

    survivors = filtered[:advance_top_k]
    if len(survivors) < advance_top_k:
        if not filtered and logger:
            logger.warning(
                f'[Prudent] seed={seed_idx} r={rnd + 1}/{n_rounds} | '
                f'无分子满足完整晋级条件（min_qed/min_sa、max_vina_affinity_for_advance、'
                f'vina_weight 对应须对接等）'
            )
        if scored:
            hard, _hard_dropped = _prudent_hard_pick_top_k(
                scored, advance_top_k, dedup_survivors, ligand_atom_mode=ligand_atom_mode,
            )
            seen_ids = {id(t[2]) for t in survivors}
            for t in hard:
                if len(survivors) >= advance_top_k:
                    break
                if id(t[2]) in seen_ids:
                    continue
                seen_ids.add(id(t[2]))
                survivors.append(t)
            if logger:
                logger.info(
                    f'[Prudent] seed={seed_idx} r={rnd + 1}/{n_rounds} | '
                    f'晋级候选不足 {advance_top_k} 条，硬保底选中 {len(survivors)} 条'
                    f'（含 {len(survivors) - min(advance_top_k, len(filtered))} 条未过门槛的 top 链）'
                )

    if _atk_mode in ('strict', 'require_full_k', 'full_k') and len(filtered) < advance_top_k and not survivors:
        if logger:
            logger.warning(
                f'[Prudent] seed={seed_idx} r={rnd + 1}/{n_rounds} | '
                f'advance_top_k_mode=strict：满足晋级条件 {len(filtered)} 条 < '
                f'advance_top_k={advance_top_k}，且无任何可硬保底候选，该 seed 后续轮次跳过'
            )
        _annotate_pool(set())
        return [], [], True

    if dedup_survivors:
        survivors, _final_dropped = _prudent_hard_pick_top_k(
            survivors, advance_top_k, True, ligand_atom_mode=ligand_atom_mode,
        )
        if logger and _final_dropped:
            logger.info(
                f'[Prudent] seed={seed_idx} r={rnd + 1}/{n_rounds} | '
                f'dedup_survivors：跳过 {_final_dropped} 条 SMILES 重复候选，选中 {len(survivors)} 条'
            )

    pool_out = [t[2] for t in survivors]
    survivor_ids = {id(c) for c in pool_out}
    for t in survivors:
        _, det, c, _ci, *_ = t
        c['prudent_score_detail'] = det

    _annotate_pool(survivor_ids)

    pending = []
    if vina_advance_thr is not None:
        for rank, t in enumerate(survivors):
            comp, det, c, ci, *_ = t
            aff = det.get('vina_affinity')
            try:
                aff_f = float(aff) if aff is not None and np.isfinite(float(aff)) else None
            except (TypeError, ValueError):
                aff_f = None
            pending.append({
                'seed_idx': seed_idx,
                'round': rnd + 1,
                'n_rounds': n_rounds,
                'chain_idx': ci,
                'rank_in_round': rank,
                'vina_affinity': aff_f,
                'composite': float(comp) if np.isfinite(float(comp)) else None,
                'qed': det.get('qed'),
                'sa': det.get('sa'),
                'max_vina_affinity_for_advance': vina_advance_thr,
            })
    return pool_out, pending, False


def _prudent_vina_ok_for_next_round(detail, max_aff_thr):
    """对接完成后：finite Vina affinity（kcal/mol）且严格小于阈值才可晋级下一轮；未对接/nan 淘汰。

    ``max_aff_thr`` 为 ``None`` 时不做此项筛选（与旧行为兼容）。
    """
    if max_aff_thr is None:
        return True
    if not detail or detail.get('vina_skipped'):
        return False
    aff = detail.get('vina_affinity')
    if aff is None:
        return False
    try:
        a = float(aff)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(a):
        return False
    return a < float(max_aff_thr)


def _prudent_eligible_for_advance(mol, qed, sa, detail, prudent_cfg, vina_advance_thr):
    """是否满足晋级下一轮的全部条件（QED/SA 已在前面一票否决，这里只检查 Vina 相关）。"""
    if mol is None:
        return False
    # QED/SA 已在 _prudent_score_one_round 中一票否决，这里只需检查 Vina 相关
    if not _prudent_vina_ok_for_next_round(detail, vina_advance_thr):
        return False
    wv = float(prudent_cfg.get('vina_weight', 0.6))
    if wv > 0.0 and (not detail or detail.get('vina_skipped')):
        return False
    return True


def _prudent_chem_scores(mol, logger=None):
    """单分子 QED/SA/LogP；mol 无效或 get_chem 失败时返回 (nan, nan, nan, error_or_none)。"""
    if mol is None:
        return float('nan'), float('nan'), float('nan'), 'mol_reconstruct_failed_or_none'
    try:
        chem = scoring_func.get_chem(mol)
        qed_raw = chem.get('qed')
        sa_raw = chem.get('sa')
        logp_raw = chem.get('logp')
        qed = float(qed_raw) if qed_raw is not None else float('nan')
        sa = float(sa_raw) if sa_raw is not None else float('nan')
        logp = float(logp_raw) if logp_raw is not None else float('nan')
        return qed, sa, logp, None
    except Exception as exc:
        if logger:
            logger.debug(f'[Prudent] get_chem 失败（无效价态等），按 QED/SA/LogP=nan 处理: {exc}')
        return float('nan'), float('nan'), float('nan'), str(exc)


def _prudent_struct_scores(mol):
    """分子量 / 重原子数 / 环数；mol 无效时返回 nan。"""
    if mol is None:
        return float('nan'), float('nan'), float('nan')
    try:
        from rdkit.Chem import Descriptors, Lipinski
        return (
            float(Descriptors.MolWt(mol)),
            float(mol.GetNumHeavyAtoms()),
            float(Lipinski.RingCount(mol)),
        )
    except Exception:
        return float('nan'), float('nan'), float('nan')


def _prudent_parse_size_gates(prudent_cfg):
    """解析可选的分子尺寸硬门控阈值。"""
    gates = {}
    for key, cfg_key in (
        ('max_molwt', 'max_molwt'),
        ('max_heavy_atoms', 'max_heavy_atoms'),
        ('max_rings', 'max_rings'),
    ):
        val = prudent_cfg.get(cfg_key, None)
        if val is not None:
            gates[key] = float(val)
    max_logp = prudent_cfg.get('max_logp', None)
    if max_logp is not None:
        gates['max_logp'] = float(max_logp)
    return gates


def _prudent_size_gate_ok(mol, qed, sa, logp, prudent_cfg, gates=None):
    """QED/SA/LogP 及可选 molwt / heavy_atoms / rings 硬门控。"""
    if gates is None:
        gates = _prudent_parse_size_gates(prudent_cfg)
    min_q = float(prudent_cfg.get('min_qed_for_docking', 0.15))
    min_s = float(prudent_cfg.get('min_sa_for_docking', 0.15))
    if mol is None:
        return False, ['mol=None']
    reasons = []
    if not np.isfinite(qed) or qed < min_q:
        reasons.append(f'QED={qed:.4f}<{min_q}' if np.isfinite(qed) else f'QED=nan<{min_q}')
    if not np.isfinite(sa) or sa < min_s:
        reasons.append(f'SA={sa:.4f}<{min_s}' if np.isfinite(sa) else f'SA=nan<{min_s}')
    max_logp = gates.get('max_logp')
    if max_logp is not None and np.isfinite(logp) and logp > max_logp:
        reasons.append(f'LogP={logp:.2f}>{max_logp}')
    molwt, heavy, rings = _prudent_struct_scores(mol)
    max_molwt = gates.get('max_molwt')
    if max_molwt is not None and np.isfinite(molwt) and molwt > max_molwt:
        reasons.append(f'MolWt={molwt:.0f}>{max_molwt:.0f}')
    max_heavy = gates.get('max_heavy_atoms')
    if max_heavy is not None and np.isfinite(heavy) and heavy > max_heavy:
        reasons.append(f'HeavyAtoms={heavy:.0f}>{max_heavy:.0f}')
    max_rings = gates.get('max_rings')
    if max_rings is not None and np.isfinite(rings) and rings > max_rings:
        reasons.append(f'Rings={rings:.0f}>{max_rings:.0f}')
    return len(reasons) == 0, reasons


def _prudent_composite_score(
    mol, prudent_cfg, protein_path, ligand_filename, logger=None, protein_root=None,
    prefetched_chem=None,
    skip_vina=False,
):
    """Prudent 综合分：Vina_norm、QED、SA 及可选 Lipinski（规则条数/5）、Lilly（扣分归一化）加权，权重自 YAML 归一化。

    Prudent 采样阶段在此直接 ``VinaDockingTask.run``，不通过 ``subprocess`` 调用 ``evaluate_pt`` /
    ``evaluate_pt_with_correct_reconstruct``（后两者用于离线评测流程）。

    ``protein_path`` 为可直接对接的 PDB；若无效则与 ``evaluate_pt`` 相同，用 ``protein_root`` +
    配体相对路径推断受体（``VinaDockingTask.from_generated_mol``）。

    ``prefetched_chem``：``(qed, sa, logp, chem_error)`` 或旧的 ``(qed, sa, chem_error)``，避免重复 get_chem；``skip_vina=True`` 时不跑对接（每轮先统一算 QED/SA/LogP，未过阈者跳过 Vina 以省算力）。
    """
    w_v = max(0.0, float(prudent_cfg.get('vina_weight', 0.6)))
    w_q = max(0.0, float(prudent_cfg.get('qed_weight', 0.2)))
    w_s = max(0.0, float(prudent_cfg.get('sa_weight', 0.2)))
    w_li = max(0.0, float(prudent_cfg.get('lipinski_weight', 0.0)))
    w_ly = max(0.0, float(prudent_cfg.get('lilly_demerit_weight', 0.0)))
    s = w_v + w_q + w_s + w_li + w_ly
    if s <= 0:
        w_v, w_q, w_s, w_li, w_ly = 0.6, 0.2, 0.2, 0.0, 0.0
        s = 1.0
    else:
        w_v, w_q, w_s, w_li, w_ly = w_v / s, w_q / s, w_s / s, w_li / s, w_ly / s

    if mol is None:
        return float('-inf'), {'reason': 'mol_none'}

    _prefetched_logp = float('nan')
    if prefetched_chem is not None:
        if len(prefetched_chem) >= 4:
            qed, sa, _prefetched_logp, chem_error = prefetched_chem[:4]
        else:
            qed, sa, chem_error = prefetched_chem
        qed = float(qed or 0.0)
        sa = float(sa or 0.0)
        _prefetched_logp = float(_prefetched_logp or float('nan'))
    else:
        qed, sa, _prefetched_logp, chem_error = _prudent_chem_scores(mol, logger=logger)

    aff = float('nan')
    vina_norm = 0.0
    pp = str(protein_path) if protein_path else None
    use_pp = pp and os.path.isfile(pp)
    root = protein_root or os.environ.get('PROTEIN_ROOT', '').strip() or None
    if root:
        try:
            root = str(Path(root).expanduser().resolve())
        except Exception:
            pass
    if not skip_vina:
        r = None
        try:
            from diffdynamic.evaluation.docking_vina import VinaDockingTask

            task = VinaDockingTask.from_generated_mol(
                mol,
                ligand_filename or 'N/A',
                protein_path=pp if use_pp else None,
                protein_root=root if root else './data/crossdocked',
                tmp_dir=prudent_cfg.get('vina_tmp_dir', './tmp'),
            )
            ex = int(prudent_cfg.get('vina_exhaustiveness', 8))
            # 子线程不能用 SIGALRM 超时（signal 仅主线程）；timeout_sec=0 走无闹钟路径
            _vina_ts = None
            if threading.current_thread() is not threading.main_thread():
                _vina_ts = 0
            r = task.run(
                mode='score_only', exhaustiveness=ex, n_poses=1, timeout_sec=_vina_ts,
            )
            if r and r[0].get('affinity') is not None:
                aff = float(r[0]['affinity'])
        except Exception as exc:
            if logger:
                logger.warning(f'[Prudent] Vina score_only 失败: {exc}')
            aff = float('nan')
        if not use_pp and not root and logger:
            logger.debug(
                '[Prudent] 未提供有效 protein_path 与 protein_root，Vina 依赖 from_generated_mol 默认 protein_root'
            )
    elif logger:
        logger.debug(
            '[Prudent] 跳过 Vina（QED/SA 未同时达到 dock阈值，prefetched QED=%.4f SA=%.4f）',
            qed, sa,
        )

    if np.isfinite(aff):
        vina_norm = (-aff) / 10.0
        vina_norm = max(0.0, min(1.0, float(vina_norm)))
    else:
        vina_norm = 0.0

    lipinski_count = 0
    lipinski_frac = 0.0
    try:
        lip_d = scoring_func.get_lipinski(mol)
        raw_li = lip_d.get('lipinski', 0) if isinstance(lip_d, dict) else 0
        if raw_li == 'N/A' or raw_li is None:
            lipinski_count = 0
        else:
            lipinski_count = int(raw_li)
        lipinski_frac = max(0.0, min(1.0, float(lipinski_count) / 5.0))
    except Exception:
        lipinski_frac = 0.0

    lilly_norm = 0.0
    lilly_demerit = 0
    lilly_passed = False
    lilly_cutoff = 100
    try:
        from diffdynamic.evaluation.lilly_medchem_rules import evaluate_lilly_medchem_rules

        lr = evaluate_lilly_medchem_rules(mol, debug=False)
        if lr:
            lilly_demerit = int(lr.get('demerit', 0) or 0)
            lilly_cutoff = max(1, int(lr.get('demerit_cutoff', 100) or 100))
            lilly_passed = bool(lr.get('passed', False))
            if lilly_passed:
                lilly_norm = max(0.0, 1.0 - float(lilly_demerit) / float(lilly_cutoff))
            else:
                lilly_norm = 0.0
    except Exception:
        pass

    composite = (
        w_v * vina_norm + w_q * qed + w_s * sa + w_li * lipinski_frac + w_ly * lilly_norm
    )
    detail = {
        'vina_affinity': aff,
        'vina_norm': vina_norm,
        'qed': qed,
        'sa': sa,
        'logp': _prefetched_logp,
        'lipinski': lipinski_count,
        'lipinski_frac': lipinski_frac,
        'lilly_demerit': lilly_demerit,
        'lilly_passed': lilly_passed,
        'lilly_norm': lilly_norm,
        'lilly_demerit_cutoff': lilly_cutoff,
        'weights': {
            'vina': w_v, 'qed': w_q, 'sa': w_s,
            'lipinski': w_li, 'lilly_demerit': w_ly,
        },
    }
    if chem_error is not None:
        detail['chem_error'] = chem_error
    if skip_vina:
        detail['vina_skipped'] = 'qed_sa_gate'
    return composite, detail
