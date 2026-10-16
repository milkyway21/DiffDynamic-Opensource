from __future__ import annotations

"""Complete-rollout Prudent gates with exact rewind and immutable incumbents.

The sampler helpers own diffusion. This module owns search bookkeeping and the
terminal scoring contract; the archived direct-intermediate branch is separate.
"""

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch

from diffdynamic.sampling.prudent.round_sampling import _finite, _sync, split_round_schedule

LEGACY_SHA256 = '2d89a69cb9cae97e7d1e14b7c7e097c6d7aa25efb351c649c84893ec8f755d9b'
MODE = 'terminal_rollout_backtrack'


def state_hash(pos, types):
    """Hash exact scored/output arrays, including dtype and shape."""
    digest = hashlib.sha256()
    for array in (pos, types):
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _worker_init():
    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      MKL_NUM_THREADS='1', DIFFDYNAMIC_VINA_CPU='1')
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')


def terminal_score(payload):
    """Use the offline evaluator's exact reconstruction and fragment functions."""
    from rdkit import Chem
    from diffdynamic.evaluation.molecule.reconstruction import (
        reconstruct_molecule, remove_small_fragments,
    )
    from diffdynamic.evaluation import scoring_func
    from diffdynamic.evaluation.docking_vina import VinaDockingTask

    pos, types, atom_mode, protein, ligand, cfg = payload
    started = time.perf_counter()
    result = dict(reconstructed=False, complete=False, eligible=False,
                  smiles=None, raw_smiles=None, qed=None, sa=None,
                  vina_affinity=None, composite=None, raw_composite=None,
                  reconstruction_method='targetdiff_legacy',
                  state_sha256=state_hash(pos, types), reason=None,
                  cpu_affinity=sorted(os.sched_getaffinity(0)), vina_cpu=1)
    try:
        mol, error = reconstruct_molecule(
            pos, types, atom_mode=atom_mode,
            reconstruction_method='targetdiff_legacy')
        if mol is None:
            result['reason'] = 'reconstruction_failure'
            result['error'] = error
            return result
        result['reconstructed'] = True
        result['raw_smiles'] = Chem.MolToSmiles(mol, canonical=True)
        mol = remove_small_fragments(mol)
        smiles = Chem.MolToSmiles(mol, canonical=True) if mol else None
        result['smiles'] = smiles
        parsed = Chem.MolFromSmiles(smiles) if smiles else None
        valid = parsed is not None and len(Chem.GetMolFrags(parsed)) == 1
        if valid:
            Chem.SanitizeMol(Chem.Mol(mol))
        result['complete'] = valid
        chem = scoring_func.get_chem(mol) if mol else {}
        result.update(qed=_finite(chem.get('qed')), sa=_finite(chem.get('sa')))
        q, sa = result['qed'], result['sa']
        if not valid:
            result['reason'] = 'invalid_or_multiple_fragments'
            return result
        if (q is None or sa is None or q < cfg['min_qed_for_docking']
                or sa < cfg['min_sa_for_docking']):
            result['reason'] = 'qed_sa_gate'
            return result
        with tempfile.TemporaryDirectory(prefix='prudent_gate_') as tmp:
            task = VinaDockingTask.from_generated_mol(
                mol, ligand, protein_root=cfg['protein_root'],
                protein_path=protein if protein and Path(protein).is_file() else None,
                tmp_dir=tmp, size_factor=1.0, buffer=5.0)
            values = task.run(mode='score_only', exhaustiveness=8, n_poses=1,
                              timeout_sec=cfg.get('vina_timeout', 20))
        vina = _finite(values[0]['affinity']) if values else None
        result['vina_affinity'] = vina
        result['raw_composite'] = (
            .15*q + .15*sa + .70*float(np.clip(-vina/10, 0, 1))
            if vina is not None else .15*q + .15*sa)
        result['eligible'] = vina is not None and -15 < vina < 0
        if result['eligible']:
            result['composite'] = result['raw_composite']
        else:
            result['reason'] = 'missing_or_out_of_range_vina'
    except TimeoutError as exc:
        result['reason'], result['error'] = 'timeout', str(exc)
    except Exception as exc:
        result['reason'], result['error'] = type(exc).__name__, str(exc)
    finally:
        result['wall_s'] = time.perf_counter() - started
    return result


class TerminalScorer:
    """Separate CPU workers keep Vina alarms active and never touch CUDA."""

    def __init__(self, data, atom_mode, cfg):
        self.data, self.atom_mode, self.cfg = data, atom_mode, cfg
        self.executor = None

    def __enter__(self):
        self.executor = ProcessPoolExecutor(
            max_workers=min(10, int(self.cfg.get('scoring_workers', 10))),
            mp_context=mp.get_context('spawn'), initializer=_worker_init)
        return self

    def score(self, candidates):
        payloads = [(c['pos'], c['v'], self.atom_mode,
                     getattr(self.data, 'protein_filename', None),
                     getattr(self.data, 'ligand_filename', None), self.cfg)
                    for c in candidates]
        # Process-level watchdog in the runner also covers native-code hangs.
        return list(self.executor.map(terminal_score, payloads, chunksize=1))

    def __exit__(self, *args):
        self.executor.shutdown(wait=True)


def select_family(candidates, keep, fallback_keep=1):
    """Scan the entire eligible pool before truncation; ties retain the elite.

    Ineligible candidates have composite=-inf (JSON null). Their diagnostic
    raw score does not compete with qualified candidates. A marked fallback
    is used only when the entire family has no eligible candidate.
    """
    def key(candidate):
        score = candidate['gate']
        return (score['composite'] if score['composite'] is not None
                else float('-inf'), bool(candidate.get('is_elite')))
    eligible = sorted((c for c in candidates if c['gate']['eligible']),
                      key=key, reverse=True)
    selected, seen = [], set()
    for candidate in eligible:
        smiles = candidate['gate']['smiles']
        if smiles not in seen:
            selected.append(candidate)
            seen.add(smiles)
        if len(selected) == keep:
            break
    if not eligible:
        elites = [c for c in candidates if c.get('is_elite')]
        pool = elites or sorted(
            candidates,
            key=lambda c: c['gate'].get('raw_composite') or float('-inf'),
            reverse=True)
        for candidate in pool:
            smiles = candidate['gate'].get('smiles')
            if smiles and smiles in seen:
                continue
            selected.append(candidate)
            if smiles:
                seen.add(smiles)
            if len(selected) == min(keep, fallback_keep):
                break
    return selected


def rewind(candidate, offset, copy_helper=deepcopy):
    """Return this winner's own pre-TBR state after exactly offset updates."""
    snapshot = candidate['checkpoints'][offset]
    count = candidate['large_step_frames'] + offset
    result = copy_helper(candidate)
    for key in ('pos', 'v', 'log_v'):
        result[key] = np.array(snapshot[key], copy=True)
    for key in ('pos_traj', 'v_traj', 'log_v_traj'):
        result[key] = list(candidate[key][:count])
    result['time_indices'] = list(candidate['time_indices'][:count])
    # No gate or TBR from the previous terminal is part of the continued path.
    return result


def _checkpoint(candidate, offset, schedule):
    idx = candidate['large_step_frames'] + offset - 1
    if candidate['time_indices'][idx] != schedule[offset-1]:
        raise ValueError('Checkpoint time index does not belong to its path')
    snapshot = {key: np.array(candidate[trajectory][idx], copy=True)
                for key, trajectory in [('pos', 'pos_traj'), ('v', 'v_traj'),
                                        ('log_v', 'log_v_traj')]}
    snapshot.update(refine_steps=offset, time_index=schedule[offset-1],
                    owner_id=candidate['lineage']['candidate_id'])
    snapshot['state_sha256'] = state_hash(snapshot['pos'], snapshot['v'])
    return snapshot


def run_terminal_rounds(model, data, config, ligand_atom_mode, ls_out,
                        total_candidates, *, helpers, device='cuda:0',
                        logger=None, skip_targetdiff_baseline_refine=False):
    """Search completed paths, rewind winners, and retain a scored incumbent."""
    h = SimpleNamespace(**helpers)
    dynamic = config.sample['dynamic']
    cfg, refine = dynamic['prudent'], dynamic['refine']
    schedule, segments, top_k = split_round_schedule(model, dynamic)
    if (len(schedule) != 24 or cfg['n_sampling'] != 10
            or top_k != [2] + [1]*(len(segments)-1)
            or skip_targetdiff_baseline_refine
            or not config.sample['targetdiff_baseline_refine']['enable']
            or config.sample['targetdiff_baseline_refine']['start_t'] != 9):
        raise ValueError('Terminal Prudent requires 24 updates, 10 candidates and TBR10')
    if any(abs(cfg.get(k+'_weight', 0)-w) > 1e-12
           for k, w in [('qed', .15), ('sa', .15), ('vina', .7),
                        ('lipinski', 0), ('lilly_demerit', 0)]):
        raise ValueError('Unexpected terminal gate weights')
    from diffdynamic.chemistry import reconstruct_targetdiff_legacy
    legacy = Path(reconstruct_targetdiff_legacy.__file__)
    if hashlib.sha256(legacy.read_bytes()).hexdigest() != LEGACY_SHA256:
        raise ValueError('Frozen TargetDiff reconstruction hash changed')
    parents = []
    for root, item in enumerate(total_candidates):
        parent = h._prudent_copy_candidate(item)
        parent.update(lineage=dict(root_seed=root, candidate_id=f's{root}',
                                   parent_id=None),
                      large_step_frames=len(parent.get('time_indices') or []))
        parents.append(parent)
    archives, lineage_rows, round_rows, shortages = {}, [], [], []
    started = time.perf_counter()
    scorer_factory = helpers.get('_terminal_scorer_factory', TerminalScorer)
    with scorer_factory(data, ligand_atom_mode, cfg) as scorer:
        for rnd, keep in enumerate(top_k):
            offset = sum(map(len, segments[:rnd]))
            suffix = schedule[offset:]
            children, families = [], {}
            for family, parent in enumerate(parents):
                families[family] = []
                if rnd:
                    elite = dict(parent, is_elite=True)
                    families[family].append(elite)
                for j in range(10 if rnd == 0 else 9):
                    child = (h._prudent_copy_candidate(parent) if not rnd
                             else rewind(parent, offset, h._prudent_copy_candidate))
                    child.update(
                        large_step_frames=parent['large_step_frames'],
                        is_elite=False,
                        lineage=dict(root_seed=parent['lineage']['root_seed'],
                                     parent_id=parent['lineage']['candidate_id'],
                                     candidate_id=f"{parent['lineage']['candidate_id']}/r{rnd+1}c{j}",
                                     round=rnd+1, family=family, child_index=j))
                    children.append(child)
                    families[family].append(child)
            _sync(device)
            begin = time.perf_counter()
            updates = h._prudent_refine_batch(
                model, data, children, dynamic, refine,
                config.sample.get('center_pos_mode', 'protein'),
                config.sample.get('pos_only', False), device,
                time_upper=suffix[0], time_lower=suffix[-1],
                max_grad_cap=len(suffix), explicit_time_indices=suffix,
                continuation=rnd > 0, logger=logger)
            _sync(device)
            gpu_s = time.perf_counter() - begin
            for child, update in zip(children, updates):
                child.update(update)
                if child['time_indices'][-24:] != schedule:
                    raise ValueError('Final path is not the original 24-step schedule')
                child['checkpoints'] = {
                    k: _checkpoint(child, k, schedule) for k in (8, 12, 16)}
            begin = time.perf_counter()
            tbr = h.apply_targetdiff_baseline_refinement(
                model, data, [c['pos'] for c in children],
                [c['v'] for c in children], config, device=device, logger=logger)
            _sync(device)
            if len(tbr) != 5 or tbr[4] != list(range(9, -1, -1)):
                raise ValueError('Full terminal rollout did not receive TBR10')
            for child, pos, types, pt, vt in zip(children, *tbr[:4]):
                if len(pt) != 10 or len(vt) != 10:
                    raise ValueError('TBR trajectory must contain exactly ten updates')
                child.update(pos=pos, v=types, tbr_steps=10, final_refine_steps=24)
                child['pos_traj'] = list(child['pos_traj']) + list(pt)
                child['v_traj'] = list(child['v_traj']) + list(vt)
            baseline_s = time.perf_counter() - begin
            begin = time.perf_counter()
            scores = scorer.score(children)
            if len(scores) != len(children):
                raise ValueError('Missing terminal gate results')
            for child, score in zip(children, scores):
                if score['state_sha256'] != state_hash(child['pos'], child['v']):
                    raise ValueError('Gate score does not match terminal coordinates')
                child['gate'] = score
                child['score_source_id'] = child['lineage']['candidate_id']
                archives[child['score_source_id']] = {
                    **child['lineage'], 'checkpoints': child['checkpoints'],
                    'terminal_pos': child['pos'], 'terminal_v': child['v'],
                    'gate': score, 'refine_time_indices': schedule,
                    'suffix_start_offset': offset, 'tbr_steps': 10,
                    'parent_snapshot_sha256': (
                        None if not rnd else parents[child['lineage']['family']]
                        ['checkpoints'][offset]['state_sha256']),
                }
            parents_next = []
            for family, pool in families.items():
                selected = select_family(pool, keep, cfg.get('fallback_keep_top_k', 1))
                if rnd and selected:
                    incumbent = pool[0]
                    before = incumbent['gate']['composite']
                    after = selected[0]['gate']['composite']
                    if before is not None and (after is None or after < before):
                        raise ValueError('Incumbent composite score decreased')
                if len(selected) < keep:
                    shortages.append(dict(round=rnd+1, family=family,
                                          requested=keep, actual=len(selected),
                                          reason='no_eligible_candidates' if not any(
                                              c['gate']['eligible'] for c in pool)
                                          else 'insufficient_unique_eligible_smiles'))
                selected_ids = {id(c) for c in selected}
                ranked = sorted(pool, key=lambda c: (
                    c['gate']['composite'] if c['gate']['composite'] is not None
                    else float('-inf'), bool(c.get('is_elite'))), reverse=True)
                for rank, candidate in enumerate(ranked, 1):
                    gate = candidate['gate']
                    lineage_rows.append(dict(
                        dict(candidate['lineage'], round=rnd+1, family=family),
                        occurrence_id=f'r{rnd+1}f{family}p{rank}',
                        score_source_id=candidate['score_source_id'],
                        parent_id=(pool[0]['lineage']['candidate_id'] if rnd
                                   else candidate['lineage']['parent_id']),
                        is_elite=bool(candidate.get('is_elite')),
                        selected=id(candidate) in selected_ids,
                        fallback=id(candidate) in selected_ids and not gate['eligible'],
                        rank_in_family=rank, cumulative_refine_steps=24,
                        rewind_from_offset=offset, scored_after_tbr_steps=10,
                        score_state='post_TBR10_terminal', **gate))
                parents_next.extend(selected)
            parents = parents_next
            scoring_s = time.perf_counter() - begin
            round_rows.append(dict(
                round=rnd+1, rewind_offset=offset, time_indices=suffix,
                n_parents=len(families), n_candidates=sum(map(len, families.values())),
                n_new_candidates=len(children), n_selected=len(parents),
                advance_top_k=keep, gpu_s=gpu_s, baseline_s=baseline_s,
                scoring_s=scoring_s, physical_refine_updates=len(children)*len(suffix),
                physical_tbr_updates=len(children)*10))
            if logger:
                logger.info('[TerminalPrudent] round=%s offset=%s new=%s selected=%s',
                            rnd+1, offset, len(children), len(parents))
    ls_s = float(ls_out['meta']['prudent_large_step_wall_s'])
    total_s = ls_s + time.perf_counter() - started
    records = []
    for candidate in parents:
        records.append(dict(
            candidate, source_index=candidate['lineage']['root_seed'],
            gate_score_state='post_TBR10_terminal', final_state='scored_terminal',
            prudent_composite_detail=candidate['gate']))
    timing = dict(large_step_s=ls_s, gpu_s=sum(r['gpu_s'] for r in round_rows),
                  cpu_s=sum(r['scoring_s'] for r in round_rows),
                  baseline_s=sum(r['baseline_s'] for r in round_rows),
                  total_s=total_s, n_mol=len(records), target_n_mol=100,
                  normalized_100_s=100*total_s/len(records) if records else None,
                  status='ok' if len(records) == 100 else 'shortfall',
                  rounds=round_rows, shortage_reasons=shortages)
    archive_root = Path(os.environ.get('DIFFDYNAMIC_RECORD_DIR', '.'))
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_path = archive_root / 'terminal_candidates.pt'
    torch.save(dict(mode=MODE, candidates=archives, occurrences=lineage_rows), archive_path)
    meta = dict(ls_out['meta'])
    meta.update(refined_candidates=records, timing=timing,
                baseline_refine_time_indices=list(range(9, -1, -1)))
    meta['prudent'] = dict(
        mode=MODE, round_grad_fusion_steps=[len(s) for s in segments],
        round_advance_top_k=top_k, refine_time_indices=schedule,
        max_grad_fusion_iterations=24, n_rounds=len(segments), n_sampling=10,
        weights={k: cfg.get(k+'_weight', 0) for k in
                 ('qed', 'sa', 'vina', 'lipinski', 'lilly_demerit')},
        reconstruction_method='targetdiff_legacy', reconstruction_sha256=LEGACY_SHA256,
        candidate_lineage=lineage_rows, candidate_archive=str(archive_path.resolve()),
        candidate_archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        final_tbr_calls=1, final_tbr_after_selection_calls=0,
        shortages=shortages, dedup_scope='entire eligible pool within parent family',
        gate_filter='single fragment; QED/SA>=0.2; finite -15<Vina<0',
        elite_policy='unchanged scored incumbent plus 9 new; ties retain incumbent')
    return dict(pos_list=[c['pos'] for c in records],
                v_list=[c['v'] for c in records],
                pos_traj=[c['pos_traj'] for c in records],
                v_traj=[c['v_traj'] for c in records],
                log_v_traj=[c['log_v_traj'] for c in records],
                time_list=[ls_s]+[r['gpu_s'] for r in round_rows], meta=meta)
