"""Tests for full-rollout gates, own-path snapshots and cached incumbents."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from models.molopt_score_model import ScorePosNet3D


class ScheduleModel:
    num_timesteps = 1000
    _build_lambda_schedule = ScorePosNet3D._build_lambda_schedule
from utils.prudent_terminal_rollout import (
    run_terminal_rounds, select_family, state_hash,
)


def scored(smiles, score, elite=False, eligible=True):
    return dict(is_elite=elite, gate=dict(
        smiles=smiles, eligible=eligible,
        composite=score if eligible else None, raw_composite=score))


def test_dedup_scans_entire_pool_and_ties_keep_elite():
    pool = [scored('CC', .9), scored('CC', .8), scored('CCC', .7)]
    assert [c['gate']['smiles'] for c in select_family(pool, 2)] == ['CC', 'CCC']
    elite = scored('CC', .8, elite=True)
    assert select_family([scored('CCC', .8), elite], 1)[0] is elite
    assert select_family([scored('CCC', .7), elite], 1)[0] is elite


def test_all_failed_fallback_is_marked_and_partial_pool_is_not_padded():
    elite = scored('CC', .1, elite=True, eligible=False)
    assert select_family([scored('CCC', .9, eligible=False), elite], 1) == [elite]
    assert len(select_family([scored('CC', .6), scored('CC', .5)], 2)) == 1
    assert len(select_family([scored(None, None, eligible=False)]*10, 2)) == 1


@pytest.mark.parametrize('improving', [False, True])
@pytest.mark.parametrize('name,expected', [
    ('prudent_24', [20]), ('prudent_12_12', [20, 36]),
    ('prudent_8_8_8', [20, 36, 36])])
def test_full_rollout_snapshots_and_elite_immutable(tmp_path, monkeypatch, name, expected, improving):
    monkeypatch.setenv('DIFFDYNAMIC_RECORD_DIR', str(tmp_path))
    config_path = Path(__file__).parent / 'fixtures/configs/prudent_terminal_rollout'
    cfg = yaml.safe_load((config_path / (name + '.yml')).read_text())
    events, original = [], {}

    def refine(model, data, candidates, *args, **kw):
        indices = kw['explicit_time_indices']
        events.append(('refine', len(candidates), len(indices), kw['continuation']))
        result = []
        for i, item in enumerate(candidates):
            c = deepcopy(item)
            if kw['continuation']:
                start = 24-len(indices)
                assert len(c['pos_traj']) == start
                assert c['pos'][0, 0] == start
                # Full category vector, not argmax reinitialization.
                np.testing.assert_array_equal(c['log_v'], c['log_v_traj'][-1])
            for t in indices:
                step = len(c['time_indices'])+1
                c['pos'] = np.full((2, 3), step, dtype=float)
                c['pos'][:, 1] = c['lineage']['round']*100 + c['lineage']['child_index']
                c['v'] = np.array([0, 1])
                c['log_v'] = np.log(np.array([[.2, .3, .5], [.6, .1, .3]]))
                for key, traj in [('pos', 'pos_traj'), ('v', 'v_traj'),
                                  ('log_v', 'log_v_traj')]:
                    c[traj].append(c[key].copy())
                c['time_indices'].append(t)
            result.append(c)
        return result

    def tbr(model, data, positions, types, *args, **kw):
        assert all(p[0, 0] == 24 for p in positions)
        events.append(('tbr', len(positions)))
        positions = [p+100 for p in positions]
        return (positions, types, [[p.copy()]*10 for p in positions],
                [[v.copy()]*10 for v in types], list(range(9, -1, -1)))

    class Scorer:
        def __init__(self, *args):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def score(self, candidates):
            events.append(('score', len(candidates)))
            result = []
            for i, c in enumerate(candidates):
                assert c['pos'][0, 0] == 124 and c['tbr_steps'] == 10
                cid = c['lineage']['candidate_id']
                score = (.9-i/1000 if c['lineage']['round'] == 1 else
                         .9+c['lineage']['round']/100-i/10000 if improving else .1)
                g = dict(smiles=cid, reconstructed=True, complete=True,
                         eligible=True, composite=score, raw_composite=score,
                         state_sha256=state_hash(c['pos'], c['v']),
                         qed=.5, sa=.5, vina_affinity=-7)
                original[cid] = deepcopy(c)
                result.append(g)
            return result

    roots = [dict(pos=np.zeros((2, 3)), v=np.array([0, 1]), num_atoms=2,
                  log_v=np.zeros((2, 3)), time_indices=[], pos_traj=[],
                  v_traj=[], log_v_traj=[]) for _ in range(2)]
    helpers = dict(_prudent_copy_candidate=deepcopy, _prudent_refine_batch=refine,
                   apply_targetdiff_baseline_refinement=tbr,
                   _terminal_scorer_factory=Scorer)
    output = run_terminal_rounds(
        ScheduleModel(), SimpleNamespace(), SimpleNamespace(sample=cfg['sample']),
        'add_aromatic', dict(meta=dict(prudent_large_step_wall_s=1)), roots,
        helpers=helpers, device='cpu')
    assert [e[1] for e in events if e[0] == 'refine'] == expected
    assert [e[0] for e in events] == ['refine', 'tbr', 'score']*len(expected)
    assert len(output['pos_list']) == 4
    for c in output['meta']['refined_candidates']:
        assert c['final_refine_steps'] == 24 and c['tbr_steps'] == 10
        assert len(c['pos_traj']) == 34 and len(c['time_indices']) == 24
        assert c['gate']['composite'] >= .88
        assert c['lineage']['round'] == (len(expected) if improving else 1)
        if improving:
            cp = c['checkpoints'][16]
            assert cp['pos'][0, 1] == c['pos_traj'][15][0, 1]
        for offset in (8, 12, 16):
            assert c['checkpoints'][offset]['pos'][0, 0] == offset
        assert c['gate']['state_sha256'] == state_hash(c['pos'], c['v'])
    assert output['meta']['prudent']['final_tbr_after_selection_calls'] == 0


@pytest.mark.parametrize('vina,eligible', [(-15, False), (0, False),
                                          (np.nan, False), (-7, True)])
def test_terminal_gate_uses_offline_reconstruction_and_strict_bounds(monkeypatch, vina, eligible):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import evaluate_pt_with_correct_reconstruct as evaluation
    from utils.evaluation import scoring_func
    from utils.evaluation.docking_vina import VinaDockingTask
    from utils.prudent_terminal_rollout import terminal_score

    mol = Chem.AddHs(Chem.MolFromSmiles('CCCC'))
    AllChem.EmbedMolecule(mol, randomSeed=2021)
    mol = Chem.RemoveHs(mol)
    calls = []
    def reconstruct(*args, **kwargs):
        calls.append(kwargs['reconstruction_method'])
        return mol, None
    class Task:
        def run(self, **kw):
            assert kw['mode'] == 'score_only' and kw['timeout_sec'] == 20
            return [dict(affinity=vina)]
    monkeypatch.setattr(evaluation, 'reconstruct_molecule', reconstruct)
    monkeypatch.setattr(scoring_func, 'get_chem', lambda m: dict(qed=.5, sa=.6))
    monkeypatch.setattr(VinaDockingTask, 'from_generated_mol', lambda *a, **kw: Task())
    cfg = dict(min_qed_for_docking=.2, min_sa_for_docking=.2,
               protein_root='unused', vina_timeout=20)
    out = terminal_score((np.zeros((4, 3)), np.zeros(4, dtype=int),
                          'add_aromatic', None, 'unused', cfg))
    assert calls == ['targetdiff_legacy'] and out['eligible'] == eligible
    assert out['complete'] and out['smiles'] == 'CCCC'
    assert (out['composite'] is not None) == eligible
