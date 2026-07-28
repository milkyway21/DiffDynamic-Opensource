"""Unit tests for Murcko side-chain site extraction and atom allocation."""

import os
import sys

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.scaffold_sites import (
    DEFAULT_FRAGMENT_SIZES,
    allocate_atoms_random_per_site,
    allocate_atoms_sequential_random,
    allocate_atoms_to_sites,
    build_extra_atom_positions,
    cap_n_extra_for_sites,
    compute_site_capacity,
    extract_exit_vector_sites,
    extract_murcko_attachment_sites,
    init_extra_positions_at_sites,
    merge_attachment_and_exit_vector_sites,
    sample_fragment_atom_count,
    uses_site_budget_placement,
)


def _mol_with_conformer(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol, addCoords=True)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def test_extract_toluene_methyl_site():
    """甲苯：Murcko 骨架为苯环，甲基侧链应产生 1 个质心位点。"""
    mol = _mol_with_conformer('Cc1ccccc1')
    from scripts.sample_diffusion import _detect_murcko_scaffold_indices
    scaffold_indices = _detect_murcko_scaffold_indices(mol)
    assert len(scaffold_indices) >= 6

    sites = extract_murcko_attachment_sites(mol, scaffold_indices)
    assert len(sites) >= 1
    centroid = np.array(sites[0]['centroid_pos'])
    assert centroid.shape == (3,)
    assert sites[0]['removed_atom_count'] >= 1


def test_allocate_sum_equals_n_extra():
    sites = [
        {'site_id': 0, 'centroid_pos': [0, 0, 0]},
        {'site_id': 1, 'centroid_pos': [3, 0, 0]},
        {'site_id': 2, 'centroid_pos': [0, 3, 0]},
    ]
    rng = np.random.default_rng(0)
    for n_extra in [1, 5, 12, 30, 60]:
        active, counts, overflow = allocate_atoms_to_sites(
            n_extra, sites, p_active=1.0, max_per_site=20, rng=rng,
        )
        assert sum(counts) + overflow == n_extra
        for c in counts:
            assert 0 <= c <= 20


def test_build_extra_positions_count():
    sites = [{'site_id': 0, 'centroid_pos': [1.0, 2.0, 3.0],
              'anchor_pos': [0.0, 2.0, 3.0]}]
    cfg = {'p_active': 1.0, 'max_per_site': 20, 'jitter_std': 0.1}
    center = torch.zeros(3)
    pos, meta = build_extra_atom_positions(5, sites, cfg, center, 'cpu')
    assert pos.shape == (5, 3)
    assert meta['placement'] == 'murcko_sites'


def test_overflow_mode_cap():
    sites = [
        {'site_id': 0, 'centroid_pos': [0, 0, 0], 'anchor_pos': [-1, 0, 0]},
        {'site_id': 1, 'centroid_pos': [3, 0, 0], 'anchor_pos': [2, 0, 0]},
    ]
    cfg = {
        'p_active': 1.0,
        'max_per_site': 2,
        'overflow_mode': 'cap',
        'jitter_mode': 'directional',
    }
    center = torch.zeros(3)
    n_req = 10
    capped = cap_n_extra_for_sites(n_req, sites, cfg)
    assert capped == 4  # 2 sites * 2 per site

    pos, meta = build_extra_atom_positions(n_req, sites, cfg, center, 'cpu')
    assert pos.shape[0] == 4
    assert meta['n_extra_effective'] == 4
    assert meta['overflow_count'] == 0
    assert meta.get('overflow_dropped', 0) == 0


def test_directional_jitter_spreads_along_anchor():
    sites = [{
        'site_id': 0,
        'anchor_pos': [0.0, 0.0, 0.0],
        'centroid_pos': [2.0, 0.0, 0.0],
    }]
    rng = np.random.default_rng(0)
    pos = init_extra_positions_at_sites(
        sites, [2], 'cpu', jitter_std=0.1, jitter_mode='directional', rng=rng,
    )
    assert pos.shape == (2, 3)
    p0 = pos[0].numpy()
    p1 = pos[1].numpy()
    # 两点应沿 +x 方向外推，第二点更远
    assert p0[0] > 0.5
    assert p1[0] > p0[0]


def test_gaussian_jitter_clusters_at_centroid():
    """无径向外推时：额外原子应聚在去除位点质心附近。"""
    sites = [{
        'site_id': 0,
        'site_kind': 'murcko_sidechain',
        'anchor_pos': [0.0, 0.0, 0.0],
        'centroid_pos': [5.0, 0.0, 0.0],
    }]
    rng = np.random.default_rng(0)
    pos = init_extra_positions_at_sites(
        sites, [8], 'cpu', jitter_std=1.0, jitter_mode='gaussian', rng=rng,
        sites_cfg={'radial_step': 0.0, 'radial_bulk': 0.0, 'radial_bulk_start': 4},
    )
    assert pos.shape == (8, 3)
    arr = pos.numpy()
    mean = arr.mean(axis=0)
    assert abs(mean[0] - 5.0) < 1.0
    # 团内跨度应远小于 directional 的 ~1Å*idx 拉开
    span = arr.max(axis=0) - arr.min(axis=0)
    assert float(span.max()) < 6.0


def test_gaussian_radial_spreads_with_more_atoms():
    """多原子时平均 ||pos-anchor|| 应明显大于少原子。"""
    sites = [{
        'site_id': 0,
        'site_kind': 'murcko_sidechain',
        'anchor_pos': [0.0, 0.0, 0.0],
        'centroid_pos': [2.0, 0.0, 0.0],
    }]
    cfg = {
        'radial_step': 0.25,
        'radial_bulk': 0.35,
        'radial_bulk_start': 4,
    }
    rng_small = np.random.default_rng(0)
    rng_large = np.random.default_rng(0)
    pos2 = init_extra_positions_at_sites(
        sites, [2], 'cpu', jitter_std=0.05, jitter_mode='gaussian',
        rng=rng_small, sites_cfg=cfg,
    ).numpy()
    pos12 = init_extra_positions_at_sites(
        sites, [12], 'cpu', jitter_std=0.05, jitter_mode='gaussian',
        rng=rng_large, sites_cfg=cfg,
    ).numpy()
    d2 = float(np.linalg.norm(pos2, axis=1).mean())
    d12 = float(np.linalg.norm(pos12, axis=1).mean())
    assert d12 > d2 + 1.0, (d2, d12)
    # 大团应整体更靠 +x（远离锚点）
    assert float(pos12[:, 0].mean()) > float(pos2[:, 0].mean()) + 0.5


def test_concentrate_to_one_removed_site():
    from utils.scaffold_sites import concentrate_site_allocation, build_extra_atom_positions
    sites = [
        {'site_id': 0, 'site_kind': 'murcko_sidechain',
         'centroid_pos': [0, 0, 0], 'anchor_pos': [-1, 0, 0]},
        {'site_id': 1, 'site_kind': 'exit_vector',
         'centroid_pos': [10, 0, 0], 'anchor_pos': [9, 0, 0]},
        {'site_id': 2, 'site_kind': 'exit_vector',
         'centroid_pos': [0, 10, 0], 'anchor_pos': [0, 9, 0]},
    ]
    active = sites
    counts = [3, 4, 5]
    rng = np.random.default_rng(1)
    a2, c2 = concentrate_site_allocation(
        active, counts, max_active_sites=1, rng=rng, prefer_murcko=True,
    )
    assert len(a2) == 1
    assert c2 == [12]
    assert a2[0]['site_kind'] == 'murcko_sidechain'

    cfg = {
        'p_active': 1.0,
        'max_per_site': 20,
        'per_site_count_mode': 'split',
        'jitter_mode': 'gaussian',
        'jitter_std': 1.0,
        'max_active_sites': 1,
        'prefer_murcko_sites': True,
        'overflow_mode': 'cap',
    }
    pos, meta = build_extra_atom_positions(
        12, sites, cfg, torch.zeros(3), 'cpu', rng=np.random.default_rng(0),
    )
    assert pos.shape[0] == 12
    assert meta['n_sites_after_concentrate'] == 1
    assert len(meta['site_allocation']) == 1
    # 全部落在 murcko 去除位点附近
    assert meta['site_allocation'][0]['site_id'] == 0
    arr = pos.numpy()
    # 径向外推后团心沿 +x 略移，但仍应在 murcko 位点附近，远离 exit_vector(x=10)
    assert abs(arr.mean(0)[0]) < 7.0
    assert abs(arr.mean(0)[0] - 10.0) > 3.0
    assert abs(arr.mean(0)[1] - 10.0) > 3.0


def test_prudent_size_gate_helpers():
    from scripts.sample_diffusion import _prudent_parse_size_gates, _prudent_size_gate_ok

    gates = _prudent_parse_size_gates({
        'max_logp': 4.5,
        'max_molwt': 550,
        'max_rings': 7,
    })
    assert gates['max_logp'] == 4.5
    assert gates['max_molwt'] == 550

    mol = _mol_with_conformer('Cc1ccccc1')
    qed, sa, logp = 0.5, 0.5, 3.0
    ok, reasons = _prudent_size_gate_ok(mol, qed, sa, logp, {'min_qed_for_docking': 0.2, 'min_sa_for_docking': 0.2}, gates=gates)
    assert ok, reasons

    rings = Lipinski.RingCount(mol)
    ok2, reasons2 = _prudent_size_gate_ok(mol, qed, sa, logp, {'min_qed_for_docking': 0.2, 'min_sa_for_docking': 0.2}, gates={'max_rings': max(0, rings - 1)})
    assert not ok2
    assert any('Rings' in r for r in reasons2)


def test_random_per_site_diversity():
    """每位点独立随机计数应产生变化的 n_extra。"""
    sites = [
        {'site_id': 0, 'centroid_pos': [0, 0, 0], 'anchor_pos': [-1, 0, 0]},
    ]
    cfg = {
        'p_active': 1.0,
        'max_per_site': 2,
        'min_per_site': 0,
        'per_site_count_mode': 'random_per_site',
        'overflow_mode': 'cap',
        'n_extra_min_clamp': 0,
        'n_extra_max_clamp': 2,
    }
    totals = set()
    for seed in range(50):
        rng = np.random.default_rng(seed)
        _, counts, n_eff = allocate_atoms_random_per_site(
            sites, cfg, n_extra_bounds=(0, 2), rng=rng,
        )
        totals.add(n_eff)
        assert sum(counts) == n_eff
        for c in counts:
            assert 0 <= c <= 2
    assert len(totals) >= 2, f'expected varied n_extra, got {totals}'


def test_build_extra_random_per_site():
    sites = [{'site_id': 0, 'centroid_pos': [1.0, 2.0, 3.0],
              'anchor_pos': [0.0, 2.0, 3.0]}]
    cfg = {
        'p_active': 1.0,
        'max_per_site': 2,
        'min_per_site': 0,
        'per_site_count_mode': 'random_per_site',
        'overflow_mode': 'cap',
        'n_extra_min_clamp': 0,
        'n_extra_max_clamp': 2,
        'jitter_std': 0.1,
    }
    center = torch.zeros(3)
    sizes = set()
    for seed in range(30):
        pos, meta = build_extra_atom_positions(
            0, sites, cfg, center, 'cpu', rng=np.random.default_rng(seed),
        )
        sizes.add(pos.shape[0])
        assert meta['per_site_count_mode'] == 'random_per_site'
    assert len(sizes) >= 2


def test_sequential_random_ordered_p_active():
    """按序 1/2 概率、片段先验、上限 20/位点，总数在 budget 内。"""
    sites = [
        {'site_id': 0, 'centroid_pos': [0, 0, 0], 'anchor_pos': [-1, 0, 0]},
        {'site_id': 1, 'centroid_pos': [3, 0, 0], 'anchor_pos': [2, 0, 0]},
        {'site_id': 2, 'centroid_pos': [0, 3, 0], 'anchor_pos': [0, 2, 0]},
    ]
    cfg = {
        'p_active': 0.5,
        'max_per_site': 20,
        'min_per_site': 0,
        'per_site_count_mode': 'sequential_random',
        'per_site_add_mode': 'fragment_prior',
    }
    totals = set()
    allocs = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        active, counts, n_eff = allocate_atoms_sequential_random(
            sites, cfg, n_extra_bounds=(3, 12), rng=rng,
        )
        totals.add(n_eff)
        allocs.append((counts, n_eff))
        assert 3 <= n_eff <= 12
        for c in counts:
            assert 0 <= c <= 20
        assert sum(counts) == n_eff
    assert len(totals) >= 3, f'expected varied totals, got {totals}'


def test_build_extra_sequential_random():
    sites = [
        {'site_id': 0, 'centroid_pos': [1.0, 2.0, 3.0], 'anchor_pos': [0.0, 2.0, 3.0]},
        {'site_id': 1, 'centroid_pos': [4.0, 2.0, 3.0], 'anchor_pos': [3.0, 2.0, 3.0]},
    ]
    cfg = {
        'p_active': 0.5,
        'max_per_site': 20,
        'min_per_site': 0,
        'per_site_count_mode': 'sequential_random',
        'per_site_add_mode': 'fragment_prior',
        'overflow_mode': 'cap',
        'n_extra_min_clamp': 1,
        'n_extra_max_clamp': 10,
        'jitter_std': 0.1,
    }
    sizes = set()
    for seed in range(30):
        pos, meta = build_extra_atom_positions(
            0, sites, cfg, torch.zeros(3), 'cpu', rng=np.random.default_rng(seed),
        )
        sizes.add(pos.shape[0])
        assert meta['per_site_count_mode'] == 'sequential_random'
        assert 1 <= pos.shape[0] <= 10
    assert len(sizes) >= 3


def test_p_active_statistics():
    sites = [{'site_id': i, 'centroid_pos': [float(i), 0, 0]} for i in range(10)]
    rng = np.random.default_rng(123)
    active_counts = []
    for _ in range(500):
        active, _, _ = allocate_atoms_to_sites(10, sites, p_active=0.5, max_per_site=20, rng=rng)
        active_counts.append(len(active))
    mean_active = np.mean(active_counts)
    # p=0.5 → E≈5；至少保证 1 个激活
    assert 3.5 < mean_active < 6.5


def test_fragment_prior_complex_sizes():
    """片段先验：≥6 占主导；复杂档高于小基团；11/12 可观测；不超过 room。"""
    cfg = {
        'max_per_site': 20,
        'min_per_site': 0,
        'per_site_add_mode': 'fragment_prior',
    }
    rng = np.random.default_rng(42)
    samples = [sample_fragment_atom_count(rng, cfg, room=20) for _ in range(3000)]
    assert all(1 <= s <= 20 for s in samples)
    assert set(samples).issubset(set(DEFAULT_FRAGMENT_SIZES))
    small = sum(1 for s in samples if s in (1, 2, 3))
    complex_ = sum(1 for s in samples if s in (6, 10, 11, 12))
    assert complex_ > small, f'complex={complex_} should exceed small={small}'
    p_ge6 = sum(1 for s in samples if s >= 6) / len(samples)
    assert p_ge6 > 0.8, f'P(>=6)={p_ge6:.3f} should exceed 0.8'
    assert any(s == 11 for s in samples)
    assert any(s == 12 for s in samples)
    # room 过滤：room=5 时不应出现 6+
    samples_small_room = [
        sample_fragment_atom_count(rng, cfg, room=5) for _ in range(200)
    ]
    assert all(1 <= s <= 5 for s in samples_small_room)


def test_sequential_uniform_regression():
    """per_site_add_mode=uniform 仍可用。"""
    sites = [
        {'site_id': 0, 'centroid_pos': [0, 0, 0], 'anchor_pos': [-1, 0, 0]},
        {'site_id': 1, 'centroid_pos': [3, 0, 0], 'anchor_pos': [2, 0, 0]},
    ]
    cfg = {
        'p_active': 1.0,
        'max_per_site': 20,
        'min_per_site': 0,
        'per_site_add_mode': 'uniform',
        'per_site_count_mode': 'sequential_random',
    }
    for seed in range(20):
        _, counts, n_eff = allocate_atoms_sequential_random(
            sites, cfg, n_extra_bounds=(5, 15), rng=np.random.default_rng(seed),
        )
        assert 5 <= n_eff <= 15
        assert sum(counts) == n_eff
        assert all(0 <= c <= 20 for c in counts)


def test_exit_vector_sites_on_x77():
    """X77 Murcko 仅 1 个真实位点；开启 aromatic_h exit-vector 后应明显增多。"""
    x77_path = os.environ.get(
        'DD_TEST_X77_SDF',
        os.path.join(os.path.dirname(__file__), '..', 'data', 'examples', '6W63_ligand_X77_pose.sdf'),
    )
    if not os.path.exists(x77_path):
        return
    mol = Chem.RemoveHs(Chem.SDMolSupplier(x77_path, removeHs=True)[0])
    from scripts.sample_diffusion import _detect_murcko_scaffold_indices
    scaffold_indices = _detect_murcko_scaffold_indices(mol)
    murcko = extract_murcko_attachment_sites(mol, scaffold_indices)
    assert len(murcko) == 1, f'expected 1 murcko site, got {len(murcko)}'
    exit_sites = extract_exit_vector_sites(
        mol, scaffold_indices, mode='aromatic_h', offset=1.5,
        existing_sites=murcko, dedup_dist=1.2,
    )
    assert len(exit_sites) >= 5, f'expected >=5 exit vectors, got {len(exit_sites)}'
    merged = merge_attachment_and_exit_vector_sites(murcko, exit_sites)
    assert len(merged) == len(murcko) + len(exit_sites)
    assert sum(1 for s in merged if s.get('site_kind') == 'exit_vector') == len(exit_sites)
    assert all(s['site_id'] == i for i, s in enumerate(merged))


def test_resolve_max_active_sites_equals_removed_sidechains():
    from utils.scaffold_sites import (
        resolve_max_active_sites, count_removed_sidechain_sites,
    )
    sites = [
        {'site_id': 0, 'site_kind': 'murcko_sidechain', 'removed_atom_indices': [5, 6]},
        {'site_id': 1, 'site_kind': 'murcko_sidechain', 'removed_atom_indices': [7]},
        {'site_id': 2, 'site_kind': 'exit_vector', 'removed_atom_indices': []},
    ]
    assert count_removed_sidechain_sites(sites) == 2
    assert resolve_max_active_sites({'max_active_sites': 'n_removed_sidechains'}, sites) == 2
    assert resolve_max_active_sites({}, sites) == 2
    assert resolve_max_active_sites({'max_active_sites': 1}, sites) == 1


def test_preserve_zero_allocation_sidechains_meta():
    """concentrate 到 1 个位点后，其余位点的 removed 原子应进入 preserved meta。"""
    sites = [
        {
            'site_id': 0, 'site_kind': 'murcko_sidechain',
            'centroid_pos': [0.0, 0.0, 0.0], 'anchor_pos': [-1.0, 0.0, 0.0],
            'removed_atom_indices': [10, 11],
        },
        {
            'site_id': 1, 'site_kind': 'murcko_sidechain',
            'centroid_pos': [5.0, 0.0, 0.0], 'anchor_pos': [4.0, 0.0, 0.0],
            'removed_atom_indices': [20, 21, 22],
        },
    ]
    cfg = {
        'p_active': 1.0,
        'max_per_site': 20,
        'per_site_count_mode': 'split',
        'jitter_mode': 'gaussian',
        'jitter_std': 0.1,
        'max_active_sites': 1,
        'prefer_murcko_sites': True,
        'overflow_mode': 'cap',
        'preserve_zero_allocation_sidechains': True,
    }
    pos, meta = build_extra_atom_positions(
        6, sites, cfg, torch.zeros(3), 'cpu', rng=np.random.default_rng(0),
    )
    assert pos.shape[0] == 6
    assert meta['n_sites_after_concentrate'] == 1
    active_id = meta['site_allocation'][0]['site_id']
    assert active_id in (0, 1)
    assert active_id not in meta['zero_allocation_site_ids']
    assert len(meta['zero_allocation_site_ids']) == 1
    assert meta['n_preserved_atoms'] == 3 if active_id == 0 else 2

    ref_pos = torch.zeros(30, 3)
    ref_pos[10] = torch.tensor([1.0, 0.0, 0.0])
    ref_pos[11] = torch.tensor([1.5, 0.0, 0.0])
    ref_pos[20] = torch.tensor([2.0, 0.0, 0.0])
    ref_pos[21] = torch.tensor([2.5, 0.0, 0.0])
    ref_pos[22] = torch.tensor([3.0, 0.0, 0.0])
    ref_log_v = torch.zeros(30, 4)
    ref_log_v[:, 0] = 1.0
    from utils.scaffold_sites import resolve_preserved_sidechains
    ppos, plog, pidx, zids = resolve_preserved_sidechains(
        sites, meta, cfg, ref_pos, ref_log_v,
    )
    assert len(pidx) == meta['n_preserved_atoms']
    assert ppos.shape[0] == len(pidx)
    # 坐标与参考一致
    for i, idx in enumerate(pidx):
        assert torch.allclose(ppos[i], ref_pos[idx])

    # 关闭开关 → 不保留
    cfg_off = dict(cfg)
    cfg_off['preserve_zero_allocation_sidechains'] = False
    ppos2, _, pidx2, _ = resolve_preserved_sidechains(
        sites, meta, cfg_off, ref_pos, ref_log_v,
    )
    assert ppos2.shape[0] == 0 and pidx2 == []


if __name__ == '__main__':
    test_extract_toluene_methyl_site()
    test_allocate_sum_equals_n_extra()
    test_build_extra_positions_count()
    test_overflow_mode_cap()
    test_directional_jitter_spreads_along_anchor()
    test_gaussian_jitter_clusters_at_centroid()
    test_concentrate_to_one_removed_site()
    test_random_per_site_diversity()
    test_build_extra_random_per_site()
    test_sequential_random_ordered_p_active()
    test_build_extra_sequential_random()
    test_prudent_size_gate_helpers()
    test_p_active_statistics()
    test_fragment_prior_complex_sizes()
    test_sequential_uniform_regression()
    test_exit_vector_sites_on_x77()
    test_resolve_max_active_sites_equals_removed_sidechains()
    test_preserve_zero_allocation_sidechains_meta()
    print('All scaffold_sites tests passed.')
