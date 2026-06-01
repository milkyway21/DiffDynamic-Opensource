#!/usr/bin/env python3
"""
Prudent Cumulative 精简实现
只保留跨代累加模式：frame0 -> frame2 -> frame4 -> frame6 -> frame8 -> frame24
每代独立保存断点(_frame2)和完整文件(_full)，支持从断点恢复继续运行。
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

# 添加项目路径
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D
from utils.evaluation import scoring_func


def load_model(config, device):
    """加载预训练模型"""
    ckpt = torch.load(config.model.checkpoint, map_location=device)
    
    model = ScorePosNet3D(
        ckpt['config'].model,
        num_classes=10,
        num_known_ligand_atom_types=10,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    return model, ckpt


def load_data(args, ckpt, transform):
    """加载数据"""
    if args.data_id is not None:
        dataset, subsets = get_dataset(config=ckpt['config'].data, transform=transform)
        train_set, test_set = subsets['train'], subsets['test']
        data = test_set[args.data_id]
        pocket_id = str(args.data_id)
    else:
        from utils.data import PDBProtein, parse_sdf_file
        from datasets.pl_data import ProteinLigandData
        
        protein_path = Path(args.protein_path).expanduser().resolve()
        data = PDBProtein(str(protein_path)).to_dict()
        
        if args.ligand_path:
            lig_dict = parse_sdf_file(args.ligand_path)
            data.update(lig_dict)
        
        data = ProteinLigandData(**data)
        data = transform(data)
        pocket_id = 'custom'
    
    return data, pocket_id


def run_large_step(model, data, config, device):
    """运行 large_step 生成初始候选"""
    from scripts.sample_diffusion import sample_dynamic_diffusion_ligand
    
    # 临时设置只跑 large_step
    config.sample.dynamic['skip_refine'] = True
    
    output = sample_dynamic_diffusion_ligand(
        model=model,
        data=data,
        config=config,
        ligand_atom_mode='add_aromatic',
        device=device,
        logger=None,
        skip_targetdiff_baseline_refine=True,
    )
    
    candidates = output['meta'].get('large_step_candidates', [])
    return candidates


def copy_candidates_n_times(candidates: List[Dict], n: int) -> List[Dict]:
    """复制候选 n 次，用于并行链"""
    pool = []
    for c in candidates:
        for _ in range(n):
            pool.append({
                'pos': c['pos'].copy() if hasattr(c['pos'], 'copy') else c['pos'],
                'v': c['v'].copy() if hasattr(c['v'], 'copy') else c['v'],
                'source_index': c.get('source_index', 0),
            })
    return pool


def refine_pool_to_frame(
    model,
    data,
    pool: List[Dict],
    start_frame: int,
    target_frame: int,
    config,
    device: str,
) -> List[Dict]:
    """
    将 pool 从 start_frame refine 到 target_frame
    简化实现：直接返回（实际需要调用 model 的 refine）
    """
    steps = target_frame - start_frame
    print(f"    [refine] frame {start_frame} -> {target_frame} ({steps} steps)")
    
    # TODO: 实现实际的 refine 逻辑
    # 这里简化处理，直接返回 pool
    # 实际应该调用 _prudent_refine_batch 或类似函数
    
    return pool


def score_pool_and_select_winner(
    pool: List[Dict],
    data,
    protein_root: Path,
    vina_weight: float = 0.86,
    qed_weight: float = 0.07,
    sa_weight: float = 0.07,
) -> Tuple[Dict, List[Dict]]:
    """
    对 pool 打分并选出 winner
    返回: (winner, scored_pool)
    """
    # 简化实现：计算 composite 分数
    scored = []
    for i, c in enumerate(pool):
        # 这里简化处理，实际应该调用 Vina/QED/SA 评估
        composite = np.random.random()  # 占位
        scored.append({
            **c,
            'composite': composite,
            'idx': i,
        })
    
    # 按 composite 排序
    scored.sort(key=lambda x: x['composite'], reverse=True)
    winner = scored[0]
    
    return winner, scored


def save_checkpoint(
    pool: List[Dict],
    data,
    gen_idx: int,
    frame: int,
    output_dir: Path,
    is_full: bool = False,
) -> Path:
    """保存断点或完整文件"""
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    
    if is_full:
        filename = f'gen{gen_idx}_frame{frame}_full_{timestamp}.pt'
    else:
        filename = f'gen{gen_idx}_frame{frame}_checkpoint_{timestamp}.pt'
    
    filepath = output_dir / filename
    
    blob = {
        'checkpoint_kind': 'prudent_cumulative_full' if is_full else 'prudent_cumulative_checkpoint',
        'generation': gen_idx,
        'frame': frame,
        'data': data,
        'pred_ligand_pos': [c['pos'] for c in pool],
        'pred_ligand_v': [c['v'] for c in pool],
        'meta': {
            'refined_candidates': pool,
        },
    }
    
    torch.save(blob, filepath)
    print(f"    💾 保存: {filename}")
    
    return filepath


def load_checkpoint(ck_file: Path) -> Tuple[List[Dict], Dict]:
    """从断点文件加载候选和数据"""
    blob = torch.load(str(ck_file), map_location='cpu')
    
    pool = []
    for pos, v in zip(blob['pred_ligand_pos'], blob['pred_ligand_v']):
        pool.append({
            'pos': pos,
            'v': v,
        })
    
    data = blob.get('data')
    frame = blob.get('frame', 0)
    
    return pool, data, frame


def run_single_generation(
    gen_idx: int,
    start_frame: int,
    pool: List[Dict],
    data,
    model,
    config,
    device: str,
    protein_root: Path,
    output_dir: Path,
) -> Tuple[bool, Optional[Path], Optional[Path]]:
    """
    运行单代：
    1. refine 到 frame2（保存断点）
    2. 继续 refine 到 frame24（保存完整）
    3. 评估并选 winner
    
    Returns: (success, checkpoint_path, full_path)
    """
    print(f"\n{'='*60}")
    print(f"第 {gen_idx} 代 | 从 frame {start_frame} 开始")
    print(f"{'='*60}")
    
    gen_dir = output_dir / f'gen{gen_idx}'
    gen_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. refine 到 checkpoint_frame (2)
    checkpoint_frame = 2
    pool_ck = refine_pool_to_frame(
        model, data, pool, start_frame, checkpoint_frame, config, device
    )
    ck_path = save_checkpoint(pool_ck, data, gen_idx, checkpoint_frame, gen_dir, is_full=False)
    
    # 2. 继续 refine 到 total_frames (24)
    total_frames = config.sample.dynamic.prudent.get('total_frames', 24)
    pool_full = refine_pool_to_frame(
        model, data, pool_ck, checkpoint_frame, total_frames, config, device
    )
    full_path = save_checkpoint(pool_full, data, gen_idx, total_frames, gen_dir, is_full=True)
    
    # 3. 评估并选 winner
    print(f"  [评估] 第 {gen_idx} 代...")
    winner, scored = score_pool_and_select_winner(
        pool_full, data, protein_root,
        vina_weight=config.sample.dynamic.prudent.get('vina_weight', 0.86),
        qed_weight=config.sample.dynamic.prudent.get('qed_weight', 0.07),
        sa_weight=config.sample.dynamic.prudent.get('sa_weight', 0.07),
    )
    
    print(f"  🏆 Winner: composite={winner.get('composite', 0):.4f}")
    
    # 4. 复制 winner 作为下一代起点
    n_sampling = config.sample.dynamic.prudent.get('n_sampling', 4)
    next_pool = copy_candidates_n_times([winner], n_sampling)
    
    # 5. 保存下一代起点（在 gen{gen_idx+1} 目录）
    if gen_idx < config.sample.dynamic.prudent.get('max_generations', 5) - 1:
        next_gen_dir = output_dir / f'gen{gen_idx + 1}'
        next_gen_dir.mkdir(parents=True, exist_ok=True)
        next_frame = checkpoint_frame * (gen_idx + 1)  # 2, 4, 6, 8...
        
        next_ck = save_checkpoint(next_pool, data, gen_idx + 1, next_frame, next_gen_dir, is_full=False)
        print(f"  ⏩ 下一代起点: gen{gen_idx+1}/frame{next_frame}")
    
    return True, ck_path, full_path


def collect_results_to_excel(output_dir: Path, max_gens: int) -> None:
    """收集所有代结果到 Excel"""
    try:
        import pandas as pd
    except ImportError:
        print("⚠️  需要 pandas: pip install pandas openpyxl")
        return
    
    print(f"\n{'='*60}")
    print("[汇总] 收集结果到 Excel...")
    print(f"{'='*60}")
    
    all_data = []
    
    for gen_idx in range(max_gens):
        gen_dir = output_dir / f'gen{gen_idx}'
        if not gen_dir.exists():
            continue
        
        # 尝试加载 full .pt
        full_files = list(gen_dir.glob('*_full_*.pt'))
        if not full_files:
            continue
        
        try:
            blob = torch.load(str(full_files[-1]), map_location='cpu')
            candidates = blob.get('meta', {}).get('refined_candidates', [])
            
            for i, c in enumerate(candidates):
                all_data.append({
                    'generation': gen_idx,
                    'candidate_idx': i,
                    'frame': blob.get('frame', 24),
                    # 从 candidate 提取数据
                    'smiles': c.get('smiles', ''),
                    'num_atoms': len(c.get('pos', [])),
                    'composite': c.get('composite', c.get('prudent_score_detail', {}).get('composite')),
                    'qed': c.get('qed', c.get('prudent_score_detail', {}).get('qed')),
                    'sa': c.get('sa', c.get('prudent_score_detail', {}).get('sa')),
                    'vina': c.get('vina_affinity', c.get('prudent_score_detail', {}).get('vina_affinity')),
                })
            
            print(f"  ✅ 第 {gen_idx} 代: {len(candidates)} 候选")
        except Exception as e:
            print(f"  ⚠️  第 {gen_idx} 代加载失败: {e}")
    
    if not all_data:
        print("  ⚠️  无数据")
        return
    
    df = pd.DataFrame(all_data)
    
    # 保存 Excel
    batchsummary = output_dir / 'batchsummary'
    batchsummary.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    xlsx_path = batchsummary / f'prudent_cumulative_{timestamp}.xlsx'
    
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='all_candidates', index=False)
        
        # 统计
        if 'composite' in df.columns:
            stats = df.groupby('generation').agg({
                'candidate_idx': 'count',
                'composite': ['mean', 'max', 'min'],
                'qed': 'mean',
                'sa': 'mean',
            })
            stats.to_excel(writer, sheet_name='by_generation_stats')
        
        # 每代最佳
        if 'composite' in df.columns:
            best = df.loc[df.groupby('generation')['composite'].idxmax()]
            best.to_excel(writer, sheet_name='best_per_generation', index=False)
    
    print(f"\n✅ Excel: {xlsx_path}")
    print(f"   总行数: {len(df)}")


def main():
    parser = argparse.ArgumentParser(description='Prudent Cumulative 精简版')
    parser.add_argument('--data_id', type=int, default=None)
    parser.add_argument('--protein_path', type=str, default=None)
    parser.add_argument('--ligand_path', type=str, default=None)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--protein_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs/prudent_cumulative')
    parser.add_argument('--gpu', type=int, default=0)
    
    # 恢复模式：从指定断点继续
    parser.add_argument('--resume_from', type=str, default=None,
                        help='从指定断点 .pt 文件恢复运行（自动检测代数）')
    
    args = parser.parse_args()
    
    if args.data_id is None and args.protein_path is None:
        print("❌ 需要 --data_id 或 --protein_path")
        sys.exit(1)
    
    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = f'cuda:{args.gpu}'
    
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    config = misc.load_config(args.config)
    
    # 加载模型和数据（只一次）
    print("="*60)
    print("加载模型和数据...")
    print("="*60)
    
    model, ckpt = load_model(config, device)
    
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = trans.Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])
    
    data, pocket_id = load_data(args, ckpt, transform)
    
    protein_root = Path(args.protein_root).expanduser().resolve()
    
    # 确定起始状态
    if args.resume_from:
        # 恢复模式：从断点加载
        print(f"\n[恢复] 从断点: {args.resume_from}")
        pool, data, start_frame = load_checkpoint(Path(args.resume_from))
        
        # 从文件名推断代数
        ck_name = Path(args.resume_from).stem
        if 'gen' in ck_name:
            gen_start = int(ck_name.split('gen')[1].split('_')[0])
        else:
            gen_start = 0
    else:
        # 全新运行：从 large_step 开始
        print("\n[启动] 从 large_step 开始...")
        print("="*60)
        
        # 运行 large_step
        candidates = run_large_step(model, data, config, device)
        print(f"large_step: {len(candidates)} 候选")
        
        # 复制 n_sampling 份
        n_sampling = config.sample.dynamic.prudent.get('n_sampling', 4)
        pool = copy_candidates_n_times(candidates, n_sampling)
        print(f"复制 {n_sampling} 份: {len(pool)} 条链")
        
        start_frame = 0
        gen_start = 0
    
    # 运行各代
    max_gens = config.sample.dynamic.prudent.get('max_generations', 5)
    
    for gen_idx in range(gen_start, max_gens):
        success, ck_path, full_path = run_single_generation(
            gen_idx=gen_idx,
            start_frame=start_frame if gen_idx == gen_start else 2 * gen_idx,
            pool=pool,
            data=data,
            model=model,
            config=config,
            device=device,
            protein_root=protein_root,
            output_dir=output_dir,
        )
        
        if not success:
            print(f"\n❌ 第 {gen_idx} 代失败")
            break
        
        # 准备下一代（从 winner 复制）
        # 这里简化处理，实际需要从 winner 复制
        # pool = ...
    
    # 汇总结果
    collect_results_to_excel(output_dir, max_gens)
    
    print("\n" + "="*60)
    print("✅ 完成")
    print(f"输出: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
