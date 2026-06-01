#!/usr/bin/env python3
"""
Prudent Cumulative 多代运行脚本

调用 sample_diffusion.py 完成完整 Prudent 流程（内部处理所有代），
实时解析日志提取每代指标，生成统计报告。
python run_prudent_generations.py \
  --config configs/sampling.yml \
  --data_id 10 \
  --gpu 0 \
  --output_dir ./prudent_run \
  --timeout 7200
使用示例:python run_prudent_generations.py \
  --config configs/sampling.yml \
  --protein_path /path/to/pocket.pdb \
  --ligand_path /path/to/ligand.sdf \
  --gpu 0 \
  --output_dir ./outputs/prudent_run
"""


import argparse
import subprocess
import sys
import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

REPO_ROOT = Path(__file__).parent
SAMPLE_SCRIPT = REPO_ROOT / 'scripts' / 'sample_diffusion.py'

# 导入评估工具
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    import torch
    import pandas as pd
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("请安装: pip install rdkit torch pandas openpyxl")
    sys.exit(1)


class PrudentLogParser:
    """解析Prudent采样日志，提取每代指标"""
    
    def __init__(self):
        self.generations = {}  # {gen_idx: {'records': [], 'seed': int, 'round': int}}
        self.current_seed = None
        self.current_gen = None
    
    def parse_line(self, line: str):
        """解析单行日志"""
        # 检测seed切换
        # [Prudent] g0s0 t=650 global_offset=0 remain=24 phase1=2
        seed_match = re.search(r'g(\d+)s(\d+)', line)
        if seed_match:
            gen_idx = int(seed_match.group(1))
            seed_idx = int(seed_match.group(2))
            self.current_gen = gen_idx
            self.current_seed = seed_idx
            if gen_idx not in self.generations:
                self.generations[gen_idx] = {
                    'records': [],
                    'seed': seed_idx,
                    'global_offset': None,
                    'remain': None,
                }
            # 解析global_offset和remain
            offset_match = re.search(r'global_offset=(\d+)', line)
            remain_match = re.search(r'remain=(\d+)', line)
            if offset_match:
                self.generations[gen_idx]['global_offset'] = int(offset_match.group(1))
            if remain_match:
                self.generations[gen_idx]['remain'] = int(remain_match.group(1))
        
        # 解析评分行
        # [Prudent][score] seed=0 r=1/4 chain=0 | QED=0.2825 SA=0.5500 Vina_kcal=-8.176 vina_norm=0.8176 composite=0.7159
        # [Prudent][score] seed=0 r=1/4 chain=1 | QED=0.2513 SA=0.5600 | REJECTED (xxx), composite=-inf
        score_match = re.search(
            r'\[Prudent\]\[score\] seed=(\d+) r=(\d+)/(\d+) chain=(\d+) \| QED=(\S+) SA=(\S+)',
            line
        )
        if score_match:
            seed_idx = int(score_match.group(1))
            round_idx = int(score_match.group(2))
            total_rounds = int(score_match.group(3))
            chain_idx = int(score_match.group(4))
            qed_str = score_match.group(5)
            sa_str = score_match.group(6)
            
            # 解析数值，处理'nan'和'rejected'
            def parse_val(v):
                if v is None or v == 'nan' or v == '-inf' or 'REJECTED' in str(v):
                    return np.nan
                try:
                    return float(v)
                except:
                    return np.nan
            
            # 检查是否被拒绝
            is_rejected = 'REJECTED' in line
            
            # 尝试提取Vina和composite（被拒绝的可能没有）
            vina_match = re.search(r'Vina_kcal=(\S+)', line)
            vina_norm_match = re.search(r'vina_norm=(\S+)', line)
            composite_match = re.search(r'composite=([\-\d\.]+|nan|-inf)', line)
            lip_n_match = re.search(r'lipinski_n=(\S+)', line)
            lip_frac_match = re.search(r'lipinski_frac=(\S+)', line)
            lilly_dem_match = re.search(r'lilly_demerit=(\S+)', line)
            lilly_pass_match = re.search(r'lilly_passed=([01])', line)
            lilly_norm_match = re.search(r'lilly_norm=(\S+)', line)

            qed = parse_val(qed_str)
            sa = parse_val(sa_str)
            vina = parse_val(vina_match.group(1)) if vina_match else np.nan
            vina_norm = parse_val(vina_norm_match.group(1)) if vina_norm_match else np.nan
            composite = parse_val(composite_match.group(1)) if composite_match else np.nan

            def parse_maybe_int(v):
                if v is None:
                    return np.nan
                s = str(v).strip()
                if s in ('nan', '-inf', 'inf', ''):
                    return np.nan
                try:
                    return int(float(s))
                except (TypeError, ValueError):
                    return np.nan

            lipinski_n = parse_maybe_int(lip_n_match.group(1)) if lip_n_match else np.nan
            lipinski_frac = parse_val(lip_frac_match.group(1)) if lip_frac_match else np.nan
            if (lip_n_match is None or not np.isfinite(lipinski_n)) and lip_frac_match is not None:
                _fr = parse_val(lip_frac_match.group(1))
                if np.isfinite(_fr):
                    lipinski_n = int(round(min(1.0, max(0.0, float(_fr))) * 5.0))
            lilly_demerit = parse_maybe_int(lilly_dem_match.group(1)) if lilly_dem_match else np.nan
            if lilly_pass_match:
                lilly_passed = int(lilly_pass_match.group(1)) == 1
            else:
                lilly_passed = None
            lilly_norm = parse_val(lilly_norm_match.group(1)) if lilly_norm_match else np.nan

            record = {
                'seed': seed_idx,
                'round': round_idx,
                'total_rounds': total_rounds,
                'chain': chain_idx,
                'qed': qed,
                'sa': sa,
                'vina_kcal': vina,
                'vina_norm': vina_norm,
                'lipinski_n': lipinski_n,
                'lipinski_frac': lipinski_frac,
                'lilly_demerit': lilly_demerit,
                'lilly_passed': lilly_passed,
                'lilly_norm': lilly_norm,
                'composite': composite,
                'status': 'rejected' if is_rejected else 'accepted',
            }
            
            # 确定属于哪一代 (r=1对应g0, r=5对应g4)
            gen_idx = round_idx - 1
            if gen_idx not in self.generations:
                self.generations[gen_idx] = {'records': []}
            self.generations[gen_idx]['records'].append(record)
        
        # 解析winner选择
        # [Prudent] g0 winner_chain=2 @d(frame=2) → t606
        winner_match = re.search(r'g(\d+) winner_chain=(\d+) @d\(frame=(\d+)\)', line)
        if winner_match:
            gen_idx = int(winner_match.group(1))
            winner_chain = int(winner_match.group(2))
            frame = int(winner_match.group(3))
            if gen_idx in self.generations:
                if 'winners' not in self.generations[gen_idx]:
                    self.generations[gen_idx]['winners'] = []
                self.generations[gen_idx]['winners'].append({
                    'chain': winner_chain,
                    'frame': frame,
                })
    
    def parse_output(self, output_text: str):
        """解析完整输出"""
        for line in output_text.split('\n'):
            self.parse_line(line)
    
    def get_generation_stats(self) -> Dict[int, Dict]:
        """计算每代统计"""
        stats = {}
        for gen_idx, gen_data in sorted(self.generations.items()):
            records = gen_data.get('records', [])
            if not records:
                continue
            
            # 按seed分组
            seed_groups = {}
            for r in records:
                s = r['seed']
                if s not in seed_groups:
                    seed_groups[s] = []
                seed_groups[s].append(r)
            
            # 计算该代所有记录的统计
            qeds = [r['qed'] for r in records if not np.isnan(r['qed'])]
            sas = [r['sa'] for r in records if not np.isnan(r['sa'])]
            vinas = [r['vina_kcal'] for r in records if not np.isnan(r['vina_kcal'])]
            composites = [r['composite'] for r in records if not np.isnan(r['composite'])]

            def _finite_list(key):
                return [
                    r[key] for r in records
                    if key in r and r[key] is not None and not (isinstance(r[key], float) and np.isnan(r[key]))
                ]

            lipinski_ns = _finite_list('lipinski_n')
            lipinski_fracs = [
                r['lipinski_frac'] for r in records
                if 'lipinski_frac' in r and not np.isnan(r['lipinski_frac'])]
            lilly_norms = [
                r['lilly_norm'] for r in records
                if 'lilly_norm' in r and not np.isnan(r['lilly_norm'])]
            lilly_dems = []
            for r in records:
                v = r.get('lilly_demerit')
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                try:
                    lilly_dems.append(int(v))
                except (TypeError, ValueError):
                    pass
            lilly_passed_list = [r['lilly_passed'] for r in records if r.get('lilly_passed') is not None]

            # 计算每seed的winner统计
            seed_winners = []
            for s, recs in seed_groups.items():
                valid = [r for r in recs if not np.isnan(r['composite'])]
                if valid:
                    best = max(valid, key=lambda x: x['composite'])
                    seed_winners.append(best)

            def _mean_or_nan(arr):
                return float(np.mean(arr)) if arr else np.nan

            stats[gen_idx] = {
                'total_records': len(records),
                'accepted_count': len([r for r in records if r['status'] == 'accepted']),
                'rejected_count': len([r for r in records if r['status'] == 'rejected']),
                'num_seeds': len(seed_groups),
                'global_offset': gen_data.get('global_offset'),
                'remain': gen_data.get('remain'),

                'qed_mean': np.mean(qeds) if qeds else np.nan,
                'qed_std': np.std(qeds) if qeds else np.nan,
                'sa_mean': np.mean(sas) if sas else np.nan,
                'sa_std': np.std(sas) if sas else np.nan,
                'vina_mean': np.mean(vinas) if vinas else np.nan,
                'vina_std': np.std(vinas) if vinas else np.nan,
                'composite_mean': np.mean(composites) if composites else np.nan,
                'composite_std': np.std(composites) if composites else np.nan,

                'lipinski_n_mean': _mean_or_nan(lipinski_ns),
                'lipinski_frac_mean': _mean_or_nan(lipinski_fracs),
                'lilly_demerit_mean': _mean_or_nan(lilly_dems),
                'lilly_pass_rate': float(np.mean(lilly_passed_list)) if lilly_passed_list else np.nan,
                'lilly_norm_mean': _mean_or_nan(lilly_norms),

                'winner_count': len(seed_winners),
                'winner_qed_mean': np.mean([w['qed'] for w in seed_winners if not np.isnan(w['qed'])]) if seed_winners else np.nan,
                'winner_sa_mean': np.mean([w['sa'] for w in seed_winners if not np.isnan(w['sa'])]) if seed_winners else np.nan,
                'winner_vina_mean': np.mean([w['vina_kcal'] for w in seed_winners if not np.isnan(w['vina_kcal'])]) if seed_winners else np.nan,
                'winner_composite_mean': np.mean([w['composite'] for w in seed_winners if not np.isnan(w['composite'])]) if seed_winners else np.nan,
                'winner_lipinski_frac_mean': _mean_or_nan([w['lipinski_frac'] for w in seed_winners if 'lipinski_frac' in w and not np.isnan(w['lipinski_frac'])]),
                'winner_lilly_norm_mean': _mean_or_nan([w['lilly_norm'] for w in seed_winners if 'lilly_norm' in w and not np.isnan(w['lilly_norm'])]),
                'best_composite': max([w['composite'] for w in seed_winners if not np.isnan(w['composite'])]) if seed_winners else np.nan,
                'best_vina': min([w['vina_kcal'] for w in seed_winners if not np.isnan(w['vina_kcal'])]) if seed_winners else np.nan,
            }
        
        return stats
    
    def export_to_csv(self, output_path: Path):
        """导出每代统计到CSV"""
        stats = self.get_generation_stats()
        if not stats:
            return None
        
        # 创建DataFrame
        rows = []
        for gen_idx, stat in sorted(stats.items()):
            row = {'generation': gen_idx}
            row.update(stat)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        return df
    
    def print_summary(self):
        """打印统计摘要"""
        stats = self.get_generation_stats()
        if not stats:
            print("\n⚠️ 无日志数据可统计")
            return
        
        print("\n" + "="*100)
        print("Prudent 多代演化日志统计")
        print("="*100)
        
        # 打印表头
        print(f"{'代':<4} {'记录数':<8} {'通过/拒':<10} {'QED(μ±σ)':<18} {'SA(μ±σ)':<18} {'Vina(μ±σ)':<22} {'Composite(μ±σ)':<22}")
        print("-"*100)
        
        for gen_idx, stat in sorted(stats.items()):
            gen_label = f"g{gen_idx}"
            count = f"{stat['total_records']}"
            ratio = f"{stat['accepted_count']}/{stat['rejected_count']}"
            
            # QED
            qed_str = f"{stat['qed_mean']:.4f}±{stat['qed_std']:.4f}" if not np.isnan(stat['qed_mean']) else "N/A"
            
            # SA
            sa_str = f"{stat['sa_mean']:.4f}±{stat['sa_std']:.4f}" if not np.isnan(stat['sa_mean']) else "N/A"
            
            # Vina
            vina_str = f"{stat['vina_mean']:.3f}±{stat['vina_std']:.3f}" if not np.isnan(stat['vina_mean']) else "N/A"
            
            # Composite
            comp_str = f"{stat['composite_mean']:.4f}±{stat['composite_std']:.4f}" if not np.isnan(stat['composite_mean']) else "N/A"
            
            print(f"{gen_label:<4} {count:<8} {ratio:<10} {qed_str:<18} {sa_str:<18} {vina_str:<22} {comp_str:<22}")
        
        print("\n" + "="*100)
        print("各代 Lipinski / Lilly（来自 [Prudent][score] 日志解析）")
        print("="*100)
        print(
            f"{'代':<4} {'lipinski_n(μ)':<16} {'lipinski_frac(μ)':<18} "
            f"{'lilly_demerit(μ)':<18} {'lilly_pass%':<12} {'lilly_norm(μ)':<14}"
        )
        print("-" * 100)
        for gen_idx, stat in sorted(stats.items()):
            gen_label = f"g{gen_idx}"
            ln = stat.get('lipinski_n_mean')
            lf = stat.get('lipinski_frac_mean')
            dm = stat.get('lilly_demerit_mean')
            pr = stat.get('lilly_pass_rate')
            lnor = stat.get('lilly_norm_mean')
            ln_s = f"{ln:.3f}" if ln is not None and not np.isnan(ln) else "N/A"
            lf_s = f"{lf:.4f}" if lf is not None and not np.isnan(lf) else "N/A"
            dm_s = f"{dm:.2f}" if dm is not None and not np.isnan(dm) else "N/A"
            pr_s = f"{pr*100:.1f}%" if pr is not None and not np.isnan(pr) else "N/A"
            lnor_s = f"{lnor:.4f}" if lnor is not None and not np.isnan(lnor) else "N/A"
            print(f"{gen_label:<4} {ln_s:<16} {lf_s:<18} {dm_s:<18} {pr_s:<12} {lnor_s:<14}")
        
        # Winner统计
        print("\n" + "="*100)
        print("每代 Winner 统计 (每seed最佳)")
        print("="*100)
        print(f"{'代':<4} {'Winners':<8} {'QED(μ)':<10} {'SA(μ)':<10} {'Vina(μ)':<12} {'Comp(μ)':<10} {'lip_frac(μ)':<12} {'lilly_n(μ)':<12} {'最佳Comp':<12} {'最佳Vina':<12}")
        print("-"*100)
        
        for gen_idx, stat in sorted(stats.items()):
            gen_label = f"g{gen_idx}"
            winners = f"{stat['winner_count']}"
            w_qed = f"{stat['winner_qed_mean']:.4f}" if not np.isnan(stat['winner_qed_mean']) else "N/A"
            w_sa = f"{stat['winner_sa_mean']:.4f}" if not np.isnan(stat['winner_sa_mean']) else "N/A"
            w_vina = f"{stat['winner_vina_mean']:.3f}" if not np.isnan(stat['winner_vina_mean']) else "N/A"
            w_comp = f"{stat['winner_composite_mean']:.4f}" if not np.isnan(stat['winner_composite_mean']) else "N/A"
            w_lf = stat.get('winner_lipinski_frac_mean')
            w_ln = stat.get('winner_lilly_norm_mean')
            w_lf_s = f"{w_lf:.4f}" if w_lf is not None and not np.isnan(w_lf) else "N/A"
            w_ln_s = f"{w_ln:.4f}" if w_ln is not None and not np.isnan(w_ln) else "N/A"
            best_comp = f"{stat['best_composite']:.4f}" if not np.isnan(stat['best_composite']) else "N/A"
            best_vina = f"{stat['best_vina']:.3f}" if not np.isnan(stat['best_vina']) else "N/A"
            
            print(f"{gen_label:<4} {winners:<8} {w_qed:<10} {w_sa:<10} {w_vina:<12} {w_comp:<10} {w_lf_s:<12} {w_ln_s:<12} {best_comp:<12} {best_vina:<12}")
        
        print("\n")


def _sanitize_excel_sheet(name: str) -> str:
    """Excel 工作表名限制：31 字符，且不能包含 []:*?/\\"""
    for ch in '[]:*?/\\':
        name = name.replace(ch, '_')
    return name[:31]


def import_scoring_utils():
    """动态导入评分工具（避免循环导入问题）"""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import utils.evaluation.scoring_func as scoring_func
        from utils.evaluation.lilly_medchem_rules import evaluate_lilly_medchem_rules
        return scoring_func, evaluate_lilly_medchem_rules
    except ImportError as e:
        print(f"⚠️ import失败: {e}")
        return None, None
    finally:
        sys.path.pop(0)


def _sample_ligand_atom_mode(config_path: Optional[Path]) -> str:
    if config_path is None:
        return 'add_aromatic'
    try:
        p = Path(config_path).expanduser()
        if not p.is_file():
            return 'add_aromatic'
        import yaml

        with open(p, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        if isinstance(cfg, dict):
            sample = cfg.get('sample') or {}
            if isinstance(sample, dict):
                return str(sample.get('ligand_atom_mode', 'add_aromatic'))
    except Exception:
        pass
    return 'add_aromatic'


def _lilly_description_from_result(lilly_result: Optional[Dict[str, Any]]) -> Optional[str]:
    if not lilly_result:
        return None
    parts: List[str] = []
    if not lilly_result.get('passed', False):
        rr = lilly_result.get('reject_reason', '')
        if rr:
            parts.append(f'拒绝:{rr}')
    matched = lilly_result.get('matched_rules') or []
    if matched:
        tail = [str(x) for x in matched[:12]]
        suf = '…' if len(matched) > 12 else ''
        parts.append(f'命中:{",".join(tail)}{suf}')
    dem = lilly_result.get('demerit', 0)
    cutoff = lilly_result.get('demerit_cutoff', 100)
    parts.append(f'扣分={dem}/{cutoff}')
    n_ha = lilly_result.get('n_heavy_atoms', 0)
    if n_ha:
        parts.append(f'重原子={n_ha}')
    return '; '.join(parts) if parts else '通过'


def _smiles_from_pos_v(pos, v, ligand_atom_mode: str) -> str:
    try:
        import torch
        import utils.transforms as trans
        import utils.reconstruct as reconstruct

        rs = str(REPO_ROOT)
        inserted = rs not in sys.path
        if inserted:
            sys.path.insert(0, rs)
        try:
            v_tensor = torch.tensor(np.asarray(v), dtype=torch.long)
            atom_numbers = trans.get_atomic_number_from_index(v_tensor, mode=ligand_atom_mode)
            aromatic_flags = trans.is_aromatic_from_index(v_tensor, mode=ligand_atom_mode)
            mol = reconstruct.reconstruct_from_generated(
                np.asarray(pos, dtype=np.float64),
                atom_numbers,
                aromatic_flags,
            )
            if mol is not None:
                return Chem.MolToSmiles(mol)
        finally:
            if inserted:
                try:
                    sys.path.remove(rs)
                except ValueError:
                    pass
    except Exception:
        pass
    return ''


def _reference_smiles_from_blob(blob: Dict[str, Any], ligand_atom_mode: str) -> str:
    data = blob.get('data')
    if data is None:
        return ''
    try:
        if not hasattr(data, 'ligand_pos') or data.ligand_pos is None:
            return ''
        pos = data.ligand_pos.detach().cpu().numpy()
        vfeat = getattr(data, 'ligand_atom_feature_full', None)
        if vfeat is None:
            return ''
        v = vfeat.detach().cpu().numpy()
        return _smiles_from_pos_v(pos, v, ligand_atom_mode)
    except Exception:
        return ''


def _tanimoto_morgan(smiles_a: str, smiles_b: str, radius: int = 2, n_bits: int = 2048) -> float:
    if not smiles_a or not smiles_b:
        return float('nan')
    try:
        from rdkit import DataStructs
        from rdkit.Chem import AllChem

        m1 = Chem.MolFromSmiles(str(smiles_a).strip())
        m2 = Chem.MolFromSmiles(str(smiles_b).strip())
        if m1 is None or m2 is None:
            return float('nan')
        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, radius, nBits=n_bits)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, radius, nBits=n_bits)
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return float('nan')


def _reorder_excel_columns(df: pd.DataFrame) -> pd.DataFrame:
    front = [
        '分子ID',
        'SMILES',
        'TanimotoSimilarity',
        'Lipinski条数',
        'Lipinski_frac',
        'Lipinski规则得分',
        'Lilly扣分',
        'Lilly_通过',
        'Lilly_norm',
        'Lilly_Medchem_通过',
        'Lilly_Medchem_扣分',
        'Lilly_Medchem_描述',
    ]
    rest = [c for c in df.columns if c not in front]
    return df[[c for c in front if c in df.columns] + rest]


def _sync_lipinski_lilly_from_prudent_detail(result: Dict[str, Any], detail: Dict[str, Any]) -> None:
    if not detail:
        return
    lip_n = detail.get('lipinski')
    if lip_n is not None and lip_n != 'N/A':
        try:
            n = int(lip_n)
            result['Lipinski条数'] = n
            result['Lipinski规则得分'] = result.get('Lipinski规则得分') or n
        except (TypeError, ValueError):
            pass
    lf = detail.get('lipinski_frac')
    if lf is not None:
        try:
            x = float(lf)
            if np.isfinite(x):
                result['Lipinski_frac'] = x
        except (TypeError, ValueError):
            pass
    ld = detail.get('lilly_demerit')
    if ld is not None:
        try:
            d = int(ld)
            result['Lilly扣分'] = d
            result['Lilly_Medchem_扣分'] = d
        except (TypeError, ValueError):
            pass
    lp = detail.get('lilly_passed')
    if lp is not None:
        b = bool(lp)
        result['Lilly_通过'] = b
        result['Lilly_Medchem_通过'] = b
    lnorm = detail.get('lilly_norm')
    if lnorm is not None:
        try:
            x = float(lnorm)
            if np.isfinite(x):
                result['Lilly_norm'] = x
        except (TypeError, ValueError):
            pass


def run_sample_once(args) -> Tuple[Optional[Path], Optional[PrudentLogParser]]:
    """运行一次 sample_diffusion.py，完成所有 Prudent 代（实时输出日志）
    
    返回: (pt_file_path, log_parser)
    """
    print("\n" + "="*60)
    print("[采样] 启动 Prudent Cumulative（内部处理所有代）")
    print("="*60)

    cmd = [
        sys.executable,
        str(SAMPLE_SCRIPT),
        args.config,
        '--device', f'cuda:{args.gpu}',
        '--result_path', args.output_dir,
    ]

    if args.data_id is not None:
        cmd.extend(['--data_id', str(args.data_id)])
    else:
        cmd.extend(['--protein_path', args.protein_path])
        if args.ligand_path:
            cmd.extend(['--ligand_path', args.ligand_path])

    print(f"CMD: {' '.join(cmd)}")
    print("-" * 40)
    print("[OUT]")

    stdout_lines = []
    stderr_lines = []
    final_pt = None
    
    # 初始化日志解析器
    log_parser = PrudentLogParser()

    try:
        import select
        
        # 使用 Popen 启动子进程，实时输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # 实时读取 stdout 和 stderr
        while True:
            # 检查进程是否结束
            ret = process.poll()
            
            # 读取可用的输出
            reads = [process.stdout, process.stderr]
            readable, _, _ = select.select(reads, [], [], 0.1)
            
            for stream in readable:
                line = stream.readline()
                if line:
                    prefix = 'O' if stream is process.stdout else 'E'
                    print(f"[{prefix}] {line}", end='')
                    stdout_lines.append(line) if stream is process.stdout else stderr_lines.append(line)
                    # 实时解析日志
                    log_parser.parse_line(line)
            
            if ret is not None:
                for line in process.stdout:
                    print(f"[O] {line}", end=''); stdout_lines.append(line)
                    log_parser.parse_line(line)
                for line in process.stderr:
                    print(f"[E] {line}", end=''); stderr_lines.append(line)
                    log_parser.parse_line(line)
                break

        print("-" * 40)
        print(f"[END rc={ret}]")

        # 合并输出用于解析
        output_text = ''.join(stdout_lines) + '\n' + ''.join(stderr_lines)

        # 检查进程是否成功
        if ret != 0:
            print(f"❌ 采样失败，返回码: {ret}")
            return None, log_parser

        # 采样完成性校验：防止未完整跑完就进入评估
        prudent_mode = ('[Prudent]' in output_text) or ('Prudent 模式' in output_text) or ('Prudent 分段演化' in output_text)
        idx_refine_done = output_text.rfind('[Prudent] refine完成')
        idx_sample_done = output_text.rfind('Sample done!')
        idx_saved = output_text.rfind('Results saved to:')

        if idx_sample_done < 0 or idx_saved < 0:
            print("❌ 采样日志不完整：缺少 'Sample done!' 或 'Results saved to:'，中止后续评估")
            return None, log_parser
        if idx_sample_done > idx_saved:
            print("❌ 日志顺序异常：出现 'Results saved to' 早于 'Sample done!'，中止后续评估")
            return None, log_parser
        if prudent_mode:
            # 放宽检查：只要有Prudent评分日志即可，不强制要求'refine完成'标记
            has_prudent_score = '[Prudent][score]' in output_text
            if not has_prudent_score:
                print("⚠️ Prudent 日志警告：未检测到评分记录，但继续处理")
            # 如果存在refine完成标记，检查顺序
            if idx_refine_done >= 0 and idx_refine_done > idx_sample_done:
                print("❌ 日志顺序异常：'Prudent refine完成' 晚于 'Sample done!'，中止后续评估")
                return None, log_parser

        # 从输出中找到最终 .pt 文件路径
        for line in output_text.split('\n'):
            if 'Results saved to:' in line and '.pt' in line:
                parts = line.split('Results saved to:')
                if len(parts) > 1:
                    final_pt = Path(parts[-1].strip())
                    break
            elif '.pt' in line and 'result_' in line:
                words = line.split()
                for word in words:
                    if 'result_' in word and '.pt' in word:
                        final_pt = Path(word.strip().strip('.'))
                        break

        if final_pt and final_pt.exists():
            print(f"✅ {final_pt}")
            return final_pt, log_parser
        else:
            # 尝试从输出目录找最新的 result_*.pt
            output_path = Path(args.output_dir)
            pt_files = list(output_path.glob('result_*.pt'))
            if pt_files:
                pt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                print(f"✅ {pt_files[0]}")
                return pt_files[0], log_parser
            print("⚠️ 无PT路径")
            return None, log_parser

    except subprocess.TimeoutExpired:
        print(f"⏱️ 超时>{args.timeout}s")
        return None, log_parser
    except Exception as e:
        print(f"❌ 异常: {e}")
        import traceback; traceback.print_exc()
        return None, log_parser


def find_all_pt_files(final_pt: Path) -> List[Path]:
    """从采样输出目录找到与当前数据ID相关的 .pt 文件"""
    output_dir = final_pt.parent
    
    # 从final_pt提取data_id（格式: result_{data_id}_it{N}_{timestamp}.pt）
    match = re.search(r'result_(\d+)_', final_pt.name)
    data_id = match.group(1) if match else None
    
    # 找所有 result_{data_id}_*.pt（只匹配当前data_id）
    if data_id:
        pattern = f'result_{data_id}_*.pt'
        result_pts = list(output_dir.glob(pattern))
    else:
        # 如果无法确定data_id，只返回final_pt
        result_pts = [final_pt]
    
    # checkpoint / 每代全链 pool（gen*_seed*_chains*）
    if data_id:
        checkpoint_pts = list(output_dir.glob(f'result_{data_id}_*_*_frame*.pt'))
        full_pts = list(output_dir.glob(f'result_{data_id}_*_*_full.pt'))
        gen_pool_pts = list(output_dir.glob(f'result_{data_id}_*_*_gen*_seed*_chains*.pt'))
    else:
        checkpoint_pts = list(output_dir.glob('result_*_*_*_frame*.pt'))
        full_pts = list(output_dir.glob('result_*_*_*_full.pt'))
        gen_pool_pts = list(output_dir.glob('result_*_*_*_gen*_seed*_chains*.pt'))

    # 合并去重
    all_pts = list(set(result_pts + checkpoint_pts + full_pts + gen_pool_pts))
    # 按文件名中的代数和类型排序
    all_pts.sort(key=lambda x: (extract_gen_from_filename(x.name), x.name))

    print(f"\n[PT:{len(all_pts)}] (data_id={data_id})")
    for pt in all_pts:
        gen = extract_gen_from_filename(pt.name)
        print(f"  [G{gen}] {pt.name} ({pt.stat().st_size/1048576:.1f}MB)")

    return all_pts


def extract_gen_from_filename(filename: str) -> int:
    """从文件名提取 generation 编号
    
    支持格式:
    - result_10_it4_... -> 4 (从it4提取)
    - result_10_gen2_... -> 2 (从gen2提取)
    - result_10_seg2_... -> 2 (从seg2提取)
    """
    import re
    patterns = [
        r'it(\d+)[_.]',           # it4_ 或 it4.
        r'_it(\d+)[_.]',          # _it4_ 或 _it4.
        r'gen(\d+)',
        r'seg(\d+)',
        r'_gen(\d+)_',
        r'_seg(\d+)_',
    ]
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 0


def evaluate_molecule_from_pt(
    candidate: Dict[str, Any],
    scoring_func,
    evaluate_lilly,
    atom_mode: str = 'add_aromatic',
) -> Dict[str, Any]:
    """从 PT 候选数据评估分子（不跑 Vina）

    返回与 evaluate_pt_with_correct_reconstruct 类似的字段
    """
    result: Dict[str, Any] = {
        'SMILES': candidate.get('smiles') or '',
        '原子数': candidate.get('num_atoms', 0),
        'QED评分': None,
        'SA评分': None,
        'logP': None,
        'TPSA': None,
        'Lipinski条数': None,
        'Lipinski_frac': None,
        'Lipinski规则得分': None,
        'PAINS检测': None,
        '分子稳定性': None,
        '稳定原子数': None,
        '总原子数(稳定性)': None,
        '键数': None,
        '环数': None,
        '分子量': None,
        '构象能量': None,
        'Lilly扣分': None,
        'Lilly_通过': None,
        'Lilly_norm': None,
        'Lilly_Medchem_通过': None,
        'Lilly_Medchem_扣分': None,
        'Lilly_Medchem_描述': None,
        'TanimotoSimilarity': None,
        'RDKit验证': None,
        '综合模型评分': None,
    }

    detail = dict(candidate.get('prudent_score_detail') or {})
    cdetail = candidate.get('prudent_composite_detail') or {}
    if cdetail:
        detail = {**detail, **cdetail}
    if detail:
        result['QED评分'] = detail.get('qed')
        result['SA评分'] = detail.get('sa')
        result['综合模型评分'] = detail.get('composite')
        result['Prudent_Vina_kcal'] = detail.get('vina_affinity')
        result['Prudent_vina_norm'] = detail.get('vina_norm')
        result['Prudent_lipinski_count'] = detail.get('lipinski')
        result['Prudent_lipinski_frac'] = detail.get('lipinski_frac')
        result['Prudent_lilly_demerit'] = detail.get('lilly_demerit')
        result['Prudent_lilly_passed'] = detail.get('lilly_passed')
        result['Prudent_lilly_norm'] = detail.get('lilly_norm')
        _sync_lipinski_lilly_from_prudent_detail(result, detail)

    if not result['SMILES'] and candidate.get('pos') is not None and candidate.get('v') is not None:
        result['SMILES'] = _smiles_from_pos_v(candidate['pos'], candidate['v'], atom_mode)

    if not result['SMILES']:
        return result

    try:
        mol = Chem.MolFromSmiles(result['SMILES'])
        if mol is None:
            return result

        mol_with_h = Chem.AddHs(mol)

        # 1. 基础化学性质
        try:
            chem = scoring_func.get_chem(mol)
            result['QED评分'] = chem.get('qed', result['QED评分'])
            result['SA评分'] = chem.get('sa', result['SA评分'])
            result['logP'] = chem.get('logp', 'N/A')
        except Exception:
            pass

        # 2. TPSA
        try:
            result['TPSA'] = Descriptors.TPSA(mol)
        except Exception:
            pass

        # 3. Lipinski
        try:
            lip_raw = scoring_func.get_lipinski(mol)
            li_n = lip_raw.get('lipinski', 'N/A') if isinstance(lip_raw, dict) else 'N/A'
            if li_n != 'N/A' and li_n is not None:
                li_n = int(li_n)
                result['Lipinski条数'] = li_n
                result['Lipinski规则得分'] = li_n
                result['Lipinski_frac'] = max(0.0, min(1.0, float(li_n) / 5.0))
            else:
                result['Lipinski规则得分'] = 'N/A'
        except Exception:
            pass

        # 4. PAINS
        try:
            result['PAINS检测'] = scoring_func.is_pains(mol)
        except Exception:
            pass

        # 5. 基础结构信息
        try:
            basic = scoring_func.get_basic(mol)
            result['键数'] = basic[1]
            result['环数'] = basic[2]
            result['分子量'] = basic[3]
        except Exception:
            pass

        # 6. 分子稳定性
        try:
            stability = scoring_func.check_stability(mol)
            result['分子稳定性'] = stability[0]
            result['稳定原子数'] = stability[1]
            result['总原子数(稳定性)'] = stability[2]
        except Exception:
            pass

        # 7. 构象能量
        try:
            # 生成构象并计算能量
            from rdkit.Chem import AllChem
            AllChem.EmbedMolecule(mol_with_h, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol_with_h)
            energies = scoring_func.get_conformer_energies(mol_with_h, force_field='mmff')
            if energies:
                result['构象能量'] = min(energies)
        except Exception:
            pass

        # 8. Lilly Medchem Rules
        try:
            lilly_result = evaluate_lilly(mol, debug=False)
            if lilly_result:
                dem = int(lilly_result.get('demerit', 0) or 0)
                cutoff = max(1, int(lilly_result.get('demerit_cutoff', 100) or 100))
                passed = bool(lilly_result.get('passed', False))
                result['Lilly_Medchem_通过'] = passed
                result['Lilly_Medchem_扣分'] = dem
                result['Lilly_Medchem_描述'] = _lilly_description_from_result(lilly_result)
                result['Lilly扣分'] = dem
                result['Lilly_通过'] = passed
                if passed:
                    result['Lilly_norm'] = max(0.0, 1.0 - float(dem) / float(cutoff))
                else:
                    result['Lilly_norm'] = 0.0
        except Exception:
            pass

        try:
            result['RDKit验证'] = True
        except Exception:
            pass

    except Exception as e:
        print(f"    ⚠️ {e}")

    return result


def collect_and_save_excel(
    all_pt_files: List[Path],
    output_dir: Path,
    data_id: str,
    log_parser: Optional['PrudentLogParser'] = None,
    sampling_config: Optional[Path] = None,
) -> None:
    """收集所有 PT 文件并直接评估保存为 Excel。

    - 自 PT 的 ``refined_candidates``：工作表 ``g{n}``（n 来自文件名 it/gen 等）及 ``avg``（逐列平均）。
    - 自 ``data`` 晶体配体重建参考 SMILES，在各 ``g*`` 表计算与 crystal 的 Morgan TanimotoSimilarity。
    - 自采样日志：``log_g{n}`` / ``log_summary`` 亦写入同名列 ``TanimotoSimilarity``（无 SMILES 时为 NaN）。
    """
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("⚠️ pip install openpyxl")
        return

    scoring_func, evaluate_lilly = import_scoring_utils()
    if scoring_func is None:
        print("❌ 无评分工具")
        return

    atom_mode = _sample_ligand_atom_mode(
        Path(sampling_config) if sampling_config is not None else None,
    )

    print("\n" + "="*40)
    print("[EVAL]")
    print("="*40)

    generation_data: Dict[int, List[pd.DataFrame]] = {}
    ref_smiles_guess = ''

    for pt_file in all_pt_files:
        gen_idx = extract_gen_from_filename(pt_file.name)

        print(f"\n  [G{gen_idx}] {pt_file.name}")

        try:
            blob = torch.load(str(pt_file), map_location='cpu')
            if not ref_smiles_guess:
                ref_smiles_guess = _reference_smiles_from_blob(blob, atom_mode) or ''

            meta = blob.get('meta', {})
            candidates = meta.get('refined_candidates', [])

            if not candidates:
                print(f"    ⚠️ 无候选")
                continue

            records = []
            for i, c in enumerate(candidates):
                if i % 10 == 0:
                    print(f"    {i}/{len(candidates)}", end='\r')

                rec = evaluate_molecule_from_pt(
                    c, scoring_func, evaluate_lilly, atom_mode=atom_mode,
                )
                rec['分子ID'] = i
                records.append(rec)

            print(f"    ✅ {len(records)} mol")

            df = pd.DataFrame(records)
            if gen_idx not in generation_data:
                generation_data[gen_idx] = []
            generation_data[gen_idx].append(df)

        except Exception as e:
            print(f"    ❌ {e}")

    if not generation_data:
        print("  ⚠️ 无数据")
        return

    merged_by_gen: Dict[int, pd.DataFrame] = {}
    for gen_idx, dfs in sorted(generation_data.items()):
        if len(dfs) == 1:
            merged_by_gen[gen_idx] = dfs[0].copy()
        else:
            merged_by_gen[gen_idx] = pd.concat(dfs, ignore_index=True)

    ref_smiles = (ref_smiles_guess or '').strip()
    if ref_smiles:
        tail = '…' if len(ref_smiles) > 80 else ''
        print(f"\n  [REF] crystal ligand SMILES（Tanimoto 基准）: {ref_smiles[:80]}{tail}")
    else:
        print("\n  [REF] 未能从 PT 的 data 重建参考 SMILES，TanimotoSimilarity 将置空")

    for gen_idx in sorted(merged_by_gen.keys()):
        df = merged_by_gen[gen_idx]
        tans: List[float] = []
        for _, row in df.iterrows():
            smi = str(row.get('SMILES', '') or '').strip()
            if ref_smiles and smi and '.' not in smi:
                tans.append(_tanimoto_morgan(smi, ref_smiles))
            else:
                tans.append(float('nan'))
        df['TanimotoSimilarity'] = tans
        merged_by_gen[gen_idx] = _reorder_excel_columns(df)

    batchsummary = Path(output_dir) / 'batchsummary'
    batchsummary.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = batchsummary / f'prudent_generations_{data_id}_{timestamp}.xlsx'

    with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
        ref_df = pd.DataFrame([
            {
                'reference_SMILES': ref_smiles,
                'TanimotoSimilarity': float(1.0) if ref_smiles else float('nan'),
            },
        ])
        ref_df.to_excel(
            writer,
            sheet_name=_sanitize_excel_sheet('reference_ligand'),
            index=False,
        )
        print("   📊 reference_ligand(1)")

        for gen_idx, df in sorted(merged_by_gen.items()):
            sheet_name = f'g{gen_idx}'
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"   📊 {sheet_name}({len(df)})")

        avg_records = []
        for gen_idx, df in sorted(merged_by_gen.items()):
            rec = {'generation': gen_idx, 'molecule_count': len(df)}

            for col in df.columns:
                if col in ['分子ID', 'SMILES', 'Lilly_Medchem_描述']:
                    continue
                if any(kw in col.lower() for kw in ['vina', 'dock', 'jsd', 'rmsd', '亲和力']):
                    continue

                try:
                    values = pd.to_numeric(df[col].replace('N/A', np.nan), errors='coerce')
                    avg_val = values.mean()
                    if pd.notna(avg_val):
                        simple_name = col.replace('评分', '').replace('规则得分', '').replace('检测', '')
                        rec[simple_name] = avg_val
                except Exception:
                    pass

            if 'Lilly_Medchem_通过' in df.columns:
                try:
                    passed = df['Lilly_Medchem_通过'].replace('N/A', False).fillna(False).astype(bool)
                    rec['Lilly_pass_rate'] = passed.mean()
                except Exception:
                    pass

            avg_records.append(rec)

        avg_df = pd.DataFrame(avg_records)
        tan_means = []
        for _, df in sorted(merged_by_gen.items()):
            v = pd.to_numeric(df['TanimotoSimilarity'], errors='coerce')
            tan_means.append(float(v.mean()) if v.notna().any() else float('nan'))
        avg_df['TanimotoSimilarity'] = tan_means

        avg_df = avg_df.dropna(axis=1, how='all')
        avg_df.to_excel(writer, sheet_name=_sanitize_excel_sheet('avg'), index=False)
        print(f"   📊 avg({len(avg_df)})")

        if log_parser is not None:
            for gen_idx, gen_data in sorted(log_parser.generations.items()):
                recs = gen_data.get('records') or []
                if not recs:
                    continue
                log_df = pd.DataFrame(recs)
                log_df.insert(0, 'generation', gen_idx)
                log_df['TanimotoSimilarity'] = np.nan
                log_df['reference_SMILES'] = ref_smiles
                sn = _sanitize_excel_sheet(f'log_g{gen_idx}')
                log_df.to_excel(writer, sheet_name=sn, index=False)
                print(f"   📊 {sn}({len(log_df)})")

            stats = log_parser.get_generation_stats()
            if stats:
                log_sum_rows = []
                for gen_idx, stat in sorted(stats.items()):
                    row = {'generation': gen_idx}
                    row.update(stat)
                    log_sum_rows.append(row)
                log_summary_df = pd.DataFrame(log_sum_rows)
                log_summary_df['TanimotoSimilarity'] = np.nan
                log_summary_df['reference_SMILES'] = ref_smiles
                log_summary_df = log_summary_df.dropna(axis=1, how='all')
                log_summary_df.to_excel(
                    writer,
                    sheet_name=_sanitize_excel_sheet('log_summary'),
                    index=False,
                )
                print(f"   📊 log_summary({len(log_summary_df)})")

    print(f"\n✅ {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Prudent Cumulative 多代运行')
    parser.add_argument('--data_id', type=int, default=None, help='数据ID')
    parser.add_argument('--protein_path', type=str, default=None, help='蛋白路径（不使用data_id时）')
    parser.add_argument('--ligand_path', type=str, default=None, help='配体路径（可选）')
    parser.add_argument('--config', type=str, required=True, help='采样配置文件')
    parser.add_argument('--output_dir', type=str, default='./outputs/prudent_run', help='输出目录')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备（容器内实际可见的GPU编号）')
    parser.add_argument('--timeout', type=int, default=7200, help='采样超时时间（秒）')

    args = parser.parse_args()

    if args.data_id is None and args.protein_path is None:
        print("❌ 需--data_id或--protein_path")
        sys.exit(1)

    import torch
    available_gpus = torch.cuda.device_count()
    requested_gpu = args.gpu

    if requested_gpu >= available_gpus:
        print(f"⚠️ GPU {requested_gpu}不可用，仅{available_gpus}个(0~{available_gpus-1})")
        sys.exit(1)

    # 创建基础输出目录
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建本次运行的独立子文件夹: {data_id}_{timestamp}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_id = str(args.data_id) if args.data_id is not None else 'custom'
    run_dir = base_output_dir / f"{data_id}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 更新args，让采样输出到子文件夹
    args.output_dir = str(run_dir)

    print("="*40)
    print(f"cfg={args.config}")
    print(f"base_out={base_output_dir}")
    print(f"run_out={run_dir}")
    print(f"gpu={args.gpu}")
    print("="*40)

    final_pt, log_parser = run_sample_once(args)
    
    # 无论成功与否，都打印和导出日志统计到子文件夹
    if log_parser:
        log_parser.print_summary()
        
        # 导出日志统计到CSV（存储到子文件夹的batchsummary）
        batchsummary = run_dir / 'batchsummary'
        batchsummary.mkdir(parents=True, exist_ok=True)
        csv_path = batchsummary / f'prudent_log_stats_{data_id}_{timestamp}.csv'
        df = log_parser.export_to_csv(csv_path)
        if df is not None:
            print(f"✅ 日志统计已导出: {csv_path}")
    
    if not final_pt:
        print("\n❌ 采样失败或未完成")
        sys.exit(1)

    all_pt_files = find_all_pt_files(final_pt)
    collect_and_save_excel(
            all_pt_files, run_dir, data_id,
            log_parser=log_parser, sampling_config=Path(args.config),
        )

    print("\n" + "="*40)
    print(f"Done run_dir={run_dir}")
    print(f"pt={final_pt.name} n={len(all_pt_files)}")
    print("="*40)


if __name__ == '__main__':
    main()
