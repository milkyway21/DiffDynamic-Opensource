"""Sampling arguments responsibilities for ligand sampling."""

from __future__ import annotations

import argparse


def build_parser():
    """Build the unchanged historical sampling CLI."""
    parser = argparse.ArgumentParser(
        description='扩散模型分子采样脚本。支持从测试集或直接指定蛋白/配体文件进行采样。',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  # 1. 使用测试集的指定样本（推荐用于复现测试集结果）
  python scripts/sample_diffusion.py configs/sampling.yml -i 0
  #    指定 GPU：-gpu 5  或  --device cuda:5
  
  # 2. 使用自定义蛋白和配体文件
  python scripts/sample_diffusion.py configs/sampling.yml --protein_path pocket.pdb --ligand_path ligand.sdf
  
  # 3. 使用测试集配体进行骨架生长（scaffold grow）
  # 配置: sample.scaffold.enable=true, mode=grow, after_dynamic=false
  python scripts/sample_diffusion.py configs/sampling.yml -i 0 --use_test_set
  
  # 4. 使用测试集配体进行骨架进化优化（scaffold evolve）
  # 配置: sample.scaffold.enable=true, mode=evolve, after_dynamic=false
  python scripts/sample_diffusion.py configs/sampling.yml -i 0 --use_test_set
  
  # 5. 对测试集配体进行分子优化（optimization模式）
  # 配置: sample.optimization.enable=true, after_dynamic=false
  python scripts/sample_diffusion.py configs/sampling.yml -i 0 --use_test_set

注意:
  - scaffold/optimization模式且after_dynamic=false时，会直接用输入配体作为参考/起点
  - 使用--use_test_set会明确启用测试集模式，并从数据集加载蛋白和配体
  - --molecule_path优先级最高，指定后会覆盖测试集配体
        '''
    )
    parser.add_argument('config', type=str, help='采样配置文件路径（如 configs/sampling.yml）')
    parser.add_argument('-i', '--data_id', type=int, default=None,
                        help='测试集数据索引（从0开始）。不指定且使用测试集模式时默认使用0号样本')
    parser.add_argument('--use_test_set', action='store_true',
                        help='明确使用测试集模式：从数据集加载蛋白和配体。'
                             '适用于 scaffold grow/evolve 或 optimization 模式直接优化测试集配体')
    parser.add_argument('--protein_path', type=str, default=None,
                        help='自定义蛋白口袋 PDB 文件路径（与 data_id/use_test_set 二选一）')
    parser.add_argument(
        '--protein_root',
        type=str,
        default=None,
        help='蛋白数据根目录，与 evaluate_pt --protein_root 一致；用于 Prudent 内 Vina 解析相对路径或按配体推断受体 PDB',
    )
    parser.add_argument('--ligand_path', type=str, default=None,
                        help='自定义参考配体 SDF/MOL2 文件路径（用于裁剪口袋和作为参考分子）')
    parser.add_argument('--use_dataset_for_pocket', action='store_true',
                        help='当蛋白在数据集内时，使用 index.pkl 中的配体（与测试集一致，可避免生成分子不完整）')
    parser.add_argument('--pocket_radius', type=float, default=10.0,
                        help='口袋裁剪半径（Å），提供配体时自动裁剪蛋白为口袋（默认: 10.0，匹配训练数据 pocket10）')
    parser.add_argument('--device', type=str, default='cuda:0', help='运行设备（如 cuda:0, cuda:1, cpu）')
    parser.add_argument(
        '-gpu', '--gpu',
        type=int,
        default=None,
        metavar='N',
        dest='gpu_id',
        help='指定 GPU 编号，等价于 --device cuda:N；若设置则覆盖 --device（cpu 请仍用 --device cpu）',
    )
    parser.add_argument('--batch_size', type=int, default=100, help='采样批量大小')
    parser.add_argument('--result_path', type=str, default='./outputs/pt', help='结果输出目录')
    parser.add_argument(
        '--mode', type=str,
        choices=['baseline', 'dynamic', 'optimization', 'prudent'],
        default=None,
        help='采样模式（optimization.enable 或 scaffold.enable 为 True 时被覆盖）'
    )
    parser.add_argument('--molecule_path', type=str, default=None,
                        help='待优化/参考分子的 SDF 文件路径（覆盖默认配体）。'
                             'scaffold grow/evolve/optimization 模式下用作骨架/优化起点。'
                             '未指定时默认使用测试集配体或 ligand_path 指定的配体')
    
    # Prudent 分段独立运行模式参数
    parser.add_argument('--resume-from-pt', type=str, default=None,
                        help='从指定 .pt 文件恢复运行（用于分段独立运行模式）')
    parser.add_argument('--resume-frame', type=int, default=None,
                        help='从指定 frame 开始（与 --resume-from-pt 配合使用）')
    parser.add_argument('--target-frame', type=int, default=None,
                        help='跑到指定 target frame 结束（默认24）')
    parser.add_argument('--force-generation', type=int, default=None,
                        help='强制指定 generation 编号（用于输出文件名）')

    return parser
