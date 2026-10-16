"""Batch CLI arguments, preserving all historical defaults."""

from diffdynamic.batch.state import (
    argparse,
    DEFAULT_GPU_IDS,
    DEFAULT_NUM_CPU_CORES,
    os,
    REPO_ROOT,
    Path,
)


def build_parser():
    parser = argparse.ArgumentParser(description="批量采样和评估脚本（并行执行）")

    parser.add_argument("--start", type=int, default=0, help="起始 data_id（默认: 0）")
    parser.add_argument("--end", type=int, default=99, help="结束 data_id（默认: 99）")
    parser.add_argument(
        "--data_ids",
        type=str,
        default=None,
        help='指定 data_id 列表，逗号分隔（如 "5,10,20,22"），优先于 --start/--end',
    )
    parser.add_argument(
        "--protein_path",
        type=str,
        default=None,
        help="自定义蛋白口袋 PDB 路径（与 --start/--end 二选一，用于单次自定义采样）",
    )
    parser.add_argument(
        "--ligand_path",
        type=str,
        default=None,
        help="自定义参考配体 SDF 路径（可选，仅与 --protein_path 配合）",
    )
    parser.add_argument(
        "--use_dataset_for_pocket",
        action="store_true",
        help="当蛋白在数据集内时，使用 index.pkl 中的配体（与测试集一致，可避免生成分子不完整）",
    )
    parser.add_argument(
        "--pocket_radius",
        type=float,
        default=10.0,
        help="口袋裁剪半径（Å），提供配体时自动裁剪蛋白为口袋（默认: 10.0）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（默认: configs/sampling.yml）",
    )

    default_protein_root = os.environ.get("PROTEIN_ROOT", None)
    if default_protein_root is None:
        possible_paths = [
            REPO_ROOT / "data" / "crossdocked_v1.1_rmsd1.0_pocket10",
            Path("/mnt/e/DiffDynamic/data/crossdocked_v1.1_rmsd1.0_pocket10"),
            REPO_ROOT / "data" / "crossdocked_v1.1_rmsd1.0",
        ]
        for path in possible_paths:
            if path.exists():
                default_protein_root = str(path)
                break

    parser.add_argument(
        "--protein_root",
        type=str,
        default=default_protein_root,
        help=f"蛋白质数据根目录（默认: {default_protein_root if default_protein_root else '未找到，请指定'}）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="评估输出目录（已废弃，评估结果自动保存在outputs目录下）",
    )
    parser.add_argument(
        "--atom_mode",
        type=str,
        default="add_aromatic",
        help="原子模式（默认: add_aromatic）",
    )
    parser.add_argument(
        "--exhaustiveness", type=int, default=8, help="AutoDock Vina对接强度（默认: 8）"
    )
    parser.add_argument(
        "--skip_existing", action="store_true", help="跳过已存在的.pt文件（不重新采样）"
    )
    parser.add_argument(
        "--excel_file",
        type=str,
        default=None,
        help="批次汇总主 CSV 路径（默认: batchsummary/batch_evaluation_summary_{timestamp}.csv）。"
        "可写 .xlsx 后缀，运行时会被视为同主名 .csv。"
        "仅评估模式 --pt_dir/--pt_file：相对路径落入仓库 batchsummary/",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="只生成模式：只执行采样，不执行评估（默认: False）",
    )
    parser.add_argument(
        "--pt_file",
        type=str,
        default=None,
        help="单独评估模式：指定 .pt 文件路径，仅执行评估不采样。支持多个文件逗号分隔",
    )
    parser.add_argument(
        "--pt_dir",
        type=str,
        default=None,
        help="扫描目录模式：指定目录路径，按 --pt-dir-glob 匹配该目录顶层 .pt 并评估（默认仅 result_*.pt）",
    )
    parser.add_argument(
        "--pt-dir-glob",
        type=str,
        default="result_*.pt",
        help='与 --pt_dir 连用：顶层 glob 模式（默认 result_*.pt）。评估目录内全部 pt 可用 "*.pt"',
    )
    parser.add_argument(
        "--auto-cleanup", action="store_true", help="自动清理残留进程（默认: False）"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="不清理残留进程（默认: True，已废弃，保留兼容）",
    )

    # GPU配置参数
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="指定使用的GPU ID，支持多种格式：\n"
        '  - "0,1,2,3" 指定多个GPU\n'
        '  - "0-3" 指定GPU范围\n'
        '  - "0,2-4,6" 混合格式\n'
        '  - "all" 使用所有可用GPU\n'
        f"  （默认: {','.join(map(str, DEFAULT_GPU_IDS))}）",
    )
    parser.add_argument(
        "--num_cpu_cores",
        type=int,
        default=None,
        help=f"用于评估的CPU核心数（默认: {DEFAULT_NUM_CPU_CORES}）",
    )
    parser.add_argument(
        "--cores_per_task",
        type=int,
        default=1,
        help="每个评估任务分配的CPU核心数。并行任务数 = num_cpu_cores // cores_per_task。"
        "设为>1时，每个子进程会限制OMP_NUM_THREADS等，避免多任务竞争CPU（默认: 1）",
    )
    parser.add_argument(
        "--save_intermediate_interval",
        type=int,
        default=0,
        help="每处理多少个分子保存一次中间结果；0=禁用（默认，批量模式减少磁盘占用）",
    )
    parser.add_argument(
        "--eval-show-output",
        action="store_true",
        help="评估子进程把 stdout/stderr 直接打到当前终端（可看到 tqdm/逐分子进度；多任务并行时输出会交错）。",
    )
    parser.add_argument(
        "--eval-save-log",
        action="store_true",
        help="将每个评估子进程的完整输出写入对应 eval_* 目录下的 evaluate_subprocess.log（默认不保存）。",
    )
    parser.add_argument(
        "--eval-vina-modes",
        type=str,
        default="auto",
        dest="eval_vina_modes",
        help="传给 evaluate_pt 的 --vina-modes。auto（默认）：sampling.yml 为 Prudent 或 .pt 内含 "
        "meta.prudent 时为 none（不重跑 AutoDock Vina，沿用 .pt 中 Prudent 对接分并做理化性质）；"
        "否则不传参（与历史一致：dock+score_only+minimize）。可显式写 none、score_only、"
        "dock,minimize 等。",
    )
    parser.add_argument(
        "--molecular-repair",
        type=str,
        choices=["none", "mmff", "targetdiff_baseline_refine"],
        default="none",
        help="分子修复: none；mmff=评估前强制 RDKit MMFF（--force-mmff-minimize）；"
        "targetdiff_baseline_refine=按 sampling.yml 中 sample.targetdiff_baseline_refine 对 .pt 做扩散修复",
    )
    parser.add_argument(
        "--mmff-max-iters",
        type=int,
        default=None,
        help="与 --molecular-repair mmff 联用时传给 evaluate 的 MMFF 最大迭代次数（不设则与 evaluate/yaml 默认一致）",
    )
    parser.add_argument(
        "--medchem-optimize",
        action="store_true",
        help="透传给 evaluate 的 --medchem-optimize：重建后、理化性质与 Vina 之前做激进 medchem 结构优化。"
        "会改变分子式，开启后生成分子不再是模型原始输出，论文里必须作为独立 ablation 臂汇报。",
    )
    parser.add_argument(
        "--medchem-optimize-config",
        type=str,
        default=None,
        help="透传给 evaluate 的 --medchem-optimize-config（默认用 evaluate 侧的 "
        "rdkit-structure-repair/configs/medchem_optimize.yaml）",
    )
    parser.add_argument(
        "--keep-preopt-sdf",
        type=str,
        choices=["none", "structure_only", "evaluated"],
        default=None,
        help="透传给 evaluate 的 --keep-preopt-sdf：优化前分子如何保留（不设则用 evaluate 默认 structure_only）",
    )
    parser.add_argument(
        "--collect-benchmark-pocket",
        type=int,
        default=None,
        metavar="DATA_ID",
        help="跨基线对比：从各预置 outputs 复制该口袋的 .pt 到 docs/<DATA_ID>/<方法>/pt/，并对每个 .pt "
        "启动 evaluate_pt_with_correct_reconstruct（在 pt 目录下生成 eval_* 与表格）；需有效 --protein_root "
        "（未指定时尝试 data/、/workspace/data 等回退路径）。运行前会检查 torch_scatter 是否已安装。"
        "并行评估与批量模式相同，请传 --num_cpu_cores 与 --cores_per_task（不设则 num_cpu_cores 用默认值）。",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="指定要处理的方法列表，逗号分隔。可用: TargetDiff,JSDPT3010,DiffSBDD,DecompDiff,MolForm,IPDiff,GlintDM。"
        "默认处理所有方法。示例: --methods TargetDiff,JSDPT3010 只处理这两个方法，节省计算资源。"
        "注意: DiffSBDD(SDF) 和 DecompDiff 评估较慢，如不需要可用此参数跳过。",
    )

    return parser
