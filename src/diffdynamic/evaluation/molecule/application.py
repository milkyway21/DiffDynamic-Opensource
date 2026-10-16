"""Application responsibilities extracted from evaluate_pt_with_correct_reconstruct.py."""

from diffdynamic.evaluation.molecule.state import (
    Path,
    _TRUE_VALUES,
    argparse,
    datetime,
    os,
)
from diffdynamic.evaluation.molecule import state as _state


def main():
    from diffdynamic.evaluation.molecule.history import (
        record_evaluation_to_global_log,
        update_sampling_history,
    )
    from diffdynamic.evaluation.molecule.options import parse_vina_modes_arg
    from diffdynamic.evaluation.molecule.pipeline import evaluate_pt_file

    parser = argparse.ArgumentParser(
        description="使用正确的reconstruct方法评估.pt文件中的分子",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python evaluate_pt_with_correct_reconstruct.py outputs/result.pt
  
  # 指定蛋白根目录和输出目录
  python evaluate_pt_with_correct_reconstruct.py outputs/result.pt \\
      --protein_root ./data/crossdocked_v1.1_rmsd1.0 \\
      --output_dir ./eval_results
  
  # 使用basic模式（不考虑芳香性）
  python evaluate_pt_with_correct_reconstruct.py outputs/result.pt \\
      --atom_mode basic
  
  # 调整对接参数
  python evaluate_pt_with_correct_reconstruct.py outputs/result.pt \\
      --exhaustiveness 8 \\
      --size_factor 1.0 \\
      --buffer 5.0

  # 纯口袋 .pt（无 ligand_filename）：用 index 补全或用显式受体
  python evaluate_pt_with_correct_reconstruct.py docs/10/IPDiff/sample.pt \\
      --protein_root ./data/crossdocked_v1.1_rmsd1.0_pocket10 \\
      --index_pkl ./data/crossdocked_v1.1_rmsd1.0_pocket10/index.pkl \\
      --output_dir ./docs/10/IPDiff/eval_from_index

  # pocket10 测试集编号（如 docs/10 = data_id 10 = 3DZH）：务必用 100ligands.csv，勿用全量 index 行号
  python evaluate_pt_with_correct_reconstruct.py docs/10/IPDiff/result_10.pt \\
      --protein_root ./data/crossdocked_v1.1_rmsd1.0_pocket10 \\
      --benchmark_ligands_csv ./docs/100ligands.csv \\
      --output_dir ./docs/10/IPDiff/eval_3dzh

  python evaluate_pt_with_correct_reconstruct.py docs/10/IPDiff/sample.pt \\
      --protein_root ./data/crossdocked_v1.1_rmsd1.0_pocket10 \\
      --receptor_pdb ./data/crossdocked_v1.1_rmsd1.0_pocket10/xxx/xxx_pocket10.pdb \\
      --data_id 10 \\
      --output_dir ./docs/10/IPDiff/eval_explicit
        """,
    )

    parser.add_argument(
        "pt_file", type=str, help=".pt文件路径（包含pred_ligand_pos和pred_ligand_v）"
    )
    parser.add_argument(
        "--protein_root",
        type=str,
        default="./data/crossdocked_v1.1_rmsd1.0",
        help="蛋白质数据根目录（默认: ./data/crossdocked_v1.1_rmsd1.0）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="结果输出目录（默认: ./eval_results）",
    )
    parser.add_argument(
        "--atom_mode",
        type=str,
        choices=["basic", "add_aromatic"],
        default="add_aromatic",
        help="原子编码模式（默认: add_aromatic）",
    )
    parser.add_argument(
        "--reconstruction-method",
        choices=["current", "targetdiff_legacy"],
        default="current",
        help="targetdiff_legacy uses the frozen original TargetDiff "
        "de novo reconstruction (covalent factor 1.3).",
    )
    parser.add_argument(
        "--exhaustiveness", type=int, default=8, help="AutoDock Vina搜索强度（默认: 8）"
    )
    parser.add_argument(
        "--vina-timeout-seconds",
        type=float,
        default=None,
        help="每次 Vina dock/score_only/minimize 的 wall-clock 上限（秒）。"
        "不设则读环境变量 VINA_DOCK_TIMEOUT_SEC，仍默认 128；0 表示不限制。",
    )
    parser.add_argument(
        "--n_poses",
        type=int,
        default=9,
        help="生成的对接姿势数量（默认: 9，与configs/sampling.yml中的vina_poses一致）",
    )
    parser.add_argument(
        "--size_factor", type=float, default=1.0, help="搜索盒尺寸因子（默认: 1.0）"
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=5.0,
        help="搜索盒缓冲区大小（默认: 5.0，与docking_vina.py一致）",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="最大评估样本数（默认: 全部）"
    )
    parser.add_argument(
        "--save_sdf",
        action="store_true",
        default=True,
        help="保存重建成功的分子为SDF文件（默认: True）",
    )
    parser.add_argument(
        "--no_sdf", dest="save_sdf", action="store_false", help="不保存SDF文件"
    )
    parser.add_argument(
        "--remove-fragments",
        action="store_true",
        default=True,
        help="对接前去除小碎片，仅保留最大连通片段（默认: 开启）",
    )
    parser.add_argument(
        "--no-remove-fragments",
        dest="remove_fragments",
        action="store_false",
        help="禁用碎片去除",
    )
    parser.add_argument(
        "--debug", action="store_true", help="启用调试模式（显示详细信息）"
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default=None,
        help="对接临时目录（默认: 自动生成唯一目录，避免并行冲突）",
    )
    parser.add_argument(
        "--enable_isolation",
        action="store_true",
        help="启用单分子子进程隔离与超时（防底层C++崩溃；默认关闭，同进程顺序评估）",
    )
    parser.add_argument(
        "--disable_isolation", action="store_true", help=argparse.SUPPRESS
    )  # 兼容旧参数；默认已不启用隔离
    parser.add_argument(
        "--save_intermediate_interval",
        type=int,
        default=64,
        help="每处理多少个分子保存一次中间结果；0=禁用（默认64，可减少磁盘占用）",
    )
    parser.add_argument(
        "--force-mmff-minimize",
        action="store_true",
        help="对接前强制启用 RDKit MMFF（覆盖 sampling.yml 的 sample.pre_docking.mmff_minimize）",
    )
    parser.add_argument(
        "--mmff-max-iters",
        type=int,
        default=None,
        help="MMFF 最大迭代次数（默认与 yaml 一致，通常 200；需与 --force-mmff-minimize 联用）",
    )
    parser.add_argument(
        "--vina-modes",
        type=str,
        default="dock,score_only,minimize",
        help="逗号分隔：dock / score_only / minimize（默认三者均运行）；或单独写 none/off/skip 关闭全部 Vina。关闭时 Prudent .pt 从 refined_candidates 注入 Prudent Vina 到 Vina_ScoreOnly。",
    )
    parser.add_argument(
        "--receptor_pdb",
        type=str,
        default=None,
        help="显式受体 PDB 路径（.pt 无 protein_filename 时必需或与 index 联用时覆盖受体；存在则覆盖前述解析结果）",
    )
    parser.add_argument(
        "--index_pkl",
        type=str,
        default=None,
        help="CrossDocked 的 index.pkl；与 data_id 联用，补全缺失的 ligand_filename / protein_filename（路径相对 --protein_root）",
    )
    parser.add_argument(
        "--data_id",
        type=int,
        default=None,
        help="强制指定口袋编号（覆盖 .pt 内 extra_info 与文件名推断），用于 index 查表或记录",
    )
    parser.add_argument(
        "--benchmark_ligands_csv",
        type=str,
        default=None,
        help="测试集列表 CSV（如 docs/100ligands.csv）：按 data_id 取 ligand_rel，与 test_set[data_id] 一致；"
        "pocket10 基准下勿用全量 index.pkl 的第 N 行代替测试编号 N",
    )
    parser.add_argument(
        "--reference_ligand",
        type=str,
        default=None,
        help="参考配体路径，相对 --protein_root（如 shoc2ligand.sdf）；用于自定义口袋 .pt 的对接盒参考、Tanimoto、分子命名；"
        "与 --receptor_pdb 联用时勿再依赖 100ligands.csv",
    )
    parser.add_argument(
        "--no-distribution-plots",
        action="store_true",
        help="不生成键长/原子对距离等聚合分布图（bond_length_hist.png、pair_dist_hist.png）",
    )
    parser.add_argument(
        "--no-excel",
        action="store_true",
        help="不写 evaluation_results_*.xlsx / complete_molecules_*.xlsx "
        "（Web 服务端默认；数据保留在 .pt/DB，可经后端按需导出）",
    )
    parser.add_argument(
        "--rdkit-structure-repair",
        action="store_true",
        help="重建后启用确定性 RDKit 结构修复（rdkit-structure-repair；默认关闭）",
    )
    parser.add_argument(
        "--rdkit-structure-repair-config",
        type=str,
        default="rdkit-structure-repair/configs/conservative.yaml",
        help="rdkit-structure-repair YAML 配置路径（相对 DiffDynamic 根目录）",
    )
    parser.add_argument(
        "--medchem-optimize",
        action="store_true",
        help="重建后、理化性质与 Vina 之前，启用激进 medchem 结构优化（默认关闭）。"
        "会改变分子式，开启后必须作为独立 ablation 臂汇报。",
    )
    parser.add_argument(
        "--medchem-optimize-config",
        type=str,
        default="rdkit-structure-repair/configs/medchem_optimize.yaml",
        help="medchem 优化 YAML 配置路径（相对 DiffDynamic 根目录）",
    )
    parser.add_argument(
        "--keep-preopt-sdf",
        choices=["none", "structure_only", "evaluated"],
        default="structure_only",
        help="优化前分子如何保留：none 不留；structure_only 另写 "
        "reconstructed_molecules_preopt/ 仅含 3D 结构（零额外成本）；"
        "evaluated 优化前分子也跑完整评估，Excel 出双份指标（成本翻倍，含 Vina）",
    )

    args = parser.parse_args()
    pass  # process state is held in state.py
    _state._RECONSTRUCTION_METHOD = args.reconstruction_method

    # 服务端批量链路（batch_sampleandeval_parallel 子进程）通过环境变量关闭 Excel 落盘
    if os.environ.get("DIFFDYNAMIC_NO_EXCEL", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        args.no_excel = True

    pass  # process state is held in state.py
    if args.rdkit_structure_repair:
        _state._RDKIT_STRUCTURE_REPAIR_CFG = {
            "enable": True,
            "config": args.rdkit_structure_repair_config,
            "on_reject": "keep_original",
        }
        print(f"RDKit structure repair: ON ({args.rdkit_structure_repair_config})")
    else:
        _state._RDKIT_STRUCTURE_REPAIR_CFG = None

    pass  # process state is held in state.py
    if args.medchem_optimize:
        _state._MEDCHEM_OPTIMIZE_CFG = {
            "enable": True,
            "config": args.medchem_optimize_config,
        }
        print(
            f"Medchem optimize: ON ({args.medchem_optimize_config}), "
            f"keep-preopt-sdf={args.keep_preopt_sdf}"
        )
        print("⚠️  生成分子不再是模型原始输出，请作为独立 ablation 臂汇报。")
    else:
        _state._MEDCHEM_OPTIMIZE_CFG = None

    try:
        vina_modes_cli = parse_vina_modes_arg(args.vina_modes)
    except ValueError as e:
        print(f"❌ {e}")
        return 1

    if args.vina_timeout_seconds is not None:
        os.environ["VINA_DOCK_TIMEOUT_SEC"] = str(args.vina_timeout_seconds)

    use_isolation = bool(args.enable_isolation)
    if args.disable_isolation:
        use_isolation = False
        os.environ["DISABLE_EVAL_ISOLATION"] = "1"

    # 验证输入
    pt_path = Path(args.pt_file)
    if not pt_path.exists():
        print(f"❌ 错误: .pt文件不存在: {pt_path}")
        return 1

    protein_root = Path(args.protein_root)
    if not protein_root.exists():
        print(f"❌ 错误: 蛋白根目录不存在: {protein_root}")
        return 1

    print(f"\n{'=' * 70}")
    print(f"使用正确的reconstruct方法评估分子")
    print(f"{'=' * 70}")
    print(f"输入文件: {pt_path}")
    print(f"蛋白根目录: {protein_root}")
    print(f"输出目录: {args.output_dir}")
    print(f"原子编码: {args.atom_mode}")
    _vts = os.environ.get("VINA_DOCK_TIMEOUT_SEC", "").strip()
    print(
        f"Vina 单次时限(秒): {_vts if _vts else '128(默认，可设 VINA_DOCK_TIMEOUT_SEC 或 --vina-timeout-seconds)'}"
    )
    print(f"{'=' * 70}\n")

    # 记录评估开始时间
    start_time = datetime.now()
    print(f"⏰ 评估开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 执行评估
    results = evaluate_pt_file(
        pt_path=pt_path,
        protein_root=args.protein_root,
        output_dir=args.output_dir,
        atom_mode=args.atom_mode,
        exhaustiveness=args.exhaustiveness,
        n_poses=args.n_poses,
        size_factor=args.size_factor,
        buffer=args.buffer,
        max_samples=args.max_samples,
        save_sdf=args.save_sdf,
        debug=args.debug,
        tmp_dir=args.tmp_dir,  # 传递临时目录参数
        use_isolation=use_isolation,
        save_intermediate_interval=args.save_intermediate_interval,
        force_pre_docking_mmff=True if args.force_mmff_minimize else None,
        pre_docking_mmff_max_iters_override=(
            args.mmff_max_iters
            if args.force_mmff_minimize and args.mmff_max_iters is not None
            else None
        ),
        receptor_pdb=args.receptor_pdb,
        index_pkl=args.index_pkl,
        data_id_override=args.data_id,
        benchmark_ligands_csv=args.benchmark_ligands_csv,
        reference_ligand_rel=args.reference_ligand,
        vina_modes=vina_modes_cli,
        save_distribution_plots=not args.no_distribution_plots,
        remove_fragments=args.remove_fragments,
        keep_preopt_sdf=args.keep_preopt_sdf,
        save_excel=not args.no_excel,
    )

    # 记录评估结束时间
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    print(f"\n⏰ 评估结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  总耗时: {elapsed_time:.1f} 秒 ({elapsed_time / 60:.1f} 分钟)\n")

    if results:
        print(f"\n✅ 评估成功完成！")

        _eval_out_for_history = results.get("eval_output_dir") or args.output_dir
        if (
            os.environ.get("DIFFDYNAMIC_SKIP_EVAL_RECORDS", "").lower()
            not in _TRUE_VALUES
        ):
            # Parallel scaffold chunks must not concurrently rewrite the same
            # sampling_history.xlsx/global log files.
            try:
                update_sampling_history(
                    pt_file=str(pt_path),
                    statistics=results.get("statistics", {}),
                    output_dir=_eval_out_for_history,
                    num_samples=results.get("num_samples"),
                    n_reconstruct_success=results.get("n_reconstruct_success"),
                    n_eval_success=results.get("n_eval_success"),
                )
            except Exception as e:
                print(f"⚠️  更新sampling_history.xlsx失败: {e}")

            try:
                record_evaluation_to_global_log(
                    pt_file=str(pt_path),
                    ligand_filename=results.get("ligand_filename", "N/A"),
                    protein_root=args.protein_root,
                    atom_mode=args.atom_mode,
                    exhaustiveness=args.exhaustiveness,
                    start_time=start_time,
                    end_time=end_time,
                    num_samples=results.get("num_samples", 0),
                    n_reconstruct_success=results.get("n_reconstruct_success", 0),
                    n_eval_success=results.get("n_eval_success", 0),
                    n_complete=results.get("n_complete", 0),
                    statistics=results.get("statistics", {}),
                    output_dir=_eval_out_for_history,
                    data_id=results.get("data_id"),
                )
            except Exception as e:
                print(f"⚠️  记录全局日志失败: {e}")

        return 0
    else:
        print(f"\n❌ 评估失败（可能是文件加载失败或数据验证失败）")
        print(f"   请检查:")
        print(f"   1. .pt文件是否完整且未损坏")
        print(
            f"   2. 字段: dict 需 pred_ligand_pos / pred_ligand_v / data；list 需各元素含 pred_pos / pred_v（DecompDiff）"
        )
        print(f"   3. 文件路径是否正确: {pt_path}")
        return 1
