"""Application responsibilities extracted from evaluate_pocket_quality.py."""

from diffdynamic.analysis.pocket.state import (
    DEFAULT_NUM_CPU_CORES,
    EVAL_RECORDS_CSV,
    Path,
    Pool,
    VIS_ROOT,
    argparse,
    cpu_count,
    datetime,
    sys,
)


def main():
    from diffdynamic.analysis.pocket.aggregation import append_evaluation_record
    from diffdynamic.analysis.pocket.batch import (
        _eval_single_pt_task,
        find_pt_files_for_range,
        run_batch_sampleandeval,
    )
    from diffdynamic.analysis.pocket.pipeline import (
        evaluate_pocket_quality,
        print_evaluation_report,
    )

    parser = argparse.ArgumentParser(
        description="蛋白质口袋质量评估（基于扩散模型生成，含可视化）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 直接评估 .pt 文件并生成可视化
  python evaluate_pocket_quality.py --pt_file outputs/result_5.pt --visualize

  # 指定可视化输出目录
  python evaluate_pocket_quality.py --pt_file outputs/result_5.pt --visualize --vis_dir ./my_vis

  # 批量评估并可视化
  python evaluate_pocket_quality.py --run_batch --start 1 --end 10 --gpus "0" --visualize

  # 单口袋（先生成再评估）
  python evaluate_pocket_quality.py --pocket_pdb data/pocket.pdb --visualize

  # 自定义口袋+配体：仅评估已有 SDF/MOL（不依赖 .pt；想法E 跳过）
  python evaluate_pocket_quality.py --eval_ligands my/poses.sdf --vina_outputs_dir outputs --visualize
  python evaluate_pocket_quality.py --eval_ligands shoc2/8v1tligand.sdf \\
    --custom_pocket_pdb shoc2/8v1t.pdb --vina_outputs_dir outputs --visualize

  # 想法 H：FPocket 蛋白口袋参与评分；配体合并 PDB 再跑 FPocket 仅报告
  python evaluate_pocket_quality.py --pt_file outputs/result_5.pt \\
    --fpocket_protein_pdb data/target.pdb --visualize
        """,
    )

    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--pocket_pdb", type=str, help="口袋 PDB 文件路径（将调用 batch 脚本生成）"
    )
    g.add_argument(
        "--pt_file", type=str, help="已有 .pt 文件路径（跳过生成，直接评估）"
    )
    g.add_argument(
        "--pt_dir",
        type=str,
        metavar="DIR",
        help="已有 .pt 所在目录：与 --start/--end 联用，按 result_{i}_*.pt 批量评估（不重新生成）；"
        "未指定 --vina_outputs_dir 时默认用该目录定位 eval_*",
    )
    g.add_argument(
        "--run_batch",
        action="store_true",
        help="运行 batch_sampleandeval_parallel 后评估指定范围",
    )
    g.add_argument(
        "--eval_ligands",
        type=str,
        metavar="PATH",
        help="自定义配体 .sdf/.mol 或目录（仅评估，不读 .pt；需 batch 已生成 eval_* 或指定 --vina_outputs_dir）",
    )

    parser.add_argument("--protein_root", type=str, default=None)
    parser.add_argument(
        "--ligand_path",
        type=str,
        default=None,
        help="与 --pocket_pdb 联用：参考配体 SDF，传给 batch_sampleandeval_parallel",
    )
    parser.add_argument(
        "--custom_pocket_pdb",
        type=str,
        default=None,
        help="与 --eval_ligands 联用：自定义口袋 PDB，用于推测 Vina outputs 位置（可选）",
    )
    parser.add_argument(
        "--vina_outputs_dir",
        type=str,
        default=None,
        help="含 eval_* 子目录的文件夹（一般为 batch 的 outputs）；自定义评估时建议显式指定",
    )
    parser.add_argument(
        "--vina_pocket_id",
        type=str,
        default=None,
        help="匹配 eval_{id}_* 的口袋 id（默认：custom 或由 --pt_file 文件名推断）",
    )

    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=99)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument(
        "--num_cpu_cores",
        type=int,
        default=None,
        help=f"CPU 核心数（默认: min({DEFAULT_NUM_CPU_CORES}, cpu_count())）",
    )
    parser.add_argument(
        "--cores_per_task",
        type=int,
        default=1,
        help="每任务 CPU 核心数，并行数 = num_cpu_cores // cores_per_task（默认: 1）",
    )

    parser.add_argument("--atom_mode", type=str, default="add_aromatic")
    parser.add_argument(
        "--weight_a",
        type=float,
        default=0.45,
        help="层内 A Aff+LE 权重（默认 0.45，ligand 层）",
    )
    parser.add_argument(
        "--weight_b", type=float, default=0.25, help="层内 B Mode 权重（默认 0.25）"
    )
    parser.add_argument(
        "--weight_c", type=float, default=0.30, help="层内 C Drug 权重（默认 0.30）"
    )
    parser.add_argument(
        "--weight_d",
        type=float,
        default=0.45,
        help="层内 D Unique 权重（默认 0.45，compat 层）",
    )
    parser.add_argument(
        "--weight_e", type=float, default=0.55, help="层内 E Complete 权重（默认 0.55）"
    )
    parser.add_argument(
        "--idea_e_expected_n_molecules",
        type=int,
        default=None,
        metavar="N",
        help=(
            "想法E：应生成分子数（正整数）。比例=完整分子数/N，完整分子=SMILES 不含 '.'；"
            "指定后不再用 .pt 里 pred_ligand_pos 长度作分母。"
            "仅 --eval_ligands 时也可凭此启用想法E"
        ),
    )
    parser.add_argument(
        "--weight_f",
        type=float,
        default=0.20,
        help="层内 F Chem 权重（默认 0.20，pocket 层）",
    )
    parser.add_argument(
        "--weight_g", type=float, default=0.55, help="层内 G Anchor 权重（默认 0.55）"
    )
    parser.add_argument(
        "--weight_h", type=float, default=0.25, help="层内 H Geom 权重（默认 0.25）"
    )
    parser.add_argument(
        "--fpocket_protein_pdb",
        type=str,
        default=None,
        help="蛋白 PDB（F/G/H）；未指定时从 .pt 的 protein_filename 自动解析；"
        "需已安装 fpocket（F/H）",
    )
    parser.add_argument(
        "--fpocket_cmd",
        type=str,
        default="fpocket",
        help="FPocket 可执行文件；也可用环境变量 FPOCKET_CMD（默认先找命令行再 PATH/conda）",
    )
    parser.add_argument(
        "--fpocket_pocket_index",
        type=int,
        default=1,
        help="采用 info.txt 中 Pocket N 的编号 N（默认 1，一般为打分最高位点）",
    )
    parser.add_argument(
        "--fpocket_max_ligand_models",
        type=int,
        default=50,
        help="配体侧 FPocket：合并 PDB 中最多写入的构象数（默认 50）",
    )
    parser.add_argument(
        "--fpocket_timeout",
        type=int,
        default=600,
        help="单次 fpocket 子进程超时秒数（默认 600）",
    )
    parser.add_argument(
        "--idea_h_fpocket_vol_min",
        type=float,
        default=None,
        help="配体 FPocket 失败时，蛋白体积回退评分：满分下限 Å³（默认 300）",
    )
    parser.add_argument(
        "--idea_h_fpocket_vol_max",
        type=float,
        default=None,
        help="配体 FPocket 失败时，蛋白体积回退评分：满分上限 Å³（默认 2200）",
    )
    parser.add_argument(
        "--idea_h_fpocket_vol_zero_below",
        type=float,
        default=None,
        help="蛋白体积回退评分：低于该值 Å³ 为 0 分（默认 80）",
    )
    parser.add_argument(
        "--idea_h_fpocket_vol_zero_above",
        type=float,
        default=None,
        help="蛋白体积回退评分：≥该值 Å³ 为 0 分（默认 4500）",
    )
    parser.add_argument(
        "--idea_h_ligand_path",
        type=str,
        default=None,
        metavar="PATH",
        help="与 --pt_file 联用：仅想法 H 配体侧 FPocket 使用该 .sdf/.mol 或目录（A–G 仍用 .pt）；"
        "也可与 --eval_ligands 联用以覆盖 H 的配体构象来源",
    )
    parser.add_argument(
        "--idea_b_atom_subset",
        type=str,
        default="hetero_heavy",
        choices=("all", "hetero_heavy", "edge_heavy", "hetero_edge", "combined"),
        help="想法B：DBSCAN 用的坐标子集（默认 hetero_heavy）。hetero_heavy=非H非C；"
        "edge_heavy=重邻居≤阈值的表面重原子；hetero_edge=二者交；"
        "combined=(1-w)*全原子+w*hetero_edge（见下）",
    )
    parser.add_argument(
        "--idea_b_edge_max_neighbors",
        type=int,
        default=2,
        help="edge_heavy / hetero_edge：非氢重邻居数上限（默认 2）",
    )
    parser.add_argument(
        "--idea_b_combined_weight",
        type=float,
        default=0.5,
        help="atom_subset=combined 时 hetero_edge 分项权重 w（默认 0.5）",
    )

    # 可视化参数
    parser.add_argument(
        "--visualize", action="store_true", help="生成所有可视化图像（需要 matplotlib）"
    )
    parser.add_argument(
        "--vis_dir",
        type=str,
        default=None,
        help="可视化根目录（默认: pocket_quality_vis/，其下自动创建 蛋白质编号_时间戳/）",
    )
    parser.add_argument(
        "--use_tsne",
        action="store_true",
        help="聚类图中使用 t-SNE 替代 PCA（数据量大时较慢）",
    )

    args = parser.parse_args()

    if (
        args.idea_e_expected_n_molecules is not None
        and args.idea_e_expected_n_molecules <= 0
    ):
        print("❌ --idea_e_expected_n_molecules 须为正整数", file=sys.stderr)
        sys.exit(2)

    if args.eval_ligands:
        el = Path(args.eval_ligands)
        if not el.exists():
            print(f"❌ 配体路径不存在: {el}")
            sys.exit(1)

        base_root = Path(args.vis_dir) if args.vis_dir else VIS_ROOT
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vis_dir = None
        if args.visualize:
            vis_dir = base_root / f"custom_ligands_{timestamp}"
            vis_dir.mkdir(parents=True, exist_ok=True)
            print(f"Visualization output -> {base_root}")

        cpp = (
            str(Path(args.custom_pocket_pdb).resolve())
            if args.custom_pocket_pdb
            else None
        )
        result = evaluate_pocket_quality(
            pt_path=None,
            ligand_path=str(el.resolve()),
            custom_pocket_pdb=cpp,
            vina_outputs_dir=args.vina_outputs_dir,
            vina_pocket_id=args.vina_pocket_id,
            protein_root=args.protein_root,
            data_id=args.vina_pocket_id or "custom",
            atom_mode=args.atom_mode,
            weight_a=args.weight_a,
            weight_b=args.weight_b,
            weight_c=args.weight_c,
            weight_d=args.weight_d,
            weight_e=args.weight_e,
            weight_f=args.weight_f,
            weight_g=args.weight_g,
            weight_h=args.weight_h,
            idea_b_atom_coord_subset=args.idea_b_atom_subset,
            idea_b_edge_max_heavy_neighbors=args.idea_b_edge_max_neighbors,
            idea_b_combined_focus_weight=args.idea_b_combined_weight,
            visualize=args.visualize,
            vis_dir=str(vis_dir) if vis_dir else None,
            use_tsne=args.use_tsne,
            fpocket_protein_pdb=args.fpocket_protein_pdb,
            fpocket_cmd=args.fpocket_cmd,
            fpocket_pocket_index=args.fpocket_pocket_index,
            fpocket_max_ligand_models=args.fpocket_max_ligand_models,
            fpocket_timeout=args.fpocket_timeout,
            fpocket_optimal_min=args.idea_h_fpocket_vol_min,
            fpocket_optimal_max=args.idea_h_fpocket_vol_max,
            fpocket_zero_below=args.idea_h_fpocket_vol_zero_below,
            fpocket_zero_above=args.idea_h_fpocket_vol_zero_above,
            idea_h_ligand_path=args.idea_h_ligand_path,
            idea_e_expected_n_molecules=args.idea_e_expected_n_molecules,
        )
        if result.get("error"):
            print(f"❌ {result['error']}")
            sys.exit(1)
        print_evaluation_report(result)
        record_path = base_root / EVAL_RECORDS_CSV
        append_evaluation_record(result, record_path, timestamp=timestamp)
        print(f"\nEvaluation records -> {record_path}")
        sys.exit(0)

    pt_files_to_eval = []

    if args.run_batch:
        ok, pt_files, msg = run_batch_sampleandeval(
            start=args.start,
            end=args.end,
            gpus=args.gpus,
            num_cpu_cores=args.num_cpu_cores,
            cores_per_task=args.cores_per_task,
            protein_root=args.protein_root,
        )
        if not ok:
            print(f"❌ {msg}")
            sys.exit(1)
        pt_files_to_eval = find_pt_files_for_range(args.start, args.end)
        if not pt_files_to_eval:
            pt_files_to_eval = pt_files

    elif args.pt_dir:
        pt_dir = Path(args.pt_dir).resolve()
        if not pt_dir.is_dir():
            print(f"❌ --pt_dir 不是目录或不存在: {pt_dir}")
            sys.exit(1)
        if args.vina_outputs_dir is None:
            args.vina_outputs_dir = str(pt_dir)
            print(f"--vina_outputs_dir 未指定，默认使用 --pt_dir: {pt_dir}")
        pt_files_to_eval = find_pt_files_for_range(
            args.start, args.end, output_dir=pt_dir
        )
        if not pt_files_to_eval:
            print(f"❌ 在 {pt_dir} 未找到 result_{{{args.start}..{args.end}}}_*.pt")
            sys.exit(1)
        print(
            f"从 --pt_dir 收集到 {len(pt_files_to_eval)} 个 .pt（data_id {args.start}..{args.end}）"
        )

    elif args.pocket_pdb:
        protein_path = Path(args.pocket_pdb)
        if not protein_path.exists():
            print(f"❌ 口袋文件不存在: {protein_path}")
            sys.exit(1)

        ok, pt_files, msg = run_batch_sampleandeval(
            start=0,
            end=0,
            gpus=args.gpus,
            num_cpu_cores=args.num_cpu_cores,
            cores_per_task=args.cores_per_task,
            protein_path=protein_path,
            ligand_path=Path(args.ligand_path) if args.ligand_path else None,
            protein_root=args.protein_root,
        )
        if not ok:
            print(f"❌ {msg}")
            sys.exit(1)
        pt_files_to_eval = pt_files

    else:
        pt_path = Path(args.pt_file)
        if not pt_path.exists():
            print(f"❌ .pt 文件不存在: {pt_path}")
            sys.exit(1)
        pt_files_to_eval = [str(pt_path)]
        # 单文件时若未指定 vina_outputs_dir，用 .pt 所在目录
        if args.vina_outputs_dir is None:
            args.vina_outputs_dir = str(pt_path.parent.resolve())

    if not pt_files_to_eval:
        print("❌ 未找到可评估的 .pt 文件")
        sys.exit(1)

    # CPU 并行配置（与 batch_sampleandeval_parallel 一致）
    num_cpu_cores = (
        args.num_cpu_cores
        if args.num_cpu_cores is not None
        else min(DEFAULT_NUM_CPU_CORES, cpu_count())
    )
    cores_per_task = max(1, args.cores_per_task)
    max_parallel = max(1, num_cpu_cores // cores_per_task)

    print(f"将评估 {len(pt_files_to_eval)} 个 .pt 文件")
    print(
        f"CPU 并行: {num_cpu_cores} 核, 每任务 {cores_per_task} 核, 最多 {min(max_parallel, len(pt_files_to_eval))} 个并行任务"
    )

    # 可视化：主文件夹 pocket_quality_vis/，子文件夹命名为 蛋白质编号_时间戳（如 3dzh_20260309_110357）
    base_root = Path(args.vis_dir) if args.vis_dir else VIS_ROOT
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 构建任务列表
    tasks = []
    for pt in pt_files_to_eval:
        stem = Path(pt).stem
        data_id = None
        if stem.startswith("result_"):
            parts = stem.split("_")
            if len(parts) >= 3:
                try:
                    data_id = int(parts[1])
                except ValueError:
                    data_id = parts[1]

        vis_dir = None
        if args.visualize:
            pocket_name = str(data_id) if data_id is not None else stem
            folder_name = f"{pocket_name}_{timestamp}"
            vis_dir = base_root / folder_name
            vis_dir.mkdir(parents=True, exist_ok=True)
            if len(tasks) == 0:
                print(f"Visualization output -> {base_root}")

        tasks.append(
            (
                pt,
                args.protein_root,
                data_id,
                args.atom_mode,
                args.weight_a,
                args.weight_b,
                args.weight_c,
                args.weight_d,
                args.weight_e,
                args.weight_f,
                args.weight_g,
                args.weight_h,
                args.visualize,
                vis_dir,
                args.use_tsne,
                args.idea_b_atom_subset,
                args.idea_b_edge_max_neighbors,
                args.idea_b_combined_weight,
                args.vina_outputs_dir,
                args.vina_pocket_id,
                args.custom_pocket_pdb,
                args.fpocket_protein_pdb,
                args.fpocket_cmd,
                args.fpocket_pocket_index,
                args.fpocket_max_ligand_models,
                args.fpocket_timeout,
                args.idea_h_fpocket_vol_min,
                args.idea_h_fpocket_vol_max,
                args.idea_h_fpocket_vol_zero_below,
                args.idea_h_fpocket_vol_zero_above,
                args.idea_h_ligand_path,
                args.idea_e_expected_n_molecules,
            )
        )

    # 并行或串行执行
    if len(tasks) > 1 and max_parallel > 1:
        n_workers = min(max_parallel, len(tasks))
        try:
            with Pool(processes=n_workers) as pool:
                results = pool.map(_eval_single_pt_task, tasks)
        except KeyboardInterrupt:
            print("\n⚠️  收到中断信号，已停止")
            sys.exit(1)
    else:
        results = [_eval_single_pt_task(t) for t in tasks]

    for result in results:
        try:
            print_evaluation_report(result)
        except Exception as exc:
            print(f"⚠️  print_evaluation_report failed: {exc}", flush=True)
        record_path = base_root / EVAL_RECORDS_CSV
        try:
            append_evaluation_record(result, record_path, timestamp=timestamp)
        except PermissionError as e:
            print(f"⚠️  未写入 {record_path}: {e}", flush=True)
    if results:
        print(f"\nEvaluation records -> {base_root / EVAL_RECORDS_CSV}")
