"""Application responsibilities extracted from batch_sampleandeval_parallel.py."""

from diffdynamic.batch.state import (
    CONFIG,
    DEFAULT_GPU_IDS,
    DEFAULT_NUM_CPU_CORES,
    EVAL_SCRIPT,
    JSD_EVAL_SCRIPT,
    Manager,
    OUTPUT_DIR,
    Path,
    Pool,
    REPO_ROOT,
    SCRIPT,
    argparse,
    datetime,
    os,
    pd,
    subprocess,
    sys,
    time,
    timedelta,
    timezone,
    torch,
    yaml,
)
from diffdynamic.batch import state as _state


def main():
    from diffdynamic.batch.benchmark import (
        _resolve_protein_root_for_eval_fallback,
        collect_benchmark_pocket_and_eval,
    )
    from diffdynamic.batch.configuration import (
        generate_config_params_string,
        medchem_opts_from_args,
        resolve_eval_vina_modes,
    )
    from diffdynamic.batch.evaluation import (
        process_evaluation_task,
        run_single_evaluation,
    )
    from diffdynamic.batch.reports import (
        collect_all_evaluation_results,
        save_molecules_to_excel,
    )
    from diffdynamic.batch.resources import (
        cleanup_zombie_processes,
        get_available_gpus,
        parse_gpu_ids,
    )
    from diffdynamic.batch.results import (
        extract_data_id_from_pt_filename,
        find_latest_result_file,
        normalize_batch_summary_path,
        resolve_eval_only_excel_path,
    )
    from diffdynamic.batch.sampling import (
        process_baseline_refine_standalone_task,
        process_sampling_task,
        run_single_sample_custom,
    )
    from .arguments import build_parser

    args = build_parser().parse_args()

    if args.config is None:
        args.config = CONFIG
    else:
        args.config = Path(args.config)

    if getattr(args, "collect_benchmark_pocket", None) is not None:
        pid = int(args.collect_benchmark_pocket)
        if pid < 0:
            print("❌ 错误: --collect-benchmark-pocket 须为非负整数")
            sys.exit(1)
        pr = _resolve_protein_root_for_eval_fallback(
            getattr(args, "protein_root", None)
        )
        if pr is None:
            print(
                "❌ 错误: 无法解析蛋白质数据目录。请指定 --protein_root 为 CrossDocked 口袋数据根目录 "
                "（例如 /workspace/data/crossdocked_v1.1_rmsd1.0_pocket10）。"
            )
            sys.exit(1)
        if getattr(args, "molecular_repair", "none") == "targetdiff_baseline_refine":
            print(
                "⚠️  提示: --collect-benchmark-pocket 当前不对各基线 .pt 做 targetdiff_baseline_refine，"
                "将按原 .pt 直接评估（与 --molecular-repair none 等效）。"
            )
        num_cpu_collect = (
            args.num_cpu_cores
            if args.num_cpu_cores is not None
            else DEFAULT_NUM_CPU_CORES
        )
        cpt_collect = getattr(args, "cores_per_task", 1) or 1

        # 解析方法列表
        include_methods = None
        if args.methods:
            include_methods = [m.strip() for m in args.methods.split(",") if m.strip()]

        collect_benchmark_pocket_and_eval(
            pid,
            pr,
            docs_parent=REPO_ROOT / "docs",
            atom_mode=args.atom_mode,
            exhaustiveness=args.exhaustiveness,
            num_cpu_cores=num_cpu_collect,
            cores_per_task=cpt_collect,
            eval_show_output=getattr(args, "eval_show_output", False),
            eval_save_log=getattr(args, "eval_save_log", False),
            force_mmff_minimize=getattr(args, "molecular_repair", "none") == "mmff",
            mmff_max_iters=args.mmff_max_iters
            if getattr(args, "molecular_repair", "none") == "mmff"
            else None,
            include_methods=include_methods,
        )
        sys.exit(0)

    # 自定义文件模式：提前设置 protein_root（在验证前）
    use_custom_mode = args.protein_path is not None
    use_eval_only_mode = args.pt_file is not None or args.pt_dir is not None
    if use_custom_mode:
        args.protein_path = Path(args.protein_path)
        if not args.protein_path.exists():
            print(f"❌ 错误: 蛋白文件不存在: {args.protein_path}")
            sys.exit(1)
        if args.ligand_path:
            args.ligand_path = Path(args.ligand_path)
            if not args.ligand_path.exists():
                print(f"❌ 错误: 配体文件不存在: {args.ligand_path}")
                sys.exit(1)
        if args.protein_root is None:
            args.protein_root = args.protein_path.parent
            print(
                f"💡 自定义模式：未指定 --protein_root，使用蛋白所在目录: {args.protein_root}"
            )

    # 单独评估模式与自定义模式互斥
    if use_eval_only_mode and use_custom_mode:
        print(f"❌ 错误: --pt_file 与 --protein_path 不能同时指定")
        sys.exit(1)

    # 只在非只生成模式下要求protein_root（采样+评估、自定义模式、单独评估模式均需要）
    if not args.sample_only or use_eval_only_mode:
        if args.protein_root is None:
            print(f"❌ 错误: 未指定蛋白质数据根目录（--protein_root）")
            print(f"   请使用 --protein_root 参数指定蛋白质数据目录")
            sys.exit(1)

        args.protein_root = Path(args.protein_root)

        if not args.protein_root.exists():
            # 单独评估模式：尝试常见数据目录作为回退（custom .pt 的蛋白路径可能已嵌入文件）
            if use_eval_only_mode:
                orig_path = args.protein_root
                fallback_paths = [
                    REPO_ROOT / "data" / "crossdocked_v1.1_rmsd1.0_pocket10",
                    REPO_ROOT / "data" / "crossdocked_v1.1_rmsd1.0",
                    Path("/workspace/data/crossdocked_v1.1_rmsd1.0_pocket10"),
                    Path("/workspace/data"),
                    REPO_ROOT,
                ]
                for fb in fallback_paths:
                    if fb.exists():
                        args.protein_root = fb
                        print(
                            f"💡 单独评估模式：{orig_path} 不存在，使用回退路径: {args.protein_root}"
                        )
                        break
                else:
                    args.protein_root = (
                        REPO_ROOT  # 最后回退到项目根目录（custom .pt 的蛋白可能已嵌入）
                    )
                    print(
                        f"💡 单独评估模式：{orig_path} 不存在，使用项目根目录: {args.protein_root}"
                    )
                    print(
                        f"   （custom .pt 的蛋白路径通常已嵌入文件，评估脚本会优先使用）"
                    )
            else:
                print(f"❌ 错误: 蛋白质数据根目录不存在: {args.protein_root}")
                sys.exit(1)

        if not EVAL_SCRIPT.exists():
            print(f"❌ 错误: 评估脚本不存在: {EVAL_SCRIPT}")
            sys.exit(1)
    elif args.protein_root:
        # 如果只生成模式下提供了protein_root，也验证一下
        args.protein_root = Path(args.protein_root)

    if not use_eval_only_mode and not args.config.exists():
        print(f"❌ 错误: 配置文件不存在: {args.config}")
        sys.exit(1)

    if not use_eval_only_mode and not SCRIPT.exists():
        print(f"❌ 错误: 采样脚本不存在: {SCRIPT}")
        sys.exit(1)

    # 解析 data_id 列表（--data_ids 优先于 --start/--end）
    if args.data_ids and not use_custom_mode:
        try:
            data_id_list = [
                int(x.strip()) for x in args.data_ids.split(",") if x.strip()
            ]
            if not data_id_list:
                print(
                    f"❌ 错误: --data_ids 格式无效，请提供逗号分隔的整数列表（如 5,10,20）"
                )
                sys.exit(1)
        except ValueError:
            print(
                f"❌ 错误: --data_ids 格式无效，请提供逗号分隔的整数列表（如 5,10,20）"
            )
            sys.exit(1)
    else:
        data_id_list = (
            list(range(args.start, args.end + 1)) if not use_custom_mode else []
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 自动清理残留进程（默认关闭，需显式启用 --auto-cleanup）
    if args.auto_cleanup:
        # --auto-cleanup: 自动清理残留进程（不询问，直接清理）
        interactive = False  # 不询问，直接清理
        cleaned_count = cleanup_zombie_processes(interactive=interactive)
        if cleaned_count > 0:
            print(f"✓ 已清理 {cleaned_count} 个残留进程，等待资源释放...")
            time.sleep(2)  # 等待资源释放
            print()
    else:
        # 默认不清理，仅打印提示
        print("提示: 检测到 --auto-cleanup 未启用，跳过残留进程清理")
        print("      如需清理残留进程，请添加 --auto-cleanup 参数")
        print()

    # 解析GPU配置
    if args.gpus:
        try:
            gpu_ids = parse_gpu_ids(args.gpus)
            if not gpu_ids:
                # 如果指定了"all"但检测不到GPU，尝试使用nvidia-smi检测
                if args.gpus.lower() == "all":
                    try:
                        result = subprocess.run(
                            ["nvidia-smi", "--list-gpus"],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        if result.returncode == 0 and result.stdout.strip():
                            num_gpus = len(result.stdout.strip().split("\n"))
                            gpu_ids = list(range(num_gpus))
                            print(
                                f"⚠️  警告: PyTorch检测不到CUDA，但nvidia-smi检测到 {num_gpus} 个GPU"
                            )
                            print(f"   将使用GPU IDs: {gpu_ids}")
                            print(
                                f"   子进程在设置CUDA_VISIBLE_DEVICES后可能会检测到CUDA"
                            )
                        else:
                            print(
                                f"⚠️  警告: 未找到可用的GPU（PyTorch和nvidia-smi都检测不到）"
                            )
                            print(f"   将继续运行，但任务可能会失败")
                            print(f"   如果所有任务都失败，请检查Docker容器的GPU配置")
                            # 使用默认GPU列表，让程序继续运行
                            gpu_ids = DEFAULT_GPU_IDS
                    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                        print(f"⚠️  警告: 未找到可用的GPU")
                        print(f"   将继续运行，但任务可能会失败")
                        print(f"   如果所有任务都失败，请检查Docker容器的GPU配置")
                        # 使用默认GPU列表，让程序继续运行
                        gpu_ids = DEFAULT_GPU_IDS
                else:
                    print(f"❌ 错误: 未找到可用的GPU")
                    print(f"   指定的GPU: {args.gpus}")
                    sys.exit(1)
        except ValueError as e:
            print(f"❌ 错误: GPU ID格式无效: {e}")
            print(f"   支持的格式: '0,1,2,3', '0-3', '0,2-4,6', 'all'")
            sys.exit(1)
    else:
        gpu_ids = DEFAULT_GPU_IDS

    # 验证GPU是否可用
    # 注意：主进程可能检测不到CUDA（因为环境变量或驱动问题），
    # 但子进程在设置CUDA_VISIBLE_DEVICES后可能可以检测到
    # 所以这里只做警告，不阻止运行
    cuda_available_in_main = False
    if torch is not None and torch.cuda.is_available():
        cuda_available_in_main = True
        available_gpus = get_available_gpus()
        invalid_gpus = [gpu_id for gpu_id in gpu_ids if gpu_id not in available_gpus]
        if invalid_gpus:
            print(f"⚠️  警告: 以下GPU在主进程中不可用: {invalid_gpus}")
            print(f"   主进程检测到的可用GPU: {available_gpus}")
            print(f"   将继续尝试使用指定的GPU（子进程可能会检测到）")
    else:
        # 尝试使用 nvidia-smi 作为备用检测方法
        try:
            result = subprocess.run(
                ["nvidia-smi", "--list-gpus"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                num_gpus_detected = len(result.stdout.strip().split("\n"))
                print(
                    f"⚠️  警告: PyTorch检测不到CUDA，但nvidia-smi检测到 {num_gpus_detected} 个GPU"
                )
                print(
                    f"   将继续运行，子进程在设置CUDA_VISIBLE_DEVICES后可能会检测到CUDA"
                )
                print(f"   如果所有任务都失败，请检查:")
                print(f"   1. Docker容器是否正确配置了GPU支持 (--gpus all)")
                print(f"   2. PyTorch是否正确安装了CUDA版本")
                print(f"   3. CUDA运行时库是否正确安装")
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            print("⚠️  警告: CUDA不可用（PyTorch和nvidia-smi都检测不到）")
            print("   将继续运行，但任务可能会失败")
            print("   如果所有任务都失败，请检查:")
            print("   1. Docker容器是否正确配置了GPU支持 (--gpus all)")
            print("   2. NVIDIA驱动是否正确安装")
            print("   3. PyTorch是否正确安装了CUDA版本")

    num_gpus = len(gpu_ids)
    num_cpu_cores = (
        args.num_cpu_cores if args.num_cpu_cores is not None else DEFAULT_NUM_CPU_CORES
    )

    # 确保batchsummary目录存在
    BATCHSUMMARY_DIR = REPO_ROOT / "batchsummary"
    BATCHSUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    batch_start_time = time.time()

    # ========== 单独评估 .pt 文件模式 ==========
    if use_eval_only_mode:
        pt_files = []

        # 1. 处理 --pt_dir 目录扫描
        if args.pt_dir is not None:
            pt_dir = Path(args.pt_dir)
            if not pt_dir.is_absolute():
                pt_dir = (REPO_ROOT / pt_dir).resolve()
            if not pt_dir.exists():
                print(f"❌ 错误: 目录不存在: {pt_dir}")
                sys.exit(1)
            if not pt_dir.is_dir():
                print(f"❌ 错误: 指定路径不是目录: {pt_dir}")
                sys.exit(1)

            # 扫描目录顶层，按 glob 匹配 .pt（默认 result_*.pt）
            _glob_pat = (args.pt_dir_glob or "result_*.pt").strip() or "result_*.pt"
            pt_files_from_dir = list(pt_dir.glob(_glob_pat))
            if not pt_files_from_dir:
                print(f"⚠️ 警告: 目录 {pt_dir} 中未找到匹配 {_glob_pat!r} 的文件")
            else:
                pt_files.extend(sorted(pt_files_from_dir))
                print(f"📁 从目录扫描到 {len(pt_files_from_dir)} 个 .pt 文件")

        # 2. 处理 --pt_file 文件列表
        if args.pt_file is not None:
            pt_paths_raw = [p.strip() for p in args.pt_file.split(",") if p.strip()]
            for p in pt_paths_raw:
                path = Path(p)
                if not path.is_absolute():
                    path = (REPO_ROOT / p).resolve()
                if not path.exists():
                    print(f"❌ 错误: .pt 文件不存在: {path}")
                    sys.exit(1)
                pt_files.append(path)

        # 去重并验证
        pt_files = list(dict.fromkeys(pt_files))  # 保持顺序去重
        if not pt_files:
            print("❌ 错误: 未找到任何有效的 .pt 文件用于评估")
            sys.exit(1)

        if args.molecular_repair == "targetdiff_baseline_refine":
            cfg_eval = Path(args.config) if args.config else CONFIG
            if not cfg_eval.is_file():
                print(
                    f"错误: targetdiff_baseline_refine 需要有效的 --config: {cfg_eval}"
                )
                sys.exit(1)
            refine_gpu = gpu_ids[0]
            print(
                f"\n单独评估：对 {len(pt_files)} 个 .pt 执行 targetdiff_baseline_refine "
                f"（GPU {refine_gpu}，与 sampling.yml 中 sample.targetdiff_baseline_refine 一致）\n"
            )
            _rpool = None
            try:
                _rpool = Pool(processes=1)
                _ref_tasks = [
                    (str(p), str(cfg_eval), str(args.protein_root), refine_gpu)
                    for p in pt_files
                ]
                _refined_paths = _rpool.map(
                    process_baseline_refine_standalone_task, _ref_tasks
                )
                pt_files = [Path(x) for x in _refined_paths]
            finally:
                if _rpool:
                    _rpool.close()
                    _rpool.join()

        cst = timezone(timedelta(hours=8))
        timestamp = datetime.now(cst).strftime("%Y%m%d_%H%M%S")
        config_params = (
            generate_config_params_string(args.config) if args.config else ""
        )
        default_excel = (
            BATCHSUMMARY_DIR
            / f"batch_evaluation_summary_{timestamp}_{config_params}.csv"
            if config_params
            else BATCHSUMMARY_DIR / f"batch_evaluation_summary_{timestamp}.csv"
        )
        excel_file = resolve_eval_only_excel_path(
            args.excel_file, BATCHSUMMARY_DIR, default_excel
        )
        if args.excel_file and not Path(args.excel_file).is_absolute():
            print(f"📊 批次汇总相对路径已解析到 batchsummary: {excel_file}")

        print(f"\n{'=' * 60}")
        print(f"单独评估模式：仅评估 .pt 文件（不采样）")
        print(f"{'=' * 60}")
        print(f".pt 文件数: {len(pt_files)}")
        for pf in pt_files:
            print(f"  - {pf.name}")
        print(f"蛋白质数据根目录: {args.protein_root}")
        print(f"原子模式: {args.atom_mode}")
        print(f"对接强度: {args.exhaustiveness}")
        print(f"分子修复: {args.molecular_repair}")
        print(f"{'=' * 60}\n")

        cores_per_task = getattr(args, "cores_per_task", 1)
        _eval_force_mmff = args.molecular_repair == "mmff"
        _eval_mmff_iters = args.mmff_max_iters if _eval_force_mmff else None
        max_parallel = max(1, num_cpu_cores // max(1, cores_per_task))
        eval_workers = min(max_parallel, len(pt_files))
        print(
            f"并行评估: {eval_workers} 个进程（CPU 核心 {num_cpu_cores}，每任务 {cores_per_task} 核）\n"
        )
        print(
            f"评估 Vina：--eval-vina-modes={args.eval_vina_modes!r} "
            f"（auto 时对每个 .pt 单独判断是否为 Prudent）\n"
        )
        if getattr(args, "eval_show_output", False):
            print(
                "提示: 已启用 --eval-show-output，子进程输出直接打印（多任务并行时可能交错）。"
            )
        if getattr(args, "eval_save_log", False):
            print(
                "提示: 已启用 --eval-save-log，完整日志写入各 eval_* 目录下的 evaluate_subprocess.log。"
            )
        if not getattr(args, "eval_show_output", False) and not getattr(
            args, "eval_save_log", False
        ):
            print(
                "提示: 默认仅打印 [评估] 起止行；要看 evaluate 详细过程请加 --eval-save-log或 --eval-show-output。\n"
            )

        manager_eval = Manager()
        excel_lock_eval = manager_eval.Lock()
        _eval_medchem_opts = medchem_opts_from_args(args)
        if _eval_medchem_opts:
            print(
                "提示: 已启用 medchem 结构优化 → 生成分子不再是模型原始输出，"
                "请作为独立 ablation 臂汇报。\n"
            )
        evaluation_tasks = []
        for pt_path in pt_files:
            data_id = extract_data_id_from_pt_filename(pt_path)
            _vm_pt = resolve_eval_vina_modes(
                args.eval_vina_modes,
                config_path=args.config,
                pt_path=pt_path,
            )
            evaluation_tasks.append(
                (
                    data_id,
                    str(pt_path),
                    str(args.protein_root),
                    args.atom_mode,
                    args.exhaustiveness,
                    str(excel_file),
                    batch_start_time,
                    excel_lock_eval,
                    cores_per_task,
                    args.save_intermediate_interval,
                    getattr(args, "eval_show_output", False),
                    getattr(args, "eval_save_log", False),
                    _eval_force_mmff,
                    _eval_mmff_iters,
                    _vm_pt,
                    _eval_medchem_opts,
                )
            )

        all_results = []
        if evaluation_tasks:
            eval_pool = None
            try:
                eval_pool = Pool(processes=eval_workers)
                all_results = eval_pool.map(process_evaluation_task, evaluation_tasks)
            finally:
                if eval_pool:
                    eval_pool.close()
                    eval_pool.join()

        molform_out_parents = list({Path(p).resolve().parent for p in pt_files})
        molecule_records, summary_stats, pocket_stats_list = (
            collect_all_evaluation_results(
                all_results, batch_start_time, extra_eval_roots=molform_out_parents
            )
        )

        if excel_file and pd is not None:
            save_molecules_to_excel(
                excel_file,
                molecule_records,
                summary_stats,
                batch_start_time,
                pocket_stats_list,
                config_file=str(args.config),
            )
            print(f"✅ 结果已保存到: {excel_file}")

        print(f"\n{'=' * 60}")
        print(f"单独评估模式处理完成")
        print(f"{'=' * 60}")
        sys.exit(0)

    # ========== 自定义文件模式：单次采样+评估 ==========
    if use_custom_mode:
        pocket_id = "custom"
        gpu_id = gpu_ids[0]

        cst = timezone(timedelta(hours=8))
        timestamp = datetime.now(cst).strftime("%Y%m%d_%H%M%S")
        config_params = generate_config_params_string(args.config)
        excel_file = (
            normalize_batch_summary_path(Path(args.excel_file))
            if args.excel_file
            else (
                BATCHSUMMARY_DIR
                / f"batch_evaluation_summary_{timestamp}_{config_params}.csv"
                if config_params
                else BATCHSUMMARY_DIR / f"batch_evaluation_summary_{timestamp}.csv"
            )
        )

        print(f"\n{'=' * 60}")
        print(f"自定义文件模式：单次采样+评估")
        print(f"{'=' * 60}")
        # 检测是否已进入优化阶段
        try:
            config_path = Path(args.config) if args.config else CONFIG
            if config_path.exists() and yaml is not None:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                if cfg and cfg.get("sample", {}).get("optimization", {}).get(
                    "enable", False
                ):
                    after_dyn = (
                        cfg.get("sample", {})
                        .get("optimization", {})
                        .get("after_dynamic", False)
                    )
                    print(
                        f"✅ 已进入优化阶段 (optimization.enable=true)"
                        + (" [dynamic后再优化]" if after_dyn else "")
                    )
        except Exception:
            pass
        print(f"蛋白文件: {args.protein_path}")
        print(f"配体文件: {args.ligand_path or '无（仅蛋白）'}")
        print(
            f"口袋裁剪半径: {args.pocket_radius}Å"
            + (" (需提供配体)" if not args.ligand_path else "")
        )
        print(f"使用GPU: {gpu_id}")
        print(f"{'=' * 60}\n")

        if args.skip_existing:
            pt_file = find_latest_result_file(pocket_id)
            if pt_file and pt_file.exists():
                print(f"⏭️  跳过已存在的文件: {pt_file}")
                sample_success, pt_file_str, sample_msg = (
                    True,
                    str(pt_file),
                    "文件已存在",
                )
            else:
                sample_success, pt_file_str, sample_msg = run_single_sample_custom(
                    args.protein_path,
                    args.config,
                    gpu_id,
                    ligand_path=args.ligand_path,
                    use_dataset_for_pocket=args.use_dataset_for_pocket,
                    pocket_radius=getattr(args, "pocket_radius", 10.0),
                )
        else:
            sample_success, pt_file_str, sample_msg = run_single_sample_custom(
                args.protein_path,
                args.config,
                gpu_id,
                ligand_path=args.ligand_path,
                use_dataset_for_pocket=args.use_dataset_for_pocket,
                pocket_radius=getattr(args, "pocket_radius", 10.0),
            )

        if not sample_success or pt_file_str is None:
            print(f"采样失败: {sample_msg}")
            sys.exit(1)

        if args.molecular_repair == "targetdiff_baseline_refine":
            _crp = None
            try:
                _crp = Pool(processes=1)
                pt_file_str = _crp.apply(
                    process_baseline_refine_standalone_task,
                    ((pt_file_str, str(args.config), str(args.protein_root), gpu_id),),
                )
            finally:
                if _crp:
                    _crp.close()
                    _crp.join()

        if args.sample_only:
            print(f"\n自定义模式采样完成: {pt_file_str}")
            sys.exit(0)

        cores_per_task = getattr(args, "cores_per_task", 1)
        _vm_custom = resolve_eval_vina_modes(
            getattr(args, "eval_vina_modes", "auto"),
            config_path=args.config,
            pt_path=Path(pt_file_str),
        )
        eval_success, eval_msg, eval_output_dir = run_single_evaluation(
            pt_file_str,
            args.protein_root,
            pocket_id,
            args.atom_mode,
            args.exhaustiveness,
            cores_per_task=cores_per_task,
            save_intermediate_interval=args.save_intermediate_interval,
            eval_show_output=getattr(args, "eval_show_output", False),
            eval_save_subprocess_log=getattr(args, "eval_save_log", False),
            force_mmff_minimize=args.molecular_repair == "mmff",
            mmff_max_iters=args.mmff_max_iters
            if args.molecular_repair == "mmff"
            else None,
            vina_modes=_vm_custom,
            **medchem_opts_from_args(args),
        )
        all_results = [
            (pocket_id, sample_success, sample_msg, None, pt_file_str, eval_output_dir)
        ]
        molecule_records, summary_stats, pocket_stats_list = (
            collect_all_evaluation_results(all_results, batch_start_time)
        )

        if excel_file and pd is not None:
            save_molecules_to_excel(
                excel_file,
                molecule_records,
                summary_stats,
                batch_start_time,
                pocket_stats_list,
                config_file=str(args.config),
            )
            print(f"✅ 结果已保存到: {excel_file}")

        print(f"\n{'=' * 60}")
        print(f"自定义模式处理完成")
        print(f"{'=' * 60}")
        sys.exit(0)
    # 使用本地时区（CST，UTC+8）生成时间戳
    cst = timezone(timedelta(hours=8))
    timestamp = datetime.now(cst).strftime("%Y%m%d_%H%M%S")

    # 生成配置参数字符串
    config_params = generate_config_params_string(args.config)

    if args.excel_file:
        excel_file = normalize_batch_summary_path(Path(args.excel_file))
    else:
        # 原有文件名格式
        base_filename = f"batch_evaluation_summary_{timestamp}"
        # 如果成功提取了配置参数，则添加到文件名中
        if config_params:
            excel_file = BATCHSUMMARY_DIR / f"{base_filename}_{config_params}.csv"
        else:
            excel_file = BATCHSUMMARY_DIR / f"{base_filename}.csv"

    # 打印配置信息
    print(f"\n{'=' * 60}")
    if args.sample_only:
        print(f"批量采样配置（只生成模式）")
    else:
        print(f"批量采样和评估配置（并行版本）")
    print(f"{'=' * 60}")
    print(f"数据ID: {data_id_list}")
    print(f"并行配置:")
    print(f"  - GPU数量: {num_gpus} (GPU IDs: {gpu_ids})")
    if not args.sample_only:
        cores_per_task = getattr(args, "cores_per_task", 1)
        print(f"  - CPU核心数: {num_cpu_cores}")
        print(
            f"  - 每任务核心数: {cores_per_task} (并行任务数={num_cpu_cores // max(1, cores_per_task)})"
        )
    print(f"配置文件: {args.config}")
    if not args.sample_only:
        print(f"蛋白质数据根目录: {args.protein_root}")
        print(f"评估结果保存位置: {OUTPUT_DIR}")
        print(f"原子模式: {args.atom_mode}")
        print(f"对接强度: {args.exhaustiveness}")
    else:
        print(f"输出目录: {OUTPUT_DIR}")
    print(f"跳过已存在: {args.skip_existing}")
    print(f"分子修复: {args.molecular_repair}")
    if not args.sample_only:
        print(f"批次汇总文件: {excel_file}")
    print(f"{'=' * 60}\n")

    # 创建Manager用于进程间共享的锁
    manager = Manager()
    excel_lock = manager.Lock()

    total = len(data_id_list)
    start_time = time.time()

    # ========== 第一阶段：采样（4个GPU并行） ==========
    print(f"\n{'=' * 60}")
    print(f"第一阶段：并行采样")
    print(f"{'=' * 60}")
    print(f"数据ID: {data_id_list} (共 {total} 个)")
    print(f"使用 {num_gpus} 个GPU并行采样 (GPU IDs: {gpu_ids})")
    print(f"GPU列表详情: {gpu_ids}")
    print(f"{'=' * 60}\n")

    # 准备采样任务列表
    # 按GPU分组任务，确保每个GPU的任务均匀分布
    tasks_by_gpu = {gpu_id: [] for gpu_id in gpu_ids}
    for i, data_id in enumerate(data_id_list):
        # 轮询分配GPU（确保正确轮询）
        gpu_id = gpu_ids[i % num_gpus]
        tasks_by_gpu[gpu_id].append(
            (
                data_id,
                gpu_id,
                str(args.config),
                args.skip_existing,
                str(args.protein_root) if getattr(args, "protein_root", None) else "",
                getattr(args, "molecular_repair", "none") or "none",
            )
        )

    # 交错排列任务，确保所有GPU的任务均匀分布
    # 这样pool.map()处理时，前几个任务会分配给不同的GPU
    sampling_tasks = []
    max_tasks_per_gpu = max(len(tasks) for tasks in tasks_by_gpu.values())
    for i in range(max_tasks_per_gpu):
        for gpu_id in gpu_ids:
            if i < len(tasks_by_gpu[gpu_id]):
                sampling_tasks.append(tasks_by_gpu[gpu_id][i])

    # 验证GPU分配（调试信息）
    gpu_distribution = {}
    for data_id, gpu_id, *_ in sampling_tasks:
        if gpu_id not in gpu_distribution:
            gpu_distribution[gpu_id] = []
        gpu_distribution[gpu_id].append(data_id)

    print(f"GPU分配验证:")
    for gpu_id in sorted(gpu_distribution.keys()):
        count = len(gpu_distribution[gpu_id])
        print(
            f"  GPU {gpu_id}: {count} 个任务 (data_ids: {gpu_distribution[gpu_id][:10]}{'...' if count > 10 else ''})"
        )
    print(f"任务列表前10个: {[(t[0], t[1]) for t in sampling_tasks[:10]]}")

    # 验证前N个任务是否分配给所有GPU（N=GPU数量）
    if len(sampling_tasks) >= num_gpus:
        first_n_tasks = [(t[0], t[1]) for t in sampling_tasks[:num_gpus]]
        first_n_gpus = [gpu_id for _, gpu_id in first_n_tasks]
        unique_gpus_in_first_n = set(first_n_gpus)
        print(f"前{num_gpus}个任务的GPU分配: {first_n_gpus}")
        print(
            f"前{num_gpus}个任务覆盖的GPU: {sorted(unique_gpus_in_first_n)} (期望: {sorted(gpu_ids)})"
        )
        if len(unique_gpus_in_first_n) < num_gpus:
            print(f"⚠️  警告: 前{num_gpus}个任务没有覆盖所有GPU！")
            missing_gpus = set(gpu_ids) - unique_gpus_in_first_n
            print(f"   缺失的GPU: {sorted(missing_gpus)}")
    print()

    # 检测是否已进入优化阶段（config 中 optimization.enable=true）
    try:
        config_path = Path(args.config) if args.config else CONFIG
        if config_path.exists() and yaml is not None:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if cfg and cfg.get("sample", {}).get("optimization", {}).get(
                "enable", False
            ):
                after_dyn = (
                    cfg.get("sample", {})
                    .get("optimization", {})
                    .get("after_dynamic", False)
                )
                print(
                    f"✅ 已进入优化阶段 (optimization.enable=true)"
                    + (" [dynamic后再优化]" if after_dyn else "")
                )
                print()
    except Exception:
        pass

    sampling_start_time = time.time()

    # 采样阶段：限制为GPU数量的进程（每个GPU一个）
    # 确保进程数等于GPU数量，以便充分利用所有GPU
    pool_processes = num_gpus
    print(f"创建进程池: {pool_processes} 个工作进程 (对应 {num_gpus} 个GPU)")

    # 使用按GPU分组的方式，确保每个GPU都有独立的工作进程
    # 方法：为每个GPU创建一个独立的进程池，每个进程池只处理该GPU的任务
    all_sampling_results = []

    try:
        # 为每个GPU创建独立的进程池
        pools = {}
        async_results = {}

        for gpu_id in gpu_ids:
            # 获取该GPU的所有任务
            gpu_tasks = [task for task in sampling_tasks if task[1] == gpu_id]
            if not gpu_tasks:
                continue

            print(f"GPU {gpu_id}: {len(gpu_tasks)} 个任务，创建独立进程池...")
            # 为每个GPU创建一个工作进程（因为每个GPU只需要一个进程）
            pool = Pool(processes=1)
            pools[gpu_id] = pool
            # 异步执行该GPU的所有任务
            async_results[gpu_id] = pool.map_async(process_sampling_task, gpu_tasks)

        # 等待所有GPU的任务完成
        print(f"等待所有GPU任务完成...")
        for gpu_id in gpu_ids:
            if gpu_id in async_results:
                try:
                    gpu_results = async_results[gpu_id].get()
                    all_sampling_results.extend(gpu_results)
                    print(f"GPU {gpu_id} 完成: {len(gpu_results)} 个任务")
                except Exception as e:
                    print(f"GPU {gpu_id} 执行出错: {e}")

        # 按data_id排序结果，保持原始顺序
        sampling_results = sorted(all_sampling_results, key=lambda x: x[0])
    except KeyboardInterrupt:
        print("\n⚠️  收到中断信号，正在清理采样进程...")
        for gpu_id, pool in pools.items():
            if pool:
                pool.terminate()  # 强制终止所有工作进程
                pool.join()  # 等待进程完全退出
        raise
    except Exception as e:
        print(f"\n⚠️  采样阶段发生异常: {e}")
        for gpu_id, pool in pools.items():
            if pool:
                pool.terminate()
                pool.join()
        raise
    finally:
        # 清理所有进程池
        for gpu_id, pool in pools.items():
            if pool:
                pool.close()  # 关闭进程池，不再接受新任务
                pool.join()  # 等待所有工作进程完成

    sampling_time = time.time() - sampling_start_time

    # 统计采样结果
    sampling_success = [r for r in sampling_results if r[1]]
    sampling_fail = [r for r in sampling_results if not r[1]]

    print(f"\n{'=' * 60}")
    print(f"采样阶段完成")
    print(f"{'=' * 60}")
    print(f"成功: {len(sampling_success)}")
    print(f"失败: {len(sampling_fail)}")
    print(f"耗时: {sampling_time:.2f} 秒 ({sampling_time / 60:.2f} 分钟)")
    print(f"{'=' * 60}\n")

    # 如果启用只生成模式，跳过评估阶段
    if args.sample_only:
        print(f"\n{'=' * 60}")
        print(f"只生成模式：跳过评估阶段")
        print(f"{'=' * 60}")
        print(f"采样成功的任务数: {len(sampling_success)}")
        print(f"生成的pt文件位置: {OUTPUT_DIR}")
        print(f"{'=' * 60}\n")

        # 只记录采样结果
        all_results = []
        for r in sampling_success:
            data_id, success, pt_file, msg = r
            all_results.append((data_id, True, "采样成功", None, pt_file, None))
        for r in sampling_fail:
            data_id, success, pt_file, msg = r
            all_results.append((data_id, False, msg, None, pt_file, None))

        evaluation_time = 0
    else:
        # ========== 第二阶段：评估（64个CPU核心并行） ==========
        print(f"\n{'=' * 60}")
        print(f"第二阶段：并行评估")
        print(f"{'=' * 60}")
        print(f"待评估任务数: {len(sampling_success)}")
        cores_per_task = getattr(args, "cores_per_task", 1)
        max_parallel = max(1, num_cpu_cores // cores_per_task)
        print(
            f"使用最多 {max_parallel} 个并行任务（每任务 {cores_per_task} 核，共 {num_cpu_cores} 核）"
        )
        print(f"{'=' * 60}\n")

        _vm_batch = resolve_eval_vina_modes(
            getattr(args, "eval_vina_modes", "auto"),
            config_path=args.config,
            pt_path=None,
        )
        _ev_src = getattr(args, "eval_vina_modes", "auto")
        if _vm_batch == "none":
            print(
                "提示: 当前 sampling 为 Prudent（或等价）→ 评估使用 --vina-modes none，"
                "不重跑 AutoDock Vina，仅用 .pt 中 Prudent 对接分并完成重建与理化。\n"
            )
        elif (
            _vm_batch is None
            and isinstance(_ev_src, str)
            and _ev_src.strip().lower() == "auto"
        ):
            print("提示: 非 Prudent 采样 → 评估使用 evaluate 默认三种 Vina 模式。\n")

        # 准备评估任务列表（只评估采样成功的）
        cores_per_task = getattr(args, "cores_per_task", 1)
        _batch_eval_force_mmff = args.molecular_repair == "mmff"
        _batch_eval_mmff_iters = args.mmff_max_iters if _batch_eval_force_mmff else None
        _batch_medchem_opts = medchem_opts_from_args(args)
        if _batch_medchem_opts:
            print(
                "提示: 已启用 medchem 结构优化 → 生成分子不再是模型原始输出，"
                "请作为独立 ablation 臂汇报。\n"
            )
        evaluation_tasks = []
        for r in sampling_success:
            data_id, success, pt_file, msg = r
            evaluation_tasks.append(
                (
                    data_id,
                    pt_file,
                    str(args.protein_root),
                    args.atom_mode,
                    args.exhaustiveness,
                    str(excel_file),
                    batch_start_time,
                    excel_lock,
                    cores_per_task,
                    args.save_intermediate_interval,
                    getattr(args, "eval_show_output", False),
                    getattr(args, "eval_save_log", False),
                    _batch_eval_force_mmff,
                    _batch_eval_mmff_iters,
                    _vm_batch,
                    _batch_medchem_opts,
                )
            )

        # 对于采样失败的任务，也记录到结果中
        all_results = []
        for r in sampling_fail:
            data_id, success, pt_file, msg = r
            all_results.append((data_id, False, msg, None, pt_file, None))

        evaluation_time = 0
        if evaluation_tasks:
            evaluation_start_time = time.time()

            # 评估阶段：使用指定数量的CPU核心并行
            # 每个任务分配 cores_per_task 个核心，并行任务数 = num_cpu_cores // cores_per_task
            cores_per_task = getattr(args, "cores_per_task", 1)
            max_parallel = max(1, num_cpu_cores // cores_per_task)
            max_eval_workers = min(max_parallel, len(evaluation_tasks))

            eval_pool = None
            try:
                eval_pool = Pool(processes=max_eval_workers)
                evaluation_results = eval_pool.map(
                    process_evaluation_task, evaluation_tasks
                )
            except KeyboardInterrupt:
                print("\n⚠️  收到中断信号，正在清理评估进程...")
                if eval_pool:
                    eval_pool.terminate()
                    eval_pool.join()
                raise
            except Exception as e:
                print(f"\n⚠️  评估阶段发生异常: {e}")
                if eval_pool:
                    eval_pool.terminate()
                    eval_pool.join()
                raise
            finally:
                if eval_pool:
                    eval_pool.close()
                    eval_pool.join()

            evaluation_time = time.time() - evaluation_start_time

            # 合并评估结果
            all_results.extend(evaluation_results)

            print(f"\n{'=' * 60}")
            print(f"评估阶段完成")
            print(f"{'=' * 60}")
            print(f"成功: {sum(1 for r in evaluation_results if r[1])}")
            print(f"失败: {sum(1 for r in evaluation_results if not r[1])}")
            print(f"耗时: {evaluation_time:.2f} 秒 ({evaluation_time / 60:.2f} 分钟)")
            print(f"{'=' * 60}\n")
        else:
            print(f"\n{'=' * 60}")
            print(f"评估阶段：无任务（所有采样均失败）")
            print(f"{'=' * 60}\n")

    # JSD 评估：对本次批量生成的所有 .pt 文件计算 JSD（无 MMFF 优化）
    pt_files_this_batch = [
        str(r[2]) for r in sampling_success if r[2]
    ]  # r[2] = pt_file
    if pt_files_this_batch and JSD_EVAL_SCRIPT.exists():
        print(f"\n{'=' * 60}")
        print(f"JSD 评估（无对接、无 MMFF）")
        print(f"{'=' * 60}")
        print(f"本次生成的 .pt 文件数: {len(pt_files_this_batch)}")
        try:
            # 写入临时列表文件，避免命令行过长
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write("\n".join(pt_files_this_batch))
                pt_list_file = f.name
            try:
                jsd_cmd = [
                    sys.executable,
                    str(JSD_EVAL_SCRIPT),
                    "--pt_list",
                    pt_list_file,
                    "--no_mmff",
                    "--output_dir",
                    str(REPO_ROOT / "evaljsd"),
                    "--quiet",
                ]
                jsd_proc = subprocess.run(jsd_cmd, cwd=str(REPO_ROOT), timeout=86400)
                if jsd_proc.returncode == 0:
                    print(f"✅ JSD 评估完成，结果保存在 evaljsd/ 下")
                else:
                    print(f"⚠️  JSD 评估退出码: {jsd_proc.returncode}")
            finally:
                try:
                    os.unlink(pt_list_file)
                except OSError:
                    pass
        except subprocess.TimeoutExpired:
            print(f"⚠️  JSD 评估超时")
        except Exception as e:
            print(f"⚠️  JSD 评估异常: {e}")
        print(f"{'=' * 60}\n")

    # 统计最终结果
    success_count = sum(1 for r in all_results if r[1])
    fail_count = sum(1 for r in all_results if not r[1])
    skip_count = 0

    # 批量保存所有结果到批次 CSV（只生成模式下跳过）
    if excel_file and not args.sample_only:
        print(f"\n{'=' * 70}")
        print(f"收集并保存评估结果到批次 CSV...")
        print(f"{'=' * 70}")

        molecule_records, summary_stats, pocket_stats_list = (
            collect_all_evaluation_results(all_results, batch_start_time)
        )

        print(f"收集到的分子记录数: {len(molecule_records)}")
        print(f"收集到的口袋统计信息数: {len(pocket_stats_list)}")

        if save_molecules_to_excel(
            excel_file,
            molecule_records,
            summary_stats,
            batch_start_time,
            pocket_stats_list,
            config_file=str(args.config),
        ):
            print(f"✅ 成功保存 {len(molecule_records)} 个对接成功分子到 {excel_file}")
            print(f"   统计信息:")
            print(f"     - 应生成分子数: {summary_stats.get('应生成分子数', 0)}")
            print(f"     - 可重建分子数: {summary_stats.get('可重建分子数', 0)}")
            print(f"     - 对接成功分子数: {summary_stats.get('对接成功分子数', 0)}")
            if "Vina_Dock_平均亲和力" in summary_stats:
                print(
                    f"     - Vina_Dock_平均亲和力: {summary_stats['Vina_Dock_平均亲和力']:.3f} kcal/mol"
                )

            # 批量评估完成后，从 batchsummary 文件读取参数并填写到 evaall bestchoice.xlsx
            try:
                # 导入更新函数（sys已在文件顶部导入）
                eval_script_path = REPO_ROOT / "evaluate_pt_with_correct_reconstruct.py"
                if eval_script_path.exists():
                    # 动态导入模块
                    import importlib.util

                    spec = importlib.util.spec_from_file_location(
                        "evaluate_pt_module", eval_script_path
                    )
                    evaluate_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(evaluate_module)

                    # 调用更新函数
                    if hasattr(evaluate_module, "update_bestchoice_excel_with_params"):
                        evaluate_module.update_bestchoice_excel_with_params(
                            output_dir=None
                        )
            except Exception as e:
                print(f"  ⚠️  更新 evaall bestchoice.xlsx 失败: {e}")
                import traceback

                traceback.print_exc()
        else:
            print(f"⚠️  批次 CSV 保存失败")
        print(f"{'=' * 70}\n")

    # 打印总结
    elapsed_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    if args.sample_only:
        print(f"批量采样完成（只生成模式）")
    else:
        print(f"批量处理完成（并行版本）")
    print(f"{'=' * 60}")
    print(f"总计: {total}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"总耗时: {elapsed_time:.2f} 秒 ({elapsed_time / 60:.2f} 分钟)")
    print(f"  - 采样耗时: {sampling_time:.2f} 秒 ({sampling_time / 60:.2f} 分钟)")
    if evaluation_time > 0:
        print(
            f"  - 评估耗时: {evaluation_time:.2f} 秒 ({evaluation_time / 60:.2f} 分钟)"
        )
    print(f"平均每个任务: {elapsed_time / total:.2f} 秒")
    if args.sample_only:
        print(f"📁 生成的pt文件保存在: {OUTPUT_DIR}")
        print(f"   成功生成 {len(sampling_success)} 个pt文件")
    elif excel_file:
        print(f"📊 详细记录已保存至: {excel_file}（及同主名侧车 CSV）")
    print(f"{'=' * 60}\n")
