"""Benchmark responsibilities extracted from batch_sampleandeval_parallel.py."""

from diffdynamic.batch.state import (
    EVAL_SCRIPT,
    Manager,
    Path,
    Pool,
    REPO_ROOT,
    datetime,
    os,
    pickle,
    shutil,
    sys,
    time,
    timedelta,
    timezone,
    torch,
)
from diffdynamic.batch import state as _state


def _pick_result_pt_standard_layout(base_dir: Path, pocket_id: int):
    """在目录顶层查找 result_{id}.pt 或最新的 result_{id}_*.pt。"""
    if not base_dir.is_dir():
        return None
    pid = str(int(pocket_id))
    exact = base_dir / f"result_{pid}.pt"
    if exact.is_file():
        return exact
    candidates = list(base_dir.glob(f"result_{pid}_*.pt"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _pick_glintdm_bench_pt(base_dir: Path, pocket_id: int):
    """GlintDM 基准命名：..._<pocket_id>.pt"""
    if not base_dir.is_dir():
        return None
    suffix = f"_{int(pocket_id)}.pt"
    for p in base_dir.glob("*.pt"):
        if p.is_file() and p.name.endswith(suffix):
            return p
    return None


def _pick_decompdiff_bench_pt(base_dir: Path, pocket_id: int):
    """DecompDiff：sampling_drift_{idx:03d}_*/result.pt"""
    if not base_dir.is_dir():
        return None
    pat = f"sampling_drift_{int(pocket_id):03d}_*"
    dirs = sorted([d for d in base_dir.glob(pat) if d.is_dir()])
    if not dirs:
        return None
    cand = dirs[0] / "result.pt"
    return cand if cand.is_file() else None


def _resolve_protein_root_for_eval_fallback(protein_root_arg):
    """
    解析 --protein_root：若路径不存在则按单独评估模式尝试常见数据目录。
    未传入时亦尝试自动选用首个存在的回退路径。
    """
    fallback_paths = [
        REPO_ROOT / "data" / "crossdocked_v1.1_rmsd1.0_pocket10",
        REPO_ROOT / "data" / "crossdocked_v1.1_rmsd1.0",
        Path("/workspace/data/crossdocked_v1.1_rmsd1.0_pocket10"),
        Path("/workspace/data"),
        REPO_ROOT,
    ]
    if protein_root_arg:
        pr = Path(protein_root_arg).expanduser()
        if pr.exists():
            return pr
        orig = pr
        for fb in fallback_paths:
            if fb.exists():
                print(f"💡 --protein_root {orig} 不存在，使用回退路径: {fb}")
                return fb
        return None
    for fb in fallback_paths:
        if fb.exists():
            print(f"💡 未指定 --protein_root，自动使用: {fb}")
            return fb
    return None


def default_benchmark_collect_backend_dirs():
    """各基线/对比方法 outputs 根目录（相对本仓库）。"""
    return [
        ("TargetDiff", REPO_ROOT / "targetdiff-main" / "outputs"),
        ("JSDPT3010", REPO_ROOT / "jsdpt3010"),
        (
            "DiffSBDD",
            REPO_ROOT / "third_party" / "DiffSBDD" / "outputs_diffsbdd_bench_100p",
        ),
        (
            "DecompDiff",
            REPO_ROOT / "third_party" / "DecompDiff" / "outputs_decompdiff_bench_100p",
        ),
        (
            "MolForm",
            REPO_ROOT / "third_party" / "MolForm" / "outputs_molform_bench_100p",
        ),
        ("IPDiff", REPO_ROOT / "third_party" / "IPDiff" / "outputs_ipdiff_bench_100p"),
        ("GlintDM", REPO_ROOT / "glintdm" / "outputs_glintdm_bench_100p"),
    ]


def _validate_pt_compatible(pt_path):
    """
    检查 .pt 文件格式，返回格式类型和状态。
    支持格式:
    - 'targetdiff': dict 含 'pred_ligand_pos', 'pred_ligand_v'（与主仓库兼容）
    - 'decompdiff': list 含 'pred_pos', 'pred_v', 'ligand_filename'
    Returns: (format_type, info_msg)  format_type: 'targetdiff'|'decompdiff'|None
    """
    try:
        if torch is None:
            return None, "torch 未安装"
        data = torch.load(str(pt_path), map_location="cpu")

        # TargetDiff / JSDPT 格式
        if isinstance(data, dict):
            required_keys = ["pred_ligand_pos", "pred_ligand_v", "data"]
            if all(k in data for k in required_keys):
                return "targetdiff", "TargetDiff/JSDPT 格式"
            return None, f"字段缺失: {data.keys()}"

        # DecompDiff 格式: list 且元素含 pred_pos/pred_v
        if isinstance(data, list) and len(data) > 0:
            sample = data[0]
            if isinstance(sample, dict) and "pred_pos" in sample and "pred_v" in sample:
                return "decompdiff", "DecompDiff 格式"
            return (
                None,
                f"未知 list 格式: {list(sample.keys()) if isinstance(sample, dict) else type(sample)}",
            )

        return None, f"未知根类型: {type(data).__name__}"
    except Exception as e:
        return None, f"加载失败: {e}"


def _get_protein_path_from_index(pocket_id, protein_root):
    """从 CrossDocked index.pkl 获取指定 data_id 的蛋白口袋 PDB 路径。"""
    try:
        index_path = protein_root / "index.pkl"
        if not index_path.exists():
            return None
        with open(index_path, "rb") as f:
            index = pickle.load(f)
        if not isinstance(index, list) or pocket_id >= len(index):
            return None
        entry = index[pocket_id]
        protein_rel = (
            entry[0] if isinstance(entry, (list, tuple)) else entry.get("protein")
        )
        protein_abs = protein_root / protein_rel
        return protein_abs if protein_abs.exists() else None
    except Exception:
        return None


def process_sdf_evaluation_task(args_tuple):
    """
    处理单个 SDF 文件的对接评估任务（用于 DiffSBDD 等 SDF-only 基线）。

    Args:
        args_tuple: (data_id, sdf_file, protein_root, atom_mode, exhaustiveness,
            batch_start_time, excel_lock, cores_per_task[, eval_show_output])

    Returns:
        tuple: (data_id, success, message, sdf_file, eval_output_dir)
    """
    if len(args_tuple) >= 9:
        (
            data_id,
            sdf_file,
            protein_root,
            atom_mode,
            exhaustiveness,
            batch_start_time,
            excel_lock,
            cores_per_task,
            eval_show_output,
        ) = args_tuple[:9]
    else:
        (
            data_id,
            sdf_file,
            protein_root,
            atom_mode,
            exhaustiveness,
            batch_start_time,
            excel_lock,
            cores_per_task,
        ) = args_tuple[:8]
        eval_show_output = False

    from multiprocessing import current_process
    import subprocess
    import time

    task_start_time = time.time()
    sdf_path = Path(sdf_file)

    # 从 index.pkl 查找蛋白路径
    protein_path = _get_protein_path_from_index(int(data_id), Path(protein_root))
    if protein_path is None:
        return (
            data_id,
            False,
            f"无法从 index.pkl 找到 data_id={data_id} 的蛋白路径",
            str(sdf_path),
            None,
        )

    # 评估输出目录
    cst = timezone(timedelta(hours=8))
    eval_timestamp = datetime.now(cst).strftime("%Y%m%d_%H%M%S")
    eval_output_dir = sdf_path.parent / f"eval_{data_id}_{eval_timestamp}"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # 限制 CPU 核心数
    env = os.environ.copy()
    if cores_per_task >= 1:
        cores_str = str(cores_per_task)
        env["OMP_NUM_THREADS"] = cores_str
        env["MKL_NUM_THREADS"] = cores_str
        env["OPENBLAS_NUM_THREADS"] = cores_str
        env["NUMEXPR_NUM_THREADS"] = cores_str
        env["VECLIB_MAXIMUM_THREADS"] = cores_str

    try:
        print(
            f"[SDF评估] 开始对接: {sdf_path.name} (data_id={data_id}) 蛋白: {protein_path.name}"
        )

        # 使用 smina 进行对接（直接命令行调用）
        # 首先将蛋白 PDB 转换为 PDBQT（如果还没有）
        protein_pdbqt = eval_output_dir / f"{protein_path.stem}.pdbqt"
        if not protein_pdbqt.exists():
            # Open Babel 2.x: -O 与输出文件名之间不能有空格
            prep_cmd = f"obabel {protein_path} -O{protein_pdbqt} 2>/dev/null || true"
            subprocess.run(prep_cmd, shell=True, env=env, timeout=300)

        # 获取 SDF 中的分子数量并逐个对接
        from rdkit import Chem

        suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
        scores = []

        for i, mol in enumerate(suppl):
            if mol is None:
                continue
            # 计算分子中心
            conf = mol.GetConformer()
            cx = (
                sum(conf.GetAtomPosition(j).x for j in range(mol.GetNumAtoms()))
                / mol.GetNumAtoms()
            )
            cy = (
                sum(conf.GetAtomPosition(j).y for j in range(mol.GetNumAtoms()))
                / mol.GetNumAtoms()
            )
            cz = (
                sum(conf.GetAtomPosition(j).z for j in range(mol.GetNumAtoms()))
                / mol.GetNumAtoms()
            )

            # 单个分子的 SDF 和输出
            mol_sdf = eval_output_dir / f"mol_{i}.sdf"
            mol_pdbqt = eval_output_dir / f"mol_{i}.pdbqt"
            out_sdf = eval_output_dir / f"mol_{i}_out.sdf"

            # 保存单个分子
            writer = Chem.SDWriter(str(mol_sdf))
            writer.write(mol)
            writer.close()

            # 转换为 PDBQT（-O 与输出路径无空格）
            obabel_cmd = f"obabel {mol_sdf} -O{mol_pdbqt} 2>/dev/null"
            subprocess.run(obabel_cmd, shell=True, env=env, timeout=60)

            if not mol_pdbqt.exists():
                continue

            # 运行 smina 对接
            smina_bin = shutil.which("smina.static") or shutil.which("smina") or "smina"
            smina_cmd = [
                smina_bin,
                "-r",
                str(protein_pdbqt) if protein_pdbqt.exists() else str(protein_path),
                "-l",
                str(mol_pdbqt),
                "--center_x",
                str(cx),
                "--center_y",
                str(cy),
                "--center_z",
                str(cz),
                "--size_x",
                "20",
                "--size_y",
                "20",
                "--size_z",
                "20",
                "--exhaustiveness",
                str(exhaustiveness),
                "--score_only",  # 只打分，不输出构象，更快
            ]

            try:
                smina_result = subprocess.run(
                    smina_cmd, capture_output=True, text=True, env=env, timeout=300
                )
                # 解析 Affinity
                for line in smina_result.stdout.split("\n"):
                    if "Affinity:" in line:
                        try:
                            score = float(line.split()[1])
                            scores.append(score)
                            break
                        except (ValueError, IndexError):
                            pass
            except Exception:
                pass

        # 保存结果到文件
        if scores:
            import numpy as np

            results_summary = {
                "scores": scores,
                "mean": np.mean(scores),
                "median": np.median(scores),
                "min": np.min(scores),
                "num_molecules": len(scores),
            }
            import json

            with open(eval_output_dir / "smina_scores.json", "w") as f:
                json.dump(results_summary, f, indent=2)

        elapsed = time.time() - task_start_time
        print(
            f"[SDF评估] ✅ 成功 (data_id={data_id}) 耗时 {elapsed:.1f}s, 成功对接 {len(scores)} 个分子"
        )
        return (
            data_id,
            True,
            f"SDF 对接成功 ({len(scores)} 个分子)",
            str(sdf_path),
            str(eval_output_dir),
        )
    except Exception as e:
        error_msg = f"SDF 评估异常: {str(e)}"
        print(f"[SDF评估] ❌ 异常 (data_id={data_id}): {error_msg}")
        import traceback

        traceback.print_exc()
        return (data_id, False, error_msg, str(sdf_path), None)


def process_decompdiff_pt_evaluation_task(args_tuple):
    """
    处理 DecompDiff 格式 .pt 文件的评估任务。
    调用 DecompDiff 的 evaluate_mol_from_meta_full.py 脚本进行评估。

    Args:
        args_tuple: (data_id, pt_file, protein_root, atom_mode, exhaustiveness,
            batch_start_time, excel_lock, cores_per_task[, eval_show_output])

    Returns:
        tuple: (data_id, success, message, pt_file, eval_output_dir)
    """
    if len(args_tuple) >= 9:
        (
            data_id,
            pt_file,
            protein_root,
            atom_mode,
            exhaustiveness,
            batch_start_time,
            excel_lock,
            cores_per_task,
            eval_show_output,
        ) = args_tuple[:9]
    else:
        (
            data_id,
            pt_file,
            protein_root,
            atom_mode,
            exhaustiveness,
            batch_start_time,
            excel_lock,
            cores_per_task,
        ) = args_tuple[:8]
        eval_show_output = False

    import subprocess
    import time

    task_start_time = time.time()
    pt_path = Path(pt_file)

    # 评估输出目录
    cst = timezone(timedelta(hours=8))
    eval_timestamp = datetime.now(cst).strftime("%Y%m%d_%H%M%S")
    eval_output_dir = pt_path.parent / f"eval_{data_id}_{eval_timestamp}"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # DecompDiff 评估脚本路径（使用绝对路径）
    decompdiff_dir = REPO_ROOT / "third_party" / "DecompDiff"
    decompdiff_dir_abs = decompdiff_dir.resolve()
    decompdiff_eval_script = (
        decompdiff_dir_abs / "scripts" / "evaluate_mol_from_meta_full.py"
    )

    if not decompdiff_eval_script.is_file():
        return (
            data_id,
            False,
            f"DecompDiff 评估脚本不存在: {decompdiff_eval_script}",
            str(pt_path),
            None,
        )

    # 限制 CPU 核心数
    env = os.environ.copy()
    if cores_per_task >= 1:
        cores_str = str(cores_per_task)
        env["OMP_NUM_THREADS"] = cores_str
        env["MKL_NUM_THREADS"] = cores_str
        env["OPENBLAS_NUM_THREADS"] = cores_str
        env["NUMEXPR_NUM_THREADS"] = cores_str
        env["VECLIB_MAXIMUM_THREADS"] = cores_str

    # 仅保留 DecompDiff 根目录，避免继承父进程的 PYTHONPATH（主仓库 /workspace 的 utils 会遮蔽 DecompDiff）
    env["PYTHONPATH"] = str(decompdiff_dir_abs)

    # 构建命令：使用 -c 执行，在代码中设置 sys.path 和 sys.argv，然后用 runpy 运行脚本
    pt_path_abs = Path(pt_path).resolve()
    protein_root_abs = Path(protein_root).resolve()
    eval_output_dir_abs = Path(eval_output_dir).resolve()
    script_path_abs = decompdiff_eval_script.resolve()
    repo_root_abs = REPO_ROOT.resolve()

    # 从 sys.path 去掉主仓库根，防止 import utils 命中 DiffDynamic 的 utils 包
    dd_s = str(decompdiff_dir_abs)
    repo_s = str(repo_root_abs)
    script_s = str(script_path_abs)
    pt_s = str(pt_path_abs)
    pr_s = str(protein_root_abs)
    out_s = str(eval_output_dir_abs)
    exh_s = str(exhaustiveness)

    setup_code = (
        "import sys, runpy; "
        f"DD={repr(dd_s)}; REPO={repr(repo_s)}; "
        "sys.path=[DD]+[p for p in sys.path if p and p != DD and p != REPO]; "
        f"sys.argv=[{repr(script_s)}, {repr(pt_s)}, '--protein_root', {repr(pr_s)}, "
        "'--docking_mode', 'vina_score', "
        f"'--exhaustiveness', {repr(exh_s)}, "
        "'--num_workers', '1', "
        f"'--result_path', {repr(out_s)}, '--aggregate_meta', 'False']; "
        f"runpy.run_path({repr(script_s)}, run_name='__main__')"
    )

    cmd = [sys.executable, "-c", setup_code]

    try:
        print(f"[DecompDiff评估] 开始评估: {pt_path.name} (data_id={data_id})")
        print(f"[DecompDiff评估] 工作目录: {decompdiff_dir_abs}")
        if eval_show_output:
            result = subprocess.run(
                cmd, check=True, env=env, cwd=str(decompdiff_dir_abs), timeout=21600
            )
        else:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env=env,
                cwd=str(decompdiff_dir_abs),
                timeout=21600,
            )

        elapsed = time.time() - task_start_time
        print(f"[DecompDiff评估] ✅ 成功 (data_id={data_id}) 耗时 {elapsed:.1f}s")
        return (
            data_id,
            True,
            "DecompDiff 评估成功",
            str(pt_path),
            str(eval_output_dir),
        )
    except subprocess.CalledProcessError as e:
        error_msg = f"DecompDiff 评估失败: {e.stderr[-500:] if e.stderr else str(e)}"
        print(f"[DecompDiff评估] ❌ 失败 (data_id={data_id}): {error_msg}")
        return (data_id, False, error_msg, str(pt_path), None)
    except Exception as e:
        error_msg = f"DecompDiff 评估异常: {str(e)}"
        print(f"[DecompDiff评估] ❌ 异常 (data_id={data_id}): {error_msg}")
        return (data_id, False, error_msg, str(pt_path), None)


def collect_benchmark_pocket_and_eval(
    pocket_id,
    protein_root,
    docs_parent=None,
    atom_mode="add_aromatic",
    exhaustiveness=8,
    num_cpu_cores=64,
    cores_per_task=1,
    eval_show_output=False,
    eval_save_log=False,
    force_mmff_minimize=False,
    mmff_max_iters=None,
    include_methods=None,
):
    """
    将各基线 outputs 中该测试口袋的 .pt 复制到 docs/<口袋编号>/<方法>/pt/（或 SDF 到 sdf/），
    再对每个已复制的 .pt/.sdf 调用本仓库 evaluate 流程（在对应目录下生成 eval_* 及 Excel）。

    Args:
        include_methods: 要处理的方法列表，如 ['TargetDiff', 'JSDPT3010', 'DiffSBDD']。
                        None 表示处理所有方法。

    Returns:
        目标根目录 Path
    """
    from diffdynamic.batch.evaluation import _dispatch_eval_task
    from diffdynamic.batch.resources import _preflight_import_torch_scatter

    if not EVAL_SCRIPT.is_file():
        print(f"❌ 错误: 评估脚本不存在: {EVAL_SCRIPT}")
        sys.exit(1)

    if not _preflight_import_torch_scatter():
        print(
            "❌ 错误: 当前 Python 环境未安装 `torch_scatter`，evaluate 子进程会在 import 阶段失败。\n"
            "   修复（容器内需已激活 conda 环境，如 diffdynamic）：\n"
            "   • 推荐：bash scripts/repair_pyg_for_current_torch.sh\n"
            "     （按当前 torch 版本从 https://data.pyg.org/whl/ 安装匹配的 pyg-lib / torch-scatter 等）\n"
            "   • 或见仓库 初始化.md 中「torch_scatter / PyG」与 pip 安装说明。\n"
            "   安装成功后请重新运行本命令。"
        )
        sys.exit(1)

    docs_parent = docs_parent or (REPO_ROOT / "docs")
    dest_root = (docs_parent / str(int(pocket_id))).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    protein_root = Path(protein_root).resolve()

    backends = default_benchmark_collect_backend_dirs()

    # 方法筛选
    if include_methods is not None:
        allowed = set(include_methods)
        backends = [(lbl, pth) for lbl, pth in backends if lbl in allowed]
        if not backends:
            print(
                f"❌ 错误: --methods 指定了无效的方法。可用: {[_lb for _lb, _ in default_benchmark_collect_backend_dirs()]}"
            )
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print(
        f"基准口袋对比：data_id = {int(pocket_id)}（复制 .pt/.sdf → docs 下并启动评估）"
    )
    if include_methods:
        print(f"本次选择的方法: {include_methods}")
    print(f"目标目录: {dest_root}")
    print(f"protein_root: {protein_root}")
    print(f"{'=' * 60}\n")

    targetdiff_eval_jobs = []  # (label, dest_pt) 用于 TargetDiff/JSDPT 格式
    decompdiff_eval_jobs = []  # (label, dest_pt) 用于 DecompDiff 格式
    sdf_eval_jobs = []  # (label, dest_sdf) 用于 SDF 评估

    for label, root in backends:
        # 先尝试找 .pt
        pt_dst_dir = dest_root / label / "pt"
        if label == "DecompDiff":
            src_pt = _pick_decompdiff_bench_pt(root, pocket_id)
        elif label == "GlintDM":
            src_pt = _pick_glintdm_bench_pt(root, pocket_id)
        else:
            src_pt = _pick_result_pt_standard_layout(root, pocket_id)

        if src_pt is not None:
            pt_dst_dir.mkdir(parents=True, exist_ok=True)
            dest_pt = pt_dst_dir / src_pt.name
            shutil.copy2(src_pt, dest_pt)
            # 验证 .pt 格式
            fmt, info = _validate_pt_compatible(dest_pt)
            if fmt == "targetdiff":
                print(
                    f"  [{label}] 已复制 .pt (TargetDiff格式): {dest_pt.name} <- {src_pt}"
                )
                targetdiff_eval_jobs.append((label, dest_pt))
            elif fmt == "decompdiff":
                print(
                    f"  [{label}] 已复制 .pt (DecompDiff格式): {dest_pt.name} <- {src_pt}"
                )
                decompdiff_eval_jobs.append((label, dest_pt))
            else:
                print(
                    f"  [{label}] 已复制 .pt (未知格式，跳过): {dest_pt.name} <- {src_pt}"
                )
                print(f"           原因: {info}")
            continue

        # 无 .pt 时，对 DiffSBDD 尝试找 SDF
        if label == "DiffSBDD" and root.is_dir():
            sdf_candidates = list(root.glob(f"diffsbdd_pocket_{int(pocket_id)}.sdf"))
            if sdf_candidates:
                sdf_dst_dir = dest_root / label / "sdf"
                sdf_dst_dir.mkdir(parents=True, exist_ok=True)
                src_sdf = sdf_candidates[0]
                dest_sdf = sdf_dst_dir / src_sdf.name
                shutil.copy2(src_sdf, dest_sdf)
                print(f"  [{label}] 已复制 SDF: {dest_sdf.name} <- {src_sdf}")
                sdf_eval_jobs.append((label, dest_sdf))
                continue

        print(f"  [{label}] 跳过：未找到 .pt 或 SDF（{root}）")

    total_jobs = (
        len(targetdiff_eval_jobs) + len(decompdiff_eval_jobs) + len(sdf_eval_jobs)
    )
    if total_jobs == 0:
        print("\n❌ 未找到任何可评估的 .pt 或 SDF，退出\n")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(
        f"开始并行评估：{len(targetdiff_eval_jobs)} 个 TargetDiff + "
        f"{len(decompdiff_eval_jobs)} 个 DecompDiff + "
        f"{len(sdf_eval_jobs)} 个 SDF = 共 {total_jobs} 路"
    )
    max_parallel = max(1, int(num_cpu_cores) // max(1, int(cores_per_task)))
    workers = min(max_parallel, total_jobs)
    print(
        f"CPU 并行: num_cpu_cores={num_cpu_cores}, cores_per_task={cores_per_task} "
        f"→ 理论最多 {max_parallel} 路并行，实际启动 {workers} 路"
    )
    print(f"{'=' * 60}\n")

    manager = Manager()
    excel_lock = manager.Lock()
    batch_start_time = time.time()
    pid_int = int(pocket_id)

    # 构建 TargetDiff 格式 .pt 评估任务
    targetdiff_task_tuples = [
        (
            pid_int,
            str(dest_pt),
            str(protein_root),
            atom_mode,
            exhaustiveness,
            None,
            batch_start_time,
            excel_lock,
            max(1, int(cores_per_task)),
            0,
            eval_show_output,
            eval_save_log,
            force_mmff_minimize,
            mmff_max_iters,
        )
        for _label, dest_pt in targetdiff_eval_jobs
    ]

    # 构建 DecompDiff 格式 .pt 评估任务
    decompdiff_task_tuples = [
        (
            pid_int,
            str(dest_pt),
            str(protein_root),
            atom_mode,
            exhaustiveness,
            batch_start_time,
            excel_lock,
            max(1, int(cores_per_task)),
            eval_show_output,
        )
        for _label, dest_pt in decompdiff_eval_jobs
    ]

    # 构建 SDF 评估任务
    sdf_task_tuples = [
        (
            pid_int,
            str(dest_sdf),
            str(protein_root),
            atom_mode,
            exhaustiveness,
            batch_start_time,
            excel_lock,
            max(1, int(cores_per_task)),
            eval_show_output,
        )
        for _label, dest_sdf in sdf_eval_jobs
    ]

    # 合并任务并标记类型以便后续拆分结果
    all_tasks = (
        [("targetdiff", t) for t in targetdiff_task_tuples]
        + [("decompdiff", t) for t in decompdiff_task_tuples]
        + [("sdf", t) for t in sdf_task_tuples]
    )

    pool = None
    raw_results = []
    try:
        pool = Pool(processes=workers)
        # 使用统一 map，按类型分发到不同处理函数
        raw_results = pool.map(_dispatch_eval_task, all_tasks)
    finally:
        if pool:
            pool.close()
            pool.join()

    # 拆分结果
    targetdiff_results = [
        r for r, t in zip(raw_results, all_tasks) if t[0] == "targetdiff"
    ]
    decompdiff_results = [
        r for r, t in zip(raw_results, all_tasks) if t[0] == "decompdiff"
    ]
    sdf_results = [r for r, t in zip(raw_results, all_tasks) if t[0] == "sdf"]

    # 汇总输出
    ok = 0
    print(f"\n{'=' * 60}")
    print("各方法评估结果")
    print(f"{'=' * 60}")

    for (label, _dest_pt), row in zip(targetdiff_eval_jobs, targetdiff_results):
        success = bool(row[1]) if row and len(row) > 1 else False
        msg = (row[2] if row and len(row) > 2 else "") or ""
        if success:
            ok += 1
            print(f"  [{label}] ✅ 成功 (TargetDiff格式)")
        else:
            short = (msg[:400] + "…") if len(msg) > 400 else msg
            print(f"  [{label}] ❌ 失败 (TargetDiff格式): {short}")

    for (label, _dest_pt), row in zip(decompdiff_eval_jobs, decompdiff_results):
        success = bool(row[1]) if row and len(row) > 1 else False
        msg = (row[2] if row and len(row) > 2 else "") or ""
        if success:
            ok += 1
            print(f"  [{label}] ✅ 成功 (DecompDiff格式)")
        else:
            short = (msg[:400] + "…") if len(msg) > 400 else msg
            print(f"  [{label}] ❌ 失败 (DecompDiff格式): {short}")

    for (label, _dest_sdf), row in zip(sdf_eval_jobs, sdf_results):
        success = bool(row[1]) if row and len(row) > 1 else False
        msg = (row[2] if row and len(row) > 2 else "") or ""
        if success:
            ok += 1
            print(f"  [{label}] ✅ 成功 (SDF)")
        else:
            short = (msg[:400] + "…") if len(msg) > 400 else msg
            print(f"  [{label}] ❌ 失败 (SDF): {short}")

    print(f"{'=' * 60}\n")

    if ok == len(raw_results):
        print(
            f"✅ 全部 {ok}/{len(raw_results)} 路评估成功；输出在 {dest_root} 各方法子目录下的 eval_*。\n"
        )
    else:
        print(
            f"⚠️  评估结束：成功 {ok}/{len(raw_results)}，失败 {len(raw_results) - ok}。\n"
            f"   结果已保存在 {dest_root}；请根据上方失败信息排查。\n"
        )
        sys.exit(1)

    return dest_root
