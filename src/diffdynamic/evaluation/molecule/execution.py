"""Pipeline responsibilities extracted from evaluate_pt_with_correct_reconstruct.py."""

from diffdynamic.evaluation.molecule.state import (
    Chem,
    DEFAULT_SINGLE_MOL_EVAL_TIMEOUT_SEC,
    Path,
    REPO_ROOT,
    _ALL_VINA_MODES,
    _DEFAULT_VINA_MODES,
    _TRUE_VALUES,
    atexit,
    datetime,
    eval_bond_length,
    extract_protein_id,
    generate_molecule_id,
    misc,
    np,
    os,
    re,
    resolve_receptor_pdb_for_docking,
    shutil,
    similarity,
    sys,
    time,
    torch,
    tqdm,
    traceback,
)
from diffdynamic.evaluation.molecule import state as _state


from .context import EvaluatedMolecules


def evaluate_molecules(context):
    """Reconstruct, score and persist each molecule in the established order."""
    from diffdynamic.evaluation.molecule.loading import (
        decompdiff_list_to_targetdiff_dict,
        load_pt_file,
        validate_pt_data,
    )
    from diffdynamic.evaluation.molecule.optimization import (
        _apply_medchem_optimize,
        _evaluate_preopt_molecule,
        _safe_heavy_smiles,
        _write_optimize_audit,
        _write_preopt_sdf,
    )
    from diffdynamic.evaluation.molecule.options import (
        _benchmark_ligands_csv_lookup,
        _crossdocked_index_lookup,
        _infer_data_id_from_docs_layout,
        generate_eval_dir_name,
    )
    from diffdynamic.evaluation.molecule.reconstruction import (
        _scaffold_attachment_bonds,
        reconstruct_molecule,
        remove_small_fragments,
    )
    from diffdynamic.evaluation.molecule.scoring import (
        calculate_comprehensive_score,
        dock_reference_ligand,
    )
    from diffdynamic.evaluation.molecule.workers import (
        evaluate_single_molecule_isolated,
    )
    from diffdynamic.evaluation.molecule.writers import (
        record_complete_molecules_to_excel,
        record_evaluation_results_to_excel,
        save_intermediate_results,
    )

    _vina_modes = context._vina_modes
    atom_mode = context.atom_mode
    buffer = context.buffer
    data_id = context.data_id
    debug = context.debug
    eval_output_dir = context.eval_output_dir
    eval_timestamp = context.eval_timestamp
    exhaustiveness = context.exhaustiveness
    keep_preopt_sdf = context.keep_preopt_sdf
    ligand_filename = context.ligand_filename
    meta_records = context.meta_records
    n_poses = context.n_poses
    n_scaffold_pt = context.n_scaffold_pt
    num_samples = context.num_samples
    pre_docking_etkdg_reembed = context.pre_docking_etkdg_reembed
    pre_docking_mmff_max_iters = context.pre_docking_mmff_max_iters
    pre_docking_mmff_minimize = context.pre_docking_mmff_minimize
    pre_docking_use_uff_fallback = context.pre_docking_use_uff_fallback
    pred_atom_types = context.pred_atom_types
    pred_positions = context.pred_positions
    protein_filename = context.protein_filename
    protein_root = context.protein_root
    pt_path = context.pt_path
    refined_for_pt_vina = context.refined_for_pt_vina
    remove_fragments = context.remove_fragments
    resolved_protein_path = context.resolved_protein_path
    save_intermediate_interval = context.save_intermediate_interval
    save_sdf = context.save_sdf
    scaffold_bonds_pt = context.scaffold_bonds_pt
    scaffold_cfg_meta = context.scaffold_cfg_meta
    scaffold_substruct_mol = context.scaffold_substruct_mol
    score_category_dirs = context.score_category_dirs
    sdf_dir = context.sdf_dir
    sdf_dir_optimized = context.sdf_dir_optimized
    sdf_dir_original = context.sdf_dir_original
    size_factor = context.size_factor
    tmp_dir = context.tmp_dir
    use_isolation = context.use_isolation
    use_split_sdf_dirs = context.use_split_sdf_dirs

    results = []
    n_reconstruct_success = 0
    n_eval_success = 0
    n_complete = 0

    # 激进 medchem 优化（默认关闭）的产出
    optimize_results = []
    n_medchem_optimized = 0
    medchem_enabled = bool((_state._MEDCHEM_OPTIMIZE_CFG or {}).get("enable"))
    preopt_sdf_dir = None
    if medchem_enabled and save_sdf and keep_preopt_sdf != "none":
        preopt_sdf_dir = eval_output_dir / "reconstructed_molecules_preopt"
        preopt_sdf_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ 优化前结构将保存至: {preopt_sdf_dir} (模式={keep_preopt_sdf})")

    # 重建失败统计
    reconstruct_failures = []  # 记录所有重建失败的详细信息
    failure_stats = {
        "MolReconsError": 0,
        "reconstruct_returned_none": 0,
        "other_errors": {},
    }

    # 顺序处理每个分子
    pbar = tqdm(range(num_samples), desc="评估分子", file=sys.stdout)

    # 注册退出时保存中间结果的函数（save_intermediate_interval=0 时不保存）
    def save_on_exit():
        # 最终结果已落盘则不再写中间快照（否则 atexit 会在 final 之后
        # 重新生成 intermediate 文件，白白占磁盘）
        if eval_output_dir and list(
            Path(eval_output_dir).glob("eval_results_*_final_*.pt")
        ):
            return
        if save_intermediate_interval > 0 and results:  # 如果有已处理的结果
            try:
                # 计算当前统计信息
                current_vina_dock_scores = [
                    r["vina_dock"][0]["affinity"]
                    for r in results
                    if r["success"] and r.get("vina_dock") and len(r["vina_dock"]) > 0
                ]
                current_vina_score_only_scores = [
                    r["vina_score_only"]
                    for r in results
                    if r["success"] and r.get("vina_score_only") is not None
                ]
                current_vina_minimize_scores = [
                    r["vina_minimize"]
                    for r in results
                    if r["success"] and r.get("vina_minimize") is not None
                ]
                current_qed_values = [
                    r["chem"]["qed"] for r in results if r.get("chem")
                ]
                current_sa_values = [r["chem"]["sa"] for r in results if r.get("chem")]
                current_atom_type_jsd_values = [
                    r["atom_type_jsd"]
                    for r in results
                    if r.get("atom_type_jsd") is not None
                ]

                # 计算当前的重建失败数
                current_n_reconstruct_fail = num_samples - n_reconstruct_success

                save_intermediate_results(
                    eval_output_dir,
                    eval_timestamp,
                    None,
                    pt_path,
                    ligand_filename,
                    protein_root,
                    atom_mode,
                    exhaustiveness,
                    num_samples,
                    n_reconstruct_success,
                    n_eval_success,
                    n_complete,
                    data_id,
                    results,
                    current_vina_dock_scores,
                    current_vina_score_only_scores,
                    current_vina_minimize_scores,
                    current_qed_values,
                    current_sa_values,
                    current_atom_type_jsd_values,
                    {},
                    {},
                    current_n_reconstruct_fail,
                    failure_stats,
                    reconstruct_failures,
                    is_final=False,
                )
            except Exception as e:
                print(f"\n⚠️  退出时保存中间结果失败: {e}")
                traceback.print_exc()

    atexit.register(save_on_exit)

    # dynamic_then_optimization: 记录每个 dynamic 分子的 molecule_id，供优化变体 _opN_from_{原分子id} 使用
    dynamic_id_to_molecule_id = {}

    single_eval_timeout = DEFAULT_SINGLE_MOL_EVAL_TIMEOUT_SEC
    if use_isolation:
        try:
            single_eval_timeout = max(
                5,
                int(
                    os.environ.get(
                        "EVAL_SINGLE_MOL_TIMEOUT",
                        str(DEFAULT_SINGLE_MOL_EVAL_TIMEOUT_SEC),
                    )
                ),
            )
        except ValueError:
            single_eval_timeout = DEFAULT_SINGLE_MOL_EVAL_TIMEOUT_SEC

    for idx in pbar:
        mol = None
        reconstructed_this_sample = False
        reconstructed_smiles_before_fragment_removal = None
        try:
            if debug:
                print(f"\n[{idx + 1}/{num_samples}] 处理分子 {idx}...", flush=True)

            pos = pred_positions[idx]
            v = pred_atom_types[idx]

            # 3.1 重建分子
            try:
                # Site allocation is an initialization prior, not a bond
                # topology constraint. Protect only the locked CRBN scaffold
                # bonds; all extra bonds must be supported by final geometry.
                allow_forced_attachment = bool(
                    scaffold_cfg_meta.get(
                        "allow_forced_attachment_bonds",
                        False,
                    )
                )
                attachment_bonds = (
                    _scaffold_attachment_bonds(
                        meta_records[idx] if idx < len(meta_records) else None,
                        scaffold_cfg_meta,
                        len(pos),
                    )
                    if allow_forced_attachment
                    else []
                )
                scaffold_factor = None
                if scaffold_bonds_pt and n_scaffold_pt:
                    scaffold_factor = float(
                        scaffold_cfg_meta.get(
                            "scaffold_reconstruction_covalent_factor",
                            1.3,
                        )
                    )
                mol, error_info = reconstruct_molecule(
                    pos,
                    v,
                    atom_mode=atom_mode,
                    debug=debug,
                    scaffold_bonds=scaffold_bonds_pt,
                    n_scaffold=n_scaffold_pt,
                    extra_attachment_bonds=attachment_bonds,
                    covalent_factor=scaffold_factor,
                )
            except Exception as e:
                # 捕获重建过程中的异常
                error_msg = f"重建过程异常: {str(e)}"
                if debug:
                    sys.stdout.write(f"\n  ❌ 分子 {idx + 1} {error_msg}\n")
                    sys.stdout.flush()
                reconstruct_failures.append(
                    {
                        "mol_idx": idx,
                        "error_type": "reconstruct_exception",
                        "error": error_msg,
                    }
                )
                if "reconstruct_exception" not in failure_stats["other_errors"]:
                    failure_stats["other_errors"]["reconstruct_exception"] = 0
                failure_stats["other_errors"]["reconstruct_exception"] += 1
                continue

            if mol is None:
                # 记录重建失败信息
                if error_info:
                    reconstruct_failures.append({"mol_idx": idx, **error_info})
                    # 统计失败类型
                    error_type = error_info["error_type"]
                    if error_type == "MolReconsError":
                        failure_stats["MolReconsError"] += 1
                    elif error_type == "reconstruct_returned_none":
                        failure_stats["reconstruct_returned_none"] += 1
                    else:
                        if error_type not in failure_stats["other_errors"]:
                            failure_stats["other_errors"][error_type] = 0
                        failure_stats["other_errors"][error_type] += 1

                if debug:
                    error_msg = error_info["error_type"] if error_info else "未知错误"
                    sys.stdout.write(
                        f"\n  ❌ 重建失败 ({error_msg})，跳过（不记录，避免数据质量下降）\n"
                    )
                    sys.stdout.flush()
                # 每8个分子提示一次进度
                if (idx + 1) % 8 == 0:
                    sys.stdout.write(
                        f"\n  📊 进度: [{idx + 1}/{num_samples}] 重建成功: {n_reconstruct_success}, 对接成功: {n_eval_success}\n"
                    )
                    sys.stdout.flush()
                    pbar.refresh()  # 刷新进度条
                continue  # 重建失败，跳过，不记录到results、SDF或Excel

            n_reconstruct_success += 1
            reconstructed_this_sample = True

            # Preserve the original reconstruction identity when the existing
            # optional largest-fragment step changes the molecule being scored.
            try:
                reconstructed_smiles_before_fragment_removal = Chem.MolToSmiles(mol)
            except Exception:
                reconstructed_smiles_before_fragment_removal = None

            # 3.1.5 Fragment removal (optional): keep only the largest connected component
            if remove_fragments:
                mol_before = mol
                mol = remove_small_fragments(
                    mol, debug=debug, scaffold_substruct=scaffold_substruct_mol
                )
                if mol is None:
                    if debug:
                        print(
                            f"  ❌ Fragment removal returned None for molecule {idx + 1}, skipping"
                        )
                    continue

            # 3.1.6 激进 medchem 结构优化（默认关闭）
            # 必须在理化性质与 Vina 之前，这样报告的每一项指标描述的都是优化后的分子。
            mol_preopt = mol
            mol, optimize_result = _apply_medchem_optimize(
                mol, idx, protein_path=resolved_protein_path, debug=debug
            )
            if optimize_result is not None:
                optimize_results.append(optimize_result)
                if optimize_result.changed:
                    n_medchem_optimized += 1

            # 3.2 评估分子（默认同进程；可选子进程隔离）
            try:
                # 显示正在评估的分子信息
                if debug or (idx + 1) % 4 == 0:  # 每4个分子显示一次
                    sys.stdout.write(
                        f"\n  🔄 正在评估分子 {idx + 1}/{num_samples}...\n"
                    )
                    sys.stdout.flush()
                    pbar.refresh()

                eval_start_time = time.time()
                eval_result = evaluate_single_molecule_isolated(
                    mol,
                    ligand_filename,
                    protein_root,
                    exhaustiveness=exhaustiveness,
                    n_poses=n_poses,
                    size_factor=size_factor,
                    buffer=buffer,
                    debug=debug,
                    tmp_dir=tmp_dir,  # 传递临时目录参数，确保并行执行时每个任务使用独立目录
                    mol_idx=idx,  # 传递分子索引
                    timeout=single_eval_timeout,  # 仅隔离模式：子进程超时（秒）
                    use_isolation=use_isolation,
                    protein_path=resolved_protein_path,  # 自定义模式：直接指定蛋白路径
                    pre_docking_mmff_minimize=pre_docking_mmff_minimize,
                    pre_docking_mmff_max_iters=pre_docking_mmff_max_iters,
                    pre_docking_use_uff_fallback=pre_docking_use_uff_fallback,
                    pre_docking_etkdg_reembed=pre_docking_etkdg_reembed,
                    vina_modes=_vina_modes,
                )
                if (
                    not _vina_modes
                    and refined_for_pt_vina
                    and idx < len(refined_for_pt_vina)
                ):
                    rec_rc = refined_for_pt_vina[idx]
                    if isinstance(rec_rc, dict):
                        pdet = rec_rc.get("prudent_composite_detail") or {}
                        aff0 = pdet.get("vina_affinity")
                        try:
                            if aff0 is not None and np.isfinite(float(aff0)):
                                eval_result["vina_score_only"] = float(aff0)
                        except (TypeError, ValueError):
                            pass
                eval_elapsed = time.time() - eval_start_time

                # 如果评估时间过长，给出提示
                if eval_elapsed > 60 and debug:
                    sys.stdout.write(
                        f"  ⏱️  分子 {idx + 1} 评估耗时: {eval_elapsed:.1f} 秒\n"
                    )
                    sys.stdout.flush()
                    pbar.refresh()
            except KeyboardInterrupt:
                # 用户中断，保存中间结果后退出
                print(f"\n⚠️  用户中断，正在保存已处理的结果...")
                save_on_exit()
                raise
            except SystemExit:
                # 系统退出，保存中间结果
                save_on_exit()
                raise
            except Exception as e:
                # 捕获评估过程中的所有异常（包括底层C++库的异常）
                error_msg = f"评估异常: {str(e)}"
                error_type = type(e).__name__
                if debug:
                    sys.stdout.write(
                        f"\n  ❌ 分子 {idx + 1} {error_msg} (类型: {error_type})\n"
                    )
                    sys.stdout.flush()
                    traceback.print_exc()

                # 记录失败信息，但继续处理下一个分子
                eval_result = {
                    "mol_idx": idx,
                    "mol": mol,
                    "reconstructed": reconstructed_this_sample,
                    "reconstructed_smiles_before_fragment_removal": reconstructed_smiles_before_fragment_removal,
                    "smiles": None,
                    "chem": None,
                    "vina_dock": None,
                    "vina_score_only": None,
                    "vina_minimize": None,
                    "atom_type_jsd": None,
                    "bond_length_jsd": None,
                    "pair_length_jsd": None,
                    "success": False,
                    "error": error_msg,
                    # 新增评估指标（失败时设为None）
                    "stability": None,
                    "basic_info": None,
                    "logp": None,
                    "lipinski": None,
                    "pains": None,
                    "tanimoto_sim": None,
                    "rdkit_rmsd": None,
                    "conformer_energy": None,
                    "tpsa": None,
                    "rdkit_valid": None,
                    "lilly_medchem_passed": None,
                    "lilly_medchem_demerit": None,
                    "lilly_medchem_description": None,
                    "comprehensive_score": None,
                    "molecule_id": None,
                }

                # 仍然保存这个分子（标记为失败），以便后续分析
                results.append(eval_result)

                # 每8个分子提示一次进度
                if (idx + 1) % 8 == 0:
                    sys.stdout.write(
                        f"\n  📊 进度: [{idx + 1}/{num_samples}] 重建成功: {n_reconstruct_success}, 对接成功: {n_eval_success}\n"
                    )
                    sys.stdout.flush()
                    pbar.refresh()

                # 每 N 个分子保存一次中间结果（防止丢失太多数据）
                if (
                    save_intermediate_interval > 0
                    and (idx + 1) % save_intermediate_interval == 0
                ):
                    try:
                        current_vina_dock_scores = [
                            r["vina_dock"][0]["affinity"]
                            for r in results
                            if r["success"]
                            and r.get("vina_dock")
                            and len(r["vina_dock"]) > 0
                        ]
                        current_vina_score_only_scores = [
                            r["vina_score_only"]
                            for r in results
                            if r["success"] and r.get("vina_score_only") is not None
                        ]
                        current_vina_minimize_scores = [
                            r["vina_minimize"]
                            for r in results
                            if r["success"] and r.get("vina_minimize") is not None
                        ]
                        current_qed_values = [
                            r["chem"]["qed"] for r in results if r.get("chem")
                        ]
                        current_sa_values = [
                            r["chem"]["sa"] for r in results if r.get("chem")
                        ]
                        current_atom_type_jsd_values = [
                            r["atom_type_jsd"]
                            for r in results
                            if r.get("atom_type_jsd") is not None
                        ]

                        # 计算当前的重建失败数
                        current_n_reconstruct_fail = num_samples - n_reconstruct_success

                        save_intermediate_results(
                            eval_output_dir,
                            eval_timestamp,
                            None,
                            pt_path,
                            ligand_filename,
                            protein_root,
                            atom_mode,
                            exhaustiveness,
                            num_samples,
                            n_reconstruct_success,
                            n_eval_success,
                            n_complete,
                            data_id,
                            results,
                            current_vina_dock_scores,
                            current_vina_score_only_scores,
                            current_vina_minimize_scores,
                            current_qed_values,
                            current_sa_values,
                            current_atom_type_jsd_values,
                            {},
                            {},
                            current_n_reconstruct_fail,
                            failure_stats,
                            reconstruct_failures,
                            is_final=False,
                        )
                    except Exception as e:
                        if debug:
                            print(f"  ⚠️  保存中间结果失败: {e}")
                        pass  # 保存失败不影响主流程

                continue  # 继续处理下一个分子

            # 3.2.5 medchem 优化的溯源信息与（可选的）优化前对照评估
            if optimize_result is not None and optimize_result.changed:
                eval_result["medchem_optimize"] = {
                    "transform_chain": optimize_result.transform_chain,
                    "transform_tiers": ">".join(
                        a.tier for a in optimize_result.applied
                    ),
                    "total_reward": optimize_result.total_reward,
                    "preopt_smiles": _safe_heavy_smiles(mol_preopt),
                    "qed_before": optimize_result.properties_before.get("qed"),
                    "qed_after": optimize_result.properties_after.get("qed"),
                    "sa_before": optimize_result.properties_before.get("sa"),
                    "sa_after": optimize_result.properties_after.get("sa"),
                }
                if keep_preopt_sdf == "evaluated":
                    eval_result["preopt_eval"] = _evaluate_preopt_molecule(
                        mol_preopt,
                        idx,
                        ligand_filename,
                        protein_root,
                        exhaustiveness=exhaustiveness,
                        n_poses=n_poses,
                        size_factor=size_factor,
                        buffer=buffer,
                        debug=debug,
                        tmp_dir=tmp_dir,
                        timeout=single_eval_timeout,
                        use_isolation=use_isolation,
                        protein_path=resolved_protein_path,
                        pre_docking_mmff_minimize=pre_docking_mmff_minimize,
                        pre_docking_mmff_max_iters=pre_docking_mmff_max_iters,
                        pre_docking_use_uff_fallback=pre_docking_use_uff_fallback,
                        pre_docking_etkdg_reembed=pre_docking_etkdg_reembed,
                        vina_modes=_vina_modes,
                    )

            if eval_result["success"]:
                n_eval_success += 1

                # 对接成功，立即提示并输出评分（优先显示dock模式的评分）
                vina_score = None
                if eval_result.get("vina_dock") and len(eval_result["vina_dock"]) > 0:
                    vina_score = eval_result["vina_dock"][0]["affinity"]
                elif eval_result.get("vina_minimize") is not None:
                    vina_score = eval_result["vina_minimize"]
                elif eval_result.get("vina_score_only") is not None:
                    vina_score = eval_result["vina_score_only"]

                if vina_score is not None:
                    # 使用sys.stdout.write确保输出不被tqdm覆盖
                    sys.stdout.write(
                        f"\n  ✅ 分子 {idx + 1} 对接成功！评分: {vina_score:.3f} kcal/mol\n"
                    )
                    sys.stdout.flush()
                    pbar.refresh()  # 刷新进度条

                # 检查是否是完整分子（非片段）
                if eval_result["smiles"] and "." not in eval_result["smiles"]:
                    n_complete += 1
            else:
                # 对接失败，也显示（但只在debug模式或每8个时显示）
                if debug or (idx + 1) % 8 == 0:
                    error_msg = eval_result.get("error") or "未知错误"
                    if error_msg == "未知错误":
                        if (
                            not eval_result.get("vina_dock")
                            and eval_result.get("vina_score_only") is None
                            and eval_result.get("vina_minimize") is None
                        ):
                            error_msg = "所有对接模式均失败"
                    err_disp = (
                        error_msg if isinstance(error_msg, str) else str(error_msg)
                    )[:500]
                    sys.stdout.write(f"\n  ❌ 分子 {idx + 1} 对接失败: {err_disp}\n")
                    sys.stdout.flush()
                    pbar.refresh()  # 刷新进度条

            # 每8个分子提示一次进度
            if (idx + 1) % 8 == 0:
                sys.stdout.write(
                    f"\n  📊 进度: [{idx + 1}/{num_samples}] 重建成功: {n_reconstruct_success}, 对接成功: {n_eval_success}, 完整分子: {n_complete}\n"
                )
                sys.stdout.flush()
                pbar.refresh()  # 刷新进度条

            # 3.3 计算综合评分和生成分子身份证
            try:
                comprehensive_score = calculate_comprehensive_score(eval_result)
                eval_result["comprehensive_score"] = comprehensive_score

                # 生成分子身份证：蛋白质ID+生成时间+分子评分
                # 从ligand_filename或protein_filename中提取蛋白质ID
                protein_id = extract_protein_id(
                    ligand_filename=ligand_filename, protein_filename=protein_filename
                )
                generation_time = datetime.now()
                molecule_id = generate_molecule_id(
                    protein_id, generation_time, comprehensive_score
                )
                id_suffix = os.environ.get("EVAL_MOLECULE_ID_SUFFIX", "").strip()
                if id_suffix:
                    molecule_id = f"{molecule_id}_{id_suffix}_{idx:04d}"
                # dynamic_then_optimization 模式下，优化变体追加 _opN_from_{原分子id}，from_id 必须源于 dynamic 结束时的分子
                if use_split_sdf_dirs and idx < len(meta_records):
                    rec = meta_records[idx]
                    # 与 2c optimization 一致：原分子（含 dynamic 原始条 is_dynamic_raw）登记 id，变体追加 _opN_from_
                    is_orig_like = rec.get("is_original", False) or rec.get(
                        "is_dynamic_raw", False
                    )
                    if is_orig_like:
                        dyn_idx = rec.get("source_dynamic_idx")
                        if dyn_idx is not None:
                            dynamic_id_to_molecule_id[dyn_idx] = molecule_id
                    else:
                        dyn_idx = rec.get("source_dynamic_idx")
                        op_num = 1 + sum(
                            1
                            for j in range(idx)
                            if j < len(meta_records)
                            and meta_records[j].get("source_dynamic_idx") == dyn_idx
                            and not (
                                meta_records[j].get("is_original", False)
                                or meta_records[j].get("is_dynamic_raw", False)
                            )
                        )
                        from_id = dynamic_id_to_molecule_id.get(
                            dyn_idx, f"dyn{dyn_idx}" if dyn_idx is not None else "dyn?"
                        )
                        molecule_id = f"{molecule_id}_op{op_num}_from_{from_id}"
                eval_result["molecule_id"] = molecule_id

                if debug:
                    print(f"  ✅ 综合模型评分: {comprehensive_score:.2f}")
                    print(f"  ✅ 分子身份证: {molecule_id}")
            except Exception as e:
                if debug:
                    print(f"  ⚠️  计算评分或生成身份证失败: {e}")
                eval_result["comprehensive_score"] = 0.0
                # 如果提取失败，使用UNKNOWN作为蛋白质ID
                protein_id = extract_protein_id(
                    ligand_filename=ligand_filename, protein_filename=protein_filename
                )
                eval_result["molecule_id"] = (
                    f"{protein_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_0p00"
                )

            # 3.4 保存SDF文件（仅保存成功重建的分子，重建失败的不保存）
            # 优先使用预处理后的分子（MMFF/ETKDG 优化后键长合理化），否则使用原始重建分子
            mol_to_save = eval_result.get("mol") or mol
            # 不保存 comprehensive_score=0 的分子（通常表示评估失败，与 selector 筛选一致）
            # 但当 Vina 未运行时（score=0 且无 Vina 数据），仍保存分子
            comp_score = eval_result.get("comprehensive_score")
            _has_vina = bool(
                eval_result.get("vina_dock")
                or eval_result.get("vina_score_only") is not None
                or eval_result.get("vina_minimize") is not None
            )
            skip_save_low_score = (
                comp_score is not None and comp_score <= 0 and _has_vina
            )
            # 根据模式选择保存目录：分存原分子/优化变体 或 统一 reconstructed_molecules
            target_sdf_dir = None
            if save_sdf and mol_to_save is not None and not skip_save_low_score:
                if use_split_sdf_dirs and idx < len(meta_records):
                    rec = meta_records[idx]
                    is_orig_like = rec.get("is_original", False) or rec.get(
                        "is_dynamic_raw", False
                    )
                    target_sdf_dir = (
                        sdf_dir_original if is_orig_like else sdf_dir_optimized
                    )
                else:
                    target_sdf_dir = sdf_dir
            if save_sdf and target_sdf_dir and mol_to_save is not None:
                try:
                    # 使用分子身份证作为文件名
                    molecule_id = eval_result.get("molecule_id", f"molecule_{idx:04d}")
                    sdf_path = str(target_sdf_dir / f"{molecule_id}.sdf")
                    writer = Chem.SDWriter(sdf_path)

                    # 添加属性信息（使用预处理后的分子，键长已合理化）
                    if eval_result.get("smiles"):
                        mol_to_save.SetProp("SMILES", eval_result["smiles"])
                    if eval_result.get("chem"):
                        mol_to_save.SetProp(
                            "QED", f"{eval_result['chem'].get('qed', 0):.3f}"
                        )
                        mol_to_save.SetProp(
                            "SA", f"{eval_result['chem'].get('sa', 0):.3f}"
                        )
                    if (
                        eval_result.get("vina_dock")
                        and len(eval_result["vina_dock"]) > 0
                    ):
                        mol_to_save.SetProp(
                            "Vina_Dock",
                            f"{eval_result['vina_dock'][0]['affinity']:.3f}",
                        )
                    if eval_result.get("vina_score_only") is not None:
                        mol_to_save.SetProp(
                            "Vina_ScoreOnly", f"{eval_result['vina_score_only']:.3f}"
                        )
                    if eval_result.get("vina_minimize") is not None:
                        mol_to_save.SetProp(
                            "Vina_Minimize", f"{eval_result['vina_minimize']:.3f}"
                        )
                    mol_to_save.SetProp("Molecule_Index", str(idx))
                    if eval_result.get("comprehensive_score") is not None:
                        mol_to_save.SetProp(
                            "Comprehensive_Score",
                            f"{eval_result['comprehensive_score']:.2f}",
                        )
                    if eval_result.get("molecule_id"):
                        mol_to_save.SetProp("Molecule_ID", eval_result["molecule_id"])

                    writer.write(mol_to_save)
                    writer.close()

                    if debug:
                        print(f"  💾 已保存SDF: {molecule_id}.sdf")

                    # 优化前结构（同名文件，便于逐一对照）
                    if preopt_sdf_dir is not None and eval_result.get(
                        "medchem_optimize"
                    ):
                        _write_preopt_sdf(
                            preopt_sdf_dir / f"{molecule_id}.sdf",
                            mol_preopt,
                            molecule_id,
                            idx,
                            eval_result,
                        )

                    # 根据综合评分分类并复制文件
                    comprehensive_score = eval_result.get("comprehensive_score")
                    if comprehensive_score is not None:
                        try:
                            # 确定评分分类
                            category = None
                            if 65 <= comprehensive_score < 70:
                                category = "65-70"
                            elif 70 <= comprehensive_score < 80:
                                category = "70-80"
                            elif comprehensive_score >= 80:
                                category = "80+"

                            if category and category in score_category_dirs:
                                target_dir = score_category_dirs[category]

                                # 复制SDF文件到分类文件夹
                                target_sdf_path = (
                                    target_dir / "molecules" / f"{molecule_id}.sdf"
                                )
                                shutil.copy2(sdf_path, target_sdf_path)

                                # 复制对应的蛋白质文件
                                if protein_filename:
                                    # 查找蛋白质文件路径
                                    protein_path = None
                                    # 尝试多个可能的路径
                                    possible_protein_paths = [
                                        Path(protein_root) / protein_filename,
                                        Path(protein_root)
                                        / Path(ligand_filename).parent
                                        / protein_filename,
                                    ]
                                    # 如果protein_filename是相对路径，尝试从ligand_filename推断
                                    if (
                                        "/" in ligand_filename
                                        or "\\" in ligand_filename
                                    ):
                                        ligand_dir = Path(ligand_filename).parent
                                        possible_protein_paths.append(
                                            Path(protein_root)
                                            / ligand_dir
                                            / protein_filename
                                        )
                                        # 尝试从ligand文件名推断protein文件名
                                        ligand_stem = Path(ligand_filename).stem
                                        if "_ligand" in ligand_stem:
                                            protein_stem = ligand_stem.replace(
                                                "_ligand", "_rec"
                                            )
                                            possible_protein_paths.append(
                                                Path(protein_root)
                                                / ligand_dir
                                                / f"{protein_stem}.pdb"
                                            )

                                    for ppath in possible_protein_paths:
                                        if ppath.exists():
                                            protein_path = ppath
                                            break

                                    if protein_path and protein_path.exists():
                                        # 提取蛋白质文件名（不含路径）
                                        protein_basename = protein_path.name
                                        target_protein_path = (
                                            target_dir / "proteins" / protein_basename
                                        )
                                        # 如果目标文件不存在，才复制（避免重复复制相同的蛋白质文件）
                                        if not target_protein_path.exists():
                                            shutil.copy2(
                                                protein_path, target_protein_path
                                            )

                                        if debug:
                                            print(
                                                f"  📋 已复制到{category}分类: SDF和蛋白质文件"
                                            )
                                    else:
                                        if debug:
                                            print(f"  ⚠️  未找到蛋白质文件，仅复制SDF")
                                else:
                                    if debug:
                                        print(f"  ⚠️  无蛋白质文件名信息，仅复制SDF")

                                if debug:
                                    print(
                                        f"  ✅ 已分类到{category}文件夹: {molecule_id} (评分: {comprehensive_score:.2f})"
                                    )
                        except Exception as e:
                            if debug:
                                print(f"  ⚠️  分类复制文件失败: {e}")

                except Exception as e:
                    if debug:
                        print(f"  ⚠️  保存SDF失败: {e}")

            # 3.5 保存结果（仅保存成功重建的分子）
            # 注意：重建失败的分子不会被记录，避免数据质量下降
            results.append(
                {
                    "mol_idx": idx,
                    "mol": mol,
                    "reconstructed": reconstructed_this_sample,
                    "smiles": eval_result.get("smiles"),
                    "reconstructed_smiles_before_fragment_removal": reconstructed_smiles_before_fragment_removal,
                    "error": eval_result.get("error"),
                    "chem": eval_result.get("chem"),
                    "vina_dock": eval_result.get("vina_dock"),
                    "vina_score_only": eval_result.get("vina_score_only"),
                    "vina_minimize": eval_result.get("vina_minimize"),
                    "atom_type_jsd": eval_result.get("atom_type_jsd"),
                    "bond_length_jsd": eval_result.get("bond_length_jsd"),
                    "pair_length_jsd": eval_result.get("pair_length_jsd"),
                    "success": eval_result["success"],
                    "comprehensive_score": eval_result.get("comprehensive_score"),
                    "molecule_id": eval_result.get("molecule_id"),
                    # 新增评估指标
                    "stability": eval_result.get("stability"),
                    "basic_info": eval_result.get("basic_info"),
                    "logp": eval_result.get("logp"),
                    "lipinski": eval_result.get("lipinski"),
                    "pains": eval_result.get("pains"),
                    "tanimoto_sim": eval_result.get("tanimoto_sim"),
                    "rdkit_rmsd": eval_result.get("rdkit_rmsd"),
                    "conformer_energy": eval_result.get("conformer_energy"),
                    "tpsa": eval_result.get("tpsa"),
                    "rdkit_valid": eval_result.get("rdkit_valid"),
                    "lilly_medchem_passed": eval_result.get("lilly_medchem_passed"),
                    "lilly_medchem_demerit": eval_result.get("lilly_medchem_demerit"),
                    "lilly_medchem_description": eval_result.get(
                        "lilly_medchem_description"
                    ),
                    "medchem_optimize": eval_result.get("medchem_optimize"),
                    "preopt_eval": eval_result.get("preopt_eval"),
                }
            )

            # 每16个分子保存一次中间结果（防止丢失太多数据）
            if (
                save_intermediate_interval > 0
                and (idx + 1) % save_intermediate_interval == 0
            ):
                try:
                    current_vina_dock_scores = [
                        r["vina_dock"][0]["affinity"]
                        for r in results
                        if r["success"]
                        and r.get("vina_dock")
                        and len(r["vina_dock"]) > 0
                    ]
                    current_vina_score_only_scores = [
                        r["vina_score_only"]
                        for r in results
                        if r["success"] and r.get("vina_score_only") is not None
                    ]
                    current_vina_minimize_scores = [
                        r["vina_minimize"]
                        for r in results
                        if r["success"] and r.get("vina_minimize") is not None
                    ]
                    current_qed_values = [
                        r["chem"]["qed"] for r in results if r.get("chem")
                    ]
                    current_sa_values = [
                        r["chem"]["sa"] for r in results if r.get("chem")
                    ]
                    current_atom_type_jsd_values = [
                        r["atom_type_jsd"]
                        for r in results
                        if r.get("atom_type_jsd") is not None
                    ]

                    # 计算当前的重建失败数
                    current_n_reconstruct_fail = num_samples - n_reconstruct_success

                    save_intermediate_results(
                        eval_output_dir,
                        eval_timestamp,
                        None,
                        pt_path,
                        ligand_filename,
                        protein_root,
                        atom_mode,
                        exhaustiveness,
                        num_samples,
                        n_reconstruct_success,
                        n_eval_success,
                        n_complete,
                        data_id,
                        results,
                        current_vina_dock_scores,
                        current_vina_score_only_scores,
                        current_vina_minimize_scores,
                        current_qed_values,
                        current_sa_values,
                        current_atom_type_jsd_values,
                        {},
                        {},
                        current_n_reconstruct_fail,
                        failure_stats,
                        reconstruct_failures,
                        is_final=False,
                    )
                except Exception as e:
                    if debug:
                        print(f"  ⚠️  保存中间结果失败: {e}")
                    pass  # 保存失败不影响主流程

        except KeyboardInterrupt:
            # 用户中断，保存中间结果后退出
            print(f"\n⚠️  用户中断，正在保存已处理的结果...")
            save_on_exit()
            raise
        except SystemExit:
            # 系统退出，保存中间结果
            save_on_exit()
            raise
        except Exception as e:
            # 捕获循环级别的异常（理论上不应该到达这里，因为每个分子都有异常处理）
            error_msg = f"处理分子 {idx + 1} 时发生未预期的异常: {str(e)}"
            print(f"\n❌ {error_msg}")
            traceback.print_exc()

            # 记录失败信息
            eval_result = {
                "mol_idx": idx,
                "mol": mol,
                "reconstructed": reconstructed_this_sample,
                "reconstructed_smiles_before_fragment_removal": reconstructed_smiles_before_fragment_removal,
                "smiles": None,
                "chem": None,
                "vina_dock": None,
                "vina_score_only": None,
                "vina_minimize": None,
                "atom_type_jsd": None,
                "bond_length_jsd": None,
                "pair_length_jsd": None,
                "success": False,
                "error": error_msg,
            }
            results.append(eval_result)

            # 每8个分子提示一次进度
            if (idx + 1) % 8 == 0:
                sys.stdout.write(
                    f"\n  📊 进度: [{idx + 1}/{num_samples}] 重建成功: {n_reconstruct_success}, 对接成功: {n_eval_success}\n"
                )
                sys.stdout.flush()
                pbar.refresh()

            # 继续处理下一个分子
            continue

    return EvaluatedMolecules(
        failure_stats=failure_stats,
        n_complete=n_complete,
        n_eval_success=n_eval_success,
        n_medchem_optimized=n_medchem_optimized,
        n_reconstruct_success=n_reconstruct_success,
        optimize_results=optimize_results,
        reconstruct_failures=reconstruct_failures,
        results=results,
    )
