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


def finalize_evaluation(context, evaluated):
    """Aggregate metrics and write the result formats consumed by the Web UI."""
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
    data = context.data
    data_id = context.data_id
    debug = context.debug
    eval_output_dir = context.eval_output_dir
    eval_timestamp = context.eval_timestamp
    exhaustiveness = context.exhaustiveness
    ligand_filename = context.ligand_filename
    meta_records = context.meta_records
    num_samples = context.num_samples
    protein_root = context.protein_root
    pt_mode = context.pt_mode
    pt_path = context.pt_path
    ref_vina_score = context.ref_vina_score
    save_distribution_plots = context.save_distribution_plots
    save_excel = context.save_excel
    save_sdf = context.save_sdf
    sdf_dir = context.sdf_dir
    failure_stats = evaluated.failure_stats
    n_complete = evaluated.n_complete
    n_eval_success = evaluated.n_eval_success
    n_medchem_optimized = evaluated.n_medchem_optimized
    n_reconstruct_success = evaluated.n_reconstruct_success
    optimize_results = evaluated.optimize_results
    reconstruct_failures = evaluated.reconstruct_failures
    results = evaluated.results

    print(f"\n{'=' * 70}")
    print(f"评估完成！")
    print(f"{'=' * 70}")
    print(f"总样本数: {num_samples}")
    _pct = lambda n, d: (100.0 * n / d) if d else 0.0
    print(
        f"重建成功: {n_reconstruct_success} ({_pct(n_reconstruct_success, num_samples):.1f}%)"
    )
    print(f"评估成功: {n_eval_success} ({_pct(n_eval_success, num_samples):.1f}%)")
    print(f"完整分子: {n_complete} ({_pct(n_complete, num_samples):.1f}%)")

    if optimize_results:
        print(
            f"medchem 优化: {n_medchem_optimized} / {len(optimize_results)} "
            f"({_pct(n_medchem_optimized, len(optimize_results)):.1f}%)"
        )
        _write_optimize_audit(eval_output_dir / "optimize_audit.csv", optimize_results)

    # 4.1 详细的分子重建成功率统计
    n_reconstruct_fail = num_samples - n_reconstruct_success
    print(f"\n{'=' * 70}")
    print(f"分子重建成功率详细统计")
    print(f"{'=' * 70}")
    print(
        f"重建成功: {n_reconstruct_success:>6} / {num_samples:>6} ({_pct(n_reconstruct_success, num_samples):>6.2f}%)"
    )
    print(
        f"重建失败: {n_reconstruct_fail:>6} / {num_samples:>6} ({_pct(n_reconstruct_fail, num_samples):>6.2f}%)"
    )

    if n_reconstruct_fail > 0:
        print(f"\n重建失败原因分类:")
        print(f"  {'失败类型':<30} {'数量':>8} {'占比':>8}")
        print(f"  {'-' * 30} {'-' * 8} {'-' * 8}")

        if failure_stats["MolReconsError"] > 0:
            pct = failure_stats["MolReconsError"] / n_reconstruct_fail * 100
            print(
                f"  {'MolReconsError':<30} {failure_stats['MolReconsError']:>8} {pct:>7.2f}%"
            )

        if failure_stats["reconstruct_returned_none"] > 0:
            pct = failure_stats["reconstruct_returned_none"] / n_reconstruct_fail * 100
            print(
                f"  {'reconstruct返回None':<30} {failure_stats['reconstruct_returned_none']:>8} {pct:>7.2f}%"
            )

        for error_type, count in failure_stats["other_errors"].items():
            pct = count / n_reconstruct_fail * 100
            print(f"  {error_type:<30} {count:>8} {pct:>7.2f}%")

        # 统计失败分子的原子数量分布
        if reconstruct_failures:
            atom_counts = [
                f["atom_count"]
                for f in reconstruct_failures
                if "atom_count" in f and f["atom_count"] > 0
            ]
            if atom_counts:
                print(f"\n失败分子的原子数量统计:")
                print(f"  平均原子数: {np.mean(atom_counts):.1f}")
                print(f"  中位数原子数: {np.median(atom_counts):.1f}")
                print(f"  最小原子数: {np.min(atom_counts)}")
                print(f"  最大原子数: {np.max(atom_counts)}")

    print(f"{'=' * 70}")

    # 如果评估失败，提供诊断信息
    if n_reconstruct_success > 0 and n_eval_success == 0:
        print(f"\n⚠️  警告: 所有分子重建成功，但对接评估失败！")
        print(f"   可能的原因：")
        print(
            f"   1. 生成分子结构异常（Meeko无法转为PDBQT）— 自定义模式可加 --use_dataset_for_pocket 使用数据集配体"
        )
        print(f"   2. AutoDock Vina环境未正确配置（检查: conda activate adt）")
        print(f"   3. 蛋白文件路径不正确（当前: {protein_root}）")
        print(f"   4. 配体文件名: {ligand_filename}")
        print(f"   5. 检查工具是否可用: which vina, which prepare_receptor4.py")
        # 显示前几个失败的错误信息（帮助诊断是 Meeko 还是 Vina 问题）
        error_count = 0
        for r in results:
            if r.get("mol") and "error" in r and error_count < 2:
                print(
                    f"\n   失败分子 {error_count + 1} 的错误: {str(r.get('error') or '未提供错误详情')[:300]}..."
                )
                error_count += 1

    if save_sdf and sdf_dir:
        sdf_count = len([f for f in sdf_dir.glob("*.sdf")])
        print(f"已保存SDF: {sdf_count} 个文件 → {sdf_dir}")

    # 5. 对接评分统计（三种模式）
    vina_dock_scores = []  # dock模式评分
    vina_score_only_scores = []  # score_only模式评分
    vina_minimize_scores = []  # minimize模式评分

    if n_eval_success > 0:
        # 收集dock模式评分
        vina_dock_scores = [
            r["vina_dock"][0]["affinity"]
            for r in results
            if r["success"] and r.get("vina_dock") and len(r["vina_dock"]) > 0
        ]

        # 收集score_only模式评分
        vina_score_only_scores = [
            r["vina_score_only"]
            for r in results
            if r["success"] and r.get("vina_score_only") is not None
        ]

        # 收集minimize模式评分
        vina_minimize_scores = [
            r["vina_minimize"]
            for r in results
            if r["success"] and r.get("vina_minimize") is not None
        ]

        # 5.1 dock模式统计
        if vina_dock_scores:
            print(f"\n{'=' * 70}")
            print(f"AutoDock Vina Dock模式 对接评分统计")
            print(f"{'=' * 70}")
            print(f"均值 (Mean):   {np.mean(vina_dock_scores):>8.3f} kcal/mol")
            print(f"中位数 (Median): {np.median(vina_dock_scores):>8.3f} kcal/mol")
            print(f"标准差 (Std):   {np.std(vina_dock_scores):>8.3f} kcal/mol")
            print(f"最小值 (Min):   {np.min(vina_dock_scores):>8.3f} kcal/mol")
            print(f"最大值 (Max):   {np.max(vina_dock_scores):>8.3f} kcal/mol")
            print(f"{'=' * 70}")

        # 5.2 score_only模式统计
        if vina_score_only_scores:
            print(f"\n{'=' * 70}")
            print(f"AutoDock Vina Score_Only模式 评分统计")
            print(f"{'=' * 70}")
            print(f"均值 (Mean):   {np.mean(vina_score_only_scores):>8.3f} kcal/mol")
            print(
                f"中位数 (Median): {np.median(vina_score_only_scores):>8.3f} kcal/mol"
            )
            print(f"标准差 (Std):   {np.std(vina_score_only_scores):>8.3f} kcal/mol")
            print(f"最小值 (Min):   {np.min(vina_score_only_scores):>8.3f} kcal/mol")
            print(f"最大值 (Max):   {np.max(vina_score_only_scores):>8.3f} kcal/mol")
            print(f"{'=' * 70}")

        # 5.3 minimize模式统计
        if vina_minimize_scores:
            print(f"\n{'=' * 70}")
            print(f"AutoDock Vina Minimize模式 评分统计")
            print(f"{'=' * 70}")
            print(f"均值 (Mean):   {np.mean(vina_minimize_scores):>8.3f} kcal/mol")
            print(f"中位数 (Median): {np.median(vina_minimize_scores):>8.3f} kcal/mol")
            print(f"标准差 (Std):   {np.std(vina_minimize_scores):>8.3f} kcal/mol")
            print(f"最小值 (Min):   {np.min(vina_minimize_scores):>8.3f} kcal/mol")
            print(f"最大值 (Max):   {np.max(vina_minimize_scores):>8.3f} kcal/mol")
            print(f"{'=' * 70}")

        # 5.4 参考配体对比统计
        if ref_vina_score is not None:
            print(f"\n{'=' * 70}")
            print(f"参考配体对比统计（Vina score_only）")
            print(f"{'=' * 70}")
            print(f"参考配体亲和力: {ref_vina_score:.3f} kcal/mol")

            if vina_score_only_scores:
                n_better_score = sum(
                    1 for s in vina_score_only_scores if s < ref_vina_score
                )
                pct_better_score = n_better_score / len(vina_score_only_scores) * 100
                print(
                    f"Score_only 优于参考: {n_better_score}/{len(vina_score_only_scores)} ({pct_better_score:.1f}%)"
                )
                print(f"  生成分子均值: {np.mean(vina_score_only_scores):.3f} kcal/mol")
                print(f"  生成分子最佳: {np.min(vina_score_only_scores):.3f} kcal/mol")

            if vina_dock_scores:
                n_better_dock = sum(1 for s in vina_dock_scores if s < ref_vina_score)
                pct_better_dock = n_better_dock / len(vina_dock_scores) * 100
                print(
                    f"Dock 优于参考: {n_better_dock}/{len(vina_dock_scores)} ({pct_better_dock:.1f}%)"
                )
                print(f"  生成分子均值: {np.mean(vina_dock_scores):.3f} kcal/mol")
                print(f"  生成分子最佳: {np.min(vina_dock_scores):.3f} kcal/mol")

            if vina_minimize_scores:
                n_better_min = sum(
                    1 for s in vina_minimize_scores if s < ref_vina_score
                )
                pct_better_min = n_better_min / len(vina_minimize_scores) * 100
                print(
                    f"Minimize 优于参考: {n_better_min}/{len(vina_minimize_scores)} ({pct_better_min:.1f}%)"
                )
                print(f"  生成分子均值: {np.mean(vina_minimize_scores):.3f} kcal/mol")
                print(f"  生成分子最佳: {np.min(vina_minimize_scores):.3f} kcal/mol")
            print(f"{'=' * 70}")

    # 6. 化学指标统计
    qed_values = []  # 初始化
    sa_values = []  # 初始化
    if n_eval_success > 0:
        qed_values = [r["chem"]["qed"] for r in results if r.get("chem")]
        sa_values = [r["chem"]["sa"] for r in results if r.get("chem")]

        if qed_values:
            print(f"\n化学指标统计:")
            print(
                f"  QED - Mean: {np.mean(qed_values):.3f}, Median: {np.median(qed_values):.3f}"
            )
            print(
                f"  SA  - Mean: {np.mean(sa_values):.3f}, Median: {np.median(sa_values):.3f}"
            )

    # 7. 分子结构指标统计
    atom_type_jsd_values = []  # 原子类型分布JSD
    bond_length_jsd_all = {}  # 键长分布JSD（按键类型）
    pair_length_jsd_all = {}  # 原子对距离分布JSD（按类型）
    aggregate_bond_length_jsd = {}  # 聚合键长JSD（与 evaluate_diffusion 一致）
    aggregate_pair_length_jsd = {}  # 聚合原子对距离JSD（与 evaluate_diffusion 一致）

    if n_eval_success > 0:
        # 收集原子类型分布JSD
        atom_type_jsd_values = [
            r["atom_type_jsd"] for r in results if r.get("atom_type_jsd") is not None
        ]

        # 收集键长分布JSD（按键类型）
        for r in results:
            if r.get("bond_length_jsd"):
                for key, value in r["bond_length_jsd"].items():
                    if value is not None:
                        if key not in bond_length_jsd_all:
                            bond_length_jsd_all[key] = []
                        bond_length_jsd_all[key].append(value)

        # 收集原子对距离分布JSD（按类型）
        for r in results:
            if r.get("pair_length_jsd"):
                for key, value in r["pair_length_jsd"].items():
                    if value is not None:
                        if key not in pair_length_jsd_all:
                            pair_length_jsd_all[key] = []
                        pair_length_jsd_all[key].append(value)

        # 7.0 聚合JSD计算（与 TargetDiff scripts/evaluate_diffusion.py 完全一致）
        # 仅对完整分子（无 '.' 片段）聚合，避免单分子样本过少导致分布稀疏、JSD虚高
        all_bond_dist = []
        all_pair_dist = []
        for r in results:
            mol = r.get("mol")
            if mol is None or not r.get("success"):
                continue
            smiles = r.get("smiles") or ""
            if "." in smiles:  # TargetDiff: 仅 complete 分子（无片段）
                continue
            try:
                bond_dist = eval_bond_length.bond_distance_from_mol(mol)
                all_bond_dist.extend(bond_dist)
                pos = mol.GetConformer().GetPositions()
                elements = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
                pair_dist = eval_bond_length.pair_distance_from_pos_v(pos, elements)
                all_pair_dist.extend(pair_dist)
            except Exception:
                pass
        if all_bond_dist:
            agg_bond_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
            aggregate_bond_length_jsd = eval_bond_length.eval_bond_length_profile(
                agg_bond_profile
            )
            # 保存键长 JSD 分布对比图到评估目录
            if save_distribution_plots and eval_output_dir:
                try:
                    bond_plot_path = eval_output_dir / "bond_length_hist.png"
                    if eval_bond_length.plot_bond_length_hist(
                        agg_bond_profile,
                        metrics=aggregate_bond_length_jsd,
                        save_path=str(bond_plot_path),
                    ):
                        print(f"✅ 键长 JSD 分布对比图已保存: {bond_plot_path}")
                except ImportError as e:
                    print(f"⚠️  无法生成键长分布图 (matplotlib 未安装): {e}")
                except Exception as e:
                    print(f"⚠️  保存键长分布图失败: {e}")
        if all_pair_dist:
            agg_pair_profile = eval_bond_length.get_pair_length_profile(all_pair_dist)
            aggregate_pair_length_jsd = eval_bond_length.eval_pair_length_profile(
                agg_pair_profile
            )
            # 保存 JSD 分布对比图（真实分布 vs 生成分布）到评估目录
            if save_distribution_plots and eval_output_dir:
                try:
                    plot_save_path = eval_output_dir / "pair_dist_hist.png"
                    eval_bond_length.plot_distance_hist(
                        agg_pair_profile,
                        metrics=aggregate_pair_length_jsd,
                        save_path=str(plot_save_path),
                    )
                    print(f"✅ JSD 分布对比图已保存: {plot_save_path}")
                except ImportError as e:
                    print(f"⚠️  无法生成 JSD 分布图 (matplotlib 未安装): {e}")
                except Exception as e:
                    print(f"⚠️  保存 JSD 分布图失败: {e}")

        # 7.1 原子类型分布JSD统计
        if atom_type_jsd_values:
            print(f"\n{'=' * 70}")
            print(f"原子类型分布JSD统计")
            print(f"{'=' * 70}")
            print(f"均值 (Mean):   {np.mean(atom_type_jsd_values):>8.4f}")
            print(f"中位数 (Median): {np.median(atom_type_jsd_values):>8.4f}")
            print(f"标准差 (Std):   {np.std(atom_type_jsd_values):>8.4f}")
            print(f"最小值 (Min):   {np.min(atom_type_jsd_values):>8.4f}")
            print(f"最大值 (Max):   {np.max(atom_type_jsd_values):>8.4f}")
            print(f"{'=' * 70}")

        # 7.2 键长分布JSD统计
        # 优先显示聚合JSD（与 evaluate_diffusion 一致，更可靠）
        if aggregate_bond_length_jsd:
            print(f"\n{'=' * 70}")
            print(f"键长分布JSD统计（聚合，与 evaluate_diffusion 一致）")
            print(f"{'=' * 70}")
            print(f"{'键类型':<20} {'聚合JSD':>10}")
            print(f"{'-' * 20} {'-' * 10}")
            for key, value in sorted(aggregate_bond_length_jsd.items()):
                if value is not None:
                    print(f"{key:<20} {value:>10.4f}")
            print(f"{'=' * 70}")
        if bond_length_jsd_all:
            print(f"\n键长分布JSD（按分子均值，参考）:")
            for key, values in sorted(bond_length_jsd_all.items()):
                print(
                    f"  {key}: 均值={np.mean(values):.4f}, 中位数={np.median(values):.4f}, 样本数={len(values)}"
                )

        # 7.3 原子对距离分布JSD统计
        if aggregate_pair_length_jsd:
            print(f"\n{'=' * 70}")
            print(f"原子对距离分布JSD统计（聚合，与 evaluate_diffusion 一致）")
            print(f"{'=' * 70}")
            print(f"{'类型':<15} {'聚合JSD':>10}")
            print(f"{'-' * 15} {'-' * 10}")
            for key, value in sorted(aggregate_pair_length_jsd.items()):
                if value is not None:
                    print(f"{key:<15} {value:>10.4f}")
            print(f"{'=' * 70}")
        if pair_length_jsd_all:
            print(f"\n原子对距离JSD（按分子均值，参考）:")
            for key, values in sorted(pair_length_jsd_all.items()):
                print(
                    f"  {key}: 均值={np.mean(values):.4f}, 中位数={np.median(values):.4f}, 样本数={len(values)}"
                )

    # 7.4 计算唯一性和内部相似度（批量指标）
    uniqueness = None
    internal_similarity = None

    if n_eval_success > 0:
        # 收集所有有效的SMILES
        valid_smiles = []
        valid_mols = []
        for r in results:
            if (
                r.get("mol") is not None
                and r.get("smiles")
                and r.get("smiles") != "N/A"
            ):
                smiles = r["smiles"]
                if "." not in smiles:  # 排除片段化分子
                    valid_smiles.append(smiles)
                    valid_mols.append(r["mol"])

        if len(valid_smiles) > 0:
            # 计算唯一性：唯一SMILES数量 / 总数量
            unique_smiles = set(valid_smiles)
            uniqueness = (
                len(unique_smiles) / len(valid_smiles) if len(valid_smiles) > 0 else 0
            )
            print(f"\n{'=' * 70}")
            print(f"唯一性统计")
            print(f"{'=' * 70}")
            print(f"总有效分子数: {len(valid_smiles)}")
            print(f"唯一分子数: {len(unique_smiles)}")
            print(f"唯一性 (Uniqueness): {uniqueness:.4f}")
            print(f"{'=' * 70}")

            # 计算内部相似度：所有分子对之间的平均相似度
            if len(valid_mols) > 1:
                try:
                    similarity_pairs = []
                    for i in range(len(valid_mols)):
                        for j in range(i + 1, len(valid_mols)):
                            try:
                                sim = similarity.tanimoto_sim(
                                    valid_mols[i], valid_mols[j]
                                )
                                similarity_pairs.append(sim)
                            except:
                                pass

                    if len(similarity_pairs) > 0:
                        internal_similarity = np.mean(similarity_pairs)
                        print(f"\n{'=' * 70}")
                        print(f"内部相似度统计")
                        print(f"{'=' * 70}")
                        print(f"分子对数量: {len(similarity_pairs)}")
                        print(
                            f"内部相似度 (Internal Similarity): {internal_similarity:.4f}"
                        )
                        print(f"内部相似度中位数: {np.median(similarity_pairs):.4f}")
                        print(f"{'=' * 70}")
                except Exception as e:
                    print(f"⚠️  内部相似度计算失败: {e}")

    # 7. 准备输出数据（无论是否有output_dir都需要创建）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_data = {
        "pt_file": str(pt_path),
        "ligand_filename": ligand_filename,
        "protein_root": protein_root,
        "atom_mode": atom_mode,
        "reconstruction_method": _state._RECONSTRUCTION_METHOD,
        "exhaustiveness": exhaustiveness,
        "num_samples": num_samples,
        "n_reconstruct_success": n_reconstruct_success,
        "n_eval_success": n_eval_success,
        "n_complete": n_complete,
        "data_id": data_id,  # 添加data_id到输出数据
        "results": results,
        "statistics": {
            "vina_dock_scores": vina_dock_scores,
            "vina_score_only_scores": vina_score_only_scores,
            "vina_minimize_scores": vina_minimize_scores,
            "qed_values": qed_values,
            "sa_values": sa_values,
            "atom_type_jsd_values": atom_type_jsd_values,
            "bond_length_jsd_all": bond_length_jsd_all,
            "pair_length_jsd_all": pair_length_jsd_all,
            "aggregate_bond_length_jsd": aggregate_bond_length_jsd,
            "aggregate_pair_length_jsd": aggregate_pair_length_jsd,
            # 新增评估指标统计
            "molecule_stable_values": [
                r.get("stability", {}).get("molecule_stable", False)
                for r in results
                if r.get("stability")
            ],
            "nr_stable_bonds_values": [
                r.get("stability", {}).get("nr_stable_bonds", 0)
                for r in results
                if r.get("stability")
            ],
            "n_atoms_stability_values": [
                r.get("stability", {}).get("n_atoms", 0)
                for r in results
                if r.get("stability")
            ],
            "n_atoms_values": [
                r.get("basic_info", {}).get("n_atoms", 0)
                for r in results
                if r.get("basic_info")
            ],
            "n_bonds_values": [
                r.get("basic_info", {}).get("n_bonds", 0)
                for r in results
                if r.get("basic_info")
            ],
            "n_rings_values": [
                r.get("basic_info", {}).get("n_rings", 0)
                for r in results
                if r.get("basic_info")
            ],
            "weight_values": [
                r.get("basic_info", {}).get("weight", 0)
                for r in results
                if r.get("basic_info")
            ],
            "logp_values": [
                r.get("logp", 0)
                for r in results
                if r.get("logp") != "N/A" and r.get("logp") is not None
            ],
            "tpsa_values": [
                r.get("tpsa", 0)
                for r in results
                if r.get("tpsa") != "N/A" and r.get("tpsa") is not None
            ],
            "lipinski_values": [
                r.get("lipinski", 0)
                for r in results
                if r.get("lipinski") != "N/A" and r.get("lipinski") is not None
            ],
            "pains_values": [
                r.get("pains", False)
                for r in results
                if r.get("pains") != "N/A" and r.get("pains") is not None
            ],
            "tanimoto_sim_values": [
                r.get("tanimoto_sim", 0)
                for r in results
                if r.get("tanimoto_sim") != "N/A" and r.get("tanimoto_sim") is not None
            ],
            "rdkit_rmsd_max_values": [
                r.get("rdkit_rmsd", {}).get("max", 0)
                for r in results
                if r.get("rdkit_rmsd")
                and not np.isnan(r.get("rdkit_rmsd", {}).get("max", np.nan))
            ],
            "rdkit_rmsd_median_values": [
                r.get("rdkit_rmsd", {}).get("median", 0)
                for r in results
                if r.get("rdkit_rmsd")
                and not np.isnan(r.get("rdkit_rmsd", {}).get("median", np.nan))
            ],
            "conformer_energy_values": [
                r.get("conformer_energy", 0)
                for r in results
                if r.get("conformer_energy") != "N/A"
                and r.get("conformer_energy") is not None
            ],
            "rdkit_valid_values": [
                r.get("rdkit_valid", False)
                for r in results
                if r.get("rdkit_valid") is not None
            ],
            "lilly_medchem_passed_values": [
                r.get("lilly_medchem_passed", False)
                for r in results
                if r.get("lilly_medchem_passed") is not None
            ],
            "lilly_medchem_demerit_values": [
                r.get("lilly_medchem_demerit", 0)
                for r in results
                if r.get("lilly_medchem_demerit") is not None
            ],
            "lilly_medchem_description_values": [
                r.get("lilly_medchem_description", "")
                for r in results
                if r.get("lilly_medchem_description") is not None
            ],
            "uniqueness": uniqueness,
            "internal_similarity": internal_similarity,
        },
        "reconstruct_statistics": {
            "n_reconstruct_fail": n_reconstruct_fail,
            "reconstruct_success_rate": n_reconstruct_success / num_samples
            if num_samples > 0
            else 0,
            "failure_stats": failure_stats,
            "reconstruct_failures": reconstruct_failures[:100]
            if len(reconstruct_failures) > 100
            else reconstruct_failures,  # 只保存前100个失败案例
        },
    }

    # 7.1 保存结果文件（保存到评估目录）
    if eval_output_dir:
        # 使用新的保存函数保存最终结果（内部会写 summary.json 并清理中间文件）
        save_intermediate_results(
            eval_output_dir,
            eval_timestamp,
            output_data,
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
            vina_dock_scores,
            vina_score_only_scores,
            vina_minimize_scores,
            qed_values,
            sa_values,
            atom_type_jsd_values,
            bond_length_jsd_all,
            pair_length_jsd_all,
            n_reconstruct_fail,
            failure_stats,
            reconstruct_failures,
            is_final=True,
        )
        # 注：旧版会再写一份内容完全相同的 eval_results_{ts}.pt（纯冗余拷贝），
        # 已移除——所有读取方（server/molecule_ingest、data_manager、batch 脚本）
        # 均优先匹配 eval_results_*final*.pt / eval_results_*.pt 通配。

        # 保存简单的文本报告
        report_file = eval_output_dir / f"eval_report_{eval_timestamp}.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"评估报告\n")
            f.write(f"{'=' * 70}\n")
            f.write(f"输入文件: {pt_path}\n")
            f.write(f"配体文件: {ligand_filename}\n")
            f.write(f"原子编码: {atom_mode}\n")
            f.write(f"对接强度: {exhaustiveness}\n")
            f.write(f"\n统计信息:\n")
            f.write(f"  总样本数: {num_samples}\n")
            _rp = lambda n, d: (100.0 * n / d) if d else 0.0
            f.write(
                f"  重建成功: {n_reconstruct_success} ({_rp(n_reconstruct_success, num_samples):.1f}%)\n"
            )
            f.write(
                f"  评估成功: {n_eval_success} ({_rp(n_eval_success, num_samples):.1f}%)\n"
            )
            f.write(f"  完整分子: {n_complete} ({_rp(n_complete, num_samples):.1f}%)\n")

            # 添加详细的分子重建统计
            f.write(f"\n{'=' * 70}\n")
            f.write(f"分子重建成功率详细统计\n")
            f.write(f"{'=' * 70}\n")
            f.write(
                f"重建成功: {n_reconstruct_success:>6} / {num_samples:>6} ({_rp(n_reconstruct_success, num_samples):>6.2f}%)\n"
            )
            f.write(
                f"重建失败: {n_reconstruct_fail:>6} / {num_samples:>6} ({_rp(n_reconstruct_fail, num_samples):>6.2f}%)\n"
            )

            if n_reconstruct_fail > 0:
                f.write(f"\n重建失败原因分类:\n")
                f.write(f"  {'失败类型':<30} {'数量':>8} {'占比':>8}\n")
                f.write(f"  {'-' * 30} {'-' * 8} {'-' * 8}\n")

                if failure_stats["MolReconsError"] > 0:
                    pct = failure_stats["MolReconsError"] / n_reconstruct_fail * 100
                    f.write(
                        f"  {'MolReconsError':<30} {failure_stats['MolReconsError']:>8} {pct:>7.2f}%\n"
                    )

                if failure_stats["reconstruct_returned_none"] > 0:
                    pct = (
                        failure_stats["reconstruct_returned_none"]
                        / n_reconstruct_fail
                        * 100
                    )
                    f.write(
                        f"  {'reconstruct返回None':<30} {failure_stats['reconstruct_returned_none']:>8} {pct:>7.2f}%\n"
                    )

                for error_type, count in failure_stats["other_errors"].items():
                    pct = count / n_reconstruct_fail * 100
                    f.write(f"  {error_type:<30} {count:>8} {pct:>7.2f}%\n")

                # 统计失败分子的原子数量分布
                if reconstruct_failures:
                    atom_counts = [
                        fail["atom_count"]
                        for fail in reconstruct_failures
                        if "atom_count" in fail and fail["atom_count"] > 0
                    ]
                    if atom_counts:
                        f.write(f"\n失败分子的原子数量统计:\n")
                        f.write(f"  平均原子数: {np.mean(atom_counts):.1f}\n")
                        f.write(f"  中位数原子数: {np.median(atom_counts):.1f}\n")
                        f.write(f"  最小原子数: {np.min(atom_counts)}\n")
                        f.write(f"  最大原子数: {np.max(atom_counts)}\n")

            f.write(f"{'=' * 70}\n")

            # Dock模式统计
            if vina_dock_scores:
                f.write(f"\nAutoDock Vina Dock模式 评分:\n")
                f.write(f"  Mean:   {np.mean(vina_dock_scores):.3f} kcal/mol\n")
                f.write(f"  Median: {np.median(vina_dock_scores):.3f} kcal/mol\n")
                f.write(f"  Std:    {np.std(vina_dock_scores):.3f} kcal/mol\n")
                f.write(f"  Min:    {np.min(vina_dock_scores):.3f} kcal/mol\n")
                f.write(f"  Max:    {np.max(vina_dock_scores):.3f} kcal/mol\n")

            # Score_Only模式统计
            if vina_score_only_scores:
                f.write(f"\nAutoDock Vina Score_Only模式 评分:\n")
                f.write(f"  Mean:   {np.mean(vina_score_only_scores):.3f} kcal/mol\n")
                f.write(f"  Median: {np.median(vina_score_only_scores):.3f} kcal/mol\n")
                f.write(f"  Std:    {np.std(vina_score_only_scores):.3f} kcal/mol\n")
                f.write(f"  Min:    {np.min(vina_score_only_scores):.3f} kcal/mol\n")
                f.write(f"  Max:    {np.max(vina_score_only_scores):.3f} kcal/mol\n")

            # Minimize模式统计
            if vina_minimize_scores:
                f.write(f"\nAutoDock Vina Minimize模式 评分:\n")
                f.write(f"  Mean:   {np.mean(vina_minimize_scores):.3f} kcal/mol\n")
                f.write(f"  Median: {np.median(vina_minimize_scores):.3f} kcal/mol\n")
                f.write(f"  Std:    {np.std(vina_minimize_scores):.3f} kcal/mol\n")
                f.write(f"  Min:    {np.min(vina_minimize_scores):.3f} kcal/mol\n")
                f.write(f"  Max:    {np.max(vina_minimize_scores):.3f} kcal/mol\n")

            if qed_values:
                f.write(f"\n化学指标:\n")
                f.write(f"  QED Mean: {np.mean(qed_values):.3f}\n")
                f.write(f"  SA Mean:  {np.mean(sa_values):.3f}\n")

            # 分子结构指标统计
            if atom_type_jsd_values:
                f.write(f"\n原子类型分布JSD:\n")
                f.write(f"  Mean:   {np.mean(atom_type_jsd_values):.4f}\n")
                f.write(f"  Median: {np.median(atom_type_jsd_values):.4f}\n")
                f.write(f"  Std:    {np.std(atom_type_jsd_values):.4f}\n")

            if bond_length_jsd_all:
                f.write(f"\n键长分布JSD（按键类型）:\n")
                for key, values in sorted(bond_length_jsd_all.items()):
                    f.write(
                        f"  {key}: Mean={np.mean(values):.4f}, Median={np.median(values):.4f}, N={len(values)}\n"
                    )

            if pair_length_jsd_all:
                f.write(f"\n原子对距离分布JSD:\n")
                for key, values in sorted(pair_length_jsd_all.items()):
                    f.write(
                        f"  {key}: Mean={np.mean(values):.4f}, Median={np.median(values):.4f}, N={len(values)}\n"
                    )

        print(f"✅ 报告已保存至: {report_file}")

    # 8. 记录评估结果到Excel（仅记录成功重建和对接的分子）
    # 注意：重建失败的分子不会被记录到Excel，避免数据质量下降
    # save_excel=False（Web 服务端）时跳过——数据已在 .pt/DB 中，
    # 导出改走后端 /api/runs/{id}/export 按需生成，避免每次评估落盘两份 xlsx。
    if save_excel and eval_output_dir and n_eval_success > 0:
        try:
            record_evaluation_results_to_excel(
                results=results,
                meta_records=meta_records,
                pt_mode=pt_mode,
                sampling_meta=data.get("meta"),
                output_dir=str(eval_output_dir),
                pt_file=str(pt_path),
                ligand_filename=ligand_filename,
                atom_mode=atom_mode,
                exhaustiveness=exhaustiveness,
                timestamp=eval_timestamp,
                num_samples=num_samples,
                n_reconstruct_success=n_reconstruct_success,
                n_reconstruct_fail=n_reconstruct_fail,
                n_eval_success=n_eval_success,
                failure_stats=failure_stats,
                ref_vina_score=ref_vina_score,
                vina_modes=_vina_modes,
            )
            # 8.1 记录完整分子到专用Excel（分初次生成、优化后两个表）
            record_complete_molecules_to_excel(
                results=results,
                meta_records=meta_records,
                output_dir=str(eval_output_dir),
                pt_file=str(pt_path),
                ligand_filename=ligand_filename,
                atom_mode=atom_mode,
                exhaustiveness=exhaustiveness,
                timestamp=eval_timestamp,
                num_samples=num_samples,
                n_reconstruct_success=n_reconstruct_success,
                n_complete=n_complete,
            )
        except Exception as e:
            if debug:
                print(f"⚠️  记录Excel失败: {e}")
                traceback.print_exc()

    if eval_output_dir:
        output_data["eval_output_dir"] = str(eval_output_dir.resolve())
    return output_data
