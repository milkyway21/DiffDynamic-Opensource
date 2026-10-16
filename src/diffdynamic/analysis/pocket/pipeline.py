"""Pipeline responsibilities extracted from evaluate_pocket_quality.py."""

from diffdynamic.analysis.pocket.state import (
    GEO_EPS,
    LAYER_WEIGHTS_COMPAT,
    LAYER_WEIGHTS_LIGAND,
    LAYER_WEIGHTS_POCKET,
    Path,
)


def evaluate_pocket_quality(
    pt_path=None,
    ligand_path=None,
    custom_pocket_pdb=None,
    vina_outputs_dir=None,
    vina_pocket_id=None,
    protein_root=None,
    data_id=None,
    atom_mode="add_aromatic",
    weight_a=0.45,
    weight_b=0.25,
    weight_c=0.30,
    weight_d=0.45,
    weight_e=0.55,
    weight_f=0.20,
    weight_g=0.55,
    weight_h=0.25,
    idea_b_atom_coord_subset="hetero_heavy",
    idea_b_eps=1.5,
    idea_b_edge_max_heavy_neighbors=2,
    idea_b_combined_focus_weight=0.5,
    visualize=False,
    vis_dir=None,
    use_tsne=False,
    fpocket_protein_pdb=None,
    fpocket_cmd="fpocket",
    fpocket_pocket_index=1,
    fpocket_max_ligand_models=50,
    fpocket_timeout=600,
    fpocket_optimal_min=None,
    fpocket_optimal_max=None,
    fpocket_zero_below=None,
    fpocket_zero_above=None,
    idea_h_ligand_path=None,
    idea_e_expected_n_molecules=None,
):
    """
    综合评估口袋质量（八维三层）。

    字母槽：A Aff+LE / B Mode / C Drug / D Unique / E Complete / F Chem / G Anchor / H Geom。
    层：ligand=ABC，compatibility=DE，pocket=FGH。
    层内加权几何平均；层间 S_overall=(Sp·Sc·Sl)^(1/3)。

    weight_* 为层内相对权重（默认见 LAYER_WEIGHTS_*）；失败维剔除后重归一。
    """
    from diffdynamic.analysis.pocket.affinity import (
        evaluate_idea_a_vina,
        evaluate_idea_c_ligand_efficiency,
    )
    from diffdynamic.analysis.pocket.aggregation import (
        _equal_geo_mean,
        _merge_affinity_le,
        _weighted_geo_mean,
    )
    from diffdynamic.analysis.pocket.chemistry import (
        evaluate_idea_d_druglikeness,
        evaluate_idea_e_reconstruction,
        evaluate_idea_f_uniqueness,
    )
    from diffdynamic.analysis.pocket.clustering import evaluate_idea_b_clustering
    from diffdynamic.analysis.pocket.inputs import (
        load_molecules_from_ligand_paths,
        load_molecules_from_pt,
        resolve_receptor_pdb_from_pt,
        resolve_vina_outputs_dir,
    )
    from diffdynamic.analysis.pocket.plots_summary import generate_all_visualizations
    from diffdynamic.analysis.pocket.pocket_metrics import (
        evaluate_idea_f_pocket_chemistry,
        evaluate_idea_g_anchor_richness,
    )
    from diffdynamic.analysis.pocket.volume import evaluate_idea_h_pocket_size

    pp = Path(pt_path) if pt_path else None
    lp = Path(ligand_path) if ligand_path else None

    if pp is not None and pp.exists() and pp.is_file():
        molecules = load_molecules_from_pt(pp, atom_mode=atom_mode)
        eval_tag = data_id
        evaluation_source = "pt"
    elif lp is not None and lp.exists():
        molecules = load_molecules_from_ligand_paths(lp)
        eval_tag = data_id if data_id is not None else "custom"
        evaluation_source = "ligand_file"
        pp = None
    else:
        hint = pt_path or ligand_path
        return {
            "error": f"未找到有效的 .pt 或配体路径: {hint}",
            "overall_score": 0.0,
            "overall_label": "unknown",
        }

    if not molecules:
        return {
            "error": "未能从 .pt 或配体文件解析出带 3D 构象的分子",
            "overall_score": 0.0,
            "overall_label": "unknown",
        }

    molecules_for_h = molecules
    idea_h_ligand_path_resolved = None
    idea_h_ligand_override_used = False
    if idea_h_ligand_path and str(idea_h_ligand_path).strip():
        ihp = Path(idea_h_ligand_path).expanduser().resolve()
        if ihp.exists():
            alt_h = load_molecules_from_ligand_paths(
                ihp,
                max_mols=int(fpocket_max_ligand_models)
                if fpocket_max_ligand_models
                else None,
            )
            if alt_h:
                molecules_for_h = alt_h
                idea_h_ligand_path_resolved = str(ihp)
                idea_h_ligand_override_used = True

    vod = resolve_vina_outputs_dir(
        vina_outputs_dir=vina_outputs_dir,
        pt_path=str(pp) if pp is not None else None,
        ligand_path=str(lp) if lp is not None else None,
        custom_pocket_pdb=custom_pocket_pdb,
    )
    if vod is None and pp is not None and pp.is_file():
        vod = pp.parent

    # Prefer explicit fpocket pdb; else custom pocket pdb; else resolve from .pt
    pocket_pdb = fpocket_protein_pdb or custom_pocket_pdb
    if not pocket_pdb and pp is not None:
        pocket_pdb = resolve_receptor_pdb_from_pt(pp, protein_root=protein_root)
        if pocket_pdb:
            print(f"Auto-resolved receptor PDB from .pt -> {pocket_pdb}", flush=True)

    idea_a_vina = evaluate_idea_a_vina(
        pt_path=str(pp) if pp is not None else None,
        data_id=eval_tag,
        outputs_dir=str(vod) if vod is not None else None,
        pocket_id_override=vina_pocket_id,
    )
    idea_b = evaluate_idea_b_clustering(
        molecules,
        eps=idea_b_eps,
        atom_coord_subset=idea_b_atom_coord_subset,
        edge_max_heavy_neighbors=idea_b_edge_max_heavy_neighbors,
        combined_focus_weight=idea_b_combined_focus_weight,
    )
    vina_for_le = idea_a_vina.get("vina_scores") if idea_a_vina.get("success") else []
    idea_le = evaluate_idea_c_ligand_efficiency(molecules, vina_for_le)
    idea_a = _merge_affinity_le(idea_a_vina, idea_le)
    idea_c = evaluate_idea_d_druglikeness(molecules)  # slot C = druglikeness
    idea_d = evaluate_idea_f_uniqueness(molecules)  # slot D = uniqueness
    idea_e = evaluate_idea_e_reconstruction(
        str(pp) if pp is not None else None,
        molecules,
        expected_n_molecules=idea_e_expected_n_molecules,
    )
    idea_f = evaluate_idea_f_pocket_chemistry(
        fpocket_protein_pdb=pocket_pdb,
        molecules_with_pos=molecules,
        centroid_radius=10.0,
    )
    idea_g = evaluate_idea_g_anchor_richness(
        fpocket_protein_pdb=pocket_pdb,
        molecules_with_pos=molecules,
    )
    idea_h = evaluate_idea_h_pocket_size(
        molecules_for_h,
        optimal_min=350,
        optimal_max=750,
        centroid_radius=10.0,
        fpocket_protein_pdb=pocket_pdb,
    )
    if (
        evaluation_source == "pt"
        and idea_h_ligand_path
        and str(idea_h_ligand_path).strip()
    ):
        idea_h = dict(idea_h)
        if idea_h_ligand_override_used:
            idea_h["ligand_fpocket_pose_source"] = "idea_h_ligand_path"
            idea_h["ligand_fpocket_pose_path"] = idea_h_ligand_path_resolved
        else:
            idea_h["ligand_fpocket_pose_source"] = "pt"
            idea_h["ligand_fpocket_pose_path"] = None
            _ihp = Path(idea_h_ligand_path).expanduser().resolve()
            if not _ihp.exists():
                idea_h["ligand_fpocket_override_note"] = (
                    f"--idea_h_ligand_path 不存在: {idea_h_ligand_path}"
                )
            else:
                idea_h["ligand_fpocket_override_note"] = (
                    "已指定 --idea_h_ligand_path 但未解析到带 3D 构象的分子，H 配体 FPocket 仍用 .pt 分子"
                )

    # Aux LE result for LE plots (slot C is druglikeness)
    idea_le_for_viz = idea_le

    dim_map = {
        "A": idea_a,
        "B": idea_b,
        "C": idea_c,
        "D": idea_d,
        "E": idea_e,
        "F": idea_f,
        "G": idea_g,
        "H": idea_h,
    }
    w_map = {
        "A": weight_a,
        "B": weight_b,
        "C": weight_c,
        "D": weight_d,
        "E": weight_e,
        "F": weight_f,
        "G": weight_g,
        "H": weight_h,
    }

    def _layer_score(keys, default_w):
        scores, weights = [], []
        for k in keys:
            res = dim_map[k]
            if res.get("success"):
                scores.append(res.get("score", 0.0))
                weights.append(w_map.get(k, default_w.get(k, 1.0)))
        if not scores:
            return float(GEO_EPS), False
        return _weighted_geo_mean(scores, weights), True

    s_ligand, ok_l = _layer_score(["A", "B", "C"], LAYER_WEIGHTS_LIGAND)
    s_compat, ok_c = _layer_score(["D", "E"], LAYER_WEIGHTS_COMPAT)
    s_pocket, ok_p = _layer_score(["F", "G", "H"], LAYER_WEIGHTS_POCKET)

    layer_scores = []
    if ok_p:
        layer_scores.append(s_pocket)
    if ok_c:
        layer_scores.append(s_compat)
    if ok_l:
        layer_scores.append(s_ligand)
    overall_score = _equal_geo_mean(layer_scores) if layer_scores else 0.0

    if overall_score >= 0.70:
        overall_label = "high"
    elif overall_score >= 0.45:
        overall_label = "medium"
    else:
        overall_label = "low"

    cpp = Path(custom_pocket_pdb).resolve() if custom_pocket_pdb else None

    result = {
        "pt_path": str(pp.resolve()) if pp is not None else None,
        "ligand_path": str(lp.resolve()) if lp is not None else None,
        "custom_pocket_pdb": str(cpp) if cpp is not None else None,
        "evaluation_source": evaluation_source,
        "vina_outputs_dir": str(vod) if vod is not None else None,
        "data_id": eval_tag,
        "n_molecules": len(molecules),
        "idea_a": idea_a,
        "idea_b": idea_b,
        "idea_c": idea_c,
        "idea_d": idea_d,
        "idea_e": idea_e,
        "idea_f": idea_f,
        "idea_g": idea_g,
        "idea_h": idea_h,
        "idea_le": idea_le_for_viz,
        "s_pocket": s_pocket if ok_p else None,
        "s_compatibility": s_compat if ok_c else None,
        "s_ligand": s_ligand if ok_l else None,
        "overall_score": overall_score,
        "overall_label": overall_label,
        "weights": w_map,
        "layer_weights": {
            "ligand": dict(LAYER_WEIGHTS_LIGAND),
            "compatibility": dict(LAYER_WEIGHTS_COMPAT),
            "pocket": dict(LAYER_WEIGHTS_POCKET),
        },
        "visualizations": {},
        "idea_h_ligand_path": idea_h_ligand_path_resolved,
        "idea_h_ligand_override_used": idea_h_ligand_override_used,
        "idea_e_expected_n_molecules": idea_e_expected_n_molecules,
        "vis_dir": str(Path(vis_dir).resolve()) if vis_dir else None,
        "fpocket_protein_pdb": str(Path(pocket_pdb).resolve()) if pocket_pdb else None,
    }

    if visualize and len(molecules) > 0 and vis_dir:
        if data_id is not None:
            prefix = str(data_id)
        elif pp is not None:
            prefix = pp.stem
        else:
            prefix = lp.stem if lp is not None else "custom"
        vis_paths = generate_all_visualizations(
            result,
            molecules,
            vis_dir,
            title_prefix=prefix,
            use_tsne=use_tsne,
        )
        result["visualizations"] = vis_paths
        print(f"Visualization dir -> {result['vis_dir']}", flush=True)

    return result


def print_evaluation_report(result):
    """打印评估报告（含所有维度及丰富指标）"""
    print("\n" + "=" * 70)
    print("蛋白质口袋质量评估报告")
    print("=" * 70)
    if result.get("error"):
        print(f"错误: {result['error']}")
        print("=" * 70 + "\n")
        return
    src = result.get("pt_path") or result.get("ligand_path")
    print(f"数据源: {src or 'N/A'} ({result.get('evaluation_source', 'unknown')})")
    if result.get("custom_pocket_pdb"):
        print(f"自定义口袋 PDB: {result.get('custom_pocket_pdb')}")
    if result.get("vina_outputs_dir"):
        print(f"Vina 输出目录: {result.get('vina_outputs_dir')}")
    print(f"分子数: {result.get('n_molecules', 0)}")
    print()

    a = result.get("idea_a", {})
    print("【想法A】亲和力 + 配体效率（Aff+LE）")
    print(f"  状态: {'成功' if a.get('success') else '失败'}")
    if a.get("success"):
        if a.get("vina_mean") is not None:
            print(
                f"  Vina 平均/中位数/标准差: {a.get('vina_mean'):.2f} / "
                f"{a.get('vina_median'):.2f} / {a.get('vina_std'):.2f} kcal/mol"
            )
            print(
                f"  Vina 最佳/最差: {a.get('vina_best'):.2f} / {a.get('vina_worst'):.2f} kcal/mol"
            )
            print(f"  亲和力良好比例(Vina≤-7): {a.get('vina_pct_good', 0) * 100:.1f}%")
            print(f"  有效分数数量: {a.get('num_scores')}")
        le = a.get("le_component") or result.get("idea_le") or {}
        if le.get("success") and le.get("le_mean") is not None:
            print(
                f"  LE 均值/中位数: {le.get('le_mean'):.4f} / {le.get('le_median'):.4f} "
                f"kcal·mol⁻¹·重原子⁻¹"
            )
        print(f"  质量分数: {a.get('score'):.3f} ({a.get('quality_label')})")
        if a.get("message"):
            print(f"  摘要: {a.get('message')}")
    else:
        print(f"  备注: {a.get('message', '')}")
    print()

    b = result.get("idea_b", {})
    print("【想法B】原子分布聚类（结合模式收敛性）")
    print(f"  状态: {'成功' if b.get('success') else '失败'}")
    if b.get("success"):
        sub = b.get("atom_coord_subset", "all")
        print(f"  DBSCAN 坐标子集: {sub}（全原子质心 KMeans 不变）")
        nad = b.get("n_atoms_dbscan")
        nt = b.get("n_atoms_total")
        if nad is not None and nt is not None:
            print(f"  DBSCAN 用原子数: {nad} / 总原子 {nt}")
        print(
            f"  DBSCAN 簇数 / 噪声点数: {b.get('n_clusters')} / {b.get('n_noise', 'N/A')}"
        )
        sil = b.get("silhouette")
        print(f"  轮廓系数: {f'{sil:.4f}' if sil is not None else 'N/A'}")
        if sub == "combined":
            sa = b.get("score_all_atoms")
            sf = b.get("score_focus")
            w = b.get("combined_focus_weight")
            print(
                f"  分项: 全原子分={sa if sa is not None else 'N/A'}, "
                f"hetero_edge分={sf if sf is not None else 'N/A'}, w={w}"
            )
            print(
                f"  hetero_edge 簇/噪声: {b.get('n_clusters_focus')} / "
                f"{b.get('n_noise_focus', 'N/A')}"
            )
        ki = b.get("kmeans_inertia")
        print(f"  质心 KMeans 惯性: {f'{ki:.2f}' if ki is not None else 'N/A'}")
        print(f"  质量分数: {b.get('score'):.3f} ({b.get('quality_label')})")
    else:
        print(f"  备注: {b.get('message', '')}")
    print()

    c = result.get("idea_c", {})
    print("【想法C】药物相似性 (QED/SA/Lipinski/PAINS)")
    print(f"  状态: {'成功' if c.get('success') else '失败'}")
    if c.get("success"):
        print(
            f"  QED 均值±标准差: {c.get('qed_mean', 0):.3f} ± {c.get('qed_std', 0):.3f}"
        )
        print(
            f"  SA  均值±标准差: {c.get('sa_mean', 0):.3f} ± {c.get('sa_std', 0):.3f}"
        )
        print(f"  Lipinski 合规(0-1): {c.get('lipinski_mean', 0):.2f}")
        print(f"  PAINS 命中率: {c.get('pains_ratio', 0) * 100:.1f}%")
        print(f"  质量分数: {c.get('score'):.3f} ({c.get('quality_label')})")
    else:
        print(f"  备注: {c.get('message', '')}")
    print()

    d = result.get("idea_d", {})
    print("【想法D】分子唯一性（进分）与指纹相异度（参考）")
    print(f"  状态: {'成功' if d.get('success') else '失败'}")
    if d.get("success"):
        print(f"  唯一分子/有效分子: {d.get('n_unique')}/{d.get('n_total')}")
        print(f"  唯一性比例: {d.get('unique_ratio', 0) * 100:.1f}%")
        fd = d.get("fingerprint_dissimilarity", d.get("tanimoto_diversity"))
        print(
            f"  指纹余弦相异度(参考,不进分): {f'{fd:.4f}' if fd is not None else 'N/A'}"
        )
        print(f"  质量分数: {d.get('score'):.3f} ({d.get('quality_label')})")
    else:
        print(f"  备注: {d.get('message', '')}")
    print()

    e = result.get("idea_e", {})
    print("【想法E】完整分子比例（SMILES 无 '.' 断点）")
    print(f"  状态: {'成功' if e.get('success') else '失败'}")
    if e.get("success"):
        ds = e.get("denominator_source")
        if ds == "expected_n_molecules":
            print("  分母来源: 用户指定应生成分子数 (--idea_e_expected_n_molecules)")
        elif ds == "pt_pred_ligand_pos":
            print("  分母来源: .pt 中 pred_ligand_pos 条数")
        nl = e.get("n_loaded_molecules")
        if nl is not None:
            print(f"  已载入 3D 构象数: {nl}")
        print(f"  完整分子数/分母: {e.get('n_success')}/{e.get('n_total')}")
        print(f"  完整分子比例: {e.get('reconstruct_rate', 0) * 100:.1f}%")
        print(f"  质量分数: {e.get('score'):.3f} ({e.get('quality_label')})")
    else:
        print(f"  备注: {e.get('message', '')}")
    print()

    f = result.get("idea_f", {})
    print("【想法F】口袋化学（配体邻域疏水/极性/封闭度，不调用 FPocket）")
    print(f"  状态: {'成功' if f.get('success') else '失败'}")
    if f.get("success"):
        print(
            f"  邻域半径: {f.get('centroid_radius')} Å；"
            f"残基数={f.get('n_residues')}；重原子={f.get('n_heavy_atoms')}"
        )
        print(
            f"  比例: hydro={f.get('hydro_fraction')}, polar={f.get('polar_fraction')}, "
            f"enc_raw={f.get('enclosure_raw')}"
        )
        print(
            f"  mapped: hydro={f.get('druggability_mapped')}, "
            f"polar={f.get('fpocket_score_mapped')}, enc={f.get('hydrophobicity_mapped')}"
        )
        print(f"  质量分数: {f.get('score'):.3f} ({f.get('quality_label')})")
        if f.get("message"):
            print(f"  摘要: {f.get('message')}")
    else:
        print(f"  备注: {f.get('message', '')}")
    print()

    g = result.get("idea_g", {})
    print("【想法G】相互作用锚点丰富度（P4）")
    print(f"  状态: {'成功' if g.get('success') else '失败'}")
    if g.get("success"):
        print(
            f"  richness/coverage/balance: "
            f"{g.get('s_richness'):.3f} / {g.get('s_coverage'):.3f} / {g.get('s_balance'):.3f}"
        )
        print(
            f"  Neff={g.get('n_eff')} / cap={g.get('n_cap'):.0f}, "
            f"clusters={g.get('n_clusters')}, pocket_res={g.get('n_pocket_residues')}"
        )
        ac = g.get("anchor_counts") or {}
        if ac:
            print(f"  counts: {ac}")
        print(f"  质量分数: {g.get('score'):.3f} ({g.get('quality_label')})")
        if g.get("message"):
            print(f"  摘要: {g.get('message')}")
    else:
        print(f"  备注: {g.get('message', '')}")
    print()

    h = result.get("idea_h", {})
    print("【想法H】口袋几何体积（配体质心 10Å 球内 MC 占据，满分带 350–750 Å³）")
    print(f"  状态: {'成功' if h.get('success') else '失败'}")
    if h.get("success"):
        vol = h.get("volume_ang3")
        if vol is not None:
            print(f"  用于评分的体积: {vol:.0f} Å³ ({h.get('volume_method', '')})")
        print(f"  质量分数: {h.get('score'):.3f} ({h.get('quality_label')})")
        if h.get("message"):
            print(f"  摘要: {h.get('message')}")
    else:
        print(f"  备注: {h.get('message', '')}")
    print()

    print("【综合评估】三层几何平均")
    sp, sc, sl = (
        result.get("s_pocket"),
        result.get("s_compatibility"),
        result.get("s_ligand"),
    )
    print(
        f"  S_pocket={f'{sp:.3f}' if sp is not None else 'n/a'}  "
        f"S_compat={f'{sc:.3f}' if sc is not None else 'n/a'}  "
        f"S_ligand={f'{sl:.3f}' if sl is not None else 'n/a'}"
    )
    print(f"  综合质量分数: {result.get('overall_score', 0):.3f}")
    print(f"  质量等级: {result.get('overall_label', 'unknown').upper()}")

    vis = result.get("visualizations", {})
    if vis:
        print(f"\n【可视化图像】")
        for name, path in vis.items():
            print(f"  [{name}] {path}")

    print("=" * 70 + "\n")
