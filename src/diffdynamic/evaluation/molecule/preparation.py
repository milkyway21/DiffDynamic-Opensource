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


from .context import PreparedEvaluation


def prepare_evaluation(
    pt_path,
    protein_root,
    output_dir=None,
    atom_mode="add_aromatic",
    exhaustiveness=8,
    n_poses=9,
    size_factor=1.0,
    buffer=5.0,
    max_samples=None,
    save_sdf=True,
    debug=False,
    tmp_dir=None,
    use_isolation=False,
    save_intermediate_interval=64,
    eval_parallel_workers=None,
    force_pre_docking_mmff=None,
    pre_docking_mmff_max_iters_override=None,
    receptor_pdb=None,
    index_pkl=None,
    data_id_override=None,
    benchmark_ligands_csv=None,
    reference_ligand_rel=None,
    vina_modes=None,
    save_distribution_plots=True,
    remove_fragments=True,
    keep_preopt_sdf="structure_only",
    save_excel=True,
):
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

    data = load_pt_file(pt_path)
    if data is None:
        return None

    # DecompDiff 保存为 list[dict]（pred_pos / pred_v），转为与 TargetDiff 相同的 dict
    meta_decompdiff = {}
    if isinstance(data, list):
        data, meta_decompdiff = decompdiff_list_to_targetdiff_dict(data, Path(pt_path))
        if data is None:
            return None

    if not validate_pt_data(data):
        return None

    if vina_modes is None:
        _vina_modes = _DEFAULT_VINA_MODES
    else:
        _vina_modes = frozenset(vina_modes)
        if _vina_modes and not _vina_modes.issubset(_ALL_VINA_MODES):
            print(
                f"❌ 错误: vina_modes 含非法项: {sorted(_vina_modes - _ALL_VINA_MODES)}"
            )
            return None

    if (
        meta_decompdiff.get("decompdiff_use_add_aromatic")
        and atom_mode != "add_aromatic"
    ):
        print(
            f"  ⚠️  DecompDiff 列表的 pred_v 为 add_aromatic 词表，已将 atom_mode 从 {atom_mode!r} 改为 'add_aromatic'。"
        )
        atom_mode = "add_aromatic"

    # 2. 准备
    pred_positions = data["pred_ligand_pos"]
    pred_atom_types = data["pred_ligand_v"]
    # IPDiff sample_for_pocket 等仅口袋构图时可能无 ligand_filename
    ligand_filename = getattr(data["data"], "ligand_filename", None)
    # 提取蛋白质文件名（如果存在）
    protein_filename = getattr(data["data"], "protein_filename", None)
    if reference_ligand_rel:
        ligand_filename = str(reference_ligand_rel).strip()
        print(
            f"  💡 --reference_ligand 指定参考配体（相对 protein_root）: {ligand_filename}"
        )

    # 从extra_info中提取data_id（如果存在）
    data_id = None
    if "extra_info" in data and "data_id" in data["extra_info"]:
        data_id_raw = data["extra_info"]["data_id"]
        # 确保data_id是整数格式（0-99）
        try:
            data_id = int(data_id_raw)
            if data_id < 0 or data_id > 99:
                print(f"  ⚠️  警告: data_id={data_id} 超出有效范围(0-99)，将使用原值")
        except (ValueError, TypeError):
            print(f"  ⚠️  警告: data_id={data_id_raw} 无法转换为整数，将使用原值")
            data_id = data_id_raw

    # 如果data_id仍然为None，尝试从pt文件名推断（如果文件名包含data_id信息）
    if data_id is None and pt_path is not None:
        pt_filename = Path(pt_path).stem
        # MolForm/TargetDiff：result_58.pt -> 58
        if pt_filename.startswith("result_"):
            parts = pt_filename.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                try:
                    inferred_id = int(parts[1])
                    if 0 <= inferred_id <= 999:
                        data_id = inferred_id
                        print(f"  📝 从文件名推断 data_id={data_id}")
                except ValueError:
                    pass
        # 例如：result_data_5.pt -> 5
        if data_id is None:
            match = re.search(r"data[_\s]*(\d+)", pt_filename, re.IGNORECASE)
            if match:
                try:
                    inferred_id = int(match.group(1))
                    if 0 <= inferred_id <= 99:
                        data_id = inferred_id
                        print(f"  📝 从文件名推断 data_id={data_id}")
                except ValueError:
                    pass

    if data_id is None and pt_path is not None:
        inferred_docs = _infer_data_id_from_docs_layout(pt_path)
        if inferred_docs is not None:
            data_id = inferred_docs
            print(f"  📝 从路径 docs/<id>/ 推断 data_id={data_id}")

    if data_id_override is not None:
        try:
            data_id = int(data_id_override)
            print(f"  📝 使用命令行指定的 data_id={data_id}（覆盖前述推断）")
        except (TypeError, ValueError):
            print(f"  ⚠️  --data_id 无效: {data_id_override!r}，忽略覆盖")

    if benchmark_ligands_csv and data_id is not None and not reference_ligand_rel:
        hit = _benchmark_ligands_csv_lookup(benchmark_ligands_csv, int(data_id))
        if hit:
            lig_rel, _prot_rel = hit
            ligand_filename = lig_rel
            print(
                f"  💡 自测试集 CSV 绑定配体（data_id={data_id}，与 test_set[{data_id}] 一致）: {ligand_filename}"
            )
        else:
            print(
                f"  ⚠️  未在 {benchmark_ligands_csv} 中找到 data_id={data_id}，请检查 CSV"
            )
    elif index_pkl and data_id is not None and not reference_ligand_rel:
        pair = _crossdocked_index_lookup(index_pkl, int(data_id))
        if pair:
            prot_rel, lig_rel = pair
            if not ligand_filename:
                ligand_filename = lig_rel
                print(f"  💡 自 index.pkl 补全 ligand_filename={ligand_filename}")
            if not protein_filename:
                protein_filename = prot_rel
                print(f"  💡 自 index.pkl 补全 protein_filename={protein_filename}")
            print(
                "  ⚠️  提示: 全量 index.pkl 的第 N 行未必等于测试集 data_id=N（如 pocket10 基准请用 --benchmark_ligands_csv）"
            )
        else:
            print(
                f"  ⚠️  无法从 {index_pkl} 读取 data_id={data_id} 的条目（检查索引范围与格式）"
            )

    # 打印data_id信息
    if data_id is not None:
        print(f"  ✅ 数据ID: {data_id} (类型: {type(data_id).__name__})")
    else:
        print(f"  ⚠️  未找到数据ID，将记录为 'N/A'")

    num_samples = len(pred_positions)
    if max_samples:
        num_samples = min(num_samples, max_samples)

    if num_samples == 0:
        print(f"\n⚠️  无可评估样本：pred_ligand_pos 中分子数为 0。")
        print(
            f"   常见原因：采样阶段出错或未写出有效构象（如 GPU OOM、进程中断），仍生成了空的 result_*.pt。"
        )
        print(
            f"   请查看 outputs/sampling_error_<data_id>_*.log 或对应采样日志，修复采样后重新生成 .pt 再评估。"
        )
        return None

    # 从配置提取对接前预处理参数（ETKDG重嵌、MMFF能量最小化、键长合理化）
    pre_docking_mmff_minimize = False  # 默认关闭MMFF，保持与原始TargetDiff一致
    pre_docking_mmff_max_iters = 200
    pre_docking_use_uff_fallback = True
    pre_docking_etkdg_reembed = False

    # 生成评估目录名（使用 generate_eval_dir_name）
    eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir_name = None

    # 尝试从.pt文件的extra_info中获取配置，或从配置文件加载
    config = None
    if "extra_info" in data and "config_backup" in data["extra_info"]:
        config_backup_path = data["extra_info"].get("config_backup")
        if config_backup_path and Path(config_backup_path).exists():
            try:
                config = misc.load_config(config_backup_path)
            except:
                pass

    # 如果无法从extra_info获取，尝试从默认配置文件加载
    if config is None:
        default_config_path = REPO_ROOT / "configs" / "sampling.yml"
        if default_config_path.exists():
            try:
                config = misc.load_config(default_config_path)
            except:
                pass

    # 从配置提取对接前预处理参数（支持 EasyDict 兼容 dict）
    if config is not None:
        try:
            pre_docking_cfg = getattr(
                getattr(config, "sample", None), "pre_docking", None
            )
            if pre_docking_cfg is None and hasattr(config, "sample"):
                sample = config.sample
                pre_docking_cfg = (
                    sample.get("pre_docking")
                    if isinstance(sample, dict)
                    else getattr(sample, "pre_docking", None)
                )
            if pre_docking_cfg is not None:
                pre_docking_mmff_minimize = bool(
                    pre_docking_cfg.get("mmff_minimize", False)
                    if isinstance(pre_docking_cfg, dict)
                    else getattr(pre_docking_cfg, "mmff_minimize", False)
                )
                pre_docking_mmff_max_iters = int(
                    pre_docking_cfg.get("mmff_max_iters", 200)
                    if isinstance(pre_docking_cfg, dict)
                    else getattr(pre_docking_cfg, "mmff_max_iters", 200)
                )
                pre_docking_use_uff_fallback = bool(
                    pre_docking_cfg.get("use_uff_fallback", True)
                    if isinstance(pre_docking_cfg, dict)
                    else getattr(pre_docking_cfg, "use_uff_fallback", True)
                )
                pre_docking_etkdg_reembed = bool(
                    pre_docking_cfg.get("etkdg_reembed", False)
                    if isinstance(pre_docking_cfg, dict)
                    else getattr(pre_docking_cfg, "etkdg_reembed", False)
                )
        except Exception:
            pass

    if force_pre_docking_mmff is True:
        pre_docking_mmff_minimize = True
    elif force_pre_docking_mmff is False:
        pre_docking_mmff_minimize = False
    if pre_docking_mmff_max_iters_override is not None:
        pre_docking_mmff_max_iters = int(pre_docking_mmff_max_iters_override)

    # 如果data_id为None，使用0作为默认值
    eval_data_id = data_id if data_id is not None else 0

    # 生成评估目录名
    if config is not None:
        eval_dir_name = generate_eval_dir_name(
            eval_data_id, config, timestamp=eval_timestamp
        )
    else:
        # 如果无法加载配置，使用简化格式
        eval_dir_name = f"eval_{eval_timestamp}_{eval_data_id}"

    # 创建评估输出目录（在outputs目录下）
    if output_dir:
        # 如果output_dir是绝对路径，直接使用；否则在outputs目录下创建
        if Path(output_dir).is_absolute():
            eval_output_dir = Path(output_dir) / eval_dir_name
        else:
            # 如果output_dir是相对路径，在outputs目录下创建
            pt_path_obj = Path(pt_path)
            outputs_dir = pt_path_obj.parent  # outputs目录
            eval_output_dir = outputs_dir / eval_dir_name
    else:
        # 如果没有指定output_dir，在outputs目录下创建
        pt_path_obj = Path(pt_path)
        outputs_dir = pt_path_obj.parent  # outputs目录
        eval_output_dir = outputs_dir / eval_dir_name

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ 评估输出目录: {eval_output_dir}")

    # 创建SDF输出目录（在评估目录下）
    # dynamic_then_optimization 模式：分 original_molecules 和 optimized_molecules 两个子文件夹
    sdf_dir = None
    sdf_dir_original = None
    sdf_dir_optimized = None
    pt_mode = data.get("mode", "")
    meta = data.get("meta") or {}
    refined_for_pt_vina = meta.get("refined_candidates") or []
    meta_records = meta.get("records", [])

    # ---- scaffold bond extraction for reconstruct_molecule ----
    # Extract complete scaffold bonds from the reference ligand SDF + SMARTS,
    # because data.ligand_bond_index may be incomplete (missing ring bonds).
    scaffold_bonds_pt = None
    n_scaffold_pt = None
    scaffold_cfg_meta = meta.get("scaffold_cfg") or {}
    if scaffold_cfg_meta:
        n_scaffold_pt = scaffold_cfg_meta.get("n_scaffold_atoms")
        scaffold_indices = scaffold_cfg_meta.get("scaffold_indices") or []
        scaffold_smarts = scaffold_cfg_meta.get("scaffold_smarts") or ""
        # scaffold_smarts may not be saved in meta; try config_backup
        if not scaffold_smarts:
            extra_info = data.get("extra_info") or {}
            cfg_backup = extra_info.get("config_backup")
            if cfg_backup:
                try:
                    if isinstance(cfg_backup, dict):
                        sc_backup = (cfg_backup.get("sample") or {}).get("scaffold", {})
                    else:
                        sc_backup = {}
                    scaffold_smarts = sc_backup.get("scaffold_smarts", "") or ""
                except Exception:
                    scaffold_smarts = ""
        raw_data = data.get("data")
        # ligand_filename gives us the path to the reference ligand SDF
        ref_ligand_path = getattr(raw_data, "ligand_filename", None) or ""
        if (
            n_scaffold_pt
            and scaffold_indices
            and ref_ligand_path
            and int(n_scaffold_pt) > 0
        ):
            try:
                from rdkit import Chem as _Chem

                ref_mol = None
                # Try loading the reference ligand SDF
                ref_path = Path(ref_ligand_path)
                if ref_path.exists():
                    suppl = _Chem.SDMolSupplier(str(ref_path), sanitize=False)
                    ref_mol = suppl[0] if len(suppl) > 0 else None
                    if ref_mol is not None:
                        try:
                            _Chem.SanitizeMol(ref_mol)
                        except Exception:
                            pass
                # Also try with protein_root + reference_ligand_rel
                if ref_mol is None and reference_ligand_rel and protein_root:
                    alt_path = Path(protein_root) / reference_ligand_rel
                    if alt_path.exists():
                        suppl = _Chem.SDMolSupplier(str(alt_path), sanitize=False)
                        ref_mol = suppl[0] if len(suppl) > 0 else None
                        if ref_mol is not None:
                            try:
                                _Chem.SanitizeMol(ref_mol)
                            except Exception:
                                pass

                if ref_mol is not None:
                    # Extract scaffold bonds directly from the reference ligand:
                    # any bond whose both endpoints are in scaffold_indices
                    orig_to_gen = {
                        int(orig): gen for gen, orig in enumerate(scaffold_indices)
                    }
                    scaffold_bonds_pt = []
                    for bond in ref_mol.GetBonds():
                        a, b = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
                        if a in orig_to_gen and b in orig_to_gen:
                            bt_val = bond.GetBondType()
                            arom = bond.GetIsAromatic()
                            if arom:
                                order = 1
                            elif bt_val == _Chem.BondType.DOUBLE:
                                order = 2
                            elif bt_val == _Chem.BondType.TRIPLE:
                                order = 3
                            else:
                                order = 1
                            ga, gb = orig_to_gen[a], orig_to_gen[b]
                            scaffold_bonds_pt.append((ga, gb, order, arom))

                if scaffold_bonds_pt:
                    print(
                        f"  🔒 scaffold bonds from SMARTS: {len(scaffold_bonds_pt)} bonds, "
                        f"n_scaffold={n_scaffold_pt}"
                    )
                else:
                    # Fallback: try from ligand_bond_index (may be incomplete)
                    bond_index = getattr(raw_data, "ligand_bond_index", None)
                    bond_type = getattr(raw_data, "ligand_bond_type", None)
                    if bond_index is not None and bond_type is not None:
                        orig_to_gen = {
                            int(orig): gen for gen, orig in enumerate(scaffold_indices)
                        }
                        bi = (
                            bond_index
                            if not torch.is_tensor(bond_index)
                            else bond_index.cpu()
                        )
                        bt_arr = (
                            bond_type
                            if not torch.is_tensor(bond_type)
                            else bond_type.cpu()
                        )
                        seen = set()
                        scaffold_bonds_pt = []
                        for bidx in range(bi.shape[1]):
                            a, b = int(bi[0, bidx]), int(bi[1, bidx])
                            if a in orig_to_gen and b in orig_to_gen:
                                ga, gb = orig_to_gen[a], orig_to_gen[b]
                                key = (min(ga, gb), max(ga, gb))
                                if key in seen:
                                    continue
                                seen.add(key)
                                order = int(bt_arr[bidx])
                                if order == 4:
                                    scaffold_bonds_pt.append((ga, gb, 1, True))
                                else:
                                    scaffold_bonds_pt.append((ga, gb, order))
                        if scaffold_bonds_pt:
                            print(
                                f"  🔒 scaffold bonds from bond_index (fallback): "
                                f"{len(scaffold_bonds_pt)} bonds, n_scaffold={n_scaffold_pt}"
                            )

                if not scaffold_bonds_pt:
                    n_scaffold_pt = None
            except Exception as e:
                print(f"  ⚠️ scaffold bond extraction failed: {e}")
                traceback.print_exc()
                n_scaffold_pt = None
        else:
            n_scaffold_pt = None

    # Build scaffold substructure Mol for fragment preference
    scaffold_substruct_mol = None
    if n_scaffold_pt and scaffold_indices:
        _sc_smarts = scaffold_cfg_meta.get("scaffold_smarts") or ""
        if not _sc_smarts:
            extra_info = data.get("extra_info") or {}
            cfg_backup = extra_info.get("config_backup")
            if cfg_backup:
                try:
                    if isinstance(cfg_backup, dict):
                        sc_backup = (cfg_backup.get("sample") or {}).get("scaffold", {})
                    else:
                        sc_backup = {}
                    _sc_smarts = sc_backup.get("scaffold_smarts", "") or ""
                except Exception:
                    _sc_smarts = ""
        if _sc_smarts:
            try:
                scaffold_substruct_mol = Chem.MolFromSmarts(_sc_smarts)
            except Exception:
                scaffold_substruct_mol = None
        # If no SMARTS, try extracting substructure from reference ligand
        if scaffold_substruct_mol is None:
            raw_data = data.get("data")
            ref_ligand_path = getattr(raw_data, "ligand_filename", None) or ""
            ref_path = Path(ref_ligand_path)
            if not ref_path.exists() and reference_ligand_rel and protein_root:
                ref_path = Path(protein_root) / reference_ligand_rel
            if ref_path.exists():
                try:
                    suppl = Chem.SDMolSupplier(str(ref_path), sanitize=False)
                    ref_mol = suppl[0] if len(suppl) > 0 else None
                    if ref_mol is not None:
                        try:
                            Chem.SanitizeMol(ref_mol)
                        except Exception:
                            pass
                        # Use SMARTS-style match to extract scaffold as query Mol
                        scaffold_smiles = "O=C1CCC(N2Cc3ccccc3C2=O)C(=O)N1"
                        scaffold_substruct_mol = Chem.MolFromSmiles(scaffold_smiles)
                except Exception:
                    pass

    opt_style = meta.get("optimization_style_naming", True)
    use_split_sdf_dirs = (
        pt_mode in ("dynamic_then_optimization", "dynamic_then_scaffold")
        and meta_records
        and opt_style
    )
    if save_sdf:
        if use_split_sdf_dirs:
            sdf_dir_original = eval_output_dir / "original_molecules"
            sdf_dir_optimized = eval_output_dir / "optimized_molecules"
            sdf_dir_original.mkdir(parents=True, exist_ok=True)
            sdf_dir_optimized.mkdir(parents=True, exist_ok=True)
            print(
                f"✅ SDF文件将分存至: {sdf_dir_original.name}/ (原分子/dynamic 原始), "
                f"{sdf_dir_optimized.name}/ (优化或骨架变体)；模式={pt_mode}"
            )
        else:
            sdf_dir = eval_output_dir / "reconstructed_molecules"
            sdf_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ SDF文件将保存至: {sdf_dir}")

    # 创建按评分分类的文件夹（在主路径下）。统一历史重建只需要
    # evaluator 的 SDF/manifest 输出，禁止额外复制副产物；默认行为保持不变。
    skip_score_categories = (
        os.environ.get("DIFFDYNAMIC_SKIP_SCORE_CATEGORIES", "").lower() in _TRUE_VALUES
    )
    score_category_dirs = (
        {}
        if skip_score_categories
        else {
            "65-70": REPO_ROOT / "molecules_score_65-70",
            "70-80": REPO_ROOT / "molecules_score_70-80",
            "80+": REPO_ROOT / "molecules_score_80plus",
        }
    )
    if score_category_dirs:
        for category, dir_path in score_category_dirs.items():
            dir_path.mkdir(parents=True, exist_ok=True)
            # 在每个分类文件夹下创建molecules和proteins子文件夹
            (dir_path / "molecules").mkdir(parents=True, exist_ok=True)
            (dir_path / "proteins").mkdir(parents=True, exist_ok=True)
        print(f"✅ 评分分类文件夹已创建:")
        print(f"   - 65-70分: {score_category_dirs['65-70']}")
        print(f"   - 70-80分: {score_category_dirs['70-80']}")
        print(f"   - 80分以上: {score_category_dirs['80+']}")
    else:
        print("ℹ️  已跳过评分分类副产物目录（DIFFDYNAMIC_SKIP_SCORE_CATEGORIES）")

    # 解析蛋白路径：自定义模式（无配体）时，protein_filename 为绝对路径，需直接传入对接
    resolved_protein_path = None
    if protein_filename:
        pf = Path(protein_filename)
        if pf.is_absolute() and pf.exists():
            resolved_protein_path = str(pf.resolve())
        elif not pf.is_absolute():
            candidate = Path(protein_root) / protein_filename
            if candidate.exists():
                resolved_protein_path = str(candidate.resolve())
            else:
                # CrossDocked：数据里 protein_filename 多为 *_pocket10.pdb，实际在 pocket10 子数据集根下
                pr = Path(protein_root)
                if pr.name == "crossdocked_v1.1_rmsd1.0":
                    alt_root = pr.parent / "crossdocked_v1.1_rmsd1.0_pocket10"
                    if alt_root.is_dir():
                        alt = alt_root / protein_filename
                        if alt.exists():
                            resolved_protein_path = str(alt.resolve())
            # 部分 index 记录为 *_pocket10.pdb，但磁盘上仅有 {PDB}_A_rec.pdb（常见于未同步全量 pocket 文件）
            if not resolved_protein_path and protein_filename:
                rel = Path(protein_filename)
                parent_dir = Path(protein_root) / rel.parent
                base = rel.name
                m = re.match(r"^([0-9][0-9a-zA-Z]{3})", base)
                if m and parent_dir.is_dir():
                    code = m.group(1).lower()
                    for name in (f"{code}_A_rec.pdb", f"{code.upper()}_A_rec.pdb"):
                        alt_c = parent_dir / name
                        if alt_c.exists():
                            resolved_protein_path = str(alt_c.resolve())
                            print(f"  💡 未找到 {base}，改用同目录受体 {name}")
                            break
    if ligand_filename and str(ligand_filename).strip() not in ("", "N/A", "n/a"):
        try:
            merged = resolve_receptor_pdb_for_docking(
                ligand_filename,
                protein_root,
                explicit_protein_path=resolved_protein_path,
            )
            if os.path.isfile(merged):
                resolved_protein_path = merged
        except ValueError:
            pass
    if resolved_protein_path:
        mode_hint = (
            "自定义模式"
            if (
                not ligand_filename
                or ligand_filename == "N/A"
                or Path(ligand_filename).is_absolute()
            )
            else "数据集模式"
        )
        print(f"✅ {mode_hint}：使用蛋白路径 {resolved_protein_path}")
    elif protein_filename:
        print(f"⚠️  蛋白文件 {protein_filename} 不存在或无法解析，对接可能失败")

    if receptor_pdb:
        rp = Path(receptor_pdb).expanduser()
        if rp.is_file():
            resolved_protein_path = str(rp.resolve())
            print(f"✅ --receptor_pdb 指定受体，已覆盖为: {resolved_protein_path}")
        else:
            print(f"⚠️  --receptor_pdb 路径不存在: {rp}，仍使用上文解析的受体（若有）")

    print(f"\n{'=' * 70}")
    print(f"开始评估 {num_samples} 个分子")
    print(f"原子编码模式: {atom_mode}")
    print(f"对接强度: {exhaustiveness}")
    print(f"蛋白根目录: {protein_root}")
    print(f"配体文件名: {ligand_filename}")
    print(f"单分子子进程隔离: {'启用' if use_isolation else '关闭（同进程顺序评估）'}")
    if not _vina_modes:
        print(
            "  ℹ️  Vina 已关闭（--vina-modes none）：不调用 AutoDock Vina；"
            "对 Prudent .pt 从 meta.refined_candidates[*].prudent_composite_detail 注入 "
            "vina_affinity → vina_score_only（写入 SDF 的 Vina_ScoreOnly）。"
        )
        if refined_for_pt_vina and len(refined_for_pt_vina) != num_samples:
            print(
                f"  ⚠️  refined_candidates 条数 ({len(refined_for_pt_vina)}) 与样本数 ({num_samples}) 不一致，"
                f"仅在索引对齐范围内注入。"
            )
    if pre_docking_mmff_minimize or pre_docking_etkdg_reembed:
        parts = []
        if pre_docking_etkdg_reembed:
            parts.append("ETKDGv3重嵌3D")
        if pre_docking_mmff_minimize:
            parts.append("MMFF能量最小化")
        print(
            f"对接前预处理: {'+'.join(parts)} (max_iters={pre_docking_mmff_max_iters}, UFF回退={'是' if pre_docking_use_uff_fallback else '否'})"
        )
    if save_sdf:
        print(f"保存SDF: 是")
    print(f"{'=' * 70}\n")

    # 2.5 对接参考配体（获取 baseline Vina 分数用于对比）
    ref_vina_score = None
    ref_ligand_sdf_path = None
    if ligand_filename and str(ligand_filename).strip() not in ("", "N/A", "n/a"):
        candidate_paths = []
        lf = Path(ligand_filename)
        if lf.is_absolute() and lf.exists():
            candidate_paths.append(str(lf))
        else:
            candidate_paths.append(str(Path(protein_root) / ligand_filename))
            # Also try resolved_protein_path's sibling
            if resolved_protein_path:
                rp = Path(resolved_protein_path)
                sibling = rp.parent / Path(ligand_filename).name
                if sibling.exists():
                    candidate_paths.append(str(sibling))
        for cp in candidate_paths:
            if os.path.exists(cp):
                ref_ligand_sdf_path = cp
                break
    if ref_ligand_sdf_path and resolved_protein_path and "score_only" in _vina_modes:
        print(f"对接参考配体: {ref_ligand_sdf_path}")
        ref_vina_score = dock_reference_ligand(
            ref_ligand_sdf_path,
            resolved_protein_path,
            exhaustiveness=exhaustiveness,
            vina_timeout=20,
        )
        if ref_vina_score is not None:
            print(f"参考配体 Vina score_only: {ref_vina_score:.3f} kcal/mol")
        else:
            print(f"参考配体对接失败，跳过对比")

    return PreparedEvaluation(
        _vina_modes=_vina_modes,
        atom_mode=atom_mode,
        buffer=buffer,
        data=data,
        data_id=data_id,
        debug=debug,
        eval_output_dir=eval_output_dir,
        eval_timestamp=eval_timestamp,
        exhaustiveness=exhaustiveness,
        keep_preopt_sdf=keep_preopt_sdf,
        ligand_filename=ligand_filename,
        meta_records=meta_records,
        n_poses=n_poses,
        n_scaffold_pt=n_scaffold_pt,
        num_samples=num_samples,
        pre_docking_etkdg_reembed=pre_docking_etkdg_reembed,
        pre_docking_mmff_max_iters=pre_docking_mmff_max_iters,
        pre_docking_mmff_minimize=pre_docking_mmff_minimize,
        pre_docking_use_uff_fallback=pre_docking_use_uff_fallback,
        pred_atom_types=pred_atom_types,
        pred_positions=pred_positions,
        protein_filename=protein_filename,
        protein_root=protein_root,
        pt_mode=pt_mode,
        pt_path=pt_path,
        ref_vina_score=ref_vina_score,
        refined_for_pt_vina=refined_for_pt_vina,
        remove_fragments=remove_fragments,
        resolved_protein_path=resolved_protein_path,
        save_distribution_plots=save_distribution_plots,
        save_excel=save_excel,
        save_intermediate_interval=save_intermediate_interval,
        save_sdf=save_sdf,
        scaffold_bonds_pt=scaffold_bonds_pt,
        scaffold_cfg_meta=scaffold_cfg_meta,
        scaffold_substruct_mol=scaffold_substruct_mol,
        score_category_dirs=score_category_dirs,
        sdf_dir=sdf_dir,
        sdf_dir_optimized=sdf_dir_optimized,
        sdf_dir_original=sdf_dir_original,
        size_factor=size_factor,
        tmp_dir=tmp_dir,
        use_isolation=use_isolation,
        use_split_sdf_dirs=use_split_sdf_dirs,
    )
