"""Optimization responsibilities extracted from evaluate_pt_with_correct_reconstruct.py."""

from diffdynamic.evaluation.molecule.state import (
    Chem,
    Path,
    REPO_ROOT,
    sys,
)
from diffdynamic.evaluation.molecule import state as _state


def _resolve_repo_path(path_str):
    """Resolve a config path relative to the DiffDynamic root when not absolute."""
    if not path_str:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _load_medchem_optimizer():
    """Import and configure the medchem optimize engine once per process."""
    state = _state._MEDCHEM_OPTIMIZE_STATE
    if state["loaded"]:
        return state
    state["loaded"] = True

    cfg = _state._MEDCHEM_OPTIMIZE_CFG
    if not cfg or not cfg.get("enable"):
        return state

    pkg_src = REPO_ROOT / "rdkit-structure-repair" / "src"
    if pkg_src.is_dir() and str(pkg_src) not in sys.path:
        sys.path.insert(0, str(pkg_src))
    try:
        from structure_repair.config import load_config, load_optimize_config
        from structure_repair.optimize import optimize_molecule
    except ImportError as exc:
        print(f"⚠️  medchem optimize 不可用，已跳过: {exc}")
        return state

    try:
        optimize_config = load_optimize_config(_resolve_repo_path(cfg.get("config")))
    except Exception as exc:
        print(f"⚠️  medchem optimize 配置读取失败，已跳过: {exc}")
        return state
    optimize_config["enable"] = True

    repair_config = {}
    repair_cfg_path = (_state._RDKIT_STRUCTURE_REPAIR_CFG or {}).get("config")
    if repair_cfg_path:
        try:
            repair_config = load_config(_resolve_repo_path(repair_cfg_path))
        except Exception:
            repair_config = {}

    # evaluate_single_molecule_isolated forks the molecule to a subprocess;
    # without this the optimize provenance properties are lost in transit.
    try:
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    except Exception:
        pass

    state["fn"] = optimize_molecule
    state["config"] = optimize_config
    state["repair_config"] = repair_config
    return state


def _safe_heavy_smiles(mol):
    """Canonical SMILES without explicit hydrogens; None when unusable."""
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)))
    except Exception:
        try:
            return Chem.MolToSmiles(mol)
        except Exception:
            return None


def _evaluate_preopt_molecule(
    mol_preopt, mol_idx, ligand_filename, protein_root, **kwargs
):
    """Full evaluation of the pre-optimization molecule for side-by-side rows.

    Only used by ``--keep-preopt-sdf evaluated``; it doubles evaluation cost
    because it reruns docking.
    """
    from diffdynamic.evaluation.molecule.workers import (
        evaluate_single_molecule_isolated,
    )

    if mol_preopt is None:
        return None
    try:
        result = evaluate_single_molecule_isolated(
            mol_preopt, ligand_filename, protein_root, mol_idx=mol_idx, **kwargs
        )
    except Exception:
        return None
    chem = result.get("chem") or {}
    vina_dock = result.get("vina_dock") or []
    return {
        "smiles": result.get("smiles"),
        "qed": chem.get("qed"),
        "sa": chem.get("sa"),
        "logp": result.get("logp"),
        "lipinski": result.get("lipinski"),
        "tpsa": result.get("tpsa"),
        "vina_dock": vina_dock[0]["affinity"] if vina_dock else None,
        "vina_score_only": result.get("vina_score_only"),
        "vina_minimize": result.get("vina_minimize"),
        "success": result.get("success"),
        "mol": result.get("mol") or mol_preopt,
    }


def _write_optimize_audit(csv_path, optimize_results):
    """Full audit trail of the medchem layer: what changed, why, what was rejected."""
    if not optimize_results:
        return
    try:
        from structure_repair.audit import write_optimize_audit_csv

        write_optimize_audit_csv(csv_path, optimize_results)
        print(f"✅ medchem 优化审计已写入: {csv_path}")
    except Exception as exc:
        print(f"⚠️  medchem 优化审计写入失败: {exc}")


def _write_preopt_sdf(sdf_path, mol_preopt, molecule_id, mol_idx, eval_result):
    """Write the pre-optimization structure alongside the optimized one.

    Under ``structure_only`` this carries geometry and provenance but no
    QED/Vina properties, which is what makes that mode free.
    """
    if mol_preopt is None:
        return
    opt_info = eval_result.get("medchem_optimize") or {}
    preopt_eval = eval_result.get("preopt_eval") or {}
    mol_out = preopt_eval.get("mol") or mol_preopt
    try:
        mol_out = Chem.Mol(mol_out)
        mol_out.SetProp("Molecule_ID", str(molecule_id))
        mol_out.SetProp("Molecule_Index", str(mol_idx))
        mol_out.SetProp("Medchem_Stage", "pre_optimization")
        if opt_info.get("preopt_smiles"):
            mol_out.SetProp("SMILES", opt_info["preopt_smiles"])
        if opt_info.get("transform_chain"):
            mol_out.SetProp("Applied_Transforms", opt_info["transform_chain"])
        for prop, key, fmt in (
            ("QED", "qed", "{:.3f}"),
            ("SA", "sa", "{:.3f}"),
            ("Vina_Dock", "vina_dock", "{:.3f}"),
            ("Vina_ScoreOnly", "vina_score_only", "{:.3f}"),
            ("Vina_Minimize", "vina_minimize", "{:.3f}"),
        ):
            value = preopt_eval.get(key)
            if value is not None:
                mol_out.SetProp(prop, fmt.format(float(value)))
        writer = Chem.SDWriter(str(sdf_path))
        writer.write(mol_out)
        writer.close()
    except Exception:
        pass


def _apply_medchem_optimize(mol, molecule_index, protein_path=None, debug=False):
    """Return ``(mol_to_evaluate, optimize_result)``; a no-op when disabled.

    The returned molecule carries ``_dd_*`` provenance properties so downstream
    SDF and Excel writers can report what the optimizer did. They survive the
    fork in ``evaluate_single_molecule_isolated`` thanks to the AllProps pickle
    setting above.
    """
    state = _load_medchem_optimizer()
    if state["fn"] is None or mol is None:
        return mol, None

    try:
        result = state["fn"](
            mol,
            molecule_id=str(molecule_index),
            config=state["config"],
            protein_path=protein_path,
            repair_config=state["repair_config"],
        )
    except Exception as exc:
        if debug:
            print(f"  ⚠️  medchem optimize 异常（分子 {molecule_index}）: {exc}")
        return mol, None

    if not result.changed or result.optimized_mol is None:
        return mol, result

    optimized = result.optimized_mol
    try:
        optimized.SetProp(
            "_dd_preopt_smiles", Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)))
        )
        optimized.SetProp("_dd_opt_transforms", result.transform_chain)
        optimized.SetProp("_dd_opt_reward", f"{result.total_reward:.5f}")
    except Exception:
        pass
    if debug:
        print(
            f"  🧪 medchem optimize: {result.transform_chain} (reward {result.total_reward:+.3f})"
        )
    return optimized, result
