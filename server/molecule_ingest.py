"""Parse evaluation results from .pt and insert molecules into DB.

SDF files are saved to: {output_root}/sdf_store/{pocket_id}/{molecule_id}.sdf
DB stores the SDF path for each molecule.
"""

import glob
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from server.database import db_session, Molecule


OUTPUTS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "DiffDynamic", "outputs")
SDF_STORE = os.path.join(OUTPUTS_ROOT, "sdf_store")


def _find_eval_dir(pt_path: str) -> str:
    """Find the eval directory for a given .pt file. Check eval/ subfolder first."""
    parent = os.path.dirname(pt_path)
    basename = os.path.basename(pt_path)
    # Extract timestamp from result_0_YYYYMMDD_HHMMSS.pt
    m = re.search(r"result_\d+_(\d{8}_\d{6})", basename)
    if not m:
        return parent
    ts = m.group(1)
    # Look in outputs/eval/ for matching eval directory
    eval_root = os.path.join(parent, "eval")
    if not os.path.isdir(eval_root):
        eval_root = os.path.join(os.path.dirname(parent), "eval")
    if os.path.isdir(eval_root):
        for d in sorted(os.listdir(eval_root)):
            if ts in d and d.startswith("eval_"):
                return os.path.join(eval_root, d)
    # Fallback: look in parent for eval_* directories
    for d in sorted(os.listdir(parent)):
        if ts in d and d.startswith("eval_"):
            return os.path.join(parent, d)
    return parent


def ingest_molecules_from_eval(run_id: int, eval_dir: str, pocket_name: Optional[str] = None):
    """Parse eval_results_*.pt and SDF files, insert molecules into DB."""
    # Resolve eval_dir — may be a .pt path, find the actual eval directory
    if eval_dir.endswith(".pt"):
        eval_dir = _find_eval_dir(eval_dir)
    eval_pts = sorted(glob.glob(os.path.join(eval_dir, "eval_results_*final*.pt")))
    if not eval_pts:
        eval_pts = sorted(glob.glob(os.path.join(eval_dir, "eval_results_*.pt")))

    for pt_path in eval_pts:
        try:
            try:
                data = torch.load(pt_path, map_location="cpu", weights_only=True)
            except Exception:
                data = torch.load(pt_path, map_location="cpu")
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        mol_results = data.get("results", data.get("molecule_results", []))
        if not mol_results:
            continue

        # Use provided pocket_name, or extract from eval data
        if pocket_name:
            pocket_id = pocket_name
        else:
            pocket_id = _extract_pocket_id(pt_path, data)
        data_id = _extract_data_id(pt_path)
        gen_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Find existing SDF files from eval output
        sdf_dir = os.path.join(eval_dir, "reconstructed_molecules")
        sdf_files = sorted(glob.glob(os.path.join(sdf_dir, "*.sdf")))

        # Create pocket folder in sdf_store
        pocket_store = os.path.join(SDF_STORE, pocket_id)
        os.makedirs(pocket_store, exist_ok=True)

        with db_session() as s:
            for idx, mol in enumerate(mol_results):
                if not isinstance(mol, dict):
                    continue

                smiles = mol.get("smiles", "")
                vina = _get_vina_score(mol)
                qed = _safe_float(mol.get("chem", {}).get("qed") if mol.get("chem") else None)
                sa = _safe_float(mol.get("chem", {}).get("sa") if mol.get("chem") else None)
                logp = _safe_float(mol.get("logp"))
                tpsa = _safe_float(mol.get("tpsa"))
                score = _safe_float(mol.get("comprehensive_score"))
                lipinski_raw = mol.get("lipinski")
                lipinski = int(lipinski_raw) if lipinski_raw not in (None, 'N/A') else None
                pains = bool(mol.get("pains")) if mol.get("pains") is not None else None
                lilly_passed = mol.get("lilly_medchem_passed")
                lilly_demerit = _safe_int(mol.get("lilly_medchem_demerit"))
                lilly_desc = str(mol.get("lilly_medchem_description", ""))[:200] if mol.get("lilly_medchem_description") else None
                conformer_energy = _safe_float(mol.get("conformer_energy"))
                rdkit_valid = mol.get("rdkit_valid")
                stability = mol.get("stability")
                molecule_stable = stability.get("molecule_stable") if isinstance(stability, dict) else None
                basic_info = mol.get("basic_info")
                n_heavy_atoms = basic_info.get("n_atoms") if isinstance(basic_info, dict) else None
                tanimoto = _safe_float(mol.get("tanimoto_sim"))

                # Molecule name: pocket_proteinid_time_score
                score_str = f"{score:.1f}" if score is not None else "0"
                vina_str = f"{vina:.1f}" if vina is not None else "nodock"
                mol_name = f"{pocket_id}_{gen_time}_{vina_str}"

                # Copy SDF to organized store
                sdf_path = None
                if idx < len(sdf_files):
                    src_sdf = sdf_files[idx]
                    dst_sdf = os.path.join(pocket_store, f"{mol_name}_{idx}.sdf")
                    try:
                        import shutil
                        shutil.copy2(src_sdf, dst_sdf)
                        sdf_path = dst_sdf
                    except Exception:
                        pass

                s.add(Molecule(
                    run_id=run_id,
                    data_id=data_id,
                    pocket_id=pocket_id,
                    molecule_index=idx,
                    smiles=smiles,
                    vina_score=vina,
                    qed=qed,
                    sa=sa,
                    logp=logp,
                    tpsa=tpsa,
                    comprehensive_score=score,
                    lipinski_pass=lipinski,
                    pains_pass=pains,
                    lilly_passed=lilly_passed,
                    lilly_demerit=lilly_demerit,
                    lilly_description=lilly_desc,
                    conformer_energy=conformer_energy,
                    rdkit_valid=rdkit_valid,
                    molecule_stable=molecule_stable,
                    n_heavy_atoms=n_heavy_atoms,
                    tanimoto=tanimoto,
                    sdf_path=sdf_path,
                ))


def _get_vina_score(mol: dict) -> float:
    for mode in ("vina_score_only", "vina_dock", "vina_minimize"):
        val = mol.get(mode)
        if val is not None:
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, dict):
                return float(val.get("affinity", 0))
            if isinstance(val, list) and val:
                return float(val[0].get("affinity", 0)) if isinstance(val[0], dict) else float(val[0])
    return None


def _safe_float(val):
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val):
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _extract_pocket_id(pt_path: str, data: dict) -> str:
    # Try top-level ligand_filename (eval results format)
    lig = data.get("ligand_filename", "")
    if not lig:
        # Try meta.ligand_filename (generation results format)
        meta = data.get("meta", {})
        if isinstance(meta, dict):
            lig = meta.get("ligand_filename", "")
    if lig:
        parts = lig.replace(".sdf", "").replace(".mol2", "")
        return parts.replace("/", "_")[:40]
    basename = os.path.basename(pt_path)
    m = re.search(r"eval_(\d+)_(\w+)", basename)
    if m:
        return f"data{m.group(1)}_{m.group(2)[:20]}"
    return "unknown"


def _extract_data_id(pt_path: str) -> int:
    m = re.search(r"eval_(\d+)_", os.path.basename(pt_path))
    return int(m.group(1)) if m else 0
