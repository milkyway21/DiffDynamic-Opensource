"""Parse evaluation results from .pt and insert molecules into DB.

SDF files are saved to: {output_root}/sdf_store/{pocket_id}/{molecule_id}.sdf
DB stores the SDF path for each molecule. Duplicates (same run_id + smiles) are skipped.
"""

import glob
import json
import logging
import os
import re
import shutil
from datetime import datetime
from typing import Optional

import torch

from server.database import add_molecules, get_run

logger = logging.getLogger(__name__)


OUTPUTS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
SDF_STORE = os.path.join(OUTPUTS_ROOT, "sdf_store")


def _link_or_copy(src: str, dst: str):
    """Hardlink when possible (same filesystem); fall back to copy."""
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    shutil.copy2(src, dst)


def _sdf_store_for_run(run_id: int) -> str:
    """Prefer per-owner sdf_store under outputs/users/{owner}/sdf_store."""
    try:
        from server.auth import normalize_owner, user_sdf_store
        r = get_run(run_id)
        owner = normalize_owner(getattr(r, "owner", None) if r else None)
        return user_sdf_store(OUTPUTS_ROOT, owner)
    except Exception:
        return SDF_STORE


def _trusted_output_roots(owner: Optional[str] = None) -> list:
    """Output roots whose .pt files are produced by this repo's own jobs.

    When the run owner is known, trust only that user's tree
    (outputs/users/{owner}/); the server-wide roots remain as fallback for
    flows without an owner. This keeps one user's outputs from trusting
    another user's tree during ingestion."""
    if owner:
        try:
            from server.auth import user_output_root
            root = user_output_root(OUTPUTS_ROOT, str(owner))
            if os.path.isdir(root):
                return [os.path.realpath(root)]
        except Exception:
            pass
    cfg_output_root = ""
    try:
        from server.config import get_config
        cfg_output_root = get_config().output_root
    except Exception:
        pass
    roots = [cfg_output_root, OUTPUTS_ROOT]
    return [os.path.realpath(r) for r in roots if r and os.path.isdir(r)]


def _load_eval_pt(pt_path: str, owner: Optional[str] = None):
    """Load an eval_results .pt written by this repo's evaluator.

    These files legitimately embed RDKit Mol objects, so strict
    ``weights_only=True`` cannot unpickle them. The untrusted
    ``weights_only=False`` fallback is allowed only for files under
    trusted output roots (produced by our own jobs); anything else —
    e.g. uploads or data dirs — must load strictly or not at all.
    """
    real = os.path.realpath(pt_path)
    trusted = any(os.path.commonpath([real, root]) == root
                  for root in _trusted_output_roots(owner))
    if not trusted:
        from server.safe_pt import load_untrusted_pt
        return load_untrusted_pt(real, owner)
    try:
        return torch.load(pt_path, map_location="cpu", weights_only=True)
    except Exception:
        real = os.path.realpath(pt_path)
        for root in _trusted_output_roots(owner):
            try:
                if os.path.commonpath([real, root]) == root:
                    logger.warning("weights_only fallback for trusted .pt: %s", pt_path)
                    return torch.load(pt_path, map_location="cpu", weights_only=False)
            except ValueError:
                continue
        raise


def _find_eval_dir(pt_path: str) -> str:
    """Find the eval directory for a given .pt file. Check eval/ subfolder first."""
    parent = os.path.dirname(pt_path)
    basename = os.path.basename(pt_path)
    m = re.search(r"result_\d+_(\d{8}_\d{6})", basename)
    if not m:
        return parent
    ts = m.group(1)
    eval_root = os.path.join(parent, "eval")
    if not os.path.isdir(eval_root):
        eval_root = os.path.join(os.path.dirname(parent), "eval")
    if os.path.isdir(eval_root):
        for d in sorted(os.listdir(eval_root)):
            if ts in d and d.startswith("eval_"):
                return os.path.join(eval_root, d)
    for d in sorted(os.listdir(parent)):
        if ts in d and d.startswith("eval_"):
            return os.path.join(parent, d)
    return parent


def _eval_dirs_to_scan(eval_dir: str) -> list:
    """Resolve extract/eval output roots to directories that contain eval_results_*.pt.

    Extraction writes ``extract_<job>/eval_<ts>_.../eval_results_*.pt``; evaluation
    may write the .pt directly in ``eval_<job>/``.
    """
    if eval_dir.endswith(".pt"):
        eval_dir = _find_eval_dir(eval_dir)
    if not eval_dir or not os.path.isdir(eval_dir):
        return []
    found = []

    def _consider(d: str):
        if glob.glob(os.path.join(d, "eval_results_*final*.pt")) or glob.glob(
            os.path.join(d, "eval_results_*.pt")
        ):
            found.append(d)

    _consider(eval_dir)
    try:
        for name in sorted(os.listdir(eval_dir)):
            sub = os.path.join(eval_dir, name)
            if os.path.isdir(sub) and name.startswith("eval_"):
                _consider(sub)
    except OSError:
        pass
    out, seen = [], set()
    for d in found:
        real = os.path.realpath(d)
        if real not in seen:
            seen.add(real)
            out.append(d)
    return out


def _usable_smiles(smiles) -> Optional[str]:
    if smiles is None:
        return None
    text = str(smiles).strip()
    if not text or text.lower() in ("none", "nan", "n/a", "null"):
        return None
    return text


def _collect_sdf_files(eval_dir: str) -> list:
    files = []
    for name in ("reconstructed_molecules", "original_molecules", "optimized_molecules"):
        folder = os.path.join(eval_dir, name)
        if os.path.isdir(folder):
            files.extend(sorted(glob.glob(os.path.join(folder, "*.sdf"))))
    return files


def _read_sdf_props(path: str) -> dict:
    """Read SMILES / Molecule_Index written by the evaluator (text, no RDKit required)."""
    props = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        text = ""
    # RDKit SDWriter: `>  <SMILES>  (1) \nvalue`  — the `(1)` count is optional.
    for match in re.finditer(r">\s*<([^>]+)>\s*(?:\([^)]*\))?\s*\n([^\n]*)", text):
        props[match.group(1).strip()] = match.group(2).strip()
    idx = None
    for key in ("Molecule_Index", "molecule_index", "mol_idx"):
        if key in props:
            try:
                idx = int(float(props[key]))
            except (TypeError, ValueError):
                idx = None
            break
    smiles = _usable_smiles(props.get("SMILES") or props.get("smiles"))
    mol_id = props.get("Molecule_ID") or props.get("molecule_id") or os.path.splitext(os.path.basename(path))[0]
    return {"path": path, "index": idx, "smiles": smiles or "", "molecule_id": mol_id}


def match_sdf_for_molecule(mol_idx, smiles: Optional[str], sdf_records: list, used_paths: set) -> Optional[str]:
    """Pair a result row to an SDF by Molecule_Index, then SMILES. Never glob[idx]."""
    unused = [rec for rec in sdf_records if rec["path"] not in used_paths]
    if mol_idx is not None:
        for rec in unused:
            if rec.get("index") == mol_idx:
                return rec["path"]
    want = _usable_smiles(smiles)
    if want:
        for rec in unused:
            if _usable_smiles(rec.get("smiles")) == want:
                return rec["path"]
    return None


def ingest_molecules_from_eval(run_id: int, eval_dir: str, pocket_name: Optional[str] = None):
    """Parse eval_results_*.pt and SDF files, insert molecules into DB (deduped)."""
    owner = None
    try:
        r = get_run(run_id)
        owner = getattr(r, "owner", None) or None
    except Exception:
        owner = None
    for scan_dir in _eval_dirs_to_scan(eval_dir):
        _ingest_one_eval_dir(run_id, scan_dir, pocket_name=pocket_name, owner=owner)


def _ingest_one_eval_dir(run_id: int, eval_dir: str, pocket_name: Optional[str] = None, owner: Optional[str] = None):
    eval_pts = sorted(glob.glob(os.path.join(eval_dir, "eval_results_*final*.pt")))
    if not eval_pts:
        eval_pts = sorted(glob.glob(os.path.join(eval_dir, "eval_results_*.pt")))

    sdf_records = [_read_sdf_props(path) for path in _collect_sdf_files(eval_dir)]

    for pt_path in eval_pts:
        try:
            data = _load_eval_pt(pt_path, owner=owner)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        mol_results = data.get("results", data.get("molecule_results", []))
        if not mol_results:
            continue

        pocket_id = pocket_name or _extract_pocket_id(pt_path, data)
        data_id = _extract_data_id(pt_path)
        gen_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        pocket_store = os.path.join(_sdf_store_for_run(run_id), pocket_id or "unknown")
        os.makedirs(pocket_store, exist_ok=True)

        used_paths = set()
        mol_list = []
        for idx, mol in enumerate(mol_results):
            if not isinstance(mol, dict):
                continue

            smiles = _usable_smiles(mol.get("smiles"))
            if not smiles:
                continue

            mol_idx = mol.get("mol_idx", mol.get("molecule_index", idx))
            try:
                mol_idx = int(mol_idx)
            except (TypeError, ValueError):
                mol_idx = idx

            vina = _get_vina_score(mol)
            chem = mol.get("chem") or {}
            qed = _safe_float(chem.get("qed"))
            sa = _safe_float(chem.get("sa"))
            logp = _safe_float(mol.get("logp"))
            tpsa = _safe_float(mol.get("tpsa"))
            score = _safe_float(mol.get("comprehensive_score"))
            lipinski_raw = mol.get("lipinski")
            lipinski = int(lipinski_raw) if lipinski_raw not in (None, "N/A") else None
            pains = bool(mol.get("pains")) if mol.get("pains") is not None else None
            lilly_passed = mol.get("lilly_medchem_passed")
            lilly_demerit = _safe_int(mol.get("lilly_medchem_demerit"))
            lilly_desc = (
                str(mol.get("lilly_medchem_description", ""))[:200]
                if mol.get("lilly_medchem_description")
                else None
            )
            conformer_energy = _safe_float(mol.get("conformer_energy"))
            rdkit_valid = mol.get("rdkit_valid")
            stability = mol.get("stability")
            molecule_stable = stability.get("molecule_stable") if isinstance(stability, dict) else None
            basic_info = mol.get("basic_info")
            n_heavy_atoms = basic_info.get("n_atoms") if isinstance(basic_info, dict) else None
            tanimoto = _safe_float(mol.get("tanimoto_sim"))

            vina_str = f"{vina:.1f}" if vina is not None else "nodock"
            mol_name = f"{pocket_id}_{gen_time}_{vina_str}"

            sdf_path = None
            src_sdf = match_sdf_for_molecule(mol_idx, smiles, sdf_records, used_paths)
            if src_sdf:
                used_paths.add(src_sdf)
                dst_sdf = os.path.join(pocket_store, f"{mol_name}_{mol_idx}.sdf")
                try:
                    if not os.path.exists(dst_sdf):
                        _link_or_copy(src_sdf, dst_sdf)
                    sdf_path = dst_sdf
                except Exception:
                    sdf_path = src_sdf if os.path.isfile(src_sdf) else None

            mol_list.append({
                "data_id": data_id,
                "pocket_id": pocket_id,
                "molecule_index": mol_idx,
                "smiles": smiles,
                "vina_score": vina,
                "qed": qed,
                "sa": sa,
                "logp": logp,
                "tpsa": tpsa,
                "comprehensive_score": score,
                "lipinski_pass": lipinski,
                "pains_pass": pains,
                "lilly_passed": lilly_passed,
                "lilly_demerit": lilly_demerit,
                "lilly_description": lilly_desc,
                "conformer_energy": conformer_energy,
                "rdkit_valid": rdkit_valid,
                "molecule_stable": molecule_stable,
                "n_heavy_atoms": n_heavy_atoms,
                "tanimoto": tanimoto,
                "sdf_path": sdf_path,
            })

        if mol_list:
            add_molecules(run_id, mol_list)


def _get_vina_score(mol: dict):
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
    lig = data.get("ligand_filename", "")
    if not lig:
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


def repair_molecule_catalog() -> dict:
    """Delete empty-SMILES rows and re-link recent per-user extract dirs. Never drops the DB."""
    from server.database import Molecule, Run, db_session

    users_marker = f"{os.sep}outputs{os.sep}users{os.sep}"
    deleted = 0
    relinked_runs = []
    with db_session() as session:
        deleted = session.query(Molecule).filter(
            (Molecule.smiles == None) | (Molecule.smiles == "")  # noqa: E711
        ).delete(synchronize_session=False)

        recent = (
            session.query(Run)
            .filter(Run.run_type.in_(["extract", "evaluate", "generate_dynamic", "generate_prudent"]))
            .order_by(Run.id.desc())
            .limit(40)
            .all()
        )
        run_rows = [(r.id, r.output_path, r.parameters) for r in recent]

    for run_id, output_path, parameters in run_rows:
        targets = []
        if output_path:
            targets.append(output_path)
        params = {}
        if parameters:
            try:
                params = json.loads(parameters) if isinstance(parameters, str) else dict(parameters)
            except Exception:
                params = {}
        for key in ("result_path", "eval_dir"):
            val = params.get(key)
            if val:
                targets.append(val)
        chosen = None
        for target in targets:
            if not target or users_marker not in os.path.realpath(target):
                continue
            if os.path.isdir(target) and _eval_dirs_to_scan(target):
                chosen = target
                break
        if not chosen:
            continue
        try:
            ingest_molecules_from_eval(run_id, chosen)
            relinked_runs.append(run_id)
        except Exception:
            logger.warning("repair ingest failed run_id=%s target=%s", run_id, chosen, exc_info=True)
    return {"deleted_empty_smiles": deleted, "relinked_runs": sorted(set(relinked_runs))}
