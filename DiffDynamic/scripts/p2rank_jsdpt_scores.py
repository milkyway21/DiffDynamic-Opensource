#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 jsdpt3010 的受体 + 参考配体跑 P2Rank，按「与配体空间最一致的口袋」取 probability，
并与 fig7/pocketeval1.csv 的 p2rank 列核对。

分数规则与 Desktop ``plot_testset_pocket_triple_scatter.parse_p2rank_predictions_match_ligand`` 一致：
对每个预测口袋球心，算到配体各原子距离的均值，取最小者的 probability（无则 score）。

``--crop-radius`` > 0 时：先按参考配体邻域裁受体再跑 prank（默认 0=整蛋白）。
空预测且 ``--fill-empty-zero`` 时记 p2rank=0.0。

示例::

    python3 scripts/p2rank_jsdpt_scores.py \\
      --pt-dir jsdpt3010 --start 0 --end 99 \\
      --crop-radius 12 --fill-empty-zero \\
      --p2rank-cache outputs/p2rank_jsdpt3010_ligand_cache \\
      --out-audit fig7/p2rank_jsdpt3010_ligand_audit.csv
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pandas as pd
except ImportError:
    pd = None

from evaluate_pocket_quality import load_pt_file, resolve_receptor_pdb_from_pt  # noqa: E402

DESKTOP_ROOT = Path("/home/user/Desktop/Ye/DiffDynamic")
DEFAULT_PRANK = DESKTOP_ROOT / "third_party" / "p2rank" / "distro" / "prank"


def _find_pt(pt_dir: Path, data_id: int) -> Optional[Path]:
    matches = sorted(pt_dir.glob(f"result_{data_id}_*.pt"))
    return matches[-1] if matches else None


def resolve_reference_ligand_sdf(pt_path: Path, protein_root: Optional[Path] = None) -> Optional[Path]:
    """从 .pt 的 ligand_filename 解析 CrossDocked 参考配体 SDF。"""
    try:
        raw = load_pt_file(pt_path)
    except Exception:
        return None
    data = raw.get("data") if isinstance(raw, dict) else None
    ligand_fn = getattr(data, "ligand_filename", None) if data is not None else None
    if isinstance(data, dict):
        ligand_fn = ligand_fn or data.get("ligand_filename")
    if not ligand_fn:
        return None
    lf = Path(str(ligand_fn))
    if lf.is_file():
        return lf.resolve()

    search_roots = []
    if protein_root:
        search_roots.append(Path(protein_root))
    search_roots.extend(
        [
            REPO_ROOT / "data" / "crossdocked_pocket10_test_only",
            REPO_ROOT / "data" / "crossdocked_v1.1_rmsd1.0_pocket10",
            DESKTOP_ROOT / "data" / "crossdocked_v1.1_rmsd1.0_pocket10",
            REPO_ROOT / "data",
        ]
    )
    for root in search_roots:
        if not root.is_dir():
            continue
        cand = (root / lf).resolve()
        if cand.is_file():
            return cand
        cand2 = (root / lf.name).resolve()
        if cand2.is_file():
            return cand2
    return None


def load_ligand_positions(sdf_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (positions Nx3, centroid) from first mol in SDF."""
    try:
        from rdkit import Chem
    except ImportError:
        return None, None
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    mol = None
    for m in suppl:
        if m is not None:
            mol = m
            break
    if mol is None or mol.GetNumConformers() == 0:
        return None, None
    conf = mol.GetConformer()
    pos = np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
         for i in range(mol.GetNumAtoms())],
        dtype=np.float64,
    )
    if pos.size < 3:
        return None, None
    return pos, pos.mean(axis=0)


def _find_predictions_csv(out_dir: Path) -> Optional[Path]:
    """仅在 out_dir 根目录查找（不递归 crop 子目录）。"""
    cands = list(out_dir.glob("*_predictions.csv"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def _crop_work_dir(pocket_dir: Path, crop_radius_ang: float) -> Path:
    s = f"{crop_radius_ang:g}".replace(".", "p").replace("-", "m")
    return pocket_dir / f"p2rank_crop_{s}"


def write_receptor_ligand_vicinity_pdb(
    protein_path: Path,
    ligand_path: Path,
    radius_ang: float,
    out_pdb: Path,
) -> bool:
    """截取参考配体半径内残基写入 PDB，供 P2Rank 在结合位点附近预测。"""
    from utils.data import PDBProtein

    lig_pos, _ = load_ligand_positions(ligand_path)
    if lig_pos is None:
        return False
    try:
        protein = PDBProtein(str(protein_path))
        selected = protein.query_residues_ligand({"pos": lig_pos}, float(radius_ang))
        if not selected:
            return False
        block = protein.residues_to_pdb_block(selected, name="LIG_VICINITY")
        out_pdb.parent.mkdir(parents=True, exist_ok=True)
        out_pdb.write_text(block, encoding="utf-8")
        return True
    except Exception as exc:
        print(f"⚠️ crop failed {protein_path.name}: {exc}", flush=True)
        return False


def parse_p2rank_predictions_match_ligand(
    predictions_csv: Path,
    ligand_center: np.ndarray,
    ligand_positions: Optional[np.ndarray] = None,
) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    if pd is None:
        return None, None, None
    df = pd.read_csv(predictions_csv)
    df.columns = [str(c).strip().lower().lstrip("\ufeff") for c in df.columns]
    for need in ("center_x", "center_y", "center_z"):
        if need not in df.columns:
            return None, None, None
    centers = np.stack(
        [
            df["center_x"].astype(float).values,
            df["center_y"].astype(float).values,
            df["center_z"].astype(float).values,
        ],
        axis=1,
    )
    if centers.shape[0] == 0:
        return None, None, None
    if ligand_positions is not None and ligand_positions.size >= 3:
        lp = np.asarray(ligand_positions, dtype=np.float64).reshape(-1, 3)
        diff = lp[:, None, :] - centers[None, :, :]
        d = np.mean(np.linalg.norm(diff, axis=2), axis=0)
    else:
        d = np.linalg.norm(centers - ligand_center.reshape(1, 3), axis=1)
    j = int(np.argmin(d))
    val = None
    for col in ("probability", "score"):
        if col in df.columns:
            try:
                val = float(df.iloc[j][col])
                break
            except Exception:
                val = None
    return val, float(d[j]), j


def run_p2rank_predict(protein_pdb: Path, out_dir: Path, prank: Path, threads: int = 4) -> bool:
    """直接调用 prank，避免 Desktop wrapper 与本仓 utils 包名冲突。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(prank),
        "predict",
        "-f",
        str(protein_pdb),
        "-o",
        str(out_dir),
        "-visualizations",
        "0",
        "-threads",
        str(threads),
    ]
    try:
        r = subprocess.run(
            cmd,
            check=False,
            cwd=str(prank.parent),
            env=os.environ,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            err = (r.stderr or r.stdout or "").strip()
            print(f"⚠️ P2Rank failed {protein_pdb.name}: rc={r.returncode} {err[-500:]}", flush=True)
            return False
        return True
    except (FileNotFoundError, OSError) as e:
        print(f"⚠️ P2Rank failed {protein_pdb.name}: {e}", flush=True)
        return False


def _summary_path(cache_root: Path, data_id: int) -> Path:
    return cache_root / f"pocket_{data_id}" / "p2rank_summary.json"


def _eval_one(args: tuple) -> dict[str, Any]:
    (
        data_id,
        pt_path,
        cache_root,
        run_prank,
        prank_path,
        threads,
        protein_root,
        crop_radius,
        fill_empty_zero,
    ) = args
    out: dict[str, Any] = {
        "pocket_id": int(data_id),
        "pt_path": str(pt_path) if pt_path else "",
        "protein": "",
        "ligand": "",
        "ok": False,
        "message": "",
        "p2rank": np.nan,
        "match_dist": np.nan,
        "pocket_row": -1,
        "pred_protein": "",
        "crop_radius": float(crop_radius),
        "from_cache": False,
    }
    if not pt_path or not Path(pt_path).is_file():
        out["message"] = "missing_pt"
        return out

    cache_root = Path(cache_root)
    pocket_dir = cache_root / f"pocket_{data_id}"
    spath = _summary_path(cache_root, data_id)
    use_crop = float(crop_radius) > 0
    if spath.is_file():
        try:
            loaded = json.loads(spath.read_text(encoding="utf-8"))
            cache_crop = float(loaded.get("crop_radius", 0) or 0)
            msg = str(loaded.get("message") or "")
            if (
                loaded.get("ok")
                and loaded.get("p2rank") is not None
                and abs(cache_crop - float(crop_radius)) < 1e-9
                and "empty" not in msg.lower()
                and "fill" not in msg.lower()
            ):
                out.update(
                    {
                        "ok": True,
                        "from_cache": True,
                        "message": str(loaded.get("message") or "ok"),
                        "protein": loaded.get("protein", ""),
                        "ligand": loaded.get("ligand", ""),
                        "p2rank": float(loaded["p2rank"]),
                        "match_dist": float(loaded.get("match_dist", np.nan)),
                        "pocket_row": int(loaded.get("pocket_row", -1)),
                        "pred_protein": loaded.get("pred_protein", ""),
                    }
                )
                return out
        except Exception:
            pass

    protein = resolve_receptor_pdb_from_pt(pt_path, protein_root=protein_root)
    ligand = resolve_reference_ligand_sdf(
        Path(pt_path), protein_root=Path(protein_root) if protein_root else None
    )
    if not protein:
        out["message"] = "protein_resolve_failed"
        return out
    if not ligand:
        out["message"] = "ligand_resolve_failed"
        out["protein"] = protein
        return out
    out["protein"] = str(Path(protein).resolve())
    out["ligand"] = str(Path(ligand).resolve())

    lig_pos, lig_cent = load_ligand_positions(Path(ligand))
    if lig_cent is None:
        out["message"] = "ligand_coords_failed"
        return out

    work_dir = _crop_work_dir(pocket_dir, float(crop_radius)) if use_crop else pocket_dir
    pred = _find_predictions_csv(work_dir)

    if pred is None and run_prank:
        work_dir.mkdir(parents=True, exist_ok=True)
        for stale in work_dir.glob("*_predictions.csv"):
            try:
                stale.unlink()
            except Exception:
                pass
        run_pdb = Path(protein)
        if use_crop:
            vicinity = work_dir / f"{Path(protein).stem}_lig_vicinity_{float(crop_radius):g}A.pdb"
            if not write_receptor_ligand_vicinity_pdb(
                Path(protein), Path(ligand), float(crop_radius), vicinity
            ):
                out["message"] = "crop_failed"
                if fill_empty_zero:
                    out["ok"] = True
                    out["p2rank"] = 0.0
                    out["message"] = "crop_failed_filled_zero"
                payload = {
                    "ok": out["ok"],
                    "message": out["message"],
                    "protein": out["protein"],
                    "ligand": out["ligand"],
                    "p2rank": out["p2rank"] if out["ok"] else None,
                    "crop_radius": float(crop_radius),
                    "pt_path": out["pt_path"],
                }
                pocket_dir.mkdir(parents=True, exist_ok=True)
                spath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                return out
            run_pdb = vicinity
        ok = run_p2rank_predict(run_pdb, work_dir, Path(prank_path), threads=int(threads))
        if ok:
            pred = _find_predictions_csv(work_dir)

    if pred is None:
        out["message"] = "empty_predictions"
        if fill_empty_zero:
            out["ok"] = True
            out["p2rank"] = 0.0
            out["message"] = "empty_predictions_filled_zero"
        payload = {
            "ok": out["ok"],
            "message": out["message"],
            "protein": out["protein"],
            "ligand": out["ligand"],
            "p2rank": out["p2rank"] if out["ok"] else None,
            "crop_radius": float(crop_radius),
            "pt_path": out["pt_path"],
        }
        pocket_dir.mkdir(parents=True, exist_ok=True)
        spath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    out["pred_protein"] = pred.name.replace("_predictions.csv", "")
    val, dist, row = parse_p2rank_predictions_match_ligand(pred, lig_cent, ligand_positions=lig_pos)
    if val is None:
        out["message"] = "empty_predictions"
        if fill_empty_zero:
            out["ok"] = True
            out["p2rank"] = 0.0
            out["message"] = "empty_predictions_filled_zero"
    else:
        out["ok"] = True
        out["message"] = "ok"
        out["p2rank"] = float(val)
        out["match_dist"] = float(dist) if dist is not None else np.nan
        out["pocket_row"] = int(row) if row is not None else -1

    payload = {
        "ok": out["ok"],
        "message": out["message"],
        "protein": out["protein"],
        "ligand": out["ligand"],
        "p2rank": out["p2rank"] if out["ok"] else None,
        "match_dist": out["match_dist"] if np.isfinite(out["match_dist"]) else None,
        "pocket_row": out["pocket_row"],
        "pred_protein": out["pred_protein"],
        "crop_radius": float(crop_radius),
        "pt_path": out["pt_path"],
    }
    pocket_dir.mkdir(parents=True, exist_ok=True)
    spath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="jsdpt3010 P2Rank 评分核对")
    ap.add_argument("--pt-dir", type=Path, default=REPO_ROOT / "jsdpt3010")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=99)
    ap.add_argument("--protein-root", type=Path, default=None)
    ap.add_argument(
        "--p2rank-cache",
        type=Path,
        default=REPO_ROOT / "outputs" / "p2rank_jsdpt3010_cache",
    )
    ap.add_argument("--run-p2rank", dest="run_p2rank", action="store_true", default=True)
    ap.add_argument("--no-run-p2rank", dest="run_p2rank", action="store_false")
    ap.add_argument("--prank", type=Path, default=DEFAULT_PRANK)
    ap.add_argument("--threads-per-job", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--crop-radius",
        type=float,
        default=0.0,
        help=">0 时按配体邻域裁受体再跑 prank（Å）",
    )
    ap.add_argument(
        "--fill-empty-zero",
        action="store_true",
        help="空预测 / 解析失败时记 p2rank=0.0",
    )
    ap.add_argument("--compare-csv", type=Path, default=REPO_ROOT / "fig7" / "pocketeval1.csv")
    ap.add_argument("--out-audit", type=Path, default=REPO_ROOT / "fig7" / "p2rank_jsdpt3010_audit.csv")
    ap.add_argument(
        "--out-corrected",
        type=Path,
        default=REPO_ROOT / "fig7" / "pocketeval1_p2rank_jsdpt_corrected.csv",
    )
    args = ap.parse_args()

    if pd is None:
        print("需要 pandas", file=sys.stderr)
        return 2
    if not args.prank.is_file():
        print(f"❌ prank 不存在: {args.prank}", file=sys.stderr)
        return 2
    if shutil.which("java") is None:
        print("❌ 未找到 java", file=sys.stderr)
        return 2

    pt_dir = args.pt_dir.resolve()
    tasks = []
    for i in range(int(args.start), int(args.end) + 1):
        pt = _find_pt(pt_dir, i)
        tasks.append(
            (
                i,
                str(pt) if pt else "",
                str(args.p2rank_cache.resolve()),
                bool(args.run_p2rank),
                str(args.prank.resolve()),
                int(args.threads_per_job),
                str(args.protein_root.resolve()) if args.protein_root else None,
                float(args.crop_radius),
                bool(args.fill_empty_zero),
            )
        )

    print(
        f"P2Rank on {len(tasks)} pockets (workers={args.workers}, "
        f"crop={args.crop_radius}, fill0={args.fill_empty_zero}, cache={args.p2rank_cache})",
        flush=True,
    )
    results = []
    n_workers = max(1, int(args.workers))
    if n_workers == 1:
        for t in tasks:
            r = _eval_one(t)
            results.append(r)
            print(
                f"  [{r['pocket_id']}] ok={r['ok']} p2={r['p2rank']} "
                f"dist={r['match_dist']} prot={Path(r['protein']).name if r['protein'] else '-'} "
                f"({'cache' if r['from_cache'] else 'run'}) {r['message']}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_eval_one, t): t[0] for t in tasks}
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                print(
                    f"  [{r['pocket_id']}] ok={r['ok']} p2={r['p2rank']} "
                    f"dist={r['match_dist']} prot={Path(r['protein']).name if r['protein'] else '-'} "
                    f"({'cache' if r['from_cache'] else 'run'}) {r['message']}",
                    flush=True,
                )

    results = sorted(results, key=lambda x: int(x["pocket_id"]))
    df = pd.DataFrame(results)

    if args.compare_csv.is_file():
        old = pd.read_csv(args.compare_csv)
        n = min(len(old), len(df))
        df["csv_p2rank_old"] = old["p2rank"].astype(float).values[:n]
        if len(df) > n:
            # pad
            pass
        df["abs_diff"] = (df["p2rank"] - df["csv_p2rank_old"]).abs()
    else:
        df["csv_p2rank_old"] = np.nan
        df["abs_diff"] = np.nan

    out_audit = args.out_audit.resolve()
    out_audit.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_audit, index=False)

    if args.compare_csv.is_file():
        base = pd.read_csv(args.compare_csv).copy()
        n = min(len(base), len(df))
        base = base.iloc[:n].copy()
        base["p2rank"] = df["p2rank"].astype(float).values[:n]
        cols = [c for c in ("this_work", "p2rank", "fpocket", "Sitemap") if c in base.columns]
        out_corr = args.out_corrected.resolve()
        base[cols].to_csv(out_corr, index=False)
        print(f"Corrected draft -> {out_corr}")

    ok_n = int(df["ok"].sum())
    finite = df["p2rank"].to_numpy(dtype=float)
    print("\n=== Summary ===")
    print(f"ok: {ok_n}/{len(df)}")
    print(f"finite: {int(np.isfinite(finite).sum())}/{len(df)}")
    if np.isfinite(finite).any():
        print(
            f"p2rank mean±std: {np.nanmean(finite):.4f} ± {np.nanstd(finite):.4f} "
            f"[{np.nanmin(finite):.4f}, {np.nanmax(finite):.4f}]"
        )
    if np.isfinite(df["abs_diff"]).any():
        ad = df["abs_diff"].to_numpy(dtype=float)
        print(
            f"vs old CSV: MAE={np.nanmean(ad):.4f} max={np.nanmax(ad):.4f} "
            f"exact_match={int(np.nansum(ad < 1e-6))}/{int(np.isfinite(ad).sum())}"
        )
    # protein vs old cache overlap note
    print(f"Audit -> {out_audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
