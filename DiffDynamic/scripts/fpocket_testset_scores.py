#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 jsdpt3010（或任意 result_{i}_*.pt 目录）的受体 PDB 运行 FPocket，
缓存 Score / Druggability / Volume，并按 fig7 约定计算
``fpocket_plot = (score + druggability) / 2``。

选袋模式：
  - 默认：固定 Pocket N（--fpocket-pocket-index，默认 1）
  - ``--match-ligand``：保留 *_out，用 pocketN_vert.pqr 球心与参考配体匹配，取最近口袋

示例::

    python3 scripts/fpocket_testset_scores.py \\
      --pt-dir jsdpt3010 --start 0 --end 99 --match-ligand \\
      --fpocket-cache outputs/fpocket_jsdpt3010_ligand_cache \\
      --out-audit fig7/fpocket_jsdpt3010_ligand_audit.csv
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pandas as pd
except ImportError:
    pd = None

from evaluate_pocket_quality import (  # noqa: E402
    _fpocket_pick_pocket,
    _parse_fpocket_info_txt,
    resolve_fpocket_executable,
    resolve_receptor_pdb_from_pt,
    run_fpocket_on_pdb,
)
from scripts.p2rank_jsdpt_scores import (  # noqa: E402
    load_ligand_positions,
    resolve_reference_ligand_sdf,
)


def _cache_path(cache_root: Path, data_id: Any) -> Path:
    return cache_root.resolve() / f"pocket_{data_id}" / "fpocket_summary.json"


def _read_cache(p: Path) -> Optional[dict[str, Any]]:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(p: Path, payload: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _find_pt(pt_dir: Path, data_id: int) -> Optional[Path]:
    matches = sorted(pt_dir.glob(f"result_{data_id}_*.pt"))
    return matches[-1] if matches else None


def _parse_xyz_from_pdb_like(path: Path) -> Optional[np.ndarray]:
    """Parse ATOM/HETATM coordinates from PDB/PQR (flexible whitespace)."""
    coords = []
    rx = re.compile(
        r"^(?:ATOM|HETATM)\s+\d+\s+\S+\s+\S+\s+\S*\s*-?\d*\s+"
        r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
    )
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    for line in text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        # Fixed-width PDB first
        if len(line) >= 54:
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append((x, y, z))
                continue
            except ValueError:
                pass
        m = rx.match(line)
        if m:
            coords.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
    if not coords:
        return None
    return np.asarray(coords, dtype=np.float64)


def _pocket_center_from_out(out_dir: Path, pocket_id: int) -> Optional[np.ndarray]:
    pockets_dir = out_dir / "pockets"
    for name in (f"pocket{pocket_id}_vert.pqr", f"pocket{pocket_id}_atm.pdb"):
        p = pockets_dir / name
        if not p.is_file():
            continue
        xyz = _parse_xyz_from_pdb_like(p)
        if xyz is not None and xyz.size >= 3:
            return xyz.mean(axis=0)
    return None


def run_fpocket_keep_out(
    pdb_path: Path,
    out_parent: Path,
    fpocket_cmd: str = "fpocket",
    timeout: int = 600,
) -> tuple[bool, str, list[dict], Optional[Path]]:
    """
    Run fpocket under out_parent, keep ``{stem}_out`` for center parsing.
    Returns (ok, message, pockets, out_dir).
    """
    pdb_path = Path(pdb_path).resolve()
    if not pdb_path.is_file():
        return False, f"PDB 不存在: {pdb_path}", [], None
    exe = resolve_fpocket_executable(fpocket_cmd)
    if not exe:
        return False, "未找到 fpocket 可执行文件", [], None

    out_parent = Path(out_parent)
    out_parent.mkdir(parents=True, exist_ok=True)
    stem = pdb_path.stem
    work_pdb = out_parent / f"{stem}.pdb"
    if work_pdb.resolve() != pdb_path:
        shutil.copy2(pdb_path, work_pdb)
    out_dir = out_parent / f"{stem}_out"
    if out_dir.is_dir():
        shutil.rmtree(out_dir, ignore_errors=True)

    cmd = [exe, "-f", str(work_pdb)]
    try:
        r = subprocess.run(
            cmd, cwd=str(out_parent), capture_output=True, text=True, timeout=int(timeout)
        )
    except FileNotFoundError:
        return False, f"无法执行: {exe}", [], None
    except subprocess.TimeoutExpired:
        return False, f"fpocket 超时 ({timeout}s)", [], None
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "")[:800]
        return False, f"fpocket 退出码 {r.returncode}: {err}", [], None

    if not out_dir.is_dir():
        return False, "未生成 *_out 目录", [], None
    info_path = out_dir / f"{stem}_info.txt"
    if not info_path.is_file():
        infos = sorted(out_dir.glob("*_info.txt"))
        info_path = infos[0] if infos else None
    if info_path is None or not info_path.is_file():
        return False, "未找到 *_info.txt", [], out_dir
    pockets = _parse_fpocket_info_txt(info_path.read_text(encoding="utf-8", errors="replace"))
    return True, "ok", pockets, out_dir


def _pick_nearest_to_ligand(
    pockets: list[dict],
    out_dir: Path,
    ligand_positions: np.ndarray,
) -> tuple[Optional[dict], float, Optional[list]]:
    """Return (picked_pocket, match_dist, center_xyz)."""
    lp = np.asarray(ligand_positions, dtype=np.float64).reshape(-1, 3)
    best = None
    best_d = float("inf")
    best_c = None
    for p in pockets:
        pid = int(p.get("pocket_id") or 0)
        if pid <= 0:
            continue
        c = _pocket_center_from_out(out_dir, pid)
        if c is None:
            continue
        d = float(np.mean(np.linalg.norm(lp - c.reshape(1, 3), axis=1)))
        if d < best_d:
            best_d = d
            best = p
            best_c = [float(c[0]), float(c[1]), float(c[2])]
    if best is None:
        return None, float("nan"), None
    return best, best_d, best_c


def _apply_pick(out: dict, pick: dict) -> None:
    try:
        if pick.get("score") is not None:
            out["score"] = float(pick["score"])
        if pick.get("druggability") is not None:
            out["druggability"] = float(pick["druggability"])
        if pick.get("volume") is not None:
            out["volume"] = float(pick["volume"])
    except (TypeError, ValueError):
        pass
    if np.isfinite(out["score"]) and np.isfinite(out["druggability"]):
        out["fpocket_plot"] = 0.5 * (out["score"] + out["druggability"])
        out["ok"] = True
        out["message"] = "ok"
    else:
        out["ok"] = False
        out["message"] = "incomplete_scores"


def _eval_one_pocket(args: tuple) -> dict[str, Any]:
    (
        data_id,
        pt_path,
        cache_root,
        run_fpocket,
        fpocket_cmd,
        pocket_index,
        timeout,
        protein_root,
        match_ligand,
    ) = args
    out: dict[str, Any] = {
        "pocket_id": int(data_id),
        "pt_path": str(pt_path) if pt_path else "",
        "protein": "",
        "ligand": "",
        "ok": False,
        "message": "",
        "score": np.nan,
        "druggability": np.nan,
        "volume": np.nan,
        "fpocket_plot": np.nan,
        "pocket_id_picked": -1,
        "match_dist": np.nan,
        "from_cache": False,
    }
    if not pt_path or not Path(pt_path).is_file():
        out["message"] = "missing_pt"
        return out

    cpath = _cache_path(Path(cache_root), data_id)
    loaded = _read_cache(cpath)
    need_ligand = bool(match_ligand)
    cache_ok = (
        loaded
        and loaded.get("ok")
        and (not need_ligand or loaded.get("match_mode") == "ligand_nearest")
    )
    if cache_ok:
        out["from_cache"] = True
        out["ok"] = True
        out["message"] = str(loaded.get("message") or "ok")
        out["protein"] = str(loaded.get("protein") or "")
        out["ligand"] = str(loaded.get("ligand") or "")
        try:
            if loaded.get("score") is not None:
                out["score"] = float(loaded["score"])
            if loaded.get("druggability") is not None:
                out["druggability"] = float(loaded["druggability"])
            if loaded.get("volume") is not None:
                out["volume"] = float(loaded["volume"])
            if loaded.get("pocket_id") is not None:
                out["pocket_id_picked"] = int(loaded["pocket_id"])
            if loaded.get("match_dist") is not None:
                out["match_dist"] = float(loaded["match_dist"])
        except (TypeError, ValueError):
            pass
        if np.isfinite(out["score"]) and np.isfinite(out["druggability"]):
            out["fpocket_plot"] = 0.5 * (out["score"] + out["druggability"])
        return out

    protein = resolve_receptor_pdb_from_pt(pt_path, protein_root=protein_root)
    if not protein:
        out["message"] = "protein_resolve_failed"
        _write_cache(cpath, {"ok": False, "message": out["message"], "pt_path": str(pt_path)})
        return out
    out["protein"] = str(Path(protein).resolve())

    ligand = None
    lig_pos = None
    if match_ligand:
        ligand = resolve_reference_ligand_sdf(
            Path(pt_path), protein_root=Path(protein_root) if protein_root else None
        )
        if not ligand:
            out["message"] = "ligand_resolve_failed"
            _write_cache(
                cpath,
                {
                    "ok": False,
                    "message": out["message"],
                    "protein": out["protein"],
                    "pt_path": str(Path(pt_path).resolve()),
                    "match_mode": "ligand_nearest",
                },
            )
            return out
        out["ligand"] = str(Path(ligand).resolve())
        lig_pos, _ = load_ligand_positions(Path(ligand))
        if lig_pos is None:
            out["message"] = "ligand_coords_failed"
            _write_cache(
                cpath,
                {
                    "ok": False,
                    "message": out["message"],
                    "protein": out["protein"],
                    "ligand": out["ligand"],
                    "match_mode": "ligand_nearest",
                },
            )
            return out

    if not run_fpocket:
        out["message"] = "no_cache_skip_run"
        return out

    pocket_dir = Path(cache_root) / f"pocket_{data_id}"
    if match_ligand:
        ok, msg, pockets, out_dir = run_fpocket_keep_out(
            Path(protein),
            pocket_dir,
            fpocket_cmd=fpocket_cmd,
            timeout=int(timeout),
        )
        pick = None
        match_dist = float("nan")
        center = None
        if ok and out_dir is not None and lig_pos is not None:
            pick, match_dist, center = _pick_nearest_to_ligand(pockets, out_dir, lig_pos)
        payload: dict[str, Any] = {
            "ok": False,
            "message": msg,
            "protein": out["protein"],
            "ligand": out["ligand"],
            "pt_path": str(Path(pt_path).resolve()),
            "match_mode": "ligand_nearest",
            "n_pockets": len(pockets) if pockets else 0,
        }
        if pick is not None:
            _apply_pick(out, pick)
            out["pocket_id_picked"] = int(pick.get("pocket_id") or -1)
            out["match_dist"] = float(match_dist)
            payload.update(
                {
                    "ok": out["ok"],
                    "message": out["message"],
                    "pocket_id": pick.get("pocket_id"),
                    "score": pick.get("score"),
                    "druggability": pick.get("druggability"),
                    "volume": pick.get("volume"),
                    "hydrophobicity_score": pick.get("hydrophobicity_score"),
                    "match_dist": match_dist if np.isfinite(match_dist) else None,
                    "center": center,
                }
            )
        else:
            out["message"] = msg if not ok else "no_ligand_match_pocket"
            payload["message"] = out["message"]
        _write_cache(cpath, payload)
        return out

    # Fixed pocket index mode (legacy)
    ok, msg, pockets = run_fpocket_on_pdb(
        Path(protein), fpocket_cmd=fpocket_cmd, timeout=int(timeout)
    )
    pick = _fpocket_pick_pocket(pockets, int(pocket_index)) if ok else None
    payload = {
        "ok": bool(ok and pick is not None),
        "message": msg,
        "protein": out["protein"],
        "pt_path": str(Path(pt_path).resolve()),
        "pocket_index_requested": int(pocket_index),
        "match_mode": "fixed_index",
    }
    if pick:
        _apply_pick(out, pick)
        out["pocket_id_picked"] = int(pick.get("pocket_id") or -1)
        payload["pocket_id"] = pick.get("pocket_id")
        payload["score"] = pick.get("score")
        payload["druggability"] = pick.get("druggability")
        payload["volume"] = pick.get("volume")
        payload["hydrophobicity_score"] = pick.get("hydrophobicity_score")
        payload["ok"] = out["ok"]
        payload["message"] = out["message"]
    else:
        out["message"] = msg if not ok else "no_pocket_pick"
        payload["message"] = out["message"]
    _write_cache(cpath, payload)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="jsdpt3010 受体 FPocket 评分（可选配体最近口袋）"
    )
    ap.add_argument("--pt-dir", type=Path, default=REPO_ROOT / "jsdpt3010")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=99)
    ap.add_argument("--protein-root", type=Path, default=None)
    ap.add_argument(
        "--fpocket-cache",
        type=Path,
        default=REPO_ROOT / "outputs" / "fpocket_jsdpt3010_cache",
    )
    ap.add_argument("--run-fpocket", dest="run_fpocket", action="store_true", default=True)
    ap.add_argument("--no-run-fpocket", dest="run_fpocket", action="store_false")
    ap.add_argument("--fpocket-cmd", type=str, default="fpocket")
    ap.add_argument("--fpocket-pocket-index", type=int, default=1)
    ap.add_argument(
        "--match-ligand",
        action="store_true",
        help="按参考配体最近口袋选袋（需解析配体 SDF + 保留 fpocket out）",
    )
    ap.add_argument("--fpocket-timeout", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--compare-csv",
        type=Path,
        default=REPO_ROOT / "fig7" / "pocketeval1.csv",
    )
    ap.add_argument(
        "--out-audit",
        type=Path,
        default=REPO_ROOT / "fig7" / "fpocket_jsdpt3010_audit.csv",
    )
    ap.add_argument(
        "--out-corrected",
        type=Path,
        default=REPO_ROOT / "fig7" / "pocketeval1_fpocket_jsdpt_corrected.csv",
    )
    ap.add_argument(
        "--out-xlsx",
        type=Path,
        default=REPO_ROOT / "outputs" / "fpocket_jsdpt3010_scores.xlsx",
    )
    args = ap.parse_args()

    if pd is None:
        print("需要 pandas", file=sys.stderr)
        return 2

    pt_dir = args.pt_dir.resolve()
    if not pt_dir.is_dir():
        print(f"❌ --pt-dir 不存在: {pt_dir}", file=sys.stderr)
        return 2

    tasks = []
    for i in range(int(args.start), int(args.end) + 1):
        pt = _find_pt(pt_dir, i)
        tasks.append(
            (
                i,
                str(pt) if pt else "",
                str(args.fpocket_cache.resolve()),
                bool(args.run_fpocket),
                args.fpocket_cmd,
                int(args.fpocket_pocket_index),
                int(args.fpocket_timeout),
                str(args.protein_root.resolve()) if args.protein_root else None,
                bool(args.match_ligand),
            )
        )

    results: list[dict[str, Any]] = []
    n_workers = max(1, int(args.workers))
    mode = "ligand_nearest" if args.match_ligand else f"index={args.fpocket_pocket_index}"
    print(
        f"Evaluating FPocket on {len(tasks)} pockets "
        f"(workers={n_workers}, mode={mode}, cache={args.fpocket_cache})",
        flush=True,
    )
    if n_workers == 1 or len(tasks) == 1:
        for t in tasks:
            r = _eval_one_pocket(t)
            results.append(r)
            print(
                f"  [{r['pocket_id']}] ok={r['ok']} plot={r['fpocket_plot']} "
                f"picked={r['pocket_id_picked']} dist={r['match_dist']} "
                f"prot={Path(r['protein']).name if r['protein'] else '-'} "
                f"({'cache' if r['from_cache'] else 'run'}) {r['message']}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_eval_one_pocket, t): t[0] for t in tasks}
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                print(
                    f"  [{r['pocket_id']}] ok={r['ok']} plot={r['fpocket_plot']} "
                    f"picked={r['pocket_id_picked']} dist={r['match_dist']} "
                    f"prot={Path(r['protein']).name if r['protein'] else '-'} "
                    f"({'cache' if r['from_cache'] else 'run'}) {r['message']}",
                    flush=True,
                )

    results = sorted(results, key=lambda x: int(x["pocket_id"]))
    df = pd.DataFrame(results)

    if args.compare_csv and Path(args.compare_csv).is_file():
        old = pd.read_csv(args.compare_csv)
        if "fpocket" in old.columns and len(old) >= len(df):
            df["csv_fpocket_old"] = old["fpocket"].astype(float).values[: len(df)]
            df["abs_diff"] = (df["fpocket_plot"] - df["csv_fpocket_old"]).abs()
        else:
            df["csv_fpocket_old"] = np.nan
            df["abs_diff"] = np.nan
    else:
        df["csv_fpocket_old"] = np.nan
        df["abs_diff"] = np.nan

    audit_cols = [
        "pocket_id",
        "protein",
        "ligand",
        "pt_path",
        "ok",
        "message",
        "pocket_id_picked",
        "match_dist",
        "score",
        "druggability",
        "volume",
        "fpocket_plot",
        "csv_fpocket_old",
        "abs_diff",
        "from_cache",
    ]
    for c in audit_cols:
        if c not in df.columns:
            df[c] = np.nan if c != "ligand" else ""
    audit = df[audit_cols].copy()
    out_audit = args.out_audit.resolve()
    out_audit.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_audit, index=False)

    if args.compare_csv and Path(args.compare_csv).is_file():
        base = pd.read_csv(args.compare_csv).copy()
        n = min(len(base), len(df))
        base = base.iloc[:n].copy()
        base["fpocket"] = df["fpocket_plot"].astype(float).values[:n]
        out_corr = args.out_corrected.resolve()
        out_corr.parent.mkdir(parents=True, exist_ok=True)
        cols = [c for c in ("this_work", "p2rank", "fpocket", "Sitemap") if c in base.columns]
        base[cols].to_csv(out_corr, index=False)
        print(f"Corrected draft -> {out_corr}")

    out_xlsx = args.out_xlsx.resolve()
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    meta = pd.DataFrame(
        [
            ("fpocket_cmd", args.fpocket_cmd),
            ("match_ligand", bool(args.match_ligand)),
            ("fpocket_pocket_index", args.fpocket_pocket_index),
            ("cache", str(args.fpocket_cache.resolve())),
            ("pt_dir", str(pt_dir)),
            ("n_pockets", len(df)),
            ("n_ok", int(df["ok"].sum())),
            ("n_finite_plot", int(np.isfinite(df["fpocket_plot"]).sum())),
            ("formula", "(score + druggability) / 2"),
        ],
        columns=["key", "value"],
    )
    try:
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="fpocket_scores")
            meta.to_excel(w, index=False, sheet_name="run_info")
        print(f"Excel -> {out_xlsx}")
    except Exception as exc:
        print(f"⚠️ Excel write skipped: {exc}")

    finite = df["fpocket_plot"].to_numpy(dtype=float)
    ok_n = int(df["ok"].sum())
    print("\n=== Summary ===")
    print(f"ok: {ok_n}/{len(df)}")
    print(f"finite fpocket_plot: {int(np.isfinite(finite).sum())}/{len(df)}")
    if np.isfinite(finite).any():
        print(
            f"fpocket_plot mean±std: {np.nanmean(finite):.4f} ± {np.nanstd(finite):.4f} "
            f"[{np.nanmin(finite):.4f}, {np.nanmax(finite):.4f}]"
        )
    if "abs_diff" in df.columns and np.isfinite(df["abs_diff"]).any():
        ad = df["abs_diff"].to_numpy(dtype=float)
        print(
            f"vs old CSV abs_diff: MAE={np.nanmean(ad):.4f} "
            f"max={np.nanmax(ad):.4f} "
            f"exact_match={int(np.nansum(ad < 1e-9))}/{int(np.isfinite(ad).sum())}"
        )
    print(f"Audit -> {out_audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
