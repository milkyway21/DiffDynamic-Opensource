"""Aggregation responsibilities extracted from evaluate_pocket_quality.py."""

from diffdynamic.analysis.pocket.state import GEO_EPS

from diffdynamic.analysis.pocket.state import (
    EVAL_RECORD_FIELDS,
    Path,
    csv,
    datetime,
    np,
)


def _weighted_geo_mean(scores, weights, eps=GEO_EPS):
    """Weighted geometric mean; skip non-finite / missing; renormalize weights."""
    vals, wts = [], []
    for s, w in zip(scores, weights):
        if s is None or w is None or w <= 0:
            continue
        try:
            sf = float(s)
            wf = float(w)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(sf) or not np.isfinite(wf) or wf <= 0:
            continue
        vals.append(max(sf, eps))
        wts.append(wf)
    if not vals:
        return float(eps)
    wts = np.asarray(wts, dtype=np.float64)
    wts = wts / wts.sum()
    vals = np.asarray(vals, dtype=np.float64)
    return float(np.exp(np.sum(wts * np.log(vals))))


def _equal_geo_mean(scores, eps=GEO_EPS):
    """Equal-weight geometric mean (cube root of product when len==3)."""
    vals = []
    for s in scores:
        if s is None:
            continue
        try:
            sf = float(s)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(sf):
            continue
        vals.append(max(sf, eps))
    if not vals:
        return float(eps)
    return float(np.exp(np.mean(np.log(np.asarray(vals, dtype=np.float64)))))


def _merge_affinity_le(idea_vina, idea_le, w_v=0.55, w_le=0.45):
    """Merge Vina (idea A raw) and LE into slot-A Affinity+LE score."""
    idea_vina = idea_vina or {}
    idea_le = idea_le or {}
    v_ok = bool(idea_vina.get("success"))
    le_ok = bool(idea_le.get("success"))
    out = dict(idea_vina)
    out["le_component"] = idea_le
    out["vina_component_score"] = idea_vina.get("score") if v_ok else None
    out["le_component_score"] = idea_le.get("score") if le_ok else None
    if v_ok and le_ok:
        score = w_v * float(idea_vina["score"]) + w_le * float(idea_le["score"])
        out["score"] = float(min(1.0, score))
        out["success"] = True
        out["message"] = (
            f"Aff+LE={out['score']:.3f} "
            f"(vina={idea_vina['score']:.3f}×{w_v}+LE={idea_le['score']:.3f}×{w_le})"
        )
    elif v_ok:
        out["score"] = float(idea_vina["score"])
        out["success"] = True
        out["message"] = f"仅 Vina={out['score']:.3f}（无有效 LE）"
    elif le_ok:
        out = {
            "score": float(idea_le["score"]),
            "success": True,
            "le_component": idea_le,
            "vina_scores": idea_vina.get("vina_scores") or [],
            "vina_component_score": None,
            "le_component_score": idea_le.get("score"),
            "message": f"仅 LE={idea_le['score']:.3f}（无有效 Vina）",
        }
    else:
        out["score"] = 0.0
        out["success"] = False
        out["message"] = "Vina 与 LE 均失败"
    s = float(out.get("score") or 0.0)
    if out.get("success"):
        out["quality_label"] = "high" if s >= 0.5 else ("medium" if s >= 0.2 else "low")
    else:
        out["quality_label"] = "unknown"
    return out


def append_evaluation_record(result, record_path, timestamp=None):
    """
    Append one evaluation record to the pocket evaluation log (CSV).

    Columns: pocket_id, timestamp, n_molecules, score_a..h, overall_score,
    overall_label, pt_path, vis_dir
    """
    record_path = Path(record_path)
    record_path.parent.mkdir(parents=True, exist_ok=True)

    pocket_id = result.get("data_id")
    if pocket_id is None:
        pt_stem = Path(result.get("pt_path", "")).stem
        if pt_stem.startswith("result_") and "_" in pt_stem:
            pocket_id = pt_stem.split("_")[1]
        else:
            pocket_id = pt_stem
    pocket_id = str(pocket_id)

    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    def _score(k):
        r = result.get(k, {})
        return f"{r.get('score', 0):.4f}" if r.get("success") else "N/A"

    row = {
        "pocket_id": pocket_id,
        "timestamp": ts,
        "n_molecules": result.get("n_molecules", 0),
        "score_a": _score("idea_a"),
        "score_b": _score("idea_b"),
        "score_c": _score("idea_c"),
        "score_d": _score("idea_d"),
        "score_e": _score("idea_e"),
        "score_f": _score("idea_f"),
        "score_g": _score("idea_g"),
        "score_h": _score("idea_h"),
        "s_pocket": f"{result.get('s_pocket', 0):.4f}"
        if result.get("s_pocket") is not None
        else "",
        "s_compatibility": (
            f"{result.get('s_compatibility', 0):.4f}"
            if result.get("s_compatibility") is not None
            else ""
        ),
        "s_ligand": f"{result.get('s_ligand', 0):.4f}"
        if result.get("s_ligand") is not None
        else "",
        "overall_score": f"{result.get('overall_score', 0):.4f}",
        "overall_label": result.get("overall_label", "unknown"),
        "pt_path": result.get("pt_path") or result.get("ligand_path") or "",
        "vis_dir": result.get("vis_dir") or "",
    }

    file_exists = record_path.exists()
    if file_exists:
        with open(record_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fields = list(reader.fieldnames or [])
            old_rows = list(reader)
        if "vis_dir" not in old_fields:
            with open(record_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=EVAL_RECORD_FIELDS)
                writer.writeheader()
                for old in old_rows:
                    writer.writerow({k: old.get(k, "") for k in EVAL_RECORD_FIELDS})
                writer.writerow({k: row.get(k, "") for k in EVAL_RECORD_FIELDS})
            return

    with open(record_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_RECORD_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in EVAL_RECORD_FIELDS})
