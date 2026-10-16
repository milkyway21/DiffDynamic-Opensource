"""Pocket Metrics responsibilities extracted from evaluate_pocket_quality.py."""

from diffdynamic.analysis.pocket.state import (
    DBSCAN,
    Path,
    _ANCHOR_ARO,
    _ANCHOR_HYDRO,
    _ANCHOR_NEG,
    _ANCHOR_POS,
    _POCKET_POLAR,
    np,
)


def _fpocket_score_map(score_raw):
    """Map FPocket Score (~0–40+) to [0,1]."""
    if score_raw is None or not np.isfinite(float(score_raw)):
        return None
    return float(np.clip(float(score_raw) / 40.0, 0.0, 1.0))


def _fpocket_hydrophobicity_midpeak(h_raw):
    """
    Hydrophobicity: mid-range full score; extremes penalized.
    FPocket hydrophobicity often roughly in [-2, 2] or wider — normalize via tanh then mid-peak.
    """
    if h_raw is None or not np.isfinite(float(h_raw)):
        return None
    # map to ~[0,1] via sigmoid-ish on raw
    x = 0.5 * (np.tanh(float(h_raw) / 2.0) + 1.0)
    if 0.30 <= x <= 0.70:
        return 1.0
    if x < 0.30:
        return float(0.40 + (x / 0.30) * 0.60)  # 0→0.4, 0.3→1.0
    # x > 0.70
    return float(1.0 - (x - 0.70) / 0.30 * 0.60)  # 0.7→1, 1→0.4


def _fpocket_druggability_map(d_raw):
    if d_raw is None or not np.isfinite(float(d_raw)):
        return None
    d = float(np.clip(float(d_raw), 0.0, 1.0))
    if d < 0.2:
        return float(d / 0.2 * 0.35)  # <0.2 → ≤0.35
    return d


def evaluate_idea_f_pocket_chemistry(
    fpocket_protein_pdb=None,
    molecules_with_pos=None,
    centroid_radius=10.0,
    fpocket_cmd="fpocket",
    fpocket_pocket_index=1,
    fpocket_timeout=600,
):
    """
    想法F（口袋化学）：配体邻域残基的生物学代理分（疏水 / 极性 / 封闭度）。

    不调用 FPocket。``fpocket_protein_pdb`` 参数名保留以兼容旧调用，实为受体 PDB。
    默认邻域半径 **10 Å**。无蛋白 PDB、无配体质心、或邻域无残基 → success=False。
    """
    from diffdynamic.analysis.pocket.aggregation import _weighted_geo_mean

    empty = {
        "score": 0.0,
        "success": False,
        "quality_label": "unknown",
        "hydro_fraction": None,
        "polar_fraction": None,
        "enclosure_score": None,
        "n_residues": 0,
        "n_heavy_atoms": 0,
        "message": "未指定蛋白 PDB 或配体坐标不可用",
        # legacy keys kept empty for old viz/CSV consumers
        "druggability": None,
        "fpocket_score": None,
        "hydrophobicity_score": None,
        "druggability_mapped": None,
        "fpocket_score_mapped": None,
        "hydrophobicity_mapped": None,
    }
    if not fpocket_protein_pdb or not str(fpocket_protein_pdb).strip():
        return empty
    prot = Path(fpocket_protein_pdb).expanduser().resolve()
    if not prot.is_file():
        empty["message"] = f"蛋白 PDB 不存在: {prot}"
        return empty
    center = _ligand_centroid_from_molecules(molecules_with_pos)
    if center is None:
        empty["message"] = "无法从分子估计口袋中心"
        return empty

    data, err = _collect_pocket_residue_composition(
        prot, center, radius=float(centroid_radius)
    )
    if data is None:
        empty["message"] = err or "残基组成收集失败"
        return empty
    n_res = int(data.get("n_residues") or 0)
    if n_res <= 0:
        empty["message"] = (
            f"口袋邻域（{float(centroid_radius):g}Å）内无蛋白残基；"
            "请确认 PDB 与配体坐标同一参考系"
        )
        return empty

    n_hydro = int(data.get("n_hydrophobic") or 0)
    n_polar = int(data.get("n_polar") or 0)
    n_heavy = int(data.get("n_heavy_atoms") or 0)
    hydro_frac = float(n_hydro) / float(n_res)
    polar_frac = float(n_polar) / float(n_res)
    # enclosure: saturate around ~350 heavy atoms in 10Å shell
    enc_raw = float(np.clip(n_heavy / 350.0, 0.0, 1.0))
    hydro_m = _fraction_midpeak_01(hydro_frac)
    polar_m = _fraction_midpeak_01(polar_frac)
    enc_m = _fraction_midpeak_01(enc_raw)

    parts = [hydro_m, polar_m, enc_m]
    score = float(_weighted_geo_mean(parts, [1.0, 1.0, 1.0]))
    label = "high" if score >= 0.6 else ("medium" if score >= 0.3 else "low")
    return {
        "score": score,
        "success": True,
        "quality_label": label,
        "hydro_fraction": hydro_frac,
        "polar_fraction": polar_frac,
        "enclosure_score": enc_m,
        "enclosure_raw": enc_raw,
        "n_residues": n_res,
        "n_hydrophobic": n_hydro,
        "n_polar": n_polar,
        "n_heavy_atoms": n_heavy,
        "centroid_radius": float(centroid_radius),
        "ligand_centroid": [float(center[0]), float(center[1]), float(center[2])],
        "druggability": None,
        "fpocket_score": None,
        "hydrophobicity_score": None,
        "druggability_mapped": hydro_m,
        "fpocket_score_mapped": polar_m,
        "hydrophobicity_mapped": enc_m,
        "message": (
            f"bio_chem={score:.3f} (hydro={hydro_m:.3f}/{hydro_frac:.2f}, "
            f"polar={polar_m:.3f}/{polar_frac:.2f}, enc={enc_m:.3f}, "
            f"n_res={n_res}, n_heavy={n_heavy}, r={float(centroid_radius):g}Å)"
        ),
    }


def _fraction_midpeak_01(x):
    """Map fraction in [0,1] with mid-range peak (0.30–0.70 → 1.0)."""
    if x is None or not np.isfinite(float(x)):
        return 0.0
    x = float(np.clip(float(x), 0.0, 1.0))
    if 0.30 <= x <= 0.70:
        return 1.0
    if x < 0.30:
        return float(0.40 + (x / 0.30) * 0.60)
    return float(1.0 - (x - 0.70) / 0.30 * 0.60)


def _collect_pocket_residue_composition(protein_pdb, center, radius=10.0):
    """
    Count residues / hydrophobic / polar / heavy atoms within radius of ligand center.
    Default radius 10 Å. Returns (dict, err_msg).
    """
    try:
        from Bio.PDB import PDBParser, is_aa
    except ImportError:
        return None, "Bio.PDB 未安装"

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("prot", str(protein_pdb))
    except Exception as exc:
        return None, f"PDB 解析失败: {exc}"

    center = np.asarray(center, dtype=np.float64).reshape(3)
    n_res = 0
    n_hydro = 0
    n_polar = 0
    n_heavy = 0
    r = float(radius)

    for model in structure:
        for chain in model:
            for res in chain:
                if not is_aa(res, standard=True):
                    resname = res.get_resname().strip().upper()
                    if resname not in ("HID", "HIE", "HIP"):
                        continue
                else:
                    resname = res.get_resname().strip().upper()
                try:
                    if "CA" in res:
                        ca = np.asarray(res["CA"].coord, dtype=np.float64)
                    else:
                        coords = [a.coord for a in res.get_atoms()]
                        if not coords:
                            continue
                        ca = np.mean(coords, axis=0)
                except Exception:
                    continue
                if np.linalg.norm(ca - center) > r:
                    continue
                n_res += 1
                if resname in _ANCHOR_HYDRO:
                    n_hydro += 1
                if resname in _POCKET_POLAR:
                    n_polar += 1
                for atom in res.get_atoms():
                    el = atom.element.strip().upper() if atom.element else ""
                    if el and el != "H":
                        try:
                            pos = np.asarray(atom.coord, dtype=np.float64)
                        except Exception:
                            continue
                        if np.linalg.norm(pos - center) <= r:
                            n_heavy += 1

    return {
        "n_residues": n_res,
        "n_hydrophobic": n_hydro,
        "n_polar": n_polar,
        "n_heavy_atoms": n_heavy,
    }, None


def _ligand_centroid_from_molecules(molecules_with_pos):
    pts = []
    for mol, pos in molecules_with_pos or []:
        if pos is None:
            continue
        try:
            arr = np.asarray(pos, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] >= 3 and len(arr) > 0:
                pts.append(arr[:, :3].mean(axis=0))
        except Exception:
            pass
    if not pts:
        return None
    return np.mean(np.vstack(pts), axis=0)


def _collect_pocket_anchor_sites(protein_pdb, center, radius=12.0):
    """
    Collect anchor site coordinates + category counts from protein PDB
    within radius of center. Uses Bio.PDB.
    Returns dict with counts and list of (xyz, category).
    """
    try:
        from Bio.PDB import PDBParser, is_aa
    except ImportError:
        return None, "Bio.PDB 未安装"

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("prot", str(protein_pdb))
    except Exception as exc:
        return None, f"PDB 解析失败: {exc}"

    center = np.asarray(center, dtype=np.float64).reshape(3)
    sites = []  # (xyz, cat) cat in donor,acceptor,aromatic,ionic,hydrophobic
    n_res_in_pocket = 0

    for model in structure:
        for chain in model:
            for res in chain:
                if not is_aa(res, standard=True):
                    # still allow HIS variants
                    resname = res.get_resname().strip().upper()
                    if resname not in ("HID", "HIE", "HIP"):
                        continue
                else:
                    resname = res.get_resname().strip().upper()
                try:
                    if "CA" in res:
                        ca = res["CA"].coord
                    else:
                        coords = [a.coord for a in res.get_atoms()]
                        if not coords:
                            continue
                        ca = np.mean(coords, axis=0)
                except Exception:
                    continue
                ca = np.asarray(ca, dtype=np.float64)
                if np.linalg.norm(ca - center) > radius:
                    continue
                n_res_in_pocket += 1

                # backbone donor/acceptor
                if "N" in res:
                    sites.append(
                        (np.asarray(res["N"].coord, dtype=np.float64), "donor")
                    )
                if "O" in res:
                    sites.append(
                        (np.asarray(res["O"].coord, dtype=np.float64), "acceptor")
                    )

                if resname in _ANCHOR_NEG:
                    for aname in ("OD1", "OD2", "OE1", "OE2"):
                        if aname in res:
                            sites.append(
                                (
                                    np.asarray(res[aname].coord, dtype=np.float64),
                                    "ionic",
                                )
                            )
                            sites.append(
                                (
                                    np.asarray(res[aname].coord, dtype=np.float64),
                                    "acceptor",
                                )
                            )
                if resname in _ANCHOR_POS or resname == "LYS":
                    for aname in ("NZ", "NH1", "NH2", "NE"):
                        if aname in res:
                            sites.append(
                                (
                                    np.asarray(res[aname].coord, dtype=np.float64),
                                    "ionic",
                                )
                            )
                            sites.append(
                                (
                                    np.asarray(res[aname].coord, dtype=np.float64),
                                    "donor",
                                )
                            )
                if resname in _ANCHOR_ARO:
                    ring_atoms = []
                    for a in res.get_atoms():
                        if a.element.strip().upper() == "C" and a.name.strip() not in (
                            "CA",
                            "C",
                            "CB",
                        ):
                            ring_atoms.append(a.coord)
                    if ring_atoms:
                        sites.append((np.mean(ring_atoms, axis=0), "aromatic"))
                # sidechain H-bond for Ser/Thr/Tyr/Asn/Gln/Trp
                if resname in ("SER", "THR", "TYR"):
                    for aname in ("OG", "OG1", "OH"):
                        if aname in res:
                            sites.append(
                                (
                                    np.asarray(res[aname].coord, dtype=np.float64),
                                    "donor",
                                )
                            )
                            sites.append(
                                (
                                    np.asarray(res[aname].coord, dtype=np.float64),
                                    "acceptor",
                                )
                            )
                if resname in ("ASN", "GLN", "TRP"):
                    for aname in ("ND2", "NE1", "NE2"):
                        if aname in res:
                            sites.append(
                                (
                                    np.asarray(res[aname].coord, dtype=np.float64),
                                    "donor",
                                )
                            )
                    for aname in ("OD1", "OE1"):
                        if aname in res:
                            sites.append(
                                (
                                    np.asarray(res[aname].coord, dtype=np.float64),
                                    "acceptor",
                                )
                            )
                if resname in _ANCHOR_HYDRO:
                    if "CB" in res:
                        sites.append(
                            (
                                np.asarray(res["CB"].coord, dtype=np.float64),
                                "hydrophobic",
                            )
                        )
                    else:
                        sites.append((ca, "hydrophobic"))

    return {
        "sites": sites,
        "n_pocket_residues": n_res_in_pocket,
    }, None


def evaluate_idea_g_anchor_richness(
    fpocket_protein_pdb=None, molecules_with_pos=None, centroid_radius=12.0
):
    """
    想法G（P4）：相互作用锚点丰富度 = 0.35 richness + 0.35 coverage + 0.30 balance。
    仅用蛋白原子类型；需要蛋白 PDB + 配体质心（或失败）。
    """
    empty = {
        "score": 0.0,
        "success": False,
        "quality_label": "unknown",
        "s_richness": None,
        "s_coverage": None,
        "s_balance": None,
        "n_eff": 0,
        "n_cap": None,
        "n_clusters": 0,
        "message": "需要蛋白 PDB 与配体坐标以估计口袋锚点",
    }
    if not fpocket_protein_pdb or not str(fpocket_protein_pdb).strip():
        return empty
    prot = Path(fpocket_protein_pdb).expanduser().resolve()
    if not prot.is_file():
        empty["message"] = f"蛋白 PDB 不存在: {prot}"
        return empty
    center = _ligand_centroid_from_molecules(molecules_with_pos)
    if center is None:
        empty["message"] = "无法从分子估计口袋中心"
        return empty

    data, err = _collect_pocket_anchor_sites(prot, center, radius=centroid_radius)
    if err or not data:
        empty["message"] = err or "锚点收集失败"
        return empty

    sites = data["sites"]
    n_res = int(data["n_pocket_residues"])
    if n_res <= 0 or not sites:
        empty["message"] = (
            f"口袋邻域（{centroid_radius}Å）内无蛋白残基/锚点；请确认 PDB 与配体坐标同一参考系"
        )
        return empty
    n_res = max(1, n_res)
    cats = {"donor": 0, "acceptor": 0, "aromatic": 0, "ionic": 0, "hydrophobic": 0}
    coords = []
    for xyz, cat in sites:
        if cat in cats:
            cats[cat] += 1
        if cat in ("donor", "acceptor", "aromatic", "ionic"):
            coords.append(xyz)

    n_eff = cats["donor"] + cats["acceptor"] + cats["aromatic"] + cats["ionic"]
    n_cap = float(np.clip(0.8 * n_res, 12, 40))
    s_rich = float(min(1.0, n_eff / n_cap)) if n_cap > 0 else 0.0

    # Coverage via DBSCAN on polar/ionic/aromatic anchors
    s_cov = 0.30
    n_clusters = 0
    if len(coords) >= 2 and DBSCAN is not None:
        X = np.asarray(coords, dtype=np.float64)
        labels = DBSCAN(eps=3.5, min_samples=2).fit(X).labels_
        uniq = [l for l in set(labels) if l >= 0]
        n_clusters = len(uniq)
        if n_clusters < 2:
            s_cov = 0.45
            # tighten if single tiny cluster
            if n_clusters == 1:
                mask = labels == uniq[0]
                rad = float(
                    np.max(np.linalg.norm(X[mask] - X[mask].mean(axis=0), axis=1))
                )
                if rad < 2.0:
                    s_cov = 0.30
        else:
            cents = [X[labels == lab].mean(axis=0) for lab in uniq]
            dists = []
            for i in range(len(cents)):
                for j in range(i + 1, len(cents)):
                    dists.append(float(np.linalg.norm(cents[i] - cents[j])))
            dmin = min(dists) if dists else 0.0
            if 4.0 <= dmin <= 14.0:
                s_cov = 1.0
            elif dmin < 4.0:
                s_cov = 0.55
            else:
                s_cov = 0.70
    elif len(coords) < 2:
        s_cov = 0.25

    # Balance: Shannon entropy over 4 interaction classes (ignore hydrophobic for balance core)
    counts = np.array(
        [cats["donor"], cats["acceptor"], cats["aromatic"], cats["ionic"]],
        dtype=np.float64,
    )
    if counts.sum() <= 0:
        s_bal = 0.0
    else:
        p = counts / counts.sum()
        p = p[p > 0]
        ent = float(-np.sum(p * np.log(p)) / np.log(4.0))
        s_bal = ent
        if ent < 0.5:
            s_bal = min(s_bal, 0.5)

    score = 0.35 * s_rich + 0.35 * s_cov + 0.30 * s_bal
    label = "high" if score >= 0.6 else ("medium" if score >= 0.3 else "low")
    # Keep a capped list of sites for 2D viz (xyz + category)
    site_xyz = []
    site_cat = []
    for xyz, cat in sites[:800]:
        site_xyz.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
        site_cat.append(cat)
    return {
        "score": float(min(1.0, score)),
        "success": True,
        "quality_label": label,
        "s_richness": s_rich,
        "s_coverage": s_cov,
        "s_balance": s_bal,
        "n_eff": int(n_eff),
        "n_cap": n_cap,
        "n_clusters": n_clusters,
        "n_pocket_residues": n_res,
        "anchor_counts": cats,
        "ligand_centroid": [float(center[0]), float(center[1]), float(center[2])],
        "centroid_radius": float(centroid_radius),
        "site_xyz": site_xyz,
        "site_categories": site_cat,
        "message": (
            f"anchor={score:.3f} rich={s_rich:.2f} cov={s_cov:.2f} bal={s_bal:.2f} "
            f"Neff={n_eff}/{n_cap:.0f} clusters={n_clusters}"
        ),
    }
