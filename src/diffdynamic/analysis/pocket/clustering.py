"""Clustering responsibilities extracted from evaluate_pocket_quality.py."""

from diffdynamic.analysis.pocket.state import (
    Chem,
    DBSCAN,
    DataStructs,
    HAS_SKLEARN,
    KMeans,
    np,
    rdMolDescriptors,
    silhouette_score,
)


def _collect_atom_data(molecules_with_pos):
    """
    收集所有分子的原子坐标、元素信息和分子质心

    Returns:
        dict:
            'coords'       : np.ndarray (N_total, 3) — 所有原子坐标
            'mol_idx'      : np.ndarray (N_total,)   — 原子所属分子索引
            'atom_nums'    : np.ndarray (N_total,)   — 原子序数
            'centroids'    : np.ndarray (n_mols, 3)  — 每分子质心
            'n_atoms_each' : list[int]               — 每分子原子数
            'n_mols'       : int
    """
    all_coords, mol_indices, atom_nums_list, centroids = [], [], [], []
    n_atoms_each = []

    for mol_i, (mol, pos) in enumerate(molecules_with_pos):
        if pos is None or len(pos) == 0:
            continue
        pos_arr = np.array(pos, dtype=np.float32)
        if pos_arr.ndim == 1:
            pos_arr = pos_arr.reshape(-1, 3)

        n = len(pos_arr)
        all_coords.append(pos_arr)
        mol_indices.extend([mol_i] * n)
        n_atoms_each.append(n)
        centroids.append(pos_arr.mean(axis=0))

        # 必须与 pos 行数 n 一一对应；mol.GetAtoms() 数量常与 n 不一致（H/重建差异等），
        # 否则 idea H 等处 coords 与 radii 长度不齐会广播失败。
        if mol is not None and Chem is not None:
            n_mol = mol.GetNumAtoms()
            if n_mol == n:
                for atom in mol.GetAtoms():
                    atom_nums_list.append(atom.GetAtomicNum())
            elif n_mol > n:
                for i in range(n):
                    atom_nums_list.append(mol.GetAtomWithIdx(i).GetAtomicNum())
            else:
                for atom in mol.GetAtoms():
                    atom_nums_list.append(atom.GetAtomicNum())
                atom_nums_list.extend([6] * (n - n_mol))
        else:
            atom_nums_list.extend([6] * n)  # 默认碳

    if not all_coords:
        return None

    return {
        "coords": np.vstack(all_coords),
        "mol_idx": np.array(mol_indices, dtype=np.int32),
        "atom_nums": np.array(atom_nums_list, dtype=np.int32),
        "centroids": np.array(centroids, dtype=np.float32),
        "n_atoms_each": n_atoms_each,
        "n_mols": len(centroids),
    }


def _build_idea_b_coord_mask(
    molecules_with_pos, subset_mode, edge_max_heavy_neighbors=2
):
    """
    与 _collect_atom_data 相同的分子遍历顺序，生成与拼接后 coords 对齐的布尔掩码。

    subset_mode:
        hetero_heavy — 非氢非碳重原子（药效/极性位点）
        edge_heavy   — 重原子且非氢邻居数 <= edge_max_heavy_neighbors（近似分子表面/边缘）
        hetero_edge  — 同时满足 hetero 与 edge（非碳边缘重原子，默认推荐用于结合模式）
    """
    if subset_mode not in ("hetero_heavy", "edge_heavy", "hetero_edge"):
        return None
    masks = []
    max_nn = int(edge_max_heavy_neighbors)
    for mol, pos in molecules_with_pos:
        if pos is None or len(pos) == 0:
            continue
        pos_arr = np.asarray(pos, dtype=np.float32)
        if pos_arr.ndim == 1:
            pos_arr = pos_arr.reshape(-1, 3)
        n = len(pos_arr)
        if mol is None or Chem is None or mol.GetNumAtoms() != n:
            masks.append(np.ones(n, dtype=bool))
            continue
        m = np.zeros(n, dtype=bool)
        for i, atom in enumerate(mol.GetAtoms()):
            z = atom.GetAtomicNum()
            hn = sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() > 1)
            if subset_mode == "hetero_heavy":
                m[i] = z > 1 and z != 6
            elif subset_mode == "edge_heavy":
                m[i] = z > 1 and hn <= max_nn
            else:  # hetero_edge
                m[i] = z > 1 and z != 6 and hn <= max_nn
        masks.append(m)
    if not masks:
        return None
    return np.concatenate(masks)


def _idea_b_cluster_score_from_n_clusters(
    n_clusters, max_clusters_for_high, silhouette=None, compactness=None
):
    if n_clusters <= 1:
        base = 1.0
    elif n_clusters <= max_clusters_for_high:
        base = max(0.0, 1.0 - (n_clusters - 1) / max_clusters_for_high)
    else:
        base = max(0.0, 0.5 - (n_clusters - max_clusters_for_high) * 0.1)
    # 轮廓系数修正（多簇时）
    if silhouette is not None and np.isfinite(silhouette):
        if silhouette < 0.1:
            base *= 0.6
        elif silhouette < 0.2:
            base *= 0.8
        elif silhouette > 0.3:
            base = min(1.0, base * 1.1)
    # 紧凑度修正（单簇或轮廓系数不可用时）：avg_dist_to_centroid
    # 越小越紧凑，说明结合模式越收敛
    if compactness is not None and np.isfinite(compactness):
        if compactness < 2.0:
            base *= 1.0  # 非常紧凑，不扣分
        elif compactness < 3.0:
            base *= 0.9
        elif compactness < 4.0:
            base *= 0.75
        elif compactness < 5.0:
            base *= 0.6
        else:
            base *= 0.4  # 非常分散，大幅扣分
    return base


def _dbscan_metrics_idea_b(X, eps, min_samples, max_clusters_for_high):
    """对 3D 坐标矩阵 X 运行 DBSCAN，返回簇数、噪声、轮廓系数与想法 B 密度收敛分。"""
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    silhouette = None
    compactness = None
    unique_labels = set(labels) - {-1}
    if len(unique_labels) >= 2 and len(X) > len(unique_labels):
        try:
            n_sample = min(len(X), 5000)
            idx = (
                np.random.choice(len(X), n_sample, replace=False)
                if len(X) > n_sample
                else np.arange(len(X))
            )
            mask_valid = labels[idx] != -1
            if mask_valid.sum() > len(unique_labels):
                silhouette = float(
                    silhouette_score(X[idx][mask_valid], labels[idx][mask_valid])
                )
        except Exception:
            pass
    # 紧凑度：per-cluster 平均点到质心距离（Å），仅多簇时有意义
    # 单簇不做 compactness 惩罚（大口袋中分子天然分散，不代表聚类差）
    if n_clusters >= 2:
        compactness_vals = []
        for lbl in unique_labels:
            mask = labels == lbl
            if mask.sum() > 1:
                X_cluster = X[mask]
                centroid = X_cluster.mean(axis=0)
                dists = np.linalg.norm(X_cluster - centroid, axis=1)
                compactness_vals.append(float(np.mean(dists)))
        if compactness_vals:
            compactness = float(np.mean(compactness_vals))
    score = _idea_b_cluster_score_from_n_clusters(
        n_clusters, max_clusters_for_high, silhouette, compactness
    )
    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "silhouette": silhouette,
        "compactness": compactness,
        "dbscan_labels": labels,
        "score": score,
    }


def _compute_morgan_fingerprints(molecules_with_pos, radius=2, n_bits=1024):
    """
    计算 Morgan 指纹矩阵（用于多样性分析）

    Returns:
        fps_matrix: np.ndarray (n_valid, n_bits) 或 None
        valid_mols: list of mol
    """
    if Chem is None or rdMolDescriptors is None:
        return None, []

    fps, valid_mols = [], []
    for mol, _ in molecules_with_pos:
        if mol is None:
            continue
        try:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, radius, nBits=n_bits
            )
            arr = np.zeros(n_bits, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
            valid_mols.append(mol)
        except Exception:
            pass

    if not fps:
        return None, []
    return np.vstack(fps).astype(np.float32), valid_mols


def evaluate_idea_b_clustering(
    molecules_with_pos,
    eps=1.5,
    min_samples=3,
    max_clusters_for_high=5,
    n_kmeans=5,
    atom_coord_subset="hetero_heavy",
    edge_max_heavy_neighbors=2,
    combined_focus_weight=0.5,
):
    """
    想法B：对生成分子的原子坐标做 DBSCAN（密度/簇数）并结合质心 KMeans。

    atom_coord_subset（DBSCAN 所用 3D 点集）：
        all          — 全部原子
        hetero_heavy — 非氢、非碳重原子（默认，聚焦药效团相关原子）
        edge_heavy   — 重原子且非氢重邻居数 <= edge_max_heavy_neighbors（近似分子边缘）
        hetero_edge  — 非碳边缘重原子（极性/表面位点，常用于结合模式）
        combined     — (1-w)*全原子 + w*hetero_edge 加权分，w=combined_focus_weight

    eps=1.5 A（默认）比 2.0 A 更好区分不同结合模式（2.0 A 约为 C-C 键长，过于粗糙）。

    KMeans 始终对整条分子的质心，不受子集影响。
    """
    if not molecules_with_pos or not HAS_SKLEARN:
        return {
            "score": 0.0,
            "n_clusters": 0,
            "n_noise": 0,
            "silhouette": None,
            "centroid_kmeans_labels": None,
            "kmeans_inertia": None,
            "quality_label": "unknown",
            "success": False,
            "atom_data": None,
            "message": "无分子数据或 sklearn 未安装",
        }

    valid_subsets = ("all", "hetero_heavy", "edge_heavy", "hetero_edge", "combined")
    if atom_coord_subset not in valid_subsets:
        atom_coord_subset = "all"

    atom_data = _collect_atom_data(molecules_with_pos)
    if atom_data is None:
        return {
            "score": 0.0,
            "n_clusters": 0,
            "n_noise": 0,
            "silhouette": None,
            "centroid_kmeans_labels": None,
            "kmeans_inertia": None,
            "quality_label": "unknown",
            "success": False,
            "atom_data": None,
            "message": "无有效坐标",
        }

    X_all = atom_data["coords"]
    centroids = atom_data["centroids"]
    n_total = len(X_all)

    def _quality_label_from_score(sc):
        if sc >= 0.6:
            return "high"
        if sc >= 0.3:
            return "medium"
        return "low"

    def _kmeans_block():
        kmeans_labels, kmeans_inertia = None, None
        if len(centroids) >= n_kmeans and KMeans is not None:
            k = min(n_kmeans, len(centroids))
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_labels = km.fit_predict(centroids).tolist()
            kmeans_inertia = float(km.inertia_)
        return kmeans_labels, kmeans_inertia

    if n_total < min_samples * 2:
        kmeans_labels, kmeans_inertia = _kmeans_block()
        return {
            "score": 0.5,
            "n_clusters": 1,
            "n_noise": 0,
            "silhouette": None,
            "centroid_kmeans_labels": kmeans_labels,
            "kmeans_inertia": kmeans_inertia,
            "quality_label": "medium",
            "success": True,
            "atom_data": atom_data,
            "atom_coord_subset": atom_coord_subset,
            "n_atoms_dbscan": n_total,
            "n_atoms_total": n_total,
            "coords_focus": None,
            "score_all_atoms": None,
            "score_focus": None,
            "n_clusters_focus": None,
            "n_noise_focus": None,
            "silhouette_focus": None,
            "dbscan_eps": eps,
            "dbscan_min_samples": min_samples,
            "message": "样本点过少，无法可靠聚类",
        }

    try:
        np.random.seed(42)
        kmeans_labels, kmeans_inertia = _kmeans_block()

        def _run_on_subset(X_sub, _label_tag):
            if len(X_sub) < min_samples * 2:
                return None
            return _dbscan_metrics_idea_b(
                X_sub, eps, min_samples, max_clusters_for_high
            )

        if atom_coord_subset == "all":
            m_all = _run_on_subset(X_all, "all")
            if m_all is None:
                raise RuntimeError("DBSCAN 子集为空")
            n_clusters = m_all["n_clusters"]
            n_noise = m_all["n_noise"]
            silhouette = m_all["silhouette"]
            score = m_all["score"]
            db_labels = m_all["dbscan_labels"]
            out_focus = None
            score_all_atoms = score
            score_focus = None
            n_cf = n_noise_focus = sil_f = None
            message = f"簇数={n_clusters}, 噪声点={n_noise}, 轮廓系数={silhouette}"
        elif atom_coord_subset == "combined":
            m_all = _run_on_subset(X_all, "all")
            if m_all is None:
                raise RuntimeError("DBSCAN 子集为空")
            mask_fe = _build_idea_b_coord_mask(
                molecules_with_pos, "hetero_edge", edge_max_heavy_neighbors
            )
            if mask_fe is None or mask_fe.sum() < min_samples * 2:
                n_clusters = m_all["n_clusters"]
                n_noise = m_all["n_noise"]
                silhouette = m_all["silhouette"]
                score = m_all["score"]
                db_labels = m_all["dbscan_labels"]
                score_all_atoms = m_all["score"]
                score_focus = None
                n_cf = n_noise_focus = sil_f = None
                out_focus = None
                note = "（hetero_edge 点数不足，仅用全原子）"
            else:
                X_fe = X_all[mask_fe]
                m_fe = _run_on_subset(X_fe, "hetero_edge")
                w = float(np.clip(combined_focus_weight, 0.0, 1.0))
                score_all_atoms = m_all["score"]
                score_focus = m_fe["score"] if m_fe else None
                if m_fe is None:
                    score = m_all["score"]
                    n_clusters = m_all["n_clusters"]
                    n_noise = m_all["n_noise"]
                    silhouette = m_all["silhouette"]
                    db_labels = m_all["dbscan_labels"]
                    n_cf = n_noise_focus = sil_f = None
                    out_focus = None
                    note = "（hetero_edge 子集聚类不可用，仅用全原子）"
                else:
                    score = (1.0 - w) * m_all["score"] + w * m_fe["score"]
                    n_clusters = m_all["n_clusters"]
                    n_noise = m_all["n_noise"]
                    silhouette = m_all["silhouette"]
                    db_labels = m_all["dbscan_labels"]
                    n_cf = m_fe["n_clusters"]
                    n_noise_focus = m_fe["n_noise"]
                    sil_f = m_fe["silhouette"]
                    out_focus = np.asarray(X_fe, dtype=np.float64)
                    note = f"（combined w={w:.2f} 全原子+hetero_edge）"
            message = (
                f"簇数={n_clusters}, 噪声={n_noise}, 轮廓={silhouette}; "
                f"focus簇={n_cf}, focus噪声={n_noise_focus}{note}"
            )
        else:
            mask = _build_idea_b_coord_mask(
                molecules_with_pos, atom_coord_subset, edge_max_heavy_neighbors
            )
            if mask is None or mask.sum() < min_samples * 2:
                return {
                    "score": 0.5,
                    "n_clusters": 1,
                    "n_noise": 0,
                    "silhouette": None,
                    "centroid_kmeans_labels": kmeans_labels,
                    "kmeans_inertia": kmeans_inertia,
                    "quality_label": "medium",
                    "success": True,
                    "atom_data": atom_data,
                    "atom_coord_subset": atom_coord_subset,
                    "n_atoms_dbscan": int(mask.sum()) if mask is not None else 0,
                    "n_atoms_total": n_total,
                    "coords_focus": None,
                    "score_all_atoms": None,
                    "score_focus": None,
                    "n_clusters_focus": None,
                    "n_noise_focus": None,
                    "silhouette_focus": None,
                    "dbscan_eps": eps,
                    "dbscan_min_samples": min_samples,
                    "message": f"子集 {atom_coord_subset} 点数不足，无法可靠聚类",
                }
            X_sub = X_all[mask]
            m_sub = _run_on_subset(X_sub, atom_coord_subset)
            if m_sub is None:
                raise RuntimeError("DBSCAN 失败")
            n_clusters = m_sub["n_clusters"]
            n_noise = m_sub["n_noise"]
            silhouette = m_sub["silhouette"]
            score = m_sub["score"]
            db_labels = m_sub["dbscan_labels"]
            out_focus = np.asarray(X_sub, dtype=np.float64)
            score_all_atoms = None
            score_focus = None
            n_cf = n_noise_focus = sil_f = None
            message = (
                f"[{atom_coord_subset}] 簇数={n_clusters}, 噪声={n_noise}, "
                f"轮廓={silhouette}, DBSCAN点数={len(X_sub)}/{n_total}"
            )

        label = _quality_label_from_score(score)

        if atom_coord_subset in ("all", "combined"):
            n_atoms_dbscan = n_total
        else:
            n_atoms_dbscan = int(mask.sum())

        ret = {
            "score": score,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "silhouette": silhouette,
            "dbscan_labels": db_labels.tolist(),
            "centroid_kmeans_labels": kmeans_labels,
            "kmeans_inertia": kmeans_inertia,
            "quality_label": label,
            "success": True,
            "atom_data": atom_data,
            "atom_coord_subset": atom_coord_subset,
            "n_atoms_dbscan": n_atoms_dbscan,
            "n_atoms_total": n_total,
            "coords_focus": out_focus,
            "dbscan_eps": eps,
            "dbscan_min_samples": min_samples,
            "message": message,
        }

        if atom_coord_subset == "combined":
            ret["score_all_atoms"] = score_all_atoms
            ret["score_focus"] = score_focus
            ret["n_clusters_focus"] = n_cf
            ret["n_noise_focus"] = n_noise_focus
            ret["silhouette_focus"] = sil_f
            ret["combined_focus_weight"] = float(
                np.clip(combined_focus_weight, 0.0, 1.0)
            )
        else:
            ret["score_all_atoms"] = score if atom_coord_subset == "all" else None
            ret["score_focus"] = score if atom_coord_subset != "all" else None
            ret["n_clusters_focus"] = n_cf
            ret["n_noise_focus"] = n_noise_focus
            ret["silhouette_focus"] = sil_f

        return ret
    except Exception as e:
        return {
            "score": 0.0,
            "n_clusters": 0,
            "n_noise": 0,
            "silhouette": None,
            "centroid_kmeans_labels": None,
            "kmeans_inertia": None,
            "quality_label": "unknown",
            "success": False,
            "atom_data": atom_data,
            "atom_coord_subset": atom_coord_subset,
            "message": str(e),
        }
