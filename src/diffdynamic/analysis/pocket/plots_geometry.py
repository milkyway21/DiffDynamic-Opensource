"""Plots Geometry responsibilities extracted from evaluate_pocket_quality.py."""

from diffdynamic.analysis.pocket.state import (
    DBSCAN,
    HAS_MATPLOTLIB,
    HAS_SCIPY,
    HAS_SKLEARN,
    IDEA_H_VIZ_REF_OPT_MAX,
    IDEA_H_VIZ_REF_OPT_MIN,
    IDEA_H_VIZ_REF_ZERO_ABOVE,
    IDEA_H_VIZ_REF_ZERO_BELOW,
    IDEA_H_VIZ_X_MAX_DEFAULT,
    KMeans,
    MaxNLocator,
    PCA,
    Path,
    StandardScaler,
    _CPK_COLORS,
    _ELEMENT_NAMES,
    gaussian_kde,
    np,
    plt,
)


def visualize_clustering_2d(
    molecules_with_pos,
    idea_b_result=None,
    output_dir=None,
    title_prefix="",
    n_kmeans=5,
    use_tsne=False,
):
    """
    Idea B · Mode — visualizations aligned with scoring.

    Primary figure (B_clustering_dbscan.png):
      Uses the **same 3D point set + DBSCAN labels** as ``evaluate_idea_b_clustering``
      (default subset=hetero_heavy, eps=1.5), projected to 2D with PCA for display only.
      Does **not** re-run DBSCAN in 2D on all atoms.

    Auxiliary figures: molecule-index / centroid KMeans / CPK / raw XY
    (not used for Mode score).
    """
    from diffdynamic.analysis.pocket.clustering import (
        _build_idea_b_coord_mask,
        _collect_atom_data,
    )
    from diffdynamic.analysis.pocket.plotting_common import _safe_savefig, _viz_title

    if not HAS_MATPLOTLIB:
        print("⚠️ visualize_clustering_2d needs matplotlib")
        return {}
    if not HAS_SKLEARN:
        print("⚠️ visualize_clustering_2d needs scikit-learn")
        return {}

    try:
        from diffdynamic.analysis.pocket_visualization import (
            ACCENT,
            FONT,
            ICEBLUE_EDGE,
            ICEBLUE_FILL,
            NEUTRAL,
            apply_nature_axes,
        )
    except ImportError:
        ACCENT, ICEBLUE_FILL, ICEBLUE_EDGE, NEUTRAL = (
            "#F56E1A",
            "#C5DDF0",
            "#5A7FA0",
            "#4A4A4A",
        )
        FONT = {"title": 11, "label": 9, "tick": 8, "annot": 7}

        def apply_nature_axes(ax, grid_y=True, grid_x=False):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    atom_data = (
        idea_b_result.get("atom_data") if idea_b_result else None
    ) or _collect_atom_data(molecules_with_pos)
    if atom_data is None or len(atom_data["coords"]) < 5:
        print("⚠️ Insufficient atom coordinates for clustering plot")
        return {}

    coords_all = atom_data["coords"]
    mol_idx_all = atom_data["mol_idx"]
    atom_nums_all = atom_data["atom_nums"]
    centroids = atom_data["centroids"]
    n_mols = atom_data["n_mols"]

    # ---- Mode-scored point set (must match evaluate_idea_b_clustering) ----
    subset = "hetero_heavy"
    eps_scored = 1.5
    min_s_scored = 3
    X_mode = None
    labels_mode = None
    score_b = None
    n_clusters_b = None
    n_noise_b = None

    if idea_b_result and idea_b_result.get("success"):
        subset = str(idea_b_result.get("atom_coord_subset") or subset)
        eps_scored = float(idea_b_result.get("dbscan_eps", eps_scored))
        min_s_scored = int(idea_b_result.get("dbscan_min_samples", min_s_scored))
        score_b = idea_b_result.get("score")
        n_clusters_b = idea_b_result.get("n_clusters")
        n_noise_b = idea_b_result.get("n_noise")
        Xf = idea_b_result.get("coords_focus")
        labs = idea_b_result.get("dbscan_labels")
        if Xf is not None and labs is not None:
            X_mode = np.asarray(Xf, dtype=np.float64)
            labels_mode = np.asarray(labs, dtype=int)
            if len(X_mode) != len(labels_mode):
                X_mode, labels_mode = None, None
        if X_mode is None and subset == "all" and labs is not None:
            X_mode = np.asarray(coords_all, dtype=np.float64)
            labels_mode = np.asarray(labs, dtype=int)
            if len(X_mode) != len(labels_mode):
                X_mode, labels_mode = None, None

    # Fallback: rebuild subset + run same 3D DBSCAN as scoring (never 2D all-atom)
    if X_mode is None and DBSCAN is not None:
        if subset == "all":
            X_mode = np.asarray(coords_all, dtype=np.float64)
        else:
            mask = _build_idea_b_coord_mask(molecules_with_pos, subset, 2)
            if mask is not None and mask.sum() >= 5:
                X_mode = np.asarray(coords_all, dtype=np.float64)[mask]
            else:
                X_mode = np.asarray(coords_all, dtype=np.float64)
        clustering = DBSCAN(eps=eps_scored, min_samples=min_s_scored).fit(X_mode)
        labels_mode = clustering.labels_
        n_clusters_b = len(set(labels_mode)) - (1 if -1 in labels_mode else 0)
        n_noise_b = int(np.sum(labels_mode == -1))

    output_dir = Path(output_dir) if output_dir else None
    if not output_dir:
        return {}
    saved = {}

    def _save(name, fig):
        p = output_dir / name
        _safe_savefig(fig, p)
        saved[name.replace(".png", "")] = str(p)

    # ---- Fig 1 (PRIMARY): Mode DBSCAN labels → PCA 2D (display only) ----
    fig1, ax1 = plt.subplots(figsize=(6.2, 5.0))
    if X_mode is not None and len(X_mode) >= 3 and labels_mode is not None:
        pca_m = PCA(n_components=2, random_state=42)
        xy = pca_m.fit_transform(X_mode)
        var = pca_m.explained_variance_ratio_
        unique_lbls = sorted(set(labels_mode.tolist()))
        n_colors = max(len(unique_lbls), 2)
        palette = plt.cm.get_cmap("tab10", n_colors)
        for li, lbl in enumerate(unique_lbls):
            mask = labels_mode == lbl
            color = "#B0B0B0" if lbl == -1 else palette(li % n_colors)
            alpha = 0.35 if lbl == -1 else 0.70
            s = 10 if lbl == -1 else 14
            lname = (
                f"noise ({mask.sum()})" if lbl == -1 else f"mode {lbl} ({mask.sum()})"
            )
            ax1.scatter(
                xy[mask, 0],
                xy[mask, 1],
                c=[color],
                s=s,
                alpha=alpha,
                label=lname,
                rasterized=True,
                edgecolors="none",
                zorder=3,
            )
        apply_nature_axes(ax1, grid_y=True, grid_x=True)
        ax1.set_xlabel(f"PC1 ({var[0] * 100:.1f}%)", fontsize=FONT["label"])
        ax1.set_ylabel(f"PC2 ({var[1] * 100:.1f}%)", fontsize=FONT["label"])
        sc_txt = f"{float(score_b):.3f}" if score_b is not None else "n/a"
        ax1.set_title(
            _viz_title(title_prefix, "B · Mode (3D DBSCAN→PCA)"),
            fontsize=FONT["title"],
            pad=8,
            color=NEUTRAL,
        )
        ax1.text(
            0.02,
            0.98,
            f"subset={subset}  eps={eps_scored:g}  min_s={min_s_scored}\n"
            f"clusters={n_clusters_b}  noise={n_noise_b}  B={sc_txt}",
            transform=ax1.transAxes,
            ha="left",
            va="top",
            fontsize=FONT["annot"],
            color="#555555",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="#DDDDDD",
                alpha=0.92,
            ),
        )
        if len(unique_lbls) <= 12:
            ax1.legend(
                fontsize=FONT["annot"],
                loc="lower right",
                framealpha=0.9,
                markerscale=1.4,
            )
    else:
        ax1.text(
            0.5,
            0.5,
            "Mode DBSCAN unavailable",
            ha="center",
            va="center",
            transform=ax1.transAxes,
            color="#888888",
        )
    plt.tight_layout()
    _save("B_clustering_dbscan.png", fig1)

    # Drop obsolete focus duplicate (main plot is already the scored subset)
    for obsolete in (
        "B_clustering_dbscan_focus.png",
        "B_clustering_dbscan_focus_notext.png",
    ):
        op = output_dir / obsolete
        if op.exists():
            try:
                op.unlink()
            except OSError:
                pass

    # ---- Aux: all-atom PCA colored by molecule index (NOT scored) ----
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords_all)
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(coords_scaled)
    pca_var = pca.explained_variance_ratio_
    centroids_scaled = scaler.transform(centroids)
    centroids_2d = pca.transform(centroids_scaled)
    ax_xlabel = f"PC1 ({pca_var[0] * 100:.1f}%)"
    ax_ylabel = f"PC2 ({pca_var[1] * 100:.1f}%)"

    fig2, ax2 = plt.subplots(figsize=(6.2, 5.0))
    sc2 = ax2.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=mol_idx_all,
        cmap="viridis",
        s=5,
        alpha=0.35,
        rasterized=True,
        vmin=0,
        vmax=max(n_mols - 1, 1),
    )
    plt.colorbar(sc2, ax=ax2, label="Molecule index", shrink=0.85)
    ax2.scatter(
        centroids_2d[:, 0],
        centroids_2d[:, 1],
        c=ACCENT,
        s=42,
        marker="*",
        zorder=6,
        alpha=0.9,
        label=f"Centroids (n={len(centroids_2d)})",
        edgecolors="white",
        linewidths=0.4,
    )
    apply_nature_axes(ax2, grid_y=True, grid_x=True)
    ax2.legend(fontsize=FONT["annot"], loc="upper right")
    ax2.set_xlabel(ax_xlabel, fontsize=FONT["label"])
    ax2.set_ylabel(ax_ylabel, fontsize=FONT["label"])
    ax2.set_title(
        _viz_title(title_prefix, "B aux · all atoms by mol (not scored)"),
        fontsize=FONT["title"],
        pad=8,
        color=NEUTRAL,
    )
    plt.tight_layout()
    _save("B_clustering_molindex.png", fig2)

    # ---- Aux: soft density of Mode points ----
    fig3, ax3 = plt.subplots(figsize=(6.2, 5.0))
    if X_mode is not None and len(X_mode) >= 8:
        pca_d = PCA(n_components=2, random_state=42)
        xy_d = pca_d.fit_transform(X_mode)
        try:
            import seaborn as sns

            sns.kdeplot(
                x=xy_d[:, 0],
                y=xy_d[:, 1],
                ax=ax3,
                levels=6,
                thresh=0.1,
                fill=True,
                cmap="Blues",
                alpha=0.85,
            )
            sns.kdeplot(
                x=xy_d[:, 0],
                y=xy_d[:, 1],
                ax=ax3,
                levels=1,
                thresh=0.12,
                fill=False,
                color=ICEBLUE_EDGE,
                linewidths=1.0,
            )
        except Exception:
            ax3.scatter(xy_d[:, 0], xy_d[:, 1], s=8, c=ICEBLUE_FILL, alpha=0.5)
        apply_nature_axes(ax3, grid_y=True, grid_x=True)
        ax3.set_xlabel("PC1", fontsize=FONT["label"])
        ax3.set_ylabel("PC2", fontsize=FONT["label"])
        ax3.set_title(
            _viz_title(title_prefix, f"B aux · {subset} density"),
            fontsize=FONT["title"],
            pad=8,
            color=NEUTRAL,
        )
    else:
        ax3.text(
            0.5,
            0.5,
            "Density unavailable",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )
    plt.tight_layout()
    _save("B_clustering_kde.png", fig3)

    # ---- Aux: centroid KMeans (stored from scoring; not Mode primary score) ----
    fig4, ax4 = plt.subplots(figsize=(6.2, 5.0))
    kmeans_labels = None
    if idea_b_result and idea_b_result.get("centroid_kmeans_labels"):
        kmeans_labels = np.array(idea_b_result["centroid_kmeans_labels"])
    elif KMeans is not None and len(centroids_2d) >= n_kmeans:
        k = min(n_kmeans, len(centroids_2d))
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_labels = km.fit_predict(centroids_2d)
    if kmeans_labels is not None:
        n_km = len(set(kmeans_labels))
        pal_km = plt.cm.get_cmap("Set2", max(n_km, 2))
        for ki in range(n_km):
            mk = kmeans_labels == ki
            ax4.scatter(
                centroids_2d[mk, 0],
                centroids_2d[mk, 1],
                c=[pal_km(ki)],
                s=70,
                marker="o",
                label=f"cent {ki} ({mk.sum()})",
                edgecolors="white",
                linewidths=0.7,
                zorder=4,
            )
        ax4.legend(fontsize=FONT["annot"], loc="best", ncol=2 if n_km > 5 else 1)
    else:
        ax4.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c=ICEBLUE_EDGE, s=50)
    apply_nature_axes(ax4, grid_y=True, grid_x=True)
    ax4.set_xlabel("PC1", fontsize=FONT["label"])
    ax4.set_ylabel("PC2", fontsize=FONT["label"])
    ax4.set_title(
        _viz_title(title_prefix, "B aux · mol centroids KMeans"),
        fontsize=FONT["title"],
        pad=8,
        color=NEUTRAL,
    )
    plt.tight_layout()
    _save("B_clustering_kmeans.png", fig4)

    # ---- Aux: CPK on Mode atoms when possible ----
    fig5, ax5 = plt.subplots(figsize=(6.2, 5.0))
    if X_mode is not None and subset != "all":
        mask = _build_idea_b_coord_mask(molecules_with_pos, subset, 2)
        if mask is not None and mask.sum() == len(X_mode):
            atom_nums_plot = atom_nums_all[mask]
            pca_c = PCA(n_components=2, random_state=42)
            xy_c = pca_c.fit_transform(X_mode)
        else:
            atom_nums_plot = atom_nums_all
            xy_c = coords_2d
    else:
        atom_nums_plot = atom_nums_all
        xy_c = coords_2d
    unique_atypes = sorted(set(np.asarray(atom_nums_plot).tolist()))
    for anum in unique_atypes:
        mask_a = np.asarray(atom_nums_plot) == anum
        color_a = _CPK_COLORS.get(int(anum), "#888888")
        name_a = _ELEMENT_NAMES.get(int(anum), f"Z={anum}")
        ax5.scatter(
            xy_c[mask_a, 0],
            xy_c[mask_a, 1],
            c=color_a,
            s=10,
            alpha=0.65,
            label=f"{name_a} ({mask_a.sum()})",
            rasterized=True,
        )
    apply_nature_axes(ax5, grid_y=True, grid_x=True)
    ax5.set_xlabel("PC1", fontsize=FONT["label"])
    ax5.set_ylabel("PC2", fontsize=FONT["label"])
    ax5.set_title(
        _viz_title(title_prefix, f"B aux · CPK ({subset})"),
        fontsize=FONT["title"],
        pad=8,
        color=NEUTRAL,
    )
    ax5.legend(
        fontsize=FONT["annot"],
        loc="best",
        markerscale=1.6,
        framealpha=0.92,
        ncol=2 if len(unique_atypes) > 6 else 1,
    )
    plt.tight_layout()
    _save("B_clustering_cpk.png", fig5)

    # ---- Aux: raw XY of Mode coords ----
    fig6, ax6 = plt.subplots(figsize=(6.2, 5.0))
    if X_mode is not None:
        sc6 = ax6.scatter(
            X_mode[:, 0],
            X_mode[:, 1],
            c=labels_mode if labels_mode is not None else "steelblue",
            cmap="tab10",
            s=8,
            alpha=0.55,
            rasterized=True,
        )
        if labels_mode is not None:
            plt.colorbar(sc6, ax=ax6, label="DBSCAN label", shrink=0.85)
        ax6.scatter(
            centroids[:, 0],
            centroids[:, 1],
            c=ACCENT,
            s=28,
            marker="D",
            alpha=0.85,
            zorder=5,
            label="Mol centroids",
            edgecolors="white",
            linewidths=0.4,
        )
        ax6.legend(fontsize=FONT["annot"])
    apply_nature_axes(ax6, grid_y=True, grid_x=True)
    ax6.set_xlabel("X (Å)", fontsize=FONT["label"])
    ax6.set_ylabel("Y (Å)", fontsize=FONT["label"])
    ax6.set_title(
        _viz_title(title_prefix, f"B aux · {subset} XY (scored labels)"),
        fontsize=FONT["title"],
        pad=8,
        color=NEUTRAL,
    )
    plt.tight_layout()
    _save("B_clustering_xy.png", fig6)

    _ = (use_tsne, ICEBLUE_FILL)
    return saved


def visualize_size_distribution(idea_g_result, output_dir=None, title_prefix=""):
    """
    Auxiliary — molecular size distribution (not scored as G; anchors are G)

    Subplots: MW hist / atom count hist / MW vs atom count scatter
    """
    from diffdynamic.analysis.pocket.plotting_common import _safe_savefig, _viz_title

    if not HAS_MATPLOTLIB:
        return {}
    if not idea_g_result.get("success"):
        return {}
    mw_list = np.array(idea_g_result.get("mw_list", []))
    natoms_list = np.array(idea_g_result.get("n_atoms_list", []))
    if len(mw_list) < 2:
        return {}
    output_dir = Path(output_dir) if output_dir else None
    if not output_dir:
        return {}
    saved = {}
    mw_mean = float(np.mean(mw_list))
    mw_cv = idea_g_result.get("mw_cv", 0) or 0

    # ---- Fig 1: MW distribution ----
    fig1, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        mw_list,
        bins=25,
        color="cornflowerblue",
        edgecolor="white",
        alpha=0.8,
        density=True,
    )
    if HAS_SCIPY and gaussian_kde is not None and len(mw_list) > 3:
        kde = gaussian_kde(mw_list, bw_method="scott")
        xm = np.linspace(mw_list.min() * 0.9, mw_list.max() * 1.1, 300)
        ax.plot(xm, kde(xm), "r-", lw=2, label="KDE")
    ax.axvline(
        mw_mean, color="orange", lw=1.5, linestyle="--", label=f"Mean={mw_mean:.0f}"
    )
    ax.axvline(500, color="green", lw=1.2, linestyle=":", label="Lipinski MW<=500")
    ax.set_xlabel("Molecular weight (Da)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title(_viz_title(title_prefix, "Aux · MW"), fontsize=10, pad=6)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _safe_savefig(fig1, output_dir / "aux_size_mw.png")
    saved["aux_size_mw"] = str(output_dir / "aux_size_mw.png")

    # ---- Fig 2: Atom count distribution ----
    fig2, ax = plt.subplots(figsize=(6, 4))
    if len(natoms_list) > 0:
        bins_a = range(int(natoms_list.min()), int(natoms_list.max()) + 2)
        ax.hist(
            natoms_list, bins=bins_a, color="mediumorchid", edgecolor="white", alpha=0.8
        )
        ax.axvline(
            float(np.mean(natoms_list)),
            color="orange",
            lw=1.5,
            linestyle="--",
            label=f"Mean={np.mean(natoms_list):.1f}",
        )
        ax.set_xlabel("Atom count (incl. H)", fontsize=9)
        ax.set_ylabel("Molecule count", fontsize=9)
        ax.legend(fontsize=7)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(
            0.5,
            0.5,
            "No atom count data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )
    plt.tight_layout()
    _safe_savefig(fig2, output_dir / "aux_size_atoms.png")
    saved["aux_size_atoms"] = str(output_dir / "aux_size_atoms.png")

    # ---- Fig 3: MW vs atom count ----
    fig3, ax = plt.subplots(figsize=(6, 4))
    mw_arr = np.array(mw_list)
    if len(natoms_list) >= len(mw_arr):
        natoms_aligned = natoms_list[: len(mw_arr)]
    else:
        natoms_aligned = natoms_list
        mw_arr = mw_arr[: len(natoms_list)]
    sc = ax.scatter(
        natoms_aligned,
        mw_arr,
        c=mw_arr,
        cmap="coolwarm",
        s=25,
        alpha=0.6,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="MW (Da)", shrink=0.85)
    if len(natoms_aligned) > 2:
        z = np.polyfit(natoms_aligned, mw_arr, 1)
        p = np.poly1d(z)
        x_line = np.linspace(natoms_aligned.min(), natoms_aligned.max(), 100)
        ax.plot(
            x_line, p(x_line), "r--", lw=1.5, label=f"Linear trend (slope={z[0]:.1f})"
        )
        ax.legend(fontsize=7)
    ax.set_xlabel("Atom count", fontsize=9)
    ax.set_ylabel("Molecular weight (Da)", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _safe_savefig(fig3, output_dir / "aux_size_mw_vs_atoms.png")
    saved["aux_size_mw_vs_atoms"] = str(output_dir / "aux_size_mw_vs_atoms.png")
    return saved


def visualize_pocket_volume(idea_h_result, output_dir=None, title_prefix=""):
    """
    Idea H — pocket cavity volume (ligand-centered, protein-excluded MC when available).
    Full-score band 350–750 Å³; Nature iceblue / accent styling.
    """
    from diffdynamic.analysis.pocket.plotting_common import _safe_savefig, _viz_title

    if not HAS_MATPLOTLIB:
        return {}
    if not idea_h_result.get("success"):
        return {}
    vol = idea_h_result.get("volume_ang3")
    if vol is None:
        return {}
    output_dir = Path(output_dir) if output_dir else None
    if not output_dir:
        return {}
    try:
        from diffdynamic.analysis.pocket_visualization import (
            ACCENT,
            BAND_HIGH,
            BAND_MED,
            FONT,
            ICEBLUE_EDGE,
            ICEBLUE_FILL,
            apply_nature_axes,
        )
    except ImportError:
        ACCENT, ICEBLUE_FILL, ICEBLUE_EDGE = "#F56E1A", "#C5DDF0", "#5A7FA0"
        BAND_HIGH, BAND_MED = "#A8C5A0", "#F0D9A8"
        FONT = {"title": 11, "label": 9, "tick": 8, "annot": 7}

        def apply_nature_axes(ax, grid_y=True, grid_x=False):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    score = float(idea_h_result.get("score") or 0)
    vol = float(vol)
    rmin = float(idea_h_result.get("optimal_vol_min", IDEA_H_VIZ_REF_OPT_MIN))
    rmax = float(idea_h_result.get("optimal_vol_max", IDEA_H_VIZ_REF_OPT_MAX))
    rzb = float(idea_h_result.get("vol_score_zero_below", IDEA_H_VIZ_REF_ZERO_BELOW))
    rza = float(idea_h_result.get("vol_score_zero_above", IDEA_H_VIZ_REF_ZERO_ABOVE))
    radius = float(idea_h_result.get("centroid_radius", 10.0) or 10.0)
    method = str(idea_h_result.get("volume_method") or "")
    if "cavity" in method or "protein" in method:
        method_tag = "cav·prot+lig"
    elif "fallback" in method:
        method_tag = "MC·lig-only(fallback)"
    else:
        method_tag = "MC·lig-only"

    x_min = 0.0
    x_cap = float(IDEA_H_VIZ_X_MAX_DEFAULT)
    x_max = x_cap if vol <= x_cap else float(vol) * 1.06

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    # zero tails
    if rzb > 0:
        ax.axvspan(x_min, min(rzb, x_max), color="#E8C4C0", alpha=0.35, zorder=1)
    if rza < x_max:
        ax.axvspan(max(rza, x_min), x_max, color="#E8C4C0", alpha=0.35, zorder=1)
    # sub-optimal
    if rzb < rmin:
        ax.axvspan(
            max(x_min, rzb), min(rmin, x_max), color=BAND_MED, alpha=0.45, zorder=1
        )
    if rmax < rza:
        ax.axvspan(
            max(x_min, rmax), min(rza, x_max), color=BAND_MED, alpha=0.45, zorder=1
        )
    # full-score band
    ax.axvspan(rmin, rmax, color=BAND_HIGH, alpha=0.55, zorder=2)
    ax.axvline(rmin, color="#6B8F71", ls=":", lw=1.0, alpha=0.8, zorder=3)
    ax.axvline(rmax, color="#6B8F71", ls=":", lw=1.0, alpha=0.8, zorder=3)

    ax.barh(
        0,
        vol,
        left=0,
        height=0.34,
        color=ICEBLUE_FILL,
        edgecolor=ICEBLUE_EDGE,
        linewidth=1.0,
        zorder=4,
    )
    ax.axvline(vol, color=ACCENT, ls="-", lw=1.6, alpha=0.95, zorder=5)
    ax.plot(
        [vol],
        [0],
        marker="o",
        markersize=7,
        color=ACCENT,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=6,
    )

    apply_nature_axes(ax, grid_y=False, grid_x=True)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([])
    ax.set_xlabel("Volume (Å³)", fontsize=FONT["label"])
    ttl = _viz_title(title_prefix, "H · Pocket volume")
    ax.set_title(ttl, fontsize=FONT["title"], pad=8)
    ax.text(
        0.02,
        0.96,
        f"{method_tag} · r={radius:g} Å  ·  band {rmin:.0f}–{rmax:.0f}  ·  H={score:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT["annot"],
        color="#666666",
    )
    ax.text(
        vol,
        0.28,
        f"V={vol:.0f}",
        ha="center",
        va="bottom",
        fontsize=FONT["label"],
        color=ACCENT,
    )
    plt.tight_layout()
    path = output_dir / "H_pocket_volume.png"
    _safe_savefig(fig, path, dpi=180)
    return {"H_pocket_volume": str(path)}
