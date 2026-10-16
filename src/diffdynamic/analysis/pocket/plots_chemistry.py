"""Plots Chemistry responsibilities extracted from evaluate_pocket_quality.py."""

from diffdynamic.analysis.pocket.state import (
    HAS_MATPLOTLIB,
    HAS_SCIPY,
    HAS_SKLEARN,
    IDEA_C_LE_SCORE_HIGH,
    IDEA_C_LE_SCORE_LOW,
    MaxNLocator,
    PCA,
    Path,
    TSNE,
    gaussian_kde,
    np,
    plt,
)


def visualize_vina_le_combined(
    idea_a_result=None,
    idea_le_result=None,
    output_dir=None,
    title_prefix="",
):
    """
    Idea A: Vina + LE as two narrow box+strip columns on one frame.

    Twin y-axes only (no colorbars): left = Vina, right = LE.
    Points keep RdYlGn coloring; scales are read from the axes.
    """
    from diffdynamic.analysis.pocket.plotting_common import _safe_savefig, _viz_title

    if not HAS_MATPLOTLIB:
        return {}
    output_dir = Path(output_dir) if output_dir else None
    if not output_dir:
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
            if grid_y:
                ax.grid(True, axis="y", alpha=0.3, zorder=0)

    # ---- prepare Vina ----
    vina_ok = False
    vina = np.array([], dtype=np.float64)
    if idea_a_result and idea_a_result.get("success"):
        raw = (
            idea_a_result.get("vina_scores_valid")
            or idea_a_result.get("vina_scores")
            or []
        )
        vina = np.asarray(raw, dtype=np.float64)
        vina = vina[np.isfinite(vina)]
        vina = vina[vina < 0]
        vina_ok = len(vina) > 0

    # ---- prepare LE ----
    le_src = idea_le_result or {}
    if not le_src.get("success") and idea_a_result:
        le_src = idea_a_result.get("le_component") or {}
    le_ok = False
    le = np.array([], dtype=np.float64)
    lo = float(IDEA_C_LE_SCORE_LOW)
    hi = float(IDEA_C_LE_SCORE_HIGH)
    if le_src and le_src.get("success"):
        le = np.asarray(le_src.get("le_list") or [], dtype=np.float64)
        le = le[np.isfinite(le)]
        lo = float(le_src.get("le_low_for_score", lo))
        hi = float(le_src.get("le_high_for_score", hi))
        le_ok = len(le) > 0

    if not vina_ok and not le_ok:
        return {}

    def _box_strip(ax, scores, *, x, cmap, vmin, vmax, seed, facecolor):
        rng = np.random.default_rng(seed)
        ax.boxplot(
            scores,
            vert=True,
            patch_artist=True,
            widths=0.28,
            positions=[x],
            manage_ticks=False,
            boxprops=dict(facecolor=facecolor, edgecolor=ICEBLUE_EDGE, alpha=0.85),
            medianprops=dict(color="red", lw=2.0),
            whiskerprops=dict(color="#888888", lw=1.0),
            capprops=dict(color="#888888", lw=1.0),
            flierprops=dict(
                marker="o",
                markersize=3.5,
                markerfacecolor="none",
                markeredgecolor="#555555",
                markeredgewidth=0.7,
            ),
        )
        jitter = rng.uniform(-0.10, 0.10, len(scores))
        # colored points; no colorbar — twin y-axes carry the numeric scales
        ax.scatter(
            x + jitter,
            scores,
            c=scores,
            cmap=cmap,
            s=18,
            alpha=0.65,
            zorder=3,
            vmin=vmin,
            vmax=vmax,
            edgecolors="none",
        )

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax_r = ax.twinx()
    xticks, xlabels = [], []
    x_v, x_l = 0.0, 1.0

    if vina_ok:
        mean_v = float(np.mean(vina))
        pct_hit = float(np.mean(vina <= -7.0) * 100)
        _box_strip(
            ax,
            vina,
            x=x_v,
            cmap="RdYlGn_r",
            vmin=float(vina.min()),
            vmax=float(vina.max()),
            seed=42,
            facecolor=ICEBLUE_FILL,
        )
        ax.hlines(
            -7.0,
            x_v - 0.38,
            x_v + 0.38,
            colors="red",
            linestyles="--",
            lw=1.15,
            alpha=0.85,
            zorder=5,
        )
        y_pad = max(0.5, float(np.ptp(vina)) * 0.08 + 0.1)
        ax.set_ylim(float(vina.min()) - y_pad, float(vina.max()) + y_pad)
        ax.set_ylabel("Vina (kcal/mol)", fontsize=FONT["label"], color=ICEBLUE_EDGE)
        ax.tick_params(axis="y", labelcolor=ICEBLUE_EDGE, labelsize=FONT["tick"])
        xticks.append(x_v)
        xlabels.append("Vina")
        ax.text(
            0.02,
            0.98,
            f"Vina mean={mean_v:.2f}  n={len(vina)}  P(≤−7)={pct_hit:.0f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONT["annot"],
            color="#666666",
        )
    else:
        ax.set_yticks([])
        ax.set_ylabel("")

    if le_ok:
        mean_le = float(np.mean(le))
        _box_strip(
            ax_r,
            le,
            x=x_l,
            cmap="RdYlGn",
            vmin=lo,
            vmax=hi,
            seed=43,
            facecolor="#FBC9A6",
        )
        ax_r.hlines(
            lo,
            x_l - 0.38,
            x_l + 0.38,
            colors="red",
            linestyles="--",
            lw=1.0,
            alpha=0.8,
            zorder=5,
        )
        ax_r.hlines(
            hi,
            x_l - 0.38,
            x_l + 0.38,
            colors="green",
            linestyles="--",
            lw=1.0,
            alpha=0.8,
            zorder=5,
        )
        y_pad = max(0.02, float(np.ptp(le)) * 0.08 + 0.01)
        ax_r.set_ylim(float(le.min()) - y_pad, float(le.max()) + y_pad)
        ax_r.set_ylabel("LE (kcal·mol⁻¹·heavy⁻¹)", fontsize=FONT["label"], color=ACCENT)
        ax_r.tick_params(axis="y", labelcolor=ACCENT, labelsize=FONT["tick"])
        xticks.append(x_l)
        xlabels.append("LE")
        ax.text(
            0.98,
            0.98,
            f"LE mean={mean_le:.3f}  n={len(le)}  map {lo:.2f}–{hi:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=FONT["annot"],
            color="#666666",
        )
    else:
        ax_r.set_yticks([])
        ax_r.set_ylabel("")

    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=FONT["label"])

    # Left axis: Nature style but keep room for twin right axis
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(ICEBLUE_EDGE)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_color("#888888")
    ax.spines["bottom"].set_linewidth(0.8)
    ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Right axis: LE numeric scale only (no colorbar)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["left"].set_visible(False)
    ax_r.spines["bottom"].set_visible(False)
    ax_r.spines["right"].set_visible(True)
    ax_r.spines["right"].set_color(ACCENT)
    ax_r.spines["right"].set_linewidth(1.0)

    ax.set_title(
        _viz_title(title_prefix, "A · Vina & LE"),
        fontsize=FONT["title"],
        pad=8,
        color=NEUTRAL,
    )
    fig.tight_layout()
    path = output_dir / "A_vina_le.png"
    _safe_savefig(fig, path, dpi=180)

    for obsolete in (
        "A_vina_hist.png",
        "A_vina_hist_notext.png",
        "A_vina_box.png",
        "A_vina_box_notext.png",
        "A_vina_cdf.png",
        "A_vina_cdf_notext.png",
        "A_le_hist.png",
        "A_le_hist_notext.png",
        "A_le_box.png",
        "A_le_box_notext.png",
        "A_le_cdf.png",
        "A_le_cdf_notext.png",
    ):
        op = output_dir / obsolete
        if op.exists():
            try:
                op.unlink()
            except OSError:
                pass

    return {"A_vina_le": str(path)}


def visualize_vina_distribution(
    idea_a_result, output_dir=None, title_prefix="", idea_le_result=None
):
    """Backward-compatible wrapper → combined Vina+LE figure."""
    return visualize_vina_le_combined(
        idea_a_result=idea_a_result,
        idea_le_result=idea_le_result,
        output_dir=output_dir,
        title_prefix=title_prefix,
    )


def visualize_le_distribution(
    idea_c_result, output_dir=None, title_prefix="", idea_a_result=None
):
    """Backward-compatible wrapper → combined Vina+LE figure."""
    return visualize_vina_le_combined(
        idea_a_result=idea_a_result,
        idea_le_result=idea_c_result,
        output_dir=output_dir,
        title_prefix=title_prefix,
    )


def visualize_druglikeness(
    idea_d_result, output_dir=None, title_prefix="", vina_scores=None
):
    """
    Idea D - Druglikeness multi-dimensional visualization (5 separate figures)

    Subplots: QED vs SA scatter (color=vinadock) / QED dist / SA dist / Lipinski / radar
    """
    from diffdynamic.analysis.pocket.plotting_common import _safe_savefig, _viz_title

    if not HAS_MATPLOTLIB:
        return {}
    if not idea_d_result.get("success"):
        return {}
    qed_list = np.array(idea_d_result.get("qed_list", []))
    sa_list = np.array(idea_d_result.get("sa_list", []))
    lip_list = np.array(idea_d_result.get("lipinski_list", []))
    if len(qed_list) == 0:
        return {}
    output_dir = Path(output_dir) if output_dir else None
    if not output_dir:
        return {}
    saved = {}
    vina_arr = None
    if vina_scores and len(vina_scores) > 0:
        try:
            vina_arr = np.asarray(vina_scores, dtype=np.float64)
        except (TypeError, ValueError):
            vina_arr = None

    # ---- Fig 1: QED vs SA scatter (color=vinadock) ----
    n_pts = min(len(qed_list), len(sa_list))
    try:
        from diffdynamic.analysis.pocket_visualization import (
            ACCENT,
            FONT,
            ICEBLUE_EDGE,
            ICEBLUE_FILL,
            apply_nature_axes,
        )
        from matplotlib.colors import LinearSegmentedColormap

        _cmap = LinearSegmentedColormap.from_list(
            "ice_accent_vina",
            [ACCENT, "#FBC9A6", ICEBLUE_FILL, ICEBLUE_EDGE],
        )
    except ImportError:
        ACCENT, ICEBLUE_FILL, ICEBLUE_EDGE = "#F56E1A", "#C5DDF0", "#5A7FA0"
        FONT = {"title": 11, "label": 9, "tick": 8, "annot": 7}
        _cmap = "cividis_r"

        def apply_nature_axes(ax, grid_y=True, grid_x=False):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    use_vina = (
        vina_arr is not None
        and len(vina_arr) > 0
        and np.isfinite(vina_arr[:n_pts]).any()
    )
    if use_vina:
        c_vals = np.asarray(vina_arr[:n_pts], dtype=np.float64)
        # if shorter than n_pts, trim scatter to vina length
        n_use = min(n_pts, len(vina_arr))
        c_vals = np.asarray(vina_arr[:n_use], dtype=np.float64)
        q_use = qed_list[:n_use]
        s_use = sa_list[:n_use]
        finite = np.isfinite(c_vals)
        if finite.any():
            c_label = "Vina (kcal/mol)"
            c_vmin = float(np.nanmin(c_vals))
            c_vmax = float(np.nanmax(c_vals))
            c_cmap = _cmap
        else:
            use_vina = False
    if not use_vina:
        n_use = n_pts
        q_use = qed_list[:n_use]
        s_use = sa_list[:n_use]
        c_vals = lip_list[:n_use] if len(lip_list) >= n_use else np.ones(n_use)
        c_label = "Lipinski (0–1)"
        c_vmin, c_vmax = 0, 1
        c_cmap = "Blues"
    fig1, ax1 = plt.subplots(figsize=(5.8, 4.6))
    sc = ax1.scatter(
        q_use,
        s_use,
        c=c_vals,
        cmap=c_cmap,
        s=36,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.3,
        vmin=c_vmin,
        vmax=c_vmax,
        zorder=3,
    )
    cb = fig1.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
    cb.set_label(c_label, fontsize=FONT["annot"])
    cb.ax.tick_params(labelsize=FONT["tick"])
    ax1.axvline(0.5, color="#BBBBBB", linestyle=":", lw=1.0, alpha=0.8)
    ax1.axhline(0.5, color="#BBBBBB", linestyle=":", lw=1.0, alpha=0.8)
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (0.5, 0.5),
        0.5,
        0.5,
        linewidth=0,
        edgecolor="none",
        facecolor="#A8C5A0",
        alpha=0.12,
        zorder=1,
    )
    ax1.add_patch(rect)
    qed_m = idea_d_result.get("qed_mean", 0)
    sa_m = idea_d_result.get("sa_mean", 0)
    ax1.scatter(
        [qed_m],
        [sa_m],
        c=ACCENT,
        s=90,
        marker="*",
        zorder=10,
        edgecolors="white",
        linewidths=0.5,
    )
    apply_nature_axes(ax1)
    ax1.set_xlabel("QED", fontsize=FONT["label"])
    ax1.set_ylabel("SA", fontsize=FONT["label"])
    ax1.set_xlim(0, 1.02)
    ax1.set_ylim(0, 1.02)
    ax1.set_title(
        _viz_title(title_prefix, "C · QED vs SA"), fontsize=FONT["title"], pad=8
    )
    plt.tight_layout()
    _safe_savefig(fig1, output_dir / "C_druglikeness_qed_vs_sa.png", dpi=180)
    saved["C_druglikeness_qed_vs_sa"] = str(output_dir / "C_druglikeness_qed_vs_sa.png")

    # ---- Fig 2: QED distribution ----
    fig2, ax2 = plt.subplots(figsize=(5.0, 3.8))
    ax2.hist(
        qed_list,
        bins=20,
        color=ICEBLUE_FILL,
        edgecolor="white",
        alpha=0.95,
        density=True,
        zorder=3,
    )
    if HAS_SCIPY and gaussian_kde is not None and len(qed_list) > 3:
        kde = gaussian_kde(qed_list, bw_method=0.25)
        xq = np.linspace(0, 1, 200)
        ax2.plot(xq, kde(xq), color=ICEBLUE_EDGE, lw=1.6)
    ax2.axvline(float(np.mean(qed_list)), color=ACCENT, lw=1.2, ls="--")
    apply_nature_axes(ax2)
    ax2.set_xlabel("QED", fontsize=FONT["label"])
    ax2.set_ylabel("Density", fontsize=FONT["label"])
    ax2.set_title(_viz_title(title_prefix, "C · QED"), fontsize=FONT["title"], pad=8)
    plt.tight_layout()
    _safe_savefig(fig2, output_dir / "C_druglikeness_qed_dist.png", dpi=180)
    saved["C_druglikeness_qed_dist"] = str(output_dir / "C_druglikeness_qed_dist.png")

    # ---- Fig 3: SA distribution ----
    fig3, ax3 = plt.subplots(figsize=(5.0, 3.8))
    ax3.hist(
        sa_list,
        bins=20,
        color=ICEBLUE_FILL,
        edgecolor="white",
        alpha=0.95,
        density=True,
        zorder=3,
    )
    if HAS_SCIPY and gaussian_kde is not None and len(sa_list) > 3:
        kde = gaussian_kde(sa_list, bw_method=0.25)
        xs = np.linspace(0, 1, 200)
        ax3.plot(xs, kde(xs), color=ICEBLUE_EDGE, lw=1.6)
    ax3.axvline(float(np.mean(sa_list)), color=ACCENT, lw=1.2, ls="--")
    apply_nature_axes(ax3)
    ax3.set_xlabel("SA", fontsize=FONT["label"])
    ax3.set_ylabel("Density", fontsize=FONT["label"])
    ax3.set_title(_viz_title(title_prefix, "C · SA"), fontsize=FONT["title"], pad=8)
    plt.tight_layout()
    _safe_savefig(fig3, output_dir / "C_druglikeness_sa_dist.png", dpi=180)
    saved["C_druglikeness_sa_dist"] = str(output_dir / "C_druglikeness_sa_dist.png")

    # ---- Fig 4: Lipinski compliance ----
    fig4, ax4 = plt.subplots(figsize=(5.0, 3.8))
    if len(lip_list) > 0:
        ax4.hist(
            lip_list * 5,
            bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            color=ICEBLUE_EDGE,
            edgecolor="white",
            alpha=0.9,
            zorder=3,
        )
        ax4.set_xlabel("Lipinski rules passed", fontsize=FONT["label"])
        ax4.set_ylabel("Count", fontsize=FONT["label"])
        ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax4.axvline(4, color=ACCENT, lw=1.2, linestyle="--")
        apply_nature_axes(ax4)
        ax4.set_title(
            _viz_title(title_prefix, "C · Lipinski"), fontsize=FONT["title"], pad=8
        )
    else:
        ax4.text(
            0.5,
            0.5,
            "No Lipinski data",
            ha="center",
            va="center",
            transform=ax4.transAxes,
            fontsize=10,
        )
    plt.tight_layout()
    _safe_savefig(fig4, output_dir / "C_druglikeness_lipinski.png", dpi=180)
    saved["C_druglikeness_lipinski"] = str(output_dir / "C_druglikeness_lipinski.png")

    # ---- Fig 5: 5-axis druglikeness radar (pentagon); overall only in caption ----
    fig5 = plt.figure(figsize=(5.6, 5.4))
    ax5 = fig5.add_subplot(111, polar=True)
    categories = ["QED", "SA", "Lipinski", "Veber", "PAINS-free"]
    qed_m = float(idea_d_result.get("qed_mean", 0) or 0)
    sa_m = float(idea_d_result.get("sa_mean", 0) or 0)
    lip_m = float(idea_d_result.get("lipinski_mean", 0) or 0)
    veber_r = float(idea_d_result.get("veber_ratio", 0) or 0)
    pains_r = float(idea_d_result.get("pains_ratio", 0) or 0)
    overall = float(idea_d_result.get("score", 0) or 0)
    values = [
        float(min(1.0, qed_m)),
        float(min(1.0, sa_m)),
        float(np.clip(lip_m, 0.0, 1.0)),
        float(np.clip(veber_r, 0.0, 1.0)),
        float(np.clip(1.0 - pains_r, 0.0, 1.0)),
    ]
    N = 5
    assert len(categories) == N and len(values) == N
    # equal 72° spacing → closed pentagon
    angles = (np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)).tolist()
    angles_c = angles + angles[:1]
    values_c = values + values[:1]

    ax5.set_facecolor("#FAFBFC")
    # clear default polar spokes so only our 5 axes remain
    ax5.set_xticks([])
    ax5.plot(
        angles_c,
        values_c,
        "o-",
        lw=1.9,
        color=ICEBLUE_EDGE,
        markersize=6,
        markerfacecolor=ICEBLUE_FILL,
        markeredgecolor=ICEBLUE_EDGE,
        zorder=3,
    )
    ax5.fill(angles_c, values_c, alpha=0.28, color=ICEBLUE_FILL, zorder=2)
    ax5.set_thetagrids(np.degrees(angles), categories, fontsize=FONT["tick"])
    ax5.set_ylim(0, 1.0)
    ax5.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax5.set_yticklabels([])
    ax5.spines["polar"].set_color("#B0B0B0")
    ax5.grid(True, color="#E8E8E8", linewidth=0.7)
    ax5.set_title(
        _viz_title(title_prefix, "C · Druglikeness"),
        fontsize=FONT["title"],
        pad=16,
    )
    # keep caption inside figure so bbox_inches='tight' does not crop it
    fig5.text(
        0.5,
        0.035,
        f"C overall = {overall:.3f}",
        ha="center",
        va="bottom",
        fontsize=FONT["annot"] + 1,
        color="#555555",
    )
    fig5.subplots_adjust(top=0.86, bottom=0.12, left=0.10, right=0.90)
    path = output_dir / "C_druglikeness_radar.png"
    _safe_savefig(fig5, path, dpi=180)
    saved["C_druglikeness_radar"] = str(path)
    return saved


def visualize_fingerprint_diversity(
    molecules_with_pos,
    idea_f_result=None,
    output_dir=None,
    title_prefix="",
    color_by_qed=None,
    max_mols=150,
):
    """
    Idea F - Chemical space diversity (Morgan FP) (4 separate figures)

    Subplots: PCA / t-SNE / Tanimoto heatmap / uniqueness
    """
    from diffdynamic.analysis.pocket.clustering import _compute_morgan_fingerprints
    from diffdynamic.analysis.pocket.plotting_common import _safe_savefig

    if not HAS_MATPLOTLIB or not HAS_SKLEARN:
        return {}
    fps_matrix = None
    if idea_f_result and idea_f_result.get("fps_matrix") is not None:
        fps_matrix = idea_f_result["fps_matrix"]
    if fps_matrix is None:
        fps_matrix, _ = _compute_morgan_fingerprints(molecules_with_pos)
    if fps_matrix is None or len(fps_matrix) < 4:
        return {}
    output_dir = Path(output_dir) if output_dir else None
    if not output_dir:
        return {}
    saved = {}

    n_fps = len(fps_matrix)
    if n_fps > max_mols:
        idx_sub = np.random.choice(n_fps, max_mols, replace=False)
        fps_sub = fps_matrix[idx_sub]
    else:
        idx_sub = np.arange(n_fps)
        fps_sub = fps_matrix

    qed_colors = None
    if color_by_qed is not None and len(color_by_qed) == n_fps:
        qed_colors = np.array(color_by_qed)[idx_sub]
    c_vals = qed_colors if qed_colors is not None else np.arange(len(fps_sub))

    # ---- Fig 1: PCA ----
    pca_fp = PCA(n_components=2, random_state=42)
    fps_2d_pca = pca_fp.fit_transform(fps_sub)
    var_ratio = pca_fp.explained_variance_ratio_
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    sc1 = ax1.scatter(
        fps_2d_pca[:, 0],
        fps_2d_pca[:, 1],
        c=c_vals,
        cmap="plasma",
        s=30,
        alpha=0.7,
        edgecolors="none",
    )
    plt.colorbar(
        sc1,
        ax=ax1,
        label="QED" if qed_colors is not None else "Molecule index",
        shrink=0.85,
    )
    ax1.set_xlabel(f"PC1 ({var_ratio[0] * 100:.1f}%)", fontsize=9)
    ax1.set_ylabel(f"PC2 ({var_ratio[1] * 100:.1f}%)", fontsize=9)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    _safe_savefig(fig1, output_dir / "F_fingerprint_pca.png")
    saved["F_fingerprint_pca"] = str(output_dir / "F_fingerprint_pca.png")

    # ---- Fig 2: t-SNE ----
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    if TSNE is not None and len(fps_sub) >= 4:
        perp = min(30, len(fps_sub) - 1)
        try:
            tsne_fp = TSNE(
                n_components=2, perplexity=perp, random_state=42, n_jobs=1, max_iter=500
            )
        except TypeError:
            tsne_fp = TSNE(
                n_components=2, perplexity=perp, random_state=42, n_jobs=1, n_iter=500
            )
        fps_2d_tsne = tsne_fp.fit_transform(fps_sub)
        sc2 = ax2.scatter(
            fps_2d_tsne[:, 0],
            fps_2d_tsne[:, 1],
            c=c_vals,
            cmap="plasma",
            s=30,
            alpha=0.7,
            edgecolors="none",
        )
        plt.colorbar(
            sc2,
            ax=ax2,
            label="QED" if qed_colors is not None else "Molecule index",
            shrink=0.85,
        )
        ax2.set_xlabel("t-SNE 1", fontsize=9)
        ax2.set_ylabel("t-SNE 2", fontsize=9)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            "t-SNE unavailable",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=10,
        )
    plt.tight_layout()
    _safe_savefig(fig2, output_dir / "F_fingerprint_tsne.png")
    saved["F_fingerprint_tsne"] = str(output_dir / "F_fingerprint_tsne.png")

    # ---- Fig 3: Tanimoto heatmap ----
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    n_heat = min(len(fps_sub), 60)
    fps_heat = fps_sub[:n_heat]
    norms = np.linalg.norm(fps_heat, axis=1, keepdims=True) + 1e-8
    fps_normed = fps_heat / norms
    sim_mat = fps_normed @ fps_normed.T
    np.fill_diagonal(sim_mat, 1.0)
    im = ax3.imshow(sim_mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax3, label="Cosine similarity", shrink=0.85)
    ax3.set_xlabel("Molecule index", fontsize=9)
    ax3.set_ylabel("Molecule index", fontsize=9)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    _safe_savefig(fig3, output_dir / "F_fingerprint_heatmap.png")
    saved["F_fingerprint_heatmap"] = str(output_dir / "F_fingerprint_heatmap.png")

    # ---- Fig 4: Uniqueness ----
    fig4, ax4 = plt.subplots(figsize=(5, 5))
    if idea_f_result and idea_f_result.get("success"):
        stats_labels = ["Unique", "Duplicate"]
        n_unique = idea_f_result.get("n_unique", 0)
        n_total = idea_f_result.get("n_total", n_unique)
        unique_ratio = idea_f_result.get("unique_ratio", n_unique / max(n_total, 1))
        full_at = float(idea_f_result.get("unique_full_at", 0.95))
        stats_vals = [n_unique, max(0, n_total - n_unique)]
        colors_pie = ["#4CAF50", "#FF5722"]
        wedges, texts, autotexts = ax4.pie(
            stats_vals,
            labels=stats_labels,
            colors=colors_pie,
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=1.5),
        )
        for at in autotexts:
            at.set_fontsize(10)
        ax4.set_title(
            f"unique_ratio={unique_ratio * 100:.1f}% ({n_unique}/{n_total})\n"
            f"full score @ ≥{full_at * 100:.0f}% unique",
            fontsize=9,
        )
    else:
        ax4.text(
            0.5,
            0.5,
            "No uniqueness data",
            ha="center",
            va="center",
            transform=ax4.transAxes,
            fontsize=10,
        )
    plt.tight_layout()
    _safe_savefig(fig4, output_dir / "F_fingerprint_uniqueness.png")
    saved["F_fingerprint_uniqueness"] = str(output_dir / "F_fingerprint_uniqueness.png")
    return saved


def _mol_atom_type_counts(mol):
    """Count heavy atoms by element label (excl. H). Used for chemspace series stats."""
    if mol is None:
        return {}
    try:
        zs = [
            int(a.GetAtomicNum()) for a in mol.GetAtoms() if int(a.GetAtomicNum()) > 1
        ]
    except Exception:
        return {}
    z_to_lab = {
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        15: "P",
        16: "S",
        17: "Cl",
        35: "Br",
        53: "I",
    }
    out = {}
    for z in zs:
        lab = z_to_lab.get(z, "Other")
        out[lab] = out.get(lab, 0) + 1
    return out


def visualize_chemspace_umap(
    molecules_with_pos,
    idea_a_result=None,
    output_dir=None,
    title_prefix="",
    native_molecules=None,  # kept for API compat; intentionally unused
):
    """
    Fig.2a contour grammar (J Cheminform 10.1186/s13321-026-01230-5):

    Morgan2048 → UMAP; **one outermost density ring per atom type** (thin line,
    no nested levels). Density is weighted by **atom counts across all molecules**
    (legend n = total atoms of that type, not molecule presence). Sparse types
    drawn as open circles. No Native overlay.
    """
    from diffdynamic.analysis.pocket.plotting_common import _safe_savefig, _viz_title

    if not HAS_MATPLOTLIB:
        return {}
    output_dir = Path(output_dir) if output_dir else None
    if not output_dir:
        return {}
    try:
        from diffdynamic.analysis.pocket_visualization import (
            FONT,
            NEUTRAL,
            embed_2d,
            morgan_fps_from_molecules,
        )
    except ImportError:
        return {}

    from matplotlib.lines import Line2D

    # Element colors: chemistry / PyMOL-CPK common (C green as requested)
    # Ref: CPK + PyMOL defaults (O red, N blue, S yellow, P orange; C→green)
    ATOM_COLORS = {
        "C": "#33AA33",  # green (common chem ball-stick / user request)
        "N": "#0000FF",  # blue (PyMOL / CPK)
        "O": "#FF0000",  # red (PyMOL / CPK)
        "F": "#7CFC00",  # light green (halogen)
        "P": "#FF8C00",  # orange (PyMOL / CPK)
        "S": "#E6C200",  # yellow (PyMOL / CPK)
        "Cl": "#00AA00",  # green (CPK chlorine)
        "Br": "#A52A2A",  # dark red-brown
        "I": "#9400D3",  # purple
        "Other": "#808080",
    }
    ATOM_ORDER = ("C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Other")
    MIN_KDE = 8
    LINE_W = 0.50  # thin single outer ring only

    gen_fps, gen_keep = morgan_fps_from_molecules(
        molecules_with_pos, n_bits=2048, radius=2
    )
    if gen_fps is None or len(gen_keep) < 8:
        return {}

    xy_gen, method = embed_2d(gen_fps)

    # per-molecule atom counts → series weighted by total atoms (not molecule presence)
    # groups[lab] = list of (umap_row_index, atom_count_in_that_mol)
    groups = {lab: [] for lab in ATOM_ORDER}
    for i, idx in enumerate(gen_keep):
        mol = molecules_with_pos[idx][0] if idx < len(molecules_with_pos) else None
        counts = _mol_atom_type_counts(mol)
        for lab, cnt in counts.items():
            if cnt <= 0:
                continue
            if lab in groups:
                groups[lab].append((i, int(cnt)))
            else:
                groups.setdefault("Other", []).append((i, int(cnt)))

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    legend_handles = []

    def _draw_small_rings(xy, color, zorder=4):
        """Even n=1: open circle ('小圈子'), not a filled dot."""
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=55,
            facecolors="none",
            edgecolors=color,
            linewidths=LINE_W,
            zorder=zorder,
        )
        return "ring"

    def _draw_outer_ring(xy, weights, color, zorder=2):
        """Single lowest-density contour; KDE weighted by atom counts."""
        n_atoms = int(np.sum(weights)) if weights is not None else len(xy)
        if n_atoms < MIN_KDE or len(xy) < 1:
            return _draw_small_rings(xy, color, zorder=zorder + 2)
        w = np.asarray(weights, dtype=np.float64)
        # Expand to atom-count points + tiny jitter so mass (not just support) shapes the ring.
        # Shared molecule support (C/O≈all mols) otherwise yields near-identical outer edges.
        reps = np.maximum(1, np.round(w).astype(int))
        if int(reps.sum()) > 8000:
            reps = np.maximum(1, np.round(reps * (8000.0 / reps.sum())).astype(int))
        xy_exp = np.repeat(xy, reps, axis=0)
        rng = np.random.RandomState(42 + int(zorder) + int(n_atoms) % 97)
        span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-6)
        jitter = 0.012 * span
        xy_exp = xy_exp + rng.normal(0.0, jitter, size=xy_exp.shape)
        if HAS_SCIPY and gaussian_kde is not None:
            try:
                # lower bw → sharper, less soft outline
                kde = gaussian_kde(xy_exp.T, bw_method=0.16)
                xmin, xmax = float(xy_exp[:, 0].min()), float(xy_exp[:, 0].max())
                ymin, ymax = float(xy_exp[:, 1].min()), float(xy_exp[:, 1].max())
                pad_x = 0.28 * (xmax - xmin + 1e-6)
                pad_y = 0.28 * (ymax - ymin + 1e-6)
                xx, yy = np.meshgrid(
                    np.linspace(xmin - pad_x, xmax + pad_x, 180),
                    np.linspace(ymin - pad_y, ymax + pad_y, 180),
                )
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                # slightly higher than pure outer fringe → weight differences show more
                level = float(zz.max() * 0.22)
                ax.contour(
                    xx,
                    yy,
                    zz,
                    levels=[level],
                    colors=[color],
                    linewidths=LINE_W,
                    zorder=zorder,
                )
                return "contour"
            except Exception:
                pass
        try:
            import seaborn as sns

            sns.kdeplot(
                x=xy_exp[:, 0],
                y=xy_exp[:, 1],
                ax=ax,
                levels=1,
                thresh=0.22,
                fill=False,
                color=color,
                linewidths=LINE_W,
                bw_adjust=0.35,
                zorder=zorder,
            )
            return "contour"
        except Exception:
            return _draw_small_rings(xy, color, zorder=zorder + 2)

    for zi, lab in enumerate(ATOM_ORDER):
        entries = groups.get(lab) or []
        if not entries:
            continue
        color = ATOM_COLORS.get(lab, "#A0A0A0")
        row_idx = np.asarray([e[0] for e in entries], dtype=int)
        weights = np.asarray([e[1] for e in entries], dtype=np.float64)
        xy = xy_gen[row_idx]
        n_atoms = int(weights.sum())
        style = _draw_outer_ring(xy, weights, color, zorder=2 + zi)
        if style == "contour":
            legend_handles.append(
                Line2D(
                    [0], [0], color=color, lw=LINE_W + 0.3, label=f"{lab} (n={n_atoms})"
                ),
            )
        else:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=color,
                    markerfacecolor="none",
                    markeredgecolor=color,
                    markersize=6,
                    linestyle="None",
                    label=f"{lab} (n={n_atoms})",
                ),
            )

    # Frame + grid like Fig.2a
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#444444")
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(
        axis="both", labelsize=FONT["tick"], colors=NEUTRAL, length=3.5, width=0.7
    )
    ax.grid(True, color="#D0D0D0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    pad_x = 0.18 * (xy_gen[:, 0].max() - xy_gen[:, 0].min() + 1e-6)
    pad_y = 0.18 * (xy_gen[:, 1].max() - xy_gen[:, 1].min() + 1e-6)
    ax.set_xlim(xy_gen[:, 0].min() - pad_x, xy_gen[:, 0].max() + pad_x)
    ax.set_ylim(xy_gen[:, 1].min() - pad_y, xy_gen[:, 1].max() + pad_y)

    ax.set_xlabel(f"{method} 1", fontsize=FONT["label"], color=NEUTRAL)
    ax.set_ylabel(f"{method} 2", fontsize=FONT["label"], color=NEUTRAL)
    ax.set_title(
        _viz_title(title_prefix, "Chemical space"),
        fontsize=FONT["title"],
        color=NEUTRAL,
        pad=10,
    )

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="lower left",
            fontsize=FONT["annot"],
            frameon=True,
            fancybox=False,
            edgecolor="#CCCCCC",
            framealpha=0.95,
        )

    _ = (idea_a_result, native_molecules)

    plt.tight_layout()
    path = output_dir / "chemspace_umap.png"
    _safe_savefig(fig, path, dpi=180, save_notext=True)
    notext = output_dir / "chemspace_umap_notext.png"
    out = {"chemspace_umap": str(path)}
    if notext.exists():
        out["chemspace_umap_notext"] = str(notext)
    return out
