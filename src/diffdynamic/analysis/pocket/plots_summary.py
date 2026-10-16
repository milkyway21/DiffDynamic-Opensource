"""Plots Summary responsibilities extracted from evaluate_pocket_quality.py."""

from diffdynamic.analysis.pocket.state import (
    HAS_MATPLOTLIB,
    Path,
    np,
    plt,
)


def visualize_pocket_chemistry(idea_f_result, output_dir=None, title_prefix=""):
    """
    Idea F - ligand-neighborhood bio chemistry (hydrophobic / polar / enclosure), 10 Å.
    """
    from diffdynamic.analysis.pocket.plotting_common import _safe_savefig, _viz_title

    if not HAS_MATPLOTLIB:
        return {}
    if not idea_f_result or not idea_f_result.get("success"):
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
            apply_nature_axes,
        )
    except ImportError:
        ACCENT, ICEBLUE_FILL, ICEBLUE_EDGE = "#F56E1A", "#C5DDF0", "#5A7FA0"
        FONT = {"title": 11, "label": 9, "tick": 8, "annot": 7}

        def apply_nature_axes(ax, grid_y=True, grid_x=False):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    labels = ["Hydrophobic", "Polar", "Enclosure"]
    # Show raw bio fractions as primary (readable when mapped saturates at 1)
    raws = [
        idea_f_result.get("hydro_fraction"),
        idea_f_result.get("polar_fraction"),
        idea_f_result.get("enclosure_raw"),
    ]
    mapped = [
        idea_f_result.get("druggability_mapped"),
        idea_f_result.get("fpocket_score_mapped"),
        idea_f_result.get("hydrophobicity_mapped"),
    ]
    vals = [float(v) if v is not None and np.isfinite(v) else 0.0 for v in raws]
    mapped_f = [float(v) if v is not None and np.isfinite(v) else 0.0 for v in mapped]
    overall = float(idea_f_result.get("score") or 0.0)
    radius = idea_f_result.get("centroid_radius", 10.0)

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    colors = [ICEBLUE_EDGE, "#7BA3C4", ACCENT]
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, edgecolor="white", width=0.62, zorder=3)
    for bar, raw, mp in zip(bars, vals, mapped_f):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            raw + 0.045,
            f"{raw:.2f}  (→{mp:.2f})",
            ha="center",
            va="bottom",
            fontsize=FONT["annot"],
            color="#4A4A4A",
        )
    apply_nature_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT["label"])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Raw fraction", fontsize=FONT["label"])
    ax.set_title(
        _viz_title(title_prefix, "F · Pocket chemistry"), fontsize=FONT["title"], pad=8
    )
    ax.text(
        0.98,
        0.98,
        f"r={float(radius):g} Å  ·  F={overall:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=FONT["annot"],
        color="#666666",
    )
    plt.tight_layout()
    _safe_savefig(fig, output_dir / "F_pocket_chemistry.png", dpi=180)
    return {"F_pocket_chemistry": str(output_dir / "F_pocket_chemistry.png")}


def visualize_pocket_chemistry_legacy_fpocket_bars(
    idea_f_result, output_dir=None, title_prefix=""
):
    """Deprecated stub — redirects to ligand-neighborhood chemistry plot."""
    return visualize_pocket_chemistry(
        idea_f_result, output_dir=output_dir, title_prefix=title_prefix
    )


def visualize_anchor_richness(idea_g_result, output_dir=None, title_prefix=""):
    """Idea G — P4 interaction anchors (Nature style)."""
    from diffdynamic.analysis.pocket.plotting_common import _safe_savefig, _viz_title

    if not HAS_MATPLOTLIB:
        return {}
    if not idea_g_result or not idea_g_result.get("success"):
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
            apply_nature_axes,
        )
    except ImportError:
        ACCENT, ICEBLUE_FILL, ICEBLUE_EDGE = "#F56E1A", "#C5DDF0", "#5A7FA0"
        FONT = {"title": 11, "label": 9, "tick": 8, "annot": 7}

        def apply_nature_axes(ax, grid_y=True, grid_x=False):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    saved = {}
    counts = idea_g_result.get("anchor_counts") or {}
    cat_order = ["donor", "acceptor", "aromatic", "ionic", "hydrophobic"]
    cat_vals = [int(counts.get(c, 0) or 0) for c in cat_order]
    cat_colors = [ICEBLUE_EDGE, "#7BA3C4", ACCENT, "#4A4A4A", ICEBLUE_FILL]

    fig1, ax1 = plt.subplots(figsize=(5.8, 3.8))
    x = np.arange(len(cat_order))
    bars = ax1.bar(
        x, cat_vals, color=cat_colors, edgecolor="white", width=0.62, zorder=3
    )
    for bar, val in zip(bars, cat_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.4,
            str(val),
            ha="center",
            va="bottom",
            fontsize=FONT["annot"],
            color="#4A4A4A",
        )
    apply_nature_axes(ax1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cat_order, fontsize=FONT["tick"])
    ax1.set_ylabel("Anchor count", fontsize=FONT["label"])
    n_eff = idea_g_result.get("n_eff")
    n_cap = idea_g_result.get("n_cap")
    ttl = _viz_title(title_prefix, "G · Anchors")
    ax1.set_title(ttl, fontsize=FONT["title"], pad=8)
    ax1.text(
        0.98,
        0.98,
        f"Neff={n_eff}/{n_cap:.0f}" if n_cap else f"Neff={n_eff}",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=FONT["annot"],
        color="#666666",
    )
    plt.tight_layout()
    _safe_savefig(fig1, output_dir / "G_anchor_categories.png", dpi=180)
    saved["G_anchor_categories"] = str(output_dir / "G_anchor_categories.png")

    sub_labels = ["Richness", "Coverage", "Balance", "Overall"]
    sub_vals = [
        float(idea_g_result.get("s_richness") or 0.0),
        float(idea_g_result.get("s_coverage") or 0.0),
        float(idea_g_result.get("s_balance") or 0.0),
        float(idea_g_result.get("score") or 0.0),
    ]
    fig2, ax2 = plt.subplots(figsize=(5.4, 3.6))
    y = np.arange(len(sub_labels))
    colors2 = [ICEBLUE_FILL, "#7BA3C4", ICEBLUE_EDGE, ACCENT]
    bars2 = ax2.barh(
        y, sub_vals, color=colors2, edgecolor="white", height=0.55, zorder=3
    )
    for bar, val in zip(bars2, sub_vals):
        ax2.text(
            val + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=FONT["annot"],
            color="#4A4A4A",
        )
    apply_nature_axes(ax2, grid_y=False, grid_x=True)
    ax2.set_yticks(y)
    ax2.set_yticklabels(sub_labels, fontsize=FONT["label"])
    ax2.set_xlim(0, 1.18)
    ax2.set_xlabel("Score", fontsize=FONT["label"])
    ax2.set_title(
        _viz_title(title_prefix, "G · Anchor scores"),
        fontsize=FONT["title"],
        pad=8,
    )
    plt.tight_layout()
    _safe_savefig(fig2, output_dir / "G_anchor_scores.png", dpi=180)
    saved["G_anchor_scores"] = str(output_dir / "G_anchor_scores.png")

    xyz = idea_g_result.get("site_xyz") or []
    cats = idea_g_result.get("site_categories") or []
    if len(xyz) >= 2 and len(cats) == len(xyz):
        arr = np.asarray(xyz, dtype=np.float64)
        fig3, ax3 = plt.subplots(figsize=(5.4, 5.0))
        # High-contrast categorical colors (types must be separable at a glance)
        cat_style = {
            "donor": {"c": "#2E6EB5", "marker": "o", "z": 3},  # blue
            "acceptor": {"c": "#D94A3D", "marker": "s", "z": 3},  # red
            "aromatic": {"c": "#E6A817", "marker": "D", "z": 5},  # gold
            "ionic": {"c": "#7B2D8E", "marker": "^", "z": 4},  # purple
            "hydrophobic": {"c": "#2A9D6E", "marker": "o", "z": 2},  # green
        }
        # draw rarer / more important types last so they sit on top
        draw_order = ["hydrophobic", "donor", "acceptor", "ionic", "aromatic"]
        for cat in draw_order:
            mask = [c == cat for c in cats]
            n_cat = sum(1 for m in mask if m)
            if n_cat < 1:
                continue
            pts = arr[np.asarray(mask)]
            st = cat_style[cat]
            ax3.scatter(
                pts[:, 0],
                pts[:, 1],
                s=36 if cat != "aromatic" else 48,
                alpha=0.88,
                c=st["c"],
                marker=st["marker"],
                label=f"{cat} ({n_cat})",
                edgecolors="white",
                linewidths=0.35,
                zorder=st["z"],
            )
        # Pocket neighborhood ring only (no centroid star)
        cent = idea_g_result.get("ligand_centroid")
        if cent is not None and len(cent) >= 2:
            rad = float(idea_g_result.get("centroid_radius") or 12.0)
            circ = plt.Circle(
                (cent[0], cent[1]),
                rad,
                fill=False,
                ls="--",
                color="#9A9A9A",
                lw=1.1,
                alpha=0.75,
                zorder=1,
            )
            ax3.add_patch(circ)
        apply_nature_axes(ax3, grid_y=False, grid_x=False)
        ax3.set_aspect("equal", adjustable="datalim")
        ax3.set_xlabel("X (Å)", fontsize=FONT["label"])
        ax3.set_ylabel("Y (Å)", fontsize=FONT["label"])
        ax3.set_title(
            _viz_title(title_prefix, "G · Sites XY"),
            fontsize=FONT["title"],
            pad=8,
        )
        ax3.legend(
            fontsize=FONT["annot"],
            loc="best",
            frameon=True,
            fancybox=False,
            edgecolor="#DDDDDD",
            framealpha=0.92,
            markerscale=1.15,
        )
        plt.tight_layout()
        _safe_savefig(fig3, output_dir / "G_anchor_xy.png", dpi=180)
        saved["G_anchor_xy"] = str(output_dir / "G_anchor_xy.png")

    return saved


def visualize_layer_summary(result, output_dir=None, title_prefix=""):
    """Three-layer scores Sp / Sc / Sl + overall (Nature style)."""
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
            apply_nature_axes,
        )
    except ImportError:
        ACCENT, ICEBLUE_FILL, ICEBLUE_EDGE = "#F56E1A", "#C5DDF0", "#5A7FA0"
        FONT = {"title": 11, "label": 9, "tick": 8, "annot": 7}

        def apply_nature_axes(ax, grid_y=True, grid_x=False):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    labels = ["Sp", "Sc", "Sl", "Overall"]
    raw = [
        result.get("s_pocket"),
        result.get("s_compatibility"),
        result.get("s_ligand"),
        result.get("overall_score"),
    ]
    vals_f = [float(v) if v is not None else 0.0 for v in raw]
    present = [v is not None for v in raw]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    colors = [ICEBLUE_EDGE, "#7BA3C4", ICEBLUE_FILL, ACCENT]
    x = np.arange(len(labels))
    bars = ax.bar(x, vals_f, color=colors, edgecolor="white", width=0.62, zorder=3)
    ax.axhline(0.70, color="#6B8F71", ls=":", lw=1.0, alpha=0.8, zorder=2)
    ax.axhline(0.45, color="#C4A35A", ls=":", lw=1.0, alpha=0.8, zorder=2)
    for bar, val, ok in zip(bars, vals_f, present):
        y_txt = min(val + 0.03, 1.08)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_txt,
            f"{val:.3f}" if ok else "n/a",
            ha="center",
            va="bottom",
            fontsize=FONT["annot"],
            color="#4A4A4A",
        )
    apply_nature_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT["label"])
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Score", fontsize=FONT["label"])
    ax.set_title(
        _viz_title(title_prefix, "Layer summary"), fontsize=FONT["title"], pad=8
    )
    ax.text(
        0.02,
        0.98,
        "Sp=pocket  Sc=compat  Sl=ligand",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT["annot"],
        color="#999999",
    )
    plt.tight_layout()
    _safe_savefig(fig, output_dir / "summary_layers.png", dpi=180)
    return {"summary_layers": str(output_dir / "summary_layers.png")}


def visualize_uniqueness_completeness(
    idea_d_result=None,
    idea_e_result=None,
    output_dir=None,
    title_prefix="",
):
    """
    D + E combined — H-style score-band gauge, single iceblue tone.

    Per-row bands match scoring breakpoints:
      D Unique: low <0.40 / mid 0.40–0.70 / high 0.70–0.95 / full ≥0.95
      E Complete: low <0.75 / mid 0.75–0.90 / high 0.90–0.98 / full ≥0.98
    """
    from diffdynamic.analysis.pocket.plotting_common import _safe_savefig, _viz_title

    if not HAS_MATPLOTLIB:
        return {}
    output_dir = Path(output_dir) if output_dir else None
    if not output_dir:
        return {}

    d_ok = bool(idea_d_result and idea_d_result.get("success"))
    e_ok = bool(idea_e_result and idea_e_result.get("success"))
    if not d_ok and not e_ok:
        return {}

    try:
        from diffdynamic.analysis.pocket_visualization import (
            FONT,
            ICEBLUE_EDGE,
            ICEBLUE_FILL,
            NEUTRAL,
            apply_nature_axes,
        )
    except ImportError:
        ICEBLUE_FILL, ICEBLUE_EDGE, NEUTRAL = "#C5DDF0", "#5A7FA0", "#4A4A4A"
        FONT = {"title": 11, "label": 9, "tick": 8, "annot": 7}

        def apply_nature_axes(ax, grid_y=True, grid_x=False):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    BAND_LOW = "#EEF4F8"
    BAND_MID = "#D5E6F2"
    BAND_HIGH = "#B7D4E8"
    BAND_FULL = "#8FB8D4"
    BAR_FACE = ICEBLUE_FILL
    BAR_EDGE = ICEBLUE_EDGE
    MARK = "#3F6F8F"
    x_max = 1.05

    rows = []
    if d_ok:
        n_u = int(idea_d_result.get("n_unique") or 0)
        n_t = int(idea_d_result.get("n_total") or n_u)
        ratio = float(idea_d_result.get("unique_ratio") or (n_u / max(n_t, 1)))
        score = float(idea_d_result.get("score") or 0.0)
        full_at = float(idea_d_result.get("unique_full_at") or 0.95)
        rows.append(
            {
                "label": "D · Unique",
                "ratio": float(np.clip(ratio, 0.0, 1.0)),
                "score": score,
                "annot": f"{n_u}/{n_t}  D={score:.3f}",
                "full_at": full_at,
                "breaks": (0.0, 0.40, 0.70, 0.95, 1.0),
                "band_note": "D full≥0.95",
            }
        )
    if e_ok:
        n_ok = int(idea_e_result.get("n_success") or 0)
        n_tot = int(idea_e_result.get("n_total") or n_ok)
        rate = idea_e_result.get("reconstruct_rate")
        if rate is None:
            rate = n_ok / max(n_tot, 1)
        rate = float(rate)
        score = float(idea_e_result.get("score") or 0.0)
        rows.append(
            {
                "label": "E · Complete",
                "ratio": float(np.clip(rate, 0.0, 1.0)),
                "score": score,
                "annot": f"{n_ok}/{n_tot}  E={score:.3f}",
                "full_at": 0.98,
                "breaks": (0.0, 0.75, 0.90, 0.98, 1.0),
                "band_note": "E full≥0.98",
            }
        )

    n_rows = len(rows)
    fig_h = 2.2 + 1.15 * n_rows
    fig, ax = plt.subplots(figsize=(6.0, fig_h))
    band_colors = (BAND_LOW, BAND_MID, BAND_HIGH, BAND_FULL)

    y_positions = list(range(n_rows - 1, -1, -1))
    for y, row in zip(y_positions, rows):
        b0, b1, b2, b3, b4 = row["breaks"]
        for (xa, xb), col in zip(((b0, b1), (b1, b2), (b2, b3), (b3, b4)), band_colors):
            ax.fill_between(
                [xa, xb],
                y - 0.48,
                y + 0.48,
                color=col,
                alpha=0.95,
                zorder=0,
                linewidth=0,
            )
        ax.plot(
            [row["full_at"], row["full_at"]],
            [y - 0.48, y + 0.48],
            color=BAR_EDGE,
            ls=":",
            lw=1.0,
            alpha=0.75,
            zorder=1,
        )
        r = row["ratio"]
        ax.barh(
            y,
            r,
            left=0,
            height=0.42,
            color=BAR_FACE,
            edgecolor=BAR_EDGE,
            linewidth=1.0,
            zorder=3,
        )
        ax.plot([r, r], [y - 0.28, y + 0.28], color=MARK, lw=1.6, zorder=5)
        ax.plot(
            [r],
            [y],
            marker="o",
            markersize=7,
            color=MARK,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=6,
        )
        ax.text(
            0.01,
            y + 0.30,
            row["annot"],
            ha="left",
            va="bottom",
            fontsize=FONT["annot"],
            color="#666666",
        )
        ax.text(
            min(max(r, 0.08), 0.92),
            y,
            f"{r:.2f}",
            ha="center",
            va="center",
            fontsize=FONT["annot"],
            color=MARK,
            fontweight="bold",
        )

    apply_nature_axes(ax, grid_y=False, grid_x=True)
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(-0.70, n_rows - 0.15)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=FONT["label"])
    ax.set_xlabel("Ratio", fontsize=FONT["label"])
    ax.set_title(
        _viz_title(title_prefix, "D/E · Uniqueness & Completeness"),
        fontsize=FONT["title"],
        pad=8,
    )
    ax.text(
        0.02,
        0.02,
        "bands  D: <0.40 / 0.40–0.70 / 0.70–0.95 / ≥0.95   ·   E: <0.75 / 0.75–0.90 / 0.90–0.98 / ≥0.98",
        transform=ax.transAxes,
        fontsize=FONT["annot"],
        color="#888888",
        va="bottom",
        ha="left",
    )

    plt.tight_layout()
    path = output_dir / "DE_uniqueness_completeness.png"
    _safe_savefig(fig, path, dpi=180)
    return {"DE_uniqueness_completeness": str(path)}


def visualize_uniqueness(
    idea_d_result, output_dir=None, title_prefix="", idea_e_result=None
):
    """Backward-compatible wrapper → combined D/E figure."""
    return visualize_uniqueness_completeness(
        idea_d_result=idea_d_result,
        idea_e_result=idea_e_result,
        output_dir=output_dir,
        title_prefix=title_prefix,
    )


def visualize_completeness(
    idea_e_result, output_dir=None, title_prefix="", idea_d_result=None
):
    """Backward-compatible wrapper → combined D/E figure (no-op if D already drawn)."""
    # Prefer single combined call from generate_all_visualizations.
    if idea_d_result is not None:
        return visualize_uniqueness_completeness(
            idea_d_result=idea_d_result,
            idea_e_result=idea_e_result,
            output_dir=output_dir,
            title_prefix=title_prefix,
        )
    return visualize_uniqueness_completeness(
        idea_d_result=None,
        idea_e_result=idea_e_result,
        output_dir=output_dir,
        title_prefix=title_prefix,
    )


def visualize_radar_summary(result, output_dir=None, title_prefix=""):
    """Summary radar — 8-axis A–H (summary_radar8.png) + horizontal bars."""
    from diffdynamic.analysis.pocket.plotting_common import _safe_savefig, _viz_title

    if not HAS_MATPLOTLIB:
        return {}
    idea_keys = [
        "idea_a",
        "idea_b",
        "idea_c",
        "idea_d",
        "idea_e",
        "idea_f",
        "idea_g",
        "idea_h",
    ]
    idea_labels = [
        "A Aff",
        "B Mode",
        "C Drug",
        "D Uniq",
        "E Comp",
        "F Chem",
        "G Anch",
        "H Vol",
    ]
    scores = []
    for k in idea_keys:
        r = result.get(k, {})
        scores.append(r.get("score", 0.0) if r.get("success") else 0.0)
    overall = float(result.get("overall_score") or 0.0)
    output_dir = Path(output_dir) if output_dir else None
    if not output_dir:
        return {}
    try:
        from diffdynamic.analysis.pocket_visualization import (
            ACCENT,
            FONT,
            ICEBLUE_EDGE,
            ICEBLUE_FILL,
            apply_nature_axes,
        )
    except ImportError:
        ACCENT, ICEBLUE_FILL, ICEBLUE_EDGE = "#F56E1A", "#C5DDF0", "#5A7FA0"
        FONT = {"title": 11, "label": 9, "tick": 8, "annot": 7}

        def apply_nature_axes(ax, grid_y=True, grid_x=False):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    saved = {}

    # ---- Fig 1: Radar ----
    fig1, ax_radar = plt.subplots(figsize=(5.6, 5.2), subplot_kw=dict(polar=True))
    N = len(idea_labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals = scores + scores[:1]
    ax_radar.set_facecolor("#FAFBFC")
    ax_radar.fill(angles, [0.70] * (N + 1), color="#A8C5A0", alpha=0.12)
    ax_radar.fill(angles, [0.45] * (N + 1), color="#F0D9A8", alpha=0.10)
    ax_radar.plot(
        angles,
        vals,
        "o-",
        lw=1.8,
        color=ICEBLUE_EDGE,
        markersize=5,
        markerfacecolor=ICEBLUE_FILL,
        markeredgecolor=ICEBLUE_EDGE,
        zorder=5,
    )
    ax_radar.fill(angles, vals, alpha=0.22, color=ICEBLUE_FILL)
    ax_radar.set_thetagrids(
        np.degrees(angles[:-1]), idea_labels, fontsize=FONT["label"]
    )
    ax_radar.set_ylim(0, 1)
    ax_radar.spines["polar"].set_color("#B0B0B0")
    ax_radar.spines["polar"].set_linewidth(0.8)
    ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_radar.set_yticklabels([])  # avoid collision with value labels
    ax_radar.grid(True, color="#E8E8E8", linewidth=0.7)
    for angle, val in zip(angles[:-1], scores):
        r_txt = val - 0.12 if val >= 0.92 else val + 0.10
        r_txt = float(np.clip(r_txt, 0.08, 0.98))
        ax_radar.annotate(
            f"{val:.2f}",
            xy=(angle, val),
            xytext=(angle, r_txt),
            fontsize=FONT["annot"],
            ha="center",
            va="center",
            color="#4A4A4A",
        )
    ax_radar.set_title(
        _viz_title(title_prefix, "A–H radar"),
        fontsize=FONT["title"],
        pad=14,
        color="#4A4A4A",
    )
    plt.tight_layout()
    _safe_savefig(fig1, output_dir / "summary_radar8.png", dpi=180)
    saved["summary_radar8"] = str(output_dir / "summary_radar8.png")

    # ---- Fig 2: Horizontal A–H score bars (beautified) ----
    fig2, ax_bar = plt.subplots(figsize=(6.0, 4.6))
    # Per-dimension palette (ligand / compatibility / pocket layers)
    DIM_COLORS = {
        "A Aff": "#3D7A9C",  # ligand — deep iceblue
        "B Mode": "#5A9E8F",  # teal
        "C Drug": "#7EB3C9",  # light iceblue
        "D Uniq": "#6B9E78",  # compatibility — soft green
        "E Comp": "#8FB89A",
        "F Chem": "#D4894A",  # pocket — warm amber (not purple)
        "G Anch": "#E0A060",
        "H Vol": "#C4A06A",
    }
    # Score-aware lighten: low scores fade toward neutral
    colors_bar = []
    for lab, s in zip(idea_labels, scores):
        base = DIM_COLORS.get(lab, ICEBLUE_EDGE)
        # blend with light gray when score is low
        fade = float(np.clip(s, 0.0, 1.0))
        r = int(base[1:3], 16)
        g = int(base[3:5], 16)
        b = int(base[5:7], 16)
        nr = int(r * fade + 200 * (1 - fade))
        ng = int(g * fade + 200 * (1 - fade))
        nb = int(b * fade + 200 * (1 - fade))
        colors_bar.append(f"#{nr:02X}{ng:02X}{nb:02X}")

    y = np.arange(len(idea_labels))
    # soft track + score bars (A at top via invert_yaxis)
    ax_bar.barh(
        y,
        [1.0] * len(scores),
        color="#F0F2F4",
        edgecolor="none",
        height=0.68,
        zorder=1,
    )
    bars = ax_bar.barh(
        y,
        scores,
        color=colors_bar,
        edgecolor="white",
        linewidth=0.8,
        height=0.68,
        zorder=3,
    )
    _ = bars

    ax_bar.axvline(0.70, color="#6B8F71", ls=":", lw=1.1, alpha=0.80, zorder=2)
    ax_bar.axvline(0.45, color="#C4A35A", ls=":", lw=1.1, alpha=0.80, zorder=2)
    ax_bar.axvline(overall, color=ACCENT, ls="-", lw=1.6, alpha=0.95, zorder=4)
    ax_bar.plot(
        [overall],
        [len(idea_labels) - 0.45],
        marker="D",
        markersize=6,
        color=ACCENT,
        markeredgecolor="white",
        markeredgewidth=0.6,
        zorder=5,
        clip_on=False,
    )

    for yi, val in zip(y, scores):
        txt_x = val + 0.02 if val < 0.88 else val - 0.02
        ha = "left" if val < 0.88 else "right"
        tc = "#4A4A4A" if val < 0.88 else "white"
        ax_bar.text(
            txt_x,
            yi,
            f"{val:.2f}",
            va="center",
            ha=ha,
            fontsize=FONT["annot"],
            color=tc,
            fontweight="medium",
            zorder=6,
        )

    apply_nature_axes(ax_bar, grid_y=False, grid_x=True)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(idea_labels, fontsize=FONT["label"])
    ax_bar.invert_yaxis()  # A Aff on top
    ax_bar.set_xlim(0, 1.12)
    ax_bar.set_ylim(len(idea_labels) - 0.15, -0.85)
    ax_bar.set_xlabel("Score", fontsize=FONT["label"])
    ax_bar.set_title(
        _viz_title(title_prefix, "A–H scores"),
        fontsize=FONT["title"],
        pad=8,
    )
    ax_bar.text(
        0.98,
        0.02,
        f"overall = {overall:.3f}",
        transform=ax_bar.transAxes,
        ha="right",
        va="bottom",
        fontsize=FONT["annot"],
        color=ACCENT,
        fontweight="bold",
    )
    # compact layer legend
    ax_bar.text(
        0.02,
        0.02,
        "ligand A–C  ·  compatibility D–E  ·  pocket F–H",
        transform=ax_bar.transAxes,
        ha="left",
        va="bottom",
        fontsize=FONT["annot"],
        color="#888888",
    )
    plt.tight_layout()
    _safe_savefig(fig2, output_dir / "summary_bar.png", dpi=180)
    saved["summary_bar"] = str(output_dir / "summary_bar.png")
    return saved


def generate_all_visualizations(
    result, molecules_with_pos, output_dir, title_prefix="", use_tsne=False
):
    """
    一键生成所有可视化图像

    Args:
        result            : evaluate_pocket_quality() 的返回值
        molecules_with_pos: [(mol, pos_array), ...]
        output_dir        : 输出目录
        title_prefix      : 图标题前缀（通常为口袋 ID）
        use_tsne          : 是否在聚类图中使用 t-SNE

    Returns:
        dict: {图名: 文件路径}
    """
    from diffdynamic.analysis.pocket.chemistry import evaluate_idea_g_size_consistency
    from diffdynamic.analysis.pocket.plots_chemistry import (
        visualize_chemspace_umap,
        visualize_druglikeness,
        visualize_vina_le_combined,
    )
    from diffdynamic.analysis.pocket.plots_geometry import (
        visualize_clustering_2d,
        visualize_pocket_volume,
        visualize_size_distribution,
    )

    if not HAS_MATPLOTLIB:
        print("⚠️ matplotlib 未安装，跳过可视化")
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    print(f"\n{'=' * 60}")
    print(f"Generating visualizations -> {output_dir}")
    print("=" * 60)

    # 0. Chemical space UMAP (Fig.2a contours, colored by primary atom type; no Native)
    r = visualize_chemspace_umap(
        molecules_with_pos,
        idea_a_result=result.get("idea_a", {}),
        output_dir=output_dir,
        title_prefix=title_prefix,
    )
    saved.update(r)

    # 1. Atom clustering (Idea B)
    idea_b = result.get("idea_b", {})
    r = visualize_clustering_2d(
        molecules_with_pos,
        idea_b_result=idea_b if idea_b.get("success") else None,
        output_dir=output_dir,
        title_prefix=title_prefix,
        n_kmeans=5,
        use_tsne=use_tsne,
    )
    saved.update(r)

    # 2. Vina + LE on one figure (Idea A components)
    r = visualize_vina_le_combined(
        idea_a_result=result.get("idea_a", {}),
        idea_le_result=result.get("idea_le")
        or (result.get("idea_a") or {}).get("le_component")
        or {},
        output_dir=output_dir,
        title_prefix=title_prefix,
    )
    saved.update(r)

    # 3. Druglikeness (Idea C slot; keep D_* filenames for continuity)
    idea_a = result.get("idea_a", {})
    vina_scores = idea_a.get("vina_scores") if idea_a.get("success") else None
    r = visualize_druglikeness(
        result.get("idea_c", {}),
        output_dir=output_dir,
        title_prefix=title_prefix,
        vina_scores=vina_scores,
    )
    saved.update(r)

    # 4. Uniqueness + Completeness (D/E combined, H-style bands, one tone)
    r = visualize_uniqueness_completeness(
        idea_d_result=result.get("idea_d", {}),
        idea_e_result=result.get("idea_e", {}),
        output_dir=output_dir,
        title_prefix=title_prefix,
    )
    saved.update(r)

    # 4b. Pocket chemistry F
    r = visualize_pocket_chemistry(
        result.get("idea_f", {}),
        output_dir=output_dir,
        title_prefix=title_prefix,
    )
    saved.update(r)

    # 5. Size distribution — auxiliary only (not scored; G = anchors)
    try:
        size_aux = evaluate_idea_g_size_consistency(molecules_with_pos)
        r = visualize_size_distribution(
            size_aux,
            output_dir=output_dir,
            title_prefix=title_prefix,
        )
        saved.update(r)
    except Exception:
        pass

    # 5b. Anchor richness G
    r = visualize_anchor_richness(
        result.get("idea_g", {}),
        output_dir=output_dir,
        title_prefix=title_prefix,
    )
    saved.update(r)

    # 6. Pocket volume (Idea H)
    r = visualize_pocket_volume(
        result.get("idea_h", {}),
        output_dir=output_dir,
        title_prefix=title_prefix,
    )
    saved.update(r)

    # 7. Summary radar / bar
    r = visualize_radar_summary(
        result,
        output_dir=output_dir,
        title_prefix=title_prefix,
    )
    saved.update(r)

    # 7b. three-layer summary
    r = visualize_layer_summary(
        result,
        output_dir=output_dir,
        title_prefix=title_prefix,
    )
    saved.update(r)

    # Drop obsolete filenames from earlier naming schemes
    for obsolete in (
        "C_le_hist.png",
        "C_le_hist_notext.png",
        "C_le_box.png",
        "C_le_box_notext.png",
        "C_le_cdf.png",
        "C_le_cdf_notext.png",
        "G_size_mw.png",
        "G_size_mw_notext.png",
        "G_size_atoms.png",
        "G_size_atoms_notext.png",
        "G_size_mw_vs_atoms.png",
        "G_size_mw_vs_atoms_notext.png",
        "D_uniqueness.png",
        "D_uniqueness_notext.png",
        "E_completeness.png",
        "E_completeness_notext.png",
    ):
        op = output_dir / obsolete
        if op.exists():
            try:
                op.unlink()
            except OSError:
                pass

    print(f"\nGenerated {len(saved)} images:")
    for name, path in saved.items():
        print(f"  [{name}] {path}")
    return saved
