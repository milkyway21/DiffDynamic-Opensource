"""Plotting Common responsibilities extracted from evaluate_pocket_quality.py."""

from diffdynamic.analysis.pocket.state import (
    HAS_MATPLOTLIB,
    Path,
    _NOTEXT_SUFFIX,
    plt,
)


def _strip_figure_text(fig):
    """
    去掉图中全部可读文字（轴标题、刻度数字、图例、colorbar 标签、
    散点/柱状图上的标注与饼图百分比等），仅保留几何与颜色，便于在排版软件中自行配字。
    """
    if not HAS_MATPLOTLIB:
        return
    from matplotlib.projections.polar import PolarAxes

    supt = getattr(fig, "_suptitle", None)
    if supt is not None:
        try:
            supt.set_visible(False)
        except Exception:
            try:
                supt.set_text("")
            except Exception:
                pass

    for lg in list(getattr(fig, "legends", []) or []):
        try:
            lg.remove()
        except Exception:
            pass

    for ax in fig.axes:
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        leg = ax.get_legend()
        if leg is not None:
            try:
                leg.remove()
            except Exception:
                pass
        ax.tick_params(
            axis="both",
            which="both",
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )
        if isinstance(ax, PolarAxes):
            try:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            except Exception:
                pass
        for txt in list(ax.texts):
            try:
                txt.remove()
            except Exception:
                try:
                    txt.set_visible(False)
                except Exception:
                    pass


def _safe_savefig(fig, path, dpi=180, save_notext=True):
    """
    安全保存图像，创建父目录。

    在保存带完整标注的原图后，默认再保存一份仅保留图形元素、无文字的同分辨率图：
    文件名在原 stem 后追加 ``_notext``（如 ``A_vina_hist_notext.png``），便于科研排版自行控制字体。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path}")

    ext = path.suffix.lower()
    if save_notext and ext in (
        ".png",
        ".pdf",
        ".svg",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".webp",
    ):
        notext_path = path.parent / f"{path.stem}{_NOTEXT_SUFFIX}{path.suffix}"
        # 去字后再 tight_layout 会破坏「主图 + 窄 colorbar」等布局；先冻结各轴位置再恢复
        pos_bounds = [tuple(ax.get_position().bounds) for ax in fig.axes]
        _strip_figure_text(fig)
        if len(fig.axes) == len(pos_bounds):
            for ax, b in zip(fig.axes, pos_bounds):
                ax.set_position(b)
        try:
            fig.tight_layout()
        except Exception:
            pass
        if len(fig.axes) == len(pos_bounds):
            for ax, b in zip(fig.axes, pos_bounds):
                ax.set_position(b)
        fig.savefig(notext_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {notext_path}")

    plt.close(fig)
    return str(path)


def _viz_title(title_prefix, core):
    """Journal-style title: 'Pocket 0 · Vina' or just 'Vina'."""
    p = str(title_prefix).strip() if title_prefix is not None else ""
    if not p:
        return core
    if p.isdigit() or (
        p.startswith("pocket") is False and p.replace("_", "").isalnum() and len(p) <= 8
    ):
        # numeric / short id → Pocket N
        if p.isdigit():
            return f"Pocket {p} · {core}"
        return f"{p} · {core}"
    return f"{p} · {core}"
