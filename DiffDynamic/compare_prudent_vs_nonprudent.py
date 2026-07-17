#!/usr/bin/env python3
"""Compare non-prudent vs Murcko-prudent generation results for 6W63 and 8RI2.

Data sources:
- Non-prudent: /data/ye/protein-ligand/{6W63,8RI2}/generated_dock_qed_sa.csv
- Murcko-prudent: /data/ye/DiffDynamic/{6w63prudent_murcko,8ri2prudent_murcko}/generated_dock_qed_sa.csv
- Active ligands: /data/ye/protein-ligand/{6W63,8RI2}/active_dock_qed_sa.csv
"""

import sys, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, mannwhitneyu

warnings.filterwarnings("ignore")

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED, Lipinski

# Style constants (same as protein-ligand violin plots)
COLOR_NONPRUDENT = "#a7d8e9"       # blue-ish (like known actives)
COLOR_MURCKO = "#fff0f0"            # pink-ish (like generated)
COLOR_NONPRUDENT_MEDIAN = "#5a9ab5"
COLOR_MURCKO_MEDIAN = "#c97a7a"
COLOR_VIOLIN_EDGE = "#444444"
COLOR_VIOLIN_SPINE = "#DDDDDD"
MEDIAN_LINEWIDTH = 1.8


def compute_properties(smiles_list):
    """Compute LogP and MW from SMILES using RDKit."""
    logps, mws = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            logps.append(Crippen.MolLogP(mol))
            mws.append(Descriptors.MolWt(mol))
        else:
            logps.append(np.nan)
            mws.append(np.nan)
    return np.array(logps), np.array(mws)


def load_data(csv_path):
    """Load CSV and compute additional properties."""
    df = pd.read_csv(csv_path)
    logps, mws = compute_properties(df['smiles'].values)
    df['logP'] = logps
    df['MW'] = mws

    # Rename columns for consistency
    if 'vina_dock' in df.columns:
        df = df.rename(columns={'vina_dock': 'vina', 'qed': 'QED', 'sa': 'SA'})

    return df


def load_active_data(csv_path):
    """Load active ligand CSV."""
    df = pd.read_csv(csv_path)
    logps, mws = compute_properties(df['smiles'].values)
    df['logP'] = logps
    df['MW'] = mws
    if 'vina_dock' in df.columns:
        df = df.rename(columns={'vina_dock': 'vina', 'qed': 'QED', 'sa': 'SA'})
    return df


def compute_stats(df, label):
    """Compute statistics for a dataset."""
    stats = {}
    for col, clean_name in [('vina', 'Vina Dock'), ('QED', 'QED'), ('SA', 'SA'),
                              ('logP', 'LogP'), ('MW', 'MW')]:
        vals = pd.to_numeric(df[col], errors='coerce').dropna()
        # Filter vina to -15~0
        if col == 'vina':
            vals = vals[(vals > -15) & (vals < 0)]

        stats[clean_name] = {
            'count': len(vals),
            'mean': vals.mean(),
            'std': vals.std(),
            'min': vals.min(),
            'max': vals.max(),
            'median': vals.median(),
        }

    # Top-5 and Top-10 mean Vina
    vina_vals = pd.to_numeric(df['vina'], errors='coerce').dropna()
    vina_vals = vina_vals[(vina_vals > -15) & (vina_vals < 0)]
    vina_sorted = vina_vals.sort_values()
    stats['Vina Top-5 mean'] = vina_sorted.head(5).mean() if len(vina_sorted) >= 5 else np.nan
    stats['Vina Top-10 mean'] = vina_sorted.head(10).mean() if len(vina_sorted) >= 10 else np.nan

    # LogP <= 5 filter
    logp_vals = pd.to_numeric(df['logP'], errors='coerce').dropna()
    stats['LogP ≤ 5 count'] = (logp_vals <= 5).sum()
    stats['LogP ≤ 5 pct'] = (logp_vals <= 5).mean() * 100

    # Lipinski: HBD≤5, HBA≤10, MW≤500, LogP≤5
    lipinski_pass = 0
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row['smiles']))
        if mol is not None:
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            if hbd <= 5 and hba <= 10 and mw <= 500 and logp <= 5:
                lipinski_pass += 1

    stats['Lipinski pass'] = lipinski_pass
    stats['Lipinski pass pct'] = lipinski_pass / max(len(df), 1) * 100
    stats['Total molecules'] = len(df)

    return stats


def _violin_with_style(ax, data, pos, color, y_min, y_max, median_color=None):
    """Draw a styled violin at the given position (from plot_property_violins.py)."""
    if len(data) < 2:
        return

    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    # Clip extremes
    q1, q3 = np.percentile(data, [1, 99])
    data = data[(data >= q1) & (data <= q3)]

    if len(data) < 2:
        return

    # KDE
    try:
        kde = gaussian_kde(data, bw_method='scott')
    except Exception:
        return

    bw = kde.factor * np.std(data)
    y_grid = np.linspace(y_min, y_max, 500)
    density = kde.evaluate(y_grid)
    density = density / density.max() if density.max() > 0 else density

    width = 0.32
    x = pos + density * width
    x_neg = pos - density * width

    ax.fill_betweenx(y_grid, x_neg, x, alpha=0.85, facecolor=color, edgecolor=COLOR_VIOLIN_EDGE, linewidth=0.7, zorder=3)

    # Median line
    med = np.median(data)
    ax.plot([pos - width, pos + width], [med, med], color=median_color or color,
            linewidth=MEDIAN_LINEWIDTH, zorder=6, solid_capstyle='butt')


def make_violin_comparison(nonprudent_df, murcko_df, protein_name, out_dir, active_df=None):
    """Create comparison violin plots: non-prudent vs Murcko-prudent."""
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 14, "axes.labelsize": 15, "axes.titlesize": 16,
        "figure.dpi": 300, "savefig.dpi": 300, "axes.linewidth": 1.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 13, "ytick.labelsize": 12,
    })

    # Filter vina -15~0
    np_filt = nonprudent_df.copy()
    mk_filt = murcko_df.copy()

    metrics = [
        ("vina", "Vina Dock", "Vina Dock (kcal/mol)", (-15, 0), True),
        ("QED", "QED", "QED", (0, 1), False),
        ("sa" if "sa" in nonprudent_df.columns else "SA", "SA Score", "SA Score", (0, 1), False),
        ("logP", "LogP", "LogP", (-5, 15), False),
        ("MW", "MW", "Molecular Weight", (0, 800), False),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5), gridspec_kw={"wspace": 0.35})

    for ax, (col, display, y_label, (y_lo, y_hi), invert) in zip(axes, metrics):
        # Get non-prudent data
        if col in np_filt.columns:
            np_data = pd.to_numeric(np_filt[col], errors='coerce').dropna()
        else:
            np_data = pd.Series(dtype=float)

        # Vina filter
        if col == 'vina':
            np_data = np_data[(np_data > -15) & (np_data < 0)]

        # Get Murcko data
        if col in mk_filt.columns:
            mk_data = pd.to_numeric(mk_filt[col], errors='coerce').dropna()
        else:
            mk_data = pd.Series(dtype=float)

        if col == 'vina':
            mk_data = mk_data[(mk_data > -15) & (mk_data < 0)]

        # Draw violins
        for arr, pos, color, med_col, lbl in [
            (np_data.values, 0.85, COLOR_NONPRUDENT, COLOR_NONPRUDENT_MEDIAN, "Non-Prudent"),
            (mk_data.values, 1.65, COLOR_MURCKO, COLOR_MURCKO_MEDIAN, "Murcko-Prudent"),
        ]:
            if len(arr) >= 2:
                _violin_with_style(ax, arr, pos, color, y_lo, y_hi, median_color=med_col)
            elif len(arr) == 1:
                ax.scatter([pos], [arr[0]], color=color, s=40, zorder=5,
                          edgecolors=COLOR_VIOLIN_EDGE, linewidth=0.8)

        # Add mean markers
        if len(np_data) > 0:
            ax.plot(0.85, np_data.mean(), 'D', color=COLOR_NONPRUDENT_MEDIAN, markersize=5, zorder=7)
        if len(mk_data) > 0:
            ax.plot(1.65, mk_data.mean(), 'D', color=COLOR_MURCKO_MEDIAN, markersize=5, zorder=7)

        ax.set_xlim(0.35, 2.15)
        ax.set_xticks([0.85, 1.65])
        ax.set_xticklabels(["Non-\nPrudent", "Murcko-\nPrudent"], fontsize=10)
        ax.set_title(display, fontsize=13, fontweight="medium", pad=8)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_ylim(y_lo, y_hi)

        if invert:
            ax.invert_yaxis()

        ax.yaxis.grid(True, linestyle="--", alpha=0.25, color="#AAAAAA")
        ax.set_axisbelow(True)

    # Legend
    from matplotlib.patches import Patch
    n_count = len(np_filt)
    m_count = len(mk_filt)
    fig.legend(handles=[
        Patch(facecolor=COLOR_NONPRUDENT, edgecolor=COLOR_VIOLIN_EDGE, alpha=1.0,
              label=f"Non-Prudent (n={n_count})"),
        Patch(facecolor=COLOR_MURCKO, edgecolor=COLOR_VIOLIN_EDGE, alpha=1.0,
              label=f"Murcko-Prudent (n={m_count})"),
    ], loc="upper center", ncol=2, frameon=True, fancybox=False, edgecolor="#CCCCCC",
       bbox_to_anchor=(0.5, -0.02), framealpha=0.95, fontsize=9)

    plt.suptitle(f"{protein_name}: Non-Prudent vs Murcko-Prudent", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout(rect=(0, 0.06, 1, 0.96))

    out_png = out_dir / "comparison_violin.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved violin: {out_png}")


def make_logp_mw_scatter(nonprudent_df, murcko_df, protein_name, out_dir, active_df=None):
    """Create side-by-side LogP vs MW scatter plots."""
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
        "figure.dpi": 300, "savefig.dpi": 300,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

    for ax, df, title, color in [
        (ax1, nonprudent_df, "Non-Prudent", COLOR_NONPRUDENT_MEDIAN),
        (ax2, murcko_df, "Murcko-Prudent", COLOR_MURCKO_MEDIAN),
    ]:
        mw = pd.to_numeric(df['MW'], errors='coerce')
        logp = pd.to_numeric(df['logP'], errors='coerce')
        valid = mw.notna() & logp.notna()

        ax.scatter(mw[valid], logp[valid], c=color, alpha=0.5, s=15, edgecolors='none')
        ax.axhline(y=5, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
        ax.text(790, 5.3, "LogP=5", color='red', fontsize=9, ha='right', va='bottom')

        ax.set_xlim(0, 800)
        ax.set_ylim(-5, 15)
        ax.set_xlabel("Molecular Weight", fontsize=12)
        if ax == ax1:
            ax.set_ylabel("LogP", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="medium")
        ax.grid(True, linestyle="--", alpha=0.25, color="#AAAAAA")

        # Stats annotation
        pass_logp5 = (logp <= 5).sum()
        total = valid.sum()
        ax.text(0.02, 0.97, f"LogP≤5: {pass_logp5}/{total} ({pass_logp5/total*100:.1f}%)",
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle(f"{protein_name}: LogP vs MW Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.94))

    out_png = out_dir / "comparison_logp_mw.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved scatter: {out_png}")


def print_comparison_table(nonprudent_stats, murcko_stats, protein_name):
    """Print Markdown comparison table."""
    print(f"\n{'='*80}")
    print(f"  COMPARISON: {protein_name} — Non-Prudent vs Murcko-Prudent")
    print(f"{'='*80}")

    properties = ['Vina Dock', 'QED', 'SA', 'LogP', 'MW']

    print(f"\n| Property | Non-Prudent (mean±std) | Murcko-Prudent (mean±std) |")
    print(f"|----------|------------------------|---------------------------|")

    for prop in properties:
        ns = nonprudent_stats[prop]
        ms = murcko_stats[prop]
        if ms['count'] > 0 and ns['count'] > 0:
            print(f"| {prop} | {ns['mean']:.3f}±{ns['std']:.3f} (n={ns['count']}) | {ms['mean']:.3f}±{ms['std']:.3f} (n={ms['count']}) |")

    # Extended stats
    print(f"\n| Metric | Non-Prudent | Murcko-Prudent |")
    print(f"|--------|-------------|----------------|")

    for key in ['Total molecules', 'Vina Top-5 mean', 'Vina Top-10 mean',
                'LogP ≤ 5 count', 'LogP ≤ 5 pct', 'Lipinski pass', 'Lipinski pass pct']:
        nv = nonprudent_stats[key]
        mv = murcko_stats[key]
        if isinstance(nv, float):
            print(f"| {key} | {nv:.3f} | {mv:.3f} |")
        else:
            print(f"| {key} | {nv} | {mv} |")

    # Vina range details
    for prop in properties:
        ns = nonprudent_stats[prop]
        ms = murcko_stats[prop]
        if prop == 'Vina Dock':
            print(f"\n| {prop} Stats | Non-Prudent | Murcko-Prudent |")
            print(f"|{'─'*15}|{'─'*13}|{'─'*16}|")
            for stat in ['min', 'max', 'median']:
                print(f"| {stat} | {ns[stat]:.2f} | {ms[stat]:.2f} |")


def main():
    # Create output plot directories
    out_dirs = {
        '6W63': Path('/data/ye/DiffDynamic/6w63prudent_murcko/plots'),
        '8RI2': Path('/data/ye/DiffDynamic/8ri2prudent_murcko/plots'),
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    proteins = [
        {
            'name': '6W63',
            'nonprudent_csv': '/data/ye/protein-ligand/6W63/generated_dock_qed_sa.csv',
            'murcko_csv': '/data/ye/DiffDynamic/6w63prudent_murcko/generated_dock_qed_sa.csv',
            'active_csv': '/data/ye/protein-ligand/6W63/active_dock_qed_sa.csv',
        },
        {
            'name': '8RI2',
            'nonprudent_csv': '/data/ye/protein-ligand/8RI2/generated_dock_qed_sa.csv',
            'murcko_csv': '/data/ye/DiffDynamic/8ri2prudent_murcko/generated_dock_qed_sa.csv',
            'active_csv': '/data/ye/protein-ligand/8RI2/active_dock_qed_sa.csv',
        },
    ]

    for p in proteins:
        print(f"\n{'#'*80}")
        print(f"# Processing {p['name']}")
        print(f"{'#'*80}")

        # Load data
        nonprudent_df = load_data(p['nonprudent_csv'])
        murcko_df = load_data(p['murcko_csv'])
        active_df = load_active_data(p['active_csv'])

        print(f"Non-Prudent: {len(nonprudent_df)} molecules")
        print(f"Murcko-Prudent: {len(murcko_df)} molecules")
        print(f"Active ligands: {len(active_df)} molecules")

        # Compute stats
        np_stats = compute_stats(nonprudent_df, "Non-Prudent")
        mk_stats = compute_stats(murcko_df, "Murcko-Prudent")

        # Print table
        print_comparison_table(np_stats, mk_stats, p['name'])

        # Generate plots
        make_violin_comparison(nonprudent_df, murcko_df, p['name'], out_dirs[p['name']], active_df)
        make_logp_mw_scatter(nonprudent_df, murcko_df, p['name'], out_dirs[p['name']], active_df)

    print(f"\n{'='*80}")
    print("All plots saved successfully!")


if __name__ == "__main__":
    main()
