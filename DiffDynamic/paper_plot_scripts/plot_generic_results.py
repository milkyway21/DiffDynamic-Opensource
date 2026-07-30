#!/usr/bin/env python3
"""Generic t-SNE + LogP-MW plots for pocket comparison.
Filters generated molecules to Vina in (-15, 0).

Usage:
  python3 plot_generic_results.py --gen_sdf GEN_SDF_DIR --act_sdf ACT_SDF_DIR \
      --out_dir OUT_DIR --tag TAG [--gen_csv GEN_CSV]
"""
import sys, warnings, random, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, AllChem
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

COLOR_BG = "#9a9a9a"
COLOR_ACTUAL = "#C51610"
COLOR_GENERATED = "#2563eb"


def load_mols_from_sdf_dir(sdf_dir):
    mols = []
    sdf_dir = Path(sdf_dir)
    if not sdf_dir.exists():
        return mols
    for sf in sorted(sdf_dir.glob("*.sdf")):
        try:
            m = Chem.SDMolSupplier(str(sf))[0]
            if m: mols.append(m)
        except Exception: pass
    return mols


def load_bg_mols(n=5000, seed=42):
    mols = []
    bg_path = Path("/home/user/Desktop/Ye/ccp_roco4_pipeline/.venv/lib/python3.12/site-packages/rdkit/Data/NCI/first_5K.smi")
    if not bg_path.exists():
        # Try conda env path
        import glob
        candidates = glob.glob("/home/user/anaconda3/envs/diffdynamic/lib/python*/site-packages/rdkit/Data/NCI/first_5K.smi")
        if candidates: bg_path = Path(candidates[0])
        else: return mols
    with open(bg_path, encoding="utf-8", errors="replace") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    random.seed(seed)
    random.shuffle(lines)
    for line in lines[:n * 3]:
        smi = line.split()[0]
        try:
            m = Chem.MolFromSmiles(smi)
            if m:
                mols.append(m)
                if len(mols) >= n: break
        except Exception: pass
    return mols


def morgan_fp(mol, radius=2, nbits=1024):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.int8)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr


def plot_tsne(gen_mols, ref_mols, out_png, bg_mols=None, show_text=True, tag=""):
    print(f"  t-SNE: gen={len(gen_mols)}, ref={len(ref_mols)}, bg={len(bg_mols) if bg_mols else 0}")
    blocks, labels = [], []
    if bg_mols:
        blocks.append(np.vstack([morgan_fp(m) for m in bg_mols]))
        labels.append("bg")
    if ref_mols:
        blocks.append(np.vstack([morgan_fp(m) for m in ref_mols]))
        labels.append("ref")
    if gen_mols:
        blocks.append(np.vstack([morgan_fp(m) for m in gen_mols]))
        labels.append("gen")
    if not blocks:
        print("  No molecules to plot for t-SNE")
        return
    X = np.vstack(blocks)
    n = X.shape[0]
    perp = min(30.0, max(5.0, float(min(n - 1, 500))))
    print(f"  Total mols: {n}, perplexity={perp:.0f}")
    tsne = TSNE(n_components=2, perplexity=perp, metric="cosine",
                random_state=42, init="pca", learning_rate="auto",
                early_exaggeration=24.0 if n >= 80 else 12.0, n_iter=1000)
    Y = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    idx = 0
    if bg_mols:
        ax.scatter(Y[:len(bg_mols), 0], Y[:len(bg_mols), 1],
                   s=10, marker="o", c=COLOR_BG, edgecolors="none",
                   alpha=0.3, label=f"Background ({len(bg_mols)})", zorder=1)
        idx += len(bg_mols)
    if gen_mols:
        ax.scatter(Y[idx:idx+len(gen_mols), 0], Y[idx:idx+len(gen_mols), 1],
                   s=18, marker="o", c=COLOR_GENERATED, edgecolors="none",
                   alpha=0.6, label=f"Generated ({len(gen_mols)})", zorder=3)
        idx += len(gen_mols)
    if ref_mols:
        ax.scatter(Y[idx:idx+len(ref_mols), 0], Y[idx:idx+len(ref_mols), 1],
                   s=18, marker="o", c=COLOR_ACTUAL, edgecolors="none",
                   alpha=1.0, label=f"Known Actives ({len(ref_mols)})", zorder=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    if show_text:
        ax.set_xlabel("t-SNE 1 (cosine, Morgan FP)", fontsize=12)
        ax.set_ylabel("t-SNE 2 (cosine, Morgan FP)", fontsize=12)
        ax.set_title(f"t-SNE — {tag} Generated vs Actives", fontsize=12)
        ax.legend(loc="best", framealpha=0.92, prop={"size": 9})
    else:
        ax.tick_params(labelbottom=False, labelleft=False)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_png}")


def mols_to_df(mols):
    rows = []
    for m in mols:
        try:
            rows.append({"smiles": Chem.MolToSmiles(m),
                         "molwt": float(Descriptors.MolWt(m)),
                         "logp": float(Crippen.MolLogP(m))})
        except Exception: pass
    return pd.DataFrame(rows)


def plot_logp_mw(gen_df, act_df, out_png, bg_df=None, show_text=True, tag=""):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    if bg_df is not None and len(bg_df) > 0:
        ax.scatter(bg_df["molwt"], bg_df["logp"], s=10, marker="o", c=COLOR_BG,
                   edgecolors="none", alpha=0.3, label=f"Background ({len(bg_df)})", zorder=1)
    if len(gen_df) > 0:
        ax.scatter(gen_df["molwt"], gen_df["logp"], s=18, marker="o", c=COLOR_GENERATED,
                   edgecolors="none", alpha=0.6, label=f"Generated ({len(gen_df)})", zorder=3)
    if len(act_df) > 0:
        ax.scatter(act_df["molwt"], act_df["logp"], s=18, marker="o", c=COLOR_ACTUAL,
                   edgecolors="none", alpha=1.0, label=f"Known Actives ({len(act_df)})", zorder=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    if show_text:
        ax.set_xlabel("Molecular Weight", fontsize=12)
        ax.set_ylabel("LogP (Crippen)", fontsize=12)
        ax.set_title(f"LogP vs Molecular Weight — {tag}", fontsize=12)
        ax.legend(loc="best", framealpha=0.92, prop={"size": 9})
    else:
        ax.tick_params(labelbottom=False, labelleft=False)
    ax.grid(True, linestyle="--", alpha=0.25, color="#AAAAAA")
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Generic t-SNE + LogP-MW plots")
    parser.add_argument("--gen_sdf", required=True, help="Generated molecules SDF directory")
    parser.add_argument("--act_sdf", required=True, help="Active molecules SDF directory")
    parser.add_argument("--gen_csv", default=None, help="Optional: generated CSV (filters Vina -15~0 by SMILES)")
    parser.add_argument("--out_dir", required=True, help="Output directory for plots")
    parser.add_argument("--tag", required=True, help="Plot label prefix (e.g. 8RI2, 6W63)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load generated molecules ---
    print(f"\n[{args.tag}] Loading generated molecules...")
    gen_mols = load_mols_from_sdf_dir(args.gen_sdf)

    # Always load generated molecules from CSV SMILES (SDF SMILES may not match due to H/tautomers)
    gen_mols_from_csv = []
    if args.gen_csv:
        gen_csv_path = Path(args.gen_csv)
        if gen_csv_path.exists():
            gen_df = pd.read_csv(gen_csv_path)
            gen_vina_ok = gen_df[(gen_df["vina_dock"].notna()) &
                                  (gen_df["vina_dock"] < 0) &
                                  (gen_df["vina_dock"] > -15)]
            print(f"  CSV filter: {len(gen_vina_ok)} molecules with Vina -15~0")
            for smi in gen_vina_ok["smiles"].dropna():
                try:
                    m = Chem.MolFromSmiles(str(smi))
                    if m: gen_mols_from_csv.append(m)
                except: pass
            print(f"  Loaded from CSV SMILES: {len(gen_mols_from_csv)} molecules")
    gen_mols = gen_mols_from_csv if gen_mols_from_csv else gen_mols

    # --- Load active molecules ---
    print(f"[{args.tag}] Loading known actives...")
    ref_mols = load_mols_from_sdf_dir(args.act_sdf)
    print(f"  Known actives: {len(ref_mols)}")

    # --- Background ---
    print(f"[{args.tag}] Loading background...")
    bg_mols = load_bg_mols(n=5000)
    print(f"  Background: {len(bg_mols)}")

    # --- t-SNE ---
    print(f"\n[{args.tag}] t-SNE plots...")
    plot_tsne(gen_mols, ref_mols, out_dir / f"{args.tag}_tsne.png",
              bg_mols, show_text=True, tag=args.tag)
    plot_tsne(gen_mols, ref_mols, out_dir / f"{args.tag}_tsne_notext.png",
              bg_mols, show_text=False, tag=args.tag)

    # --- LogP-MW ---
    print(f"\n[{args.tag}] LogP vs MW plots...")
    act_df = mols_to_df(ref_mols)
    gen_df = mols_to_df(gen_mols)
    bg_df = mols_to_df(bg_mols)
    plot_logp_mw(gen_df, act_df, out_dir / f"{args.tag}_logp_vs_molweight.png",
                 bg_df, show_text=True, tag=args.tag)
    plot_logp_mw(gen_df, act_df, out_dir / f"{args.tag}_logp_vs_molweight_no_text.png",
                 bg_df, show_text=False, tag=args.tag)

    print(f"\n[{args.tag}] Done!")


if __name__ == "__main__":
    main()
