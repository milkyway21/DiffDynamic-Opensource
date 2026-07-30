#!/usr/bin/env python3
"""Visualize vina residue contact matrices → PNG + viz_summary.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "experiments" / "vina_residue_contacts_gen1000"
PLOT = ROOT / "plots"
PLOT.mkdir(parents=True, exist_ok=True)
META = ["pocket", "molecule_id", "smiles", "vina_dock"]
C_BAR = "#5A7FA0"
C_ORANGE = "#F56E1A"


def load(pocket: str):
    df = pd.read_csv(ROOT / f"{pocket}_vina_residue_contacts_cutoff5.0A.csv")
    res = [c for c in df.columns if c not in META]
    return df, res


def short_res(name: str) -> str:
    parts = name.split("_")
    if len(parts) >= 3:
        return f"{parts[0]}{parts[1]} {parts[2]}"
    return name


def main() -> None:
    canvas_payload = {}
    for pocket, title in [("8ri2", "8RI2"), ("6w63", "6W63")]:
        df, res = load(pocket)
        n = len(df)
        rates, means = [], []
        for c in res:
            s = df[c]
            rates.append(float(s.notna().mean()))
            means.append(float(s.mean()) if s.notna().any() else float("nan"))
        order = np.argsort(-np.asarray(rates))
        res_o = [res[i] for i in order]
        rates_o = [rates[i] for i in order]
        means_o = [means[i] for i in order]
        x = np.arange(len(res_o))
        labels = [short_res(r) for r in res_o]

        fig, ax = plt.subplots(figsize=(max(10, len(res) * 0.28), 4.2))
        ax.bar(x, [r * 100 for r in rates_o], color=C_BAR, width=0.8, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylabel("% molecules with contact (≤5 Å)")
        ax.set_xlabel("Residue")
        ax.set_title(f"{title}: residue contact frequency (n={n} molecules with Vina)")
        ax.set_ylim(0, 105)
        ax.axhline(50, color="#999", lw=0.6, ls="--")
        fig.tight_layout()
        p1 = PLOT / f"{pocket}_residue_contact_freq.png"
        fig.savefig(p1, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("saved", p1)

        df_s = df.sort_values("vina_dock").reset_index(drop=True)
        mat = df_s[res_o].to_numpy(dtype=float)
        fig_h = max(6.0, min(14.0, n * 0.008))
        fig, ax = plt.subplots(figsize=(max(10, len(res_o) * 0.28), fig_h))
        cmap = plt.cm.get_cmap("YlGnBu_r").copy()
        cmap.set_bad(color="#f0f0f0")
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=2.0, vmax=5.0, interpolation="nearest")
        ax.set_xticks(np.arange(len(res_o)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        y_idx = np.linspace(0, n - 1, num=min(8, n), dtype=int)
        ax.set_yticks(y_idx)
        ax.set_yticklabels([f"{df_s.loc[i, 'vina_dock']:.1f}" for i in y_idx], fontsize=7)
        ax.set_ylabel("Molecules (sorted by Vina affinity, kcal/mol)")
        ax.set_xlabel("Residue")
        ax.set_title(f"{title}: ligand–residue min distance heatmap (Å; blank = no contact ≤5 Å)")
        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        cbar.set_label("Min heavy-atom distance (Å)")
        fig.tight_layout()
        p2 = PLOT / f"{pocket}_residue_distance_heatmap.png"
        fig.savefig(p2, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print("saved", p2)

        fig, ax = plt.subplots(figsize=(max(10, len(res_o) * 0.28), 4.0))
        ax.bar(x, means_o, color=C_ORANGE, width=0.8, alpha=0.85, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylabel("Mean min distance among contacts (Å)")
        ax.set_xlabel("Residue (ordered by contact frequency)")
        ax.set_title(f"{title}: mean contact distance by residue")
        ax.set_ylim(0, 5.2)
        fig.tight_layout()
        p3 = PLOT / f"{pocket}_residue_mean_distance.png"
        fig.savefig(p3, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("saved", p3)

        n_c = df[res].notna().sum(axis=1)
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        ax.scatter(df["vina_dock"], n_c, s=12, alpha=0.35, c=C_BAR, edgecolors="none")
        ax.set_xlabel("Vina dock affinity (kcal/mol)")
        ax.set_ylabel("Number of contacting residues (≤5 Å)")
        ax.set_title(f"{title}: contact breadth vs Vina")
        fig.tight_layout()
        p4 = PLOT / f"{pocket}_contacts_vs_vina.png"
        fig.savefig(p4, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("saved", p4)

        canvas_payload[pocket] = {
            "title": title,
            "n_mols": int(n),
            "n_residues": len(res),
            "vina_min": float(df.vina_dock.min()),
            "vina_median": float(df.vina_dock.median()),
            "vina_max": float(df.vina_dock.max()),
            "contacts_per_mol_mean": float(n_c.mean()),
            "residues": [
                {
                    "residue": labels[i],
                    "residue_key": res_o[i],
                    "contact_pct": round(rates_o[i] * 100, 1),
                    "mean_dist": None
                    if not np.isfinite(means_o[i])
                    else round(float(means_o[i]), 3),
                    "n_contact": int(df[res_o[i]].notna().sum()),
                }
                for i in range(len(res_o))
            ],
        }

    (ROOT / "viz_summary.json").write_text(json.dumps(canvas_payload, indent=2))
    print("wrote", ROOT / "viz_summary.json")


if __name__ == "__main__":
    main()
