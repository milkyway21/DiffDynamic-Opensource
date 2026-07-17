#!/usr/bin/env python3
"""Draw top 20 molecules without text labels — clean version."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from PIL import Image
import numpy as np

COLS = 4
ROWS = 5
SUB_IMG_W, SUB_IMG_H = 400, 320


def make_grid_image(top20):
    mols = []
    for m in top20:
        mol = Chem.MolFromSmiles(m['smiles'])
        if mol is None:
            mols.append(None)
            continue
        rdDepictor.Compute2DCoords(mol)
        mols.append(mol)
    while len(mols) < 20:
        mols.append(None)

    # Draw without legends
    # Filter None and draw only valid mols, then manually arrange in grid
    img = Draw.MolsToGridImage(
        [m for m in mols if m is not None],
        molsPerRow=COLS,
        subImgSize=(SUB_IMG_W, SUB_IMG_H),
        useSVG=False,
    )
    return np.array(img)


def main():
    base = Path('/data/ye/DiffDynamic/outputs/seedforge_bayesian')

    grids = []
    for pk in ['8R12', '7RPZ_KRAS']:
        with open(base / f'top20_{pk}.json') as f:
            top20 = json.load(f)[:20]
        grids.append(make_grid_image(top20))

    # Pad to same width
    max_w = max(g.shape[1] for g in grids)
    def pad_width(img, target_w):
        if img.shape[1] >= target_w:
            return img
        pad = target_w - img.shape[1]
        left = pad // 2
        padded = np.ones((img.shape[0], target_w, 3), dtype=np.uint8) * 255
        padded[:, left:left+img.shape[1], :] = img
        return padded

    grids = [pad_width(g, max_w) for g in grids]
    sep = np.ones((6, max_w, 3), dtype=np.uint8) * 220
    combined = np.vstack([grids[0], sep, grids[1]])

    outpath = base / 'top20_combined_8R12_7RPZ_KRAS_notext.png'
    Image.fromarray(combined).save(str(outpath))
    print(f"Saved: {outpath}")
    print(f"  Size: {combined.shape[1]} x {combined.shape[0]} px")


if __name__ == '__main__':
    main()
