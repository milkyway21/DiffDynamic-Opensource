#!/usr/bin/env python3
"""Draw top 20 molecules for two pockets combined in one 4x10 grid figure."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np

POCKET_LABELS = {'8R12': '8R12', '7RPZ_KRAS': '7RPZ_KRAS (KRAS G12D)'}
COLS = 4
ROWS = 5
SUB_IMG_W, SUB_IMG_H = 400, 320
LEGEND_FONT = 26


def make_grid_image(pocket_key, top20, title):
    """Generate a single 4x5 grid image for one pocket."""
    mols = []
    legends = []
    for i, m in enumerate(top20):
        mol = Chem.MolFromSmiles(m['smiles'])
        if mol is None:
            mols.append(None)
            legends.append(f"#{i+1}  (invalid)")
            continue
        rdDepictor.Compute2DCoords(mol)
        mols.append(mol)
        leg = (f"#{i+1} Vina={m['vina']:.1f} "
               f"QED={m['qed']:.2f} SA={m['sa']:.2f}\n"
               f"  Composite={m['composite']:.3f}")
        legends.append(leg)

    # Fill to 20 with blanks
    while len(mols) < 20:
        mols.append(None)
        legends.append("")

    img = Draw.MolsToGridImage(
        [m for m in mols if m is not None] if None in mols else mols,
        molsPerRow=COLS,
        subImgSize=(SUB_IMG_W, SUB_IMG_H),
        legends=legends,
        legendFontSize=LEGEND_FONT,
        useSVG=False,
    )
    # Add title area
    arr = np.array(img)
    title_h = 60
    titled = np.ones((arr.shape[0] + title_h, arr.shape[1], 3), dtype=np.uint8) * 255
    titled[title_h:, :, :] = arr

    # Use PIL to draw title
    pil_img = Image.fromarray(titled)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((titled.shape[1] - tw) // 2, 10), title, fill=(0, 0, 0), font=font)
    return np.array(pil_img)


def main():
    base = Path('/data/ye/DiffDynamic/outputs/seedforge_bayesian')

    top_grid = None
    bot_grid = None
    titles = []

    for pk in ['8R12', '7RPZ_KRAS']:
        with open(base / f'top20_{pk}.json') as f:
            top20 = json.load(f)

        n_total = len(top20)
        top20 = top20[:20]
        print(f"{pk}: top 20 selected from {n_total} total molecules")

        title = f"{POCKET_LABELS.get(pk, pk)}  —  Top 20 Molecules by Composite Score"
        titles.append(title)
        grid_img = make_grid_image(pk, top20, title)

        if pk == '8R12':
            top_grid = grid_img
        else:
            bot_grid = grid_img

    # Combine into one figure (stack vertically)
    max_w = max(top_grid.shape[1], bot_grid.shape[1])
    # Pad narrower image
    def pad_width(img, target_w):
        if img.shape[1] >= target_w:
            return img
        pad = target_w - img.shape[1]
        left = pad // 2
        padded = np.ones((img.shape[0], target_w, 3), dtype=np.uint8) * 255
        padded[:, left:left+img.shape[1], :] = img
        return padded

    top_grid = pad_width(top_grid, max_w)
    bot_grid = pad_width(bot_grid, max_w)

    separator = np.ones((8, max_w, 3), dtype=np.uint8) * 200
    combined = np.vstack([top_grid, separator, bot_grid])

    outpath = base / 'top20_combined_8R12_7RPZ_KRAS.png'
    Image.fromarray(combined).save(str(outpath), dpi=(200, 200))
    print(f"\nCombined figure saved: {outpath}")
    print(f"  Size: {combined.shape[1]} x {combined.shape[0]} px")


if __name__ == '__main__':
    main()
