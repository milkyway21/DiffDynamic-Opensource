#!/usr/bin/env python3
"""Create the coarse GSPT1 scaffold prior used by scaffold-only sampling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.gspt1_scaffold_prior import build_scaffold_profile, write_scaffold_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-sdf", required=True)
    parser.add_argument("--native-ligand-sdf", required=True)
    parser.add_argument("--scaffold-smarts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exploration-floor", type=float, default=1.0)
    args = parser.parse_args()

    profile = build_scaffold_profile(
        args.reference_sdf,
        args.native_ligand_sdf,
        args.scaffold_smarts,
        exploration_floor=args.exploration_floor,
    )
    write_scaffold_profile(profile, args.output)
    print(json.dumps(profile, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
