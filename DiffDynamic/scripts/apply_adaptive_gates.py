#!/usr/bin/env python3
"""根据配体/活性分子自适应计算 Prudent gate，写出 JSON/YAML，可选合并进运行配置。

Examples:
  python3 scripts/apply_adaptive_gates.py \\
    --ligand /data/ye/protein-ligand/6W63_ligand_4WI_pose.sdf \\
    --actives /data/ye/protein-ligand/6W63/active_dock_qed_sa.csv \\
    --config configs/run_6w63_4wi_ligandsize_normal_gen100_prudent_tbr10.yml \\
    --out_dir experiments/.../adaptive_gates
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from utils.adaptive_gates import (
    apply_gates_to_prudent_cfg,
    compute_adaptive_gates,
    write_gates_json,
    write_gates_yaml_snippet,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ligand", required=True)
    ap.add_argument("--actives", default=None)
    ap.add_argument("--config", default=None, help="sampling yml；用于读取 grow/base prudent 并写回")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--write_config", default=None, help="写出合并后的完整 config 路径")
    ap.add_argument("--qed_slack", type=float, default=None)
    ap.add_argument("--logp_slack", type=float, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = {}
    grow_cfg = {}
    base_prudent = {}
    if args.config:
        with open(args.config) as f:
            base_cfg = yaml.safe_load(f)
        sc = (((base_cfg or {}).get("sample") or {}).get("scaffold") or {})
        grow_cfg = sc.get("grow") or {}
        base_prudent = sc.get("prudent") or {}

    margins = {}
    if args.qed_slack is not None:
        margins["qed_slack"] = args.qed_slack
    if args.logp_slack is not None:
        margins["logp_slack"] = args.logp_slack

    gates = compute_adaptive_gates(
        args.ligand,
        actives_csv=args.actives,
        margins=margins or None,
        grow_cfg=grow_cfg,
        base_prudent=base_prudent,
    )
    jpath = write_gates_json(gates, out_dir / "adaptive_gates.json")
    ypath = write_gates_yaml_snippet(gates, out_dir / "adaptive_gates_snippet.yml")
    print("gates:", gates.as_prudent_dict())
    for r in gates.rationale:
        print(" -", r)
    print("wrote", jpath)
    print("wrote", ypath)

    if args.write_config or args.config:
        cfg = copy.deepcopy(base_cfg) if base_cfg else {"sample": {"scaffold": {"prudent": {}}}}
        sc = cfg.setdefault("sample", {}).setdefault("scaffold", {})
        sc["prudent"] = apply_gates_to_prudent_cfg(sc.get("prudent") or {}, gates)
        # 去掉仅运行时元数据，保持 YAML 干净；元数据另存 json
        sc["prudent"].pop("_adaptive_gates", None)
        sc["adaptive_gates"] = {
            "enable": True,
            "source_json": str(jpath),
            "ligand": args.ligand,
            "actives": args.actives,
        }
        out_cfg = Path(args.write_config) if args.write_config else out_dir / "sampling_with_adaptive_gates.yml"
        with open(out_cfg, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print("wrote", out_cfg)


if __name__ == "__main__":
    main()
