#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对已保存的采样结果 .pt 执行 TargetDiff baseline refine（sampling.yml 中
``sample.targetdiff_baseline_refine``：enable / start_t），再按
``dd0120/batch_sampleandeval_parallel.py`` 中 ``run_single_evaluation`` 的方式
调用 ``evaluate_pt_with_correct_reconstruct.py`` 重新评估。

典型用法（仓库根目录执行）::

    python refine_saved_pt_baseline_then_eval.py \\
        --input_dir ./dd0414base \\
        --output_dir ./dd0414base_refined \\
        --config ./configs/sampling.yml \\
        --protein_root ./data/crossdocked_v1.1_rmsd1.0_pocket10 \\
        --device cuda:0

说明：
- 不重新跑 dynamic 采样，只读取 .pt 中的 ``pred_ligand_pos`` / ``pred_ligand_v``，
  调用与 ``scripts/sample_diffusion.py`` 相同的 ``apply_targetdiff_baseline_refine_to_sampling_result``。
- ``--config`` 须与生成原始 .pt 时一致的模型检查点（``model.checkpoint``），且 YAML 中已启用
  ``targetdiff_baseline_refine``；``start_t`` 以当前 YAML 为准。
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import utils.misc as misc
import utils.transforms as trans
from models.molopt_score_model import DiffDynamic, ScorePosNet3D
from scripts.sample_diffusion import (
    apply_targetdiff_baseline_refine_to_sampling_result,
    resolve_and_set_absolute_protein_path,
)

EVAL_SCRIPT = REPO_ROOT / "evaluate_pt_with_correct_reconstruct.py"


def _torch_load_pt(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _parse_data_id(name: str) -> int | None:
    m = re.match(r"result_(\d+)_", name)
    if m:
        return int(m.group(1))
    return None


def _build_model(config, ckpt, device: str, logger):
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt["config"].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)

    model_cfg = ckpt["config"].model
    if hasattr(config.model, "use_grad_fusion"):
        model_cfg.use_grad_fusion = config.model.use_grad_fusion
    if hasattr(config.model, "grad_fusion_lambda"):
        model_cfg.grad_fusion_lambda = config.model.grad_fusion_lambda

    model_name = getattr(model_cfg, "name", "score").lower()
    model_cls = DiffDynamic if model_name in ("glintdm", "diffdynamic") else ScorePosNet3D
    model = model_cls(
        model_cfg,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if hasattr(config, "sample") and hasattr(config.sample, "dynamic"):
        dynamic_cfg = config.sample.dynamic
        if hasattr(dynamic_cfg, "large_step"):
            model_cfg.dynamic_large_step = dynamic_cfg.large_step
        if hasattr(dynamic_cfg, "refine"):
            model_cfg.dynamic_refine = dynamic_cfg.refine
        model.dynamic_large_step_defaults = getattr(model_cfg, "dynamic_large_step", {})
        model.dynamic_refine_defaults = getattr(model_cfg, "dynamic_refine", {})

    return model, ligand_atom_mode


def load_stack(config_path: Path, device: str, logger):
    cfg_abs = str(config_path.resolve())
    if not os.path.isfile(cfg_abs):
        raise FileNotFoundError(f"配置文件不存在: {cfg_abs}")
    config = misc.load_config(cfg_abs)
    misc.seed_all(config.sample.seed)

    ckpt_path = config.model.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = str((REPO_ROOT / ckpt_path).resolve())
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"模型检查点不存在: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model, _ligand_atom_mode = _build_model(config, ckpt, device, logger)
    dataset_root = None
    try:
        dp = getattr(ckpt["config"].data, "path", None)
        if dp:
            dataset_root = str(Path(dp).expanduser().resolve())
    except Exception:
        pass
    return config, model, ckpt, dataset_root


def refine_one_pt(
    pt_in: Path,
    out_dir: Path,
    model,
    config,
    dataset_root: str | None,
    protein_root: str | None,
    device: str,
    logger,
) -> Path | None:
    result = _torch_load_pt(pt_in)
    if not isinstance(result, dict):
        print(f"[跳过] 非字典结构: {pt_in}")
        return None
    data = result.get("data")
    if data is None:
        print(f"[跳过] 无 data 字段: {pt_in}")
        return None

    resolve_and_set_absolute_protein_path(
        data,
        dataset_root=dataset_root,
        protein_root=protein_root,
        logger=logger,
    )

    refine_cfg = config.sample.get("targetdiff_baseline_refine", {})
    if not refine_cfg.get("enable", False):
        print(
            "[错误] sampling.yml 中 targetdiff_baseline_refine.enable 为 false，"
            "请改为 true 后再运行（与第 294–297 行配置一致）。"
        )
        sys.exit(2)

    apply_targetdiff_baseline_refine_to_sampling_result(
        model, data, result, config, device=device, logger=logger
    )

    meta = result.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        result["meta"] = meta
    start_t = int(refine_cfg.get("start_t", 10))
    meta["posthoc_targetdiff_baseline_refine"] = {
        "start_t": start_t,
        "source_pt": str(pt_in.resolve()),
    }

    cst = timezone(timedelta(hours=8))
    ts = datetime.now(cst).strftime("%Y%m%d_%H%M%S")
    data_id = _parse_data_id(pt_in.name)
    if data_id is not None:
        out_name = f"result_{data_id}_tbr{start_t}_{ts}.pt"
    else:
        out_name = f"{pt_in.stem}_tbr{start_t}_{ts}.pt"
    out_path = out_dir / out_name
    extra = result.get("extra_info")
    if isinstance(extra, dict):
        extra["posthoc_refined_from"] = str(pt_in.resolve())
        extra["result_file"] = str(out_path.resolve())

    torch.save(result, str(out_path))
    print(f"[保存] {out_path}")
    return out_path


def run_single_evaluation(
    pt_file: Path, protein_root: Path, data_id: int, atom_mode: str, exhaustiveness: int
) -> tuple[bool, str, str | None]:
    """对齐 dd0120/batch_sampleandeval_parallel.run_single_evaluation。"""
    print(f"[评估] 开始: {pt_file.name} (data_id={data_id})")
    cst = timezone(timedelta(hours=8))
    eval_ts = datetime.now(cst).strftime("%Y%m%d_%H%M%S")
    eval_output_dir = pt_file.parent / f"eval_{data_id}_{eval_ts}"
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        str(pt_file),
        "--protein_root",
        str(protein_root),
        "--output_dir",
        str(eval_output_dir),
        "--atom_mode",
        atom_mode,
        "--exhaustiveness",
        str(exhaustiveness),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=7200)
        print(f"[评估] 完成: {eval_output_dir}")
        return True, "ok", str(eval_output_dir)
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or str(e))[:2000]
        print(f"[评估] 失败 data_id={data_id}:\n{msg}")
        return False, msg, None
    except subprocess.TimeoutExpired:
        print(f"[评估] 超时 data_id={data_id}")
        return False, "timeout", None


def main():
    parser = argparse.ArgumentParser(
        description="对目录内 result_*.pt 做 TargetDiff baseline refine 并重新评估"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(REPO_ROOT / "dd0414base"),
        help="含 result_*.pt 的目录（默认: ./dd0414base）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(REPO_ROOT / "dd0414base_refined"),
        help="精炼后 .pt 输出目录（默认: ./dd0414base_refined）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "configs" / "sampling.yml"),
        help="sampling.yml 路径（须含 targetdiff_baseline_refine）",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--protein_root",
        type=str,
        default=None,
        help="与 evaluate_pt 一致；默认尝试 data/crossdocked_v1.1_rmsd1.0_pocket10",
    )
    parser.add_argument("--atom_mode", type=str, default="add_aromatic")
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="只做 refine 保存 .pt，不调用 evaluate_pt",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    protein_root = args.protein_root
    if not protein_root:
        for cand in (
            REPO_ROOT / "data" / "crossdocked_v1.1_rmsd1.0_pocket10",
            Path("/mnt/e/DiffDynamic/data/crossdocked_v1.1_rmsd1.0_pocket10"),
            REPO_ROOT / "data" / "crossdocked_v1.1_rmsd1.0",
        ):
            if cand.is_dir():
                protein_root = str(cand)
                break
    if not protein_root or not Path(protein_root).is_dir():
        print("请使用 --protein_root 指定存在的蛋白数据根目录。")
        sys.exit(1)
    protein_root = str(Path(protein_root).resolve())

    if not EVAL_SCRIPT.is_file():
        print(f"缺少评估脚本: {EVAL_SCRIPT}")
        sys.exit(1)

    pt_files = sorted(input_dir.glob("result_*.pt"))
    if not pt_files:
        print(f"在 {input_dir} 未找到 result_*.pt")
        sys.exit(1)

    logger = misc.get_logger("posthoc_refine")
    config_path = Path(args.config)
    config, model, _ckpt, dataset_root = load_stack(config_path, args.device, logger)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        did = int(args.device.split(":")[-1]) if ":" in args.device else 0
        torch.cuda.set_device(did)

    ok_refine = 0
    refined_paths: list[tuple[Path, int | None]] = []
    for pt_in in pt_files:
        try:
            outp = refine_one_pt(
                pt_in,
                out_dir,
                model,
                config,
                dataset_root,
                protein_root,
                args.device,
                logger,
            )
            if outp is not None:
                ok_refine += 1
                did = _parse_data_id(pt_in.name)
                refined_paths.append((outp, did if did is not None else -1))
        except Exception as e:
            print(f"[失败] {pt_in.name}: {e}")
            traceback.print_exc()
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n精炼完成: {ok_refine}/{len(pt_files)} 写入 {out_dir}")

    if args.skip_eval:
        return

    ok_eval = 0
    for pt_path, data_id in refined_paths:
        if data_id < 0:
            print(f"[评估跳过] 无法从文件名解析 data_id: {pt_path.name}")
            continue
        ok, _, _ = run_single_evaluation(
            pt_path,
            Path(protein_root),
            data_id,
            args.atom_mode,
            args.exhaustiveness,
        )
        if ok:
            ok_eval += 1
    print(f"\n评估完成: {ok_eval}/{len([p for p, d in refined_paths if d >= 0])}")


if __name__ == "__main__":
    main()
