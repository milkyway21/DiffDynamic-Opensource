#!/usr/bin/env python3
"""
从 .pt 提取分子为 SDF，并生成与 batch_sampleandeval_parallel 一致的评估 Excel。

实现方式：调用仓库内 evaluate_pt_with_correct_reconstruct.py（与 batch 中 run_single_evaluation 相同），
输出目录内会得到 evaluation_results_<时间戳>.xlsx（含「评估结果」「正常分子」「统计信息」等）与 reconstructed_molecules/*.sdf。

用法示例（本机仓库根目录；与 Docker 内 /workspace 等价时把前缀换成 /workspace）:

  cd /home/user/Desktop/Ye/DiffDynamic
  python extract_pt_to_sdf_excel.py \\
    third_party/DecompDiff/outputs_decompdiff_gpu5_pocket10_run/\\
    sampling_drift_pocketbench_010_3dzh_A_rec_3u4i_cvr_lig_tt_docked_0_pocket/eval_out/\\
    eval_000_3dzh_A_rec_3u4i_cvr_lig_tt_docked_0.pt \\
    --protein_root ./data/crossdocked_v1.1_rmsd1.0

  # Docker（仓库挂载为 /workspace 时）:
  # cd /workspace && python3 extract_pt_to_sdf_excel.py \
  #   third_party/DecompDiff/outputs_decompdiff_gpu5_pocket10_run/sampling_drift_pocketbench_010_3dzh_A_rec_3u4i_cvr_lig_tt_docked_0_pocket/eval_out/eval_000_3dzh_A_rec_3u4i_cvr_lig_tt_docked_0.pt \
  #   --protein_root ./data/crossdocked_v1.1_rmsd1.0

  # 指定输出目录（默认: .pt 同目录下 eval_<data_id>_<CST时间戳>/）
  python extract_pt_to_sdf_excel.py path/to/result.pt --output_dir ./my_eval_out

其余参数（如 --exhaustiveness、--force-mmff-minimize）与 evaluate_pt_with_correct_reconstruct.py 一致，可追加在命令末尾。
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
EVAL_SCRIPT = REPO_ROOT / "evaluate_pt_with_correct_reconstruct.py"


def _default_data_id_from_pt_name(stem: str) -> int:
    """从文件名如 eval_000_xxx.pt 解析口袋编号，默认 0。"""
    m = re.match(r"eval_(\d+)_", stem, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m2 = re.match(r"result_(\d+)", stem)
    if m2:
        return int(m2.group(1))
    return 0


def _default_output_dir(pt_path: Path, data_id: int) -> Path:
    cst = timezone(timedelta(hours=8))
    ts = datetime.now(cst).strftime("%Y%m%d_%H%M%S")
    return pt_path.parent / f"eval_{data_id}_{ts}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从 .pt 提取 SDF + Excel（与 batch_sampleandeval_parallel 评估流程一致）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "pt_file",
        type=str,
        help=".pt 路径：TargetDiff 的 dict 或 DecompDiff 的 list[dict]（pred_pos/pred_v，评估时会转为 TargetDiff 形状）",
    )
    parser.add_argument(
        "--protein_root",
        type=str,
        default="./data/crossdocked_v1.1_rmsd1.0",
        help="蛋白数据根目录（与 evaluate 一致）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="评估输出目录；默认与 batch 相同：<pt 目录>/eval_<data_id>_<时间戳>/",
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default=None,
        help="已有评估目录（跳过评估，直接从该目录提取 SDF/Excel）",
    )
    parser.add_argument(
        "--data_id",
        type=int,
        default=None,
        help="用于默认输出目录命名；默认从文件名 eval_<id>_ 或 result_<id> 解析",
    )
    parser.add_argument(
        "--atom_mode",
        type=str,
        choices=["basic", "add_aromatic"],
        default="add_aromatic",
    )
    parser.add_argument(
        "--exhaustiveness",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--save_intermediate_interval",
        type=int,
        default=0,
        help="与 batch 一致默认 0（不写中间结果，节省磁盘）",
    )
    parser.add_argument(
        "--cores_per_task",
        type=int,
        default=1,
        help="限制 OpenMP/MKL 线程数（与 batch 评估子进程一致）",
    )
    parser.add_argument(
        "--force-mmff-minimize",
        action="store_true",
        help="对接前 MMFF（传给 evaluate）",
    )
    parser.add_argument(
        "--mmff-max-iters",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--remove-fragments",
        action="store_true",
        default=False,
        help="对接前去除小碎片，仅保留最大连通片段",
    )
    parser.add_argument(
        "extra_eval_args",
        nargs=argparse.REMAINDER,
        help="其余参数原样传给 evaluate_pt_with_correct_reconstruct.py（勿以 -- 开头时可省略 --）",
    )

    args = parser.parse_args()
    pt_path = Path(args.pt_file).expanduser().resolve()
    if not pt_path.is_file():
        print(f"❌ .pt 不存在: {pt_path}", file=sys.stderr)
        return 1
    if not EVAL_SCRIPT.is_file():
        print(f"❌ 评估脚本不存在: {EVAL_SCRIPT}", file=sys.stderr)
        return 1

    protein_root = Path(args.protein_root).expanduser().resolve()
    if not protein_root.is_dir():
        print(f"❌ protein_root 不存在: {protein_root}", file=sys.stderr)
        return 1

    data_id = args.data_id if args.data_id is not None else _default_data_id_from_pt_name(pt_path.stem)
    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = _default_output_dir(pt_path, data_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.cores_per_task >= 1:
        c = str(args.cores_per_task)
        env["OMP_NUM_THREADS"] = c
        env["MKL_NUM_THREADS"] = c
        env["OPENBLAS_NUM_THREADS"] = c
        env["NUMEXPR_NUM_THREADS"] = c
        env["VECLIB_MAXIMUM_THREADS"] = c
    if "EVAL_SINGLE_MOL_TIMEOUT" not in env:
        env["EVAL_SINGLE_MOL_TIMEOUT"] = "10800"

    # 如果指定了 --eval_dir，跳过评估，直接从已有目录提取
    if args.eval_dir:
        eval_dir = Path(args.eval_dir).expanduser().resolve()
        if not eval_dir.is_dir():
            print(f"❌ eval_dir 不存在: {eval_dir}", file=sys.stderr)
            return 1
        print(f"跳过评估，使用已有评估目录: {eval_dir}")
        print(f"输入: {pt_path}")
        return 0

    cmd: list[str] = [
        sys.executable,
        str(EVAL_SCRIPT),
        str(pt_path),
        "--protein_root",
        str(protein_root),
        "--output_dir",
        str(out_dir),
        "--atom_mode",
        args.atom_mode,
        "--exhaustiveness",
        str(args.exhaustiveness),
        "--save_intermediate_interval",
        str(args.save_intermediate_interval),
    ]
    if args.force_mmff_minimize:
        cmd.append("--force-mmff-minimize")
    if args.mmff_max_iters is not None:
        cmd.extend(["--mmff-max-iters", str(args.mmff_max_iters)])
    if args.remove_fragments:
        cmd.append("--remove-fragments")

    # REMAINDER 可能带前导的 "--"；去掉与上面已显式传入等价的重复项（避免命令行里出现两次 --protein_root）
    rest = list(args.extra_eval_args or [])
    if rest and rest[0] == "--":
        rest = rest[1:]
    pr_resolved = str(protein_root)
    rest_f: list[str] = []
    i = 0
    while i < len(rest):
        t = rest[i]
        if t == "--protein_root" and i + 1 < len(rest):
            nxt = rest[i + 1]
            if str(Path(nxt).expanduser().resolve()) == pr_resolved:
                i += 2
                continue
        if t == "--output_dir" and i + 1 < len(rest):
            if str(Path(rest[i + 1]).expanduser().resolve()) == str(out_dir):
                i += 2
                continue
        rest_f.append(t)
        i += 1
    cmd.extend(rest_f)

    print(f"输入: {pt_path}")
    print(f"蛋白根目录: {protein_root}")
    print(f"输出目录（SDF + evaluation_results_*.xlsx）: {out_dir}")
    print(f"命名用 data_id={data_id}")
    print(f"命令: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)
    except subprocess.CalledProcessError as e:
        return e.returncode or 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
