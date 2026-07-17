#!/usr/bin/env bash
# Master orchestrator: 6 projects x 100 molecules (sequential_random allocation)
# Wave1: baseline30 (6W63 + 8RI2)
# Wave2: baseline50 (6W63 + 8RI2)
# Wave3: prudent murckosites (6W63 + 8RI2)
set -eo pipefail
cd /data/ye/DiffDynamic
export VERSION=v2
export PYTHONPATH=/data/ye/DiffDynamic:$PYTHONPATH

GPU_6W63="${GPU_6W63:-0}"
GPU_8RI2="${GPU_8RI2:-2}"
export GPU_6W63 GPU_8RI2

LOG="outputs/murckosites_v2_master.log"
mkdir -p outputs

echo "========== $(date -Iseconds) Murckosites v2 master start ==========" | tee -a "$LOG"
echo "GPU_6W63=$GPU_6W63 GPU_8RI2=$GPU_8RI2" | tee -a "$LOG"

fail=0

echo "" | tee -a "$LOG"
echo "========== Wave1: baseline30 ==========" | tee -a "$LOG"
bash run_dl_baseline_pipeline.sh 6w63_b30 8ri2_b30 2>&1 | tee -a "$LOG" || fail=1

echo "" | tee -a "$LOG"
echo "========== Wave2: baseline50 ==========" | tee -a "$LOG"
bash run_dl_baseline_pipeline.sh 6w63_b50 8ri2_b50 2>&1 | tee -a "$LOG" || fail=1

echo "" | tee -a "$LOG"
echo "========== Wave3: prudent murckosites ==========" | tee -a "$LOG"
bash run_prudent_murckosites_pipeline.sh all 2>&1 | tee -a "$LOG" || fail=1

echo "" | tee -a "$LOG"
echo "========== Verification ==========" | tee -a "$LOG"
python3 - <<'PY' | tee -a "$LOG"
from pathlib import Path
import pandas as pd

dirs = [
    ("experiments/6w63_dl_baseline30_v2", "6W63_dl_baseline30_v2"),
    ("experiments/6w63_dl_baseline50_v2", "6W63_dl_baseline50_v2"),
    ("experiments/8ri2_dl_baseline30_v2", "8RI2_dl_baseline30_v2"),
    ("experiments/8ri2_dl_baseline50_v2", "8RI2_dl_baseline50_v2"),
    ("experiments/6w63prudent_murckosites_v2", "6W63_murckosites_v2"),
    ("experiments/8ri2prudent_murckosites_v2", "8RI2_murckosites_v2"),
]
root = Path("/data/ye/DiffDynamic")
ok = True
for d, tag in dirs:
    p = root / d
    pt = list(p.glob("result_custom_*.pt"))
    xlsx = list(p.glob("evaluation_results_*.xlsx"))
    active_png = list((p / "plots").glob(f"{tag}_active_ligands.png")) if (p / "plots").exists() else []
    n_eval = 0
    if xlsx:
        try:
            n_eval = len(pd.read_excel(xlsx[-1], sheet_name="评估结果"))
        except Exception:
            pass
    status = "OK" if pt and xlsx and active_png and n_eval >= 90 else "INCOMPLETE"
    if status != "OK":
        ok = False
    print(f"  {d}: pt={len(pt)} eval={n_eval} active_png={len(active_png)} -> {status}")
print("ALL_OK" if ok else "SOME_INCOMPLETE")
PY

echo "========== $(date -Iseconds) Master done (fail=$fail) ==========" | tee -a "$LOG"
exit $fail
