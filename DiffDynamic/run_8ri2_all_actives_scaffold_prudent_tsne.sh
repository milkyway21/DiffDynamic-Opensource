#!/usr/bin/env bash
# 8RI2 骨架库 t-SNE：已知活性 + 口袋配体(A1H02) 各做 Murcko-scaffold prudent，
# 每骨架取 N_SAMPLES(=100) → 共 n×100 分子作 library t-SNE。
#
# Usage:
#   GPU=1 bash run_8ri2_all_actives_scaffold_prudent_tsne.sh
#   N_SAMPLES=100 SKIP_GEN=1 bash ...   # 仅重建+出图（复用已有 .pt）
#   FORCE_REGEN=1 bash ...              # 忽略已有 .pt 重跑生成
set -eo pipefail
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate diffdynamic
export PYTHONPATH=/data/ye/DiffDynamic:$PYTHONPATH
cd /data/ye/DiffDynamic

OUT="experiments/8ri2_all_actives_scaffold_prudent_tsne"
CFG="configs/run_8ri2_all_actives_scaffold_prudent.yml"
PROTEIN="/data/ye/protein-ligand/8RI2/8RI2_apo.pdb"
REF_IN_POCKET="/data/ye/protein-ligand/8RI2/8RI2_ligand.sdf"
ACT_SDF_DIR="/data/ye/protein-ligand/8RI2/active_ligands/sdf"
GPU="${GPU:-1}"
N_SAMPLES="${N_SAMPLES:-100}"
# 生成时略多于目标，给重建失败留余量；FILL_SHORT=1 只补缺额骨架
GEN_BUFFER="${GEN_BUFFER:-40}"
SKIP_GEN="${SKIP_GEN:-0}"
FORCE_REGEN="${FORCE_REGEN:-0}"
FILL_SHORT="${FILL_SHORT:-0}"
# 逗号分隔骨架名；空=全部。例：ONLY_SCAFFOLDS=A1A4L,A1D79,WTN,XE3
ONLY_SCAFFOLDS="${ONLY_SCAFFOLDS:-}"
# 并行补缺：GPU 列表，与 ONLY_SCAFFOLDS 一一对应（或轮询）
FILL_GPUS="${FILL_GPUS:-1,2,3,5}"

mkdir -p "$OUT"/{active_poses,runs,generated_sdf,plots}
GEN_NUM=$((N_SAMPLES + GEN_BUFFER))
echo "OUT=$OUT GPU=$GPU N_SAMPLES=$N_SAMPLES GEN_NUM=$GEN_NUM SKIP_GEN=$SKIP_GEN FORCE_REGEN=$FORCE_REGEN FILL_SHORT=$FILL_SHORT ONLY_SCAFFOLDS=$ONLY_SCAFFOLDS START=$(date -Iseconds)"

# ---------- 1) Dock / place all known actives into pocket ----------
python3 - <<PY
from pathlib import Path
import json, shutil, subprocess, tempfile
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from utils.evaluation.docking_vina import VinaDockingTask

OUT = Path("$OUT")
POSE_DIR = OUT / "active_poses"
POSE_DIR.mkdir(parents=True, exist_ok=True)
ACT_DIR = Path("$ACT_SDF_DIR")
PROTEIN = "$PROTEIN"
REF = "$REF_IN_POCKET"
OBABEL = "/home/user/anaconda3/envs/diffdynamic/bin/obabel"

# 口袋配体 / 已知晶体 pose 直接复用（A1H02=原生配体，RM5=已有 pose）
KNOWN_POSES = {
    "A1H02": Path("/data/ye/protein-ligand/8RI2/8RI2_ligand.sdf"),
    "RM5": Path("/data/ye/protein-ligand/8RI2/8RI2_ligand_RM5_pose.sdf"),
}

ref = Chem.SDMolSupplier(REF, removeHs=False)[0]
Chem.SanitizeMol(ref)
ref_h = Chem.AddHs(ref, addCoords=True)
ref_pos = ref_h.GetConformer().GetPositions()
ref_center = (ref_pos.max(0) + ref_pos.min(0)) / 2


def pdbqt_pose_to_sdf(pose_str: str, out_sdf: Path, name: str, aff: float):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pq = td / "pose.pdbqt"
        pq.write_text(pose_str if isinstance(pose_str, str) else pose_str.decode())
        sdf_tmp = td / "pose.sdf"
        subprocess.check_call([OBABEL, str(pq), "-O", str(sdf_tmp)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        mol = Chem.SDMolSupplier(str(sdf_tmp), removeHs=False)[0]
        if mol is None:
            raise RuntimeError("obabel produced empty mol")
        mol.SetProp("_Name", f"{name}_docked_8RI2")
        if np.isfinite(aff):
            mol.SetProp("vina_dock", f"{aff:.3f}")
        w = Chem.SDWriter(str(out_sdf)); w.write(mol); w.close()
        return mol

manifest = []
for sf in sorted(ACT_DIR.glob("*.sdf")):
    name = sf.stem
    out_sdf = POSE_DIR / f"{name}_pose.sdf"
    if out_sdf.exists() and name not in ("__force__",):
        # reuse existing pose unless FORCE — poses are stable
        mol = Chem.SDMolSupplier(str(out_sdf), removeHs=False)[0]
        print(f"[keep] {name} -> {out_sdf.name}")
    elif name in KNOWN_POSES and KNOWN_POSES[name].exists():
        shutil.copy(KNOWN_POSES[name], out_sdf)
        mol = Chem.SDMolSupplier(str(out_sdf), removeHs=False)[0]
        print(f"[reuse] {name} -> {out_sdf.name}")
    else:
        mol0 = Chem.SDMolSupplier(str(sf), removeHs=False)[0]
        if mol0 is None:
            print(f"[skip] {name}: invalid"); continue
        try:
            Chem.SanitizeMol(mol0)
        except Exception as e:
            print(f"[warn] sanitize {name}: {e}")
        mol = Chem.AddHs(mol0)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass
        conf = mol.GetConformer()
        pos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        delta = ref_center - ((pos.max(0) + pos.min(0)) / 2)
        for i in range(mol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            conf.SetAtomPosition(i, (p.x + delta[0], p.y + delta[1], p.z + delta[2]))
        task = VinaDockingTask(protein_path=PROTEIN, ligand_rdmol=mol)
        dock = task.run(mode="dock", exhaustiveness=8, n_poses=1)
        aff = float(dock[0]["affinity"]) if dock else float("nan")
        pose_str = dock[0].get("pose") if dock else None
        if not pose_str:
            raise RuntimeError(f"no pose for {name}")
        if "ENDMDL" in pose_str:
            pose_str = pose_str.split("ENDMDL")[0] + "ENDMDL\n"
        mol = pdbqt_pose_to_sdf(pose_str, out_sdf, name, aff)
        print(f"[dock] {name} vina={aff:.3f} -> {out_sdf.name}")

    conf = mol.GetConformer()
    xyz = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                    for i in range(mol.GetNumAtoms())])
    print(f"  centroid={xyz.mean(0).round(2)} atoms={mol.GetNumAtoms()}")
    manifest.append({"name": name, "pose_sdf": str(out_sdf)})

(OUT / "active_poses_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Poses ready: {len(manifest)} (= n scaffolds; library size target = n * $N_SAMPLES)")
PY

# ---------- 2) Sync config num_samples (GEN_NUM = target + buffer) ----------
python3 - <<PY
from pathlib import Path
import re
cfg = Path("$CFG")
text = cfg.read_text() if cfg.exists() else Path("configs/run_8ri2_ligand_instant_gen100_prudent_tbr10.yml").read_text()
text2, n = re.subn(r"(num_samples:\s*)\d+", r"\g<1>$GEN_NUM", text, count=1)
if n == 0:
    raise SystemExit("failed to set num_samples in $CFG")
if not text2.lstrip().startswith("#"):
    text2 = "# 8RI2 all-known-actives + ligand Murcko × prudent library t-SNE (n×$N_SAMPLES)\n" + text2
cfg.write_text(text2)
print(f"Config $CFG num_samples=$GEN_NUM (library keep $N_SAMPLES each)")
PY

should_gen_scaffold() {
  local name="$1"
  if [[ -n "$ONLY_SCAFFOLDS" ]]; then
    [[ ",$ONLY_SCAFFOLDS," == *",$name,"* ]] || return 1
  fi
  return 0
}

# ---------- 3) Prudent generation per active/ligand pose (no eval) ----------
if [[ "$SKIP_GEN" != "1" ]]; then
  for pose in "$OUT"/active_poses/*_pose.sdf; do
    name=$(basename "$pose" _pose.sdf)
    should_gen_scaffold "$name" || continue
    run_dir="$OUT/runs/$name"
    mkdir -p "$run_dir"
    if [[ "$FORCE_REGEN" != "1" ]] && [[ "$FILL_SHORT" != "1" ]] && ls "$run_dir"/result_custom_*.pt >/dev/null 2>&1; then
      echo "[skip gen] $name already has .pt"
      continue
    fi
    if [[ "$FORCE_REGEN" == "1" ]] || [[ "$FILL_SHORT" == "1" ]]; then
      rm -f "$run_dir"/result_custom_*.pt
    fi
    echo "========== Generate prudent scaffold=$name (gen=$GEN_NUM, keep=$N_SAMPLES) =========="
    python -u scripts/sample_diffusion.py "$CFG" \
      --protein_path "$PROTEIN" --ligand_path "$pose" \
      --result_path "$run_dir" --device "cuda:${GPU}" \
      2>&1 | tee "$OUT/runs/${name}_generation.log"
  done
else
  echo "[SKIP_GEN] reuse existing .pt under $OUT/runs"
fi

# ---------- 4) Reconstruct until N_SAMPLES success per scaffold ----------
recon_library() {
N_SAMPLES="$N_SAMPLES" OUT="$OUT" python3 - <<'PY'
import json
import os
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from utils import reconstruct
import utils.transforms as trans

OUT = Path(os.environ["OUT"])
N_SAMPLES = int(os.environ["N_SAMPLES"])
GEN = OUT / "generated_sdf"
GEN.mkdir(parents=True, exist_ok=True)
for f in GEN.glob("*.sdf"):
    f.unlink()

def recon_one(pos, v):
    pos = pos.numpy() if hasattr(pos, "numpy") else np.asarray(pos)
    v = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
    atom_type = v.argmax(-1) if getattr(v, "ndim", 1) > 1 else v.astype(int)
    atomic_nums = trans.get_atomic_number_from_index(atom_type, mode="add_aromatic")
    aromatic = trans.is_aromatic_from_index(atom_type, mode="add_aromatic")
    return reconstruct.reconstruct_from_generated(pos, atomic_nums, aromatic)

per_scaffold = {}
n_ok = 0
short = []
for pt in sorted((OUT / "runs").glob("*/result_custom_*.pt")):
    tag = pt.parent.name
    data = torch.load(pt, map_location="cpu", weights_only=False)
    pos_list, v_list = data["pred_ligand_pos"], data["pred_ligand_v"]
    wrote = 0
    for i in range(len(pos_list)):
        if wrote >= N_SAMPLES:
            break
        try:
            mol = recon_one(pos_list[i], v_list[i])
            if mol is None:
                continue
            mol.SetProp("_Name", f"{tag}_{wrote:03d}")
            mol.SetProp("scaffold_source", tag)
            path = GEN / f"{tag}_gen_{wrote:03d}.sdf"
            w = Chem.SDWriter(str(path)); w.write(mol); w.close()
            wrote += 1
            n_ok += 1
        except Exception:
            continue
    per_scaffold[tag] = {"pt_n": len(pos_list), "wrote": wrote, "target": N_SAMPLES}
    status = "OK" if wrote >= N_SAMPLES else "SHORT"
    print(f"[recon] {tag}: wrote {wrote}/{N_SAMPLES} (pt has {len(pos_list)}) [{status}]")
    if wrote < N_SAMPLES:
        short.append(tag)

n_scaf = len(per_scaffold)
manifest = {
    "n_scaffolds": n_scaf,
    "n_samples_target": N_SAMPLES,
    "library_target": n_scaf * N_SAMPLES,
    "library_actual": n_ok,
    "short_scaffolds": short,
    "per_scaffold": per_scaffold,
    "note": "scaffold library = known actives + pocket ligand (A1H02); exact n×N_SAMPLES required",
}
(OUT / "library_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print(f"Library: scaffolds={n_scaf} target={n_scaf*N_SAMPLES} actual={n_ok} short={short} → {GEN}")
if short:
    raise SystemExit(2)
PY
}

# If FILL_SHORT + parallel GPU list: regenerate only short scaffolds in parallel
fill_short_parallel() {
  local short_csv="$1"
  IFS=',' read -r -a scafs <<< "$short_csv"
  IFS=',' read -r -a gpus <<< "$FILL_GPUS"
  local pids=()
  local i=0
  for name in "${scafs[@]}"; do
    [[ -z "$name" ]] && continue
    local g="${gpus[$((i % ${#gpus[@]}))]}"
    local pose="$OUT/active_poses/${name}_pose.sdf"
    local run_dir="$OUT/runs/$name"
    mkdir -p "$run_dir"
    rm -f "$run_dir"/result_custom_*.pt
    echo "========== FILL PARALLEL scaffold=$name on GPU=$g (gen=$GEN_NUM) =========="
    (
      python -u scripts/sample_diffusion.py "$CFG" \
        --protein_path "$PROTEIN" --ligand_path "$pose" \
        --result_path "$run_dir" --device "cuda:${g}" \
        > "$OUT/runs/${name}_generation.log" 2>&1
      echo "FILL_DONE $name gpu=$g exit=$?"
    ) &
    pids+=($!)
    i=$((i + 1))
  done
  local fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done
  return $fail
}

# Recon; on shortage, parallel refill up to 3 rounds
MAX_FILL_ROUNDS="${MAX_FILL_ROUNDS:-3}"
round=0
while true; do
  set +e
  recon_library
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    echo "Library complete: exact $N_SAMPLES per scaffold"
    break
  fi
  short=$(python3 -c "import json; print(','.join(json.load(open('$OUT/library_manifest.json'))['short_scaffolds']))")
  if [[ -z "$short" ]]; then
    echo "ERROR: recon failed without short list"; exit 1
  fi
  round=$((round + 1))
  if [[ $round -gt $MAX_FILL_ROUNDS ]]; then
    echo "ERROR: still short after $MAX_FILL_ROUNDS fill rounds: $short"
    cat "$OUT/library_manifest.json"
    exit 1
  fi
  echo "========== FILL ROUND $round short=[$short] =========="
  fill_short_parallel "$short"
done

# ---------- 5) t-SNE (scaffold library vs known actives) ----------
OUT="$OUT" python3 - <<'PY'
import os
import sys
from pathlib import Path
sys.path.insert(0, "/data/ye/protein-ligand")
from plot_generic_results import load_mols_from_sdf_dir, load_bg_mols, plot_tsne

OUT = Path(os.environ["OUT"])
plots = OUT / "plots"
plots.mkdir(parents=True, exist_ok=True)
tag = "8RI2_all_actives_library_n100"

gen_mols = load_mols_from_sdf_dir(OUT / "generated_sdf")
ref_mols = load_mols_from_sdf_dir(OUT / "active_poses")
if not ref_mols:
    ref_mols = load_mols_from_sdf_dir("/data/ye/protein-ligand/8RI2/active_ligands/sdf")
bg_mols = load_bg_mols(n=5000)
print(f"t-SNE inputs: gen={len(gen_mols)} ref={len(ref_mols)} bg={len(bg_mols)}")
plot_tsne(gen_mols, ref_mols, plots / f"{tag}_tsne.png", bg_mols, show_text=True, tag=tag)
plot_tsne(gen_mols, ref_mols, plots / f"{tag}_tsne_notext.png", bg_mols, show_text=False, tag=tag)
# also keep classic tag name for compatibility
plot_tsne(gen_mols, ref_mols, plots / "8RI2_all_actives_library_tsne.png", bg_mols, show_text=True, tag="8RI2_all_actives_library")
plot_tsne(gen_mols, ref_mols, plots / "8RI2_all_actives_library_tsne_notext.png", bg_mols, show_text=False, tag="8RI2_all_actives_library")
print("DONE", plots)
PY

echo "END=$(date -Iseconds)"
echo "ALL_DONE: $OUT"
cat "$OUT/library_manifest.json"
ls -la "$OUT/plots"
