# GSPT1 Scaffold RePaint v3 Handoff

Status: prepared only. No sampling or reconstruction process was started.

## Launch

```bash
cd /data/ye/DiffDynamic
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate diffdynamic
export PYTHONPATH="/data/ye/DiffDynamic${PYTHONPATH:+:$PYTHONPATH}"
python -u scripts/gspt1_class_campaign.py \
  --run --hours 10 --samples 1000 --gpus 3,4,5
```

Do not pass `--resume` for the first v3 launch. The dedicated output root is:

`/data/zhang/Ye/DiffDynamic_outputs/hsvpol/molglue_ikzf2_gspt1/diffdynamic/gspt1_scaffold_repaint_v3`

## GPU Split

- GPU 3: no-F core, `n_extra` exactly 13 or 14.
- GPU 4: F-main core, `n_extra` exactly 14 or 16.
- GPU 5: F-large core, `n_extra` exactly 24.
- Reference 15 is excluded because its Br atom is outside the checkpoint vocabulary.

## Diffusion And Placement

- This behavior is isolated behind `sample.scaffold.enable=true` and
  `sample.scaffold.mode=dynamic_locked`; de novo dispatch is unchanged.
- The clean scaffold plus site-placed extra atoms is forward-diffused to
  `q(x_999 | x_0)` before the first reverse model call.
- RePaint injects the scaffold at the matching noise level throughout reverse
  diffusion. Original scaffold positions and types are hard constrained.
- Extra positions use the reference exit-site allocation profile but no target
  graph, atom order, bond, coordinate, or SMILES is copied.
- Extra atom types use a randomly permuted per-molecule element/aromatic-count
  multiset. The RePaint type strength is 0.2, so identities remain model-driven.
- DiffDynamic refine is capped at 30 evaluations sampled across the complete
  refine schedule, including `t=0`. TargetDiff baseline repair also runs 30
  inclusive steps (`t=29..0`).

## Reconstruction And Audit

- Reconstruction is automatic after each generation job.
- Each job uses 30 one-thread evaluator workers; three concurrent jobs can use
  90 CPU cores total.
- Vina is disabled explicitly with `--vina-modes none`.
- Output keeps per-job directories, so timestamp-named SDF files cannot collide.
- The campaign audits each class only against its reachable non-Br references.

Prepared configs and profiles are under the output root in `configs/` and
`profiles/`. The manifest is `campaign_manifest.json`; runtime progress is
written to `state.json`, `job_results.jsonl`, and `latest_summary.json`.
