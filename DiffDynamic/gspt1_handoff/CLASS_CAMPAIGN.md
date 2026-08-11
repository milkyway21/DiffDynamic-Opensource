# GSPT1 trusted-class campaign

This campaign replaces the mixed GSPT1 scaffold prior with three explicit
reference classes. It changes only scaffold generation. De novo generation is
unchanged.

## Fixed experiment design

| GPU | Class | Reference indices (zero-based) | Locked core | Allowed added heavy atoms | Joint exit allocation |
| --- | --- | --- | --- | --- | --- |
| 3 | `no_f` | 0-5, 7-11 | 18-atom CRBN core | 13, 14 | slot 0: 13 (weight 4); slot 0: 14 (weight 6); slot 0: 13 + slot 10: 1 (weight 1) |
| 4 | `f_main` | 14-16 | 19-atom CRBN core including F | 14, 15, 16 | slot 1: 14, 15, or 16 (equal weights) |
| 5 | `f_large` | 13 | 19-atom CRBN core including F | 24 | slot 1: 23 + slot 7: 1 |

Reference 6 (`*`) and reference 12 (`[U]`) are excluded from profile building
and scoring. Reference 15 contains Br. It remains in the size/allocation and
similarity reference set, but the checkpoint has no Br output class, so it is
excluded from the reachable-exact metric. Br is never substituted with another
element.

The original core coordinates and atom types are locked. Added coordinates and
types are not locked. The aggregate class element counts only initialize added
type logits; they are not restored during denoising. TargetDiff baseline refine
uses `start_t: 19`, which runs reverse timesteps 19 through 0 (20 steps), and
restores only the core prefix.

Initial added positions use the empirical joint exit allocation and a
pocket-aware placement pass. The preflight requires every allowed atom count and
joint allocation to appear and requires at least 1.5 A clearance from protein
heavy atoms.

## Leakage boundary

Generation profiles contain only class membership, added-heavy-atom counts,
joint counts by scaffold exit slot, and aggregate element/aromatic counts. They
do not contain a reference target-side graph, bond list, coordinates, SMILES,
or atom order. Known-reference SMILES and fingerprints are used only by the
post-generation audit. No generated graph is edited toward a reference.

## No-docking policy

Reconstruction always invokes the evaluator with `--vina-modes none`.
Three class jobs reconstruct concurrently with 30 workers each, for at most 90
evaluator workers. The campaign does not call Vina or use docking scores.

## Launch sequence

Run all commands from `/data/ye/DiffDynamic` in the `diffdynamic` environment.
The command without `--run` is CPU-only preparation and exits after writing
profiles, configs, a manifest, and `preflight.json`:

```bash
python scripts/gspt1_class_campaign.py \
  --root /data/zhang/Ye/DiffDynamic_outputs/hsvpol/molglue_ikzf2_gspt1/diffdynamic/gspt1_class_campaign \
  --samples 20
```

Start with one 20-molecule smoke round. This is the first command that is
allowed to use GPUs:

```bash
python scripts/gspt1_class_campaign.py --run \
  --root /data/zhang/Ye/DiffDynamic_outputs/hsvpol/molglue_ikzf2_gspt1/diffdynamic/gspt1_class_campaign \
  --gpus 3,4,5 --samples 20 --max-rounds 1 --hours 1 \
  --start-seed 20273000
```

Inspect `latest_summary.json`, all three `orchestrator.log` files, and each
parallel reconstruction manifest. Proceed only when all jobs report `ok`, no
chunk failed, each class produced SDF records, core retention is at least 90%,
and allowed-count retention is at least 90%. A low similarity in this smoke is
not a failure; it only validates the pipeline.

Then run a 200-molecule pilot as the second absolute round:

```bash
python scripts/gspt1_class_campaign.py --run --resume \
  --root /data/zhang/Ye/DiffDynamic_outputs/hsvpol/molglue_ikzf2_gspt1/diffdynamic/gspt1_class_campaign \
  --gpus 3,4,5 --samples 200 --max-rounds 2 --hours 2 \
  --start-seed 20273000
```

For the pilot, require all jobs and reconstruction chunks to succeed, core and
allowed-count retention to remain at least 90%, and at least one class to show a
clear side-chain similarity tail rather than only near-zero matches. Compare
`best_full_similarity` with the previous whole-molecule maximum 0.6575, but do
not require a 200-sample pilot to exceed a result obtained from 14,215 unique
molecules.

After the pilot passes, continue with 1,000 samples per class per round for up
to 10 hours:

```bash
python scripts/gspt1_class_campaign.py --run --resume \
  --root /data/zhang/Ye/DiffDynamic_outputs/hsvpol/molglue_ikzf2_gspt1/diffdynamic/gspt1_class_campaign \
  --gpus 3,4,5 --samples 1000 --hours 10 \
  --start-seed 20273000
```

`--max-rounds` is an absolute round limit when resuming. Omitting it means no
round limit. The loop stops early only when `exact_reachable_count` becomes
positive; otherwise it stops at the wall-clock deadline. `state.json` and
per-job done markers provide resume behavior.

## Main outputs

- `campaign_manifest.json`: input hashes, Git SHA, GPU mapping, and policy.
- `preflight.json`: discrete allocation and protein-clearance checks.
- `rounds/<class>/run_XXXX/`: generation and reconstruction for one class round.
- `rounds/<class>/audit/summary.json`: class-specific no-docking audit.
- `latest_summary.json`: latest combined class result.
- `EXACT_MATCH_FOUND`: written only for a reachable exact canonical-SMILES hit.
