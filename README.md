# DiffDynamic

DiffDynamic generates and evaluates 3D molecules in protein pockets using pretrained
diffusion models. It includes baseline/dynamic sampling, Prudent selection,
scaffold generation, optimization, batch orchestration, pocket analysis and a Web UI.

[中文说明](README.zh-CN.md) · [Architecture](docs/ARCHITECTURE.md) · [Medchem](docs/MEDCHEM.md) · [Publication](docs/PUBLICATION.md)

## Maintained code

```text
src/diffdynamic/
├── models/       Networks, probability, schedules and diffusion
├── data/         Protein/ligand parsing, graph containers and features
├── sampling/     Baseline, dynamic, Prudent, scaffold and optimization
├── chemistry/    Reconstruction, fragments, profiles and placement
├── evaluation/   Metrics, docking, isolated workers and result writers
├── analysis/     Pocket metrics, visualization and reusable plots
├── batch/        Resource selection, subprocess tasks and aggregation
├── io/           Paths, configuration, PT interchange and records
└── cli/          Public command adapters
```

`server/` holds API schemas/routes, job configuration, scheduling and runners.
`ui/` holds the workbench and landing-page assets. Old commands/imports are
compatibility adapters; algorithms are maintained once in the library.
Historical scripts/results are local-only under `local/archive` and
`local/research`. File-location maps and migration journals live in `local/inventory`.

## Install in the existing scientific environment

This refactor supports the validated Python 3.8 environment and does not upgrade
PyTorch, RDKit or GPU extensions. See `requirements-validated.txt` for the versions.

```bash
conda activate diffdynamic
python -m pip install --no-deps --no-build-isolation -e .
python tools/project.py weights --download
# Optional comparison checkpoint:
python tools/project.py weights --download --include-optional
```

Downloads are pinned and verified using `weights-manifest.json`. The default stays
`pretrained_models/pretrained_diffusion.pt`. Mismatching existing files are never replaced.
New machines need scientific dependencies plus OpenBabel/AutoDock Vina; geometric
extensions must match PyTorch/CUDA. The version list is a record, not an automatic migration.

## Commands

```bash
diffdynamic sample configs/sampling.yml --data_id 0 --device cuda:0
diffdynamic sample configs/sampling.yml --protein_path pocket.pdb --ligand_path ligand.sdf --device cuda:0
diffdynamic batch --start 0 --end 9 --gpus 0,1
diffdynamic evaluate outputs/result.pt --protein_root data/crossdocked_pocket10_test_only
diffdynamic extract outputs/result.pt
diffdynamic pocket --help
diffdynamic cascade --help
diffdynamic plot --help
```

Established commands such as `python scripts/sample_diffusion.py` and
`python evaluate_pt_with_correct_reconstruct.py` keep their arguments/defaults,
progress messages and result formats. Optional benchmarks require external baseline installations.

## Web

```bash
bash start_server.sh
# Explicit loopback-only local development:
DD_AUTH_MODE=off DD_HOST=127.0.0.1 bash start_server.sh
```

Default bind: `127.0.0.1:7860`. Platform authentication and reverse proxy prefixes
retain their behavior. Actual settings in `server/server_config.json` remain local;
see `server/server_config.example.json`. Explicit `DD_ROOT`, `DD_DATA`, `DD_OUTPUT`,
`DD_UPLOAD`, `DD_MODELS`, `DD_SAMPLING_YML` override saved paths.

The UI retains custom/dataset pockets, batch generation, scaffold placement,
optimization, evaluation/extraction, molecular downloads, progress/cancellation,
history/reruns, user isolation and administrative limits.

## Verify and prepare a release

```bash
python tools/checks.py
node ui/tests/check-contracts.cjs
python tools/project.py audit
python tools/project.py export --destination /path/to/new-release-candidate
```

Checks use temporary runtime storage. Export uses an explicit allowlist and builds
a flat repository candidate plus checksums. It never stages, commits, pushes,
changes remotes or includes private research/runtime data.

Checkpoint keys, historical PT graph names, strict loading, reconstruction policies,
metrics, result directories and API role contracts are regression-tested.
See `LICENSE`, source headers and `THIRD_PARTY_NOTICES.md` for retained attribution.
