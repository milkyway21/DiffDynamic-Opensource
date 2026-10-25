# DiffDynamic

DiffDynamic generates and evaluates 3D molecules in protein pockets using pretrained
diffusion models. It includes baseline/dynamic sampling, Prudent selection,
scaffold generation, optimization, batch orchestration, pocket analysis and a Web UI.

[中文说明](README.zh-CN.md) · [Project guide](PROJECT.md) · [Agent guide](AGENTS.md) · [Architecture](docs/ARCHITECTURE.md) · [Medchem](docs/MEDCHEM.md) · [Publication](docs/PUBLICATION.md)

DiffDynamic is an inference framework: it ships no training pipeline and runs the
pretrained diffusion checkpoints fetched by `tools/project.py weights --download`.

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

## Set up from scratch

The validated environment is Python 3.8 with PyTorch 2.4.1 (CUDA 11.8) and a CUDA
GPU. Install the torch stack first — the graph-extension wheels must match the
PyTorch/CUDA build — then the pinned dependencies, then the package:

```bash
conda create -n diffdynamic python=3.8 -y
conda activate diffdynamic

# 1. PyTorch (CUDA 11.8 build)
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118

# 2. Graph extensions — wheels must match torch 2.4.1 + cu118
pip install torch_geometric==2.6.1
pip install torch_scatter==2.1.2 torch_sparse==0.6.18 \
    -f https://data.pyg.org/whl/torch-2.4.1+cu118.html

# 3. OpenBabel Python bindings (used by reconstruction and docking)
conda install -c conda-forge openbabel=3.1.0 -y

# 4. Remaining pinned dependencies, then the package
pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .

# 5. Pretrained checkpoints (URL + SHA256 pinned in weights-manifest.json)
python tools/project.py weights --download
# Optional comparison checkpoint (GlintDM):
python tools/project.py weights --download --include-optional
```

`requirements-validated.txt` is the exact record of the validated environment and is
not an automatic migration path. Docking uses the `vina` Python package; the optional
QVina benchmark additionally needs an external `qvina2` binary. Downloaded weights
never replace an existing file whose checksum differs.

Machines that already run the validated environment only need the editable install
and the weights:

```bash
conda activate diffdynamic
python -m pip install --no-deps --no-build-isolation -e .
python tools/project.py weights --download
```

## Data

Sampling supports two modes; only the first needs no dataset:

- **Custom pocket:** `--protein_path pocket.pdb --ligand_path ligand.sdf` works
  right after setup with any protein/ligand pair. The repository ships a ready-made
  pair (landing-page demo structures) for a first run:

  ```bash
  diffdynamic sample configs/sampling.yml \
      --protein_path ui/static/data/landing/protein.pdb \
      --ligand_path ui/static/data/landing/reference-ligand.sdf --device cuda:0
  ```

- **Test-set pockets:** `--data_id N` indexes the CrossDocked test split (100
  pockets). The checkpoint stores the expected locations, so the files must be
  placed as:

  ```text
  data/
  ├── crossdocked_pocket10_test_only/                    # pocket PDBs + ligand SDFs
  │   ├── index.pkl                                      # (pocket, ligand, ...) entries
  │   └── ..._processed.lmdb                             # built automatically on first use
  └── crossdocked_pocket10_test_only_split.pt            # subset name -> data ids
  ```

  The dataset pipeline is adopted from **TargetDiff**, the upstream diffusion
  project (see `THIRD_PARTY_NOTICES.md`). The downloads below are hosted by the
  TargetDiff / CrossDocked2020 authors and follow their license terms
  (research use); DiffDynamic only reuses them:

  | Resource | Link | What it provides |
  |---|---|---|
  | TargetDiff repository | https://github.com/guanjq/targetdiff | Upstream code and `scripts/data_preparation/` used to build the pocket10 dataset |
  | TargetDiff processed data (Google Drive) | https://drive.google.com/drive/folders/1j21cc7-97TedKh_El5E34yI8o5ckI7eK?usp=share_link | `crossdocked_pocket10_pose_split.pt` (train 99,990 / test 100 split), `crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb` (full training LMDB), `test_set.zip` (test structures), `split_by_name.pt` |
  | Raw CrossDocked2020 v1.1 | https://bits.csb.pitt.edu/files/crossdock2020/ | Original dataset; only needed to rebuild everything from scratch |

  To reproduce the layout above: put the 100 test pockets (structures in
  TargetDiff's `test_set.zip`, pocket/ligand layout as produced by their
  `extract_pockets.py`) under `data/crossdocked_pocket10_test_only/` with an
  `index.pkl`, and copy their pose split file to
  `data/crossdocked_pocket10_test_only_split.pt` (only its `test` subset is used
  against the test-only directory). DiffDynamic is inference-only: the training
  LMDB and raw CrossDocked2020 downloads are referenced for completeness and are
  not required. The LMDB cache is built on first use by walking every index
  entry, so the first dataset run takes noticeably longer. Evaluation passes the
  same directory explicitly: `--protein_root data/crossdocked_pocket10_test_only`.

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
# First run: create the local config from the example, then edit
# gpu_ids / default_gpu_id / max_concurrent_jobs for your machine.
cp server/server_config.example.json server/server_config.json

bash start_server.sh
# Explicit loopback-only local development:
DD_AUTH_MODE=off DD_HOST=127.0.0.1 bash start_server.sh
```

Authentication modes:

- `DD_AUTH_MODE=platform` (default) delegates sign-in to the external platform
  gateway that fronts the production deployment (URL overridable via
  `DD_PLATFORM_SESSION_URL`) and isolates each user's uploads and outputs.
  It requires that gateway to be reachable.
- `DD_AUTH_MODE=off` is for single-user local development only and must bind to
  loopback; visiting the page sets an internal cookie and
  `server/.internal_api_key` is generated on first start.

The server must run as a single process (the in-memory scheduler, queue caps and
rate limiter do not survive multiple uvicorn workers); scale through `gpu_ids` and
`max_concurrent_jobs` instead. Per-user caps live in `server_config.json`:
`max_generate_per_day` / `max_eval_per_day` (0 disables, administrators are exempt)
and upload/queue limits; an administrator can tune them at runtime via
`PUT /api/config`. For a single-user local setup, either set the daily caps to 0 or
launch with `DD_ADMIN_USERS=web_ui`.

Default bind: `127.0.0.1:7860`. Reverse proxy prefixes retain their behavior.
Actual settings in `server/server_config.json` remain local; see
`server/server_config.example.json`. Explicit `DD_ROOT`, `DD_DATA`, `DD_OUTPUT`,
`DD_UPLOAD`, `DD_MODELS`, `DD_SAMPLING_YML` override saved paths.

The UI retains custom/dataset pockets, batch generation, scaffold placement,
optimization, evaluation/extraction, molecular downloads, progress/cancellation,
history/reruns, user isolation and administrative limits.

## Verify and prepare a release

```bash
python tools/checks.py
node --experimental-vm-modules ui/tests/check-contracts.cjs
python tools/project.py audit
python tools/project.py export --destination /path/to/new-release-candidate
```

Checks use temporary runtime storage. Export uses an explicit allowlist and builds
a flat repository candidate plus checksums. It never stages, commits, pushes,
changes remotes or includes private research/runtime data.

Checkpoint keys, historical PT graph names, strict loading, reconstruction policies,
metrics, result directories and API role contracts are regression-tested.
See `LICENSE`, source headers and `THIRD_PARTY_NOTICES.md` for retained attribution.
