# DiffDynamic Refactoring Project

## Goal

Refactor the DiffDynamic codebase into a clean, minimal, deployable system. Strip unnecessary features, keep only core functionality, and build a fully functional web frontend/backend as the **demonstration interface for the research paper**.

## Final Deliverable: Paper Demo Web UI

This is NOT a superficial wrapper. The web UI must be a **production-grade interactive platform** that allows anyone (reviewer, collaborator, user) to:

### Core Operations (from the browser)
- **Run DiffDynamic generation** — trigger `dynamic` mode batch sampling or `prudent` cumulative generation directly from the UI, configure parameters (GPU, data_id range, sampling config), and monitor progress in real-time.
- **Evaluate results** — launch pocket quality evaluation, view 8-dimension scores, compare across runs.
- **Extract & reconstruct molecules** — convert `.pt` outputs to SDF/Excel, perform scaffold reconstruction, download results.

### Data Layer
- **Read/browse all project data** — proteins, ligands, generated molecules, evaluation results, docking scores. Not just file listing; structured views with filtering, sorting, and search.
- **Result persistence** — every generation/evaluation run is recorded with full metadata (config snapshot, timestamps, parameters, outputs). The system maintains a complete experiment database.

### History & Reproducibility
- **Operation history log** — every action (generation, evaluation, extraction) is logged with: who triggered it, when, with what parameters, what outputs were produced. Full audit trail.
- **Run comparison** — side-by-side comparison of different runs' metrics, molecules, docking scores.
- **Re-run from history** — pick any past run, view its config, re-execute or modify and re-run.

### Architecture Requirements
- **Real backend** — not Gradio-only. A proper API layer (FastAPI or similar) that the UI talks to. The backend manages GPU job scheduling, file I/O, database, and exposes clean endpoints.
- **Database** — SQLite or similar for run records, molecule catalogs, evaluation results. Not just reading files from disk each time.
- **Async job system** — long-running tasks (generation, evaluation) run as background jobs with status tracking, not blocking the UI.
- **Multi-user** — multiple people can use it simultaneously without conflicts. Job queue, not first-come blocking.
- **Network-accessible** — bind to 0.0.0.0, serve on a port, usable by paper reviewers and collaborators on the network.

## Task 1: Core Pipeline Preservation

Keep these entry points working in `/data/ye/DiffDynamic/`:

- **`batch_sampleandeval_parallel.py`** — `dynamic` mode fast generation. Must preserve GPU-parallel batch sampling via `scripts/sample_diffusion.py` with `configs/sampling.yml`.
- **`run_prudent_generations.py`** — Prudent cumulative generation. Multi-round generate → filter → dock → seed upgrade loop.
- **`evaluate_pocket_quality.py`** — 8-dimension pocket quality assessment (Vina, clustering, LE, QED/SA/Lipinski/PAINS, completeness, etc.).
- **`extract_pt_to_sdf_excel.py`** / **`sampling.yml` scaffold section** — Molecular extraction from `.pt` files and scaffold reconstruction.

### Supporting modules that must remain functional:
- `scripts/sample_diffusion.py` — core diffusion sampler
- `scripts/evaluate_mol_from_meta_full.py` / `evaluate_pt_with_correct_reconstruct.py` — evaluation pipeline
- `configs/sampling.yml` — all config knobs for dynamic/prudent/scaffold modes
- `models/` — EGNN, UniTransformer, MolOptScoreModel
- `utils/` — data loading, transforms, reconstruct, docking helpers
- `datasets/` — protein/lidand data handling

### What can be removed:
- Experimental scripts (parameter sweeps, benchmark comparisons, chart generators)
- Duplicate/redundant analysis scripts
- Backup directories (`hsvpol.bak_before_symlink`, `dd0120`)
- One-off diagnostic/visualization scripts not part of the core pipeline

## Task 2: Build Web Frontend/Backend

Source reference: `/data/ye/diffdynamic-ui/diffdynamic/app.py` (existing Gradio prototype) and its dependencies.

Existing code in `diffdynamic-ui/diffdynamic/` to learn from:
- `app.py` — Gradio web interface (prototype, to be evolved)
- `cli_batch.py`, `cli_eval.py` — CLI wrappers
- `config/` — paths, defaults
- `io/` — pt_loader, excel_recorder
- `mol/` — reconstruct, preparation, scoring
- `eval/` — pt_file, single_mol, vina_task
- `batch/` — batch processing
- `records/` — catalog, result scanning
- `upstream/` — copies of DiffDynamic scripts (to be eliminated)

### Architecture to build:

```
┌─────────────────────────────────────────────────┐
│                  Web Frontend                    │
│         (Gradio / or lightweight HTML+JS)        │
├─────────────────────────────────────────────────┤
│              API Layer (FastAPI)                  │
│  /api/generate  /api/evaluate  /api/molecules    │
│  /api/history   /api/jobs       /api/config      │
├─────────────────────────────────────────────────┤
│           Backend Services                       │
│  JobScheduler │ DataManager │ HistoryLogger      │
├─────────────────────────────────────────────────┤
│  SQLite (runs, molecules, evals) │ File Storage  │
└─────────────────────────────────────────────────┘
```

### Key modules to build:
1. **`server/api.py`** — FastAPI app with REST endpoints for all operations
2. **`server/jobs.py`** — Async job queue (threading or asyncio) for GPU tasks
3. **`server/database.py`** — SQLite ORM (SQLAlchemy or raw sqlite3) for run records, molecules, evaluations
4. **`server/history.py`** — Operation audit log (who, when, what, outputs)
5. **`server/data_manager.py`** — Structured access to proteins, ligands, .pt results, SDF molecules
6. **`ui/`** — Frontend that talks to the API (Gradio as component library, or custom HTML)
7. **`server/config.py`** — Runtime config (GPU list, paths, defaults) separate from sampling.yml

### Target:
- Self-contained web service deployable on any server with GPUs
- Remove `upstream/` copies; import directly from the refactored DiffDynamic core
- Every operation is recorded in the database with full metadata
- Real-time job status updates (WebSocket or polling)
- Multi-user safe: concurrent requests handled via job queue

## Task 3: Code Cleanup

- Remove dead imports and unused functions
- Consolidate duplicate logic between DiffDynamic and diffdynamic-ui
- Ensure `configs/sampling.yml` remains the single source of truth for generation parameters
- Keep `requirements.txt` / dependency list minimal and accurate

## Environment

- **Conda env**: `diffdynamic` at `/home/user/anaconda3/envs/diffdynamic` (Python 3.8, PyTorch, RDKit, Gradio 4.44.1)
- **Activation**: `conda activate diffdynamic`
- **Refactoring tools** (installed globally via pipx):
  - `ruff` — lint + format (also in conda env)
  - `vulture` — dead code detection
  - `autoflake` — remove unused imports/variables
  - `pipreqs` — generate requirements.txt from imports
- **Run all scripts with**: `conda run -n diffdynamic python3 <script>`

## Constraints

- Do NOT break the three core entry points (batch dynamic, prudent, evaluate_pocket_quality)
- Preserve all config options in sampling.yml — they control generation behavior
- The web UI must be a real application, not a thin wrapper — database-backed, multi-user, with proper API
- All operations must be logged and reproducible from the UI
- Bind to 0.0.0.0, accessible to paper reviewers and collaborators on the network

## Testing Rules (MUST FOLLOW)

- **测试生成用 `batch_size=5`**，不要用默认的 100
- **测试评估用 `max_samples=5`**，只评估 5 个分子
- **Vina 对接超时 20 秒**（`vina_timeout=20`），超时自动跳过
- **自动链式执行**：生成后自动评估+提取，都用 `max_samples=5`
- **永远不删数据库**：`diffdynamic.db` 必须持久化，重启保留历史记录，**绝对不能 rm diffdynamic.db***
- API 参数：`{"mode":"dynamic","data_id":0,"batch_size":5,"auto_evaluate":true,"auto_extract":true}`
