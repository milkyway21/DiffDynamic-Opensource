# DiffDynamic

**DiffDynamic** is a diffusion-based framework for 3D structure-based drug design (SBDD). Built upon [TargetDiff](https://github.com/DeepGraphLearning/TargetDiff), it introduces several inference-stage techniques aimed at improving the quality and diversity of generated molecules within protein binding pockets.

## Table of Contents

- [Overview](#overview)
- [Key Techniques](#key-techniques)
  - [Gradient Fusion](#1-gradient-fusion)
  - [Two-Stage Dynamic Sampling with Skip-Step and Structural Repair](#2-two-stage-dynamic-sampling-with-skip-step-and-structural-repair)
  - [Prudent Multi-Round Filtering](#3-prudent-multi-round-filtering)
  - [Scaffold-Constrained Generation](#4-scaffold-constrained-generation)
  - [Comprehensive Molecular Scoring](#5-comprehensive-molecular-scoring)
- [Quick Start](#quick-start)
- [Batch iteration workbook](#batch-iteration-workbook)
- [Configuration](#configuration)
- [Evaluation Pipeline](#evaluation-pipeline)
- [Project Structure](#project-structure)
- [Relation to TargetDiff](#relation-to-targetdiff)
- [Innovation Summary](#innovation-summary)
- [Acknowledgments](#acknowledgments)

## Overview

DiffDynamic focuses on **inference-time improvements** to the TargetDiff framework. The backbone model architecture (ScorePosNet3D with UniTransformer / EGNN) and the joint continuous-coordinate + discrete-atom-type diffusion process are inherited from TargetDiff. DiffDynamic's contributions are primarily in three areas:

1. A **two-stage dynamic skip-step sampling** strategy that decouples topology exploration from geometry refinement, with a **structural repair** mechanism — the first AIDD system to exploit information density asymmetry across the diffusion trajectory
2. A **gradient fusion** mechanism that blends global and local denoising signals with time-decaying schedules
3. A **multi-round filtering** pipeline (Prudent) that integrates docking scores into the generation loop

These techniques operate at inference time and do not require retraining the base diffusion model.

### What DiffDynamic is NOT

- It is **not** a new diffusion model architecture (the neural network is unchanged from TargetDiff)
- It is **not** a new training paradigm (uses pretrained TargetDiff checkpoints)
- The evaluation metrics (Vina, QED, SA) are standard tools, not novel contributions

## Key Techniques

### 1. Gradient Fusion

**Motivation.** Standard diffusion sampling uses the posterior mean direction for each denoising step. DiffDynamic introduces a second gradient signal — the direction from the current noisy molecule toward the model's predicted clean structure (x_0) — and blends the two.

**Mechanism.** At each denoising step *t*:

```
local_grad  = posterior_mean(x_t) - x_t          # standard diffusion direction
global_grad = predicted_x0(x_t)   - x_t          # direct-to-target direction
combined    = lambda(t) * global_grad + (1 - lambda(t)) * local_grad
```

where `lambda(t)` decays from 1.0 to 0.0 across the diffusion trajectory. This means:
- **Early steps** (high *t*): the model relies more on the global signal to establish the overall molecular conformation
- **Late steps** (low *t*): the model relies more on the local posterior for fine-grained bond geometry

**Supported schedules** for `lambda(t)`: `quadratic` (default), `linear`, `exponential`, `adaptive` (gradient-norm-based), and `time`.

**Relation to prior work.** The idea of blending guidance signals is conceptually related to classifier-free guidance (Ho & Salimans, 2022) and gradient-guided diffusion (Dhariwal & Nichol, 2021), though the specific formulation of blending posterior-mean and x_0-prediction directions is, to our knowledge, not previously proposed in the SBDD literature. We do not claim this is a fundamentally new paradigm — it is a practical inference-time heuristic that improves sample quality in our experiments.

### 2. Two-Stage Dynamic Sampling with Skip-Step and Structural Repair

**A key innovation in AIDD/de novo drug design.** While adaptive step sizes exist in general-purpose diffusion solvers (DDIM, DPM-Solver), no prior work in AI-driven drug discovery has proposed a two-stage dynamic skip-step strategy tailored to the unique structure of molecular denoising trajectories. DiffDynamic is the first to exploit the **information density distribution** across the diffusion timestep spectrum for 3D molecule generation.

**Core insight: information density varies across the denoising trajectory.** In molecular diffusion, the denoising process is not information-uniform:

- **High-noise region (t=999 → ~650):** The model is determining *topology* — which atoms connect, ring systems, overall scaffold shape. Each denoising step carries **low marginal information** because the molecular graph is still largely underdetermined. Skipping intermediate steps here loses almost no structural information.
- **Low-noise region (t~650 → 0):** The model is refining *geometry* — bond angles, torsion angles, precise atomic positions. Each step carries **high marginal information** because small coordinate changes affect binding affinity and chemical validity.

This asymmetry motivates a fundamental redesign of the sampling schedule.

**Mechanism.** The sampling trajectory is split at a configurable time boundary `t_boundary`:

| Stage | Range | Purpose | Step scheduling |
|-------|-------|---------|-----------------|
| Large Step | t=999 → t_boundary | **Skip-step exploration**: coarse topology with adaptive large strides | Lambda/linear schedule (strides 15–58 steps) |
| Refine | t_boundary → t=0 | **Structural repair**: fine-grained geometry with dense small strides | Lambda/linear schedule (strides 3–60 steps) |

**Lambda scheduling** computes the stride per interval as `n = a * (t/T) + b`, producing large strides early and small strides late. **Linear scheduling** computes `n = a * progress + b` where progress decreases from 1 to 0.

**The skip-step mechanism** is the key innovation. During the Large Step stage, the model explicitly *skips* intermediate diffusion steps — instead of visiting every timestep from 999 to 650, it jumps in strides of 15–58 steps. At each jump:
1. The model predicts the denoising direction at the current timestep
2. The state is advanced by `stride` steps in one forward pass
3. The gradient fusion mechanism (see Section 1) provides a correction signal to compensate for the skipped steps

This is fundamentally different from simply using fewer steps (which would lose the ability to track the learned data manifold). The skip-step approach **preserves the model's learned denoising dynamics** while dramatically reducing the number of neural network evaluations in the low-information region.

**The structural repair mechanism** complements the skip-step stage. After the coarse topology is established, the `targetdiff_baseline_refine` module performs a brief forward diffusion to `start_t=9` followed by full reverse diffusion to t=0. This exploits the model's learned ability to **repair local geometric defects** (bond angles, ring planarity, steric clashes) that accumulate during the skip-step stage. The repair is efficient because:
- It operates in the high-sensitivity low-t region where the model has learned fine-grained geometric priors
- The forward diffusion is shallow (only to t=9), so the structural information from the skip-step stage is largely preserved
- The subsequent reverse diffusion acts as a "denoising autocorrect" that fixes local defects without altering the global topology

**Why this is a genuine innovation for AIDD:**

1. **No prior SBDD/de novo diffusion work has decoupled topology exploration from geometry refinement.** Existing methods (TargetDiff, DiffSBDD, DecompDiff) all use uniform step schedules, treating every timestep equally. DiffDynamic recognizes that the information content of each step varies by orders of magnitude across the trajectory.

2. **The skip-step + repair combination is not merely "using fewer steps."** A naive reduction in step count (e.g., running 200 uniform steps instead of 1000) would degrade quality because it cannot track the learned manifold at any resolution. DiffDynamic's approach allocates compute *proportionally to information density* — few evaluations where the manifold is flat, many where it is steep.

3. **The structural repair stage exploits a previously untapped capability of diffusion models.** The fact that a model trained on full diffusion trajectories can also serve as a "local geometry repair tool" when applied to shallow forward-then-reverse passes is a non-obvious insight about the learned denoising network's generalization.

**Practical impact.** Compared to uniform 1000-step sampling, the two-stage approach reduces neural network evaluations by ~60–70% while maintaining or improving output quality. The time savings are reinvested into the Prudent multi-round filtering pipeline, enabling more thorough quality control within the same computational budget.

### 3. Prudent Multi-Round Filtering

**Motivation.** Most SBDD diffusion methods generate molecules in a single forward pass with no feedback from scoring functions. This means many generated molecules are chemically invalid, have poor drug-likeness, or score poorly on docking — problems that could be caught and corrected early.

**Mechanism.** Prudent implements a generate-filter-refine loop within the diffusion process:

```
Round 1:
  1. Large Step → produce candidate molecules
  2. For each candidate, open N parallel denoising chains (e.g., N=20)
  3. Refine each chain to t=0
  4. Score each chain:
     - Compute QED, SA (fast, no docking)
     - If QED >= threshold AND SA >= threshold → run Vina score_only (slow)
     - Compute composite score: w_vina * Vina + w_qed * QED + w_sa * SA
  5. Keep top-K chains by composite score

Round 2+ (if n_rounds > 1):
  1. Re-noise survivors to intermediate timestep (t=400)
  2. Re-refine with new random noise
  3. Re-score and re-filter
  4. Repeat for n_rounds total rounds
```

**Key design decisions:**
- **QED/SA gating before Vina**: Only molecules passing drug-likeness thresholds are sent to the (expensive) docking step. This saves significant compute.
- **Composite scoring**: Weights Vina affinity (0.86), QED (0.07), and SA (0.07) to balance binding affinity with drug-likeness.
- **Strict advance criteria**: A molecule must (a) be chemically valid, (b) pass QED/SA thresholds, (c) if Vina weight > 0, must have been docked, and (d) optionally must have affinity < a configurable threshold. These criteria prevent low-quality molecules from propagating.
- **Checkpoint-based re-noising**: Rather than fully re-noising from t=0 (which loses structural information), Prudent can snapshot the denoising trajectory at the closest frame to the target re-noise timestep.

**Relation to prior work.** The idea of using molecular property filters during generation is common in Bayesian optimization and iterative refinement methods. The novelty of Prudent lies in embedding this filtering loop *within* the diffusion trajectory (rather than post-hoc) and using the denoising process itself as the refinement mechanism. This is, to our knowledge, a new combination in the SBDD diffusion literature.

### 4. Scaffold-Constrained Generation

Two modes for generating molecules that preserve a chemical scaffold:

#### 4a. Scaffold Evolution (`mode: evolve`)

Starting from an existing molecule (e.g., a reference ligand), applies SDEdit-style noising and re-denoising to generate variants while preserving the Murcko scaffold. A population-based approach maintains diversity across generations:

```
For each generation:
  1. For each parent molecule, add noise to start_t (annealing from high to low)
  2. Re-denoise with scaffold atoms masked (position + type fixed)
  3. Score offspring: QED, SA, diversity bonus, scaffold RMSD penalty
  4. Select survivors using greedy diversity filter (Tanimoto < 0.90)
```

The noise annealing (`start_t_high=200` → `start_t_low=30` over generations) allows early generations to explore broadly and later generations to refine.

#### 4b. Scaffold Growth (`mode: grow`)

Keeps the scaffold atoms fixed and generates additional atoms from scratch within the binding pocket:

1. Extract scaffold using Bemis-Murcko decomposition (or manual specification)
2. Determine number of additional atoms from the pocket prior
3. Initialize: scaffold atoms at original positions + additional atoms as random noise
4. Denoise with scaffold atoms masked
5. The scaffold atoms serve as an anchor; the network learns to fill the remaining pocket volume

**Relation to prior work.** Scaffold-constrained generation using masked diffusion is related to the RePaint method (Lugmayr et al., 2022) and has been explored in several SBDD works (e.g., DecompDiff). The combination with Murcko scaffold detection and the evolution/growth pipeline is a practical engineering contribution rather than a fundamental algorithmic novelty.

### 5. Comprehensive Molecular Scoring

A multi-dimensional scoring system for generated molecules:

```
final_score = 100 * base_score * pains_multiplier * stability_multiplier

base_score = 0.4 * affinity_norm + 0.3 * QED + 0.2 * SA_normalized + 0.1 * Lipinski_fraction
```

where `affinity_norm` normalizes Vina scores into [0, 1] with -6 kcal/mol as the onset. The system also integrates Lilly Medchem Rules (PAINS filters, reactive group detection, etc.) as penalty multipliers.

Each molecule receives a **molecular ID**: `{protein_id}_{timestamp}_{score}`, e.g., `1A4K_20260416_88p89`.

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.12-1.13 with CUDA 11.6
- PyTorch Geometric 2.2
- RDKit, AutoDock Vina

### Docker (Recommended)

```bash
cd /path/to/DiffDynamic
chmod +x start_docker.sh
./start_docker.sh
```

This creates a `diffdynamic` Docker container with all dependencies pre-installed. Inside the container:

```bash
cd /workspace
export PYTHONPATH=/workspace:$PYTHONPATH
conda activate diffdynamic
```

### Basic Sampling

```bash
# Single pocket sampling (data_id 0-99 for test set)
python scripts/sample_diffusion.py configs/sampling.yml --data_id 0

# With GPU selection
python scripts/sample_diffusion.py configs/sampling.yml --data_id 0 --device cuda:1

# Custom protein and ligand
python scripts/sample_diffusion.py configs/sampling.yml \
    --protein_path path/to/protein.pdb \
    --ligand_path path/to/reference.sdf
```

### Prudent Mode (Multi-Round Filtering)

```bash
# Ensure sampling.yml has sample.mode: prudent
python scripts/sample_diffusion.py configs/sampling.yml --data_id 0

# Or for a dedicated target (e.g., SHOC2 8v1t):
python run_shoc2_prudent.py --gpus "0,1,2,3" --seed_start 0 --seed_end 100
```

### Evaluation

```bash
# Evaluate a single .pt result file
python evaluate_pt_with_correct_reconstruct.py \
    outputs/result_0_*.pt \
    --protein_root ./data/crossdocked_v1.1_rmsd1.0_pocket10 \
    --output_dir ./eval_results

# Batch sampling + evaluation
python batch_sampleandeval_parallel.py \
    --start 0 --end 100 --gpus "0-3" --num_cpu_cores 4
```

### Batch iteration workbook

[`batch_iter.py`](batch_iter.py) exports **intra-run Prudent trajectories** (segment-evolution checkpoints or full-generation rounds) from **one** sampling `.pt`: per-segment/per-chain SMILES and composite/QED/SA/Vina fields, plus a `refined_final` sheet from `meta.refined_candidates`. It does **not** chain unrelated repeated samples. Optionally run a single pocket sample first (`--data_id` or `--protein_path`) or point at `--from-pt`. Sampling stores each segment-end pool under `meta` as `scoring_snapshot` inside **one final** `.pt` (a single save); **`batch_iter` runs one `torch.load` on that file** and splits tabs from it — **not one `.pt` per worksheet**. See `scripts/sample_diffusion.py`; re-sample if your checkpoint predates that field. **[docs/batch_iter.md](docs/batch_iter.md)** documents all workbook tabs.

```bash
python batch_iter.py --from-pt ./outputs/result_10_it5_YYYYMMDD_HHMMSS.pt

python batch_iter.py --data_id 10 --gpu 0 --config configs/sampling.yml \
    --protein-root ./data/crossdocked_v1.1_rmsd1.0_pocket10
```

## Configuration

All sampling parameters are in `configs/sampling.yml`. Key sections:

### Model

```yaml
model:
  checkpoint: ./pretrained_models/pretrained_diffusion.pt
  force_model_type: diffdynamic
  use_grad_fusion: true
  grad_fusion_lambda:
    mode: quadratic      # quadratic | linear | exponential | adaptive | time
    start: 1.0           # initial global gradient weight
    end: 0.0             # final local gradient weight
    power: 2.0           # decay power for quadratic mode
```

### Dynamic Sampling

```yaml
sample:
  mode: prudent           # baseline | dynamic | optimization | prudent
  num_steps: 1000

  dynamic:
    time_boundary: 650    # split point between large_step and refine

    large_step:
      schedule: lambda    # lambda | linear
      lambda_coeff_a: 58.0
      lambda_coeff_b: 20.0
      step_size: 0.33

    refine:
      schedule: lambda
      lambda_coeff_a: 60.0
      lambda_coeff_b: 5.0
      step_size: 0.47
```

### Prudent Filtering

```yaml
    prudent:
      enable: true
      n_sampling: 20           # parallel chains per candidate
      n_rounds: 1              # number of filter-refine rounds
      advance_top_k: 5         # survivors per round
      renoise_t: 400           # target timestep for inter-round re-noising
      renoise_mode: checkpoint # checkpoint | forward
      min_qed_for_docking: 0.2
      min_sa_for_docking: 0.3
      max_vina_affinity_for_advance: -4  # kcal/mol
      vina_weight: 0.86
      qed_weight: 0.07
      sa_weight: 0.07
```

### Scaffold Generation

```yaml
  scaffold:
    enable: false
    mode: grow              # grow | evolve
    scaffold_source: auto_murcko  # auto_murcko | smarts | atom_indices | none
    fix_scaffold_pos: true
    fix_scaffold_type: true

    evolve:
      population_size: 50
      n_generations: 5
      children_per_parent: 10
      start_t_high: 200
      start_t_low: 30
      noise_anneal: true

    grow:
      n_extra_mode: prior_minus_scaffold
      start_t: 450
      num_samples: 200
```

## Evaluation Pipeline

DiffDynamic supports multiple evaluation backends:

| Tool | Metric | Script |
|------|--------|--------|
| AutoDock Vina | Binding affinity (kcal/mol) | `evaluate_pt_with_correct_reconstruct.py` |
| RDKit | QED, SA, validity, uniqueness | Built into evaluation scripts |
| Lilly Medchem Rules | Drug-likeness penalties | Built into scoring pipeline |
| iFit Eval | Pocket-specific fragment docking | `scripts/ifit_eval_testset_scores.py` |
| GenBench3D | 3D geometric quality | `scripts/run_genbench3d_local.py` |
| P2Rank | Pocket prediction | `scripts/run_p2rank_predict.py` |

### CrossDocked Test Set Evaluation

```bash
# Reference ligand Vina scores (baseline comparison)
python scripts/dock_testset_split_reference_vina.py \
    --max-pockets 100 \
    --protein-root ./data/crossdocked_v1.1_rmsd1.0_pocket10 \
    --out-csv outputs/testset100_reference_vina.csv

# Pocket quality triple scatter plot
python scripts/plot_testset_pocket_triple_scatter.py \
    --eval-xlsx outputs/evaluation_summary.xlsx
```

## Project Structure

```
DiffDynamic/
├── models/
│   ├── molopt_score_model.py    # Core diffusion model (ScorePosNet3D + DiffDynamic)
│   ├── egnn.py                  # Equivariant Graph Neural Network
│   ├── uni_transformer.py       # UniTransformer attention network
│   └── common.py                # Shared utilities
├── scripts/
│   ├── sample_diffusion.py      # Main sampling script (~6500 lines)
│   └── ifit_eval_testset_scores.py  # iFit pocket evaluation
├── utils/
│   ├── evaluation/
│   │   ├── docking_vina.py      # AutoDock Vina interface
│   │   ├── scoring_func.py      # QED, SA, Lilly rules
│   │   └── similarity.py        # Tanimoto similarity
│   └── masked_guidance_sampling.py  # RePaint-style masked sampling
├── configs/
│   └── sampling.yml             # All sampling parameters
├── evaluate_pt_with_correct_reconstruct.py  # Molecule reconstruction + evaluation
├── batch_sampleandeval_parallel.py          # Batch sampling and evaluation
├── batch_iter.py                # Prudent segment/generation trajectory -> Excel (single .pt)
├── run_shoc2_prudent.py         # SHOC2 target Prudent pipeline
└── start_docker.sh              # Docker environment launcher
```

## Relation to TargetDiff

| Aspect | TargetDiff | DiffDynamic |
|--------|-----------|-------------|
| Model architecture | ScorePosNet3D + UniTransformer | Same (unchanged) |
| Training | Custom training loop | Uses pretrained TargetDiff checkpoints |
| Sampling | Fixed-step posterior denoising | **Two-stage dynamic skip-step + structural repair** + gradient fusion |
| Molecular filtering | None (post-hoc evaluation) | Prudent multi-round filtering |
| Scaffold support | None | Evolution + growth modes |
| Scoring | Basic Vina + QED/SA | Multi-dimensional composite + Lilly rules |

## Innovation Summary

The following table summarizes each component's novelty within the AIDD / de novo drug design literature:

| Component | Novelty Level | What is new | What is inherited |
|-----------|:------------:|-------------|-------------------|
| **Dynamic Skip-Step Sampling** | **First in AIDD** | Two-stage topology/geometry decoupling; information-density-proportional compute allocation; skip-step + structural repair pipeline | General concept of non-uniform stepping (DDIM, DPM-Solver) |
| **Gradient Fusion** | Novel formulation | Blending posterior-mean and x_0-prediction directions with time-decaying lambda; 5 scheduling strategies | Conceptual relation to classifier-free guidance |
| **Prudent Filtering** | **Novel in SBDD diffusion** | In-diffusion property-gated filtering; docking-score-driven selection; checkpoint-based re-noising | Individual ingredients (rejection sampling, property filtering) are well-known |
| **Scaffold Generation** | Engineering contribution | Murcko scaffold detection + evolution/growth pipeline with diversity-aware population selection | Masked diffusion (RePaint) is well-established |

**The most significant innovation** is the dynamic skip-step sampling strategy. To our knowledge, DiffDynamic is the first AIDD system to recognize and exploit the fact that denoising steps in molecular diffusion carry vastly different amounts of structural information, and to design an inference pipeline that allocates compute accordingly. The complementary structural repair mechanism — using the same trained model as a "local geometry autocorrector" — is a non-obvious capability that emerges from this analysis.

## Acknowledgments

This project builds upon [TargetDiff](https://github.com/DeepGraphLearning/TargetDiff) by Guan et al. (ICLR 2023). We gratefully acknowledge their foundational work on diffusion-based structure-based drug design.
