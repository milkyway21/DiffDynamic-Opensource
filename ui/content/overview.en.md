## Platform Overview

DiffDynamic is a diffusion-based framework for **3D structure-based drug design (SBDD)**. Built on [TargetDiff](https://github.com/DeepGraphLearning/TargetDiff), it introduces inference-time optimizations that significantly improve molecular quality and diversity.

> **Paper Demo** — This interface supports generation, pocket quality evaluation, docking evaluation, extraction, and full experiment history for paper demonstration and reproducibility.

---

## Core Innovations

### 1 · Two-Stage Dynamic Skip-Step Sampling

**First in AIDD.** Information density varies dramatically across the denoising trajectory — high-noise steps determine topology, low-noise steps refine geometry. DiffDynamic splits sampling into **Large Step** (topology exploration) and **Refine** (geometry repair), achieving **~60–70% fewer neural network evaluations** while maintaining quality.

| Stage | Range | Purpose |
|-------|-------|---------|
| Large Step | t=999 → t_boundary | Topology exploration with adaptive strides |
| Refine | t_boundary → 0 | Geometry refinement with dense steps |

### 2 · Gradient Fusion

At each denoising step, blends **local posterior gradient** with **global x₀-prediction gradient**. λ(t) decays from 1.0 to 0.0 — global signal early, local refinement late. Five schedules: quadratic, linear, exponential, adaptive, time.

### 3 · Prudent Multi-Round Filtering

Embeds property filtering inside the diffusion loop: **generate → QED/SA gate → Vina docking → Top-K selection**. The QED/SA gate avoids expensive docking on low-quality candidates. Supports **adaptive gates** that relax/tighten thresholds from the reference ligand's properties.

### 4 · Scaffold-Constrained Generation (four modes)

| Mode | Description |
|------|-------------|
| **Grow** | Fix Murcko scaffold atoms; grow side chains / fragments inside the pocket |
| **Evolve** | SDEdit noise annealing + population evolution across the whole molecule |
| **Dynamic Locked** | Lock scaffold atom types and coordinates |
| **Prudent** | Scaffold constraint + multi-round property gating + docking-driven promotion |

Scaffolds can be auto-detected from SDF/PDB (Murcko / generic), or specified via atom indices / SMARTS. **ScaffoldCascade** merges multiple randomized rounds for higher quality and diversity.

### 5 · Pocket Quality Multi-Dimension Evaluation

Runs an 8-dimension assessment on generated results:

A · Vina docking + ligand efficiency · B · Atom clustering · C · Ligand efficiency · D · Drug-likeness · E · Completeness ratio · F · Uniqueness · G · Size consistency · H · Pocket volume

> Results include a radar chart and pocket visualization images.

### 6 · Comprehensive Molecular Scoring

Integrates Vina affinity, QED, SA, Lipinski, and Lilly Medchem Rules (PAINS filtering) into a 0–100 composite score for multi-dimensional filtering.

---

## Innovation Summary

| Component | Novelty | Key Idea |
|-----------|---------|----------|
| **Dynamic skip-step** | First in AIDD | Topology/geometry decoupling, information-density scheduling |
| **Gradient fusion** | Novel formulation | Posterior-mean + x₀ direction blend, 5 schedules |
| **Prudent filtering** | First in SBDD diffusion | In-loop property gating + docking-driven selection |
| **Scaffold generation** | Engineering | Murcko detection + grow/evolve/locked/prudent pipeline |
| **Cascade generation** | Engineering | Multi-round randomization merged for diversity |

---

## Platform Guide

| Tab | Function |
|-----|----------|
| **Generate** | Dynamic fast / Prudent optimization / preset configs, auto-eval & extract |
| **Scaffold** | Grow / Evolve / Dynamic Locked / Prudent with adaptive gates |
| **Cascade** | Multi-round randomized Grow merged into one run |
| **Optimize** | Iterative refinement of seed molecules (QED/SA/Vina) |
| **Evaluate** | Vina docking + chemistry metrics on `.pt` outputs, export SDF/Excel |
| **Pocket Eval** | 8-dimension pocket quality scoring + radar chart + images |
| **Molecules** | Search evaluated molecules by pocket, SMILES, Vina score, etc. |
| **History** | Full audit trail with result summary and one-click re-run |
| **Config** | Edit `sampling.yml`, view GPU status |

**Recommended workflow:** Upload/select protein–ligand → choose a mode to generate (batch_size=5 for testing) → auto evaluate → auto extract → inspect pocket evaluation / molecules. Every operation is recorded automatically.

---

## Operations

- The server listens on `0.0.0.0`; multi-user concurrency is handled by the job queue.
- All generation/evaluation/extraction tasks run asynchronously with real-time progress and cancellation.
- Runs, molecule catalogs and eval results persist in SQLite (`diffdynamic.db`) across restarts.