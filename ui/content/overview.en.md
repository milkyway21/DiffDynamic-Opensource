## Platform Overview

DiffDynamic is a diffusion-based framework for **3D structure-based drug design (SBDD)**. Built on [TargetDiff](https://github.com/DeepGraphLearning/TargetDiff), it introduces inference-time optimizations that significantly improve molecular quality and diversity.

> **Paper Demo** — This interface supports generation, docking evaluation, extraction, and full experiment history for paper demonstration and reproducibility.

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

Embeds property filtering inside the diffusion loop: **generate → QED/SA gate → Vina docking → Top-K selection**. The QED/SA gate avoids expensive docking on low-quality candidates.

### 4 · Scaffold-Constrained Generation

- **Evolve** — SDEdit noise annealing + population evolution for scaffold diversity
- **Grow** — Fixed scaffold atoms with de novo atom generation in the pocket

### 5 · Comprehensive Molecular Scoring

Integrates Vina affinity, QED, SA, Lipinski, and Lilly Medchem Rules (PAINS filtering) into a 0–100 composite score for multi-dimensional filtering.

---

## Innovation Summary

| Component | Novelty | Key Idea |
|-----------|---------|----------|
| **Dynamic skip-step** | First in AIDD | Topology/geometry decoupling, information-density scheduling |
| **Gradient fusion** | Novel formulation | Posterior-mean + x₀ direction blend, 5 schedules |
| **Prudent filtering** | First in SBDD diffusion | In-loop property gating + docking-driven selection |
| **Scaffold generation** | Engineering | Murcko detection + evolve/grow pipeline |

---

## Platform Guide

| Tab | Function |
|-----|----------|
| **Generate** | Dynamic fast sampling / Prudent multi-round optimization with auto-eval & extract |
| **Evaluate** | Vina docking + chemistry metrics on `.pt` outputs, export SDF/Excel |
| **Molecules** | Search evaluated molecules by pocket, SMILES, Vina score, etc. |
| **History** | Full operation audit trail with parameters and outputs |
| **Config** | Edit `sampling.yml`, view GPU status |

**Recommended workflow:** Generate (batch_size=5 for testing) → Auto evaluate → Auto extract → Browse results in Molecules
