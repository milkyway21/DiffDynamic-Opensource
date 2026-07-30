# rdkit-structure-repair

Deterministic RDKit molecular structure repair for reconstructed molecules.
All repairs use chemical rules, graph algorithms, valence constraints, and
candidate validation — no machine learning models.

## Install

```bash
cd rdkit-structure-repair
pip install -e .
# or use DiffDynamic conda env (RDKit already present)
conda run -n diffdynamic pip install -e .
```

## CLI

```bash
structure-repair validate reconstructed.sdf --output validation/
structure-repair repair reconstructed.sdf --config configs/conservative.yaml --output results/
structure-repair explain results/audit.jsonl --molecule-id GEN_00128
```

## Pipeline position (DiffDynamic)

```text
sample → targetdiff_baseline_refine → reconstruct → [structure_repair] → eval/SDF
```

Enable via `sample.rdkit_structure_repair.enable` in `configs/sampling.yml` (default: false).
