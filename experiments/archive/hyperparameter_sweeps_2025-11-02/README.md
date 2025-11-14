# Hyperparameter Sweep Results - November 2, 2025

## Provenance

- **Created**: 2025-11-02
- **Script**: `preprocessing/boughter/train_hyperparameter_sweep.py`
- **Dataset**: Boughter VH training data (914 sequences)
- **Model**: ESM-1v + Logistic Regression
- **Methodology**: 10-fold stratified cross-validation

## Two Sweep Runs

This archive contains results from **two separate sweep runs** conducted on Nov 2, 2025:

### Run 1: 17:05:16

- **Best Configuration**: C=0.01, L2 penalty, lbfgs solver
- **CV Accuracy**: 67.06% ± 4.70%
- **Config File**: `best_config_20251102_170516.yaml`
- **Results**: `final_sweep_results_20251102_170516.csv`

### Run 2: 18:25:42

- **Best Configuration**: C=1.0, L2 penalty, lbfgs solver
- **CV Accuracy**: 67.50% ± 4.45%
- **Config File**: `best_config_20251102_182542.yaml`
- **Results**: `final_sweep_results_20251102_182542.csv`

## Contents

- `sweep_results_20251102_*.csv` - Intermediate results after each config (18 files)
- `final_sweep_results_20251102_*.csv` - Complete sweep results (2 files)
- `best_config_20251102_*.yaml` - Optimal hyperparameters found (2 files)

## Parameter Grid Tested

- **C values**: [0.001, 0.01, 0.1, 1.0, 10, 100]
- **Penalties**: L1, L2
- **Solvers**: lbfgs, liblinear, saga
- **Total configurations**: 12 per sweep

## Key Findings

1. **Optimal C range**: 0.01 - 1.0 (both sweeps converged here)
2. **Penalty**: L2 consistently outperformed L1
3. **Solver**: lbfgs showed best performance
4. **Performance**: ~67% CV accuracy (both sweeps)
5. **Overfitting gap**: Higher C values (1.0) showed more overfitting

## Archive Notes

These files were moved from root `hyperparameter_sweep_results/` during
repository cleanup (2025-11-14) to improve directory organization.

Original git commit history preserved via `git mv`.

## Future Sweeps

New hyperparameter sweeps should write to:
`experiments/hyperparameter_sweeps/`

**NOT** to this archive directory.
