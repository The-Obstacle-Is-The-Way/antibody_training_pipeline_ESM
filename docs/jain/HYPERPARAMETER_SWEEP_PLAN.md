# Hyperparameter Sweep Execution Plan

**Date**: 2025-11-02
**Status**: 🚀 **IN PROGRESS**
**Goal**: Match Novo's 71% 10-CV accuracy

---

## Current Status

**Baseline Performance** (from `logs/boughter_training.log`):
```
Configuration:
  C: 1.0 (sklearn default)
  penalty: 'l2'
  solver: 'lbfgs'
  max_iter: 1000
  class_weight: 'balanced'
  random_state: 42

Results:
  10-fold CV accuracy: 67.5% ± 8.9%
  Training accuracy: 95.6%
  Overfitting gap: 28.1%
```

**Target**: 71.0% (Novo's benchmark from Figure S2)
**Gap**: -3.5%

---

## Sweep Configuration

### Test Grid (12 configurations)

**Priority 1: C parameter sweep with L2 regularization**
```python
[
    {'C': 0.001, 'penalty': 'l2', 'solver': 'lbfgs'},  # Very strong regularization
    {'C': 0.01,  'penalty': 'l2', 'solver': 'lbfgs'},  # Strong regularization
    {'C': 0.1,   'penalty': 'l2', 'solver': 'lbfgs'},  # Moderate regularization
    {'C': 1.0,   'penalty': 'l2', 'solver': 'lbfgs'},  # Current default ← BASELINE
    {'C': 10,    'penalty': 'l2', 'solver': 'lbfgs'},  # Weak regularization
    {'C': 100,   'penalty': 'l2', 'solver': 'lbfgs'},  # Very weak regularization
]
```

**Priority 2: L1 regularization (feature selection)**
```python
[
    {'C': 0.01, 'penalty': 'l1', 'solver': 'liblinear'},
    {'C': 0.1,  'penalty': 'l1', 'solver': 'liblinear'},
    {'C': 1.0,  'penalty': 'l1', 'solver': 'liblinear'},
]
```

**Priority 3: Alternative solver (saga)**
```python
[
    {'C': 0.01, 'penalty': 'l2', 'solver': 'saga'},
    {'C': 0.1,  'penalty': 'l2', 'solver': 'saga'},
    {'C': 1.0,  'penalty': 'l2', 'solver': 'saga'},
]
```

---

## Implementation

### Script: `train_hyperparameter_sweep.py`

**Features**:
- Extract embeddings ONCE (cached for all experiments)
- 10-fold stratified cross-validation (matches Novo)
- Track training accuracy to measure overfitting
- Compute multiple metrics (accuracy, F1, ROC-AUC, precision, recall)
- Save intermediate results after each configuration
- Automatically identify best configuration

**Execution Time Estimate**: 2-4 hours
- ~12 configurations
- ~10-15 minutes per configuration
- Parallel fold processing (n_jobs=-1)

---

## Launch Instructions

### Option 1: Tmux (Recommended)

```bash
./run_sweep_tmux.sh
```

**Tmux commands**:
- Attach: `tmux attach -t hyperparam_sweep`
- Detach: `Ctrl+B, then D`
- View logs: `tail -f logs/hyperparam_sweep_*.log`

### Option 2: Direct Execution

```bash
python3 train_hyperparameter_sweep.py 2>&1 | tee logs/hyperparam_sweep.log
```

---

## Expected Outcomes

### Hypothesis 1: Optimal C in range [0.01, 0.1]

**Reasoning**:
- Current C=1.0 → 28% overfitting gap (95.6% train, 67.5% CV)
- ESM embeddings are 1280-dimensional → high-dimensional feature space
- Stronger regularization should reduce overfitting

**Expected improvement**: +2-4% CV accuracy

---

### Hypothesis 2: L1 regularization may help

**Reasoning**:
- L1 can zero out irrelevant features
- ESM embeddings may have redundant dimensions
- Could improve generalization

**Expected improvement**: +1-3% CV accuracy (if effective)

---

### Hypothesis 3: Solver choice matters less

**Reasoning**:
- lbfgs, liblinear, saga should converge to similar solutions
- Main difference is computational efficiency

**Expected improvement**: Minimal (<1%)

---

## Success Criteria

### Target Metrics

| Metric | Current | Target | Stretch Goal |
|--------|---------|--------|--------------|
| **10-CV Accuracy** | 67.5% | **71.0%** | 72%+ |
| **Overfitting Gap** | 28.1% | <20% | <15% |
| **Training Accuracy** | 95.6% | 85-90% | 80-85% |
| **CV Std Dev** | 8.9% | <8% | <6% |

### Evaluation Criteria

**Minimum Success**:
- CV accuracy ≥ 70.0%
- Overfitting gap < 20%

**Full Success**:
- CV accuracy ≥ 71.0% (matches Novo)
- Overfitting gap < 15%

**Optimal Configuration**:
- Highest CV accuracy
- Lowest overfitting gap
- Lowest CV std dev (most stable)

---

## Output Files

### During Execution

| File | Description |
|------|-------------|
| `hyperparameter_sweep_results/sweep_results_*.csv` | Intermediate results (updated after each config) |
| `logs/hyperparam_sweep_*.log` | Full execution log with CV results |

### After Completion

| File | Description |
|------|-------------|
| `hyperparameter_sweep_results/final_sweep_results_*.csv` | Complete results table |
| `hyperparameter_sweep_results/best_config_*.yaml` | Best hyperparameter configuration |

### Results Format

```csv
C,penalty,solver,class_weight,cv_accuracy_mean,cv_accuracy_std,cv_f1_mean,cv_roc_auc_mean,train_accuracy_mean,overfitting_gap
0.01,l2,lbfgs,balanced,0.7123,0.0654,0.7034,0.7821,0.8734,0.1611
0.1,l2,lbfgs,balanced,0.7089,0.0712,0.6987,0.7756,0.9012,0.1923
1.0,l2,lbfgs,balanced,0.6750,0.0890,0.6790,0.7409,0.9562,0.2812
...
```

---

## Post-Sweep Actions

### If Successful (≥71% CV)

1. ✅ **Retrain final model** with best hyperparameters
2. ✅ **Test on Jain** to measure generalization improvement
3. ✅ **Update `config_boughter.yaml`** with optimal parameters
4. ✅ **Document results** in `HYPERPARAMETER_INVESTIGATION.md`

### If Unsuccessful (<70% CV)

1. ⚠️ **Investigate embedding quality**
   - Check for preprocessing differences
   - Compare embedding distributions (Boughter vs Jain)

2. ⚠️ **Try advanced techniques**
   - ElasticNet (L1 + L2 combination)
   - Different CV strategies (stratified by family)
   - Ensemble methods

3. ⚠️ **Contact Novo authors**
   - Request exact hyperparameters
   - Ask about preprocessing pipeline

---

## Monitoring Progress

### Real-time Monitoring

```bash
# Attach to tmux session
tmux attach -t hyperparam_sweep

# Or tail logs
tail -f logs/hyperparam_sweep_*.log

# Or check intermediate results
watch -n 30 'tail -20 hyperparameter_sweep_results/sweep_results_*.csv'
```

### Key Metrics to Watch

1. **CV Accuracy**: Should increase as we test stronger regularization
2. **Overfitting Gap**: Should decrease (training accuracy closer to CV)
3. **Std Dev**: Should decrease (more stable performance)

---

## Troubleshooting

### If sweep fails:

**Check**:
1. ESM model loaded correctly: `logs/hyperparam_sweep_*.log`
2. Embeddings extracted: Look for "Extracted embeddings shape"
3. sklearn compatibility: Some solver/penalty combinations invalid

**Common errors**:
- `ValueError: Solver ... does not support penalty ...` → Skip invalid combinations
- `MemoryError` → Reduce batch size in embedding extraction
- `ConvergenceWarning` → Increase max_iter (already 1000)

---

## Timeline

**Start**: 2025-11-02 (today)
**Estimated completion**: 2-4 hours
**Results review**: Same day
**Model retraining**: +1 hour after best config identified

---

## References

- Baseline config: `config_boughter.yaml`
- Baseline results: `logs/boughter_training.log`
- Sweep script: `train_hyperparameter_sweep.py`
- Launch script: `run_sweep_tmux.sh`
- Investigation: `docs/jain/HYPERPARAMETER_INVESTIGATION.md`

---

**Status**: 🚀 Sweep launched in tmux
**Next**: Monitor progress and analyze results
