# Hyperparameter Sweep - Live Status

**Started**: 2025-11-02 17:04:20
**Session**: `hyperparam_sweep`
**Log**: `logs/hyperparam_sweep_20251102_170416.log`

---

## Quick Access Commands

```bash
# View live progress
tmux attach -t hyperparam_sweep

# Follow logs
tail -f logs/hyperparam_sweep_20251102_170416.log

# Check latest results
tail -20 hyperparameter_sweep_results/sweep_results_*.csv
```

---

## Current Status

🚀 **RUNNING** - Extracting ESM embeddings (this takes ~5-10 minutes)

Progress: 3% (3/115 batches)

---

## Test Configurations (12 total)

### Priority 1: L2 Regularization Sweep (6 configs)
- [ ] C=0.001 (very strong)
- [ ] C=0.01 (strong) ← **Expected optimal**
- [ ] C=0.1 (moderate)
- [ ] C=1.0 (current baseline) → 67.5%
- [ ] C=10 (weak)
- [ ] C=100 (very weak)

### Priority 2: L1 Regularization (3 configs)
- [ ] C=0.01, L1
- [ ] C=0.1, L1
- [ ] C=1.0, L1

### Priority 3: SAGA Solver (3 configs)
- [ ] C=0.01, saga
- [ ] C=0.1, saga
- [ ] C=1.0, saga

---

## Expected Timeline

| Phase | Time | Status |
|-------|------|--------|
| Embedding extraction | 5-10 min | 🔄 IN PROGRESS |
| Config 1 (C=0.001) | 10-15 min | ⏳ Pending |
| Config 2 (C=0.01) | 10-15 min | ⏳ Pending |
| Config 3 (C=0.1) | 10-15 min | ⏳ Pending |
| ... (9 more) | ~2 hours | ⏳ Pending |
| **Total** | **2-4 hours** | - |

---

## What to Expect

### Good Signs ✅
- CV accuracy increases from 67.5%
- Overfitting gap decreases from 28.1%
- Training accuracy drops to 85-90% (healthier)

### Bad Signs ❌
- All configs stay at ~67%
- Overfitting gap remains >25%
- Errors or convergence warnings

---

## Results Will Show

For each configuration:
```
C=X, penalty=Y, solver=Z
CV Accuracy: XX.X% ± X.X%
Train Accuracy: XX.X%
Overfitting Gap: XX.X%
F1: X.XXX, ROC-AUC: X.XXX
```

Best configuration will be saved to:
- `hyperparameter_sweep_results/best_config_*.yaml`
- `hyperparameter_sweep_results/final_sweep_results_*.csv`

---

## If Successful

Next steps:
1. Retrain final model with best hyperparameters
2. Test on Jain dataset (91 antibodies)
3. Compare to baseline 55.3% accuracy
4. Update config_boughter.yaml
5. Document improvement

---

**Last Updated**: 2025-11-02 17:04:30
**Check back in**: 2-3 hours for final results
