# Phase 1 Test Results

**Date:** 2025-11-02 17:53:39.437125

## Hypothesis

StandardScaler is hurting performance. Removing it should restore CV accuracy to ~70-71% (matching Novo).

## Results

### Test 1: No StandardScaler + Our Hyperparameters (max_iter=1000, random_state=42)
- CV Accuracy: 66.62% (+/- 9.26%)
- Training Accuracy: 74.07%
- Jain Test Accuracy: 68.09%

### Test 2: No StandardScaler + sklearn Defaults (max_iter=100, no random_state)
- CV Accuracy: 66.51% (+/- 9.57%)
- Training Accuracy: 74.07%
- Jain Test Accuracy: 67.02%

## Comparison with Baseline

| Configuration | CV Accuracy | Train Accuracy | Jain Accuracy |
|---------------|-------------|----------------|---------------|
| Baseline (with StandardScaler) | 63.88% | 95.62% | 55.32% |
| Test 1 (no scaler, max_iter=1000) | 66.62% | 74.07% | 68.09% |
| Test 2 (no scaler, sklearn defaults) | 66.51% | 74.07% | 67.02% |
| **Novo Benchmark (2025)** | **71.00%** | ~75-80% | **69.00%** |

## Conclusion

⚠️ **HYPOTHESIS PARTIALLY CONFIRMED**

Removing StandardScaler improved performance, but we're still below Novo's benchmarks.
Additional investigations needed (see console output for details).
