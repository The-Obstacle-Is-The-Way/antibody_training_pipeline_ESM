# Jain Test Results - UNEXPECTED POOR PERFORMANCE

**Date**: 2025-11-02
**Model**: `models/boughter_vh_esm1v_logreg.pkl` (trained on Boughter)
**Test Set**: `test_datasets/jain/VH_only_jain_test.csv` (70 sequences)
**Status**: ⚠️ **INVESTIGATION NEEDED**

---

## Results Summary

### Confusion Matrix

```
                    Predicted
                Specific  Non-Specific
Actual  Specific      36          31
        Non-Spec        2           1
```

**Accuracy**: 52.9% (37/70)

### Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | **52.9%** |
| Precision (Class 1) | 3.1% |
| Recall (Class 1) | 33.3% |
| F1 Score (Class 1) | 5.7% |
| ROC-AUC | 37.8% |

---

## Comparison to Published Results

| Source | Confusion Matrix | Accuracy | Gap from Ours |
|--------|------------------|----------|---------------|
| **Novo (Paper)** | [[40 19][10 17]] | **69.0%** | **+16.1%** |
| **Hybri (Discord)** | [[39 19][11 17]] | **65.1%** | **+12.2%** |
| **Ours (Current)** | [[36 31][ 2  1]] | **52.9%** | **baseline** |

---

## Problem Analysis

### Key Issue: Overpredicting Non-Specific

The model is severely overpredicting non-specific sequences:

- **Predicted non-specific**: 32 sequences (31 FP + 1 TP)
- **Expected non-specific** (based on Novo/Hybri): ~19 sequences
- **Overprediction rate**: 68% more predictions than expected

### Precision Breakdown

- **Precision for non-specific class**: 3.1% (1 correct out of 32 predictions)
- **Precision for specific class**: 94.7% (36 correct out of 38 predictions)

The model is **heavily biased** toward predicting non-specific, leading to terrible precision.

---

## Potential Root Causes

### 1. **Class Imbalance Mismatch** (Most Likely)

**Training set** (Boughter):
- 914 sequences
- 443 specific (48.5%) vs 471 non-specific (51.5%)
- Nearly **balanced**

**Test set** (Jain):
- 70 sequences
- 67 specific (95.7%) vs 3 non-specific (4.3%)
- Heavily **imbalanced**

**Issue**: Model was trained on balanced data but tested on imbalanced data. Default threshold (0.5) may not be appropriate for 95:5 ratio.

### 2. **Threshold Calibration**

- Logistic regression uses 0.5 as default decision threshold
- For imbalanced test sets, this threshold should be adjusted
- Novo/Hybri may have used a different threshold optimized for Jain distribution

### 3. **Dataset Distribution Shift**

**Boughter sequences**:
- Source: Boughter polyreactivity assay
- Flags: 0-7 scale based on specific reactivity measurements
- Training protocol: Exclude 1-3 flags (mild), use 0 vs ≥4

**Jain sequences**:
- Source: 4 different assay clusters (PSR, HIC/SMAC, ELISA, stability)
- Flags: Table 1 thresholds (90th percentile of approved mAbs)
- Clinical-stage antibodies (different population)

**Issue**: Different assay methods and antibody populations may lead to embedding distribution shift.

### 4. **Preprocessing Differences**

We applied P0 fixes (V-domain reconstruction) to both datasets, but:
- Boughter preprocessing: `preprocessing/process_boughter.py`
- Jain preprocessing: `preprocessing/process_jain.py`
- Different raw data sources and annotation paths

---

## Verification Checks

### Training Performance (Sanity Check)

From `TRAINING_RESULTS.md`:

```
✅ Boughter 10-fold CV: 67.5% ± 8.9% accuracy
✅ Boughter training set: 95.6% accuracy (expected overfitting)
✅ Model saved and loaded correctly
```

**Training worked correctly** - the issue is in generalization to Jain.

### Test Set Integrity

From `tests/test_jain_embedding_compatibility.py`:

```
✅ All 5 ESM compatibility tests passed
✅ 70 test sequences (67 specific + 3 non-specific)
✅ All sequences gap-free and ESM-compatible
✅ Label distribution matches expected (67:3 ratio)
```

**Test data is valid** - P0 blockers fixed, sequences are clean.

---

## Investigation Steps (TODO)

### 1. Check Prediction Probabilities

```bash
# Examine the prediction probabilities to see if they're reasonable
head test_results/jain/predictions_boughter_vh_esm1v_logreg_VH_only_jain_test_*.csv
```

**Questions**:
- Are probabilities clustered near decision boundary?
- What's the distribution of predicted probabilities?
- Are there clear confident predictions or is everything ambiguous?

### 2. Try Different Classification Thresholds

```python
# Instead of default 0.5, try threshold that matches test set ratio
# Expected ratio: 67:3 = 95.7% specific
# Try threshold = 0.96 (predict non-specific only if P(nonspec) > 0.96)
```

**Expected Impact**: Should reduce false positives dramatically.

### 3. Compare Embeddings

```python
# Compare Boughter training embeddings vs Jain test embeddings
# Check if there's a distribution shift
# Use t-SNE or PCA to visualize
```

**Questions**:
- Do Jain embeddings cluster differently from Boughter?
- Is there a clear separation between datasets?
- Are the 3 Jain non-specific sequences similar to Boughter non-specific?

### 4. Check Hybri's Methodology

**From Discord investigation**:
- Hybri got 65.1% on Jain ([[39 19][11 17]])
- This is also lower than Novo's 69%
- Hybri likely used provisional Jain preprocessing

**Questions**:
- What threshold did Hybri use?
- Did Hybri retrain on Jain or use Boughter model?
- What preprocessing differences exist?

### 5. Review Novo's Methodology

**From paper** (Sakhnini et al. 2025):
> "An accuracy of 69% was obtained for the parsed Jain dataset"

**Questions**:
- Did Novo train on combined Boughter+Jain? (more likely)
- What was Novo's train/test split?
- Did Novo use threshold calibration?
- Did Novo exclude mild cases (1-3 flags) from Jain?

---

## Hypotheses to Test

### Hypothesis 1: Threshold Issue (Most Likely)

**Prediction**: Adjusting threshold from 0.5 to ~0.9 will significantly improve performance.

**Test**:
```python
# Rerun test with different threshold
y_proba = model.predict_proba(X)[:, 1]
y_pred_calibrated = (y_proba > 0.9).astype(int)
```

**Expected Outcome**: Confusion matrix closer to [[40-50  17-27][0-2  1-3]]

### Hypothesis 2: Domain Shift (Possible)

**Prediction**: Jain embeddings are systematically different from Boughter embeddings.

**Test**: Visualize embeddings, compute distribution statistics.

**Expected Outcome**: Some separation between datasets, but overlapping regions.

### Hypothesis 3: Novo Used Different Protocol (Likely)

**Prediction**: Novo either:
- Trained on combined data (Boughter + Jain train split)
- Used different flag thresholds
- Applied threshold calibration

**Test**: Re-read Novo paper methods section carefully.

**Expected Outcome**: Find methodological difference explaining gap.

---

## Comparison to Training Performance

| Dataset | Performance | Notes |
|---------|-------------|-------|
| **Boughter (10-fold CV)** | 67.5% ± 8.9% | ✅ Expected range |
| **Boughter (training set)** | 95.6% | ✅ Overfitting (expected) |
| **Jain (external test)** | 52.9% | ❌ **WORSE than training CV!** |

**This is highly unusual** - external test performance is 14.6 percentage points WORSE than training CV. Normally we expect:
- Training CV: ~67%
- External test: ~65-70% (slight drop due to distribution shift)
- Actual: ~53% (major drop, indicates serious issue)

---

## Next Actions

### Immediate (Do First)

1. ✅ **Document findings** - this file
2. **Check prediction probabilities** - examine distribution
3. **Try threshold adjustment** - test with threshold = 0.9, 0.95
4. **Re-read Novo methods** - find any missed details

### Short-term (This Session)

5. **Compare embeddings** - visualize Boughter vs Jain
6. **Test on Boughter held-out** - verify model works correctly
7. **Check Hybri's code** - if available on Discord/GitHub

### Long-term (Future Work)

8. **Retrain with threshold tuning** - optimize for Jain ratio
9. **Try combined training** - train on Boughter + Jain (if that's what Novo did)
10. **Contact Novo authors** - ask about methodology details

---

## Files Generated

```
test_results/jain/
├── confusion_matrix_VH_only_jain_test.png          # Confusion matrix heatmap
├── detailed_results_VH_only_jain_test_20251102_131152.yaml  # Full results
├── predictions_boughter_vh_esm1v_logreg_VH_only_jain_test_20251102_131152.csv  # Predictions
└── test_20251102_131152.log                        # Test log
```

---

## Conclusion

⚠️ **Model performance on Jain is significantly worse than expected** (52.9% vs 65-69%)

**Most Likely Cause**: Threshold calibration issue due to class imbalance mismatch (training balanced 50:50, testing imbalanced 95:5)

**Recommended Action**: Adjust classification threshold and re-evaluate before concluding the model doesn't work.

**Status**: Investigation in progress. Do not proceed with further testing until this is resolved.

---

## References

- Training results: `TRAINING_RESULTS.md`
- Jain P0 fix: `docs/jain/JAIN_P0_FIX_REPORT.md`
- Novo paper: `literature/markdown/Sakhnini_2025_Antibody_NonSpecificity_PLM_Biophysical.md`
- Test results directory: `test_results/jain/`
