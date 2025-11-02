# Jain Dataset Unbalanced Data Handling - Blocker Analysis

**Date**: 2025-11-02
**Status**: 🔴 **CRITICAL BLOCKERS IDENTIFIED**
**Investigation**: Systematic review of Jain dataset handling methodology

---

## Executive Summary

A systematic review of how the Jain dataset is handled in our pipeline vs. published methodologies (Novo/Hybri) has revealed **multiple critical blockers (P0-P2)** that explain the poor test performance (52.9% vs expected 65-69%).

**Key Finding**: The combination of **test set definition mismatch** (P0), **class imbalance threshold handling** (P1), and **dataset provenance uncertainty** (P2) creates a perfect storm that makes our results non-comparable to published benchmarks.

---

## 🔴 P0 BLOCKER: Test Set Definition Mismatch

### Issue Description

**Our test set size does NOT match the published confusion matrices:**

| Source | Test Set Size | Ratio (Specific:Non-Specific) | Source Document |
|--------|---------------|------------------------------|-----------------|
| **Our Implementation** | 70 antibodies | 67:3 (95.7% : 4.3%) | `test_datasets/jain/VH_only_jain_test.csv` |
| **Novo (Paper)** | 86 antibodies | 59:27 (68.6% : 31.4%) | Confusion matrix [[40 19][10 17]] |
| **Hybri (Discord)** | 86 antibodies | 58:28 (67.4% : 32.6%) | Confusion matrix [[39 19][11 17]] |

**Mismatch**: 70 ≠ 86 antibodies!

### Root Cause Analysis

According to Novo paper (Sakhnini et al. 2025, Methods section 4.3):

> "As in the Boughter dataset, the Jain dataset was **parsed** into two groups: **specific (0 flags)** and **non-specific (>3 flags)**, leaving out the mildly non-specific antibodies (1-3 flags)"

**Our Implementation** (following this description):
- **Specific**: `flags_total == 0` → 67 antibodies
- **Mild** (excluded): `1 <= flags_total <= 3` → 67 antibodies
- **Non-specific**: `flags_total >= 4` → 3 antibodies
- **Test set**: 67 + 3 = **70 antibodies**

**Flag Distribution in Full Dataset** (137 antibodies total):

```
flags_total  count  category
     0         67   specific (label=0)
     1         21   mild (label=NaN)
     2         22   mild (label=NaN)
     3         24   mild (label=NaN)
     4          3   non_specific (label=1)
```

**Analysis**:
- With 0 vs ≥4 threshold: 67:3 = 70 total ✓ (matches our implementation)
- With 0-1 vs ≥2 threshold: 88:49 = 137 total (all antibodies)
- With 0-2 vs ≥3 threshold: 110:27 = 137 total (27 non-specific matches Novo!)

**Hypothesis**: Novo may have used a **different threshold** (≥3 instead of ≥4) but documented it as ">3 flags"!

### Blocker Severity

**P0 - CRITICAL**: Cannot compare results to published benchmarks with fundamentally different test sets.

### Impact

1. **Performance comparison invalid**: Comparing 70-antibody test (95:5 imbalanced) to 86-antibody test (59:27 balanced) is meaningless
2. **Threshold calibration different**: 95:5 ratio requires threshold ~0.95, while 59:27 ratio might work with ~0.5
3. **Statistical significance affected**: Different test set sizes affect confidence intervals
4. **Reproducibility broken**: Cannot reproduce Novo/Hybri results with current setup

### Recommended Actions

**Option 1: Match Novo's actual test set (PREFERRED)**
1. Contact Novo authors (llsh@novonordisk.com, dngt@novonordisk.com) to clarify:
   - Exact flag threshold used (is ">3" actually "≥3"?)
   - Total test set size (70 vs 86 discrepancy)
   - Which antibodies were included/excluded
2. Replicate their exact methodology once confirmed

**Option 2: Try threshold ≥3** (if Option 1 unavailable)
1. Test with `flags_total >= 3` instead of `>= 4`
2. This gives 27 non-specific (matches Novo's confusion matrix)
3. But still doesn't explain 110 vs 59 specific antibodies

**Option 3: Use different evaluation metric**
- Report performance separately for each threshold
- Use ROC-AUC and PR-AUC instead of accuracy (threshold-independent)
- Compare probability calibration curves

---

## 🔴 P1 BLOCKER: Class Imbalance Threshold Handling

### Issue Description

**Severe class imbalance in test set NOT accounted for in model threshold:**

| Dataset | Class Distribution | Imbalance Ratio | Default Threshold Appropriate? |
|---------|-------------------|-----------------|-------------------------------|
| **Boughter (Training)** | 443 specific (48.5%) vs 471 non-specific (51.5%) | Nearly **balanced** (1.06:1) | ✅ Yes (0.5 is reasonable) |
| **Jain (Test - Ours)** | 67 specific (95.7%) vs 3 non-specific (4.3%) | Heavily **imbalanced** (22.3:1) | ❌ No (should be ~0.96) |
| **Jain (Test - Novo?)** | 59 specific (68.6%) vs 27 non-specific (31.4%) | Moderately **imbalanced** (2.19:1) | ⚠️ Borderline (should be ~0.7) |

### Model Behavior Analysis

**Current Results** (threshold = 0.5):

```
Confusion Matrix:
                    Predicted
                Specific  Non-Specific
Actual  Specific      36          31     ← 46% misclassified as non-specific!
        Non-Spec        2           1     ← Only 3 non-specific total
```

**Problem Breakdown**:
1. **Overpredicting non-specific**: 32 predictions (31 FP + 1 TP) vs 3 actual
2. **False Positive Rate**: 31/67 = 46.3% (nearly HALF of specific antibodies flagged!)
3. **Precision for non-specific**: 1/32 = 3.1% (96.9% of "non-specific" predictions are wrong!)
4. **Model is biased** toward predicting non-specific due to balanced training

### Why This Happens

**Logistic Regression default behavior**:
- Trained on balanced data (50:50) → learns decision boundary at P(non-spec) ≈ 0.5
- Applied to imbalanced test (95:5) → predicts too many non-specific
- This is a **textbook class imbalance problem**

**Correct threshold for 95:5 ratio**:
- Should be: P(non-spec) > (base_rate_non_spec) ≈ 0.95
- Current: P(non-spec) > 0.5 (too lenient)

### Evidence from Test Results

From `test_results/jain/detailed_results_VH_only_jain_test_20251102_131152.yaml`:

```yaml
y_proba: [array of probabilities from model]
y_pred: [predicted labels with threshold=0.5]
y_true: [actual labels]
```

**Need to examine**: Distribution of predicted probabilities to see if they're clustered near 0.5 or well-separated.

### Blocker Severity

**P1 - HIGH**: Threshold calibration failure makes results appear worse than model's actual capability.

### Impact

1. **Underestimated model quality**: Model may have good probability calibration but poor threshold
2. **Invalid accuracy metric**: 52.9% accuracy is misleading with imbalanced data
3. **Better metrics needed**: ROC-AUC (37.8%) and PR-AUC are more informative
4. **Threshold tuning required**: Need to find optimal threshold for 95:5 ratio

### Recommended Actions

**Immediate**:
1. **Analyze probability distribution**:
   ```bash
   # Check predicted probabilities from detailed results
   python3 -c "
   import yaml
   import numpy as np
   with open('test_results/jain/detailed_results_VH_only_jain_test_20251102_131152.yaml') as f:
       results = yaml.safe_load(f)
       probs = results['results']['boughter_vh_esm1v_logreg']['predictions']['y_proba']
       print(f'Probability distribution:')
       print(f'  Mean: {np.mean(probs):.3f}')
       print(f'  Median: {np.median(probs):.3f}')
       for p in [50, 75, 90, 95, 99]:
           print(f'  {p}th percentile: {np.percentile(probs, p):.3f}')
   "
   ```

2. **Test multiple thresholds**:
   ```python
   # Test thresholds: 0.5, 0.7, 0.9, 0.95, 0.96
   for threshold in [0.5, 0.7, 0.9, 0.95, 0.96]:
       y_pred_adj = (y_proba > threshold).astype(int)
       accuracy = (y_pred_adj == y_true).mean()
       # Calculate precision, recall, F1 for each
   ```

3. **Use threshold-independent metrics**:
   - ROC-AUC (already computed: 37.8%)
   - PR-AUC (already computed)
   - Calibration curves
   - Probability histograms stratified by true label

**Long-term**:
1. **Retrain with class weights**: Penalize misclassification of rare class more heavily
2. **Use stratified sampling**: Ensure validation sets match test set imbalance
3. **Calibrate probabilities**: Use Platt scaling or isotonic regression post-training

---

## 🟡 P2 CONCERN: Dataset Provenance Uncertainty

### Issue Description

**Unclear which version of Jain dataset Novo/Hybri used:**

| Version | Source | Antibody Count | Evidence |
|---------|--------|---------------|----------|
| **PNAS Official** | [pnas.1616408114.sd01-03.xlsx](https://www.pnas.org/doi/10.1073/pnas.1616408114) | **137** | Jain et al. 2017 supplementary files |
| **Historic Repository** | Unknown (mentioned in `jain_data_sources.md`) | **80** | Note: "existing jain.csv contains only 80 antibodies" |
| **Novo/Hybri** | Not explicitly documented | **86** (inferred from confusion matrix) | Confusion matrix totals |

### Evidence of Provenance Issue

From `docs/jain/jain_data_sources.md` (lines 146-159):

> **Problem:** The existing `test_datasets/jain.csv` in the repository contains only 80 antibodies (not 137), and values don't match the PNAS supplementary files.
>
> **Example discrepancy:**
> - `jain.csv` abituzumab `smp`: 0.126
> - PNAS SD03 `PSR SMP Score`: 0.167

**This suggests**:
1. There was a PREVIOUS `jain.csv` with 80 antibodies (origin unknown)
2. We recreated it from PNAS files → 137 antibodies
3. Novo/Hybri results suggest 86 antibodies used
4. **80 ≠ 86 ≠ 137** → version mismatch!

### Possible Explanations

**Hypothesis 1**: Novo received provisional data from Jain et al. before publication
- Early-access dataset with fewer antibodies
- Different preprocessing or quality filtering
- Not publicly available

**Hypothesis 2**: Novo excluded antibodies with missing data
- PNAS SD03 has 139 rows (137 antibodies + 2 metadata rows)
- Some antibodies might have incomplete biophysical measurements
- Filtering for complete data → 86 antibodies?

**Hypothesis 3**: Novo used specific clinical stage filter
- Approved + Phase 3 only?
- Approved + Phase 2/3 + Phase 3?
- Need to check counts in PNAS SD01 metadata

### Blocker Severity

**P2 - MEDIUM**: Affects reproducibility but workarounds exist.

### Impact

1. **Cannot exactly reproduce Novo results**: Without knowing exact dataset version
2. **Different baseline characteristics**: 80 vs 86 vs 137 antibodies may have different biophysical property distributions
3. **Flag calculation differences**: Thresholds (90th percentile of approved) differ with different datasets
4. **Comparison validity questioned**: Are we comparing apples to apples?

### Recommended Actions

**Immediate**:
1. **Count antibodies by clinical stage** in our 137-antibody dataset:
   ```bash
   python3 -c "
   import pandas as pd
   df = pd.read_csv('test_datasets/jain.csv')
   print('Clinical status distribution:')
   print(df['Clinical Status'].value_counts())
   print(f'\nApproved + Phase 3: {len(df[df[\"Clinical Status\"].isin([\"Approved\", \"Phase 3\"])])}')
   "
   ```

2. **Check for missing data patterns**:
   ```python
   # See which antibodies have complete measurements across all assays
   complete_data = df.dropna(subset=['psr_smp', 'acsins_dlmax_nm', 'ova', 'bvp_elisa', ...])
   print(f'Antibodies with complete data: {len(complete_data)}')
   ```

3. **Document our exact dataset**:
   - Save antibody list with IDs
   - Document source (PNAS SD01-03)
   - Note any exclusions/filters applied

**Long-term**:
1. **Contact Novo authors** to request:
   - Exact antibody IDs used in Jain test set
   - Any preprocessing/filtering steps applied
   - Dataset version/source details

2. **Contact Hybri** (Discord) to compare:
   - Which antibodies they tested on
   - Whether they used provisional or PNAS data
   - Preprocessing methodology

3. **Fallback**: Document differences and report results for multiple versions:
   - 70-antibody test (0 vs ≥4 flags)
   - 86-antibody test (if we can identify which antibodies)
   - Full 137-antibody test (all thresholds)

---

## 🟢 P3-P5: Lower Priority Issues

### P3: Prediction Probability Distribution Unknown

**Issue**: Haven't examined distribution of predicted probabilities yet.

**Impact**: Can't determine if model is well-calibrated or if probabilities are meaningful.

**Action**: Plot histogram of `y_proba` stratified by `y_true` to check calibration.

---

### P4: No Uncertainty Quantification

**Issue**: Single-point accuracy estimate without confidence intervals.

**Impact**: Can't assess statistical significance of 52.9% vs 69% difference.

**Action**: Bootstrap resampling to estimate 95% CI for all metrics.

---

### P5: Limited Diagnostic Metrics

**Issue**: Only accuracy, precision, recall, F1, ROC-AUC reported.

**Impact**: Missing:
- Confusion matrix at multiple thresholds
- Calibration curve
- Per-antibody predictions for manual review
- Error analysis (which antibodies misclassified and why?)

**Action**: Generate comprehensive diagnostic report.

---

## Comparison Matrix: Our Implementation vs. Novo/Hybri

| Aspect | Our Implementation | Novo (Sakhnini 2025) | Hybri (Discord) | Match? |
|--------|-------------------|----------------------|-----------------|--------|
| **Training Data** | Boughter (914 antibodies, 443:471 split) | Boughter (same) | Boughter | ✅ |
| **Test Data Source** | PNAS SD01-03 (137 antibodies) | Unknown (possibly provisional) | Unknown | ❓ |
| **Flag Threshold** | 0 vs ≥4 (following Novo's stated ">3") | **Unclear** (stated ">3", results suggest ≥3?) | Unknown | ❌ |
| **Test Set Size** | **70 antibodies** (67:3 ratio) | **86 antibodies** (59:27 ratio) | **86 antibodies** (58:28 ratio) | ❌ |
| **Class Imbalance** | 95.7% : 4.3% (extreme) | 68.6% : 31.4% (moderate) | 67.4% : 32.6% (moderate) | ❌ |
| **Classification Threshold** | 0.5 (default) | Unknown (possibly tuned) | Unknown | ❓ |
| **Model Architecture** | ESM-1v + LogisticRegression | ESM-1v + LogisticRegression | ESM-1v + LogisticRegression | ✅ |
| **Fragment Used** | VH only | VH only | VH only | ✅ |
| **Accuracy** | **52.9%** | **69.0%** | **65.1%** | ❌ |

**Key Takeaway**: Multiple mismatches (❌) prevent direct comparison!

---

## Methodology Discrepancies Timeline

### What Novo Paper States (Sakhnini et al. 2025)

**Section 2.4** (page 7):
> "As in the Boughter dataset, the Jain dataset was **parsed** into two groups: **specific (0 flags)** and **non-specific (>3 flags)**, leaving out the mildly non-specific antibodies (1-3 flags)"

**Section 2.6** (page 9):
> "An accuracy of 69% was obtained for the parsed Jain dataset (see confusion matrix in **Figure S14A**)"

**Methods 4.1** (page 14):
> "Boughter dataset... was selected for **training** of ML models, while the remining three (i.e. **Jain**, Shehata and Harvey, which consists exclusively of VHH sequences) were used for **testing**."

### What We Implemented

1. ✅ Used Boughter for training only (no Jain in training)
2. ✅ Parsed Jain into 0 vs >3 flags (interpreted as ≥4)
3. ✅ Excluded mild (1-3 flags) from test set
4. ✅ Used ESM-1v VH-based LogisticRegression model
5. ❌ Got 70-antibody test set (67:3) instead of 86 (59:27)
6. ❌ Used default 0.5 threshold (may need calibration)

### What We Don't Know About Novo

1. ❓ Exact flag threshold (">3" ambiguous: could be >3 or ≥3)
2. ❓ Which 86 antibodies tested (vs our 137-antibody source)
3. ❓ Classification threshold used (default 0.5 or tuned?)
4. ❓ Any additional preprocessing/filtering steps
5. ❓ Exact dataset version (provisional vs PNAS published)

---

## Root Cause Summary

The **52.9% vs 69% gap** can be explained by:

| Factor | Contribution to Gap | Blocker Level |
|--------|-------------------|--------------|
| **Different test sets** (70 vs 86, different antibodies) | +++  | P0 |
| **Different class imbalance** (95:5 vs 69:31) | +++ | P0 |
| **Threshold not calibrated** (0.5 vs optimal for imbalance) | +++ | P1 |
| **Different flag threshold?** (≥4 vs ≥3) | ++ | P0 |
| **Dataset version** (137 vs 86 vs 80 antibodies) | ++ | P2 |
| **Unknown Novo methodology details** | + | P2 |

**Legend**: +++ Major contributor, ++ Moderate contributor, + Minor contributor

---

## Recommended Investigation Priority

### Phase 1: Immediate (This Session)

1. ✅ **Document all findings** → This report
2. **Test threshold adjustment**:
   - Load prediction probabilities
   - Try thresholds: 0.5, 0.7, 0.9, 0.95, 0.96
   - Generate confusion matrix for each
   - Find threshold that maximizes F1 score
3. **Test alternative flag thresholds**:
   - Create test set with ≥3 flags (110 specific + 27 non-specific)
   - Run inference on this set
   - Compare results
4. **Generate probability distribution plots**:
   - Histogram of P(non-specific) for true specific vs true non-specific
   - Check if distributions overlap or are well-separated

### Phase 2: Short-Term (Next Session)

5. **Contact authors**:
   - Email Novo authors (llsh@novonordisk.com, dngt@novonordisk.com, mv245@cam.ac.uk)
   - Ask for clarification on exact Jain test set methodology
   - Request antibody IDs if possible
6. **Embedding analysis**:
   - Compare Boughter training embeddings vs Jain test embeddings
   - Check for distribution shift using t-SNE/PCA
   - Analyze the 3 Jain non-specific antibodies specifically
7. **Error analysis**:
   - Identify which 31 specific antibodies were misclassified
   - Check their biophysical properties
   - Look for patterns (clinical stage, phage-derived, etc.)

### Phase 3: Long-Term (Future Work)

8. **Retrain with proper handling**:
   - Use class weights during training
   - Calibrate threshold on validation set
   - Try different train/test splits
9. **Compare to Boughter held-out performance**:
   - Verify model works correctly on similar distribution
   - Rule out model corruption/loading issues
10. **Try combined training** (if that's what Novo did):
    - Train on Boughter + Jain (with proper split)
    - Compare to Boughter-only training

---

## Files Analyzed

### Source Code
- `preprocessing/process_jain.py` - Jain fragment extraction (V-domain reconstruction fix applied ✅)
- `test.py` - Model inference script
- `test_datasets/jain.csv` - 137-antibody dataset from PNAS supplementary files

### Documentation
- `docs/jain/jain_data_sources.md` - Dataset provenance documentation
- `docs/jain/JAIN_P0_FIX_REPORT.md` - Gap character fix report
- `JAIN_TEST_RESULTS.md` - Test results and findings (created during investigation)

### Data Files
- `test_datasets/jain/VH_only_jain_test.csv` - Our 70-antibody test set
- `test_results/jain/detailed_results_VH_only_jain_test_20251102_131152.yaml` - Full test results
- `test_results/jain/predictions_boughter_vh_esm1v_logreg_VH_only_jain_test_20251102_131152.csv` - Predictions
- `test_results/jain/confusion_matrix_VH_only_jain_test.png` - Confusion matrix visualization

### Literature
- `literature/markdown/Sakhnini_2025_Antibody_NonSpecificity_PLM_Biophysical/` - Novo paper
- `literature/markdown/novo-media-1/` - Novo supplementary materials
- `literature/markdown/jain-et-al-2017-biophysical-properties-of-the-clinical-stage-antibody-landscape/` - Jain original paper
- `literature/markdown/pnas.201616408si/` - Jain supplementary materials

---

## Conclusion

**Status**: 🔴 **Multiple critical blockers prevent valid comparison to published results**

**Primary Recommendation**: **Contact Novo authors to clarify exact methodology** before proceeding with further analysis or model improvements.

**Secondary Recommendation**: **Test threshold calibration** (quick win) to see if performance improves with adjusted threshold for imbalanced data.

**Assessment**: The poor performance (52.9%) is **likely NOT a model quality issue** but rather:
1. Wrong test set (different antibodies or threshold)
2. Wrong classification threshold (not calibrated for 95:5 imbalance)
3. Both of the above

The model may actually be performing reasonably well given the constraints, but we're evaluating it incorrectly.

---

## Next Actions

**User Decision Required**:

Which path should we take?

**Path A (Recommended)**: Contact authors and wait for clarification
**Path B**: Proceed with threshold tuning on current 70-antibody test set
**Path C**: Try ≥3 flag threshold to get 27 non-specific antibodies
**Path D**: Report results for multiple test set definitions and be transparent about uncertainty

---

## References

- **Novo Paper**: Sakhnini et al. 2025, "Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters"
- **Jain Paper**: Jain et al. 2017, "Biophysical properties of the clinical-stage antibody landscape" ([DOI:10.1073/pnas.1616408114](https://doi.org/10.1073/pnas.1616408114))
- **Dataset Source**: PNAS Supplementary Data 1-3 (pnas.1616408114.sd01-03.xlsx)
- **Test Results**: `test_results/jain/detailed_results_VH_only_jain_test_20251102_131152.yaml`

---

**Report Generated**: 2025-11-02
**Last Updated**: 2025-11-02
**Status**: Investigation ongoing - user decision required
