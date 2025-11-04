# Novo Nordisk Benchmark Confusion Matrices

## Overview

This document contains the complete benchmark confusion matrices from Sakhnini et al. (2025) for all three test datasets. These benchmarks represent the published performance of the ESM-1v VH-based LogisticReg model trained on Boughter data.

**Reference:** Sakhnini et al. (2025) "Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters" - Figure S14

## Methodology

**Training data:** Boughter dataset (parsed: 0 flags = specific, >3 flags = non-specific, excluding 1-3 flags)
**Model:** ESM-1v mean-mode VH-based LogisticRegression
**Validation:** 10-fold cross-validation on Boughter
**Test sets:** Jain (clinical antibodies), Shehata (B-cell antibodies), Harvey (nanobodies)

## Confusion Matrix Format

```
                Predicted
                0    1
True    0      TN   FP
        1      FN   TP
```

Where:
- 0 = Specific antibody
- 1 = Non-specific antibody
- TN = True Negative (specific correctly predicted)
- FP = False Positive (specific incorrectly predicted as non-specific)
- FN = False Negative (non-specific incorrectly predicted as specific)
- TP = True Positive (non-specific correctly predicted)

---

## 1. Jain Dataset (Clinical Antibodies)

**Source:** Jain et al. (2017) - 137 clinical-stage IgG1 antibodies
**Assay:** ELISA with panel of 6 ligands (ssDNA, dsDNA, insulin, LPS, cardiolipin, KLH)
**Parsed dataset:** 86 antibodies (0 flags or >3 flags, excluding 1-3 flags)

### Novo Benchmark

```
Confusion Matrix: [[40, 17], [10, 19]]

                Predicted
                Spec  Non-spec
True    Spec      40      17      (57 specific)
        Non-spec  10      19      (29 non-specific)

Total antibodies: 86
```

### Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 68.6% (59/86) |
| Sensitivity (Recall) | 65.5% (19/29) |
| Specificity | 70.2% (40/57) |
| Precision | 52.8% (19/36) |
| F1-Score | 58.5% |

---

## 2. Shehata Dataset (B-cell Antibodies)

**Source:** Shehata et al. (2019) - 398 antibodies from naïve, IgG memory, and long-lived plasma cells
**Assay:** Poly-specific reagent (PSR) assay
**Dataset size:** 398 antibodies (391 specific, 7 non-specific - highly imbalanced)

### Novo Benchmark

```
Confusion Matrix: [[229, 162], [2, 5]]

                Predicted
                Spec  Non-spec
True    Spec     229     162      (391 specific)
        Non-spec   2       5      (7 non-specific)

Total antibodies: 398
```

### Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 58.8% (234/398) |
| Sensitivity (Recall) | 71.4% (5/7) |
| Specificity | 58.6% (229/391) |
| Precision | 3.0% (5/167) |
| F1-Score | 5.7% |

### Notes

- Highly imbalanced dataset (98.2% specific, 1.8% non-specific)
- PSR assay may represent different non-specificity spectrum than ELISA
- Low precision due to class imbalance
- Model tends to over-predict non-specificity

---

## 3. Harvey Dataset (Nanobodies)

**Source:** Harvey et al. (2022) - >140,000 naïve nanobody (VHH) clones
**Assay:** Poly-specific reagent (PSR) assay
**Dataset size:** 141,559 nanobodies

### Novo Benchmark

```
Confusion Matrix: [[19778, 49962], [4186, 67633]]

                Predicted
                Spec    Non-spec
True    Spec    19778     49962      (69,740 specific)
        Non-spec 4186     67633      (71,819 non-specific)

Total nanobodies: 141,559
```

### Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 61.7% (87,411/141,559) |
| Sensitivity (Recall) | 94.2% (67,633/71,819) |
| Specificity | 28.4% (19,778/69,740) |
| Precision | 57.5% (67,633/117,595) |
| F1-Score | 71.4% |

### Notes

- Large-scale dataset (>140k sequences)
- Nearly balanced (49.3% specific, 50.7% non-specific)
- PSR assay (different from ELISA-based Jain/Boughter)
- Model has high sensitivity but low specificity on this dataset
- Nanobodies (VHH) may have different non-specificity patterns than full antibodies

---

## Cross-Dataset Comparison

| Dataset | Size | Balance | Accuracy | Sensitivity | Specificity |
|---------|------|---------|----------|-------------|-------------|
| **Jain** | 86 | 66.3% / 33.7% | **68.6%** | 65.5% | 70.2% |
| **Shehata** | 398 | 98.2% / 1.8% | 58.8% | 71.4% | 58.6% |
| **Harvey** | 141,559 | 49.3% / 50.7% | 61.7% | 94.2% | 28.4% |

### Key Observations

1. **Best performance on Jain:** Balanced dataset with ELISA assay (same as training data)
2. **Poor performance on Shehata:** Extreme class imbalance (7 non-specific out of 398)
3. **Mixed performance on Harvey:** High sensitivity but low specificity; PSR assay difference
4. **Assay dependency:** ELISA-based predictions may not transfer well to PSR-based datasets

---

## Expected Results for Our Model

Our trained model (`boughter_vh_esm1v_logreg.pkl`) was trained on Boughter with identical methodology to Novo. We should expect:

1. **Jain:** Near-identical results (we already achieved 66.28% vs Novo's 68.6%)
2. **Harvey:** Similar confusion matrix pattern to Novo benchmark
3. **Shehata:** Similar poor performance due to extreme class imbalance

---

## Testing Protocol

To validate our model against these benchmarks:

1. Load trained model: `models/boughter_vh_esm1v_logreg.pkl`
2. Extract ESM-1v embeddings for each test dataset
3. Run inference and generate confusion matrices
4. Compare to Novo benchmarks above
5. Calculate performance metrics

---

## References

Sakhnini, L.I. et al. (2025). Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters. *bioRxiv*. Figure S14.

Jain, T. et al. (2017). Biophysical properties of the clinical-stage antibody landscape. *PNAS*, 114(5), 944-949.

Shehata, L. et al. (2019). Affinity maturation enhances antibody specificity but compromises conformational stability. *Cell Reports*, 28(13), 3300-3308.

Harvey, E.P. et al. (2022). An in silico method to assess antibody fragment polyreactivity. *Nat Commun*, 13, 7554.
