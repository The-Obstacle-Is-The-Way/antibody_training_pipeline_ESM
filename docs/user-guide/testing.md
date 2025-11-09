# Testing Guide

This guide covers how to evaluate trained antibody non-specificity prediction models on test datasets.

---

## Overview

Testing involves:

1. **Load Trained Model** - Load a previously trained `.pkl` model
2. **Load Test Data** - Load test dataset (CSV format)
3. **Extract Embeddings** - Generate ESM-1v embeddings for test sequences
4. **Predict** - Classify sequences as specific (0) or non-specific (1)
5. **Evaluate** - Compute performance metrics (accuracy, precision, recall, confusion matrix)

---

## Quick Testing Commands

### Test with Default Config

If you trained with default config, test on Jain dataset:

```bash
uv run antibody-test \
  --model models/boughter_train_jain_test_vh.pkl \
  --dataset jain
```

### Test with Custom Dataset

```bash
uv run antibody-test \
  --model models/my_model.pkl \
  --test-file test_datasets/harvey/fragments/harvey_VHH_only.csv \
  --fragment VHH_only
```

---

## Test Dataset Options

The pipeline includes three pre-configured test datasets:

### Jain Dataset (Novo Parity Benchmark)

```bash
uv run antibody-test --model models/boughter_train_jain_test_vh.pkl --dataset jain
```

**Details:**

- **Size:** 86 clinical antibodies
- **Assay:** ELISA (per-antigen binding)
- **Fragment:** VH
- **Expected Accuracy:** 66.28% (Novo Nordisk exact parity)
- **Expected Confusion Matrix:** [[40, 19], [10, 17]]

---

### Harvey Dataset (Nanobodies)

```bash
uv run antibody-test --model models/boughter_harvey_vhh.pkl --dataset harvey
```

**Details:**

- **Size:** 141,021 nanobody sequences
- **Assay:** PSR (polyspecific reagent)
- **Fragment:** VHH_only (full nanobody VHH domain)
- **Note:** Large-scale test, may take 10-30 minutes

**Fragment-Level Testing:**

```bash
# Test on VHH CDRs only
uv run antibody-test \
  --model models/boughter_harvey_vhh_cdrs.pkl \
  --test-file test_datasets/harvey/fragments/harvey_VHH-CDRs.csv \
  --fragment VHH-CDRs
```

---

### Shehata Dataset (PSR Cross-Validation)

```bash
uv run antibody-test --model models/boughter_shehata_vh.pkl --dataset shehata
```

**Details:**

- **Size:** 398 human antibodies
- **Assay:** PSR (polyspecific reagent)
- **Fragment:** VH
- **Note:** Cross-assay validation (train ELISA, test PSR)

---

## Fragment-Level Testing

Test models on specific antibody fragments:

### CDR Testing

```bash
# Test on H-CDRs (Heavy Chain CDRs)
uv run antibody-test \
  --model models/boughter_jain_h_cdrs.pkl \
  --test-file test_datasets/jain/fragments/jain_H-CDRs.csv \
  --fragment H-CDRs

# Test on All-CDRs (Heavy + Light)
uv run antibody-test \
  --model models/boughter_jain_all_cdrs.pkl \
  --test-file test_datasets/jain/fragments/jain_All-CDRs.csv \
  --fragment All-CDRs
```

---

### FWR Testing (Framework Regions)

```bash
# Test on H-FWRs (Heavy Framework Regions)
uv run antibody-test \
  --model models/boughter_jain_h_fwrs.pkl \
  --test-file test_datasets/jain/fragments/jain_H-FWRs.csv \
  --fragment H-FWRs
```

---

### VH + VL Combined Testing

```bash
# Test on combined VH+VL sequences
uv run antibody-test \
  --model models/boughter_jain_vh_vl.pkl \
  --test-file test_datasets/jain/fragments/jain_VH_VL.csv \
  --fragment VH_VL
```

---

## Understanding Test Results

### Standard Output

```
✅ Loaded model: models/boughter_train_jain_test_vh.pkl
✅ Loaded test data: 86 samples
✅ Extracted embeddings (86 x 1280)
✅ Predictions complete

Test Set Performance:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Accuracy:        66.28%
Precision:       47.22%
Recall:          62.96%
F1 Score:        54.05%
ROC-AUC:         0.6384
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Confusion Matrix:
              Predicted
              Neg    Pos
Actual  Neg  [40     19]
        Pos  [10     17]

Classification Report:
              precision    recall  f1-score   support
           0       0.80      0.68      0.73        59
           1       0.47      0.63      0.54        27
    accuracy                           0.66        86
   macro avg       0.64      0.65      0.64        86
weighted avg       0.69      0.66      0.67        86
```

---

### Interpreting Metrics

**Accuracy: 66.28%**

- Percentage of correct predictions
- **Baseline:** Random guessing = ~50%
- **Novo Parity:** 66.28% matches published result

---

**Precision: 47.22%**

- Of predicted non-specific, 47% are truly non-specific
- Low precision = many false positives
- **Interpretation:** Model is conservative (predicts non-specific often)

---

**Recall: 62.96%**

- Of truly non-specific, 63% were detected
- Moderate recall = misses some non-specific antibodies
- **Interpretation:** Model catches majority but not all

---

**F1 Score: 54.05%**

- Harmonic mean of precision and recall
- Balances false positives and false negatives
- **Interpretation:** Moderate overall performance

---

**ROC-AUC: 0.6384**

- Area under ROC curve
- **0.5 = random**, **1.0 = perfect**
- **0.64 = weak positive discrimination**

---

**Confusion Matrix:**

```
              Predicted
              Neg    Pos
Actual  Neg  [40     19]   ← True Neg: 40, False Pos: 19
        Pos  [10     17]   ← False Neg: 10, True Pos: 17
```

**Key Observations:**

- **True Negatives (40):** Correctly identified specific antibodies
- **False Positives (19):** Specific antibodies mislabeled as non-specific
- **False Negatives (10):** Non-specific antibodies mislabeled as specific
- **True Positives (17):** Correctly identified non-specific antibodies

**Class Imbalance:** 59 specific vs 27 non-specific (2.2:1 ratio)

- High precision on class 0 (80%) vs low precision on class 1 (47%)
- Model biased toward predicting "specific" (majority class)

---

## Cross-Assay Testing

### ELISA → PSR Prediction

Training on ELISA (Boughter) and testing on PSR (Harvey/Shehata) requires **assay-specific threshold tuning**:

```python
# Adjust threshold for PSR assay
classifier = BinaryClassifier.load("models/boughter_train_jain_test_vh.pkl")
classifier.threshold = 0.5495  # Novo Nordisk PSR threshold

# Predict on PSR dataset
predictions = classifier.predict(test_embeddings)
```

**Why different thresholds?**

- **ELISA threshold:** 0.5 (standard)
- **PSR threshold:** 0.5495 (empirically derived for Novo parity)
- Assays measure different binding properties

See [Research Notes - Assay-Specific Thresholds](../research/ASSAY_SPECIFIC_THRESHOLDS.md) for details.

---

## Batch Testing (Multiple Datasets)

Test a single model on multiple datasets:

```bash
# Test on all three datasets
for dataset in jain harvey shehata; do
  echo "Testing on $dataset..."
  uv run antibody-test \
    --model models/boughter_train_jain_test_vh.pkl \
    --dataset $dataset
done
```

---

## Custom CSV Testing

Test on your own antibody dataset:

### CSV Format Requirements

```csv
sequence,label
EVQLVESGGGLVQPGGSLRLSCAASGFTFS,0
QVQLQESGPGLVKPSQTLSLTCTVSGGSLS,1
```

**Required columns:**

- `sequence`: Antibody amino acid sequence
- `label`: Ground truth (0=specific, 1=non-specific)

**Optional columns:**

- `id`: Antibody identifier
- `name`: Antibody name
- `source`: Data source

---

### Test Command

```bash
uv run antibody-test \
  --model models/my_model.pkl \
  --test-file /path/to/my_test_data.csv \
  --fragment VH
```

---

## Performance Benchmarking

### Time and Memory Usage

**Small Dataset (Jain, 86 sequences):**

- **CPU:** ~30 seconds
- **GPU (CUDA/MPS):** ~10 seconds
- **Memory:** ~2 GB

**Large Dataset (Harvey, 141k sequences):**

- **CPU:** ~15-20 minutes
- **GPU (CUDA/MPS):** ~5-8 minutes
- **Memory:** ~8-12 GB

**Tip:** Use GPU for large-scale testing (10x speedup).

---

### Embedding Caching

Test embeddings are cached (same as training):

```
embeddings_cache/
└── {SHA256_hash}.npy
```

**Benefits:**

- Second test run on same dataset = instant
- Cache shared with training (no duplication)

---

## Comparing Models

Compare performance of different models on same test set:

```bash
# Compare VH vs H-CDRs vs All-CDRs
for fragment in VH H-CDRs All-CDRs; do
  echo "Testing $fragment model..."
  uv run antibody-test \
    --model models/boughter_jain_${fragment}.pkl \
    --test-file test_datasets/jain/canonical/jain_p5e_s2.csv \
    --fragment $fragment
done
```

**Expected Ranking (Novo Nordisk findings):**

1. **VH** - Best performance (full variable domain)
2. **H-CDRs** - Moderate performance (binding sites only)
3. **H-FWRs** - Lower performance (structural framework)

---

## Troubleshooting

### Issue: Model fails to load

**Symptoms:** `FileNotFoundError` or `UnpicklingError`

**Solution:**

```bash
# Check model exists
ls -lh models/

# Verify model is valid pickle
file models/boughter_train_jain_test_vh.pkl
```

---

### Issue: Fragment column not found in test CSV

**Symptoms:** `KeyError: 'VH'`

**Solution:** Ensure fragment column matches model training:

```bash
# Check available fragments
head -n 1 test_datasets/jain/canonical/jain_p5e_s2.csv

# Use correct fragment name
uv run antibody-test --model models/my_model.pkl --dataset jain --fragment VH
```

---

### Issue: Poor test performance

**Symptoms:** Accuracy < 60% on Jain dataset

**Possible causes:**

1. **Model trained on wrong fragment:** Train VH, test VH (not VL)
2. **Model trained on different dataset:** Cross-dataset generalization is hard
3. **Assay mismatch:** ELISA ≠ PSR (adjust threshold)
4. **Overfitting:** High train CV, low test (increase regularization)

See [Troubleshooting Guide](troubleshooting.md) for detailed debugging.

---

### Issue: Test takes too long (large datasets)

**Solution:** Use GPU acceleration:

```bash
# Verify GPU available
uv run python -c "import torch; print(torch.cuda.is_available())"

# Force GPU usage
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
uv run antibody-test --model models/my_model.pkl --dataset harvey
```

Or reduce batch size:

```yaml
# In training config (if retraining)
hardware:
  batch_size: 8  # Reduce from 16
```

---

## Advanced Testing

### Prediction Probability Thresholds

Adjust prediction threshold for different precision/recall tradeoffs:

```python
# Load model
from antibody_training_esm.core.classifier import BinaryClassifier
classifier = BinaryClassifier.load("models/boughter_train_jain_test_vh.pkl")

# Get prediction probabilities
probs = classifier.predict_proba(test_embeddings)[:, 1]  # P(non-specific)

# Custom threshold
predictions = (probs > 0.6).astype(int)  # More conservative (higher precision)
```

---

### ROC Curve Analysis

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get probabilities
probs = classifier.predict_proba(test_embeddings)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

# Plot
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Random baseline
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Jain Test Set')
plt.legend()
plt.savefig('roc_curve.png')
```

---

## Next Steps

- **Preprocessing:** See [Preprocessing Guide](preprocessing.md) to prepare new test datasets
- **Training:** See [Training Guide](training.md) to train new models
- **Research Methodology:** See [Research Notes](../research/methodology.md) for scientific validation

---

**Last Updated:** 2025-11-09
**Branch:** `docs/canonical-structure`
