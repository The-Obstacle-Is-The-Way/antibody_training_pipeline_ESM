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

## Understanding Dataset File Types

Before testing, it's important to understand the two types of CSV files in the pipeline:

### Canonical Files vs Fragment Files

**Fragment Files** (`test_datasets/{dataset}/fragments/*.csv`) - **RECOMMENDED**:
- Standardized column names: `sequence`, `label`
- Ready for testing with default CLI (no config override needed)
- Created by preprocessing scripts
- **Use these for most testing workflows**

**Canonical Files** (`test_datasets/{dataset}/canonical/*.csv`) - **ADVANCED**:
- Original column names from source data (`vh_sequence`, `vl_sequence`)
- Includes all metadata (flags, PSR scores, etc.)
- Requires config override with `sequence_column: "vh_sequence"`
- Use for custom analysis requiring full metadata

**Which to use?**
- **Quick testing:** Use fragment files (work with `--model` and `--data` CLI flags)
- **Metadata analysis:** Use canonical files with test config YAML

---

## Quick Testing Commands

### Test with Model and Data Paths (Recommended)

```bash
# Test trained model on Jain dataset (using fragment file)
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/jain/fragments/VH_only_jain.csv
```

**Note:** Fragment files have standardized `sequence` column - no config override needed.

### Test with Configuration File

```bash
# Create sample test config
uv run antibody-test --create-config

# Test using config
uv run antibody-test --config test_config.yaml
```

**Example test_config.yaml (fragment file):**

```yaml
model_paths:
  - "models/boughter_vh_esm1v_logreg.pkl"

data_paths:
  - "test_datasets/jain/fragments/VH_only_jain.csv"  # Fragment file

output_dir: "./test_results"
device: "auto"  # or "cpu", "cuda", "mps"
batch_size: 16
```

---

## Test Dataset Options

The pipeline includes three test datasets with preprocessed fragment files:

### Jain Dataset (Novo Parity Benchmark)

```bash
# Using fragment file (recommended - standardized columns)
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/jain/fragments/VH_only_jain.csv
```

**Details:**

- **Size:** 137 antibodies (fragment file includes full Jain dataset)
  - 86 antibodies from P5e-S2 subset (Novo parity benchmark)
- **Assay:** ELISA (per-antigen binding)
- **Fragment:** VH
- **File:** `test_datasets/jain/fragments/VH_only_jain.csv` (standardized `sequence` column)
- **Alternative:** `test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv` (86 only, requires config override)
- **Expected Accuracy:** ~66% on P5e-S2 subset (Novo Nordisk parity)

---

### Harvey Dataset (Nanobodies)

```bash
# Test on full VHH sequences
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/harvey/fragments/VHH_only_harvey.csv
```

**Details:**

- **Size:** 141,021 nanobody sequences
- **Assay:** PSR (polyspecific reagent)
- **Fragment:** VHH_only (full nanobody VHH domain)
- **File:** `test_datasets/harvey/fragments/VHH_only_harvey.csv`
- **Note:** Large-scale test, may take 10-30 minutes

**Fragment-Level Testing:**

```bash
# Test on VHH CDRs only
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/harvey/fragments/H-CDRs_harvey.csv
```

**Available Harvey Fragments:**

- `VHH_only_harvey.csv` - Full VHH domain
- `H-CDR1_harvey.csv`, `H-CDR2_harvey.csv`, `H-CDR3_harvey.csv` - Individual CDRs
- `H-CDRs_harvey.csv` - Concatenated CDRs
- `H-FWRs_harvey.csv` - Concatenated FWRs

---

### Shehata Dataset (PSR Cross-Validation)

```bash
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/shehata/fragments/VH_only_shehata.csv
```

**Details:**

- **Size:** 398 human antibodies
- **Assay:** PSR (polyspecific reagent)
- **Fragment:** VH
- **File:** `test_datasets/shehata/fragments/VH_only_shehata.csv`
- **Note:** Cross-assay validation (train ELISA, test PSR)

---

## Fragment-Level Testing

All datasets provide fragment-specific CSV files. Test on specific antibody regions:

### Shehata Fragments (Most Complete)

```bash
# Test on H-CDRs (Heavy Chain CDRs)
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/shehata/fragments/H-CDRs_shehata.csv

# Test on All-CDRs (Heavy + Light)
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/shehata/fragments/All-CDRs_shehata.csv

# Test on H-FWRs (Heavy Framework Regions)
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/shehata/fragments/H-FWRs_shehata.csv

# Test on combined VH+VL
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/shehata/fragments/VH+VL_shehata.csv
```

**Available Shehata Fragments:**

- `VH_only_shehata.csv`, `VL_only_shehata.csv` - Variable domains
- `H-CDR1_shehata.csv`, `H-CDR2_shehata.csv`, `H-CDR3_shehata.csv` - Heavy CDRs
- `L-CDR1_shehata.csv`, `L-CDR2_shehata.csv`, `L-CDR3_shehata.csv` - Light CDRs
- `H-CDRs_shehata.csv`, `L-CDRs_shehata.csv`, `All-CDRs_shehata.csv` - Concatenated CDRs
- `H-FWRs_shehata.csv`, `L-FWRs_shehata.csv`, `All-FWRs_shehata.csv` - Framework regions
- `VH+VL_shehata.csv`, `Full_shehata.csv` - Combined sequences

### Boughter Fragments (Training Set)

```bash
# Test on training set fragments (for cross-validation)
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data train_datasets/boughter/annotated/VH_only_boughter.csv
```

**Available Boughter Fragments:**

- `train_datasets/boughter/annotated/VH_only_boughter.csv` - VH domain (914 sequences)
- `train_datasets/boughter/annotated/H-CDRs_boughter.csv` - Heavy CDRs
- `train_datasets/boughter/annotated/All-CDRs_boughter.csv` - All CDRs
- (See `train_datasets/boughter/annotated/` for all 16 fragments)

---

## Using Canonical Files (Advanced)

Canonical files preserve original column names and full metadata. To use them, create a test config:

**Example: Test with Jain canonical file**

```yaml
# test_config_jain_canonical.yaml
model_paths:
  - "models/boughter_vh_esm1v_logreg.pkl"

data_paths:
  - "test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv"

sequence_column: "vh_sequence"  # Override for canonical file
label_column: "label"
output_dir: "./test_results"
device: "auto"
batch_size: 16
```

Then run:

```bash
uv run antibody-test --config test_config_jain_canonical.yaml
```

**Why the override?**
- Canonical files use `vh_sequence` instead of `sequence` (original source data columns)
- Fragment files use standardized `sequence` column (preprocessed for training/testing)
- Config override tells the CLI which column to read

**When to use canonical files:**
- Access to full metadata (ELISA flags, PSR scores, source annotations)
- Reproducing exact paper methodology with original data structure
- Custom analysis requiring features beyond sequence + label

---

## Understanding Test Results

### Standard Output

```
✅ Loaded model: models/boughter_vh_esm1v_logreg.pkl
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

Training on ELISA (Boughter) and testing on PSR (Harvey/Shehata) requires **assay-specific threshold tuning**.

**Method 1: Test Configuration (Recommended)**

Create a test config with PSR-specific threshold:

```yaml
# test_config_psr.yaml
model_paths:
  - "models/boughter_vh_esm1v_logreg.pkl"

data_paths:
  - "test_datasets/shehata/fragments/VH_only_shehata.csv"

output_dir: "./test_results"
device: "auto"
batch_size: 16

# PSR assay-specific threshold
threshold: 0.5495  # Novo Nordisk PSR threshold (default ELISA: 0.5)
```

```bash
uv run antibody-test --config test_config_psr.yaml
```

**Method 2: Manual Threshold Adjustment (Python)**

Load model and adjust threshold manually:

```python
import pickle
import numpy as np
from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor

# Load model
with open("models/boughter_vh_esm1v_logreg.pkl", "rb") as f:
    classifier = pickle.load(f)

# Extract embeddings for test data
extractor = ESMEmbeddingExtractor(
    model_name="facebook/esm1v_t33_650M_UR90S_1",
    device="auto",
    batch_size=16
)
test_embeddings = extractor.extract_embeddings(test_sequences)

# Get prediction probabilities
probs = classifier.predict_proba(test_embeddings)[:, 1]

# Apply PSR-specific threshold
psr_threshold = 0.5495  # Novo Nordisk PSR threshold
predictions = (probs > psr_threshold).astype(int)
```

**Why different thresholds?**

- **ELISA threshold:** 0.5 (standard)
- **PSR threshold:** 0.5495 (empirically derived for Novo parity)
- Assays measure different binding properties

See [Research Notes - Assay-Specific Thresholds](../research/assay-thresholds.md) for details.

---

## Batch Testing (Multiple Datasets)

**Method 1: Multiple Data Paths (Recommended)**

Test a single model on multiple datasets in one command:

```bash
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data \
    test_datasets/jain/fragments/VH_only_jain.csv \
    test_datasets/shehata/fragments/VH_only_shehata.csv \
    test_datasets/harvey/fragments/VHH_only_harvey.csv
```

**Method 2: Test Configuration File**

```yaml
# test_config_multi.yaml
model_paths:
  - "models/boughter_vh_esm1v_logreg.pkl"

data_paths:
  - "test_datasets/jain/fragments/VH_only_jain.csv"
  - "test_datasets/shehata/fragments/VH_only_shehata.csv"
  - "test_datasets/harvey/fragments/VHH_only_harvey.csv"

output_dir: "./test_results"
```

```bash
uv run antibody-test --config test_config_multi.yaml
```

**Method 3: Shell Loop**

```bash
# Test on multiple datasets sequentially
for data_file in \
  test_datasets/jain/fragments/VH_only_jain.csv \
  test_datasets/shehata/fragments/VH_only_shehata.csv; do
  echo "Testing on $data_file..."
  uv run antibody-test \
    --model models/boughter_vh_esm1v_logreg.pkl \
    --data "$data_file"
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
  --data /path/to/my_test_data.csv
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

**Method 1: Multiple Models on Same Dataset**

Compare performance of different models on same test set:

```bash
uv run antibody-test \
  --model \
    models/boughter_vh_esm1v_logreg.pkl \
    models/boughter_vh_esm2_logreg.pkl \
  --data test_datasets/jain/fragments/VH_only_jain.csv
```

**Method 2: Test Configuration**

```yaml
# test_config_compare.yaml
model_paths:
  - "models/boughter_vh_esm1v_logreg.pkl"
  - "models/boughter_vh_esm2_logreg.pkl"

data_paths:
  - "test_datasets/jain/fragments/VH_only_jain.csv"

output_dir: "./test_results"
```

**Method 3: Compare Fragment Performance**

Test same model on different fragments to evaluate which regions are most predictive:

```bash
# Compare VH vs CDRs vs FWRs performance
for fragment_file in \
  VH_only_shehata.csv \
  H-CDRs_shehata.csv \
  H-FWRs_shehata.csv; do
  echo "Testing on $fragment_file..."
  uv run antibody-test \
    --model models/boughter_vh_esm1v_logreg.pkl \
    --data test_datasets/shehata/fragments/$fragment_file
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
file models/boughter_vh_esm1v_logreg.pkl
```

---

### Issue: Sequence column not found in test CSV

**Symptoms:** `KeyError: 'sequence'`

**Solution:** Ensure test CSV has standardized `sequence` column:

```bash
# Check CSV structure (use fragment file)
head -n 5 test_datasets/jain/fragments/VH_only_jain.csv

# Expected format:
# id,sequence,label,elisa_flags,source
# abituzumab,QVQLQQSGGELAKPGASVKVSCKASGYTFSSFWMHWVRQAPGQGLEWIGYINPRSGYTEYNEIFRDKATMTTDTSTSTAYMELSSLRSEDTAVYYCASFLGRGAMDYWGQGTTVTVSS,0.0,0,jain2017_pnas
```

**Note:** Fragment CSVs from preprocessing already have standardized `sequence` column. Canonical CSVs use `vh_sequence` instead (see "Using Canonical Files" section above for config override).

---

### Issue: Poor test performance

**Symptoms:** Accuracy < 60% on Jain dataset

**Possible causes:**

1. **Model trained on different fragment:** Train on VH, test on VH (not CDRs/FWRs)
2. **Cross-dataset generalization:** Models trained on one dataset may not generalize to others
3. **Assay mismatch:** ELISA ≠ PSR (adjust threshold to 0.5495 for PSR)
4. **Overfitting:** High train CV, low test accuracy (increase regularization in training config)

See [Troubleshooting Guide](troubleshooting.md) for detailed debugging.

---

### Issue: Test takes too long (large datasets)

**Solution:** Use GPU acceleration:

```bash
# Verify GPU available
uv run python -c "import torch; print(torch.cuda.is_available())"

# Force GPU usage
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/harvey/fragments/VHH_only_harvey.csv \
  --device cuda
```

Or reduce batch size:

```bash
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/harvey/fragments/VHH_only_harvey.csv \
  --batch-size 8  # Reduce from default (16)
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
