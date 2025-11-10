# Training Guide

This guide covers how to train antibody non-specificity prediction models using the pipeline.

---

## Overview

Training involves:

1. **Configuration** - Define datasets, model, classifier, and experiment parameters in YAML
2. **Embedding Extraction** - Generate ESM-1v embeddings for training sequences
3. **Cross-Validation** - Evaluate model performance via 10-fold stratified CV
4. **Final Training** - Train on full training set
5. **Test Evaluation** - Evaluate on hold-out test set
6. **Model Persistence** - Save trained model to `.pkl` file

---

## Quick Training Commands

### Train with Default Config

```bash
make train
# OR
uv run antibody-train --config configs/config.yaml
```

### Train with Custom Config

```bash
uv run antibody-train --config configs/my_experiment.yaml
```

---

## Configuration File Structure

Training is controlled via YAML configuration files in `configs/`. Here's the default config structure:

```yaml
# configs/config.yaml

model:
  name: "facebook/esm1v_t33_650M_UR90S_1"  # ESM-1v model from HuggingFace
  revision: "main"                         # Model revision (for reproducibility)

data:
  train_file: "train_datasets/boughter/canonical/VH_only_boughter_training.csv"
  sequence_column: "sequence"  # Column containing antibody sequences
  label_column: "label"        # Column containing binary labels (0=specific, 1=non-specific)

classifier:
  type: "logistic_regression"
  C: 1.0                      # Regularization strength (inverse)
  penalty: "l2"               # Regularization type (l1, l2, elasticnet, none)
  solver: "lbfgs"             # Optimization algorithm
  max_iter: 1000              # Maximum iterations
  random_state: 42            # Seed for reproducibility
  cv_folds: 10                # Number of cross-validation folds

training:
  save_model: true            # Save trained model to disk
  model_name: "boughter_vh_esm1v_logreg"
  model_save_dir: "./models"

experiment:
  name: "boughter_novo_reproduction"

hardware:
  device: "auto"              # "auto", "cpu", "cuda", "mps"
  batch_size: 16              # Embedding extraction batch size
```

---

## Training on Different Datasets

### Boughter → Jain (Default)

Train on Boughter (914 VH, ELISA), test on Jain (86 clinical, ELISA):

**Training config:**
```yaml
data:
  train_file: "train_datasets/boughter/canonical/VH_only_boughter_training.csv"
  sequence_column: "sequence"
  label_column: "label"
```

**Training:**
```bash
uv run antibody-train --config configs/config.yaml
```

**Testing** (after training):
```bash
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/jain/fragments/VH_only_jain.csv
```

**Expected Accuracy:** ~66.28% (Novo Nordisk exact parity)

---

### Boughter → Harvey (Nanobodies)

Train on Boughter (914 VH, ELISA), test on Harvey (141k nanobodies, PSR):

**Training config:**
```yaml
data:
  train_file: "train_datasets/boughter/canonical/VH_only_boughter_training.csv"
  sequence_column: "sequence"
  label_column: "label"
```

**Training:**
```bash
uv run antibody-train --config configs/config.yaml
```

**Testing** (after training):
```bash
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/harvey/fragments/VHH_only_harvey.csv
```

**Note:** Cross-assay (ELISA → PSR) and cross-species (human antibodies → nanobodies) may reduce performance.

---

### Boughter → Shehata (PSR Cross-Validation)

Train on Boughter (914 VH, ELISA), test on Shehata (398, PSR):

**Training config:**
```yaml
data:
  train_file: "train_datasets/boughter/canonical/VH_only_boughter_training.csv"
  sequence_column: "sequence"
  label_column: "label"
```

**Training:**
```bash
uv run antibody-train --config configs/config.yaml
```

**Testing** (after training):
```bash
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/shehata/fragments/VH_only_shehata.csv
```

**Note:** Cross-assay prediction (ELISA → PSR) requires assay-specific threshold tuning.

---

## Training on Different Fragments

The pipeline supports training on various antibody sequence fragments by using different fragment CSV files.

**How it works:** Fragment files are pre-generated during preprocessing and stored in `train_datasets/{dataset}/annotated/` or `train_datasets/{dataset}/fragments/`.

### Example: Train on Heavy CDRs Only

```yaml
data:
  train_file: "train_datasets/boughter/annotated/H-CDRs_boughter.csv"
  sequence_column: "sequence"
  label_column: "label"
```

```bash
uv run antibody-train --config configs/h_cdrs_config.yaml
```

### Available Fragments

**Boughter (Training Set):**
- `VH_only_boughter_training.csv` - Variable Heavy chain (default)
- `H-CDR1_boughter.csv`, `H-CDR2_boughter.csv`, `H-CDR3_boughter.csv` - Individual Heavy CDRs
- `H-CDRs_boughter.csv` - All Heavy CDRs concatenated
- `H-FWRs_boughter.csv` - Heavy Framework Regions
- (See `train_datasets/boughter/annotated/` for all 16 fragments)

**Fragment Naming Pattern:**
- Format: `{fragmentName}_{dataset}.csv`
- Examples: `VH_only_boughter.csv`, `All-CDRs_jain.csv`, `VHH_only_harvey.csv`

**Note:** Fragment availability depends on dataset. See `docs/datasets/{dataset}/` for preprocessing details and `train_datasets/{dataset}/annotated/` for available files.

---

## Hyperparameter Tuning

### Regularization Strength (C)

**Smaller C = stronger regularization (simpler model)**

```yaml
classifier:
  C: 0.1   # Strong regularization (underfitting risk)
  # OR
  C: 1.0   # Default (balanced)
  # OR
  C: 10.0  # Weak regularization (overfitting risk)
```

**Use cases:**

- **Small datasets:** Use stronger regularization (C=0.1)
- **Large datasets:** Can use weaker regularization (C=10.0)

---

### Regularization Type (penalty)

```yaml
classifier:
  penalty: "l2"          # Ridge (default, works well for most cases)
  # OR
  penalty: "l1"          # Lasso (feature selection, requires solver="liblinear")
  # OR
  penalty: "elasticnet"  # Elastic Net (L1 + L2, requires solver="saga")
  # OR
  penalty: "none"        # No regularization (overfitting risk)
```

**Note:** Penalty type must match solver:

- **l2:** Use `solver: "lbfgs"` (default)
- **l1:** Use `solver: "liblinear"`
- **elasticnet:** Use `solver: "saga"`

---

### Hyperparameter Sweep

For systematic hyperparameter search, see example sweep script:

```bash
# See reference implementation
cat preprocessing/boughter/train_hyperparameter_sweep.py
```

**Sweep strategy:**

1. Define parameter grid (e.g., C=[0.01, 0.1, 1.0, 10.0])
2. Train model for each configuration
3. Embeddings are cached (fast re-runs)
4. Compare cross-validation metrics

---

## Understanding Training Output

### Cross-Validation Metrics

```
✅ 10-Fold Cross-Validation:
   - Accuracy: 71.2% ± 3.5%
   - Precision: 68.3% ± 4.2%
   - Recall: 72.1% ± 5.1%
   - F1 Score: 70.1% ± 3.8%
   - ROC-AUC: 0.75 ± 0.04
```

**Interpretation:**

- **Accuracy:** Overall correct predictions (71.2% on Boughter)
- **Precision:** Of predicted non-specific, how many are truly non-specific
- **Recall:** Of truly non-specific, how many were detected
- **F1 Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under ROC curve (0.5 = random, 1.0 = perfect)

**Standard Deviation (±):** Variability across folds (lower = more stable)

---

### Test Set Metrics

```
✅ Test Set (Jain):
   - Accuracy: 66.28%
   - Confusion Matrix: [[40, 19], [10, 17]]
```

**Confusion Matrix:**

```
                 Predicted
                 Neg    Pos
Actual  Neg     [40     19]   ← True Neg: 40, False Pos: 19
        Pos     [10     17]   ← False Neg: 10, True Pos: 17
```

**Performance Drops:**

- CV accuracy (71%) vs Test accuracy (66%) - **Expected!**
- Cross-dataset generalization is challenging (different assays, antibody sources)
- Novo Nordisk reported 66.28% - we achieve **exact parity**

---

## Training Best Practices

### 1. Start with Default Config

Use the validated default configuration as a baseline:

```bash
make train
```

This ensures reproducibility and provides a reference point for comparisons.

---

### 2. Use Appropriate Test Sets

- **Same-assay testing:** Boughter → Jain (both ELISA)
- **Cross-assay testing:** Boughter → Harvey/Shehata (ELISA → PSR)
- **Match fragment types:** Train on VH, test on VH (not VL)

---

### 3. Monitor Overfitting

If cross-validation accuracy is high but test accuracy is low:

- **Increase regularization:** Decrease `C` (e.g., 1.0 → 0.1)
- **Use L1 penalty:** Feature selection via Lasso
- **Simplify model:** Consider simpler classifier

---

### 4. Leverage Embedding Caching

Embeddings are cached automatically:

```
embeddings_cache/
└── {SHA256_hash}.npy  # Cached embeddings for dataset + model
```

**Benefits:**

- Hyperparameter sweeps run 10-100x faster
- Cache invalidates automatically when data/model changes
- No manual cache management required

**Note:** First run downloads ESM-1v (~700 MB) and extracts embeddings (~5-10 min). Subsequent runs are instant.

---

### 5. Save All Experiments

Always enable model saving:

```yaml
training:
  save_model: true

experiment:
  name: "descriptive_experiment_name"  # Use meaningful names
  output_dir: "models/"
```

Models are saved as:

```
models/{experiment_name}_{fragment}.pkl
```

---

## Advanced Training

### Custom Dataset Paths

```yaml
data:
  train_file: "/absolute/path/to/training_data.csv"
  sequence_column: "sequence"  # Column name for sequences
  label_column: "label"        # Column name for binary labels
```

**CSV Format Requirements:**

- Must have `sequence` column (antibody amino acid sequence)
- Must have `label` column (0=specific, 1=non-specific)
- Column names can be customized via `sequence_column` and `label_column` config keys

---

### Custom ESM Model

Use different ESM model versions:

```yaml
model:
  name: "facebook/esm2_t33_650M_UR50D"  # ESM-2 (newer)
  revision: "main"
```

**Available ESM models:**

- `facebook/esm1v_t33_650M_UR90S_1` - ESM-1v (default, validated)
- `facebook/esm2_t33_650M_UR50D` - ESM-2 (experimental)
- `facebook/esm2_t36_3B_UR50D` - ESM-2 Large (requires 24+ GB GPU)

---

### GPU Memory Management

Reduce batch size if encountering OOM errors:

```yaml
hardware:
  batch_size: 8   # Reduce from default (16)
  device: "cuda"  # or "mps" for Apple Silicon
```

**Memory Requirements:**

| Batch Size | GPU Memory | Speed |
|------------|-----------|-------|
| 4          | 4 GB      | Slow  |
| 8          | 8 GB      | Medium |
| 16         | 12 GB     | Fast (default) |
| 32         | 24 GB     | Fastest |

---

## Troubleshooting

### Training Fails with "Label column not found"

**Solution:** Ensure CSV has `label` column with 0/1 values:

```csv
sequence,label
EVQLVESGGGLV...,0
QVQLQESGPGLV...,1
```

---

### Embeddings Cache Out of Sync

**Solution:** Clear cache and retrain:

```bash
rm -rf embeddings_cache/
uv run antibody-train --config configs/config.yaml
```

---

### Poor Test Performance

**Possible causes:**

1. **Cross-assay mismatch:** Train ELISA, test PSR → tune threshold
2. **Cross-species mismatch:** Train human, test nanobodies → expect lower accuracy
3. **Overfitting:** High CV accuracy, low test accuracy → increase regularization
4. **Underfitting:** Low CV and test accuracy → decrease regularization

See [Troubleshooting Guide](troubleshooting.md) for more solutions.

---

## Next Steps

- **Testing Models:** See [Testing Guide](testing.md) for evaluating trained models
- **Preprocessing Data:** See [Preprocessing Guide](preprocessing.md) for preparing new datasets
- **Hyperparameter Sweeps:** See reference script in `preprocessing/boughter/train_hyperparameter_sweep.py`

---

**Last Updated:** 2025-11-09
**Branch:** `docs/canonical-structure`
