# Getting Started

This 5-minute quickstart will get you running a simple training experiment to verify your installation and familiarize yourself with the pipeline.

---

## Overview

This quickstart will:

1. ✅ Verify your installation works
2. ✅ Train a model on the Boughter dataset (914 VH sequences)
3. ✅ Test on the Jain dataset (86 clinical antibodies)
4. ✅ Generate performance metrics

**Time:** ~5-10 minutes (depending on hardware)

---

## Prerequisites

- ✅ Completed [Installation Guide](installation.md)
- ✅ Virtual environment activated (`.venv`)
- ✅ Internet connection (for downloading ESM-1v model on first run)

---

## Quick Start: Train Your First Model

### Step 1: Verify Installation

```bash
# Check that commands are available
uv run antibody-train --help
uv run antibody-test --help
```

You should see help messages for both commands.

### Step 2: Review Default Configuration

The pipeline includes a default Hydra configuration for Novo Nordisk parity validation:

```bash
cat src/antibody_training_esm/conf/config.yaml
```

**Key settings:**

- **Training Dataset:** Boughter (914 VH sequences, ELISA assay)
- **Test Dataset:** Jain (86 clinical antibodies, ELISA assay)
- **Model:** ESM-1v (facebook/esm1v_t33_650M_UR90S_1)
- **Classifier:** Logistic Regression (C=1.0, max_iter=1000)

### Step 3: Train the Model

Run training with the default Hydra configuration:

```bash
make train
# OR
uv run antibody-train
```

**What happens:**

1. **Download ESM-1v** - Downloads ~700 MB model from HuggingFace (first run only)
2. **Extract Embeddings** - Generates 1280-dimensional embeddings for all sequences
3. **Cache Embeddings** - Saves embeddings to `experiments/cache/` (SHA-256 hashed)
4. **10-Fold Cross-Validation** - Trains and evaluates on Boughter dataset
5. **Train Final Model** - Trains on full Boughter dataset
6. **Test on Jain** - Evaluates on hold-out test set
7. **Save Model** - Saves trained model to `experiments/checkpoints/`

**Expected output:**

```
✅ Loaded 914 training samples from Boughter dataset
✅ Loaded 86 test samples from Jain dataset
✅ Extracted embeddings (shape: 914 x 1280)
✅ 10-Fold Cross-Validation:
   - Accuracy: 71.2% ± 3.5%
   - Precision: 68.3% ± 4.2%
   - Recall: 72.1% ± 5.1%
✅ Test Set (Jain):
   - Accuracy: 68.60% - EXACT NOVO PARITY
   - Confusion Matrix: [[40, 17], [10, 19]]
✅ Model saved to: experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl
```

**Training Time:**

- **CPU:** ~10-15 minutes
- **GPU (CUDA/MPS):** ~3-5 minutes

### Step 4: Verify Results

Check that outputs were created:

```bash
# Trained model
ls -lh experiments/checkpoints/**/*.pkl

# Cached embeddings
ls -lh experiments/cache/

# Logs (Hydra outputs)
ls -lh experiments/runs/*/logs/
```

---

## Understanding the Results

### Cross-Validation Metrics

**10-Fold Cross-Validation** on Boughter dataset:

- **Accuracy: ~71%** - Percentage of correct predictions
- **Precision:** True positives / (true positives + false positives)
- **Recall:** True positives / (true positives + false negatives)
- **F1 Score:** Harmonic mean of precision and recall

These metrics estimate how well the model generalizes to unseen data.

### Test Set Performance (Jain Dataset)

**Accuracy: 68.60%** - EXACT NOVO PARITY (matches Figure S14A)

**Our Confusion Matrix:**
```
[[40, 17],   ← True Negatives: 40, False Positives: 17
 [10, 19]]   ← False Negatives: 10, True Positives: 19
```

**Novo's Target (Figure S14A):** [[40, 17], [10, 19]] - IDENTICAL to our result.

---

## What Just Happened?

### 1. ESM-1v Embedding Extraction

The pipeline loaded the pre-trained ESM-1v protein language model and generated 1280-dimensional embeddings for each antibody sequence. These embeddings capture:

- **Evolutionary information** - Patterns learned from 250M protein sequences
- **Structural information** - Predicted secondary structure and contacts
- **Functional properties** - Biophysical characteristics encoded in the sequence

### 2. Logistic Regression Classification

A simple logistic regression classifier was trained on the embeddings to predict:

- **Specific (class 0):** Antibody binds only to intended target
- **Non-Specific (class 1):** Antibody exhibits polyreactivity (binds to unintended targets)

### 3. Embedding Caching

Embeddings were cached to `experiments/cache/` with SHA-256 hashed filenames. This enables:

- **Fast hyperparameter sweeps** - No need to re-extract embeddings
- **Automatic invalidation** - Cache updates when model/data changes

---

## Next Steps

Now that you've trained your first model, explore:

### 1. Training Custom Models

See [Training Guide](training.md) to:

- Train on different datasets (Harvey, Shehata)
- Tune hyperparameters (C, max_iter, penalty)
- Run hyperparameter sweeps

### 2. Testing on New Data

See [Testing Guide](testing.md) to:

- Evaluate trained models on hold-out test sets
- Test on fragment-level sequences (CDRs, FWRs)
- Compare performance across assays (ELISA vs PSR)

### 3. Preprocessing New Datasets

See [Preprocessing Guide](preprocessing.md) to:

- Prepare your own antibody datasets
- Convert Excel/CSV to canonical format
- Extract sequence fragments (VH, CDRs, FWRs)

### 4. Troubleshooting

See [Troubleshooting Guide](troubleshooting.md) if you encounter:

- MPS memory issues on Apple Silicon
- CUDA out-of-memory errors
- Cache invalidation problems

---

## Common Quick Start Issues

### Issue: ESM-1v download fails

**Symptoms:** `ConnectionError` or `HTTPError` during model download

**Solution:**

```bash
# Set HuggingFace cache directory (if needed)
export HF_HOME=/path/to/cache

# Retry download
uv run antibody-train
```

### Issue: Out of memory during embedding extraction

**Symptoms:** `RuntimeError: CUDA out of memory` or `MPS out of memory`

**Solution:** Reduce batch size via CLI override:

```bash
# Override batch size from CLI
uv run antibody-train training.batch_size=8
```

Or edit `src/antibody_training_esm/conf/config.yaml`:

```yaml
# src/antibody_training_esm/conf/config.yaml
training:
  batch_size: 8  # Reduce from default
```

### Issue: Training takes too long

**Solution:** Use GPU acceleration:

```bash
# Verify GPU is available
uv run python -c "import torch; print(torch.cuda.is_available())"  # CUDA
uv run python -c "import torch; print(torch.backends.mps.is_available())"  # MPS

# GPU should auto-detect, but can force with:
export PYTORCH_ENABLE_MPS_FALLBACK=1  # macOS
```

---

## Understanding the Pipeline

For a deeper understanding of how the pipeline works:

- **System Architecture:** See [System Overview](../overview.md)
- **Core Components:** See [CLAUDE.md](../../CLAUDE.md) (Architecture section)
- **Research Methodology:** See [Research - Methodology](../research/methodology.md)

---

**Last Updated:** 2025-11-18
**Branch:** `dev`
