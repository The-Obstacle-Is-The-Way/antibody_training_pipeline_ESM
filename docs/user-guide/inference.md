# Antibody Non-Specificity Prediction: Comprehensive Inference Guide

**Target Audience:** Researchers and developers using the prediction pipeline
**Prerequisites:** Trained model checkpoint (`.pkl`), Python environment set up
**Quick Start:** See root [`INFERENCE_GUIDE.md`](../../INFERENCE_GUIDE.md) for fast reference

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Input Specification](#input-specification)
4. [Command-Line Usage](#command-line-usage)
5. [Output Specification](#output-specification)
6. [Advanced Usage](#advanced-usage)
7. [Batch Processing](#batch-processing)
8. [Troubleshooting](#troubleshooting)
9. [Performance Tuning](#performance-tuning)
10. [Integration with Downstream Tools](#integration-with-downstream-tools)

---

## Overview

The `antibody-predict` CLI provides production-ready inference for screening antibody sequences for non-specificity (polyreactivity). It uses pre-trained ESM-1v protein language models combined with logistic regression or XGBoost classifiers to predict whether an antibody sequence is:
- **Specific**: Low polyreactivity (safe for therapeutic use)
- **Non-specific**: High polyreactivity (potential developability issues)

### Key Features

- ✅ **Pre-trained models**: No training required - use published checkpoints
- ✅ **CSV input/output**: Standard data formats
- ✅ **Batch processing**: Process hundreds of sequences efficiently
- ✅ **GPU acceleration**: Automatic device selection (CUDA, MPS, CPU)
- ✅ **Assay-specific thresholds**: Optimized for ELISA or PSR assays
- ✅ **Flexible column names**: Works with any CSV schema

---

## Prerequisites

### 1. Trained Model Checkpoint

Obtain a trained classifier (`.pkl` file) via:

**Option A: Train Your Own Model**
```bash
make train  # Saves to experiments/checkpoints/esm1v/logreg/
```
See [`training.md`](training.md) for detailed instructions.

**Option B: Download Published Checkpoint**
```bash
# Download from GitHub releases
wget https://github.com/the-obstacle-is-the-way/antibody_training_pipeline_ESM/releases/download/v0.6.0/boughter_vh_esm1v_logreg.pkl
mkdir -p experiments/checkpoints/esm1v/logreg/
mv boughter_vh_esm1v_logreg.pkl experiments/checkpoints/esm1v/logreg/
```

### 2. Python Environment

```bash
# Install dependencies
make install  # or: uv sync --all-extras
```

### 3. Hardware Recommendations

| Hardware | Batch Size | Throughput | Notes |
|----------|-----------|------------|-------|
| **CPU only** | 8-16 | ~10 seq/min | Works, but slow |
| **GPU (8GB VRAM)** | 32 | ~100 seq/min | Recommended |
| **GPU (16GB+ VRAM)** | 64 | ~200 seq/min | Optimal |
| **Apple Silicon (M1/M2)** | 32 | ~80 seq/min | MPS acceleration |

---

## Input Specification

### File Format

**Required:** CSV file (`.csv`) with amino acid sequences

### Column Requirements

| Column | Required? | Default Name | Description |
|--------|-----------|--------------|-------------|
| **Sequence** | **YES** | `sequence` | Amino acid sequence (20 standard AAs + X) |
| **ID** | No | `id` | Unique identifier (auto-generated if missing) |
| **Other columns** | No | Any | Preserved in output |

### Sequence Requirements

✅ **Valid:**
```
EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGKGLEWVSA...
```

❌ **Invalid:**
```
EVQ-LVESGGGLVQPG---SLRLSCAASGFTFSSYAMSWVRQAPGKGKGLEWVSA...  # Gap characters (-)
EVQLVESGGGLV*QPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGKGLEWVSA...  # Stop codons (*)
EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGKGLEWVSA123  # Non-amino acid characters
```

**Cleaning:**
- Leading/trailing whitespace automatically stripped
- Case-insensitive (auto-normalized to uppercase)
- Gap characters (`-`) and stop codons (`*`) **NOT supported** - will cause validation errors

### Example Input File

**`my_candidates.csv`:**
```csv
id,sequence,description,source
mAb-001,EVQLVESGGGLVQPGGSLRLSCAASGFTFSDYAMHWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR,Primary candidate,Lab A
mAb-002,QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCAR,Backup sequence,Lab B
mAb-003,QVQLQQSGPGLVKPSQTLSLTCAISGDSVSSNSAAWNWIRQSPSRGLEWLGRTYYRSKWYNDYAVSVKSRITINPDTSKNQFSLQLNSVTPEDTAVYYCAR,Negative control,Lab A
```

---

## Command-Line Usage

### Basic Command

```bash
uv run antibody-predict \
    input_file="path/to/input.csv" \
    output_file="path/to/output.csv" \
    classifier.path="path/to/model.pkl"
```

### All Arguments

| Argument | Type | Required? | Default | Description |
|----------|------|-----------|---------|-------------|
| `input_file` | Path | **YES** | — | Input CSV path |
| `output_file` | Path | No | `predictions.csv` | Output CSV path |
| `classifier.path` | Path | **YES** | — | Trained model checkpoint |
| `sequence_column` | String | No | `sequence` | Column name for sequences |
| `assay_type` | String | No | — | `ELISA` or `PSR` (calibrated thresholds) |
| `threshold` | Float | No | `0.5` | Manual probability threshold (0.0-1.0) |
| `model.name` | String | No | Auto-detect | ESM model architecture |
| `hardware.device` | String | No | Auto | `cuda`, `mps`, or `cpu` |
| `hardware.batch_size` | Int | No | `32` | Embedding batch size |

### Example Commands

**1. Basic prediction:**
```bash
uv run antibody-predict \
    input_file=data/candidates.csv \
    classifier.path=experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl
```

**2. Custom sequence column:**
```bash
uv run antibody-predict \
    input_file=data/canonical.csv \
    sequence_column="vh_sequence" \
    classifier.path=experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl
```

**3. PSR assay-specific threshold:**
```bash
uv run antibody-predict \
    input_file=data/nanobodies.csv \
    assay_type="PSR" \
    classifier.path=experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl
```

**4. Manual threshold tuning:**
```bash
uv run antibody-predict \
    input_file=data/high_specificity_batch.csv \
    threshold=0.8 \
    classifier.path=experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl
```

**5. Force CPU (no GPU):**
```bash
uv run antibody-predict \
    input_file=data/small_batch.csv \
    hardware.device=cpu \
    classifier.path=experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl
```

---

## Output Specification

### Output CSV Format

The output CSV contains **all original columns** plus two new inference columns:

| Column Name | Type | Description |
|-------------|------|-------------|
| `prediction` | String | Binary classification: `specific` or `non-specific` |
| `probability` | Float (0.0-1.0) | Confidence score for **non-specificity** |

**Probability Interpretation:**
- **0.0 → 0.3**: Highly specific (low polyreactivity)
- **0.3 → 0.5**: Borderline specific
- **0.5 → 0.7**: Borderline non-specific
- **0.7 → 1.0**: Highly non-specific (high polyreactivity)

### Example Output

**`predictions.csv`:**
```csv
id,sequence,description,source,prediction,probability
mAb-001,EVQL...YCAR,Primary candidate,Lab A,specific,0.04
mAb-002,QVQL...YCAR,Backup sequence,Lab B,non-specific,0.89
mAb-003,QVQL...YCAR,Negative control,Lab A,specific,0.12
```

### Result Files

When running with default settings, the following files are created:

```
output_directory/
├── predictions.csv           # Main output with predictions
├── prediction_log.txt        # Detailed run log
└── prediction_metadata.json  # Model and config metadata
```

---

## Advanced Usage

### Assay-Specific Thresholds

Different assays have different optimal thresholds:

**ELISA (Standard):**
```bash
uv run antibody-predict \
    assay_type="ELISA" \
    ...  # Uses threshold = 0.5
```

**PSR (Polyspecificity Reagent):**
```bash
uv run antibody-predict \
    assay_type="PSR" \
    ...  # Uses threshold = 0.5495 (Novo Nordisk exact parity)
```

### Threshold Tuning for Precision/Recall Trade-offs

**High Precision (Minimize False Positives):**
```bash
uv run antibody-predict \
    threshold=0.8 \
    ...  # More conservative - fewer "non-specific" calls
```

**High Recall (Minimize False Negatives):**
```bash
uv run antibody-predict \
    threshold=0.3 \
    ...  # More sensitive - catch all potential non-specific sequences
```

### Using Different Model Checkpoints

**XGBoost Classifier (v0.6.0+):**
```bash
uv run antibody-predict \
    classifier.path=experiments/checkpoints/esm1v/xgboost/boughter_vh_esm1v_xgboost.pkl \
    ...
```

**ESM-2 Backbone:**
```bash
uv run antibody-predict \
    classifier.path=experiments/checkpoints/esm2_650m/logreg/boughter_vh_esm2_650m_logreg.pkl \
    ...
```

---

## Batch Processing

### Processing Large Datasets

For datasets with >1000 sequences, use these strategies:

**1. Increase batch size (GPU):**
```bash
uv run antibody-predict \
    input_file=large_dataset.csv \
    hardware.batch_size=64 \
    hardware.device=cuda \
    classifier.path=...
```

**2. Process in chunks (CPU):**
```bash
# Split input CSV into chunks
split -l 500 large_dataset.csv chunk_

# Process each chunk
for chunk in chunk_*; do
    uv run antibody-predict \
        input_file=$chunk \
        output_file=predictions_$chunk.csv \
        classifier.path=...
done

# Merge results
cat predictions_chunk_*.csv > all_predictions.csv
```

### Parallel Processing

For maximum throughput on multi-GPU systems:

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 uv run antibody-predict \
    input_file=batch1.csv output_file=pred1.csv ... &

# GPU 1
CUDA_VISIBLE_DEVICES=1 uv run antibody-predict \
    input_file=batch2.csv output_file=pred2.csv ... &

wait  # Wait for both to finish
```

---

## Troubleshooting

### Common Errors

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `Input CSV must contain a 'sequence' column` | Column name mismatch | Use `sequence_column=your_column_name` |
| `FileNotFoundError: classifier.path` | Model path incorrect | Check path exists: `ls experiments/checkpoints/...` |
| `torch.cuda.OutOfMemoryError` | GPU memory exceeded | Reduce batch size: `hardware.batch_size=16` |
| `ValueError: Sequence contains invalid characters` | Gap characters or stop codons | Remove `-` and `*` from sequences |
| `KeyError: 'sequence'` | Missing sequence column | Add `sequence_column` argument |

### Debugging Tips

**1. Enable verbose logging:**
```bash
uv run antibody-predict \
    input_file=... \
    +log_level=DEBUG  # See detailed embedding extraction logs
```

**2. Check model compatibility:**
```bash
# Load model in Python to inspect metadata
python -c "
import pickle
with open('model.pkl', 'rb') as f:
    clf = pickle.load(f)
    print(clf.model_name)  # Should match ESM backbone
"
```

**3. Validate input CSV:**
```bash
# Check CSV schema
head -1 my_data.csv  # Print headers

# Check for gap characters
grep -c '\-' my_data.csv  # Should return 0
```

---

## Performance Tuning

### GPU Memory Optimization

| Batch Size | VRAM Usage | Throughput | Recommendation |
|------------|------------|------------|----------------|
| 8 | ~2GB | Slow | Small GPU or testing |
| 16 | ~4GB | Moderate | 8GB VRAM cards |
| 32 | ~6GB | Good | **Default (recommended)** |
| 64 | ~12GB | Excellent | 16GB+ VRAM cards |
| 128 | ~24GB | Maximum | A100 GPUs only |

### CPU vs GPU Performance

**Benchmark (100 sequences, ESM-1v):**
- **CPU (12-core)**: ~15 minutes
- **GPU (RTX 3090)**: ~45 seconds
- **Apple M2 Max (MPS)**: ~90 seconds

**Recommendation:** Use GPU for >50 sequences

---

## Integration with Downstream Tools

### Exporting to Excel

```bash
# Run prediction
uv run antibody-predict input_file=data.csv output_file=predictions.csv ...

# Convert to Excel
python -c "
import pandas as pd
df = pd.read_csv('predictions.csv')
df.to_excel('predictions.xlsx', index=False)
"
```

### Filtering Specific Sequences

```bash
# Get only high-confidence specific sequences
python -c "
import pandas as pd
df = pd.read_csv('predictions.csv')
specific = df[(df['prediction'] == 'specific') & (df['probability'] < 0.2)]
specific.to_csv('high_confidence_specific.csv', index=False)
"
```

### Visualization

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load predictions
df = pd.read_csv('predictions.csv')

# Plot probability distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='probability', hue='prediction', bins=20)
plt.xlabel('Non-Specificity Probability')
plt.ylabel('Count')
plt.title('Prediction Distribution')
plt.savefig('prediction_distribution.png')
```

---

## See Also

- **Quick Reference:** [`INFERENCE_GUIDE.md`](../../INFERENCE_GUIDE.md) (root)
- **Training Guide:** [`training.md`](training.md)
- **Model Zoo:** [`../research/model-zoo-roadmap.md`](../research/model-zoo-roadmap.md)
- **Troubleshooting:** [`troubleshooting.md`](troubleshooting.md)

---

**Last Updated:** 2025-11-19
**Version:** v0.6.0+
