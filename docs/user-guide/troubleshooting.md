# Troubleshooting Guide

This guide provides solutions to common issues encountered when using the antibody training pipeline.

---

## Installation Issues

### `uv` command not found after installation

**Symptoms:**

```bash
$ uv --version
bash: uv: command not found
```

**Solution:**

Restart your terminal or manually add `uv` to your PATH:

```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH="$HOME/.cargo/bin:$PATH"

# Reload shell config
source ~/.bashrc  # or source ~/.zshrc
```

---

### Python version mismatch

**Symptoms:**

```bash
error: Python 3.12 is required but Python 3.10 is installed
```

**Solution:**

Install Python 3.12:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.12

# macOS (Homebrew)
brew install python@3.12

# Or use pyenv
pyenv install 3.12
pyenv local 3.12
```

---

### Permission denied on macOS/Linux

**Symptoms:**

```bash
PermissionError: [Errno 13] Permission denied
```

**Solution:**

**Never use `sudo` with `uv`**. Fix ownership instead:

```bash
# Fix ~/.local ownership
sudo chown -R $USER:$USER ~/.local

# Fix .venv ownership (if needed)
sudo chown -R $USER:$USER .venv
```

---

## GPU / Hardware Issues

### MPS Memory Issues (Apple Silicon)

**Symptoms:**

```python
RuntimeError: MPS backend out of memory
```

**Solution 1: Reduce Batch Size**

```yaml
# configs/config.yaml
hardware:
  batch_size: 4  # Reduce from default (16)
```

**Solution 2: Clear MPS Cache**

```python
import torch
torch.mps.empty_cache()
```

**Solution 3: Use CPU Instead**

```yaml
# configs/config.yaml
hardware:
  device: "cpu"
```

**Permanent Fix:**

The MPS memory leak was fixed in commit `9c8e5f2`. If still encountering issues, see `docs/archive/MPS_MEMORY_LEAK_FIX.md` for historical context.

---

### CUDA Out of Memory

**Symptoms:**

```python
RuntimeError: CUDA out of memory. Tried to allocate XX.XX MiB
```

**Solution 1: Reduce Batch Size**

```yaml
# configs/config.yaml
hardware:
  batch_size: 8  # Reduce from default (16)
```

**Solution 2: Clear CUDA Cache**

```python
import torch
torch.cuda.empty_cache()
```

**Solution 3: Use Smaller Model**

```yaml
# configs/config.yaml
model:
  name: "facebook/esm1v_t33_650M_UR90S_1"  # 650M parameters
  # Instead of:
  # name: "facebook/esm2_t36_3B_UR50D"  # 3B parameters
```

**Solution 4: Use CPU**

```yaml
# configs/config.yaml
hardware:
  device: "cpu"
```

---

### GPU Not Detected

**Symptoms:**

```bash
CUDA available: False
MPS available: False
```

**Solution (CUDA):**

```bash
# Check GPU is visible
nvidia-smi

# Verify PyTorch CUDA installation
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Solution (MPS):**

```bash
# Verify macOS version (≥12.3 required)
sw_vers

# Check MPS is available
python -c "import torch; print(torch.backends.mps.is_available())"
```

---

## Training Issues

### ESM-1v Download Fails

**Symptoms:**

```python
ConnectionError: HTTPSConnectionPool(host='huggingface.co', port=443)
```

**Solution:**

```bash
# Set HuggingFace cache directory
export HF_HOME=/path/with/space

# Use HuggingFace mirror (if in region with restrictions)
export HF_ENDPOINT=https://hf-mirror.com

# Retry download
uv run antibody-train --config configs/config.yaml
```

---

### "Label column not found" Error

**Symptoms:**

```python
KeyError: 'label'
```

**Solution:**

Ensure training CSV has `label` column with 0/1 values:

```csv
sequence,label
EVQLVESGGGLV...,0
QVQLQESGPGLV...,1
```

If using different column name, update CSV:

```python
import pandas as pd
df = pd.read_csv("data.csv")
df = df.rename(columns={"polyreactivity": "label"})
df.to_csv("data_fixed.csv", index=False)
```

---

### Embedding Cache Out of Sync

**Symptoms:**

- Embeddings don't match expected shape
- Predictions are random/nonsensical
- Cache from old model version

**Solution:**

Clear embeddings cache and retrain:

```bash
rm -rf embeddings_cache/
uv run antibody-train --config configs/config.yaml
```

Cache is SHA-256 hashed by:
- Model name
- Dataset path
- Model revision

Any change invalidates cache automatically, but manual clearing ensures fresh start.

---

### Poor Cross-Validation Performance

**Symptoms:**

```
10-Fold CV Accuracy: 52% ± 8%  # Near random
```

**Possible Causes & Solutions:**

**1. Wrong Column Names**

```yaml
# Ensure column names match CSV file
data:
  sequence_column: "sequence"  # Default column name for sequences
  label_column: "label"        # Default column name for labels
```

Check your CSV has these columns:
```python
import pandas as pd
df = pd.read_csv("train.csv")
print(df.columns)  # Should include 'sequence' and 'label'
```

**2. Label Encoding Error**

```python
# Check label distribution
import pandas as pd
df = pd.read_csv("train.csv")
print(df["label"].value_counts())
# Should show: 0: XXX, 1: YYY (binary labels)
```

**3. Sequence Quality Issues**

```python
# Check for invalid sequences
df["valid"] = df["VH"].str.match(r'^[ACDEFGHIKLMNPQRSTVWY]+$')
print(f"Invalid: {(~df['valid']).sum()}")
```

**4. Model Not Loaded Correctly**

```bash
# Verify ESM-1v downloaded
ls ~/.cache/huggingface/hub/models--facebook--esm1v_t33_650M_UR90S_1/
```

---

### Training Takes Too Long

**Symptoms:**

- Training runs for hours on small dataset
- Embedding extraction stuck

**Solution 1: Use GPU**

```yaml
# configs/config.yaml
hardware:
  device: "cuda"  # or "mps" for Apple Silicon
```

**Solution 2: Check Dataset Size**

```python
import pandas as pd
df = pd.read_csv("train.csv")
print(f"Dataset size: {len(df)}")
# Boughter: 914 sequences
# Jain: 86 sequences
# If >10k, expect longer training
```

**Solution 3: Verify Embeddings Are Cached**

```bash
# Check cache directory
ls -lh embeddings_cache/
# Should see .npy files after first run
```

---

## Testing Issues

### Model Fails to Load

**Symptoms:**

```python
FileNotFoundError: models/my_model.pkl not found
```

**Solution:**

```bash
# Check model exists
ls -lh models/

# Verify model path in command (using fragment file for compatibility)
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \  # Correct path
  --data test_datasets/jain/fragments/VH_only_jain.csv
```

---

### "Sequence column not found" in Test CSV

**Symptoms:**

```python
ValueError: Sequence column 'sequence' not found in dataset. Available columns: ['id', 'vh_sequence', 'vl_sequence', ...]
```

**Root Cause:**

You're trying to test with a **canonical file** using default config:

```bash
# THIS FAILS (canonical file has vh_sequence, not sequence)
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv
```

**Solution 1: Use Fragment Files (Recommended)**

Fragment files have standardized `sequence` column:

```bash
# THIS WORKS (fragment file has sequence column)
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/jain/fragments/VH_only_jain.csv
```

**Solution 2: Create Config for Canonical Files**

If you need to use canonical files (for metadata access):

```yaml
# test_config_canonical.yaml
model_paths:
  - "models/boughter_vh_esm1v_logreg.pkl"
data_paths:
  - "test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv"
sequence_column: "vh_sequence"  # Override for canonical file
label_column: "label"
```

Then run:
```bash
uv run antibody-test --config test_config_canonical.yaml
```

**Understanding File Types:**

| File Type | Location | Columns | Use Case |
|-----------|----------|---------|----------|
| Canonical | `test_datasets/{dataset}/canonical/` | `vh_sequence`, `vl_sequence` | Full metadata, requires config |
| Fragment | `test_datasets/{dataset}/fragments/` | `sequence`, `label` | Standardized, works with defaults |

**Check CSV columns:**
```bash
head -n 1 test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv
# Output: id,vh_sequence,label (needs sequence_column: "vh_sequence")

head -n 1 test_datasets/jain/fragments/VH_only_jain.csv
# Output: id,sequence,label (works with defaults)
```

---

### Poor Test Performance (Cross-Dataset)

**Symptoms:**

- Train CV: 71% accuracy
- Test (different dataset): 55% accuracy

**Expected Behavior:**

Cross-dataset generalization is **inherently challenging**:

- **Cross-assay:** ELISA → PSR (different binding measurements)
- **Cross-species:** Human antibodies → Nanobodies (different structure)
- **Cross-source:** Different labs, protocols, quality control

**Solutions:**

**1. Use Correct Dataset Files**

```bash
# Train ELISA, test ELISA (Boughter → Jain)
# Use fragment file for compatibility with default config
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/jain/fragments/VH_only_jain.csv
```

**2. Tune Assay-Specific Thresholds (PSR Assays)**

For ELISA → PSR prediction, adjust threshold in test config:

```yaml
# test_config_psr.yaml
model_paths:
  - "models/boughter_vh_esm1v_logreg.pkl"

data_paths:
  - "test_datasets/shehata/fragments/VH_only_shehata.csv"

threshold: 0.5495  # Novo Nordisk PSR threshold (default ELISA: 0.5)
```

Or manually in Python:

```python
import pickle

# Load model
with open("models/boughter_vh_esm1v_logreg.pkl", "rb") as f:
    classifier = pickle.load(f)

# Get prediction probabilities
probs = classifier.predict_proba(test_embeddings)[:, 1]

# Apply PSR threshold
psr_predictions = (probs > 0.5495).astype(int)
```

**3. Match Fragment Types**

Train and test on same fragment type:

```bash
# If trained on VH, test on VH (not CDRs or FWRs)
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/shehata/fragments/VH_only_shehata.csv  # VH only
```

**4. Accept Lower Performance**

Cross-assay accuracy typically drops 5-10%:

- **Same-assay (ELISA→ELISA):** 66-71%
- **Cross-assay (ELISA→PSR):** 60-65%

See [Research Notes - Benchmark Results](../research/benchmark-results.md) for expected performance ranges.

---

### Test Takes Too Long (Large Datasets)

**Symptoms:**

Testing Harvey (141k sequences) takes >30 minutes

**Solution:**

Use GPU acceleration:

```bash
# Verify GPU available
python -c "import torch; print(torch.cuda.is_available())"

# Force GPU usage
export CUDA_VISIBLE_DEVICES=0
uv run antibody-test --model models/my_model.pkl --dataset harvey
```

**Expected Times:**

| Dataset | Size | CPU | GPU (CUDA/MPS) |
|---------|------|-----|----------------|
| Jain    | 86   | 30s | 10s            |
| Shehata | 398  | 2m  | 30s            |
| Harvey  | 141k | 20m | 5-8m           |

---

## Preprocessing Issues

### ANARCI Annotation Fails

**Symptoms:**

```bash
Command 'anarci' not found
```

**Solution:**

Install ANARCI (requires Conda/Mamba):

```bash
# Create conda environment with ANARCI
conda create -n anarci python=3.12
conda activate anarci
conda install -c bioconda anarci

# Verify installation
anarci -h
```

**Note:** ANARCI is required for CDR/FWR extraction but not for training on pre-extracted VH sequences.

---

### Excel File Won't Open

**Symptoms:**

```python
ImportError: Missing optional dependency 'openpyxl'
```

**Solution:**

Install Excel reading libraries:

```bash
uv pip install openpyxl xlrd
```

---

### Fragment Extraction Produces Empty Sequences

**Symptoms:**

Fragment CSVs have `NaN` or empty strings

**Solution:**

Check ANARCI annotation success:

```python
import pandas as pd
df = pd.read_csv("canonical/my_dataset.csv")

# Check annotation rate
df["has_vh"] = df["VH"].notna() & (df["VH"] != "")
print(f"VH annotation rate: {df['has_vh'].mean():.1%}")

# Expected: >90% for high-quality data
```

**If annotation rate is low (<80%):**

1. Check sequence quality (valid amino acids only)
2. Verify ANARCI is installed correctly
3. Inspect failed sequences manually

---

## Configuration Issues

### YAML Syntax Error

**Symptoms:**

```python
yaml.scanner.ScannerError: while scanning a simple key
```

**Solution:**

Check YAML syntax:

```yaml
# INCORRECT (missing space after colon)
data:
  train_file:"path/to/file.csv"

# CORRECT (space after colon)
data:
  train_file: "path/to/file.csv"
```

Validate YAML:

```bash
python -c "import yaml; yaml.safe_load(open('configs/config.yaml'))"
```

---

### Config File Not Found

**Symptoms:**

```bash
FileNotFoundError: configs/my_config.yaml not found
```

**Solution:**

Use absolute path or ensure working directory is correct:

```bash
# From repository root
uv run antibody-train --config configs/config.yaml

# Or use absolute path
uv run antibody-train --config /full/path/to/config.yaml
```

---

## Development / CI Issues

### Pre-commit Hook Blocks Commit

**Symptoms:**

```bash
ruff....................................Failed
- hook id: ruff
- exit code: 1
```

**Solution:**

This is **intentional** - hooks enforce code quality. Fix the errors:

```bash
# Auto-fix formatting
make format

# Check remaining issues
make lint

# Run all quality checks
make all

# Try commit again
git commit -m "Your message"
```

**To bypass hooks (NOT RECOMMENDED):**

```bash
git commit --no-verify -m "Your message"
```

---

### Type Checking Fails

**Symptoms:**

```bash
error: Function is missing a return type annotation
```

**Solution:**

Add type annotations:

```python
# INCORRECT (no return type)
def my_function(x):
    return x * 2

# CORRECT (with return type)
def my_function(x: int) -> int:
    return x * 2
```

This repository enforces 100% type safety. See [Developer Guide - Type Checking](../developer-guide/type-checking.md) for details (pending Phase 4).

---

### Tests Fail in CI but Pass Locally

**Symptoms:**

```bash
# Local
pytest  # All pass

# CI
pytest  # Some fail
```

**Possible Causes:**

1. **Environment differences** - CI uses fresh environment
2. **Cached data** - Local has cached embeddings, CI doesn't
3. **Random seeds** - Non-deterministic test behavior

**Solution:**

```bash
# Test in fresh environment
rm -rf .venv embeddings_cache/
uv venv
source .venv/bin/activate
uv sync
pytest
```

---

## Common Error Messages

### `RuntimeError: Cannot re-initialize CUDA in forked subprocess`

**Solution:**

Set multiprocessing start method:

```python
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
```

---

### `pickle.UnpicklingError: invalid load key`

**Symptoms:**

Model file is corrupted

**Solution:**

Retrain model:

```bash
rm models/corrupted_model.pkl
uv run antibody-train --config configs/config.yaml
```

---

### `torch.cuda.OutOfMemoryError: CUDA out of memory`

See [CUDA Out of Memory](#cuda-out-of-memory) section above.

---

## Getting Help

If you encounter an issue not covered here:

1. **Check existing documentation:**
   - [System Overview](../overview.md)
   - [Installation Guide](installation.md)
   - [Training Guide](training.md)
   - [Testing Guide](testing.md)
   - [Preprocessing Guide](preprocessing.md)

2. **Check CI/CD logs:**
   - See [Developer Guide - CI/CD](../developer-guide/ci-cd.md) (pending Phase 4)

3. **Review historical issues:**
   - See [Archive](../archive/) for past debugging sessions

4. **File a GitHub issue:**
   - Include: OS, Python version, GPU type, error message, minimal reproducible example

---

## Quick Diagnostic Commands

Run these commands to diagnose common issues:

```bash
# Check Python version
python --version  # Should be 3.12+

# Check uv installation
uv --version

# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# Check installed packages
uv pip list

# Check repository structure
ls -lh configs/ models/ train_datasets/ test_datasets/

# Check embeddings cache
ls -lh embeddings_cache/

# Run full quality pipeline
make all
```

---

**Last Updated:** 2025-11-09
**Branch:** `docs/canonical-structure`
