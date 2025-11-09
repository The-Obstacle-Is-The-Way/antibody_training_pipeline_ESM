# Column Naming Fix Plan

**Date:** 2025-11-09
**Parent Document:** `INVESTIGATION_COLUMN_NAMING.md`
**Status:** Ready for execution

---

## Problem Statement

Documentation examples show testing commands that will fail with `ValueError: Sequence column 'sequence' not found` because:

1. Examples use canonical CSV files (with `vh_sequence` column)
2. But don't specify `sequence_column: "vh_sequence"` in config
3. Testing CLI defaults to `sequence_column: "sequence"`

**Root Cause:** Documentation doesn't explain the difference between:
- **Canonical files** (`canonical/`): Original column names (`vh_sequence`, `vl_sequence`)
- **Fragment files** (`fragments/`): Standardized column names (`sequence`, `label`)

---

## What NOT to Fix

### Code: NO CHANGES ✅

- Training code is correct (400/403 tests passing)
- Testing code is correct
- Config loading is correct
- **Do not modify any Python code**

---

## What TO Fix

### 1. Documentation Updates (Required)

#### docs/user-guide/testing.md

**Section to add after line 50:**
```markdown
## Understanding Dataset File Types

### Canonical Files vs Fragment Files

The pipeline uses two types of CSV files:

**Canonical Files** (`test_datasets/{dataset}/canonical/*.csv`):
- Original column names from source data
- Jain: `vh_sequence`, `vl_sequence`
- Includes all metadata (flags, scores, etc.)
- Requires config override to use

**Fragment Files** (`test_datasets/{dataset}/fragments/*.csv`):
- Standardized column names: `sequence`, `label`
- Ready for training/testing with default config
- Created by preprocessing scripts
- **Recommended for most users**

### Which File Type to Use?

Use **fragment files** for:
- Quick testing with CLI (no config override needed)
- Training on specific fragments (VH, CDRs, FWRs)
- Consistent column naming across datasets

Use **canonical files** for:
- Access to full metadata (flags, PSR scores)
- Custom analysis requiring original data structure
- Must create test config with `sequence_column` override
```

**Fix all CLI examples (replace canonical with fragment):**

Before (line 80):
```bash
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv
```

After:
```bash
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/jain/fragments/VH_only_jain.csv
```

**Add section showing how to use canonical files (after line 150):**
```markdown
## Testing with Canonical Files (Advanced)

Canonical files preserve original column names and metadata. To use them, create a test config:

```yaml
# test_config_jain_canonical.yaml
model_paths:
  - "models/boughter_vh_esm1v_logreg.pkl"
data_paths:
  - "test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv"
sequence_column: "vh_sequence"  # Override default
label_column: "label"
output_dir: "./test_results"
```

Then run:
```bash
uv run antibody-test --config test_config_jain_canonical.yaml
```

**Why?** Canonical files use `vh_sequence` instead of `sequence`. The config override tells the testing CLI which column to read.
```

---

#### docs/user-guide/preprocessing.md

**Fix dataset output documentation (lines 56-117):**

**Jain section (line 73-76):**
Before:
```markdown
**Outputs:**
- `test_datasets/jain/canonical/jain_86_novo_parity.csv` - Full sequences (86 antibodies)
- `test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv` - VH only (Novo parity benchmark)
```

After:
```markdown
**Outputs:**
- `test_datasets/jain/canonical/jain_86_novo_parity.csv` - Full data (columns: `id`, `vh_sequence`, `vl_sequence`, ...)
- `test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv` - VH only (columns: `id`, `vh_sequence`, `label`)
- `test_datasets/jain/fragments/VH_only_jain.csv` - VH fragment (columns: `id`, `sequence`, `label`) **← Use this for testing**

**Column Naming:**
- **Canonical files** use `vh_sequence`/`vl_sequence` (original column names)
- **Fragment files** use `sequence` (standardized for training/testing)
```

**Harvey section (line 89-96):**
Before:
```markdown
**Outputs:**
- `test_datasets/harvey/processed/harvey.csv` - Combined raw data
- `test_datasets/harvey/fragments/VHH_only_harvey.csv` - Full VHH domain
- `test_datasets/harvey/fragments/H-CDR1_harvey.csv` - Individual CDRs
- `test_datasets/harvey/fragments/H-CDRs_harvey.csv` - Concatenated CDRs
- `test_datasets/harvey/fragments/H-FWRs_harvey.csv` - Concatenated FWRs

**Note:** Fragment naming pattern: `{fragmentName}_harvey.csv` (not `harvey_{fragmentName}.csv`)
```

After:
```markdown
**Outputs:**
- `test_datasets/harvey/processed/harvey.csv` - Combined raw data (columns: varies)
- `test_datasets/harvey/fragments/VHH_only_harvey.csv` - Full VHH (columns: `id`, `sequence`, `label`, ...)
- `test_datasets/harvey/fragments/H-CDR1_harvey.csv` - Individual CDRs
- `test_datasets/harvey/fragments/H-CDRs_harvey.csv` - Concatenated CDRs
- `test_datasets/harvey/fragments/H-FWRs_harvey.csv` - Concatenated FWRs

**Column Naming:**
- All fragment files use standardized `sequence` column (ready for testing)

**Note:** Fragment naming pattern: `{fragmentName}_harvey.csv` (not `harvey_{fragmentName}.csv`)
```

**Shehata section (line 110-117):**
Before:
```markdown
**Outputs:**
- `test_datasets/shehata/processed/shehata.csv` - Combined processed data
- `test_datasets/shehata/fragments/VH_only_shehata.csv` - VH domain
- `test_datasets/shehata/fragments/H-CDRs_shehata.csv` - Heavy CDRs
- `test_datasets/shehata/fragments/All-CDRs_shehata.csv` - All CDRs
- (16 fragment files total, pattern: `{fragmentName}_shehata.csv`)

**Note:** No `shehata_full.csv` in canonical/ - processed data in `processed/` directory
```

After:
```markdown
**Outputs:**
- `test_datasets/shehata/processed/shehata.csv` - Combined processed data (columns: varies)
- `test_datasets/shehata/fragments/VH_only_shehata.csv` - VH domain (columns: `id`, `sequence`, `label`, ...)
- `test_datasets/shehata/fragments/H-CDRs_shehata.csv` - Heavy CDRs
- `test_datasets/shehata/fragments/All-CDRs_shehata.csv` - All CDRs
- (16 fragment files total, pattern: `{fragmentName}_shehata.csv`)

**Column Naming:**
- All fragment files use standardized `sequence` column (ready for testing)

**Note:** No canonical/ directory with CSVs - only fragments/ (processed outputs only)
```

---

#### docs/user-guide/troubleshooting.md

**Add new section after line 382 (after "Sequence column not found" in Test CSV):**
```markdown
### "Sequence column not found" in Test CSV (Updated)

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
```

---

### 2. Config File Updates (Optional but Recommended)

#### configs/config.yaml

**Option A: Remove misleading test_file (Recommended)**
```yaml
# configs/config.yaml
data:
  source: "local"
  train_file: ./train_datasets/boughter/canonical/VH_only_boughter_training.csv
  # test_file: REMOVED - not used by training code (testing uses --data or test config)
  sequence_column: "sequence"
  label_column: "label"
  embeddings_cache_dir: ./embeddings_cache
```

**Option B: Add explanatory comments**
```yaml
# configs/config.yaml
data:
  source: "local"
  train_file: ./train_datasets/boughter/canonical/VH_only_boughter_training.csv
  test_file: ./test_datasets/jain/canonical/jain_86_novo_parity.csv  # NOTE: Not used by training - for reference only
  sequence_column: "sequence"  # Training file column name
  label_column: "label"
  embeddings_cache_dir: ./embeddings_cache

# NOTE: For testing, use `antibody-test` CLI with --data (fragment files)
# or create separate test_config.yaml for canonical files with sequence_column override
```

---

### 3. Create Test Config Templates (Recommended)

**Create:** `configs/test_jain_canonical.yaml`
```yaml
# Test configuration for Jain canonical dataset
# Canonical files use vh_sequence column instead of sequence

model_paths:
  - "models/boughter_vh_esm1v_logreg.pkl"
data_paths:
  - "test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv"
sequence_column: "vh_sequence"  # Canonical file column name
label_column: "label"
output_dir: "./test_results/jain_canonical"
metrics:
  - "accuracy"
  - "precision"
  - "recall"
  - "f1"
  - "roc_auc"
  - "pr_auc"
save_predictions: true
device: "mps"  # or "cuda" or "cpu"
batch_size: 16
```

**Create:** `configs/test_jain_fragment.yaml`
```yaml
# Test configuration for Jain fragment dataset
# Fragment files use standardized sequence column

model_paths:
  - "models/boughter_vh_esm1v_logreg.pkl"
data_paths:
  - "test_datasets/jain/fragments/VH_only_jain.csv"
sequence_column: "sequence"  # Standard fragment column name (default)
label_column: "label"
output_dir: "./test_results/jain_fragment"
metrics:
  - "accuracy"
  - "precision"
  - "recall"
  - "f1"
  - "roc_auc"
  - "pr_auc"
save_predictions: true
device: "mps"  # or "cuda" or "cpu"
batch_size: 16
```

**Update:** `docs/user-guide/testing.md` to reference these templates:
```markdown
## Using Test Config Templates

The repository includes example test configs:

**For fragment files (recommended):**
```bash
uv run antibody-test --config configs/test_jain_fragment.yaml
```

**For canonical files (advanced):**
```bash
uv run antibody-test --config configs/test_jain_canonical.yaml
```
```

---

### 4. Create Dataset File Types Documentation (Recommended)

**Create:** `docs/datasets/FILE_TYPES.md`

```markdown
# Dataset File Types

This document explains the two types of CSV files in the repository and when to use each.

---

## File Types

### Canonical Files

**Location:** `test_datasets/{dataset}/canonical/*.csv`

**Purpose:** Preserve original data structure and metadata from source papers

**Column Naming:**
- Original column names from source (e.g., `vh_sequence`, `vl_sequence`)
- May vary by dataset based on source publication
- Includes all metadata (flags, scores, source annotations)

**Example (Jain):**
```csv
id,vh_sequence,vl_sequence,elisa_flags,total_flags,flag_category,label,...
abrilumab,QVQLVQ...,DIQMTQ...,0,0,specific,0.0,...
```

**When to Use:**
- Custom analysis requiring full metadata
- Reproducing exact paper methodology
- Exploring additional features beyond sequence

**How to Use:**
```bash
# Requires test config with sequence_column override
uv run antibody-test --config configs/test_jain_canonical.yaml
```

---

### Fragment Files

**Location:** `test_datasets/{dataset}/fragments/*.csv`

**Purpose:** Standardized, ready-to-use files for training and testing

**Column Naming:**
- Standardized: `sequence`, `label`
- Consistent across all datasets
- May include additional columns (`id`, `source`, etc.)

**Example (Jain):**
```csv
id,sequence,label,elisa_flags,source
abituzumab,QVQLQQ...,0.0,0,jain2017_pnas
```

**When to Use:**
- Training models
- Testing models (recommended)
- Quick experiments with CLI

**How to Use:**
```bash
# Works with default config
uv run antibody-test \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/jain/fragments/VH_only_jain.csv
```

---

## Dataset-Specific Details

### Boughter (Training)
- **Canonical:** `train_datasets/boughter/canonical/VH_only_boughter_training.csv` (has `sequence` column)
- **Fragments:** `train_datasets/boughter/annotated/*_boughter.csv` (has `sequence` column)
- **Note:** Training dataset already uses standardized column names

### Jain (Test - Novo Parity)
- **Canonical:** `test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv` (has `vh_sequence` column) ❌
- **Fragments:** `test_datasets/jain/fragments/VH_only_jain.csv` (has `sequence` column) ✅
- **Recommendation:** Use fragment file for testing

### Harvey (Test - Nanobodies)
- **Canonical:** None (no canonical/ directory)
- **Fragments:** `test_datasets/harvey/fragments/VHH_only_harvey.csv` (has `sequence` column) ✅
- **Note:** Only fragment files available

### Shehata (Test - PSR Assay)
- **Canonical:** None (no canonical/ CSV files)
- **Fragments:** `test_datasets/shehata/fragments/VH_only_shehata.csv` (has `sequence` column) ✅
- **Note:** Only fragment files available

---

## Quick Reference

| Task | File Type | Command |
|------|-----------|---------|
| Training | Training file (standardized) | `uv run antibody-train --config configs/config.yaml` |
| Testing (simple) | Fragment file | `antibody-test --model MODEL --data FRAGMENT_FILE` |
| Testing (metadata) | Canonical file | `antibody-test --config test_config_canonical.yaml` |

---

## Checking CSV Columns

To check which columns a CSV has:

```bash
head -n 1 test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv
# Output: id,vh_sequence,label

head -n 1 test_datasets/jain/fragments/VH_only_jain.csv
# Output: id,sequence,label,elisa_flags,source
```

If you see `vh_sequence`, use `sequence_column: "vh_sequence"` in your test config.
If you see `sequence`, you can use defaults.
```

---

## Execution Checklist

### Phase 1: Documentation Fixes (Priority 1)
- [ ] Update `docs/user-guide/testing.md`:
  - [ ] Add "Understanding Dataset File Types" section
  - [ ] Replace canonical file examples with fragment file examples
  - [ ] Add section on testing with canonical files (advanced)
- [ ] Update `docs/user-guide/preprocessing.md`:
  - [ ] Add column naming notes to Jain outputs
  - [ ] Add column naming notes to Harvey outputs
  - [ ] Add column naming notes to Shehata outputs
- [ ] Update `docs/user-guide/troubleshooting.md`:
  - [ ] Expand "Sequence column not found" section
  - [ ] Add file type comparison table
  - [ ] Add command to check CSV columns

### Phase 2: Config Files (Priority 2)
- [ ] Update `configs/config.yaml`:
  - [ ] Remove `test_file` entry OR add explanatory comments
- [ ] Create `configs/test_jain_canonical.yaml`
- [ ] Create `configs/test_jain_fragment.yaml`
- [ ] Update `docs/user-guide/testing.md` to reference new configs

### Phase 3: Additional Documentation (Priority 3)
- [ ] Create `docs/datasets/FILE_TYPES.md`
- [ ] Add link to FILE_TYPES.md in main docs

---

## Validation

After fixes, verify:

1. **All tests still pass:**
   ```bash
   uv run pytest tests/ -v
   # Expected: 400/403 passed
   ```

2. **Training works:**
   ```bash
   uv run antibody-train --config configs/config.yaml
   ```

3. **Testing with fragment files works:**
   ```bash
   uv run antibody-test \
     --model models/boughter_vh_esm1v_logreg.pkl \
     --data test_datasets/jain/fragments/VH_only_jain.csv
   ```

4. **Testing with new configs works:**
   ```bash
   uv run antibody-test --config configs/test_jain_fragment.yaml
   uv run antibody-test --config configs/test_jain_canonical.yaml
   ```

---

## Success Criteria

- [ ] All documentation examples use correct file paths
- [ ] File type differences are clearly explained
- [ ] Users can successfully test without config override (using fragment files)
- [ ] Users understand when and how to use canonical files
- [ ] No confusion about `test_file` in config.yaml
- [ ] All 400 tests still pass after changes

---

**Last Updated:** 2025-11-09
**Status:** Ready for execution
**Priority:** High (blocks documentation Phase 3 completion)
