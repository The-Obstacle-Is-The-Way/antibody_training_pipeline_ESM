# Validation Plan

**Purpose:** Ensure refactoring preserves all functionality without introducing regressions.

**Philosophy:** "Trust, but verify" - Run tests, but also validate real outputs.

**Status:** 🟢 ACTIVE - Use this for **Phase F (Pydantic Integration)**.

---

## Table of Contents
1. [Quick Validation (5 min)](#quick-validation-5-min)
2. [Full Validation (30 min)](#full-validation-30-min)
3. [Deep Validation (2 hours)](#deep-validation-2-hours)
4. [Phase-Specific Validation](#phase-specific-validation)
5. [Baseline Snapshots](#baseline-snapshots)
6. [Troubleshooting](#troubleshooting)

---

## Quick Validation (5 min)

**When to run:** After every significant change (e.g., adding a Pydantic model).

```bash
# 1. Test suite
make test
# Expected: 513 passed, 20 deselected in ~95s

# 2. Type checking
make typecheck
# Expected: Success: no issues found in source files

# 3. Lint
make lint
# Expected: All checks passed!

# 4. Import smoke tests
python3 -c "
from antibody_training_esm.core.trainer import train_pipeline
from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from antibody_training_esm.core.classifier import BinaryClassifier
from antibody_training_esm.datasets.boughter import BoughterDataset
from antibody_training_esm.datasets.jain import JainDataset
from antibody_training_esm.datasets.harvey import HarveyDataset
from antibody_training_esm.datasets.shehata import ShehataDataset
print('✅ All imports OK')
"
```

**Pass Criteria:** All 4 checks pass with expected output.

---

## Full Validation (30 min)

**When to run:** Before merging the Pydantic integration phase.

### Step 1: Test Suite (10 min)
```bash
# Run full test suite
make test-all

# Expected output:
# - 513+ tests passed
# - 20 deselected (e2e/slow/gpu)
# - No failures or errors
# - Coverage ≥ 70%
```

### Step 2: Preprocessing Pipeline Validation (10 min)

**Test that preprocessing scripts still work:**

```bash
# Navigate to project root
cd /Users/ray/Desktop/CLARITY-DIGITAL-TWIN/antibody_training_pipeline_ESM

# Test Boughter preprocessing (quick dry-run)
echo "=== Testing Boughter preprocessing ==="
python3 preprocessing/boughter/validate_stages2_3.py
# Expected: ✅ All validation checks pass

# Test Jain preprocessing
echo "=== Testing Jain preprocessing ==="
python3 preprocessing/jain/validate_conversion.py
# Expected: ✅ Conversion validated successfully

# Test Harvey preprocessing
echo "=== Testing Harvey preprocessing ==="
ls -lh data/test/harvey/fragments/VHH_only_harvey.csv
# Expected: File exists with ~141k rows

# Test Shehata preprocessing
echo "=== Testing Shehata preprocessing ==="
python3 preprocessing/shehata/validate_conversion.py
# Expected: ✅ Conversion validated successfully
```

### Step 3: Training Pipeline Smoke Test (10 min)

**Verify training still works (small test run):**

```bash
# Create test config for quick training run
cat > /tmp/test_training_config.yaml << 'EOF'
defaults:
  - base_config

experiment:
  name: validation_test
  description: Quick validation run

training:
  batch_size: 8
  max_iter: 10  # Quick run
  cv_folds: 2   # Minimal folds

data:
  train_file: data/train/boughter/canonical/VH_only_boughter_training.csv
  test_file: data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv

hardware:
  device: cpu
EOF

# Run quick training
uv run antibody-train --config-path /tmp --config-name test_training_config

# Expected output:
# - Training completes without errors
# - Model saved to experiments/checkpoints/
# - Test metrics reported
```

**Pass Criteria:** All 3 steps complete successfully.

---

## Deep Validation (2 hours)

**When to run:**
1.  **NOW (Pre-Pydantic):** Establish the "Gold Standard" baseline.
2.  **LATER (Post-Pydantic):** Compare against baseline to ensure zero regressions.

### Step 1: Baseline Creation (Pre-Pydantic)

**Run BEFORE starting Pydantic integration to capture known-good outputs:**

```bash
# Create baseline directory
mkdir -p validation/baseline
mkdir -p validation/baseline/preprocessed_data
mkdir -p validation/baseline/model_outputs
mkdir -p validation/baseline/checksums

# 1. Save preprocessed data checksums
echo "=== Capturing baseline preprocessed data ==="
find data/train/boughter/canonical -name "*.csv" -type f -exec md5 {} + > validation/baseline/checksums/boughter_preprocessed.md5
find data/test/jain/canonical -name "*.csv" -type f -exec md5 {} + > validation/baseline/checksums/jain_preprocessed.md5
find data/test/harvey/fragments -name "*.csv" -type f -exec md5 {} + > validation/baseline/checksums/harvey_preprocessed.md5
find data/test/shehata/canonical -name "*.csv" -type f -exec md5 {} + > validation/baseline/checksums/shehata_preprocessed.md5

# 2. Run full training pipeline (baseline model)
echo "=== Training baseline model ==="
uv run antibody-train experiment.name=baseline_validation

# Save baseline model
cp experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl \
   validation/baseline/model_outputs/baseline_model.pkl

# 3. Test baseline model on all datasets
echo "=== Testing baseline model ==="
for dataset in jain harvey shehata; do
  echo "Testing on $dataset..."
  uv run antibody-test \
    --model validation/baseline/model_outputs/baseline_model.pkl \
    --dataset $dataset \
    > validation/baseline/model_outputs/${dataset}_results.txt 2>&1
done

# 4. Save baseline metrics
echo "=== Saving baseline metrics ==="
cat validation/baseline/model_outputs/*_results.txt | \
  grep -E "Accuracy|Precision|Recall|F1" > validation/baseline/model_outputs/baseline_metrics.txt

echo "✅ Baseline captured successfully"
echo "Baseline location: validation/baseline/"
```

### Step 2: Post-Refactoring Comparison

**Run AFTER Pydantic integration to validate outputs match:**

```bash
# 1. Check preprocessed data integrity (should be identical)
echo "=== Validating preprocessed data ==="
find data/train/boughter/canonical -name "*.csv" -type f -exec md5 {} + > /tmp/boughter_current.md5
diff validation/baseline/checksums/boughter_preprocessed.md5 /tmp/boughter_current.md5
# Expected: No differences

# 2. Run training pipeline (post-refactoring)
echo "=== Training post-refactoring model ==="
uv run antibody-train experiment.name=validation_post_pydantic

# Save post-refactoring model
cp experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl \
   validation/post_refactor_model.pkl

# 3. Test post-refactoring model
echo "=== Testing post-refactoring model ==="
for dataset in jain harvey shehata; do
  echo "Testing on $dataset..."
  uv run antibody-test \
    --model validation/post_refactor_model.pkl \
    --dataset $dataset \
    > validation/post_refactor_${dataset}_results.txt 2>&1
done

# 4. Compare metrics (Should match baseline EXACTLY or within tiny variance)
# Note: Pydantic changes are structural, not mathematical, so results should be identical.
echo "=== Comparing baseline vs post-refactoring metrics ==="
paste validation/baseline/model_outputs/baseline_metrics.txt \
      validation/post_refactor_metrics.txt
```

---

## Phase-Specific Validation

### Completed Phases (History)
- **Phase A (Quick Wins):** ✅ COMPLETE
- **Phase B (Path Centralization):** ✅ COMPLETE
- **Phase C (File Splitting):** ✅ COMPLETE
- **Phase D (Code Deduplication):** ✅ COMPLETE
- **Phase E (Polish & Documentation):** ✅ COMPLETE

### Phase F (Pydantic Integration) - ACTIVE 🟡
**Goal:** Harden boundaries with runtime validation.

**Critical Checks:**
1.  **Config Loading:** Verify `antibody-train` still accepts all CLI overrides (e.g., `hardware.device=cpu`).
2.  **Model Loading:** Verify `antibody-predict` and the Web App can still load old `.pkl` and new `.npz` models.
3.  **Error Messages:** Verify invalid inputs (e.g., bad amino acids) produce clear Pydantic ValidationErrors, not cryptic stack traces.
4.  **Serialization:** Verify `model_config.json` contains the new structured metadata.

---

## Baseline Snapshots

**Current Known-Good State (Post-Refactor):**

```bash
# Test Results
Tests passing: 513
Tests deselected: 20
Test duration: ~95s

# Data Structure
boughter/canonical/VH_only_boughter_training.csv: 915 lines
jain/canonical/VH_only_jain_86_p5e_s2.csv: 87 lines
shehata/canonical/shehata_398.csv: 399 lines

# Metrics (Approximate)
Jain accuracy: ~66.28%
Shehata accuracy: ~58.29%
```

---

## Validation Checklist for Phase F

**Before starting Phase F:**
- [ ] Run **Deep Validation Step 1** (Capture Baseline).
- [ ] Commit baseline artifacts to `validation/baseline/`.

**During Phase F:**
- [ ] Run **Quick Validation** after adding each model.
- [ ] Verify `mypy` passes (Pydantic typing can be tricky).

**After completing Phase F:**
- [ ] Run **Full Validation**.
- [ ] Run **Deep Validation Step 2** (Compare against Baseline).
- [ ] Verify `antibody-app` (Gradio) still works with valid inputs.
- [ ] Verify `antibody-app` gracefully handles invalid inputs.

---

**THIS DOCUMENT IS YOUR SAFETY NET. USE IT. 🛡️**

```