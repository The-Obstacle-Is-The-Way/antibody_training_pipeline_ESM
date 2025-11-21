# Refactoring Validation Plan

**Purpose:** Ensure refactoring preserves all functionality without introducing regressions.

**Philosophy:** "Trust, but verify" - Run tests, but also validate real outputs.

**Status:** 🟢 ACTIVE - Use this before/after each refactoring phase

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

**When to run:** After every significant change

```bash
# 1. Test suite
make test
# Expected: 513 passed, 20 deselected in ~95s

# 2. Type checking
make typecheck
# Expected: Success: no issues found in 115 source files

# 3. Lint
make lint
# Expected: All checks passed!

# 4. Import smoke tests
python3 -c "
from antibody_training_esm.core.trainer import train_model
from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from antibody_training_esm.core.classifier import BinaryClassifier
from antibody_training_esm.datasets.boughter import BoughterDataset
from antibody_training_esm.datasets.jain import JainDataset
from antibody_training_esm.datasets.harvey import HarveyDataset
from antibody_training_esm.datasets.shehata import ShehataDataset
print('✅ All imports OK')
"
```

**Pass Criteria:** All 4 checks pass with expected output

---

## Full Validation (30 min)

**When to run:** Before merging a phase (A, B, C, D, E)

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
# - Test metrics reported (accuracy ~60-70%)
```

**Pass Criteria:** All 3 steps complete successfully

---

## Deep Validation (2 hours)

**When to run:** Before major releases, after Phase C/D/E

### Step 1: Baseline Creation (30 min)

**Run BEFORE refactoring to capture known-good outputs:**

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

### Step 2: Post-Refactoring Comparison (30 min)

**Run AFTER refactoring to validate outputs match:**

```bash
# 1. Check preprocessed data integrity
echo "=== Validating preprocessed data (should be unchanged) ==="
find data/train/boughter/canonical -name "*.csv" -type f -exec md5 {} + > /tmp/boughter_current.md5
diff validation/baseline/checksums/boughter_preprocessed.md5 /tmp/boughter_current.md5
# Expected: No differences (files unchanged)

find data/test/jain/canonical -name "*.csv" -type f -exec md5 {} + > /tmp/jain_current.md5
diff validation/baseline/checksums/jain_preprocessed.md5 /tmp/jain_current.md5
# Expected: No differences

# 2. Run training pipeline (post-refactoring)
echo "=== Training post-refactoring model ==="
uv run antibody-train experiment.name=validation_post_refactor

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

# 4. Compare metrics
echo "=== Comparing baseline vs post-refactoring metrics ==="
cat validation/post_refactor_*_results.txt | \
  grep -E "Accuracy|Precision|Recall|F1" > validation/post_refactor_metrics.txt

# Side-by-side comparison
paste validation/baseline/model_outputs/baseline_metrics.txt \
      validation/post_refactor_metrics.txt

# 5. Statistical comparison (allow ±1% variance due to random seeds)
python3 << 'EOF'
import re

def parse_metrics(filepath):
    """Extract accuracy from results file."""
    with open(filepath, 'r') as f:
        content = f.read()
    match = re.search(r'Accuracy:\s+([\d.]+)', content)
    return float(match.group(1)) if match else None

datasets = ['jain', 'harvey', 'shehata']
print("\n=== Metric Comparison ===")
print(f"{'Dataset':<15} {'Baseline':<12} {'Post-Refactor':<15} {'Δ':<10} {'Status'}")
print("-" * 65)

all_pass = True
for dataset in datasets:
    baseline = parse_metrics(f'validation/baseline/model_outputs/{dataset}_results.txt')
    post = parse_metrics(f'validation/post_refactor_{dataset}_results.txt')

    if baseline and post:
        delta = abs(baseline - post)
        status = "✅ PASS" if delta <= 0.01 else "❌ FAIL"
        if status == "❌ FAIL":
            all_pass = False
        print(f"{dataset:<15} {baseline:.4f}       {post:.4f}          {delta:+.4f}     {status}")
    else:
        print(f"{dataset:<15} Missing metrics")
        all_pass = False

print()
if all_pass:
    print("✅ All metrics within acceptable variance (±1%)")
else:
    print("❌ Some metrics exceed acceptable variance - investigate!")
    exit(1)
EOF
```

### Step 3: Integration Test (30 min)

**Test full end-to-end workflows:**

```bash
# Test 1: Preprocessing → Training → Testing pipeline
echo "=== Test 1: Full E2E Pipeline ==="

# Use small subset for speed
head -100 data/train/boughter/canonical/VH_only_boughter_training.csv > /tmp/test_train.csv
head -20 data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv > /tmp/test_test.csv

# Run training on subset
uv run antibody-train \
  data.train_file=/tmp/test_train.csv \
  data.test_file=/tmp/test_test.csv \
  training.max_iter=10 \
  experiment.name=e2e_validation

# Expected: Completes without errors

# Test 2: Predict CLI
echo "=== Test 2: Predict CLI ==="
echo "QVQLVQSGAEVKKPGASVKVSCKASGYTFT" > /tmp/test_sequence.txt

uv run antibody-predict \
  --model experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl \
  --sequences /tmp/test_sequence.txt

# Expected: Returns prediction (0 or 1) with probability

# Test 3: Web App (if using Gradio)
echo "=== Test 3: Web App Smoke Test ==="
timeout 10 uv run antibody-app &
sleep 5
curl -s http://127.0.0.1:7860 > /dev/null && echo "✅ Gradio app responds" || echo "❌ Gradio app not responding"
pkill -f "antibody-app"

echo "✅ All integration tests passed"
```

### Step 4: Regression Test Checklist (30 min)

**Manual verification of key workflows:**

- [ ] **Preprocessing Scripts**
  - [ ] Boughter DNA translation runs without errors
  - [ ] Boughter annotation/QC produces expected output files
  - [ ] Jain Excel → CSV conversion works
  - [ ] Harvey CSV combination works
  - [ ] Shehata Excel → CSV conversion works

- [ ] **Training Pipeline**
  - [ ] Embedding extraction completes
  - [ ] Embedding cache works (second run faster)
  - [ ] Cross-validation runs all folds
  - [ ] Model saves to correct location
  - [ ] Metrics logged correctly

- [ ] **Testing Pipeline**
  - [ ] Model loads from checkpoint
  - [ ] Predictions generated for test set
  - [ ] Metrics calculated correctly
  - [ ] Assay-specific thresholds applied (ELISA 0.5, PSR 0.5495)

- [ ] **CLI Commands**
  - [ ] `make train` works
  - [ ] `make test` works
  - [ ] `antibody-train` works with Hydra overrides
  - [ ] `antibody-test` works with model checkpoint
  - [ ] `antibody-predict` works with single sequence

**Pass Criteria:** All checklist items verified manually

---

## Phase-Specific Validation

### Phase A (Quick Wins) - COMPLETE ✅
**What changed:** File permissions, type ignores, bare except blocks

**Validation:**
```bash
make test  # 513 passed → Same as before
# No functional changes, low risk
```

### Phase B (Path Centralization) - COMPLETE ✅
**What changed:** Hardcoded paths → centralized preprocessing/paths.py

**Validation:**
```bash
# Verify imports work
python3 -c "from preprocessing.paths import BOUGHTER_TRAINING_SUBSET; print(BOUGHTER_TRAINING_SUBSET)"

# Verify preprocessing scripts work
python3 preprocessing/boughter/validate_stages2_3.py
python3 preprocessing/jain/validate_conversion.py

make test  # 513 passed → Same as before
```

### Phase C (File Splitting) - IN PROGRESS 🟡
**What will change:** Massive files split into modules

**Critical Validation (run AFTER Phase C):**
```bash
# 1. All imports still work
python3 -c "
from antibody_training_esm.core.trainer import train_model
from antibody_training_esm.core.training.cache import save_embeddings_cache
from antibody_training_esm.core.training.metrics import calculate_metrics
from antibody_training_esm.datasets.base import AntibodyDataset
print('✅ All imports OK')
"

# 2. Training still works
uv run antibody-train training.max_iter=10 experiment.name=phase_c_validation

# 3. Test suite unchanged
make test  # Should still be 513 passed, 20 deselected

# 4. Compare training output to baseline
# (Use Deep Validation Step 2 above)
```

### Phase D (Code Deduplication) - PENDING 🔴
**What will change:** Duplicated code extracted to shared utilities

**Critical Validation (run AFTER Phase D):**
```bash
# Same as Phase C validation
# Focus on: utility functions work correctly, no regressions
```

### Phase E (Polish & Documentation) - PENDING 🔴
**What will change:** Documentation updates, minor polish

**Validation:**
```bash
# Low risk - mostly docs
make docs-build  # Verify docs build
make test        # Verify no accidental changes
```

---

## Baseline Snapshots

**Current Known-Good State (as of Phase B completion):**

```bash
# Test Results
Tests passing: 513
Tests deselected: 20
Test duration: ~95s
Coverage: 88.57%

# Preprocessed Data Files
data/train/boughter/canonical/VH_only_boughter_training.csv: 915 lines (914 sequences + header)
data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv: 87 lines (86 antibodies + header)
data/test/harvey/fragments/VHH_only_harvey.csv: ~141k nanobodies
data/test/shehata/canonical/shehata_398.csv: 399 lines (398 antibodies + header)

# Model Performance (Baseline - Novo Nordisk Methodology)
Jain accuracy: 66.28% (Novo range: 66-71%) ✅
Harvey accuracy: ~55-60% (PSR threshold 0.5495)
Shehata accuracy: 58.29% (near-parity with Novo)

# File Sizes (for sanity checks)
trainer.py: 961 lines
base.py: 627 lines
stage1_dna_translation.py: 598 lines
stage2_stage3_annotation_qc.py: 519 lines
```

**How to capture your own baseline:**
```bash
# Before refactoring
git checkout dev
make test 2>&1 | tee validation/baseline_test_output.txt

# Save line counts
cloc src/ preprocessing/ > validation/baseline_line_counts.txt

# Save data file info
find data/ -name "*.csv" -exec wc -l {} + > validation/baseline_data_files.txt

# Save checksums
find data/train data/test -name "*.csv" -type f -exec md5 {} + > validation/baseline_checksums.md5
```

---

## Troubleshooting

### Issue: Test count changed after refactoring
**Symptom:** `make test` shows different number passed/deselected

**Diagnosis:**
```bash
# Compare test discovery
git diff dev -- tests/
pytest --collect-only | wc -l
```

**Likely causes:**
- Accidentally deleted test file
- Test file not properly imported
- New test added unintentionally

**Fix:**
- Restore deleted test files
- Verify all test files in `tests/` directory
- Check git diff for unintended test changes

### Issue: Import errors after file splitting
**Symptom:** `ModuleNotFoundError` or `ImportError`

**Diagnosis:**
```bash
# Check what's importing the moved code
grep -r "from.*trainer import" src/ tests/

# Check if __init__.py exports are correct
cat src/antibody_training_esm/core/training/__init__.py
```

**Likely causes:**
- Forgot to update imports in dependent files
- Missing `__init__.py` re-exports
- Circular import

**Fix:**
- Update all imports to use new module paths
- Add re-exports to `__init__.py`
- Restructure to avoid circular imports

### Issue: Different model performance after refactoring
**Symptom:** Accuracy changed significantly (>2%)

**Diagnosis:**
```bash
# Compare random seeds
grep -r "random_state\|seed" src/

# Check if data loading changed
python3 -c "
from antibody_training_esm.datasets.boughter import BoughterDataset
df = BoughterDataset().load_data(include_mild=False)
print(f'Shape: {df.shape}')
print(f'Label dist: {df[\"label\"].value_counts()}')
"
```

**Likely causes:**
- Random seed changed
- Data loading logic changed
- Feature extraction changed

**Fix:**
- Revert data loading changes
- Ensure random seeds preserved
- Compare embeddings between versions

### Issue: Preprocessed data checksums don't match
**Symptom:** MD5 checksums differ before/after refactoring

**Diagnosis:**
```bash
# Check if preprocessing was re-run
ls -ltr data/train/boughter/canonical/

# Compare file contents (ignore timestamps)
diff -w validation/baseline/boughter.csv data/train/boughter/canonical/VH_only_boughter_training.csv
```

**Expected:** Data files should NOT change during refactoring (unless Phase C touches preprocessing)

**If changed:**
- Re-run preprocessing from baseline branch
- Copy baseline data files
- Investigate why preprocessing ran

---

## Success Criteria Summary

| Validation Level | Pass Criteria | When to Run |
|------------------|---------------|-------------|
| **Quick** (5 min) | All tests pass, imports work | After every commit |
| **Full** (30 min) | Test suite + preprocessing + training smoke test | Before merging phase |
| **Deep** (2 hours) | Baseline comparison, metrics within ±1% | Before Phase C/D/E, releases |

---

## Validation Checklist for Phase C

**Before starting Phase C:**
- [ ] Run baseline capture (Deep Validation Step 1)
- [ ] Save current test results
- [ ] Save current line counts
- [ ] Commit baseline to validation/baseline/

**After each file split in Phase C:**
- [ ] Run Quick Validation (5 min)
- [ ] Verify imports work
- [ ] make test shows 513 passed

**After completing Phase C:**
- [ ] Run Full Validation (30 min)
- [ ] Run Deep Validation Step 2 (compare to baseline)
- [ ] Verify metrics within ±1% of baseline
- [ ] Manual regression test checklist

**Before merging Phase C to dev:**
- [ ] All validation checks pass
- [ ] Git diff reviewed (only intended changes)
- [ ] Documentation updated
- [ ] PR created with validation results

---

## Professional Best Practices

**What Google/DeepMind engineers do:**

1. **Golden Files**: Store known-good outputs in `testdata/` directories
2. **Snapshot Tests**: Compare outputs before/after changes
3. **Regression Suites**: Automated tests that run on every commit
4. **Staged Rollouts**: Deploy changes incrementally with monitoring
5. **A/B Testing**: Run old and new code side-by-side, compare results

**What we're doing (appropriate for this project):**

✅ **Baseline Snapshots**: Capture known-good state before refactoring
✅ **Incremental Validation**: Test after each phase
✅ **Metric Comparison**: Ensure performance unchanged (±1%)
✅ **Checksum Validation**: Verify data files unchanged
✅ **Test Suite**: Comprehensive unit/integration/e2e tests

---

## Next Steps

**For current refactoring (Phase C):**

1. **Before Phase C**: Run baseline capture (if not done)
2. **During Phase C**: Run Quick Validation after each file split
3. **After Phase C**: Run Full Validation + Deep Validation Step 2
4. **Before merge**: Verify all checks pass

**Commands to run NOW (before Phase C starts):**

```bash
# 1. Capture baseline (30 min)
mkdir -p validation/baseline/checksums
find data/train data/test -name "*.csv" -type f -exec md5 {} + > validation/baseline/checksums/baseline_data.md5

# 2. Save current test results
make test 2>&1 | tee validation/baseline/test_results.txt

# 3. Save current metrics (full training)
uv run antibody-train experiment.name=baseline_before_phase_c

# 4. Test baseline model
for dataset in jain harvey shehata; do
  uv run antibody-test --model experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl --dataset $dataset \
    > validation/baseline/${dataset}_baseline.txt 2>&1
done

# 5. Commit baseline
git add validation/
git commit -m "validation: Capture baseline before Phase C"
```

---

**THIS DOCUMENT IS YOUR SAFETY NET. USE IT. 🛡️**
