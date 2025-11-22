# Validation Plan (Post-Pydantic Integration)

**Purpose:** Ensure Pydantic integration (Phases 1-4) and future changes preserve functionality without introducing regressions.

**Philosophy:** "Trust, but verify" - Run tests, but also validate real outputs against known benchmarks.

**Status:** 🟢 ACTIVE - **Updated for Post-Pydantic Integration** (Phases 1-4 complete)

**Last Updated:** 2025-11-21

---

## Table of Contents
1. [Quick Validation (5 min)](#quick-validation-5-min)
2. [Full Validation (30 min)](#full-validation-30-min)
3. [Deep Validation (2 hours)](#deep-validation-2-hours)
4. [Pydantic-Specific Validation](#pydantic-specific-validation)
5. [Benchmark Validation (Gold Standard)](#benchmark-validation-gold-standard)
6. [Baseline Snapshots](#baseline-snapshots)
7. [Troubleshooting](#troubleshooting)

---

## Quick Validation (5 min)

**When to run:** After every significant change (e.g., adding a Pydantic model, refactoring data loaders).

```bash
# 1. Test suite
make test
# Expected: ~567 passed, 20 deselected in ~90s (coverage ~90%)

# 2. Type checking
make typecheck
# Expected: Success: no issues found in 148 source files

# 3. Lint
make lint
# Expected: All checks passed!

# 4. Import smoke tests (verify no circular imports, syntax errors)
python3 -c "
from antibody_training_esm.core.trainer import train_pipeline
from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from antibody_training_esm.core.classifier import BinaryClassifier
from antibody_training_esm.datasets.boughter import BoughterDataset
from antibody_training_esm.datasets.jain import JainDataset
from antibody_training_esm.datasets.harvey import HarveyDataset
from antibody_training_esm.datasets.shehata import ShehataDataset
from antibody_training_esm.schemas.dataset import (
    get_sequence_dataset_schema,
    get_boughter_schema,
    get_jain_schema,
    get_harvey_schema,
    get_shehata_schema,
)
from antibody_training_esm.settings import settings
from antibody_training_esm.models.artifact import (
    ModelArtifactMetadata,
    EvaluationMetrics,
    CVResults
)
print('✅ All imports OK')
"

# 5. Pandera schema validation smoke test
python3 -c "
import pandas as pd
from antibody_training_esm.schemas.dataset import get_jain_schema

# Test valid data
df = pd.DataFrame({
    'sequence': ['QVQLVQSG'],
    'label': [0],
    'id': ['test001']
})
validated = get_jain_schema().validate(df)
print('✅ Pandera validation OK')
"
```

**Pass Criteria:** All 5 checks pass with expected output.

---

## Full Validation (30 min)

**When to run:** Before merging any phase, before releases, after major refactoring.

### Step 1: Test Suite (10 min)
```bash
# Run full test suite (unit + integration, excludes e2e/slow/gpu)
make test-all

# Expected output:
# - ~567 tests passed
# - ~20 deselected (e2e/slow/gpu)
# - No failures or errors
# - Coverage ≥ 70% (current ~90%)

# Run coverage report (if needed)
make coverage
# Expected: Coverage ≥ 70% (current: ~90%)
```

### Step 2: Preprocessing Pipeline Validation (10 min)

**Test that preprocessing scripts work with Pandera validation:**

```bash
# Navigate to project root
cd /Users/ray/Desktop/CLARITY-DIGITAL-TWIN/antibody_training_pipeline_ESM

# Test Boughter preprocessing (Pandera-integrated)
echo "=== Testing Boughter preprocessing ==="
python3 preprocessing/boughter/validate_stages2_3.py
# Expected: ✓ VALIDATION PASSED
# Expected: 16 fragment files, 1,065 rows each
# Expected: Training subset: 914 rows (443 specific + 471 non-specific)

# Test Jain preprocessing
echo "=== Testing Jain preprocessing ==="
python3 preprocessing/jain/validate_conversion.py
# Expected: ✓ Conversion validated successfully
# Expected: VH_only_jain_86_p5e_s2.csv with 86 antibodies

# Test Harvey preprocessing (large dataset ~141k rows)
echo "=== Testing Harvey preprocessing ==="
wc -l data/test/harvey/fragments/VHH_only_harvey.csv
# Expected: ~141,000 lines (nanobody dataset)

# Test Shehata preprocessing
echo "=== Testing Shehata preprocessing ==="
python3 preprocessing/shehata/validate_conversion.py
# Expected: ✓ Conversion validated successfully
# Expected: shehata.csv with 398 antibodies
```

### Step 3: Training Pipeline Smoke Test (10 min)

**Verify training works with Pydantic/Pandera integration (quick run):**

```bash
# Quick training run (reduced folds/iterations for speed)
uv run antibody-train \
  experiment.name=validation_smoke_test \
  training.batch_size=8 \
  training.max_iter=50 \
  training.n_splits=2 \
  data.train_file=data/train/boughter/canonical/VH_only_boughter_training.csv \
  data.test_file=data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv \
  model.device=cpu

# Expected output:
# - Training completes without errors
# - Model saved to experiments/checkpoints/esm1v/logreg/
# - Test metrics reported (Jain accuracy ~60-70%, exact value varies with reduced training)
# - No Pydantic ValidationErrors
# - No Pandera SchemaErrors
```

**Pass Criteria:** All 3 steps complete successfully.

---

## Deep Validation (2 hours)

**When to run:**
- After completing Pydantic integration (Phases 1-4)
- Before major releases
- After significant refactoring of data loading/training pipelines

### Step 1: Data Integrity Verification

**Verify preprocessed data unchanged by Pydantic integration:**

```bash
# 1. Check preprocessed data checksums (should match baseline)
echo "=== Validating preprocessed data integrity ==="

# Boughter
find data/train/boughter/canonical -name "*.csv" -type f -exec shasum -a 256 {} + > /tmp/boughter_current.sha
diff validation/baseline/checksums/boughter_preprocessed.sha /tmp/boughter_current.sha
# Expected: No differences (Pandera validates but doesn't modify data)

# Jain
find data/test/jain/canonical -name "*.csv" -type f -exec shasum -a 256 {} + > /tmp/jain_current.sha
diff validation/baseline/checksums/jain_preprocessed.sha /tmp/jain_current.sha
# Expected: No differences

# Harvey
find data/test/harvey/fragments -name "*.csv" -type f -exec shasum -a 256 {} + > /tmp/harvey_current.sha
diff validation/baseline/checksums/harvey_preprocessed.sha /tmp/harvey_current.sha
# Expected: No differences

# Shehata
find data/test/shehata/processed -name "*.csv" -type f -exec shasum -a 256 {} + > /tmp/shehata_current.sha
diff validation/baseline/checksums/shehata_preprocessed.sha /tmp/shehata_current.sha
# Expected: No differences
```

### Step 2: Batch Size Independence Test 🔥

**Critical: Verify results are deterministic and batch-size independent**

```bash
echo "=== Batch Size Independence Test ==="

# Test 1: Determinism (same batch size, same results)
uv run antibody-train \
  experiment.name=batch_determinism_run1 \
  training.batch_size=8 \
  training.random_state=42 \
  model.device=cpu

uv run antibody-train \
  experiment.name=batch_determinism_run2 \
  training.batch_size=8 \
  training.random_state=42 \
  model.device=cpu

# Compare final test accuracies (should be IDENTICAL)
for run in batch_determinism_run1 batch_determinism_run2; do
  find ./experiments/runs -name "${run}*" -type f -name "training.log" -exec grep -h "Accuracy" {} \;
done
# Expected: Exact match (e.g., both 0.6628)

# Test 2: Batch size independence (different batch sizes, same results within tolerance)
for bs in 4 8 16; do
  echo "Testing batch_size=$bs..."
  uv run antibody-train \
    experiment.name=batch_independence_bs${bs} \
    training.batch_size=$bs \
    training.random_state=42 \
    model.device=cpu
done

# Extract accuracies
find ./experiments/runs -name "batch_independence_bs*" -type f -name "training.log" -exec grep -h "Accuracy" {} \;

# Expected: All three runs produce same accuracy within ±0.5%
# Example: bs4=0.6628, bs8=0.6628, bs16=0.6628 (identical or tiny variance)
```

**⚠️ If batch size affects results:** This indicates a bug in embedding extraction or batching logic. Must be fixed before production.

### Step 3: Dtype Verification (Pandera Coercion Check)

**Verify Pandera's `coerce=True` doesn't introduce unexpected dtype changes:**

```bash
python3 << 'EOF'
import pandas as pd
from antibody_training_esm.datasets.jain import JainDataset
from antibody_training_esm.datasets.boughter import BoughterDataset
from antibody_training_esm.datasets.shehata import ShehataDataset

print("=== Dtype Verification (Pandera coerce=True) ===\n")

# Test Jain
df_jain = JainDataset().load_data(full_csv_path="data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv", stage="full")
print("Jain dtypes:")
print(df_jain.dtypes)
assert df_jain['label'].dtype == 'int64', f"❌ Jain label dtype: {df_jain['label'].dtype}"
assert str(df_jain['sequence'].dtype) in ['string', 'object'], f"❌ Jain sequence dtype: {df_jain['sequence'].dtype}"
print("✅ Jain dtypes correct\n")

# Test Boughter
df_boughter = BoughterDataset().load_data("data/train/boughter/canonical/VH_only_boughter_training.csv")
print("Boughter dtypes:")
print(df_boughter.dtypes)
assert df_boughter['label'].dtype == 'int64', f"❌ Boughter label dtype: {df_boughter['label'].dtype}"
print("✅ Boughter dtypes correct\n")

# Test Shehata
df_shehata = ShehataDataset().load_data("data/test/shehata/processed/shehata.csv")
print("Shehata dtypes:")
print(df_shehata.dtypes)
assert df_shehata['label'].dtype == 'int64', f"❌ Shehata label dtype: {df_shehata['label'].dtype}"
if 'psr_measurement' in df_shehata.columns:
    assert df_shehata['psr_measurement'].dtype == 'float64', f"❌ Shehata PSR dtype: {df_shehata['psr_measurement'].dtype}"
print("✅ Shehata dtypes correct\n")

print("✅ All dtypes verified - Pandera coercion safe")
EOF
```

### Step 4: Full Production Training Run

**Run FULL training pipeline with production config (not toy config):**

```bash
echo "=== Full Production Training Run ==="

# Train with FULL production config
uv run antibody-train \
  experiment.name=validation_full_production \
  training.batch_size=8 \
  training.n_splits=10 \
  training.max_iter=1000 \
  data.train_file=data/train/boughter/canonical/VH_only_boughter_training.csv \
  data.test_file=data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv \
  model.device=cpu

# Test on all datasets
for dataset in jain harvey shehata; do
  echo "Testing on $dataset..."
  uv run antibody-test \
    --model experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl \
    --dataset $dataset
done

# Save results for comparison (Hydra prints output dir on start)
find ./experiments/runs -name "validation_full_production*" -maxdepth 2 -type d \
  -exec grep -h "Accuracy" {}/training.log \; > /tmp/production_run_metrics.txt 2>/dev/null || true
cat /tmp/production_run_metrics.txt
```

**Expected metrics (within ±1% tolerance):**
- **Jain**: 66.28% (Novo parity benchmark)
- **Shehata**: 58.29% (Novo parity benchmark)
- **Harvey**: 61.33% (baseline)

---

## Pydantic-Specific Validation

**Checks introduced by Pydantic Phases 1-4:**

### Phase 1: Prediction Hardening
```bash
# Test PredictionRequest validation
python3 -c "
from antibody_training_esm.models.prediction import PredictionRequest

# Valid input
req = PredictionRequest(sequence='QVQLVQSG')
print(f'✅ Valid sequence accepted: {req.sequence}')

# Invalid input (should raise ValidationError)
try:
    req = PredictionRequest(sequence='QVQL123')  # Invalid amino acids
    print('❌ Invalid sequence accepted (BUG!)')
except Exception as e:
    print(f'✅ Invalid sequence rejected: {type(e).__name__}')
"
```

### Phase 2: Configuration Safety
```bash
# Test config validation
uv run antibody-train \
  experiment.name=config_validation_test \
  hardware.device=invalid_device \
  || echo "✅ Invalid config rejected (expected)"
```

### Phase 3: Data Integrity (Pandera)
```bash
# Test schema rejection of invalid data
python3 << 'EOF'
import pandas as pd
from antibody_training_esm.schemas.dataset import get_jain_schema
from pandera.errors import SchemaError

# Valid data
df_valid = pd.DataFrame({
    'sequence': ['QVQLVQSG'],
    'label': [0],
    'id': ['test001']
})
get_jain_schema().validate(df_valid)
print("✅ Valid data accepted")

# Invalid data (bad amino acids)
df_invalid = pd.DataFrame({
    'sequence': ['QVQL123'],  # Invalid
    'label': [0],
    'id': ['test001']
})
try:
    get_jain_schema().validate(df_invalid)
    print("❌ Invalid data accepted (BUG!)")
except SchemaError:
    print("✅ Invalid data rejected")

# Invalid data (missing required column)
df_missing = pd.DataFrame({
    'sequence': ['QVQLVQSG'],
    # Missing 'label' and 'id'
})
try:
    get_jain_schema().validate(df_missing)
    print("❌ Missing columns accepted (BUG!)")
except SchemaError:
    print("✅ Missing columns rejected")
EOF
```

---

## Benchmark Validation (Gold Standard)

**Use published benchmarks as authoritative reference (more reliable than pre-Pydantic baseline).**

### Known Benchmarks (from docs/research/)

| Dataset | Metric | Benchmark Value | Source |
|---------|--------|----------------|--------|
| Jain | Accuracy | 66.28% | Novo Nordisk parity (docs/research/novo-parity.md) |
| Shehata | Accuracy | 58.29% | Novo Nordisk parity (docs/datasets/shehata/threshold_calibration_discovery.md) |
| Harvey | Accuracy | 61.33% | Baseline (validation/baseline/model_outputs/baseline_metrics.txt) |

### Validation Script

```bash
echo "=== Benchmark Validation ==="

# Train and test
uv run antibody-train experiment.name=benchmark_validation
for dataset in jain shehata harvey; do
  uv run antibody-test \
    --model experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl \
    --dataset $dataset \
    > /tmp/benchmark_${dataset}.txt
done

# Extract accuracies
echo "=== Results vs. Benchmarks ==="
echo "Dataset       | Result  | Benchmark | Delta   | Status"
echo "--------------|---------|-----------|---------|-------"

# Jain
jain_acc=$(grep "accuracy:" /tmp/benchmark_jain.txt | awk '{print $2}')
jain_delta=$(python3 -c "print(f'{abs(${jain_acc:-0} - 0.6628):.4f}')")
jain_status=$(python3 -c "print('✅ PASS' if abs(${jain_acc:-0} - 0.6628) < 0.01 else '❌ FAIL')")
echo "Jain          | ${jain_acc} | 0.6628    | ${jain_delta} | ${jain_status}"

# Shehata
shehata_acc=$(grep "accuracy:" /tmp/benchmark_shehata.txt | awk '{print $2}')
shehata_delta=$(python3 -c "print(f'{abs(${shehata_acc:-0} - 0.5829):.4f}')")
shehata_status=$(python3 -c "print('✅ PASS' if abs(${shehata_acc:-0} - 0.5829) < 0.01 else '❌ FAIL')")
echo "Shehata       | ${shehata_acc} | 0.5829    | ${shehata_delta} | ${shehata_status}"

# Harvey
harvey_acc=$(grep "accuracy:" /tmp/benchmark_harvey.txt | awk '{print $2}')
harvey_delta=$(python3 -c "print(f'{abs(${harvey_acc:-0} - 0.6133):.4f}')")
harvey_status=$(python3 -c "print('✅ PASS' if abs(${harvey_acc:-0} - 0.6133) < 0.01 else '❌ FAIL')")
echo "Harvey        | ${harvey_acc} | 0.6133    | ${harvey_delta} | ${harvey_status}"

echo ""
echo "Tolerance: ±1% (0.01)"
echo "If all PASS → Pydantic integration safe ✅"
echo "If any FAIL → Investigate regression 🚨"
```

---

## Baseline Snapshots

**Current Known-Good State (Post-Pydantic Phase 4):**

```bash
# Test Results (as of 2025-11-21)
Tests passing: 567
Tests deselected: 20
Test duration: ~90s
Coverage: ~90% (make test)

# Data Structure
boughter/canonical/VH_only_boughter_training.csv: 915 lines (914 sequences)
jain/canonical/VH_only_jain_86_p5e_s2.csv: 87 lines (86 antibodies)
shehata/processed/shehata.csv: 399 lines (398 antibodies)
harvey/fragments/VHH_only_harvey.csv: 141,022 lines

# Benchmark Metrics (Authoritative)
Jain accuracy: 66.28% (Novo parity)
Shehata accuracy: 58.29% (Novo parity)
Harvey accuracy: 61.33% (baseline)

# Pydantic Integration Status
Phase 1 (Prediction Hardening): ✅ COMPLETE
Phase 2 (Configuration Safety): ✅ COMPLETE
Phase 3 (Data Integrity): ✅ COMPLETE
Phase 4 (Artifacts & Metrics): ✅ COMPLETE
```

---

## Troubleshooting

### Issue: Batch size affects results
**Symptoms:** Different batch sizes (4, 8, 16) produce different accuracies.

**Root cause:** Batch padding or floating-point rounding in embedding extraction.

**Fix:** Investigate `ESMEmbeddingExtractor.extract_embeddings()` batching logic.

### Issue: Pandera SchemaError on valid data
**Symptoms:** `SchemaError: valid_amino_acids` on sequences that should be valid.

**Root cause:** Pandera check function returning scalar instead of bool.

**Fix:** Ensure all check functions use `bool(series.all())` pattern (see `schemas/dataset.py`).

### Issue: Metrics don't match benchmarks
**Symptoms:** Jain accuracy is 60% instead of 66.28%.

**Possible causes:**
1. Wrong dataset file (using wrong fragment or subset)
2. Model not fully trained (using toy config instead of production config)
3. Random seed variation (check if `training.random_state` is set)

**Fix:** Verify:
- Using correct dataset: `VH_only_jain_86_p5e_s2.csv` (not archive files)
- Production config: `n_splits=10`, `max_iter=1000`
- Same batch size as benchmark runs

### Issue: Data checksums differ
**Symptoms:** `diff` shows differences between baseline and current checksums.

**Root cause:** Preprocessed data files were regenerated.

**Fix:** Verify preprocessing scripts haven't changed. If preprocessing is correct, update baseline checksums.

---

## Validation Checklist

**After completing Pydantic Phases 1-4:**

- [ ] Run **Quick Validation** (5 min)
- [ ] Run **Full Validation** (30 min)
- [ ] Run **Batch Size Independence Test** (30 min)
- [ ] Run **Dtype Verification** (5 min)
- [ ] Run **Full Production Training** (1 hour)
- [ ] Run **Benchmark Validation** (1 hour)
- [ ] Verify all benchmarks within ±1% tolerance
- [ ] Update baseline checksums (if preprocessing changed)
- [ ] Document any deviations in validation log

**Before major releases:**

- [ ] All of the above
- [ ] Run e2e tests: `make test-e2e` (opt-in with env vars)
- [ ] Test Gradio app: `uv run antibody-app`
- [ ] Verify invalid inputs produce clear error messages (not stack traces)
- [ ] Check documentation reflects current behavior

---

**THIS DOCUMENT IS YOUR SAFETY NET. USE IT. 🛡️**

**Key Changes from Pre-Pydantic Plan:**
1. ✅ Updated for Pydantic Phases 1-4 completion
2. ✅ Added batch size independence test (your concern!)
3. ✅ Added dtype verification (Pandera coercion check)
4. ✅ Added benchmark validation (gold standard comparison)
5. ✅ Updated test counts (~567 tests, ~90% coverage)
6. ✅ Added Pydantic-specific validation sections
7. ✅ Removed outdated pre-Pydantic baseline references
8. ✅ Added Phase 4 artifact imports to smoke tests
