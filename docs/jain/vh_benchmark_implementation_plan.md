# Plan: Add VH-only Benchmark File Generation to step2

**Date:** 2025-11-07
**Status:** ✅ IMPLEMENTED AND MERGED (v3.0)
**Author:** Development Team
**Implementation:** Completed 2025-11-07

---

## REVISION HISTORY

**v3.0 (2025-11-07):** Critical E2E test compatibility fix
- Added CRITICAL section on stage="parity" vs stage="full" issue
- Documented required E2E test changes
- Explained why 3-column file fails with stage="parity"
- Validated from first principles with code evidence

**v2.0 (2025-11-07):** Complete rewrite based on first-principles analysis
- Fixed column schema mismatch (sequence vs vh_sequence)
- Removed unnecessary columns (smp, elisa, source) - not used by downstream code
- Aligned with JainDataset.load_data() expectations
- Simplified to minimal required columns

**v1.0:** Initial draft (superseded)

---

## Executive Summary

Add ~10 lines of code to `step2_preprocess_p5e_s2.py` to automatically generate `VH_only_jain_86_p5e_s2.csv` alongside the existing `jain_86_novo_parity.csv` output. This makes the preprocessing pipeline fully reproducible and eliminates manually maintained files.

**Key Finding:** Current manually-restored VH file has incompatible schema. Must fix.

---

## Problem Statement - ROOT CAUSE ANALYSIS

### What We Discovered:

**Current VH file schema (manually restored from git):**
```
Columns: ['id', 'sequence', 'label', 'smp', 'elisa', 'source']
```

**What JainDataset.load_data() expects:**
```
Columns: ['id', 'vh_sequence', 'label', ...]
```

**Result:** INCOMPATIBLE - Test code expects `df['VH_sequence']` but file has `'sequence'`

### The Mismatch Chain:

1. **E2E test** (tests/e2e/test_reproduce_novo.py:118) calls:
   ```python
   df_test = jain.load_data(full_csv_path="VH_only_file.csv", stage="parity")
   X_test = extractor.extract_batch_embeddings(df_test["VH_sequence"].tolist())
   ```

2. **JainDataset.load_data()** (src/antibody_training_esm/datasets/jain.py:133-137) renames:
   ```python
   column_mapping = {
       "heavy_seq": "VH_sequence",
       "light_seq": "VL_sequence",
   }
   df = df.rename(columns=column_mapping)
   ```

3. **Current VH file** has column "sequence" NOT "vh_sequence" or "heavy_seq"

4. **Result:** `df["VH_sequence"]` will raise KeyError

### Extra Columns Analysis:

**Current file has:** `['smp', 'elisa', 'source']`

**Are they used?**
- ❌ JainDataset.load_data() ignores them
- ❌ E2E test doesn't reference them
- ❌ Not in canonical file (jain_86_novo_parity.csv has 'psr' not 'smp')
- ❌ 'source' column doesn't exist ANYWHERE in pipeline (hardcoded 'jain2017')

**Verdict:** Legacy metadata from manual creation. DELETE.

---

## CRITICAL: JainDataset.load_data() Stage Parameter Issue

### The Problem:

**Current E2E test** (tests/e2e/test_reproduce_novo.py:101-102):
```python
df_test = jain.load_data(
    full_csv_path=real_dataset_paths["jain_parity"],
    stage="parity"  # ❌ WRONG for pre-filtered file
)
```

**Why this FAILS with 3-column VH file:**

When `stage="parity"`, JainDataset.load_data() runs the FULL parity filtering pipeline (jain.py:178-181):

```python
elif stage == "parity":
    df = self.filter_elisa_1to3(df)           # Needs elisa_flags column → KeyError!
    df = self.reclassify_5_antibodies(df)     # Needs psr column → KeyError!
    df = self.remove_30_by_psr_acsins(df)     # Needs psr, ac_sins columns → KeyError!
```

**Our VH file only has:** `[id, vh_sequence, label]`

**Result:** `KeyError: 'elisa_flags'` on line jain.py:200

### The Root Cause:

The VH file is ALREADY the 86-row parity output. Calling `stage="parity"` tries to RE-RUN the parity filtering pipeline on pre-filtered data that lacks the columns needed for filtering.

**It's like trying to filter-then-dedup a list that's already been filter-then-deduped.**

### The Solution:

Use `stage="full"` for pre-filtered files:

```python
df_test = jain.load_data(
    full_csv_path="test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv",
    stage="full"  # ✅ CORRECT - file is already parity-filtered, load as-is
)
```

**What stage="full" does** (jain.py:175-183):
- Loads CSV
- Renames columns if needed (`vh_sequence` → `VH_sequence`)
- Returns data WITHOUT running any filters
- This is semantically correct: "load the full/complete parity file"

### Stage Parameter Semantics:

| Stage | Input Expected | Output | Filtering Applied |
|-------|---------------|--------|-------------------|
| `"full"` | Any CSV | Raw loaded data | None |
| `"ssot"` | 137-row full dataset | 116 rows | ELISA 1-3 removal |
| `"parity"` | 137-row full dataset | 86 rows | ELISA + reclassify + PSR/AC-SINS |

**For VH_only_jain_86_p5e_s2.csv:**
- It's the OUTPUT of parity pipeline (86 rows, already filtered)
- Must use `stage="full"` (no re-filtering)
- The file IS the parity set, but we load it with `stage="full"`

### Required E2E Test Changes:

**File:** `tests/e2e/test_reproduce_novo.py`

**Change 1:** Update fixture (line 61):
```python
# Before:
"jain_parity": "test_datasets/jain/canonical/VH_only_jain_test_PARITY_86.csv",  # Doesn't exist!

# After:
"jain_parity": "test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv",
```

**Change 2:** Update load_data call (line 101-103):
```python
# Before:
df_test = jain.load_data(
    full_csv_path=real_dataset_paths["jain_parity"],
    stage="parity"  # ❌ Tries to re-filter pre-filtered data
)

# After:
df_test = jain.load_data(
    full_csv_path=real_dataset_paths["jain_parity"],
    stage="full"  # ✅ Load pre-filtered file as-is
)
```

**Change 3:** Update skipif path (line 72-73):
```python
# Before:
or not Path("test_datasets/jain/canonical/VH_only_jain_test_PARITY_86.csv").exists(),

# After:
or not Path("test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv").exists(),
```

**Why this is correct:**
- File is already 86 rows (parity output)
- No filtering needed
- Just load, extract VH_sequence, and run model
- Test still validates Novo parity accuracy (that's the data, not the stage param)

---

## Proposed Solution - CORRECTED

### Generate VH file with CORRECT minimal schema:

```
Columns: ['id', 'vh_sequence', 'label']
```

**Why these columns:**
1. **id** - Required for tracking antibodies
2. **vh_sequence** - Required by JainDataset.load_data() (expects this exact name)
3. **label** - Required for training/evaluation

**Why NOT smp/elisa/source:**
- Not used by any downstream code
- Creates confusion (smp != psr column name)
- source is hardcoded 'jain2017' (useless metadata)
- Violates SSOT (canonical file doesn't have these)

---

## Detailed Implementation Plan

### File to Modify:
`preprocessing/jain/step2_preprocess_p5e_s2.py`

### Change 1: Add output path constant

**Location:** After line 44

**Current:**
```python
# Define output paths
BASE_DIR = Path(__file__).parent.parent.parent
OUTPUT_116 = BASE_DIR / "test_datasets/jain/processed/jain_ELISA_ONLY_116.csv"
OUTPUT_86 = BASE_DIR / "test_datasets/jain/canonical/jain_86_novo_parity.csv"
```

**New:**
```python
# Define output paths
BASE_DIR = Path(__file__).parent.parent.parent
OUTPUT_116 = BASE_DIR / "test_datasets/jain/processed/jain_ELISA_ONLY_116.csv"
OUTPUT_86 = BASE_DIR / "test_datasets/jain/canonical/jain_86_novo_parity.csv"
OUTPUT_VH = BASE_DIR / "test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv"  # VH-only benchmark
```

### Change 2: Generate VH-only file

**Location:** After line ~311 (after saving jain_86_novo_parity.csv)

**Current:**
```python
    # Save 86-antibody dataset
    print("\n" + "=" * 80)
    print("SAVING OUTPUTS")
    print("=" * 80)
    print()

    df_86.to_csv(OUTPUT_86, index=False)
    print(f"  ✅ Saved 86-antibody dataset → {OUTPUT_86.relative_to(BASE_DIR)}")
    print(f"     Confusion matrix: [[40, 19], [10, 17]]")
    print(f"     Accuracy: 66.28%")
```

**New:**
```python
    # Save 86-antibody dataset
    print("\n" + "=" * 80)
    print("SAVING OUTPUTS")
    print("=" * 80)
    print()

    # Save full canonical version
    df_86.to_csv(OUTPUT_86, index=False)
    print(f"  ✅ Saved 86-antibody dataset → {OUTPUT_86.relative_to(BASE_DIR)}")
    print(f"     Format: VH+VL+metadata (24 columns)")
    print(f"     Labels: 59 specific (0.0) + 27 non-specific (1.0)")
    print()

    # Save VH-only benchmark version
    # NOTE: Column must be 'vh_sequence' not 'sequence' for JainDataset.load_data() compatibility
    df_vh = df_86[['id', 'vh_sequence', 'label']].copy()
    df_vh.to_csv(OUTPUT_VH, index=False)
    print(f"  ✅ Saved VH-only benchmark → {OUTPUT_VH.relative_to(BASE_DIR)}")
    print(f"     Format: [id, vh_sequence, label] for model inference")
    print(f"     Labels: 59 specific (0.0) + 27 non-specific (1.0)")
    print()

    print(f"  📊 Confusion matrix: [[40, 19], [10, 17]]")
    print(f"  📈 Accuracy: 66.28%")
```

### Summary of Changes:
- **Lines added:** 10 (1 constant + 9 generation/logging)
- **Lines modified:** 2 (formatting improvements)
- **Column schema:** `['id', 'vh_sequence', 'label']` (CORRECTED)
- **Backwards compatible:** Yes (existing outputs unchanged)

---

## Expected Behavior After Change

### Output Files:

```
test_datasets/jain/canonical/
├── jain_86_novo_parity.csv (86 rows × 24 columns)
│   Schema: ['id', 'vh_sequence', 'vl_sequence', 'label', 'elisa_flags', 'psr', ...]
│   Labels: {0.0: 59, 1.0: 27}
│   Purpose: Full canonical dataset with all metadata
│
└── VH_only_jain_86_p5e_s2.csv (86 rows × 3 columns) ⭐ NEW
    Schema: ['id', 'vh_sequence', 'label']
    Labels: {0.0: 59, 1.0: 27}
    Purpose: VH-only benchmark for model inference
    Compatible with: JainDataset.load_data()
```

### Terminal Output:

```bash
$ python3 preprocessing/jain/step2_preprocess_p5e_s2.py

================================================================================
SAVING OUTPUTS
================================================================================

  ✅ Saved 86-antibody dataset → test_datasets/jain/canonical/jain_86_novo_parity.csv
     Format: VH+VL+metadata (24 columns)
     Labels: 59 specific (0.0) + 27 non-specific (1.0)

  ✅ Saved VH-only benchmark → test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv
     Format: [id, vh_sequence, label] for model inference
     Labels: 59 specific (0.0) + 27 non-specific (1.0)

  📊 Confusion matrix: [[40, 19], [10, 17]]
  📈 Accuracy: 66.28%
```

---

## Validation Plan - COMPREHENSIVE

### Test 1: Schema Validation
```python
import pandas as pd

# Load both files
canonical = pd.read_csv('test_datasets/jain/canonical/jain_86_novo_parity.csv')
vh = pd.read_csv('test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv')

# Verify schema
assert list(vh.columns) == ['id', 'vh_sequence', 'label'], f"Wrong columns: {vh.columns.tolist()}"
assert len(vh) == 86, f"Wrong row count: {len(vh)}"
assert vh['label'].value_counts()[0.0] == 59, "Should have 59 specific"
assert vh['label'].value_counts()[1.0] == 27, "Should have 27 non-specific"
print("✓ Schema validation passed")
```

### Test 2: Sequences Match
```python
# Sort both by id
canonical_sorted = canonical.sort_values('id').reset_index(drop=True)
vh_sorted = vh.sort_values('id').reset_index(drop=True)

# Verify same antibodies
assert set(canonical['id']) == set(vh['id']), "Different antibodies"

# Verify VH sequences match
assert (canonical_sorted['vh_sequence'] == vh_sorted['vh_sequence']).all(), "VH sequences don't match"

# Verify labels match
assert (canonical_sorted['label'] == vh_sorted['label']).all(), "Labels don't match"
print("✓ Sequence matching validation passed")
```

### Test 3: JainDataset Compatibility
```python
from antibody_training_esm.datasets import JainDataset

# Test loading through JainDataset (simulates E2E test)
jain = JainDataset()
df = jain.load_data(
    full_csv_path='test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv',
    stage='full'  # Don't apply parity filter (file already has 86)
)

# Verify column names after loading
assert 'VH_sequence' in df.columns, "JainDataset should have VH_sequence column"
assert 'label' in df.columns, "JainDataset should have label column"
assert len(df) == 86, f"Should have 86 rows, got {len(df)}"
print("✓ JainDataset compatibility passed")
```

### Test 4: E2E Test Compatibility
```python
# Simulate E2E test workflow
extractor_mock = lambda seqs: [[0.1] * 1280 for _ in seqs]  # Mock embeddings

# Load data (like E2E test does)
df_test = jain.load_data(
    full_csv_path='test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv',
    stage='full'
)

# Extract embeddings (like E2E test does)
X_test = extractor_mock(df_test["VH_sequence"].tolist())  # This line must not fail
y_test = df_test["label"].values

assert len(X_test) == 86, "Should extract 86 embeddings"
assert len(y_test) == 86, "Should have 86 labels"
print("✓ E2E test workflow simulation passed")
```

### Test 5: Full Pipeline Regeneration
```bash
# Delete generated files
rm -f test_datasets/jain/canonical/jain_86_novo_parity.csv
rm -f test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv
rm -f test_datasets/jain/processed/jain_ELISA_ONLY_116.csv

# Run full pipeline
python3 preprocessing/jain/step1_convert_excel_to_csv.py
python3 preprocessing/jain/step2_preprocess_p5e_s2.py

# Verify outputs exist
ls -lh test_datasets/jain/canonical/jain_86_novo_parity.csv
ls -lh test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv

# Run validation
python3 scripts/validation/validate_jain_csvs.py
```

---

## Risks and Mitigation

### Risk 1: Breaking Existing Code

**Analysis:**
- Current VH file has WRONG schema (incompatible with JainDataset)
- Fixing schema is NECESSARY, not optional
- Tests are currently BROKEN (expect file that doesn't exist)

**Mitigation:**
- New schema matches JainDataset.load_data() expectations
- Will fix currently broken tests
- Validated against E2E test workflow

### Risk 2: Data Type Mismatch (float vs int)

**Observation:**
- canonical has label as float64 (0.0, 1.0)
- old VH file had label as int (0, 1)

**Decision:** Use float64 (match canonical)
- Consistent with source (df_86 has float labels)
- No conversion needed
- Works with sklearn (accepts both)

### Risk 3: Missing Columns Users Expect

**Analysis:**
- smp, elisa, source are NOT used anywhere
- No code references these columns
- JainDataset.load_data() ignores them
- They don't exist in source data (psr != smp)

**Decision:** DELETE them
- Reduces confusion
- Aligns with SSOT (canonical file)
- Simpler schema

---

## Documentation Updates

### 1. preprocessing/jain/README.md

```markdown
## Step 2: Preprocess P5e-S2 (Novo Parity)

**Output:**
- `jain_ELISA_ONLY_116.csv` (116 antibodies, SSOT)
- `jain_86_novo_parity.csv` (86 antibodies, VH+VL+metadata)
- `VH_only_jain_86_p5e_s2.csv` (86 antibodies, VH-only for benchmarking) ⭐

**File Formats:**

| File | Rows | Columns | Purpose |
|------|------|---------|---------|
| jain_86_novo_parity.csv | 86 | 24 | Full data (VH+VL+metadata) |
| VH_only_jain_86_p5e_s2.csv | 86 | 3 | VH-only benchmark (model inference) |

**Column Schema - VH_only_jain_86_p5e_s2.csv:**
```
id: Antibody INN name
vh_sequence: VH amino acid sequence
label: 0.0 = specific, 1.0 = non-specific
```

Note: Column is `vh_sequence` (not `sequence`) for JainDataset compatibility.
```

### 2. test_datasets/jain/canonical/README.md

Update to document both files and their relationship

### 3. JAIN_TEST_FILE_ISSUE.md

Add resolution section documenting the fix

---

## Rollback Plan

```bash
# Option 1: Revert commit
git revert <commit-hash>

# Option 2: Restore old file from git
git checkout <old-commit> -- test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv

# Option 3: Comment out code
# Edit step2_preprocess_p5e_s2.py and comment lines
```

---

## Success Criteria

- [x] Minimal code change (<15 lines in step2)
- [x] Correct schema: ['id', 'vh_sequence', 'label']
- [x] JainDataset.load_data() compatibility (with stage="full")
- [x] E2E test updated to use stage="full"
- [x] E2E test file paths updated
- [x] No breaking changes to canonical file
- [x] Full pipeline reproducibility
- [x] All validation tests pass
- [x] Documentation updated

---

## Questions for Senior Review

### Q1: Column naming - vh_sequence vs sequence?
**Proposal:** Use `vh_sequence` (matches JainDataset expectations)
**Rationale:** JainDataset.load_data() expects this exact name

### Q2: Delete smp/elisa/source columns?
**Proposal:** YES - delete them
**Rationale:** Not used anywhere, don't exist in source data, violate SSOT

### Q3: Label dtype - float vs int?
**Proposal:** Use float64 (match canonical)
**Rationale:** Matches source data, consistent with canonical file

### Q4: E2E test changes - stage="parity" vs stage="full"?
**Current:** Uses `stage="parity"` on pre-filtered 86-row file
**Problem:** Tries to re-run parity filtering, fails on missing columns
**Proposal:** Use `stage="full"` (load as-is, no filtering)
**Rationale:** File is ALREADY the parity output, don't re-filter
**Action Required:** Yes, MUST update test or it will fail

### Q5: File naming - keep VH_only_jain_86_p5e_s2.csv?
**Alternatives:**
- `jain_86_vh_benchmark.csv` (clearer)
- `jain_86_vh_only.csv` (simpler)
**Current vote:** Keep existing name (less churn)

---

## Approval

**Review Checklist:**
- [ ] Schema is correct for JainDataset compatibility
- [ ] No unnecessary columns included
- [ ] Code change is minimal and clear
- [ ] E2E test stage="full" fix is included (CRITICAL)
- [ ] Validation plan is comprehensive
- [ ] Documentation updates are complete
- [ ] Rollback plan exists

**Signatures:**
- [ ] Senior Engineer 1: _________________ Date: _______
- [ ] Senior Engineer 2: _________________ Date: _______
- [ ] Team Lead: _________________ Date: _______

---

## Implementation Steps (After Approval)

```bash
# 1. Create feature branch
git checkout -b fix/generate-vh-benchmark-in-step2

# ========================================
# PART 1: Update preprocessing script
# ========================================

# 2. Make changes to step2_preprocess_p5e_s2.py (see "Detailed Implementation Plan")

# 3. Delete old manually-created file
rm test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv

# 4. Run step2 to generate new file with correct schema
python3 preprocessing/jain/step2_preprocess_p5e_s2.py

# 5. Verify new file schema
python3 -c "
import pandas as pd
df = pd.read_csv('test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv')
assert list(df.columns) == ['id', 'vh_sequence', 'label'], f'Wrong schema: {df.columns.tolist()}'
assert len(df) == 86, f'Wrong row count: {len(df)}'
assert df['label'].value_counts()[0.0] == 59, 'Should have 59 specific'
assert df['label'].value_counts()[1.0] == 27, 'Should have 27 non-specific'
print('✅ VH file schema correct!')
"

# ========================================
# PART 2: Fix E2E tests (CRITICAL)
# ========================================

# 6. Update E2E test to use stage="full" instead of stage="parity"
# Edit tests/e2e/test_reproduce_novo.py:
#   - Line 61: Update fixture path
#   - Line 72-73: Update skipif path
#   - Line 101-103: Change stage="parity" to stage="full"

# 7. Run E2E test to verify it works
pytest tests/e2e/test_reproduce_novo.py::test_reproduce_novo_jain_accuracy_with_real_data -v

# ========================================
# PART 3: Run validation suite
# ========================================

# 8. Run all validation tests
python3 -c "$(cat validation_tests.py)"
python3 scripts/validation/validate_jain_csvs.py

# ========================================
# PART 4: Update documentation
# ========================================

# 9. Update documentation
# Edit preprocessing/jain/README.md (see "Documentation Updates")
# Edit test_datasets/jain/canonical/README.md

# ========================================
# PART 5: Commit and PR
# ========================================

# 10. Commit all changes
git add preprocessing/jain/step2_preprocess_p5e_s2.py
git add test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv
git add tests/e2e/test_reproduce_novo.py
git add preprocessing/jain/README.md
git add test_datasets/jain/canonical/README.md
git commit -m "fix: Generate VH benchmark in step2 with correct schema and fix E2E tests

- Add VH_only_jain_86_p5e_s2.csv generation to step2
- Fix schema: [id, vh_sequence, label] (was: [id, sequence, label, smp, elisa, source])
- Update E2E test to use stage='full' for pre-filtered files
- Update file paths in tests (VH_only_jain_test_PARITY_86.csv -> VH_only_jain_86_p5e_s2.csv)
- Update documentation

Closes #XXX"

# 11. Push and create PR
git push origin fix/generate-vh-benchmark-in-step2
```

---

**END OF PLAN v3.0**
