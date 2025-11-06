# Shehata Dataset Cleanup - Complete Plan

**Date:** 2025-11-05
**Branch:** `leroy-jenkins/shehata-cleanup`
**Status:** ✅ **READY FOR EXECUTION**

---

## Executive Summary

Complete cleanup of Shehata dataset AND script organization, following Rob C. Martin discipline.

**Scope:**
1. Reorganize Shehata to 4-tier structure (raw/ → processed/ → canonical/ → fragments/)
2. Update 8 Python scripts with new paths
3. Update 35+ documentation references
4. Clean up duplicate scripts (scripts/ root vs scripts/*/subdirectories)

**Validation:** All claims verified from first principles.

---

## Decisions Made (Final)

1. **Script duplication:** **Option B** - Delete root duplicates, use subdirectory versions
2. **canonical/ directory:** **Empty with README** (Shehata is external test set)
3. **mmc3/4/5 files:** **Move to raw/** (complete provenance)

---

## Part 1: File Reorganization

### Current State
```
test_datasets/
├── shehata-mmc2.xlsx          (Main data - 402 rows)
├── shehata-mmc3.xlsx          (Unused)
├── shehata-mmc4.xlsx          (Unused)
├── shehata-mmc5.xlsx          (Unused)
├── shehata.csv                (398 antibodies)
└── shehata/                   (16 fragment CSVs)
```

### Target State
```
test_datasets/shehata/
├── README.md
├── raw/
│   ├── README.md
│   ├── shehata-mmc2.xlsx
│   ├── shehata-mmc3.xlsx
│   ├── shehata-mmc4.xlsx
│   └── shehata-mmc5.xlsx
├── processed/
│   ├── README.md
│   └── shehata.csv (398 antibodies)
├── canonical/
│   └── README.md (empty - external test set)
└── fragments/
    ├── README.md
    └── [16 fragment CSVs]
```

### File Move Commands
```bash
# Create structure
mkdir -p test_datasets/shehata/{raw,processed,canonical,fragments}

# Move Excel files to raw/
mv test_datasets/shehata-mmc2.xlsx test_datasets/shehata/raw/
mv test_datasets/shehata-mmc3.xlsx test_datasets/shehata/raw/
mv test_datasets/shehata-mmc4.xlsx test_datasets/shehata/raw/
mv test_datasets/shehata-mmc5.xlsx test_datasets/shehata/raw/

# Move processed CSV
mv test_datasets/shehata.csv test_datasets/shehata/processed/

# Move fragments
mv test_datasets/shehata/*.csv test_datasets/shehata/fragments/
```

---

## Part 2: Python Script Updates (8 files)

### Core Scripts (3 files)

**1. preprocessing/shehata/step1_convert_excel_to_csv.py**
```python
# Line 275-276:
# OLD:
excel_path = Path("test_datasets/mmc2.xlsx")  # ❌ File doesn't exist
output_path = Path("test_datasets/shehata.csv")

# NEW:
excel_path = Path("test_datasets/shehata/raw/shehata-mmc2.xlsx")
output_path = Path("test_datasets/shehata/processed/shehata.csv")
```

**2. preprocessing/shehata/step2_extract_fragments.py**
```python
# Line 220-221:
# OLD:
csv_path = Path("test_datasets/shehata.csv")
output_dir = Path("test_datasets/shehata")

# NEW:
csv_path = Path("test_datasets/shehata/processed/shehata.csv")
output_dir = Path("test_datasets/shehata/fragments")
```

**3. scripts/validation/validate_shehata_conversion.py**
```python
# Line 194-195, 287:
# OLD:
excel_path = Path("test_datasets/mmc2.xlsx")  # ❌ File doesn't exist
csv_path = Path("test_datasets/shehata.csv")
fragments_dir = Path("test_datasets/shehata")

# NEW:
excel_path = Path("test_datasets/shehata/raw/shehata-mmc2.xlsx")
csv_path = Path("test_datasets/shehata/processed/shehata.csv")
fragments_dir = Path("test_datasets/shehata/fragments")
```

### Analysis/Testing Scripts (5 files)

**4. scripts/analysis/analyze_threshold_optimization.py**
```python
# Line 170, 194:
"test_datasets/shehata/VH_only_shehata.csv"
→ "test_datasets/shehata/fragments/VH_only_shehata.csv"
```

**5. scripts/testing/demo_assay_specific_thresholds.py**
```python
# Line 96:
"test_datasets/shehata/VH_only_shehata.csv"
→ "test_datasets/shehata/fragments/VH_only_shehata.csv"
```

**6. scripts/validate_fragments.py**
```python
# Line 193:
("shehata", Path("test_datasets/shehata"), 16)
→ ("shehata", Path("test_datasets/shehata/fragments"), 16)
```

**7. scripts/validation/validate_fragments.py**
```python
# Line 193 (same as #6, duplicate file):
("shehata", Path("test_datasets/shehata"), 16)
→ ("shehata", Path("test_datasets/shehata/fragments"), 16)
```

**8. tests/test_shehata_embedding_compatibility.py**
```python
# Lines 25, 59, 114, 163, 200:
fragments_dir = Path("test_datasets/shehata")
→ fragments_dir = Path("test_datasets/shehata/fragments")
```

---

## Part 3: Script Duplication Cleanup

### Duplicates to Delete (4 files)

**Delete these root-level scripts:**
```bash
rm scripts/convert_jain_excel_to_csv.py        # Use scripts/conversion/ version
rm scripts/convert_harvey_csvs.py              # Use scripts/conversion/ version
rm scripts/validate_jain_conversion.py         # Use scripts/validation/ version
rm scripts/validate_fragments.py               # Use scripts/validation/ version
```

**Canonical versions (keep these):**
- ✅ `preprocessing/jain/step1_convert_excel_to_csv.py`
- ✅ `preprocessing/harvey/step1_convert_raw_csvs.py`
- ✅ `scripts/validation/validate_jain_conversion.py`
- ✅ `scripts/validation/validate_fragments.py`

---

## Part 4: Documentation Updates (35+ files)

### Root README.md (2 lines)

**File:** `README.md`

**Lines 109-110:**
```markdown
# OLD:
- `test_datasets/shehata.csv` - Full paired VH+VL sequences (398 antibodies)
- `test_datasets/shehata/*.csv` - 16 fragment-specific files (...)

# NEW:
- `test_datasets/shehata/processed/shehata.csv` - Full paired VH+VL sequences (398 antibodies)
- `test_datasets/shehata/fragments/*.csv` - 16 fragment-specific files (...)
```

### Top-Level Docs (3 files)

**File:** `docs/COMPLETE_VALIDATION_RESULTS.md`

**Line 128:**
```markdown
# OLD:
**Test file**: `test_datasets/shehata/VH_only_shehata.csv`

# NEW:
**Test file**: `test_datasets/shehata/fragments/VH_only_shehata.csv`
```

**File:** `docs/BENCHMARK_TEST_RESULTS.md`

**Line 75:**
```markdown
# OLD:
**Test file:** `test_datasets/shehata/VH_only_shehata.csv`

# NEW:
**Test file:** `test_datasets/shehata/fragments/VH_only_shehata.csv`
```

**File:** `docs/ASSAY_SPECIFIC_THRESHOLDS.md`

**Line 143:**
```python
# OLD:
df = pd.read_csv("test_datasets/shehata/VH_only_shehata.csv")

# NEW:
df = pd.read_csv("test_datasets/shehata/fragments/VH_only_shehata.csv")
```

### Shehata-Specific Docs (7 files in docs/shehata/)

**Update all references in:**
1. `docs/shehata/shehata_preprocessing_implementation_plan.md`
2. `docs/shehata/shehata_data_sources.md`
3. `docs/shehata/shehata_phase2_completion_report.md`
4. `docs/shehata/shehata_conversion_verification_report.md`
5. `docs/shehata/SHEHATA_BLOCKER_ANALYSIS.md`
6. `docs/shehata/P0_BLOCKER_FIRST_PRINCIPLES_VALIDATION.md`
7. `docs/shehata/shehata_preprocessing_implementation_plan.md`

**Pattern (apply to all 7 files):**
```bash
# Find and replace across all docs/shehata/ files:
test_datasets/shehata.csv → test_datasets/shehata/processed/shehata.csv
test_datasets/shehata/*.csv → test_datasets/shehata/fragments/*.csv
test_datasets/shehata/VH_only → test_datasets/shehata/fragments/VH_only
test_datasets/mmc2.xlsx → test_datasets/shehata/raw/shehata-mmc2.xlsx
```

### Harvey Docs (if Option B - script cleanup)

**Update references to root-level scripts:**

**File:** `docs/harvey_data_cleaning_log.md` (7 references)
```bash
scripts/convert_harvey_csvs.py → preprocessing/harvey/step1_convert_raw_csvs.py
scripts/validate_fragments.py → scripts/validation/validate_fragments.py
```

**File:** `docs/harvey_data_sources.md` (3 references)
```bash
scripts/convert_harvey_csvs.py → preprocessing/harvey/step1_convert_raw_csvs.py
```

**File:** `docs/harvey/harvey_script_status.md` (3 references)
```bash
scripts/validate_fragments.py → scripts/validation/validate_fragments.py
```

---

## Part 5: Create New READMEs (5 files)

### 1. test_datasets/shehata/README.md (Master)

**Content:**
- Citation (Shehata 2019 + Sakhnini 2025)
- Quick start guide
- Data flow diagram
- Link to subdirectory READMEs
- Verification commands

### 2. test_datasets/shehata/raw/README.md

**Content:**
- Original Excel files description
- Citation details
- DO NOT MODIFY warning
- Note: mmc3/4/5 unused but archived for provenance
- Conversion instructions

### 3. test_datasets/shehata/processed/README.md

**Content:**
- shehata.csv description (398 antibodies)
- PSR score thresholding (98.24th percentile)
- Label distribution (391 specific, 7 non-specific)
- Regeneration instructions

### 4. test_datasets/shehata/canonical/README.md

**Content:**
- Explanation: Shehata is external test set
- No subsampling needed (unlike Jain)
- Full 398-antibody dataset in processed/ is canonical
- canonical/ kept empty for consistency with Jain structure

### 5. test_datasets/shehata/fragments/README.md

**Content:**
- 16 fragment types explained
- ANARCI/IMGT numbering methodology
- Usage examples for each fragment
- Fragment type use cases
- **CRITICAL:** Note about P0 blocker (gap characters) and fix

---

## Verification Plan (Complete)

### 1. File Move Verification
```bash
echo "Raw files (4):" && ls -1 test_datasets/shehata/raw/*.xlsx | wc -l
echo "Processed files (1):" && ls -1 test_datasets/shehata/processed/*.csv | wc -l
echo "Fragment files (16):" && ls -1 test_datasets/shehata/fragments/*.csv | wc -l
echo "Total CSVs (17):" && find test_datasets/shehata -name "*.csv" | wc -l
```

### 2. P0 Blocker Regression Check (CRITICAL)
```bash
grep -c '\-' test_datasets/shehata/fragments/*.csv | grep -v ':0$'
# Should return NOTHING (all files have 0 gaps)
```

### 3. Row Count Validation
```bash
for f in test_datasets/shehata/fragments/*.csv; do
  count=$(wc -l < "$f")
  if [ "$count" -ne 399 ]; then
    echo "ERROR: $f has $count lines (expected 399)"
  fi
done
```

### 4. Label Distribution Check
```bash
python3 -c "
import pandas as pd
files = ['processed/shehata.csv', 'fragments/VH_only_shehata.csv', 'fragments/Full_shehata.csv']
for f in files:
    path = f'test_datasets/shehata/{f}'
    df = pd.read_csv(path)
    dist = df['label'].value_counts().sort_index().to_dict()
    expected = {0: 391, 1: 7}
    status = '✅' if dist == expected else '❌'
    print(f'{status} {f}: {dist}')
"
```

### 5. Script Regeneration Test
```bash
python3 preprocessing/shehata/step1_convert_excel_to_csv.py  # Should work
python3 preprocessing/shehata/step2_extract_fragments.py                     # Should work
```

### 6. Comprehensive Validation
```bash
python3 scripts/validation/validate_shehata_conversion.py  # Should pass all checks
python3 tests/test_shehata_embedding_compatibility.py      # Should pass all tests
```

### 7. Model Test
```bash
python3 test.py --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/shehata/fragments/VH_only_shehata.csv
# Should load and run successfully
```

### 8. Documentation Validation
```bash
# Check no references to old paths remain
grep -rn "test_datasets/shehata\.csv" docs/ README.md --include="*.md" | grep -v "processed/"
# Should return NOTHING

grep -rn "test_datasets/mmc2\.xlsx" docs/ --include="*.md" | grep -v "raw/"
# Should return NOTHING

# Check no references to deleted root scripts
grep -rn "scripts/convert_jain_excel" docs/ --include="*.md" | grep -v "conversion/"
grep -rn "scripts/convert_harvey" docs/ --include="*.md" | grep -v "conversion/"
grep -rn "scripts/validate_jain" docs/ --include="*.md" | grep -v "validation/"
grep -rn "scripts/validate_fragments" docs/ --include="*.md" | grep -v "validation/"
# All should return NOTHING
```

---

## Execution Order (Critical)

**Execute in this exact order:**

### Phase 1: Prepare
1. ✅ Create branch `leroy-jenkins/shehata-cleanup`
2. Create directory structure
3. Create all 5 READMEs

### Phase 2: Move Files
4. Move 4 Excel files → raw/
5. Move shehata.csv → processed/
6. Move 16 fragments → fragments/

### Phase 3: Update Python Scripts
7. Update 3 core scripts (conversion, processing, validation)
8. Update 5 analysis/testing scripts

### Phase 4: Clean Duplicate Scripts
9. Delete 4 root-level duplicate scripts
10. Verify no broken imports

### Phase 5: Update Documentation
11. Update README.md (2 lines)
12. Update 3 top-level docs
13. Update 7 shehata docs
14. Update 3 harvey docs (script references)

### Phase 6: Verify
15. Run all 8 verification checks
16. Confirm all pass

### Phase 7: Commit
17. Commit with detailed message
18. Push to branch

---

## Complete File Checklist

### Python Files to Modify (8)
- [ ] `preprocessing/shehata/step1_convert_excel_to_csv.py`
- [ ] `preprocessing/shehata/step2_extract_fragments.py`
- [ ] `scripts/validation/validate_shehata_conversion.py`
- [ ] `scripts/analysis/analyze_threshold_optimization.py`
- [ ] `scripts/testing/demo_assay_specific_thresholds.py`
- [ ] `scripts/validate_fragments.py`
- [ ] `scripts/validation/validate_fragments.py`
- [ ] `tests/test_shehata_embedding_compatibility.py`

### Python Files to Delete (4)
- [ ] `scripts/convert_jain_excel_to_csv.py`
- [ ] `scripts/convert_harvey_csvs.py`
- [ ] `scripts/validate_jain_conversion.py`
- [ ] `scripts/validate_fragments.py`

### READMEs to Create (5)
- [ ] `test_datasets/shehata/README.md`
- [ ] `test_datasets/shehata/raw/README.md`
- [ ] `test_datasets/shehata/processed/README.md`
- [ ] `test_datasets/shehata/canonical/README.md`
- [ ] `test_datasets/shehata/fragments/README.md`

### Documentation Files to Update (13)
- [ ] `README.md` (2 lines)
- [ ] `docs/COMPLETE_VALIDATION_RESULTS.md` (1 line)
- [ ] `docs/BENCHMARK_TEST_RESULTS.md` (1 line)
- [ ] `docs/ASSAY_SPECIFIC_THRESHOLDS.md` (1 line)
- [ ] `docs/shehata/shehata_preprocessing_implementation_plan.md`
- [ ] `docs/shehata/shehata_data_sources.md`
- [ ] `docs/shehata/shehata_phase2_completion_report.md`
- [ ] `docs/shehata/shehata_conversion_verification_report.md`
- [ ] `docs/shehata/SHEHATA_BLOCKER_ANALYSIS.md`
- [ ] `docs/shehata/P0_BLOCKER_FIRST_PRINCIPLES_VALIDATION.md`
- [ ] `docs/harvey_data_cleaning_log.md` (7 refs)
- [ ] `docs/harvey_data_sources.md` (3 refs)
- [ ] `docs/harvey/harvey_script_status.md` (3 refs)

---

## Time Estimate

**Total: 60-75 minutes**

- Phase 1 (Prepare): 5 min
- Phase 2 (Move): 5 min
- Phase 3 (Scripts): 15 min
- Phase 4 (Duplicates): 5 min
- Phase 5 (Docs): 20 min
- Phase 6 (Verify): 10 min
- Phase 7 (Commit): 5 min

---

## Citation (Correct)

**Dataset Source:**
Shehata L, Thaventhiran JED, Engelhardt KR, et al. (2019). "Affinity Maturation Enhances Antibody Specificity but Compromises Conformational Stability." *Cell Reports* 28(13):3300-3308.e4.
DOI: 10.1016/j.celrep.2019.08.056

**Methodology Source:**
Sakhnini A, et al. (2025). "Antibody Non-Specificity Prediction using Protein Language Models and Biophysical Features." *Cell*.
DOI: 10.1016/j.cell.2024.12.025

---

**Status:** ✅ **COMPLETE PLAN - READY TO EXECUTE**

All feedback validated from first principles. Plan now includes:
- ✅ All 8 Python script updates
- ✅ All 4 duplicate script deletions
- ✅ All 35+ documentation updates
- ✅ Complete verification checklist
- ✅ Clear execution order

**Awaiting your go-ahead to execute, boss.**
