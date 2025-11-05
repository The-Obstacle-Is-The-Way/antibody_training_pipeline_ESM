# Harvey Dataset Cleanup - Senior Investigation

**Date:** 2025-11-05
**Branch:** leroy-jenkins/full-send
**Status:** 🔍 **INVESTIGATION - AWAITING SENIOR APPROVAL**

---

## Executive Summary

Harvey dataset structure is **MESSY** and requires cleanup similar to Shehata/Jain reorganization.

**Current Problems:**
1. ❌ Raw source files NOT in test_datasets/ (in reference_repos/)
2. ❌ Processed files scattered (3 CSVs at root, 6 in subdirectory)
3. ❌ No clear data flow documentation
4. ❌ No README files in test_datasets/harvey/
5. ❌ Inconsistent with Shehata/Jain 4-tier structure

**Recommendation:** Apply same 4-tier cleanup (raw → processed → canonical → fragments)

---

## Current State (MESSY)

### File Layout

```
reference_repos/harvey_official_repo/backend/app/experiments/
├── high_polyreactivity_high_throughput.csv (71,772 + header)
├── low_polyreactivity_high_throughput.csv (69,702 + header)
└── low_throughput_polyspecificity_scores_w_exp.csv (48 + header)

test_datasets/  (ROOT LEVEL - BAD)
├── harvey.csv (141,474 antibodies + header = 141,475 lines)
├── harvey_high.csv (71,772 + header = 71,773 lines)
├── harvey_low.csv (69,702 + header = 69,703 lines)
└── harvey/  (SUBDIRECTORY - MIXED PURPOSE)
    ├── H-CDR1_harvey.csv (141,021 + header)
    ├── H-CDR2_harvey.csv (141,021 + header)
    ├── H-CDR3_harvey.csv (141,021 + header)
    ├── H-CDRs_harvey.csv (141,021 + header)
    ├── H-FWRs_harvey.csv (141,021 + header)
    ├── VHH_only_harvey.csv (141,021 + header)
    └── failed_sequences.txt (453 failed ANARCI annotations)
```

### Problems Identified

**P1: Raw sources outside test_datasets/**
- Raw data in `reference_repos/` not version controlled with dataset
- Should be copied/symlinked to `test_datasets/harvey/raw/`
- Breaking principle: "All data sources in test_datasets/"

**P2: Processed files at root level**
- `harvey.csv`, `harvey_high.csv`, `harvey_low.csv` at test_datasets/ root
- Should be in `test_datasets/harvey/processed/`
- Breaking principle: "Organized by dataset, not scattered"

**P3: No canonical/ directory**
- Harvey is training set (not external test like Shehata)
- Should have canonical benchmarks similar to Boughter
- Breaking principle: "Consistent 4-tier structure"

**P4: Mixed purpose harvey/ directory**
- Currently contains only fragments
- Should be `harvey/fragments/` specifically
- Breaking principle: "Single Responsibility - one dir, one purpose"

**P5: No README documentation**
- No provenance documentation in harvey/ directory
- No data flow explanation
- Breaking principle: "Self-documenting structure"

**P6: Inconsistent with Shehata/Jain cleanup**
- Shehata/Jain now have clean 4-tier structure
- Harvey still has old messy structure
- Breaking principle: "Consistent patterns across datasets"

---

## Data Flow Analysis

### Current Flow (Undocumented)

```
reference_repos/harvey_official_repo/backend/app/experiments/
  ├── high_polyreactivity_high_throughput.csv (71,772)
  └── low_polyreactivity_high_throughput.csv (69,702)
    ↓ [scripts/conversion/convert_harvey_csvs.py]
test_datasets/harvey.csv (141,474 combined)
  ↓ [preprocessing/process_harvey.py + ANARCI]
test_datasets/harvey/ fragments (141,021 - 453 failures)
```

**Missing intermediate files:**
- harvey_high.csv and harvey_low.csv appear to be copies from reference_repos
- Purpose unclear (are they needed? duplicates?)
- No documentation explaining their role

**ANARCI failures:**
- 453 sequences failed annotation (0.32% failure rate)
- Documented in failed_sequences.txt
- Acceptable loss, but should be tracked in README

---

## Proposed Structure (CLEAN)

### Target Layout

```
test_datasets/harvey/
├── README.md                  ← Master guide
├── raw/                       ← Original sources (DO NOT MODIFY)
│   ├── README.md
│   ├── high_polyreactivity_high_throughput.csv (71,772)
│   ├── low_polyreactivity_high_throughput.csv (69,702)
│   └── low_throughput_polyspecificity_scores_w_exp.csv (48 - optional)
├── processed/                 ← Converted datasets
│   ├── README.md
│   ├── harvey.csv (141,474 combined - SSOT)
│   ├── harvey_high.csv (71,772 - intermediate, optional)
│   └── harvey_low.csv (69,702 - intermediate, optional)
├── canonical/                 ← Final benchmarks
│   ├── README.md
│   └── [TO BE DETERMINED - training splits? balanced subsets?]
└── fragments/                 ← Region-specific extracts
    ├── README.md
    ├── VHH_only_harvey.csv (141,021)
    ├── H-CDR1/2/3_harvey.csv
    ├── H-CDRs_harvey.csv
    ├── H-FWRs_harvey.csv
    └── failed_sequences.txt (453 failures logged)
```

---

## Comparison with Clean Datasets

### Shehata (CLEAN) ✅

```
shehata/
├── raw/ (4 Excel files)
├── processed/ (shehata.csv - 398 antibodies)
├── canonical/ (empty - external test set)
└── fragments/ (16 fragments)
```

**Benefits:**
- Clear separation of stages
- Complete provenance documentation
- Reproducible pipelines
- Self-documenting with READMEs

### Jain (CLEAN) ✅

```
jain/
├── raw/ (3 PNAS Excel + 1 private ELISA)
├── processed/ (jain.csv, jain_ELISA_ONLY_116.csv)
├── canonical/ (jain_86_novo_parity.csv)
└── fragments/ (16 fragments + extras)
```

**Benefits:**
- Same 4-tier structure
- Benchmarks in canonical/
- All derived files reproducible

### Harvey (MESSY) ❌

```
reference_repos/harvey_official_repo/ (raw - WRONG LOCATION)
test_datasets/harvey.csv (root - WRONG LOCATION)
test_datasets/harvey_high.csv (root - WRONG LOCATION)
test_datasets/harvey_low.csv (root - WRONG LOCATION)
test_datasets/harvey/ (fragments only - MIXED PURPOSE)
```

**Problems:**
- No consistent structure
- Files scattered across locations
- No provenance documentation
- Inconsistent with other datasets

---

## Cleanup Scope

### Files to Move (10 files)

**From reference_repos → raw/:**
- high_polyreactivity_high_throughput.csv (copy or symlink)
- low_polyreactivity_high_throughput.csv (copy or symlink)
- low_throughput_polyspecificity_scores_w_exp.csv (optional)

**From test_datasets/ root → processed/:**
- harvey.csv
- harvey_high.csv (decision: keep or delete?)
- harvey_low.csv (decision: keep or delete?)

**From test_datasets/harvey/ → fragments/:**
- H-CDR1_harvey.csv
- H-CDR2_harvey.csv
- H-CDR3_harvey.csv
- H-CDRs_harvey.csv
- H-FWRs_harvey.csv
- VHH_only_harvey.csv
- failed_sequences.txt

### Scripts to Update (2 files)

**1. scripts/conversion/convert_harvey_csvs.py**
```python
# OLD:
input_high = "reference_repos/harvey_official_repo/backend/app/experiments/high_polyreactivity_high_throughput.csv"
input_low = "reference_repos/harvey_official_repo/backend/app/experiments/low_polyreactivity_high_throughput.csv"
output = "test_datasets/harvey.csv"

# NEW:
input_high = "test_datasets/harvey/raw/high_polyreactivity_high_throughput.csv"
input_low = "test_datasets/harvey/raw/low_polyreactivity_high_throughput.csv"
output = "test_datasets/harvey/processed/harvey.csv"
```

**2. preprocessing/process_harvey.py**
```python
# OLD:
csv_path = "test_datasets/harvey.csv"
output_dir = "test_datasets/harvey"

# NEW:
csv_path = "test_datasets/harvey/processed/harvey.csv"
output_dir = "test_datasets/harvey/fragments"
```

### Documentation to Update (8+ files)

**Harvey-specific docs (5 files in docs/harvey/):**
- harvey_data_sources.md
- harvey_data_cleaning_log.md
- harvey_preprocessing_implementation_plan.md
- harvey_script_status.md
- HARVEY_P0_FIX_REPORT.md

**Root-level docs (2 files):**
- docs/harvey_data_sources.md
- docs/harvey_data_cleaning_log.md

**README references:**
- README.md (root)

### READMEs to Create (5 files)

1. **test_datasets/harvey/README.md** (master guide)
   - Dataset overview
   - Data flow diagram
   - Citation information
   - Quick start guide
   - Verification commands

2. **test_datasets/harvey/raw/README.md**
   - Original source files
   - Data provenance
   - Citation (Harvey et al., Mason et al.)
   - DO NOT MODIFY warning
   - Conversion instructions

3. **test_datasets/harvey/processed/README.md**
   - CSV conversion details
   - Label assignment (0=low poly, 1=high poly)
   - Label distribution (49.1% / 50.9%)
   - harvey_high/low.csv purpose
   - Regeneration instructions

4. **test_datasets/harvey/canonical/README.md**
   - Purpose: Training benchmarks
   - Decision needed: balanced subsets? cross-validation splits?
   - Comparison with Boughter canonical/

5. **test_datasets/harvey/fragments/README.md**
   - 6 fragment types (VHH only - nanobodies)
   - ANARCI annotation details
   - Failed sequences (453 - 0.32%)
   - Fragment use cases

---

## Key Decisions Required

### Decision 1: Raw Data Location

**Question:** Copy or symlink reference_repos files to test_datasets/harvey/raw/?

**Options:**
- **Option A: Copy files** (15MB + 15MB = 30MB)
  - ✅ Self-contained test_datasets/
  - ✅ No external dependencies
  - ❌ Duplicated data (uses more space)

- **Option B: Symlink files**
  - ✅ No duplication
  - ✅ Single source of truth
  - ❌ Breaks if reference_repos/ moved

- **Option C: Keep in reference_repos, update paths**
  - ✅ No duplication
  - ❌ External dependency
  - ❌ Inconsistent with Shehata/Jain

**Recommendation:** **Option A (Copy)** - Consistency with Shehata/Jain, self-contained

### Decision 2: harvey_high.csv and harvey_low.csv

**Question:** Keep or delete intermediate files?

**Current state:**
- harvey_high.csv = copy of raw/high_polyreactivity_high_throughput.csv
- harvey_low.csv = copy of raw/low_polyreactivity_high_throughput.csv
- Both used as input to scripts/conversion/convert_harvey_csvs.py

**Options:**
- **Option A: Keep in processed/**
  - ✅ Explicit intermediate files
  - ✅ Can regenerate harvey.csv from these
  - ❌ Duplicated data (3x storage)

- **Option B: Delete, use raw/ directly**
  - ✅ DRY principle (no duplication)
  - ✅ Scripts read directly from raw/
  - ❌ Loses intermediate checkpoint

**Recommendation:** **Option B (Delete)** - Scripts should read from raw/, output to processed/harvey.csv

### Decision 3: canonical/ Contents

**Question:** What benchmarks belong in harvey/canonical/?

**Harvey characteristics:**
- 141,021 nanobodies (training set)
- Balanced classes (49.1% / 50.9%)
- High-throughput dataset (not curated like Jain)

**Options:**
- **Option A: Empty (like Shehata)**
  - Use full 141,021 dataset directly
  - No subsampling needed

- **Option B: Balanced subset**
  - Create 10k balanced subset for quick testing
  - Similar to Boughter canonical/

- **Option C: Cross-validation splits**
  - Pre-defined train/val splits
  - Ensures consistent benchmarking

**Recommendation:** **Option A (Empty)** - Full dataset is already balanced, no need for canonical subsets

---

## Verification Plan

### 1. File Move Verification
```bash
echo "Raw files (3):" && ls -1 test_datasets/harvey/raw/*.csv | wc -l
echo "Processed files (1):" && ls -1 test_datasets/harvey/processed/*.csv | wc -l
echo "Fragment files (6):" && ls -1 test_datasets/harvey/fragments/*.csv | wc -l
echo "Total CSVs (10):" && find test_datasets/harvey -name "*.csv" | wc -l
```

### 2. Row Count Validation
```bash
# Processed should have 141,474 + header
wc -l test_datasets/harvey/processed/harvey.csv  # Should be 141,475

# All fragments should have 141,021 + header
for f in test_datasets/harvey/fragments/*.csv; do
  count=$(wc -l < "$f")
  if [ "$count" -ne 141022 ]; then
    echo "ERROR: $f has $count lines (expected 141022)"
  fi
done
```

### 3. Label Distribution Check
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('test_datasets/harvey/processed/harvey.csv')
dist = df['label'].value_counts().sort_index().to_dict()
expected = {0: 69702, 1: 71772}  # low (0) and high (1) polyreactivity
print(f'Label distribution: {dist}')
print(f'Expected: {expected}')
print('Match:', dist == expected)
"
```

### 4. Script Regeneration Test
```bash
# Test conversion script
python3 scripts/conversion/convert_harvey_csvs.py

# Test fragment extraction
python3 preprocessing/process_harvey.py
```

### 5. Fragment Validation
```bash
python3 scripts/validation/validate_fragments.py
# Should validate harvey fragments
```

### 6. Model Test
```bash
python3 test.py --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/harvey/fragments/VHH_only_harvey.csv
# Should load and run successfully
```

### 7. Failed Sequences Check
```bash
# Verify failed_sequences.txt has 453 entries
wc -l test_datasets/harvey/fragments/failed_sequences.txt  # Should be 453
```

### 8. Documentation Validation
```bash
# Check no references to old paths remain
grep -rn "test_datasets/harvey\.csv" docs/ README.md --include="*.md" | grep -v "processed/"
# Should return NOTHING

grep -rn "reference_repos/harvey_official_repo" scripts/ --include="*.py"
# Should return NOTHING
```

---

## Execution Plan (7 Phases)

**Estimated time:** 45-60 minutes

### Phase 1: Prepare (5 min)
- Create directory structure: `test_datasets/harvey/{raw,processed,canonical,fragments}`
- Create 5 comprehensive READMEs

### Phase 2: Move Raw Files (5 min)
- Copy 3 CSV files from reference_repos/ → raw/

### Phase 3: Move Processed Files (2 min)
- Move harvey.csv → processed/
- Delete harvey_high.csv and harvey_low.csv (Decision 2)

### Phase 4: Move Fragments (2 min)
- Move 6 fragment CSVs → fragments/
- Move failed_sequences.txt → fragments/

### Phase 5: Update Scripts (10 min)
- Update scripts/conversion/convert_harvey_csvs.py (paths)
- Update preprocessing/process_harvey.py (paths)

### Phase 6: Update Documentation (20 min)
- Update 8+ markdown files with new paths
- Update README.md

### Phase 7: Verify (10 min)
- Run all 8 verification checks
- Ensure all pass

---

## Risk Assessment

### Low Risk ✅
- **Harvey has good docs** (5 docs in docs/harvey/, 2 at root)
- **Simple structure** (only 2 scripts, 6 fragments)
- **No P0 blockers** (ANARCI issues already resolved)
- **Balanced dataset** (no label issues)
- **Reference implementation** (Shehata cleanup already done)

### Medium Risk ⚠️
- **Raw data dependency** (reference_repos/ outside version control)
- **Intermediate files** (harvey_high/low.csv purpose unclear)
- **canonical/ decision** (empty vs. subsets?)

### Mitigation
- Copy raw files to test_datasets/ (self-contained)
- Delete intermediate files (simplify)
- Start with empty canonical/ (add later if needed)

---

## Comparison with Shehata Cleanup

### Similarities
- Both need 4-tier structure
- Both have fragments in subdirectory
- Both need README documentation
- Both need script path updates
- Both need doc updates

### Differences
- **Harvey is SIMPLER:**
  - Only 6 fragments (vs 16 for Shehata)
  - Only 2 scripts (vs 3 for Shehata)
  - Raw files are CSVs (vs Excel for Shehata)
  - No canonical benchmarks needed
  - No duplicate script cleanup needed

**Estimated complexity:** 60% of Shehata cleanup effort

---

## Rob C. Martin Principles Applied

✅ **Single Responsibility Principle** - Each directory serves ONE purpose
✅ **DRY (Don't Repeat Yourself)** - No duplicate files
✅ **Clean Code** - Clear naming, self-documenting structure
✅ **Traceability** - Complete provenance documentation
✅ **Reproducibility** - Scripts regenerate all derived files
✅ **Consistency** - Same 4-tier pattern as Shehata/Jain

---

## Recommendation

**PROCEED WITH CLEANUP** following Shehata pattern.

**Rationale:**
1. Harvey structure is inconsistent with cleaned Shehata/Jain
2. Cleanup is SIMPLER than Shehata (fewer files, no duplicates)
3. Low risk (good docs, no P0 blockers)
4. High benefit (consistent dataset organization)
5. Fast execution (45-60 minutes estimated)

**Proposed branch:** `leroy-jenkins/harvey-cleanup`

**Execution:** Same disciplined approach as Shehata:
1. Senior review this document ✅
2. Get approval for decisions
3. Create branch
4. Execute 7 phases
5. Verify with 8 checks
6. Merge to leroy-jenkins/full-send

---

## Questions for Senior Approval

**Q1:** Approve Decision 1 (Copy raw files to test_datasets/harvey/raw/)?

**Q2:** Approve Decision 2 (Delete harvey_high/low.csv intermediates)?

**Q3:** Approve Decision 3 (Empty canonical/ directory)?

**Q4:** Proceed with harvey-cleanup branch creation?

**Q5:** Any additional concerns or requirements before execution?

---

**Status:** ⏸️ **AWAITING SENIOR APPROVAL**

**Next step:** Get approval for all 5 questions, then execute cleanup.

---

**Date:** 2025-11-05 16:45
**Investigator:** Claude Code (Senior Review Mode)
**Reviewer:** [PENDING]
