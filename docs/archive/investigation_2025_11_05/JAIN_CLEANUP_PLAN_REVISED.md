# Jain CSV Cleanup Plan - REVISED (Based on Actual Tree Structure)

**Date:** 2025-11-05
**Branch:** clean-jain
**Status:** Ready to execute

---

## Current Structure (72 files total)

```
REPO ROOT
├── test_datasets/
│   ├── jain.csv (137)
│   ├── jain_86_elisa_1.3.csv (86)
│   ├── jain_ELISA_ONLY_116.csv (116)
│   ├── jain_sd01.csv (137)
│   ├── jain_sd02.csv (137)
│   ├── jain_sd03.csv (139)
│   ├── jain_with_private_elisa_FULL.csv (137)
│   ├── jain-pnas.*.xlsx (3 Excel files - original PNAS supplemental)
│   │
│   └── jain/
│       ├── README.md
│       ├── Full_jain.csv (137)
│       ├── VH_only_jain.csv, VL_only_jain.csv, VH+VL_jain.csv (137 each)
│       ├── [12 CDR/FWR variant files] (137 each)
│       ├── VH_only_jain_test_FULL.csv (94)
│       ├── VH_only_jain_test_QC_REMOVED.csv (91)
│       ├── VH_only_jain_test_PARITY_86.csv (86) ✅ GOLD STANDARD
│       ├── VH_only_jain_86_p5e_s2.csv (86) ✅ ALSO CORRECT
│       ├── VH_only_jain_86_p5e_s4.csv (86)
│       ├── jain_86_novo_parity.csv (86)
│       │
│       ├── archive/
│       │   ├── jain_116_qc_candidates.csv (24)
│       │   └── jain_ELISA_ONLY_116_with_zscores.csv (116)
│       │
│       ├── legacy_reverse_engineered/ (DUPLICATES)
│       │   ├── VH_only_jain_test_FULL.csv (94)
│       │   ├── VH_only_jain_test_PARITY_86.csv (86)
│       │   └── VH_only_jain_test_QC_REMOVED.csv (91)
│       │
│       └── legacy_total_flags_methodology/ (WRONG METHOD - 13 files)
│           └── [13 CSV files using incorrect total_flags column]
│
├── experiments/novo_parity/
│   ├── [7 MD documentation files]
│   │
│   ├── datasets/
│   │   ├── jain_86_exp05.csv (86)
│   │   ├── jain_86_p5.csv (86)
│   │   ├── jain_86_p5d.csv, p5e, p5e_s2, p5e_s4, p5f, p5g, p5h (86 each)
│   │   └── [9 permutation files total] ✅ KEEP ALL
│   │
│   ├── results/
│   │   ├── permutations/ (17 JSON/CSV result files)
│   │   └── [4 other result files]
│   │
│   └── scripts/
│       └── [5 Python experiment scripts]
│
└── test_results/
    ├── [12 Jain-related test result folders]
    ├── [3 loose Jain CSV predictions]
    ├── [2 confusion matrix PNGs]
    ├── [3 YAML detailed results]
    ├── [3 test log files]
    ├── jain_benchmark_matrix_20251105.txt
    └── shehata_rerun/ (not Jain - keep)
```

---

## Key Insight: Where Are the "Correct" Files?

**Answer:** They're in BOTH places now (and they match)! ✅

### experiments/novo_parity/datasets/jain_86_p5e_s2.csv
- ✅ Result: [[40, 19], [10, 17]]
- Has full metadata (PSR, AC-SINS, biophysics, predictions, etc.)
- 36 columns of rich data
- Source of truth for P5e-S2 methodology

### test_datasets/jain/VH_only_jain_86_p5e_s2.csv
- ✅ Result: [[40, 19], [10, 17]]
- Minimal VH-only version (6 columns: id, sequence, label, smp, elisa, source)
- Created from experiments source
- For quick testing/benchmarking

**Both are correct!** The file in test_datasets was fixed to match experiments source.

---

## Cleanup Strategy

### ✅ KEEP - Production Files (test_datasets/)

**Keep in `test_datasets/` root (7 files):**
```
test_datasets/
├── jain.csv (137) - base dataset
├── jain_sd01.csv, jain_sd02.csv, jain_sd03.csv - biophysical data
├── jain_with_private_elisa_FULL.csv (137) - with private ELISA
├── jain_ELISA_ONLY_116.csv (116) - after ELISA filter
├── jain_86_elisa_1.3.csv (86) - ELISA threshold experiment
└── jain-pnas.*.xlsx (3) - original Excel supplemental data
```

**Keep in `test_datasets/jain/` (23 files):**
```
test_datasets/jain/
├── README.md ✅ already exists
│
├── SOURCE (137 antibodies - 1 file)
│   └── Full_jain.csv
│
├── FEATURE VARIANTS (137 antibodies - 15 files)
│   ├── VH_only_jain.csv, VL_only_jain.csv, VH+VL_jain.csv
│   ├── H-CDR1_jain.csv, H-CDR2_jain.csv, H-CDR3_jain.csv
│   ├── L-CDR1_jain.csv, L-CDR2_jain.csv, L-CDR3_jain.csv
│   ├── H-CDRs_jain.csv, L-CDRs_jain.csv, All-CDRs_jain.csv
│   └── H-FWRs_jain.csv, L-FWRs_jain.csv, All-FWRs_jain.csv
│
├── FILTERED DATASETS (3 files)
│   ├── VH_only_jain_test_FULL.csv (94)
│   ├── VH_only_jain_test_QC_REMOVED.csv (91)
│   └── VH_only_jain_test_PARITY_86.csv (86) ⭐ PRIMARY BENCHMARK
│
├── NOVO PARITY ALTERNATIVES (3 files)
│   ├── VH_only_jain_86_p5e_s2.csv (86) ⭐ ALSO ACHIEVES PARITY
│   ├── VH_only_jain_86_p5e_s4.csv (86)
│   └── jain_86_novo_parity.csv (86) - P5e-S2 with full metadata
│
└── archive/ (2 files - already archived)
    ├── jain_116_qc_candidates.csv (24)
    └── jain_ELISA_ONLY_116_with_zscores.csv (116)
```

**Total to keep: 33 files** (10 in root, 23 in jain/)

---

### 📦 MOVE TO ARCHIVE

**Create new archive structure:**
```bash
mkdir -p test_datasets/jain/archive/legacy_reverse_engineered
mkdir -p test_datasets/jain/archive/legacy_total_flags_methodology
```

**Move legacy files:**

1. **legacy_reverse_engineered/ → archive/legacy_reverse_engineered/** (3 files)
   - These are DUPLICATES of files in main directory
   - Keep for historical record but clearly deprecated

2. **legacy_total_flags_methodology/ → archive/legacy_total_flags_methodology/** (13 files)
   - These use WRONG ELISA methodology (total_flags vs elisa_flags column)
   - Keep for historical record but clearly deprecated
   - Label as "INCORRECT - Do not use"

**Total in archive after: 18 files**

---

### 🔬 KEEP - Experiments Folder (AS-IS)

**Location:** `experiments/novo_parity/`

```
experiments/novo_parity/
├── [7 MD files] ✅ Keep all documentation
├── datasets/ [9 CSV files] ✅ Keep all permutations
├── results/ [21 JSON/TXT/CSV/MD files] ✅ Keep all results
└── scripts/ [5 Python files] ✅ Keep all scripts
```

**Total: 42 files**

**Why keep everything:**
- Complete experimental provenance
- Shows methodology exploration
- Reproducible research
- Well-documented with MD files
- **This is RESEARCH OUTPUT** - don't delete!

---

### 🗑️ DELETE - Test Results

**Delete ALL Jain test results in `test_results/`:**

```bash
# Delete Jain-related test result folders (12 folders)
rm -rf test_results/jain
rm -rf test_results/jain_94_full
rm -rf test_results/jain_fixed_94ab
rm -rf test_results/jain_novo_parity_86_20251103_182349
rm -rf test_results/jain_old_verification
rm -rf test_results/jain_old_verification_strict
rm -rf test_results/jain_p5e_s2_verification
rm -rf test_results/jain_p5e_s2_verification_strict
rm -rf test_results/jain_p5e_s4_verification
rm -rf test_results/jain_qc3_91ab
rm -rf test_results/matrix_test_new_old
rm -rf test_results/matrix_test_new_p5e

# Delete loose Jain files (9 files)
rm -f test_results/predictions_*jain*.csv
rm -f test_results/confusion_matrix_*jain*.png
rm -f test_results/detailed_results_*jain*.yaml
rm -f test_results/jain_94_full_inference.log
rm -f test_results/jain_novo_parity_86_*_inference.log
rm -f test_results/jain_benchmark_matrix_*.txt
rm -f test_results/test_*.log (if Jain-related only)

# KEEP shehata_rerun/ (not Jain)
```

**Total deleted: ~60 files/folders**

**Why delete:**
- Ephemeral outputs (can regenerate anytime)
- Take up space and add clutter
- Not source data
- Models are what matter, not old predictions

---

## Execution Plan

### Step 1: Move Legacy Files to Archive

```bash
# Move reverse-engineered duplicates
mv test_datasets/jain/legacy_reverse_engineered test_datasets/jain/archive/

# Move incorrect total_flags files
mv test_datasets/jain/legacy_total_flags_methodology test_datasets/jain/archive/
```

### Step 2: Delete Test Results

```bash
# Delete ALL Jain test result folders
rm -rf test_results/jain*
rm -rf test_results/matrix_test_*

# Delete loose Jain prediction files
rm -f test_results/predictions_*jain*.csv
rm -f test_results/confusion_matrix_*jain*.png
rm -f test_results/detailed_results_*jain*.yaml

# Delete Jain-specific logs
rm -f test_results/jain*.log
rm -f test_results/jain*.txt
```

### Step 3: Update Archive README

Create `test_datasets/jain/archive/README.md`:

```markdown
# Jain Dataset Archive

## ⚠️ DEPRECATED FILES - DO NOT USE

This archive contains historical versions and incorrect methodologies.
Use files in the parent directory (`test_datasets/jain/`) instead.

### legacy_reverse_engineered/
- **Status:** DUPLICATES of files in parent directory
- **Reason archived:** Redundant
- **Created:** Nov 2, 2025
- **Files:** VH_only_jain_test_FULL.csv, VH_only_jain_test_PARITY_86.csv, VH_only_jain_test_QC_REMOVED.csv

### legacy_total_flags_methodology/
- **Status:** ❌ INCORRECT METHODOLOGY
- **Reason archived:** Uses wrong ELISA column (total_flags instead of elisa_flags)
- **Created:** Nov 2-3, 2025
- **Files:** 13 CSV files (94 and 86 antibody variants)
- **Do NOT use these files!**

### Other Archive Files:
- `jain_116_qc_candidates.csv` - QC removal candidates (research)
- `jain_ELISA_ONLY_116_with_zscores.csv` - Intermediate analysis with z-scores
```

### Step 4: Update Main README

Update `test_datasets/jain/README.md` to document current structure.

### Step 5: Verify Cleanup

```bash
# Count files in each location
echo "Main directory:"
ls test_datasets/jain/*.csv 2>/dev/null | wc -l  # Should be ~22

echo "Archive:"
find test_datasets/jain/archive -name "*.csv" | wc -l  # Should be ~18

echo "Experiments:"
ls experiments/novo_parity/datasets/*.csv | wc -l  # Should be 9

echo "Test results (Jain):"
ls test_results/*jain* 2>/dev/null | wc -l  # Should be 0

echo "Test results (total):"
ls test_results/ | wc -l  # Should be minimal
```

---

## Final Structure After Cleanup

```
REPO ROOT
├── test_datasets/ (10 files in root + jain/ folder)
│   ├── jain*.csv (7 CSV files)
│   ├── jain-pnas*.xlsx (3 Excel files)
│   │
│   └── jain/ (23 files)
│       ├── README.md (UPDATED)
│       ├── [1 source file: Full_jain.csv]
│       ├── [15 feature variant files]
│       ├── [3 filtered datasets: 94, 91, 86]
│       ├── [3 parity alternatives: p5e_s2, p5e_s4, novo_parity]
│       │
│       └── archive/ (18 files total)
│           ├── README.md (NEW)
│           ├── jain_116_qc_candidates.csv
│           ├── jain_ELISA_ONLY_116_with_zscores.csv
│           ├── legacy_reverse_engineered/ (3 files)
│           └── legacy_total_flags_methodology/ (13 files)
│
├── experiments/novo_parity/ (42 files - unchanged)
│   ├── [7 MD documentation files]
│   ├── datasets/ (9 CSV permutation files)
│   ├── results/ (21 result files)
│   └── scripts/ (5 Python files)
│
└── test_results/ (CLEANED - minimal files)
    ├── shehata_rerun/ (kept - not Jain)
    └── [other non-Jain test results if any]
```

**Total Jain files after cleanup: ~51 files**
- Production: 33 files (test_datasets)
- Archive: 18 files (test_datasets/jain/archive)
- Experiments: 42 files (experiments/novo_parity) [includes 9 CSVs + docs/results/scripts]
- Test results: 0 files (deleted)

**Reduction: 72 → 51 files (29% reduction)**

---

## Verification Tests After Cleanup

Run these to ensure both parity methods still work:

```bash
# Test 1: OLD reverse-engineered method
python test.py \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/jain/VH_only_jain_test_PARITY_86.csv
# Expected: [[40, 19], [10, 17]] ✅

# Test 2: P5e-S2 canonical method
python test.py \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/jain/VH_only_jain_86_p5e_s2.csv
# Expected: [[40, 19], [10, 17]] ✅

# Both should pass!
```

---

## Summary

### Why This Makes Sense Now:

1. **experiments/ is the RESEARCH WORKSPACE**
   - Contains full experimental provenance
   - 9 permutation datasets with rich metadata
   - Complete documentation and results
   - **Keep everything** - it's valuable research

2. **test_datasets/jain/ is the PRODUCTION LIBRARY**
   - Canonical datasets for benchmarking
   - Clean, minimal files
   - Both parity methods available
   - Legacy/incorrect files archived

3. **test_results/ was CLUTTERED**
   - 60+ ephemeral test outputs
   - Can regenerate anytime
   - **Delete everything Jain-related**

### Why You Were Confused:

1. Files existed in BOTH experiments and test_datasets
2. P5e-S2 in test_datasets was CORRUPTED until 10:49 AM today
3. 72 total files with unclear purpose
4. Legacy files not clearly marked
5. Test results mixed with source data

### After Cleanup:

- ✅ Clear separation: research vs production
- ✅ Both parity methods work (verified)
- ✅ Legacy files archived (not deleted)
- ✅ Test clutter removed
- ✅ Documented and systematic

---

## Ready to Execute?

Reply with "execute" and I'll run all cleanup commands!

