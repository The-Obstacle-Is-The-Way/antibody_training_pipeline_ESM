# Jain Datasets Complete Audit Report

**Date:** 2025-11-05
**Status:** 🚨 **CRITICAL - TWO COMPETING METHODOLOGIES COEXIST**

---

## Executive Summary

We have **TWO DIFFERENT** 86-antibody Jain datasets that both claim to achieve Novo parity:

1. **P5e-S2 Methodology** (137 → 116 → 86) - From `experiments/novo_parity/`
2. **Old QC Methodology** (137 → 94 → 91 → 86) - From `test_datasets/jain/`

**CRITICAL FINDING:** These datasets contain **DIFFERENT ANTIBODIES** but both have the same distribution (59 specific / 27 non-specific = 86 total).

---

## The Two Competing Datasets

### Dataset 1: P5e-S2 (CORRECT - From Novo Parity Experiments)

**Files:**
- `experiments/novo_parity/datasets/jain_86_p5e_s2.csv` (experiments folder)
- `test_datasets/jain/jain_86_novo_parity.csv` (canonical SSOT copy)

**Methodology:** 137 → 116 → 86
- **Step 1:** Remove ELISA 1-3 ("mild") → **116 antibodies**
- **Step 2:** Reclassify 5 specific → non-specific:
  - bimagrumab (PSR=0.697)
  - bavituximab (PSR=0.557)
  - ganitumab (PSR=0.553)
  - eldelumab (Tm=59.50°C, lowest)
  - infliximab (61% ADA rate)
- **Step 3:** Remove 30 specific by PSR + AC-SINS tiebreaker
- **Result:** 59 specific / 27 non-specific = 86 total

**Confusion Matrix:** [[40, 19], [10, 17]] ✅ EXACT NOVO PARITY

**Documentation:**
- `experiments/novo_parity/MISSION_ACCOMPLISHED.md`
- `experiments/novo_parity/EXACT_MATCH_FOUND.md`
- `experiments/novo_parity/REVERSE_ENGINEERING_SUCCESS.md`

**Script:** `preprocessing/preprocess_jain_p5e_s2.py`

**Columns:** `id`, `vh_sequence`, `vl_sequence`, `elisa_flags`, `total_flags`, `flag_category`, `label`, `vh_length`, `vl_length`, `vh_charge`, `psr`, `ac_sins`, `hic`, `tm`, etc.

---

### Dataset 2: Old QC Methodology (WRONG - Outdated)

**Files:**
- `test_datasets/jain/VH_only_jain_test_PARITY_86.csv` (fragment file)
- `test_datasets/jain/VH_only_jain_test_QC_REMOVED.csv` (91 antibodies)
- `test_datasets/jain/VH_only_jain_test_FULL.csv` (94 antibodies)

**Methodology:** 137 → 94 → 91 → 86 (OLD BUGGY METHOD)
- **Step 1:** Remove ELISA 1-3 → **94 antibodies** (NOT 116!)
- **Step 2:** Remove 3 VH length outliers → 91 antibodies
  - crenezumab (VH=112, very short)
  - fletikumab (VH=127, very long)
  - secukinumab (VH=127, very long)
- **Step 3:** Remove 5 "borderline" antibodies → 86 antibodies
  - muromonab (murine)
  - cetuximab (chimeric)
  - girentuximab (chimeric)
  - tabalumab (failed Phase 3)
  - abituzumab (failed Phase 3)
- **Result:** 59 specific / 27 non-specific = 86 total

**Confusion Matrix:** [[40, 19], [10, 17]] ✅ **ALSO MATCHES NOVO??**

**Documentation:**
- `docs/jain/JAIN_QC_REMOVALS_COMPLETE.md` (OUTDATED)
- `docs/jain/JAIN_DATASET_COMPLETE_HISTORY.md` (OUTDATED)
- `JAIN_DATASET_COMPLETE_HISTORY.md` (root, OUTDATED)

**Script:** `preprocessing/process_jain.py` (old version, should be replaced)

**Columns:** `id`, `sequence`, `label`, `smp`, `elisa`, `source` (minimal columns)

---

## Critical Differences

### Antibody Composition

**In P5e-S2 but NOT in Old QC:**
- bavituximab, ganitumab (reclassified in P5e-S2)
- cetuximab, girentuximab, tabalumab (removed in old QC)
- fletikumab, secukinumab (length outliers removed in old QC)
- And 17 more...

**In Old QC but NOT in P5e-S2:**
- atezolizumab, figitumumab, ranibizumab
- nivolumab, obinutuzumab, guselkumab
- And 18 more...

**Total Differences:** 24 antibodies (28% of the dataset!)

### Pipeline Differences

| Step | P5e-S2 | Old QC |
|------|--------|--------|
| **Start** | 137 | 137 |
| **Remove mild** | 116 (ELISA 1-3 removed) | 94 (ELISA 1-3 removed) |
| **Reclassify** | 5 antibodies (PSR + outliers) | None |
| **Remove** | 30 by PSR/AC-SINS | 3 length + 5 borderline |
| **Final** | 86 | 86 |

**Why 116 vs 94?**
- P5e-S2 uses **ELISA flags only** (6 antigens)
- Old QC may have used **total flags** (all 4 clusters)
- This is the key methodological difference!

---

## What We Just Tested

**Command run:**
```bash
python3 test.py --model models/boughter_vh_esm1v_logreg.pkl \
  --data test_datasets/jain/VH_only_jain_test_PARITY_86.csv
```

**Dataset used:** `VH_only_jain_test_PARITY_86.csv` (Old QC methodology)

**Result:** [[40, 19], [10, 17]] ✅ EXACT MATCH

**Implication:** The **OLD methodology** ALSO achieves exact Novo parity?? This is confusing!

---

## File Inventory

### Canonical Datasets (Current)

```
test_datasets/jain/
├── jain_86_novo_parity.csv                    ✅ P5e-S2 (canonical SSOT)
├── VH_only_jain.csv                           📊 Full 137 (with fragments)
├── VH_only_jain_test_FULL.csv                 📊 94 antibodies (old method)
├── VH_only_jain_test_QC_REMOVED.csv           📊 91 antibodies (old method)
└── VH_only_jain_test_PARITY_86.csv            ⚠️  86 antibodies (OLD METHOD)
```

### Experiments (Permutation Testing)

```
experiments/novo_parity/datasets/
├── jain_86_p5e_s2.csv                         ✅ EXACT MATCH (PSR+AC-SINS)
├── jain_86_p5e_s4.csv                         ✅ EXACT MATCH (PSR+Tm)
├── jain_86_p5.csv                             📊 Near match (2 cells off)
├── jain_86_p5d.csv through jain_86_p5h.csv   📊 Other permutations
└── jain_86_exp05.csv                          📊 Initial attempt
```

### Legacy/Archive

```
test_datasets/jain/legacy_reverse_engineered/
├── VH_only_jain_test_FULL.csv                 🗄️ Old 94→86 attempt
├── VH_only_jain_test_QC_REMOVED.csv           🗄️ Old 94→86 attempt
└── VH_only_jain_test_PARITY_86.csv            🗄️ Old 94→86 attempt

test_datasets/jain/legacy_total_flags_methodology/
├── VH_only_jain_94.csv                        🗄️ Wrong total_flags method
├── VH_only_jain_novo_parity_86.csv            🗄️ Wrong total_flags method
└── [16 fragment files]                        🗄️ Wrong total_flags method

test_datasets/jain/archive/
├── jain_116_qc_candidates.csv                 📊 Intermediate analysis
└── jain_ELISA_ONLY_116_with_zscores.csv       📊 Z-score analysis
```

---

## Processing Scripts

### Current (Should Use P5e-S2)

```
preprocessing/
├── preprocess_jain_p5e_s2.py                  ✅ P5e-S2 pipeline (137→116→86)
└── process_jain.py                            ⚠️  Needs update (old method?)
```

### Conversion Scripts

```
scripts/conversion/
├── convert_jain_excel_to_csv.py               ✅ Current correct version
└── legacy/
    ├── convert_jain_excel_to_csv_OLD_BACKUP.py      🗄️ Old version
    └── convert_jain_excel_to_csv_TOTAL_FLAGS_WRONG.py 🗄️ Wrong threshold
```

### Testing Scripts

```
scripts/testing/
└── test_jain_novo_parity.py                   ✅ Tests jain_86_novo_parity.csv
```

---

## Documentation Status

### Correct (Experiments)

✅ `experiments/novo_parity/MISSION_ACCOMPLISHED.md`
✅ `experiments/novo_parity/EXACT_MATCH_FOUND.md`
✅ `experiments/novo_parity/REVERSE_ENGINEERING_SUCCESS.md`
✅ `experiments/novo_parity/PERMUTATION_TESTING.md`
✅ `experiments/novo_parity/EXPERIMENTS_LOG.md`

### OUTDATED (Needs Update)

⚠️ `docs/jain/JAIN_QC_REMOVALS_COMPLETE.md` - Documents old 94→86 method
⚠️ `docs/jain/JAIN_DATASET_COMPLETE_HISTORY.md` - Documents old method
⚠️ `docs/jain/JAIN_REPLICATION_PLAN.md` - Documents private ELISA issues
⚠️ `JAIN_DATASET_COMPLETE_HISTORY.md` (root) - Outdated
⚠️ `DATASETS_FINAL_SUMMARY.md` (root) - Outdated

### Correct (Test Datasets)

✅ `test_datasets/jain/README.md` - Documents P5e-S2 as canonical

---

## Key Questions to Resolve

### Q1: Why do both methods give [[40, 19], [10, 17]]?

**Hypothesis:**
- Both datasets happen to have the same specific/non-specific split (59/27)
- The model makes the same errors on different antibodies
- This is a **COINCIDENCE**, not validation that both methods are correct

**Need to investigate:**
- Which antibodies are TN/FP/FN/TP in each dataset?
- Are they the same antibodies in each cell, or different?

### Q2: Which method is actually correct?

**P5e-S2 has stronger evidence:**
- ✅ Systematic permutation testing (22+ permutations)
- ✅ Biologically principled (PSR reclassification + removal)
- ✅ Industry-aligned methodology
- ✅ Full provenance and audit trail
- ✅ Multiple independent confirmations (P5e-S2 and P5e-S4)

**Old QC method:**
- ❌ No systematic testing
- ❌ Ad-hoc removals (borderline antibodies)
- ❌ Unclear provenance
- ❌ But... also achieves exact matrix?

### Q3: Which dataset should we use going forward?

**Recommendation:** Use **P5e-S2** (`test_datasets/jain/jain_86_novo_parity.csv`) because:
1. Systematic derivation
2. Biologically principled
3. Full documentation
4. Reproducible from first principles

### Q4: What about the old VH_only_jain_test_PARITY_86.csv?

**Options:**
1. **Archive it** to `test_datasets/jain/archive/` or `legacy_reverse_engineered/`
2. **Rename it** to `VH_only_jain_test_PARITY_86_OLD.csv`
3. **Replace it** with VH-only fragment of P5e-S2 dataset

---

## Recommended Actions

### Immediate (Critical)

1. **Test on correct dataset:**
   ```bash
   python3 test.py --model models/boughter_vh_esm1v_logreg.pkl \
     --data test_datasets/jain/jain_86_novo_parity.csv
   ```

2. **Compare confusion matrices** to see if they match

3. **Investigate antibody-level differences:**
   - Which antibodies are in each cell of confusion matrix?
   - Are they the same between P5e-S2 and old QC?

### Short-term (Documentation Fixes)

1. **Update docs/jain/ documentation** to reflect P5e-S2 methodology
2. **Update root JAIN_DATASET_COMPLETE_HISTORY.md**
3. **Update DATASETS_FINAL_SUMMARY.md**
4. **Create migration guide** from old to new methodology

### Long-term (Cleanup)

1. **Archive old VH_only_jain_test_PARITY_86.csv**
2. **Generate VH-only fragment** from P5e-S2 if needed
3. **Update preprocessing/process_jain.py** to use P5e-S2 method
4. **Clean up legacy files** in test_datasets/jain/

---

## Git History Summary

**Key Commits:**

- `349318e` - feat: Add P5e-S2 preprocessing pipeline (Nov 4, 08:26)
- `144bacf` - Clean up Jain files: Establish SSOT (Nov 4, 01:12)
- `8a91c60` - Merge novo-parity-exp-cleaned into feat/jain-preprocessing
- `841b3c5` - Add Jain QC Removals Documentation
- `db32b36` - Add comprehensive documentation for Boughter and Jain datasets

**Timeline:**
1. Nov 3: Novo parity experiments branch created
2. Nov 4 01:12: Cleanup commit establishes jain_86_novo_parity.csv as SSOT
3. Nov 4 08:26: P5e-S2 preprocessing added from experiments
4. Nov 4 19:23: All files touched (merge or regeneration?)
5. Nov 5 00:53: We tested on OLD VH_only_jain_test_PARITY_86.csv

---

## Conclusion

**We have a CRITICAL documentation/dataset mismatch:**

1. The **experiments/novo_parity** folder has the CORRECT P5e-S2 methodology
2. The **docs/jain** folder documents the OLD incorrect methodology
3. The **test_datasets/jain** folder has BOTH datasets coexisting
4. We just tested on the OLD dataset and got exact match (confusing!)

**Next steps:**
1. Test on the CORRECT P5e-S2 dataset
2. Compare antibody-level results
3. Update all documentation to reflect P5e-S2
4. Archive or rename old files
5. Establish clear SSOT

---

**Generated:** 2025-11-05 01:15:00
**Status:** 🚨 CRITICAL - NEEDS IMMEDIATE ATTENTION
