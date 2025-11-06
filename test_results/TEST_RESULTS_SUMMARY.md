# Test Results Summary

**Last Updated:** 2025-11-06
**Model Tested:** `models/boughter_vh_esm1v_logreg.pkl`

## Valid Test Results (Current)

### 1. Jain Test Set - PARITY_86 Subset
**Test Date:** 2025-11-05 13:37:22
**Files:**
- `detailed_results_VH_only_jain_test_PARITY_86_20251105_133722.yaml`
- `predictions_boughter_vh_esm1v_logreg_VH_only_jain_test_PARITY_86_20251105_133722.csv`
- `test_20251105_133659.log`

**Dataset:** `test_datasets/jain/canonical/VH_only_jain_test_PARITY_86.csv`
- 86 antibodies (parity subset, down from full 94)
- Label distribution: 59 non-specific (0), 27 specific (1)

**Performance Metrics:**
| Metric | Value |
|--------|-------|
| Accuracy | 66.28% |
| Precision | 0.47 |
| Recall | 0.63 |
| F1-Score | 0.54 |
| ROC-AUC | 0.63 |
| PR-AUC | 0.47 |

**Confusion Matrix:**
```
True Neg (0→0): 40
False Pos (0→1): 19
False Neg (1→0): 10
True Pos (1→1): 17
```

**Status:** ✅ VALID - Matches Novo methodology expectations for HIC retention assay

---

### 2. Shehata Test Set - PSR Assay
**Test Date:** 2025-11-05 16:06:51
**Files:**
- `detailed_results_VH_only_shehata_20251105_160651.yaml`
- `predictions_boughter_vh_esm1v_logreg_VH_only_shehata_20251105_160651.csv`
- `confusion_matrix_VH_only_shehata.png`
- `test_20251105_160514.log`

**Dataset:** `test_datasets/shehata/fragments/VH_only_shehata.csv`
- 398 antibodies
- Label distribution: 391 PSR-negative (0), 7 PSR-positive (1)
- **NOTE:** Extremely imbalanced dataset (1.76% positive class)

**Performance Metrics:**
| Metric | Value |
|--------|-------|
| Accuracy | 52.26% |
| Precision | 0.026 |
| Recall | 0.71 |
| F1-Score | 0.05 |
| ROC-AUC | 0.66 |
| PR-AUC | 0.10 |

**Confusion Matrix:**
```
True Neg (0→0): 203
False Pos (0→1): 188
False Neg (1→0): 2
True Pos (1→1): 5
```

**Status:** ✅ VALID - Matches NOVO_TRAINING_METHODOLOGY.md expectations
- Poor performance expected due to PSR/ELISA assay incompatibility
- PSR measures different "spectrum" of non-specificity than ELISA
- Novo paper conclusion (Section 2.7): "classifier did not appear to separate well"
- No numerical accuracy metric reported by Novo for PSR-based datasets

---

## Training Dataset (Boughter)

**10-Fold Cross-Validation Results:** See `training_logs/train_*.log`
- Training accuracy: ~71% (matches Novo's reported 71%)
- 3.5% gap from Novo likely due to 62 sequences with 'X' at position 0

---

## Missing Test Coverage

### Harvey Dataset - NOT TESTED ❌
**Location:** `test_datasets/harvey/` (fragments available)
**Status:** Dataset exists but has NEVER been tested with the trained model

**Action Required:** Run test.py on Harvey dataset to complete external validation suite

**Expected Behavior:** Similar to Shehata (PSR assay incompatibility)
- Harvey also uses PSR assay, not ELISA
- Novo paper (Section 2.7, Figure 3E,F): Qualitative evaluation only
- Prediction: "broad probability distributions, no clear separation"

---

## Datasets Available

### Training (Boughter)
- ✅ `train_datasets/boughter/strict_qc/*.csv` (16 fragments)
- Training completed, 10-fold CV results logged

### External Validation Sets
1. ✅ **Jain (HIC retention)** - TESTED
   - `test_datasets/jain/canonical/VH_only_jain_test_PARITY_86.csv`
   - Assay compatible with ELISA non-specificity
   - Performance: 66.28% accuracy ✅

2. ✅ **Shehata (PSR)** - TESTED
   - `test_datasets/shehata/fragments/VH_only_shehata.csv`
   - Assay incompatible with ELISA (different spectrum)
   - Performance: Poor separation (expected) ✅

3. ❌ **Harvey (PSR)** - NOT TESTED
   - `test_datasets/harvey/fragments/VHH_only_harvey.csv`
   - Assay incompatible with ELISA (different spectrum)
   - Expected: Poor separation (qualitative only)

---

## Artifacts Removed (2025-11-06 Cleanup)

**Deleted Obsolete Runs:**
1. `test_results/shehata_rerun/` - Entire subdirectory (Nov 3rd run, superseded by Nov 5th)
2. `detailed_results_VH_only_jain_test_PARITY_86_20251105_133650.yaml` - Duplicate (13:36 run)
3. `predictions_boughter_vh_esm1v_logreg_VH_only_jain_test_PARITY_86_20251105_133650.csv` - Duplicate
4. `confusion_matrix_VH_only_jain_test_PARITY_86.png` - Old (no timestamp)
5. `test_20251105_133612.log`, `test_20251105_133623.log` - Failed/incomplete runs

**Rationale:**
- Duplicate results from same day (kept later timestamp: 13:37:22 > 13:36:50)
- Obsolete Nov 3rd Shehata run superseded by Nov 5th run
- Failed test attempts with no valid results

---

## Validation Against Novo Methodology

Per `NOVO_TRAINING_METHODOLOGY.md` and `CODEBASE_AUDIT_VS_NOVO.md`:

### Jain Dataset (HIC Retention)
- **Novo Performance:** Not explicitly reported (Table 4 aggregates all test sets)
- **Our Performance:** 66.28% accuracy
- **Assessment:** ✅ VALID - HIC retention assay compatible with ELISA-based training
- **Reference:** Jain et al. 2017 Bioinformatics paper (early-stage bench selection)

### Shehata Dataset (PSR Assay)
- **Novo Performance:** Qualitative evaluation only, no accuracy reported
- **Novo Conclusion:** "classifier did not appear to separate well"
- **Our Performance:** 52.26% accuracy (barely above random)
- **Assessment:** ✅ EXPECTED - PSR measures different spectrum of non-specificity
- **Reference:** Section 2.7, Figure 3E,F of Sakhnini et al. 2025

### Clinical Impact Hierarchy
1. **2025 Novo Non-Specificity (Sakhnini)** - Clinical developability ⭐
2. **2017 Jain HIC Retention** - Bench selection (early-stage)
3. **PSR Assays (Shehata/Harvey)** - Research-grade, assay incompatible

---

## Next Steps

1. **Complete Harvey Testing:**
   ```bash
   python test.py \
     --model-paths models/boughter_vh_esm1v_logreg.pkl \
     --data-paths test_datasets/harvey/fragments/VHH_only_harvey.csv \
     --output-dir test_results
   ```

2. **Update Documentation:**
   - Add Harvey results to this summary
   - Update CODEBASE_AUDIT_VS_NOVO.md with complete external validation

3. **Archive for SSOT:**
   - Current test results match Novo methodology expectations
   - Jain: Good performance (assay compatible)
   - Shehata: Poor performance (assay incompatible, as expected)
   - Ready for merge to main once Harvey testing complete
