# Scripts Directory Audit

**Purpose:** Comprehensive audit of all scripts to identify:
1. ✅ **VALID** - Production-ready scripts still needed
2. 🧪 **EXPERIMENTAL** - One-off experiments that may be deleted
3. 📦 **DATASET-SPECIFIC** - Should be moved to `preprocessing/{dataset}/`
4. 🗑️ **DEPRECATED** - Replaced by newer methods, safe to delete

**Date:** 2025-11-05
**Status:** Awaiting senior approval before reorganization

---

## Summary

**Total scripts audited:** 13 scripts (+ 3 README files)

**Breakdown:**
- ✅ VALID (keep in `scripts/`): 2 scripts
- 📦 DATASET-SPECIFIC (move to `preprocessing/`): 6 scripts
- 🧪 EXPERIMENTAL (delete): 4 scripts
- 🗑️ DEPRECATED (delete): 1 script

**Actions:**
- **DELETE:** 5 scripts (4 experimental + 1 deprecated)
- **MOVE:** 6 scripts (2 Boughter + 2 Jain + 1 Shehata + 1 Harvey)
- **KEEP:** 2 scripts (cross-dataset utilities only)

---

## scripts/analysis/ (5 scripts)

### ✅ `analyze_threshold_optimization.py` - KEEP

**Status:** VALID - Cross-dataset utility
**Purpose:** Finds optimal decision thresholds for different assay types (ELISA vs PSR)

**What it does:**
- Loads trained model
- Extracts prediction probabilities for Jain (ELISA) and Shehata (PSR)
- Finds optimal thresholds that match Novo's confusion matrices
- Demonstrates why single threshold can't work for both assays

**Why keep:**
- Cross-dataset analysis tool
- Documents ELISA vs PSR threshold differences
- Production utility for threshold calibration

**References:**
- Novo's Section 2.7: "PSR and ELISA measure different non-specificity spectrums"
- Used to discover PSR threshold = 0.549

---

### 🧪 `compare_jain_methodologies.py` - DELETE

**Status:** EXPERIMENTAL - One-off validation
**Purpose:** Compare OLD (94→91→86) vs P5e-S2 (137→116→86) Jain methodologies

**What it does:**
- Compares VH_only_jain_test_PARITY_86.csv (OLD reverse-engineered)
- vs VH_only_jain_86_p5e_s2.csv (P5e-S2 canonical)
- Shows antibody composition differences
- Shows label differences
- Compares model predictions and confusion matrices

**Why delete:**
- ✅ We now KNOW P5e-S2 is the correct methodology (achieves Novo parity)
- ✅ OLD dataset was from incorrect reverse engineering attempts
- ✅ Historical record preserved in docs/jain/
- ✅ No longer needed for production or validation

**Historical context:**
- Created during Nov 3-4, 2025 reverse engineering attempts
- Helped us validate that P5e-S2 achieves [[40, 19], [10, 17]]
- Purpose fulfilled - can be deleted

---

### 🧪 `rethreshold_harvey.py` - DELETE

**Status:** EXPERIMENTAL - One-off threshold discovery
**Purpose:** Find optimal PSR threshold for Harvey dataset

**What it does:**
- Loads trained model
- Re-runs predictions on Harvey (VHH_only_harvey.csv)
- Compares threshold 0.5 vs 0.5495
- Shows metrics changes and Novo benchmark comparison

**Why delete:**
- ✅ PSR threshold (0.549) already discovered and documented
- ✅ Now implemented in model.predict(assay_type='PSR')
- ✅ Experimental one-off to find the threshold
- ✅ Results documented in `analyze_threshold_optimization.py`

**Note:** Reads embeddings from CSV (old approach), doesn't generate fresh embeddings

---

### 🧪 `zscore_jain_116_outliers.py` - DELETE

**Status:** EXPERIMENTAL - Failed reverse engineering attempt
**Purpose:** Reverse-engineer which 30 antibodies Novo removed (116 → 86)

**What it does:**
- Z-score analysis of Jain 116-antibody ELISA-only set
- Flags sequence length outliers, charge outliers
- Flags chimeric/discontinued/withdrawn antibodies
- Attempts to identify QC removal candidates

**Why delete:**
- ✅ We now use P5e-S2 methodology (137 → 116 → 86)
- ✅ This was a failed attempt to reverse-engineer Novo's QC
- ✅ P5e-S2 doesn't use z-score outliers - uses PSR reclassification
- ✅ Creates jain_116_qc_candidates.csv which is NOT used
- ✅ References `test_datasets/jain_ELISA_ONLY_116.csv` (old path)

**Historical context:**
- Created Nov 3, 2025 during reverse engineering phase
- Problem identified: "Novo has 27 non-specific (we have 22)" - this was before P5e-S2
- Now obsolete with P5e-S2 methodology

---

### 📄 `README.md` - UPDATE

**Current content:** Only documents `analyze_threshold_optimization.py`

**Action:** Update after cleanup to reflect remaining scripts

---

## scripts/testing/ (3 scripts)

### ✅ `test_jain_novo_parity.py` - KEEP (but consider moving)

**Status:** VALID - Production test script
**Purpose:** Verify Novo parity on Jain 86-antibody test set

**What it does:**
- Loads boughter_vh_esm1v_logreg.pkl model
- Tests on jain_86_novo_parity.csv
- Verifies exact match: CM [[40, 19], [10, 17]], accuracy 66.28%
- Generates detailed classification report

**Why keep:**
- ✅ Production test for Novo parity verification
- ✅ Critical validation that we match published results
- ✅ Used in CI/CD pipeline verification

**Note:** Jain-specific - could move to `preprocessing/jain/` under dataset-centric organization

---

### ✅ `test_harvey_psr_threshold.py` - KEEP (but consider moving)

**Status:** VALID - Production test script
**Purpose:** Test Harvey dataset with PSR threshold (0.5495)

**What it does:**
- Loads model with assay_type='PSR'
- Tests on VHH_only_harvey.csv (141k nanobodies)
- Uses PSR-specific threshold
- Compares to Novo benchmark from Figure S14

**Why keep:**
- ✅ Production test for Harvey validation
- ✅ Long-running test (~20-30 min for 141k sequences)
- ✅ Validates PSR threshold implementation

**Note:** Harvey-specific - could move to `preprocessing/harvey/` under dataset-centric organization

---

### ✅ `demo_assay_specific_thresholds.py` - KEEP

**Status:** VALID - Educational/demo script
**Purpose:** Tutorial showing how to use assay-specific thresholds

**What it does:**
- Tests Jain with assay_type='ELISA'
- Tests Shehata with assay_type='PSR'
- Shows usage examples for developers
- Documents API: `model.predict(X, assay_type='ELISA'|'PSR')`

**Why keep:**
- ✅ Educational documentation for users
- ✅ Shows API usage patterns
- ✅ Cross-dataset demo (not specific to one dataset)
- ✅ Useful for onboarding and examples

---

### 📄 `README.md` - OK

**Current content:** Documents all 3 test scripts

**Action:** Update if we move dataset-specific tests to preprocessing/

---

## scripts/training/ (1 script)

### 📦 `train_hyperparameter_sweep.py` - MOVE to preprocessing/boughter/

**Status:** DATASET-SPECIFIC - Boughter only
**Purpose:** Hyperparameter tuning for Boughter LogisticRegression model

**What it does:**
- Loads Boughter training data
- Extracts ESM-1v embeddings once (cached)
- Tests different C, penalty, solver combinations
- 10-fold cross-validation (matching Novo methodology)
- Goal: Find params that match Novo's 71% CV accuracy

**Why move:**
- 📦 100% Boughter-specific (loads boughter_training.csv)
- 📦 Part of Boughter training pipeline
- 📦 Should live with other Boughter scripts

**Proposed location:** `preprocessing/boughter/train_hyperparameter_sweep.py`

---

## scripts/validation/ (5 scripts)

### 📦 `audit_boughter_training_qc.py` - MOVE to preprocessing/boughter/

**Status:** DATASET-SPECIFIC - Boughter only
**Purpose:** Comprehensive QC audit of Boughter training set (914 sequences)

**What it does:**
- Checks for stop codons (*), gaps (-), unknown AA (X)
- Checks for unusual sequence lengths
- Checks for repeated residues (homopolymers)
- Checks for suspicious CDR lengths
- Searches for potential QC issues explaining 3.5% gap with Novo

**Why move:**
- 📦 100% Boughter-specific
- 📦 Part of Boughter QC pipeline
- 📦 Already has validate_stage*.py scripts in preprocessing/boughter/

**Proposed location:** `preprocessing/boughter/audit_training_qc.py`

---

### ✅ `validate_fragments.py` - KEEP

**Status:** VALID - Cross-dataset utility
**Purpose:** Generic fragment extraction validation

**What it does:**
- Validates fragment CSV files for any dataset
- Checks expected fragment count (default 16)
- Verifies file structure and integrity
- Generic validation function

**Why keep:**
- ✅ Cross-dataset utility (works for Jain, Shehata, Harvey)
- ✅ Generic validation tool
- ✅ Not specific to one dataset

---

### 📦 `validate_jain_conversion.py` - MOVE to preprocessing/jain/

**Status:** DATASET-SPECIFIC - Jain only
**Purpose:** Validation harness for Jain Excel→CSV conversion

**What it does:**
- Re-runs conversion pipeline in-memory
- Compares against jain.csv
- Verifies flag counts, label distribution
- Validates amino acid sequences
- SHA256 checksum for provenance

**Why move:**
- 📦 100% Jain-specific
- 📦 Validates step1_convert_excel_to_csv.py
- 📦 Part of Jain preprocessing pipeline
- 📦 Already imports from preprocessing.jain.step1_convert_excel_to_csv

**Proposed location:** `preprocessing/jain/validate_conversion.py`

---

### 📦 `validate_shehata_conversion.py` - MOVE to preprocessing/shehata/

**Status:** DATASET-SPECIFIC - Shehata only
**Purpose:** Multi-method validation of Shehata Excel→CSV conversion

**What it does:**
- Validates using pandas (openpyxl)
- Validates using openpyxl direct
- Compares to generated CSV
- Ensures data integrity across methods

**Why move:**
- 📦 100% Shehata-specific
- 📦 Validates step1_convert_excel_to_csv.py
- 📦 Part of Shehata preprocessing pipeline

**Proposed location:** `preprocessing/shehata/validate_conversion.py`

---

### 🗑️ `verify_novo_parity.py` - DELETE

**Status:** DEPRECATED - Superseded by test_jain_novo_parity.py
**Purpose:** OLD version of Novo parity verification

**What it does:**
- IDENTICAL logic to scripts/testing/test_jain_novo_parity.py
- BUT uses OLD file path: `VH_only_jain_test_PARITY_86.csv` (deprecated)
- Uses OLD column name: `sequence` (should be `vh_sequence`)

**Why delete:**
- 🚨 **DUPLICATE** of scripts/testing/test_jain_novo_parity.py (newer version)
- 🚨 **DEPRECATED** - references OLD file and column names
- ✅ Newer version in scripts/testing/ uses correct paths
- ✅ Same functionality, just outdated references

**Confirmed via diff:** Only differences are file path and column name

---

### 📄 `README.md` - UPDATE

**Current content:** Documents validation scripts

**Action:** Update after moving dataset-specific scripts to preprocessing/

---

## Recommended Actions

### Phase 1: DELETE Experimental + Deprecated Scripts (4 scripts)

```bash
# Experimental scripts (3)
git rm scripts/analysis/compare_jain_methodologies.py
git rm scripts/analysis/rethreshold_harvey.py
git rm scripts/analysis/zscore_jain_116_outliers.py

# Deprecated script (1)
git rm scripts/validation/verify_novo_parity.py
```

**Rationale:**
- **Experimental (3):** One-off experiments from reverse engineering phase, results documented, no longer needed
- **Deprecated (1):** verify_novo_parity.py superseded by test_jain_novo_parity.py (uses old paths/columns)

---

### Phase 2: Move Dataset-Specific Scripts (6 scripts)

**Boughter (2 scripts):**
```
scripts/training/train_hyperparameter_sweep.py
  → preprocessing/boughter/train_hyperparameter_sweep.py

scripts/validation/audit_boughter_training_qc.py
  → preprocessing/boughter/audit_training_qc.py
```

**Jain (2 scripts):**
```
scripts/validation/validate_jain_conversion.py
  → preprocessing/jain/validate_conversion.py

scripts/testing/test_jain_novo_parity.py
  → preprocessing/jain/test_novo_parity.py
```

**Shehata (1 script):**
```
scripts/validation/validate_shehata_conversion.py
  → preprocessing/shehata/validate_conversion.py
```

**Harvey (1 script):**
```
scripts/testing/test_harvey_psr_threshold.py
  → preprocessing/harvey/test_psr_threshold.py
```

---

### Phase 3: Final scripts/ Structure (Minimal & Clean)

```
scripts/
├── analysis/
│   ├── analyze_threshold_optimization.py  (cross-dataset utility)
│   └── README.md
├── testing/
│   ├── demo_assay_specific_thresholds.py  (educational demo)
│   └── README.md
└── validation/
    ├── validate_fragments.py  (cross-dataset utility)
    └── README.md
```

**Only 3 scripts remain:**
- All are cross-dataset utilities
- All are production-ready
- All serve different purposes (analysis, demo, validation)

---

## Notes

### Confusion About "validation/"

**User concern:** ML people use "validation" for train/dev/validation splits

**Resolution:**
- Rename `scripts/validation/` → `scripts/checks/` or `scripts/verify/`?
- Or just delete it entirely since all validation scripts move to preprocessing/

**Recommendation:** Just delete `scripts/validation/` folder after moving scripts to preprocessing/

---

## Files That Reference Old Paths

**From analysis:**
- `zscore_jain_116_outliers.py` references `test_datasets/jain_ELISA_ONLY_116.csv` (old path)
  - But script is being deleted anyway ✅

---

## Senior Approval Questions

1. **Approve Phase 1: DELETE 4 scripts?**
   - 3 experimental (compare_jain_methodologies, rethreshold_harvey, zscore_jain_116_outliers)
   - 1 deprecated (verify_novo_parity.py - superseded by test_jain_novo_parity.py)

2. **Approve Phase 2: MOVE 6 dataset-specific scripts to preprocessing/?**
   - Follows dataset-centric organization (consistent with recent refactor)
   - Boughter (2), Jain (2), Shehata (1), Harvey (1)

3. **Approve Phase 3: Final structure with only 3 cross-dataset utilities?**
   - scripts/ becomes minimal: only cross-dataset tools
   - Dataset-specific work lives in preprocessing/{dataset}/

---

**Generated:** 2025-11-05
**Next step:** Senior review and approval before executing cleanup
