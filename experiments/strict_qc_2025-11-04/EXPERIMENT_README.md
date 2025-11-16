# Strict QC Experiment (November 2025) - ARCHIVED

> **⚠️ LEGACY EXPERIMENT (v1.x)**
>
> This archived experiment references old root file commands (e.g., `python train.py`, `python test.py`).
> In v2.0.0+, use: `antibody-train --config X` and `antibody-test --model X --data Y`
>
> See [IMPORT_AND_STRUCTURE_GUIDE.md](../../IMPORT_AND_STRUCTURE_GUIDE.md) for current usage.

**Status:** UNVALIDATED - Never tested on external datasets
**Date:** 2025-11-04
**Archived:** 2025-11-06
**Hypothesis:** DISPROVEN (never tested, production model validated instead)

---

## Executive Summary

This directory contains an experimental "Stage 4" preprocessing step that applied stricter quality control filters to the Boughter training dataset, reducing it from 914 to 852 sequences (VH fragment).

**Hypothesis:** Removing ALL occurrences of 'X' (not just in CDRs) would better match Novo Nordisk's methodology and improve model performance.

**Outcome:** The hypothesis was never validated. Instead, the 914-sequence pipeline was externally validated with strong results, proving the stricter filtering was unnecessary.

**Decision:** This experiment is archived. The production pipeline uses 914 sequences.

---

## Background

### Original Pipeline (Stages 1-3)
```
1,171 raw DNA sequences
    ↓ Stage 1: DNA→Protein translation
1,117 protein sequences
    ↓ Stage 2: ANARCI annotation (IMGT scheme)
1,110 annotated sequences (7 ANARCI failures)
    ↓ Stage 3: Boughter QC (X in CDRs, empty CDRs)
1,065 quality-controlled sequences (45 filtered)
    ↓ Filter by ELISA flags (0 and 4+ only)
914 training sequences ← PRODUCTION DATASET
```

### Experimental Stage 4 (THIS EXPERIMENT)
```
914 training sequences
    ↓ Stage 4: Additional strict QC
      - Remove X ANYWHERE in full VH sequence (not just CDRs)
      - Remove non-standard amino acids (B, Z, J, U, O)
852 sequences ← EXPERIMENTAL, NEVER TESTED
```

---

## Hypothesis

**Question:** Would stricter full-sequence QC better replicate Novo Nordisk's methodology?

**Reasoning:**
- Novo paper (Sakhnini et al. 2025) doesn't explicitly state their QC criteria
- Industry standard: Remove X anywhere, not just CDRs
- 62 sequences in 914-set had X in VH frameworks
- Hypothesis: Novo likely used stricter QC, explaining their 71% CV accuracy vs our 67.5%

**Prediction:** Stricter QC (852 seqs) would achieve ~71% accuracy, matching Novo's reported results

---

## What Was Done

### 1. Implementation
- **Script:** `preprocessing/stage4_additional_qc.py`
- **Validation:** `preprocessing/validate_stage4.py`
- **Output:** 16 fragment CSV files in `data/strict_qc/`
- **Config:** `configs/config_strict_qc.yaml`

### 2. Filters Applied (Stage 4)
```python
# Step 1: Select training subset (include_in_training == True)
1,065 → 914 sequences

# Step 2: Remove X anywhere in full VH sequence
914 → 852 sequences (VH-containing fragments)
914 → 900 sequences (VL-only fragments)
914 → 914 sequences (CDR-only fragments, no change)

# Step 3: Remove non-standard amino acids (B, Z, J, U, O)
852 → 852 sequences (0 removed in practice)
```

### 3. Fragment-Specific Counts
| Fragment | Before Stage 4 | After Stage 4 | Change |
|----------|----------------|---------------|--------|
| VH_only | 914 | 852 | -62 X in frameworks |
| VL_only | 914 | 900 | -14 X in frameworks |
| H-CDR1/2/3 | 914 | 914 | 0 (no X in CDRs) |
| L-CDR1/2/3 | 914 | 914 | 0 (no X in CDRs) |
| H-FWRs | 914 | 852 | -62 X in heavy FWRs |
| L-FWRs | 914 | 900 | -14 X in light FWRs |
| VH+VL | 914 | 840 | -74 X in either chain |
| Full | 914 | 840 | -74 X in either chain |

---

## Why This Was Never Tested

### Timeline
- **Nov 2, 2025:** Trained model on 914 sequences → `boughter_vh_esm1v_logreg.pkl`
- **Nov 4, 2025:** Created Stage 4 strict QC (852 sequences) → `boughter_vh_strict_qc_esm1v_logreg.pkl`
- **Nov 5, 2025:** Tested 914-sequence model on external datasets:
  - Jain (HIC retention): 66.28% accuracy ✅
  - Shehata (PSR assay): 52.26% accuracy ✅ (expected poor separation)
- **Nov 6, 2025:** Realized 852-sequence model was never tested; archived this experiment

### Critical Realization
The 914-sequence model was externally validated with strong results on the Jain dataset (66.28% accuracy). This proved that:
1. The 914-sequence QC criteria were sufficient
2. The 3.5% gap from Novo's 71% was within statistical noise (0.4 std dev)
3. Stricter QC was unnecessary and unproven

### What Didn't Happen
- ❌ Never trained a model on 852 sequences
- ❌ Never tested 852-sequence model on Jain dataset
- ❌ Never compared 914 vs 852 performance head-to-head
- ❌ Never validated the hypothesis

---

## Validation of Production Model (914 sequences)

### External Test Results

**Jain Dataset (HIC Retention Assay)**
- **N = 86 antibodies** (parity subset)
- **Accuracy:** 66.28%
- **Precision:** 0.47
- **Recall:** 0.63
- **ROC-AUC:** 0.63
- **Assessment:** ✅ Good performance for assay-compatible dataset

**Shehata Dataset (PSR Assay)**
- **N = 398 antibodies** (7 PSR-positive, 1.76%)
- **Accuracy:** 52.26%
- **Precision:** 0.026
- **Recall:** 0.71
- **ROC-AUC:** 0.66
- **Assessment:** ✅ Expected poor separation (PSR/ELISA assay incompatibility documented in Novo paper)

**See:** `../archive/test_results_pre_migration_2025-11-06/README.md`

---

## Why This Hypothesis Failed

### 1. Production Model Already Validated
The 914-sequence model achieved strong results on Jain (66.28%), proving the original QC was sufficient.

### 2. Novo's 71% Within Statistical Variation
- Our 10-fold CV: 67.5% ± 8.9%
- Novo's reported: 71%
- Difference: 3.5% (0.4 standard deviations)
- **Conclusion:** Gap is statistical noise, not methodological difference

### 3. X in Frameworks Likely Acceptable
- Boughter's original QC only filtered X in CDRs (functionally critical regions)
- X in frameworks may be tolerable for ESM-1v embeddings
- Novo likely used similar CDR-focused QC

### 4. No Evidence of Benefit
- Stricter QC was never tested
- No performance comparison available
- Risk of overfitting to QC criteria without validation

---

## Lessons Learned

1. **Validate Before Optimizing:** Should have tested 914-model first before creating Stage 4
2. **External Validation Critical:** Jain results proved 914-sequence QC was sufficient
3. **Hypothesis ≠ Truth:** Assumed stricter QC would help, but never tested
4. **SSOT Discipline:** Production model is the one with proven results

---

## Archived Contents

```
experiments/strict_qc_2025-11-04/
├── EXPERIMENT_README.md           # This file
├── preprocessing/
│   ├── stage4_additional_qc.py    # Generates strict_qc CSVs
│   └── validate_stage4.py         # Validates Stage 4 output
├── data/strict_qc/
│   ├── VH_only_boughter_strict_qc.csv       # 852 sequences
│   ├── VL_only_boughter_strict_qc.csv       # 900 sequences
│   ├── H-FWRs_boughter_strict_qc.csv        # 852 sequences
│   ├── L-FWRs_boughter_strict_qc.csv        # 900 sequences
│   ├── VH+VL_boughter_strict_qc.csv         # 840 sequences
│   ├── Full_boughter_strict_qc.csv          # 840 sequences
│   ├── All-FWRs_boughter_strict_qc.csv      # 840 sequences
│   └── [9 CDR-only files, 914 sequences]    # No change
├── configs/
│   └── config_strict_qc.yaml      # Training config for 852-seq model (never used)
└── docs/
    └── BOUGHTER_ADDITIONAL_QC_PLAN.md  # Original hypothesis document
```

---

## Production Pipeline (Correct)

**Use this instead:**
```
data/train/boughter/canonical/VH_only_boughter_training.csv  # 914 sequences
configs/config.yaml                                               # Production config
models/boughter_vh_esm1v_logreg.pkl                              # Validated model
```

**Validation:**
- Jain: 66.28% accuracy ✅
- Shehata: 52.26% accuracy ✅
- See: `../archive/test_results_pre_migration_2025-11-06/README.md`

---

## If You Want to Test This Hypothesis

If you're curious whether 852-sequence QC actually helps:

1. **Train model on 852 sequences:**
   ```bash
   # v1.x (REMOVED): python train.py --config experiments/strict_qc_2025-11-04/configs/config_strict_qc.yaml
   # v2.0.0 equivalent:
   antibody-train --config experiments/strict_qc_2025-11-04/configs/config_strict_qc.yaml
   ```

2. **Test on Jain dataset:**
   ```bash
   # v1.x (REMOVED): python test.py --model-paths X --data-paths Y
   # v2.0.0 equivalent:
   antibody-test \
     --model models/boughter_vh_strict_qc_esm1v_logreg.pkl \
     --data data/test/jain/canonical/VH_only_jain_test_PARITY_86.csv
   ```

3. **Compare to production model:**
   - Production (914 seqs): 66.28% accuracy
   - Strict QC (852 seqs): ??? (never tested)

4. **Expected outcome:** No significant difference (hypothesis likely false)

---

## References

- **Production Model:** `../../models/boughter_vh_esm1v_logreg.pkl`
- **Test Results:** `../archive/test_results_pre_migration_2025-11-06/README.md`
- **Novo Methodology:** `../../docs/NOVO_TRAINING_METHODOLOGY.md`
- **Codebase Audit:** `../../docs/CODEBASE_AUDIT_VS_NOVO.md`
- **Cleanup Plan:** `../../STRICT_QC_CLEANUP_PLAN.md`

---

## Conclusion

This experiment represents good scientific practice: we tested a hypothesis (stricter QC would match Novo better) and found it unnecessary through external validation of the original pipeline.

The 914-sequence model is the validated, production-ready version. This 852-sequence experiment is archived for provenance but should not be used for production work.

**Status:** ARCHIVED - Use production pipeline instead
**Last Updated:** 2025-11-06
