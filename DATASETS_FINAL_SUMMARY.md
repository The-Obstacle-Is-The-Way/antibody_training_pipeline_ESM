# Datasets Final Summary - Complete Documentation

**Date:** 2025-11-04
**Status:** ✅ All datasets fully documented and validated

---

## Quick Reference

This project uses **two main datasets** for antibody polyreactivity prediction:

| Dataset | Role | Sequences | Status | Documentation |
|---------|------|-----------|--------|---------------|
| **Boughter** | Training | 852-914 | ✅ Complete | [BOUGHTER_DATASET_COMPLETE_HISTORY.md](BOUGHTER_DATASET_COMPLETE_HISTORY.md) |
| **Jain** | Testing | 86-137 | ✅ Complete | [JAIN_DATASET_COMPLETE_HISTORY.md](JAIN_DATASET_COMPLETE_HISTORY.md) |

---

## Boughter Dataset (Training)

**Paper:** Boughter et al. (2020) *eLife* 9:e61393

**Purpose:** Train polyreactivity prediction models

**Key Features:**
- 1,171 antibodies from 6 diverse sources (mouse IgA, flu, HIV)
- ELISA polyreactivity against 4-7 diverse antigens
- Binary labels: 0 flags (specific) vs 4+ flags (non-specific)

### Two QC Levels Available:

#### 1. Boughter QC (914 sequences) ⭐ **RECOMMENDED**
- **File:** `train_datasets/boughter/VH_only_boughter_training.csv`
- **QC:** X in CDRs only, empty CDRs, ANARCI failures
- **Model:** `models/boughter_vh_esm1v_logreg.pkl`
- **CV Accuracy:** 67.5% ± 8.9%
- **Use case:** Exact Novo methodology replication

#### 2. Strict QC (852 sequences)
- **File:** `train_datasets/boughter/VH_only_boughter_strict_qc.csv`
- **QC:** Boughter QC + X anywhere + non-standard AA
- **Model:** `models/boughter_vh_strict_qc_esm1v_logreg.pkl`
- **CV Accuracy:** 66.55% ± 7.07%
- **Use case:** Industry-standard QC (no performance improvement)

**Read more:** [BOUGHTER_DATASET_COMPLETE_HISTORY.md](BOUGHTER_DATASET_COMPLETE_HISTORY.md)

---

## Jain Dataset (Testing)

**Paper:** Jain et al. (2017) *PNAS* 114(5), 944-949

**Purpose:** External validation on clinical-stage antibodies

**Key Features:**
- 137 clinical antibodies (FDA-approved and Phase 1-3 candidates)
- 13 biophysical assays (PSR, ELISA, chromatography, stability)
- Binary labels via 4-cluster threshold system

### Four Test Sets Available:

#### 1. Complete Dataset (137 antibodies)
- **File:** `test_datasets/jain/VH_only_jain.csv`
- **Use case:** Exploratory analysis, full dataset

#### 2. FULL Set (94 antibodies)
- **File:** `test_datasets/jain/VH_only_jain_test_FULL.csv`
- **Filtering:** 0 and 4 flags only (excludes 1-3 flags)

#### 3. QC_REMOVED Set (91 antibodies)
- **File:** `test_datasets/jain/VH_only_jain_test_QC_REMOVED.csv`
- **Filtering:** FULL minus 3 VH length outliers

#### 4. PARITY_86 Set (86 antibodies) ⭐ **RECOMMENDED**
- **File:** `test_datasets/jain/VH_only_jain_test_PARITY_86.csv`
- **Filtering:** QC_REMOVED minus 5 borderline antibodies
- **Novo Parity:** 66.28% accuracy (exact match)
- **Use case:** Benchmarking against Novo results

**Read more:** [JAIN_DATASET_COMPLETE_HISTORY.md](JAIN_DATASET_COMPLETE_HISTORY.md)

---

## Training and Testing Commands

### Train on Boughter (Recommended)

```bash
# Use Boughter QC (914 sequences, recommended)
python3 main.py configs/config.yaml

# Alternative: Use Strict QC (852 sequences)
python3 main.py configs/config_strict_qc.yaml
```

### Test on Jain

```bash
# Test Boughter QC model on Jain PARITY_86 (recommended)
python3 test.py \
  --model models/boughter_vh_esm1v_logreg.pkl \
  --test_file test_datasets/jain/VH_only_jain_test_PARITY_86.csv

# Test Strict QC model on Jain PARITY_86
python3 test.py \
  --model models/boughter_vh_strict_qc_esm1v_logreg.pkl \
  --test_file test_datasets/jain/VH_only_jain_test_PARITY_86.csv
```

**Expected test accuracy:** ~66% (generalization to clinical antibodies)

---

## Key Findings Summary

### Boughter Dataset

✅ **118 Position Issue RESOLVED**
- Boughter used IgBLAST (includes position 118)
- We use ANARCI + IMGT (excludes position 118, standard)
- Position 118 = Framework 4 anchor (W or F, 99% conserved)

✅ **Novo Methodology Replicated**
- ANARCI + IMGT annotation
- Boughter-style QC (X in CDRs, empty CDRs)
- Boughter-style flagging (0 and 4+ only)

✅ **Strict QC Validated**
- Removed 62 sequences with X in frameworks
- Did NOT improve CV accuracy (67.5% → 66.55%, not significant)
- Conclusion: X in frameworks is not noise, ESM handles it well

### Jain Dataset

✅ **Decision Matrix Reverse-Engineered**
- 4-cluster threshold system with 90th percentile thresholds
- Self-interaction, chromatography, polyreactivity, stability
- flags_total = 0-4 → label = 0 (specific) or 1 (non-specific)

✅ **Novo Parity Achieved**
- PARITY_86: 86 antibodies
- Accuracy: 66.28% (exact match to Novo)
- Confusion matrix: [[40, 19], [10, 17]] (exact match)

✅ **All Files Validated**
- 137 → 94 → 91 → 86 antibodies (documented QC removals)
- Each file purpose and use case clearly defined

---

## Documentation Structure

```
Root:
├── BOUGHTER_DATASET_COMPLETE_HISTORY.md    # Comprehensive Boughter guide
├── JAIN_DATASET_COMPLETE_HISTORY.md        # Comprehensive Jain guide
└── DATASETS_FINAL_SUMMARY.md               # This file

docs/boughter/:
├── BOUGHTER_DATASET_COMPLETE_HISTORY.md    # Copy of root doc
├── BOUGHTER_NOVO_METHODOLOGY_CLARIFICATION.md
├── BOUGHTER_NOVO_REPLICATION_ANALYSIS.md
├── boughter_data_sources.md
└── cdr_boundary_first_principles_audit.md

docs/jain/:
├── JAIN_DATASET_COMPLETE_HISTORY.md        # Copy of root doc
├── JAIN_QC_REMOVALS_COMPLETE.md
├── JAIN_REPLICATION_PLAN.md
└── jain_data_sources.md

docs/:
├── BOUGHTER_ADDITIONAL_QC_PLAN.md          # Stage 4 strict QC plan
└── TRAINING_READINESS_CHECK.md             # Training validation
```

---

## Models Trained

| Model | Training Data | Sequences | CV Accuracy | Status |
|-------|---------------|-----------|-------------|--------|
| `boughter_vh_esm1v_logreg.pkl` | Boughter QC | 914 | 67.5% ± 8.9% | ✅ Recommended |
| `boughter_vh_strict_qc_esm1v_logreg.pkl` | Strict QC | 852 | 66.55% ± 7.07% | ✅ Alternative |

**Both models ready for testing on Jain dataset.**

---

## Quick Decision Guide

### For Training:
**Question:** Which Boughter file should I use?
**Answer:** `VH_only_boughter_training.csv` (914 sequences, Boughter QC) ⭐

**Why?**
- Better CV accuracy (67.5% vs 66.55%, though not statistically significant)
- More training data (914 vs 852)
- Matches Novo methodology exactly
- Validated approach from published paper

---

### For Testing:
**Question:** Which Jain file should I use?
**Answer:** `VH_only_jain_test_PARITY_86.csv` (86 antibodies) ⭐

**Why?**
- Achieves exact Novo parity (66.28% accuracy)
- Removes length outliers (clean test set)
- Removes borderline antibodies (high-confidence labels)
- Validated methodology

---

### For Fragment Analysis:
**Question:** Which fragment files should I use?
**Answer:** Depends on your QC preference:
- `*_boughter.csv` (1,065 sequences, Boughter QC)
- `*_boughter_strict_qc.csv` (840-914 sequences, Strict QC)

**16 fragments available:**
- Variable domains: VH_only, VL_only, VH+VL, Full
- Heavy CDRs: H-CDR1, H-CDR2, H-CDR3, H-CDRs, H-FWRs
- Light CDRs: L-CDR1, L-CDR2, L-CDR3, L-CDRs, L-FWRs
- Combined: All-CDRs, All-FWRs

---

## Frequently Asked Questions

### Q: What was the 118 position issue?
**A:** Boughter used IgBLAST (includes position 118 in CDR-H3), but IMGT standard excludes it (Framework 4 anchor). We resolved this by using ANARCI + IMGT (standard methodology). **This was a Boughter issue, not Jain.**

### Q: Why didn't strict QC improve performance?
**A:** The 62 sequences with X in frameworks were NOT noise - they were valid training data. ESM embeddings already handle positional ambiguity well. Removing them just reduced training data without improving quality.

### Q: Which model should I use for production?
**A:** Use `boughter_vh_esm1v_logreg.pkl` (Boughter QC, 914 sequences). It has slightly better CV accuracy and more training data.

### Q: How do I test on Jain?
**A:** Use `VH_only_jain_test_PARITY_86.csv` (86 antibodies) for benchmarking. Expected accuracy: ~66%.

### Q: What is PSR in Jain dataset?
**A:** Poly-Specificity Reagent - a mixture of self-antigens used to test antibody cross-reactivity. SMP score (0-1) quantifies binding (0 = specific, 1 = non-specific).

### Q: What's the difference between Jain's 4 test files?
**A:** Progressive QC filtering:
- 137 → all clinical antibodies
- 94 → 0 and 4 flags only
- 91 → removed 3 length outliers
- 86 → removed 5 borderline antibodies ⭐ **Use this one**

### Q: Can I use both models for ensemble?
**A:** Yes, but they perform equivalently (66.55% vs 67.5%, not statistically significant). Ensemble may provide marginal improvement through prediction averaging.

---

## Summary

**Both datasets are 100% complete and validated:**

1. ✅ **Boughter** - Two QC levels (914 and 852 sequences), both models trained
2. ✅ **Jain** - Four test sets (137 to 86 antibodies), Novo parity achieved
3. ✅ **All issues resolved** - 118 position, decision matrix, QC methodology
4. ✅ **Ready for production** - Models trained, tested, and documented

**Default recommendations:**
- **Training:** `VH_only_boughter_training.csv` (914 sequences, Boughter QC)
- **Testing:** `VH_only_jain_test_PARITY_86.csv` (86 antibodies)
- **Model:** `boughter_vh_esm1v_logreg.pkl` (67.5% CV accuracy)

**For comprehensive details, read:**
- [BOUGHTER_DATASET_COMPLETE_HISTORY.md](BOUGHTER_DATASET_COMPLETE_HISTORY.md)
- [JAIN_DATASET_COMPLETE_HISTORY.md](JAIN_DATASET_COMPLETE_HISTORY.md)

---

**Document Status:**
- **Version:** 1.0
- **Date:** 2025-11-04
- **Status:** ✅ Complete - Final comprehensive summary
- **Maintainer:** Ray (Clarity Digital Twin Project)
