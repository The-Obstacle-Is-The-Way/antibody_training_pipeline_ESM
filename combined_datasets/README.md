# Combined Datasets

This directory contains merged/preprocessed datasets used for transfer learning experiments.

## Files

### `boughter_training.csv` (268K, 914 antibodies)
- **Source:** Boughter et al. (2020) eLife
- **License:** Public data (eLife Open Access)
- **Processing:** Filtered to `include_in_training==True` (0 and 4+ flags only)
- **Format:** VH+VL sequences + binary labels (ELISA polyreactivity)
- **Safe to commit:** ✅ YES (our data, already in repo)

### `ginkgo_labeled.csv` (56K, 197 antibodies)
- **Source:** Derived from GDPa1 v1.2 (Ginkgo/HuggingFace)
- **License:** Public (HuggingFace datasets)
- **Processing:** Filtered to labeled antibodies only, added binary labels (median split)
- **Format:** VH+VL sequences + PR_CHO labels + folds
- **Safe to commit:** ✅ YES (preprocessed/derived, not raw competition data)
- **Note:** This is a DERIVED dataset for research purposes. Raw GDPa1 data in `train_datasets/ginkgo/` is NOT committed.

### `boughter_ginkgo_combined.csv` (325K, 1111 antibodies)
- **Source:** Merge of above two datasets
- **Purpose:** Transfer learning experiments (failed - see GINKGO_2025_IMPROVEMENT_ROADMAP.md)
- **Format:** Union of Boughter + GDPa1 labeled sets
- **Safe to commit:** ✅ YES (derived from committed datasets)

## Why These Files Exist

### Experiment: Boughter → GDPa1 Transfer Learning

**Hypothesis:** Pre-training on Boughter (914 samples) should improve GDPa1 prediction (197 samples)

**Results:** ❌ FAILED
- Transfer learning: 0.491 Spearman (-1.93% vs baseline)
- Combined training: 0.461 Spearman (-7.88% vs baseline)

**Root cause:**
- ELISA (Boughter) ≠ PR_CHO (GDPa1)
- Different assays, different antigens, different label semantics
- Boughter patterns don't transfer to GDPa1

**See:** `GINKGO_2025_IMPROVEMENT_ROADMAP.md` for full analysis

## Provenance

All data in this directory is:
1. ✅ From public sources (eLife, HuggingFace)
2. ✅ Preprocessed/derived (not raw competition data)
3. ✅ Small enough for version control (< 1 MB total)
4. ✅ Necessary for reproducing experiments

Raw GDPa1 competition data (`train_datasets/ginkgo/*.csv`) is NOT committed (see `.gitignore`).

---

**Created:** 2025-11-13
**Purpose:** Transfer learning experiments
**Status:** Archived (experiments complete, transfer learning failed)
