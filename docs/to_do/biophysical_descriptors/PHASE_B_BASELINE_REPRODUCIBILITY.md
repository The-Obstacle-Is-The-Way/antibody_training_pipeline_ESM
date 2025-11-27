# Phase B: Biophysical Baseline Reproducibility

**Date**: 2025-11-27
**Status**: PENDING
**Parent**: [BIOPHYSICAL_IMPLEMENTATION_SPECS.md](BIOPHYSICAL_IMPLEMENTATION_SPECS.md)

---

## 1. Objective

Reproduce the "Track B" baseline results from Sakhnini et al. 2025 using our Phase A `BiophysicalExtractor` (Biopython Trio).

**Target Metrics (from Paper Table S2):**
*   **Theoretical pI ALONE**: 65.2% Accuracy (single descriptor baseline)
*   **3 Descriptors COMBINED**: Unknown - need to establish empirically
    * Note: Paper's Table S2 excluded Charge@pH6/7.4 from "all descriptors" model due to correlation with pI
    * Our 3-descriptor model may perform similarly to pI-only (~65%) due to high correlation

**What We're Testing:**
*   **Cross-Validation (Boughter)**: Expect ~64-67% range (pI-dominated)
*   **Test (Jain)**: Establish baseline for comparison with ESM's ~71%

## 2. Implementation Strategy

Instead of refactoring the heavy `Trainer` class (designed for 1280-d embeddings + PyTorch/SKLearn hybrids), we will create a focused **reproduction script** that utilizes our existing `Dataset` classes and the new `BiophysicalExtractor`.

This approach aligns with "Vertical Slice" architecture—delivering value (scientific verification) without premature optimization of the main pipeline.

### 2.1 New Script: `src/antibody_training_esm/cli/reproduce_track_b.py`

This script will:
1.  Load the **Boughter** dataset (Train) using `BoughterDataset`.
2.  Load the **Jain** dataset (Test) using `JainDataset`.
3.  Extract biophysical features (3-d vector) for all sequences using `BiophysicalExtractor`.
4.  Standardize features (StandardScaler) - **Critical for Logistic Regression**.
5.  Train `LogisticRegression` with 10-fold Cross-Validation on Boughter.
6.  Train a final model on full Boughter and evaluate on Jain.
7.  Save metrics to `results/track_b_baseline.json`.
8.  Print a clean report comparing results to the Paper's claims.

### 2.2 Dependencies

*   `src/antibody_training_esm/core/biophysical.py` (Phase A)
*   `src/antibody_training_esm/datasets/boughter.py`
*   `src/antibody_training_esm/datasets/jain.py`
*   `scikit-learn` (already in project)
*   `numpy`

## 3. TDD Plan

We will write an integration test *before* the script to ensure the components wire together correctly.

**Test File**: `tests/integration/test_track_b_reproducibility.py`

*   **Test 1**: `test_biophysical_extraction_on_boughter_sample`
    *   Load small subset of Boughter.
    *   Extract features.
    *   Assert shape is (N, 3).
    *   Assert no NaNs.
*   **Test 2**: `test_model_training_flow`
    *   Create synthetic (X, y) from extractor.
    *   Fit LogisticRegression.
    *   Predict.
    *   Assert pipeline runs end-to-end.

## 4. Acceptance Criteria

*   [ ] Integration test `tests/integration/test_track_b_reproducibility.py` passes.
*   [ ] Script `src/antibody_training_esm/cli/reproduce_track_b.py` implemented.
*   [ ] CLI command registered (e.g., `uv run reproduce-track-b` or via module).
*   [ ] **Experiment Run**:
    *   10-fold CV Accuracy is within 64-67% range.
    *   Results saved to `results/track_b_baseline.json`.
*   [ ] Documentation updated with findings.

## 5. Scientific Validation (The "Why")

If this script achieves ~65% accuracy with just 3 numbers (Charge@6, Charge@7.4, pI), it confirms:
1.  Our dataset (Boughter) is correctly processed.
2.  The `BiophysicalExtractor` is working as expected.
3.  We have a solid "dumb baseline" that the ESM model *must* beat significantly to justify its cost.

If it fails (<60% or >70%):
*   **<60%**: Our pI calculation might differ from the paper's (Biopython uses specific pK values), or the dataset filtering is different.
*   **>70%**: Something is wrong (data leakage?).

---

**Next Step**: Implement Integration Tests.
