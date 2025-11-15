# Harvey Dataset Documentation

**Status:** ✅ Complete and validated
**Pipeline:** `data/test/harvey/` → `preprocessing/harvey/`

---

## Quick Start

**For the complete Harvey dataset pipeline, see the authoritative documentation:**

👉 **[`data/test/harvey/README.md`](../../../data/test/harvey/README.md)** ← **SSOT**

That README contains:
- Complete 2-step pipeline (convert raw CSVs → extract fragments)
- Dataset statistics (141,474 → 141,021 sequences)
- Data source information (official Harvey Lab repo)
- Execution instructions
- Validation procedures

---

## Current Documentation Structure

### Active Reference Docs

**Technical Reports:**
- [`harvey_p0_fix_report.md`](harvey_p0_fix_report.md) - Gap character fix (ANARCI sequence_aa vs sequence_alignment_aa)
- [`harvey_test_results.md`](harvey_test_results.md) - Benchmark validation (61.5% vs Novo's 61.7%)

**Methodology & Status:**
- [`harvey_data_sources.md`](harvey_data_sources.md) - Data provenance and source verification
- [`harvey_preprocessing_implementation_plan.md`](harvey_preprocessing_implementation_plan.md) - Implementation methodology
- [`harvey_script_status.md`](harvey_script_status.md) - Current script status

### Historical Archive

**Development logs (November 2025):**
- [`archive/harvey_data_cleaning_log.md`](archive/harvey_data_cleaning_log.md) - Initial data discovery
- [`archive/harvey_script_audit_request.md`](archive/harvey_script_audit_request.md) - External code audit
- [`archive/harvey_cleanup_investigation.md`](archive/harvey_cleanup_investigation.md) - File reorganization plan
- [`archive/harvey_cleanup_verification.md`](archive/harvey_cleanup_verification.md) - Cleanup verification results

⚠️ **Note:** Archived docs contain historical status warnings that do not reflect current state.

---

## Harvey Dataset Summary

**Source:** Harvey et al. (2022) - Nanobody polyreactivity from FACS + deep sequencing
**Size:** 141,474 nanobodies (VHH only)
**Processed:** 141,021 sequences (453 ANARCI failures, 0.32%)
**Labels:** Binary (0=low polyreactivity, 1=high polyreactivity)
**Assay:** PSR (Poly-Specificity Reagent)

**Official Repository:** `debbiemarkslab/nanobody-polyreactivity`

---

## Key Results

### P0 Fix (2025-11-02)
- **Issue:** Gap characters in VHH sequences (8.6% affected)
- **Fix:** Changed `annotation.sequence_alignment_aa` → `annotation.sequence_aa`
- **Result:** All 141,021 sequences now ESM-1v compatible

### Benchmark Validation (2025-11-03)
- **Our result:** 61.5% accuracy
- **Novo Nordisk:** 61.7% accuracy
- **Difference:** Only 0.2 percentage points
- **Sensitivity:** 95.4% (better than Novo's 94.2%)

✅ **Near-perfect replication achieved**

---

## Quick Links

**Data:**
- Raw CSVs: `data/test/harvey/raw/`
- Processed: `data/test/harvey/processed/harvey.csv`
- Fragments: `data/test/harvey/fragments/*.csv` (6 fragment types)

**Scripts:**
- Step 1: `preprocessing/harvey/step1_convert_raw_csvs.py`
- Step 2: `preprocessing/harvey/step2_extract_fragments.py`
- Testing: `preprocessing/harvey/test_psr_threshold.py`

**Tests:**
- Embedding compatibility: `tests/test_harvey_embedding_compatibility.py`
- Fragment validation: `scripts/validation/validate_fragments.py`

---

## References

- **Harvey et al. (2022):** An in silico method to assess antibody fragment polyreactivity. *Nat Commun* 13, 7554. [DOI: 10.1038/s41467-022-35276-4](https://doi.org/10.1038/s41467-022-35276-4)
- **Sakhnini et al. (2025):** Prediction of Antibody Non-Specificity using Protein Language Models. *bioRxiv* [DOI: 10.1101/2025.04.28.650927](https://doi.org/10.1101/2025.04.28.650927)

---

**Last Updated:** 2025-11-06
**Status:** ✅ Production ready
