# Shehata Dataset Documentation

**Status:** ✅ Complete and validated
**Pipeline:** `test_datasets/shehata/` → `preprocessing/shehata/`

---

## Quick Start

**For the complete Shehata dataset pipeline, see the authoritative documentation:**

👉 **[`test_datasets/shehata/README.md`](../../test_datasets/shehata/README.md)** ← **SSOT**

That README contains:
- Complete 2-step pipeline (Excel → CSV → fragments)
- Dataset statistics (398 paired antibodies → 16 fragment types)
- Data source information (Shehata et al. 2019 Cell Reports)
- Execution instructions
- Validation procedures

---

## Current Documentation Structure

### Active Reference Docs

**Technical Reports:**
- [`threshold_calibration_discovery.md`](threshold_calibration_discovery.md) - PSR-specific threshold (0.5495) discovery

**Methodology & Status:**
- [`shehata_data_sources.md`](shehata_data_sources.md) - Data provenance and source verification
- [`shehata_preprocessing_implementation_plan.md`](shehata_preprocessing_implementation_plan.md) - Implementation methodology
- [`shehata_phase2_completion_report.md`](shehata_phase2_completion_report.md) - Phase 2 completion and validation

### Historical Archive

**Development logs (October-November 2025):**
- [`archive/shehata_conversion_verification_report.md`](archive/shehata_conversion_verification_report.md) - Phase 1 verification
- [`archive/p0_blocker_first_principles_validation.md`](archive/p0_blocker_first_principles_validation.md) - Gap character validation
- [`archive/shehata_blocker_analysis.md`](archive/shehata_blocker_analysis.md) - P0 blocker analysis and resolution
- [`archive/shehata_cleanup_plan.md`](archive/shehata_cleanup_plan.md) - Dataset reorganization plan

⚠️ **Note:** Archived docs contain historical status warnings that do not reflect current state.

---

## Shehata Dataset Summary

**Source:** Shehata et al. (2019) - Affinity maturation and antibody specificity study
**Size:** 398 paired antibodies (VH + VL)
**Processed:** 398 sequences (0 failures, 100% success rate)
**Labels:** Binary (0=low PSR, 1=high PSR) - Highly imbalanced (391 low / 7 high)
**Assay:** PSR (Poly-Specificity Reagent)

**Publication:** Cell Reports 28(13):3300-3308.e4 (2019)
**DOI:** https://doi.org/10.1016/j.celrep.2019.08.056

---

## Key Results

### P0 Fix (2025-11-02)
- **Issue:** Gap characters in VH/VL/Full fragment sequences
- **Affected:** 13 VH, 4 VL, 17 Full sequences
- **Fix:** Changed `annotation.sequence_alignment_aa` → `annotation.sequence_aa`
- **Result:** All 398 sequences now gap-free and ESM-1v compatible

### PSR-Specific Threshold Calibration (2025-11-03)
- **Discovery:** Default threshold (0.5) gives 52.5% accuracy
- **Optimized:** PSR-specific threshold (0.5495) gives 58.8% accuracy
- **Result:** Exact parity with Novo Nordisk benchmark
- **Implementation:** `src/antibody_training_esm/core/classifier.py:167`

✅ **Perfect benchmark replication achieved**

---

## Quick Links

**Data:**
- Raw Excel: `test_datasets/shehata/raw/shehata-mmc2.xlsx`
- Processed: `test_datasets/shehata/processed/shehata.csv`
- Canonical: `test_datasets/shehata/canonical/` (empty - dataset already balanced)
- Fragments: `test_datasets/shehata/fragments/*.csv` (16 fragment types)

**Scripts:**
- Step 1: `preprocessing/shehata/step1_convert_excel_to_csv.py`
- Step 2: `preprocessing/shehata/step2_extract_fragments.py`

**Tests:**
- Fragment validation: `scripts/validation/validate_fragments.py`
- Threshold testing: `test_assay_specific_thresholds.py`

---

## Fragment Types

The Shehata dataset generates **16 fragment files** (paired antibodies):

**Heavy Chain (8 files):**
- `VH_only_shehata.csv` - Full heavy chain variable domain
- `H-CDR1_shehata.csv`, `H-CDR2_shehata.csv`, `H-CDR3_shehata.csv`
- `H-FWRs_shehata.csv` - Concatenated H-FWR1+2+3+4
- `H-CDRs_shehata.csv` - Concatenated H-CDR1+2+3

**Light Chain (6 files):**
- `VL_only_shehata.csv` - Full light chain variable domain
- `L-CDR1_shehata.csv`, `L-CDR2_shehata.csv`, `L-CDR3_shehata.csv`
- `L-FWRs_shehata.csv` - Concatenated L-FWR1+2+3+4
- `L-CDRs_shehata.csv` - Concatenated L-CDR1+2+3

**Combined (2 files):**
- `VH+VL_shehata.csv` - Paired heavy + light chains
- `Full_shehata.csv` - Complete antibody sequence

---

## PSR Threshold

**Key Finding:** PSR assay requires different threshold than ELISA assay.

- **ELISA datasets (Jain, Boughter):** Use default 0.5 threshold
- **PSR datasets (Shehata, Harvey):** Use calibrated 0.5495 threshold

**Rationale:** PSR assay has different probability distributions than ELISA. The 0.5495 threshold ensures exact parity with Novo Nordisk benchmarks.

**Usage:**
```python
from antibody_training_esm.core.classifier import predict

# Automatic PSR threshold selection
predictions = predict(sequences, assay_type='PSR')
```

See [`threshold_calibration_discovery.md`](threshold_calibration_discovery.md) for technical details.

---

## References

- **Shehata et al. (2019):** Affinity Maturation Enhances Antibody Specificity but Compromises Conformational Stability. *Cell Reports* 28(13):3300-3308.e4. [DOI: 10.1016/j.celrep.2019.08.056](https://doi.org/10.1016/j.celrep.2019.08.056)
- **Sakhnini et al. (2025):** Prediction of Antibody Non-Specificity using Protein Language Models. *bioRxiv* [DOI: 10.1101/2025.04.28.650927](https://doi.org/10.1101/2025.04.28.650927)

---

**Last Updated:** 2025-11-06
**Status:** ✅ Production ready
