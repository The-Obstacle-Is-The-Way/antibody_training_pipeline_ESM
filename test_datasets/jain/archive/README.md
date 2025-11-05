# Jain Dataset Archive

## ⚠️ DEPRECATED FILES - DO NOT USE

This archive contains historical versions and incorrect methodologies.
Use files in the parent directory (`test_datasets/jain/`) instead.

### jain_116_qc_candidates.csv
- **Status:** Intermediate analysis
- **Purpose:** List of 24 QC removal candidates

### jain_ELISA_ONLY_116_with_zscores.csv
- **Status:** Intermediate analysis
- **Purpose:** 116 antibodies after ELISA filter, with z-scores

### legacy_reverse_engineered/
- **Status:** ✅ DUPLICATES of files in parent directory
- **Reason archived:** Redundant (same as files in test_datasets/jain/)
- **Created:** Nov 2, 2025
- **Files:** 
  - VH_only_jain_test_FULL.csv (94)
  - VH_only_jain_test_PARITY_86.csv (86)
  - VH_only_jain_test_QC_REMOVED.csv (91)

### legacy_total_flags_methodology/
- **Status:** ❌ **INCORRECT METHODOLOGY - DO NOT USE**
- **Reason archived:** Uses wrong ELISA column (`total_flags` instead of `elisa_flags`)
- **Created:** Nov 2-3, 2025
- **Files:** 13 CSV files (94 and 86 antibody variants)
- **⚠️ These files should NEVER be used for analysis!**

---

**Use instead:** Files in `test_datasets/jain/` (parent directory)

**See:** `JAIN_COMPLETE_GUIDE.md` for full documentation

**Date archived:** 2025-11-05
