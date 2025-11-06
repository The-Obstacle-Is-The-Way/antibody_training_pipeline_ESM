# Legacy Jain Fragments (DEPRECATED)

**Status**: ⚠️ DEPRECATED - DO NOT USE FOR TRAINING

## Why These Are Deprecated

These fragment files were generated from the **wrong source** and use **incorrect labels**.

### The Problem

- **Source Used**: `jain.csv` (paper-based `flags_total` labeling)
- **Label Distribution**: 67 specific / 27 non-specific / 43 mild
- **Error Rate**: 53/137 antibodies (38.7%) have wrong labels
- **Root Cause**: Used paper-based flags (4 assays) instead of ELISA flags (6 reagents)

### The Fix

New fragments generated from correct source:
- **Source**: `jain_with_private_elisa_FULL.csv` (ELISA-based labeling)
- **Label Distribution**: 94 specific / 22 non-specific / 21 mild
- **Location**: `test_datasets/jain/fragments/` (replaced these files)

## Labeling Systems Comparison

### DEPRECATED (flags_total - Paper-based)
- **Flags**: 4 independent assays (self-interaction, chromatography, polyreactivity, stability)
- **Range**: 0-4 flags total
- **Labeling**:
  - `flags_total = 0` → Specific (label 0)
  - `flags_total = 1-3` → Mild (label NaN)
  - `flags_total ≥ 4` → Non-specific (label 1)
- **Distribution**: 67/27/43

### CORRECT (elisa_flags - ELISA-based SSOT)
- **Flags**: 6 independent ELISA reagents (cardiolipin, KLH, LPS, ssDNA, dsDNA, insulin)
- **Range**: 0-6 flags total
- **Labeling**:
  - `elisa_flags = 0` → Specific (label 0)
  - `elisa_flags = 1-3` → Mild (label NaN)
  - `elisa_flags ≥ 4` → Non-specific (label 1)
- **Distribution**: 94/22/21

## Files in This Directory

All 16 fragment files with WRONG labels (flags_total-based):

1. `VH_only_jain.csv` - Full heavy variable domain
2. `VL_only_jain.csv` - Full light variable domain
3. `H-CDR1_jain.csv` - Heavy CDR1
4. `H-CDR2_jain.csv` - Heavy CDR2
5. `H-CDR3_jain.csv` - Heavy CDR3
6. `L-CDR1_jain.csv` - Light CDR1
7. `L-CDR2_jain.csv` - Light CDR2
8. `L-CDR3_jain.csv` - Light CDR3
9. `H-CDRs_jain.csv` - Concatenated H-CDR1+2+3
10. `L-CDRs_jain.csv` - Concatenated L-CDR1+2+3
11. `H-FWRs_jain.csv` - Concatenated H-FWR1+2+3+4
12. `L-FWRs_jain.csv` - Concatenated L-FWR1+2+3+4
13. `VH+VL_jain.csv` - Paired variable domains
14. `All-CDRs_jain.csv` - All CDRs concatenated
15. `All-FWRs_jain.csv` - All FWRs concatenated
16. `Full_jain.csv` - Full antibody

## Example Label Discrepancies

| Antibody ID | Legacy Label | Correct Label | ELISA Flags | Flags Total | Issue |
|-------------|--------------|---------------|-------------|-------------|-------|
| atezolizumab | 1 (non-spec) | 0 (specific) | 0 | 3 | Has 3 paper flags, 0 ELISA flags |
| bapineuzumab | NaN (mild) | 0 (specific) | 0 | 1 | Has 1 paper flag, 0 ELISA flags |
| belimumab | NaN (mild) | 1 (non-spec) | 6 | 0 | Has 0 paper flags, 6 ELISA flags |
| bimagrumab | 1 (non-spec) | 0 (specific) | 0 | 4 | Has 4 paper flags, 0 ELISA flags |
| carlumab | NaN (mild) | 1 (non-spec) | 4 | 2 | Has 2 paper flags, 4 ELISA flags |

**Pattern**: Paper flags and ELISA flags are independent measurements. An antibody can have high paper flags but low ELISA flags (or vice versa).

## History

- **Created**: Nov 2, 2025 (commit `aa1c4da`)
- **Moved**: Nov 5, 2025 (commit `9de5687` - directory reorganization)
- **Deprecated**: Nov 6, 2025 (discovered 38.7% label error rate)
- **Replaced**: Nov 6, 2025 (regenerated from ELISA-based source)

## References

- Investigation findings: `JAIN_LABEL_DISCREPANCY_FINDINGS.md`
- Correct source: `test_datasets/jain/processed/jain_with_private_elisa_FULL.csv`
- Regeneration script: `preprocessing/jain/step3_extract_fragments.py`
- New fragments: `test_datasets/jain/fragments/`

---

**DO NOT USE THESE FILES FOR TRAINING OR EVALUATION**

Use the corrected fragments in `test_datasets/jain/fragments/` instead.
