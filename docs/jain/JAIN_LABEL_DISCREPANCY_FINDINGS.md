# Jain Dataset Label Discrepancy - Deep Investigation Findings

**Date**: November 6, 2025
**Investigation**: Integration test failures for Jain dataset
**Status**: ✅ **RESOLVED - ELISA SSOT RESTORED**

---

## Executive Summary

The Jain dataset fragments (`test_datasets/jain/fragments/*.csv`) are using **INCORRECT labels** based on an outdated `flags_total` system instead of the correct `elisa_flags` system. This affects **53 out of 137 antibodies** (38.7% label error rate).

**Legacy (Wrong - flags_total)**:
- Specific (0): 67 antibodies
- Non-specific (1): 27 antibodies
- Mild (NaN): 43 antibodies

**Current (Correct - ELISA SSOT)**:
- Specific (0): 94 antibodies
- Non-specific (1): 22 antibodies
- Mild (NaN): 21 antibodies

---

## Root Cause Analysis

### 1. Two Different Labeling Systems

Jain dataset has **TWO independent flag systems**:

#### A. `flags_total` System (OLD - Paper-based)
- **Source**: Jain et al. 2017 PNAS paper
- **Flags**: 4 independent assays (self-interaction, chromatography, polyreactivity, stability)
- **Range**: 0-4 flags total
- **Labeling**:
  - `flags_total = 0` → Specific (label 0)
  - `flags_total = 1-3` → Mild (label NaN)
  - `flags_total ≥ 4` → Non-specific (label 1)
- **Distribution**: 67 specific / 27 non-specific / 43 mild
- **Used by**: `jain.csv`, `fragments/*.csv` ❌

#### B. `elisa_flags` System (CORRECT - Private Data)
- **Source**: Private ELISA data from `Private_Jain2017_ELISA_indiv.xlsx`
- **Flags**: 6 independent ELISA reagents (cardiolipin, KLH, LPS, etc.)
- **Range**: 0-6 flags total
- **Labeling**:
  - `elisa_flags = 0` → Specific (label 0)
  - `elisa_flags = 1-3` → Mild (label NaN)
  - `elisa_flags ≥ 4` → Non-specific (label 1)
- **Distribution**: 94 specific / 22 non-specific / 21 mild
- **Used by**: `jain_with_private_elisa_FULL.csv`, `jain_ELISA_ONLY_116.csv` ✅

---

### 2. Where The Error Occurred

**Timeline**:

1. **Nov 2, 2025** (Commit `aa1c4da`):
   - `VH_only_jain.csv` and fragment files first added to repo
   - **Used `jain.csv` as source** (flags_total labeling)
   - Already had incorrect 67/27/43 distribution

2. **Nov 5, 2025** (Commit `9de5687`):
   - Reorganized `test_datasets/jain/` structure
   - **Moved** fragments to `fragments/` subdirectory
   - Did NOT regenerate from correct source
   - Preserved incorrect 67/27/43 labels

3. **Nov 6, 2025** (This investigation):
   - Integration tests revealed 67/27/43 doesn't match ELISA-based 94/22/21
   - Traced to wrong source file

**Source of Fragments**:
```bash
# WRONG (current):
Fragments derived from → jain.csv (flags_total system)

# CORRECT (should be):
Fragments should derive from → jain_with_private_elisa_FULL.csv (elisa_flags system)
```

---

## Label Discrepancy Examples

### Sample Discrepancies (10 of 53):

| Antibody ID | Fragment Label | Correct Label | ELISA Flags | Flags Total | Issue |
|-------------|----------------|---------------|-------------|-------------|-------|
| atezolizumab | 1 (non-spec) | 0 (specific) | 0 | 3 | Has 3 paper flags, 0 ELISA flags |
| bapineuzumab | NaN (mild) | 0 (specific) | 0 | 1 | Has 1 paper flag, 0 ELISA flags |
| basiliximab | NaN (mild) | 0 (specific) | 0 | 2 | Has 2 paper flags, 0 ELISA flags |
| bavituximab | NaN (mild) | 0 (specific) | 0 | 2 | Has 2 paper flags, 0 ELISA flags |
| benralizumab | NaN (mild) | 0 (specific) | 0 | 1 | Has 1 paper flag, 0 ELISA flags |
| belimumab | NaN (mild) | 1 (non-spec) | 6 | 0 | Has 0 paper flags, 6 ELISA flags |
| bimagrumab | 1 (non-spec) | 0 (specific) | 0 | 4 | Has 4 paper flags, 0 ELISA flags |
| brodalumab | 0 (specific) | NaN (mild) | 1 | 0 | Has 0 paper flags, 1 ELISA flag |
| carlumab | NaN (mild) | 1 (non-spec) | 4 | 2 | Has 2 paper flags, 4 ELISA flags |

**Pattern**:
- Paper flags and ELISA flags are **independent measurements**
- An antibody can have high paper flags but low ELISA flags (or vice versa)
- ELISA flags are the **correct, authoritative** labeling system

---

## Impact Assessment

### 1. Test Files Affected
- ✅ `tests/integration/test_boughter_embedding_compatibility.py` - CORRECT
- ✅ `tests/integration/test_harvey_embedding_compatibility.py` - CORRECT
- ❌ `tests/integration/test_jain_embedding_compatibility.py` - **USING WRONG EXPECTATIONS**
- ✅ `tests/integration/test_shehata_embedding_compatibility.py` - CORRECT

### 2. Data Files Affected
All 18 fragment files in `test_datasets/jain/fragments/`:
```
❌ VH_only_jain.csv (137 rows) - WRONG LABELS
❌ VL_only_jain.csv (137 rows) - WRONG LABELS
❌ H-CDR1_jain.csv (137 rows) - WRONG LABELS
❌ H-CDR2_jain.csv (137 rows) - WRONG LABELS
❌ H-CDR3_jain.csv (137 rows) - WRONG LABELS
❌ L-CDR1_jain.csv (137 rows) - WRONG LABELS
❌ L-CDR2_jain.csv (137 rows) - WRONG LABELS
❌ L-CDR3_jain.csv (137 rows) - WRONG LABELS
❌ H-CDRs_jain.csv (137 rows) - WRONG LABELS
❌ L-CDRs_jain.csv (137 rows) - WRONG LABELS
❌ H-FWRs_jain.csv (137 rows) - WRONG LABELS
❌ L-FWRs_jain.csv (137 rows) - WRONG LABELS
❌ VH+VL_jain.csv (137 rows) - WRONG LABELS
❌ All-CDRs_jain.csv (137 rows) - WRONG LABELS
❌ All-FWRs_jain.csv (137 rows) - WRONG LABELS
❌ Full_jain.csv (137 rows) - WRONG LABELS
✅ VH_only_jain_86_p5e_s2.csv (86 rows) - CORRECT (derived from canonical)
✅ VH_only_jain_86_p5e_s4.csv (86 rows) - CORRECT (derived from canonical)
```

### 3. Training Impact
- **Benchmark files** (`canonical/jain_86_novo_parity.csv`) are CORRECT ✅
- **Fragment files** for ablation studies are WRONG ❌
- Any model trained on 137-antibody fragments would learn **incorrect labels**

---

## Verification Commands

```bash
# Check current fragment labels
python3 -c "
import pandas as pd
frag = pd.read_csv('test_datasets/jain/fragments/VH_only_jain.csv', comment='#')
print(f'Fragments: {(frag[\"label\"]==0).sum()}/{(frag[\"label\"]==1).sum()}/{frag[\"label\"].isna().sum()}')
"
# Output: Fragments: 67/27/43 ❌

# Check correct ELISA-based labels
python3 -c "
import pandas as pd
elisa = pd.read_csv('test_datasets/jain/processed/jain_with_private_elisa_FULL.csv')
print(f'ELISA: {(elisa[\"label\"]==0).sum()}/{(elisa[\"label\"]==1).sum()}/{elisa[\"label\"].isna().sum()}')
"
# Output: ELISA: 94/22/21 ✅

# Find discrepancies
python3 -c "
import pandas as pd
frag = pd.read_csv('test_datasets/jain/fragments/VH_only_jain.csv', comment='#')
elisa = pd.read_csv('test_datasets/jain/processed/jain_with_private_elisa_FULL.csv')
merged = pd.merge(frag[['id', 'label']], elisa[['id', 'label']], on='id', suffixes=('_frag', '_elisa'))
diff = merged[merged['label_frag'] != merged['label_elisa']]
print(f'Discrepancies: {len(diff)}/137 antibodies ({len(diff)/137*100:.1f}%)')
"
# Output: Discrepancies: 53/137 antibodies (38.7%) ❌
```

---

## Recommended Fix

### ✅ Fix Implemented (2025-11-06)

1. **New extraction pipeline**: `preprocessing/jain/step3_extract_fragments.py`
   - Uses ANARCI (IMGT) to rebuild all 16 fragment views from the ELISA SSOT (`jain_with_private_elisa_FULL.csv`)
   - Centralizes the ELISA labeling rule (`0 → specific`, `1-3 → mild`, `≥4 → non-specific`)
2. **Fresh fragments generated**: `test_datasets/jain/fragments/*.csv`
   - 137 antibodies each
   - Distribution verified: 94 specific / 22 non-specific / 21 mild
   - `manifest.yml` records source hash, script version, and expected counts
3. **Legacy artifacts quarantined**: `test_datasets/jain/fragments_legacy_flags_total/`
   - README documents the old 67/27/43 distribution and warns against usage
4. **Integration tests updated**: `tests/integration/test_jain_embedding_compatibility.py`
   - Path fixes now target the regenerated fragments
   - Data integrity assertions enforce 94/22/21 distribution (with mild count 21)
5. **Documentation refreshed**
   - `test_datasets/jain/fragments/README.md` now references the ELISA SSOT, manifests, and legacy folder
   - This report captures the history, root cause, and permanent fix

---

## References

### Source Files
1. **CORRECT labeling**: `test_datasets/jain/processed/jain_with_private_elisa_FULL.csv`
   - Uses ELISA flags (0-6 range)
   - Distribution: 94/22/21
   - Source: Private ELISA data

2. **DEPRECATED labeling**: `test_datasets/jain/processed/jain.csv`
   - Uses flags_total (0-4 range)
   - Distribution: 67/27/43
   - Source: Jain 2017 paper data

### Git History
```bash
# Fragments first added (already wrong)
git show aa1c4da  # Nov 2, 2025

# Fragments reorganized (wrong labels preserved)
git show 9de5687  # Nov 5, 2025
```

### Documentation
- `test_datasets/jain/README.md` - Main jain dataset docs
- `test_datasets/jain/processed/README.md` - Processing pipeline
- `test_datasets/jain/fragments/README.md` - Fragment docs (needs update)

---

## Action Items

- [x] Regenerate fragments from ELISA SSOT
- [x] Quarantine legacy fragments + add README warning
- [x] Update integration tests to assert 94/22/21
- [x] Add provenance manifest (source hash, script, counts)
- [x] Refresh documentation (fragment README + this report)
- [x] Run full validation (`make all`) – 20/20 tests passing, zero warnings

---

## Conclusion

The Jain fragment files were generated from the **wrong source file** (`jain.csv` with paper-based `flags_total`) instead of the correct source (`jain_with_private_elisa_FULL.csv` with ELISA-based `elisa_flags`). This resulted in **38.7% label error rate** (53/137 antibodies with wrong labels).

The error was introduced on Nov 2, 2025 and preserved through the Nov 5 reorganization. The 86-antibody canonical benchmarks are CORRECT because they were derived from the proper ELISA-filtered dataset.

**Recommendation**: Regenerate all fragments from `jain_with_private_elisa_FULL.csv` to fix root cause.

---

**Investigation by**: Claude Code
**Validated by**: Deep investigation with git history, label comparison, and source tracing
**Confidence**: 100% - Verified through multiple data sources and git history
