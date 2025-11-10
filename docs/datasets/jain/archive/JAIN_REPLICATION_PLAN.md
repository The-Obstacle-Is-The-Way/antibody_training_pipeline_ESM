# Jain Dataset: Novo Nordisk Replication Plan (CORRECTED)

**Date**: 2025-11-03
**Status**: Bugs identified, ready to implement correct methodology
**Goal**: Replicate Novo Nordisk's Jain test set using private ELISA data + public BVP data

---

## Executive Summary

**BUGS FOUND IN PREVIOUS IMPLEMENTATION:**
1. **Missing BVP flag** - We completely omitted BVP ELISA data from public SD03
2. **Collapsed flags** - We incorrectly aggregated self-interaction, chromatography, and stability into 1 flag instead of keeping them separate

**Impact**: Wrong flag range (0-7 instead of 0-10), incorrect test set composition, off parity with Novo

**Correct Methodology**: 0-10 flags, threshold >=4, mild (1-3) excluded

---

## Data Sources

### Private Data (from Jain authors via email)
- **Private_Jain2017_ELISA_indiv.xlsx** - 6 disaggregated ELISA antigens
  - Cardiolipin, KLH, LPS, ssDNA, dsDNA, Insulin
  - 137 antibodies
  - Threshold: 1.9 OD for each antigen

### Public Data (Jain et al. 2017 supplementary)
- **jain-pnas.1616408114.sd01.xlsx** - Metadata (clinical status, provenance)
- **jain-pnas.1616408114.sd02.xlsx** - VH/VL sequences
- **jain-pnas.1616408114.sd03.xlsx** - Biophysical assays including:
  - **BVP ELISA** (threshold: 4.3 fold-over-background) ⚠️ WE WERE MISSING THIS!
  - Self-interaction assays (PSR, AC-SINS, CSI-BLI, CIC)
  - Chromatography assays (HIC, SMAC, SGAC-SINS)
  - Stability assay (AS slope)

---

## Correct Novo Methodology (Validated)

### Flag Calculation (0-10 range)

**Total: 10 individual flags**

#### Flags 1-6: ELISA (from private data)
Each antigen gets its own flag (threshold: 1.9 OD):
```python
df['flag_cardiolipin'] = (df['ELISA Cardiolipin'] > 1.9).astype(int)
df['flag_klh'] = (df['ELISA KLH'] > 1.9).astype(int)
df['flag_lps'] = (df['ELISA LPS'] > 1.9).astype(int)
df['flag_ssdna'] = (df['ELISA ssDNA'] > 1.9).astype(int)
df['flag_dsdna'] = (df['ELISA dsDNA'] > 1.9).astype(int)
df['flag_insulin'] = (df['ELISA Insulin'] > 1.9).astype(int)
```

#### Flag 7: BVP (from public SD03) ⚠️ CRITICAL - WE WERE MISSING THIS!
```python
df['flag_bvp'] = (df['BVP ELISA'] > 4.3).astype(int)
```

**Impact of missing BVP**:
- 35 antibodies exceed BVP threshold
- 3 antibodies flagged ONLY by BVP (atezolizumab, bimagrumab, ganitumab)
- Missing this flag breaks parity!

#### Flag 8: Self-Interaction (from public SD03)
ANY of these 4 assays fails → 1 flag:
```python
df['flag_self_interaction'] = (
    (df['PSR'] > 0.27) |
    (df['AC-SINS'] > 11.8) |
    (df['CSI-BLI'] > 0.01) |
    (df['CIC'] > 10.1)
).astype(int)
```

#### Flag 9: Chromatography (from public SD03)
ANY of these 3 assays fails → 1 flag:
```python
df['flag_chromatography'] = (
    (df['HIC'] > 11.7) |
    (df['SMAC'] > 12.8) |
    (df['SGAC-SINS'] < 370)
).astype(int)
```

#### Flag 10: Stability (from public SD03)
```python
df['flag_stability'] = (df['AS slope'] > 0.08).astype(int)
```

### Total Flags and Label Assignment

```python
# Sum ALL 10 flags (NOT 7!)
df['total_flags'] = (
    df['flag_cardiolipin'] + df['flag_klh'] + df['flag_lps'] +
    df['flag_ssdna'] + df['flag_dsdna'] + df['flag_insulin'] +
    df['flag_bvp'] +
    df['flag_self_interaction'] + df['flag_chromatography'] + df['flag_stability']
)

# Label assignment (threshold >=4)
df['label'] = (df['total_flags'] >= 4).astype(int)

# Test set: Exclude mild (1-3 flags)
test_set = df[(df['total_flags'] == 0) | (df['total_flags'] >= 4)]
```

### Expected Distribution

With correct 0-10 methodology:
- **Specific** (0 flags): ~60-70 antibodies (label=0)
- **Mild** (1-3 flags): EXCLUDED from test set
- **Non-specific** (>=4 flags): ~20-30 antibodies (label=1)
- **Test set size**: ~80-90 antibodies (target: 86 to match Novo)

---

## What We Did Wrong (Bug Analysis)

### Bug 1: Missing BVP Flag

**What we did**: Completely ignored BVP ELISA data
**What we should do**: Include BVP from public SD03 as Flag 7

**Code location**: `preprocessing/jain/step1_convert_excel_to_csv.py` (currently missing BVP entirely)

**Evidence BVP exists**:
```python
# From SD03:
sd03['BVP ELISA'].describe()
# count    137.000000
# mean       4.072574
# std        1.867419
# max       11.600000

# Antibodies with BVP > 4.3: 35 total
# Antibodies flagged ONLY by BVP: 3 (atezolizumab, bimagrumab, ganitumab)
```

### Bug 2: Collapsed Flags

**What we did** (lines 111-129 of buggy script):
```python
# WRONG: Aggregated 3 flags into 1
df['flag_other_aggregated'] = (
    (df['flag_self_interaction'] == 1) |
    (df['flag_chromatography'] == 1) |
    (df['flag_stability'] == 1)
).astype(int)

# WRONG: Results in 0-7 range
df['total_flags'] = df['elisa_flags'] + df['flag_other_aggregated']
```

**What we should do**: Keep all 3 as SEPARATE flags
```python
# CORRECT: Keep flags separate
df['total_flags'] = (
    df['elisa_flags'] +           # 0-6
    df['flag_bvp'] +              # 0-1
    df['flag_self_interaction'] + # 0-1
    df['flag_chromatography'] +   # 0-1
    df['flag_stability']          # 0-1
)  # Total: 0-10 range
```

**Why this matters**: Collapsing 3 flags into 1 artificially reduces flag counts:
- Antibody with self=1, chrom=1, stability=1 should have 3 flags
- But our buggy code gives it only 1 flag
- This promotes non-specific antibodies to "specific" incorrectly

### Bug Impact on Test Set

**Buggy output** (0-7 range):
- 62 specific (0 flags)
- 50 mild (1-3 flags)
- 25 non-specific (>=4 flags)
- Test set: 87 antibodies

**Expected correct output** (0-10 range):
- ~60-70 specific (0 flags)
- ~30-40 mild (1-3 flags) - EXCLUDED
- ~20-30 non-specific (>=4 flags)
- Test set: ~80-90 antibodies

---

## Implementation Plan

### Step 1: Fix Flag Calculation Script

**File**: `preprocessing/jain/step1_convert_excel_to_csv.py`

**Changes needed**:
1. Load BVP from SD03 alongside other public assays
2. Add BVP flag calculation: `df['flag_bvp'] = (df['BVP ELISA'] > 4.3).astype(int)`
3. Remove `flag_other_aggregated` entirely
4. Update `total_flags` to sum all 10 individual flags
5. Update docstrings to reflect 0-10 range

### Step 2: Regenerate All Output Files

Run corrected script to generate:
- `test_datasets/jain_with_private_elisa_FULL.csv` (all 137 antibodies)
- `test_datasets/jain_with_private_elisa_TEST.csv` (specific + non-specific only)
- `test_datasets/jain/canonical/VH_only_jain_test_PARITY_86.csv` (for model inference)

### Step 3: Validate Distributions

Check that corrected outputs show:
- Test set size close to 86 antibodies
- Reasonable specific/non-specific split
- Flag distribution that makes sense (0-10 range)

### Step 4: Run Model Inference

Test model on corrected dataset:
- Target accuracy: ~68.6% (Novo's reported performance)
- Compare to previous buggy results

### Step 5: Clean Up Documentation

Update/delete files:
- ✅ Keep: `docs/JAIN_QC_REMOVALS_COMPLETE.md`
- ✅ Keep: `docs/research/novo-parity.md`
- ✅ Keep: `docs/jain/jain_conversion_verification_report.md`
- ✅ Keep: `docs/jain/jain_data_sources.md`
- ❌ Delete: All internal debug/investigation docs (see list in background bash)

---

## Source of Truth References

### Novo Paper (Sakhnini et al. 2025)
- **Table 2, line 201**: "137 clinical-stage IgG1-formatted antibodies with ELISA with a panel of 6 ligands"
- **Line 127**: "non-specific (>3 flags)" → means >=4 threshold
- **Methods**: Excludes mild (1-3 flags) from test set

### Original Jain Paper (Jain et al. 2017)
- **Table 1**: Shows BVP threshold of 4.3 (90th percentile of approved antibodies)
- **Supplementary Data 03**: Contains all 12 biophysical assays including BVP
- **Methodology**: Uses grouped flags (0-4) where polyreactivity = ELISA OR BVP

### Key Distinction
- **Jain 2017**: Grouped methodology (0-4 flags)
- **Novo 2025**: Disaggregated methodology (0-10 flags) using private ELISA + public BVP

---

## Success Criteria

✅ **All 10 flags calculated correctly** (6 ELISA + BVP + self + chrom + stability)
✅ **Test set size ~86 antibodies** (matching Novo's reported size)
✅ **No aggregation bugs** (all flags kept separate)
✅ **BVP data included** (from public SD03)
✅ **Model accuracy ~68-70%** (close to Novo's 68.6%)
✅ **Clean provenance** (scripts, intermediate files, documentation)

---

## Next Actions

1. **Fix** `preprocessing/jain/step1_convert_excel_to_csv.py` (add BVP, remove aggregation)
2. **Regenerate** all Jain output files with corrected methodology
3. **Validate** flag distributions and test set size
4. **Test** model inference on corrected dataset
5. **Clean up** internal debug documentation
6. **Document** final parity analysis

---

**Author**: Claude Code
**Reviewed by**: Ray Wu
**Status**: Ready to implement fixes
**Last Updated**: 2025-11-03
