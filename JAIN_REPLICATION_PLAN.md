# Jain Dataset: Novo Nordisk Replication Plan

**Date**: 2025-11-03
**Status**: Clean implementation plan (ignoring all reverse-engineering confusion)
**Goal**: Replicate Novo Nordisk's Jain test set using private ELISA data

---

## What We Have Now

### Data Files
1. **Private_Jain2017_ELISA_indiv.xlsx** - 6 disaggregated ELISA antigens from Jain authors
   - Cardiolipin, KLH, LPS, ssDNA, dsDNA, Insulin
   - 137 antibodies

2. **jain-pnas.1616408114.sd01.xlsx** - Metadata (clinical status, provenance)
3. **jain-pnas.1616408114.sd02.xlsx** - VH/VL sequences
4. **jain-pnas.1616408114.sd03.xlsx** - Other biophysical assays

5. **Novo paper + supplementary** - Their reported methodology

---

## What Novo Nordisk Did (From Their Paper)

### Flag Calculation Methodology

**From Sakhnini et al. 2025:**

#### Data Sources (Table 2, line 201):
> "137 clinical-stage IgG1-formatted antibodies with their respective non-specificity flag from ELISA with a panel of **6 ligands**"

This confirms they used **6 disaggregated ELISA values**, not the aggregated public data!

#### Threshold (INCONSISTENT in paper):
- **Line 51**: "non-specific (>4 flags)"  ← means >=5
- **Line 127**: "non-specific (>3 flags)" ← means >=4
- **Line 51**: "mildly non-specific (1-3 flags)"

**Best interpretation**: Use >=4 threshold (">3" in mathematical notation)

#### Flag Range:
- **0-7 total flags** (confirmed by Figure S1D histogram in supplementary)
  - 6 ELISA flags (one per antigen)
  - 1 "other assay" flag (likely aggregated self-interaction/chromatography/stability)

### Dataset Splits

**From line 127:**
- **Specific**: 0 flags
- **Mild**: 1-3 flags (EXCLUDED from test set)
- **Non-specific**: >=4 flags (">3")

**Test set**: Specific + Non-specific only (mild excluded)

---

## Novo's Likely Implementation

### Option 1: 6 ELISA + 1 Aggregated Other (0-7 range)
```
Flag calculation:
- ELISA flags: 6 individual antigens (0-6 range)
  - flag_cardiolipin (threshold 1.9 OD)
  - flag_klh (threshold 1.9 OD)
  - flag_lps (threshold 1.9 OD)
  - flag_ssdna (threshold 1.9 OD)
  - flag_dsdna (threshold 1.9 OD)
  - flag_insulin (threshold 1.9 OD)

- Other flag: 1 aggregated from self-interaction/chromatography/stability
  - ANY of these assays failing → 1 flag

Total: 0-7 flags
```

**Threshold**: >=4 for non-specific

**Expected splits**:
- Specific (0 flags): ~50-70 antibodies
- Mild (1-3 flags): EXCLUDED
- Non-specific (>=4 flags): ~15-35 antibodies
- **Test set**: ~65-100 antibodies (specific + non-specific)

---

### Option 2: 6 ELISA + 3 Individual Other (0-9 range)
```
Flag calculation:
- ELISA flags: 6 individual antigens (0-6 range)
- Other flags: 3 individual groups (0-3 range)
  - flag_self_interaction
  - flag_chromatography
  - flag_stability

Total: 0-9 flags
```

**Problem**: Figure S1D shows 0-7 range, NOT 0-9, so this is less likely.

---

## Our Implementation Plan

### Step 1: Calculate Flags (Both Options)

Create `scripts/conversion/convert_jain_with_private_elisa.py`:

```python
# Load private ELISA + public SD03
# Calculate flags for BOTH options:

# Option 1: 0-7 range (6 ELISA + 1 other)
df['elisa_flags'] = sum of 6 individual ELISA flags
df['other_flag'] = (self_interaction OR chromatography OR stability)
df['total_flags_0to7'] = elisa_flags + other_flag

# Option 2: 0-9 range (6 ELISA + 3 other)
df['total_flags_0to9'] = elisa_flags + flag_self + flag_chrom + flag_stab

# Apply threshold >=4
df['label_0to7'] = (total_flags_0to7 >= 4).astype(int)
df['label_0to9'] = (total_flags_0to9 >= 4).astype(int)

# Exclude mild (1-3 flags)
test_set_0to7 = df[(total_flags_0to7 == 0) | (total_flags_0to7 >= 4)]
test_set_0to9 = df[(total_flags_0to9 == 0) | (total_flags_0to9 >= 4)]
```

### Step 2: Compare Flag Distributions

Check which option gives:
- Histogram that matches Figure S1D (0-7 bins)
- Test set size close to 86 antibodies
- Balanced specific/non-specific split

### Step 3: Alignment & Validation

**Before implementing**, verify:
1. Flag distribution histogram matches Novo Figure S1D
2. Test set size is reasonable (~86 antibodies)
3. Thresholds are correctly implemented (>=4 means ">3")

### Step 4: Document & Implement

Once aligned:
- Update `scripts/conversion/convert_jain_with_private_elisa.py`
- Generate intermediate datasets
- Run model inference
- Compare to Novo's 68.6% accuracy

---

## Key Decisions Needed

### Decision 1: 0-7 vs 0-9 Flag Range?
- **Evidence for 0-7**: Figure S1D histogram shows 0-7 bins
- **Evidence for 0-9**: We have 6 ELISA + 3 other = 9 possible flags

**Recommendation**: Start with 0-7 (6 ELISA + 1 aggregated other) to match Figure S1D

### Decision 2: Threshold >=4 or >=5?
- **Line 51**: ">4" (means >=5)
- **Line 127**: ">3" (means >=4)

**Recommendation**: Use >=4 (">3") based on line 127 being more specific methodology section

### Decision 3: How to Aggregate "Other" Flag?
If using 0-7 range, how to combine self-interaction/chromatography/stability into 1 flag?

**Option A**: ANY assay fails → 1 flag
**Option B**: MAJORITY (>=2/3) fail → 1 flag
**Option C**: Use Jain Table 1 grouping rules

**Recommendation**: Start with Option A (ANY fails) as most stringent

---

## Success Criteria

✅ Flag distribution matches Novo Figure S1D
✅ Test set size ~86 antibodies (specific + non-specific)
✅ Balanced split (~60/40 or 70/30)
✅ Model accuracy ~68-70% (close to Novo's 68.6%)

---

## Next Steps

1. **Review & Align** on this plan before coding
2. **Implement** flag calculation script
3. **Generate** flag distributions for both options (0-7 and 0-9)
4. **Compare** to Novo Figure S1D
5. **Select** correct methodology
6. **Regenerate** all intermediate datasets
7. **Test** model performance

---

## Files to Audit/Revise

### Keep & Use:
- `Private_Jain2017_ELISA_indiv.xlsx` ✅
- `jain-pnas.1616408114.sd0X.xlsx` ✅
- `literature/markdown/Sakhnini_2025_*.md` ✅

### Delete (old reverse-engineering):
- `test_datasets/jain_flags_disaggregated.csv` ❌ (wrong thresholds)
- `docs/jain/CRITICAL_BUGS_FOUND.md` ❌
- `docs/jain/HYPERPARAMETER_SWEEP_PLAN.md` ❌
- All debug/investigation docs ❌

### Create Fresh:
- `scripts/conversion/convert_jain_with_private_elisa.py` 🆕
- `JAIN_REPLICATION_PLAN.md` (this file) 🆕

---

**Author**: Claude Code
**Reviewed by**: Ray Wu
**Status**: Awaiting alignment before implementation
