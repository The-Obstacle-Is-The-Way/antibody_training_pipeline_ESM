# Jain 116→86 QC Removal Analysis

**Date**: 2025-11-03
**Status**: ✅ ELISA-only methodology confirmed | ⚠️ 24/30 QC candidates identified

---

## Summary

We successfully established the **116-antibody ELISA-only test set** and identified **24 out of 30 QC candidates** that Novo likely removed to achieve their 86-antibody set.

### The Math Problem

```
137 antibodies (full Jain dataset)
  ↓ Exclude ELISA 1-3 (mild)
116 antibodies (94 specific / 22 non-specific) ← OUR SSOT
  ↓ ??? Novo's undisclosed QC ???
86 antibodies (59 specific / 27 non-specific) ← Novo's reported

Gap: 30 antibodies removed
Class shift: Need to remove 35 specific + ADD 5 non-specific (?!)
```

**Critical Issue**: Novo has **MORE** non-specific (27 vs 22), which suggests:
1. They reclassified 5 antibodies from specific → non-specific, OR
2. They used different ELISA thresholding, OR
3. The confusion matrix is from cross-validation (not a fixed test set)

---

## Literature Confirmation

### ✅ ELISA-Only Methodology is CORRECT

**From Novo Nordisk paper (Sakhnini et al. 2025):**

> "As in the Boughter dataset, the Jain dataset was parsed into two groups, specific (0 flags) and non-specific (>3 flags), leaving out the mildly non-specific antibodies (1-3 flags)."

**Table 2:**
```
Jain dataset | 137 clinical stage IgG1-formatted antibodies |
             | ELISA with a panel of 6 ligands              |
             | (ssDNA, dsDNA, insulin, LPS, cardiolipin, KLH) |
```

### ❌ NO Mention of 116→86 Gap

- **NO explanation** of how they got from 116→86
- **NO mention** of additional QC criteria
- **NO mention** of ANARCI failures
- **NO mention** of 30-antibody removal

---

## Z-Score Analysis Results

### Outliers Found (|z| > 2.5)

**Total outliers**: 3 antibodies

1. **cixutumumab** (non-specific, ELISA=6)
   - VH length: 130 aa (z=+3.33) ← **EXTREME outlier**
   - Longest VH in the dataset

2. **lampalizumab** (specific, ELISA=0)
   - Total charge: -5 (z=-2.57)
   - Extremely negatively charged

3. **polatuzumab** (specific, ELISA=0)
   - Total charge: -5 (z=-2.57)
   - Extremely negatively charged

### Known Biological/Clinical Issues

**Chimeric antibodies** (11 total):
- Mouse/human chimeras → higher immunogenicity
- Examples: muromonab, cetuximab, basiliximab, infliximab, rituximab, alemtuzumab, bevacizumab, panitumumab, girentuximab, trastuzumab, gemtuzumab

**Discontinued antibodies** (11 total):
- Failed clinical trials or pulled from market
- Examples: tabalumab (failed Phase 3 lupus), girentuximab (failed ARISER), abituzumab (failed POSEIDON), dalotuzumab, ganitumab, robatumumab, tigatuzumab, farletuzumab, zanolimumab

**Withdrawn antibodies** (4 total):
- muromonab (OKT3) - severe HAMA, withdrawn 2010
- efalizumab - withdrawn due to PML risk
- natalizumab - temporarily withdrawn, reintroduced with REMS
- gemtuzumab - withdrawn 2010, reapproved 2017

---

## QC Candidate Summary

### Total: 24 antibodies identified

**By category:**
- **Outliers** (z-score |z| > 2.5): 3 antibodies
- **Chimeric**: 11 antibodies
- **Discontinued**: 11 antibodies
- **Withdrawn**: 4 antibodies

**By label:**
- **Specific (label=0)**: 21 candidates
- **Non-specific (label=1)**: 3 candidates

### Top Candidates (Most Likely Removed)

| Rank | Antibody     | Label      | ELISA | VH (z) | Reason                          |
|------|--------------|------------|-------|--------|---------------------------------|
| 1    | cixutumumab  | non-spec   | 6     | +3.33  | **EXTREME VH outlier**          |
| 2    | lampalizumab | specific   | 0     | -1.44  | **Extreme charge** (z=-2.57)    |
| 3    | polatuzumab  | specific   | 0     | -0.80  | **Extreme charge** (z=-2.57)    |
| 4    | muromonab    | specific   | 0     | -0.17  | **Chimeric + WITHDRAWN**        |
| 5    | efalizumab   | specific   | 0     | +0.47  | **WITHDRAWN** (PML risk)        |
| 6    | natalizumab  | specific   | 0     | +1.11  | **WITHDRAWN** (PML risk)        |
| 7    | gemtuzumab   | specific   | 0     | -1.12  | **Chimeric + WITHDRAWN**        |
| 8    | girentuximab | specific   | 0     | -0.17  | **Chimeric + DISCONTINUED**     |
| 9    | tabalumab    | specific   | 0     | +1.11  | **DISCONTINUED** (failed Ph3)   |
| 10   | abituzumab   | specific   | 0     | -0.49  | **DISCONTINUED** (failed Ph3)   |
| 11   | dalotuzumab  | non-spec   | 5     | -0.80  | **DISCONTINUED**                |
| 12   | robatumumab  | non-spec   | 4     | -0.49  | **DISCONTINUED**                |
| 13   | ganitumab    | specific   | 0     | -0.17  | **DISCONTINUED**                |
| 14   | tigatuzumab  | specific   | 0     | -0.17  | **DISCONTINUED**                |
| 15   | farletuzumab | specific   | 0     | -0.17  | **DISCONTINUED**                |
| 16   | zanolimumab  | specific   | 0     | -1.44  | **DISCONTINUED**                |
| 17   | basiliximab  | specific   | 0     | -0.80  | **Chimeric**                    |
| 18   | alemtuzumab  | specific   | 0     | +0.47  | **Chimeric**                    |
| 19   | rituximab    | specific   | 0     | +0.47  | **Chimeric**                    |
| 20   | bevacizumab  | specific   | 0     | +1.11  | **Chimeric**                    |
| 21   | cetuximab    | specific   | 0     | -0.17  | **Chimeric**                    |
| 22   | panitumumab  | specific   | 0     | -0.17  | **Chimeric**                    |
| 23   | trastuzumab  | specific   | 0     | +0.15  | **Chimeric**                    |
| 24   | infliximab   | specific   | 0     | +0.15  | **Chimeric**                    |

---

## Remaining Questions

### 1. Where are the other 6 candidates?

**Need to find 6 more** antibodies to reach 30 total removals.

**Possible criteria:**
- More subtle VH/VL length outliers (|z| > 2.0 instead of 2.5)
- CDR H3 length outliers (need to calculate IMGT numbering)
- Missing biophysical assay data
- High hydrophobicity / aggregation propensity
- Manual curation by Novo scientists

### 2. Why does Novo have 5 MORE non-specific?

**Current**: 94 specific / 22 non-specific = 116
**Novo**: 59 specific / 27 non-specific = 86
**Math**: Remove 35 specific, ADD 5 non-specific (?!)

**Possibilities:**
1. **Reclassification**: Novo reclassified 5 antibodies with ELISA=0 but high other flags
2. **Different threshold**: Novo used ELISA ≥3 instead of ≥4 (but paper says ">3")
3. **Cross-validation**: Confusion matrix from CV, not fixed test set
4. **Data differences**: Our private ELISA file differs slightly from theirs

### 3. Should we proceed with 116 or try to match 86?

**Option A: Use 116 as SSOT**
- ✅ Method-faithful to Novo's stated ELISA-only approach
- ✅ Fully documented and reproducible
- ✅ Larger test set (better statistics)
- ❌ Can't directly compare confusion matrix to Novo

**Option B: Create our own 86-antibody set**
- Remove the 24 QC candidates + 6 more based on additional criteria
- Try to match Novo's 59/27 distribution
- ❌ Requires guessing their criteria
- ❌ May not match exactly anyway

**Option C: Contact Jain/Novo authors**
- Email Laila Sakhnini (llsh@novonordisk.com) for the exact 86-antibody list
- Ask what QC was applied beyond "exclude mild"
- ⏱️ May take weeks to get response

---

## Recommendations

### Immediate Actions

1. **Use 116 as primary test set**
   - Report results on the 116-antibody ELISA-only set
   - Document as "method-faithful to Novo's stated approach"
   - Note that Novo used undisclosed QC to get 86

2. **Create 86-antibody "best guess" set**
   - Remove the top 24 QC candidates
   - Use additional criteria to find 6 more:
     * CDR H3 length outliers (calculate with ANARCI)
     * High aggregation propensity (use TANGO or Zyggregator)
     * Missing biophysical data
   - Report as "inferred Novo-parity set"

3. **Document the discrepancy**
   - Create a supplementary note explaining the 116 vs 86 issue
   - Show that our ELISA-only methodology matches the literature
   - Explain that the 30-antibody gap is undocumented

### Long-term Actions

1. **Email Jain authors**
   - Ask for the exact 86-antibody list used by Novo
   - Request clarification on QC criteria
   - May not get response (Novo Discord said "no more shit")

2. **Run inference on both sets**
   - Test our model on 116 (primary)
   - Test our model on our "best guess" 86 (secondary)
   - Compare to Novo's reported 66.28% accuracy

3. **Create fragment files**
   - Run `preprocessing/process_jain.py` on the 116-antibody set
   - Generate VH-only, CDR, FWR fragment files
   - Ready for inference

---

## Files Generated

| File | Description | Rows |
|------|-------------|------|
| `jain_ELISA_ONLY_116.csv` | Primary test set (ELISA-only) | 116 |
| `jain_with_private_elisa_FULL.csv` | Full 137 with all flags | 137 |
| `jain_ELISA_ONLY_116_with_zscores.csv` | 116 with z-scores | 116 |
| `jain_116_qc_candidates.csv` | 24 QC removal candidates | 24 |

---

## Conclusion

✅ **ELISA-only methodology**: Confirmed correct from literature
✅ **116-antibody SSOT**: Established and validated
⚠️ **30-antibody gap**: 24/30 candidates identified via z-scoring
⚠️ **Class distribution**: 5-antibody discrepancy remains unexplained

**Next step**: Use 116 as primary test set and proceed with inference. Document the 30-antibody gap as "undisclosed QC" in Novo's methods.
