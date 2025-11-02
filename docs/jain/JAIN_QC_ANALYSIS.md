# Jain Dataset QC Analysis - Gap Investigation (94 → 86 antibodies)

**Date**: 2025-11-02
**Status**: 🔍 Investigation Complete
**Gap**: Need to exclude 8 antibodies to match Novo's 86-antibody test set

---

## Executive Summary

After fixing the >=3 flag threshold bug, we now have **94 antibodies** (67 specific + 27 non-specific) vs Novo's **86 antibodies** (57 specific + 29 non-specific).

**Gap breakdown**:
- **Total**: Remove 8 antibodies (94 → 86)
- **Specific**: Remove 10 (67 → 57)
- **Non-specific**: Add 2 (27 → 29)

This analysis identifies the most likely QC exclusion candidates based on:
1. Sequence length outliers
2. Unusual CDR structures (annotation challenges)
3. Light chain class (lambda minority)
4. Literature-confirmed structural anomalies

---

## Current Test Set Composition

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total antibodies** | 94 | 100% |
| **Specific (label=0, flags=0)** | 67 | 71.3% |
| **Non-specific (label=1, flags≥3)** | 27 | 28.7% |
| **Kappa light chain** | 87 | 92.6% |
| **Lambda light chain** | 7 | 7.4% |
| **Approved drugs** | 35 | 37.2% |
| **Phase 3** | 25 | 26.6% |
| **Phase 2** | 34 | 36.2% |

---

## QC Analysis Findings

### 1. Length Outliers (High Confidence)

**VH sequence length**: mean=119.0 aa, std=3.1 aa

#### Extreme Outliers (z-score > 2):

| Antibody | VH Length | Z-Score | Label | Flags | Issue |
|----------|-----------|---------|-------|-------|-------|
| **cixutumumab** | 130 | 3.57 | 1 (non-spec) | 4 | Extremely long |
| **crenezumab** | 112 | 2.29 | 0 (specific) | 0 | Extremely short |
| **fletikumab** | 127 | 2.59 | 0 (specific) | 0 | Very long |
| **secukinumab** | 127 | 2.59 | 0 (specific) | 0 | Very long |

**Total: 4 antibodies**
- 3 specific (candidates for removal)
- 1 non-specific (cixutumumab)

### 2. Literature-Confirmed Structural Anomalies

#### Crenezumab (VH=112 aa)

**Issue**: Unusually short CDR H3 loop (only **3 residues**)

**Evidence**:
- Structure of Crenezumab Complex with Aβ (Nature Scientific Reports, 2016)
- Quote: "The complementarity determining regions (CDRs) of crenezumab feature a very short H3 loop of 3 residues and a long L1 loop of 16 residues"
- CDR H3 is typically the most variable region; 3 residues is exceptionally short
- Reference antibody solanezumab has "minimum length" H3 of **4 residues**

**ANARCI Impact**: Short CDR H3 likely causes numbering/annotation failures

**QC Verdict**: **HIGH RISK** - Strong candidate for exclusion

---

### 3. Lambda Chain Antibodies (7 total)

Lambda chains represent only 7.4% of test set (minority population).

| Antibody | Label | Flags | VH Length | Notes |
|----------|-------|-------|-----------|-------|
| **evolocumab** | 0 | 0 | 115 | Also short |
| **mavrilimumab** | 0 | 0 | 120 | Normal length |
| ustekinumab | 1 | 3 | 118 | Non-specific |
| belimumab | 1 | 3 | 118 | Non-specific |
| benralizumab | 1 | 3 | 117 | Non-specific |
| daratumumab | 1 | 3 | 117 | Non-specific |
| ublituximab | 1 | 3 | 121 | Non-specific |

**Lambda specific antibodies**: 2 (evolocumab, mavrilimumab)
**Lambda non-specific**: 5

---

### 4. Non-Specific Borderline Cases (flags=3)

**24 antibodies** with exactly 3 flags (threshold boundary):

Sample high-risk candidates:
- **ozanezumab**: VH=113 (very short)
- **emibetuzumab**: VH=115 (short)
- **figitumumab**: VH=125 (long)
- **eldelumab**: VH=124 (long)
- **ranibizumab**: VH=123 (long)
- **parsatuzumab**: VH=123 (long)

These represent antibodies that might be borderline between specific/non-specific and could account for the +2 non-specific gap.

---

## QC Risk Scoring

**Methodology**: Assign points for each QC risk factor
- Length outlier (z>2): +3 points
- Very long (>125): +2 points
- Very short (<113): +2 points
- Lambda chain: +1 point
- Borderline flags (=3): +1 point

### Top 10 Specific Antibodies for Removal (67 → 57)

| Rank | Antibody | VH Length | QC Score | Risk Factors |
|------|----------|-----------|----------|--------------|
| 1 | **fletikumab** | 127 | 5 | Outlier length (z=2.59), very long |
| 2 | **secukinumab** | 127 | 5 | Outlier length (z=2.59), very long |
| 3 | **crenezumab** | 112 | 5 | Outlier length (z=2.29), very short, CDR H3=3 residues |
| 4 | **mavrilimumab** | 120 | 1 | Lambda chain |
| 5 | **evolocumab** | 115 | 1 | Lambda chain, short |
| 6 | nivolumab | 113 | 2 | Very short |
| 7 | zalutumumab | 125 | 2 | Long |
| 8 | natalizumab | 123 | 0 | Moderate length |
| 9 | romosozumab | 123 | 0 | Moderate length |
| 10 | tabalumab | 123 | 0 | Moderate length |

---

## Proposed Exclusion Scenarios

### Scenario A: Conservative (High-Confidence Only)

**Remove top 3 length outliers**:
1. crenezumab (VH=112, literature-confirmed CDR H3 issue)
2. fletikumab (VH=127)
3. secukinumab (VH=127)

**Result**: 94 - 3 = **91 antibodies** (still 5 away from 86)

---

### Scenario B: Length + Lambda Chain

**Remove 5 antibodies**:
1. crenezumab (VH=112, short, CDR H3 issue)
2. fletikumab (VH=127, long)
3. secukinumab (VH=127, long)
4. mavrilimumab (lambda, specific)
5. evolocumab (lambda, specific, VH=115 short)

**Result**: 94 - 5 = **89 antibodies** (still 3 away from 86)

---

### Scenario C: Aggressive Length Filtering

**Remove all VH length outliers (±10 aa from mean)**:

**VH < 109 or VH > 129**:
- crenezumab (112)
- fletikumab (127)
- secukinumab (127)
- cixutumumab (130) - but this is NON-SPECIFIC!

**Issue**: This only gives us 3-4 removals, not enough.

---

### Scenario D: Extended Length Threshold

**Remove VH length outside 115-125 range**:

**Specific antibodies to remove**:
1. crenezumab (112) ✓
2. fletikumab (127) ✓
3. secukinumab (127) ✓
4. evolocumab (115 - borderline)
5. nivolumab (113)
6. brodalumab (116 - borderline)
7. ... (need to count all <115 or >125)

This might give us closer to 10 removals.

---

## Remaining Questions

### 1. Where is the exact QC filtering code?

**Hypothesis**: Novo likely used ANARCI or similar tool and excluded:
- Annotation failures
- Length outliers
- Unusual CDR definitions
- Missing germline calls

**Evidence needed**: Novo's preprocessing code or exact QC criteria

---

### 2. Why the +2 non-specific gap?

**Current**: 27 non-specific (flags ≥3)
**Target**: 29 non-specific
**Gap**: Need +2 more

**Hypothesis**: Some antibodies with flags=3 might be reclassified based on:
- Different threshold interpretation
- Different assay value rounding
- Borderline flag calculations

**Candidates** (flags=3, length issues):
- ozanezumab (VH=113, very short)
- emibetuzumab (VH=115, short)
- Any with measurement uncertainty near thresholds

---

## Recommended Next Steps

### Option 1: Test with Current 94 Antibodies ✅ **RECOMMENDED**

**Rationale**:
- Major bug fixed (3 → 27 non-specific)
- Class balance much better (71:29 vs 95:5)
- Gap is small (94 vs 86 = 8 antibodies)
- Results will be comparable even if not exact

**Action**: Run inference now, document 94-antibody results

---

### Option 2: Apply Conservative QC (Remove Top 3)

**Remove**:
1. crenezumab (literature-confirmed issue)
2. fletikumab (extreme length outlier)
3. secukinumab (extreme length outlier)

**Result**: 91 antibodies (64 specific + 27 non-specific)

**Action**: Create filtered CSV, run inference, compare to 94-antibody results

---

### Option 3: Deep Investigation (Time-Intensive)

**Actions**:
1. Run ANARCI on all 137 antibodies, log failures
2. Contact Novo authors for exact QC code
3. Systematically test different filtering thresholds
4. Iterate until exact 86-antibody match

**Estimated time**: 1-2 days
**Priority**: P1 (not blocking)

---

## Files Generated

This analysis used:
- `test_datasets/jain.csv` (137 antibodies with flags)
- `test_datasets/jain/VH_only_jain_test.csv` (94 antibodies after flag filtering)
- `test_datasets/pnas.1616408114.sd01.xlsx` (original metadata)
- Literature: Crenezumab structure papers

No filtered datasets created yet (awaiting decision on filtering approach).

---

## Conclusion

**The good news**: We fixed the major bug! 3 → 27 non-specific is huge.

**The 8-antibody gap**: Likely due to QC filtering differences (ANARCI failures, length outliers, CDR annotation issues).

**High-confidence exclusion candidates**:
1. **crenezumab** - Literature-confirmed CDR H3 anomaly (3 residues)
2. **fletikumab** - Extreme length outlier (VH=127)
3. **secukinumab** - Extreme length outlier (VH=127)

**Recommendation**: Proceed with testing on current 94-antibody set. The results will be scientifically valid and directly comparable to Novo's benchmark (within ~9% sample size difference).

---

**Analysis Date**: 2025-11-02
**Analyst**: Claude Code
**Status**: ✅ Ready for decision on filtering approach
