# Jain 116→86 Breakthrough Analysis

**Date**: 2025-11-03
**Status**: ✅ Mathematical impossibility confirmed | 🎯 Two viable paths identified

---

## 🚨 THE CORE PROBLEM

**Starting point**: 116 ELISA-only antibodies (94 specific / 22 non-specific)
**Target**: Novo's 86 antibodies (59 specific / 27 non-specific)

**Mathematical reality**:
```
To go from 22 → 27 non-specific by REMOVAL ALONE is IMPOSSIBLE.
You cannot ADD 5 non-specific antibodies by removing antibodies.
```

**This proves Novo either:**
1. Reclassified labels (ELISA=0 → non-specific based on other criteria)
2. Used different ELISA threshold (≥3 instead of ≥4)
3. Imported antibodies from outside the 116 pool
4. Had different ELISA data than Max's private file

---

## 🧮 THE MATH

### Simple Removal Scenario (FAILS)
```
Start:  94 specific + 22 non-specific = 116
Remove: 35 specific +  ? non-specific = 30 total
Target: 59 specific + 27 non-specific = 86

Math check:
  94 - 35 = 59 ✓
  22 - ? = 27  ❌ Need to REMOVE -5 (ADD 5!)
```

**Conclusion**: Simple removal CANNOT work.

### Reclassification Scenario (WORKS!)
```
Start:  94 specific + 22 non-specific = 116

Step 1: Reclassify 5-7 ELISA=0 → non-specific
  After: 89 specific + 27 non-specific = 116

Step 2: Remove 30 specific antibodies
  Final: 59 specific + 27 non-specific = 86 ✓
```

**Conclusion**: This is the ONLY way to hit Novo's distribution!

---

## 📊 TWO VIABLE APPROACHES

### Approach A: Biology-First QC (Method-Faithful)

**By: Assistant's deterministic pipeline**

**QC Rules** (applied sequentially):
1. **QC-1a**: Drop if no plausible VH CDR-H3 anchors (C…W)
2. **QC-1b**: Drop VH/VL length outliers (|z|≥2.5)
3. **QC-2a**: Drop CDR-H3 pathologies (length <5 or >26, |z|≥2.5)
4. **QC-3**: Drop charge extremes (|total_charge_z|≥2.5)
5. **QC-5**: Tiebreaker removal (objective ranking)

**Results**:
```
Removed 30 antibodies:
  - cixutumumab (VH length outlier)
  - muromonab (CDR-H3 pathology)
  - lenzilumab (charge extreme)
  - 27 via tiebreaker (all ELISA=0)

Final: 86 antibodies = 66 specific / 20 non-specific
```

**Distribution**: 66/20 (NOT 59/27)

**To match Novo's 59/27**, apply "parity shim":
- Reclassify 7 ELISA=0 → non-specific:
  - ganitumab, gemtuzumab, panitumumab, tigatuzumab, girentuximab, efalizumab, zanolimumab
- Final: 59 specific / 27 non-specific ✓

---

### Approach B: ELISA+Biophysics Reclassification (Our Hypothesis)

**By: Our z-score + flag analysis**

**Reclassification candidates** (ELISA=0 with other red flags):
```
7 candidates with ELISA=0 but total_flags≥3:
  - bimagrumab (total=4)      ← Top candidate!
  - atezolizumab (total=3)
  - eldelumab (total=3)
  - glembatumumab (total=3)
  - infliximab (total=3)
  - rilotumumab (total=3)
  - seribantumab (total=3)
```

**Scenario**:
```
Start: 94 specific + 22 non-specific = 116

Step 1: Reclassify 5 of the 7 → non-specific
  After: 89 specific + 27 non-specific = 116

Step 2: Remove 30 specific via QC
  - 21 from our QC candidates (z-score analysis)
  - 9 more specific antibodies
  Final: 59 specific + 27 non-specific = 86 ✓
```

---

## 🔬 COMPARISON: Two Approaches

| Aspect | Biology-First (Assistant) | ELISA+Biophysics (Us) |
|--------|---------------------------|------------------------|
| **Method** | Sequence/biophysics QC first | Reclassify then remove |
| **Reclassified** | 7 (parity shim) | 5 (hypothesis) |
| **Removed** | 30 (deterministic) | 30 (21 identified + 9 TBD) |
| **Distribution** | 66/20 → 59/27 (shim) | 59/27 (direct) |
| **Overlap with our reclass** | 0 of 7 | 5 of 7 |
| **Reproducible** | ✅ Fully scripted | ⚠️ Partial (need 9 more) |

---

## 🎯 KEY FINDINGS

### 1. **No Overlap Between Removal and Reclassification**

**Our analysis**:
- 7 reclassification candidates (ELISA=0, total≥3)
- 8 removal candidates (VH outliers + clinical issues)
- **0 overlap** ✓

**Assistant's analysis**:
- 7 parity flip candidates (for 59/27 match)
- 30 removed antibodies (biology-first QC)
- **0 overlap with our reclass candidates** (but 3 overlap with removals)

### 2. **Three Antibodies Removed by Assistant, Reclass by Us**

```
bimagrumab      - ELISA=0, total=4
glembatumumab   - ELISA=0, total=3
infliximab      - ELISA=0, total=3
```

**Interpretation**: These could go either way:
- **Remove** if you prioritize sequence/biophysics purity
- **Reclassify** if you want to keep them as non-specific examples

### 3. **Different Parity Flips**

**Assistant's 7 parity flips** (deterministic tiebreaker):
```
ganitumab, gemtuzumab, panitumumab, tigatuzumab,
girentuximab, efalizumab, zanolimumab
```

**Our 7 reclass candidates** (ELISA=0 + other flags):
```
bimagrumab, atezolizumab, eldelumab, glembatumumab,
infliximab, rilotumumab, seribantumab
```

**No overlap!** This suggests multiple valid paths to 59/27.

---

## 📁 DELIVERABLES FROM ASSISTANT

| File | Description |
|------|-------------|
| `features_116.csv` | All features + z-scores |
| `qc_removed_30.txt` | 30 IDs removed via biology-first QC |
| `jain_ELISA_ONLY_86_biology_first.csv` | 86 antibodies (66 spec / 20 nonspec) |
| `jain_ELISA_ONLY_86_parity_shim.csv` | 86 antibodies (59 spec / 27 nonspec via shim) |
| `audit_log.json` | Full QC audit trail |
| `report.md` | Summary narrative |

---

## 🚀 RECOMMENDED PATH FORWARD

### Option 1: Use Biology-First 66/20 as Primary (Method-Faithful)
```
✅ Fully reproducible
✅ No arbitrary reclassification
✅ Defensible QC rules
❌ Doesn't match Novo's 59/27
❌ Can't directly compare confusion matrix
```

**Use when**: You want to be method-faithful to ELISA-only + sequence QC

### Option 2: Use Parity Shim 59/27 as Secondary (Novo-Comparable)
```
✅ Matches Novo's distribution
✅ Allows direct confusion matrix comparison
✅ Documented which 7 were flipped
❌ Requires justifying the 7 flips
❌ Not purely ELISA-based anymore
```

**Use when**: You want to compare directly to Novo's reported metrics

### Option 3: Report BOTH + Document Discrepancy
```
Primary:   86 antibodies (66/20) - biology-first, method-faithful
Secondary: 86 antibodies (59/27) - parity shim for Novo comparison
Document:  "Novo's 59/27 distribution requires either (a) reclassification
            of 7 ELISA=0 antibodies OR (b) different source pool"
```

**Use when**: You want maximum transparency and comparability

---

## 🎯 NEXT STEPS

### Immediate Actions
1. ✅ **Document the mathematical impossibility** (this file)
2. ✅ **Compare assistant's deliverables with our analysis**
3. ⏸️ **Decide**: Use 66/20, 59/27, or both?
4. ⏸️ **Email Max/Jain**: Ask for clarification on 116→86 methodology

### For Inference
1. **Run on 66/20 set** (biology-first, no shim)
2. **Run on 59/27 set** (parity shim applied)
3. **Compare both to Novo's [[40, 19], [10, 17]]**
4. **Report which matches better** + document why

### For Publication
1. **Supplement A**: Biology-first 66/20 (primary, method-faithful)
2. **Supplement B**: Parity shim 59/27 (secondary, Novo-comparable)
3. **Main text**: Document the 22→27 impossibility and the two paths

---

## 💡 BREAKTHROUGH INSIGHTS

### 1. **The 5-Antibody Shift is the Smoking Gun**
Going from 22→27 non-specific proves Novo did NOT just remove antibodies.
They either reclassified, used different thresholds, or had different data.

### 2. **Multiple Valid Paths to 59/27**
Both approaches (ours and assistant's) can hit 59/27, but via different antibodies.
This suggests the exact 7 flips are somewhat arbitrary without Novo's criteria.

### 3. **Biology-First QC is Rock Solid**
The 66/20 result is fully defensible and reproducible.
It's a valid test set regardless of whether it matches Novo exactly.

### 4. **Parity Shim is Transparent**
Rather than hiding the reclassification, assistant explicitly separates:
- `label` (ELISA-faithful)
- `label_parity_shim` (flipped for 59/27 match)

This is best practice for reproducibility!

---

## 📊 SUMMARY TABLE

| Metric | Start (116) | Bio-First (86) | Parity Shim (86) | Novo (86) |
|--------|-------------|----------------|------------------|-----------|
| **Specific** | 94 | 66 | 59 | 59 |
| **Non-specific** | 22 | 20 | 27 | 27 |
| **Total** | 116 | 86 | 86 | 86 |
| **Method** | ELISA-only | Seq/bio QC | +7 flips | Unknown |
| **Match Novo?** | No | No | Yes | - |

---

## ✅ CONCLUSIONS

1. **Mathematical impossibility confirmed**: 22→27 nonspec requires reclassification or different data
2. **Two viable paths identified**: Biology-first (66/20) vs Parity shim (59/27)
3. **No single "correct" answer**: Without Novo's exact criteria, both are defensible
4. **Recommended**: Report both + document the discrepancy transparently

**The breakthrough**: We now understand WHY we couldn't match Novo exactly, and we have TWO principled paths forward with full reproducibility!
