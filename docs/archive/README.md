# Documentation Archive

This directory contains historical documentation from the Novo Nordisk parity reverse engineering process (Nov 2-3, 2025).

---

## Why These Docs Are Archived

During the reverse engineering process, we explored **multiple hypotheses** before finding the exact method Novo Nordisk used. These documents represent:

- ❌ **Incorrect attempts** that were superseded
- 📚 **Historical progress** showing our journey
- ✅ **Key insights** that led to breakthroughs
- 📝 **Intermediate results** (like P5, which was close but not exact)

**All docs are preserved for provenance and reproducibility.**

---

## Archive Structure

### `failed_attempts/` ❌

**Documents based on incorrect reverse engineering assumptions:**

- `NOVO_PARITY_ANALYSIS.md` - 91-antibody approach (removed 5 by model confidence)
  - Claimed "66.28% accuracy, cell-for-cell match" ❌ WRONG
  - Actually 67.03% on 91 antibodies
  - Superseded by: P5e-S2

- `NOVO_ALIGNMENT_COMPLETE.md` - Pre-reverse-engineering assumptions
  - Superseded by: experiments/novo_parity/

- `NOVO_BENCHMARK_CONFUSION_MATRICES.md` - Old 91-antibody confusion matrices
  - Superseded by: P5e-S2

- `JAIN_FLAG_DISCREPANCY_INVESTIGATION.md` - Shows 54/32 split (wrong distribution)
  - Before we discovered the reclassification approach
  - Superseded by: P5e-S2

---

### `p5_close_attempt/` 📚

**P5 Result: Our first major breakthrough (2 cells off from exact match)**

- `JAIN_NOVO_PARITY_VALIDATION_REPORT.md`
  - Result: [[40, 19], [9, 18]] (Novo: [[40, 19], [10, 17]])
  - Perfect top row! (TN=40, FP=19)
  - Off by 1 on bottom row
  - Date: 2025-11-03
  - Value: Showed we were on the right track with PSR-based approach
  - Superseded by: P5e-S2 [[40, 19], [10, 17]] exact match

---

### `key_insights/` ✅

**Important discoveries that led to success:**

- `JAIN_BREAKTHROUGH_ANALYSIS.md` ✅ **CORRECT**
  - Mathematical proof: "To go from 22 → 27 non-specific by REMOVAL ALONE is IMPOSSIBLE"
  - This KEY insight led to the reclassification approach
  - Date: 2025-11-03
  - Status: Historically valuable, scientifically correct

---

### `preprocessing/` 📝

**Documentation of the 137→116 antibody QC step:**

- `JAIN_QC_REMOVALS_COMPLETE.md` - 8-antibody QC removal (VH length, missing VL)
  - Correct methodology for 137→116 step
  - Still valid, just not critical for main docs

- `JAIN_116_QC_ANALYSIS.md` - Analysis of 116-antibody starting point
  - Background on the dataset we started with

---

### `historical/` 📚

**Pre-parity training results and old analyses:**

- `TRAINING_RESULTS.md` - Training results with pre-parity Jain numbers
  - Shows 66.28% accuracy on OLD Jain split (before reverse engineering)
  - Superseded by: P5e-S2 exact match

---

## The Final Answer

**For the authoritative, final reverse engineering results, see:**

📁 **`experiments/novo_parity/`**

- `MISSION_ACCOMPLISHED.md` - Executive summary
- `EXACT_MATCH_FOUND.md` - Technical details
- `datasets/jain_86_p5e_s2.csv` - Final dataset

**Result**: [[40, 19], [10, 17]] ✅ EXACT MATCH (66.28% accuracy)

**Method**:
- Reclassification: 3 PSR >0.4 + eldelumab (extreme Tm) + infliximab (clinical)
- Removal: PSR primary, AC-SINS tiebreaker
- Distribution: 59 specific / 27 non-specific = 86 total

---

**Archived**: November 3, 2025
**Branch**: `novo-parity-exp-cleaned`
