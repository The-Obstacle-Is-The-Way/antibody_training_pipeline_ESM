# Documentation Cleanup Plan

**Date**: November 3, 2025
**Branch**: `novo-parity-exp-cleaned`
**Goal**: Archive outdated/incorrect docs from failed reverse-engineering attempts

---

## FINAL CATEGORIZATION

### ✅ KEEP IN docs/ (Core Technical Docs)

1. **excel_to_csv_conversion_methods.md** - Data processing methods
2. **FIXES_APPLIED.md** - Bug fix documentation
3. **MPS_MEMORY_LEAK_FIX.md** - Technical fix
4. **TRAINING_RESULTS.md** - Model training results
5. **TRAINING_SETUP_STATUS.md** - Training setup
6. **METHODOLOGY_AND_DIVERGENCES.md** - Overall methodology comparison
7. **ASSAY_SPECIFIC_THRESHOLDS.md** - ELISA vs PSR thresholds
8. **DOCS_AUDIT_STATUS.md** - This audit (NEW)
9. **CLEANUP_PLAN.md** - This plan (NEW)

---

### 📦 ARCHIVE (Move to docs/archive/)

#### archive/failed_attempts/ (Incorrect Reverse Engineering)

**NOVO_PARITY_ANALYSIS.md**
- ❌ WRONG: Claims "66.28% accuracy, cell-for-cell confusion matrix match"
- Reality: Was 67.03% on 91 antibodies, removed 5 by model confidence
- Date: 2025-11-02
- Superseded by: P5e-S2 exact match

**NOVO_ALIGNMENT_COMPLETE.md**
- ⚠️ OUTDATED: Pre-reverse-engineering assumptions
- Date: Unknown
- Superseded by: experiments/novo_parity/

**NOVO_BENCHMARK_CONFUSION_MATRICES.md**
- ⚠️ OUTDATED: Based on old 91-antibody approach
- Date: Unknown
- Superseded by: P5e-S2

**JAIN_FLAG_DISCREPANCY_INVESTIGATION.md**
- ⚠️ OUTDATED: Shows confusion matrix with 54 specific / 32 non-specific
- Reality: Was before we figured out the reclassification approach
- Date: 2025-11-03
- Superseded by: P5e-S2

#### archive/p5_close_attempt/ (Near Miss - 2 Cells Off)

**JAIN_NOVO_PARITY_VALIDATION_REPORT.md**
- 📚 HISTORICAL: Documents P5 result [[40,19],[9,18]]
- Claims "✅ PARITY ACHIEVED" but was 2 cells off
- Date: 2025-11-03
- Value: Shows our first major breakthrough (perfect top row)
- Superseded by: P5e-S2 [[40,19],[10,17]] exact match

#### archive/key_insights/ (Correct Historical Insights)

**JAIN_BREAKTHROUGH_ANALYSIS.md**
- ✅ CORRECT: Mathematical proof that simple removal is impossible
- Quote: "To go from 22 → 27 non-specific by REMOVAL ALONE is IMPOSSIBLE"
- Date: 2025-11-03
- Value: KEY insight that led to reclassification approach
- Status: Keep for historical value

#### archive/preprocessing/ (137→116 Step Documentation)

**JAIN_QC_REMOVALS_COMPLETE.md**
- ✅ CORRECT: Documents 8-antibody QC removal (VH length, missing VL)
- Date: Unknown
- Value: Correct methodology for 137→116 preprocessing
- Status: Valid but not critical for main docs

**JAIN_116_QC_ANALYSIS.md**
- 📚 HISTORICAL: Analysis of 116-antibody starting point
- Date: Unknown
- Value: Background on starting dataset
- Status: Archive for reference

#### archive/needs_review/ (Uncertain Status)

**COMPLETE_VALIDATION_RESULTS.md**
- 🤔 UNKNOWN: Need to check what validation this documents
- Action: Read and categorize

**BENCHMARK_TEST_RESULTS.md**
- 🤔 UNKNOWN: Need to check which benchmark
- Action: Read and categorize

**JAIN_REPLICATION_PLAN.md**
- 🤔 UNKNOWN: Need to check what approach this documents
- Action: Read and categorize

---

## DELETION CANDIDATES

**NOVO_PARITY_EXPERIMENTS.md** (27K)
- **Status**: DUPLICATE
- **Why**: Already exists in `experiments/novo_parity/NOVO_PARITY_EXPERIMENTS.md`
- **Action**: DELETE (keep only in experiments/)

---

## EXECUTION PLAN

### Step 1: Create Archive Structure
```bash
mkdir -p docs/archive/{failed_attempts,p5_close_attempt,key_insights,preprocessing,needs_review}
```

### Step 2: Create Archive README
Create `docs/archive/README.md` explaining:
- What's in each folder
- Why docs were archived
- How to find the correct/final docs

### Step 3: Move Files (Preserve Git History)
```bash
# Failed attempts
git mv docs/NOVO_PARITY_ANALYSIS.md docs/archive/failed_attempts/
git mv docs/NOVO_ALIGNMENT_COMPLETE.md docs/archive/failed_attempts/
git mv docs/NOVO_BENCHMARK_CONFUSION_MATRICES.md docs/archive/failed_attempts/
git mv docs/JAIN_FLAG_DISCREPANCY_INVESTIGATION.md docs/archive/failed_attempts/

# P5 close attempt
git mv docs/JAIN_NOVO_PARITY_VALIDATION_REPORT.md docs/archive/p5_close_attempt/

# Key insights
git mv docs/JAIN_BREAKTHROUGH_ANALYSIS.md docs/archive/key_insights/

# Preprocessing
git mv docs/JAIN_QC_REMOVALS_COMPLETE.md docs/archive/preprocessing/
git mv docs/JAIN_116_QC_ANALYSIS.md docs/archive/preprocessing/

# Needs review
git mv docs/COMPLETE_VALIDATION_RESULTS.md docs/archive/needs_review/
git mv docs/BENCHMARK_TEST_RESULTS.md docs/archive/needs_review/
git mv docs/JAIN_REPLICATION_PLAN.md docs/archive/needs_review/
```

### Step 4: Delete Duplicate
```bash
git rm docs/NOVO_PARITY_EXPERIMENTS.md
```

### Step 5: Create Main README
Create `docs/README.md` explaining:
- What docs are here
- Where to find reverse engineering results (experiments/novo_parity/)
- What's in archive/

---

## FINAL docs/ STRUCTURE

```
docs/
├── README.md (NEW - points to correct docs)
├── DOCS_AUDIT_STATUS.md (NEW - this audit)
├── CLEANUP_PLAN.md (NEW - this plan)
├── excel_to_csv_conversion_methods.md ✅
├── FIXES_APPLIED.md ✅
├── MPS_MEMORY_LEAK_FIX.md ✅
├── TRAINING_RESULTS.md ✅
├── TRAINING_SETUP_STATUS.md ✅
├── METHODOLOGY_AND_DIVERGENCES.md ✅
├── ASSAY_SPECIFIC_THRESHOLDS.md ✅
├── archive/
│   ├── README.md (explains archive)
│   ├── failed_attempts/
│   │   ├── NOVO_PARITY_ANALYSIS.md
│   │   ├── NOVO_ALIGNMENT_COMPLETE.md
│   │   ├── NOVO_BENCHMARK_CONFUSION_MATRICES.md
│   │   └── JAIN_FLAG_DISCREPANCY_INVESTIGATION.md
│   ├── p5_close_attempt/
│   │   └── JAIN_NOVO_PARITY_VALIDATION_REPORT.md
│   ├── key_insights/
│   │   └── JAIN_BREAKTHROUGH_ANALYSIS.md ✅
│   ├── preprocessing/
│   │   ├── JAIN_QC_REMOVALS_COMPLETE.md
│   │   └── JAIN_116_QC_ANALYSIS.md
│   └── needs_review/
│       ├── COMPLETE_VALIDATION_RESULTS.md
│       ├── BENCHMARK_TEST_RESULTS.md
│       └── JAIN_REPLICATION_PLAN.md
├── boughter/ (existing)
├── harvey/ (existing)
├── investigation/ (existing)
├── jain/ (existing)
└── shehata/ (existing)
```

---

## READY TO EXECUTE?

**Confirm cleanup plan looks good, then we'll:**
1. Create archive structure
2. Create README files
3. Move files with git mv
4. Delete duplicate
5. Commit with clear message

---

**Status**: PLAN READY
**Next**: Execute cleanup
