# Root Documentation Cleanup Plan

**Date:** 2025-11-05
**Branch:** clean-jain

---

## Problem: Documentation Slop

**9 MD files in root** - overlapping, contradictory, outdated:

```
BOUGHTER_JAIN_FINAL_RESOLUTION.md (12K)
COMPLETE_JAIN_MODEL_RESOLUTION.md (13K)
DATASETS_FINAL_SUMMARY.md (9.5K)
FINAL_RESOLUTION_AND_PATH_FORWARD.md (8.7K)
JAIN_CLEANUP_PLAN_REVISED.md (13K) ← KEEP THIS (cleanup plan)
JAIN_DATASETS_AUDIT_REPORT.md (11K)
JAIN_DATASET_COMPLETE_HISTORY.md (19K)
JAIN_EXPERIMENTS_DISCREPANCY.md (6K) ← NOW OBSOLETE (P5e-S2 works!)
JAIN_MODELS_DATASETS_COMPLETE_ANALYSIS.md (10K)

README.md (6.3K) ← KEEP
USAGE.md (8.4K) ← KEEP
```

**Issues:**
1. Multiple "FINAL" resolution docs (which is final??)
2. JAIN_EXPERIMENTS_DISCREPANCY.md says P5e-S2 is off by 1 (NO LONGER TRUE!)
3. Overlapping content across 7 different Jain docs
4. No single source of truth

---

## Solution: Consolidate to 1 Master Doc + 1 Cleanup Plan

### ✅ KEEP (2 files)

1. **JAIN_CLEANUP_PLAN_REVISED.md** ✅
   - Action plan for cleaning up CSV files
   - Executable steps
   - Keep as-is

2. **JAIN_COMPLETE_GUIDE.md** (NEW - create this)
   - Consolidates ALL accurate information
   - Single source of truth
   - Replaces 6 redundant docs

### 📦 ARCHIVE (7 files)

Move to `docs/archive/investigation_2025_11_05/`:

```
BOUGHTER_JAIN_FINAL_RESOLUTION.md
COMPLETE_JAIN_MODEL_RESOLUTION.md
DATASETS_FINAL_SUMMARY.md
FINAL_RESOLUTION_AND_PATH_FORWARD.md
JAIN_DATASETS_AUDIT_REPORT.md
JAIN_DATASET_COMPLETE_HISTORY.md
JAIN_EXPERIMENTS_DISCREPANCY.md (OBSOLETE)
JAIN_MODELS_DATASETS_COMPLETE_ANALYSIS.md
```

**Why archive, not delete:**
- Shows investigation process
- Historical record
- Might be useful for reference
- Just get them out of root

### ✅ KEEP (user-facing docs)

```
README.md (main repo docs)
USAGE.md (how to use the pipeline)
```

---

## Create: JAIN_COMPLETE_GUIDE.md

**Single source of truth with 4 sections:**

### 1. Quick Start (TL;DR)
- Which files to use for benchmarking
- Expected results
- Links to key datasets

### 2. Dataset Inventory
- Complete list of all Jain datasets
- What each file is for
- Which achieve Novo parity

### 3. Methodology Comparison
- OLD reverse-engineered (simple QC)
- P5e-S2 canonical (PSR-based)
- When to use each

### 4. Reproducibility Notes
- **IMPORTANT:** P5e-S2 has 1 borderline antibody (nimotuzumab)
- Probability ~0.5 can flip due to embedding nondeterminism
- Use stored predictions for exact reproducibility
- Both methods achieve parity in practice

---

## Execution Steps

### Step 1: Create Archive Directory

```bash
mkdir -p docs/archive/investigation_2025_11_05
```

### Step 2: Move Old Docs to Archive

```bash
mv BOUGHTER_JAIN_FINAL_RESOLUTION.md docs/archive/investigation_2025_11_05/
mv COMPLETE_JAIN_MODEL_RESOLUTION.md docs/archive/investigation_2025_11_05/
mv DATASETS_FINAL_SUMMARY.md docs/archive/investigation_2025_11_05/
mv FINAL_RESOLUTION_AND_PATH_FORWARD.md docs/archive/investigation_2025_11_05/
mv JAIN_DATASETS_AUDIT_REPORT.md docs/archive/investigation_2025_11_05/
mv JAIN_DATASET_COMPLETE_HISTORY.md docs/archive/investigation_2025_11_05/
mv JAIN_EXPERIMENTS_DISCREPANCY.md docs/archive/investigation_2025_11_05/
mv JAIN_MODELS_DATASETS_COMPLETE_ANALYSIS.md docs/archive/investigation_2025_11_05/
```

### Step 3: Create Archive README

```bash
cat > docs/archive/investigation_2025_11_05/README.md << 'EOF'
# Jain Dataset Investigation - Nov 5, 2025

This archive contains documentation from the investigation into Jain dataset
methodologies and Novo parity.

**Conclusion:** Both OLD reverse-engineered and P5e-S2 canonical methods achieve
[[40, 19], [10, 17]] parity with Novo Nordisk results.

See `JAIN_COMPLETE_GUIDE.md` in the repo root for the consolidated, accurate guide.

## Archived Files:
- BOUGHTER_JAIN_FINAL_RESOLUTION.md
- COMPLETE_JAIN_MODEL_RESOLUTION.md
- DATASETS_FINAL_SUMMARY.md
- FINAL_RESOLUTION_AND_PATH_FORWARD.md
- JAIN_DATASETS_AUDIT_REPORT.md
- JAIN_DATASET_COMPLETE_HISTORY.md
- JAIN_EXPERIMENTS_DISCREPANCY.md (OBSOLETE - P5e-S2 works!)
- JAIN_MODELS_DATASETS_COMPLETE_ANALYSIS.md

**Date archived:** 2025-11-05
EOF
```

### Step 4: Create JAIN_COMPLETE_GUIDE.md

(See next section for contents)

---

## After Cleanup: Root Directory

```
REPO ROOT/
├── README.md ✅ (main docs)
├── USAGE.md ✅ (how to use)
├── JAIN_COMPLETE_GUIDE.md ✅ (NEW - single source of truth)
├── JAIN_CLEANUP_PLAN_REVISED.md ✅ (CSV cleanup plan)
│
├── docs/
│   └── archive/
│       └── investigation_2025_11_05/
│           ├── README.md (explains what's archived)
│           └── [8 old investigation docs]
│
└── [rest of repo...]
```

**Result:**
- 11 root MD files → 4 root MD files
- Clear separation: guides vs plans vs archives
- One source of truth for Jain datasets

---

## Verification

```bash
# Count root MD files (should be 4)
ls *.md | wc -l

# Verify archive created
ls docs/archive/investigation_2025_11_05/

# Check new guide exists
ls -lh JAIN_COMPLETE_GUIDE.md
```

---

**Ready to execute? I'll create the consolidated guide next.**

