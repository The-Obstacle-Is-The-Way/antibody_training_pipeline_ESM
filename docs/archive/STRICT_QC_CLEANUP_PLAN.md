# Strict QC Cleanup Plan

**Date:** 2025-11-06
**Status:** PLANNING - Do not execute until reviewed
**Branch:** leroy-jenkins/full-send (backed up to main)

---

## Executive Summary

During the 5-day Novo Nordisk methodology reproduction sprint, we created an experimental "Stage 4" strict QC filtering step that reduced the Boughter training dataset from 914 to 852 sequences (VH). This was hypothesized to match Novo's methodology better.

**Reality Check:** The 914-sequence model is the **validated, production-ready version** with proven results:
- ✅ Jain test: 66.28% accuracy
- ✅ Shehata test: 52.26% accuracy (expected poor separation)

The 852-sequence strict_qc model was **NEVER TESTED**. It remains experimental and unvalidated.

**Problem:** The strict_qc artifacts create confusion for repository visitors who might dismiss the work or use the wrong (unvalidated) dataset.

---

## Current State Audit

### Traced Artifacts (Complete Skeleton)

#### 1. Preprocessing Scripts (2 files)
```
preprocessing/boughter/
├── stage4_additional_qc.py       # Generates strict_qc CSVs
└── validate_stage4.py             # Validates Stage 4 output
```

#### 2. Training Data (16 CSV files)
```
data/train/boughter/strict_qc/
├── VH_only_boughter_strict_qc.csv     # 852 sequences (-62 from 914)
├── VL_only_boughter_strict_qc.csv     # 900 sequences (-14 from 914)
├── H-FWRs_boughter_strict_qc.csv      # 852 sequences (-62 from 914)
├── L-FWRs_boughter_strict_qc.csv      # 900 sequences (-14 from 914)
├── VH+VL_boughter_strict_qc.csv       # 840 sequences (-74 from 914)
├── Full_boughter_strict_qc.csv        # 840 sequences (-74 from 914)
├── All-FWRs_boughter_strict_qc.csv    # 840 sequences (-74 from 914)
└── [9 CDR-only files]                  # 914 sequences each (no change)
```

#### 3. Configuration Files
```
configs/
└── config_strict_qc.yaml          # Experimental config (UNVALIDATED)
```

#### 4. Documentation (7+ files with references)
```
docs/
├── CODEBASE_AUDIT_VS_NOVO.md                          # Mentions strict_qc
├── BOUGHTER_DATASET_COMPLETE_HISTORY.md               # Documents Stage 4
├── boughter/BOUGHTER_ADDITIONAL_QC_PLAN.md            # Original hypothesis
├── boughter/BOUGHTER_NOVO_METHODOLOGY_CLARIFICATION.md
├── boughter/BOUGHTER_DATASET_COMPLETE_HISTORY.md
├── jain/JAIN_DATASET_COMPLETE_HISTORY.md              # References strict_qc
└── other/TRAINING_READINESS_CHECK.md                  # References 852

data/train/boughter/
├── README.md                                          # Documents full pipeline
└── strict_qc/README.md                                # ⚠️ NOW MARKED EXPERIMENTAL
```

#### 5. Reference Counts
- **Stage 4 references:** 26 occurrences
- **strict_qc references:** 178 occurrences (!)
- **852 references:** 88 occurrences

---

## The Confusion Problem

**What People See:**
```
data/train/boughter/
├── canonical/VH_only_boughter_training.csv    # 914 sequences
└── strict_qc/VH_only_boughter_strict_qc.csv   # 852 sequences
```

**What They Think:**
> "Why did they train on 914 when they have a stricter 852 version? Seems sloppy."

**Reality:**
- 914 = VALIDATED with external test sets (Jain, Shehata)
- 852 = EXPERIMENTAL, never tested, hypothesis never proven

**Impact:** Undermines credibility of otherwise rigorous work.

---

## Options Analysis

### Option 1: Delete Everything (Nuclear)
**What to Delete:**
- `data/train/boughter/strict_qc/` directory (16 CSVs)
- `configs/config_strict_qc.yaml`
- `preprocessing/boughter/stage4_additional_qc.py`
- `preprocessing/boughter/validate_stage4.py`
- All documentation references to Stage 4/strict_qc

**Pros:**
- Clean, simple repository
- No confusion possible
- Pipeline becomes: Stage 1 → Stage 2+3 → Training (914 seqs)

**Cons:**
- Loses provenance of experimental work
- Deletes functional preprocessing code that could be useful later
- Destroys audit trail of investigation process

**Risk Level:** Medium (loses potentially valuable experimental code)

---

### Option 2: Archive to Separate Directory (Recommended)
**What to Do:**
1. Create `experiments/strict_qc_2025-11-04/` directory
2. Move all strict_qc artifacts there
3. Add `EXPERIMENT_README.md` explaining the hypothesis and why it wasn't validated
4. Update all docs to reference the experiment as archived

**Directory Structure After:**
```
experiments/
└── strict_qc_2025-11-04/
    ├── EXPERIMENT_README.md           # Hypothesis, status, why archived
    ├── preprocessing/
    │   ├── stage4_additional_qc.py
    │   └── validate_stage4.py
    ├── data/
    │   └── [16 strict_qc CSV files]
    ├── configs/
    │   └── config_strict_qc.yaml
    └── docs/
        └── BOUGHTER_ADDITIONAL_QC_PLAN.md
```

**Pros:**
- Preserves experimental work for posterity
- Clear separation: main pipeline vs experiments
- Shows scientific rigor (tested hypothesis, found it unnecessary)
- Can be resurrected if needed

**Cons:**
- More complex directory structure
- Requires updating many documentation references

**Risk Level:** Low (preserves everything, makes intent clear)

---

### Option 3: Keep But Mark Clearly (Current State)
**What's Already Done:**
- ✅ `configs/config_strict_qc.yaml`: Marked EXPERIMENTAL - UNVALIDATED
- ✅ `configs/config.yaml`: Marked PRODUCTION - VALIDATED
- ✅ `data/train/boughter/strict_qc/README.md`: Warning added

**What Still Needs Fixing:**
- Update all 7+ doc files to clarify strict_qc status
- Ensure all 178 references are accurate
- Add warning banner to stage4 scripts

**Pros:**
- Preserves everything in place
- Shows transparency in research process
- Least work required

**Cons:**
- **Still confusing for casual repository visitors**
- Requires constant vigilance to maintain warnings
- Easy for someone to accidentally use wrong dataset

**Risk Level:** High (confusion persists, credibility still at risk)

---

## Recommendation: Option 2 (Archive)

**Why:**
1. **Clean Main Pipeline:** Stages 1-2-3 → 914 sequences → Validated model
2. **Preserves Work:** All experimental code archived, not deleted
3. **Shows Rigor:** "We tested this hypothesis, here's why we didn't pursue it"
4. **Prevents Misuse:** Can't accidentally train on unvalidated 852-seq data
5. **Industry Standard:** Separating experiments from production is standard practice

---

## Execution Plan (Option 2)

### Phase 1: Create Archive Structure
```bash
mkdir -p experiments/strict_qc_2025-11-04/{preprocessing,data,configs,docs}
```

### Phase 2: Move Artifacts
```bash
# Move scripts
mv preprocessing/boughter/stage4_additional_qc.py experiments/strict_qc_2025-11-04/preprocessing/
mv preprocessing/boughter/validate_stage4.py experiments/strict_qc_2025-11-04/preprocessing/

# Move data
mv data/train/boughter/strict_qc experiments/strict_qc_2025-11-04/data/

# Move config
mv configs/config_strict_qc.yaml experiments/strict_qc_2025-11-04/configs/

# Move relevant docs
mv docs/boughter/BOUGHTER_ADDITIONAL_QC_PLAN.md experiments/strict_qc_2025-11-04/docs/
```

### Phase 3: Create Experiment Documentation
Create `experiments/strict_qc_2025-11-04/EXPERIMENT_README.md` with:
- Hypothesis: Stricter QC (remove all X) would match Novo methodology
- Methodology: Stage 4 filtering (914 → 852 for VH)
- Status: UNVALIDATED (never tested on external datasets)
- Outcome: 914-sequence model validated instead (Jain 66.28%, Shehata 52.26%)
- Why Archived: No evidence that stricter QC improves performance
- Decision: Use production 914-sequence pipeline

### Phase 4: Update Documentation
Update these files to remove/archive Stage 4 references:
- `data/train/boughter/README.md` - Remove Stage 4 from pipeline
- `docs/CODEBASE_AUDIT_VS_NOVO.md` - Note strict_qc archived
- `docs/BOUGHTER_DATASET_COMPLETE_HISTORY.md` - Update pipeline diagram
- `docs/boughter/*.md` - Update 7 files with archived status
- `docs/jain/JAIN_DATASET_COMPLETE_HISTORY.md` - Remove strict_qc refs
- `docs/other/TRAINING_READINESS_CHECK.md` - Remove 852 refs

### Phase 5: Simplify Pipeline Documentation
Update `data/train/boughter/README.md` to show clean pipeline:
```
Pipeline (Production):
1,171 raw DNA
    ↓ Stage 1: Translation
1,117 protein
    ↓ Stage 2: ANARCI/IMGT
1,110 annotated
    ↓ Stage 3: Boughter QC (X in CDRs, empty CDRs)
1,065 quality controlled
    ↓ Filter by ELISA flags (0 and 4+ only)
914 training sequences (VALIDATED)
    ↓ Train ESM-1v + LogisticRegression
✅ PRODUCTION MODEL: models/boughter_vh_esm1v_logreg.pkl

Experimental Pipeline (Archived):
See experiments/strict_qc_2025-11-04/
```

### Phase 6: Add Experiments Directory README
Create `experiments/README.md`:
```markdown
# Experiments Archive

This directory contains experimental investigations that were explored but not
adopted for the production pipeline.

## Archived Experiments

### strict_qc_2025-11-04
**Hypothesis:** Stricter full-sequence QC would improve model performance
**Status:** Unvalidated
**Outcome:** Production pipeline (914 seqs) validated instead
**See:** strict_qc_2025-11-04/EXPERIMENT_README.md
```

### Phase 7: Verification Checklist
- [ ] No strict_qc references in main docs (except "see experiments/")
- [ ] Pipeline diagram shows 914 as final validated step
- [ ] Config.yaml is only training config
- [ ] data/train/boughter/ has no strict_qc subdirectory
- [ ] preprocessing/boughter/ has no stage4 scripts
- [ ] All 178 strict_qc references accounted for (archived or updated)

---

## Timeline Estimate

**Estimated Time:** 2-3 hours
- Phase 1-2: 15 minutes (create dirs, move files)
- Phase 3: 30 minutes (write experiment README)
- Phase 4: 60-90 minutes (update 7+ doc files)
- Phase 5-6: 30 minutes (update pipeline docs, experiments README)
- Phase 7: 15 minutes (verification)

---

## Rollback Plan

If we need to restore strict_qc:
```bash
# All artifacts are in experiments/strict_qc_2025-11-04/
# Just move them back:
mv experiments/strict_qc_2025-11-04/preprocessing/* preprocessing/boughter/
mv experiments/strict_qc_2025-11-04/data/strict_qc data/train/boughter/
mv experiments/strict_qc_2025-11-04/configs/* configs/
mv experiments/strict_qc_2025-11-04/docs/* docs/boughter/
```

---

## Questions Before Execution

1. **Do we want to preserve the strict_qc data for potential future testing?**
   - If yes → Option 2 (Archive)
   - If no → Option 1 (Delete)

2. **Is there value in showing the experimental process?**
   - If yes → Option 2 (demonstrates scientific rigor)
   - If no → Option 1 (simpler is better)

3. **Are there other experiments we should archive alongside this?**
   - Check for other unvalidated experimental code

4. **Should we test the 852-seq model before archiving?**
   - Might validate the hypothesis (unlikely)
   - Or confirm it's unnecessary (more likely)
   - Adds time but provides closure

---

## Next Steps

1. **Decision:** Choose Option 1, 2, or 3
2. **Review:** Walk through execution plan
3. **Execute:** Run phases 1-7 if Option 2 chosen
4. **Verify:** Complete Phase 7 checklist
5. **Commit:** Single clean commit archiving the experiment
6. **Document:** Update CHANGELOG with rationale

---

## Contact

Questions or concerns? Discuss before executing this plan.

**Document Author:** Claude Code
**Last Updated:** 2025-11-06
**Status:** AWAITING USER DECISION
