# Boughter Documentation Archive

**Purpose:** Historical documentation from the Boughter dataset development process.

**Note:** These documents are archived because they are either:
1. Historical investigation/debugging documents superseded by current documentation
2. Status reports from intermediate development stages
3. Accurate but redundant with current primary documentation

---

## Archived Documents

### BOUGHTER_NOVO_REPLICATION_ANALYSIS.md
**Date:** 2025-11-04
**Status:** Historical investigation
**Why Archived:** Investigation process document that led to `BOUGHTER_NOVO_METHODOLOGY_CLARIFICATION.md`. Still accurate but superseded by the clarification document.

**Contains:**
- Investigation of apparent contradictions in Novo's paper
- Analysis of "Boughter methodology" vs "ANARCI/IMGT"
- Hybri's concerns and responses
- Resolution of position 118 issue

**Current Reference:** See `../BOUGHTER_NOVO_METHODOLOGY_CLARIFICATION.md` for the final clarification.

---

### accuracy_verification_report.md
**Date:** Pre-2025-11-02
**Status:** Progress report (pre-P0 fix)
**Why Archived:** Written before P0 fix discovery (gap characters and stop codon contamination). Pipeline statistics are accurate but document doesn't mention critical bugs that were later discovered.

**Contains:**
- Pipeline statistics (1,171 → 1,117 → 1,110 → 1,065)
- Stage descriptions
- Fragment file validation
- Label distribution

**Issues:** Claims "✅ VALIDATED" but was written before P0 blocker was discovered.

**Current Reference:** See `preprocessing/boughter/README.md` for current pipeline status.

---

### boughter_processing_status.md
**Date:** 2025-11-02
**Status:** Status report (includes P0 fix)
**Why Archived:** Comprehensive status report that is now superseded by `preprocessing/boughter/README.md` as the single source of truth. Document is accurate but redundant.

**Contains:**
- Complete pipeline statistics
- P0 fix documentation (gap characters, stop codons)
- Stage-by-stage breakdown
- Novo parity check
- Training subset details

**Current Reference:** See `preprocessing/boughter/README.md` for current pipeline documentation.

---

## Active Documentation

For current, actively maintained Boughter documentation, see:

### Primary Reference Documents (in docs/boughter/)
1. **BOUGHTER_DATASET_COMPLETE_HISTORY.md** - Master historical reference
2. **BOUGHTER_NOVO_METHODOLOGY_CLARIFICATION.md** - Critical methodological clarification
3. **BOUGHTER_P0_FIX_REPORT.md** - Essential bug fix documentation (gap characters, stop codons)
4. **boughter_cdr_boundary_investigation.md** - CDR boundary technical analysis
5. **boughter_data_sources.md** - Novo methodology requirements
6. **cdr_boundary_first_principles_audit.md** - Gold standard first-principles analysis

### Implementation Reference (in preprocessing/boughter/)
- **README.md** - Complete preprocessing pipeline guide (SINGLE SOURCE OF TRUTH)
- **stage1_dna_translation.py** - DNA translation implementation
- **stage2_stage3_annotation_qc.py** - ANARCI annotation + QC implementation
- **validate_stage1.py** - Stage 1 validation
- **validate_stages2_3.py** - Stages 2+3 validation
- **train_hyperparameter_sweep.py** - Hyperparameter optimization

---

**Last Updated:** 2025-11-06
**Archive Created:** 2025-11-06
**Reason:** Documentation cleanup to establish clear hierarchy and single source of truth
