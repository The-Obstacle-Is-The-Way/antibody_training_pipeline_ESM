# Boughter Dataset Documentation

**Purpose:** Reference documentation for the Boughter 2020 antibody polyreactivity dataset processing and methodology.

**Dataset:** Boughter et al. (2020) - 1,171 mouse antibodies with ELISA polyreactivity measurements

---

## Quick Start

### For Implementation
**→ See [`preprocessing/boughter/README.md`](../../../preprocessing/boughter/README.md)** - SINGLE SOURCE OF TRUTH for pipeline implementation, scripts, and usage

### For Understanding the Dataset
Start here with this directory's documentation ↓

---

## Documentation Structure

### Core Reference Documents

#### 1. complete_history.md
**Purpose:** Master historical reference and QC level comparison
- Complete dataset evolution (1,171 → 1,117 → 1,110 → 1,065 → 914 training)
- Two QC levels explained (Boughter QC vs Strict QC)
- Position 118 resolution
- Training file selection guidance
- Why strict QC was archived (no performance improvement)

**Read this if:** You want comprehensive context on dataset processing decisions

---

#### 2. novo_methodology_clarification.md
**Purpose:** Critical methodological insight resolving apparent contradiction in Novo's paper
- "Boughter methodology" = QC filtering + flagging (NOT CDR boundaries)
- Evidence from Boughter's actual `seq_loader.py` code
- Novo's pipeline: ANARCI/IMGT → Boughter QC → Boughter flagging
- Resolves: How can you use "Boughter methodology" AND "ANARCI/IMGT"?

**Read this if:** You're replicating Novo Nordisk methodology or confused about CDR boundaries

---

#### 3. p0_fix_report.md
**Purpose:** Essential bug fix documentation (P0 blocker)
- Gap character contamination (13 sequences, 1.2%)
- Stop codon contamination (241 sequences, 22.6%)
- Fix: V-domain reconstruction from fragments
- Test suite: 5/5 tests passing
- Why bug only affected Boughter (DNA input) not Harvey/Shehata (protein input)

**Read this if:** You want to understand why V-domain reconstruction was necessary

---

### Technical Analysis Documents

#### 4. cdr_boundary_investigation.md
**Purpose:** CDR boundary technical analysis
- Position 118 discrepancy (Boughter's IgBLAST vs IMGT standard)
- Biological rationale (position 118 is Framework 4 anchor, W or F, 99% conserved)
- Harvey et al. 2022 validation (CDR2 variable lengths are normal)
- Resolution: Use strict IMGT (CDR-H3 = positions 105-117)

**Read this if:** You need technical justification for CDR boundary decisions

---

#### 5. data_sources.md
**Purpose:** Novo Nordisk methodology requirements specification
- Complete requirements from Sakhnini et al. 2025 paper
- ELISA polyreactivity panel (7 antigens)
- Flagging strategy (0, 1-3 exclude, 4+)
- What IS and is NOT specified in Novo's paper
- Updates through 2025-11-04 clarification

**Read this if:** You're implementing Novo's methodology from scratch

---

#### 6. cdr_boundary_first_principles_audit.md
**Purpose:** Gold standard first-principles analysis of CDR boundaries
- Rigorous analysis from IMGT.org official documentation
- Multi-source validation (IMGT, Boughter code, Harvey paper)
- Biological + ML rationale for excluding position 118
- 2025 best practices (post-annotation QC)

**Read this if:** You need the most authoritative technical reference

---

## Document Hierarchy

```
preprocessing/boughter/README.md ← SINGLE SOURCE OF TRUTH (implementation)
         ↑
         └── References these docs for methodology and context

docs/boughter/
├── README.md (THIS FILE) ← Documentation index
│
├── complete_history.md ← Master reference
├── novo_methodology_clarification.md ← Key insight
├── p0_fix_report.md ← Critical bug fix
│
├── cdr_boundary_investigation.md ← CDR boundaries
├── data_sources.md ← Novo requirements
├── cdr_boundary_first_principles_audit.md ← Gold standard reference
│
└── archive/ ← Historical investigation and status reports
    ├── README.md ← Archive index
    ├── BOUGHTER_NOVO_REPLICATION_ANALYSIS.md ← Investigation process
    ├── accuracy_verification_report.md ← Pre-P0 fix report
    └── boughter_processing_status.md ← Status report (2025-11-02)
```

---

## Key Facts (from current implementation)

### Pipeline Statistics
- **Raw input:** 1,171 DNA sequences (6 subsets)
- **Stage 1 (DNA translation):** 1,117 protein sequences (95.4% success)
- **Stage 2 (ANARCI annotation):** 1,110 annotated (99.4% success)
- **Stage 3 (QC filtering):** 1,065 clean sequences (95.9% retention)
- **Training subset:** 914 sequences (Novo flagging: 0 and 4+ flags only)

### Novo Flagging Strategy
- **0 flags** → Specific (label=0, include in training)
- **1-3 flags** → Mildly polyreactive (EXCLUDE from training)
- **4+ flags** → Non-specific (label=1, include in training)

### 16 Fragment Types
1. VH_only, VL_only (full variable domains)
2. H-CDR1, H-CDR2, H-CDR3 (heavy chain CDRs)
3. L-CDR1, L-CDR2, L-CDR3 (light chain CDRs)
4. H-CDRs, L-CDRs (concatenated CDRs)
5. H-FWRs, L-FWRs (concatenated frameworks)
6. VH+VL (paired variable domains)
7. All-CDRs, All-FWRs (all concatenated)
8. Full (alias for VH+VL)

### Training Files
- **Production:** `train_datasets/boughter/canonical/VH_only_boughter_training.csv` (914 sequences)
- **Fragment files:** `train_datasets/boughter/annotated/*_boughter.csv` (1,065 sequences each, 16 files)

---

## References

### Primary Papers
- **Boughter et al. (2020)** - "Biochemical patterns of antibody polyreactivity revealed through a bioinformatics-based analysis of CDR loops." *eLife* 9:e61393. DOI: https://doi.org/10.7554/eLife.61393

- **Sakhnini et al. (2025)** - "Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters." *Cell* 188(1):112-123. DOI: https://doi.org/10.1016/j.cell.2024.12.025

### Supporting Papers
- **Harvey et al. (2022)** - Validation of CDR2 variable lengths
- **IMGT documentation** - CDR/Framework boundary definitions

---

## For Contributors

### When to Update Documentation

1. **Implementation changes** → Update `preprocessing/boughter/README.md` (SSOT)
2. **Methodology insights** → Update relevant docs/boughter/*.md files
3. **Bug fixes** → Create new report (follow p0_fix_report.md pattern)
4. **Historical/debugging docs** → Move to archive/ with explanation

### Documentation Principles

- **preprocessing/boughter/README.md** is the SINGLE SOURCE OF TRUTH for implementation
- **docs/boughter/*.md** provide context, rationale, and technical justification
- **docs/boughter/archive/** contains historical investigation and status reports
- Keep docs DRY (Don't Repeat Yourself) - reference other docs instead of duplicating

---

**Last Updated:** 2025-11-06
**Documentation Version:** 2.0 (post-cleanup)
**Status:** ✅ Active and maintained
