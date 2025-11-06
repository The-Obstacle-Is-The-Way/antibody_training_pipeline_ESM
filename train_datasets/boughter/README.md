# Boughter Dataset - Processed Antibody Sequences

This directory contains processed antibody sequences from the Boughter et al. 2020 polyreactivity study, prepared according to the Novo Nordisk methodology (Sakhnini et al. 2025).

---

## Dataset Overview

**Source:** Boughter et al. (2020) eLife 9:e61393
**Reference Implementation:** Sakhnini et al. (2025) - Novo Nordisk antibody non-specificity prediction
**Total Antibodies:** 1,171 (raw) → 1,065 (after ANARCI + Boughter QC) → 914 (training subset, PRODUCTION)

**Experimental Strict QC:** Archived in `experiments/strict_qc_2025-11-04/` (never validated)

**Assay:** ELISA polyreactivity against 4-7 diverse antigens (DNA, insulin, LPS, flagellin, albumin, cardiolipin, KLH)
**Classification:** Binary (0 flags = specific, 4+ flags = non-specific; 1-3 flags excluded from training)

---

## File Organization

### QC Levels

This dataset is available at **two QC levels**:

#### 1. Boughter QC (Standard)
**Files:** `*_boughter.csv`
**Sequences:** 1,065 (all flags), 914 (training subset with `include_in_training==True`)
**QC Filters:**
- ✅ X in CDRs only (L1, L2, L3, H1, H2, H3)
- ✅ Empty CDR removal
- ✅ Flagging strategy (0 and 4+ flags only, exclude 1-3)

**Use Case:** Exact replication of Boughter's published methodology
**Source:** Boughter et al. 2020, seq_loader.py lines 10-33

#### 2. Strict QC (Industry Standard)
**Files:** `*_boughter_strict_qc.csv`
**Sequences:** 840-914 (fragment-dependent, see table below)
**QC Filters:**
- ✅ All Boughter QC filters (above)
- ✅ **PLUS:** X filtering in FULL sequence (not just CDRs)
- ✅ **PLUS:** Non-standard AA filtering (B, Z, J, U, O)

**Use Case:** Matching Novo's likely methodology + modern ML best practices
**Source:** Industry standard practice + inferred from Novo Nordisk paper

---

## Fragment Types

Each dataset is available in **16 different antibody fragments** (from Novo Table 4):

### Sequence Counts by Fragment and QC Level

| Fragment      | Boughter QC (All) | Boughter QC (Training) | Strict QC | Difference |
|---------------|-------------------|------------------------|-----------|------------|
| **Variable Domains** |
| VH_only       | 1,065             | 914                    | **852**   | -62 (VH frameworks) |
| VL_only       | 1,065             | 914                    | **900**   | -14 (VL frameworks) |
| VH+VL         | 1,065             | 914                    | **840**   | -74 (both chains) |
| Full          | 1,065             | 914                    | **840**   | -74 (both chains) |
| **Heavy Chain CDRs** |
| H-CDR1        | 1,065             | 914                    | **914**   | No change |
| H-CDR2        | 1,065             | 914                    | **914**   | No change |
| H-CDR3        | 1,065             | 914                    | **914**   | No change |
| H-CDRs        | 1,065             | 914                    | **914**   | No change |
| **Light Chain CDRs** |
| L-CDR1        | 1,065             | 914                    | **914**   | No change |
| L-CDR2        | 1,065             | 914                    | **914**   | No change |
| L-CDR3        | 1,065             | 914                    | **914**   | No change |
| L-CDRs        | 1,065             | 914                    | **914**   | No change |
| **Combined** |
| All-CDRs      | 1,065             | 914                    | **914**   | No change |
| H-FWRs        | 1,065             | 914                    | **852**   | -62 (VH frameworks) |
| L-FWRs        | 1,065             | 914                    | **900**   | -14 (VL frameworks) |
| All-FWRs      | 1,065             | 914                    | **840**   | -74 (both chains) |

**Key Insight:** CDR-only fragments have no change in strict QC because Boughter's QC already filters X in CDRs. Framework and full-sequence fragments show reductions due to X in framework regions.

---

## File Naming Convention

```
{fragment}_boughter.csv           # Boughter QC (1,065 sequences, all flags)
{fragment}_boughter_strict_qc.csv # Strict QC (840-914 sequences, training subset + industry standard)
VH_only_boughter_training.csv     # Special export (914 sequences, flattened [sequence, label] only)
```

---

## Column Descriptions

### Standard Fragment Files (*_boughter.csv, *_boughter_strict_qc.csv)

| Column | Type | Description |
|--------|------|-------------|
| `id` | str | Unique sequence identifier (format: `{subset}_{index}`) |
| `sequence` | str | Amino acid sequence for this fragment |
| `label` | int | Binary classification: 0 = specific (0 flags), 1 = non-specific (4+ flags) |
| `subset` | str | Source dataset: flu, hiv_nat, hiv_cntrl, hiv_plos, gut_hiv, mouse_iga |
| `num_flags` | int | Number of polyreactive antigens bound (0-7) |
| `flag_category` | str | Category: "specific" (0), "mildly_poly" (1-3), "poly" (4+) |
| `include_in_training` | bool | True if 0 or 4+ flags (included in training), False if 1-3 flags (excluded) |
| `source` | str | Fragment type (e.g., "VH_only", "H-CDR3", "Full") |
| `sequence_length` | int | Length of the sequence in amino acids |

### Training Export File (VH_only_boughter_training.csv)

**Columns:** `sequence`, `label` only (flattened for ML training)
**Note:** This file has **NO id, NO fragments, NO metadata** - use fragment CSVs for full provenance

---

## Processing Pipeline

```
Raw DNA FASTA (6 subsets, 1,171 sequences)
   ↓
Stage 1: DNA → Protein translation
   ↓ preprocessing/boughter/stage1_dna_translation.py
   ↓ Output: train_datasets/boughter/processed/boughter.csv (1,171 sequences)
   ↓
Stage 2+3: ANARCI annotation + Boughter QC
   ↓ preprocessing/boughter/stage2_stage3_annotation_qc.py
   ↓ QC: X in CDRs only, empty CDRs
   ↓ Outputs: 16 fragment CSVs (1,065 sequences each)
   ↓    *_boughter.csv (all flags, has include_in_training column)
   ↓
   ↓ Filter to training subset (include_in_training == True)
   ↓    VH_only_boughter_training.csv (914 sequences, flattened export)
   ↓
Stage 4: Additional QC (Industry Standard)
   ↓ preprocessing/boughter/stage4_additional_qc.py
   ↓ QC: X ANYWHERE in sequence, non-standard AA
   ↓ Outputs: 16 fragment CSVs (840-914 sequences each)
       *_boughter_strict_qc.csv (training subset + industry standard filters)
```

**Validation:**
- `preprocessing/boughter/validate_stage1.py` - Stage 1 validation
- `preprocessing/boughter/validate_stages2_3.py` - Stages 2+3 validation
- `preprocessing/boughter/validate_stage4.py` - Stage 4 validation

---

## Usage Recommendations

### For Exact Boughter Replication
```python
import pandas as pd

# Use standard Boughter QC files
df = pd.read_csv('train_datasets/boughter/annotated/VH_only_boughter.csv', comment='#')

# Filter to training subset
df_train = df[df['include_in_training'] == True]  # 914 sequences

# Or use the pre-exported training file (no metadata)
df_train = pd.read_csv('train_datasets/boughter/canonical/VH_only_boughter_training.csv')  # 914 sequences
```

### For Novo Nordisk Parity (Recommended)
```python
import pandas as pd

# Use strict QC files (matches Novo's likely methodology)
df_train = pd.read_csv('train_datasets/boughter/strict_qc/VH_only_boughter_strict_qc.csv', comment='#')  # 852 sequences

# For CDR-only analysis (no filtering difference)
df_cdr3 = pd.read_csv('train_datasets/boughter/strict_qc/H-CDR3_boughter_strict_qc.csv', comment='#')  # 914 sequences
```

### For Multi-Fragment Analysis
```python
import pandas as pd

# Load multiple fragments with consistent IDs
vh = pd.read_csv('train_datasets/boughter/strict_qc/VH_only_boughter_strict_qc.csv', comment='#')
vl = pd.read_csv('train_datasets/boughter/strict_qc/VL_only_boughter_strict_qc.csv', comment='#')
cdr3 = pd.read_csv('train_datasets/boughter/strict_qc/H-CDR3_boughter_strict_qc.csv', comment='#')

# Merge on 'id' column (available in all fragment CSVs)
merged = vh.merge(vl, on='id', suffixes=('_vh', '_vl'))
```

---

## Label Distribution

**Boughter QC (914 training sequences):**
- Label 0 (specific, 0 flags): 457 sequences (50.0%)
- Label 1 (non-specific, 4+ flags): 457 sequences (50.0%)

**Strict QC (VH_only, 852 sequences):**
- Label 0 (specific, 0 flags): 425 sequences (49.9%)
- Label 1 (non-specific, 4+ flags): 427 sequences (50.1%)

**Note:** Labels remain balanced after strict QC filtering.

---

## Performance Expectations

**Boughter QC (914 sequences):**
- Reported accuracy: 67.5% ± 8.9% (10-fold CV, our implementation)

**Strict QC (852 sequences):**
- Expected accuracy: ~71% (hypothesis: match Novo's reported performance)
- Reasoning: Remove noisy sequences with X in frameworks

**Novo Nordisk (Sakhnini et al. 2025):**
- Reported accuracy: 71%
- Methodology: ANARCI + IMGT + Boughter-style filtering (likely includes strict QC)

---

## Quality Control Summary

### What Was Filtered

**Stage 2+3 (Boughter QC):** 1,171 → 1,065 sequences (-106, -9.0%)
- X in CDRs (L1, L2, L3, H1, H2, H3)
- Empty CDRs
- ANARCI annotation failures

**Training subset filter:** 1,065 → 914 sequences (-151, -14.2%)
- Exclude 1-3 flags (mildly polyreactive)
- Keep 0 flags (specific) and 4+ flags (non-specific)

**Stage 4 (Strict QC):** 914 → 840-914 sequences (fragment-dependent)
- **VH_only/H-FWRs:** 914 → 852 (-62, -6.8%) - X in VH frameworks
- **VL_only/L-FWRs:** 914 → 900 (-14, -1.5%) - X in VL frameworks
- **Full/VH+VL/All-FWRs:** 914 → 840 (-74, -8.1%) - X in either chain
- **CDR-only fragments:** No change (X already filtered)

**Total reduction (raw → strict QC, VH_only):** 1,171 → 852 (-319, -27.2%)

---

## Documentation

**Complete methodology:**
- `BOUGHTER_ADDITIONAL_QC_PLAN.md` - Stage 4 implementation plan
- `docs/boughter/BOUGHTER_NOVO_METHODOLOGY_CLARIFICATION.md` - Novo methodology analysis
- `docs/boughter/BOUGHTER_NOVO_REPLICATION_ANALYSIS.md` - Replication strategy
- `docs/boughter/boughter_data_sources.md` - Data source specification
- `docs/boughter/cdr_boundary_first_principles_audit.md` - CDR boundary validation
- `train_datasets/BOUGHTER_DATA_PROVENANCE.md` - Full pipeline documentation

---

## Citations

### Primary Dataset

**Boughter et al. (2020) - Main Reference:**
Boughter CT, Borowska MT, Guthmiller JJ, Bendelac A, Wilson PC, Roux B, Adams EJ. Biochemical Patterns of Antibody Polyreactivity Revealed Through a Bioinformatics-Based Analysis of CDR Loops. *eLife* 9:e61393 (2020). DOI: 10.7554/eLife.61393

**Novo Nordisk Methodology:**
Sakhnini A et al. Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters. *bioRxiv* (2025, pending publication).

### Data Subsets

**Flu:**
Guthmiller JJ et al. Polyreactive Broadly Neutralizing B cells Are Selected to Provide Defense against Pandemic Threat Influenza Viruses. *Immunity* (2020).

**HIV Gut:**
Prigent J et al. Conformational plasticity in broadly neutralizing HIV-1 antibodies triggers polyreactivity. *Cell Reports* (2018).
Planchais C et al. HIV-1 envelope recognition by polyreactive and cross-reactive intestinal B cells. *Cell Reports* (2019).

**HIV Nat (Nature):**
Mouquet H et al. Polyreactivity increases the apparent affinity of anti-HIV antibodies by heteroligation. *Nature* (2010).

**HIV PLOS:**
Mouquet H et al. Memory B cell antibodies to HIV-1 gp140 cloned from individuals infected with clade A and B viruses. *PLOS ONE* (2011).

**Mouse IgA:**
Bunker JJ et al. Natural polyreactive IgA antibodies coat the intestinal microbiota. *Science* (2017).

### Methods

**ANARCI:**
Dunbar J, Deane CM. ANARCI: antigen receptor numbering and receptor classification. *Bioinformatics* 32:298-300 (2016).

**IMGT Numbering:**
Lefranc MP et al. IMGT unique numbering for immunoglobulin and T cell receptor variable domains and Ig superfamily V-like domains. *Dev Comp Immunol* 27:55-77 (2003).

---

**Document Status:**
- **Version:** 2.0 (Post-Stage 4 implementation)
- **Date:** 2025-11-04
- **Status:** ✅ Complete - All processing stages validated
- **Last Updated:** Stage 4 strict QC filtering complete
