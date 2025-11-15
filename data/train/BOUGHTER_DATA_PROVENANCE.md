# Boughter Dataset - Complete Data Provenance & Processing Pipeline

**Document Purpose:** This document traces the complete lineage of the Boughter dataset from original source through all processing stages to final training data.

**Last Updated:** 2025-11-04
**Branch:** leroy-jenkins/full-send

---

## Table of Contents

1. [Data Source & Origin](#data-source--origin)
2. [Directory Structure Overview](#directory-structure-overview)
3. [Processing Pipeline](#processing-pipeline)
4. [Citations & References](#citations--references)
5. [Quality Control](#quality-control)

---

## Data Source & Origin

### Original Publication

**Primary Paper:**
Boughter CT, Borowska MT, Guthmiller JJ, et al. (2020)
*"Biochemical Patterns of Antibody Polyreactivity Revealed Through a Bioinformatics-Based Analysis of CDR Loops"*
**eLife** 9:e61393
DOI: [10.7554/eLife.61393](https://doi.org/10.7554/eLife.61393)

### Raw Data Source

**GitHub Repository:**
[ctboughter/AIMS_manuscripts](https://github.com/ctboughter/AIMS_manuscripts)

**Specific Location:**
`app_data/full_sequences/`

**Local Copy:**
`reference_repos/AIMS_manuscripts/app_data/full_sequences/`

### Data Verification

✅ **Confirmed:** All files in `data/train/boughter/raw/` are **byte-for-byte identical** to the original source at `reference_repos/AIMS_manuscripts/app_data/full_sequences/`

**Verification Command:**
```bash
diff -r data/train/boughter/raw/ \
        reference_repos/AIMS_manuscripts/app_data/full_sequences/ \
        --brief | grep -v "README.md\|translation_failures.log"
# Result: No differences (only local processing artifacts differ)
```

---

## Directory Structure Overview

```
data/train/
├── README.md                             ← Overview of training datasets
├── BOUGHTER_DATA_PROVENANCE.md           ← This document
│
└── boughter/                             ← Dataset-centric organization
    │
    ├── README.md                         ← Boughter-specific documentation
    │
    ├── raw/                              ← Stage 0: Raw DNA sequences from AIMS
    │   ├── flu_fastaH.txt                │  (19 files, copied from AIMS_manuscripts)
    │   ├── flu_fastaL.txt                │
    │   ├── flu_NumReact.txt              │  Source: ctboughter/AIMS_manuscripts
    │   ├── gut_hiv_fastaH.txt            │          app_data/full_sequences/
    │   ├── gut_hiv_fastaL.txt            │
    │   ├── gut_hiv_NumReact.txt          │  Format: DNA nucleotide sequences (FASTA)
    │   ├── nat_cntrl_fastaH.txt          │          + polyreactivity flags
    │   ├── nat_cntrl_fastaL.txt          │
    │   ├── nat_cntrl_NumReact.txt        │
    │   ├── nat_hiv_fastaH.txt            │
    │   ├── nat_hiv_fastaL.txt            │
    │   ├── nat_hiv_NumReact.txt          │
    │   ├── plos_hiv_fastaH.txt           │
    │   ├── plos_hiv_fastaL.txt           │
    │   ├── plos_hiv_YN.txt               │
    │   ├── mouse_fastaH.dat              │
    │   ├── mouse_fastaL.dat              │
    │   ├── mouse_YN.txt                  │
    │   └── translation_failures.log      ← Generated during Stage 1 processing
    │
    ├── processed/                        ← Stage 1: Translated protein sequences
    │   └── boughter.csv                  │  (1,117 sequences from 1,171 raw)
    │
    ├── annotated/                        ← Stages 2+3: ANARCI-annotated fragments
    │   ├── VH_only_boughter.csv          │  (16 fragment files, 1,065 sequences each)
    │   ├── VL_only_boughter.csv          │
    │   ├── H-CDR1_boughter.csv           │  Each file contains:
    │   ├── H-CDR2_boughter.csv           │  - Annotated sequences (ANARCI/IMGT)
    │   ├── H-CDR3_boughter.csv           │  - Polyreactivity labels
    │   ├── L-CDR1_boughter.csv           │  - include_in_training flag
    │   ├── L-CDR2_boughter.csv           │
    │   ├── L-CDR3_boughter.csv           │
    │   ├── H-CDRs_boughter.csv           │
    │   ├── L-CDRs_boughter.csv           │
    │   ├── H-FWRs_boughter.csv           │
    │   ├── L-FWRs_boughter.csv           │
    │   ├── VH+VL_boughter.csv            │
    │   ├── All-CDRs_boughter.csv         │
    │   ├── All-FWRs_boughter.csv         │
    │   ├── Full_boughter.csv             │
    │   ├── annotation_failures.log       ← Stage 2 failures (7 sequences)
    │   ├── qc_filtered_sequences.txt     ← Stage 3 filtered (45 sequences)
    │   └── validation_report.txt         ← QC metrics
    │
    ├── canonical/                        ← Authoritative training file
    │   ├── VH_only_boughter_training.csv │  (914 sequences: 443 specific, 471 non-specific)
    │   └── README.md                     ← Training file documentation
    │
    └── (strict QC experiment archived)   ← See experiments/strict_qc_2025-11-04/
        ├── data/strict_qc/*.csv          │  (16 experimental fragment files)
        ├── preprocessing/stage4_*.py     │  (Stage 4 scripts)
        └── EXPERIMENT_README.md          ← Experimental hypothesis & results
```

---

## Processing Pipeline

### Overview

```
data/train/boughter/raw/           (Stage 0: Raw DNA FASTA)
    ↓
[Stage 1: DNA Translation & Novo Flagging]
    ↓
data/train/boughter/processed/boughter.csv   (1,117 protein sequences)
    ↓
[Stages 2+3: ANARCI Annotation + QC Filtering]
    ↓
data/train/boughter/annotated/*_boughter.csv (16 fragment types × 1,065 sequences)
    ↓
data/train/boughter/canonical/VH_only_boughter_training.csv (914 sequences for training)
```

### Stage 1: DNA Translation & Novo Nordisk Flagging

**Script:** `preprocessing/boughter/stage1_dna_translation.py`

**Input:**
- `data/train/boughter/raw/*.txt` - 6 subsets, 1,171 antibody sequences
  - flu (379), hiv_nat (134), hiv_cntrl (50), hiv_plos (52), gut_hiv (75), mouse_iga (481)

**Processing:**
1. **DNA → Protein Translation**
   - BioPython translation (all 6 reading frames tested)
   - Validates protein sequences (valid amino acids only)
   - Logs translation failures

2. **Novo Nordisk Flagging Strategy**
   - 0 flags → Specific (label=0, include in training)
   - 1-3 flags → Mild polyreactive (exclude from training)
   - 4+ flags → Non-specific (label=1, include in training)

**Output:**
- `data/train/boughter/processed/boughter.csv` - 1,117 sequences (95.4% success rate)
- `data/train/boughter/raw/translation_failures.log` - 54 failures (4.6%)

**Columns:**
```
id, subset, heavy_seq, light_seq, num_flags, flag_category, label, include_in_training, source
```

**Statistics:**
```
Total translated:  1,117 sequences (95.4%)
Translation loss:  54 sequences (4.6%)

Novo Flagging Results:
  Specific (0 flags):       461 (41.3%) → Training
  Mild (1-3 flags):         169 (15.1%) → Excluded
  Non-specific (4+ flags):  487 (43.6%) → Training

Training Set: 948 sequences (461 specific + 487 non-specific)
Excluded:     169 sequences (mild polyreactivity)
```

**Reference:** See `docs/boughter/boughter_data_sources.md` for detailed Novo Nordisk methodology

---

### Stages 2+3: ANARCI Annotation & Quality Control

**Script:** `preprocessing/boughter/stage2_stage3_annotation_qc.py`

#### Stage 2: ANARCI Annotation

**Input:**
- `data/train/boughter/processed/boughter.csv` - 1,117 sequences

**Processing:**
1. **ANARCI Annotation** (IMGT numbering scheme)
   - Annotates heavy and light chain CDRs/FWRs
   - Uses strict IMGT boundaries
   - Logs annotation failures

**Output:**
- 1,110 successfully annotated sequences (99.4% success)
- `data/train/boughter/annotated/annotation_failures.log` - 7 failures (0.6%)

**Failure Breakdown:**
```
flu:      6 failures (1.7%)
hiv_plos: 1 failure (2.3%)
```

#### Stage 3: Post-Annotation Quality Control

**Input:**
- 1,110 annotated sequences from Stage 2

**Processing:**
1. **Quality Filters:**
   - Remove sequences with 'X' (ambiguous residues) in ANY CDR
   - Remove sequences with empty CDRs
   - Maintain separate log of filtered sequences

**Output:**
- 1,065 clean sequences (95.9% retention from Stage 2)
- `data/train/boughter/annotated/qc_filtered_sequences.txt` - 45 filtered sequences
- `data/train/boughter/canonical/VH_only_boughter_training.csv` - 914 training sequences (sequence + label)

**QC Statistics:**
```
Input:     1,110 sequences
Filtered:  45 sequences
  - X in CDRs:    21 sequences
  - Empty CDRs:   25 sequences
Output:    1,065 sequences (95.9% retention)
```

#### Fragment File Generation

**Processing:**
1. **Extract 16 Fragment Types:**
   - Individual chains: VH_only, VL_only
   - Individual CDRs: H-CDR1, H-CDR2, H-CDR3, L-CDR1, L-CDR2, L-CDR3
   - CDR groups: H-CDRs, L-CDRs, All-CDRs
   - FWR groups: H-FWRs, L-FWRs, All-FWRs
   - Combined: VH+VL, Full

2. **Add Metadata:**
   - All original columns from boughter.csv
   - include_in_training flag (based on Novo flagging)
   - source = "Boughter2020"
   - sequence_length

**Output:**
- 16 fragment CSV files in `data/train/boughter/`
- Each file: 1,065 rows (all sequences represented)
- Training subset: `VH_only_boughter_training.csv` (914 rows)

**Fragment File Statistics:**
```
Total sequences per fragment:  1,065
Training eligible (flag=True): 914 sequences
Excluded (flag=False):         151 sequences (1-3 flags)

Training Set Label Balance:
  Specific (0):     443 (48.5%)
  Non-specific (1): 471 (51.5%)
```

---

### Complete Pipeline Statistics

```
Raw Input:      1,171 DNA sequences (100.0%)
   ↓ Stage 1 (Translation)
After Stage 1:  1,117 protein sequences (95.4%)
   ↓ Stage 2 (ANARCI)
After Stage 2:  1,110 annotated (99.4% of Stage 1)
   ↓ Stage 3 (QC)
After Stage 3:  1,065 clean sequences (95.9% of Stage 2)
   ↓ Novo Filtering (1-3 flags excluded)
Training Data:  914 sequences (86.0% of QC-passed)

Overall: 914 / 1,171 = 78.1% of raw data used for training
```

**Attrition Breakdown:**
```
Translation failures:     54 sequences (4.6%)
ANARCI failures:          7 sequences (0.6%)
QC filtering:             45 sequences (4.1%)
Novo exclusion (1-3 flags): 151 sequences (13.8%)

Total excluded:           257 sequences (21.9%)
Final training set:       914 sequences (78.1%)
```

---

## Citations & References

### Dataset Components

The Boughter dataset is composed of antibodies from multiple published studies:

#### Influenza (flu subset)
1. **Boughter et al. (2020)**
   *"Biochemical Patterns of Antibody Polyreactivity Revealed Through a Bioinformatics-Based Analysis of CDR Loops"*
   eLife 9:e61393
   DOI: [10.7554/eLife.61393](https://doi.org/10.7554/eLife.61393)

2. **Guthmiller et al. (2020)**
   *"Polyreactive Broadly Neutralizing B cells Are Selected to Provide Defense against Pandemic Threat Influenza Viruses"*
   Immunity 53(6):1230-1244.e6
   DOI: [10.1016/j.immuni.2020.10.005](https://doi.org/10.1016/j.immuni.2020.10.005)

#### HIV Gut (gut_hiv subset)
1. **Prigent et al. (2018)**
   *"Conformational plasticity in broadly neutralizing HIV-1 antibodies triggers polyreactivity"*
   Cell Reports 23(9):2568-2581
   DOI: [10.1016/j.celrep.2018.04.101](https://doi.org/10.1016/j.celrep.2018.04.101)

2. **Planchais et al. (2019)**
   *"HIV-1 envelope recognition by polyreactive and cross-reactive intestinal B cells"*
   Cell Reports 27(4):1048-1058.e6
   DOI: [10.1016/j.celrep.2019.03.032](https://doi.org/10.1016/j.celrep.2019.03.032)

#### HIV Nature (hiv_nat subset)
**Mouquet et al. (2010)**
*"Polyreactivity increases the apparent affinity of anti-HIV antibodies by heteroligation"*
Nature 467(7315):591-595
DOI: [10.1038/nature09385](https://doi.org/10.1038/nature09385)

#### HIV PLOS (hiv_plos subset)
**Mouquet et al. (2011)**
*"Memory B cell antibodies to HIV-1 gp140 cloned from individuals infected with clade A and B viruses"*
PLOS ONE 6(11):e27621
DOI: [10.1371/journal.pone.0027621](https://doi.org/10.1371/journal.pone.0027621)

#### Mouse IgA (mouse_iga subset)
**Bunker et al. (2017)**
*"Natural polyreactive IgA antibodies coat the intestinal microbiota"*
Science 358(6361):eaan6619
DOI: [10.1126/science.aan6619](https://doi.org/10.1126/science.aan6619)

### Processing Methodology

**Sakhnini et al. (2025)**
*"Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters"*
**(Novo Nordisk flagging strategy)**
DOI: pending

---

## Quality Control

### Validation Scripts

All processing stages have corresponding validation scripts:

1. **Stage 1 Validation:**
   `preprocessing/boughter/validate_stage1.py`
   - Validates boughter.csv sequence counts (expect 1,117)
   - Checks protein sequence validity
   - Verifies Novo flagging distribution

2. **Stages 2+3 Validation:**
   `preprocessing/boughter/validate_stages2_3.py`
   - Validates all 16 fragment files exist
   - Checks sequence counts (expect 1,065 each)
   - Verifies include_in_training flag
   - Validates training subset (expect 914)

### Reproducibility

✅ **Pipeline is 100% reproducible** - Validated 2025-11-04

**Validation Results:**
- Stage 1 output: **IDENTICAL** (byte-for-byte match)
- Stage 2+3 outputs: **IDENTICAL** (all 16 fragment files)
- Training subset: **IDENTICAL** (914 sequences)

**Validation Report:** See full validation results in preprocessing pipeline execution logs

### Data Integrity Checks

**File Counts:**
```bash
# Raw files
ls data/train/boughter/raw/*.txt | wc -l
# Expected: 18 files (6 subsets × 3 files each)

# Fragment files
ls data/train/boughter/annotated/*_boughter.csv | wc -l
# Expected: 17 files (16 fragments + 1 training subset)
```

**Sequence Counts:**
```python
import pandas as pd

# Stage 1 output
df_stage1 = pd.read_csv('data/train/boughter/processed/boughter.csv')
assert len(df_stage1) == 1117, "Stage 1 count mismatch"

# Fragment files (each should have 1,065)
df_vh = pd.read_csv('data/train/boughter/annotated/VH_only_boughter.csv', comment='#')
assert len(df_vh) == 1065, "Fragment count mismatch"

# Training subset
df_train = pd.read_csv('data/train/boughter/canonical/VH_only_boughter_training.csv', comment='#')
assert len(df_train) == 914, "Training count mismatch"
```

---

## Summary

**Data Lineage:**
```
AIMS_manuscripts/app_data/full_sequences/  (GitHub: ctboughter/AIMS_manuscripts)
    ↓ [Copied to]
data/train/boughter/raw/  (19 files, DNA sequences)
    ↓ [Stage 1: preprocessing/boughter/stage1_dna_translation.py]
data/train/boughter/processed/boughter.csv  (1,117 protein sequences)
    ↓ [Stages 2+3: preprocessing/boughter/stage2_stage3_annotation_qc.py]
data/train/boughter/*.csv  (16 fragments × 1,065 sequences)
    ↓ [Novo filtering: include_in_training flag]
data/train/boughter/canonical/VH_only_boughter_training.csv  (914 sequences)
```

**Key Points:**
- ✅ Raw data is **unmodified** copy from AIMS_manuscripts repository
- ✅ All processing is **fully documented** and **reproducible**
- ✅ Pipeline produces **identical outputs** when re-run
- ✅ All citations and references are **traceable**
- ✅ Quality control metrics are **validated**

**For detailed processing methodology, see:**
- `preprocessing/boughter/README.md` - Pipeline documentation
- `docs/boughter/boughter_data_sources.md` - Novo Nordisk methodology
- `BOUGHTER_PREPROCESSING_REORGANIZATION_PLAN.md` - Reorganization plan

---

**Document Version:** 1.0
**Last Validated:** 2025-11-04
**Maintainer:** Ray (Clarity Digital Twin Project)
