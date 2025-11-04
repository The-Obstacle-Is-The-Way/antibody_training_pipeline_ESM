# Harvey Dataset – Data Sources & Methodology

**Date:** 2025-11-01
**Issue:** #4 – Harvey dataset preprocessing
**Status:** ✅ **DATASET VERIFIED AND PROCESSED**

---

## Executive Summary

The Harvey dataset contains **141,474 nanobody sequences** with binary polyreactivity labels from deep sequencing of FACS-sorted pools. This dataset is used by Novo Nordisk (Sakhnini et al. 2025) to **test** their ESM-1v VH-based logistic regression model for predicting antibody non-specificity.

**Key Points:**
- ✅ Dataset source: Official Harvey repository (`reference_repos/harvey_official_repo/backend/app/experiments/`)
- ✅ 71,772 high polyreactivity + 69,702 low polyreactivity = 141,474 total sequences
- ✅ Binary labels: 0 = low polyreactivity, 1 = high polyreactivity
- ✅ Successfully processed: 141,021 sequences (99.68% success rate)
- ✅ Failed sequences: 453 (0.32%) logged in `test_datasets/harvey/failed_sequences.txt`

---

## Dataset Source

### Official Harvey Repository

**Location:** `reference_repos/harvey_official_repo/backend/app/experiments/`

**Source Files:**
1. `high_polyreactivity_high_throughput.csv` - 71,772 sequences
2. `low_polyreactivity_high_throughput.csv` - 69,702 sequences

**Total:** 141,474 nanobody sequences

**Data Format:**
- IMGT-numbered positions (columns 1-128)
- Pre-extracted CDRs: CDR1_nogaps, CDR2_nogaps, CDR3_nogaps
- No explicit labels (labels assigned during conversion: high=1, low=0)

---

## Primary Literature Sources

### 1. Harvey et al. 2022 – Original Dataset Paper

**Title:** An in silico method to assess antibody fragment polyreactivity
**Journal:** Nature Communications, Volume 13, Article 7554
**DOI:** 10.1038/s41467-022-35276-4
**Date:** December 7, 2022

**Local Files:**
- Main paper: `literature/markdown/harvey-et-al-2022-in-silico-method-to-assess-antibody-fragment-polyreactivity/harvey-et-al-2022-in-silico-method-to-assess-antibody-fragment-polyreactivity.md`
- Supplementary: `literature/markdown/harvey-et-al-2022-supplementary-information/harvey-et-al-2022-supplementary-information.md`
- PDF: `literature/pdf/harvey-et-al-2022-in-silico-method-to-assess-antibody-fragment-polyreactivity.pdf`

**Key Dataset Information:**

Initial deep sequencing (from main paper):
> "Given this validation, we deep-sequenced the two FACS sorted pools and obtained **65,147 unique low polyreactivity** sequences and **69,155 unique highly polyreactive** sequences."

Extended deep sequencing:
> "Through additional rounds of FACS selection, we collected **1,221,800 unique low polyreactivity clones** and **1,058,842 unique high polyreactivity clones**."

**Experimental Method:**
- Started with >2 × 10⁹ synthetic yeast display nanobody library
- MACS enrichment for polyreactive clones
- FACS sorting with PSR (polyspecificity reagent from Sf9 insect cell membranes)
- Deep sequencing of high and low polyreactivity pools

---

### 2. Sakhnini et al. 2025 – Novo Nordisk Methodology (SSOT for Issue #4)

**Title:** Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters
**Journal:** bioRxiv preprint
**DOI:** 10.1101/2025.04.28.650927
**Date:** May 2025

**Local Files:**
- Main paper: `literature/markdown/Sakhnini_2025_Antibody_NonSpecificity_PLM_Biophysical/Sakhnini_2025_Antibody_NonSpecificity_PLM_Biophysical.md`
- PDF: `literature/pdf/Sakhnini_2025_Antibody_NonSpecificity_PLM_Biophysical.pdf`

**Harvey Dataset Usage:**

From Sakhnini Table 4:
| Dataset | Description | Assay | Reference |
|---------|-------------|-------|-----------|
| Harvey dataset | **>140 000 naïve nanobodies** | Poly-specific reagent (PSR) assay | [45] |

From Section 2.1:
> "Four different datasets were retrieved from public sources... (iv) **140 000 nanobody (Nb) clones assessed by the PSR assay from a naïve Nb library**."

> "In this study, the most balanced dataset (i.e. Boughter one) was selected for **training of ML models**, while the remining three (i.e. **Jain, Shehata and Harvey**, which consists exclusively of VHH sequences) were **used for testing**."

**Model Architecture:**
- Top performer: **ESM 1v mean-mode VH-based LogisticReg**
- 10-fold CV accuracy: 71%
- Tested on Harvey nanobodies (VHH sequences only)

---

## Data Processing Pipeline

### Step 1: Copy Official CSVs

**Command:**
```bash
# Copy high polyreactivity sequences
cp reference_repos/harvey_official_repo/backend/app/experiments/high_polyreactivity_high_throughput.csv \
   test_datasets/harvey_high.csv

# Copy low polyreactivity sequences
cp reference_repos/harvey_official_repo/backend/app/experiments/low_polyreactivity_high_throughput.csv \
   test_datasets/harvey_low.csv
```

**Result:** 141,474 total sequences (71,772 + 69,702)

### Step 2: CSV Conversion

**Script:** `scripts/convert_harvey_csvs.py`

**Functionality:**
1. Read high/low polyreactivity CSVs from official repo
2. Extract full sequences from IMGT-numbered position columns (1-128)
3. Preserve pre-extracted CDRs (CDR1_nogaps, CDR2_nogaps, CDR3_nogaps)
4. Assign binary labels (0=low polyreactivity, 1=high polyreactivity)
5. Combine into single unified CSV

**Command:**
```bash
python3 scripts/convert_harvey_csvs.py
```

**Output:** `test_datasets/harvey.csv` (141,474 sequences)

### Step 3: Fragment Extraction

**Script:** `preprocessing/process_harvey.py`

**Method:**
- ANARCI (riot_na) with IMGT numbering scheme
- VHH-specific (nanobody only, no light chain)
- Error handling: Skip failures, log IDs, continue processing

**Command:**
```bash
python3 preprocessing/process_harvey.py
```

**Output:** 6 fragment CSV files in `test_datasets/harvey/`

| Fragment File | Description | Row Count | Length Range |
|---------------|-------------|-----------|--------------|
| `VHH_only_harvey.csv` | Full nanobody variable domain | 141,021 | 102-125 aa (mean: 119.1) |
| `H-CDR1_harvey.csv` | Heavy chain CDR1 | 141,021 | 1-10 aa (mean: 8.0) |
| `H-CDR2_harvey.csv` | Heavy chain CDR2 | 141,021 | 1-11 aa (mean: 7.7) |
| `H-CDR3_harvey.csv` | Heavy chain CDR3 | 141,021 | 1-17 aa (mean: 12.3) |
| `H-CDRs_harvey.csv` | Concatenated H-CDR1+2+3 | 141,021 | 13-33 aa (mean: 28.0) |
| `H-FWRs_harvey.csv` | Concatenated H-FWR1+2+3+4 | 141,021 | 78-99 aa (mean: 91.0) |

**Processing Statistics:**
- Input sequences: 141,474
- Successfully annotated: 141,021 (99.68%)
- Failed sequences: 453 (0.32%)
- Runtime: ~14 minutes
- Throughput: ~235 sequences/second

---

## Dataset Statistics

### Label Distribution

**Input (harvey.csv):**
- High polyreactivity (label=1): 71,772 sequences (50.7%)
- Low polyreactivity (label=0): 69,702 sequences (49.3%)
- **Balanced dataset** - excellent for binary classification

**Output (after ANARCI processing):**
- High polyreactivity: 71,759 sequences (50.9%)
- Low polyreactivity: 69,262 sequences (49.1%)
- Total: 141,021 sequences

### Sequence Lengths (VHH_only_harvey.csv)

- Minimum: 102 amino acids
- Maximum: 125 amino acids
- Mean: 119.1 amino acids
- **Typical nanobody range:** 110-130 aa

### Fragment Lengths

| Fragment | Min (aa) | Max (aa) | Mean (aa) |
|----------|----------|----------|-----------|
| VHH_only | 102 | 125 | 119.1 |
| H-CDR1 | 1 | 10 | 8.0 |
| H-CDR2 | 1 | 11 | 7.7 |
| H-CDR3 | 1 | 17 | 12.3 |
| H-CDRs | 13 | 33 | 28.0 |
| H-FWRs | 78 | 99 | 91.0 |

---

## Failed Sequences Analysis

### Overview

**Total failures:** 453 sequences (0.32%)
**Log file:** `test_datasets/harvey/failed_sequences.txt`

### Root Cause

ANARCI (IMGT numbering) requires valid VHH domain structure to annotate sequences. Failures occur when:

1. **Missing CDR fields in source data** - Sequences with null/missing CDR1/CDR2/CDR3 values in official CSVs
2. **Non-IMGT-numberable sequences** - Sequences that don't conform to standard IMGT VHH structure
3. **Incomplete IMGT position data** - Gaps in columns 1-128 prevent full sequence reconstruction

### Impact

- 99.68% success rate exceeds Novo Nordisk's ">140,000" threshold (141,021 > 140,000)
- Failed sequences are acceptable data quality losses
- Label distribution remains balanced (49.1% / 50.9%)

### Failed Sequence IDs

First 10 failures: harvey_014076, harvey_014372, harvey_016053, harvey_022050, harvey_033141, harvey_044910, harvey_049180, harvey_049181, harvey_052106, harvey_052117

Full list: See `test_datasets/harvey/failed_sequences.txt` (453 IDs)

---

## Data Provenance

```
Official Harvey Repository
reference_repos/harvey_official_repo/backend/app/experiments/
    ↓
high_polyreactivity_high_throughput.csv (71,772)
low_polyreactivity_high_throughput.csv (69,702)
    ↓ scripts/convert_harvey_csvs.py
test_datasets/harvey.csv (141,474)
    ↓ preprocessing/process_harvey.py (ANARCI)
test_datasets/harvey/ fragment CSVs (141,021 each)
```

---

## References

- [Harvey et al. 2022] Harvey EP, et al. An in silico method to assess antibody fragment polyreactivity. *Nat Commun* 13, 7554 (2022). https://doi.org/10.1038/s41467-022-35276-4

- [Sakhnini et al. 2025] Sakhnini LI, et al. Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters. *bioRxiv* (2025). https://doi.org/10.1101/2025.04.28.650927

- [Official Harvey Repo] reference_repos/harvey_official_repo/backend/app/experiments/
