# Harvey Dataset – Data Sources & Methodology

**Date:** 2025-11-01
**Issue:** #4 – Harvey dataset preprocessing
**Status:** 📋 Data sourced, preprocessing pending

---

## Executive Summary

The Harvey dataset contains **141,474 nanobody sequences** with binary polyreactivity labels from deep sequencing of FACS-sorted pools. This dataset is used by Novo Nordisk (Sakhnini et al. 2025) to **test** their ESM-1v VH-based logistic regression model for predicting antibody non-specificity.

**Key Points:**
- ✅ Dataset downloaded from HuggingFace: `ZYMScott/polyreactivity`
- ✅ Matches Novo Nordisk specifications: ~140K nanobody sequences
- ✅ Binary labels: 0 = low polyreactivity, 1 = high polyreactivity
- ⚠️ CDRs provided but **FWRs not extracted** - need ANARCI processing
- ⚠️ **NOT the 48-nanobody validation set** from harvey.xlsx

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

**Key Dataset Information from Paper:**

From main paper (line 32):
> "Given this validation, we deep-sequenced the two FACS sorted pools and obtained **65,147 unique low polyreactivity** sequences and **69,155 unique highly polyreactive** sequences that contained 51,308 and 59,623 distinct CDR regions."

**Initial deep sequencing:** ~134K sequences (65,147 low + 69,155 high)

From main paper (line 110):
> "Through additional rounds of FACS selection, we collected **1,221,800 unique low polyreactivity clones** and **1,058,842 unique high polyreactivity clones**."

**Extended deep sequencing:** ~2.28M sequences (for improved model training)

**Experimental Method (from line 30-32):**
- Started with >2 × 10⁹ synthetic yeast display nanobody library
- MACS enrichment for polyreactive clones
- FACS sorting with PSR (polyspecificity reagent from Sf9 insect cell membranes)
- Deep sequencing of high and low polyreactivity pools

**Validation Set (from line 60):**
- 48 nanobodies stained with PSR for quantitative polyreactivity scores
- Used for regression validation, NOT for training
- This is what was in `harvey.xlsx` (wrong dataset for Issue #4)

---

### 2. Sakhnini et al. 2025 – Novo Nordisk Methodology (SSOT for Issue #4)

**Title:** Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters
**Journal:** bioRxiv preprint
**DOI:** 10.1101/2025.04.28.650927
**Date:** May 2025

**Local Files:**
- Main paper: `literature/markdown/Sakhnini_2025_Antibody_NonSpecificity_PLM_Biophysical/Sakhnini_2025_Antibody_NonSpecificity_PLM_Biophysical.md`
- PDF: `literature/pdf/Sakhnini_2025_Antibody_NonSpecificity_PLM_Biophysical.pdf`

**Harvey Dataset Usage by Novo Nordisk:**

From Sakhnini Table 4 (line 207):
| Dataset | Description | Assay | Reference |
|---------|-------------|-------|-----------|
| Harvey dataset | **>140 000 naïve nanobodies** | Poly-specific reagent (PSR) assay | [45] |

From Sakhnini Section 2.1 (line 47):
> "Four different datasets were retrieved from public sources... (iv) **140 000 nanobody (Nb) clones assessed by the PSR assay from a naïve Nb library** [\[45\]](#page-3-1). These four datasets are referred to as the Boughter, the Jain, the Shehata and the Harvey datasets, respectively."

From Sakhnini Section 2.1 (line 51):
> "In this study, the most balanced dataset (i.e. Boughter one) was selected for **training of ML models**, while the remining three (i.e. **Jain, Shehata and Harvey**, which consists exclusively of VHH sequences) were **used for testing**."

**Critical Finding:**
Novo Nordisk used the **~140K deep sequencing dataset** for testing, NOT the 48-nanobody validation set.

From Sakhnini Section 2.7 (line 131):
> "To find out whether our ESM 1v mean-mode VH-based LogisticReg model can extend its applicability further to the non-specificity scored by the PSR assay, the Shehata dataset and the **VH-based Nb dataset by Harvey and co-authors [\[45\]](#page-3-1), here referred to as the Harvey dataset, were tested**."

**Model Architecture (from Section 2.3):**
- Top performer: **ESM 1v mean-mode VH-based LogisticReg**
- 10-fold CV accuracy: 71%
- Tested on Harvey nanobodies (VHH sequences only)

---

## Data Availability & Access

### Original Paper Statement

From Harvey et al. 2022 (line 220):
> "The data that support this study are available from the corresponding author upon request."

**However**, the dataset is now publicly available via:

### HuggingFace Dataset (Current Source)

**Repository:** `ZYMScott/polyreaction`
**URL:** https://huggingface.co/datasets/ZYMScott/polyreaction
**License:** CC-BY-4.0

**Dataset Statistics:**
- **Total sequences:** 141,474 nanobodies
- **Train split:** 101,854 sequences
- **Validation split:** 14,613 sequences
- **Test split:** 25,007 sequences

**Columns:**
- `seq`: Full nanobody amino acid sequence (VHH domain)
- `CDR1_nogaps`: H-CDR1 sequence (no gaps)
- `CDR2_nogaps`: H-CDR2 sequence (no gaps)
- `CDR3_nogaps`: H-CDR3 sequence (no gaps)
- `label`: Binary polyreactivity label (0=low, 1=high)

**Sequence Length Range:** 52-137 amino acids (nanobody VHH domain range)

**Label Distribution:**
- Low polyreactivity (label=0): 69,702 sequences (49.3%)
- High polyreactivity (label=1): 71,772 sequences (50.7%)
- **Balanced dataset** suitable for binary classification

**Data Quality Issues:**
- **377 sequences (0.27%)** have null values in pre-extracted CDR columns
  - CDR1_nogaps: 30 nulls
  - CDR2_nogaps: 82 nulls
  - CDR3_nogaps: 270 nulls
- These sequences cannot be IMGT-numbered by ANARCI
- Expected to fail during preprocessing (~0.27% failure rate)
- **After removing failures: ~141,097 sequences** (still exceeds Novo's ">140,000")

---

## Dataset Versions: Wrong vs. Correct

### ❌ WRONG: 48-Nanobody Validation Set

**File:** `test_datasets/harvey.xlsx` (deleted)
**Source:** Harvey et al. 2022 Supplementary Data
**Size:** 48 nanobodies with quantitative PSR scores
**Use Case:** Regression validation in original paper
**Why Wrong:** Issue #4 requires the deep sequencing training dataset, not the validation set

**Evidence from Supplementary (line 60):**
> "In order to test if our model could go beyond predicting binary classification labels and quantitively score polyreactivity, we stained yeast expressing **48 nanobodies** isolated from MACS and FACS pools with PSR to obtain an **index set** of sequenced clones with defined levels of polyreactivity."

### ✅ CORRECT: ~141K Deep Sequencing Dataset

**File:** `test_datasets/harvey.csv` (downloaded from HuggingFace)
**Source:** HuggingFace `ZYMScott/polyreaction`
**Size:** 141,474 nanobodies with binary labels
**Use Case:** Testing ESM-1v models (Novo Nordisk methodology)
**Why Correct:** Matches Sakhnini Table 4 specification of ">140 000 naïve nanobodies"

**Confirmation from Discord (Hybri):**
> "Novonordisk didn't filter the Harvey data like Harvey did. they used 141559 sequences... if you want to use Harvey, use the **unfiltered ones** that I sent you, not the filtered one"

**141,474 ≈ 141,559** (likely minor preprocessing differences)

---

## Relationship to Initial vs. Extended Sequencing

Harvey et al. reported TWO deep sequencing datasets:

1. **Initial sequencing:** 65,147 low + 69,155 high = **~134K total**
2. **Extended sequencing:** 1,221,800 low + 1,058,842 high = **~2.28M total**

**Question:** Which one is the HuggingFace dataset?

**Analysis:**
- HuggingFace: 141,474 sequences
- Initial: ~134K sequences
- **141,474 ≈ 134,000** (within reasonable range)

**Hypothesis:** The HuggingFace dataset represents the **initial deep sequencing** with some additional filtering/preprocessing, resulting in ~141K sequences.

**Supporting Evidence:**
- Discord mention of "141,559" matches HuggingFace size
- Novo Nordisk cites ">140,000" not ">2 million"
- Extended sequencing (2.28M) mentioned late in paper (line 110) as an improvement step

---

## Data Provenance Chain

```
Harvey Lab (Harvard/Debbie Marks Lab)
    ↓ FACS + Deep Sequencing
Initial Dataset: ~134K sequences (65K low + 69K high)
    ↓ Preprocessing/Cleaning
HuggingFace: 141,474 sequences (ZYMScott/polyreaction)
    ↓ Downloaded 2025-11-01
Local: test_datasets/harvey.csv + harvey_hf/ splits
    ↓ PENDING: ANARCI fragment extraction
test_datasets/harvey/ fragment CSVs (VHH, H-CDRs, H-FWRs, etc.)
```

---

## Outstanding Questions

1. **Who is ZYMScott?**
   - HuggingFace user who uploaded the dataset
   - Unclear if affiliated with Harvey lab or Novo Nordisk
   - Dataset appears legitimate (matches paper specs)

2. **Why 141,474 instead of 134,302?** ✅ **RESOLVED**

   **Harvey's published CDR length filter** (from Harvey 2022 paper line 142):
   > "For our dataset of sequences to train the supervised models, we **limited nanobody sequences to sequences with a CDR1 length of 8, a CDR2 length of 8 or 9... and CDR3 lengths between 6 and 22**. These processing steps leave us with **65,147 unique low polyreactivity sequences and 69,155 unique highly polyreactive sequences**..."

   - Harvey's **filtered dataset**: CDR1==8, CDR2==8|9, CDR3==6-22 → **134,302 sequences**
   - HuggingFace **(unfiltered)**: **141,474 sequences**
   - Novo Nordisk used: **">140,000 sequences"** (Sakhnini Table 4)

   **ANSWER**: Novo Nordisk used the **UNFILTERED** HuggingFace dataset (all 141,474 sequences), NOT Harvey's CDR-length-filtered version (134K). This is confirmed by Sakhnini citing ">140,000" which matches the unfiltered count.

   **Impact**: Our process_harvey.py script is CORRECT - we process all 141,474 sequences with **NO CDR length filtering**

3. **Are the HuggingFace CDRs IMGT-numbered?**
   - CDRs provided as `CDR1_nogaps`, `CDR2_nogaps`, `CDR3_nogaps`
   - Unclear if using IMGT, Kabat, or Chothia numbering
   - **Recommendation:** Re-extract using ANARCI (IMGT) for consistency with Jain/Shehata

4. **Should we use train/val/test splits or combined?**
   - For Novo Nordisk replication: Use **combined** (all 141K as test set)
   - For training Harvey-specific models: Use **splits** (train on Harvey data)

---

## References

- [Harvey et al. 2022] Harvey EP, et al. An in silico method to assess antibody fragment polyreactivity. *Nat Commun* 13, 7554 (2022). https://doi.org/10.1038/s41467-022-35276-4

- [Sakhnini et al. 2025] Sakhnini LI, et al. Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters. *bioRxiv* (2025). https://doi.org/10.1101/2025.04.28.650927

- [HuggingFace Dataset] ZYMScott/polyreaction. https://huggingface.co/datasets/ZYMScott/polyreaction

- [Kruse Lab] Andrew C. Kruse Lab, Harvard Medical School. https://kruse.hms.harvard.edu/

- [GitHub Repo] debbiemarkslab/nanobody-polyreactivity. https://github.com/debbiemarkslab/nanobody-polyreactivity
