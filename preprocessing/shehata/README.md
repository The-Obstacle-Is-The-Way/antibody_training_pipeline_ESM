# Shehata Dataset Preprocessing Pipeline

**Source:** Shehata et al. (2019) - PSR assay dataset
**Test Set:** 398 human antibodies with polyspecific reagent (PSR) measurements

---

## Pipeline Overview

```
raw/*.xlsx → processed/shehata.csv → fragments/*.csv
  (Step 1)         (Step 2)
```

---

## Step 1: Convert Excel to CSV

**Script:** `step1_convert_excel_to_csv.py`

**Purpose:** Convert Shehata Excel file to standardized CSV format.

**Input:**
- `data/test/shehata/raw/shehata-mmc2.xlsx`

**Output:**
- `data/test/shehata/processed/shehata.csv` (398 antibodies)

**Run:**
```bash
python3 preprocessing/shehata/step1_convert_excel_to_csv.py
```

**What it does:**
1. Loads Shehata Excel supplementary file (mmc2.xlsx)
2. Extracts VH and VL sequences
3. Extracts PSR assay measurements
4. Assigns binary labels by thresholding PSR scores (default: 98.24th percentile, 7/398)
5. Exports standardized CSV

---

## Step 2: Extract Fragments

**Script:** `step2_extract_fragments.py`

**Purpose:** Annotate with ANARCI and extract paired antibody fragments.

**Input:**
- `data/test/shehata/processed/shehata.csv` (398 antibodies)

**Output:**
- `data/test/shehata/fragments/*.csv` (16 fragment files)

**Fragment types:**
1. VH_only, VL_only (full variable domains)
2. H-CDR1, H-CDR2, H-CDR3 (heavy chain CDRs)
3. L-CDR1, L-CDR2, L-CDR3 (light chain CDRs)
4. H-CDRs, L-CDRs (concatenated CDRs per chain)
5. H-FWRs, L-FWRs (concatenated frameworks per chain)
6. VH+VL (paired variable domains)
7. All-CDRs, All-FWRs (all concatenated)
8. Full (alias for VH+VL)

**Run:**
```bash
python3 preprocessing/shehata/step2_extract_fragments.py
```

**What it does:**
1. Annotates VH and VL sequences with ANARCI (IMGT numbering) via `preprocessing/fragment_utils.py`
2. Extracts CDR and FWR regions using strict IMGT boundaries
3. Creates 16 fragment-specific CSV files for downstream analysis
4. Preserves PSR measurements and labels

**Note:** Uses shared `fragment_utils.py` to ensure consistent ANARCI annotation across all datasets. See [preprocessing/README.md](../README.md#shared-utilities-phase-d-refactoring---nov-2025) for details on shared utilities.

---

## Full Pipeline Execution

**Run both steps sequentially:**
```bash
# Step 1: Convert Excel to CSV
python3 preprocessing/shehata/step1_convert_excel_to_csv.py

# Step 2: Extract fragments
python3 preprocessing/shehata/step2_extract_fragments.py
```

---

## Dataset Statistics

- **Total sequences:** 398 human antibodies
- **Sequence type:** Paired VH+VL (full antibodies)
- **Assay:** PSR (polyspecific reagent)
- **Fragment files:** 16 (all combinations of CDRs, FWRs, paired/unpaired)

---

## Labeling Threshold (Dataset)

**Default labeling:** 98.24th percentile of PSR scores (treats 7/398 antibodies as non-specific)

This is the threshold used in `preprocessing/shehata/step1_convert_excel_to_csv.py` when generating the `label` column.

---

## Prediction Decision Threshold (Evaluation)

**PSR decision threshold:** 0.5495 (binarizes predicted probabilities for PSR datasets)

**Note:** This is a model decision threshold (not a dataset labeling rule). It is used to reproduce Sakhnini et al. (2025)
benchmarks on PSR datasets (Shehata/Harvey) when testing ELISA-trained models.

See `docs/research/assay-thresholds.md` for details.

---

## Dependencies

- pandas
- numpy
- openpyxl (for Excel reading)
- riot_na (ANARCI for annotation)
- tqdm

---

## References

- Shehata, L. et al. (2019). Affinity maturation enhances antibody specificity but compromises conformational stability.
  *Cell Reports*. DOI: 10.1016/j.celrep.2019.08.056
- Sakhnini, L.I. et al. (2025). Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical
  Parameters. *bioRxiv*. DOI: 10.1101/2025.04.28.650927
- Dunbar, J. and Deane, C.M. (2016). ANARCI: antigen receptor numbering and receptor classification. *Bioinformatics*.
  DOI: 10.1093/bioinformatics/btv552

---

**Last Updated:** 2025-11-20 (added shared utilities documentation)
**Status:** ✅ Production Ready
