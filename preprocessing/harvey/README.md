# Harvey Dataset Preprocessing Pipeline

**Source:** Harvey et al. (2022) - Nanobody polyreactivity dataset
**Test Set:** 141,474 raw/processed nanobody sequences (141,021 ANARCI-validated; VHH only)

---

## Pipeline Overview

```
raw/*.csv → processed/harvey.csv → fragments/*.csv
  (Step 1)         (Step 2)
```

---

## Step 1: Convert Raw CSVs

**Script:** `step1_convert_raw_csvs.py`

**Purpose:** Combines high/low polyreactivity CSVs into single processed file.

**Input:**
- `data/test/harvey/raw/high_polyreactivity_high_throughput.csv` (71,772 sequences)
- `data/test/harvey/raw/low_polyreactivity_high_throughput.csv` (69,702 sequences)

**Output:**
- `data/test/harvey/processed/harvey.csv` (141,474 sequences)

**Run:**
```bash
python3 preprocessing/harvey/step1_convert_raw_csvs.py
```

**What it does:**
1. Extracts full sequences from IMGT position columns (1-128)
2. Extracts pre-annotated CDRs (CDR1_nogaps, CDR2_nogaps, CDR3_nogaps)
3. Assigns binary labels (0=low polyreactivity, 1=high polyreactivity)
4. Combines into single CSV

---

## Step 2: Extract Fragments

**Script:** `step2_extract_fragments.py`

**Purpose:** Annotate with ANARCI and extract VHH fragments (nanobody-specific).

**Input:**
- `data/test/harvey/processed/harvey.csv` (141,474 sequences)

**Output:**
- `data/test/harvey/fragments/*.csv` (6 fragment files)
  - VHH_only_harvey.csv
  - H-CDR1_harvey.csv
  - H-CDR2_harvey.csv
  - H-CDR3_harvey.csv
  - H-CDRs_harvey.csv (concatenated CDR1+2+3)
  - H-FWRs_harvey.csv (concatenated FWR1+2+3+4)

**Run:**
```bash
python3 preprocessing/harvey/step2_extract_fragments.py
```

**What it does:**
1. Annotates sequences with ANARCI (IMGT numbering scheme) via `preprocessing/fragment_utils.py`
2. Extracts CDR regions (CDR1, CDR2, CDR3) using strict IMGT boundaries
3. Extracts framework regions (FWR1, FWR2, FWR3, FWR4)
4. Creates fragment-specific CSV files for downstream analysis

**Note:** Uses shared `fragment_utils.py` to ensure consistent ANARCI annotation across all datasets. See [preprocessing/README.md](../README.md#shared-utilities-phase-d-refactoring---nov-2025) for details on shared utilities.

---

## Full Pipeline Execution

**Run both steps sequentially:**
```bash
# Step 1: Convert raw CSVs
python3 preprocessing/harvey/step1_convert_raw_csvs.py

# Step 2: Extract fragments
python3 preprocessing/harvey/step2_extract_fragments.py
```

---

## Dataset Statistics

- **Total sequences (processed/harvey.csv):** 141,474 nanobodies
- **Successfully annotated (fragments/*.csv):** 141,021 nanobodies (99.68%)
- **ANARCI failures:** 453 sequences (`data/test/harvey/fragments/failed_sequences.txt`)
- **High polyreactivity:** 71,772 (label=1)
- **Low polyreactivity:** 69,702 (label=0)
- **Sequence type:** VHH only (nanobodies, no light chain)
- **Fragment files:** 6 (VHH, 3 CDRs, concatenated CDRs, concatenated FWRs)

---

## Dependencies

- pandas
- numpy
- riot_na (ANARCI for annotation)
- tqdm

---

## References

- Harvey, E.P. et al. (2022). An in silico method to assess antibody fragment polyreactivity. *Nature Communications*.
  DOI: 10.1038/s41467-022-35276-4
- Sakhnini, L.I. et al. (2025). Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical
  Parameters. *bioRxiv*. DOI: 10.1101/2025.04.28.650927
- Dunbar, J. and Deane, C.M. (2016). ANARCI: antigen receptor numbering and receptor classification. *Bioinformatics*.
  DOI: 10.1093/bioinformatics/btv552

---

**Last Updated:** 2025-11-20 (added shared utilities documentation)
**Status:** ✅ Production Ready
