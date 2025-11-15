# Harvey Dataset Preprocessing Pipeline

**Source:** Harvey et al. (2022) - Nanobody polyreactivity dataset
**Test Set:** 141,474 nanobody sequences (VHH only)

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
1. Annotates sequences with ANARCI (IMGT numbering scheme)
2. Extracts CDR regions (CDR1, CDR2, CDR3) using IMGT boundaries
3. Extracts framework regions (FWR1, FWR2, FWR3, FWR4)
4. Creates fragment-specific CSV files

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

- **Total sequences:** 141,474 nanobodies
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

- **Harvey et al. (2022):** [Citation needed - add when available]
- **Sakhnini et al. (2025):** Prediction of Antibody Non-Specificity using Protein Language Models
- **ANARCI:** IMGT numbering scheme for antibody annotation

---

**Last Updated:** 2025-11-05
**Status:** ✅ Production Ready
