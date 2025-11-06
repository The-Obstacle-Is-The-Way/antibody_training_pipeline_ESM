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
- `test_datasets/shehata/raw/shehata-mmc2.xlsx`

**Output:**
- `test_datasets/shehata/processed/shehata.csv` (398 antibodies)

**Run:**
```bash
python3 preprocessing/shehata/step1_convert_excel_to_csv.py
```

**What it does:**
1. Loads Shehata Excel supplementary file (mmc2.xlsx)
2. Extracts VH and VL sequences
3. Extracts PSR assay measurements
4. Assigns binary labels based on PSR threshold
5. Exports standardized CSV

---

## Step 2: Extract Fragments

**Script:** `step2_extract_fragments.py`

**Purpose:** Annotate with ANARCI and extract paired antibody fragments.

**Input:**
- `test_datasets/shehata/processed/shehata.csv` (398 antibodies)

**Output:**
- `test_datasets/shehata/fragments/*.csv` (16 fragment files)

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
1. Annotates VH and VL sequences with ANARCI (IMGT numbering)
2. Extracts CDR and FWR regions using IMGT boundaries
3. Creates 16 fragment-specific CSV files
4. Preserves PSR measurements and labels

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

## Assay-Specific Threshold

**PSR Threshold:** 0.549 (optimized for PSR assay)

**Note:** Different assays require different classification thresholds:
- ELISA (Jain): 0.5 (default)
- PSR (Shehata): 0.549

See `scripts/analysis/analyze_threshold_optimization.py` for details.

---

## Dependencies

- pandas
- numpy
- openpyxl (for Excel reading)
- riot_na (ANARCI for annotation)
- tqdm

---

## References

- **Shehata et al. (2019):** [Citation needed - add when available]
- **Sakhnini et al. (2025):** Prediction of Antibody Non-Specificity using Protein Language Models
- **ANARCI:** IMGT numbering scheme for antibody annotation

---

**Last Updated:** 2025-11-05
**Status:** ✅ Production Ready
