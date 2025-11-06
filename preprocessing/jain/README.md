# Jain Dataset Preprocessing Pipeline

**Source:** Jain et al. (2017) PNAS - Biophysical properties of clinical-stage antibodies
**Test Set:** 86 antibodies (Novo Nordisk parity benchmark)

---

## Pipeline Overview

```
raw/*.xlsx → processed/*.csv → canonical/*.csv
  (Step 1)         (Step 2)
```

---

## Step 1: Convert Excel to CSV

**Script:** `step1_convert_excel_to_csv.py`

**Purpose:** Convert Jain Excel files to standardized CSV format using ELISA-only methodology.

**Input:**
- `test_datasets/jain/raw/Private_Jain2017_ELISA_indiv.xlsx`
- `test_datasets/jain/raw/jain-pnas.1616408114.sd01.xlsx`
- `test_datasets/jain/raw/jain-pnas.1616408114.sd02.xlsx`
- `test_datasets/jain/raw/jain-pnas.1616408114.sd03.xlsx`

**Output:**
- `test_datasets/jain/processed/jain_with_private_elisa_FULL.csv` (137 antibodies)
- `test_datasets/jain/processed/jain_sd01.csv`
- `test_datasets/jain/processed/jain_sd02.csv`
- `test_datasets/jain/processed/jain_sd03.csv`

**Run:**
```bash
python3 preprocessing/jain/step1_convert_excel_to_csv.py
```

**What it does:**
1. Loads private ELISA data (137 antibodies)
2. Loads public supplement data (SD01, SD02, SD03)
3. Applies ELISA-only flag calculation (0-6 range, NOT total flags 0-10)
4. Exports processed CSVs for downstream use

**Key Methodology:**
- **ELISA-only flags:** Uses ONLY 6 ELISA antigens (NOT all 10 assays)
- **Threshold:** ≥4 ELISA flags = non-specific
- **Corrected approach:** Fixes previous "total_flags" bug

---

## Step 2: Preprocess P5e-S2 (Novo Parity)

**Script:** `step2_preprocess_p5e_s2.py`

**Purpose:** Apply P5e-S2 methodology to achieve EXACT Novo Nordisk parity.

**Input:**
- `test_datasets/jain/processed/jain_with_private_elisa_FULL.csv` (137 antibodies)
- `test_datasets/jain/processed/jain_sd03.csv` (PSR/AC-SINS data)

**Output:**
- `test_datasets/jain/processed/jain_ELISA_ONLY_116.csv` (116 antibodies)
- `test_datasets/jain/canonical/jain_86_novo_parity.csv` (86 antibodies)

**Run:**
```bash
python3 preprocessing/jain/step2_preprocess_p5e_s2.py
```

**What it does:**

**Pipeline:**
```
137 antibodies (FULL)
  ↓ Remove ELISA 1-3 (mild aggregators)
116 antibodies (ELISA_ONLY_116.csv) ✅ OUTPUT 1
  ↓ Reclassify 5 spec→nonspec (3 PSR>0.4 + eldelumab + infliximab)
89 spec / 27 nonspec
  ↓ Remove 30 by PSR primary, AC-SINS tiebreaker
86 antibodies (59 spec / 27 nonspec) ✅ OUTPUT 2
```

**Result:** Confusion matrix [[40, 19], [10, 17]] - **EXACT MATCH** (66.28% accuracy)

**Method:** P5e-S2 (PSR reclassification + PSR/AC-SINS removal)

---

## Full Pipeline Execution

**Run both steps sequentially:**
```bash
# Step 1: Convert Excel to CSV
python3 preprocessing/jain/step1_convert_excel_to_csv.py

# Step 2: Preprocess to Novo parity
python3 preprocessing/jain/step2_preprocess_p5e_s2.py
```

---

## Dataset Statistics

- **Source:** 137 antibodies with private ELISA data
- **After ELISA filtering:** 116 antibodies
- **Final benchmark:** 86 antibodies (59 specific / 27 non-specific)
- **Novo parity:** 66.28% accuracy (EXACT match)

---

## Methodology Notes

**CRITICAL:** This preprocessing uses **ELISA-only flags** (0-6 range), NOT total flags (0-10).

**Evidence:**
- Figure S13: x-axis shows "ELISA flag" (singular) with range 0-6
- Table 2: "ELISA with a panel of 6 ligands"
- Paper text: "non-specificity ELISA flags"

**Retired Approach:**
- Previous 94→86 methodology (VH length outliers + biology removals) did NOT match Novo
- total_flags approach was INCORRECT (used all 10 assays instead of 6 ELISA)

---

## Dependencies

- pandas
- numpy
- openpyxl (for Excel reading)

---

## References

- **Jain et al. (2017) PNAS:** Biophysical properties of the clinical-stage antibody landscape
- **Sakhnini et al. (2025):** Prediction of Antibody Non-Specificity using Protein Language Models

---

**Last Updated:** 2025-11-05
**Status:** ✅ Production Ready (Novo Parity Achieved)
