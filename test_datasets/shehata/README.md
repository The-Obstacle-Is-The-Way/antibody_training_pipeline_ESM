# Shehata Dataset

**DO NOT MODIFY** the `raw/` directory - Original sources only

---

## Overview

High-quality polyspecificity dataset from affinity maturation study. 398 paired VH+VL antibody sequences with PSR (Polyspecific Reagent) scores and binary non-specificity labels.

**Use case:** External test set for non-specificity prediction models.

---

## Quick Start

```bash
# Full paired sequences (398 antibodies)
test_datasets/shehata/processed/shehata.csv

# VH-only sequences for testing
test_datasets/shehata/fragments/VH_only_shehata.csv

# Other fragments (16 total)
test_datasets/shehata/fragments/*.csv
```

---

## Directory Structure

```
shehata/
├── README.md              (this file)
├── raw/                   Original Excel files (DO NOT MODIFY)
│   ├── README.md
│   ├── shehata-mmc2.xlsx  Main data (402 rows)
│   ├── shehata-mmc3.xlsx  (Unused - archived)
│   ├── shehata-mmc4.xlsx  (Unused - archived)
│   └── shehata-mmc5.xlsx  (Unused - archived)
├── processed/             Converted and processed datasets
│   ├── README.md
│   └── shehata.csv        Full paired VH+VL (398 antibodies)
├── canonical/             Final benchmarks
│   └── README.md          (Empty - Shehata is external test set)
└── fragments/             Region-specific extracts (16 files)
    ├── README.md
    ├── Full_shehata.csv, VH_only_shehata.csv, VL_only_shehata.csv
    ├── H-CDR1/2/3_shehata.csv, L-CDR1/2/3_shehata.csv
    ├── H-CDRs/FWRs_shehata.csv, L-CDRs/FWRs_shehata.csv
    └── All-CDRs/FWRs_shehata.csv
```

---

## Data Flow

```
raw/shehata-mmc2.xlsx (402 rows)
  ↓ [convert_shehata_excel_to_csv.py]
processed/shehata.csv (398 antibodies, 4 removed)
  ↓ [process_shehata.py + ANARCI annotation]
fragments/*.csv (16 fragment types)
```

**Removed:** 4 sequences with incomplete pairing or missing PSR data

---

## Label Information

**Binary classification:** 0 = specific, 1 = non-specific

**Label distribution:**
- 391 specific (label=0, PSR < 98.24th percentile)
- 7 non-specific (label=1, PSR ≥ 98.24th percentile)

**Threshold:** 98.24th percentile of PSR score (stringent cutoff from Sakhnini 2025)

**Highly imbalanced:** 98.2% specific, 1.8% non-specific

---

## Citations

**Dataset Source:**
Shehata L, Thaventhiran JED, Engelhardt KR, et al. (2019). "Affinity Maturation Enhances Antibody Specificity but Compromises Conformational Stability." *Cell Reports* 28(13):3300-3308.e4.
DOI: 10.1016/j.celrep.2019.08.056

**Methodology Source:**
Sakhnini A, et al. (2025). "Antibody Non-Specificity Prediction using Protein Language Models and Biophysical Features." *Cell*.
DOI: 10.1016/j.cell.2024.12.025

---

## Regenerating Data

To regenerate all derived files from raw sources:

```bash
# Step 1: Convert Excel to CSV
python3 preprocessing/shehata/step1_convert_excel_to_csv.py
# Creates: processed/shehata.csv (398 antibodies)

# Step 2: Extract fragments
python3 preprocessing/shehata/step2_extract_fragments.py
# Creates: fragments/*.csv (16 fragment types)
```

---

## Verification

```bash
# Check file counts
echo "Raw files (4):" && ls -1 test_datasets/shehata/raw/*.xlsx | wc -l
echo "Processed files (1):" && ls -1 test_datasets/shehata/processed/*.csv | wc -l
echo "Fragment files (16):" && ls -1 test_datasets/shehata/fragments/*.csv | wc -l

# Validate conversion
python3 scripts/validation/validate_shehata_conversion.py

# Test embedding compatibility
python3 tests/test_shehata_embedding_compatibility.py

# CRITICAL: Check for gap characters (P0 blocker)
grep -c '\-' test_datasets/shehata/fragments/*.csv | grep -v ':0'
# Should return NOTHING (all files should have 0 gaps)
```

---

## Key Details

- **Total antibodies:** 398
- **Sequence type:** Paired VH+VL
- **Annotation:** ANARCI with IMGT numbering
- **Fragment types:** 16 (VH-only, VL-only, CDRs, FWRs, Full)
- **All files:** 398 antibodies + 1 header = 399 lines
- **No gap characters:** Gap-free sequences (see P0 blocker history)

---

## See Also

- `raw/README.md` - Original Excel file details
- `processed/README.md` - CSV conversion and filtering
- `fragments/README.md` - Fragment extraction methodology
- `canonical/README.md` - Why canonical/ is empty for Shehata
- `docs/shehata/` - Complete documentation
