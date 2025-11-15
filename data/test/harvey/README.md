# Harvey Dataset

**DO NOT MODIFY** the `raw/` directory - Original sources only

---

## Overview

Large-scale polyreactivity dataset from high-throughput nanobody screening. 141,021 VHH (nanobody) sequences with binary polyreactivity labels (0=low, 1=high).

**Use case:** Training set for polyreactivity prediction models, external validation against Novo Nordisk benchmark.

---

## Quick Start

```bash
# Full nanobody dataset (141,474 antibodies before ANARCI annotation)
data/test/harvey/processed/harvey.csv

# VHH-only sequences for testing (141,021 after ANARCI)
data/test/harvey/fragments/VHH_only_harvey.csv

# Other fragments (6 total - nanobody-specific)
data/test/harvey/fragments/*.csv
```

---

## Directory Structure

```
harvey/
├── README.md              (this file)
├── raw/                   Original CSV files (DO NOT MODIFY)
│   ├── README.md
│   ├── high_polyreactivity_high_throughput.csv (71,772)
│   ├── low_polyreactivity_high_throughput.csv (69,702)
│   └── low_throughput_polyspecificity_scores_w_exp.csv (48 - optional)
├── processed/             Converted and combined datasets
│   ├── README.md
│   └── harvey.csv         Combined VHH dataset (141,474 nanobodies)
├── canonical/             Training benchmarks
│   └── README.md          (Empty - full dataset already balanced)
└── fragments/             Region-specific extracts (6 files)
    ├── README.md
    ├── VHH_only_harvey.csv (141,021 - most common use case)
    ├── H-CDR1/2/3_harvey.csv
    ├── H-CDRs/FWRs_harvey.csv
    └── failed_sequences.txt (453 ANARCI annotation failures)
```

---

## Data Flow

```
raw/high_polyreactivity_high_throughput.csv (71,772)
raw/low_polyreactivity_high_throughput.csv (69,702)
  ↓ [preprocessing/harvey/step1_convert_raw_csvs.py]
processed/harvey.csv (141,474 combined)
  ↓ [preprocessing/harvey/step2_extract_fragments.py + ANARCI annotation]
fragments/*.csv (141,021 - 453 ANARCI failures)
```

**ANARCI failures:** 453 sequences (0.32% failure rate) - logged in `fragments/failed_sequences.txt`

---

## Label Information

**Binary classification:** 0 = low polyreactivity, 1 = high polyreactivity

**Label distribution (processed/harvey.csv):**
- 69,702 low polyreactivity (label=0, 49.1%)
- 71,772 high polyreactivity (label=1, 50.9%)

**Balanced dataset:** Nearly 50/50 split, no resampling needed

**After ANARCI annotation (fragments/):**
- 141,021 total sequences
- 453 failures (0.32%)
- Label balance preserved

---

## Citations

**Dataset Source:**

Harvey, E.P., et al. (2022). "A biophysical basis for mucophilic antigen binding." *Journal of Experimental Medicine* 219(3):e20211671.
DOI: 10.1084/jem.20211671

Mason, D.M., et al. (2021). "Optimization of therapeutic antibodies by predicting antigen specificity from antibody sequence via deep learning." *Nature Biomedical Engineering* 5:600-612.
DOI: 10.1038/s41551-021-00699-9

**Methodology Source:**

Sakhnini, A., et al. (2025). "Antibody Non-Specificity Prediction using Protein Language Models and Biophysical Features." *Cell*.
DOI: 10.1016/j.cell.2024.12.025

---

## Regenerating Data

To regenerate all derived files from raw sources:

```bash
# Step 1: Convert raw CSVs to combined dataset
python3 preprocessing/harvey/step1_convert_raw_csvs.py
# Creates: processed/harvey.csv (141,474 nanobodies)

# Step 2: Extract fragments with ANARCI
python3 preprocessing/harvey/step2_extract_fragments.py
# Creates: fragments/*.csv (6 fragment types, 141,021 sequences)
```

---

## Verification

```bash
# Check file counts
echo "Raw files (3):" && ls -1 data/test/harvey/raw/*.csv | wc -l
echo "Processed files (1):" && ls -1 data/test/harvey/processed/*.csv | wc -l
echo "Fragment files (6):" && ls -1 data/test/harvey/fragments/*.csv | wc -l

# Validate fragments
python3 scripts/validation/validate_fragments.py

# Test embedding compatibility (P0 blocker check)
python3 tests/test_harvey_embedding_compatibility.py

# CRITICAL: Check for gap characters (should return NOTHING)
grep -c '\-' data/test/harvey/fragments/*.csv | grep -v ':0'
# Should return NOTHING (all files should have 0 gaps)
```

---

## Key Details

- **Total nanobodies:** 141,474 (before ANARCI) → 141,021 (after ANARCI)
- **Sequence type:** VHH only (nanobodies - no light chain)
- **Annotation:** ANARCI with IMGT numbering
- **Fragment types:** 6 (VHH-only, H-CDR1/2/3, H-CDRs, H-FWRs)
- **All fragment files:** 141,021 nanobodies + 1 header = 141,022 lines
- **No gap characters:** Gap-free sequences (P0 fix applied)
- **ANARCI failures:** 453 sequences (0.32%) logged in failed_sequences.txt

---

## See Also

- `raw/README.md` - Original CSV file details
- `processed/README.md` - CSV conversion and label assignment
- `fragments/README.md` - Fragment extraction methodology
- `canonical/README.md` - Why canonical/ is empty for Harvey
- `docs/harvey/` - Complete documentation
- `docs/harvey/HARVEY_P0_FIX_REPORT.md` - Gap character fix history

---

**Last Updated:** 2025-11-05
**Cleanup:** leroy-jenkins/harvey-cleanup branch
