# Preprocessing Guide

This guide covers how to preprocess antibody datasets for training and testing with the pipeline.

---

## Overview

Preprocessing transforms raw antibody data into the canonical CSV format required by the pipeline:

```csv
sequence,label
EVQLVESGGGLVQPGGSLRLSCAASGFTFS,0
QVQLQESGPGLVKPSQTLSLTCTVSGGSLS,1
```

**Pipeline Steps:**

1. **Data Acquisition** - Download raw data (Excel, CSV, FASTA)
2. **Format Conversion** - Convert to canonical CSV format
3. **Sequence Extraction** - Extract antibody fragments (VH, CDRs, FWRs)
4. **Quality Control** - Validate sequences, check labels
5. **Fragment Generation** - Create fragment-specific datasets

---

## When to Preprocess

You need to preprocess data if:

- ✅ **Using provided datasets for the first time** - Raw data → canonical format
- ✅ **Adding a new dataset** - Your own antibody data
- ✅ **Extracting new fragments** - VH, CDRs, FWRs from existing datasets
- ✅ **Updating data** - New version of published dataset

You DON'T need to preprocess if:

- ❌ **Using pre-processed canonical files** - Already in `train_datasets/` or `test_datasets/canonical/`
- ❌ **Using provided fragment files** - Already in `test_datasets/fragments/`

---

## Quick Preprocessing Commands

### Boughter Dataset (Training Set)

```bash
# Stage 1: Translate DNA to protein sequences
python3 preprocessing/boughter/stage1_dna_translation.py

# Stage 2 & 3: Annotate sequences (ANARCI) + Quality control
python3 preprocessing/boughter/stage2_stage3_annotation_qc.py
```

**Outputs:**
- `train_datasets/boughter/annotated/VH_only_boughter.csv` - VH sequences (1,065 rows)
- `train_datasets/boughter/annotated/*_boughter.csv` - 16 fragment CSVs (H-CDRs, L-CDRs, etc.)

**Note:** Training subset (914 sequences) selected from VH_only_boughter.csv based on polyreactivity labels.

---

### Jain Dataset (Test Set - Novo Parity)

```bash
# Step 1: Convert Excel to CSV
python3 preprocessing/jain/step1_convert_excel_to_csv.py

# Step 2: Preprocess P5e-S2 subset (Novo parity benchmark)
python3 preprocessing/jain/step2_preprocess_p5e_s2.py
```

**Outputs:**

**Canonical files** (original column names):
- `test_datasets/jain/canonical/jain_86_novo_parity.csv` - Full data (columns: `id`, `vh_sequence`, `vl_sequence`, ...)
- `test_datasets/jain/canonical/VH_only_jain_86_p5e_s2.csv` - VH only (columns: `id`, `vh_sequence`, `label`)

**Fragment files** (standardized columns):
- `test_datasets/jain/fragments/VH_only_jain.csv` - VH fragment (columns: `id`, `sequence`, `label`) **← Use for testing**
- Additional fragment files in `test_datasets/jain/fragments/` (H-CDRs, L-CDRs, etc.)

**Column Naming:**
- **Canonical files** use `vh_sequence`/`vl_sequence` (original source data)
- **Fragment files** use `sequence` (standardized for training/testing)

---

### Harvey Dataset (Nanobody Test Set)

```bash
# Step 1: Combine raw CSVs
python3 preprocessing/harvey/step1_convert_raw_csvs.py

# Step 2: Extract nanobody fragments (VHH, CDRs, FWRs)
python3 preprocessing/harvey/step2_extract_fragments.py
```

**Outputs:**

**Processed files:**
- `test_datasets/harvey/processed/harvey.csv` - Combined raw data (intermediate)

**Fragment files** (standardized columns):
- `test_datasets/harvey/fragments/VHH_only_harvey.csv` - Full VHH (columns: `id`, `sequence`, `label`, ...)
- `test_datasets/harvey/fragments/H-CDR1_harvey.csv` - Individual CDRs
- `test_datasets/harvey/fragments/H-CDRs_harvey.csv` - Concatenated CDRs
- `test_datasets/harvey/fragments/H-FWRs_harvey.csv` - Concatenated FWRs

**Column Naming:**
- All fragment files use standardized `sequence` column (ready for testing)

**Note:** Fragment naming pattern: `{fragmentName}_harvey.csv` (not `harvey_{fragmentName}.csv`)

---

### Shehata Dataset (PSR Assay Test Set)

```bash
# Step 1: Convert Excel to CSV
python3 preprocessing/shehata/step1_convert_excel_to_csv.py

# Step 2: Extract antibody fragments (VH, CDRs, FWRs)
python3 preprocessing/shehata/step2_extract_fragments.py
```

**Outputs:**

**Processed files:**
- `test_datasets/shehata/processed/shehata.csv` - Combined processed data (intermediate)

**Fragment files** (standardized columns):
- `test_datasets/shehata/fragments/VH_only_shehata.csv` - VH domain (columns: `id`, `sequence`, `label`, ...)
- `test_datasets/shehata/fragments/H-CDRs_shehata.csv` - Heavy CDRs
- `test_datasets/shehata/fragments/All-CDRs_shehata.csv` - All CDRs
- (16 fragment files total, pattern: `{fragmentName}_shehata.csv`)

**Column Naming:**
- All fragment files use standardized `sequence` column (ready for testing)

**Note:** No canonical/ directory with CSVs - only fragments/ (processed outputs only)

---

## Canonical CSV Format

All datasets must be converted to this format:

```csv
sequence,label
EVQLVESGGGLVQPGGSLRLSCAASGFTFS,0
QVQLQESGPGLVKPSQTLSLTCTVSGGSLS,1
```

**Required Columns:**

- `sequence`: Antibody amino acid sequence (single-letter code)
- `label`: Binary classification (0=specific, 1=non-specific)

**Optional Columns:**

- `id`: Unique identifier
- `name`: Antibody name
- `source`: Data source
- `assay`: Assay type (ELISA, PSR)
- Any other metadata

---

## Preprocessing Workflows by Dataset

### Boughter (Training Set)

**Source:** Boughter et al. (2020) - 914 VH sequences, ELISA assay

**Preprocessing Steps:**

1. **DNA → Protein Translation** - Translate DNA sequences to amino acids
2. **ANARCI Annotation** - Annotate CDRs using IMGT numbering scheme
3. **Quality Control** - Remove sequences with annotation failures
4. **Label Assignment** - Binary labels from polyreactivity scores

**Documentation:** See `docs/datasets/boughter/README.md` for detailed steps

**Output Files:**

- `train_datasets/boughter/canonical/boughter_processed_stage3.csv` (final)
- `train_datasets/boughter/processed/` (intermediate stages)

---

### Jain (Test Set - Novo Parity)

**Source:** Jain et al. (2017) - 86 clinical antibodies, per-antigen ELISA

**Preprocessing Steps:**

1. **Excel → CSV Conversion** - Extract data from supplementary Excel file
2. **P5e-S2 Subset Selection** - Select 86 antibodies matching Novo Nordisk's test set
3. **Threshold Application** - Binary labels from Table 1 PSR scores
4. **Sequence Validation** - Verify sequences match published data

**Documentation:** See `docs/datasets/jain/README.md` for detailed steps

**Output Files:**

- `test_datasets/jain/canonical/jain_p5e_s2.csv` (86 antibodies)
- `test_datasets/jain/raw/` (original Excel file)

**Critical Note:** Threshold selection (PSR > 0.5) matches Novo Nordisk's exact parity analysis.

---

### Harvey (Nanobody Test Set)

**Source:** Harvey et al. (2022) / Mason et al. (2021) - 141,021 nanobody sequences, PSR assay

**Preprocessing Steps:**

1. **CSV Combination** - Merge multiple raw CSV files
2. **Nanobody Fragment Extraction** - Extract VHH, VHH-CDRs, VHH-FWRs
3. **PSR Label Assignment** - Binary labels from PSR binding scores
4. **Validation** - Verify sequence counts match published data

**Documentation:** See `docs/datasets/harvey/README.md` for detailed steps

**Output Files:**

- `test_datasets/harvey/canonical/harvey_full.csv` (141k sequences)
- `test_datasets/harvey/fragments/harvey_VHH_only.csv`
- `test_datasets/harvey/fragments/harvey_VHH-CDRs.csv`
- `test_datasets/harvey/fragments/harvey_VHH-FWRs.csv`

---

### Shehata (PSR Cross-Validation)

**Source:** Shehata et al. (2019) - 398 human antibodies, PSR assay

**Preprocessing Steps:**

1. **Excel → CSV Conversion** - Extract data from supplementary Excel file
2. **Antibody Fragment Extraction** - Extract VH, VL, CDRs, FWRs
3. **PSR Label Assignment** - Binary labels from PSR scores (threshold: 0.5495)
4. **Validation** - Cross-check with published confusion matrices

**Documentation:** See `docs/datasets/shehata/README.md` for detailed steps

**Output Files:**

- `test_datasets/shehata/canonical/shehata_full.csv` (398 sequences)
- `test_datasets/shehata/fragments/shehata_VH.csv`
- `test_datasets/shehata/fragments/shehata_All-CDRs.csv`
- `test_datasets/shehata/fragments/shehata_H-FWRs.csv`

---

## Fragment Extraction

### Standard Fragments

All datasets support extraction of standard antibody fragments:

**Variable Chains:**
- `VH` - Variable Heavy chain
- `VL` - Variable Light chain
- `VH_VL` - Combined VH + VL

**CDRs (Complementarity-Determining Regions):**
- `H-CDR1`, `H-CDR2`, `H-CDR3` - Individual Heavy CDRs
- `L-CDR1`, `L-CDR2`, `L-CDR3` - Individual Light CDRs
- `H-CDRs` - All Heavy CDRs concatenated
- `L-CDRs` - All Light CDRs concatenated
- `All-CDRs` - All CDRs (Heavy + Light)

**FWRs (Framework Regions):**
- `H-FWR1`, `H-FWR2`, `H-FWR3`, `H-FWR4` - Individual Heavy FWRs
- `L-FWR1`, `L-FWR2`, `L-FWR3`, `L-FWR4` - Individual Light FWRs
- `H-FWRs` - All Heavy FWRs concatenated
- `L-FWRs` - All Light FWRs concatenated
- `All-FWRs` - All FWRs (Heavy + Light)

---

### Nanobody-Specific Fragments (Harvey)

Nanobodies have single-domain VHH sequences:

- `VHH_only` - Full VHH domain
- `VHH-CDR1`, `VHH-CDR2`, `VHH-CDR3` - Individual VHH CDRs
- `VHH-CDRs` - All VHH CDRs concatenated
- `VHH-FWRs` - All VHH FWRs concatenated

---

## Adding a New Dataset

### Step 1: Create Preprocessing Directory

```bash
mkdir -p preprocessing/my_dataset/
```

### Step 2: Write Preprocessing Scripts

```python
# preprocessing/my_dataset/step1_convert_to_csv.py
import pandas as pd

# Load raw data (Excel, CSV, FASTA, etc.)
df = pd.read_excel("path/to/raw_data.xlsx")

# Convert to canonical format
canonical_df = pd.DataFrame({
    "sequence": df["antibody_sequence"],
    "label": (df["polyreactivity_score"] > threshold).astype(int)
})

# Save canonical CSV
canonical_df.to_csv("test_datasets/my_dataset/canonical/my_dataset.csv", index=False)
```

---

### Step 3: Extract Fragments

```python
# preprocessing/my_dataset/step2_extract_fragments.py
from antibody_training_esm.datasets.base import AntibodyDataset

# Load canonical dataset
df = pd.read_csv("test_datasets/my_dataset/canonical/my_dataset.csv")

# Create dataset instance
dataset = MyDataset()  # Implement AntibodyDataset interface

# Extract fragments
fragments = dataset.extract_all_fragments(df)

# Save fragment CSVs
for fragment_name, fragment_df in fragments.items():
    fragment_df.to_csv(
        f"test_datasets/my_dataset/fragments/my_dataset_{fragment_name}.csv",
        index=False
    )
```

---

### Step 4: Create Dataset Documentation

Create `docs/datasets/my_dataset/README.md` documenting:

- Data source + citation
- Preprocessing steps
- File locations
- Known issues
- Example usage

---

### Step 5: Register Dataset

Add dataset class to `src/antibody_training_esm/datasets/`:

```python
# src/antibody_training_esm/datasets/my_dataset.py
from .base import AntibodyDataset

class MyDataset(AntibodyDataset):
    def __init__(self):
        super().__init__(
            name="my_dataset",
            canonical_path="test_datasets/my_dataset/canonical/my_dataset.csv",
            fragments_dir="test_datasets/my_dataset/fragments/"
        )

    # Implement required methods...
```

---

## Quality Control Checks

### 1. Sequence Validation

```python
import re

def is_valid_sequence(seq: str) -> bool:
    """Check if sequence contains only valid amino acids."""
    return bool(re.match(r'^[ACDEFGHIKLMNPQRSTVWY]+$', seq))

# Validate all sequences
df["valid"] = df["sequence"].apply(is_valid_sequence)
print(f"Invalid sequences: {(~df['valid']).sum()}")
```

---

### 2. Label Distribution

```python
# Check class balance
print(df["label"].value_counts())
print(f"Positive rate: {df['label'].mean():.2%}")
```

**Expected distributions:**

- **Boughter:** ~40% non-specific
- **Jain:** ~31% non-specific (27/86)
- **Harvey:** ~variable (depends on PSR threshold)

---

### 3. Sequence Length Distribution

```python
import matplotlib.pyplot as plt

df["seq_length"] = df["sequence"].str.len()
df["seq_length"].hist(bins=50)
plt.xlabel("Sequence Length")
plt.ylabel("Count")
plt.title("Sequence Length Distribution")
plt.savefig("seq_length_dist.png")
```

**Expected ranges:**

- **VH:** 110-130 amino acids
- **VL:** 100-120 amino acids
- **CDRs:** 5-20 amino acids each

---

## Troubleshooting

### Issue: ANARCI annotation fails

**Symptoms:** Many sequences skipped during annotation

**Solution:** Install ANARCI correctly:

```bash
# Install ANARCI dependencies
conda install -c bioconda anarci

# Verify installation
anarci -h
```

---

### Issue: Excel file won't open

**Symptoms:** `openpyxl` or `xlrd` errors

**Solution:** Install Excel reading libraries:

```bash
uv pip install openpyxl xlrd
```

---

### Issue: Fragment extraction produces empty sequences

**Symptoms:** Fragment CSVs have `NaN` or empty strings

**Solution:** Check ANARCI annotation success:

```python
# Check annotation status
df["has_vh"] = df["VH"].notna() & (df["VH"] != "")
print(f"VH annotation rate: {df['has_vh'].mean():.1%}")
```

---

### Issue: Label threshold unclear

**Symptoms:** Don't know how to assign binary labels from scores

**Solution:** Check original paper methods section:

- **Boughter:** Threshold from paper (polyreactivity score)
- **Jain:** Table 1 PSR > 0.5
- **Harvey:** PSR binding scores (various thresholds)
- **Shehata:** PSR > 0.5495 (Novo Nordisk exact parity)

See dataset-specific docs for details.

---

## Next Steps

After preprocessing:

- **Training:** See [Training Guide](training.md) to train models on preprocessed data
- **Testing:** See [Testing Guide](testing.md) to evaluate on preprocessed test sets
- **Dataset Documentation:** See `docs/datasets/` for dataset-specific preprocessing details

---

**Last Updated:** 2025-11-09
**Branch:** `docs/canonical-structure`
