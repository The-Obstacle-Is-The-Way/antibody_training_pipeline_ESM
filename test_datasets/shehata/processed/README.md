# Shehata Dataset - Processed Files

Converted and filtered datasets, reproducible from raw sources.

---

## Files

### shehata.csv

**Description:** Full paired VH+VL sequences with PSR scores and binary labels

**Source:** Converted from `raw/shehata-mmc2.xlsx` (402 rows) → 398 antibodies

**Columns:**
- `antibody_id` - Unique identifier
- `VH` - Heavy chain variable region sequence
- `VL` - Light chain variable region sequence
- `PSR` - Polyspecific Reagent score (continuous)
- `label` - Binary non-specificity label (0=specific, 1=non-specific)
- Additional biophysical properties (Tm, charge, pI, etc.)

**Rows:** 398 antibodies + 1 header = 399 lines

**Filtering:** 4 sequences removed from original 402 due to:
- Incomplete VH/VL pairing
- Missing PSR scores
- Data quality issues

---

## Label Assignment

**Threshold:** 98.24th percentile of PSR score distribution

**Binary classification:**
- `label=0` (specific): PSR < 98.24th percentile → **391 antibodies** (98.2%)
- `label=1` (non-specific): PSR ≥ 98.24th percentile → **7 antibodies** (1.8%)

**Methodology:** Sakhnini et al. 2025 (Cell) - stringent threshold for non-specificity prediction

**Note:** Highly imbalanced dataset. Use stratified sampling for training/validation splits.

---

## Regeneration

To regenerate this file from raw sources:

```bash
python3 preprocessing/shehata/step1_convert_excel_to_csv.py
# Input:  test_datasets/shehata/raw/shehata-mmc2.xlsx
# Output: test_datasets/shehata/processed/shehata.csv
```

**Processing steps:**
1. Read Excel file (sheet: "Table S1")
2. Extract VH, VL, PSR, and metadata columns
3. Filter out 4 sequences with incomplete data
4. Calculate 98.24th percentile threshold
5. Assign binary labels (0/1)
6. Save to CSV

---

## Validation

Verify the processed CSV is correct:

```bash
python3 scripts/validation/validate_shehata_conversion.py
```

**Checks:**
- Row count: 398 antibodies
- Label distribution: 391 specific, 7 non-specific
- No missing values in VH/VL/PSR columns
- PSR threshold correctly applied
- Sequence format validation

---

## Usage

```python
import pandas as pd

# Load full dataset
df = pd.read_csv("test_datasets/shehata/processed/shehata.csv")

# Check label distribution
print(df['label'].value_counts())
# 0    391  (specific)
# 1      7  (non-specific)

# Filter by label
specific = df[df['label'] == 0]
nonspecific = df[df['label'] == 1]

# Use for testing models
from sklearn.metrics import classification_report
# ... (model predictions)
# print(classification_report(df['label'], predictions))
```

---

## Next Steps

To extract region-specific fragments (CDRs, FWRs, VH-only, etc.):

```bash
python3 preprocessing/shehata/step2_extract_fragments.py
# Input:  test_datasets/shehata/processed/shehata.csv
# Output: test_datasets/shehata/fragments/*.csv (16 files)
```

See `../fragments/README.md` for fragment details.

---

**See:** `../README.md` for complete dataset documentation
