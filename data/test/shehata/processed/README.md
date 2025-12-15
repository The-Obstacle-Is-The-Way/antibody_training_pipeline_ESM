# Shehata Dataset - Processed Files

Converted and filtered datasets, reproducible from raw sources.

---

## Files

### shehata.csv

**Description:** Full paired VH+VL sequences with PSR scores and binary labels

**Source:** Converted from `raw/shehata-mmc2.xlsx` (402 rows) → 398 antibodies

**Columns:**
- `id` - Antibody identifier (`Clone name` in the supplementary table)
- `heavy_seq` - Heavy chain variable region amino acid sequence (VH, gap-free)
- `light_seq` - Light chain variable region amino acid sequence (VL, gap-free)
- `label` - Binary label (0 = low PSR/specific, 1 = high PSR/non-specific)
- `psr_score` - Continuous PSR score (flow cytometry; normalized 0–1)
- `b_cell_subset` - B cell subset (IgG memory, IgM memory, Naïve, LLPCs)
- `source` - Data source identifier (`shehata2019`)

**Rows:** 398 antibodies + 1 header = 399 lines

**Filtering:** 4 rows removed from the original 402:
- 2 legend/metadata rows without VH/VL sequences
- 2 antibodies missing PSR scores

---

## Label Assignment

**Threshold:** High polyreactivity (label=1) corresponds to `psr_score > 0.33` (Shehata et al. 2019). The conversion script computes a cutoff at the 98.24th percentile to enforce exactly 7/398 non-specific antibodies (Sakhnini et al. 2025); in this dataset that is equivalent to `> 0.33`.

**Binary classification:**
- `label=0` (specific): PSR ≤ 0.33 → **391 antibodies** (98.2%)
- `label=1` (non-specific): PSR > 0.33 → **7 antibodies** (1.8%)

**Benchmark alignment:** Sakhnini et al. (2025) treat 7/398 as non-specific in their PSR benchmark; this binarization matches that count while following Shehata et al. (2019) high-polyreactivity definition (`psr_score > 0.33`).

**Note:** Highly imbalanced dataset. Use stratified sampling for training/validation splits.

---

## Regeneration

To regenerate this file from raw sources:

```bash
python3 preprocessing/shehata/step1_convert_excel_to_csv.py
# Input:  data/test/shehata/raw/shehata-mmc2.xlsx
# Output: data/test/shehata/processed/shehata.csv
```

**Processing steps:**
1. Read Excel file (Sheet1; Supplementary Table S1)
2. Extract VH, VL, PSR, and metadata columns
3. Drop non-sequence legend/metadata rows
4. Drop antibodies without numeric PSR scores
5. Assign binary labels (0/1) from PSR scores
6. Save to CSV (398 antibodies)

---

## Validation

Verify the processed CSV is correct:

```bash
python3 scripts/validation/validate_shehata_conversion.py
```

**Checks:**
- Row count: 398 antibodies
- Label distribution: 391 specific, 7 non-specific
- No missing values in `heavy_seq`, `light_seq`, or `psr_score`
- Sequence format validation

---

## Usage

```python
import pandas as pd

# Load full dataset
df = pd.read_csv("data/test/shehata/processed/shehata.csv")

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
# Input:  data/test/shehata/processed/shehata.csv
# Output: data/test/shehata/fragments/*.csv (16 files)
```

See `../fragments/README.md` for fragment details.

---

**See:** `../README.md` for complete dataset documentation
