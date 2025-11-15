# Harvey Dataset - Processed Files

Converted and combined datasets, reproducible from raw sources.

---

## Files

### harvey.csv

**Description:** Combined VHH (nanobody) dataset with binary polyreactivity labels

**Source:** Converted from `raw/` CSVs (71,772 + 69,702 = 141,474 nanobodies)

**Columns:**
- `seq` - Full VHH sequence (gap-free, extracted from IMGT positions)
- `CDR1_nogaps` - Pre-extracted H-CDR1
- `CDR2_nogaps` - Pre-extracted H-CDR2
- `CDR3_nogaps` - Pre-extracted H-CDR3
- `label` - Binary polyreactivity label (0=low, 1=high)

**Rows:** 141,474 nanobodies + 1 header = 141,475 lines

**Processing:** No sequences removed during conversion (all raw data preserved)

---

## Label Assignment

**Binary classification:**
- `label=0` (low polyreactivity): 69,702 nanobodies (49.1%)
- `label=1` (high polyreactivity): 71,772 nanobodies (50.9%)

**Methodology:** Novo Nordisk high-throughput polyreactivity screening

**Threshold:** Raw files pre-labeled by experimental assay (no threshold applied during conversion)

**Note:** Balanced dataset (nearly 50/50 split). No resampling needed.

---

## Removed Files

**harvey_high.csv and harvey_low.csv**

These intermediate files were DELETED during cleanup (2025-11-05):
- **Why:** Duplicates of raw source files
- **Decision:** Scripts now read directly from `raw/` instead
- **Rationale:** DRY principle - single source of truth in `raw/`
- **Impact:** No functionality lost, cleaner structure

---

## Regeneration

To regenerate this file from raw sources:

```bash
python3 preprocessing/harvey/step1_convert_raw_csvs.py
# Input:  data/test/harvey/raw/high_polyreactivity_high_throughput.csv
#         data/test/harvey/raw/low_polyreactivity_high_throughput.csv
# Output: data/test/harvey/processed/harvey.csv
```

**Processing steps:**
1. Read high polyreactivity CSV (71,772 nanobodies)
2. Read low polyreactivity CSV (69,702 nanobodies)
3. Extract full sequences from IMGT position columns (1-128)
4. Remove gap characters (`-`) to create gap-free sequences
5. Assign binary labels (0=low, 1=high)
6. Combine into single dataset
7. Save to harvey.csv (141,474 nanobodies)

---

## Validation

Verify the processed CSV is correct:

```bash
# Check row count
wc -l data/test/harvey/processed/harvey.csv
# Should be 141,475 (141,474 + header)

# Check label distribution
python3 -c "
import pandas as pd
df = pd.read_csv('data/test/harvey/processed/harvey.csv')
print(df['label'].value_counts().sort_index())
# Expected: 0    69702
#           1    71772
"

# Verify no missing values in critical columns
python3 -c "
import pandas as pd
df = pd.read_csv('data/test/harvey/processed/harvey.csv')
print('Missing seq:', df['seq'].isna().sum())  # Should be 0
print('Missing label:', df['label'].isna().sum())  # Should be 0
"
```

---

## Usage

```python
import pandas as pd

# Load full dataset
df = pd.read_csv("data/test/harvey/processed/harvey.csv")

# Check label distribution
print(df['label'].value_counts())
# 0    69702  (low polyreactivity)
# 1    71772  (high polyreactivity)

# Filter by label
low_poly = df[df['label'] == 0]
high_poly = df[df['label'] == 1]

# Sequence statistics
print(f"Mean sequence length: {df['seq'].str.len().mean():.1f} aa")
```

---

## Next Steps

To extract region-specific fragments (CDRs, FWRs, VHH-only):

```bash
python3 preprocessing/harvey/step2_extract_fragments.py
# Input:  data/test/harvey/processed/harvey.csv
# Output: data/test/harvey/fragments/*.csv (6 files)
```

See `../fragments/README.md` for fragment details.

---

**See:** `../README.md` for complete dataset documentation
