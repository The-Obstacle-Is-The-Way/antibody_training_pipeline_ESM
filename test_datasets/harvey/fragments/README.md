# Harvey Dataset - Fragment-Specific Files

Region-specific extracts for ablation studies and targeted model training.

---

## Overview

6 fragment types extracted from `processed/harvey.csv` using ANARCI annotation with IMGT numbering scheme.

**All files:** 141,021 nanobodies + 1 header = 141,022 lines

**Annotation:** ANARCI v1.4 with IMGT numbering (Dunbar & Deane 2016)

**ANARCI failures:** 453 sequences (0.32% failure rate) logged in `failed_sequences.txt`

---

## Fragment Types (6 files)

### Full Sequence (1 file)

- `VHH_only_harvey.csv` - Full nanobody variable domain (most common use case)

### Heavy Chain CDRs (3 files)

- `H-CDR1_harvey.csv` - Heavy chain CDR1 (IMGT positions 27-38)
- `H-CDR2_harvey.csv` - Heavy chain CDR2 (IMGT positions 56-65)
- `H-CDR3_harvey.csv` - Heavy chain CDR3 (IMGT positions 105-117)

### Combined Regions (2 files)

- `H-CDRs_harvey.csv` - All heavy chain CDRs concatenated (H-CDR1 + H-CDR2 + H-CDR3)
- `H-FWRs_harvey.csv` - All heavy chain frameworks (H-FWR1 + H-FWR2 + H-FWR3 + H-FWR4)

---

## Nanobody-Specific Notes

**No light chain fragments:**
- Nanobodies (VHH) are single-domain antibodies
- Only heavy chain variable region present
- No VL, L-CDRs, or L-FWRs (unlike Shehata/Jain/Boughter)

**Fragment count:**
- Harvey: **6 fragments** (VHH only)
- Shehata/Jain/Boughter: 16 fragments (VH+VL)

---

## Regeneration

To regenerate all fragments from processed CSV:

```bash
python3 preprocessing/process_harvey.py
# Input:  test_datasets/harvey/processed/harvey.csv
# Output: test_datasets/harvey/fragments/*.csv (6 files)
```

**Processing steps:**
1. Read `processed/harvey.csv` (141,474 nanobodies)
2. For each nanobody, run ANARCI annotation on VHH
3. Extract regions using IMGT numbering
4. Save each fragment type to separate CSV
5. Log failed annotations to `failed_sequences.txt`

**Time:** ~30-40 minutes (ANARCI annotation is slow for 141k sequences)

---

## CRITICAL: P0 Blocker History

**Issue:** Gap characters (`-`) in sequences caused ESM-1v embedding failures

**Root cause:** Used `annotation.sequence_alignment_aa` instead of `annotation.sequence_aa`

**Fix:** Line 48 in `preprocessing/process_harvey.py`
```python
# WRONG (has gaps):
"full_seq_H": annotation.sequence_alignment_aa

# CORRECT (gap-free):
"full_seq_H": annotation.sequence_aa
```

**Verification:**
```bash
# Check for gap characters (should return NOTHING)
grep -c '\-' test_datasets/harvey/fragments/*.csv | grep -v ':0'
```

**See:** `docs/harvey/HARVEY_P0_FIX_REPORT.md` for complete history

---

## ANARCI Annotation Failures

**File:** `failed_sequences.txt`

**Content:** 453 sequence IDs that failed ANARCI annotation (0.32% failure rate)

**Example:**
```
harvey_000042
harvey_000157
harvey_000891
...
```

**Why failures occur:**
- Incomplete sequences
- Non-standard amino acids
- Sequences too short/long for IMGT numbering
- Structural issues

**Impact:** Acceptable loss (< 1%), full dataset still representative

**Verification:**
```bash
# Count failures
wc -l test_datasets/harvey/fragments/failed_sequences.txt
# Should be 453

# Verify fragment row counts
for f in test_datasets/harvey/fragments/*.csv; do
  count=$(wc -l < "$f")
  echo "$f: $count lines (expected 141,022)"
done
```

---

## Usage Examples

### VHH-only Testing (Most Common)

```python
import pandas as pd

# Load VHH-only sequences
df = pd.read_csv("test_datasets/harvey/fragments/VHH_only_harvey.csv")

# Use for testing
from sklearn.metrics import classification_report
predictions = model.predict(df['sequence'])
print(classification_report(df['label'], predictions))
```

### CDR-specific Analysis

```python
# Compare H-CDR regions
h_cdr1_df = pd.read_csv("test_datasets/harvey/fragments/H-CDR1_harvey.csv")
h_cdr2_df = pd.read_csv("test_datasets/harvey/fragments/H-CDR2_harvey.csv")
h_cdr3_df = pd.read_csv("test_datasets/harvey/fragments/H-CDR3_harvey.csv")

# Train models on each
# ... (compare performance)
```

### Full CDRs vs Frameworks

```python
# Test CDR-only vs framework-only models
cdrs_df = pd.read_csv("test_datasets/harvey/fragments/H-CDRs_harvey.csv")
fwrs_df = pd.read_csv("test_datasets/harvey/fragments/H-FWRs_harvey.csv")
```

---

## Fragment Use Cases

| Fragment | Use Case |
|----------|----------|
| `VHH_only_harvey.csv` | Standard VHH-only model testing (most common) |
| `H-CDR3_harvey.csv` | CDR3 dominance hypothesis testing |
| `H-CDRs_harvey.csv` | CDR-only model testing |
| `H-FWRs_harvey.csv` | Framework contribution analysis |
| `H-CDR1/2_harvey.csv` | Individual CDR region analysis |

---

## Validation

Verify all fragments are correct:

```bash
python3 scripts/validation/validate_fragments.py
# Checks: row counts, label distributions, no missing values

python3 tests/test_harvey_embedding_compatibility.py
# Tests: ESM-1v embedding generation for all fragments
# CRITICAL: Validates no gap characters (P0 regression check)
```

---

## Column Schema

All fragment CSVs have the same schema:

```
id,sequence,label,source,[sequence_length]
```

- `id` - Unique identifier (harvey_XXXXXX)
- `sequence` - Fragment-specific amino acid sequence (gap-free)
- `label` - Binary polyreactivity label (0=low, 1=high)
- `source` - Dataset source ("harvey2022")
- `sequence_length` - Length of sequence in amino acids (optional)

---

**See:** `../README.md` for complete dataset documentation
