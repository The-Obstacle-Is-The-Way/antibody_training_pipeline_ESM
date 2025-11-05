# Shehata Dataset - Fragment-Specific Files

Region-specific extracts for ablation studies and targeted model training.

---

## Overview

16 fragment types extracted from `processed/shehata.csv` using ANARCI annotation with IMGT numbering scheme.

**All files:** 398 antibodies + 1 header = 399 lines

**Annotation:** ANARCI v1.4 with IMGT numbering (Dunbar & Deane 2016)

---

## Fragment Types (16 files)

### Full Sequences (3 files)

- `Full_shehata.csv` - Concatenated VH + VL (both chains)
- `VH_only_shehata.csv` - Heavy chain variable region only (most common)
- `VL_only_shehata.csv` - Light chain variable region only

### Heavy Chain CDRs (3 files)

- `H-CDR1_shehata.csv` - Heavy chain CDR1 (IMGT positions 27-38)
- `H-CDR2_shehata.csv` - Heavy chain CDR2 (IMGT positions 56-65)
- `H-CDR3_shehata.csv` - Heavy chain CDR3 (IMGT positions 105-117)

### Light Chain CDRs (3 files)

- `L-CDR1_shehata.csv` - Light chain CDR1 (IMGT positions 27-38)
- `L-CDR2_shehata.csv` - Light chain CDR2 (IMGT positions 56-65)
- `L-CDR3_shehata.csv` - Light chain CDR3 (IMGT positions 105-117)

### Combined Regions (7 files)

- `H-CDRs_shehata.csv` - All heavy chain CDRs concatenated (H-CDR1 + H-CDR2 + H-CDR3)
- `H-FWRs_shehata.csv` - All heavy chain frameworks (H-FWR1 + H-FWR2 + H-FWR3 + H-FWR4)
- `L-CDRs_shehata.csv` - All light chain CDRs concatenated (L-CDR1 + L-CDR2 + L-CDR3)
- `L-FWRs_shehata.csv` - All light chain frameworks (L-FWR1 + L-FWR2 + L-FWR3 + L-FWR4)
- `All-CDRs_shehata.csv` - All CDRs from both chains (H-CDRs + L-CDRs)
- `All-FWRs_shehata.csv` - All frameworks from both chains (H-FWRs + L-FWRs)
- `VH_VL_shehata.csv` - Same as Full_shehata.csv (alias)

---

## Regeneration

To regenerate all fragments from processed CSV:

```bash
python3 preprocessing/process_shehata.py
# Input:  test_datasets/shehata/processed/shehata.csv
# Output: test_datasets/shehata/fragments/*.csv (16 files)
```

**Processing steps:**
1. Read `processed/shehata.csv` (398 antibodies)
2. For each antibody, run ANARCI annotation on VH and VL
3. Extract regions using IMGT numbering
4. Save each fragment type to separate CSV

**Time:** ~2 minutes (ANARCI annotation is slow)

---

## CRITICAL: P0 Blocker History

**Issue:** Gap characters (`-`) in sequences caused embedding failures

**Root cause:** Used `annotation.sequence_alignment_aa` instead of `annotation.sequence_aa`

**Fix:** Line 63 in `preprocessing/process_shehata.py`
```python
# WRONG (has gaps):
f"full_seq_{chain}": annotation.sequence_alignment_aa

# CORRECT (gap-free):
f"full_seq_{chain}": annotation.sequence_aa
```

**Verification:**
```bash
# Check for gap characters (should return NOTHING)
grep -c '\-' test_datasets/shehata/fragments/*.csv | grep -v ':0'
```

**See:** `docs/shehata/SHEHATA_BLOCKER_ANALYSIS.md` for complete history

---

## Usage Examples

### VH-only Testing (Most Common)

```python
import pandas as pd

# Load VH-only sequences
df = pd.read_csv("test_datasets/shehata/fragments/VH_only_shehata.csv")

# Use for testing
from sklearn.metrics import classification_report
predictions = model.predict(df['sequence'])
print(classification_report(df['label'], predictions))
```

### CDR-specific Analysis

```python
# Compare H-CDR3 vs L-CDR3 importance
h_cdr3_df = pd.read_csv("test_datasets/shehata/fragments/H-CDR3_shehata.csv")
l_cdr3_df = pd.read_csv("test_datasets/shehata/fragments/L-CDR3_shehata.csv")

# Train models on each
# ... (compare performance)
```

### Full Sequence Testing

```python
# Test on paired VH+VL
full_df = pd.read_csv("test_datasets/shehata/fragments/Full_shehata.csv")
```

---

## Fragment Use Cases

| Fragment | Use Case |
|----------|----------|
| `VH_only_shehata.csv` | Standard VH-only model testing |
| `VL_only_shehata.csv` | Light chain contribution analysis |
| `H-CDR3_shehata.csv` | CDR3 dominance hypothesis testing |
| `All-CDRs_shehata.csv` | CDR-only model testing |
| `All-FWRs_shehata.csv` | Framework contribution analysis |
| `Full_shehata.csv` | Paired VH+VL model testing |

---

## Validation

Verify all fragments are correct:

```bash
python3 scripts/validation/validate_shehata_conversion.py
# Checks: row counts, label distributions, no gaps, etc.

python3 tests/test_shehata_embedding_compatibility.py
# Tests: ESM embedding generation for all fragments
```

---

## Column Schema

All fragment CSVs have the same schema:

```
antibody_id,sequence,label,PSR,[optional metadata]
```

- `antibody_id` - Unique identifier
- `sequence` - Fragment-specific amino acid sequence
- `label` - Binary non-specificity label (0=specific, 1=non-specific)
- `PSR` - Polyspecific Reagent score
- `[optional metadata]` - Additional columns from source data

---

**See:** `../README.md` for complete dataset documentation
