# Harvey Dataset - Canonical Benchmarks

**Status:** Empty (use full dataset directly)

---

## Why Empty?

Harvey dataset does **not** require canonical subsets because:

1. **Already balanced:** 49.1% / 50.9% split (low/high polyreactivity)
2. **Large dataset:** 141,021 nanobodies after ANARCI annotation
3. **Training set:** Used for model training, not external validation
4. **No stratification needed:** Full dataset is representative

---

## Comparison with Other Datasets

### Shehata (canonical/ empty) ✅

- **Why empty:** External test set, use full 398 antibodies
- **Pattern:** All sequences equally important for validation

### Jain (canonical/ populated) ✅

- **Why populated:** Benchmarks for Novo Nordisk parity
- **Files:** `jain_86_novo_parity.csv` (86 VH-only sequences)
- **Pattern:** Specific benchmark subset for reproducibility

### Boughter (canonical/ populated) ✅

- **Why populated:** Training benchmarks, balanced subsets
- **Files:** Strict/lenient QC benchmarks
- **Pattern:** Curated subsets for consistent training

### Harvey (canonical/ empty) ✅

- **Why empty:** Full dataset already balanced, no subsampling needed
- **Pattern:** Use full 141,021 nanobodies directly

---

## Usage

For Harvey dataset, **use the full dataset directly:**

```python
import pandas as pd

# Load full Harvey dataset (already balanced)
df = pd.read_csv("test_datasets/harvey/fragments/VHH_only_harvey.csv")

# Split for training/validation as needed
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
```

---

## Alternative: Create Custom Benchmarks

If you need specific benchmarks (e.g., balanced subsets for quick testing):

```python
import pandas as pd

# Load full dataset
df = pd.read_csv("test_datasets/harvey/fragments/VHH_only_harvey.csv")

# Create 10k balanced subset
low = df[df['label'] == 0].sample(5000, random_state=42)
high = df[df['label'] == 1].sample(5000, random_state=42)
subset = pd.concat([low, high]).sample(frac=1, random_state=42)  # shuffle

# Save benchmark
subset.to_csv("test_datasets/harvey/canonical/harvey_10k_balanced.csv", index=False)
```

**Note:** Currently no canonical benchmarks defined. Use full dataset or create custom splits as needed.

---

## Decision Log

**Decision Date:** 2025-11-05

**Question:** Should Harvey have canonical benchmarks?

**Decision:** **Keep canonical/ empty**

**Rationale:**
- Full dataset is already balanced (49.1% / 50.9%)
- Large enough for robust training (141,021 sequences)
- No specific benchmark requirements identified
- Can add later if needed for reproducibility

**Alternative considered:**
- Balanced subsets (e.g., 10k, 50k) for quick testing
- Cross-validation splits for consistent benchmarking
- **Rejected:** Not needed initially, can add later if demand arises

---

**See:** `../README.md` for complete dataset documentation
