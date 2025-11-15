# Shehata Dataset - Canonical Benchmarks

**This directory is intentionally empty.**

---

## Why Empty?

The Shehata dataset serves as an **external test set** for non-specificity prediction models. Unlike the Jain dataset, which requires preprocessing to match Novo Nordisk parity benchmarks, the Shehata dataset is used as-is.

---

## Comparison with Jain

### Jain (canonical/ populated)
- **Purpose:** Training and validation
- **Preprocessing:** P5e-S2 filtering to match Novo Nordisk benchmark
- **Canonical files:** `jain_86_novo_parity.csv` (86 antibodies)
- **Why:** Reproducible benchmark for model comparisons

### Shehata (canonical/ empty)
- **Purpose:** External test set (never seen during training)
- **Preprocessing:** None (use full 398-antibody dataset)
- **Canonical files:** None (use `processed/shehata.csv` directly)
- **Why:** Unbiased evaluation on independent data

---

## Where to Find Shehata Data

**For testing models:**
```bash
# Full paired sequences (398 antibodies)
data/test/shehata/processed/shehata.csv

# VH-only sequences (most common use case)
data/test/shehata/fragments/VH_only_shehata.csv

# Other fragments
data/test/shehata/fragments/*.csv
```

---

## Directory Structure Consistency

This directory exists to maintain **consistent structure** across all datasets:

```
data/test/
├── jain/
│   ├── raw/
│   ├── processed/
│   ├── canonical/      ← Populated (training/validation)
│   └── fragments/
├── shehata/
│   ├── raw/
│   ├── processed/
│   ├── canonical/      ← Empty (external test set)
│   └── fragments/
└── boughter/
    ├── raw/
    ├── processed/
    ├── canonical/      ← Populated (training/validation)
    └── fragments/
```

**Principle:** Every dataset follows the 4-tier structure (raw → processed → canonical → fragments), even if some tiers are empty.

---

## Usage Example

```python
import pandas as pd

# CORRECT: Use processed/ for Shehata
shehata_df = pd.read_csv("data/test/shehata/processed/shehata.csv")

# CORRECT: Use fragments/ for region-specific testing
vh_only_df = pd.read_csv("data/test/shehata/fragments/VH_only_shehata.csv")

# For comparison:
# Jain uses canonical/ for benchmark testing
jain_df = pd.read_csv("data/test/jain/canonical/jain_86_novo_parity.csv")
```

---

**See:** `../README.md` for complete dataset documentation
