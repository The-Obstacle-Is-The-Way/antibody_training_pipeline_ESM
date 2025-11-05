# Jain Test Dataset

**Source**: Jain et al. (2017) PNAS - Biophysical properties of the clinical-stage antibody landscape

---

## Directory Structure

```
jain/
├── raw/               Original Excel files (NEVER MODIFY)
├── processed/         Converted CSVs + intermediate datasets
├── canonical/         Final benchmarks for Novo parity
├── fragments/         Region-specific extracts (CDRs, FWRs, etc.)
└── README.md          This file
```

Each subdirectory has its own README explaining contents and provenance.

---

## Quick Start

**For benchmarking with Novo Nordisk parity:**

```python
import pandas as pd
import pickle

# Load recommended P5e-S2 canonical benchmark
df = pd.read_csv('test_datasets/jain/canonical/jain_86_novo_parity.csv')

# Load model
with open('models/boughter_vh_esm1v_logreg.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Test
sequences = df['vh_sequence'].tolist()
y_true = df['label'].values

X = classifier.embedding_extractor.extract_batch_embeddings(sequences)
y_pred = classifier.predict(X)

# Expected: [[40, 19], [10, 17]], 66.28% accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)
```

---

## Canonical Benchmarks (86 antibodies each)

### 1. P5e-S2 Canonical ✅ **RECOMMENDED**

**File:** `canonical/jain_86_novo_parity.csv`

- **Method:** ELISA filter → PSR reclassification (5 antibodies) → PSR/AC-SINS removal (30 antibodies)
- **Result:** [[40, 19], [10, 17]], 66.28% accuracy
- **Distribution:** 59 specific / 27 non-specific
- **Reproducibility:** 99% deterministic (1 borderline antibody at ~0.5 probability)
- **Columns:** Full-length VH+VL + biophysical properties
- **Provenance:** Full documentation in `experiments/novo_parity/`

**Use when:** Training new models or need biologically principled methodology.

### 2. OLD Reverse-Engineered (Alternative)

**File:** `canonical/VH_only_jain_test_PARITY_86.csv`

- **Method:** ELISA filter → VH length outliers → borderline removals
- **Result:** [[40, 19], [10, 17]], 66.28% accuracy (fully deterministic)
- **Distribution:** 59 specific / 27 non-specific
- **Columns:** VH fragment only

**Use when:** You need 100% deterministic reproducibility.

---

## Data Flow

```
raw/ (Excel)
  ↓ [scripts/conversion/convert_jain_excel_to_csv.py]
processed/ (CSV base files)
  ↓ [preprocessing/preprocess_jain_p5e_s2.py]
canonical/ (86-antibody benchmarks)
  ↓ [scripts/fragmentation/extract_jain_fragments.py]
fragments/ (region-specific)
```

---

## File Counts

- **raw/**: 4 Excel files (original sources)
- **processed/**: 7 CSV files (137 → 116 → intermediates)
- **canonical/**: 4 CSV files (86-antibody benchmarks + OLD method intermediates)
- **fragments/**: 20 CSV files (CDRs, FWRs, VH/VL extracts)

**Total:** 35 files organized by purpose

---

## Verification

To verify Novo parity:

```bash
# Test P5e-S2 canonical
python3 scripts/testing/test_jain_novo_parity.py

# Expected output: [[40, 19], [10, 17]], 66.28%
```

---

## Regenerating Files

To regenerate from raw sources:

```bash
# Step 1: Convert Excel → CSV
python3 scripts/conversion/convert_jain_excel_to_csv.py

# Step 2: Create 86-antibody benchmarks
python3 preprocessing/preprocess_jain_p5e_s2.py

# Step 3: (Optional) Extract fragments
# python3 scripts/fragmentation/extract_jain_fragments.py
```

---

## For More Information

- **Complete guide:** `JAIN_COMPLETE_GUIDE.md` (repo root)
- **Experiment provenance:** `experiments/novo_parity/`
- **Subdirectory READMEs:** Each subdirectory has detailed documentation

---

**Last Updated**: November 5, 2025
**Branch**: `leroy-jenkins/full-send`
