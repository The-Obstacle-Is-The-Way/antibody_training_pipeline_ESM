# Jain Dataset - Canonical Benchmarks

Final curated datasets for reproducible benchmarking against Novo Nordisk results.

---

## Novo Nordisk Parity Datasets (86 antibodies each)

### 1. `jain_86_novo_parity.csv` (P5e-S2 Canonical) ✅ **RECOMMENDED**

- **Method:** ELISA filter → PSR reclassification (5 antibodies) → PSR/AC-SINS removal (30 antibodies)
- **Script:** `preprocessing/jain/step2_preprocess_p5e_s2.py`
- **Result:** [[40, 19], [10, 17]], 66.28% accuracy
- **Distribution:** 59 specific / 27 non-specific
- **Columns:** Full-length VH+VL sequences + all biophysical properties
- **Reproducibility:** 1 borderline antibody (nimotuzumab, probability ≈0.5) may occasionally flip due to ESM-1v embedding nondeterminism. Use stored `prediction` column for exact reproducibility.

**Pipeline:**
```
116 antibodies (ELISA-filtered)
  ↓ Reclassify 5 spec→nonspec (PSR >0.4, Tm outliers, clinical)
89 spec / 27 nonspec
  ↓ Remove 30 specific by PSR + AC-SINS tiebreaker
59 spec / 27 nonspec = 86 total
```

**Use when:** Training new models with biophysical features, or when you need the most biologically principled dataset.

---

### 2. `VH_only_jain_test_PARITY_86.csv` (OLD Reverse-Engineered)

- **Method:** ELISA filter → VH length outlier removal (3 antibodies) → borderline removals (5 antibodies)
- **Script:** Legacy `preprocessing/process_jain.py` (removed; see git history)
- **Result:** [[40, 19], [10, 17]], 66.28% accuracy (deterministic)
- **Distribution:** 59 specific / 27 non-specific
- **Columns:** VH fragment only (minimal columns)
- **Reproducibility:** Fully deterministic (no borderline cases)

**Pipeline:**
```
94 antibodies (OLD ELISA filter - different from 116!)
  ↓ Remove 3 VH length outliers (crenezumab, fletikumab, secukinumab)
91 antibodies
  ↓ Remove 5 borderline (muromonab, cetuximab, girentuximab, tabalumab, abituzumab)
86 antibodies
```

**Use when:** You need guaranteed deterministic reproducibility or VH-only benchmarking.

---

## Intermediate Datasets (OLD Method)

- `VH_only_jain_test_FULL.csv` **(94 antibodies)** - After OLD ELISA filter
  - Starting point for OLD reverse-engineered method
  - VH fragment only

- `VH_only_jain_test_QC_REMOVED.csv` **(91 antibodies)** - After VH length outlier removal
  - Removed: crenezumab (VH=112), fletikumab (VH=127), secukinumab (VH=127)
  - VH fragment only

---

## Usage Examples

### Recommended: P5e-S2 Canonical

```python
import pandas as pd
import pickle

# Load canonical benchmark
df = pd.read_csv('data/test/jain/canonical/jain_86_novo_parity.csv')

# Load model
with open('models/boughter_vh_esm1v_logreg.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Test
sequences = df['vh_sequence'].tolist()
y_true = df['label'].values

X = classifier.embedding_extractor.extract_batch_embeddings(sequences)
y_pred = classifier.predict(X)

# Expected: [[40, 19], [10, 17]]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)
```

### Alternative: OLD Deterministic

```python
# For guaranteed deterministic results
df = pd.read_csv('data/test/jain/canonical/VH_only_jain_test_PARITY_86.csv')

# Same testing procedure...
# Expected: [[40, 19], [10, 17]] (always deterministic)
```

---

## Comparison: P5e-S2 vs OLD

| Aspect | P5e-S2 Canonical | OLD Reverse-Engineered |
|--------|------------------|------------------------|
| **Result** | [[40, 19], [10, 17]] | [[40, 19], [10, 17]] |
| **Methodology** | Biologically principled (PSR-based) | Ad-hoc (length + borderline) |
| **Starting point** | 116 antibodies (ELISA-only) | 94 antibodies (different ELISA filter) |
| **Determinism** | 99% (1 borderline at ~0.5) | 100% (fully deterministic) |
| **Antibody overlap** | 62/86 same (24 different) | 62/86 same (24 different) |
| **Documentation** | Full provenance in experiments/ | Minimal documentation |
| **Biophysical data** | Full (PSR, AC-SINS, HIC, Tm, etc.) | Minimal (VH only) |

**Recommendation:** Use P5e-S2 for all new work. Use OLD only for exact reproducibility needs.

---

## Verification

To verify parity:

```bash
# Test P5e-S2
python3 preprocessing/jain/test_novo_parity.py

# Test OLD method
python3 test.py --model models/boughter_vh_esm1v_logreg.pkl \
  --data data/test/jain/canonical/VH_only_jain_test_PARITY_86.csv
```

Both should give [[40, 19], [10, 17]], 66.28% accuracy.

---

**See:** `JAIN_COMPLETE_GUIDE.md` (repo root) for complete methodology documentation
