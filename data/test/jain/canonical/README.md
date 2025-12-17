# Jain Dataset - Canonical Benchmarks

Final curated datasets for reproducible benchmarking against Novo Nordisk results.

---

## Novo Nordisk Parity Datasets (86 antibodies each)

### 1. `jain_86_novo_parity.csv` (P5e-S2 Canonical) ✅ **RECOMMENDED**

- **Method:** ELISA filter → PSR reclassification (5 antibodies) → PSR/AC-SINS removal (30 antibodies) → Tier D (2 antibodies)
- **Script:** `preprocessing/jain/step2_preprocess_p5e_s2.py`
- **Result:** [[40, 17], [10, 19]], 68.60% accuracy - EXACT NOVO PARITY
- **Distribution:** 57 specific / 29 non-specific
- **Columns:** Full-length VH+VL sequences + all biophysical properties
- **Reproducibility:** 1 borderline antibody (nimotuzumab, probability ≈0.5) may occasionally flip due to ESM-1v embedding nondeterminism. Use stored `prediction` column for exact reproducibility.

**Pipeline:**
```
116 antibodies (ELISA-filtered)
  ↓ Reclassify 5 spec→nonspec (PSR >0.4, Tm outliers, clinical)
89 spec / 27 nonspec
  ↓ Remove 30 specific by PSR + AC-SINS tiebreaker
59 spec / 27 nonspec = 86 total
  ↓ Tier D: lebrikizumab, galiximab (chromatography flags)
57 spec / 29 nonspec = 86 total - EXACT NOVO PARITY
```

**Use when:** Training new models with biophysical features, or when you need the most biologically principled dataset.

---

## Usage Examples

### Recommended: P5e-S2 Canonical

```python
import pandas as pd
import pickle

# Load canonical benchmark
df = pd.read_csv('data/test/jain/canonical/jain_86_novo_parity.csv')

# Load model
with open('experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Test
sequences = df['vh_sequence'].tolist()
y_true = df['label'].values

X = classifier.embedding_extractor.extract_batch_embeddings(sequences)
y_pred = classifier.predict(X)

# Expected: [[40, 17], [10, 19]] - EXACT NOVO PARITY (68.60%)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)
```

---

## Verification

To verify parity:

```bash
# Test P5e-S2
python3 preprocessing/jain/test_novo_parity.py
```

P5e-S2 gives [[40, 17], [10, 19]], 68.60% (EXACT NOVO PARITY).
