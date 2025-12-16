# SPEC: Phase 3 — Verify with Inference

**Status:** DRAFT
**Parent:** [JAIN_PARITY_REMEDIATION_PLAN.md](./JAIN_PARITY_REMEDIATION_PLAN.md)
**Depends On:** Phase 2 (Artifact Regeneration)
**Blocks:** Phase 4 (Documentation Fix)

---

## Objective

Run inference on the regenerated Jain dataset and verify that the confusion matrix exactly matches Novo's reported results: `[[40, 17], [10, 19]]` with 68.60% accuracy.

**This is the critical validation step.** If this fails, our hypothesis is wrong and we must re-analyze.

---

## Success Criteria

```python
# MUST achieve EXACTLY these values
expected_confusion_matrix = [[40, 17], [10, 19]]
expected_accuracy = 59 / 86  # 0.6860465...
expected_accuracy_display = "68.60%"
```

---

## Inference Process

### Prerequisites

1. Trained model exists: `experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl`
2. Regenerated dataset: `data/test/jain/canonical/jain_86_novo_parity.csv`
3. Python environment with required packages

### Step 1: Load Model and Data

```python
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

# Load trained model
MODEL_PATH = "experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl"
with open(MODEL_PATH, "rb") as f:
    classifier = pickle.load(f)

# Load regenerated dataset
DATASET_PATH = "data/test/jain/canonical/jain_86_novo_parity.csv"
df = pd.read_csv(DATASET_PATH)

print(f"Loaded model: {MODEL_PATH}")
print(f"Loaded dataset: {len(df)} antibodies")
print(f"Label distribution: {(df['label']==0).sum()} specific, {(df['label']==1).sum()} non-specific")
```

### Step 2: Generate Embeddings

```python
# Extract sequences
sequences = df["vh_sequence"].tolist()

# Generate ESM-1v embeddings
print("Generating ESM-1v embeddings...")
X_test = classifier.embedding_extractor.extract_batch_embeddings(sequences)
print(f"Embeddings shape: {X_test.shape}")
```

### Step 3: Run Inference

```python
# Get true labels
y_true = df["label"].values.astype(int)

# Make predictions
y_pred = classifier.predict(X_test)
y_proba = classifier.predict_proba(X_test)

print(f"Predictions made for {len(y_pred)} antibodies")
```

### Step 4: Compute Metrics

```python
# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(f"\nConfusion Matrix:")
print(f"  [[{cm[0,0]}, {cm[0,1]}], [{cm[1,0]}, {cm[1,1]}]]")

# Compute accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Expected values
NOVO_CM = np.array([[40, 17], [10, 19]])
NOVO_ACC = 59 / 86

print(f"\nNovo Target:")
print(f"  [[{NOVO_CM[0,0]}, {NOVO_CM[0,1]}], [{NOVO_CM[1,0]}, {NOVO_CM[1,1]}]]")
print(f"  Accuracy: {NOVO_ACC:.4f} ({NOVO_ACC*100:.2f}%)")
```

### Step 5: Verify Match

```python
# Check exact match
cm_match = np.array_equal(cm, NOVO_CM)
acc_match = abs(accuracy - NOVO_ACC) < 0.0001

print("\n" + "="*60)
if cm_match and acc_match:
    print("✅ EXACT NOVO PARITY ACHIEVED!")
    print("="*60)
    print(f"  Confusion Matrix: MATCH")
    print(f"  Accuracy: MATCH ({accuracy*100:.2f}%)")
else:
    print("❌ PARITY NOT ACHIEVED")
    print("="*60)
    if not cm_match:
        print(f"  CM difference: {cm - NOVO_CM}")
    if not acc_match:
        print(f"  Accuracy difference: {(accuracy - NOVO_ACC)*100:.2f}pp")
    raise AssertionError("Verification failed!")
```

---

## Verification Script

Create a standalone verification script:

**File:** `experiments/benchmarks/novo_parity/scripts/verify_parity.py`

```python
#!/usr/bin/env python3
"""
Verify that regenerated Jain dataset achieves exact Novo parity.

Usage:
    python -m experiments.benchmarks.novo_parity.scripts.verify_parity
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODEL_PATH = PROJECT_ROOT / "experiments/checkpoints/esm1v/logreg/boughter_vh_esm1v_logreg.pkl"
DATASET_PATH = PROJECT_ROOT / "data/test/jain/canonical/jain_86_novo_parity.csv"

# Novo target
NOVO_CM = np.array([[40, 17], [10, 19]])
NOVO_ACC = 59 / 86


def main():
    print("="*60)
    print("JAIN PARITY VERIFICATION")
    print("="*60)
    print()

    # Load model
    print(f"Loading model: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        classifier = pickle.load(f)

    # Load dataset
    print(f"Loading dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    # Verify label distribution
    n_spec = (df["label"] == 0).sum()
    n_nonspec = (df["label"] == 1).sum()
    print(f"Label distribution: {n_spec} specific, {n_nonspec} non-specific")

    assert n_spec == 57, f"Expected 57 specific, got {n_spec}"
    assert n_nonspec == 29, f"Expected 29 non-specific, got {n_nonspec}"
    print("✅ Label distribution correct (57/29)")
    print()

    # Generate embeddings
    sequences = df["vh_sequence"].tolist()
    print("Generating ESM-1v embeddings...")
    X_test = classifier.embedding_extractor.extract_batch_embeddings(sequences)
    print(f"Embeddings shape: {X_test.shape}")
    print()

    # Run inference
    y_true = df["label"].values.astype(int)
    y_pred = classifier.predict(X_test)

    # Compute metrics
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print("RESULTS:")
    print(f"  Confusion Matrix: [[{cm[0,0]}, {cm[0,1]}], [{cm[1,0]}, {cm[1,1]}]]")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    print("NOVO TARGET:")
    print(f"  Confusion Matrix: [[{NOVO_CM[0,0]}, {NOVO_CM[0,1]}], [{NOVO_CM[1,0]}, {NOVO_CM[1,1]}]]")
    print(f"  Accuracy: {NOVO_ACC:.4f} ({NOVO_ACC*100:.2f}%)")
    print()

    # Verify match
    cm_match = np.array_equal(cm, NOVO_CM)
    acc_match = abs(accuracy - NOVO_ACC) < 0.0001

    print("="*60)
    if cm_match and acc_match:
        print("✅ EXACT NOVO PARITY ACHIEVED!")
        print("="*60)
        return True
    else:
        print("❌ PARITY NOT ACHIEVED")
        print("="*60)
        if not cm_match:
            print(f"  CM difference: {cm - NOVO_CM}")
        if not acc_match:
            print(f"  Accuracy difference: {(accuracy - NOVO_ACC)*100:.2f}pp")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
```

---

## Expected Output

```
============================================================
JAIN PARITY VERIFICATION
============================================================

Loading model: .../boughter_vh_esm1v_logreg.pkl
Loading dataset: .../jain_86_novo_parity.csv
Label distribution: 57 specific, 29 non-specific
✅ Label distribution correct (57/29)

Generating ESM-1v embeddings...
Embeddings shape: (86, 1280)

RESULTS:
  Confusion Matrix: [[40, 17], [10, 19]]
  Accuracy: 0.6860 (68.60%)

NOVO TARGET:
  Confusion Matrix: [[40, 17], [10, 19]]
  Accuracy: 0.6860 (68.60%)

============================================================
✅ EXACT NOVO PARITY ACHIEVED!
============================================================
```

---

## Failure Analysis

### If Confusion Matrix Doesn't Match

**Possible Causes:**
1. ESM-1v embedding nondeterminism (rare, ~0.1% variance)
2. Model file corrupted or wrong version
3. Dataset labels incorrect
4. lebrikizumab/galiximab not properly reclassified

**Actions:**
1. Re-run verification (may be embedding variance)
2. Verify model file hash matches expected
3. Manually check lebrikizumab/galiximab labels in CSV
4. Check if Phase 2 completed successfully

### If Off by 1 in CM

This could indicate borderline predictions flipping due to embedding variance.

**Actions:**
1. Check prediction probabilities for borderline antibodies
2. Re-run 3-5 times to check consistency
3. If consistently off by 1, investigate which antibody flipped

---

## Commit Message

```
test(verification): add Novo parity verification script

Adds verify_parity.py script that:
- Loads trained ESM-1v VH LogReg model
- Loads regenerated Jain dataset (57/29 split)
- Generates embeddings and runs inference
- Verifies confusion matrix matches [[40, 17], [10, 19]]
- Verifies accuracy matches 68.60%

Result: ✅ Exact Novo parity achieved

Issue: #33
```

---

## Exit Criteria

Phase 3 is complete when:

1. [ ] Verification script created
2. [ ] Script runs successfully
3. [ ] Confusion matrix exactly matches `[[40, 17], [10, 19]]`
4. [ ] Accuracy exactly matches 68.60% (59/86)
5. [ ] Results documented
6. [ ] Committed to fix branch

---

## Critical Gate

**DO NOT PROCEED TO PHASE 4 IF VERIFICATION FAILS.**

Phase 3 is the critical validation step. If it fails, the remediation hypothesis is wrong and we must:

1. Re-analyze the data
2. Check for errors in Phase 1 or Phase 2
3. Potentially reconsider which antibody pair to use
4. Return to research phase

---

**End of Spec**
