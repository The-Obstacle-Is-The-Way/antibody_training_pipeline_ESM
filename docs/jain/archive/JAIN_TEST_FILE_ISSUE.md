# Jain Test File Issue - Root Cause Analysis

**Date:** 2025-11-07
**Status:** 🔴 CRITICAL - Test code has architectural mismatch

---

## THE PROBLEM

We have a **fundamental mismatch** between:
1. What the test code expects (VH-only file with 86 rows)
2. What the dataset loader expects (FULL file with 137 rows)
3. What we actually have (canonical file with VH+VL and 86 rows)

---

## CURRENT STATE

### What Files Exist:
```
test_datasets/jain/canonical/
└── jain_86_novo_parity.csv
    - 86 antibodies (59 specific + 27 non-specific) ✅
    - Has BOTH VH and VL sequences
    - Has full metadata (PSR, AC-SINS, etc.)
    - Created by step2_preprocess_p5e_s2.py
```

### What Files DON'T Exist (Deleted):
```
test_datasets/jain/canonical/
├── VH_only_jain_test_FULL.csv (DELETED - orphan from step4)
├── VH_only_jain_test_QC_REMOVED.csv (DELETED - orphan from step4)
└── VH_only_jain_test_PARITY_86.csv (DELETED - orphan from step4)
    - 86 antibodies, VH-only format
    - Had columns: [id, sequence, label]
    - This is what E2E tests reference!
```

---

## THE TEST CODE (tests/e2e/test_reproduce_novo.py)

### What It Does:
```python
# Line 61: Define test file path
real_dataset_paths = {
    "jain_parity": "test_datasets/jain/canonical/VH_only_jain_test_PARITY_86.csv",
}

# Line 100-102: Load the file
jain = JainDataset()
df_test = jain.load_data(
    full_csv_path=real_dataset_paths["jain_parity"],  # ← Passes VH-only file
    stage="parity"  # ← Tells loader to filter 137 → 86
)
```

### The Bug:
1. Test passes `VH_only_jain_test_PARITY_86.csv` (86 rows, VH-only)
2. Passes `stage="parity"` which tells loader to filter
3. But the file is ALREADY filtered to 86 rows!

---

## THE DATASET LOADER (src/antibody_training_esm/datasets/jain.py)

### What It Expects:
```python
def load_data(self, full_csv_path=None, sd03_csv_path=None, stage="full"):
    """
    Args:
        full_csv_path: Path to jain_with_private_elisa_FULL.csv (137 antibodies)
        stage: "full" (137) | "ssot" (116) | "parity" (86)

    Returns:
        DataFrame with VH_sequence, VL_sequence, label, etc.
    """
    # Load FULL 137-antibody file
    df = pd.read_csv(full_csv_path)  # Expects 137 rows

    # Apply filtering based on stage
    if stage == "parity":
        df = self.filter_elisa_1to3(df)       # 137 → 116
        df = self.reclassify_5_antibodies(df) # Relabel 5
        df = self.remove_30_by_psr_acsins(df) # 116 → 86

    return df
```

### The Bug:
- Loader expects `full_csv_path` to have 137 rows
- Test passes a file with 86 rows
- Loader tries to filter 86 → should crash or return wrong data

---

## WHAT THE MODEL WEIGHTS NEED

### Boughter Model (facebook/esm1v_t33_650M_UR90S_1):
```python
# From test code line 109-114
extractor = ESMEmbeddingExtractor(model_name="facebook/esm1v_t33_650M_UR90S_1")
train_embeddings = extractor.extract(df_train["sequence"].tolist())
test_embeddings = extractor.extract(df_test["sequence"].tolist())
```

**The model needs:**
- A column named `"sequence"`
- VH-only sequences (single chain)
- NOT VH+VL together

### Current File Has:
```
jain_86_novo_parity.csv columns:
- id
- vh_sequence  ← VH is here
- vl_sequence  ← VL is here
- label
- ...23 more columns
```

**Problem:** No column named `"sequence"` - has `"vh_sequence"` instead!

---

## WHY NOV 6 TESTS WORKED

The Nov 6 test results show correct 59/27 split:
```
predictions_boughter_vh_esm1v_logreg_VH_only_jain_test_PARITY_86_20251106_211815.csv
y_true: 59 zeros + 27 ones ✅
```

**Why?** The test ran BEFORE step4 contamination (Nov 7 16:47).

**What file did it use?**
- It used `VH_only_jain_test_PARITY_86.csv` which existed then
- That file had correct 59/27 labels
- That file had VH-only format: [id, sequence, label]

**Now that file is DELETED** because it was an orphan from step4.

---

## THE ARCHITECTURAL ISSUE

We have **THREE DIFFERENT USE CASES** with conflicting requirements:

### Use Case 1: Preprocessing (step2)
- Input: FULL 137-antibody file
- Process: Filter → Reclassify → Remove
- Output: canonical/jain_86_novo_parity.csv (VH+VL+metadata)

### Use Case 2: Dataset Loading (JainDataset)
- Input: FULL 137-antibody file
- Process: Apply stage filtering
- Output: DataFrame with VH+VL

### Use Case 3: Model Inference (E2E tests)
- Input: **VH-ONLY file with [id, sequence, label]**
- Process: Extract embeddings from VH
- Output: Predictions

**The problem:** Use Case 3 needs a DIFFERENT file format than what Use Cases 1 & 2 produce!

---

## QUESTIONS FOR SENIOR ENGINEERS

### Q1: Should we have separate files for different use cases?
**Option A:** Keep one canonical file (VH+VL+metadata) and extract VH in test code
**Option B:** Create separate VH-only file for inference/benchmarking
**Option C:** Refactor JainDataset.load_data() to handle both formats

### Q2: What should VH_only_jain_test_PARITY_86.csv contain?
If we recreate it:
- Should it be extracted from jain_86_novo_parity.csv?
- Or should step2 create it as a separate output?
- Or should we delete it and update test code?

### Q3: Is the test code architecture correct?
The test passes a parity file but uses `stage="parity"` which expects to filter.
Should it use `stage="full"` or `stage=None` instead?

### Q4: What's the single source of truth?
- Is it `jain_86_novo_parity.csv` (VH+VL)?
- Or should there be a separate VH-only canonical file?

### Q5: How do we prevent this from happening again?
- Should we have a test that validates file formats?
- Should we document which files are for which use cases?
- Should we refactor to have clearer separation?

---

## IMMEDIATE IMPACT

### Tests That Will FAIL:
```python
test_reproduce_novo_jain_accuracy_with_real_data()
  ↓
  FileNotFoundError: VH_only_jain_test_PARITY_86.csv not found
```

### Training Pipelines That May Break:
- Any code expecting VH-only format
- Any code using JainDataset with stage="parity" on wrong file

---

## WHAT NOT TO DO (Without Senior Review)

❌ Don't blindly recreate `VH_only_jain_test_PARITY_86.csv` without understanding why
❌ Don't change test code without understanding the architecture
❌ Don't modify JainDataset.load_data() without considering all use cases
❌ Don't commit anything that "works on my machine" without root cause analysis

---

## RECOMMENDED NEXT STEPS

1. **Document this issue** (✅ this file)
2. **Senior engineer review** - Decide on proper architecture
3. **Fix root cause** - Not just symptoms
4. **Add tests** - Prevent regression
5. **Update documentation** - Clear file usage guidelines

---

**This is a design decision that needs team consensus, not a quick fix.**
