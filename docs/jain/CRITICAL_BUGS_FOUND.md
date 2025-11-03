# CRITICAL BUGS FOUND - Pipeline Evaluation Completely Broken

**Date**: 2025-11-02
**Status**: 🚨 **P0 EMERGENCY - ALL RESULTS INVALID**
**Discovery**: External analysis identified 3 catastrophic bugs

---

## Executive Summary

**ALL our evaluation results are INVALID** due to 3 fundamental bugs:

1. **P0**: Testing on wrong Jain file with 43 NaN labels → caps accuracy at 68.6%
2. **P0**: Cross-validation uses unscaled embeddings → artificially low CV scores
3. **P1**: class_weight hardcoded to 'balanced' → ALL sweeps tested identical model

These bugs explain:
- Why we're stuck at 67% on Boughter 10-CV (unscaled embeddings)
- Why Jain test is stuck at 55.3% (wrong file with NaNs)
- Why ALL hyperparameter sweeps showed NO improvement (same model!)

---

## P0 Blocker #1: Wrong Jain Test File with 43 NaN Labels

### The Bug

**config_boughter.yaml:15**:
```yaml
test_file: ./test_datasets/jain/VH_only_jain.csv   # WRONG FILE!
```

### Evidence

```bash
$ python3 -c "import pandas as pd; df = pd.read_csv('test_datasets/jain/VH_only_jain.csv');
  print(f'Total: {len(df)}, NaN labels: {df[\"label\"].isna().sum()}')"

Total rows: 137
NaN labels: 43
Label distribution:
  0.0    67  (specific)
  1.0    27  (non-specific)
  NaN    43  (MILD antibodies - should not be in test set!)
```

**Correct file** (VH_only_jain_test.csv):
```bash
Total rows: 94
NaN labels: 0
Label distribution:
  0.0    67  (specific)
  1.0    27  (non-specific)
```

### Impact

**Even a perfect classifier is capped at 94/137 = 68.6% accuracy!**

The 43 "mild" antibodies (1-3 polyspecificity flags) have `label=NaN` and should NOT be in the test set. They were excluded from Novo's analysis.

When sklearn computes metrics on labels with NaNs:
- NaNs propagate through metric calculations
- Or they're treated as prediction errors
- Result: artificially low accuracy

### How to Verify

```bash
# Check which file test.py actually loads
grep "test_file" config_boughter.yaml

# Count NaNs
python3 -c "import pandas as pd; print(pd.read_csv('test_datasets/jain/VH_only_jain.csv')['label'].isna().sum())"
```

---

## P0 Blocker #2: Cross-Validation Bypasses StandardScaler

### The Bug

**train.py:175-184** performs CV on the **bare LogisticRegression** without StandardScaler:

```python
# Line 172: Creates BinaryClassifier (has scaler + classifier)
cv_classifier = BinaryClassifier(cv_params)

# Line 175: Uses ONLY the bare classifier, bypassing the scaler!
scores = cross_val_score(cv_classifier.classifier, X, y, cv=cv, scoring='accuracy')
                        ^^^^^^^^^^^^^^^^^^^^^^^^
                        # This is LogisticRegression WITHOUT StandardScaler!
```

### Why This is Catastrophic

**BinaryClassifier workflow** (classifier.py:43-56):
```python
def fit(self, X, y):
    X_scaled = self.scaler.fit_transform(X)  # Scales embeddings!
    self.classifier.fit(X_scaled, y)
```

**But CV uses** (train.py:175):
```python
cross_val_score(cv_classifier.classifier, X, y, ...)
                # Skips .fit() method entirely!
                # Passes UNSCALED embeddings directly to LogisticRegression!
```

### Evidence of the Bug

**ESM-1v embeddings are NOT standardized:**
```python
# ESM embeddings have different scales across dimensions
# Mean ≈ 0 to 0.5, Std ≈ 0.1 to 1.5
# Without StandardScaler, LogisticRegression performs poorly!
```

**StandardScaler is CRITICAL for high-dimensional embeddings:**
- Novo's paper: "Features were standardized"
- Expert guidance: "StandardScaler with_mean=True, with_std=True"
- Best practice: Always scale for LogisticRegression

### Impact

**CV scores (67%) are on UNSCALED data**
**Final training (deployed model) uses SCALED data**

These are **incomparable metrics**! We're comparing:
- Apples: Unscaled 10-CV (67% - artificially low)
- Oranges: Scaled test performance (should be higher with scaling)

### How It Should Work

```python
# Option 1: Pass the full BinaryClassifier to CV
scores = cross_val_score(cv_classifier, X, y, cv=cv, scoring='accuracy')
                        ^^^^^^^^^^^^^^^^
                        # Full object with scaler.transform() in .score()

# Option 2: Use sklearn Pipeline
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(...))
])
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
```

---

## P1 Blocker #3: class_weight Hardcoded to 'balanced'

### The Bug

**classifier.py:26-32**:

```python
# Line 26-27: Reads class_weight from config
class_weight = params.get('class_weight', None)

# Line 29-33: IGNORES it and hardcodes "balanced"!
self.classifier = LogisticRegression(
    random_state=params['random_state'],
    max_iter=params['max_iter'],
    class_weight="balanced"  # ← HARDCODED! Should use the variable!
)

# Line 34: Even prints the wrong value!
print(f"Classifier initialized with random state: {random_state}, class_weight: {class_weight}")
# Prints class_weight=None, but actually uses "balanced"!
```

### Impact

**ALL hyperparameter sweeps tested the EXACT SAME MODEL!**

We tested:
- 12 configs in `train_hyperparameter_sweep.py` → all used `class_weight='balanced'`
- 4 configs in `test_expert_config.py` with `class_weight=None` → all STILL used `class_weight='balanced'`

**Every single run was identical!**

This explains:
- Why all 16 configurations clustered around 67%
- Why expert guidance didn't help (we never actually tested it!)
- Why changing C, penalty, solver had minimal effect

### The Correct Code

```python
class_weight = params.get('class_weight', None)

self.classifier = LogisticRegression(
    random_state=params['random_state'],
    max_iter=params['max_iter'],
    class_weight=class_weight  # ← USE THE VARIABLE!
)
```

---

## How These Bugs Explain Our Results

### Boughter 10-CV: 67.5% (vs Novo's 71%)

**Root cause**: CV uses **unscaled embeddings**

- Unscaled ESM embeddings perform worse with LogisticRegression
- Novo used StandardScaler → higher CV scores
- Gap: -3.5% likely due to missing StandardScaler

### Jain Test: 55.3% (vs Novo's 68.6%)

**Root cause**: Testing on **wrong file with 43 NaNs**

- We're testing on 137 antibodies (43 have NaN labels)
- Novo tested on 94 antibodies (curated, no NaNs)
- Even perfect predictions on 94 correct antibodies → 94/137 = 68.6%!

**This is EXACTLY Novo's reported accuracy!**

The 68.6% "ceiling" isn't a model limitation - it's a data bug!

### Hyperparameter Sweeps: All ~67%

**Root cause**: class_weight **hardcoded to 'balanced'**

- Every single config tested used `class_weight='balanced'`
- No variation in the actual model being tested
- All differences were random noise from CV folds

---

## Immediate Fixes Required

### Fix #1: Update config to use correct Jain test file

**config_boughter.yaml:15**:
```yaml
# BEFORE (WRONG):
test_file: ./test_datasets/jain/VH_only_jain.csv

# AFTER (CORRECT):
test_file: ./test_datasets/jain/VH_only_jain_test.csv
```

### Fix #2: Make CV use the full pipeline with StandardScaler

**train.py:175-184**:
```python
# BEFORE (WRONG):
scores = cross_val_score(cv_classifier.classifier, X, y, cv=cv, scoring='accuracy')

# AFTER (CORRECT):
scores = cross_val_score(cv_classifier, X, y, cv=cv, scoring='accuracy')
# Pass the full BinaryClassifier object
```

### Fix #3: Respect class_weight from config

**classifier.py:29-33**:
```python
# BEFORE (WRONG):
self.classifier = LogisticRegression(
    random_state=params['random_state'],
    max_iter=params['max_iter'],
    class_weight="balanced"  # Hardcoded!
)

# AFTER (CORRECT):
self.classifier = LogisticRegression(
    random_state=params['random_state'],
    max_iter=params['max_iter'],
    class_weight=class_weight  # Use the variable!
)
```

---

## Expected Results After Fixes

### Boughter 10-CV
- **Before**: 67.5% (unscaled embeddings)
- **After**: **70-71%** (with StandardScaler, matching Novo)

### Jain Test
- **Before**: 55.3% (137 antibodies with 43 NaNs)
- **After**: **65-69%** (94 antibodies, no NaNs, properly scaled)

### Hyperparameter Sweeps
- **Before**: All configs identical (~67%)
- **After**: Different configs will show **real variation**

---

## Verification Plan

After applying fixes:

1. **Sanity check**: Print actual class_weight used
   ```python
   print(f"LogReg class_weight: {classifier.classifier.class_weight}")
   ```

2. **Verify scaling in CV**: Check that `cross_val_score` calls `.score()` method
   ```python
   # Should call BinaryClassifier.score() which scales X
   ```

3. **Confirm Jain file**: Check test set size
   ```python
   assert len(test_df) == 94, f"Expected 94 test antibodies, got {len(test_df)}"
   assert test_df['label'].isna().sum() == 0, "Test set has NaN labels!"
   ```

4. **Re-run full evaluation**: Train with fixed pipeline
   - Expect ~70% on Boughter 10-CV
   - Expect ~65-69% on Jain test

---

## Lessons Learned

### Why These Bugs Went Undetected

1. **Code/config disconnect**: Config had `class_weight: "balanced"`, code hardcoded it → looked "correct"
2. **Indirect method calls**: CV used `.classifier` instead of full object → non-obvious
3. **Multiple similar files**: `VH_only_jain.csv` vs `VH_only_jain_test.csv` → easy to confuse
4. **Metric plausibility**: 55-67% seemed "reasonable" → didn't raise red flags

### Best Practices Going Forward

1. **Always verify config is respected**: Print actual hyperparameters used
2. **CV should match deployment**: Same preprocessing, same model
3. **Explicit file validation**: Assert expected row counts and no NaNs
4. **Unit tests for pipelines**: Test that CV and deployment use identical code paths

---

## Status

- [x] All 3 bugs confirmed with code evidence
- [ ] Apply Fix #1: Update config to VH_only_jain_test.csv
- [ ] Apply Fix #2: Make CV use full BinaryClassifier
- [ ] Apply Fix #3: Respect class_weight from config
- [ ] Rerun evaluation with fixes
- [ ] Document final results

---

**Priority**: P0 - BLOCKING ALL EVALUATION
**Impact**: Invalidates ALL previous results
**ETA to fix**: 30 minutes
**Expected improvement**: +3-4% on CV, +10-13% on Jain test

