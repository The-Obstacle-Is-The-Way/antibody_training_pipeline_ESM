# Critical Implementation Analysis: Novo Nordisk Methodology Alignment

**Date:** 2025-11-02
**Status:** INVESTIGATION - DO NOT FIX UNTIL 100% CONFIRMED
**Priority:** P0 - Pipeline Architecture Issue

---

## Executive Summary

Our current implementation achieves **63.88% CV accuracy** and **55.32% Jain test accuracy**, significantly underperforming Novo Nordisk's published benchmarks (**71% CV**, **69% Jain test**).

After comprehensive analysis of the Novo paper (Sakhnini et al. 2025), **we have identified a critical architectural mismatch**: we added StandardScaler preprocessing that Novo **never used**, and we're using a complex BinaryClassifier wrapper that introduces sklearn compatibility issues.

**Root cause**: Over-engineering the pipeline with StandardScaler + custom sklearn wrapper instead of following Novo's simple approach: ESM 1v embeddings → LogisticRegression (no scaling).

---

## 1. Novo Nordisk's Published Methodology

### From Sakhnini et al. (2025) - Section 4.3 "Training and validation of binary classification models"

**Exact methodology (lines 235-244 of paper):**

```
First, the Boughter dataset was parsed into three groups:
- Specific group (0 flags)
- Mildly poly-reactive group (1-3 flags) [EXCLUDED FROM TRAINING]
- Poly-reactive group (>3 flags)

The primary sequences were annotated in the CDRs using ANARCI following the IMGT numbering scheme.
Following this, 16 different antibody fragment sequences were assembled and embedded by three
state-of-the-art protein language models (PLMs), ESM 1v, Protbert bfd, and AbLang2, for
representation of the physico-chemical properties and secondary/tertiary structure.

For the embeddings from the PLMs, *mean* (average of all token vectors) was used.

The vectorised embeddings were served as features for training of binary classification models
(e.g. LogisticReg, RandomForest, GaussianProcess, GradeintBoosting and SVM algorithms) for
non-specificity (class 0: specific group, and class 1: poly-reactive group).
```

**Key Points:**
1. ✅ ESM 1v embeddings with **mean pooling** (average of all token vectors)
2. ✅ Binary classification: 0 flags vs >3 flags (exclude 1-3 flags)
3. ✅ LogisticRegression classifier
4. ❌ **NO mention of StandardScaler or feature normalization**
5. ❌ **NO mention of hyperparameter tuning** (likely sklearn defaults)
6. ✅ 10-fold cross-validation
7. ✅ VH domain for best performance

### Published Results (Figure 1D, Section 2.3):
- **VH domain**: 71% 10-fold CV accuracy
- **Jain test** (parsed, 0 vs >3 flags): 69% accuracy
- **Consistent across 3, 5, 10-fold CV**: ~70-71%

---

## 2. Our Current Implementation

### Architecture Overview

```
BinaryClassifier (custom wrapper)
├── ESMEmbeddingExtractor (ESM 1v mean pooling) ✅ CORRECT
├── StandardScaler ❌ NOT USED BY NOVO
└── LogisticRegression(max_iter=1000, class_weight=None) ⚠️ QUESTIONABLE
```

### Code Analysis

#### `model.py` (Lines 58-62, 119-122): ESM 1v Embedding Extraction
```python
# Masked mean pooling - MATCHES NOVO'S "mean" method ✅
masked_embeddings = embeddings * attention_mask
sum_embeddings = masked_embeddings.sum(dim=1)  # Sum over sequence
sum_mask = attention_mask.sum(dim=1)  # Count valid tokens
mean_embeddings = sum_embeddings / sum_mask  # Average
```

**Analysis**: ✅ **CORRECT** - Our mean pooling implementation matches Novo's description exactly.

---

#### `classifier.py` (Lines 28-38): BinaryClassifier Initialization
```python
self.embedding_extractor = ESMEmbeddingExtractor(params['model_name'], params['device'], batch_size)
self.scaler = StandardScaler()  # ❌ NOT USED BY NOVO

class_weight = params.get('class_weight', None)

self.classifier = LogisticRegression(
    random_state=params['random_state'],
    max_iter=params['max_iter'],  # ⚠️ We use 1000, Novo doesn't specify
    class_weight=class_weight
)
```

**Analysis**:
- ❌ **StandardScaler NOT used by Novo** - We added this assuming it would help
- ⚠️ **max_iter=1000** - Novo doesn't specify (sklearn default is 100)
- ⚠️ **random_state=42** - Novo doesn't mention this
- ✅ **class_weight=None** - Correct for balanced Boughter dataset

---

#### `classifier.py` (Lines 88-102): fit() and predict() Methods
```python
def fit(self, X: np.ndarray, y: np.ndarray):
    # Scale the embeddings
    X_scaled = self.scaler.fit_transform(X)  # ❌ NOVO DIDN'T SCALE

    # Fit the classifier
    self.classifier.fit(X_scaled, y)

def predict(self, X: np.ndarray) -> np.ndarray:
    X_scaled = self.scaler.transform(X)  # ❌ NOVO DIDN'T SCALE
    return self.classifier.predict(X_scaled)
```

**Analysis**: ❌ **CRITICAL ISSUE** - We're scaling embeddings, Novo did not. ESM embeddings may not benefit from scaling or may actually be hurt by it.

---

#### `classifier.py` (Lines 52-86): sklearn Compatibility Wrapper
```python
def get_params(self, deep=True):
    valid_params = {
        'random_state': self.random_state,
        'max_iter': self.max_iter,
        'class_weight': self.class_weight,
        'model_name': self.model_name,
        'device': self.device,
        'batch_size': self.batch_size
    }
    return valid_params

def set_params(self, **params):
    self._params.update(params)
    self.__init__(self._params)  # ❌ REINITIALIZES ENTIRE OBJECT
    return self
```

**Analysis**: ⚠️ **COMPLEX & FRAGILE** - When sklearn's cross_val_score clones the classifier:
1. Calls `get_params()` to get parameters
2. Creates new instance with `BinaryClassifier(**params)`
3. Each fold creates a fresh BinaryClassifier with new ESMEmbeddingExtractor
4. Potential for parameter passing bugs
5. **This complexity is unnecessary** - Novo just used sklearn LogisticRegression directly on pre-computed embeddings

---

#### `train.py` (Lines 174-186): Cross-Validation
```python
# Create a new classifier instance for CV
cv_params = config['classifier'].copy()
cv_params['model_name'] = config['model']['name']
cv_params['device'] = config['model']['device']
cv_params['batch_size'] = config['training'].get('batch_size', 32)
cv_classifier = BinaryClassifier(cv_params)

# FIXED: Pass full BinaryClassifier object (includes StandardScaler)
# This ensures CV uses the same preprocessing pipeline as training/testing
scores = cross_val_score(cv_classifier, X, y, cv=cv, scoring='accuracy')
```

**Analysis**: ⚠️ **ARCHITECTURAL MISMATCH** - We're passing pre-computed embeddings (`X`) to a BinaryClassifier that expects sequences. The embedding_extractor is never used during CV because we already computed embeddings. This creates unnecessary complexity.

**What Novo did**: Compute embeddings once, then pass to simple `LogisticRegression()` for cross_val_score.

---

## 3. Performance Comparison

### Our Results vs Novo's Benchmarks

| Metric | Novo (2025) | Ours (Current) | Gap | Status |
|--------|-------------|----------------|-----|--------|
| Boughter 10-CV | 71.0% | **63.88%** | -7.12% | ❌ 10% worse |
| Jain Test (parsed) | 69.0% | **55.32%** | -13.68% | ❌ 20% worse |

### Historical Performance

| Implementation | CV Accuracy | Jain Accuracy | Notes |
|----------------|-------------|---------------|-------|
| **Before fixes** (bare LogReg, no scaler) | 67.5% | 55.3% | CV bypassed scaler (bug) |
| **After fixes** (with StandardScaler) | **63.88%** | 55.32% | CV uses scaler (correct implementation, but worse!) |
| **Novo (2025)** (no scaler) | **71.0%** | **69.0%** | Published benchmark |

**Critical observation**: Adding StandardScaler **made CV worse** (67.5% → 63.88%). The previous higher score was due to a bug (CV bypassing the scaler), but that "buggy" behavior was actually **closer to Novo's correct approach** (no scaling)!

---

## 4. Root Cause Analysis

### Issue #1: StandardScaler Not Used by Novo ❌ P0

**Evidence:**
- Novo's paper (Section 4.3) makes **zero mention** of feature scaling, normalization, or StandardScaler
- They state: "The vectorised embeddings were served as features for training of binary classification models"
- No preprocessing steps mentioned between embeddings and classifier

**Our implementation:**
```python
# classifier.py:29
self.scaler = StandardScaler()  # ❌ NOT IN NOVO'S METHODOLOGY

# classifier.py:97
X_scaled = self.scaler.fit_transform(X)  # ❌ NOT IN NOVO'S METHODOLOGY
```

**Impact:**
- CV degraded from 67.5% → 63.88% when StandardScaler was properly applied
- Jain test unchanged at 55.32% (both before and after)
- **StandardScaler is hurting performance**, not helping

**Why StandardScaler may hurt ESM embeddings:**
1. ESM embeddings are already normalized representations from a pretrained language model
2. Scaling may destroy important relative magnitudes learned by ESM
3. Novo tested multiple PLMs (ESM 1v, Protbert, AbLang2) without scaling and achieved 66-71% accuracy
4. ESM embeddings live in a meaningful semantic space - standardizing to mean=0, std=1 may distort this

---

### Issue #2: BinaryClassifier Wrapper Complexity ⚠️ P1

**Problem:** We created a custom wrapper class (BinaryClassifier) that combines:
- ESMEmbeddingExtractor (for sequence → embedding)
- StandardScaler (for embedding normalization)
- LogisticRegression (for classification)

**But in practice:**
- We pre-compute embeddings in `train.py` and cache them
- We pass pre-computed embeddings to `cross_val_score`
- The ESMEmbeddingExtractor inside BinaryClassifier is **never used during CV**
- We're using BinaryClassifier as a "sklearn estimator" but it's designed for end-to-end sequence classification

**What Novo did:**
```python
# Pseudo-code based on Novo's methodology
embeddings = esm_model.embed(sequences)  # Compute once
classifier = LogisticRegression()        # Simple sklearn classifier
cv_scores = cross_val_score(classifier, embeddings, labels, cv=10)
```

**Our approach:**
```python
# We pre-compute embeddings
X_train_embedded = get_or_create_embeddings(X_train, ...)

# But then wrap them in a complex BinaryClassifier
cv_classifier = BinaryClassifier(cv_params)  # Includes ESMEmbeddingExtractor that won't be used
scores = cross_val_score(cv_classifier, X_train_embedded, y_train, cv=cv)
```

**sklearn Compatibility Issues:**
When `cross_val_score` clones BinaryClassifier:
1. Calls `get_params()` - must return constructor-compatible params
2. Creates new instance: `BinaryClassifier(**params)`
3. Our `set_params()` calls `self.__init__(self._params)` - **reinitializes entire object**
4. Creates new ESMEmbeddingExtractor each fold (wasteful, though cached embeddings mitigate this)
5. Potential bugs in parameter filtering (had to exclude 'type', 'cv_folds', etc.)

---

### Issue #3: Hyperparameter Differences ⚠️ P2

**Our config:**
```yaml
classifier:
  max_iter: 1000              # We set to 1000
  random_state: 42            # We set to 42
  class_weight: null          # Correct
```

**Novo's methodology:**
- No mention of `max_iter` (sklearn default is 100)
- No mention of `random_state`
- No mention of hyperparameter tuning

**Potential impact:**
- `max_iter=1000` allows more iterations than default, could lead to overfitting
- `random_state=42` ensures reproducibility (good practice, but may not match Novo)

---

## 5. Evidence StandardScaler Is The Problem

### Experimental Evidence

| Configuration | CV Implementation | StandardScaler | CV Accuracy | Jain Accuracy |
|---------------|-------------------|----------------|-------------|---------------|
| Pre-fix (buggy) | Used `cv_classifier.classifier` (bare LogReg) | Bypassed during CV | 67.5% | 55.3% |
| Post-fix (correct) | Used `cv_classifier` (full BinaryClassifier) | Applied during CV | **63.88%** ❌ | 55.32% |
| Novo benchmark | LogisticRegression on embeddings | **Not used** | **71.0%** ✅ | **69.0%** ✅ |

**Key observation:** The "bug" (bypassing StandardScaler during CV) actually produced results **closer to Novo's approach** (no scaling) and achieved **better CV accuracy** (67.5% vs 63.88%).

### Why is Jain test still low (55.32%) even with correct file?

The Jain test accuracy hasn't improved because:
1. **We're using StandardScaler** - Novo didn't use it, and it's hurting performance across the board
2. **Training accuracy is very high (95.62%)** - suggests overfitting, possibly due to max_iter=1000
3. **Domain shift between Boughter and Jain** - but Novo bridged this gap to achieve 69% on Jain

---

## 6. sklearn 2025 Issues Investigation

### Potential sklearn Compatibility Issues

**Question:** Could sklearn version differences explain the degradation?

**Investigation needed:**
```bash
# Check sklearn version
pip show scikit-learn

# Novo's paper (Table 3) shows:
# "Scikit-Learn | https://scikit-learn.org | [76]"
# Reference [76]: Pedregosa et al. (2011) - original sklearn paper
# They don't specify exact version
```

**Cross-validation changes in sklearn:**
- sklearn 1.0+ (2021): Changed some CV internals for better parallelization
- sklearn 1.3+ (2023): Improved estimator cloning
- sklearn 1.4+ (2024): Enhanced parameter validation

**Our BinaryClassifier's get_params/set_params may have issues with:**
- Parameter validation in newer sklearn
- Cloning complex nested objects
- Handling custom estimators that wrap other estimators

**But this is secondary** - the primary issue is StandardScaler, not sklearn version.

---

## 7. Recommended Investigation Plan

### Phase 1: Confirm StandardScaler Is The Problem ✅ PRIORITY 1

**Test 1: Remove StandardScaler, use simple LogisticRegression**
```python
# Compute embeddings (keep current implementation)
X_train_embedded = get_or_create_embeddings(X_train, ...)

# Simple sklearn classifier (NO BinaryClassifier wrapper, NO StandardScaler)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=42, max_iter=1000)

# Cross-validation
cv_scores = cross_val_score(classifier, X_train_embedded, y_train, cv=10, scoring='accuracy')

# Expected: CV ~70-71% (matching Novo)
```

**Test 2: Try sklearn default hyperparameters**
```python
# Use sklearn defaults (max_iter=100, no random_state)
classifier = LogisticRegression()

# Expected: May match Novo's results even better
```

**Test 3: Test on Jain with corrected approach**
```python
# Train on Boughter embeddings without scaling
classifier.fit(X_train_embedded, y_train)

# Test on Jain embeddings without scaling
X_jain_embedded = get_or_create_embeddings(X_jain, ...)
accuracy = classifier.score(X_jain_embedded, y_jain)

# Expected: ~69% (matching Novo)
```

### Phase 2: Simplify Architecture ⚠️ PRIORITY 2

**Goal:** Match Novo's simple approach exactly

**Current (complex):**
```
BinaryClassifier wrapper
├── ESMEmbeddingExtractor
├── StandardScaler ❌
└── LogisticRegression
```

**Proposed (simple, matches Novo):**
```
1. Compute embeddings (separate step, cached)
2. sklearn LogisticRegression (no wrapper, no scaling)
3. Direct cross_val_score on embeddings
```

**Benefits:**
- ✅ Matches Novo's methodology exactly
- ✅ No sklearn compatibility issues
- ✅ Simpler, more maintainable code
- ✅ Faster (no object cloning overhead)
- ✅ Easier to debug

### Phase 3: Hyperparameter Alignment ⚠️ PRIORITY 3

**Test sklearn defaults vs our config:**
```python
# Test 1: Our current settings
classifier = LogisticRegression(random_state=42, max_iter=1000)

# Test 2: sklearn defaults
classifier = LogisticRegression()  # max_iter=100, no random_state

# Test 3: Novo's likely settings (based on their results)
# They don't specify, so likely used defaults
```

---

## 8. Expected Outcomes After Fixes

### If StandardScaler is the problem:

**Removing StandardScaler should yield:**
- ✅ Boughter 10-CV: ~70-71% (matching Novo)
- ✅ Jain test: ~65-69% (matching Novo)
- ✅ Training accuracy: ~75-80% (less overfitting)

**This would explain:**
- Why our CV was worse with StandardScaler (63.88%) than without it in the buggy version (67.5%)
- Why we're 7-14% below Novo's benchmarks
- Why the "bug fix" made things worse

### If there are other issues:

**Additional investigations needed:**
1. Check Boughter dataset filtering (ensure we're using same subset as Novo)
2. Verify ESM 1v model version (facebook/esm1v_t33_650M_UR90S_1 vs others)
3. Check random seed effects
4. Verify label distribution (0 flags vs >3 flags parsing)

---

## 9. Action Items

### Before Making Any Changes:

- [ ] Run Test 1: LogisticRegression without StandardScaler (no wrapper)
- [ ] Run Test 2: sklearn default hyperparameters
- [ ] Run Test 3: Jain test with corrected approach
- [ ] Document all test results in this file
- [ ] Get alignment on findings before proceeding to fixes

### After Confirming Root Cause:

- [ ] Remove StandardScaler from BinaryClassifier
- [ ] Simplify to match Novo's architecture (embeddings → LogisticRegression)
- [ ] Update config to use sklearn defaults (unless Novo specifies otherwise)
- [ ] Re-run full training pipeline
- [ ] Verify CV ~70-71% and Jain ~65-69%
- [ ] Document final results

---

## 10. Key References

### Novo Nordisk Paper (Sakhnini et al. 2025)

**Title:** "Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters"

**Key Methods Section (4.3, lines 235-244):**
```
"First, the Boughter dataset was parsed into three groups as previously done in [44]:
specific group (0 flags), mildly poly-reactive group (1-3 flags) and poly-reactive group
(>3 flags). The primary sequences were annotated in the CDRs using ANARCI following the
IMGT numbering scheme. Following this, 16 different antibody fragment sequences were
assembled and embedded by three state-of-the-art protein language models (PLMs),
ESM 1v, Protbert bfd, and AbLang2, for representation of the physico-chemical properties
and secondary/tertiary structure, and a physico-chemical descriptor of amino acids,
the Z-scale. For the embeddings from the PLMs, *mean* (average of all token vectors) was used.
The vectorised embeddings were served as features for training of binary classification models
(e.g. LogisticReg, RandomForest, GaussianProcess, GradeintBoosting and SVM algorithms) for
non-specificity (class 0: specific group, and class 1: poly-reactive group)."
```

**Results (Figure 1D, Section 2.3):**
- VH domain ESM 1v LogisticReg: **71% 10-fold CV accuracy**
- Tested on parsed Jain dataset: **69% accuracy**
- Consistent across 3, 5, 10-fold CV: ~70-71%

**Critical observation:**
NO MENTION of StandardScaler, feature normalization, or any preprocessing between embeddings and classifier.

### Our Implementation Files

**Files to review:**
- `model.py:58-62, 119-122` - ESM embedding extraction (✅ CORRECT)
- `classifier.py:28-38` - StandardScaler initialization (❌ NOT IN NOVO)
- `classifier.py:88-102` - fit/predict with scaling (❌ NOT IN NOVO)
- `classifier.py:52-86` - sklearn compatibility wrapper (⚠️ COMPLEX)
- `train.py:174-186` - Cross-validation setup (⚠️ ARCHITECTURAL MISMATCH)
- `config_boughter.yaml` - Hyperparameter config (⚠️ DIFFERS FROM NOVO)

---

## 11. Conclusion

**We are NOT aligned with Novo's methodology.**

**Critical divergence:** We added StandardScaler preprocessing that Novo never used, and our results confirm it's hurting performance:
- Novo (no scaler): 71% CV, 69% Jain
- Ours (with scaler): 63.88% CV, 55.32% Jain
- Our buggy version (bypassing scaler): 67.5% CV - **closer to Novo!**

**Path forward:**
1. **IMMEDIATE**: Test removing StandardScaler → expect ~70-71% CV
2. **SHORT-TERM**: Simplify architecture to match Novo (embeddings → LogisticRegression)
3. **MEDIUM-TERM**: Align all hyperparameters with Novo's (likely sklearn defaults)

**DO NOT FIX UNTIL:**
- We run the three tests outlined in Section 7, Phase 1
- We confirm StandardScaler is indeed the root cause
- We document test results and get alignment

---

**Last Updated:** 2025-11-02
**Next Review:** After completing Phase 1 tests
**Status:** INVESTIGATION PHASE - NO FIXES YET
