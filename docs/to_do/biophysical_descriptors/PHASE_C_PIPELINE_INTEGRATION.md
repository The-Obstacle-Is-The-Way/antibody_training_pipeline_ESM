# Phase C: Pipeline Integration (Hybrid Model)

**Date**: 2025-11-27
**Status**: PENDING
**Parent**: [BIOPHYSICAL_IMPLEMENTATION_SPECS.md](BIOPHYSICAL_IMPLEMENTATION_SPECS.md)

---

## 1. Objective

Integrate the Phase A `BiophysicalExtractor` into the main training pipeline (`antibody-train`) to enable "Hybrid Models" (ESM Embeddings + Biophysical Descriptors).

**Goal**: Allow the classifier to learn from both deep representations (ESM: 1280-d) and explicit biophysical properties (3-d: Charge@pH6, Charge@pH7.4, pI).

---

## 2. Current Architecture (IMPORTANT)

**The `BinaryClassifier` is sklearn-based, NOT PyTorch.**

```python
# Current architecture (src/antibody_training_esm/core/classifier.py)
class BinaryClassifier:
    """sklearn-style classifier wrapping ESM embeddings + LogisticRegression/XGBoost"""

    embedding_extractor: EmbeddingExtractorProtocol  # ESM or AMPLIFY
    _classifier: ClassifierStrategy  # LogisticRegression or XGBoost

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """X is embeddings (n, 1280), y is labels"""
        self._classifier.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted labels"""
        return self._classifier.predict(X)
```

**Key insight**: Integration is about **feature concatenation** before sklearn, not PyTorch layer modifications.

---

## 3. Integration Strategy

### Option A: Feature Concatenation (Recommended)

Concatenate biophysical features to ESM embeddings before passing to sklearn classifier.

```python
# Hybrid feature vector
X_esm = embedding_extractor.extract_batch_embeddings(sequences)  # (n, 1280)
X_bio = biophysical_extractor.extract_batch_features(sequences)   # (n, 3)
X_hybrid = np.concatenate([X_esm, X_bio], axis=1)                 # (n, 1283)

# Standard sklearn fit
classifier.fit(X_hybrid, y)
```

**Pros**: Simple, uses existing infrastructure, no architectural changes needed.

### Option B: Separate Biophysical Head (Future)

Train two separate models and ensemble predictions.

**Pros**: Can weight contributions differently.
**Cons**: More complex, requires ensemble logic.

---

## 4. Implementation Plan

### 4.1 Trainer Updates (`src/antibody_training_esm/core/trainer.py`)

Add optional biophysical feature extraction to the training pipeline:

```python
def train_pipeline(cfg: DictConfig) -> dict[str, Any]:
    # ... existing code ...

    # NEW: Optional biophysical features
    if config.features.use_biophysical:
        from antibody_training_esm.core.biophysical import BiophysicalExtractor
        bio_extractor = BiophysicalExtractor()
        X_bio = bio_extractor.extract_batch_features(X_train)
        X_train_embedded = np.concatenate([X_train_embedded, X_bio], axis=1)
```

### 4.2 Configuration Updates (`src/antibody_training_esm/conf/config.yaml`)

```yaml
# config.yaml - add features section
features:
  use_biophysical: false  # Enable biophysical descriptors
  standardize_biophysical: true  # StandardScaler on biophysical features
```

### 4.3 Caching Strategy

Biophysical features are fast to compute (~0.1s for 1000 sequences), so caching is optional but nice-to-have:

```python
# Cache path: experiments/cache/{dataset}_biophysical_{hash}.npy
```

---

## 5. TDD Plan

**Test File**: `tests/integration/test_hybrid_pipeline.py`

1. **`test_feature_concatenation_shape`**:
   * ESM: (100, 1280) + Bio: (100, 3) → Hybrid: (100, 1283)
   * Verify dtypes match (float32)

2. **`test_classifier_accepts_hybrid_features`**:
   * Initialize BinaryClassifier
   * fit() with (n, 1283) features
   * predict() returns correct shape

3. **`test_config_flag_enables_biophysical`**:
   * Load config with `features.use_biophysical: true`
   * Verify pipeline extracts biophysical features

---

## 6. Acceptance Criteria

- [ ] `features.use_biophysical` config flag works
- [ ] Feature concatenation produces correct shape (n, 1283)
- [ ] Training works with hybrid features
- [ ] Integration tests pass
- [ ] Can run `uv run antibody-train features.use_biophysical=true`
- [ ] Backward compatible (default: biophysical disabled)

---

## 7. Scientific Validation

After integration, compare on Jain test set:
- **ESM-only**: ~71% accuracy (current baseline)
- **Biophysical-only**: ~65% (Phase B establishes this)
- **Hybrid (ESM + Bio)**: Target ≥71% (should not decrease)

If hybrid performs worse than ESM-only, the biophysical features may be adding noise. This would inform whether to keep the integration.

---

**Dependencies**: Requires completion of Phase B (Baseline Reproducibility) to confirm descriptors work standalone.
