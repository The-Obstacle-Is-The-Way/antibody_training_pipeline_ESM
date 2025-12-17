# Phase C: Pipeline Integration (Hybrid Model)

> **DEPRECATED (2025-11-28)**: This phase was a mistake. The Novo Nordisk paper
> runs Track A (ESM) and Track B (biophysical) as **separate parallel experiments**,
> never combined. This hybrid approach has no scientific justification and should
> be removed. See [PHASE_D_HYBRID_REMOVAL.md](PHASE_D_HYBRID_REMOVAL.md) for cleanup plan.

**Date**: 2025-11-27
**Status**: ~~IMPLEMENTED~~ **DEPRECATED - PENDING REMOVAL**
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

### Important: No StandardScaler (Novo Methodology)

**After reviewing the Sakhnini et al. 2025 paper, there is NO mention of StandardScaler
for either ESM embeddings or biophysical descriptors.** Novo Nordisk feeds raw features
directly to LogisticRegression.

We match this methodology exactly:
- ESM embeddings: NOT scaled (already normalized by model architecture)
- Biophysical features: NOT scaled (raw charge and pI values)

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

    # X_train is list of sequences (strings)
    # X_train_embedded is ESM embeddings (n, 1280)

    # NEW: Optional biophysical features (after ESM embedding extraction)
    # NOTE: No StandardScaler used - matches Novo Nordisk methodology exactly
    if config.features.use_biophysical:
        from antibody_training_esm.core.biophysical import BiophysicalExtractor
        bio_extractor = BiophysicalExtractor()
        X_bio = bio_extractor.extract_batch_features(X_train)  # X_train = sequences
        X_train_embedded = np.concatenate([X_train_embedded, X_bio], axis=1)
```

**Note**: `X_train` contains raw sequences (list of strings), while `X_train_embedded` contains ESM embeddings. The biophysical extractor operates on sequences.

### 4.2 Configuration Updates (`src/antibody_training_esm/conf/config.yaml`)

**NEW**: Create `features` config group (does not exist yet):

```yaml
# config.yaml - add features section (NEW - currently no features config exists)
features:
  use_biophysical: false  # Enable biophysical descriptors (default: ESM-only)
```

Existing config groups: `model/`, `classifier/`, `data/`, `hardware/`, `hydra/`

### 4.3 Caching Strategy

Biophysical features are fast to compute (~0.1s for 1000 sequences), so caching is optional but nice-to-have:

```python
# Cache path: experiments/cache/{dataset}_biophysical_{hash}.pkl
# (Matches existing ESM embedding cache format)
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

- [x] `features.use_biophysical` config flag works
- [x] Feature concatenation produces correct shape (n, 1283)
- [x] Training works with hybrid features
- [x] Integration tests pass (12 tests in `test_hybrid_pipeline.py`)
- [x] Can run `uv run antibody-train features=hybrid`
- [x] Backward compatible (default: biophysical disabled)

---

## 7. Scientific Validation

After integration, compare on Jain test set:
- **ESM-only**: ~66-69% accuracy (our current baseline: 68.60% test, 67.5% CV)
- **Biophysical-only**: ~63-65% (Phase B: 63.18% CV, paper pI-only: 65.2%)
- **Hybrid (ESM + Bio)**: Target ≥67% (should not decrease from ESM-only)

**Note**: Novo reported 71% CV accuracy; our reproduction achieves 67.5% ± 8.9% which is within statistical variance. The goal is to match or exceed our own ESM baseline, not the reported 71%.

If hybrid performs worse than ESM-only, the biophysical features may be adding noise. This would inform whether to keep the integration.

---

**Dependencies**: Requires completion of Phase B (Baseline Reproducibility) to confirm descriptors work standalone.
