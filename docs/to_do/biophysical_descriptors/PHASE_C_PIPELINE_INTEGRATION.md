# Phase C: Pipeline Integration (Hybrid Model)

**Date**: 2025-11-27
**Status**: PENDING
**Parent**: [BIOPHYSICAL_IMPLEMENTATION_SPECS.md](BIOPHYSICAL_IMPLEMENTATION_SPECS.md)

---

## 1. Objective

Integrate the Phase A `BiophysicalExtractor` into the main PyTorch training pipeline (`antibody-train`) to enable "Hybrid Models" (ESM Embeddings + Biophysical Descriptors).

**Goal**: Allow the classifier to learn from both deep representations (ESM) and explicit biophysical properties (Charge, pI).

## 2. Architecture Changes

### 2.1 Data Loading (`src/antibody_training_esm/datasets/`)

The current `Dataset` classes (Boughter, Jain, etc.) load sequences and labels. We need to augment them to provide biophysical features.

**Strategy**: "Lazy Extraction with Caching"
*   Add `use_biophysical: bool = False` to Dataset constructors.
*   If `True`:
    *   Check for cached descriptors (e.g., `data/cache/{dataset_name}_biophysical.npy`).
    *   If missing, compute using `BiophysicalExtractor` and save.
    *   Yield `(embedding, biophysical_features, label)` instead of `(embedding, label)`.

### 2.2 Model Architecture (`src/antibody_training_esm/core/classifier.py`)

The `BinaryClassifier` (PyTorch LightningModule) needs to accept auxiliary features.

*   **Current Forward**: `x (Batch, 1280) -> ... -> logits`
*   **New Forward**: `x (Batch, 1280), x_aux (Batch, 3) -> ... -> logits`

**Implementation Detail**:
*   Add `aux_input_dim: int = 0` to `ClassifierConfig`.
*   In `__init__`:
    *   If `aux_input_dim > 0`: Create a separate input head or plan for concatenation.
    *   Concatenate `cat([x, x_aux], dim=1)` before the first Linear layer.
    *   Input dimension becomes `embedding_dim + aux_input_dim`.

### 2.3 Configuration (`src/antibody_training_esm/conf/`)

Update Hydra configs to support this mode.

```yaml
# config.yaml
model:
  use_biophysical_descriptors: true
```

## 3. TDD Plan

**Test File**: `tests/integration/test_hybrid_pipeline.py`

1.  **`test_dataset_yields_aux_features`**:
    *   Initialize Dataset with `use_biophysical=True`.
    *   Verify `__getitem__` returns tuple of length 3 (or dict).
2.  **`test_classifier_forward_with_aux`**:
    *   Initialize `BinaryClassifier` with `aux_input_dim=3`.
    *   Pass random tensors for embedding and aux.
    *   Verify output shape.
3.  **`test_training_step_with_aux`**:
    *   Run one training step with hybrid data.
    *   Ensure loss is calculated and gradients flow.

## 4. Acceptance Criteria

*   [ ] `Dataset` classes support `use_biophysical` flag.
*   [ ] `BinaryClassifier` accepts auxiliary features.
*   [ ] Hydra config updated.
*   [ ] Integration test passes.
*   [ ] Can run `uv run antibody-train model.use_biophysical_descriptors=true`.

---

**Dependencies**: Requires completion of Phase B (Baseline Reproducibility) to confirm the descriptors are worth integrating.
