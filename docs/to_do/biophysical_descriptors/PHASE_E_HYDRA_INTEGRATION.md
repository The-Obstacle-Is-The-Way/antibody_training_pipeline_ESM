# Phase E: Hydra Integration for Track B (Biophysical)

**Date**: 2025-11-28
**Status**: PENDING
**Depends On**: Phase A (BiophysicalExtractor), Phase D (Hybrid Removal)
**Blocked By**: None

---

## 1. Objective

Integrate Track B (biophysical descriptors) into the Hydra-based training pipeline so that:
- Both Track A (ESM) and Track B (Biophysical) use identical infrastructure
- Configuration is done via YAML, not hardcoded paths
- Output structure is consistent between tracks
- A single CLI command switches between tracks: `antibody-train model=biophysical`

---

## 2. Background

### Current State (Broken)
- Track B implemented as standalone script: `cli/reproduce_track_b.py`
- Hardcoded paths to Boughter/Jain datasets
- Output to `experiments/benchmarks/track_b_baseline.json` (non-standard location)
- Does NOT use Hydra
- Only tests on Jain (missing Harvey, Shehata)

### Desired State
- Track B uses same Hydra config system as Track A
- `conf/model/biophysical.yaml` config file
- Output to `experiments/checkpoints/biophysical/logreg/*.pkl`
- Full dataset testing (Jain, Harvey, Shehata)

---

## 3. Key Discovery: Infrastructure Exists

The codebase already has extensible model type support:

```python
# models/config.py
model_type: Literal["esm", "amplify"] = "esm"

# classifier.py
SUPPORTED_MODEL_TYPES = {"esm", "amplify"}
```

We just need to add `biophysical` as a third option.

---

## 4. Files to Create

### 4.1 `src/antibody_training_esm/conf/model/biophysical.yaml`

```yaml
# Track B: Biophysical Descriptors (Biopython Trio)
# Implements Table S1 #21, #22, #66 from Sakhnini et al. 2025
#
# Features:
#   - Charge at pH 6.0 (endosome)
#   - Charge at pH 7.4 (blood)
#   - Theoretical pI (isoelectric point)
#
# Usage: uv run antibody-train model=biophysical

name: biophysical
device: cpu              # Biopython is CPU-only (no GPU acceleration)
batch_size: 1            # Not batched like neural net models
model_type: biophysical
revision: "1.0.0"        # BiophysicalExtractor version for cache invalidation
trust_remote_code: false
```

### 4.2 `src/antibody_training_esm/core/embeddings_biophysical.py`

```python
"""
Biophysical Embedding Extractor

Wrapper for BiophysicalExtractor that conforms to the embedding extractor
protocol used by BinaryClassifier. This enables Track B to use the same
training infrastructure as Track A (ESM).

Note: "embeddings" is a misnomer for biophysical features, but we use this
naming for consistency with the existing codebase architecture.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from antibody_training_esm.core.biophysical import BiophysicalExtractor


class BiophysicalEmbeddingExtractor:
    """
    Embedding-like interface for biophysical feature extraction.

    Wraps BiophysicalExtractor to match the interface expected by
    BinaryClassifier, enabling Track B to use Hydra pipeline.

    Attributes:
        model_name: Always "biophysical" for this extractor.
        device: Always "cpu" (Biopython is CPU-only).
        batch_size: Ignored (single-sequence processing).
        revision: Version string for cache invalidation.
        max_length: No limit for biophysical features.
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
        revision: str = "1.0.0",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the biophysical extractor.

        Args:
            model_name: Model identifier (should be "biophysical").
            device: Device to use (ignored, always CPU).
            batch_size: Batch size (ignored, single-sequence processing).
            revision: Version for cache key generation.
            **kwargs: Additional arguments (ignored for compatibility).
        """
        self.biophysical = BiophysicalExtractor()
        self.model_name = "biophysical"
        self.device = "cpu"  # Force CPU - Biopython has no GPU support
        self.batch_size = 1  # Not batched
        self.revision = revision
        self.max_length = float("inf")  # No sequence length limit

    def embed_sequence(self, sequence: str) -> NDArray[np.float32]:
        """
        Extract biophysical features for a single sequence.

        Args:
            sequence: Amino acid sequence (VH domain).

        Returns:
            1D array of shape (3,) containing:
                - charge_ph6: Charge at pH 6.0
                - charge_ph7_4: Charge at pH 7.4
                - theoretical_pi: Isoelectric point
        """
        return self.biophysical.extract_features(sequence)

    def extract_batch_embeddings(
        self, sequences: list[str]
    ) -> NDArray[np.float32]:
        """
        Extract biophysical features for a batch of sequences.

        Args:
            sequences: List of amino acid sequences.

        Returns:
            2D array of shape (n_sequences, 3).
        """
        return self.biophysical.extract_batch_features(sequences)
```

---

## 5. Files to Modify

### 5.1 `src/antibody_training_esm/models/config.py`

**Change**: Add `biophysical` to `model_type` Literal

```python
# Before:
model_type: Literal["esm", "amplify"] = "esm"

# After:
model_type: Literal["esm", "amplify", "biophysical"] = "esm"
```

### 5.2 `src/antibody_training_esm/core/classifier.py`

**Change 1**: Update supported types constant

```python
# Before:
SUPPORTED_MODEL_TYPES = {"esm", "amplify"}

# After:
SUPPORTED_MODEL_TYPES = {"esm", "amplify", "biophysical"}
```

**Change 2**: Add factory branch in `__init__`

```python
# In __init__, after the amplify branch:
elif model_type == "biophysical":
    from antibody_training_esm.core.embeddings_biophysical import (
        BiophysicalEmbeddingExtractor,
    )
    self.embedding_extractor = BiophysicalEmbeddingExtractor(
        model_name=params["model_name"],
        device=params["device"],
        batch_size=batch_size,
        revision=revision,
    )
```

---

## 6. Files to Delete (After Validation)

These files become obsolete once Phase E is complete:

| File | Reason |
|------|--------|
| `src/antibody_training_esm/cli/reproduce_track_b.py` | Replaced by Hydra pipeline |
| `tests/unit/cli/test_reproduce_track_b.py` | Tests for deleted script |
| `experiments/benchmarks/track_b_baseline.json` | Output moves to Hydra runs |

---

## 7. Acceptance Criteria

### 7.1 Functional
- [ ] `uv run antibody-train model=biophysical` runs without error
- [ ] 10-fold CV accuracy on Boughter ~63% (matches standalone script)
- [ ] Test accuracy on Jain ~56% (matches standalone script)
- [ ] Model saved to `experiments/checkpoints/biophysical/logreg/*.pkl`

### 7.2 Integration
- [ ] `make test` passes (all unit + integration tests)
- [ ] `make typecheck` passes (mypy strict mode)
- [ ] `make lint` passes (ruff)

### 7.3 Architecture
- [ ] No hardcoded paths in new code
- [ ] Config-driven via Hydra YAML
- [ ] Same output structure as Track A (ESM)

---

## 8. Testing Strategy

### 8.1 Unit Tests (New)
- `tests/unit/core/test_embeddings_biophysical.py`
  - Test `embed_sequence` returns shape (3,)
  - Test `extract_batch_embeddings` returns shape (N, 3)
  - Test device is always CPU
  - Test invalid sequences raise appropriate errors

### 8.2 Integration Tests (New)
- `tests/integration/test_biophysical_hydra.py`
  - Test Hydra config loading for `model=biophysical`
  - Test full training pipeline with mock data
  - Test model checkpoint is saved correctly

### 8.3 E2E Tests (Modify Existing)
- Update e2e tests to also run with `model=biophysical` option

---

## 9. Implementation Order

1. Create `conf/model/biophysical.yaml`
2. Create `core/embeddings_biophysical.py`
3. Update `models/config.py` (add Literal)
4. Update `core/classifier.py` (add factory branch)
5. Add unit tests for new wrapper
6. Run full test suite
7. Validate accuracy matches standalone script
8. Delete obsolete files (reproduce_track_b.py, etc.)

---

## 10. Rollback Plan

If Phase E causes issues:
1. Revert changes to `models/config.py` and `classifier.py`
2. Keep `reproduce_track_b.py` as fallback
3. Document issues in this spec for future reference

---

## 11. References

- **Novo Nordisk Paper**: Sakhnini et al. 2025, Table S1 (descriptors), Table S2 (importance)
- **BiophysicalExtractor**: `src/antibody_training_esm/core/biophysical.py`
- **Existing Model Config Pattern**: `conf/model/esm1v.yaml`, `conf/model/amplify_350m.yaml`
