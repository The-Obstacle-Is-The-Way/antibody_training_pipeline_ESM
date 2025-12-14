# P2 Spec: Prediction and Cache Bugs

**Status:** ALL CONFIRMED
**Priority:** P2 (Medium)
**Audit Date:** 2025-12-14

This spec covers four related P2 bugs affecting prediction and caching:

1. **P2.1** - Predictor.embedder recreation ignores biophysical
2. **P2.2** - Predictor doesn't support .xgb artifacts
3. **P2.3** - Cache key builds giant string (memory risk)
4. **P2.4** - Training doesn't use test_file (docs mismatch)

---

## P2.1: Predictor Embedder Recreation Ignores Biophysical

### Root Cause

When device mismatch triggers embedder recreation in `Predictor`, only AMPLIFY is handled explicitly:

```python
# src/antibody_training_esm/core/prediction.py lines 131-148
if model_type == "amplify":
    embedder = AMPLIFYEmbeddingExtractor(...)
else:
    embedder = ESMEmbeddingExtractor(...)  # <-- biophysical falls here!
```

### TDD Test

```python
# tests/unit/core/test_predictor_embedder_recreation.py

import pytest
from unittest.mock import MagicMock, patch

from antibody_training_esm.core.prediction import Predictor


class TestPredictorEmbedderRecreation:
    """Test embedder recreation handles all model types."""

    def test_biophysical_embedder_recreation(self):
        """Biophysical classifier should recreate BiophysicalEmbeddingExtractor."""
        # Mock classifier with biophysical type
        mock_clf = MagicMock()
        mock_clf._model_type = "biophysical"
        mock_clf.embedding_extractor.device = "cuda"  # Mismatch with requested "cpu"
        mock_clf.embedding_extractor.batch_size = 8
        mock_clf.embedding_extractor.revision = "main"

        with patch.object(Predictor, "classifier", mock_clf):
            with patch(
                "antibody_training_esm.core.prediction.BiophysicalEmbeddingExtractor"
            ) as mock_bio:
                predictor = Predictor(
                    model_name="biophysical",
                    classifier_path="/fake/path.pkl",
                    device="cpu",  # Mismatch triggers recreation
                )
                # Access embedder to trigger lazy load
                _ = predictor.embedder

                # ASSERTION: BiophysicalEmbeddingExtractor should be called
                mock_bio.assert_called_once()

    def test_amplify_embedder_recreation(self):
        """AMPLIFY classifier should recreate AMPLIFYEmbeddingExtractor."""
        # Similar test for AMPLIFY
        pass

    def test_esm_embedder_recreation(self):
        """ESM classifier should recreate ESMEmbeddingExtractor."""
        # Default case
        pass
```

### Fix

```python
# src/antibody_training_esm/core/prediction.py

# Replace lines 131-148 with:
if model_type == "amplify":
    from antibody_training_esm.core.embeddings_amplify import (
        AMPLIFYEmbeddingExtractor,
    )
    embedder = AMPLIFYEmbeddingExtractor(...)
elif model_type == "biophysical":
    from antibody_training_esm.core.embeddings_biophysical import (
        BiophysicalEmbeddingExtractor,
    )
    embedder = BiophysicalEmbeddingExtractor(
        model_name=self.model_name,
        device=self.device,
        batch_size=batch_size,
        revision=revision,
    )
else:
    embedder = ESMEmbeddingExtractor(...)
```

---

## P2.2: Predictor Doesn't Support .xgb Artifacts

### Root Cause

`Predictor.classifier` property only handles `.npz` and pickle:

```python
# src/antibody_training_esm/core/prediction.py lines 71-96
if path_obj.suffix == ".npz":
    self._classifier = load_model_from_npz(...)
else:
    # Assumes pickle
    self._classifier = joblib.load(...)
```

But training saves XGBoost to native `.xgb` format (pickle-free):

```python
# src/antibody_training_esm/core/training/serialization.py line 137
classifier.classifier.save_model(str(xgb_path))  # Saves .xgb
```

### TDD Test

```python
# tests/unit/core/test_predictor_xgb_loading.py

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from antibody_training_esm.core.prediction import Predictor


class TestPredictorXGBLoading:
    """Test Predictor can load XGBoost .xgb artifacts."""

    def test_loads_xgb_file(self, tmp_path):
        """Predictor should recognize and load .xgb files."""
        # Create fake .xgb file
        xgb_path = tmp_path / "model.xgb"
        xgb_path.touch()

        # Create accompanying JSON config
        json_path = tmp_path / "model_config.json"
        json_path.write_text('{"model_name": "test", "model_type": "xgboost"}')

        with patch(
            "antibody_training_esm.core.prediction.load_model_from_xgb"
        ) as mock_load:
            mock_load.return_value = MagicMock()

            predictor = Predictor(
                model_name="facebook/esm1v_t33_650M_UR90S_1",
                classifier_path=str(xgb_path),
                device="cpu",
            )
            _ = predictor.classifier

            # ASSERTION: load_model_from_xgb should be called
            mock_load.assert_called_once()

    def test_xgb_predictions_work(self, tmp_path):
        """Loaded XGBoost model should be able to predict."""
        pass  # Implementation test
```

### Fix

```python
# src/antibody_training_esm/core/prediction.py

# Add new function
def load_model_from_xgb(xgb_path: str, json_path: str) -> BinaryClassifier:
    """Load XGBoost model from native .xgb format."""
    import json
    from xgboost import XGBClassifier

    from antibody_training_esm.models.artifact import ModelArtifactMetadata

    # Load XGBoost model
    xgb_model = XGBClassifier()
    xgb_model.load_model(xgb_path)

    # Load metadata
    with open(json_path) as f:
        metadata_dict = json.load(f)
    metadata = ModelArtifactMetadata.model_validate(metadata_dict)

    # Reconstruct BinaryClassifier
    params = metadata.to_classifier_params()
    classifier = BinaryClassifier(params)

    # Set the loaded XGBoost model
    classifier.classifier.classifier = xgb_model
    classifier.is_fitted = True

    return classifier


# Update classifier property
@property
def classifier(self) -> BinaryClassifier | LogisticRegression:
    if self._classifier is None:
        path_obj = Path(self.classifier_path)

        if path_obj.suffix == ".npz":
            # NPZ loading...
            pass
        elif path_obj.suffix == ".xgb":
            # XGBoost native format
            json_path = self._get_json_path(path_obj)
            logger.info(f"Loading XGBoost model from: {path_obj}")
            self._classifier = load_model_from_xgb(str(path_obj), str(json_path))
        else:
            # Pickle fallback
            pass

    return self._classifier
```

---

## P2.3: Cache Key Builds Giant String (Memory Risk)

### Root Cause

```python
# src/antibody_training_esm/core/training/cache.py line 103
sequences_str = "|".join(sequences)  # <-- Creates massive string!
```

For Harvey dataset (141k sequences), this creates a string of ~15MB+ in memory.

### TDD Test

```python
# tests/unit/core/training/test_cache_memory.py

import pytest
import hashlib
from antibody_training_esm.core.training.cache import compute_cache_key


class TestCacheKeyMemory:
    """Test cache key computation doesn't blow up memory."""

    def test_large_dataset_cache_key_memory_efficient(self):
        """Cache key for 100k sequences should not create giant string."""
        # Generate 100k fake sequences
        sequences = [f"ACDEFGHIKLMNPQRSTVWY" * 10 for _ in range(100000)]

        # This should NOT create a 100MB+ string in memory
        # The function should use streaming hash
        cache_key = compute_cache_key(sequences, "test_model", "main", 1022)

        # Key should be a short hash
        assert len(cache_key) <= 64, "Cache key should be a short hash string"

    def test_cache_key_deterministic(self):
        """Same sequences should produce same cache key."""
        sequences = ["ACDE", "FGHI", "JKLM"]
        key1 = compute_cache_key(sequences, "model", "v1", 1022)
        key2 = compute_cache_key(sequences, "model", "v1", 1022)
        assert key1 == key2

    def test_different_sequences_different_key(self):
        """Different sequences should produce different cache key."""
        key1 = compute_cache_key(["ACDE"], "model", "v1", 1022)
        key2 = compute_cache_key(["FGHI"], "model", "v1", 1022)
        assert key1 != key2
```

### Fix

```python
# src/antibody_training_esm/core/training/cache.py

def compute_cache_key(
    sequences: list[str],
    model_name: str,
    revision: str,
    max_length: int,
) -> str:
    """
    Compute cache key using streaming hash (memory efficient).

    Instead of joining all sequences into one giant string,
    we update the hash incrementally.
    """
    hasher = hashlib.sha256()

    # Hash model metadata first
    hasher.update(f"{model_name}|{revision}|{max_length}|".encode())

    # Stream sequences through hash (no giant string!)
    for seq in sequences:
        hasher.update(seq.encode())
        hasher.update(b"|")  # Separator

    return hasher.hexdigest()[:12]


# Update get_or_create_embeddings to use new function:
def get_or_create_embeddings(...):
    # Replace lines 103-112 with:
    sequences_hash = compute_cache_key(
        sequences,
        embedding_extractor.model_name,
        embedding_extractor.revision,
        embedding_extractor.max_length,
    )
    # ... rest unchanged
```

---

## P2.4: Training Pipeline Doesn't Use test_file

### Root Cause

`DataConfig` requires `test_file`, and docs describe test-set evaluation, but `train_pipeline()` never loads it:

```python
# src/antibody_training_esm/core/trainer.py line 124
X_train, y_train = load_data(config)  # Only loads train_file!
# config.data.test_file is NEVER used
```

### Options

**Option A**: Implement test-set evaluation in training
**Option B**: Remove `test_file` requirement, clarify that testing is separate CLI

### TDD Test (if implementing Option A)

```python
# tests/unit/core/test_trainer_test_eval.py

import pytest
from unittest.mock import MagicMock, patch

from antibody_training_esm.core.trainer import train_pipeline


class TestTrainerTestEvaluation:
    """Test that training evaluates on test set when provided."""

    def test_train_pipeline_evaluates_test_set(self, mock_config):
        """train_pipeline should evaluate on test_file if provided."""
        with patch("antibody_training_esm.core.trainer.load_data") as mock_load:
            # Setup returns for train and test
            mock_load.side_effect = [
                (["ACDE", "FGHI"], [0, 1]),  # train
                (["JKLM"], [0]),  # test
            ]

            results = train_pipeline(mock_config)

            # ASSERTION: Should have test metrics
            assert "test_metrics" in results, (
                "train_pipeline should return test_metrics when test_file is provided"
            )

    def test_train_pipeline_skips_test_if_not_provided(self, mock_config_no_test):
        """train_pipeline should skip test evaluation if test_file not provided."""
        pass
```

### Recommended Fix (Option B - Docs Clarification)

If testing is meant to be a separate `antibody-test` CLI:

1. Make `test_file` optional in `DataConfig`:
```python
test_file: Path | None = Field(
    default=None,
    description="Path to test CSV (optional, use antibody-test for evaluation)",
)
```

2. Update docs to clarify separation of concerns:
```markdown
# Training produces a model, testing evaluates it
antibody-train  # Trains model, outputs .pkl/.npz
antibody-test --model model.pkl --dataset jain  # Evaluates on test set
```

---

## Summary of All P2 Fixes

| Bug | Fix Complexity | Risk |
|-----|---------------|------|
| P2.1 | Simple - add elif branch | Low |
| P2.2 | Medium - add loader function | Medium |
| P2.3 | Simple - streaming hash | Low |
| P2.4 | Design decision needed | Low |

---

## Acceptance Criteria

### P2.1
- [ ] BiophysicalEmbeddingExtractor is created when model_type="biophysical"
- [ ] Test covers all three model types

### P2.2
- [ ] `.xgb` files can be loaded by Predictor
- [ ] XGBoost predictions work after loading

### P2.3
- [ ] Cache key computed without creating giant string
- [ ] 100k+ sequences don't cause memory issues
- [ ] Cache key is deterministic

### P2.4
- [ ] Either test evaluation is implemented OR docs are updated
- [ ] test_file requirement matches actual usage

---

**Spec Version:** 1.0
**Author:** Bug Audit 2025-12-14
