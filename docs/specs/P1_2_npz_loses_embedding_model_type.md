# P1.2 Spec: NPZ+JSON Loses Embedding Model Type

**Status:** CONFIRMED
**Priority:** P1 (High) - Model reconstruction fails for AMPLIFY/biophysical
**Audit Date:** 2025-12-14

---

## Root Cause Analysis

### The Problem

When saving a model trained with `model=biophysical` or `model=amplify_350m` to NPZ+JSON format, the **embedding model type is lost**. On reload, the model incorrectly attempts to use ESM embeddings.

### Data Flow Breakdown

```text
SAVE PATH:
1. BinaryClassifier has _model_type = "biophysical"
2. ModelArtifactMetadata.from_classifier() extracts params
   └─> Gets: model_name, batch_size, device, revision
   └─> MISSING: Does NOT extract _model_type (esm/amplify/biophysical)
3. JSON saved without embedding model type

LOAD PATH:
1. load_model_from_npz() reads JSON
2. ModelArtifactMetadata.to_classifier_params() returns params
   └─> Returns: model_name, device, batch_size, revision
   └─> MISSING: No model_type field
3. BinaryClassifier(params) is created
   └─> Line 95: model_type = params.get("model_type", "esm")  # Defaults to ESM!
4. RESULT: Biophysical model reloaded as ESM → crashes or wrong predictions
```

### Impact by Model Type

| Original Model | After NPZ Reload | Result |
|----------------|------------------|--------|
| ESM-1v | ESM-1v | ✅ Works (default matches) |
| AMPLIFY | ESM | ❌ Feature dimension mismatch |
| Biophysical | ESM | ❌ Tries to load HF model "biophysical" → crash |

---

## Affected Files

| File | Role |
|------|------|
| `src/antibody_training_esm/models/artifact.py` | `ModelArtifactMetadata` - missing field |
| `src/antibody_training_esm/core/training/serialization.py` | `load_model_from_npz()` - reconstruction |
| `src/antibody_training_esm/core/prediction.py` | `Predictor` - also affected |

---

## TDD Test Specifications

### Test 1: Metadata should capture embedding_model_type

```python
# tests/unit/models/test_artifact_model_type.py

import pytest
from unittest.mock import MagicMock

from antibody_training_esm.models.artifact import ModelArtifactMetadata


class TestModelArtifactMetadataEmbeddingType:
    """Test that ModelArtifactMetadata captures embedding model type."""

    @pytest.fixture
    def mock_biophysical_classifier(self):
        """Create mock classifier with biophysical model type."""
        clf = MagicMock()
        clf.model_name = "biophysical"
        clf.device = "cpu"
        clf.batch_size = 8
        clf.revision = "main"
        clf._model_type = "biophysical"  # <-- This must be captured
        clf.random_state = 42
        clf.C = 1.0
        clf.penalty = "l2"
        clf.solver = "lbfgs"
        clf.class_weight = None
        clf.max_iter = 1000
        clf.classifier.to_dict.return_value = {"type": "logistic_regression"}
        return clf

    @pytest.fixture
    def mock_amplify_classifier(self):
        """Create mock classifier with AMPLIFY model type."""
        clf = MagicMock()
        clf.model_name = "chandar-lab/AMPLIFY_350M"
        clf.device = "cpu"
        clf.batch_size = 1
        clf.revision = "main"
        clf._model_type = "amplify"  # <-- This must be captured
        clf.random_state = 42
        clf.classifier.to_dict.return_value = {"type": "logistic_regression"}
        return clf

    def test_from_classifier_captures_biophysical_type(self, mock_biophysical_classifier):
        """from_classifier should capture model_type='biophysical'."""
        metadata = ModelArtifactMetadata.from_classifier(mock_biophysical_classifier)

        # CRITICAL ASSERTION: embedding_model_type should be captured
        assert hasattr(metadata, "embedding_model_type"), (
            "ModelArtifactMetadata should have 'embedding_model_type' field"
        )
        assert metadata.embedding_model_type == "biophysical", (
            f"Expected embedding_model_type='biophysical', "
            f"got '{metadata.embedding_model_type}'"
        )

    def test_from_classifier_captures_amplify_type(self, mock_amplify_classifier):
        """from_classifier should capture model_type='amplify'."""
        metadata = ModelArtifactMetadata.from_classifier(mock_amplify_classifier)

        assert metadata.embedding_model_type == "amplify"

    def test_to_classifier_params_includes_model_type(self, mock_biophysical_classifier):
        """to_classifier_params should include model_type for reconstruction."""
        metadata = ModelArtifactMetadata.from_classifier(mock_biophysical_classifier)
        params = metadata.to_classifier_params()

        # CRITICAL ASSERTION: params must include model_type
        assert "model_type" in params, (
            "to_classifier_params() must return model_type for BinaryClassifier reconstruction"
        )
        assert params["model_type"] == "biophysical"


class TestModelArtifactMetadataRoundTrip:
    """Test JSON serialization/deserialization preserves embedding_model_type."""

    def test_json_roundtrip_preserves_embedding_model_type(self):
        """embedding_model_type should survive JSON serialization."""
        original = ModelArtifactMetadata(
            model_name="biophysical",
            model_type="logistic_regression",
            sklearn_version="1.3.0",
            classifier={"type": "logistic_regression"},
            esm_model="biophysical",
            esm_revision="main",
            batch_size=8,
            device="cpu",
            embedding_model_type="biophysical",  # <-- New field
        )

        # Serialize to JSON and back
        json_str = original.model_dump_json()
        restored = ModelArtifactMetadata.model_validate_json(json_str)

        # CRITICAL: embedding_model_type must survive roundtrip
        assert restored.embedding_model_type == "biophysical"
```

### Test 2: Integration - NPZ save/load preserves model type

```python
# tests/integration/test_npz_model_type_roundtrip.py

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from antibody_training_esm.core.training.serialization import (
    save_model,
    load_model_from_npz,
)


class TestNPZModelTypeRoundtrip:
    """Integration tests for NPZ save/load with different model types."""

    @pytest.fixture
    def mock_biophysical_classifier(self):
        """Mock biophysical classifier with fitted state."""
        clf = MagicMock()
        clf.model_name = "biophysical"
        clf.device = "cpu"
        clf.batch_size = 8
        clf.revision = "main"
        clf._model_type = "biophysical"
        clf.random_state = 42
        clf.is_fitted = True

        # Mock classifier strategy
        clf.classifier.to_dict.return_value = {"type": "logistic_regression"}
        clf.classifier.to_arrays.return_value = {
            "coef": np.array([[0.1, 0.2, 0.3]]),
            "intercept": np.array([0.0]),
            "classes": np.array([0, 1]),
            "n_features_in": np.array([3]),
            "n_iter": np.array([10]),
        }

        # Mock LogReg attributes for backward compat
        clf.C = 1.0
        clf.penalty = "l2"
        clf.solver = "lbfgs"
        clf.class_weight = None
        clf.max_iter = 1000

        return clf

    def test_biophysical_model_roundtrip(self, mock_biophysical_classifier, tmp_path):
        """Save and load biophysical model should preserve model_type."""
        # Create mock config
        config = MagicMock()
        config.training.save_model = True
        config.training.model_name = "test_biophysical"
        config.training.model_save_dir = tmp_path
        config.model.name = "biophysical"
        config.classifier.model_dump.return_value = {"strategy": "logistic_regression"}
        config.classifier.strategy = "logistic_regression"
        config.train_metrics = None

        logger = MagicMock()

        # Save model
        with patch("antibody_training_esm.core.training.serialization.get_hierarchical_model_dir") as mock_dir:
            mock_dir.return_value = tmp_path
            paths = save_model(mock_biophysical_classifier, config, logger)

        # Load the JSON config
        json_path = Path(paths["config"])
        with open(json_path) as f:
            saved_config = json.load(f)

        # CRITICAL ASSERTION: JSON should contain embedding_model_type
        assert "embedding_model_type" in saved_config, (
            "Saved JSON config should contain 'embedding_model_type' field"
        )
        assert saved_config["embedding_model_type"] == "biophysical"

    @pytest.mark.skip(reason="Requires fix to run - validates the fix works")
    def test_load_biophysical_creates_biophysical_extractor(self, tmp_path):
        """Loading biophysical NPZ should create BiophysicalEmbeddingExtractor."""
        # Create NPZ and JSON files
        npz_path = tmp_path / "model.npz"
        json_path = tmp_path / "model_config.json"

        np.savez(
            npz_path,
            coef=np.array([[0.1, 0.2, 0.3]]),
            intercept=np.array([0.0]),
            classes=np.array([0, 1]),
            n_features_in=np.array([3]),
            n_iter=np.array([10]),
        )

        config = {
            "model_name": "biophysical",
            "model_type": "logistic_regression",
            "sklearn_version": "1.3.0",
            "classifier": {"type": "logistic_regression"},
            "esm_model": "biophysical",
            "esm_revision": "main",
            "batch_size": 8,
            "device": "cpu",
            "embedding_model_type": "biophysical",  # <-- Must be present
        }
        with open(json_path, "w") as f:
            json.dump(config, f)

        # Load and check
        with patch("antibody_training_esm.core.embeddings_biophysical.BiophysicalEmbeddingExtractor"):
            clf = load_model_from_npz(str(npz_path), str(json_path))

        # CRITICAL: Should have biophysical extractor, not ESM
        assert clf._model_type == "biophysical", (
            f"Loaded classifier should have _model_type='biophysical', "
            f"got '{clf._model_type}'"
        )
```

### Test 3: E2E test - Train biophysical, save NPZ, load and predict

```python
# tests/e2e/test_biophysical_npz_e2e.py

import pytest


@pytest.mark.e2e
class TestBiophysicalNPZE2E:
    """E2E test for biophysical model NPZ roundtrip."""

    def test_biophysical_npz_roundtrip_predicts_correctly(self, tmp_path):
        """
        Full roundtrip test:
        1. Train biophysical model
        2. Save to NPZ+JSON
        3. Load from NPZ+JSON
        4. Predict - should use BiophysicalEmbeddingExtractor, not ESM
        """
        # This test validates the entire flow works end-to-end
        # Key assertion: predictions after load should match predictions before save
        pass  # Implementation depends on fix
```

---

## Implementation Fix

### Step 1: Add `embedding_model_type` field to metadata

```python
# src/antibody_training_esm/models/artifact.py

class ModelArtifactMetadata(BaseModel):
    # ... existing fields ...

    # ADD THIS FIELD
    embedding_model_type: Literal["esm", "amplify", "biophysical"] = Field(
        default="esm",
        description="Type of embedding extractor (esm, amplify, or biophysical)",
    )

    @classmethod
    def from_classifier(cls, classifier: Any) -> "ModelArtifactMetadata":
        """Construct metadata from BinaryClassifier instance."""
        import sklearn

        strategy_config = classifier.classifier.to_dict()
        classifier_type = strategy_config.get("type", "logistic_regression")

        # EXTRACT embedding_model_type from classifier
        embedding_model_type = getattr(classifier, "_model_type", "esm")

        metadata_dict = {
            "model_name": classifier.model_name,
            "model_type": classifier_type,
            "sklearn_version": sklearn.__version__,
            "classifier": strategy_config,
            "esm_model": classifier.model_name,
            "esm_revision": classifier.revision,
            "batch_size": classifier.batch_size,
            "device": classifier.device,
            "embedding_model_type": embedding_model_type,  # <-- ADD THIS
        }
        # ... rest unchanged
```

### Step 2: Update `to_classifier_params()` to return model_type

```python
# src/antibody_training_esm/models/artifact.py

def to_classifier_params(self) -> dict[str, Any]:
    """Extract parameters for BinaryClassifier reconstruction."""
    params = {
        "model_name": self.esm_model,
        "device": self.device,
        "batch_size": self.batch_size,
        "revision": self.esm_revision,
        "model_type": self.embedding_model_type,  # <-- ADD THIS
        **self.classifier,
    }
    # ... rest unchanged
```

### Step 3: Backward compatibility for old JSON files

```python
# src/antibody_training_esm/models/artifact.py

from pydantic import field_validator

class ModelArtifactMetadata(BaseModel):
    # ...

    @field_validator("embedding_model_type", mode="before")
    @classmethod
    def infer_embedding_model_type(cls, v, info):
        """Infer embedding_model_type from model_name for old files."""
        if v is not None:
            return v

        # Try to infer from model_name
        model_name = info.data.get("model_name", "") or info.data.get("esm_model", "")
        if "biophysical" in model_name.lower():
            return "biophysical"
        elif "amplify" in model_name.lower():
            return "amplify"
        return "esm"  # Default for old files
```

---

## Acceptance Criteria

- [ ] `embedding_model_type` field exists in `ModelArtifactMetadata`
- [ ] `from_classifier()` extracts `_model_type` from classifier
- [ ] `to_classifier_params()` includes `model_type`
- [ ] JSON files saved contain `embedding_model_type`
- [ ] Loading old JSON files (without field) defaults to "esm"
- [ ] Biophysical model roundtrip works correctly
- [ ] AMPLIFY model roundtrip works correctly
- [ ] Tests cover all three model types

---

**Spec Version:** 1.0
**Author:** Bug Audit 2025-12-14
