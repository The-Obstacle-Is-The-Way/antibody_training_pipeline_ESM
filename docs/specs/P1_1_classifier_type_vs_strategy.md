# P1.1 Spec: Hydra Classifier Selection Mismatch

**Status:** CONFIRMED
**Priority:** P1 (High) - Silent wrong model training
**Audit Date:** 2025-12-14

---

## Root Cause Analysis

### The Problem

When a user runs `antibody-train classifier=xgboost`, the training pipeline **silently uses LogisticRegression instead of XGBoost**.

### Data Flow Breakdown

```
1. Hydra loads classifier/xgboost.yaml
   └─> Has field: `type: xgboost`

2. Pydantic ClassifierConfig validates
   └─> Expects field: `strategy` (not `type`)
   └─> Unknown field `type` is IGNORED (Pydantic v2 default)
   └─> Uses default: `strategy: "logistic_regression"`

3. Trainer passes params to BinaryClassifier
   └─> Line 163: "strategy": config.classifier.strategy
   └─> Passes: strategy="logistic_regression" (the default!)

4. BinaryClassifier calls create_classifier(params)
   └─> Line 63: classifier_type = config.get("type", "logistic_regression")
   └─> No "type" key in params → defaults to logistic_regression

5. RESULT: User thinks they're training XGBoost, but gets LogReg
```

### Why This Is Dangerous

- No error, warning, or indication that the wrong classifier is used
- Model artifacts look "successful"
- User publishes results believing they used XGBoost
- Only discovered if someone manually inspects the `.pkl` file

---

## Affected Files

| File | Role |
|------|------|
| `src/antibody_training_esm/conf/classifier/xgboost.yaml` | Uses `type:` field |
| `src/antibody_training_esm/conf/classifier/logreg.yaml` | Uses `type:` field |
| `src/antibody_training_esm/models/config.py` | Expects `strategy:` field |
| `src/antibody_training_esm/core/trainer.py` | Reads `config.classifier.strategy` |
| `src/antibody_training_esm/core/classifier_factory.py` | Reads `config.get("type")` |

---

## TDD Test Specifications

### Test 1: Pydantic should reject unknown fields OR alias `type` to `strategy`

```python
# tests/unit/models/test_config_classifier_type.py

import pytest
from pydantic import ValidationError
from antibody_training_esm.models.config import ClassifierConfig


class TestClassifierConfigTypeField:
    """Test that classifier type field is properly handled."""

    def test_type_field_maps_to_strategy(self):
        """type: xgboost in YAML should result in strategy='xgboost'."""
        # Simulate what Hydra passes after reading xgboost.yaml
        config_dict = {
            "type": "xgboost",
            "n_estimators": 100,
            "max_depth": 6,
        }
        config = ClassifierConfig.model_validate(config_dict)

        # EXPECTED: strategy should be "xgboost" (mapped from type)
        assert config.strategy == "xgboost", (
            f"Expected strategy='xgboost' when type='xgboost', "
            f"got strategy='{config.strategy}'"
        )

    def test_strategy_field_still_works(self):
        """Direct strategy field should still work (backward compat)."""
        config_dict = {"strategy": "xgboost"}
        config = ClassifierConfig.model_validate(config_dict)
        assert config.strategy == "xgboost"

    def test_type_and_strategy_conflict_raises(self):
        """If both type and strategy are provided and differ, raise error."""
        config_dict = {
            "type": "xgboost",
            "strategy": "logistic_regression",
        }
        with pytest.raises(ValidationError) as exc_info:
            ClassifierConfig.model_validate(config_dict)

        assert "conflict" in str(exc_info.value).lower() or "mismatch" in str(exc_info.value).lower()

    def test_unknown_classifier_type_raises(self):
        """Unknown classifier type should raise ValidationError."""
        config_dict = {"type": "random_forest_not_supported"}
        with pytest.raises(ValidationError):
            ClassifierConfig.model_validate(config_dict)
```

### Test 2: Integration test - XGBoost config produces XGBoost classifier

```python
# tests/integration/test_classifier_selection.py

import pytest
from omegaconf import OmegaConf

from antibody_training_esm.core.trainer import validate_config
from antibody_training_esm.core.classifier import BinaryClassifier


class TestClassifierSelectionIntegration:
    """Integration tests for classifier selection via Hydra config."""

    @pytest.fixture
    def xgboost_hydra_config(self, tmp_path):
        """Create a minimal Hydra config with classifier=xgboost."""
        # Create minimal data files
        train_file = tmp_path / "train.csv"
        test_file = tmp_path / "test.csv"
        train_file.write_text("sequence,label\nACDE,0\nFGHI,1\n")
        test_file.write_text("sequence,label\nJKLM,0\n")

        config = OmegaConf.create({
            "model": {
                "name": "facebook/esm1v_t33_650M_UR90S_1",
                "device": "cpu",
                "revision": "main",
                "batch_size": 1,
                "model_type": "esm",
            },
            "data": {
                "train_file": str(train_file),
                "test_file": str(test_file),
                "embeddings_cache_dir": str(tmp_path / "cache"),
            },
            "classifier": {
                "type": "xgboost",  # <-- This is what Hydra would load
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.3,
                "random_state": 42,
            },
            "training": {
                "n_splits": 2,
                "random_state": 42,
                "stratify": True,
                "save_model": False,
                "model_save_dir": str(tmp_path / "models"),
                "model_name": "test_model",
                "log_level": "INFO",
                "log_file": "test.log",
                "batch_size": 1,
                "num_workers": 0,
                "metrics": ["accuracy"],
            },
            "experiment": {
                "name": "test_xgboost_selection",
            },
        })
        return config

    def test_xgboost_config_produces_xgboost_classifier(self, xgboost_hydra_config):
        """Hydra config with type=xgboost should produce XGBoost classifier."""
        # Validate config through Pydantic
        pydantic_config = validate_config(xgboost_hydra_config)

        # CRITICAL ASSERTION: strategy should be xgboost
        assert pydantic_config.classifier.strategy == "xgboost", (
            f"Expected classifier.strategy='xgboost', "
            f"got '{pydantic_config.classifier.strategy}'"
        )

    def test_xgboost_classifier_is_actually_xgboost(self, xgboost_hydra_config, mocker):
        """BinaryClassifier created from xgboost config should use XGBoostStrategy."""
        # Mock ESM to avoid loading 650MB model
        mocker.patch(
            "antibody_training_esm.core.embeddings.ESMEmbeddingExtractor",
            autospec=True,
        )

        pydantic_config = validate_config(xgboost_hydra_config)

        # Build classifier params (same as trainer.py does)
        classifier_params = {
            "model_name": pydantic_config.model.name,
            "device": "cpu",
            "batch_size": 1,
            "revision": "main",
            "model_type": "esm",
            "strategy": pydantic_config.classifier.strategy,
            "random_state": 42,
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.3,
        }

        # This should create an XGBoost classifier
        from antibody_training_esm.core.classifier_factory import create_classifier

        strategy = create_classifier(classifier_params)

        # CRITICAL ASSERTION: The strategy should be XGBoostStrategy
        assert "XGBoost" in type(strategy).__name__, (
            f"Expected XGBoostStrategy, got {type(strategy).__name__}"
        )
```

### Test 3: End-to-end training with XGBoost produces XGBoost artifacts

```python
# tests/e2e/test_xgboost_training.py

import pytest
import pickle
from pathlib import Path


@pytest.mark.e2e
class TestXGBoostTrainingE2E:
    """E2E test that XGBoost training produces XGBoost model."""

    def test_xgboost_training_produces_xgb_file(self, tmp_path):
        """Training with classifier=xgboost should produce .xgb artifact."""
        # This test would run actual training
        # If .xgb file exists, XGBoost was used
        # If only .npz exists, LogReg was (incorrectly) used

        # Check for XGBoost native file
        model_dir = tmp_path / "checkpoints"
        xgb_files = list(model_dir.glob("**/*.xgb"))
        npz_files = list(model_dir.glob("**/*.npz"))

        # XGBoost saves to .xgb, LogReg saves to .npz
        if npz_files and not xgb_files:
            pytest.fail(
                "Found .npz but no .xgb files - classifier selection bug! "
                "LogReg was used instead of XGBoost."
            )

        assert len(xgb_files) > 0, "XGBoost training should produce .xgb file"
```

---

## Implementation Fix

### Option A: Add Pydantic alias (Recommended - Minimal Change)

```python
# src/antibody_training_esm/models/config.py

from pydantic import Field, model_validator

class ClassifierConfig(BaseModel):
    """Classifier configuration (strategy-agnostic)."""

    strategy: Literal["logistic_regression", "xgboost"] = Field(
        default="logistic_regression",
        description="Classification strategy",
        # ADD ALIAS: Accept both 'type' and 'strategy' from config
        validation_alias="type",  # <-- This maps YAML 'type:' to 'strategy'
    )

    # ... rest of fields ...

    @model_validator(mode="before")
    @classmethod
    def unify_type_and_strategy(cls, data: dict) -> dict:
        """Handle both 'type' and 'strategy' fields consistently."""
        if isinstance(data, dict):
            # If both are present, they must match
            if "type" in data and "strategy" in data:
                if data["type"] != data["strategy"]:
                    raise ValueError(
                        f"Conflicting classifier config: type='{data['type']}' "
                        f"but strategy='{data['strategy']}'. Use only one."
                    )
            # Map 'type' to 'strategy' if only 'type' is present
            elif "type" in data and "strategy" not in data:
                data["strategy"] = data["type"]
        return data
```

### Option B: Update YAML configs to use `strategy:`

```yaml
# src/antibody_training_esm/conf/classifier/xgboost.yaml
strategy: xgboost  # <-- Change from 'type: xgboost'
n_estimators: 100
# ... rest unchanged
```

### Option C: Update factory to check both keys

```python
# src/antibody_training_esm/core/classifier_factory.py

def create_classifier(config: dict[str, Any]) -> ClassifierStrategy:
    # Check both 'type' and 'strategy' keys
    classifier_type = config.get("type") or config.get("strategy", "logistic_regression")
    # ... rest unchanged
```

---

## Recommended Fix Order

1. **Option A** - Add Pydantic validator to map `type` → `strategy`
2. Write tests FIRST (TDD)
3. Run tests (should fail)
4. Implement fix
5. Run tests (should pass)
6. Update YAML configs to use `strategy:` (Option B) for consistency
7. Add deprecation warning for `type:` field

---

## Acceptance Criteria

- [ ] `classifier=xgboost` in CLI produces XGBoost model
- [ ] `classifier=logreg` in CLI produces LogReg model
- [ ] Tests verify classifier type matches config
- [ ] No silent fallback to default classifier
- [ ] Warning if deprecated `type:` field is used

---

**Spec Version:** 1.0
**Author:** Bug Audit 2025-12-14
