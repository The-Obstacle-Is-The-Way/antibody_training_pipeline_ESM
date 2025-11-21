# Pydantic Phase 2: Configuration Safety

**Status:** Not Started
**Priority:** HIGH (Developer Experience)
**Risk:** MEDIUM (Touches core training loop)
**Dependencies:** Phase 1 (Pydantic models package exists)

---

## Overview

Replace manual dictionary-based config validation in `trainer.py` with Pydantic models that mirror Hydra's config structure. This catches config errors at startup (not deep in training) and provides IDE autocomplete for config access.

**Key Benefits:**
- **Fail Fast:** Config errors caught before ESM model loading (~30s saved)
- **Type Safety:** `config.model.device` instead of `config["model"]["device"]`
- **Clear Errors:** "device must be 'cpu', 'cuda', or 'mps'" vs "KeyError: device"
- **Hydra Compatible:** Works with DictConfig→dict→Pydantic conversion

---

## Dependencies

**Already installed from Phase 1:**
```toml
[project.optional-dependencies]
validation = [
    "pydantic>=2.10.0",
    "pydantic-settings>=2.6.0",  # Used in this phase
]
```

---

## Implementation Scope

### Files to Modify

1. **Create:** `src/antibody_training_esm/models/config.py`
   - `ModelConfig` (ESM model, device, batch size)
   - `DataConfig` (train/test files, cache dirs)
   - `ClassifierConfig` (strategy, hyperparameters)
   - `TrainingConfig` (epochs, logging, metrics)
   - `ExperimentConfig` (name, tags)
   - `TrainingPipelineConfig` (root config)

2. **Modify:** `src/antibody_training_esm/core/trainer.py`
   - Replace `validate_config()` with Pydantic model validation
   - Convert `DictConfig` → `TrainingPipelineConfig` at entry point
   - Update all `config["key"]["nested"]` to `config.key.nested`

3. **Modify:** `src/antibody_training_esm/cli/test.py`
   - Apply same config validation pattern

4. **Update:** Hydra YAML files (add descriptions for self-documentation)

---

## Model Specifications

### 1. `ModelConfig` (ESM Configuration)

**Location:** `src/antibody_training_esm/models/config.py`

```python
from pydantic import BaseModel, Field
from typing import Literal

class ModelConfig(BaseModel):
    """
    ESM protein language model configuration.

    Controls which HuggingFace model to load and execution device.
    """
    name: str = Field(
        ...,
        description="HuggingFace model ID (e.g., facebook/esm1v_t33_650M_UR90S_1)",
        examples=["facebook/esm1v_t33_650M_UR90S_1", "facebook/esm2_t33_650M_UR50D"],
    )

    device: Literal["cpu", "cuda", "mps", "auto"] = Field(
        default="auto",
        description="Execution device (auto = CUDA > MPS > CPU)",
    )

    revision: str = Field(
        default="main",
        description="HuggingFace model revision (commit hash for reproducibility)",
    )

    batch_size: int = Field(
        default=16,
        ge=1,
        le=128,
        description="Batch size for embedding extraction",
    )
```

### 2. `DataConfig` (Data Paths)

```python
from pathlib import Path

class DataConfig(BaseModel):
    """
    Dataset configuration.

    Specifies input files and caching directories.
    """
    train_file: Path = Field(
        ...,
        description="Path to training CSV (must contain 'sequence' and 'label' columns)",
    )

    test_file: Path = Field(
        ...,
        description="Path to test CSV",
    )

    embeddings_cache_dir: Path = Field(
        default=Path("experiments/cache"),
        description="Directory for cached ESM embeddings",
    )

    @field_validator("train_file", "test_file")
    @classmethod
    def validate_file_exists(cls, v: Path) -> Path:
        """Ensure file exists at config load time."""
        if not v.exists():
            raise FileNotFoundError(f"Data file not found: {v}")
        return v

    @field_validator("embeddings_cache_dir")
    @classmethod
    def create_cache_dir(cls, v: Path) -> Path:
        """Create cache directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v
```

### 3. `ClassifierConfig` (Strategy Configuration)

```python
class ClassifierConfig(BaseModel):
    """
    Classifier configuration (strategy-agnostic).

    Supports both LogisticRegression and XGBoost strategies.
    """
    strategy: Literal["logistic_regression", "xgboost"] = Field(
        default="logistic_regression",
        description="Classification strategy",
    )

    # LogisticRegression params (ignored if strategy=xgboost)
    C: float | None = Field(
        default=1.0,
        gt=0.0,
        description="Inverse regularization strength (LogReg only)",
    )

    penalty: Literal["l1", "l2"] | None = Field(
        default="l2",
        description="Regularization type (LogReg only)",
    )

    solver: Literal["lbfgs", "liblinear", "saga"] | None = Field(
        default="lbfgs",
        description="Optimization algorithm (LogReg only)",
    )

    class_weight: Literal["balanced"] | dict[int, float] | None = Field(
        default="balanced",
        description="Class weighting strategy",
    )

    max_iter: int | None = Field(
        default=1000,
        ge=100,
        description="Maximum optimization iterations",
    )

    random_state: int | None = Field(
        default=42,
        description="Random seed for reproducibility",
    )

    # XGBoost params (ignored if strategy=logistic_regression)
    n_estimators: int | None = Field(
        default=100,
        ge=1,
        description="Number of boosting rounds (XGBoost only)",
    )

    max_depth: int | None = Field(
        default=6,
        ge=1,
        le=20,
        description="Maximum tree depth (XGBoost only)",
    )

    learning_rate: float | None = Field(
        default=0.3,
        gt=0.0,
        le=1.0,
        description="Learning rate (XGBoost only)",
    )
```

### 4. `TrainingConfig` (Training Parameters)

```python
from typing import Literal

class TrainingConfig(BaseModel):
    """
    Training orchestration configuration.

    Controls cross-validation, logging, and model persistence.
    """
    n_splits: int = Field(
        default=10,
        ge=2,
        le=20,
        description="Number of cross-validation folds",
    )

    metrics: set[Literal["accuracy", "precision", "recall", "f1", "roc_auc"]] = Field(
        default={"accuracy", "precision", "recall", "f1", "roc_auc"},
        description="Metrics to compute during evaluation",
    )

    save_model: bool = Field(
        default=True,
        description="Whether to save trained model",
    )

    model_save_dir: Path = Field(
        default=Path("experiments/checkpoints"),
        description="Base directory for saved models",
    )

    model_name: str = Field(
        ...,
        min_length=1,
        description="Name for saved model file (e.g., boughter_vh_esm1v_logreg)",
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging verbosity",
    )

    log_file: str = Field(
        default="training.log",
        description="Log file name (relative to Hydra output dir)",
    )

    @field_validator("model_save_dir")
    @classmethod
    def create_model_dir(cls, v: Path) -> Path:
        """Create model save directory if needed."""
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v
```

### 5. `ExperimentConfig` (Metadata)

```python
class ExperimentConfig(BaseModel):
    """
    Experiment tracking metadata.

    Used for organizing Hydra outputs and logging.
    """
    name: str = Field(
        ...,
        min_length=1,
        description="Experiment name (used in Hydra output directory)",
    )

    tags: list[str] = Field(
        default_factory=list,
        description="Experiment tags for filtering/search",
    )

    description: str | None = Field(
        default=None,
        description="Human-readable experiment description",
    )
```

### 6. `TrainingPipelineConfig` (Root Config)

```python
class TrainingPipelineConfig(BaseModel):
    """
    Root configuration for training pipeline.

    Mirrors Hydra's config.yaml structure.
    """
    model: ModelConfig
    data: DataConfig
    classifier: ClassifierConfig
    training: TrainingConfig
    experiment: ExperimentConfig

    # Optional hardware config (added in config.yaml)
    hardware: dict[str, Any] | None = Field(
        default=None,
        description="Hardware-specific overrides (device, num_threads)",
    )

    model_config = {
        "json_schema_extra": {
            "title": "Antibody Training Pipeline Configuration",
            "description": "Complete configuration for ESM-based antibody training",
        }
    }

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "TrainingPipelineConfig":
        """
        Convert Hydra DictConfig to Pydantic model.

        This is the main entry point for validation.
        """
        # Resolve all interpolations first
        OmegaConf.resolve(cfg)

        # Convert to dict (Pydantic doesn't accept DictConfig directly)
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        # Validate with Pydantic
        return cls.model_validate(config_dict)
```

---

## Integration Steps (TDD)

### Step 1: Write Tests FIRST

**Create:** `tests/unit/models/test_config.py`

```python
"""Unit tests for configuration models."""

import pytest
from pathlib import Path
from pydantic import ValidationError

from antibody_training_esm.models.config import (
    ModelConfig,
    DataConfig,
    ClassifierConfig,
    TrainingConfig,
    ExperimentConfig,
    TrainingPipelineConfig,
)


class TestModelConfig:
    """Test ModelConfig validation."""

    def test_valid_config(self):
        """Valid model config passes."""
        cfg = ModelConfig(
            name="facebook/esm1v_t33_650M_UR90S_1",
            device="cuda",
        )
        assert cfg.device == "cuda"
        assert cfg.batch_size == 16  # default

    def test_invalid_device_rejected(self):
        """Non-enum device values are rejected."""
        with pytest.raises(ValidationError):
            ModelConfig(
                name="facebook/esm1v_t33_650M_UR90S_1",
                device="tpu",  # type: ignore
            )

    def test_batch_size_limits(self):
        """Batch size must be 1-128."""
        with pytest.raises(ValidationError):
            ModelConfig(
                name="facebook/esm1v_t33_650M_UR90S_1",
                batch_size=0,  # too low
            )

        with pytest.raises(ValidationError):
            ModelConfig(
                name="facebook/esm1v_t33_650M_UR90S_1",
                batch_size=200,  # too high
            )


class TestDataConfig:
    """Test DataConfig validation."""

    def test_missing_file_rejected(self, tmp_path):
        """Non-existent files raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            DataConfig(
                train_file=tmp_path / "nonexistent.csv",
                test_file=tmp_path / "test.csv",
            )

    def test_cache_dir_created(self, tmp_path):
        """Cache directory is created automatically."""
        train_file = tmp_path / "train.csv"
        test_file = tmp_path / "test.csv"
        train_file.touch()
        test_file.touch()

        cache_dir = tmp_path / "cache"
        cfg = DataConfig(
            train_file=train_file,
            test_file=test_file,
            embeddings_cache_dir=cache_dir,
        )

        assert cache_dir.exists()


class TestClassifierConfig:
    """Test ClassifierConfig validation."""

    def test_valid_logreg_config(self):
        """Valid LogReg config passes."""
        cfg = ClassifierConfig(
            strategy="logistic_regression",
            C=1.0,
            penalty="l2",
        )
        assert cfg.strategy == "logistic_regression"

    def test_invalid_penalty_rejected(self):
        """Only l1/l2 penalties allowed."""
        with pytest.raises(ValidationError):
            ClassifierConfig(
                strategy="logistic_regression",
                penalty="l3",  # type: ignore
            )

    def test_invalid_C_rejected(self):
        """C must be positive."""
        with pytest.raises(ValidationError):
            ClassifierConfig(
                strategy="logistic_regression",
                C=-1.0,
            )


class TestTrainingConfig:
    """Test TrainingConfig validation."""

    def test_valid_config(self):
        """Valid training config passes."""
        cfg = TrainingConfig(
            model_name="test_model",
            n_splits=10,
        )
        assert cfg.n_splits == 10
        assert "accuracy" in cfg.metrics

    def test_invalid_metrics_rejected(self):
        """Only predefined metrics allowed."""
        with pytest.raises(ValidationError):
            TrainingConfig(
                model_name="test",
                metrics={"invalid_metric"},  # type: ignore
            )

    def test_invalid_log_level_rejected(self):
        """Only standard log levels allowed."""
        with pytest.raises(ValidationError):
            TrainingConfig(
                model_name="test",
                log_level="VERBOSE",  # type: ignore
            )


class TestTrainingPipelineConfig:
    """Test full pipeline config validation."""

    def test_from_hydra_dictconfig(self, tmp_path):
        """Can convert Hydra DictConfig to Pydantic model."""
        from omegaconf import DictConfig

        # Create minimal valid files
        train_file = tmp_path / "train.csv"
        test_file = tmp_path / "test.csv"
        train_file.touch()
        test_file.touch()

        # Simulate Hydra DictConfig
        hydra_cfg = DictConfig({
            "model": {
                "name": "facebook/esm1v_t33_650M_UR90S_1",
                "device": "cpu",
                "batch_size": 16,
            },
            "data": {
                "train_file": str(train_file),
                "test_file": str(test_file),
                "embeddings_cache_dir": str(tmp_path / "cache"),
            },
            "classifier": {
                "strategy": "logistic_regression",
                "C": 1.0,
            },
            "training": {
                "model_name": "test_model",
                "n_splits": 10,
            },
            "experiment": {
                "name": "test_experiment",
            },
        })

        # Convert to Pydantic
        cfg = TrainingPipelineConfig.from_hydra(hydra_cfg)

        assert cfg.model.device == "cpu"
        assert cfg.training.n_splits == 10
        assert cfg.experiment.name == "test_experiment"
```

**Run tests (should FAIL initially):**
```bash
uv run pytest tests/unit/models/test_config.py -xvs
```

### Step 2: Implement Models

Create `src/antibody_training_esm/models/config.py` with specifications above.

**Update:** `src/antibody_training_esm/models/__init__.py`

```python
from antibody_training_esm.models.prediction import (
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResult,
)
from antibody_training_esm.models.config import (
    ModelConfig,
    DataConfig,
    ClassifierConfig,
    TrainingConfig,
    ExperimentConfig,
    TrainingPipelineConfig,
)

__all__ = [
    # Prediction models
    "PredictionRequest",
    "BatchPredictionRequest",
    "PredictionResult",
    # Config models
    "ModelConfig",
    "DataConfig",
    "ClassifierConfig",
    "TrainingConfig",
    "ExperimentConfig",
    "TrainingPipelineConfig",
]
```

**Run tests (should PASS):**
```bash
uv run pytest tests/unit/models/test_config.py -v
```

### Step 3: Integrate into Trainer

**Modify:** `src/antibody_training_esm/core/trainer.py`

**Replace `validate_config()` function:**

```python
from antibody_training_esm.models.config import TrainingPipelineConfig

def validate_config(config: dict[str, Any] | DictConfig) -> TrainingPipelineConfig:
    """
    Validate config with Pydantic models.

    Args:
        config: Raw dict or Hydra DictConfig

    Returns:
        Validated TrainingPipelineConfig

    Raises:
        ValidationError: If config is invalid
    """
    if isinstance(config, DictConfig):
        return TrainingPipelineConfig.from_hydra(config)
    else:
        return TrainingPipelineConfig.model_validate(config)
```

**Update `train_pipeline()` function:**

```python
def train_pipeline(cfg: DictConfig) -> dict[str, Any]:
    """Core training pipeline with Pydantic validation."""
    # Validate config (now returns Pydantic model)
    config = validate_config(cfg)

    # Setup logging (accepts Pydantic model now)
    logger = setup_logging(config)

    logger.info("Starting antibody classification training")
    logger.info(f"Experiment: {config.experiment.name}")

    try:
        # Access config via dot notation (type-safe!)
        X_train, y_train = load_data(config.data.train_file)

        logger.info(f"Loaded {len(X_train)} training samples")

        # Initialize classifier
        classifier_params = {
            "model_name": config.model.name,
            "device": config.model.device,
            "batch_size": config.model.batch_size,
            "revision": config.model.revision,
            # Classifier strategy params
            "strategy": config.classifier.strategy,
            "C": config.classifier.C,
            "penalty": config.classifier.penalty,
            # ... etc
        }

        classifier = BinaryClassifier(classifier_params)

        # Get embeddings (cache_dir from config)
        cache_dir = config.data.embeddings_cache_dir
        X_train_embedded = get_or_create_embeddings(
            X_train, classifier.embedding_extractor, cache_dir, "train", logger
        )

        # Cross-validation
        cv_results = perform_cross_validation(
            X_train_embedded,
            y_train,
            config.training.n_splits,
            logger,
        )

        # Train final model
        classifier.fit(X_train_embedded, y_train)

        # Evaluate
        train_results = evaluate_model(
            classifier,
            X_train_embedded,
            y_train,
            "Training",
            config.training.metrics,
            logger,
        )

        # Save model
        if config.training.save_model:
            model_paths = save_model(classifier, config, logger)
        else:
            model_paths = {}

        return {
            "train_metrics": train_results,
            "cv_metrics": cv_results,
            "config": config.model_dump(),  # Convert back to dict for serialization
            "model_paths": model_paths,
        }

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
```

**Update `setup_logging()` to accept Pydantic model:**

```python
def setup_logging(config: TrainingPipelineConfig) -> logging.Logger:
    """
    Setup logging from Pydantic config.

    Args:
        config: Validated TrainingPipelineConfig

    Returns:
        Configured logger
    """
    log_level = getattr(logging, config.training.log_level.upper())
    log_file = config.training.log_file

    # Hydra-aware path resolution (same as before)
    try:
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)
        log_path = output_dir / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except (ValueError, AttributeError):
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )

    return logging.getLogger(__name__)
```

### Step 4: Update Data Loaders

**Modify:** `src/antibody_training_esm/data/loaders.py`

```python
from antibody_training_esm.models.config import TrainingPipelineConfig

def load_data(config: TrainingPipelineConfig) -> tuple[list[str], list[int]]:
    """
    Load training data from Pydantic config.

    Args:
        config: Validated config with data.train_file

    Returns:
        (sequences, labels)
    """
    train_file = config.data.train_file

    df = pd.read_csv(train_file)

    if "sequence" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"{train_file} must contain 'sequence' and 'label' columns. "
            f"Found: {df.columns.tolist()}"
        )

    return df["sequence"].tolist(), df["label"].tolist()
```

### Step 5: Update Hydra YAML Files (Self-Documentation)

**Enhance:** `src/antibody_training_esm/conf/config.yaml`

Add Hydra comments that match Pydantic field descriptions:

```yaml
# Antibody Non-Specificity Training Pipeline Configuration
# Auto-validated by Pydantic models (see src/antibody_training_esm/models/config.py)

model:
  # HuggingFace model ID (e.g., facebook/esm1v_t33_650M_UR90S_1)
  name: facebook/esm1v_t33_650M_UR90S_1

  # Execution device: cpu, cuda, mps, auto (auto = CUDA > MPS > CPU)
  device: auto

  # HuggingFace model revision (commit hash for reproducibility)
  revision: main

  # Batch size for embedding extraction (1-128)
  batch_size: 16

data:
  # Path to training CSV (must contain 'sequence' and 'label' columns)
  train_file: data/train/boughter/canonical/VH_only_boughter.csv

  # Path to test CSV
  test_file: data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv

  # Directory for cached ESM embeddings
  embeddings_cache_dir: experiments/cache

classifier:
  # Classification strategy: logistic_regression, xgboost
  strategy: logistic_regression

  # Inverse regularization strength (LogReg only; must be > 0)
  C: 1.0

  # Regularization type: l1, l2 (LogReg only)
  penalty: l2

  # Optimization algorithm: lbfgs, liblinear, saga (LogReg only)
  solver: lbfgs

  # Class weighting: balanced or {0: w0, 1: w1}
  class_weight: balanced

  # Maximum optimization iterations (≥100)
  max_iter: 1000

  # Random seed for reproducibility
  random_state: 42

training:
  # Number of cross-validation folds (2-20)
  n_splits: 10

  # Metrics to compute: accuracy, precision, recall, f1, roc_auc
  metrics:
    - accuracy
    - precision
    - recall
    - f1
    - roc_auc

  # Whether to save trained model
  save_model: true

  # Base directory for saved models
  model_save_dir: experiments/checkpoints

  # Name for saved model file (e.g., boughter_vh_esm1v_logreg)
  model_name: boughter_vh_esm1v_logreg

  # Logging verbosity: DEBUG, INFO, WARNING, ERROR
  log_level: INFO

  # Log file name (relative to Hydra output dir)
  log_file: training.log

experiment:
  # Experiment name (used in Hydra output directory)
  name: boughter_jain_baseline

  # Experiment tags for filtering/search
  tags:
    - baseline
    - boughter
    - jain

  # Human-readable description
  description: "Baseline training: Boughter→Jain with ESM-1v + LogReg"
```

---

## Testing Strategy

### Unit Tests

**Coverage:**
- ✅ ModelConfig validation (5 tests)
- ✅ DataConfig validation (3 tests)
- ✅ ClassifierConfig validation (4 tests)
- ✅ TrainingConfig validation (4 tests)
- ✅ ExperimentConfig validation (2 tests)
- ✅ TrainingPipelineConfig.from_hydra() (3 tests)

**Run:**
```bash
uv run pytest tests/unit/models/test_config.py -v --cov=src/antibody_training_esm/models
```

### Integration Tests

**Create:** `tests/integration/test_config_integration.py`

```python
"""Integration tests for Pydantic config + trainer."""

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

from antibody_training_esm.core.trainer import validate_config
from antibody_training_esm.models.config import TrainingPipelineConfig


def test_hydra_config_validates_with_pydantic():
    """Actual Hydra config.yaml validates successfully."""
    with initialize(config_path="../../src/antibody_training_esm/conf"):
        cfg = compose(config_name="config")

        # Should not raise ValidationError
        validated_config = validate_config(cfg)

        assert isinstance(validated_config, TrainingPipelineConfig)
        assert validated_config.model.name == "facebook/esm1v_t33_650M_UR90S_1"


def test_invalid_hydra_override_caught():
    """Invalid Hydra override raises ValidationError."""
    with initialize(config_path="../../src/antibody_training_esm/conf"):
        cfg = compose(
            config_name="config",
            overrides=["model.device=tpu"],  # Invalid device
        )

        with pytest.raises(ValidationError):
            validate_config(cfg)
```

**Run:**
```bash
uv run pytest tests/integration/test_config_integration.py -v
```

---

## Success Criteria

### Functional Requirements

- [ ] All Hydra configs validate with Pydantic
- [ ] Invalid device raises ValidationError at startup
- [ ] Missing files raise FileNotFoundError at startup
- [ ] Invalid metrics raise ValidationError
- [ ] Config access uses dot notation (`config.model.device`)
- [ ] Hydra overrides work: `antibody-train model.device=cuda`
- [ ] Type hints work in IDE (autocomplete for config fields)

### Quality Gates

- [ ] All unit tests pass (≥23 tests)
- [ ] Integration tests pass
- [ ] `make test` passes
- [ ] `make lint` passes
- [ ] `make typecheck` passes
- [ ] Code coverage ≥70%
- [ ] Training runs with validated config (smoke test)

---

## Rollout Plan

1. **PR 1: Config Models Only**
   - Add `models/config.py`
   - Add tests
   - No trainer integration

2. **PR 2: Trainer Integration**
   - Replace `validate_config()`
   - Update `train_pipeline()`
   - Update `setup_logging()`
   - Maintain backward compatibility

3. **PR 3: YAML Self-Documentation**
   - Add comments to Hydra configs
   - Match Pydantic field descriptions

---

## Backward Compatibility

**Breaking Change:** `validate_config()` now returns `TrainingPipelineConfig` instead of `dict`.

**Migration Path:**
1. Update all `config["key"]` → `config.key`
2. Convert to dict when needed: `config.model_dump()`
3. Update function signatures: `def func(config: TrainingPipelineConfig)`

**No breaking changes for:**
- Hydra CLI usage
- Existing YAML files
- Tests (they already use Hydra)

---

## Non-Goals (Out of Scope)

- ❌ Prediction validation (Phase 1)
- ❌ DataFrame schemas (Phase 3)
- ❌ Model artifact validation (Phase 4)
- ❌ Environment variable overrides (future: use `pydantic-settings`)

---

**Last Updated:** 2025-11-20
**Next Phase:** [Phase 3: Data Integrity](PYDANTIC_PHASE_3_DATA_INTEGRITY.md)
