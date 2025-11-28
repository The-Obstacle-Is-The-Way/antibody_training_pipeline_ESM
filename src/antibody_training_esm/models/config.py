from pathlib import Path
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    """
    Protein language model configuration.

    Controls which HuggingFace model to load and execution device.
    Supports ESM-1v, ESM-2, and AMPLIFY models.
    """

    name: str = Field(
        ...,
        description="HuggingFace model ID (e.g., facebook/esm1v_t33_650M_UR90S_1)",
        examples=[
            "facebook/esm1v_t33_650M_UR90S_1",
            "facebook/esm2_t33_650M_UR50D",
            "chandar-lab/AMPLIFY_350M",
        ],
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
        default=8,
        ge=1,
        le=128,
        description="Batch size for embedding extraction (AMPLIFY requires 1)",
    )

    model_type: Literal["esm", "amplify"] = Field(
        default="esm",
        description="Model type: 'esm' for ESM-1v/ESM-2, 'amplify' for AMPLIFY 350M",
    )

    trust_remote_code: bool = Field(
        default=False,
        description="Allow executing remote code from HuggingFace (required for AMPLIFY)",
    )

    @model_validator(mode="after")
    def validate_amplify_constraints(self) -> "ModelConfig":
        """
        Enforce AMPLIFY-specific requirements at config validation time.

        AMPLIFY has strict requirements due to a known padding/batching bug
        (see https://www.nature.com/articles/s41598-025-05674-x):
        - batch_size must be 1 (padding bug causes non-reproducible embeddings)
        - trust_remote_code must be True (AMPLIFY uses custom HuggingFace code)
        """
        if self.model_type == "amplify":
            if self.batch_size != 1:
                raise ValueError(
                    f"AMPLIFY models require batch_size=1 due to padding bug, got {self.batch_size}. "
                    "See: https://www.nature.com/articles/s41598-025-05674-x"
                )
            if not self.trust_remote_code:
                raise ValueError(
                    "AMPLIFY models require trust_remote_code=True "
                    "(AMPLIFY uses custom HuggingFace modeling code)"
                )
        return self


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

    random_state: int = Field(
        default=42,
        description="Random seed used for cross-validation splits",
    )

    stratify: bool = Field(
        default=True,
        description="Whether to use stratified folds during cross-validation",
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

    batch_size: int = Field(
        default=8,
        ge=1,
        le=128,
        description="Batch size for embedding extraction",
    )

    num_workers: int = Field(
        default=4,
        ge=0,
        description="Number of workers for data loading or preprocessing",
    )

    @field_validator("model_save_dir")
    @classmethod
    def create_model_dir(cls, v: Path) -> Path:
        """Create model save directory if needed."""
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v


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


class FeaturesConfig(BaseModel):
    """
    Feature extraction configuration.

    Controls whether to use biophysical descriptors alongside ESM embeddings.
    Phase C of Track B implementation (Sakhnini et al. 2025).

    NOTE: Novo Nordisk paper does NOT use StandardScaler for either ESM or
    biophysical features. We match their methodology exactly - no scaling applied.
    """

    use_biophysical: bool = Field(
        default=False,
        description="Enable biophysical descriptor extraction (Charge@pH6, Charge@pH7.4, pI)",
    )


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
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)

    # Optional hardware config (added in config.yaml)
    hardware: dict[str, Any] | None = Field(
        default=None,
        description="Hardware-specific overrides (device, num_threads)",
    )

    # Runtime metrics (attached after training)
    train_metrics: dict[str, Any] | None = Field(
        default=None,
        description="Metrics from final training run (attached at runtime)",
        exclude=True,  # Do not expect this in input config
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

        # Backwards compatibility: allow training.batch_size overrides to populate model.batch_size
        if isinstance(config_dict, dict):
            model_cfg = config_dict.get("model", {}) or {}
            training_cfg = config_dict.get("training", {}) or {}
            if (
                "batch_size" not in model_cfg
                and isinstance(training_cfg, dict)
                and "batch_size" in training_cfg
            ):
                model_cfg["batch_size"] = training_cfg["batch_size"]
            config_dict["model"] = model_cfg

        # Validate with Pydantic
        result: TrainingPipelineConfig = cls.model_validate(config_dict)
        return result
