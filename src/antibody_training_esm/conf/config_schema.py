"""
Structured configuration schemas for Hydra

Type-safe configuration using dataclasses with full field coverage
validated against current trainer.py requirements.
"""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class ModelConfig:
    """ESM model configuration (matches current model config structure)"""

    name: str = "facebook/esm1v_t33_650M_UR90S_1"
    revision: str = "main"
    device: str = MISSING  # Provided by YAML interpolation ${hardware.device}


@dataclass
class ClassifierConfig:
    """Classifier head configuration (matches current classifier config)"""

    type: str = "logistic_regression"
    C: float = 1.0
    penalty: str = "l2"
    solver: str = "lbfgs"
    max_iter: int = 1000
    random_state: int = (
        MISSING  # Provided by YAML interpolation ${training.random_state}
    )
    class_weight: str | None = None
    cv_folds: int = 10
    stratify: bool = True


@dataclass
class DataConfig:
    """Dataset configuration (ALL fields used by loaders.py + trainer.py)"""

    # REQUIRED by loaders.py
    source: str = "local"
    train_file: str = MISSING  # Required
    test_file: str = MISSING  # Required
    sequence_column: str = "sequence"
    label_column: str = "label"

    # REQUIRED by trainer.py
    embeddings_cache_dir: str = "./embeddings_cache"

    # Optional fields
    dataset_name: str = "boughter_vh"
    max_sequence_length: int = 1024
    save_embeddings: bool = True

    # Fragment metadata (testing only)
    train_fragment: str = "VH"
    test_fragment: str = "VH"
    test_assay: str = "ELISA"
    test_threshold: float = 0.5


@dataclass
class TrainingConfig:
    """Training hyperparameters (ALL fields used by trainer.py)"""

    # Cross-validation
    n_splits: int = 10
    random_state: int = 42
    stratify: bool = True

    # Evaluation metrics
    metrics: list[str] = field(
        default_factory=lambda: ["accuracy", "precision", "recall", "f1", "roc_auc"]
    )

    # Model saving
    save_model: bool = True
    model_name: str = "boughter_vh_esm1v_logreg"
    model_save_dir: str = "./models"

    # Logging (Hydra-aware: relative to Hydra output dir, or logs/ in legacy mode)
    log_level: str = "INFO"
    log_file: str = "logs/training.log"  # Routes to logs/ dir in legacy mode, Hydra output dir in Hydra mode

    # Performance optimization
    batch_size: int = 8
    num_workers: int = 4


@dataclass
class HardwareConfig:
    """Hardware settings"""

    device: str = "mps"
    gpu_memory_fraction: float = 0.8
    clear_cache_frequency: int = 100


@dataclass
class ExperimentConfig:
    """Experiment metadata"""

    name: str = "novo_replication"
    description: str = "Train ESM-1v VH-based LogisticReg on Boughter, test on Jain"
    tags: list[str] = field(default_factory=lambda: ["baseline", "esm1v", "logreg"])


@dataclass
class Config:
    """Root configuration (complete schema matching current trainer.py)"""

    model: ModelConfig = field(default_factory=ModelConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


# Register schemas with ConfigStore
# Schema names MUST match YAML filenames for automatic validation
# Hydra validates group YAML against same-named schemas
cs = ConfigStore.instance()
cs.store(name="config", node=Config)  # Matches conf/config.yaml
cs.store(group="model", name="esm1v", node=ModelConfig)  # Matches conf/model/esm1v.yaml
cs.store(
    group="classifier", name="logreg", node=ClassifierConfig
)  # Matches conf/classifier/logreg.yaml
cs.store(
    group="data", name="boughter_jain", node=DataConfig
)  # Matches conf/data/boughter_jain.yaml
