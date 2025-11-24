"""Configuration management for the testing pipeline."""

from dataclasses import dataclass

import yaml

from antibody_training_esm.core.config import DEFAULT_BATCH_SIZE


@dataclass
class TestConfig:
    """Configuration for testing pipeline"""

    model_paths: list[str]
    data_paths: list[str]
    sequence_column: str = "sequence"  # Column name for sequences in dataset
    label_column: str = "label"  # Column name for labels in dataset
    output_dir: str = "./experiments/benchmarks"
    metrics: list[str] | None = None
    save_predictions: bool = True
    batch_size: int = DEFAULT_BATCH_SIZE  # Batch size for embedding extraction
    device: str = "auto"  # Device: auto (CUDA > MPS > CPU), or explicit [cuda, cpu, mps]
    threshold: float | None = (
        None  # Manual threshold override (None = auto-detect from dataset name)
    )

    def __post_init__(self) -> None:
        if self.metrics is None:
            self.metrics = [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
            ]


def load_config_file(config_path: str) -> TestConfig:
    """Load test configuration from YAML file"""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    return TestConfig(**config_dict)


def create_sample_test_config() -> None:
    """Create a sample test configuration file"""
    sample_config = {
        "model_paths": ["./experiments/checkpoints/antibody_classifier.pkl"],
        "data_paths": ["./sample_data.csv"],
        "sequence_column": "sequence",
        "label_column": "label",
        "output_dir": "./experiments/benchmarks",
        "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"],
        "save_predictions": True,
    }

    with open("test_config.yaml", "w") as f:
        yaml.dump(sample_config, f, default_flow_style=False)

    print("Sample test configuration created: test_config.yaml")
