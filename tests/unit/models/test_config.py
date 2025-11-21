"""Unit tests for configuration models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from antibody_training_esm.models.config import (
    ClassifierConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    TrainingPipelineConfig,
)


class TestModelConfig:
    """Test ModelConfig validation."""

    def test_valid_config(self) -> None:
        """Valid model config passes."""
        cfg = ModelConfig(
            name="facebook/esm1v_t33_650M_UR90S_1",
            device="cuda",
        )
        assert cfg.device == "cuda"
        assert cfg.batch_size == 8  # default

    def test_invalid_device_rejected(self) -> None:
        """Non-enum device values are rejected."""
        with pytest.raises(ValidationError):
            ModelConfig(
                name="facebook/esm1v_t33_650M_UR90S_1",
                device="tpu",  # type: ignore
            )

    def test_batch_size_limits(self) -> None:
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

    def test_missing_file_rejected(self, tmp_path: Path) -> None:
        """Non-existent files raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            DataConfig(
                train_file=tmp_path / "nonexistent.csv",
                test_file=tmp_path / "test.csv",
            )

    def test_cache_dir_created(self, tmp_path: Path) -> None:
        """Cache directory is created automatically."""
        train_file = tmp_path / "train.csv"
        test_file = tmp_path / "test.csv"
        train_file.touch()
        test_file.touch()

        cache_dir = tmp_path / "cache"
        DataConfig(
            train_file=train_file,
            test_file=test_file,
            embeddings_cache_dir=cache_dir,
        )

        assert cache_dir.exists()


class TestClassifierConfig:
    """Test ClassifierConfig validation."""

    def test_valid_logreg_config(self) -> None:
        """Valid LogReg config passes."""

        cfg = ClassifierConfig(
            strategy="logistic_regression",
            C=1.0,
            penalty="l2",
        )

        assert cfg.strategy == "logistic_regression"

    def test_invalid_penalty_rejected(self) -> None:
        """Only l1/l2 penalties allowed."""

        with pytest.raises(ValidationError):
            ClassifierConfig(
                strategy="logistic_regression",
                penalty="l3",  # type: ignore
            )

    def test_invalid_C_rejected(self) -> None:
        """C must be positive."""

        with pytest.raises(ValidationError):
            ClassifierConfig(
                strategy="logistic_regression",
                C=-1.0,
            )


class TestTrainingConfig:
    """Test TrainingConfig validation."""

    def test_valid_config(self) -> None:
        """Valid training config passes."""

        cfg = TrainingConfig(
            model_name="test_model",
            n_splits=10,
        )

        assert cfg.n_splits == 10
        assert "accuracy" in cfg.metrics
        assert cfg.random_state == 42
        assert cfg.stratify is True
        assert cfg.batch_size == 8
        assert cfg.num_workers == 4

    def test_custom_training_hyperparameters(self) -> None:
        """TrainingConfig supports overrides for CV behavior and batching."""

        cfg = TrainingConfig(
            model_name="test_model",
            n_splits=5,
            random_state=7,
            stratify=False,
            batch_size=4,
            num_workers=2,
        )

        assert cfg.n_splits == 5
        assert cfg.random_state == 7
        assert cfg.stratify is False
        assert cfg.batch_size == 4
        assert cfg.num_workers == 2

    def test_invalid_metrics_rejected(self) -> None:
        """Only predefined metrics allowed."""

        with pytest.raises(ValidationError):
            TrainingConfig(
                model_name="test",
                metrics={"invalid_metric"},  # type: ignore
            )

    def test_invalid_log_level_rejected(self) -> None:
        """Only standard log levels allowed."""

        with pytest.raises(ValidationError):
            TrainingConfig(
                model_name="test",
                log_level="VERBOSE",  # type: ignore
            )


class TestTrainingPipelineConfig:
    """Test full pipeline config validation."""

    def test_from_hydra_dictconfig(self, tmp_path: Path) -> None:
        """Can convert Hydra DictConfig to Pydantic model."""

        from omegaconf import DictConfig

        # Create minimal valid files

        train_file = tmp_path / "train.csv"

        test_file = tmp_path / "test.csv"

        train_file.touch()

        test_file.touch()

        # Simulate Hydra DictConfig

        hydra_cfg = DictConfig(
            {
                "model": {
                    "name": "facebook/esm1v_t33_650M_UR90S_1",
                    "device": "cpu",
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
                    "batch_size": 4,
                    "random_state": 99,
                    "stratify": False,
                },
                "experiment": {
                    "name": "test_experiment",
                },
            }
        )

        # Convert to Pydantic

        cfg = TrainingPipelineConfig.from_hydra(hydra_cfg)

        assert cfg.model.device == "cpu"

        assert cfg.training.n_splits == 10

        assert cfg.experiment.name == "test_experiment"

        # training.batch_size should populate model.batch_size for backward compatibility
        assert cfg.model.batch_size == 4
        assert cfg.training.batch_size == 4
        assert cfg.training.random_state == 99
        assert cfg.training.stratify is False
