"""
Unit tests for Hydra configuration system

Tests Hydra config loading, composition, and overrides.
"""

from collections.abc import Generator

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


@pytest.fixture(autouse=True)
def cleanup_hydra() -> Generator[None, None, None]:
    """Clean up Hydra singleton between tests"""
    yield
    GlobalHydra.instance().clear()


@pytest.mark.unit
def test_config_loads() -> None:
    """Test that Hydra config loads without errors"""
    # Use relative path from test file location to config directory
    # tests/unit/core/test_hydra_config.py -> src/antibody_training_esm/conf
    with initialize(
        version_base=None, config_path="../../../src/antibody_training_esm/conf"
    ):
        cfg = compose(config_name="config")

        # Verify main sections exist
        assert "model" in cfg
        assert "classifier" in cfg
        assert "data" in cfg
        assert "training" in cfg
        assert "hardware" in cfg
        assert "experiment" in cfg

        # Verify model config
        assert cfg.model.name == "facebook/esm1v_t33_650M_UR90S_1"
        assert cfg.model.revision == "main"

        # Verify classifier config
        assert cfg.classifier.type == "logistic_regression"
        assert cfg.classifier.C == 1.0
        assert cfg.classifier.penalty == "l2"

        # Verify data config (all required fields)
        assert cfg.data.source == "local"
        assert cfg.data.sequence_column == "sequence"
        assert cfg.data.label_column == "label"
        assert cfg.data.embeddings_cache_dir == "./embeddings_cache"

        # Verify training config (all required fields)
        assert cfg.training.n_splits == 10
        assert cfg.training.metrics == [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ]
        assert cfg.training.save_model is True
        assert cfg.training.model_name == "boughter_vh_esm1v_logreg"
        assert cfg.training.batch_size == 8

        # Verify hardware config
        assert cfg.hardware.device == "mps"
        assert cfg.hardware.gpu_memory_fraction == 0.8

        # Verify experiment config
        assert cfg.experiment.name == "novo_replication"
        assert "baseline" in cfg.experiment.tags


@pytest.mark.unit
def test_config_overrides() -> None:
    """Test that Hydra overrides work"""
    with initialize(
        version_base=None, config_path="../../../src/antibody_training_esm/conf"
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "training.batch_size=16",
                "classifier.C=0.5",
                "hardware.device=cuda",
            ],
        )

        assert cfg.training.batch_size == 16
        assert cfg.classifier.C == 0.5
        assert cfg.hardware.device == "cuda"


@pytest.mark.unit
def test_config_interpolation() -> None:
    """Test that config interpolation works"""
    with initialize(
        version_base=None, config_path="../../../src/antibody_training_esm/conf"
    ):
        cfg = compose(
            config_name="config",
            overrides=["hardware.device=cuda"],
        )

        # model.device should interpolate ${hardware.device}
        assert cfg.model.device == "cuda"

        # classifier.random_state should interpolate ${training.random_state}
        assert cfg.classifier.random_state == 42


@pytest.mark.unit
def test_config_completeness() -> None:
    """Test that all required fields from current trainer.py are present"""
    with initialize(
        version_base=None, config_path="../../../src/antibody_training_esm/conf"
    ):
        cfg = compose(config_name="config")

        # Fields required by trainer.py:595-640
        assert hasattr(cfg.training, "batch_size")
        assert hasattr(cfg.training, "metrics")
        assert hasattr(cfg.training, "save_model")
        assert hasattr(cfg.training, "model_name")
        assert hasattr(cfg.training, "model_save_dir")
        assert hasattr(cfg.training, "log_file")
        assert hasattr(cfg.training, "num_workers")
        assert hasattr(cfg.training, "n_splits")
        assert hasattr(cfg.training, "log_level")

        # Fields required by loaders.py:199-216
        assert hasattr(cfg.data, "source")
        assert hasattr(cfg.data, "sequence_column")
        assert hasattr(cfg.data, "label_column")

        # Fields required by trainer.py:601
        assert hasattr(cfg.data, "embeddings_cache_dir")

        # Hardware fields
        assert hasattr(cfg.hardware, "device")
        assert hasattr(cfg.hardware, "gpu_memory_fraction")
        assert hasattr(cfg.hardware, "clear_cache_frequency")
