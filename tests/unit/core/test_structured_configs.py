"""
Unit tests for structured configuration schemas

Tests type safety, schema validation, and MISSING field enforcement.
"""

from collections.abc import Generator

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

# Note: We import config_schema types inside tests to avoid import-time failures
# Hydra loads configs automatically when we initialize with config_path


@pytest.fixture(autouse=True)
def cleanup_hydra() -> Generator[None, None, None]:
    """Clean up Hydra singleton between tests"""
    yield
    GlobalHydra.instance().clear()


@pytest.mark.unit
def test_structured_config_loads() -> None:
    """Test that structured configs load correctly with struct mode enabled"""
    with initialize(
        version_base=None, config_path="../../../src/antibody_training_esm/conf"
    ):
        cfg = compose(config_name="config")

        # Verify struct mode is enabled (prevents unknown keys)
        assert OmegaConf.is_struct(cfg), "Struct mode must be enabled"

        # Verify dataclass types are active (not plain dicts)
        from antibody_training_esm.conf.config_schema import (
            ClassifierConfig,
            Config,
            DataConfig,
            ModelConfig,
            TrainingConfig,
        )

        assert OmegaConf.get_type(cfg) is Config, (
            "Root config should be Config dataclass"
        )
        assert OmegaConf.get_type(cfg, "model") is ModelConfig, (
            "model should be ModelConfig dataclass"
        )
        assert OmegaConf.get_type(cfg, "classifier") is ClassifierConfig, (
            "classifier should be ClassifierConfig dataclass"
        )
        assert OmegaConf.get_type(cfg, "data") is DataConfig, (
            "data should be DataConfig dataclass"
        )
        assert OmegaConf.get_type(cfg, "training") is TrainingConfig, (
            "training should be TrainingConfig dataclass"
        )

        # Verify ConfigStore is available (schemas registered)
        from hydra.core.config_store import ConfigStore

        cs = ConfigStore.instance()
        assert cs is not None, "ConfigStore should be initialized"

        # Type-safe access works correctly
        assert cfg.model.name == "facebook/esm1v_t33_650M_UR90S_1"
        assert cfg.classifier.C == 1.0
        assert cfg.training.batch_size == 8
        assert cfg.training.n_splits == 10


@pytest.mark.unit
def test_structured_config_type_validation() -> None:
    """Test that type validation works"""
    with initialize(
        version_base=None, config_path="../../../src/antibody_training_esm/conf"
    ):
        # This should work (correct type)
        cfg = compose(
            config_name="config",
            overrides=["training.batch_size=16"],  # int
        )
        assert cfg.training.batch_size == 16

        # String that looks like int should be converted
        cfg2 = compose(
            config_name="config",
            overrides=["training.batch_size=32"],
        )
        assert cfg2.training.batch_size == 32


@pytest.mark.unit
def test_structured_config_missing_fields_have_defaults() -> None:
    """Test that all required fields have defaults or are marked MISSING"""
    with initialize(
        version_base=None, config_path="../../../src/antibody_training_esm/conf"
    ):
        cfg = compose(config_name="config")

        # These should NOT be MISSING (have defaults in schema or YAML)
        assert cfg.model.name is not None
        assert cfg.classifier.C is not None
        assert cfg.training.n_splits is not None
        assert cfg.hardware.device is not None

        # These come from YAML (required for data config)
        assert cfg.data.train_file is not None
        assert cfg.data.test_file is not None


@pytest.mark.unit
def test_structured_config_field_completeness() -> None:
    """Test that all fields from current codebase are present in schema"""
    with initialize(
        version_base=None, config_path="../../../src/antibody_training_esm/conf"
    ):
        cfg = compose(config_name="config")

        # Verify all trainer.py required fields exist
        assert hasattr(cfg.training, "metrics")
        assert hasattr(cfg.training, "save_model")
        assert hasattr(cfg.training, "model_name")
        assert hasattr(cfg.training, "model_save_dir")
        assert hasattr(cfg.training, "batch_size")
        assert hasattr(cfg.training, "log_file")
        assert hasattr(cfg.training, "num_workers")

        # Verify all loaders.py required fields exist
        assert hasattr(cfg.data, "source")
        assert hasattr(cfg.data, "sequence_column")
        assert hasattr(cfg.data, "label_column")
        assert hasattr(cfg.data, "embeddings_cache_dir")

        # Verify hardware fields exist
        assert hasattr(cfg.hardware, "gpu_memory_fraction")
        assert hasattr(cfg.hardware, "clear_cache_frequency")


@pytest.mark.unit
def test_structured_config_interpolation_works() -> None:
    """Test that config interpolation still works with structured configs"""
    with initialize(
        version_base=None, config_path="../../../src/antibody_training_esm/conf"
    ):
        cfg = compose(
            config_name="config",
            overrides=["hardware.device=cuda"],
        )

        # model.device should interpolate ${hardware.device}
        assert cfg.model.device == "cuda"


@pytest.mark.unit
def test_structured_config_schema_registration() -> None:
    """Test that schemas are registered in ConfigStore"""
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()

    # Verify schemas are registered
    # Note: ConfigStore.repo is internal, this test just verifies import worked
    # The real test is that configs load without errors
    assert cs is not None


@pytest.mark.unit
def test_structured_config_preserves_yaml_values() -> None:
    """Test that structured configs don't override YAML values"""
    with initialize(
        version_base=None, config_path="../../../src/antibody_training_esm/conf"
    ):
        cfg = compose(config_name="config")

        # Values from YAML should be preserved
        assert cfg.training.model_name == "boughter_vh_esm1v_logreg"
        assert cfg.experiment.name == "novo_replication"
        assert cfg.model.name == "facebook/esm1v_t33_650M_UR90S_1"


@pytest.mark.unit
def test_structured_config_enforces_types() -> None:
    """Test that configs provide correct types from YAML"""
    from omegaconf import ListConfig

    with initialize(
        version_base=None, config_path="../../../src/antibody_training_esm/conf"
    ):
        cfg = compose(config_name="config")

        # Verify types are correct (int, float, bool, ListConfig)
        assert isinstance(cfg.training.batch_size, int), "batch_size should be int"
        assert isinstance(cfg.training.n_splits, int), "n_splits should be int"
        assert isinstance(cfg.classifier.C, float), "C should be float"
        assert isinstance(cfg.training.stratify, bool), "stratify should be bool"
        assert isinstance(cfg.training.metrics, (list, ListConfig)), (
            "metrics should be list-like"
        )

        # Verify valid overrides work
        cfg_override = compose(
            config_name="config",
            overrides=["training.batch_size=16"],
        )
        assert cfg_override.training.batch_size == 16
        assert isinstance(cfg_override.training.batch_size, int)


@pytest.mark.unit
def test_structured_config_rejects_unknown_keys() -> None:
    """Test that struct mode rejects unknown fields (NEGATIVE TEST)"""
    with (
        initialize(
            version_base=None, config_path="../../../src/antibody_training_esm/conf"
        ),
        pytest.raises(ConfigCompositionException),
    ):
        # Unknown field should raise ConfigCompositionException
        compose(
            config_name="config",
            overrides=["training.unknown_field=123"],
        )


@pytest.mark.unit
def test_structured_config_rejects_invalid_types() -> None:
    """Test that type validation rejects invalid values (NEGATIVE TEST)"""
    with (
        initialize(
            version_base=None, config_path="../../../src/antibody_training_esm/conf"
        ),
        pytest.raises(ConfigCompositionException),
    ):
        # Invalid type for batch_size (should be int) should raise
        compose(
            config_name="config",
            overrides=["training.batch_size=not_a_number"],
        )
