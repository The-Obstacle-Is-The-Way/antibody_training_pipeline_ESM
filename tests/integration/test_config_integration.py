"""Integration tests for Pydantic config + trainer."""

import pytest
from hydra import compose, initialize
from pydantic import ValidationError

from antibody_training_esm.core.trainer import validate_config
from antibody_training_esm.models.config import TrainingPipelineConfig


def test_hydra_config_validates_with_pydantic() -> None:
    """Actual Hydra config.yaml validates successfully."""

    with initialize(
        config_path="../../src/antibody_training_esm/conf", version_base=None
    ):
        cfg = compose(config_name="config")

        # Should not raise ValidationError

        validated_config = validate_config(cfg)

        assert isinstance(validated_config, TrainingPipelineConfig)

        # Validating expected default model from config.yaml -> defaults -> model/esm1v.yaml

        # name: facebook/esm1v_t33_650M_UR90S_1

        assert validated_config.model.name == "facebook/esm1v_t33_650M_UR90S_1"


def test_invalid_hydra_override_caught() -> None:
    """Invalid Hydra override raises ValidationError."""

    with initialize(
        config_path="../../src/antibody_training_esm/conf", version_base=None
    ):
        cfg = compose(
            config_name="config",
            overrides=["model.device=tpu"],  # Invalid device
        )

        with pytest.raises(ValidationError):
            validate_config(cfg)
