#!/usr/bin/env python3
"""
Integration Tests: AMPLIFY + Hydra Configuration

Verifies that AMPLIFY can be loaded via Hydra config without breaking existing ESM configs.

Date: 2025-11-23
"""

from pathlib import Path
from typing import Any

import pytest
from hydra import compose, initialize_config_dir

from antibody_training_esm.core.classifier import BinaryClassifier
from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor


@pytest.mark.integration
def test_amplify_loads_via_hydra(mock_transformers_model: Any) -> None:
    """Verify AMPLIFY config loads correctly via Hydra"""
    config_dir = (
        Path(__file__).parent.parent.parent / "src" / "antibody_training_esm" / "conf"
    )

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["model=amplify_350m"])

        # Verify config values
        assert cfg.model.name == "chandar-lab/AMPLIFY_350M"
        assert cfg.model.model_type == "amplify"
        assert cfg.model.batch_size == 1
        assert cfg.model.trust_remote_code is True
        assert cfg.model.revision == "main"
        assert cfg.model.device == cfg.hardware.device  # Interpolated


@pytest.mark.integration
def test_esm1v_still_works_after_amplify_addition(mock_transformers_model: Any) -> None:
    """Verify ESM-1v config still works (backward compatibility)"""
    config_dir = (
        Path(__file__).parent.parent.parent / "src" / "antibody_training_esm" / "conf"
    )

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["model=esm1v"])

        # Should still work (no model_type field in old configs)
        assert cfg.model.name == "facebook/esm1v_t33_650M_UR90S_1"
        assert cfg.model.revision == "main"
        # model_type should be missing (backward compat)
        assert (
            not hasattr(cfg.model, "model_type") or cfg.model.get("model_type") is None
        )


@pytest.mark.integration
def test_esm2_still_works_after_amplify_addition(mock_transformers_model: Any) -> None:
    """Verify ESM-2 config still works (backward compatibility)"""
    config_dir = (
        Path(__file__).parent.parent.parent / "src" / "antibody_training_esm" / "conf"
    )

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["model=esm2_650m"])

        assert cfg.model.name == "facebook/esm2_t33_650M_UR50D"
        assert cfg.model.revision == "main"


@pytest.mark.integration
def test_binary_classifier_with_amplify_hydra_config(
    mock_transformers_model: Any,
) -> None:
    """Verify BinaryClassifier can be initialized from AMPLIFY Hydra config"""
    config_dir = (
        Path(__file__).parent.parent.parent / "src" / "antibody_training_esm" / "conf"
    )

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["model=amplify_350m", "hardware.device=cpu"],
        )

        # Convert Hydra config to dict for BinaryClassifier
        params = {
            "model_name": cfg.model.name,
            "model_type": cfg.model.model_type,
            "device": cfg.model.device,
            "batch_size": cfg.model.batch_size,
            "revision": cfg.model.revision,
            "random_state": cfg.training.random_state,
            # Classifier params
            "C": cfg.classifier.C,
            "penalty": cfg.classifier.penalty,
            "solver": cfg.classifier.solver,
            "max_iter": cfg.classifier.max_iter,
            "class_weight": cfg.classifier.class_weight,
        }

        classifier = BinaryClassifier(params)

        assert isinstance(classifier.embedding_extractor, AMPLIFYEmbeddingExtractor)
        assert classifier.embedding_extractor.batch_size == 1


@pytest.mark.integration
def test_binary_classifier_with_esm_hydra_config(mock_transformers_model: Any) -> None:
    """Verify BinaryClassifier still works with ESM Hydra config (backward compat)"""
    config_dir = (
        Path(__file__).parent.parent.parent / "src" / "antibody_training_esm" / "conf"
    )

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="config", overrides=["model=esm1v", "hardware.device=cpu"]
        )

        params = {
            "model_name": cfg.model.name,
            # No model_type (legacy)
            "device": cfg.model.device,
            "batch_size": cfg.training.batch_size,  # From training config
            "revision": cfg.model.revision,
            "random_state": cfg.training.random_state,
            # Classifier params
            "C": cfg.classifier.C,
            "penalty": cfg.classifier.penalty,
            "solver": cfg.classifier.solver,
            "max_iter": cfg.classifier.max_iter,
            "class_weight": cfg.classifier.class_weight,
        }

        classifier = BinaryClassifier(params)

        assert isinstance(classifier.embedding_extractor, ESMEmbeddingExtractor)
