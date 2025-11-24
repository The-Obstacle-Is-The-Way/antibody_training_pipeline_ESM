#!/usr/bin/env python3
"""
Integration Tests for AMPLIFY Model Integration

Tests the full integration of AMPLIFY 350M with BinaryClassifier and Hydra config.
Validates that the model_type selection works correctly.

Date: 2025-11-24
Coverage Target: AMPLIFY config + BinaryClassifier integration
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.fixtures.mock_models import create_mock_labels

# ============================================================================
# 1. BinaryClassifier with AMPLIFY model_type
# ============================================================================


@pytest.fixture
def amplify_classifier_params() -> dict[str, Any]:
    """Full params for AMPLIFY BinaryClassifier including LogReg defaults"""
    return {
        "model_name": "chandar-lab/AMPLIFY_350M",
        "device": "cpu",
        "random_state": 42,
        "model_type": "amplify",
        # LogReg classifier params
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": None,
    }


@pytest.fixture
def esm_classifier_params() -> dict[str, Any]:
    """Full params for ESM BinaryClassifier including LogReg defaults"""
    return {
        "model_name": "facebook/esm1v_t33_650M_UR90S_1",
        "device": "cpu",
        "random_state": 42,
        # No model_type - defaults to "esm"
        # LogReg classifier params
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": None,
    }


@pytest.mark.integration
def test_binary_classifier_initializes_with_amplify_model_type(
    mock_transformers_model: tuple[Any, Any],
    amplify_classifier_params: dict[str, Any],
) -> None:
    """Verify BinaryClassifier correctly selects AMPLIFYEmbeddingExtractor"""
    from antibody_training_esm.core.classifier import BinaryClassifier
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    classifier = BinaryClassifier(params=amplify_classifier_params)

    # Should have AMPLIFY extractor, not ESM
    assert isinstance(classifier.embedding_extractor, AMPLIFYEmbeddingExtractor)
    assert classifier._model_type == "amplify"


@pytest.mark.integration
def test_binary_classifier_defaults_to_esm_model_type(
    mock_transformers_model: tuple[Any, Any],
    esm_classifier_params: dict[str, Any],
) -> None:
    """Verify BinaryClassifier defaults to ESM extractor for backward compat"""
    from antibody_training_esm.core.classifier import BinaryClassifier
    from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor

    classifier = BinaryClassifier(params=esm_classifier_params)

    # Should have ESM extractor (default)
    assert isinstance(classifier.embedding_extractor, ESMEmbeddingExtractor)
    assert classifier._model_type == "esm"


@pytest.mark.integration
def test_binary_classifier_raises_on_invalid_model_type(
    mock_transformers_model: tuple[Any, Any],
    esm_classifier_params: dict[str, Any],
) -> None:
    """Verify BinaryClassifier raises ValueError for invalid model_type"""
    from antibody_training_esm.core.classifier import BinaryClassifier

    params = esm_classifier_params.copy()
    params["model_type"] = "invalid"  # Invalid model type

    with pytest.raises(ValueError, match="Unknown model_type"):
        BinaryClassifier(params=params)


@pytest.mark.integration
def test_binary_classifier_get_params_includes_model_type(
    mock_transformers_model: tuple[Any, Any],
    amplify_classifier_params: dict[str, Any],
) -> None:
    """Verify get_params() returns model_type for sklearn compatibility"""
    from antibody_training_esm.core.classifier import BinaryClassifier

    classifier = BinaryClassifier(params=amplify_classifier_params)
    retrieved_params = classifier.get_params()

    assert "model_type" in retrieved_params
    assert retrieved_params["model_type"] == "amplify"


@pytest.mark.integration
def test_binary_classifier_amplify_returns_960d_embeddings(
    mock_transformers_model: tuple[Any, Any],
    amplify_classifier_params: dict[str, Any],
) -> None:
    """Verify AMPLIFY classifier extracts 960-d embeddings"""
    from antibody_training_esm.core.classifier import BinaryClassifier

    classifier = BinaryClassifier(params=amplify_classifier_params)

    # Extract single embedding
    embedding = classifier.embedding_extractor.embed_sequence("QVQLVQSG")

    # AMPLIFY returns 960-d (not 1280-d)
    assert embedding.shape == (960,)


@pytest.mark.integration
def test_binary_classifier_amplify_fit_predict(
    mock_transformers_model: tuple[Any, Any],
    amplify_classifier_params: dict[str, Any],
) -> None:
    """Verify AMPLIFY classifier can fit and predict"""
    from antibody_training_esm.core.classifier import BinaryClassifier

    classifier = BinaryClassifier(params=amplify_classifier_params)

    # Create mock 960-d embeddings (AMPLIFY dimension)
    n_samples = 10
    X_train = np.random.rand(n_samples, 960).astype(np.float32)
    y_train = create_mock_labels(n_samples=n_samples, balanced=True)

    # Fit
    classifier.fit(X_train, y_train)
    assert classifier.is_fitted

    # Predict
    predictions = classifier.predict(X_train)
    assert predictions.shape == (n_samples,)
    assert all(p in [0, 1] for p in predictions)


# ============================================================================
# 2. Hydra Config Integration
# ============================================================================


@pytest.mark.integration
def test_amplify_config_loads_via_hydra(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify AMPLIFY config can be loaded via Hydra"""
    from hydra import compose, initialize_config_module

    # Use the config module from our package (registered via ConfigStore)
    with initialize_config_module(
        config_module="antibody_training_esm.conf",
        version_base=None,
    ):
        cfg = compose(config_name="config", overrides=["model=amplify_350m"])

    # Verify AMPLIFY config values
    assert cfg.model.name == "chandar-lab/AMPLIFY_350M"
    assert cfg.model.model_type == "amplify"
    assert cfg.model.trust_remote_code is True
    assert cfg.model.batch_size == 1  # CRITICAL: padding bug


@pytest.mark.integration
def test_amplify_config_batch_size_enforced(
    mock_transformers_model: tuple[Any, Any],
) -> None:
    """Verify AMPLIFY batch_size=1 is enforced even with training.batch_size override"""
    from hydra import compose, initialize_config_module

    # Use the config module from our package
    with initialize_config_module(
        config_module="antibody_training_esm.conf",
        version_base=None,
    ):
        cfg = compose(
            config_name="config",
            overrides=["model=amplify_350m", "training.batch_size=32"],
        )

    # model.batch_size should still be 1 (overrides training.batch_size for AMPLIFY)
    assert cfg.model.batch_size == 1


# ============================================================================
# 3. Model Serialization with AMPLIFY
# ============================================================================


@pytest.mark.integration
def test_amplify_classifier_pickle_round_trip(
    mock_transformers_model: tuple[Any, Any],
    amplify_classifier_params: dict[str, Any],
    tmp_path: Any,
) -> None:
    """Verify AMPLIFY classifier survives pickle/unpickle"""
    import pickle

    from antibody_training_esm.core.classifier import BinaryClassifier
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    classifier = BinaryClassifier(params=amplify_classifier_params)

    # Fit with mock data
    X_train = np.random.rand(10, 960).astype(np.float32)
    y_train = create_mock_labels(n_samples=10, balanced=True)
    classifier.fit(X_train, y_train)

    # Pickle
    model_path = tmp_path / "amplify_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(classifier, f)

    # Unpickle
    with open(model_path, "rb") as f:
        loaded = pickle.load(f)

    # Verify
    assert isinstance(loaded.embedding_extractor, AMPLIFYEmbeddingExtractor)
    assert loaded._model_type == "amplify"
    assert loaded.is_fitted


@pytest.mark.integration
def test_esm_classifier_pickle_backwards_compatible(
    mock_transformers_model: tuple[Any, Any],
    esm_classifier_params: dict[str, Any],
    tmp_path: Any,
) -> None:
    """Verify old ESM models (without model_type) still load correctly"""
    import pickle

    from antibody_training_esm.core.classifier import BinaryClassifier
    from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor

    classifier = BinaryClassifier(params=esm_classifier_params)

    # Fit with mock data
    X_train = np.random.rand(10, 1280).astype(np.float32)
    y_train = create_mock_labels(n_samples=10, balanced=True)
    classifier.fit(X_train, y_train)

    # Manually remove _model_type to simulate old model
    del classifier._model_type

    # Pickle
    model_path = tmp_path / "old_esm_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(classifier, f)

    # Unpickle (should trigger warning and default to ESM)
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with open(model_path, "rb") as f:
            loaded = pickle.load(f)

        # Should warn about missing _model_type
        assert any("_model_type" in str(warning.message) for warning in w)

    # Should default to ESM
    assert isinstance(loaded.embedding_extractor, ESMEmbeddingExtractor)


# ============================================================================
# 4. set_params with model_type change
# ============================================================================


@pytest.mark.integration
def test_set_params_changes_model_type_esm_to_amplify(
    mock_transformers_model: tuple[Any, Any],
    esm_classifier_params: dict[str, Any],
) -> None:
    """Verify set_params() can switch from ESM to AMPLIFY extractor"""
    from antibody_training_esm.core.classifier import BinaryClassifier
    from antibody_training_esm.core.embeddings_amplify import AMPLIFYEmbeddingExtractor

    # Start with ESM
    params = esm_classifier_params.copy()
    params["model_type"] = "esm"  # Explicit

    classifier = BinaryClassifier(params=params)
    assert classifier._model_type == "esm"

    # Switch to AMPLIFY
    classifier.set_params(
        model_name="chandar-lab/AMPLIFY_350M",
        model_type="amplify",
    )

    assert classifier._model_type == "amplify"
    assert isinstance(classifier.embedding_extractor, AMPLIFYEmbeddingExtractor)


# ============================================================================
# Total Tests: 12 integration tests
# ============================================================================
