"""
Integration Tests for Hybrid Pipeline (ESM + Biophysical)

Tests Phase C: Pipeline Integration for hybrid feature models.
Validates feature concatenation, shape compatibility, and config integration.

Date: 2025-11-27
"""

import numpy as np
import pytest

from antibody_training_esm.core.biophysical import BiophysicalExtractor

# ============================================================================
# Test Fixtures - Known Sequences
# ============================================================================

# Trastuzumab VH (Herceptin) - well-characterized therapeutic antibody
TRASTUZUMAB_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAED"
    "TAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)

# Simple test sequences
TEST_SEQUENCES = [
    TRASTUZUMAB_VH,
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAED",
    "QVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAED",
]


@pytest.fixture
def bio_extractor() -> BiophysicalExtractor:
    """Create BiophysicalExtractor instance."""
    return BiophysicalExtractor()


@pytest.fixture
def mock_esm_embeddings() -> np.ndarray:
    """Create mock ESM embeddings for testing (n=3, dim=1280)."""
    # Deterministic mock embeddings for reproducibility
    rng = np.random.default_rng(42)
    return rng.standard_normal((len(TEST_SEQUENCES), 1280)).astype(np.float32)


# ============================================================================
# Feature Concatenation Tests
# ============================================================================


@pytest.mark.integration
def test_feature_concatenation_shape(
    bio_extractor: BiophysicalExtractor,
    mock_esm_embeddings: np.ndarray,
) -> None:
    """
    Verify feature concatenation produces correct shape.

    ESM: (n, 1280) + Bio: (n, 3) → Hybrid: (n, 1283)
    """
    # Extract biophysical features
    X_bio = bio_extractor.extract_batch_features(TEST_SEQUENCES)

    # Verify individual shapes
    assert mock_esm_embeddings.shape == (3, 1280), "ESM shape should be (3, 1280)"
    assert X_bio.shape == (3, 3), "Biophysical shape should be (3, 3)"

    # Concatenate
    X_hybrid = np.concatenate([mock_esm_embeddings, X_bio], axis=1)

    # Verify hybrid shape
    assert X_hybrid.shape == (3, 1283), "Hybrid shape should be (3, 1283)"


@pytest.mark.integration
def test_feature_concatenation_dtype(
    bio_extractor: BiophysicalExtractor,
    mock_esm_embeddings: np.ndarray,
) -> None:
    """
    Verify feature concatenation maintains float32 dtype.

    Both ESM and biophysical should be float32 for sklearn compatibility.
    """
    X_bio = bio_extractor.extract_batch_features(TEST_SEQUENCES)

    # Verify dtypes
    assert mock_esm_embeddings.dtype == np.float32, "ESM should be float32"
    assert X_bio.dtype == np.float32, "Biophysical should be float32"

    # Concatenate and verify
    X_hybrid = np.concatenate([mock_esm_embeddings, X_bio], axis=1)
    assert X_hybrid.dtype == np.float32, "Hybrid should be float32"


@pytest.mark.integration
def test_feature_concatenation_no_nan(
    bio_extractor: BiophysicalExtractor,
    mock_esm_embeddings: np.ndarray,
) -> None:
    """
    Verify concatenated features contain no NaN values.
    """
    X_bio = bio_extractor.extract_batch_features(TEST_SEQUENCES)
    X_hybrid = np.concatenate([mock_esm_embeddings, X_bio], axis=1)

    assert not np.isnan(X_hybrid).any(), "Hybrid features should not contain NaN"
    assert not np.isinf(X_hybrid).any(), "Hybrid features should not contain Inf"


# ============================================================================
# Classifier Compatibility Tests
# ============================================================================


@pytest.mark.integration
def test_sklearn_logreg_accepts_hybrid_features(
    bio_extractor: BiophysicalExtractor,
    mock_esm_embeddings: np.ndarray,
) -> None:
    """
    Verify sklearn LogisticRegression can fit on hybrid features.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X_bio = bio_extractor.extract_batch_features(TEST_SEQUENCES)
    X_hybrid = np.concatenate([mock_esm_embeddings, X_bio], axis=1)

    # Create labels (alternating for test)
    y = np.array([0, 1, 0])

    # Scale features (recommended for LogReg)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_hybrid)

    # Fit LogisticRegression
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_scaled, y)

    # Verify coefficient shape matches hybrid features
    assert clf.coef_.shape == (1, 1283), (
        f"Coef shape should be (1, 1283), got {clf.coef_.shape}"
    )

    # Verify prediction shape
    y_pred = clf.predict(X_scaled)
    assert y_pred.shape == (3,), "Prediction shape should match input"


@pytest.mark.integration
def test_sklearn_logreg_predicts_probabilities(
    bio_extractor: BiophysicalExtractor,
    mock_esm_embeddings: np.ndarray,
) -> None:
    """
    Verify sklearn LogisticRegression can predict probabilities on hybrid features.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X_bio = bio_extractor.extract_batch_features(TEST_SEQUENCES)
    X_hybrid = np.concatenate([mock_esm_embeddings, X_bio], axis=1)
    y = np.array([0, 1, 0])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_hybrid)

    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_scaled, y)

    # Verify probability output
    y_prob = clf.predict_proba(X_scaled)
    assert y_prob.shape == (3, 2), "Probability shape should be (n, 2)"
    assert np.allclose(y_prob.sum(axis=1), 1.0), "Probabilities should sum to 1"


# ============================================================================
# Biophysical Contribution Tests
# ============================================================================


@pytest.mark.integration
def test_biophysical_features_in_correct_position(
    bio_extractor: BiophysicalExtractor,
    mock_esm_embeddings: np.ndarray,
) -> None:
    """
    Verify biophysical features are appended at the end of hybrid vector.

    Last 3 columns should match standalone biophysical extraction.
    """
    X_bio = bio_extractor.extract_batch_features(TEST_SEQUENCES)
    X_hybrid = np.concatenate([mock_esm_embeddings, X_bio], axis=1)

    # Extract last 3 columns
    X_bio_from_hybrid = X_hybrid[:, -3:]

    # Should match standalone extraction
    np.testing.assert_array_equal(
        X_bio_from_hybrid,
        X_bio,
        err_msg="Biophysical features should be in last 3 columns",
    )


@pytest.mark.integration
def test_esm_features_preserved_in_hybrid(
    bio_extractor: BiophysicalExtractor,
    mock_esm_embeddings: np.ndarray,
) -> None:
    """
    Verify ESM features are preserved in first 1280 columns of hybrid vector.
    """
    X_bio = bio_extractor.extract_batch_features(TEST_SEQUENCES)
    X_hybrid = np.concatenate([mock_esm_embeddings, X_bio], axis=1)

    # Extract first 1280 columns
    X_esm_from_hybrid = X_hybrid[:, :1280]

    # Should match original ESM embeddings
    np.testing.assert_array_equal(
        X_esm_from_hybrid,
        mock_esm_embeddings,
        err_msg="ESM features should be in first 1280 columns",
    )


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.integration
def test_single_sequence_hybrid(
    bio_extractor: BiophysicalExtractor,
) -> None:
    """
    Verify hybrid pipeline works with single sequence.
    """
    rng = np.random.default_rng(42)
    X_esm = rng.standard_normal((1, 1280)).astype(np.float32)
    X_bio = bio_extractor.extract_batch_features([TRASTUZUMAB_VH])
    X_hybrid = np.concatenate([X_esm, X_bio], axis=1)

    assert X_hybrid.shape == (1, 1283), (
        "Single sequence hybrid shape should be (1, 1283)"
    )


@pytest.mark.integration
def test_large_batch_hybrid(
    bio_extractor: BiophysicalExtractor,
) -> None:
    """
    Verify hybrid pipeline works with large batch (100+ sequences).
    """
    # Replicate test sequence to create large batch
    sequences = TEST_SEQUENCES * 40  # 120 sequences
    rng = np.random.default_rng(42)
    X_esm = rng.standard_normal((len(sequences), 1280)).astype(np.float32)
    X_bio = bio_extractor.extract_batch_features(sequences)
    X_hybrid = np.concatenate([X_esm, X_bio], axis=1)

    assert X_hybrid.shape == (120, 1283), (
        "Large batch hybrid shape should be (120, 1283)"
    )


# ============================================================================
# Config Integration Tests
# ============================================================================


@pytest.mark.integration
def test_features_config_default_disabled() -> None:
    """Verify default FeaturesConfig has biophysical disabled."""
    from antibody_training_esm.models.config import FeaturesConfig

    config = FeaturesConfig()
    assert config.use_biophysical is False, "Default should have biophysical disabled"


@pytest.mark.integration
def test_features_config_hybrid_enabled() -> None:
    """Verify FeaturesConfig can enable biophysical."""
    from antibody_training_esm.models.config import FeaturesConfig

    config = FeaturesConfig(use_biophysical=True)
    assert config.use_biophysical is True, "Hybrid mode should have biophysical enabled"


@pytest.mark.integration
def test_training_pipeline_config_includes_features() -> None:
    """Verify TrainingPipelineConfig has features section."""
    import tempfile
    from pathlib import Path

    from antibody_training_esm.models.config import (
        ClassifierConfig,
        DataConfig,
        ExperimentConfig,
        FeaturesConfig,
        ModelConfig,
        TrainingConfig,
        TrainingPipelineConfig,
    )

    # Create minimal valid config
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        f.write("sequence,label\nEVQL,0\nQVQL,1\n")
        temp_file = Path(f.name)

    try:
        config = TrainingPipelineConfig(
            model=ModelConfig(name="facebook/esm1v_t33_650M_UR90S_1"),
            data=DataConfig(train_file=temp_file, test_file=temp_file),
            classifier=ClassifierConfig(),
            training=TrainingConfig(model_name="test"),
            experiment=ExperimentConfig(name="test"),
            features=FeaturesConfig(use_biophysical=True),
        )

        assert config.features.use_biophysical is True
    finally:
        temp_file.unlink()
