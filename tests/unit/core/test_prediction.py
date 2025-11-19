from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from antibody_training_esm.core.prediction import Predictor, run_prediction


@pytest.fixture
def sample_input_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence": [
                "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKDRLGRYFDYWGQGTLVTVSS",
                "QVQLQESGPGLVKPSETLSLTCTVSGGSISSSYYWGWIRQPPGKGLEWIGSIYYSGSTYYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARDRLGRYFDYWGQGTLVTVSS",
            ]
        }
    )


def test_predictor_creates_embedder_when_missing(sample_input_df: pd.DataFrame) -> None:
    """Test that Predictor creates an embedder if the classifier doesn't have one."""
    with (
        patch("joblib.load") as mock_joblib_load,
        patch(
            "antibody_training_esm.core.prediction.ESMEmbeddingExtractor"
        ) as mock_embedder_cls,
    ):
        # Setup mock classifier WITHOUT an embedding_extractor
        mock_classifier = MagicMock()
        del (
            mock_classifier.embedding_extractor
        )  # Explicitly remove it to trigger fallback

        # Configure mock to support introspection (needed for predict method check)
        mock_code = MagicMock()
        mock_code.co_varnames = ("X",)
        mock_classifier.predict.__code__ = mock_code

        mock_classifier.predict.return_value = np.array([1, 0])
        mock_classifier.predict_proba.return_value = np.array(
            [[0.15, 0.85], [0.77, 0.23]]
        )
        mock_joblib_load.return_value = mock_classifier

        # Mock embedder instance
        mock_embedder_instance = mock_embedder_cls.return_value
        mock_embedder_instance.extract_batch_embeddings.return_value = np.zeros(
            (2, 1280)
        )

        # Initialize
        predictor = Predictor("model", "path")

        # Run
        output_df = predictor.predict_dataframe(sample_input_df)

        # Assertions
        mock_embedder_cls.assert_called_once()  # Should be called!
        assert "prediction" in output_df.columns


def test_predictor_reuses_embedder(sample_input_df: pd.DataFrame) -> None:
    """Test that Predictor reuses the classifier's embedder if available."""
    with (
        patch("joblib.load") as mock_joblib_load,
        patch(
            "antibody_training_esm.core.prediction.ESMEmbeddingExtractor"
        ) as mock_embedder_cls,
    ):
        # Setup mock classifier WITH an embedding_extractor
        mock_classifier = MagicMock()
        # It has .embedding_extractor by default (MagicMock behavior), so we keep it.

        # Configure mock to support introspection
        mock_code = MagicMock()
        mock_code.co_varnames = ("X",)
        mock_classifier.predict.__code__ = mock_code

        mock_classifier.predict.return_value = np.array([1, 0])
        mock_classifier.predict_proba.return_value = np.array(
            [[0.15, 0.85], [0.77, 0.23]]
        )
        mock_joblib_load.return_value = mock_classifier

        # The classifier's embedded extractor
        existing_embedder = mock_classifier.embedding_extractor
        existing_embedder.extract_batch_embeddings.return_value = np.zeros((2, 1280))

        # Initialize
        predictor = Predictor("model", "path")

        # Run
        output_df = predictor.predict_dataframe(sample_input_df)

        # Assertions
        mock_embedder_cls.assert_not_called()  # Optimization: Should NOT be called!
        existing_embedder.extract_batch_embeddings.assert_called_once()  # The existing one should be used
        assert "prediction" in output_df.columns


def test_predictor_missing_column(sample_input_df: pd.DataFrame) -> None:
    """Test error handling for missing column."""
    predictor = Predictor("model", "path")
    bad_df = pd.DataFrame({"wrong_col": ["SEQ"]})

    with pytest.raises(ValueError, match="Input DataFrame must contain"):
        predictor.predict_dataframe(bad_df)


def test_run_prediction_wrapper(sample_input_df: pd.DataFrame) -> None:
    """Test the backward-compatible wrapper function."""
    with patch("antibody_training_esm.core.prediction.Predictor") as mock_predictor_cls:
        mock_instance = MagicMock()
        mock_predictor_cls.return_value = mock_instance

        cfg = OmegaConf.create(
            {
                "model": {"name": "test_model"},
                "classifier": {"path": "test_path"},
                "sequence_column": "sequence",
                "threshold": 0.5,
                "assay_type": None,
            }
        )

        run_prediction(sample_input_df, cfg)

        mock_predictor_cls.assert_called_with(
            model_name="test_model", classifier_path="test_path"
        )
        mock_instance.predict_dataframe.assert_called_with(
            sample_input_df, sequence_col="sequence", threshold=0.5, assay_type=None
        )
