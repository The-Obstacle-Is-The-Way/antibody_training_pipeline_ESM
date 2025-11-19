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


def test_predictor_class(sample_input_df: pd.DataFrame) -> None:
    """Test the Predictor class directly."""
    with (
        patch("joblib.load") as mock_joblib_load,
        patch(
            "antibody_training_esm.core.prediction.ESMEmbeddingExtractor"
        ) as mock_embedder,
    ):
        # Mocking the classifier
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = np.array([1, 0])
        mock_classifier.predict_proba.return_value = np.array(
            [[0.15, 0.85], [0.77, 0.23]]
        )
        mock_joblib_load.return_value = mock_classifier

        # Mocking the ESMEmbeddingExtractor
        mock_embedder.return_value.extract_batch_embeddings.return_value = (
            np.random.rand(2, 1280)
        )

        # Initialize Predictor
        predictor = Predictor(
            model_name="facebook/esm1v_t33_650M_UR90S_1", classifier_path="dummy_path"
        )

        # Test predict_dataframe
        output_df = predictor.predict_dataframe(sample_input_df)

        # Assertions
        assert "prediction" in output_df.columns
        assert "probability" in output_df.columns
        assert output_df["prediction"].tolist() == ["non-specific", "specific"]
        assert np.allclose(output_df["probability"].tolist(), [0.85, 0.23])

        # Verify lazy loading
        mock_joblib_load.assert_called_once()
        mock_embedder.assert_called_once()


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
            }
        )

        run_prediction(sample_input_df, cfg)

        mock_predictor_cls.assert_called_with(
            model_name="test_model", classifier_path="test_path"
        )
        mock_instance.predict_dataframe.assert_called_with(sample_input_df)
