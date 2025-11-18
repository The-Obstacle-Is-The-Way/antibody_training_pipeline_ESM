
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from omegaconf import OmegaConf

from antibody_training_esm.core.prediction import run_prediction


@pytest.fixture
def sample_input_df():
    return pd.DataFrame(
        {
            "sequence": [
                "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKDRLGRYFDYWGQGTLVTVSS",
                "QVQLQESGPGLVKPSETLSLTCTVSGGSISSSYYWGWIRQPPGKGLEWIGSIYYSGSTYYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARDRLGRYFDYWGQGTLVTVSS",
            ]
        }
    )


def test_run_prediction(sample_input_df):
    with patch("joblib.load") as mock_joblib_load, patch(
        "transformers.AutoTokenizer.from_pretrained"
    ), patch(
        "transformers.AutoModelForMaskedLM.from_pretrained"
    ), patch(
        "antibody_training_esm.core.prediction.get_embeddings"
    ) as mock_get_embeddings:
        # Mocking the classifier
        mock_classifier = MagicMock()
        mock_classifier.predict.return_value = np.array([1, 0])
        mock_classifier.predict_proba.return_value = np.array(
            [[0.15, 0.85], [0.77, 0.23]]
        )
        mock_joblib_load.return_value = mock_classifier

        # Mocking get_embeddings to avoid actual model loading/inference
        mock_get_embeddings.return_value = np.random.rand(2, 1280)  # dummy embeddings

        # Create a mock config object
        cfg = OmegaConf.create(
            {
                "model": {"name": "facebook/esm1v_t33_650M_UR90S_1"},
                "classifier": {"path": "dummy_path"},
            }
        )

        # Call the function
        output_df = run_prediction(sample_input_df, cfg)

        # Assertions
        assert "prediction" in output_df.columns
        assert "probability" in output_df.columns
        assert output_df["prediction"].tolist() == ["non-specific", "specific"]
        assert np.allclose(output_df["probability"].tolist(), [0.85, 0.23])
