import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from antibody_training_esm.core.biophysical import BiophysicalExtractor
from antibody_training_esm.datasets.boughter import load_boughter_data

# Mark as integration test
pytestmark = pytest.mark.integration


class TestTrackBReproducibility:
    """
    Integration tests for Track B (Biophysical Descriptors) reproducibility.
    Ensures that the BiophysicalExtractor works correctly with real datasets
    and integrates into a standard scikit-learn training pipeline.
    """

    @pytest.fixture(scope="class")
    def extractor(self) -> BiophysicalExtractor:
        return BiophysicalExtractor()

    @pytest.fixture(scope="class")
    def sample_data(self) -> pd.DataFrame:
        """Load a small sample of Boughter data for testing."""
        # Load with mild flags included to get more data
        df = load_boughter_data(include_mild=True)

        # Filter for valid sequences:
        # 1. Length > 0
        # 2. No 'X' (ambiguous amino acids) - BiophysicalExtractor requires exact AAs
        valid_mask = (df["VH_sequence"].str.len() > 0) & (
            ~df["VH_sequence"].str.contains("X")
        )
        df = df[valid_mask].head(20).copy()

        return df

    def test_biophysical_extraction_on_boughter_sample(
        self, extractor: BiophysicalExtractor, sample_data: pd.DataFrame
    ) -> None:
        """
        Test that features can be extracted from real Boughter sequences
        and produce the expected shape (N, 3).
        """
        sequences = sample_data["VH_sequence"].tolist()

        # Extract features
        features = extractor.extract_batch_features(sequences)

        # Check shape: (n_samples, 3 features)
        assert features.shape == (len(sequences), 3)

        # Check for NaNs or Infs
        assert not np.isnan(features).any(), "Features contain NaNs"
        assert not np.isinf(features).any(), "Features contain Infs"

        # Check value ranges (rough sanity check)
        # Charge should be roughly between -50 and +50
        assert np.all((features[:, 0] >= -50) & (features[:, 0] <= 50))  # pH 6
        assert np.all((features[:, 1] >= -50) & (features[:, 1] <= 50))  # pH 7.4
        # pI should be between 0 and 14
        assert np.all((features[:, 2] >= 0) & (features[:, 2] <= 14))  # pI

    def test_model_training_flow(
        self, extractor: BiophysicalExtractor, sample_data: pd.DataFrame
    ) -> None:
        """
        Test the full training flow: Extraction -> Scaling -> Training -> Prediction.
        This verifies that the features are compatible with sklearn models.
        """
        sequences = sample_data["VH_sequence"].tolist()
        # Deterministic synthetic labels to guarantee both classes (avoids flaky test)
        # Using alternating pattern ensures LogisticRegression always has 2 classes
        labels = (np.arange(len(sequences)) % 2).astype(int)

        # 1. Extract
        X = extractor.extract_batch_features(sequences)

        # 2. Scale (Critical for LogReg)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Check scaling (mean ~0, std ~1)
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=0.5)
        assert np.allclose(X_scaled.std(axis=0), 1, atol=0.5)

        # 3. Train
        model = LogisticRegression(random_state=42)
        model.fit(X_scaled, labels)

        # 4. Predict
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)

        # Assertions
        assert len(preds) == len(sequences)
        assert probs.shape == (len(sequences), 2)
        assert hasattr(model, "coef_")
        assert model.coef_.shape == (1, 3)  # 1 class (binary), 3 features
