"""Unit tests for Ginkgo GDPa1 dataset loader."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from antibody_training_esm.datasets.ginkgo import GinkgoDataset, GinkgoTestSet


@pytest.mark.unit
class TestGinkgoDataset:
    """Test suite for GinkgoDataset class."""

    def test_dataset_initialization(self) -> None:
        """Test that dataset can be initialized with default paths."""
        dataset = GinkgoDataset(target_property="PR_CHO")

        assert dataset.target_property == "PR_CHO"
        assert dataset.assay_file.exists()
        assert dataset.fold_file.exists()

    def test_load_data_returns_dataframe(self) -> None:
        """Test that load_data returns a pandas DataFrame."""
        dataset = GinkgoDataset(target_property="PR_CHO")
        df = dataset.load_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "antibody_id" in df.columns
        assert "antibody_name" in df.columns
        assert "vh_protein_sequence" in df.columns
        assert "vl_protein_sequence" in df.columns
        assert "PR_CHO" in df.columns
        assert "fold" in df.columns

    def test_load_data_drops_missing_values(self) -> None:
        """Test that rows with missing target values are dropped."""
        dataset = GinkgoDataset(target_property="PR_CHO")
        df = dataset.load_data()

        # Should have no NaN values in PR_CHO column
        assert df["PR_CHO"].isna().sum() == 0

        # Based on exploration, 197 antibodies have PR_CHO values
        assert len(df) == 197

    def test_get_sequences_and_labels(self) -> None:
        """Test that get_sequences_and_labels returns correct types."""
        dataset = GinkgoDataset(target_property="PR_CHO")
        sequences, labels, folds = dataset.get_sequences_and_labels()

        # Check types
        assert isinstance(sequences, list)
        assert isinstance(labels, np.ndarray)
        assert isinstance(folds, np.ndarray)

        # Check lengths match
        assert len(sequences) == len(labels) == len(folds)

        # Check that we have 197 samples
        assert len(sequences) == 197

    def test_sequences_are_valid(self) -> None:
        """Test that sequences are non-empty strings."""
        dataset = GinkgoDataset(target_property="PR_CHO")
        sequences, _, _ = dataset.get_sequences_and_labels()

        for seq in sequences:
            assert isinstance(seq, str)
            assert len(seq) > 0
            # Check that sequence contains only valid amino acids
            assert all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in seq)

    def test_labels_are_continuous(self) -> None:
        """Test that labels are continuous floats in expected range."""
        dataset = GinkgoDataset(target_property="PR_CHO")
        _, labels, _ = dataset.get_sequences_and_labels()

        # Labels should be floats
        assert labels.dtype in [np.float64, np.float32]

        # PR_CHO range is [0.0, 0.547] based on exploration
        assert labels.min() >= 0.0
        assert labels.max() <= 1.0

    def test_folds_are_valid(self) -> None:
        """Test that folds are integers 0-4."""
        dataset = GinkgoDataset(target_property="PR_CHO")
        _, _, folds = dataset.get_sequences_and_labels()

        # Folds should be integers
        assert folds.dtype in [np.int64, np.int32]

        # Should have exactly 5 folds: 0, 1, 2, 3, 4
        unique_folds = sorted(set(folds))
        assert unique_folds == [0, 1, 2, 3, 4]

    def test_fold_distribution(self) -> None:
        """Test that folds are reasonably balanced."""
        dataset = GinkgoDataset(target_property="PR_CHO")
        _, _, folds = dataset.get_sequences_and_labels()

        # Count samples per fold
        fold_counts = pd.Series(folds).value_counts().sort_index()

        # Each fold should have at least 30 samples (197/5 ≈ 39)
        assert all(fold_counts >= 30)

        # No fold should have more than 60 samples
        assert all(fold_counts <= 60)

    def test_different_target_property(self) -> None:
        """Test that dataset can load different target properties."""
        # Try loading HIC instead of PR_CHO
        dataset = GinkgoDataset(target_property="HIC")
        df = dataset.load_data()

        # Should have HIC column
        assert "HIC" in df.columns

        # Should drop rows with missing HIC values
        assert df["HIC"].isna().sum() == 0


@pytest.mark.unit
class TestGinkgoTestSet:
    """Test suite for GinkgoTestSet class."""

    @pytest.fixture
    def create_mock_test_file(self, tmp_path: Path) -> Path:
        """Create a mock test set CSV file for testing."""
        test_csv = tmp_path / "mock_test_set.csv"

        mock_data = pd.DataFrame(
            {
                "antibody_id": ["GDPa1-test-001", "GDPa1-test-002", "GDPa1-test-003"],
                "antibody_name": [
                    "test_antibody_1",
                    "test_antibody_2",
                    "test_antibody_3",
                ],
                "vh_protein_sequence": ["QVQLQQ" * 20, "EVQLVE" * 20, "QVKLQE" * 20],
                "vl_protein_sequence": ["DIQMTQ" * 15, "DIVMTQ" * 15, "DIQLTQ" * 15],
            }
        )

        mock_data.to_csv(test_csv, index=False)
        return test_csv

    def test_test_set_initialization(self, create_mock_test_file: Path) -> None:
        """Test that test set can be initialized."""
        test_set = GinkgoTestSet(test_file=str(create_mock_test_file))

        assert test_set.test_file.exists()

    def test_load_data_returns_dataframe(self, create_mock_test_file: Path) -> None:
        """Test that load_data returns DataFrame without labels."""
        test_set = GinkgoTestSet(test_file=str(create_mock_test_file))
        df = test_set.load_data()

        assert isinstance(df, pd.DataFrame)
        assert "antibody_id" in df.columns
        assert "antibody_name" in df.columns
        assert "vh_protein_sequence" in df.columns
        assert "vl_protein_sequence" in df.columns

        # Should NOT have target columns
        assert "PR_CHO" not in df.columns

    def test_get_sequences(self, create_mock_test_file: Path) -> None:
        """Test that get_sequences returns sequences and names."""
        test_set = GinkgoTestSet(test_file=str(create_mock_test_file))
        sequences, antibody_names = test_set.get_sequences()

        # Check types
        assert isinstance(sequences, list)
        assert isinstance(antibody_names, list)

        # Check lengths match
        assert len(sequences) == len(antibody_names)

        # Check we have 3 samples
        assert len(sequences) == 3

        # Check sequences are valid
        for seq in sequences:
            assert isinstance(seq, str)
            assert len(seq) > 0

    def test_real_test_file_if_exists(self) -> None:
        """Test loading real test file if it exists."""
        test_file = Path("test_datasets/gingko/heldout-set-sequences.csv")

        if not test_file.exists():
            pytest.skip("Real test file not downloaded yet")

        test_set = GinkgoTestSet(test_file=str(test_file))
        sequences, antibody_names = test_set.get_sequences()

        # Competition test set has 80 antibodies
        assert len(sequences) == 80
        assert len(antibody_names) == 80
