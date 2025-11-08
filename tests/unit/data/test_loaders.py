"""
Unit tests for data loading utilities.

Tests cover:
- CSV loading (load_local_data)
- Pickle save/load (store_preprocessed_data, load_preprocessed_data)
- Embedding preprocessing (preprocess_raw_data)
- HuggingFace dataset loading (load_hf_dataset)
- Config-based data loading (load_data)
- Error handling (missing files, malformed CSVs, invalid configs)

Testing philosophy:
- Test behaviors, not implementation
- Mock external dependencies (HuggingFace, file I/O)
- Use tmp_path for file operations
- Follow AAA pattern (Arrange-Act-Assert)

Date: 2025-11-07
Phase: 5 (Coverage Gap Closure)
"""

import pickle
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from antibody_training_esm.data.loaders import (
    load_data,
    load_hf_dataset,
    load_local_data,
    load_preprocessed_data,
    preprocess_raw_data,
    store_preprocessed_data,
)

# ==================== Fixtures ====================


@pytest.fixture
def mock_embedding_extractor():
    """Mock embedding extractor with batch method"""
    extractor = Mock()
    extractor.extract_batch_embeddings = Mock(
        return_value=np.random.rand(5, 1280).astype(np.float32)
    )
    extractor.embed_sequence = Mock(
        return_value=np.random.rand(1280).astype(np.float32)
    )
    return extractor


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing"""
    return pd.DataFrame(
        {
            "sequence": ["ACDEFGHIKLMNPQRSTVWY"] * 5,
            "label": [0, 1, 0, 1, 0],
            "id": [f"AB{i:03d}" for i in range(5)],
        }
    )


# ==================== load_local_data Tests ====================


@pytest.mark.unit
def test_load_local_data_reads_csv(tmp_path, sample_csv_data):
    """Verify load_local_data reads CSV correctly"""
    # Arrange
    csv_path = tmp_path / "test.csv"
    sample_csv_data.to_csv(csv_path, index=False)

    # Act
    X, y = load_local_data(csv_path, "sequence", "label")

    # Assert
    assert len(X) == 5
    assert len(y) == 5
    assert X[0] == "ACDEFGHIKLMNPQRSTVWY"
    assert y == [0, 1, 0, 1, 0]


@pytest.mark.unit
def test_load_local_data_handles_comments(tmp_path):
    """Verify load_local_data handles comment lines in CSV"""
    # Arrange
    csv_path = tmp_path / "test_comments.csv"
    csv_path.write_text(
        "# This is a comment\nsequence,label\nACDEF,0\nGHIKL,1\n# Another comment\n"
    )

    # Act
    X, y = load_local_data(csv_path, "sequence", "label")

    # Assert
    assert len(X) == 2
    assert X == ["ACDEF", "GHIKL"]
    assert y == [0, 1]


@pytest.mark.unit
def test_load_local_data_raises_on_missing_file():
    """Verify load_local_data raises error for missing file"""
    # Act & Assert
    with pytest.raises(FileNotFoundError):
        load_local_data("nonexistent.csv", "sequence", "label")


@pytest.mark.unit
def test_load_local_data_raises_on_missing_column(tmp_path, sample_csv_data):
    """Verify load_local_data raises error for missing column"""
    # Arrange
    csv_path = tmp_path / "test.csv"
    sample_csv_data.to_csv(csv_path, index=False)

    # Act & Assert
    with pytest.raises(KeyError):
        load_local_data(csv_path, "nonexistent_column", "label")


# ==================== preprocess_raw_data Tests ====================


@pytest.mark.unit
def test_preprocess_raw_data_uses_batch_method(mock_embedding_extractor):
    """Verify preprocess_raw_data uses batch embedding if available"""
    # Arrange
    X = ["ACDEF", "GHIKL", "MNPQR"]
    y = [0, 1, 0]

    # Act
    X_embedded, y_out = preprocess_raw_data(X, y, mock_embedding_extractor)

    # Assert
    mock_embedding_extractor.extract_batch_embeddings.assert_called_once_with(X)
    assert X_embedded.shape == (5, 1280)  # Mock returns shape (5, 1280)
    np.testing.assert_array_equal(y_out, np.array([0, 1, 0]))


@pytest.mark.unit
def test_preprocess_raw_data_falls_back_to_single_embedding():
    """Verify preprocess_raw_data falls back to embed_sequence"""
    # Arrange
    extractor = Mock(spec=["embed_sequence"])  # Only has embed_sequence method
    extractor.embed_sequence = Mock(
        return_value=np.random.rand(1280).astype(np.float32)
    )
    X = ["ACDEF", "GHIKL"]
    y = [0, 1]

    # Act
    X_embedded, y_out = preprocess_raw_data(X, y, extractor)

    # Assert
    assert extractor.embed_sequence.call_count == 2
    assert X_embedded.shape == (2, 1280)
    np.testing.assert_array_equal(y_out, np.array([0, 1]))


@pytest.mark.unit
def test_preprocess_raw_data_converts_labels_to_numpy(mock_embedding_extractor):
    """Verify preprocess_raw_data converts labels to numpy array"""
    # Arrange
    X = ["ACDEF"]
    y = [0]  # List input

    # Act
    _, y_out = preprocess_raw_data(X, y, mock_embedding_extractor)

    # Assert
    assert isinstance(y_out, np.ndarray)
    assert y_out.dtype == np.int64


# ==================== store_preprocessed_data Tests ====================


@pytest.mark.unit
def test_store_preprocessed_data_saves_all_data(tmp_path):
    """Verify store_preprocessed_data saves all provided data"""
    # Arrange
    X = ["ACDEF", "GHIKL"]
    y = [0, 1]
    X_embedded = np.random.rand(2, 1280)
    filepath = tmp_path / "test.pkl"

    # Act
    store_preprocessed_data(X=X, y=y, X_embedded=X_embedded, filename=str(filepath))

    # Assert
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    assert "X" in data
    assert "y" in data
    assert "X_embedded" in data
    assert data["X"] == X
    assert data["y"] == y
    np.testing.assert_array_equal(data["X_embedded"], X_embedded)


@pytest.mark.unit
def test_store_preprocessed_data_saves_embeddings_only(tmp_path):
    """Verify store_preprocessed_data can save embeddings only"""
    # Arrange
    X_embedded = np.random.rand(3, 1280)
    filepath = tmp_path / "embeddings.pkl"

    # Act
    store_preprocessed_data(X_embedded=X_embedded, filename=str(filepath))

    # Assert
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    assert "X_embedded" in data
    assert "X" not in data
    assert "y" not in data


@pytest.mark.unit
def test_store_preprocessed_data_requires_filename():
    """Verify store_preprocessed_data raises error if filename missing"""
    # Act & Assert
    with pytest.raises(ValueError, match="filename is required"):
        store_preprocessed_data(X=["ACDEF"])


# ==================== load_preprocessed_data Tests ====================


@pytest.mark.unit
def test_load_preprocessed_data_loads_pickle(tmp_path):
    """Verify load_preprocessed_data loads pickle correctly"""
    # Arrange
    data = {"X": ["ACDEF"], "y": [0], "X_embedded": np.random.rand(1, 1280)}
    filepath = tmp_path / "test.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

    # Act
    loaded = load_preprocessed_data(str(filepath))

    # Assert
    assert "X" in loaded
    assert "y" in loaded
    assert "X_embedded" in loaded
    assert loaded["X"] == ["ACDEF"]


@pytest.mark.unit
def test_load_preprocessed_data_raises_on_missing_file():
    """Verify load_preprocessed_data raises error for missing file"""
    # Act & Assert
    with pytest.raises(FileNotFoundError):
        load_preprocessed_data("nonexistent.pkl")


# ==================== load_hf_dataset Tests ====================


@pytest.mark.unit
@patch("antibody_training_esm.data.loaders.load_dataset")
def test_load_hf_dataset_calls_datasets_library(mock_load_dataset):
    """Verify load_hf_dataset calls HuggingFace datasets library"""
    # Arrange
    mock_dataset = {
        "sequence": ["ACDEF", "GHIKL"],
        "label": [0, 1],
    }
    mock_load_dataset.return_value = mock_dataset

    # Act
    X, y = load_hf_dataset("test_dataset", "train", "sequence", "label")

    # Assert
    mock_load_dataset.assert_called_once_with(
        "test_dataset", split="train", revision="main"
    )
    assert X == ["ACDEF", "GHIKL"]
    assert y == [0, 1]


@pytest.mark.unit
@patch("antibody_training_esm.data.loaders.load_dataset")
def test_load_hf_dataset_returns_lists(mock_load_dataset):
    """Verify load_hf_dataset returns lists, not numpy arrays"""
    # Arrange
    mock_dataset = {
        "text": ["A", "B", "C"],
        "label": [0, 1, 0],
    }
    mock_load_dataset.return_value = mock_dataset

    # Act
    X, y = load_hf_dataset("dataset", "train", "text", "label")

    # Assert
    assert isinstance(X, list)
    assert isinstance(y, list)


# ==================== load_data Tests ====================


@pytest.mark.unit
@patch("antibody_training_esm.data.loaders.load_hf_dataset")
def test_load_data_dispatches_to_hf(mock_load_hf):
    """Verify load_data dispatches to load_hf_dataset"""
    # Arrange
    config = {
        "data": {
            "source": "hf",
            "dataset_name": "test_dataset",
            "train_split": "train",
            "sequence_column": "seq",
            "label_column": "label",
        }
    }
    mock_load_hf.return_value = (["ACDEF"], [0])

    # Act
    X, y = load_data(config)

    # Assert
    mock_load_hf.assert_called_once_with(
        dataset_name="test_dataset",
        split="train",
        text_column="seq",
        label_column="label",
    )
    assert X == ["ACDEF"]
    assert y == [0]


@pytest.mark.unit
@patch("antibody_training_esm.data.loaders.load_local_data")
def test_load_data_dispatches_to_local(mock_load_local):
    """Verify load_data dispatches to load_local_data"""
    # Arrange
    config = {
        "data": {
            "source": "local",
            "train_file": "train.csv",
            "sequence_column": "seq",
            "label_column": "label",
        }
    }
    mock_load_local.return_value = (["GHIKL"], [1])

    # Act
    X, y = load_data(config)

    # Assert
    mock_load_local.assert_called_once_with(
        "train.csv", text_column="seq", label_column="label"
    )
    assert X == ["GHIKL"]
    assert y == [1]


@pytest.mark.unit
def test_load_data_raises_on_unknown_source():
    """Verify load_data raises error for unknown source"""
    # Arrange
    config = {"data": {"source": "unknown"}}

    # Act & Assert
    with pytest.raises(ValueError, match="Unknown data source: unknown"):
        load_data(config)


# ==================== Integration Tests ====================


@pytest.mark.unit
def test_end_to_end_csv_to_embeddings(tmp_path, mock_embedding_extractor):
    """Verify end-to-end workflow: CSV → load → preprocess → save → load"""
    # Arrange
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({"sequence": ["ACDEF", "GHIKL"], "label": [0, 1]}).to_csv(
        csv_path, index=False
    )
    pkl_path = tmp_path / "embeddings.pkl"

    # Act: Load CSV
    X, y = load_local_data(csv_path, "sequence", "label")

    # Act: Preprocess
    X_embedded, y_array = preprocess_raw_data(X, y, mock_embedding_extractor)

    # Act: Save
    store_preprocessed_data(X=X, y=y, X_embedded=X_embedded, filename=str(pkl_path))

    # Act: Load
    loaded = load_preprocessed_data(str(pkl_path))

    # Assert
    assert loaded["X"] == ["ACDEF", "GHIKL"]
    assert loaded["y"] == [0, 1]
    assert "X_embedded" in loaded


@pytest.mark.unit
def test_load_local_data_handles_empty_csv(tmp_path):
    """Verify load_local_data handles empty CSV gracefully"""
    # Arrange
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("sequence,label\n")

    # Act
    X, y = load_local_data(csv_path, "sequence", "label")

    # Assert
    assert X == []
    assert y == []


@pytest.mark.unit
def test_preprocess_raw_data_handles_single_sequence(mock_embedding_extractor):
    """Verify preprocess_raw_data handles single sequence"""
    # Arrange
    X = ["ACDEF"]
    y = [0]

    # Act
    X_embedded, y_out = preprocess_raw_data(X, y, mock_embedding_extractor)

    # Assert
    assert X_embedded.shape[0] == 5  # Mock returns (5, 1280)
    assert len(y_out) == 1
