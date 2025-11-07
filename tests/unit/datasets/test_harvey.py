"""
Unit tests for Harvey nanobody dataset loader.

Tests cover:
- Dataset initialization (output dir, nanobody fragment types)
- CSV loading from high/low polyreactivity files
- IMGT position extraction (removes gaps, handles missing values)
- Binary label assignment (high=1, low=0)
- Nanobody-specific behavior (VHH only, 6 fragments)
- Error handling (missing files, empty sequences)

Testing philosophy:
- Test behaviors, not implementation
- Mock only I/O boundaries (file system when necessary)
- Use real pandas operations (no mocking domain logic)
- Follow AAA pattern (Arrange-Act-Assert)
"""

from pathlib import Path

import pandas as pd
import pytest

from antibody_training_esm.datasets.harvey import HarveyDataset

# ==================== Fixtures ====================


@pytest.fixture
def harvey_high_csv():
    """Path to mock Harvey high polyreactivity CSV (5 sequences)."""
    return (
        Path(__file__).parent.parent.parent
        / "fixtures/mock_datasets/harvey_high_sample.csv"
    )


@pytest.fixture
def harvey_low_csv():
    """Path to mock Harvey low polyreactivity CSV (7 sequences)."""
    return (
        Path(__file__).parent.parent.parent
        / "fixtures/mock_datasets/harvey_low_sample.csv"
    )


# ==================== Initialization Tests ====================


@pytest.mark.unit
def test_harvey_dataset_initializes_with_default_output_dir():
    """Verify HarveyDataset initializes with default output directory."""
    # Arrange & Act
    dataset = HarveyDataset()

    # Assert
    assert dataset.dataset_name == "harvey"
    assert dataset.output_dir == Path("train_datasets/harvey/fragments")


@pytest.mark.unit
def test_harvey_dataset_initializes_with_custom_output_dir(tmp_path):
    """Verify HarveyDataset accepts custom output directory."""
    # Arrange
    custom_dir = tmp_path / "custom_harvey"

    # Act
    dataset = HarveyDataset(output_dir=custom_dir)

    # Assert
    assert dataset.output_dir == custom_dir
    assert custom_dir.exists()  # Base class creates directory


@pytest.mark.unit
def test_harvey_dataset_returns_nanobody_fragments():
    """Verify HarveyDataset returns 6 nanobody fragment types (VHH only)."""
    # Arrange
    dataset = HarveyDataset()

    # Act
    fragment_types = dataset.get_fragment_types()

    # Assert
    assert len(fragment_types) == 6  # Nanobodies have 6 fragments, not 16
    assert "VHH_only" in fragment_types
    assert "H-CDR3" in fragment_types
    # Should NOT have light chain fragments
    assert "VL_only" not in fragment_types
    assert "L-CDR1" not in fragment_types


# ==================== Dataset Loading Tests ====================


@pytest.mark.unit
def test_load_data_reads_both_csv_files(harvey_high_csv, harvey_low_csv):
    """Verify load_data reads and combines high/low polyreactivity CSVs."""
    # Arrange
    dataset = HarveyDataset()

    # Act
    df = dataset.load_data(
        high_csv_path=str(harvey_high_csv), low_csv_path=str(harvey_low_csv)
    )

    # Assert
    assert len(df) == 12  # 5 high + 7 low
    assert isinstance(df, pd.DataFrame)
    assert "VH_sequence" in df.columns
    assert "label" in df.columns
    assert "id" in df.columns


@pytest.mark.unit
def test_load_data_assigns_correct_labels(harvey_high_csv, harvey_low_csv):
    """Verify high polyreactivity → label=1, low polyreactivity → label=0."""
    # Arrange
    dataset = HarveyDataset()

    # Act
    df = dataset.load_data(
        high_csv_path=str(harvey_high_csv), low_csv_path=str(harvey_low_csv)
    )

    # Assert
    assert all(df["label"].isin([0, 1]))
    # First 5 sequences are from high_csv (label=1)
    # Last 7 sequences are from low_csv (label=0)
    assert (df["label"] == 1).sum() == 5
    assert (df["label"] == 0).sum() == 7


@pytest.mark.unit
def test_load_data_extracts_valid_sequences(harvey_high_csv, harvey_low_csv):
    """Verify IMGT extraction produces valid antibody sequences."""
    # Arrange
    dataset = HarveyDataset()

    # Act
    df = dataset.load_data(
        high_csv_path=str(harvey_high_csv), low_csv_path=str(harvey_low_csv)
    )

    # Assert
    assert all(df["VH_sequence"].str.len() > 0)
    # Check for valid amino acids (no gaps)
    valid_aa = set("ACDEFGHIKLMNPQRSTVWYX")
    for seq in df["VH_sequence"]:
        assert all(aa in valid_aa for aa in seq)
        assert "-" not in seq  # Gaps should be removed


@pytest.mark.unit
def test_load_data_raises_error_for_missing_high_csv():
    """Verify load_data raises FileNotFoundError for missing high CSV."""
    # Arrange
    dataset = HarveyDataset()

    # Act & Assert
    with pytest.raises(FileNotFoundError, match="High polyreactivity CSV not found"):
        dataset.load_data(
            high_csv_path="nonexistent_high.csv", low_csv_path="dummy.csv"
        )


@pytest.mark.unit
def test_load_data_raises_error_for_missing_low_csv(harvey_high_csv):
    """Verify load_data raises FileNotFoundError for missing low CSV."""
    # Arrange
    dataset = HarveyDataset()

    # Act & Assert
    with pytest.raises(FileNotFoundError, match="Low polyreactivity CSV not found"):
        dataset.load_data(
            high_csv_path=str(harvey_high_csv), low_csv_path="nonexistent_low.csv"
        )


@pytest.mark.unit
def test_load_data_creates_sequence_ids(harvey_high_csv, harvey_low_csv):
    """Verify load_data creates unique sequence IDs."""
    # Arrange
    dataset = HarveyDataset()

    # Act
    df = dataset.load_data(
        high_csv_path=str(harvey_high_csv), low_csv_path=str(harvey_low_csv)
    )

    # Assert
    assert "id" in df.columns
    assert len(df["id"].unique()) == len(df)  # All IDs are unique
    assert all(df["id"].str.startswith("harvey_"))


# ==================== IMGT Extraction Tests ====================


@pytest.mark.unit
def test_extract_sequence_from_imgt_removes_gaps():
    """Verify IMGT extraction removes gap characters (-)."""
    # Arrange
    dataset = HarveyDataset()
    row = pd.Series({str(i): "-" if i % 3 == 0 else "A" for i in range(1, 129)})
    imgt_cols = [str(i) for i in range(1, 129)]

    # Act
    sequence = dataset.extract_sequence_from_imgt(row, imgt_cols)

    # Assert
    assert "-" not in sequence
    assert len(sequence) > 0
    assert all(aa == "A" for aa in sequence)


@pytest.mark.unit
def test_extract_sequence_from_imgt_handles_missing_values():
    """Verify IMGT extraction handles NaN/missing values."""
    # Arrange
    dataset = HarveyDataset()
    row = pd.Series({str(i): pd.NA if i % 2 == 0 else "M" for i in range(1, 129)})
    imgt_cols = [str(i) for i in range(1, 129)]

    # Act
    sequence = dataset.extract_sequence_from_imgt(row, imgt_cols)

    # Assert
    assert len(sequence) > 0
    assert all(aa == "M" for aa in sequence)


@pytest.mark.unit
def test_extract_sequence_from_imgt_concatenates_correctly():
    """Verify IMGT extraction concatenates positions in order."""
    # Arrange
    dataset = HarveyDataset()
    # Create sequence "QVQLVQ" spread across positions 1-6
    row = pd.Series(
        {str(i): aa for i, aa in enumerate(["Q", "V", "Q", "L", "V", "Q"], start=1)}
    )
    imgt_cols = [str(i) for i in range(1, 7)]

    # Act
    sequence = dataset.extract_sequence_from_imgt(row, imgt_cols)

    # Assert
    assert sequence == "QVQLVQ"


# ==================== Nanobody-Specific Tests ====================


@pytest.mark.unit
def test_harvey_dataset_has_no_light_chain(harvey_high_csv, harvey_low_csv):
    """Verify Harvey dataset contains VH_sequence only (no VL_sequence)."""
    # Arrange
    dataset = HarveyDataset()

    # Act
    df = dataset.load_data(
        high_csv_path=str(harvey_high_csv), low_csv_path=str(harvey_low_csv)
    )

    # Assert
    assert "VH_sequence" in df.columns
    assert "VL_sequence" not in df.columns  # Nanobodies have no light chain


@pytest.mark.unit
def test_harvey_dataset_uses_nanobody_fragment_constant():
    """Verify Harvey uses NANOBODY_FRAGMENTS constant."""
    # Arrange
    dataset = HarveyDataset()

    # Act
    fragment_types = dataset.get_fragment_types()

    # Assert
    assert fragment_types == dataset.NANOBODY_FRAGMENTS
    assert len(fragment_types) == 6


# ==================== Integration Workflow Test ====================


@pytest.mark.unit
def test_complete_harvey_workflow(harvey_high_csv, harvey_low_csv):
    """Verify complete Harvey dataset workflow: init → load → validate."""
    # Arrange
    dataset = HarveyDataset()

    # Act - Load data from both high and low CSVs
    df = dataset.load_data(
        high_csv_path=str(harvey_high_csv), low_csv_path=str(harvey_low_csv)
    )

    # Assert - Verify complete workflow
    assert len(df) == 12
    assert "VH_sequence" in df.columns
    assert "label" in df.columns
    assert all(df["label"].isin([0, 1]))

    # Verify sequences are valid
    assert all(df["VH_sequence"].str.len() > 0)
    assert not any("-" in seq for seq in df["VH_sequence"])

    # Verify fragment types can be retrieved
    fragment_types = dataset.get_fragment_types()
    assert len(fragment_types) == 6
