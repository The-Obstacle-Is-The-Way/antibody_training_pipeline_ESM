"""
Unit tests for Shehata dataset loader.

Tests cover:
- Dataset initialization (output dir, fragment types)
- Excel loading (required columns, valid sequences)
- PSR threshold calculation (98.24th percentile)
- Binary label assignment (PSR > threshold → label=1)
- Sequence sanitization (gap removal)
- B cell subset metadata
- Error handling (missing files, invalid PSR scores)

Testing philosophy:
- Test behaviors, not implementation
- Mock only I/O boundaries (file system when necessary)
- Use real pandas operations (no mocking domain logic)
- Follow AAA pattern (Arrange-Act-Assert)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from antibody_training_esm.datasets.shehata import ShehataDataset

# ==================== Fixtures ====================


@pytest.fixture
def shehata_sample_csv() -> Path:
    """Path to mock Shehata CSV (15 antibodies with PSR scores)."""
    return (
        Path(__file__).parent.parent.parent
        / "fixtures/mock_datasets/shehata_sample.csv"
    )


@pytest.fixture
def shehata_sample_excel(shehata_sample_csv: Path, tmp_path: Path) -> Path:
    """Convert CSV to Excel for testing Excel loading."""
    df = pd.read_csv(shehata_sample_csv)
    excel_path = tmp_path / "shehata_sample.xlsx"
    df.to_excel(excel_path, index=False)
    return excel_path


# ==================== Initialization Tests ====================


@pytest.mark.unit
def test_shehata_dataset_initializes_with_default_output_dir() -> None:
    """Verify ShehataDataset initializes with default output directory."""
    # Arrange & Act
    dataset = ShehataDataset()

    # Assert
    assert dataset.dataset_name == "shehata"
    assert dataset.output_dir == Path("data/test/shehata/fragments")


@pytest.mark.unit
def test_shehata_dataset_initializes_with_custom_output_dir(tmp_path: Path) -> None:
    """Verify ShehataDataset accepts custom output directory."""
    # Arrange
    custom_dir = tmp_path / "custom_shehata"

    # Act
    dataset = ShehataDataset(output_dir=custom_dir)

    # Assert
    assert dataset.output_dir == custom_dir
    assert custom_dir.exists()  # Base class creates directory


@pytest.mark.unit
def test_shehata_dataset_returns_full_antibody_fragments() -> None:
    """Verify ShehataDataset returns 16 full antibody fragment types."""
    # Arrange
    dataset = ShehataDataset()

    # Act
    fragment_types = dataset.get_fragment_types()

    # Assert
    assert len(fragment_types) == 16
    assert "VH_only" in fragment_types
    assert "VL_only" in fragment_types
    assert "VH+VL" in fragment_types
    assert "H-CDR3" in fragment_types


# ==================== Dataset Loading Tests ====================


@pytest.mark.unit
def test_load_data_reads_excel_successfully(shehata_sample_excel: Path) -> None:
    """Verify load_data reads Excel file and returns DataFrame."""
    # Arrange
    dataset = ShehataDataset()

    # Act
    df = dataset.load_data(excel_path=str(shehata_sample_excel))

    # Assert
    assert len(df) == 15
    assert isinstance(df, pd.DataFrame)
    assert "VH_sequence" in df.columns
    assert "VL_sequence" in df.columns
    assert "label" in df.columns
    assert "psr_score" in df.columns
    assert "b_cell_subset" in df.columns


@pytest.mark.unit
def test_load_data_requires_valid_sequences(shehata_sample_excel: Path) -> None:
    """Verify loaded data contains valid antibody sequences."""
    # Arrange
    dataset = ShehataDataset()

    # Act
    df = dataset.load_data(excel_path=str(shehata_sample_excel))

    # Assert
    assert all(df["VH_sequence"].str.len() > 0)
    assert all(df["VL_sequence"].str.len() > 0)
    # Check for valid amino acids
    valid_aa = set("ACDEFGHIKLMNPQRSTVWYX")
    for seq in df["VH_sequence"]:
        assert all(aa in valid_aa for aa in seq)


@pytest.mark.unit
def test_load_data_creates_binary_labels_from_psr(shehata_sample_excel: Path) -> None:
    """Verify labels are created from PSR threshold (PSR > threshold → label=1)."""
    # Arrange
    dataset = ShehataDataset()

    # Act
    df = dataset.load_data(excel_path=str(shehata_sample_excel), psr_threshold=1.0)

    # Assert - Verify label creation logic
    assert all(df["label"].isin([0, 1]))

    # Check specific PSR mappings with threshold=1.0
    low_psr_rows = df[df["psr_score"] <= 1.0]
    high_psr_rows = df[df["psr_score"] > 1.0]

    assert all(low_psr_rows["label"] == 0)  # PSR ≤ 1.0 → specific (label=0)
    assert all(high_psr_rows["label"] == 1)  # PSR > 1.0 → non-specific (label=1)


@pytest.mark.unit
def test_load_data_raises_error_for_missing_file() -> None:
    """Verify load_data raises FileNotFoundError for missing Excel file."""
    # Arrange
    dataset = ShehataDataset()

    # Act & Assert
    with pytest.raises(FileNotFoundError, match="Shehata Excel file not found"):
        dataset.load_data(excel_path="nonexistent_file.xlsx")


@pytest.mark.unit
def test_load_data_sanitizes_sequences_removes_gaps(
    shehata_sample_csv: Path, tmp_path: Path
) -> None:
    """Verify load_data removes IMGT gap characters from sequences."""
    # Arrange - Add gaps to sequences
    df_with_gaps = pd.read_csv(shehata_sample_csv)
    df_with_gaps["VH Protein"] = (
        df_with_gaps["VH Protein"].str[:50]
        + "---"
        + df_with_gaps["VH Protein"].str[50:]
    )
    df_with_gaps["VL Protein"] = (
        df_with_gaps["VL Protein"].str[:30] + "--" + df_with_gaps["VL Protein"].str[30:]
    )

    excel_with_gaps = tmp_path / "shehata_with_gaps.xlsx"
    df_with_gaps.to_excel(excel_with_gaps, index=False)

    dataset = ShehataDataset()

    # Act
    df = dataset.load_data(excel_path=str(excel_with_gaps))

    # Assert - Gaps should be removed
    assert not any("-" in seq for seq in df["VH_sequence"])
    assert not any("-" in seq for seq in df["VL_sequence"])


# ==================== PSR Threshold Calculation Tests ====================


@pytest.mark.unit
def test_calculate_psr_threshold_uses_default_percentile() -> None:
    """Verify calculate_psr_threshold uses 98.24th percentile by default."""
    # Arrange
    dataset = ShehataDataset()
    psr_scores = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Act
    threshold = dataset.calculate_psr_threshold(psr_scores)

    # Assert
    expected = psr_scores.quantile(0.9824)
    assert abs(threshold - expected) < 0.001  # Close enough


@pytest.mark.unit
def test_calculate_psr_threshold_accepts_custom_percentile() -> None:
    """Verify calculate_psr_threshold accepts custom percentile."""
    # Arrange
    dataset = ShehataDataset()
    psr_scores = pd.Series([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    # Act
    threshold = dataset.calculate_psr_threshold(psr_scores, percentile=0.75)

    # Assert
    expected = psr_scores.quantile(0.75)
    assert abs(threshold - expected) < 0.001


@pytest.mark.unit
def test_load_data_uses_manual_psr_threshold(shehata_sample_excel: Path) -> None:
    """Verify load_data accepts manual PSR threshold."""
    # Arrange
    dataset = ShehataDataset()
    manual_threshold = 0.5

    # Act
    df = dataset.load_data(
        excel_path=str(shehata_sample_excel), psr_threshold=manual_threshold
    )

    # Assert
    # Verify labels created with manual threshold
    low_psr = df[df["psr_score"] <= manual_threshold]
    high_psr = df[df["psr_score"] > manual_threshold]

    assert all(low_psr["label"] == 0)
    assert all(high_psr["label"] == 1)


# ==================== Label Assignment Tests ====================


@pytest.mark.unit
def test_psr_above_threshold_is_nonspecific(shehata_sample_excel: Path) -> None:
    """Verify PSR > threshold → label=1 (non-specific)."""
    # Arrange
    dataset = ShehataDataset()

    # Act
    df = dataset.load_data(excel_path=str(shehata_sample_excel), psr_threshold=1.5)

    # Assert
    high_psr_rows = df[df["psr_score"] > 1.5]
    assert all(high_psr_rows["label"] == 1)


@pytest.mark.unit
def test_psr_below_threshold_is_specific(shehata_sample_excel: Path) -> None:
    """Verify PSR ≤ threshold → label=0 (specific)."""
    # Arrange
    dataset = ShehataDataset()

    # Act
    df = dataset.load_data(excel_path=str(shehata_sample_excel), psr_threshold=1.5)

    # Assert
    low_psr_rows = df[df["psr_score"] <= 1.5]
    assert all(low_psr_rows["label"] == 0)


# ==================== B Cell Subset Metadata Tests ====================


@pytest.mark.unit
def test_load_data_includes_b_cell_subset(shehata_sample_excel: Path) -> None:
    """Verify load_data includes B cell subset metadata."""
    # Arrange
    dataset = ShehataDataset()

    # Act
    df = dataset.load_data(excel_path=str(shehata_sample_excel))

    # Assert
    assert "b_cell_subset" in df.columns
    assert df["b_cell_subset"].notna().all()
    # Check expected subset values
    expected_subsets = {"Memory", "Naive", "Plasmablast"}
    assert set(df["b_cell_subset"].unique()).issubset(expected_subsets)


# ==================== Integration Workflow Test ====================


@pytest.mark.unit
def test_complete_shehata_workflow(shehata_sample_excel: Path) -> None:
    """Verify complete Shehata dataset workflow: init → load → validate."""
    # Arrange
    dataset = ShehataDataset()

    # Act - Load data with automatic PSR threshold
    df = dataset.load_data(excel_path=str(shehata_sample_excel))

    # Assert - Verify complete workflow
    assert len(df) == 15
    assert "VH_sequence" in df.columns
    assert "VL_sequence" in df.columns
    assert "label" in df.columns
    assert "psr_score" in df.columns
    assert "b_cell_subset" in df.columns
    assert all(df["label"].isin([0, 1]))

    # Verify sequences are valid
    assert all(df["VH_sequence"].str.len() > 0)
    assert all(df["VL_sequence"].str.len() > 0)

    # Verify fragment types can be retrieved
    fragment_types = dataset.get_fragment_types()
    assert len(fragment_types) == 16
