"""
Unit tests for Boughter dataset loader.

Tests cover:
- Dataset initialization (output dir, fragment types)
- CSV loading (required columns, valid sequences)
- Novo flagging strategy (0 vs 1-3 vs 4+ flags)
- Subset filtering (flu, hiv_nat, etc.)
- Label creation from flags
- Quality filtering (X in CDRs, empty CDRs)
- Error handling (missing files, invalid subsets)

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

from antibody_training_esm.datasets.boughter import BoughterDataset

# ==================== Fixtures ====================


@pytest.fixture
def boughter_sample_csv() -> Path:
    """Path to mock Boughter CSV file (20 antibodies, balanced labels)."""
    return (
        Path(__file__).parent.parent.parent
        / "fixtures/mock_datasets/boughter_sample.csv"
    )


# ==================== Initialization Tests ====================


@pytest.mark.unit
def test_boughter_dataset_initializes_with_default_output_dir() -> None:
    """Verify BoughterDataset initializes with default output directory."""
    # Arrange & Act
    dataset = BoughterDataset()

    # Assert
    assert dataset.dataset_name == "boughter"
    assert dataset.output_dir == Path("train_datasets/boughter/annotated")


@pytest.mark.unit
def test_boughter_dataset_initializes_with_custom_output_dir(tmp_path: Path) -> None:
    """Verify BoughterDataset accepts custom output directory."""
    # Arrange
    custom_dir = tmp_path / "custom_output"

    # Act
    dataset = BoughterDataset(output_dir=custom_dir)

    # Assert
    assert dataset.output_dir == custom_dir
    assert custom_dir.exists()  # Base class creates directory


@pytest.mark.unit
def test_boughter_dataset_returns_full_antibody_fragments() -> None:
    """Verify BoughterDataset returns 16 full antibody fragment types."""
    # Arrange
    dataset = BoughterDataset()

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
def test_load_data_reads_csv_successfully(boughter_sample_csv: Path) -> None:
    """Verify load_data reads CSV file and returns DataFrame."""
    # Arrange
    dataset = BoughterDataset()

    # Act
    df = dataset.load_data(processed_csv=str(boughter_sample_csv), include_mild=True)

    # Assert
    assert len(df) == 20
    assert isinstance(df, pd.DataFrame)
    assert "VH_sequence" in df.columns
    assert "VL_sequence" in df.columns
    assert "label" in df.columns


@pytest.mark.unit
def test_load_data_requires_valid_sequences(boughter_sample_csv: Path) -> None:
    """Verify loaded data contains valid antibody sequences."""
    # Arrange
    dataset = BoughterDataset()

    # Act
    df = dataset.load_data(processed_csv=str(boughter_sample_csv), include_mild=True)

    # Assert
    assert all(df["VH_sequence"].str.len() > 0)
    assert all(df["VL_sequence"].str.len() > 0)
    # Check for valid amino acids (no gaps, no invalid characters)
    valid_aa = set("ACDEFGHIKLMNPQRSTVWYX")
    for seq in df["VH_sequence"]:
        assert all(aa in valid_aa for aa in seq)


@pytest.mark.unit
def test_load_data_creates_binary_labels_from_flags(boughter_sample_csv: Path) -> None:
    """Verify labels are created from flags (0 flags → 0, 4+ flags → 1)."""
    # Arrange
    dataset = BoughterDataset()

    # Act
    df = dataset.load_data(processed_csv=str(boughter_sample_csv), include_mild=True)

    # Assert - Verify label creation logic
    assert all(df["label"].isin([0, 1]))

    # Check specific flag mappings
    flag_0_rows = df[df["flags"] == 0]
    assert all(flag_0_rows["label"] == 0)  # 0 flags → specific (label=0)

    flag_4plus_rows = df[df["flags"] >= 4]
    assert all(flag_4plus_rows["label"] == 1)  # 4+ flags → non-specific (label=1)


@pytest.mark.unit
def test_load_data_raises_error_for_missing_file() -> None:
    """Verify load_data raises FileNotFoundError for missing CSV."""
    # Arrange
    dataset = BoughterDataset()

    # Act & Assert
    with pytest.raises(FileNotFoundError, match="Boughter processed CSV not found"):
        dataset.load_data(processed_csv="nonexistent_file.csv")


@pytest.mark.unit
def test_load_data_renames_columns_correctly(
    boughter_sample_csv: Path, tmp_path: Path
) -> None:
    """Verify heavy_seq/light_seq columns are renamed to VH_sequence/VL_sequence."""
    # Arrange - Create CSV with old column names
    dataset = BoughterDataset()
    df_original = pd.read_csv(boughter_sample_csv)
    df_old_names = df_original.rename(
        columns={"VH_sequence": "heavy_seq", "VL_sequence": "light_seq"}
    )

    temp_csv = tmp_path / "boughter_old_names.csv"
    df_old_names.to_csv(temp_csv, index=False)

    # Act
    df = dataset.load_data(processed_csv=str(temp_csv), include_mild=True)

    # Assert
    assert "VH_sequence" in df.columns
    assert "VL_sequence" in df.columns
    assert "heavy_seq" not in df.columns
    assert "light_seq" not in df.columns

    # tmp_path auto-cleanup by pytest


# ==================== Flag Filtering Tests ====================


@pytest.mark.unit
def test_load_data_excludes_mild_flags_by_default(boughter_sample_csv: Path) -> None:
    """Verify include_mild=False excludes flags 1-3 (Novo methodology)."""
    # Arrange
    dataset = BoughterDataset()

    # Act
    df_no_mild = dataset.load_data(
        processed_csv=str(boughter_sample_csv), include_mild=False
    )
    df_with_mild = dataset.load_data(
        processed_csv=str(boughter_sample_csv), include_mild=True
    )

    # Assert - Mild flags should be excluded
    assert len(df_no_mild) < len(df_with_mild)
    assert not df_no_mild["flags"].isin([1, 2, 3]).any()

    # Verify flags 0 and 4+ are kept
    assert df_no_mild["flags"].isin([0, 4, 5, 6, 7]).all()


@pytest.mark.unit
def test_load_data_includes_mild_flags_when_requested(
    boughter_sample_csv: Path,
) -> None:
    """Verify include_mild=True keeps all flags including 1-3."""
    # Arrange
    dataset = BoughterDataset()

    # Act
    df = dataset.load_data(processed_csv=str(boughter_sample_csv), include_mild=True)

    # Assert - All flags should be present, including mild flags (1-3)
    assert df["flags"].isin([0, 1, 2, 3, 4, 5, 6]).all()  # All flags are valid
    assert df["flags"].isin([1, 2, 3]).any()  # At least one mild flag present
    assert len(df) == 20  # All 20 sequences kept


@pytest.mark.unit
def test_flag_0_is_specific() -> None:
    """Verify FLAG_SPECIFIC constant is 0."""
    # Arrange & Act
    dataset = BoughterDataset()

    # Assert
    assert dataset.FLAG_SPECIFIC == 0


@pytest.mark.unit
def test_flag_mild_is_1_to_3() -> None:
    """Verify FLAG_MILD constant is [1, 2, 3]."""
    # Arrange & Act
    dataset = BoughterDataset()

    # Assert
    assert dataset.FLAG_MILD == [1, 2, 3]


@pytest.mark.unit
def test_flag_nonspecific_is_4_plus() -> None:
    """Verify FLAG_NONSPECIFIC constant is [4, 5, 6, 7]."""
    # Arrange & Act
    dataset = BoughterDataset()

    # Assert
    assert dataset.FLAG_NONSPECIFIC == [4, 5, 6, 7]


# ==================== Subset Filtering Tests ====================


@pytest.mark.unit
def test_load_data_filters_by_valid_subset(
    boughter_sample_csv: Path, tmp_path: Path
) -> None:
    """Verify subset parameter filters data when valid subset provided."""
    # Arrange - Add subset column to mock data
    dataset = BoughterDataset()
    df_original = pd.read_csv(boughter_sample_csv)

    # Add subset column (alternate flu/hiv_nat)
    df_original["subset"] = [
        "flu" if i % 2 == 0 else "hiv_nat" for i in range(len(df_original))
    ]
    temp_csv = tmp_path / "boughter_with_subset.csv"
    df_original.to_csv(temp_csv, index=False)

    # Act
    df_flu = dataset.load_data(
        processed_csv=str(temp_csv), subset="flu", include_mild=True
    )
    df_hiv = dataset.load_data(
        processed_csv=str(temp_csv), subset="hiv_nat", include_mild=True
    )

    # Assert
    assert len(df_flu) < len(df_original)
    assert len(df_hiv) < len(df_original)
    assert len(df_flu) + len(df_hiv) == len(df_original)

    # tmp_path auto-cleanup by pytest


@pytest.mark.unit
def test_load_data_raises_error_for_invalid_subset(boughter_sample_csv: Path) -> None:
    """Verify load_data raises ValueError for invalid subset name."""
    # Arrange
    dataset = BoughterDataset()

    # Act & Assert
    with pytest.raises(ValueError, match="Unknown subset"):
        dataset.load_data(
            processed_csv=str(boughter_sample_csv), subset="invalid_subset"
        )


@pytest.mark.unit
def test_load_data_returns_all_data_when_no_subset(boughter_sample_csv: Path) -> None:
    """Verify load_data returns all data when subset=None."""
    # Arrange
    dataset = BoughterDataset()

    # Act
    df = dataset.load_data(
        processed_csv=str(boughter_sample_csv), subset=None, include_mild=True
    )

    # Assert
    assert len(df) == 20  # All sequences


# ==================== translate_dna_to_protein Tests ====================


@pytest.mark.unit
def test_translate_dna_to_protein_raises_not_implemented() -> None:
    """Verify translate_dna_to_protein always raises NotImplementedError."""
    # Arrange
    dataset = BoughterDataset()

    # Act & Assert
    with pytest.raises(NotImplementedError, match="DNA translation is not implemented"):
        dataset.translate_dna_to_protein("ATGGCTAGC")


# ==================== filter_quality_issues Tests ====================


@pytest.mark.unit
def test_filter_quality_issues_removes_x_in_cdrs() -> None:
    """Verify filter_quality_issues removes sequences with X in CDRs."""
    # Arrange
    dataset = BoughterDataset()
    df = pd.DataFrame(
        {
            "id": ["AB001", "AB002", "AB003"],
            "VH_CDR1": ["GFTFS", "GFTFX", "GFTFS"],  # AB002 has X
            "VL_CDR1": ["RASQS", "RASQS", "RASQS"],
            "VH_sequence": ["QVQLVQSG"] * 3,
            "VL_sequence": ["DIQMTQSP"] * 3,
        }
    )

    # Act
    df_filtered = dataset.filter_quality_issues(df)

    # Assert
    assert len(df_filtered) == 2
    assert "AB002" not in df_filtered["id"].values


@pytest.mark.unit
def test_filter_quality_issues_removes_empty_cdrs() -> None:
    """Verify filter_quality_issues removes sequences with empty CDRs."""
    # Arrange
    dataset = BoughterDataset()
    df = pd.DataFrame(
        {
            "id": ["AB001", "AB002", "AB003"],
            "VH_CDR1": ["GFTFS", "", "GFTFS"],  # AB002 has empty CDR
            "VL_CDR1": ["RASQS", "RASQS", "RASQS"],
            "VH_sequence": ["QVQLVQSG"] * 3,
            "VL_sequence": ["DIQMTQSP"] * 3,
        }
    )

    # Act
    df_filtered = dataset.filter_quality_issues(df)

    # Assert
    assert len(df_filtered) == 2
    assert "AB002" not in df_filtered["id"].values


@pytest.mark.unit
def test_filter_quality_issues_returns_dataframe() -> None:
    """Verify filter_quality_issues returns a DataFrame."""
    # Arrange
    dataset = BoughterDataset()
    df = pd.DataFrame(
        {
            "id": ["AB001", "AB002"],
            "VH_CDR1": ["GFTFS", "GFTFS"],
            "VL_CDR1": ["RASQS", "RASQS"],
            "VH_sequence": ["QVQLVQSG"] * 2,
            "VL_sequence": ["DIQMTQSP"] * 2,
        }
    )

    # Act
    df_filtered = dataset.filter_quality_issues(df)

    # Assert
    assert isinstance(df_filtered, pd.DataFrame)
    assert len(df_filtered) == 2  # No filtering needed


# ==================== Integration Workflow Test ====================


@pytest.mark.unit
def test_complete_boughter_workflow(boughter_sample_csv: Path) -> None:
    """Verify complete Boughter dataset workflow: init → load → validate."""
    # Arrange
    dataset = BoughterDataset()

    # Act - Load data with Novo flagging
    df = dataset.load_data(processed_csv=str(boughter_sample_csv), include_mild=False)

    # Assert - Verify complete workflow
    assert len(df) > 0
    assert "VH_sequence" in df.columns
    assert "VL_sequence" in df.columns
    assert "label" in df.columns
    assert all(df["label"].isin([0, 1]))
    assert not df["flags"].isin([1, 2, 3]).any()  # Mild flags excluded

    # Verify fragment types can be retrieved
    fragment_types = dataset.get_fragment_types()
    assert len(fragment_types) == 16
