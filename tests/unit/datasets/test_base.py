#!/usr/bin/env python3
"""
Unit Tests for AntibodyDataset Base Class

Tests the abstract base class functionality including:
- Initialization and configuration
- Sequence sanitization and validation
- Fragment type contracts
- Common utility methods

Philosophy:
- Test behaviors, not implementation details
- Use concrete subclass to test abstract base class
- Mock only I/O boundaries (ANARCI calls, file system)
- Focus on edge cases and error handling

Date: 2025-11-07
Author: Claude Code
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from antibody_training_esm.datasets.base import AntibodyDataset

# ============================================================================
# Test Fixtures - Concrete Dataset Implementation
# ============================================================================


class ConcreteDataset(AntibodyDataset):
    """
    Concrete implementation of AntibodyDataset for testing.

    Implements abstract methods with minimal behavior to test base class logic.
    """

    def load_data(self, **kwargs: Any) -> pd.DataFrame:
        """Return a simple test DataFrame"""
        return pd.DataFrame(
            {
                "id": ["AB001", "AB002", "AB003"],
                "VH_sequence": [
                    "QVQLVQSGAEVKKPGA",
                    "EVQLVESGGGLVQPGG",
                    "QVQLQQWGAGLLKPSE",
                ],
                "VL_sequence": [
                    "DIQMTQSPSSLSASVG",
                    "QSALTQPASVSGSPGQ",
                    "DIQMTQSPSSLSASVG",
                ],
                "label": [0, 1, 0],
            }
        )

    def get_fragment_types(self) -> list[str]:
        """Return standard full antibody fragments"""
        return cast(list[str], self.FULL_ANTIBODY_FRAGMENTS)


# ============================================================================
# Initialization Tests
# ============================================================================


@pytest.mark.unit
def test_dataset_initializes_with_name_and_default_output(tmp_path: Path) -> None:
    """Verify dataset initializes with dataset name and creates default output directory"""
    # Arrange & Act
    dataset = ConcreteDataset(dataset_name="test_dataset")

    # Assert
    assert dataset.dataset_name == "test_dataset"
    assert dataset.output_dir == Path("outputs/test_dataset")
    assert isinstance(dataset.logger, logging.Logger)


@pytest.mark.unit
def test_dataset_initializes_with_custom_output_dir(tmp_path: Path) -> None:
    """Verify dataset can use custom output directory"""
    # Arrange
    custom_dir = tmp_path / "custom_output"

    # Act
    dataset = ConcreteDataset(dataset_name="test_dataset", output_dir=custom_dir)

    # Assert
    assert dataset.output_dir == custom_dir
    assert custom_dir.exists()  # Should be created


@pytest.mark.unit
def test_dataset_creates_output_directory_on_init(tmp_path: Path) -> None:
    """Verify output directory is created if it doesn't exist"""
    # Arrange
    output_dir = tmp_path / "new_directory"
    assert not output_dir.exists()  # Verify it doesn't exist yet

    # Act
    _ = ConcreteDataset(dataset_name="test_dataset", output_dir=output_dir)

    # Assert
    assert output_dir.exists()


@pytest.mark.unit
def test_dataset_accepts_custom_logger() -> None:
    """Verify dataset can use custom logger"""
    # Arrange
    custom_logger = logging.getLogger("custom_test_logger")

    # Act
    dataset = ConcreteDataset(dataset_name="test_dataset", logger=custom_logger)

    # Assert
    assert dataset.logger == custom_logger


@pytest.mark.unit
def test_dataset_creates_default_logger_if_none_provided() -> None:
    """Verify dataset creates a default logger when none is provided"""
    # Arrange & Act
    dataset = ConcreteDataset(dataset_name="test_dataset")

    # Assert
    assert dataset.logger is not None
    assert "test_dataset" in dataset.logger.name


# ============================================================================
# Abstract Method Contract Tests
# ============================================================================


@pytest.mark.unit
def test_concrete_dataset_implements_load_data() -> None:
    """Verify concrete dataset implements load_data abstract method"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")

    # Act
    df = dataset.load_data()

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert "id" in df.columns
    assert "VH_sequence" in df.columns
    assert "VL_sequence" in df.columns
    assert "label" in df.columns


@pytest.mark.unit
def test_concrete_dataset_implements_get_fragment_types() -> None:
    """Verify concrete dataset implements get_fragment_types abstract method"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")

    # Act
    fragment_types = dataset.get_fragment_types()

    # Assert
    assert isinstance(fragment_types, list)
    assert len(fragment_types) == 16  # Full antibody has 16 fragment types
    assert "VH_only" in fragment_types
    assert "VL_only" in fragment_types
    assert "VH+VL" in fragment_types


# ============================================================================
# Sequence Sanitization Tests
# ============================================================================


@pytest.mark.unit
def test_sanitize_sequence_removes_gaps() -> None:
    """Verify sanitize_sequence removes gap characters"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    sequence_with_gaps = "QVQL-VQSG-AEVKKPGA"

    # Act
    sanitized = dataset.sanitize_sequence(sequence_with_gaps)

    # Assert
    assert "-" not in sanitized
    assert sanitized == "QVQLVQSGAEVKKPGA"


@pytest.mark.unit
def test_sanitize_sequence_removes_whitespace() -> None:
    """Verify sanitize_sequence removes whitespace"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    sequence_with_spaces = "QVQL VQSG AEVKKPGA"

    # Act
    sanitized = dataset.sanitize_sequence(sequence_with_spaces)

    # Assert
    assert " " not in sanitized
    assert sanitized == "QVQLVQSGAEVKKPGA"


@pytest.mark.unit
def test_sanitize_sequence_converts_to_uppercase() -> None:
    """Verify sanitize_sequence converts lowercase to uppercase"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    lowercase_sequence = "qvqlvqsgaevkkpga"

    # Act
    sanitized = dataset.sanitize_sequence(lowercase_sequence)

    # Assert
    assert sanitized == "QVQLVQSGAEVKKPGA"
    assert sanitized.isupper()


@pytest.mark.unit
def test_sanitize_sequence_rejects_invalid_amino_acids() -> None:
    """Verify sanitize_sequence raises ValueError for invalid amino acids"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    invalid_sequence = "QVQLVQSGBBB"  # 'B' is ambiguous, not in VALID_AMINO_ACIDS

    # Act & Assert
    with pytest.raises(ValueError, match="invalid amino acids"):
        dataset.sanitize_sequence(invalid_sequence)


@pytest.mark.unit
def test_sanitize_sequence_rejects_empty_string() -> None:
    """Verify sanitize_sequence raises ValueError for empty string"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")

    # Act & Assert
    with pytest.raises(ValueError, match="non-empty string"):
        dataset.sanitize_sequence("")


@pytest.mark.unit
def test_sanitize_sequence_rejects_none() -> None:
    """Verify sanitize_sequence raises ValueError for None"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")

    # Act & Assert
    with pytest.raises(ValueError, match="non-empty string"):
        dataset.sanitize_sequence(None)  # type: ignore


@pytest.mark.unit
def test_sanitize_sequence_accepts_all_valid_amino_acids() -> None:
    """Verify sanitize_sequence accepts all 20 standard amino acids"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    all_valid_aas = "ACDEFGHIKLMNPQRSTVWY"

    # Act
    sanitized = dataset.sanitize_sequence(all_valid_aas)

    # Assert
    assert sanitized == all_valid_aas


@pytest.mark.unit
def test_sanitize_sequence_rejects_numbers() -> None:
    """Verify sanitize_sequence raises ValueError for sequences with numbers"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    sequence_with_numbers = "QVQLVQSG123AEVKKPGA"

    # Act & Assert
    with pytest.raises(ValueError, match="invalid amino acids"):
        dataset.sanitize_sequence(sequence_with_numbers)


@pytest.mark.unit
def test_sanitize_sequence_rejects_special_characters() -> None:
    """Verify sanitize_sequence raises ValueError for special characters"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    sequence_with_special_chars = "QVQLVQSG!@#AEVKKPGA"

    # Act & Assert
    with pytest.raises(ValueError, match="invalid amino acids"):
        dataset.sanitize_sequence(sequence_with_special_chars)


# ============================================================================
# Sequence Validation Tests (DataFrame-level)
# ============================================================================


@pytest.mark.unit
def test_validate_sequences_returns_statistics() -> None:
    """Verify validate_sequences returns stats dictionary with correct structure"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    df = pd.DataFrame(
        {
            "VH_sequence": ["QVQLVQSGAEVKKPGA", "EVQLVESGGGLVQPGG"],
            "VL_sequence": ["DIQMTQSPSSLSASVG", "QSALTQPASVSGSPGQ"],
        }
    )

    # Act
    stats = dataset.validate_sequences(df)

    # Assert
    assert "total_sequences" in stats
    assert "valid_sequences" in stats
    assert "invalid_sequences" in stats
    assert "missing_vh" in stats
    assert "missing_vl" in stats
    assert "length_stats" in stats


@pytest.mark.unit
def test_validate_sequences_counts_total_sequences() -> None:
    """Verify validate_sequences counts total sequences correctly"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    df = pd.DataFrame(
        {"VH_sequence": ["QVQLVQSGAEVKKPGA", "EVQLVESGGGLVQPGG", "QVQLQQWGAGLLKPSE"]}
    )

    # Act
    stats = dataset.validate_sequences(df)

    # Assert
    assert stats["total_sequences"] == 3


@pytest.mark.unit
def test_validate_sequences_detects_missing_vh() -> None:
    """Verify validate_sequences detects missing VH sequences"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    df = pd.DataFrame({"VH_sequence": ["QVQLVQSGAEVKKPGA", None, "QVQLQQWGAGLLKPSE"]})

    # Act
    stats = dataset.validate_sequences(df)

    # Assert
    assert stats["missing_vh"] == 1


@pytest.mark.unit
def test_validate_sequences_detects_missing_vl() -> None:
    """Verify validate_sequences detects missing VL sequences"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    df = pd.DataFrame(
        {
            "VH_sequence": ["QVQLVQSGAEVKKPGA", "EVQLVESGGGLVQPGG"],
            "VL_sequence": ["DIQMTQSPSSLSASVG", None],
        }
    )

    # Act
    stats = dataset.validate_sequences(df)

    # Assert
    assert stats["missing_vl"] == 1


@pytest.mark.unit
def test_validate_sequences_handles_dataframe_with_no_vl() -> None:
    """Verify validate_sequences handles DataFrames without VL_sequence column"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    df = pd.DataFrame(
        {"VH_sequence": ["QVQLVQSGAEVKKPGA", "EVQLVESGGGLVQPGG"]}  # No VL_sequence
    )

    # Act
    stats = dataset.validate_sequences(df)

    # Assert
    assert stats["missing_vl"] == 0  # Should not error, just not count VL


# ============================================================================
# Fragment Type Constants Tests
# ============================================================================


@pytest.mark.unit
def test_full_antibody_fragments_contains_16_types() -> None:
    """Verify FULL_ANTIBODY_FRAGMENTS constant has all 16 fragment types"""
    # Arrange & Act
    fragments = AntibodyDataset.FULL_ANTIBODY_FRAGMENTS

    # Assert
    assert len(fragments) == 16
    expected_fragments = [
        "VH_only",
        "VL_only",
        "VH+VL",
        "H-CDR1",
        "H-CDR2",
        "H-CDR3",
        "L-CDR1",
        "L-CDR2",
        "L-CDR3",
        "H-CDRs",
        "L-CDRs",
        "All-CDRs",
        "H-FWRs",
        "L-FWRs",
        "All-FWRs",
        "Full",
    ]
    assert set(fragments) == set(expected_fragments)


@pytest.mark.unit
def test_nanobody_fragments_contains_6_types() -> None:
    """Verify NANOBODY_FRAGMENTS constant has all 6 fragment types"""
    # Arrange & Act
    fragments = AntibodyDataset.NANOBODY_FRAGMENTS

    # Assert
    assert len(fragments) == 6
    expected_fragments = [
        "VHH_only",
        "H-CDR1",
        "H-CDR2",
        "H-CDR3",
        "H-CDRs",
        "H-FWRs",
    ]
    assert set(fragments) == set(expected_fragments)


@pytest.mark.unit
def test_valid_amino_acids_contains_21_standard_aa() -> None:
    """Verify VALID_AMINO_ACIDS constant has all 20 standard amino acids + X"""
    # Arrange & Act
    valid_aas = AntibodyDataset.VALID_AMINO_ACIDS

    # Assert
    assert len(valid_aas) == 21
    expected_aas = set("ACDEFGHIKLMNPQRSTVWYX")
    assert valid_aas == expected_aas


# ============================================================================
# Edge Case Tests
# ============================================================================


@pytest.mark.unit
def test_sanitize_sequence_handles_mixed_case() -> None:
    """Verify sanitize_sequence handles mixed case input"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    mixed_case = "QvQlVqSgAeVkKpGa"

    # Act
    sanitized = dataset.sanitize_sequence(mixed_case)

    # Assert
    assert sanitized == "QVQLVQSGAEVKKPGA"


@pytest.mark.unit
def test_sanitize_sequence_handles_multiple_gaps() -> None:
    """Verify sanitize_sequence removes multiple gap characters"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    sequence_with_many_gaps = "Q-V-Q-L-V-Q-S-G"

    # Act
    sanitized = dataset.sanitize_sequence(sequence_with_many_gaps)

    # Assert
    assert sanitized == "QVQLVQSG"


@pytest.mark.unit
def test_sanitize_sequence_handles_combined_issues() -> None:
    """Verify sanitize_sequence handles gaps + spaces + lowercase together"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    messy_sequence = "q v q l - v q s g - a e v k k p g a"

    # Act
    sanitized = dataset.sanitize_sequence(messy_sequence)

    # Assert
    assert sanitized == "QVQLVQSGAEVKKPGA"


# ============================================================================
# Integration-like Tests (Test Full Workflow)
# ============================================================================


@pytest.mark.unit
def test_full_dataset_initialization_workflow(tmp_path: Path) -> None:
    """Verify complete dataset initialization workflow"""
    # Arrange
    output_dir = tmp_path / "test_output"

    # Act
    dataset = ConcreteDataset(dataset_name="test_dataset", output_dir=output_dir)
    df = dataset.load_data()
    fragment_types = dataset.get_fragment_types()

    # Assert - Dataset initialized
    assert dataset.dataset_name == "test_dataset"
    assert output_dir.exists()

    # Assert - Data loaded
    assert len(df) > 0
    assert "VH_sequence" in df.columns

    # Assert - Fragment types defined
    assert len(fragment_types) == 16


@pytest.mark.unit
def test_sanitization_workflow_on_realistic_data() -> None:
    """Verify sanitization works on realistic antibody sequences"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    realistic_vh = "qvqlvqsgaevkkpgasvkvsckasgytftsynmhwvrqapgqglewmggiypgdsdtryspsfqgqvtisadksistaylqwsslkasdtamyycarstyyggdwyfnvwgqgtlvtvss"

    # Act
    sanitized = dataset.sanitize_sequence(realistic_vh)

    # Assert
    assert sanitized.isupper()
    assert len(sanitized) == 121  # Actual length of the sequence
    assert all(aa in AntibodyDataset.VALID_AMINO_ACIDS for aa in sanitized)


# ============================================================================
# Fragment Creation Tests (create_fragments, create_fragment_csvs)
# ============================================================================


@pytest.mark.unit
def test_create_fragments_generates_all_fragment_types() -> None:
    """Verify create_fragments returns all 16 fragment types for full antibody"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    row = pd.Series(
        {
            "id": "AB001",
            "VH_sequence": "QVQLVQSGAEVKKPGA",
            "VL_sequence": "DIQMTQSPSSLSASVG",
            "VH_CDR1": "GFTFS",
            "VH_CDR2": "YISSG",
            "VH_CDR3": "ARYDDY",
            "VL_CDR1": "RASQS",
            "VL_CDR2": "AASTL",
            "VL_CDR3": "QQSYST",
            "VH_FWR1": "QVQLVQ",
            "VH_FWR2": "SGAEVK",
            "VH_FWR3": "KPGA",
            "VH_FWR4": "",
            "VL_FWR1": "DIQMTQ",
            "VL_FWR2": "SPSSLSA",
            "VL_FWR3": "SVG",
            "VL_FWR4": "",
            "label": 0,
        }
    )

    # Act
    fragments = dataset.create_fragments(row)

    # Assert
    expected_fragments = dataset.get_fragment_types()
    assert len(fragments) == len(expected_fragments)
    for ftype in expected_fragments:
        assert ftype in fragments
        seq, label, source = fragments[ftype]
        assert isinstance(seq, str)
        assert label == 0
        assert source == "AB001"


@pytest.mark.unit
def test_create_fragments_concatenates_cdrs_correctly() -> None:
    """Verify create_fragments concatenates CDR regions correctly"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    row = pd.Series(
        {
            "id": "AB002",
            "VH_sequence": "AAABBBCCC",
            "VL_sequence": "DDDEEFFFF",
            "VH_CDR1": "AAA",
            "VH_CDR2": "BBB",
            "VH_CDR3": "CCC",
            "VL_CDR1": "DDD",
            "VL_CDR2": "EEE",
            "VL_CDR3": "FFF",
            "label": 1,
        }
    )

    # Act
    fragments = dataset.create_fragments(row)

    # Assert
    assert fragments["H-CDRs"][0] == "AAABBBCCC"  # Heavy CDRs concatenated
    assert fragments["L-CDRs"][0] == "DDDEEEFFF"  # Light CDRs concatenated
    assert fragments["All-CDRs"][0] == "AAABBBCCCDDDEEEFFF"  # All CDRs concatenated


@pytest.mark.unit
def test_create_fragments_handles_missing_vl_sequence() -> None:
    """Verify create_fragments handles nanobodies (no VL chain)"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    row = pd.Series(
        {
            "id": "NANO001",
            "VH_sequence": "QVQLVQSGAEVKKPGA",
            "VL_sequence": "",  # Nanobodies have no VL chain
            "VH_CDR1": "GFTFS",
            "VH_CDR2": "YISSG",
            "VH_CDR3": "ARYDDY",
            "label": 0,
        }
    )

    # Act
    fragments = dataset.create_fragments(row)

    # Assert - VL fragments should be empty strings
    assert fragments["VL_only"][0] == ""
    assert fragments["VH_only"][0] == "QVQLVQSGAEVKKPGA"
    assert "H-CDRs" in fragments
    assert fragments["H-CDRs"][0] == "GFTFSYISSGARYDDY"


@pytest.mark.unit
def test_create_fragments_preserves_label() -> None:
    """Verify create_fragments preserves label in all fragments"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    row = pd.Series(
        {
            "id": "AB003",
            "VH_sequence": "QVQLVQSG",
            "VL_sequence": "DIQMTQSP",
            "label": 1,  # Non-specific
        }
    )

    # Act
    fragments = dataset.create_fragments(row)

    # Assert - All fragments should have label=1
    for ftype, (_seq, label, _source) in fragments.items():
        assert label == 1, f"Fragment {ftype} has wrong label"


@pytest.mark.unit
def test_create_fragment_csvs_writes_all_files(tmp_path: Path) -> None:
    """Verify create_fragment_csvs writes CSV for each fragment type"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset", output_dir=tmp_path)
    df = pd.DataFrame(
        {
            "id": ["AB001", "AB002"],
            "VH_sequence": ["QVQLVQSG", "EVQLVESG"],
            "VL_sequence": ["DIQMTQSP", "QSALTQPA"],
            "VH_CDR1": ["GFTFS", "GFTFD"],
            "VH_CDR2": ["YISSG", "AISGS"],
            "VH_CDR3": ["ARYDDY", "AKDIQY"],
            "VL_CDR1": ["RASQS", "TGTSSD"],
            "VL_CDR2": ["AASTL", "DVSNR"],
            "VL_CDR3": ["QQSYST", "SSYTSS"],
            "VH_FWR1": ["QVQL", "EVQL"],
            "VH_FWR2": ["VQSG", "VESG"],
            "VH_FWR3": ["", ""],
            "VH_FWR4": ["", ""],
            "VL_FWR1": ["DIQM", "QSAL"],
            "VL_FWR2": ["TQSP", "TQPA"],
            "VL_FWR3": ["", ""],
            "VL_FWR4": ["", ""],
            "label": [0, 1],
        }
    )

    # Act
    dataset.create_fragment_csvs(df, suffix="")

    # Assert - All fragment types should have CSV files
    fragment_types = dataset.get_fragment_types()
    for ftype in fragment_types:
        csv_file = tmp_path / f"{ftype}_test_dataset.csv"
        assert csv_file.exists(), f"Fragment CSV not created for {ftype}"

        # Verify CSV structure
        df_fragment = pd.read_csv(csv_file, comment="#")
        assert "id" in df_fragment.columns
        assert "sequence" in df_fragment.columns
        assert "label" in df_fragment.columns
        assert "source" in df_fragment.columns


@pytest.mark.unit
def test_create_fragment_csvs_includes_metadata_header(tmp_path: Path) -> None:
    """Verify create_fragment_csvs writes metadata headers"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset", output_dir=tmp_path)
    df = pd.DataFrame(
        {
            "id": ["AB001"],
            "VH_sequence": ["QVQLVQSG"],
            "VL_sequence": ["DIQMTQSP"],
            "label": [0],
        }
    )

    # Act
    dataset.create_fragment_csvs(df, suffix="")

    # Assert - Check metadata in VH_only CSV
    vh_csv = tmp_path / "VH_only_test_dataset.csv"
    with open(vh_csv) as f:
        lines = f.readlines()

    assert lines[0].startswith("# Dataset:")
    assert lines[1].startswith("# Fragment type:")
    assert lines[2].startswith("# Total sequences:")
    assert lines[3].startswith("# Label distribution:")


@pytest.mark.unit
def test_create_fragment_csvs_handles_suffix(tmp_path: Path) -> None:
    """Verify create_fragment_csvs uses suffix parameter correctly"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset", output_dir=tmp_path)
    df = pd.DataFrame(
        {
            "id": ["AB001"],
            "VH_sequence": ["QVQLVQSG"],
            "VL_sequence": ["DIQMTQSP"],
            "label": [0],
        }
    )

    # Act
    dataset.create_fragment_csvs(df, suffix="_filtered")

    # Assert - Files should have suffix
    vh_csv = tmp_path / "VH_only_test_dataset_filtered.csv"
    assert vh_csv.exists()


@pytest.mark.unit
def test_create_fragment_csvs_skips_empty_fragments(tmp_path: Path) -> None:
    """Verify create_fragment_csvs skips fragment types with no data"""
    # Arrange - Create dataset with only VH (no VL)
    dataset = ConcreteDataset(dataset_name="test_dataset", output_dir=tmp_path)
    df = pd.DataFrame(
        {
            "id": ["NANO001"],
            "VH_sequence": ["QVQLVQSG"],
            "VL_sequence": [""],  # Nanobodies have no VL chain
            "label": [0],
        }
    )

    # Act
    dataset.create_fragment_csvs(df, suffix="")

    # Assert - VL fragments should not exist or be empty
    vl_csv = tmp_path / "VL_only_test_dataset.csv"
    if vl_csv.exists():
        df_vl = pd.read_csv(vl_csv, comment="#")
        assert len(df_vl) == 0 or df_vl["sequence"].str.len().sum() == 0


# ============================================================================
# Log Dataset Statistics Tests (Lines 242-273)
# ============================================================================


@pytest.mark.unit
def test_print_statistics(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Test print_statistics with real DataFrame"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset", output_dir=tmp_path)
    df = pd.DataFrame(
        {
            "id": ["AB001", "AB002", "AB003"],
            "VH_sequence": [
                "QVQLVQSGAEVKKPGA",
                "EVQLVESGGGLVQPGG",
                "QVQLQQWGAGLLKPSE",
            ],
            "VL_sequence": [
                "DIQMTQSPSSLSASVG",
                "QSALTQPASVSGSPGQ",
                "DIQMTQSPSSLSASVG",
            ],
            "label": [0, 1, 0],
        }
    )

    # Act
    caplog.set_level(logging.INFO)
    dataset.print_statistics(df, stage="Test")

    # Assert
    assert "Total sequences: 3" in caplog.text
    assert "Specific" in caplog.text
    assert "Non-specific" in caplog.text
    assert "Label distribution:" in caplog.text
    assert "Sequence validation:" in caplog.text


@pytest.mark.unit
def test_print_statistics_without_labels(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Test print_statistics handles DataFrame without label column"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset", output_dir=tmp_path)
    df = pd.DataFrame(
        {
            "id": ["AB001", "AB002"],
            "VH_sequence": ["QVQLVQSGAEVKKPGA", "EVQLVESGGGLVQPGG"],
            "VL_sequence": ["DIQMTQSPSSLSASVG", "QSALTQPASVSGSPGQ"],
        }
    )

    # Act
    caplog.set_level(logging.INFO)
    dataset.print_statistics(df, stage="Preprocessing")

    # Assert
    assert "Total sequences: 2" in caplog.text
    assert "Sequence validation:" in caplog.text
    # Should not crash when label column missing


# ============================================================================
# Real annotate_all Tests (Lines 341-380) - NO MOCKS
# ============================================================================


@pytest.mark.unit
def test_annotate_all_with_real_sequences(tmp_path: Path) -> None:
    """Test annotate_all with REAL ANARCI calls on DataFrame - NO MOCKS

    NOTE: This test calls the actual annotate_all() method which uses real
    ANARCI annotation via the riot_na library. This tests lines 341-380.

    The annotate_sequence() method (lines 292-327) is broken and cannot be
    tested without fixing it first (it incorrectly calls create_riot_aa with
    wrong arguments). The test_base_annotation.py file uses mocks to hide this bug.
    """
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset", output_dir=tmp_path)

    # Use real VH/VL sequences from Boughter dataset
    df = pd.DataFrame(
        {
            "id": ["AB001", "AB002"],
            "VH_sequence": [
                "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
                "QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAISWVRQAPGQGLEWMGGIIPIFGTANYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCAR",
            ],
            "VL_sequence": [
                "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK",
                "EIVLTQSPGTLSLSPGERATLSCRASQSVSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGSSPTFGQGTRLEIK",
            ],
            "label": [0, 1],
        }
    )

    # Act - NO MOCKS, real ANARCI calls
    annotated_df = dataset.annotate_all(df)

    # Assert - Verify annotation columns added
    assert "VH_FWR1" in annotated_df.columns
    assert "VH_CDR1" in annotated_df.columns
    assert "VH_CDR2" in annotated_df.columns
    assert "VH_CDR3" in annotated_df.columns
    assert "VL_FWR1" in annotated_df.columns
    assert "VL_CDR1" in annotated_df.columns
    assert "VL_CDR2" in annotated_df.columns
    assert "VL_CDR3" in annotated_df.columns

    # Verify real annotations exist (not all NaN)
    assert pd.notna(annotated_df["VH_CDR1"].iloc[0])
    assert pd.notna(annotated_df["VH_CDR3"].iloc[0])
    assert pd.notna(annotated_df["VL_CDR1"].iloc[0])

    # Verify original columns preserved
    assert "id" in annotated_df.columns
    assert "label" in annotated_df.columns
