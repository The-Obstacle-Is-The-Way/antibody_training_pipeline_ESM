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

import logging
from pathlib import Path
from typing import cast

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

    def load_data(self, **kwargs) -> pd.DataFrame:
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
def test_dataset_initializes_with_name_and_default_output(tmp_path):
    """Verify dataset initializes with dataset name and creates default output directory"""
    # Arrange & Act
    dataset = ConcreteDataset(dataset_name="test_dataset")

    # Assert
    assert dataset.dataset_name == "test_dataset"
    assert dataset.output_dir == Path("outputs/test_dataset")
    assert isinstance(dataset.logger, logging.Logger)


@pytest.mark.unit
def test_dataset_initializes_with_custom_output_dir(tmp_path):
    """Verify dataset can use custom output directory"""
    # Arrange
    custom_dir = tmp_path / "custom_output"

    # Act
    dataset = ConcreteDataset(dataset_name="test_dataset", output_dir=custom_dir)

    # Assert
    assert dataset.output_dir == custom_dir
    assert custom_dir.exists()  # Should be created


@pytest.mark.unit
def test_dataset_creates_output_directory_on_init(tmp_path):
    """Verify output directory is created if it doesn't exist"""
    # Arrange
    output_dir = tmp_path / "new_directory"
    assert not output_dir.exists()  # Verify it doesn't exist yet

    # Act
    _ = ConcreteDataset(dataset_name="test_dataset", output_dir=output_dir)

    # Assert
    assert output_dir.exists()


@pytest.mark.unit
def test_dataset_accepts_custom_logger():
    """Verify dataset can use custom logger"""
    # Arrange
    custom_logger = logging.getLogger("custom_test_logger")

    # Act
    dataset = ConcreteDataset(dataset_name="test_dataset", logger=custom_logger)

    # Assert
    assert dataset.logger == custom_logger


@pytest.mark.unit
def test_dataset_creates_default_logger_if_none_provided():
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
def test_concrete_dataset_implements_load_data():
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
def test_concrete_dataset_implements_get_fragment_types():
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
def test_sanitize_sequence_removes_gaps():
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
def test_sanitize_sequence_removes_whitespace():
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
def test_sanitize_sequence_converts_to_uppercase():
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
def test_sanitize_sequence_rejects_invalid_amino_acids():
    """Verify sanitize_sequence raises ValueError for invalid amino acids"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    invalid_sequence = "QVQLVQSGBBB"  # 'B' is ambiguous, not in VALID_AMINO_ACIDS

    # Act & Assert
    with pytest.raises(ValueError, match="invalid amino acids"):
        dataset.sanitize_sequence(invalid_sequence)


@pytest.mark.unit
def test_sanitize_sequence_rejects_empty_string():
    """Verify sanitize_sequence raises ValueError for empty string"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")

    # Act & Assert
    with pytest.raises(ValueError, match="non-empty string"):
        dataset.sanitize_sequence("")


@pytest.mark.unit
def test_sanitize_sequence_rejects_none():
    """Verify sanitize_sequence raises ValueError for None"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")

    # Act & Assert
    with pytest.raises(ValueError, match="non-empty string"):
        dataset.sanitize_sequence(None)  # type: ignore


@pytest.mark.unit
def test_sanitize_sequence_accepts_all_valid_amino_acids():
    """Verify sanitize_sequence accepts all 20 standard amino acids"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    all_valid_aas = "ACDEFGHIKLMNPQRSTVWY"

    # Act
    sanitized = dataset.sanitize_sequence(all_valid_aas)

    # Assert
    assert sanitized == all_valid_aas


@pytest.mark.unit
def test_sanitize_sequence_rejects_numbers():
    """Verify sanitize_sequence raises ValueError for sequences with numbers"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    sequence_with_numbers = "QVQLVQSG123AEVKKPGA"

    # Act & Assert
    with pytest.raises(ValueError, match="invalid amino acids"):
        dataset.sanitize_sequence(sequence_with_numbers)


@pytest.mark.unit
def test_sanitize_sequence_rejects_special_characters():
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
def test_validate_sequences_returns_statistics():
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
def test_validate_sequences_counts_total_sequences():
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
def test_validate_sequences_detects_missing_vh():
    """Verify validate_sequences detects missing VH sequences"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    df = pd.DataFrame({"VH_sequence": ["QVQLVQSGAEVKKPGA", None, "QVQLQQWGAGLLKPSE"]})

    # Act
    stats = dataset.validate_sequences(df)

    # Assert
    assert stats["missing_vh"] == 1


@pytest.mark.unit
def test_validate_sequences_detects_missing_vl():
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
def test_validate_sequences_handles_dataframe_with_no_vl():
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
def test_full_antibody_fragments_contains_16_types():
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
def test_nanobody_fragments_contains_6_types():
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
def test_valid_amino_acids_contains_20_standard_aa():
    """Verify VALID_AMINO_ACIDS constant has all 20 standard amino acids"""
    # Arrange & Act
    valid_aas = AntibodyDataset.VALID_AMINO_ACIDS

    # Assert
    assert len(valid_aas) == 20
    expected_aas = set("ACDEFGHIKLMNPQRSTVWY")
    assert valid_aas == expected_aas


# ============================================================================
# Edge Case Tests
# ============================================================================


@pytest.mark.unit
def test_sanitize_sequence_handles_mixed_case():
    """Verify sanitize_sequence handles mixed case input"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    mixed_case = "QvQlVqSgAeVkKpGa"

    # Act
    sanitized = dataset.sanitize_sequence(mixed_case)

    # Assert
    assert sanitized == "QVQLVQSGAEVKKPGA"


@pytest.mark.unit
def test_sanitize_sequence_handles_multiple_gaps():
    """Verify sanitize_sequence removes multiple gap characters"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    sequence_with_many_gaps = "Q-V-Q-L-V-Q-S-G"

    # Act
    sanitized = dataset.sanitize_sequence(sequence_with_many_gaps)

    # Assert
    assert sanitized == "QVQLVQSG"


@pytest.mark.unit
def test_sanitize_sequence_handles_combined_issues():
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
def test_full_dataset_initialization_workflow(tmp_path):
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
def test_sanitization_workflow_on_realistic_data():
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
