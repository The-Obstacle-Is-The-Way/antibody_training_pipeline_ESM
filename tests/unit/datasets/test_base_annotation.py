"""
Unit Tests for AntibodyDataset ANARCI Annotation Logic

Tests cover:
- Single sequence annotation (annotate_sequence)
- Batch annotation (annotate_all)
- Error handling for ANARCI failures
- Import error handling for riot_na

Testing philosophy:
- Test behaviors, not implementation
- Mock riot_na to avoid external dependency
- Focus on error paths and edge cases
- Follow AAA pattern (Arrange-Act-Assert)

This addresses Priority 2 gaps from TEST_COVERAGE_GAPS.md:
- ANARCI annotation error handling (lines 274-326)
- Batch annotation (lines 328-379)

Date: 2025-11-08
Phase: Coverage improvement - Phase 2
"""

from unittest.mock import patch

import pandas as pd
import pytest

from antibody_training_esm.datasets.base import AntibodyDataset

# ============================================================================
# Test Fixtures - Concrete Dataset Implementation
# ============================================================================


class ConcreteDataset(AntibodyDataset):
    """Concrete implementation of AntibodyDataset for testing."""

    def load_data(self, **kwargs) -> pd.DataFrame:
        """Return a simple test DataFrame"""
        return pd.DataFrame(
            {
                "id": ["AB001", "AB002"],
                "VH_sequence": ["QVQLVQSGAEVKKPGA", "EVQLVESGGGLVQPGG"],
                "VL_sequence": ["DIQMTQSPSSLSASVG", "QSALTQPASVSGSPGQ"],
                "label": [0, 1],
            }
        )

    def get_fragment_types(self) -> list[str]:
        """Return standard full antibody fragments"""
        return list(self.FULL_ANTIBODY_FRAGMENTS)


# ============================================================================
# ANARCI Single Sequence Annotation Tests (annotate_sequence)
# ============================================================================


@pytest.mark.unit
def test_annotate_sequence_success():
    """Test successful ANARCI annotation of single sequence"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")

    mock_result = {
        "FWR1": "QVQLVQ",
        "CDR1": "SGAEVK",
        "FWR2": "KPGA",
        "CDR2": "SVKV",
        "FWR3": "SCKAS",
        "CDR3": "GYTFTS",
        "FWR4": "YNMH",
    }

    with patch("riot_na.create_riot_aa") as mock_riot:
        mock_riot.return_value = mock_result

        # Act
        annotations = dataset.annotate_sequence(
            "AB001", "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYNMH", "H"
        )

        # Assert
        assert annotations is not None
        assert annotations["FWR1"] == "QVQLVQ"
        assert annotations["CDR1"] == "SGAEVK"
        assert annotations["CDR3"] == "GYTFTS"
        assert len(annotations) == 7  # 4 FWRs + 3 CDRs


@pytest.mark.unit
def test_annotate_sequence_riot_na_returns_none():
    """Test annotate_sequence when riot_na fails to annotate"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")

    with patch("riot_na.create_riot_aa") as mock_riot:
        mock_riot.return_value = None  # ANARCI failed

        # Act
        annotations = dataset.annotate_sequence("AB001", "INVALIDSEQ", "H")

        # Assert
        assert annotations is None


@pytest.mark.unit
def test_annotate_sequence_all_annotations_empty():
    """Test annotate_sequence when ANARCI returns all empty strings"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")

    mock_result = {
        "FWR1": "",
        "CDR1": "",
        "FWR2": "",
        "CDR2": "",
        "FWR3": "",
        "CDR3": "",
        "FWR4": "",
    }

    with patch("riot_na.create_riot_aa") as mock_riot:
        mock_riot.return_value = mock_result

        # Act
        annotations = dataset.annotate_sequence("AB001", "QVQLVQSG", "H")

        # Assert
        assert annotations is None  # Should return None for all-empty annotations


@pytest.mark.unit
def test_annotate_sequence_riot_na_raises_exception():
    """Test annotate_sequence handles exceptions from riot_na"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")

    with patch("riot_na.create_riot_aa") as mock_riot:
        mock_riot.side_effect = RuntimeError("ANARCI internal error")

        # Act
        annotations = dataset.annotate_sequence("AB001", "QVQLVQSG", "H")

        # Assert
        assert annotations is None  # Should handle exception gracefully


@pytest.mark.unit
def test_annotate_sequence_riot_na_import_error():
    """Test annotate_sequence handles ImportError when riot_na not installed"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")

    # Mock import failure
    with patch("riot_na.create_riot_aa") as mock_riot:
        mock_riot.side_effect = ImportError("No module named 'riot_na'")

        # Act
        annotations = dataset.annotate_sequence("AB001", "QVQLVQSG", "H")

        # Assert
        assert annotations is None  # Should handle missing dependency gracefully


@pytest.mark.unit
def test_annotate_sequence_heavy_chain():
    """Test annotate_sequence correctly passes heavy chain ('H') to riot_na"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")

    mock_result = {
        "FWR1": "Q",
        "CDR1": "V",
        "FWR2": "Q",
        "CDR2": "L",
        "FWR3": "V",
        "CDR3": "Q",
        "FWR4": "S",
    }

    with patch("riot_na.create_riot_aa") as mock_riot:
        mock_riot.return_value = mock_result

        # Act
        dataset.annotate_sequence("AB001", "QVQLVQSG", "H")

        # Assert
        mock_riot.assert_called_once_with("AB001", "QVQLVQSG", chain="H")


@pytest.mark.unit
def test_annotate_sequence_light_chain():
    """Test annotate_sequence correctly passes light chain ('L') to riot_na"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")

    mock_result = {
        "FWR1": "D",
        "CDR1": "I",
        "FWR2": "Q",
        "CDR2": "M",
        "FWR3": "T",
        "CDR3": "Q",
        "FWR4": "S",
    }

    with patch("riot_na.create_riot_aa") as mock_riot:
        mock_riot.return_value = mock_result

        # Act
        dataset.annotate_sequence("VL001", "DIQMTQSP", "L")

        # Assert
        mock_riot.assert_called_once_with("VL001", "DIQMTQSP", chain="L")


@pytest.mark.unit
def test_annotate_sequence_handles_missing_fields_in_result():
    """Test annotate_sequence fills in missing fields with empty strings"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")

    # ANARCI returns incomplete result (missing some regions)
    mock_result = {
        "FWR1": "QVQLVQ",
        "CDR1": "SGAEVK",
        # Missing FWR2, CDR2, FWR3, CDR3, FWR4
    }

    with patch("riot_na.create_riot_aa") as mock_riot:
        mock_riot.return_value = mock_result

        # Act
        annotations = dataset.annotate_sequence("AB001", "QVQLVQSG", "H")

        # Assert
        assert annotations is not None
        assert annotations["FWR1"] == "QVQLVQ"
        assert annotations["CDR1"] == "SGAEVK"
        assert annotations["FWR2"] == ""  # Should default to empty string
        assert annotations["CDR2"] == ""
        assert annotations["FWR3"] == ""
        assert annotations["CDR3"] == ""
        assert annotations["FWR4"] == ""


# ============================================================================
# ANARCI Batch Annotation Tests (annotate_all)
# ============================================================================


@pytest.mark.unit
def test_annotate_all_vh_only():
    """Test annotate_all annotates VH-only dataset (nanobody)"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    df = pd.DataFrame(
        {
            "id": ["NANO001", "NANO002"],
            "VH_sequence": ["QVQLVQSGAEVKKPGA", "EVQLVESGGGLVQPGG"],
            "label": [0, 1],
        }
    )

    mock_vh_result = {
        "FWR1": "QVQL",
        "CDR1": "VQSG",
        "FWR2": "AEVK",
        "CDR2": "KPGA",
        "FWR3": "",
        "CDR3": "",
        "FWR4": "",
    }

    with patch("riot_na.create_riot_aa") as mock_riot:
        mock_riot.return_value = mock_vh_result

        # Act
        df_annotated = dataset.annotate_all(df)

        # Assert
        assert "VH_FWR1" in df_annotated.columns
        assert "VH_CDR1" in df_annotated.columns
        assert "VH_CDR2" in df_annotated.columns
        assert "VH_CDR3" in df_annotated.columns
        assert "VH_FWR2" in df_annotated.columns
        assert "VH_FWR3" in df_annotated.columns
        assert "VH_FWR4" in df_annotated.columns

        # Should not have VL annotation columns (VH-only dataset)
        assert "VL_FWR1" not in df_annotated.columns
        assert "VL_CDR1" not in df_annotated.columns


@pytest.mark.unit
def test_annotate_all_vh_vl_paired():
    """Test annotate_all annotates paired VH+VL dataset"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    df = pd.DataFrame(
        {
            "id": ["AB001", "AB002"],
            "VH_sequence": ["QVQLVQSGAEVKKPGA", "EVQLVESGGGLVQPGG"],
            "VL_sequence": ["DIQMTQSPSSLSASVG", "QSALTQPASVSGSPGQ"],
            "label": [0, 1],
        }
    )

    mock_vh_result = {
        "FWR1": "QVH",
        "CDR1": "VH1",
        "FWR2": "FH2",
        "CDR2": "CH2",
        "FWR3": "FH3",
        "CDR3": "CH3",
        "FWR4": "FH4",
    }
    mock_vl_result = {
        "FWR1": "QVL",
        "CDR1": "VL1",
        "FWR2": "FL2",
        "CDR2": "CL2",
        "FWR3": "FL3",
        "CDR3": "CL3",
        "FWR4": "FL4",
    }

    def mock_riot_side_effect(seq_id, sequence, chain):
        if chain == "H":
            return mock_vh_result
        elif chain == "L":
            return mock_vl_result
        return None

    with patch("riot_na.create_riot_aa") as mock_riot:
        mock_riot.side_effect = mock_riot_side_effect

        # Act
        df_annotated = dataset.annotate_all(df)

        # Assert - VH annotations
        assert "VH_FWR1" in df_annotated.columns
        assert "VH_CDR1" in df_annotated.columns
        assert df_annotated["VH_FWR1"].iloc[0] == "QVH"
        assert df_annotated["VH_CDR1"].iloc[0] == "VH1"

        # Assert - VL annotations
        assert "VL_FWR1" in df_annotated.columns
        assert "VL_CDR1" in df_annotated.columns
        assert df_annotated["VL_FWR1"].iloc[0] == "QVL"
        assert df_annotated["VL_CDR1"].iloc[0] == "VL1"


@pytest.mark.unit
def test_annotate_all_partial_failures():
    """Test annotate_all handles partial annotation failures gracefully"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    df = pd.DataFrame(
        {
            "id": ["AB001", "AB002_FAIL", "AB003"],
            "VH_sequence": ["QVQLVQSGAEVKKPGA", "INVALIDSEQ", "EVQLVESGGGLVQPGG"],
            "label": [0, 1, 0],
        }
    )

    mock_vh_result = {
        "FWR1": "QVQL",
        "CDR1": "VQSG",
        "FWR2": "AEVK",
        "CDR2": "KPGA",
        "FWR3": "",
        "CDR3": "",
        "FWR4": "",
    }

    def mock_riot_side_effect(seq_id, sequence, chain):
        if "FAIL" in seq_id:
            return None  # Annotation failed for this sequence
        return mock_vh_result

    with patch("riot_na.create_riot_aa") as mock_riot:
        mock_riot.side_effect = mock_riot_side_effect

        # Act
        df_annotated = dataset.annotate_all(df)

        # Assert - First sequence annotated successfully
        assert df_annotated["VH_FWR1"].iloc[0] == "QVQL"

        # Assert - Second sequence failed, should have empty annotations
        assert df_annotated["VH_FWR1"].iloc[1] == ""
        assert df_annotated["VH_CDR1"].iloc[1] == ""

        # Assert - Third sequence annotated successfully
        assert df_annotated["VH_FWR1"].iloc[2] == "QVQL"


@pytest.mark.unit
def test_annotate_all_handles_nan_vh_sequences():
    """Test annotate_all skips NaN VH sequences"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    df = pd.DataFrame(
        {
            "id": ["AB001", "AB002_MISSING", "AB003"],
            "VH_sequence": ["QVQLVQSGAEVKKPGA", None, "EVQLVESGGGLVQPGG"],
            "label": [0, 1, 0],
        }
    )

    mock_vh_result = {
        "FWR1": "QVQL",
        "CDR1": "VQSG",
        "FWR2": "AEVK",
        "CDR2": "KPGA",
        "FWR3": "",
        "CDR3": "",
        "FWR4": "",
    }

    with patch("riot_na.create_riot_aa") as mock_riot:
        mock_riot.return_value = mock_vh_result

        # Act
        df_annotated = dataset.annotate_all(df)

        # Assert - riot_na should only be called for non-NaN sequences
        assert (
            mock_riot.call_count == 2
        )  # Called for AB001 and AB003, not AB002_MISSING

        # Assert - NaN sequence should have empty annotations
        assert df_annotated["VH_FWR1"].iloc[1] == ""
        assert df_annotated["VH_CDR1"].iloc[1] == ""


@pytest.mark.unit
def test_annotate_all_handles_nan_vl_sequences():
    """Test annotate_all skips NaN VL sequences"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    df = pd.DataFrame(
        {
            "id": ["AB001", "NANO001"],
            "VH_sequence": ["QVQLVQSGAEVKKPGA", "EVQLVESGGGLVQPGG"],
            "VL_sequence": ["DIQMTQSPSSLSASVG", None],  # Second is nanobody (no VL)
            "label": [0, 1],
        }
    )

    mock_vh_result = {
        "FWR1": "QVH",
        "CDR1": "VH1",
        "FWR2": "FH2",
        "CDR2": "CH2",
        "FWR3": "FH3",
        "CDR3": "CH3",
        "FWR4": "FH4",
    }
    mock_vl_result = {
        "FWR1": "QVL",
        "CDR1": "VL1",
        "FWR2": "FL2",
        "CDR2": "CL2",
        "FWR3": "FL3",
        "CDR3": "CL3",
        "FWR4": "FL4",
    }

    def mock_riot_side_effect(seq_id, sequence, chain):
        if chain == "H":
            return mock_vh_result
        elif chain == "L":
            return mock_vl_result
        return None

    with patch("riot_na.create_riot_aa") as mock_riot:
        mock_riot.side_effect = mock_riot_side_effect

        # Act
        df_annotated = dataset.annotate_all(df)

        # Assert - VH should be annotated for both
        assert df_annotated["VH_FWR1"].iloc[0] == "QVH"
        assert df_annotated["VH_FWR1"].iloc[1] == "QVH"

        # Assert - VL only annotated for first (AB001), second (NANO001) should be empty
        assert df_annotated["VL_FWR1"].iloc[0] == "QVL"
        assert df_annotated["VL_FWR1"].iloc[1] == ""  # Nanobody has no VL


@pytest.mark.unit
def test_annotate_all_adds_all_seven_annotation_columns_per_chain():
    """Test annotate_all adds all 7 annotation columns (4 FWRs + 3 CDRs) per chain"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    df = pd.DataFrame(
        {
            "id": ["AB001"],
            "VH_sequence": ["QVQLVQSGAEVKKPGA"],
            "VL_sequence": ["DIQMTQSPSSLSASVG"],
            "label": [0],
        }
    )

    mock_result = {
        "FWR1": "A",
        "CDR1": "B",
        "FWR2": "C",
        "CDR2": "D",
        "FWR3": "E",
        "CDR3": "F",
        "FWR4": "G",
    }

    with patch("riot_na.create_riot_aa") as mock_riot:
        mock_riot.return_value = mock_result

        # Act
        df_annotated = dataset.annotate_all(df)

        # Assert - VH annotations (7 columns)
        expected_vh_cols = [
            "VH_FWR1",
            "VH_CDR1",
            "VH_FWR2",
            "VH_CDR2",
            "VH_FWR3",
            "VH_CDR3",
            "VH_FWR4",
        ]
        for col in expected_vh_cols:
            assert col in df_annotated.columns, f"Missing column: {col}"

        # Assert - VL annotations (7 columns)
        expected_vl_cols = [
            "VL_FWR1",
            "VL_CDR1",
            "VL_FWR2",
            "VL_CDR2",
            "VL_FWR3",
            "VL_CDR3",
            "VL_FWR4",
        ]
        for col in expected_vl_cols:
            assert col in df_annotated.columns, f"Missing column: {col}"


@pytest.mark.unit
def test_annotate_all_preserves_original_columns():
    """Test annotate_all preserves original DataFrame columns"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    df = pd.DataFrame(
        {
            "id": ["AB001"],
            "VH_sequence": ["QVQLVQSGAEVKKPGA"],
            "VL_sequence": ["DIQMTQSPSSLSASVG"],
            "label": [0],
            "custom_column": ["custom_value"],
        }
    )

    mock_result = {
        "FWR1": "A",
        "CDR1": "B",
        "FWR2": "C",
        "CDR2": "D",
        "FWR3": "E",
        "CDR3": "F",
        "FWR4": "G",
    }

    with patch("riot_na.create_riot_aa") as mock_riot:
        mock_riot.return_value = mock_result

        # Act
        df_annotated = dataset.annotate_all(df)

        # Assert - Original columns preserved
        assert "id" in df_annotated.columns
        assert "VH_sequence" in df_annotated.columns
        assert "VL_sequence" in df_annotated.columns
        assert "label" in df_annotated.columns
        assert "custom_column" in df_annotated.columns
        assert df_annotated["custom_column"].iloc[0] == "custom_value"


@pytest.mark.unit
def test_annotate_all_uses_id_column_for_sequence_id():
    """Test annotate_all uses 'id' column for sequence identifiers"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    df = pd.DataFrame(
        {
            "id": ["CUSTOM_ID_001"],
            "VH_sequence": ["QVQLVQSGAEVKKPGA"],
            "label": [0],
        }
    )

    mock_result = {
        "FWR1": "A",
        "CDR1": "B",
        "FWR2": "C",
        "CDR2": "D",
        "FWR3": "E",
        "CDR3": "F",
        "FWR4": "G",
    }

    with patch("riot_na.create_riot_aa") as mock_riot:
        mock_riot.return_value = mock_result

        # Act
        dataset.annotate_all(df)

        # Assert - Should use custom ID from 'id' column
        mock_riot.assert_called_with("CUSTOM_ID_001", "QVQLVQSGAEVKKPGA", chain="H")


@pytest.mark.unit
def test_annotate_all_falls_back_to_row_index_if_no_id_column():
    """Test annotate_all uses row index as fallback when 'id' column missing"""
    # Arrange
    dataset = ConcreteDataset(dataset_name="test_dataset")
    df = pd.DataFrame(
        {
            # No 'id' column
            "VH_sequence": ["QVQLVQSGAEVKKPGA"],
            "label": [0],
        }
    )

    mock_result = {
        "FWR1": "A",
        "CDR1": "B",
        "FWR2": "C",
        "CDR2": "D",
        "FWR3": "E",
        "CDR3": "F",
        "FWR4": "G",
    }

    with patch("riot_na.create_riot_aa") as mock_riot:
        mock_riot.return_value = mock_result

        # Act
        dataset.annotate_all(df)

        # Assert - Should use "seq_{row_index}" as fallback
        mock_riot.assert_called_with("seq_0", "QVQLVQSGAEVKKPGA", chain="H")
