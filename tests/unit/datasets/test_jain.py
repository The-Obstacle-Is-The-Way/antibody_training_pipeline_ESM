#!/usr/bin/env python3
"""
Unit Tests for JainDataset

Tests the Jain therapeutic antibody dataset loader including:
- Dataset initialization
- Multi-stage loading (full, ssot, parity)
- ELISA filtering
- 5-antibody reclassification
- PSR/AC-SINS ranking removal
- Fragment type configuration

Philosophy:
- Test behaviors, not implementation details
- Use mock CSV files for fast tests
- Mock only file I/O when testing error cases
- Focus on filtering logic and stage transitions

Date: 2025-11-07
Author: Claude Code
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from antibody_training_esm.datasets.jain import JainDataset

# ============================================================================
# Initialization Tests
# ============================================================================


@pytest.mark.unit
def test_jain_dataset_initializes_with_default_output_dir() -> None:
    """Verify JainDataset initializes with default output directory"""
    # Arrange & Act
    dataset = JainDataset()

    # Assert
    assert dataset.dataset_name == "jain"
    assert dataset.output_dir == Path("test_datasets/jain/fragments")


@pytest.mark.unit
def test_jain_dataset_initializes_with_custom_output_dir(tmp_path: Path) -> None:
    """Verify JainDataset can use custom output directory"""
    # Arrange
    custom_dir = tmp_path / "custom_jain_output"

    # Act
    dataset = JainDataset(output_dir=custom_dir)

    # Assert
    assert dataset.output_dir == custom_dir
    assert custom_dir.exists()


@pytest.mark.unit
def test_jain_dataset_returns_full_antibody_fragments() -> None:
    """Verify Jain dataset uses 16 full antibody fragment types"""
    # Arrange
    dataset = JainDataset()

    # Act
    fragment_types = dataset.get_fragment_types()

    # Assert
    assert len(fragment_types) == 16
    assert fragment_types == dataset.FULL_ANTIBODY_FRAGMENTS


# ============================================================================
# Dataset Loading Tests (Using Mock CSV)
# ============================================================================


@pytest.mark.unit
def test_load_data_reads_mock_csv(jain_sample_csv: Path) -> None:
    """Verify load_data can read mock Jain CSV file"""
    # Arrange
    dataset = JainDataset()

    # Act
    df = dataset.load_data(full_csv_path=str(jain_sample_csv), stage="full")

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert "id" in df.columns
    assert "VH_sequence" in df.columns
    assert "VL_sequence" in df.columns
    assert "label" in df.columns


@pytest.mark.unit
def test_load_data_has_required_columns(jain_sample_csv: Path) -> None:
    """Verify loaded data has all required columns"""
    # Arrange
    dataset = JainDataset()

    # Act
    df = dataset.load_data(full_csv_path=str(jain_sample_csv), stage="full")

    # Assert - Core columns
    required_columns = [
        "id",
        "VH_sequence",
        "VL_sequence",
        "label",
        "elisa_flags",
        "psr",
        "ac_sins",
        "hic",
        "fab_tm",
    ]
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"


@pytest.mark.unit
def test_load_data_contains_valid_labels(jain_sample_csv: Path) -> None:
    """Verify labels are binary (0 or 1)"""
    # Arrange
    dataset = JainDataset()

    # Act
    df = dataset.load_data(full_csv_path=str(jain_sample_csv), stage="full")

    # Assert
    assert set(df["label"].unique()).issubset({0, 1})


@pytest.mark.unit
def test_load_data_contains_valid_sequences(jain_sample_csv: Path) -> None:
    """Verify sequences are non-empty strings"""
    # Arrange
    dataset = JainDataset()

    # Act
    df = dataset.load_data(full_csv_path=str(jain_sample_csv), stage="full")

    # Assert
    assert df["VH_sequence"].notna().all()
    assert df["VL_sequence"].notna().all()
    assert all(len(seq) > 0 for seq in df["VH_sequence"])
    assert all(len(seq) > 0 for seq in df["VL_sequence"])


# ============================================================================
# ELISA Filtering Tests
# ============================================================================


@pytest.mark.unit
def test_filter_elisa_1to3_removes_mild_aggregators(jain_sample_csv: Path) -> None:
    """Verify filter_elisa_1to3 removes ELISA flags 1-3"""
    # Arrange
    dataset = JainDataset()
    df = dataset.load_data(full_csv_path=str(jain_sample_csv), stage="full")
    initial_count = len(df)

    # Act
    df_filtered = dataset.filter_elisa_1to3(df)

    # Assert
    # Check that all ELISA 1-3 are removed
    assert not df_filtered["elisa_flags"].isin([1, 2, 3]).any()
    # Check that some antibodies were removed
    assert len(df_filtered) < initial_count


@pytest.mark.unit
def test_filter_elisa_1to3_keeps_flag_0(jain_sample_csv: Path) -> None:
    """Verify filter_elisa_1to3 keeps ELISA flag 0 (clean)"""
    # Arrange
    dataset = JainDataset()
    df = dataset.load_data(full_csv_path=str(jain_sample_csv), stage="full")

    # Act
    df_filtered = dataset.filter_elisa_1to3(df)

    # Assert
    # ELISA flag 0 should still be present
    assert 0 in df_filtered["elisa_flags"].values


@pytest.mark.unit
def test_filter_elisa_1to3_keeps_flag_4plus(jain_sample_csv: Path) -> None:
    """Verify filter_elisa_1to3 keeps ELISA flags 4+ (high polyreactivity)"""
    # Arrange
    dataset = JainDataset()
    df = dataset.load_data(full_csv_path=str(jain_sample_csv), stage="full")

    # Act
    df_filtered = dataset.filter_elisa_1to3(df)

    # Assert
    # ELISA flags 4+ should still be present
    high_flags = df_filtered["elisa_flags"] >= 4
    assert high_flags.any()


@pytest.mark.unit
def test_filter_elisa_1to3_preserves_dataframe_columns(jain_sample_csv: Path) -> None:
    """Verify filtering doesn't drop required columns"""
    # Arrange
    dataset = JainDataset()
    df = dataset.load_data(full_csv_path=str(jain_sample_csv), stage="full")
    original_columns = set(df.columns)

    # Act
    df_filtered = dataset.filter_elisa_1to3(df)

    # Assert
    assert set(df_filtered.columns) == original_columns


# ============================================================================
# Multi-Stage Loading Tests
# ============================================================================


@pytest.mark.unit
def test_load_data_full_stage_returns_all_antibodies(jain_sample_csv: Path) -> None:
    """Verify 'full' stage loads all 10 mock antibodies"""
    # Arrange
    dataset = JainDataset()

    # Act
    df = dataset.load_data(full_csv_path=str(jain_sample_csv), stage="full")

    # Assert
    assert len(df) == 10


@pytest.mark.unit
def test_load_data_ssot_stage_filters_elisa_1to3(jain_sample_csv: Path) -> None:
    """Verify 'ssot' stage applies ELISA 1-3 filtering"""
    # Arrange
    dataset = JainDataset()

    # Act
    df_full = dataset.load_data(full_csv_path=str(jain_sample_csv), stage="full")
    df_ssot = dataset.load_data(full_csv_path=str(jain_sample_csv), stage="ssot")

    # Assert
    # SSOT should have fewer antibodies (ELISA 1-3 removed)
    assert len(df_ssot) < len(df_full)
    # SSOT should not contain ELISA 1-3
    assert not df_ssot["elisa_flags"].isin([1, 2, 3]).any()


@pytest.mark.unit
def test_load_data_parity_stage_is_smallest(jain_sample_csv: Path) -> None:
    """Verify 'parity' stage has fewest antibodies (most filtering)"""
    # Arrange
    dataset = JainDataset()

    # Act - Use non-existent sd03 path to avoid duplicate column merge
    df_full = dataset.load_data(
        full_csv_path=str(jain_sample_csv),
        sd03_csv_path="nonexistent.csv",
        stage="full",
    )
    df_ssot = dataset.load_data(
        full_csv_path=str(jain_sample_csv),
        sd03_csv_path="nonexistent.csv",
        stage="ssot",
    )
    df_parity = dataset.load_data(
        full_csv_path=str(jain_sample_csv),
        sd03_csv_path="nonexistent.csv",
        stage="parity",
    )

    # Assert
    # Parity should have fewest antibodies (multiple filters applied)
    assert len(df_parity) <= len(df_ssot) <= len(df_full)


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.unit
def test_load_data_raises_error_for_missing_file() -> None:
    """Verify load_data raises FileNotFoundError for missing CSV"""
    # Arrange
    dataset = JainDataset()

    # Act & Assert
    with pytest.raises(FileNotFoundError, match="Jain FULL CSV not found"):
        dataset.load_data(full_csv_path="nonexistent_file.csv")


# ============================================================================
# Reclassification Tests
# ============================================================================


@pytest.mark.unit
def test_reclassify_5_antibodies_adds_metadata_columns() -> None:
    """Verify reclassify_5_antibodies adds metadata columns"""
    # Arrange
    dataset = JainDataset()
    df = pd.DataFrame(
        {
            "id": ["bimagrumab", "other_antibody"],
            "label": [0, 0],
            "psr": [0.697, 0.1],
        }
    )

    # Act
    df_reclassified = dataset.reclassify_5_antibodies(df)

    # Assert
    assert "label_original" in df_reclassified.columns
    assert "reclassified" in df_reclassified.columns
    assert "reclassification_reason" in df_reclassified.columns


@pytest.mark.unit
def test_reclassify_5_antibodies_tier_a_psr() -> None:
    """Verify Tier A antibodies (PSR >0.4) are reclassified"""
    # Arrange
    dataset = JainDataset()
    tier_a_antibodies = ["bimagrumab", "bavituximab", "ganitumab"]
    df = pd.DataFrame(
        {
            "id": tier_a_antibodies + ["other_antibody"],
            "label": [0, 0, 0, 0],  # All initially specific
            "psr": [0.697, 0.557, 0.553, 0.1],
        }
    )

    # Act
    df_reclassified = dataset.reclassify_5_antibodies(df)

    # Assert
    # Tier A antibodies should be reclassified to non-specific (label=1)
    for ab_id in tier_a_antibodies:
        ab_row = df_reclassified[df_reclassified["id"] == ab_id]
        assert ab_row["label"].values[0] == 1
        assert bool(ab_row["reclassified"].values[0]) is True
        assert "Tier A" in ab_row["reclassification_reason"].values[0]


@pytest.mark.unit
def test_reclassify_5_antibodies_tier_b_tm() -> None:
    """Verify Tier B antibody (extreme Tm) is reclassified"""
    # Arrange
    dataset = JainDataset()
    df = pd.DataFrame(
        {
            "id": ["eldelumab", "other_antibody"],
            "label": [0, 0],
            "fab_tm": [59.50, 75.0],
        }
    )

    # Act
    df_reclassified = dataset.reclassify_5_antibodies(df)

    # Assert
    eldelumab_row = df_reclassified[df_reclassified["id"] == "eldelumab"]
    assert eldelumab_row["label"].values[0] == 1
    assert bool(eldelumab_row["reclassified"].values[0]) is True
    assert "Tier B" in eldelumab_row["reclassification_reason"].values[0]


@pytest.mark.unit
def test_reclassify_5_antibodies_tier_c_clinical() -> None:
    """Verify Tier C antibody (clinical evidence) is reclassified"""
    # Arrange
    dataset = JainDataset()
    df = pd.DataFrame({"id": ["infliximab", "other_antibody"], "label": [0, 0]})

    # Act
    df_reclassified = dataset.reclassify_5_antibodies(df)

    # Assert
    infliximab_row = df_reclassified[df_reclassified["id"] == "infliximab"]
    assert infliximab_row["label"].values[0] == 1
    assert bool(infliximab_row["reclassified"].values[0]) is True
    assert "Tier C" in infliximab_row["reclassification_reason"].values[0]


@pytest.mark.unit
def test_reclassify_5_antibodies_preserves_other_antibodies() -> None:
    """Verify reclassification doesn't affect other antibodies"""
    # Arrange
    dataset = JainDataset()
    df = pd.DataFrame(
        {
            "id": ["bimagrumab", "other_antibody_1", "other_antibody_2"],
            "label": [0, 0, 1],
            "psr": [0.697, 0.1, 0.3],
        }
    )

    # Act
    df_reclassified = dataset.reclassify_5_antibodies(df)

    # Assert
    # Non-reclassified antibodies should keep original labels
    other_1 = df_reclassified[df_reclassified["id"] == "other_antibody_1"]
    other_2 = df_reclassified[df_reclassified["id"] == "other_antibody_2"]

    assert other_1["label"].values[0] == 0  # Still specific
    assert other_2["label"].values[0] == 1  # Still non-specific
    assert bool(other_1["reclassified"].values[0]) is False
    assert bool(other_2["reclassified"].values[0]) is False


# ============================================================================
# PSR/AC-SINS Removal Tests
# ============================================================================


@pytest.mark.unit
def test_remove_30_by_psr_acsins_keeps_all_nonspecific() -> None:
    """Verify removal only affects specific antibodies, not non-specific"""
    # Arrange
    dataset = JainDataset()
    # Create 89 specific + 27 non-specific = 116 total
    specific = pd.DataFrame(
        {
            "id": [f"spec_{i:03d}" for i in range(89)],
            "label": [0] * 89,
            "psr": [0.0] * 89,  # All PSR=0
            "ac_sins": [i * 0.01 for i in range(89)],  # Varying AC-SINS
        }
    )
    nonspecific = pd.DataFrame(
        {
            "id": [f"nonspec_{i:03d}" for i in range(27)],
            "label": [1] * 27,
            "psr": [0.5] * 27,
            "ac_sins": [0.5] * 27,
        }
    )
    df = pd.concat([specific, nonspecific], ignore_index=True)

    # Act
    df_86 = dataset.remove_30_by_psr_acsins(df)

    # Assert
    # Should have exactly 86 antibodies (59 spec + 27 nonspec)
    assert len(df_86) == 86
    assert (df_86["label"] == 0).sum() == 59  # Specific
    assert (df_86["label"] == 1).sum() == 27  # Non-specific


@pytest.mark.unit
def test_remove_30_by_psr_acsins_sorts_by_psr_primary() -> None:
    """Verify removal prioritizes PSR over AC-SINS"""
    # Arrange
    dataset = JainDataset()
    # Create 89 specific with varying PSR
    specific = pd.DataFrame(
        {
            "id": [f"spec_{i:03d}" for i in range(89)],
            "label": [0] * 89,
            "psr": [
                0.1 * (i // 10) for i in range(89)
            ],  # PSR groups: 0.0, 0.1, 0.2, ...
            "ac_sins": [0.5] * 89,  # Same AC-SINS
        }
    )
    nonspecific = pd.DataFrame(
        {
            "id": [f"nonspec_{i:03d}" for i in range(27)],
            "label": [1] * 27,
            "psr": [0.5] * 27,
            "ac_sins": [0.5] * 27,
        }
    )
    df = pd.concat([specific, nonspecific], ignore_index=True)

    # Act
    df_86 = dataset.remove_30_by_psr_acsins(df)

    # Assert
    specific_kept = df_86[df_86["label"] == 0]
    # Kept specific antibodies should have lower PSR values (bottom 59)
    assert specific_kept["psr"].max() <= 0.6  # Lower PSR preferred


@pytest.mark.unit
def test_remove_30_by_psr_acsins_uses_acsins_tiebreaker() -> None:
    """Verify AC-SINS is used as tiebreaker when PSR is equal"""
    # Arrange
    dataset = JainDataset()
    # Create 89 specific with PSR=0.0, varying AC-SINS
    specific = pd.DataFrame(
        {
            "id": [f"spec_{i:03d}" for i in range(89)],
            "label": [0] * 89,
            "psr": [0.0] * 89,  # All same PSR
            "ac_sins": [i * 0.01 for i in range(89)],  # Varying AC-SINS
        }
    )
    nonspecific = pd.DataFrame(
        {
            "id": [f"nonspec_{i:03d}" for i in range(27)],
            "label": [1] * 27,
            "psr": [0.5] * 27,
            "ac_sins": [0.5] * 27,
        }
    )
    df = pd.concat([specific, nonspecific], ignore_index=True)

    # Act
    df_86 = dataset.remove_30_by_psr_acsins(df)

    # Assert
    specific_kept = df_86[df_86["label"] == 0]
    # When PSR is equal, keep antibodies with lower AC-SINS (bottom 59)
    assert specific_kept["ac_sins"].max() <= 0.6


# ============================================================================
# Integration-like Tests (Full Workflow)
# ============================================================================


@pytest.mark.unit
def test_full_jain_dataset_workflow(jain_sample_csv: Path, tmp_path: Path) -> None:
    """Verify complete Jain dataset initialization and loading workflow"""
    # Arrange
    output_dir = tmp_path / "jain_output"

    # Act - Initialize
    dataset = JainDataset(output_dir=output_dir)

    # Act - Load full stage
    df_full = dataset.load_data(full_csv_path=str(jain_sample_csv), stage="full")

    # Act - Get fragment types
    fragment_types = dataset.get_fragment_types()

    # Assert - Initialization
    assert dataset.dataset_name == "jain"
    assert output_dir.exists()

    # Assert - Data loaded
    assert len(df_full) == 10
    assert "VH_sequence" in df_full.columns
    assert "VL_sequence" in df_full.columns

    # Assert - Fragment types
    assert len(fragment_types) == 16
    assert "VH_only" in fragment_types


@pytest.mark.unit
def test_jain_constants_match_novo_parity() -> None:
    """Verify Jain constants match Novo Nordisk parity requirements"""
    # Arrange & Act
    dataset = JainDataset()

    # Assert
    assert dataset.PSR_THRESHOLD == 0.4
    assert len(dataset.TIER_A_PSR) == 3
    assert dataset.TIER_B_EXTREME_TM == "eldelumab"
    assert dataset.TIER_C_CLINICAL == "infliximab"
