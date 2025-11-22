"""
Integration tests for Jain dataset stage filtering.

Tests the real data pipeline through different filtering stages:
- full: 137 antibodies (no filtering)
- ssot: 116 antibodies (ELISA 1-3 removed)
- parity: 86 antibodies (ELISA 1-3 + reclassify 5 + remove 30)

No mocks - tests actual data files and filtering logic.
"""

from pathlib import Path

import pytest

from antibody_training_esm.datasets.jain import JainDataset


class TestJainStageFiltering:
    """Test Jain dataset filtering stages with real data."""

    @pytest.fixture
    def jain_dataset(self) -> JainDataset:
        """Create JainDataset instance."""
        return JainDataset()

    @pytest.fixture
    def jain_full_csv(self) -> str:
        """Path to Jain full dataset."""
        path = Path("data/test/jain/processed/Therapeutics_VH_VL_with_ELISA_labels.csv")
        if not path.exists():
            pytest.skip(f"Jain full CSV not found: {path}")
        return str(path)

    @pytest.fixture
    def jain_sd03_csv(self) -> str | None:
        """Path to Jain biophysical data."""
        path = Path("data/test/jain/processed/Therapeutics_SD03.csv")
        # SD03 is optional - don't skip if missing
        return str(path) if path.exists() else None

    def test_full_stage_no_filtering(
        self, jain_dataset: JainDataset, jain_full_csv: str, jain_sd03_csv: str | None
    ) -> None:
        """
        Test stage='full' returns unfiltered dataset.

        Expected: 137 antibodies (all from raw data)
        """
        df = jain_dataset.load_data(
            full_csv_path=jain_full_csv, sd03_csv_path=jain_sd03_csv, stage="full"
        )

        # Should return all 137 antibodies
        assert len(df) == 137, f"Expected 137 antibodies in full stage, got {len(df)}"

        # Must have required columns
        assert "sequence" in df.columns, "Missing 'sequence' column"
        assert "label" in df.columns, "Missing 'label' column"
        assert "id" in df.columns, "Missing 'id' column"

        # Both classes should be present
        assert (df["label"] == 0).sum() > 0, "No specific antibodies found"
        assert (df["label"] == 1).sum() > 0, "No non-specific antibodies found"

    def test_ssot_stage_removes_elisa_1_to_3(
        self, jain_dataset: JainDataset, jain_full_csv: str, jain_sd03_csv: str | None
    ) -> None:
        """
        Test stage='ssot' filters ELISA 1-3 (mild aggregators).

        Expected: 116 antibodies (137 - 21 ELISA 1-3)
        Filters: ELISA flags 1, 2, 3 (mild to moderate aggregation)
        """
        df = jain_dataset.load_data(
            full_csv_path=jain_full_csv, sd03_csv_path=jain_sd03_csv, stage="ssot"
        )

        # Should filter down to 116 antibodies
        assert len(df) == 116, f"Expected 116 antibodies in ssot stage, got {len(df)}"

        # No ELISA 1-3 flags should remain
        if "elisa_flags" in df.columns:
            assert not df["elisa_flags"].isin([1, 2, 3]).any(), (
                "Found ELISA 1-3 flags after filtering (should be removed)"
            )

        # Both classes still present
        specific_count = (df["label"] == 0).sum()
        nonspecific_count = (df["label"] == 1).sum()

        assert specific_count > 0, "No specific antibodies after ELISA filtering"
        assert nonspecific_count > 0, "No non-specific antibodies after ELISA filtering"

        # Sanity: total should equal 116
        assert specific_count + nonspecific_count == 116

    def test_parity_stage_novo_nordisk_filtering(
        self, jain_dataset: JainDataset, jain_full_csv: str, jain_sd03_csv: str | None
    ) -> None:
        """
        Test stage='parity' applies full Novo Nordisk filtering.

        Expected: 86 antibodies (Novo parity benchmark)
        Filters:
          1. ELISA 1-3 removed (137 → 116)
          2. Reclassify 5 specific → non-specific (116 → 116, labels change)
          3. Remove 30 by PSR/AC-SINS (116 → 86)

        Final distribution: 59 specific, 27 non-specific
        """
        df = jain_dataset.load_data(
            full_csv_path=jain_full_csv, sd03_csv_path=jain_sd03_csv, stage="parity"
        )

        # Should produce Novo parity set: 86 antibodies
        assert len(df) == 86, (
            f"Expected 86 antibodies in parity stage (Novo benchmark), got {len(df)}"
        )

        # Check class distribution (Novo parity)
        specific_count = (df["label"] == 0).sum()
        nonspecific_count = (df["label"] == 1).sum()

        # Novo Nordisk parity distribution
        assert specific_count == 59, (
            f"Expected 59 specific antibodies (Novo parity), got {specific_count}"
        )
        assert nonspecific_count == 27, (
            f"Expected 27 non-specific antibodies (Novo parity), got {nonspecific_count}"
        )

        # Sanity: total should equal 86
        assert specific_count + nonspecific_count == 86

    def test_stage_filtering_is_deterministic(
        self, jain_dataset: JainDataset, jain_full_csv: str, jain_sd03_csv: str | None
    ) -> None:
        """
        Test that stage filtering produces deterministic results.

        Multiple calls to the same stage should produce identical counts.
        """
        # Load parity stage twice
        df1 = jain_dataset.load_data(
            full_csv_path=jain_full_csv, sd03_csv_path=jain_sd03_csv, stage="parity"
        )
        df2 = jain_dataset.load_data(
            full_csv_path=jain_full_csv, sd03_csv_path=jain_sd03_csv, stage="parity"
        )

        # Should produce identical results
        assert len(df1) == len(df2), (
            "Stage filtering is non-deterministic (count differs)"
        )
        assert (df1["label"] == 0).sum() == (df2["label"] == 0).sum(), (
            "Stage filtering is non-deterministic (specific count differs)"
        )
        assert (df1["label"] == 1).sum() == (df2["label"] == 1).sum(), (
            "Stage filtering is non-deterministic (non-specific count differs)"
        )

    def test_stages_form_strict_subset_hierarchy(
        self, jain_dataset: JainDataset, jain_full_csv: str, jain_sd03_csv: str | None
    ) -> None:
        """
        Test that filtering stages form strict subsets.

        full ⊃ ssot ⊃ parity
        (137 ⊃ 116 ⊃ 86)
        """
        df_full = jain_dataset.load_data(
            full_csv_path=jain_full_csv, sd03_csv_path=jain_sd03_csv, stage="full"
        )
        df_ssot = jain_dataset.load_data(
            full_csv_path=jain_full_csv, sd03_csv_path=jain_sd03_csv, stage="ssot"
        )
        df_parity = jain_dataset.load_data(
            full_csv_path=jain_full_csv, sd03_csv_path=jain_sd03_csv, stage="parity"
        )

        # Verify hierarchy: full > ssot > parity
        assert len(df_full) > len(df_ssot), (
            f"full stage ({len(df_full)}) should have more antibodies than ssot ({len(df_ssot)})"
        )
        assert len(df_ssot) > len(df_parity), (
            f"ssot stage ({len(df_ssot)}) should have more antibodies than parity ({len(df_parity)})"
        )

        # Verify exact counts
        assert len(df_full) == 137
        assert len(df_ssot) == 116
        assert len(df_parity) == 86

    def test_invalid_stage_raises_value_error(
        self, jain_dataset: JainDataset, jain_full_csv: str, jain_sd03_csv: str | None
    ) -> None:
        """Test that invalid stage parameter raises ValueError."""
        with pytest.raises(ValueError, match="Invalid stage"):
            jain_dataset.load_data(
                full_csv_path=jain_full_csv,
                sd03_csv_path=jain_sd03_csv,
                stage="invalid_stage",  # Should raise
            )

    def test_p5e_s2_suffix_preserves_parity_count(
        self, jain_dataset: JainDataset
    ) -> None:
        """
        Test that _p5e_s2 suffix files contain Novo parity set.

        Files with _p5e_s2 suffix should already be filtered to 86 antibodies.
        """
        parity_path = Path("data/test/jain/canonical/VH_only_jain_86_p5e_s2.csv")
        if not parity_path.exists():
            pytest.skip(f"Parity file not found: {parity_path}")

        df = jain_dataset.load_data(str(parity_path))

        # Should be Novo parity set
        assert len(df) == 86, (
            f"_p5e_s2 suffix file should contain 86 antibodies, got {len(df)}"
        )

        # Check distribution
        specific = (df["label"] == 0).sum()
        nonspecific = (df["label"] == 1).sum()

        assert specific == 59, f"Expected 59 specific in parity set, got {specific}"
        assert nonspecific == 27, (
            f"Expected 27 non-specific in parity set, got {nonspecific}"
        )
