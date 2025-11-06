"""
Shehata Dataset Loader

Loads preprocessed Shehata HIV antibody polyreactivity dataset.

IMPORTANT: This module is for LOADING preprocessed data, not for running
the preprocessing pipeline. The preprocessing scripts that CREATE the data
are in: preprocessing/shehata/step2_extract_fragments.py

Dataset characteristics:
- Full antibodies (VH + VL)
- 398 HIV-specific antibodies from 8 donors
- Binary classification based on PSR (Polyreactivity Screening Reagent) scores
- B cell subset metadata (memory, naive, plasmablast)
- 16 fragment types (full antibody)

Source:
- test_datasets/shehata/raw/shehata-mmc2.xlsx

Reference:
- Shehata et al. (2019), "Affinity Maturation Enhances Antibody Specificity but Compromises Conformational Stability"
  Supplementary Material mmc2.xlsx
"""

from pathlib import Path

import pandas as pd

from .base import AntibodyDataset


class ShehataDataset(AntibodyDataset):
    """
    Loader for Shehata HIV antibody dataset.

    This class provides an interface to LOAD preprocessed Shehata dataset files.
    It does NOT run the preprocessing pipeline - use preprocessing/shehata/step2_extract_fragments.py for that.

    The Shehata dataset contains HIV-specific antibodies from 8 donors, with PSR scores
    measuring polyreactivity. The paper reports 7/398 (1.76%) as non-specific,
    corresponding to the 98.24th percentile threshold.
    """

    # Default PSR threshold (98.24th percentile based on paper: 7/398 non-specific)
    DEFAULT_PSR_PERCENTILE = 0.9824

    def __init__(self, output_dir: Path | None = None, logger=None):
        """
        Initialize Shehata dataset loader.

        Args:
            output_dir: Directory containing preprocessed fragment files
            logger: Logger instance
        """
        super().__init__(
            dataset_name="shehata",
            output_dir=output_dir or Path("test_datasets/shehata/fragments"),
            logger=logger,
        )

    def get_fragment_types(self) -> list[str]:
        """
        Return full antibody fragment types.

        Shehata contains VH + VL sequences, so we generate all 16 fragment types.

        Returns:
            List of 16 full antibody fragment types
        """
        return self.FULL_ANTIBODY_FRAGMENTS

    def calculate_psr_threshold(
        self,
        psr_scores: pd.Series,
        percentile: float | None = None,
    ) -> float:
        """
        Calculate PSR score threshold for binary classification.

        Based on paper: "7 out of 398 antibodies characterised as non-specific"
        This is 1.76% = 98.24th percentile

        Args:
            psr_scores: Series of PSR scores (numeric)
            percentile: Percentile to use (default: 0.9824 for 7/398)

        Returns:
            PSR threshold value
        """
        if percentile is None:
            percentile = self.DEFAULT_PSR_PERCENTILE

        threshold = psr_scores.quantile(percentile)

        self.logger.info("\nPSR Score Analysis:")
        self.logger.info(f"  Valid PSR scores: {psr_scores.notna().sum()}")
        self.logger.info(f"  Mean: {psr_scores.mean():.4f}")
        self.logger.info(f"  Median: {psr_scores.median():.4f}")
        self.logger.info(f"  75th percentile: {psr_scores.quantile(0.75):.4f}")
        self.logger.info(f"  95th percentile: {psr_scores.quantile(0.95):.4f}")
        self.logger.info(f"  Max: {psr_scores.max():.4f}")
        self.logger.info(f"\n  PSR = 0: {(psr_scores == 0).sum()} antibodies")
        self.logger.info(f"  PSR > 0: {(psr_scores > 0).sum()} antibodies")
        self.logger.info(
            "\n  Paper reports: 7/398 non-specific (~1.76%, 98.24th percentile)"
        )
        self.logger.info(f"  Calculated threshold: {threshold:.4f}")

        return threshold

    def load_data(  # type: ignore[override]
        self,
        excel_path: str | None = None,
        psr_threshold: float | None = None,
    ) -> pd.DataFrame:
        """
        Load Shehata dataset from Excel file.

        Args:
            excel_path: Path to shehata-mmc2.xlsx
            psr_threshold: PSR score threshold for binary classification.
                          If None, calculates 98.24th percentile automatically.

        Returns:
            DataFrame with columns: id, VH_sequence, VL_sequence, label, psr_score, b_cell_subset

        Raises:
            FileNotFoundError: If Excel file not found
        """
        # Default path
        if excel_path is None:
            excel_path = "test_datasets/shehata/raw/shehata-mmc2.xlsx"

        excel_file = Path(excel_path)
        if not excel_file.exists():
            raise FileNotFoundError(
                f"Shehata Excel file not found: {excel_file}\n"
                f"Please ensure mmc2.xlsx is in test_datasets/shehata/raw/"
            )

        # Load Excel
        self.logger.info(f"Reading Excel file: {excel_file}")
        df = pd.read_excel(excel_file)
        self.logger.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

        # Sanitize sequences (remove IMGT gap characters)
        self.logger.info("Sanitizing sequences (removing gaps)...")
        vh_original = df["VH Protein"].copy()
        vl_original = df["VL Protein"].copy()

        df["VH Protein"] = df["VH Protein"].apply(
            lambda x: self.sanitize_sequence(x) if pd.notna(x) else x
        )
        df["VL Protein"] = df["VL Protein"].apply(
            lambda x: self.sanitize_sequence(x) if pd.notna(x) else x
        )

        # Count gaps removed
        gaps_vh = sum(str(s).count("-") if pd.notna(s) else 0 for s in vh_original)
        gaps_vl = sum(str(s).count("-") if pd.notna(s) else 0 for s in vl_original)

        if gaps_vh > 0 or gaps_vl > 0:
            self.logger.info(f"  Removed {gaps_vh} gap characters from VH sequences")
            self.logger.info(f"  Removed {gaps_vl} gap characters from VL sequences")

        # Drop rows without sequences (Excel metadata/footnotes)
        before_drop = len(df)
        df = df.dropna(subset=["VH Protein", "VL Protein"], how="all")
        dropped = before_drop - len(df)
        if dropped:
            self.logger.info(
                f"  Dropped {dropped} rows without VH/VL sequences (metadata)"
            )

        # Convert PSR scores to numeric
        psr_numeric = pd.to_numeric(df["PSR Score"], errors="coerce")
        invalid_psr_mask = psr_numeric.isna()

        if invalid_psr_mask.any():
            dropped_ids = df.loc[invalid_psr_mask, "Clone name"].tolist()
            self.logger.warning(
                f"  Dropping {invalid_psr_mask.sum()} antibodies without numeric PSR scores: "
                f"{', '.join(dropped_ids)}"
            )
            df = df.loc[~invalid_psr_mask].reset_index(drop=True)
            psr_numeric = psr_numeric.loc[~invalid_psr_mask].reset_index(drop=True)

        # Calculate PSR threshold if not provided
        if psr_threshold is None:
            psr_threshold = self.calculate_psr_threshold(psr_numeric)
        else:
            self.logger.info(f"Using provided PSR threshold: {psr_threshold}")

        # Create standardized DataFrame
        df_output = pd.DataFrame(
            {
                "id": df["Clone name"],
                "VH_sequence": df["VH Protein"],
                "VL_sequence": df["VL Protein"],
                "label": (psr_numeric > psr_threshold).astype(int),
                "psr_score": psr_numeric,
                "b_cell_subset": df["B cell subset"],
            }
        )

        # Label distribution
        self.logger.info("\nLabel distribution:")
        label_counts = df_output["label"].value_counts().sort_index()
        for label, count in label_counts.items():
            label_name = "Specific" if label == 0 else "Non-specific"
            percentage = (count / len(df_output)) * 100
            self.logger.info(
                f"  {label_name} (label={label}): {count} ({percentage:.1f}%)"
            )

        # B cell subset distribution
        self.logger.info("\nB cell subset distribution:")
        subset_counts = df_output["b_cell_subset"].value_counts()
        for subset, count in subset_counts.items():
            self.logger.info(f"  {subset}: {count}")

        return df_output


# ========== CONVENIENCE FUNCTIONS FOR LOADING DATA ==========


def load_shehata_data(
    excel_path: str | None = None,
    psr_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Convenience function to load preprocessed Shehata dataset.

    IMPORTANT: This loads PREPROCESSED data. To preprocess raw data, use:
    preprocessing/shehata/step2_extract_fragments.py

    Args:
        excel_path: Path to shehata-mmc2.xlsx
        psr_threshold: PSR threshold for classification (None = auto-calculate)

    Returns:
        DataFrame with preprocessed data

    Example:
        >>> from antibody_training_esm.datasets.shehata import load_shehata_data
        >>> df = load_shehata_data()
        >>> print(f"Loaded {len(df)} sequences")
    """
    dataset = ShehataDataset()
    return dataset.load_data(excel_path=excel_path, psr_threshold=psr_threshold)
