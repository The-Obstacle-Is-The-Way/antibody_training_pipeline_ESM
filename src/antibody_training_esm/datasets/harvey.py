"""
Harvey Dataset Preprocessor

Handles preprocessing of the Harvey nanobody polyreactivity dataset.

Dataset characteristics:
- Nanobodies (VHH only, no light chain)
- High-throughput screen data (141,474 sequences)
- Binary classification: high/low polyreactivity
- IMGT-numbered positions in raw data
- 6 fragment types (VHH-specific)

Source:
- test_datasets/harvey/raw/high_polyreactivity_high_throughput.csv
- test_datasets/harvey/raw/low_polyreactivity_high_throughput.csv

Reference:
- Harvey et al., Engineering highly expressed antibodies for nanobody discovery platforms
"""

from pathlib import Path

import pandas as pd

from .base import AntibodyDataset


class HarveyDataset(AntibodyDataset):
    """
    Preprocessor for Harvey nanobody dataset.

    This dataset contains VHH sequences (heavy chain only, no light chain) from
    a high-throughput polyreactivity screen. Sequences are provided with IMGT
    numbering and pre-extracted CDR regions.
    """

    def __init__(self, output_dir: Path | None = None, logger=None):
        """
        Initialize Harvey dataset preprocessor.

        Args:
            output_dir: Directory to write processed outputs
            logger: Logger instance
        """
        super().__init__(
            dataset_name="harvey",
            output_dir=output_dir or Path("train_datasets/harvey"),
            logger=logger,
        )

    def get_fragment_types(self) -> list[str]:
        """
        Return nanobody-specific fragment types.

        Harvey contains VHH sequences only (no light chain), so we generate
        6 fragment types instead of the full 16.

        Returns:
            List of 6 nanobody fragment types
        """
        return self.NANOBODY_FRAGMENTS

    def extract_sequence_from_imgt(self, row: pd.Series, imgt_cols: list[str]) -> str:
        """
        Extract full sequence from IMGT-numbered position columns.

        The Harvey raw data contains columns "1" through "128" representing
        IMGT numbering positions. This method concatenates non-gap positions
        to reconstruct the full sequence.

        Args:
            row: DataFrame row with IMGT position columns
            imgt_cols: List of column names ['1', '2', ..., '128']

        Returns:
            Full sequence string with gaps removed
        """
        positions = []
        for col in imgt_cols:
            if col in row and pd.notna(row[col]) and row[col] != "-":
                positions.append(row[col])
        return "".join(positions)

    def load_data(
        self,
        high_csv_path: str | None = None,
        low_csv_path: str | None = None,
    ) -> pd.DataFrame:
        """
        Load Harvey dataset from high/low polyreactivity CSV files.

        Args:
            high_csv_path: Path to high_polyreactivity_high_throughput.csv
            low_csv_path: Path to low_polyreactivity_high_throughput.csv

        Returns:
            DataFrame with columns: id, VH_sequence, label

        Raises:
            FileNotFoundError: If input CSV files not found
        """
        # Default paths
        if high_csv_path is None:
            high_csv_path = (
                "test_datasets/harvey/raw/high_polyreactivity_high_throughput.csv"
            )
        if low_csv_path is None:
            low_csv_path = (
                "test_datasets/harvey/raw/low_polyreactivity_high_throughput.csv"
            )

        # Validate paths
        high_csv = Path(high_csv_path)
        low_csv = Path(low_csv_path)

        if not high_csv.exists():
            raise FileNotFoundError(
                f"High polyreactivity CSV not found: {high_csv}\n"
                f"Please ensure raw files are in test_datasets/harvey/raw/"
            )

        if not low_csv.exists():
            raise FileNotFoundError(
                f"Low polyreactivity CSV not found: {low_csv}\n"
                f"Please ensure raw files are in test_datasets/harvey/raw/"
            )

        # Load datasets
        self.logger.info(f"Reading high polyreactivity data from {high_csv}...")
        df_high = pd.read_csv(high_csv)
        self.logger.info(f"  Loaded {len(df_high)} high polyreactivity sequences")

        self.logger.info(f"Reading low polyreactivity data from {low_csv}...")
        df_low = pd.read_csv(low_csv)
        self.logger.info(f"  Loaded {len(df_low)} low polyreactivity sequences")

        # IMGT position columns (1-128)
        imgt_cols = [str(i) for i in range(1, 129)]

        # Extract full sequences from IMGT positions
        self.logger.info("Extracting sequences from IMGT positions...")
        df_high["VH_sequence"] = df_high.apply(
            lambda row: self.extract_sequence_from_imgt(row, imgt_cols), axis=1
        )
        df_low["VH_sequence"] = df_low.apply(
            lambda row: self.extract_sequence_from_imgt(row, imgt_cols), axis=1
        )

        # Add binary labels
        df_high["label"] = 1  # high polyreactivity = non-specific
        df_low["label"] = 0  # low polyreactivity = specific

        # Combine datasets
        self.logger.info("Combining high and low polyreactivity datasets...")
        df_combined = pd.concat([df_high, df_low], ignore_index=True)

        # Create sequence IDs
        df_combined["id"] = [f"harvey_{i:06d}" for i in range(len(df_combined))]

        # Select standardized columns
        df_output = df_combined[["id", "VH_sequence", "label"]].copy()

        # Filter out empty sequences
        empty_mask = df_output["VH_sequence"].str.len() == 0
        if empty_mask.any():
            n_empty = empty_mask.sum()
            self.logger.warning(f"Removing {n_empty} sequences with zero length")
            df_output = df_output[~empty_mask].reset_index(drop=True)

        self.logger.info(f"Combined dataset: {len(df_output)} sequences")
        self.logger.info(
            f"  High polyreactivity (label=1): {(df_output['label'] == 1).sum()}"
        )
        self.logger.info(
            f"  Low polyreactivity (label=0): {(df_output['label'] == 0).sum()}"
        )

        # Sequence length stats
        seq_lengths = df_output["VH_sequence"].str.len()
        self.logger.info(
            f"Sequence length range: {seq_lengths.min()}-{seq_lengths.max()} aa "
            f"(mean: {seq_lengths.mean():.1f})"
        )

        return df_output


# ========== CONVENIENCE FUNCTIONS FOR STANDALONE USE ==========


def preprocess_harvey(
    high_csv: str | None = None,
    low_csv: str | None = None,
    output_dir: str | None = None,
) -> pd.DataFrame:
    """
    Convenience function to preprocess Harvey dataset.

    This function provides a simple interface for standalone preprocessing scripts.

    Args:
        high_csv: Path to high polyreactivity CSV
        low_csv: Path to low polyreactivity CSV
        output_dir: Output directory for processed files

    Returns:
        Processed DataFrame

    Example:
        >>> from antibody_training_esm.datasets.harvey import preprocess_harvey
        >>> df = preprocess_harvey()
        >>> print(f"Processed {len(df)} sequences")
    """
    dataset = HarveyDataset(output_dir=Path(output_dir) if output_dir else None)
    return dataset.process(high_csv_path=high_csv, low_csv_path=low_csv)
