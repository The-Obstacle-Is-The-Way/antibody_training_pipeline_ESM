"""GDPa1 dataset loader for Ginkgo 2025 Antibody Developability Competition.

This module provides dataset loaders for the Ginkgo competition training set (GDPa1)
and private test set.

References:
    - Competition page: https://huggingface.co/spaces/ginkgo-datapoints/abdev-benchmark
    - Dataset: https://huggingface.co/datasets/ginkgo-datapoints/GDPa1
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from antibody_training_esm.datasets.base import AntibodyDataset


class GinkgoDataset(AntibodyDataset):
    """
    Ginkgo GDPa1 dataset for antibody developability competition.

    This dataset contains 246 antibodies with measurements for 5 developability properties:
    - PR_CHO: Polyreactivity (CHO cell lysate)
    - HIC: Hydrophobic interaction chromatography
    - AC-SINS_pH7.4: Self-association
    - Tm2: Thermostability (melting temperature)
    - Titer: Expression yield

    The competition requires using predefined cross-validation folds to prevent
    data leakage from similar antibodies.

    Example:
        >>> dataset = GinkgoDataset(target_property='PR_CHO')
        >>> sequences, labels, folds = dataset.get_sequences_and_labels()
        >>> print(f"Loaded {len(sequences)} antibodies")
        Loaded 197 antibodies

    Attributes:
        assay_file: Path to CSV with assay measurements and sequences
        fold_file: Path to CSV with predefined fold assignments
        target_property: Property to predict (PR_CHO, HIC, Tm2, etc.)
    """

    DEFAULT_ASSAY_FILE = "train_datasets/ginkgo/GDPa1_v1.2_20250814.csv"
    DEFAULT_FOLD_FILE = "train_datasets/ginkgo/GDPa1_v1.2_sequences.csv"

    def __init__(
        self,
        assay_file: str | None = None,
        fold_file: str | None = None,
        target_property: str = "PR_CHO",
    ) -> None:
        """
        Initialize GDPa1 dataset loader.

        Args:
            assay_file: Path to GDPa1_v1.2_20250814.csv (assay data + sequences)
            fold_file: Path to GDPa1_v1.2_sequences.csv (fold assignments)
            target_property: Property to predict (PR_CHO, HIC, AC-SINS_pH7.4, Tm2, Titer)

        Raises:
            FileNotFoundError: If dataset files don't exist
        """
        self.assay_file = Path(assay_file or self.DEFAULT_ASSAY_FILE)
        self.fold_file = Path(fold_file or self.DEFAULT_FOLD_FILE)
        self.target_property = target_property

        if not self.assay_file.exists():
            raise FileNotFoundError(
                f"Assay file not found: {self.assay_file}\n"
                f"Download from: https://huggingface.co/datasets/ginkgo-datapoints/GDPa1"
            )

        if not self.fold_file.exists():
            raise FileNotFoundError(
                f"Fold file not found: {self.fold_file}\n"
                f"Download from: https://huggingface.co/datasets/ginkgo-datapoints/GDPa1"
            )

    def get_fragment_types(self) -> list[str]:
        """
        Return available antibody fragment types.

        GDPa1 contains VH + VL sequences, so we support all 16 fragment types
        (VH, VL, CDRs, FWRs, and combinations).

        Returns:
            List of 16 full antibody fragment types
        """
        return self.FULL_ANTIBODY_FRAGMENTS

    def load_data(self, **_kwargs: Any) -> pd.DataFrame:
        """
        Load GDPa1 dataset with fold assignments.

        Merges assay data with predefined fold assignments and filters out
        rows with missing target values.

        Returns:
            DataFrame with columns:
                - antibody_id: Unique identifier (e.g., "GDPa1-001")
                - antibody_name: Clinical name (e.g., "abagovomab")
                - vh_protein_sequence: Variable heavy chain sequence
                - vl_protein_sequence: Variable light chain sequence
                - {target_property}: Target measurement (e.g., PR_CHO)
                - fold: CV fold assignment (0, 1, 2, 3, 4)
                - ... other assay measurements

        Notes:
            - Rows with missing target values are automatically dropped
            - The competition requires using the predefined folds for CV
        """
        # Load assay data (contains target measurements)
        assay_data = pd.read_csv(self.assay_file)

        # Load fold assignments from sequences file
        fold_data = pd.read_csv(self.fold_file)

        # Drop fold columns from assay_data to avoid merge conflicts
        # (both CSVs have hierarchical_cluster_IgG_isotype_stratified_fold)
        fold_cols_to_drop = [col for col in assay_data.columns if "fold" in col.lower()]
        assay_data = assay_data.drop(columns=fold_cols_to_drop, errors="ignore")

        # Merge assay data with fold assignments
        merged = assay_data.merge(
            fold_data[
                ["antibody_id", "hierarchical_cluster_IgG_isotype_stratified_fold"]
            ],
            on="antibody_id",
            how="inner",
        )

        # Rename fold column for clarity
        merged = merged.rename(
            columns={"hierarchical_cluster_IgG_isotype_stratified_fold": "fold"}
        )

        # Filter out rows with missing target values
        before_count = len(merged)
        merged = merged.dropna(subset=[self.target_property])
        after_count = len(merged)

        if before_count != after_count:
            dropped = before_count - after_count
            print(
                f"Dropped {dropped} antibodies with missing {self.target_property} "
                f"({after_count}/{before_count} remaining)"
            )

        return merged

    def get_sequences_and_labels(
        self,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """
        Get sequences, labels, and fold assignments for training.

        Returns:
            Tuple of (sequences, labels, folds):
                - sequences: List of VH protein sequences
                - labels: Array of continuous target values
                - folds: Array of fold assignments (0-4)

        Example:
            >>> dataset = GinkgoDataset(target_property='PR_CHO')
            >>> sequences, labels, folds = dataset.get_sequences_and_labels()
            >>> print(f"Label range: [{labels.min():.3f}, {labels.max():.3f}]")
            Label range: [0.000, 0.547]
        """
        df = self.load_data()

        sequences = df["vh_protein_sequence"].tolist()
        labels = df[self.target_property].values.astype(np.float64)
        folds = df["fold"].values.astype(np.int64)

        return sequences, labels, folds


class GinkgoTestSet:
    """
    Loader for Ginkgo competition private test set.

    The test set contains 80 antibodies with sequences but no labels.
    Predictions on this set are used for final competition scoring.

    Example:
        >>> test_set = GinkgoTestSet()
        >>> sequences, names = test_set.get_sequences()
        >>> print(f"Loaded {len(sequences)} test antibodies")
        Loaded 80 test antibodies

    Attributes:
        test_file: Path to heldout-set-sequences.csv
    """

    DEFAULT_TEST_FILE = "test_datasets/gingko/heldout-set-sequences.csv"

    def __init__(self, test_file: str | None = None) -> None:
        """
        Initialize test set loader.

        Args:
            test_file: Path to heldout-set-sequences.csv

        Raises:
            FileNotFoundError: If test file doesn't exist
        """
        self.test_file = Path(test_file or self.DEFAULT_TEST_FILE)

        if not self.test_file.exists():
            raise FileNotFoundError(
                f"Test file not found: {self.test_file}\n"
                f"Download from competition page 'Submit' tab:\n"
                f"https://huggingface.co/spaces/ginkgo-datapoints/abdev-benchmark"
            )

    def load_data(self) -> pd.DataFrame:
        """
        Load private test set sequences.

        Returns:
            DataFrame with columns:
                - antibody_id: Unique identifier
                - antibody_name: Name for submission CSV
                - vh_protein_sequence: Variable heavy chain
                - vl_protein_sequence: Variable light chain

        Note:
            No target labels are included - this is for prediction only.
        """
        return pd.read_csv(self.test_file)

    def get_sequences(self) -> tuple[list[str], list[str]]:
        """
        Get test set sequences and antibody names.

        Returns:
            Tuple of (sequences, antibody_names):
                - sequences: List of VH protein sequences
                - antibody_names: List of antibody names (for submission CSV)

        Example:
            >>> test_set = GinkgoTestSet()
            >>> sequences, names = test_set.get_sequences()
            >>> print(f"First antibody: {names[0]}")
        """
        df = self.load_data()

        sequences = df["vh_protein_sequence"].tolist()
        antibody_names = df["antibody_name"].tolist()

        return sequences, antibody_names
