"""
Jain Dataset Loader

Loads preprocessed Jain 2017 therapeutic antibody dataset.

IMPORTANT: This module is for LOADING preprocessed data, not for running
the preprocessing pipeline. The preprocessing scripts that CREATE the data
are in: preprocessing/jain/step2_preprocess_p5e_s2.py

Dataset characteristics:
- Full antibodies (VH + VL)
- 137 FDA-approved/clinical-stage therapeutics
- Multi-stage filtering with biophysical parameters
- Novo Nordisk parity requirements (86 antibodies, [[40, 19], [10, 17]])
- 16 fragment types (full antibody)

Processing Pipeline:
  137 antibodies (FULL)
    ↓ Remove ELISA 1-3 (mild aggregators)
  116 antibodies (SSOT)
    ↓ Reclassify 5 spec→nonspec (PSR>0.4, Tm, clinical)
  89 spec / 27 nonspec
    ↓ Remove 30 by PSR/AC-SINS ranking
  86 antibodies (59 spec / 27 nonspec) - NOVO PARITY

Source:
- test_datasets/jain/processed/jain_with_private_elisa_FULL.csv
- test_datasets/jain/processed/jain_sd03.csv (biophysical data)

Reference:
- Jain et al. (2017), "Biophysical properties of the clinical-stage antibody landscape"
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .base import AntibodyDataset
from .default_paths import JAIN_FULL_CSV, JAIN_OUTPUT_DIR, JAIN_SD03_CSV


class JainDataset(AntibodyDataset):
    """
    Loader for Jain therapeutic antibody dataset.

    This class provides an interface to LOAD preprocessed Jain dataset files.
    It does NOT run the preprocessing pipeline - use preprocessing/jain/step2_preprocess_p5e_s2.py for that.

    The Jain dataset contains FDA-approved and clinical-stage therapeutic antibodies
    with complex multi-stage filtering to achieve Novo Nordisk parity.
    """

    # P5e-S2 Method Constants (Novo Nordisk parity)
    PSR_THRESHOLD = 0.4

    # Reclassification tiers
    TIER_A_PSR = ["bimagrumab", "bavituximab", "ganitumab"]  # PSR >0.4
    TIER_B_EXTREME_TM = "eldelumab"  # Extreme Tm outlier (59.50°C)
    TIER_C_CLINICAL = "infliximab"  # 61% ADA rate + chimeric

    def __init__(
        self, output_dir: Path | None = None, logger: logging.Logger | None = None
    ):
        """
        Initialize Jain dataset loader.

        Args:
            output_dir: Directory containing preprocessed fragment files
            logger: Logger instance
        """
        super().__init__(
            dataset_name="jain",
            output_dir=output_dir or JAIN_OUTPUT_DIR,
            logger=logger,
        )

    def get_fragment_types(self) -> list[str]:
        """
        Return full antibody fragment types.

        Jain contains VH + VL sequences, so we generate all 16 fragment types.

        Returns:
            List of 16 full antibody fragment types
        """
        return self.FULL_ANTIBODY_FRAGMENTS

    def load_data(
        self,
        full_csv_path: str | Path | None = None,
        sd03_csv_path: str | Path | None = None,
        stage: str = "full",
        **_: Any,
    ) -> pd.DataFrame:
        """
        Load Jain dataset from CSV files.

        Args:
            full_csv_path: Path to jain_with_private_elisa_FULL.csv (137 antibodies)
            sd03_csv_path: Path to jain_sd03.csv (biophysical data)
            stage: Which processing stage to load:
                   "full" - 137 antibodies (raw)
                   "ssot" - 116 antibodies (ELISA-filtered)
                   "parity" - 86 antibodies (Novo parity)

        Returns:
            DataFrame with columns: id, VH_sequence, VL_sequence, label, elisa_flags, psr, ac_sins, hic, fab_tm

        Raises:
            FileNotFoundError: If input CSV files not found
        """
        # Default paths
        if full_csv_path is None:
            full_csv_path = JAIN_FULL_CSV
        if sd03_csv_path is None:
            sd03_csv_path = JAIN_SD03_CSV

        full_csv = Path(full_csv_path)
        sd03_csv = Path(sd03_csv_path)

        if not full_csv.exists():
            raise FileNotFoundError(
                f"Jain FULL CSV not found: {full_csv}\n"
                f"Please ensure source data is in test_datasets/jain/processed/"
            )

        # Load main dataset
        self.logger.info(f"Reading Jain FULL dataset from {full_csv}...")
        df = pd.read_csv(full_csv)
        self.logger.info(f"  Loaded {len(df)} antibodies")
        self.logger.info(f"    Specific: {(df['label'] == 0).sum()}")
        self.logger.info(f"    Non-specific: {(df['label'] == 1).sum()}")

        # Standardize column names
        column_mapping = {
            "heavy_seq": "VH_sequence",
            "light_seq": "VL_sequence",
            "vh_sequence": "VH_sequence",  # Support VH-only files
            "vl_sequence": "VL_sequence",  # Support VL-only files
        }
        df = df.rename(columns=column_mapping)

        # Load biophysical data if available
        if sd03_csv.exists():
            self.logger.info(f"Loading biophysical data from {sd03_csv}...")
            sd03 = pd.read_csv(sd03_csv)

            # Merge biophysical columns
            df = df.merge(
                sd03[
                    [
                        "Name",
                        "Poly-Specificity Reagent (PSR) SMP Score (0-1)",
                        "Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average",
                        "HIC Retention Time (Min)a",
                        "Fab Tm by DSF (°C)",
                    ]
                ],
                left_on="id",
                right_on="Name",
                how="left",
            )

            # Rename for easier handling
            df = df.rename(
                columns={
                    "Poly-Specificity Reagent (PSR) SMP Score (0-1)": "psr",
                    "Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average": "ac_sins",
                    "HIC Retention Time (Min)a": "hic",
                    "Fab Tm by DSF (°C)": "fab_tm",
                }
            )
            df = df.drop(columns=["Name"])

            self.logger.info("  Biophysical data merged")
            self.logger.info(f"    Missing PSR: {df['psr'].isna().sum()}")
            self.logger.info(f"    Missing AC-SINS: {df['ac_sins'].isna().sum()}")

        # Apply stage-specific filtering
        if stage == "ssot":
            df = self.filter_elisa_1to3(df)
        elif stage == "parity":
            df = self.filter_elisa_1to3(df)
            df = self.reclassify_5_antibodies(df)
            df = self.remove_30_by_psr_acsins(df)

        return df

    def filter_elisa_1to3(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove ELISA 1-3 (mild aggregators) → 116 antibodies (SSOT).

        ELISA flags 1-3 indicate mild to moderate aggregation in ELISA assays.
        These are filtered out as they don't represent strong enough
        polyreactivity signal for training.

        Args:
            df: Full dataset (137 antibodies)

        Returns:
            Filtered dataset (116 antibodies)
        """
        initial_count = len(df)
        df_filtered = df[~df["elisa_flags"].isin([1, 2, 3])].copy()
        removed_count = initial_count - len(df_filtered)

        self.logger.info("\nFiltering ELISA 1-3 (mild aggregators):")
        self.logger.info(f"  Removed: {removed_count} antibodies")
        self.logger.info(f"  Remaining: {len(df_filtered)} antibodies")

        return df_filtered

    def reclassify_5_antibodies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reclassify 5 specific → non-specific.

        Tier A (PSR-based, 3 antibodies):
          - bimagrumab (PSR=0.697)
          - bavituximab (PSR=0.557)
          - ganitumab (PSR=0.553)
          All have ELISA=0 but PSR >0.4, indicating polyreactivity

        Tier B (Multi-metric, 1 antibody):
          - eldelumab (Tm=59.50°C, extreme thermal instability outlier)

        Tier C (Clinical, 1 antibody):
          - infliximab (61% ADA rate in NEJM study + chimeric + aggregation)

        Args:
            df: 116-antibody dataset

        Returns:
            Dataset with 5 antibodies reclassified (89 spec, 27 nonspec)
        """
        df = df.copy()
        df["label_original"] = df["label"]
        df["reclassified"] = False
        df["reclassification_reason"] = ""

        # Tier A: PSR >0.4
        for ab_id in self.TIER_A_PSR:
            idx = df[df["id"] == ab_id].index
            if len(idx) > 0:
                df.loc[idx, "label"] = 1
                df.loc[idx, "reclassified"] = True
                df.loc[idx, "reclassification_reason"] = "Tier A: PSR >0.4"

        # Tier B: Extreme Tm
        idx = df[df["id"] == self.TIER_B_EXTREME_TM].index
        if len(idx) > 0:
            df.loc[idx, "label"] = 1
            df.loc[idx, "reclassified"] = True
            df.loc[idx, "reclassification_reason"] = "Tier B: Extreme Tm"

        # Tier C: Clinical evidence
        idx = df[df["id"] == self.TIER_C_CLINICAL].index
        if len(idx) > 0:
            df.loc[idx, "label"] = 1
            df.loc[idx, "reclassified"] = True
            df.loc[idx, "reclassification_reason"] = "Tier C: Clinical (61% ADA)"

        spec_count = (df["label"] == 0).sum()
        nonspec_count = (df["label"] == 1).sum()

        self.logger.info("\nReclassified 5 antibodies:")
        self.logger.info(f"  Specific: {spec_count} (expected 89)")
        self.logger.info(f"  Non-specific: {nonspec_count} (expected 27)")

        return df

    def remove_30_by_psr_acsins(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove 30 specific antibodies by PSR primary, AC-SINS tiebreaker.

        Removal strategy:
          1. Sort specific antibodies by PSR descending (primary)
          2. For PSR=0 antibodies, use AC-SINS descending (tiebreaker)
          3. Remove top 30

        Args:
            df: Dataset with 89 specific + 27 non-specific = 116 total

        Returns:
            Final 86-antibody dataset (59 spec + 27 nonspec)
        """
        # Get remaining specific and non-specific antibodies
        specific = df[df["label"] == 0].copy()
        nonspecific = df[df["label"] == 1].copy()

        # Sort by PSR (descending), then AC-SINS (descending), then id (alphabetical)
        specific_sorted = specific.sort_values(
            by=["psr", "ac_sins", "id"], ascending=[False, False, True]
        )

        # Keep bottom 59 specific + all 27 non-specific
        specific_keep = specific_sorted.tail(59)
        df_86 = pd.concat([specific_keep, nonspecific], ignore_index=True)

        # Sort by id for consistency
        df_86 = df_86.sort_values("id").reset_index(drop=True)

        spec_count = (df_86["label"] == 0).sum()
        nonspec_count = (df_86["label"] == 1).sum()

        self.logger.info("\nRemoved 30 specific by PSR/AC-SINS:")
        self.logger.info(f"  Final: {len(df_86)} antibodies")
        self.logger.info(f"  Specific: {spec_count} (expected 59)")
        self.logger.info(f"  Non-specific: {nonspec_count} (expected 27)")

        return df_86


# ========== CONVENIENCE FUNCTIONS FOR LOADING DATA ==========


def load_jain_data(
    full_csv: str | None = None,
    sd03_csv: str | None = None,
    stage: str = "parity",
) -> pd.DataFrame:
    """
    Convenience function to load preprocessed Jain dataset.

    IMPORTANT: This loads PREPROCESSED data. To preprocess raw data, use:
    preprocessing/jain/step2_preprocess_p5e_s2.py

    Args:
        full_csv: Path to jain_with_private_elisa_FULL.csv
        sd03_csv: Path to jain_sd03.csv (biophysical data)
        stage: Processing stage ("full", "ssot", or "parity")

    Returns:
        DataFrame with preprocessed data

    Example:
        >>> from antibody_training_esm.datasets.jain import load_jain_data
        >>> df = load_jain_data(stage="parity")  # 86 antibodies (Novo parity)
        >>> print(f"Loaded {len(df)} sequences")
    """
    dataset = JainDataset()
    return dataset.load_data(
        full_csv_path=full_csv, sd03_csv_path=sd03_csv, stage=stage
    )
