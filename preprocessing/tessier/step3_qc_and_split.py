"""Quality control and train/val split for Tessier 2024 dataset.

Stage 3 of 3: tessier_annotated.csv → canonical fragment CSVs

Input:
- train_datasets/tessier/annotated/tessier_annotated.csv (~244k sequences)

Outputs:
- train_datasets/tessier/canonical/VH_only_tessier_training.csv (~196k sequences)
- train_datasets/tessier/canonical/VH_only_tessier_validation.csv (~49k sequences)
- train_datasets/tessier/canonical/[14 other fragment CSVs]
- train_datasets/tessier/annotated/qc_filtered_sequences.txt

Quality filters:
1. Remove sequences with 'X' in CDRs (ambiguous residues)
2. Remove sequences with empty CDRs (annotation artifacts)
3. Remove sequences with CDR3 length > 30 aa (outliers)

Train/val split: 80% train, 20% val (stratified by label)
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
INPUT_FILE = Path("train_datasets/tessier/annotated/tessier_annotated.csv")
OUTPUT_DIR = Path("train_datasets/tessier/annotated")
QC_LOG = Path("train_datasets/tessier/annotated/qc_filtered_sequences.txt")

# QC parameters
MAX_CDR3_LENGTH = 30
TRAIN_RATIO = 0.8
RANDOM_STATE = 42

# Fragment definitions (16 total)
# Following standard naming convention: H-CDR1, VH+VL (matches Boughter/Jain/Harvey)
FRAGMENT_DEFINITIONS = {
    # Individual fragments
    "VH_only": "full_seq_H",
    "VL_only": "full_seq_L",
    "H-CDR1": "cdr1_aa_H",
    "H-CDR2": "cdr2_aa_H",
    "H-CDR3": "cdr3_aa_H",
    "L-CDR1": "cdr1_aa_L",
    "L-CDR2": "cdr2_aa_L",
    "L-CDR3": "cdr3_aa_L",
    "H-FWRs": "fwrs_H",
    "L-FWRs": "fwrs_L",
    # Combined fragments
    "VH+VL": "vh_vl",
    "H-CDRs": "cdrs_H",
    "L-CDRs": "cdrs_L",
    "All-CDRs": "all_cdrs",
    "All-FWRs": "all_fwrs",
    # Full (same as VH+VL for Tessier - no nanobodies)
    "Full": "vh_vl",
}


def apply_qc_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply quality control filters.

    Args:
        df: Annotated DataFrame

    Returns:
        Tuple of (filtered_df, filtered_ids)
    """
    filtered_ids = []
    n_original = len(df)

    # Filter 1: Remove sequences with 'X' in CDRs
    has_x_in_cdrs = df.apply(
        lambda row: any(
            "X" in str(row[col])
            for col in [
                "cdr1_aa_H",
                "cdr2_aa_H",
                "cdr3_aa_H",
                "cdr1_aa_L",
                "cdr2_aa_L",
                "cdr3_aa_L",
            ]
        ),
        axis=1,
    )
    n_x_filtered = has_x_in_cdrs.sum()
    filtered_ids.extend(df[has_x_in_cdrs]["antibody_id"].tolist())
    df = df[~has_x_in_cdrs]
    logger.info(f"  Filter 1 (X in CDRs): Removed {n_x_filtered} sequences")

    # Filter 2: Remove sequences with empty CDRs
    has_empty_cdrs = df.apply(
        lambda row: any(
            len(str(row[col])) == 0
            for col in [
                "cdr1_aa_H",
                "cdr2_aa_H",
                "cdr3_aa_H",
                "cdr1_aa_L",
                "cdr2_aa_L",
                "cdr3_aa_L",
            ]
        ),
        axis=1,
    )
    n_empty_filtered = has_empty_cdrs.sum()
    filtered_ids.extend(df[has_empty_cdrs]["antibody_id"].tolist())
    df = df[~has_empty_cdrs]
    logger.info(f"  Filter 2 (empty CDRs): Removed {n_empty_filtered} sequences")

    # Filter 3: Remove sequences with CDR3 length > 30 aa
    cdr3_h_too_long = df["cdr3_aa_H"].str.len() > MAX_CDR3_LENGTH
    cdr3_l_too_long = df["cdr3_aa_L"].str.len() > MAX_CDR3_LENGTH
    cdr3_outliers = cdr3_h_too_long | cdr3_l_too_long
    n_cdr3_filtered = cdr3_outliers.sum()
    filtered_ids.extend(df[cdr3_outliers]["antibody_id"].tolist())
    df = df[~cdr3_outliers]
    logger.info(
        f"  Filter 3 (CDR3 length > {MAX_CDR3_LENGTH}): Removed {n_cdr3_filtered} sequences"
    )

    # Summary
    n_final = len(df)
    n_total_filtered = n_original - n_final
    retention_rate = n_final / n_original * 100
    logger.info(f"  Total filtered: {n_total_filtered} ({100 - retention_rate:.1f}%)")
    logger.info(f"  Retention rate: {retention_rate:.1f}%")

    return df, filtered_ids


def create_train_val_split(df: pd.DataFrame) -> pd.DataFrame:
    """Create stratified train/val split.

    Args:
        df: Filtered DataFrame

    Returns:
        DataFrame with 'split' column ('train' or 'val')
    """
    logger.info("Creating train/val split...")

    # Stratified split by label
    train_df_raw, val_df_raw = train_test_split(
        df,
        test_size=1 - TRAIN_RATIO,
        stratify=df["label_binary"],
        random_state=RANDOM_STATE,
    )

    # Add split column (explicit DataFrame typing for mypy)
    train_df: pd.DataFrame = train_df_raw.copy()  # type: ignore
    val_df: pd.DataFrame = val_df_raw.copy()  # type: ignore
    train_df["split"] = "train"
    val_df["split"] = "val"

    # Combine
    df_split: pd.DataFrame = pd.concat([train_df, val_df], ignore_index=True)

    # Log split statistics
    n_train = len(train_df)
    n_val = len(val_df)
    n_train_poly = (train_df["label_binary"] == 1).sum()
    n_train_spec = (train_df["label_binary"] == 0).sum()
    n_val_poly = (val_df["label_binary"] == 1).sum()
    n_val_spec = (val_df["label_binary"] == 0).sum()

    logger.info(
        f"  Train: {n_train} sequences ({n_train_poly} poly, {n_train_spec} spec)"
    )
    logger.info(f"  Val:   {n_val} sequences ({n_val_poly} poly, {n_val_spec} spec)")

    return df_split


def create_fragment_csvs(df: pd.DataFrame) -> None:
    """Create 16 fragment-specific CSV files.

    Each CSV contains columns: ['id', 'sequence', 'label', 'split']
    Following standard structure (matches Boughter/Jain/Harvey datasets).

    Args:
        df: DataFrame with split column and all fragments
    """
    logger.info(f"Creating {len(FRAGMENT_DEFINITIONS)} fragment CSV files...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for fragment_name, column_name in FRAGMENT_DEFINITIONS.items():
        # Create fragment DataFrame with split column (single file per fragment)
        fragment_df = pd.DataFrame(
            {
                "id": df["antibody_id"],
                "sequence": df[column_name],
                "label": df["label_binary"],
                "split": df["split"],
            }
        )

        # Save single CSV per fragment (NOT separate train/val files)
        output_file = OUTPUT_DIR / f"{fragment_name}_tessier.csv"
        fragment_df.to_csv(output_file, index=False)

        n_train = (fragment_df["split"] == "train").sum()
        n_val = (fragment_df["split"] == "val").sum()
        logger.info(
            f"  {fragment_name}: {len(fragment_df)} total ({n_train} train, {n_val} val)"
        )


def main() -> None:
    """Execute Stage 3: QC and train/val split."""
    logger.info("=" * 80)
    logger.info("TESSIER PREPROCESSING - STAGE 3: QC and Train/Val Split")
    logger.info("=" * 80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load input
    logger.info(f"Loading input: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    logger.info(f"  Loaded {len(df)} annotated sequences")

    # Apply QC filters
    logger.info("Applying quality control filters...")
    df_filtered, filtered_ids = apply_qc_filters(df)

    # Save filtered IDs log
    if filtered_ids:
        logger.info(f"Saving {len(filtered_ids)} filtered sequences to {QC_LOG}...")
        with open(QC_LOG, "w") as f:
            f.write("# QC Filtered Sequences\n")
            f.write(f"# Total filtered: {len(filtered_ids)}\n")
            f.write(f"# Retention rate: {len(df_filtered) / len(df) * 100:.2f}%\n\n")
            for filtered_id in filtered_ids:
                f.write(f"{filtered_id}\n")

    # Create train/val split
    df_split = create_train_val_split(df_filtered)

    # Create fragment CSV files
    create_fragment_csvs(df_split)

    # Summary
    n_train = (df_split["split"] == "train").sum()
    n_val = (df_split["split"] == "val").sum()

    logger.info("=" * 80)
    logger.info("STAGE 3 COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Input (annotated): {len(df)}")
    logger.info(f"After QC: {len(df_filtered)}")
    logger.info(f"Training set: {n_train}")
    logger.info(f"Validation set: {n_val}")
    logger.info(
        f"Fragment CSVs: {len(FRAGMENT_DEFINITIONS)} types × 2 splits = {len(FRAGMENT_DEFINITIONS) * 2} files"
    )
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("Next: Implement transfer learning script")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
