"""Extract sequences from Tessier 2024 Excel files and assign binary labels.

Stage 1 of 3: Excel → tessier_raw.csv

Input files:
- external_datasets/tessier_2024_polyreactivity/Supplemental Datasets/Human Ab Poly Dataset S1_v2.xlsx
- external_datasets/tessier_2024_polyreactivity/Supplemental Datasets/Human Ab Poly Dataset S2_v2.xlsx

Output:
- train_datasets/tessier/processed/tessier_raw.csv

Expected: ~246,295 sequences (115,039 polyreactive + 131,256 specific)
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
DATASET_DIR = Path(
    "external_datasets/tessier_2024_polyreactivity/Supplemental Datasets"
)
S1_FILE = DATASET_DIR / "Human Ab Poly Dataset S1_v2.xlsx"
S2_FILE = DATASET_DIR / "Human Ab Poly Dataset S2_v2.xlsx"
OUTPUT_DIR = Path("train_datasets/tessier/processed")
OUTPUT_FILE = OUTPUT_DIR / "tessier_raw.csv"

# Validation constants (S1 + S2 combined)
# S1: 115,038 poly + 131,255 spec = 246,293
# S2: 93,079 poly + 34,085 spec = 127,164
# Total: 208,117 poly + 165,340 spec = 373,457
EXPECTED_POSITIVE = 208_117
EXPECTED_NEGATIVE = 165_340
EXPECTED_TOTAL = EXPECTED_POSITIVE + EXPECTED_NEGATIVE

# Sequence length ranges (typical for antibody variable regions)
VH_MIN_LEN = 110
VH_MAX_LEN = 130
VL_MIN_LEN = 105
VL_MAX_LEN = 115


def load_and_label_dataset(file_path: Path, dataset_name: str) -> pd.DataFrame:
    """Load Excel file and assign binary labels based on name prefix.

    Args:
        file_path: Path to Excel file
        dataset_name: Name for logging (S1 or S2)

    Returns:
        DataFrame with columns: ['antibody_id', 'vh_sequence', 'vl_sequence', 'label_binary', 'source_name']
    """
    logger.info(f"Loading {dataset_name}: {file_path}")

    # Read Excel (header is in row 2, skip rows 0-1)
    df = pd.read_excel(file_path, sheet_name=0, header=2)
    logger.info(f"  Loaded {len(df)} rows")
    logger.info(f"  Columns: {df.columns.tolist()}")

    # Check expected columns
    required_cols = ["Name", "VH", "VL"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {dataset_name}")

    # Assign labels based on name prefix
    def assign_label(name: str) -> int:
        """Assign binary label from name prefix.

        'high non-spec' → 1 (polyreactive)
        'low non-spec'  → 0 (specific)
        """
        name_lower = str(name).lower()
        if "high non-spec" in name_lower:
            return 1
        elif "low non-spec" in name_lower:
            return 0
        else:
            raise ValueError(f"Unexpected name format: {name}")

    df["label_binary"] = df["Name"].apply(assign_label)

    # Count labels
    n_poly = (df["label_binary"] == 1).sum()
    n_spec = (df["label_binary"] == 0).sum()
    logger.info(f"  Labels: {n_poly} polyreactive, {n_spec} specific")

    # Standardize column names
    result = pd.DataFrame(
        {
            "antibody_id": [f"{dataset_name}_{i}" for i in range(len(df))],
            "vh_sequence": df["VH"],
            "vl_sequence": df["VL"],
            "label_binary": df["label_binary"],
            "source_name": df["Name"],
        }
    )

    return result


def validate_sequences(df: pd.DataFrame) -> None:
    """Validate sequence quality and log statistics."""
    logger.info("Validating sequences...")

    # Check for NaN sequences
    n_nan_vh = df["vh_sequence"].isna().sum()
    n_nan_vl = df["vl_sequence"].isna().sum()
    if n_nan_vh > 0 or n_nan_vl > 0:
        raise ValueError(f"Found NaN sequences: {n_nan_vh} VH, {n_nan_vl} VL")

    # Check for empty sequences
    n_empty_vh = (df["vh_sequence"].str.len() == 0).sum()
    n_empty_vl = (df["vl_sequence"].str.len() == 0).sum()
    if n_empty_vh > 0 or n_empty_vl > 0:
        raise ValueError(f"Found empty sequences: {n_empty_vh} VH, {n_empty_vl} VL")

    # Sequence length statistics
    vh_lengths = df["vh_sequence"].str.len()
    vl_lengths = df["vl_sequence"].str.len()

    logger.info(
        f"VH sequence lengths: min={vh_lengths.min()}, max={vh_lengths.max()}, mean={vh_lengths.mean():.1f}"
    )
    logger.info(
        f"VL sequence lengths: min={vl_lengths.min()}, max={vl_lengths.max()}, mean={vl_lengths.mean():.1f}"
    )

    # Warn about outliers (but don't fail - ANARCI will handle these)
    vh_outliers = ((vh_lengths < VH_MIN_LEN) | (vh_lengths > VH_MAX_LEN)).sum()
    vl_outliers = ((vl_lengths < VL_MIN_LEN) | (vl_lengths > VL_MAX_LEN)).sum()
    if vh_outliers > 0:
        logger.warning(
            f"  {vh_outliers} VH sequences outside typical range ({VH_MIN_LEN}-{VH_MAX_LEN} aa)"
        )
    if vl_outliers > 0:
        logger.warning(
            f"  {vl_outliers} VL sequences outside typical range ({VL_MIN_LEN}-{VL_MAX_LEN} aa)"
        )

    logger.info("✅ Sequence validation passed")


def main() -> None:
    """Execute Stage 1: Extract sequences from Excel files."""
    logger.info("=" * 80)
    logger.info("TESSIER PREPROCESSING - STAGE 1: Extract Sequences")
    logger.info("=" * 80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets
    df_s1 = load_and_label_dataset(S1_FILE, "S1")
    df_s2 = load_and_label_dataset(S2_FILE, "S2")

    # Merge datasets
    logger.info("Merging datasets...")
    df_merged = pd.concat([df_s1, df_s2], ignore_index=True)
    logger.info(f"  Total after merge: {len(df_merged)} sequences")

    # Deduplicate by (VH, VL) pair
    logger.info("Deduplicating by (VH, VL) pair...")
    n_before = len(df_merged)
    df_merged = df_merged.drop_duplicates(
        subset=["vh_sequence", "vl_sequence"], keep="first"
    )
    n_after = len(df_merged)
    n_dup = n_before - n_after
    logger.info(f"  Removed {n_dup} duplicates ({n_dup / n_before * 100:.1f}%)")
    logger.info(f"  Unique sequences: {n_after}")

    # Reassign antibody IDs after deduplication
    df_merged["antibody_id"] = [f"tessier_{i:06d}" for i in range(len(df_merged))]

    # Count final labels
    n_poly = (df_merged["label_binary"] == 1).sum()
    n_spec = (df_merged["label_binary"] == 0).sum()
    logger.info("Final label distribution:")
    logger.info(f"  Polyreactive (label=1): {n_poly}")
    logger.info(f"  Specific (label=0): {n_spec}")
    logger.info(f"  Total: {n_poly + n_spec}")

    # Validate against expected counts
    pct_diff_total = abs(len(df_merged) - EXPECTED_TOTAL) / EXPECTED_TOTAL * 100

    if pct_diff_total > 5:
        logger.warning(
            f"⚠️  Total count differs from expected S1+S2 by {pct_diff_total:.1f}%"
        )
    else:
        logger.info(
            f"✅ Total count matches expected S1+S2 within 5% (diff: {pct_diff_total:.1f}%)"
        )

    # Validate sequences
    validate_sequences(df_merged)

    # Save to CSV
    logger.info(f"Saving to {OUTPUT_FILE}...")
    df_merged.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"✅ Saved {len(df_merged)} sequences to {OUTPUT_FILE}")

    # Summary
    logger.info("=" * 80)
    logger.info("STAGE 1 COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output: {OUTPUT_FILE}")
    logger.info(f"Sequences: {len(df_merged)}")
    logger.info(f"Polyreactive: {n_poly}")
    logger.info(f"Specific: {n_spec}")
    logger.info("Next: Run stage 2 (ANARCI annotation)")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
