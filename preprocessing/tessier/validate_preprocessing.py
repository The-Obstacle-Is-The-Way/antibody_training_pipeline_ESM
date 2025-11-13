"""Validation script for Tessier 2024 preprocessing pipeline.

Validates all 3 stages of preprocessing to ensure:
1. Stage 1: Correct sequence extraction and deduplication
2. Stage 2: ANARCI annotation with all required fragments
3. Stage 3: QC filters, train/val split, and fragment CSV structure

Run after preprocessing completes to verify SSOT compliance.
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Expected counts
EXPECTED_TOTAL = 373_457
EXPECTED_TRAIN = 298_765
EXPECTED_VAL = 74_692
TRAIN_RATIO = 0.8
EXPECTED_FRAGMENTS = 16

# Paths
STAGE1_OUTPUT = Path("train_datasets/tessier/processed/tessier_raw.csv")
STAGE2_OUTPUT = Path("train_datasets/tessier/annotated/tessier_annotated.csv")
STAGE3_OUTPUT_DIR = Path("train_datasets/tessier/annotated")

# Fragment definitions (must match step3_qc_and_split.py)
FRAGMENT_NAMES = [
    "VH_only",
    "VL_only",
    "H-CDR1",
    "H-CDR2",
    "H-CDR3",
    "L-CDR1",
    "L-CDR2",
    "L-CDR3",
    "H-FWRs",
    "L-FWRs",
    "VH+VL",
    "H-CDRs",
    "L-CDRs",
    "All-CDRs",
    "All-FWRs",
    "Full",
]


def validate_stage1() -> bool:
    """Validate Stage 1 output: sequence extraction and deduplication."""
    logger.info("=" * 80)
    logger.info("VALIDATING STAGE 1: Sequence Extraction")
    logger.info("=" * 80)

    if not STAGE1_OUTPUT.exists():
        logger.error(f"❌ Stage 1 output not found: {STAGE1_OUTPUT}")
        return False

    # Load Stage 1 output
    df = pd.read_csv(STAGE1_OUTPUT)
    logger.info(f"Loaded {len(df)} sequences from {STAGE1_OUTPUT}")

    # Check required columns
    required_cols = ["antibody_id", "vh_sequence", "vl_sequence", "label_binary"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"❌ Missing required columns: {missing_cols}")
        return False
    logger.info(f"✅ All required columns present: {required_cols}")

    # Check sequence count
    if len(df) != EXPECTED_TOTAL:
        logger.error(
            f"❌ Unexpected sequence count: {len(df)} (expected {EXPECTED_TOTAL})"
        )
        return False
    logger.info(f"✅ Sequence count matches expected: {EXPECTED_TOTAL}")

    # Check for duplicates
    duplicates = df.duplicated(subset=["vh_sequence", "vl_sequence"]).sum()
    if duplicates > 0:
        logger.error(f"❌ Found {duplicates} duplicate sequences")
        return False
    logger.info("✅ No duplicate sequences found")

    # Check label distribution
    n_poly = (df["label_binary"] == 1).sum()
    n_spec = (df["label_binary"] == 0).sum()
    logger.info(f"Label distribution: {n_poly} polyreactive, {n_spec} specific")

    # Check for NaN sequences
    n_nan_vh = df["vh_sequence"].isna().sum()
    n_nan_vl = df["vl_sequence"].isna().sum()
    if n_nan_vh > 0 or n_nan_vl > 0:
        logger.error(f"❌ Found NaN sequences: {n_nan_vh} VH, {n_nan_vl} VL")
        return False
    logger.info("✅ No NaN sequences found")

    logger.info("✅ STAGE 1 VALIDATION PASSED")
    return True


def validate_stage2() -> bool:
    """Validate Stage 2 output: ANARCI annotation."""
    logger.info("=" * 80)
    logger.info("VALIDATING STAGE 2: ANARCI Annotation")
    logger.info("=" * 80)

    if not STAGE2_OUTPUT.exists():
        logger.error(f"❌ Stage 2 output not found: {STAGE2_OUTPUT}")
        return False

    # Load Stage 2 output
    df = pd.read_csv(STAGE2_OUTPUT)
    logger.info(f"Loaded {len(df)} annotated sequences from {STAGE2_OUTPUT}")

    # Check sequence count
    if len(df) != EXPECTED_TOTAL:
        logger.error(
            f"❌ Unexpected sequence count: {len(df)} (expected {EXPECTED_TOTAL})"
        )
        return False
    logger.info(f"✅ Sequence count matches expected: {EXPECTED_TOTAL}")

    # Check required fragment columns
    required_fragments = [
        "cdr1_aa_H",
        "cdr2_aa_H",
        "cdr3_aa_H",
        "cdr1_aa_L",
        "cdr2_aa_L",
        "cdr3_aa_L",
        "fwr1_aa_H",
        "fwr2_aa_H",
        "fwr3_aa_H",
        "fwr4_aa_H",
        "fwr1_aa_L",
        "fwr2_aa_L",
        "fwr3_aa_L",
        "fwr4_aa_L",
        "full_seq_H",
        "full_seq_L",
    ]
    missing_fragments = [col for col in required_fragments if col not in df.columns]
    if missing_fragments:
        logger.error(f"❌ Missing fragment columns: {missing_fragments}")
        return False
    logger.info(f"✅ All {len(required_fragments)} fragment columns present")

    # Check for empty CDRs
    has_empty_cdrs = df.apply(
        lambda row: any(
            len(str(row[col])) == 0 for col in ["cdr1_aa_H", "cdr2_aa_H", "cdr3_aa_H"]
        ),
        axis=1,
    ).sum()
    if has_empty_cdrs > 0:
        logger.warning(f"⚠️  {has_empty_cdrs} sequences with empty CDRs")
    else:
        logger.info("✅ No sequences with empty CDRs")

    logger.info("✅ STAGE 2 VALIDATION PASSED")
    return True


def validate_stage3() -> bool:
    """Validate Stage 3 output: QC filters, train/val split, fragment CSVs."""
    logger.info("=" * 80)
    logger.info("VALIDATING STAGE 3: QC Filters and Train/Val Split")
    logger.info("=" * 80)

    # Check that all 16 fragment CSVs exist
    fragment_files = [
        STAGE3_OUTPUT_DIR / f"{fragment}_tessier.csv" for fragment in FRAGMENT_NAMES
    ]
    missing_files = [f for f in fragment_files if not f.exists()]
    if missing_files:
        logger.error(f"❌ Missing fragment CSVs: {[f.name for f in missing_files]}")
        return False
    logger.info(f"✅ All {EXPECTED_FRAGMENTS} fragment CSVs found")

    # Validate each fragment CSV
    all_valid = True
    total_train = None
    total_val = None

    for fragment_name in FRAGMENT_NAMES:
        fragment_file = STAGE3_OUTPUT_DIR / f"{fragment_name}_tessier.csv"
        df = pd.read_csv(fragment_file)

        # Check required columns
        required_cols = ["id", "sequence", "label", "split"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(
                f"❌ {fragment_name}: Missing columns: {missing_cols} (columns: {df.columns.tolist()})"
            )
            all_valid = False
            continue

        # Check split column values
        split_values = set(df["split"].unique())
        if split_values != {"train", "val"}:
            logger.error(
                f"❌ {fragment_name}: Unexpected split values: {split_values} (expected {{'train', 'val'}})"
            )
            all_valid = False
            continue

        # Count train/val splits
        n_train = (df["split"] == "train").sum()
        n_val = (df["split"] == "val").sum()

        # Check consistency across fragments
        if total_train is None:
            total_train = n_train
            total_val = n_val
        else:
            if n_train != total_train or n_val != total_val:
                logger.error(
                    f"❌ {fragment_name}: Inconsistent split counts: {n_train} train, {n_val} val (expected {total_train}, {total_val})"
                )
                all_valid = False
                continue

        # Check total count
        if len(df) != EXPECTED_TOTAL:
            logger.error(
                f"❌ {fragment_name}: Unexpected total count: {len(df)} (expected {EXPECTED_TOTAL})"
            )
            all_valid = False
            continue

        logger.info(
            f"  {fragment_name}: {len(df)} total ({n_train} train, {n_val} val) ✅"
        )

    # Validate train/val counts (ensure totals were set)
    if total_train is None or total_val is None:
        logger.error("❌ Failed to extract train/val counts from fragment CSVs")
        return False

    if total_train != EXPECTED_TRAIN:
        logger.error(
            f"❌ Training count mismatch: {total_train} (expected {EXPECTED_TRAIN})"
        )
        all_valid = False
    else:
        logger.info(f"✅ Training count matches expected: {EXPECTED_TRAIN}")

    if total_val != EXPECTED_VAL:
        logger.error(
            f"❌ Validation count mismatch: {total_val} (expected {EXPECTED_VAL})"
        )
        all_valid = False
    else:
        logger.info(f"✅ Validation count matches expected: {EXPECTED_VAL}")

    # Check train/val ratio (totals are guaranteed non-None here)
    actual_ratio: float = total_train / (total_train + total_val)
    if abs(actual_ratio - TRAIN_RATIO) > 0.01:
        logger.error(
            f"❌ Train ratio mismatch: {actual_ratio:.3f} (expected {TRAIN_RATIO})"
        )
        all_valid = False
    else:
        logger.info(
            f"✅ Train ratio matches expected: {TRAIN_RATIO} ({actual_ratio:.3f})"
        )

    # Check stratification (load first fragment for detailed check)
    first_fragment = pd.read_csv(STAGE3_OUTPUT_DIR / f"{FRAGMENT_NAMES[0]}_tessier.csv")
    train_df = first_fragment[first_fragment["split"] == "train"]
    val_df = first_fragment[first_fragment["split"] == "val"]

    train_poly_ratio = (train_df["label"] == 1).sum() / len(train_df)
    val_poly_ratio = (val_df["label"] == 1).sum() / len(val_df)

    if abs(train_poly_ratio - val_poly_ratio) > 0.02:
        logger.warning(
            f"⚠️  Stratification may be imperfect: train={train_poly_ratio:.3f}, val={val_poly_ratio:.3f}"
        )
    else:
        logger.info(
            f"✅ Stratification looks good: train={train_poly_ratio:.3f}, val={val_poly_ratio:.3f}"
        )

    if all_valid:
        logger.info("✅ STAGE 3 VALIDATION PASSED")
    else:
        logger.error("❌ STAGE 3 VALIDATION FAILED")

    return all_valid


def main() -> None:
    """Execute full validation pipeline."""
    logger.info("=" * 80)
    logger.info("TESSIER PREPROCESSING VALIDATION")
    logger.info("=" * 80)

    # Run all validations
    stage1_valid = validate_stage1()
    stage2_valid = validate_stage2()
    stage3_valid = validate_stage3()

    # Summary
    logger.info("=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Stage 1 (Extraction):     {'✅ PASS' if stage1_valid else '❌ FAIL'}")
    logger.info(f"Stage 2 (Annotation):     {'✅ PASS' if stage2_valid else '❌ FAIL'}")
    logger.info(f"Stage 3 (QC + Split):     {'✅ PASS' if stage3_valid else '❌ FAIL'}")

    if stage1_valid and stage2_valid and stage3_valid:
        logger.info("=" * 80)
        logger.info("🎉 ALL VALIDATIONS PASSED - PREPROCESSING COMPLETE 🎉")
        logger.info("=" * 80)
    else:
        logger.error("=" * 80)
        logger.error("❌ VALIDATION FAILED - PLEASE REVIEW ERRORS ABOVE")
        logger.error("=" * 80)


if __name__ == "__main__":
    main()
