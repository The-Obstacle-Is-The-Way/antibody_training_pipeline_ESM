"""ANARCI annotation for Tessier 2024 dataset.

Stage 2 of 3: tessier_raw.csv → tessier_annotated.csv

Input:
- train_datasets/tessier/processed/tessier_raw.csv (~246k sequences)

Output:
- train_datasets/tessier/annotated/tessier_annotated.csv (~244k sequences, 99% success rate)
- train_datasets/tessier/annotated/annotation_failures.log

Extracts 16 fragment types using ANARCI with IMGT numbering scheme.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import riot_na

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize ANARCI for amino acid annotation (IMGT scheme)
logger.info("Initializing ANARCI with IMGT numbering scheme...")
annotator = riot_na.create_riot_aa()

# Paths
INPUT_FILE = Path("train_datasets/tessier/processed/tessier_raw.csv")
OUTPUT_DIR = Path("train_datasets/tessier/annotated")
OUTPUT_FILE = OUTPUT_DIR / "tessier_annotated.csv"
FAILURES_LOG = OUTPUT_DIR / "annotation_failures.log"

# Batch processing settings
BATCH_SIZE = 1000  # Process 1000 sequences per batch
LOG_INTERVAL = 10000  # Log progress every 10k sequences


def annotate_sequence(seq_id: str, sequence: str, chain: str) -> dict[str, str] | None:
    """Annotate a single amino acid sequence using ANARCI (IMGT).

    Args:
        seq_id: Unique identifier for the sequence
        sequence: Amino acid sequence string
        chain: 'H' for heavy or 'L' for light

    Returns:
        Dictionary with extracted fragments, or None if annotation fails
    """
    assert chain in ("H", "L"), f"chain must be 'H' or 'L', got {chain}"

    try:
        annotation = annotator.run_on_sequence(seq_id, sequence)

        # Extract all fragments, converting None to empty string
        def safe_str(value: str | None) -> str:
            return value if value is not None else ""

        # Extract individual fragments
        fragments = {
            f"fwr1_aa_{chain}": safe_str(annotation.fwr1_aa),
            f"cdr1_aa_{chain}": safe_str(annotation.cdr1_aa),
            f"fwr2_aa_{chain}": safe_str(annotation.fwr2_aa),
            f"cdr2_aa_{chain}": safe_str(annotation.cdr2_aa),
            f"fwr3_aa_{chain}": safe_str(annotation.fwr3_aa),
            f"cdr3_aa_{chain}": safe_str(annotation.cdr3_aa),
            f"fwr4_aa_{chain}": safe_str(annotation.fwr4_aa),
        }

        # Reconstruct full V-domain from fragments
        fragments[f"full_seq_{chain}"] = "".join(
            [
                fragments[f"fwr1_aa_{chain}"],
                fragments[f"cdr1_aa_{chain}"],
                fragments[f"fwr2_aa_{chain}"],
                fragments[f"cdr2_aa_{chain}"],
                fragments[f"fwr3_aa_{chain}"],
                fragments[f"cdr3_aa_{chain}"],
                fragments[f"fwr4_aa_{chain}"],
            ]
        )

        # Validate that we got at least SOME CDRs
        if not any(
            [
                fragments[f"cdr1_aa_{chain}"],
                fragments[f"cdr2_aa_{chain}"],
                fragments[f"cdr3_aa_{chain}"],
            ]
        ):
            return None

        # Create concatenated fragments
        fragments[f"cdrs_{chain}"] = "".join(
            [
                fragments[f"cdr1_aa_{chain}"],
                fragments[f"cdr2_aa_{chain}"],
                fragments[f"cdr3_aa_{chain}"],
            ]
        )

        fragments[f"fwrs_{chain}"] = "".join(
            [
                fragments[f"fwr1_aa_{chain}"],
                fragments[f"fwr2_aa_{chain}"],
                fragments[f"fwr3_aa_{chain}"],
                fragments[f"fwr4_aa_{chain}"],
            ]
        )

        return fragments

    except Exception as e:
        logger.debug(f"  ANARCI failed for {seq_id} ({chain} chain): {e}")
        return None


def process_antibody(row: pd.Series) -> dict[str, Any] | None:
    """Annotate heavy and light chains, create all 16 fragments.

    Args:
        row: DataFrame row with vh_sequence, vl_sequence, and metadata

    Returns:
        Dictionary with all fragments and metadata, or None if annotation fails
    """
    seq_id = row["antibody_id"]

    # Annotate heavy chain
    heavy_frags = annotate_sequence(seq_id, row["vh_sequence"], "H")
    if heavy_frags is None:
        return None

    # Annotate light chain
    light_frags = annotate_sequence(seq_id, row["vl_sequence"], "L")
    if light_frags is None:
        return None

    # Combine metadata and fragments
    result = {
        "antibody_id": row["antibody_id"],
        "label_binary": row["label_binary"],
        "source_name": row["source_name"],
    }

    result.update(heavy_frags)
    result.update(light_frags)

    # Create paired/combined fragments (16 fragment types total)
    result["vh_vl"] = result["full_seq_H"] + result["full_seq_L"]
    result["all_cdrs"] = result["cdrs_H"] + result["cdrs_L"]
    result["all_fwrs"] = result["fwrs_H"] + result["fwrs_L"]

    return result


def annotate_batch(df_batch: pd.DataFrame) -> tuple[list[dict[str, Any]], list[str]]:
    """Annotate a batch of antibodies.

    Args:
        df_batch: DataFrame batch to process

    Returns:
        Tuple of (successful_annotations, failed_ids)
    """
    successful = []
    failed_ids = []

    for _, row in df_batch.iterrows():
        result = process_antibody(row)
        if result is not None:
            successful.append(result)
        else:
            failed_ids.append(row["antibody_id"])

    return successful, failed_ids


def main() -> None:
    """Execute Stage 2: ANARCI annotation."""
    logger.info("=" * 80)
    logger.info("TESSIER PREPROCESSING - STAGE 2: ANARCI Annotation")
    logger.info("=" * 80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load input
    logger.info(f"Loading input: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    logger.info(f"  Loaded {len(df)} sequences")

    # Process in batches for memory efficiency
    logger.info(f"Processing in batches of {BATCH_SIZE}...")
    all_successful = []
    all_failed_ids = []

    n_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(df), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        df_batch = df.iloc[i : i + BATCH_SIZE]

        # Process batch
        successful, failed_ids = annotate_batch(df_batch)
        all_successful.extend(successful)
        all_failed_ids.extend(failed_ids)

        # Log progress
        if batch_num % 10 == 0 or batch_num == n_batches:
            n_processed = min(i + BATCH_SIZE, len(df))
            n_success = len(all_successful)
            success_rate = n_success / n_processed * 100
            logger.info(
                f"  Batch {batch_num}/{n_batches}: {n_processed}/{len(df)} processed "
                f"({success_rate:.1f}% success rate)"
            )

    # Create annotated DataFrame
    logger.info("Creating annotated DataFrame...")
    df_annotated = pd.DataFrame(all_successful)

    # Save successful annotations
    logger.info(f"Saving {len(df_annotated)} annotated sequences to {OUTPUT_FILE}...")
    df_annotated.to_csv(OUTPUT_FILE, index=False)

    # Save failures log
    if all_failed_ids:
        logger.info(
            f"Saving {len(all_failed_ids)} annotation failures to {FAILURES_LOG}..."
        )
        with open(FAILURES_LOG, "w") as f:
            f.write("# ANARCI Annotation Failures\n")
            f.write(f"# Total failures: {len(all_failed_ids)}\n")
            f.write(f"# Failure rate: {len(all_failed_ids) / len(df) * 100:.2f}%\n\n")
            for failed_id in all_failed_ids:
                f.write(f"{failed_id}\n")

    # Summary
    n_success = len(df_annotated)
    n_failure = len(all_failed_ids)
    success_rate = n_success / len(df) * 100
    failure_rate = n_failure / len(df) * 100

    logger.info("=" * 80)
    logger.info("STAGE 2 COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Input sequences: {len(df)}")
    logger.info(f"Successfully annotated: {n_success} ({success_rate:.1f}%)")
    logger.info(f"Failed annotations: {n_failure} ({failure_rate:.1f}%)")
    logger.info(f"Output: {OUTPUT_FILE}")
    if all_failed_ids:
        logger.info(f"Failures logged: {FAILURES_LOG}")
    logger.info("Next: Run stage 3 (QC and train/val split)")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
