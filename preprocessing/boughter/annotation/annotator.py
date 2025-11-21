from __future__ import annotations

from typing import Any, cast

import pandas as pd
import riot_na

from preprocessing.logging_config import setup_logger
from preprocessing.paths import BOUGHTER_ANNOTATED_DIR

logger = setup_logger(__name__)

# Initialize ANARCI for amino acid annotation (IMGT scheme)
annotator = riot_na.create_riot_aa()


def annotate_sequence(seq_id: str, sequence: str, chain: str) -> dict[str, str] | None:
    """
    Annotate a single amino acid sequence using ANARCI (IMGT).

    Uses strict IMGT boundaries per Sakhnini et al. 2025:
    "The primary sequences were annotated in the CDRs using ANARCI
    following the IMGT numbering scheme"

    CDR boundaries (strict IMGT):
    - CDR-H3: positions 105-117 (EXCLUDES position 118, which is FR4 J-anchor)
    - CDR-H2: positions 56-65 (fixed IMGT positions, variable lengths OK)
    - CDR-H1: positions 27-38 (fixed IMGT)

    Note: Position 118 (J-Trp/Phe) is conserved FR4, NOT CDR.
    Boughter's published .dat files include position 118, but we use
    strict IMGT for biological correctness and ML best practices.

    Note: CDR2 length naturally varies (8-11 residues typical).
    IMGT positions are fixed (56-65), but sequences can have gaps
    for deletions or insertion codes. Harvey et al. 2022 confirms
    this is expected with ANARCI/IMGT numbering. This is normal
    antibody diversity, not an error.

    Args:
        seq_id: Unique identifier for the sequence
        sequence: Amino acid sequence string
        chain: 'H' for heavy or 'L' for light

    Returns:
        Dictionary with extracted fragments, or None if annotation fails
    """
    assert chain in ("H", "L"), "chain must be 'H' or 'L'"

    try:
        annotation = annotator.run_on_sequence(seq_id, sequence)

        # Extract all fragments, converting None to empty string
        # ANARCI returns None for regions it cannot identify
        def safe_str(value: str | None) -> str:
            """Convert None to empty string, preserve actual strings."""
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

        # Reconstruct full V-domain from fragments (avoids constant region garbage)
        # This is gap-free and clean (P0 fix + constant region removal)
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

        # Validate that we got at least SOME fragments
        # If all CDRs are empty, annotation failed
        if not any(
            [
                fragments[f"cdr1_aa_{chain}"],
                fragments[f"cdr2_aa_{chain}"],
                fragments[f"cdr3_aa_{chain}"],
            ]
        ):
            logger.info(f"  ANARCI returned no CDRs for {seq_id} ({chain} chain)")
            return None

        # Create concatenated fragments (safe now - no None values)
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
        logger.info(f"  ANARCI failed for {seq_id} ({chain} chain): {e}")
        return None


def process_antibody(row: pd.Series) -> dict[str, Any] | None:
    """
    Annotate heavy and light chains, create all 16 fragments.

    Args:
        row: DataFrame row with heavy_seq, light_seq, and metadata

    Returns:
        Dictionary with all fragments and metadata, or None if annotation fails
    """
    seq_id = row["id"]

    # Annotate heavy chain
    heavy_frags = annotate_sequence(seq_id, row["heavy_seq"], "H")
    if heavy_frags is None:
        return None

    # Annotate light chain
    light_frags = annotate_sequence(seq_id, row["light_seq"], "L")
    if light_frags is None:
        return None

    # Combine metadata and fragments
    result = {
        "id": row["id"],
        "subset": row["subset"],
        "num_flags": row["num_flags"],
        "flag_category": row["flag_category"],
        "label": row["label"],
        "include_in_training": row["include_in_training"],
        "source": row["source"],
    }

    result.update(heavy_frags)
    result.update(light_frags)

    # Create paired/combined fragments
    result["vh_vl"] = result["full_seq_H"] + result["full_seq_L"]
    result["all_cdrs"] = result["cdrs_H"] + result["cdrs_L"]
    result["all_fwrs"] = result["fwrs_H"] + result["fwrs_L"]

    return result


def annotate_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate all antibodies in the dataset.

    Args:
        df: DataFrame from Stage 1 (boughter.csv)

    Returns:
        DataFrame with all fragments annotated
    """
    logger.info(f"\nAnnotating {len(df)} antibodies with ANARCI (strict IMGT)...")

    results = []
    failures = []

    for idx, row in df.iterrows():
        result = process_antibody(row)

        if result is None:
            failures.append(row["id"])
        else:
            results.append(result)

        # Progress indicator
        idx_int = cast(int, idx)
        if (idx_int + 1) % 100 == 0:
            logger.info(
                f"  Progress: {idx_int + 1}/{len(df)} ({len(failures)} failures)"
            )

    df_annotated = pd.DataFrame(results)

    logger.info(f"\n✓ Successfully annotated: {len(df_annotated)}/{len(df)} antibodies")

    if failures:
        logger.info(f"✗ Failures: {len(failures)}")
        failure_rate = len(failures) / len(df) * 100
        logger.info(f"  Failure rate: {failure_rate:.2f}%")

        # Write failures to log
        failure_log = BOUGHTER_ANNOTATED_DIR / "annotation_failures.log"
        failure_log.parent.mkdir(parents=True, exist_ok=True)
        failure_log.write_text("\n".join(failures))
        logger.info(f"  Failed IDs written to: {failure_log}")

    return df_annotated
