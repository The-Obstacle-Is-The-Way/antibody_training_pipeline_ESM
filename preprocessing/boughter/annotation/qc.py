from __future__ import annotations

import pandas as pd

from preprocessing.logging_config import setup_logger
from preprocessing.paths import BOUGHTER_ANNOTATED_DIR

logger = setup_logger(__name__)


def filter_quality_issues(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 3: Post-annotation quality control.

    Following Boughter et al. 2020 methodology (seq_loader.py)
    and 2025 industry best practices (Harvey 2022, AbSet 2024).

    Removes:
    1. Sequences with X (unknown amino acid) in ANY CDR
    2. Sequences with empty ("") CDRs

    This is done AFTER ANARCI annotation to maximize
    information extraction from raw data.

    See docs/accuracy_verification_report.md for rationale.
    """
    logger.info("\n" + "=" * 70)
    logger.info("Stage 3: Post-annotation Quality Control")
    logger.info("=" * 70)
    logger.info(f"Input sequences: {len(df)}")

    cdr_columns = [
        "cdr1_aa_H",
        "cdr2_aa_H",
        "cdr3_aa_H",
        "cdr1_aa_L",
        "cdr2_aa_L",
        "cdr3_aa_L",
    ]

    df_clean = df.copy()

    # Track sequences removed for each reason
    sequences_with_X = set()
    sequences_with_empty = set()

    # First pass: identify sequences with X in ANY CDR
    for col in cdr_columns:
        has_X = df_clean[df_clean[col].str.contains("X", na=False)]
        if len(has_X) > 0:
            sequences_with_X.update(has_X["id"].tolist())

    # Second pass: identify sequences with empty CDRs
    for col in cdr_columns:
        is_empty = df_clean[df_clean[col] == ""]
        if len(is_empty) > 0:
            sequences_with_empty.update(is_empty["id"].tolist())

    # Remove all problematic sequences
    problematic_ids = sequences_with_X | sequences_with_empty
    df_clean = df_clean[~df_clean["id"].isin(problematic_ids)]

    logger.info(f"Sequences with X in ANY CDR: {len(sequences_with_X)}")
    logger.info(f"Sequences with empty CDRs:    {len(sequences_with_empty)}")
    logger.info(f"Total unique sequences removed: {len(problematic_ids)}")

    if problematic_ids:
        qc_log = BOUGHTER_ANNOTATED_DIR / "qc_filtered_sequences.txt"
        qc_log.parent.mkdir(parents=True, exist_ok=True)
        qc_log.write_text("\n".join(sorted(problematic_ids)))
        logger.info(f"Filtered IDs written to: {qc_log}")

    logger.info("")
    logger.info(f"Output sequences: {len(df_clean)}")
    logger.info(f"Retention rate: {len(df_clean) / len(df) * 100:.1f}%")

    return df_clean
