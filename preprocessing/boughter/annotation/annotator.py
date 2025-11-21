from __future__ import annotations

import pandas as pd

from preprocessing.fragment_utils import process_sequences_to_fragments
from preprocessing.logging_config import setup_logger
from preprocessing.paths import BOUGHTER_ANNOTATED_DIR

logger = setup_logger(__name__)


def annotate_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate all antibodies in the dataset.

    Args:
        df: DataFrame from Stage 1 (boughter.csv)

    Returns:
        DataFrame with all fragments annotated
    """
    df_annotated, failures = process_sequences_to_fragments(
        df, heavy_col="heavy_seq", light_col="light_seq", id_col="id"
    )

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
