"""
Shared utilities for antibody fragment extraction using ANARCI (IMGT).

Consolidates ANARCI annotation logic used across all datasets to ensure
consistent CDR/FWR definitions and numbering.

Methodology:
- Uses riot_na (ANARCI wrapper)
- Strict IMGT numbering scheme
- Reconstructs V-domains from fragments to ensure gap-free sequences (P0 fix)
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import riot_na

from preprocessing.logging_config import setup_logger

logger = setup_logger(__name__)

# Initialize ANARCI for amino acid annotation (IMGT scheme)
# Global instance to avoid re-initialization overhead
_ANNOTATOR = None


def get_annotator() -> Any:
    """Lazy load ANARCI annotator."""
    global _ANNOTATOR
    if _ANNOTATOR is None:
        _ANNOTATOR = riot_na.create_riot_aa()
    return _ANNOTATOR


def annotate_sequence(seq_id: str, sequence: str, chain: str) -> dict[str, str] | None:
    """
    Annotate a single amino acid sequence using ANARCI (IMGT).

    Uses strict IMGT boundaries per Sakhnini et al. 2025.

    Args:
        seq_id: Unique identifier for the sequence
        sequence: Amino acid sequence string
        chain: 'H' for heavy or 'L' for light

    Returns:
        Dictionary with extracted fragments, or None if annotation fails.
        Keys: fwr1_aa_H, cdr1_aa_H, ..., full_seq_H, cdrs_H, fwrs_H
    """
    assert chain in ("H", "L"), "chain must be 'H' or 'L'"
    annotator = get_annotator()

    try:
        annotation = annotator.run_on_sequence(seq_id, sequence)

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
            logger.debug(f"  ANARCI returned no CDRs for {seq_id} ({chain} chain)")
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
        logger.warning(f"  ANARCI failed for {seq_id} ({chain} chain): {e}")
        return None


def process_sequences_to_fragments(
    df: pd.DataFrame,
    heavy_col: str = "heavy_seq",
    light_col: str | None = "light_seq",
    id_col: str = "id",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Process a DataFrame of antibodies to extract fragments.

    Args:
        df: Input DataFrame
        heavy_col: Column name for heavy chain sequence
        light_col: Column name for light chain sequence (optional if not present)
        id_col: Column name for sequence ID

    Returns:
        Tuple of (Annotated DataFrame, List of failed IDs)
    """
    logger.info(f"\nAnnotating {len(df)} antibodies with ANARCI (strict IMGT)...")

    results = []
    failures = []

    has_heavy = heavy_col in df.columns
    has_light = light_col is not None and light_col in df.columns

    for i, (idx, row) in enumerate(df.iterrows()):
        seq_id = str(row.get(id_col, idx))
        result = row.to_dict()
        annotation_success = False

        # Annotate Heavy
        if has_heavy and pd.notna(row[heavy_col]):
            heavy_frags = annotate_sequence(seq_id, row[heavy_col], "H")
            if heavy_frags:
                result.update(heavy_frags)
                annotation_success = True
            else:
                # If heavy chain fails, mark as failure?
                # Or partial success? Typically we need at least one chain.
                pass

        # Annotate Light
        if has_light and light_col and pd.notna(row[light_col]):
            light_frags = annotate_sequence(seq_id, row[light_col], "L")
            if light_frags:
                result.update(light_frags)
                annotation_success = True  # At least one chain succeeded

        if annotation_success:
            # Create paired/combined fragments if possible
            if "full_seq_H" in result and "full_seq_L" in result:
                result["vh_vl"] = result["full_seq_H"] + result["full_seq_L"]
                result["all_cdrs"] = result.get("cdrs_H", "") + result.get("cdrs_L", "")
                result["all_fwrs"] = result.get("fwrs_H", "") + result.get("fwrs_L", "")

            # Handle Nanobodies (VHH only)
            elif "full_seq_H" in result:
                result["vh_vl"] = result["full_seq_H"]
                result["all_cdrs"] = result.get("cdrs_H", "")
                result["all_fwrs"] = result.get("fwrs_H", "")

            results.append(result)
        else:
            failures.append(seq_id)

        # Progress
        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i + 1}/{len(df)} ({len(failures)} failures)")

    df_annotated = pd.DataFrame(results)

    return df_annotated, failures
