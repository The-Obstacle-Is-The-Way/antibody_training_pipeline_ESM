#!/usr/bin/env python3
"""
Convert Harvey dataset supplementary materials into a clean CSV for polyreactivity analysis.

This script:
1. Extracts nanobody sequences from Supplementary Table 1 (markdown)
2. Merges PSR binding scores from Supplementary Data 3 (harvey.xlsx)
3. Creates binary labels based on PSR median fluorescence intensity threshold
4. Produces canonical CSV with sequence + PSR polyreactivity scores

Dataset: Harvey et al. 2022 Nature Communications
"An in silico method to assess antibody fragment polyreactivity"
Index set: 48 nanobodies with defined polyreactivity levels

Usage:
    python3 scripts/convert_harvey_to_csv.py \
        --markdown literature/markdown/harvey-et-al-2022-supplementary-information/harvey-et-al-2022-supplementary-information.md \
        --excel test_datasets/harvey.xlsx \
        --output test_datasets/harvey.csv

Reference:
- literature/markdown/harvey-et-al-2022-in-silico-method-to-assess-antibody-fragment-polyreactivity/
- Issue #4: Harvey dataset preprocessing
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

LOG = logging.getLogger("convert_harvey_to_csv")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Valid amino acids supported by ESM
VALID_AA = set("ACDEFGHIKLMNPQRSTVWYX")

# PSR threshold for polyreactivity classification (based on paper Figure 2)
# Nanobodies with PSR > 2000 are considered polyreactive
PSR_THRESHOLD = 2000.0

# Column order for output CSV
PRIMARY_COLUMNS = [
    "id",
    "sequence",
    "psr_score",
    "label",
    "polyreactivity_category",
    "source",
    "sequence_length",
]


# --------------------------------------------------------------------------- #
# Sequence parsing from Supplementary Table 1
# --------------------------------------------------------------------------- #


def parse_supplementary_table_1(markdown_path: Path) -> Dict[str, str]:
    """
    Parse nanobody sequences from Supplementary Table 1 in the markdown file.

    Each nanobody entry spans exactly 3 rows in the 2-column table:
    | A02' | QVQLVES...  |  <- row 1: name + sequence part 1
    |      | GGSTN...    |  <- row 2: empty/spaces + sequence part 2
    |      | LVDYW...    |  <- row 3: empty/spaces + sequence part 3

    Returns:
        Dictionary mapping nanobody ID → full sequence
    """
    with open(markdown_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sequences: Dict[str, str] = {}
    current_name: Optional[str] = None
    current_seq_parts: List[str] = []
    in_table = False
    rows_since_name = 0

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        # Start of table
        if "Supplementary Table 1" in line and "Nanobody Sequences" in line:
            in_table = True
            continue

        # End of table
        if in_table and "Supplementary Table 2" in line:
            # Save last entry
            if current_name and current_seq_parts:
                full_seq = clean_sequence("".join(current_seq_parts))
                if 100 < len(full_seq) < 150:
                    sequences[current_name] = full_seq
            break

        if not in_table:
            continue

        # Skip headers, separators
        if (
            not line_stripped
            or line_stripped.startswith("|---")
            or "Bold = mutation" in line
            or line_stripped.startswith("| Name |")
        ):
            continue

        # Parse table rows
        if line_stripped.startswith("|"):
            # Split by pipe - keep whitespace to detect empty cells
            raw_parts = line_stripped.split("|")
            # Remove first/last empty strings from split
            if raw_parts[0] == "":
                raw_parts = raw_parts[1:]
            if raw_parts and raw_parts[-1].strip() == "":
                raw_parts = raw_parts[:-1]

            if len(raw_parts) < 2:
                continue

            name_col = raw_parts[0].strip()
            seq_col = raw_parts[1].strip()

            # Check if this row starts a new nanobody
            # Name column is non-empty and doesn't look like a sequence
            is_new_nanobody = (
                len(name_col) > 0
                and len(name_col) < 25
                and not name_col.startswith("QVQL")
                and not name_col.startswith("EVQL")
                and not name_col.startswith("SGAST")  # continuation pattern
                and not name_col.startswith("GGSTN")  # continuation pattern
            )

            if is_new_nanobody:
                # Save previous nanobody
                if current_name and current_seq_parts:
                    full_seq = clean_sequence("".join(current_seq_parts))
                    if 100 < len(full_seq) < 150:
                        sequences[current_name] = full_seq
                    # Skip variants/mutations for now - focus on base 48
                    elif len(full_seq) > 0:
                        LOG.debug(f"Skipped {current_name}: {len(full_seq)} aa")

                # Start new nanobody
                current_name = name_col
                current_seq_parts = [seq_col]
                rows_since_name = 0
            else:
                # Continuation row
                if current_name and rows_since_name < 2:  # Max 3 rows per nanobody
                    current_seq_parts.append(seq_col)
                    rows_since_name += 1

    LOG.info(f"Parsed {len(sequences)} nanobody sequences from Supplementary Table 1")

    return sequences


def clean_sequence(seq: str) -> str:
    """
    Clean a nanobody sequence by:
    - Removing HTML tags (<br>, etc.)
    - Removing whitespace and newlines
    - Keeping only valid amino acids
    - Uppercasing
    """
    # Remove HTML tags
    seq = re.sub(r"<[^>]+>", "", seq)

    # Remove all whitespace
    seq = re.sub(r"\s+", "", seq)

    # Keep only valid amino acids
    seq = "".join(ch for ch in seq.upper() if ch in VALID_AA)

    return seq


# --------------------------------------------------------------------------- #
# PSR score extraction from Excel
# --------------------------------------------------------------------------- #


def extract_psr_scores(excel_path: Path) -> pd.DataFrame:
    """
    Extract PSR binding scores from Supplementary Data 3 (harvey.xlsx).

    Sheet: 'Supp Figure 3A' contains raw PSR staining values
    - Column 'Unnamed: 0': Nanobody ID
    - Columns 'Bioreplicate 1/2/3': PSR fluorescence values

    Returns:
        DataFrame with columns: [id, psr_score]
        psr_score is the mean of 3 bioreplicates
    """
    df = pd.read_excel(excel_path, sheet_name="Supp Figure 3A")

    # Skip header row (row 0) and read from row 1
    df = df.iloc[1:].reset_index(drop=True)

    # Rename columns
    df.columns = ["id", "rep1", "rep2", "rep3"]

    # Convert replicates to numeric
    for col in ["rep1", "rep2", "rep3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Calculate mean PSR score
    df["psr_score"] = df[["rep1", "rep2", "rep3"]].mean(axis=1)

    # Keep only id and psr_score
    df = df[["id", "psr_score"]].copy()

    # Remove any rows with missing data
    df = df.dropna()

    LOG.info(f"Extracted PSR scores for {len(df)} nanobodies")

    return df


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #


def create_harvey_dataset(sequences: Dict[str, str], psr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge sequences with PSR scores and create labels.

    Args:
        sequences: Dictionary of {id: sequence}
        psr_df: DataFrame with [id, psr_score]

    Returns:
        DataFrame with complete Harvey dataset
    """
    # Convert sequences dict to DataFrame
    seq_df = pd.DataFrame(
        [{"id": name, "sequence": seq} for name, seq in sequences.items()]
    )

    LOG.info(f"Total sequences: {len(seq_df)}")
    LOG.info(f"Total PSR scores: {len(psr_df)}")

    # Merge sequences with PSR scores
    df = seq_df.merge(psr_df, on="id", how="inner")

    LOG.info(f"Merged dataset: {len(df)} nanobodies")

    # Create binary label: 0 = specific (low PSR), 1 = polyreactive (high PSR)
    df["label"] = (df["psr_score"] > PSR_THRESHOLD).astype(int)

    # Create categorical labels for interpretability
    df["polyreactivity_category"] = df["psr_score"].apply(
        lambda x: "high" if x > 3000 else ("moderate" if x > PSR_THRESHOLD else "low")
    )

    # Add metadata
    df["source"] = "harvey2022"
    df["sequence_length"] = df["sequence"].str.len()

    return df


def prepare_output(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns for consistent output format."""
    ordered_cols = [col for col in PRIMARY_COLUMNS if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in ordered_cols]
    final_cols = ordered_cols + remaining_cols
    return df[final_cols]


def summarize(df: pd.DataFrame) -> None:
    """Print key summary statistics."""
    LOG.info("=" * 60)
    LOG.info("Harvey Dataset Summary")
    LOG.info("=" * 60)
    LOG.info(f"Total nanobodies: {len(df)}")
    LOG.info(f"Sequence length: min={df['sequence_length'].min()}, max={df['sequence_length'].max()}")
    LOG.info(
        f"PSR score range: {df['psr_score'].min():.1f} - {df['psr_score'].max():.1f}"
    )

    LOG.info("\nPolyreactivity categories:")
    for cat, count in df["polyreactivity_category"].value_counts().sort_index().items():
        LOG.info(f"  {cat}: {count} ({count/len(df)*100:.1f}%)")

    LOG.info("\nBinary labels:")
    for label, count in df["label"].value_counts().sort_index().items():
        label_name = "Specific (low PSR)" if label == 0 else "Polyreactive (high PSR)"
        LOG.info(f"  {label_name}: {count} ({count/len(df)*100:.1f}%)")

    LOG.info(f"\nPSR threshold for polyreactivity: {PSR_THRESHOLD}")
    LOG.info("=" * 60)


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Harvey supplementary materials to CSV."
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path(
            "literature/markdown/harvey-et-al-2022-supplementary-information/"
            "harvey-et-al-2022-supplementary-information.md"
        ),
        help="Path to supplementary information markdown file",
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=Path("test_datasets/harvey.xlsx"),
        help="Path to harvey.xlsx (Supplementary Data 3)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_datasets/harvey.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    # Validate inputs
    if not args.markdown.exists():
        raise FileNotFoundError(f"Markdown file not found: {args.markdown}")
    if not args.excel.exists():
        raise FileNotFoundError(f"Excel file not found: {args.excel}")

    # Parse sequences from markdown
    sequences = parse_supplementary_table_1(args.markdown)

    # Extract PSR scores from Excel
    psr_df = extract_psr_scores(args.excel)

    # Create merged dataset
    df = create_harvey_dataset(sequences, psr_df)

    # Prepare output
    df = prepare_output(df)

    # Save to CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    # Print summary
    summarize(df)

    LOG.info(f"\n✓ Saved Harvey dataset to {args.output}")


if __name__ == "__main__":
    main()
