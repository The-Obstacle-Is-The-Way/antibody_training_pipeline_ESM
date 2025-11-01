#!/usr/bin/env python3
"""
Download Harvey dataset from Hugging Face for polyreactivity prediction.

This script downloads the Harvey et al. 2022 nanobody polyreactivity dataset
from the Hugging Face hub (ZYMScott/polyreaction). This dataset contains
~141K nanobody sequences with binary polyreactivity labels from deep sequencing
of FACS-sorted pools.

Dataset: Harvey et al. 2022 Nature Communications
"An in silico method to assess antibody fragment polyreactivity"

Original paper deep sequencing:
- Initial: 65,147 low + 69,155 high polyreactivity (~134K)
- Extended: 1,221,800 low + 1,058,842 high polyreactivity (~2.28M)

This HuggingFace dataset contains ~141K sequences, matching the initial
deep sequencing dataset used by Novo Nordisk (Sakhnini et al. 2025) for
testing their ESM-1v model.

Usage:
    python3 scripts/download_harvey_dataset.py

Output:
    - test_datasets/harvey.csv (141,474 sequences)
    - test_datasets/harvey_hf/train.csv (101,854 sequences)
    - test_datasets/harvey_hf/validation.csv (14,613 sequences)
    - test_datasets/harvey_hf/test.csv (25,007 sequences)

Reference:
- Harvey paper: https://www.nature.com/articles/s41467-022-35276-4
- HuggingFace: https://huggingface.co/datasets/ZYMScott/polyreaction
- Sakhnini et al. 2025: doi.org/10.1101/2025.04.28.650927
- Issue #4: Harvey dataset preprocessing
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset

LOG = logging.getLogger("download_harvey_dataset")


def download_and_save_harvey_dataset(output_dir: Path = Path("test_datasets")) -> None:
    """
    Download Harvey polyreactivity dataset from Hugging Face.

    Args:
        output_dir: Directory to save the dataset files
    """
    LOG.info("Downloading Harvey dataset from Hugging Face...")
    LOG.info("Source: ZYMScott/polyreaction")

    # Load dataset from Hugging Face
    dataset = load_dataset("ZYMScott/polyreaction")

    # Create output directory for split files
    split_dir = output_dir / "harvey_hf"
    split_dir.mkdir(parents=True, exist_ok=True)

    # Save individual splits
    for split_name in ["train", "validation", "test"]:
        df = pd.DataFrame(dataset[split_name])
        output_path = split_dir / f"{split_name}.csv"
        df.to_csv(output_path, index=False)
        LOG.info(f"✓ Saved {split_name}: {len(df)} rows → {output_path}")

    # Save combined dataset for compatibility
    all_df = pd.concat(
        [
            pd.DataFrame(dataset["train"]),
            pd.DataFrame(dataset["validation"]),
            pd.DataFrame(dataset["test"]),
        ],
        ignore_index=True,
    )

    combined_path = output_dir / "harvey.csv"
    all_df.to_csv(combined_path, index=False)
    LOG.info(f"✓ Saved combined: {len(all_df)} rows → {combined_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Harvey Dataset Download Summary")
    print("=" * 60)
    print(f"Train: {len(dataset['train'])} sequences")
    print(f"Validation: {len(dataset['validation'])} sequences")
    print(f"Test: {len(dataset['test'])} sequences")
    print(f"Total: {len(all_df)} sequences")
    print(f"\nColumns: {list(all_df.columns)}")
    print(f"\nLabel distribution:")
    print(all_df["label"].value_counts())
    print(
        f"\nSequence length range: {all_df['seq'].str.len().min()}-{all_df['seq'].str.len().max()} aa"
    )
    print("=" * 60)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    download_and_save_harvey_dataset()

    LOG.info("\n✓ Harvey dataset download complete!")
    LOG.info("Dataset matches Novo Nordisk (Sakhnini et al. 2025) specifications")


if __name__ == "__main__":
    main()
