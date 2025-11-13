"""Prepare combined Boughter + GDPa1 dataset for transfer learning experiments.

This script creates a combined_datasets/ folder with properly merged and normalized
data from:
- Boughter: 914 VH+VL pairs, ELISA polyreactivity (binary 0/1)
- GDPa1: 197 labeled VH+VL pairs, PR_CHO polyreactivity (continuous)

Strategy:
- Use Boughter annotated files (validated on Jain/Shehata)
- Filter to include_in_training==True (914 sequences, production QC)
- Merge VH+VL pairs on 'id' column
- Normalize labels for compatibility

Data Provenance:
- Boughter: train_datasets/boughter/annotated/{VH_only,VL_only}_boughter.csv
- GDPa1: train_datasets/ginkgo/GDPa1_v1.2_*.csv
- Output: combined_datasets/boughter_ginkgo_combined.csv

Competition Rules Compliance:
- Competition allows training on external datasets
- Must report CV predictions on GDPa1 using their folds
- This is for pre-training/transfer learning only
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

print("=" * 80)
print("PREPARING COMBINED DATASET: Boughter + GDPa1")
print("=" * 80)

# ===== LOAD BOUGHTER DATA =====
logger.info("Loading Boughter data (annotated files)...")

boughter_vh = pd.read_csv(
    "train_datasets/boughter/annotated/VH_only_boughter.csv", comment="#"
)
boughter_vl = pd.read_csv(
    "train_datasets/boughter/annotated/VL_only_boughter.csv", comment="#"
)

logger.info(f"Boughter VH loaded: {len(boughter_vh)} sequences")
logger.info(f"Boughter VL loaded: {len(boughter_vl)} sequences")

# Filter to training subset (include_in_training==True, 914 sequences)
boughter_vh_train = boughter_vh[boughter_vh["include_in_training"]].copy()
boughter_vl_train = boughter_vl[boughter_vl["include_in_training"]].copy()

logger.info(
    f"Filtered to training subset: {len(boughter_vh_train)} sequences (include_in_training==True)"
)

# Merge VH+VL on 'id' column
boughter_merged = boughter_vh_train.merge(
    boughter_vl_train[["id", "sequence"]], on="id", suffixes=("_vh", "_vl")
)

logger.info(f"Merged VH+VL pairs: {len(boughter_merged)}")

# Rename columns to match GDPa1 format
boughter_combined = pd.DataFrame(
    {
        "antibody_id": boughter_merged["id"],
        "antibody_name": boughter_merged["id"],  # Use id as name
        "vh_protein_sequence": boughter_merged["sequence_vh"],
        "vl_protein_sequence": boughter_merged["sequence_vl"],
        "label_binary": boughter_merged["label"],  # 0/1
        "num_flags": boughter_merged["num_flags"],
        "flag_category": boughter_merged["flag_category"],
        "subset": boughter_merged["subset"],
        "dataset": "boughter",
    }
)

logger.info(f"Boughter combined: {len(boughter_combined)} antibodies")
logger.info(
    f"Label distribution: {boughter_combined['label_binary'].value_counts().to_dict()}"
)

# ===== LOAD GINKGO DATA =====
logger.info("\nLoading GDPa1 data...")

ginkgo_sequences = pd.read_csv("train_datasets/ginkgo/GDPa1_v1.2_sequences.csv")
ginkgo_assay = pd.read_csv("train_datasets/ginkgo/GDPa1_v1.2_20250814.csv")

# Merge sequences + assay data
ginkgo_full = ginkgo_sequences.merge(
    ginkgo_assay[["antibody_name", "PR_CHO"]], on="antibody_name", how="left"
)

# Filter to labeled only (197 sequences)
ginkgo_labeled = ginkgo_full.dropna(subset=["PR_CHO"]).copy()

logger.info(f"GDPa1 loaded: {len(ginkgo_labeled)} labeled antibodies")
logger.info(
    f"PR_CHO range: [{ginkgo_labeled['PR_CHO'].min():.3f}, {ginkgo_labeled['PR_CHO'].max():.3f}]"
)

# Convert PR_CHO to binary for Boughter-style training
# Use median split: PR_CHO > median = polyreactive (1), <= median = specific (0)
pr_cho_median = ginkgo_labeled["PR_CHO"].median()
ginkgo_labeled["label_binary"] = (ginkgo_labeled["PR_CHO"] > pr_cho_median).astype(int)

logger.info(f"PR_CHO median: {pr_cho_median:.3f}")
logger.info(f"Binary split: {ginkgo_labeled['label_binary'].value_counts().to_dict()}")

# Format GDPa1 to match Boughter structure
ginkgo_combined = pd.DataFrame(
    {
        "antibody_id": ginkgo_labeled["antibody_id"],
        "antibody_name": ginkgo_labeled["antibody_name"],
        "vh_protein_sequence": ginkgo_labeled["vh_protein_sequence"],
        "vl_protein_sequence": ginkgo_labeled["vl_protein_sequence"],
        "label_binary": ginkgo_labeled["label_binary"],
        "PR_CHO": ginkgo_labeled["PR_CHO"],
        "fold": ginkgo_labeled["hierarchical_cluster_IgG_isotype_stratified_fold"],
        "dataset": "ginkgo",
    }
)

# ===== COMBINE DATASETS =====
logger.info("\nCombining Boughter + GDPa1...")

# Add missing columns with NaN
boughter_combined["PR_CHO"] = np.nan
boughter_combined["fold"] = -1  # Mark as non-GDPa1

ginkgo_combined["num_flags"] = np.nan
ginkgo_combined["flag_category"] = "unknown"
ginkgo_combined["subset"] = "ginkgo"

# Combine
combined = pd.concat([boughter_combined, ginkgo_combined], ignore_index=True)

logger.info(f"\nCombined dataset: {len(combined)} total antibodies")
logger.info(f"  - Boughter: {len(boughter_combined)} (914 training)")
logger.info(f"  - GDPa1:    {len(ginkgo_combined)} (197 labeled)")

# ===== SAVE COMBINED DATASET =====
output_dir = Path("combined_datasets")
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "boughter_ginkgo_combined.csv"
combined.to_csv(output_file, index=False)
logger.info(f"\n✅ Combined dataset saved: {output_file}")

# ===== SAVE SEPARATE BOUGHTER/GINKGO FILES =====
boughter_file = output_dir / "boughter_training.csv"
boughter_combined.to_csv(boughter_file, index=False)
logger.info(f"✅ Boughter training saved: {boughter_file}")

ginkgo_file = output_dir / "ginkgo_labeled.csv"
ginkgo_combined.to_csv(ginkgo_file, index=False)
logger.info(f"✅ GDPa1 labeled saved: {ginkgo_file}")

# ===== SUMMARY STATS =====
logger.info("\n" + "=" * 80)
logger.info("SUMMARY STATISTICS")
logger.info("=" * 80)
logger.info("Boughter:")
logger.info(f"  - Total antibodies: {len(boughter_combined)}")
logger.info(f"  - Label 0 (specific): {(boughter_combined['label_binary'] == 0).sum()}")
logger.info(
    f"  - Label 1 (polyreactive): {(boughter_combined['label_binary'] == 1).sum()}"
)
logger.info(
    f"  - VH avg length: {boughter_combined['vh_protein_sequence'].str.len().mean():.1f}"
)
logger.info(
    f"  - VL avg length: {boughter_combined['vl_protein_sequence'].str.len().mean():.1f}"
)

logger.info("\nGDPa1:")
logger.info(f"  - Total antibodies: {len(ginkgo_combined)}")
logger.info(f"  - Label 0 (specific): {(ginkgo_combined['label_binary'] == 0).sum()}")
logger.info(
    f"  - Label 1 (polyreactive): {(ginkgo_combined['label_binary'] == 1).sum()}"
)
logger.info(f"  - PR_CHO mean: {ginkgo_combined['PR_CHO'].mean():.3f}")
logger.info(f"  - PR_CHO std: {ginkgo_combined['PR_CHO'].std():.3f}")
logger.info(
    f"  - VH avg length: {ginkgo_combined['vh_protein_sequence'].str.len().mean():.1f}"
)
logger.info(
    f"  - VL avg length: {ginkgo_combined['vl_protein_sequence'].str.len().mean():.1f}"
)

logger.info("\nCombined:")
logger.info(f"  - Total antibodies: {len(combined)}")
logger.info(f"  - Boughter: {(combined['dataset'] == 'boughter').sum()}")
logger.info(f"  - GDPa1: {(combined['dataset'] == 'ginkgo').sum()}")
logger.info(f"  - Label balance: {combined['label_binary'].value_counts().to_dict()}")

logger.info("\n" + "=" * 80)
logger.info("READY FOR TRANSFER LEARNING EXPERIMENTS!")
logger.info("=" * 80)
logger.info("Next steps:")
logger.info("  1. Pre-train on Boughter (914 samples)")
logger.info("  2. Fine-tune on GDPa1 (197 samples)")
logger.info("  3. Report CV on GDPa1 folds (competition requirement)")
