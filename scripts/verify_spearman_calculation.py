"""Verify our Spearman calculation matches what leaderboard expects.

The leaderboard says "average Spearman rank correlation across the 5 folds".
This could mean either:
1. Overall OOF Spearman: spearmanr(all_labels, all_oof_predictions)
2. Mean of per-fold Spearmans: mean([spearman_fold0, spearman_fold1, ...])

Let's test both methods and see which one matches leaderboard scores!
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from antibody_training_esm.core.trainer import get_or_create_embeddings

logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

print("=" * 80)
print("VERIFYING SPEARMAN CALCULATION METHOD")
print("=" * 80)

# Load data
train_assay_file = Path("train_datasets/ginkgo/GDPa1_v1.2_20250814.csv")
train_fold_file = Path("train_datasets/ginkgo/GDPa1_v1.2_sequences.csv")

assay_df = pd.read_csv(train_assay_file)
assay_df = assay_df.dropna(subset=["PR_CHO"])
fold_df = pd.read_csv(train_fold_file)
data_df = fold_df.merge(
    assay_df[["antibody_name", "PR_CHO"]], on="antibody_name", how="left"
)
data_df = data_df.dropna(subset=["PR_CHO"])

vh_sequences = data_df["vh_protein_sequence"].tolist()
vl_sequences = data_df["vl_protein_sequence"].tolist()
labels = data_df["PR_CHO"].values
folds = data_df["hierarchical_cluster_IgG_isotype_stratified_fold"].values

# Extract embeddings
import logging

logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

embedding_extractor = ESMEmbeddingExtractor(
    model_name="facebook/esm1v_t33_650M_UR90S_1",
    device="cpu",
    batch_size=8,
)

vh_embeddings = get_or_create_embeddings(
    sequences=vh_sequences,
    embedding_extractor=embedding_extractor,
    cache_path="./embeddings_cache",
    dataset_name="ginkgo_full_vh",
    logger=logger,
)

vl_embeddings = get_or_create_embeddings(
    sequences=vl_sequences,
    embedding_extractor=embedding_extractor,
    cache_path="./embeddings_cache",
    dataset_name="ginkgo_full_vl",
    logger=logger,
)

embeddings = np.concatenate([vh_embeddings, vl_embeddings], axis=1)

# Train with best alpha
best_alpha = 7.0
print(f"Training with Ridge (alpha={best_alpha})...\n")

unique_folds = sorted(set(folds))
fold_spearmans = []
oof_predictions = np.zeros(len(vh_sequences))

for fold_idx in unique_folds:
    train_mask = folds != fold_idx
    val_mask = folds == fold_idx

    X_train = embeddings[train_mask]
    y_train = labels[train_mask]
    X_val = embeddings[val_mask]
    y_val = labels[val_mask]

    model = Ridge(alpha=best_alpha)
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    oof_predictions[val_mask] = val_preds

    # Per-fold Spearman
    fold_spearman, _ = spearmanr(y_val, val_preds)
    fold_spearmans.append(fold_spearman)
    print(f"Fold {fold_idx}: Spearman = {fold_spearman:.4f} (n={sum(val_mask)})")

# Calculate both methods
overall_oof_spearman, _ = spearmanr(labels, oof_predictions)
mean_of_folds_spearman = np.mean(fold_spearmans)

print("\n" + "=" * 80)
print("COMPARISON OF TWO METHODS")
print("=" * 80)
print(f"Method 1 - Overall OOF Spearman: {overall_oof_spearman:.4f}")
print("  (Spearman on all 197 OOF predictions)")
print()
print(
    f"Method 2 - Mean of per-fold Spearmans: {mean_of_folds_spearman:.4f} ± {np.std(fold_spearmans):.4f}"
)
print(f"  (Average of 5 fold Spearmans: {[f'{s:.4f}' for s in fold_spearmans]})")
print()
print(f"Difference: {abs(overall_oof_spearman - mean_of_folds_spearman):.4f}")
print("=" * 80)
print()
print("🤔 WHICH ONE MATCHES THE LEADERBOARD?")
print("The leaderboard says 'average Spearman rank correlation across the 5 folds'")
print()
print("Based on typical ML competition conventions:")
print("  - 'average across folds' usually means Method 2 (mean of per-fold Spearmans)")
print()
print(f"✅ Our best score for submission is likely: {mean_of_folds_spearman:.4f}")
print("=" * 80)
