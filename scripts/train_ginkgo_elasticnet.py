"""Train Ginkgo model with ElasticNet (L1 + L2 regularization)."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNetCV

from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from antibody_training_esm.core.trainer import get_or_create_embeddings

logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

print("=" * 80)
print("GINKGO COMPETITION: VH+VL WITH ELASTICNET")
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

print(f"Loaded {len(vh_sequences)} samples")

# Extract embeddings
embedding_extractor = ESMEmbeddingExtractor(
    model_name="facebook/esm1v_t33_650M_UR90S_1",
    device="cpu",
    batch_size=8,
)

print("Extracting VH embeddings...")
vh_embeddings = get_or_create_embeddings(
    sequences=vh_sequences,
    embedding_extractor=embedding_extractor,
    cache_path="./embeddings_cache",
    dataset_name="ginkgo_full_vh",
    logger=logger,
)

print("Extracting VL embeddings...")
vl_embeddings = get_or_create_embeddings(
    sequences=vl_sequences,
    embedding_extractor=embedding_extractor,
    cache_path="./embeddings_cache",
    dataset_name="ginkgo_full_vl",
    logger=logger,
)

embeddings = np.concatenate([vh_embeddings, vl_embeddings], axis=1)
print(f"Embeddings shape: {embeddings.shape}")

# Cross-validation with ElasticNetCV
print("\n" + "=" * 80)
print("TRAINING WITH 5-FOLD CROSS-VALIDATION (ElasticNetCV)")
print("=" * 80)

unique_folds = sorted(set(folds))
fold_results = []
oof_predictions = np.zeros(len(vh_sequences))

for fold_idx in unique_folds:
    train_mask = folds != fold_idx
    val_mask = folds == fold_idx

    X_train = embeddings[train_mask]
    y_train = labels[train_mask]
    X_val = embeddings[val_mask]
    y_val = labels[val_mask]

    print(f"Fold {fold_idx}: train={X_train.shape[0]}, val={X_val.shape[0]}")

    # ElasticNetCV will find best alpha automatically
    model = ElasticNetCV(
        l1_ratio=0.5,  # 50% L1, 50% L2
        alphas=[0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
        cv=3,
        max_iter=10000,
        random_state=42,
    )
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    oof_predictions[val_mask] = val_preds

    spearman, pvalue = spearmanr(y_val, val_preds)
    fold_results.append(spearman)

    print(f"  Fold {fold_idx} Spearman: {spearman:.4f} (alpha={model.alpha_:.3f})")

overall_spearman, overall_pvalue = spearmanr(labels, oof_predictions)

print("=" * 80)
print("CROSS-VALIDATION RESULTS")
print("=" * 80)
print(f"Overall Spearman: {overall_spearman:.4f} (p-value: {overall_pvalue:.2e})")
print(f"Mean CV Spearman: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
for i, spearman in enumerate(fold_results):
    print(f"  Fold {i}: {spearman:.4f}")
print("=" * 80)
