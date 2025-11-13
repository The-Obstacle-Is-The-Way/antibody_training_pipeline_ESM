"""Rapid experimentation to beat 0.472 leaderboard score.

Try multiple approaches:
1. Alpha sweep (fine-grained)
2. TAP features without PCA
3. Different embeddings (ESM-2)
4. Simple ensembles
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from antibody_training_esm.core.trainer import get_or_create_embeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

print("=" * 80)
print("GINKGO RAPID EXPERIMENTS - BEAT 0.472!")
print("=" * 80)

# Load training data
train_assay_file = Path("train_datasets/ginkgo/GDPa1_v1.2_20250814.csv")
train_fold_file = Path("train_datasets/ginkgo/GDPa1_v1.2_sequences.csv")

assay_df = pd.read_csv(train_assay_file)
fold_df = pd.read_csv(train_fold_file)
full_df = fold_df.merge(
    assay_df[["antibody_name", "PR_CHO"]], on="antibody_name", how="left"
)
labeled_df = full_df.dropna(subset=["PR_CHO"])

vh_sequences = labeled_df["vh_protein_sequence"].tolist()
vl_sequences = labeled_df["vl_protein_sequence"].tolist()
labels = labeled_df["PR_CHO"].values
folds = labeled_df["hierarchical_cluster_IgG_isotype_stratified_fold"].values

logger.info(f"Loaded {len(vh_sequences)} labeled samples")

# ========== EXPERIMENT 1: ALPHA MICRO-TUNING ==========
logger.info("\n" + "=" * 80)
logger.info("EXPERIMENT 1: ALPHA MICRO-TUNING")
logger.info("=" * 80)

# Load cached ESM-1v embeddings
embedding_extractor = ESMEmbeddingExtractor(
    model_name="facebook/esm1v_t33_650M_UR90S_1",
    device="cpu",
    batch_size=8,
)

logger.info("Loading cached VH embeddings...")
vh_embeddings = get_or_create_embeddings(
    sequences=vh_sequences,
    embedding_extractor=embedding_extractor,
    cache_path="./embeddings_cache",
    dataset_name="ginkgo_full_vh",
    logger=logger,
)

logger.info("Loading cached VL embeddings...")
vl_embeddings = get_or_create_embeddings(
    sequences=vl_sequences,
    embedding_extractor=embedding_extractor,
    cache_path="./embeddings_cache",
    dataset_name="ginkgo_full_vl",
    logger=logger,
)

embeddings = np.concatenate([vh_embeddings, vl_embeddings], axis=1)
logger.info(f"Embeddings shape: {embeddings.shape}")

# Try alpha values around 7.0
alphas = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 10.0]
best_alpha = 7.0
best_score = 0.0

for alpha in alphas:
    fold_spearmans = []
    for fold_idx in sorted(set(folds)):
        train_mask = folds != fold_idx
        val_mask = folds == fold_idx

        X_train = embeddings[train_mask]
        y_train = labels[train_mask]
        X_val = embeddings[val_mask]
        y_val = labels[val_mask]

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)

        fold_spearman, _ = spearmanr(y_val, val_preds)
        fold_spearmans.append(fold_spearman)

    mean_spearman = np.mean(fold_spearmans)
    logger.info(f"α={alpha:5.1f} → Spearman = {mean_spearman:.5f}")

    if mean_spearman > best_score:
        best_score = mean_spearman
        best_alpha = alpha

logger.info(f"\n🏆 BEST: α={best_alpha} → {best_score:.5f}")

# ========== EXPERIMENT 2: TAP FEATURES (NO PCA) ==========
logger.info("\n" + "=" * 80)
logger.info("EXPERIMENT 2: ESM-1v + TAP FEATURES (NO PCA)")
logger.info("=" * 80)

# Load TAP features
tap_features = {
    "SFvCSP": labeled_df.get("SFvCSP", pd.Series([0] * len(labeled_df))),
    "PSH": labeled_df.get("PSH", pd.Series([0] * len(labeled_df))),
    "PPC": labeled_df.get("PPC", pd.Series([0] * len(labeled_df))),
    "PNC": labeled_df.get("PNC", pd.Series([0] * len(labeled_df))),
    "CDR_Length": labeled_df.get("CDR_Length", pd.Series([0] * len(labeled_df))),
}

# Check if TAP features exist in dataset
if "SFvCSP" in labeled_df.columns:
    logger.info("TAP features found in dataset")
    tap_array = labeled_df[["SFvCSP", "PSH", "PPC", "PNC", "CDR_Length"]].values
    embeddings_with_tap = np.concatenate([embeddings, tap_array], axis=1)
    logger.info(f"Combined shape: {embeddings_with_tap.shape}")

    # Try with best alpha from previous experiment
    fold_spearmans = []
    for fold_idx in sorted(set(folds)):
        train_mask = folds != fold_idx
        val_mask = folds == fold_idx

        X_train = embeddings_with_tap[train_mask]
        y_train = labels[train_mask]
        X_val = embeddings_with_tap[val_mask]
        y_val = labels[val_mask]

        model = Ridge(alpha=best_alpha)
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)

        fold_spearman, _ = spearmanr(y_val, val_preds)
        fold_spearmans.append(fold_spearman)

    mean_spearman = np.mean(fold_spearmans)
    logger.info(f"ESM-1v + TAP → Spearman = {mean_spearman:.5f}")
else:
    logger.warning("TAP features not found in dataset, skipping")

# ========== EXPERIMENT 3: ESM-2 650M ==========
logger.info("\n" + "=" * 80)
logger.info("EXPERIMENT 3: ESM-2 650M EMBEDDINGS")
logger.info("=" * 80)

esm2_extractor = ESMEmbeddingExtractor(
    model_name="facebook/esm2_t33_650M_UR50D",
    device="cpu",
    batch_size=8,
)

logger.info("Extracting ESM-2 VH embeddings...")
esm2_vh_embeddings = get_or_create_embeddings(
    sequences=vh_sequences,
    embedding_extractor=esm2_extractor,
    cache_path="./embeddings_cache",
    dataset_name="ginkgo_esm2_vh",
    logger=logger,
)

logger.info("Extracting ESM-2 VL embeddings...")
esm2_vl_embeddings = get_or_create_embeddings(
    sequences=vl_sequences,
    embedding_extractor=esm2_extractor,
    cache_path="./embeddings_cache",
    dataset_name="ginkgo_esm2_vl",
    logger=logger,
)

esm2_embeddings = np.concatenate([esm2_vh_embeddings, esm2_vl_embeddings], axis=1)
logger.info(f"ESM-2 embeddings shape: {esm2_embeddings.shape}")

# Test with best alpha
fold_spearmans = []
for fold_idx in sorted(set(folds)):
    train_mask = folds != fold_idx
    val_mask = folds == fold_idx

    X_train = esm2_embeddings[train_mask]
    y_train = labels[train_mask]
    X_val = esm2_embeddings[val_mask]
    y_val = labels[val_mask]

    model = Ridge(alpha=best_alpha)
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)

    fold_spearman, _ = spearmanr(y_val, val_preds)
    fold_spearmans.append(fold_spearman)

mean_spearman = np.mean(fold_spearmans)
logger.info(f"ESM-2 650M → Spearman = {mean_spearman:.5f}")

# ========== EXPERIMENT 4: ENSEMBLE ESM-1v + ESM-2 ==========
logger.info("\n" + "=" * 80)
logger.info("EXPERIMENT 4: ENSEMBLE ESM-1v + ESM-2")
logger.info("=" * 80)

# Try different ensemble weights
weights = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1)]

best_ensemble_score = 0.0
best_ensemble_weights = (0.5, 0.5)

for w1, w2 in weights:
    fold_spearmans = []
    for fold_idx in sorted(set(folds)):
        train_mask = folds != fold_idx
        val_mask = folds == fold_idx

        # Train ESM-1v model
        model1 = Ridge(alpha=best_alpha)
        model1.fit(embeddings[train_mask], labels[train_mask])
        preds1 = model1.predict(embeddings[val_mask])

        # Train ESM-2 model
        model2 = Ridge(alpha=best_alpha)
        model2.fit(esm2_embeddings[train_mask], labels[train_mask])
        preds2 = model2.predict(esm2_embeddings[val_mask])

        # Ensemble predictions
        ensemble_preds = w1 * preds1 + w2 * preds2

        fold_spearman, _ = spearmanr(labels[val_mask], ensemble_preds)
        fold_spearmans.append(fold_spearman)

    mean_spearman = np.mean(fold_spearmans)
    logger.info(f"Ensemble ({w1:.1f}*ESM1v + {w2:.1f}*ESM2) → {mean_spearman:.5f}")

    if mean_spearman > best_ensemble_score:
        best_ensemble_score = mean_spearman
        best_ensemble_weights = (w1, w2)

logger.info(f"\n🏆 BEST ENSEMBLE: {best_ensemble_weights} → {best_ensemble_score:.5f}")

# ========== FINAL SUMMARY ==========
logger.info("\n" + "=" * 80)
logger.info("FINAL RESULTS SUMMARY")
logger.info("=" * 80)
logger.info("Baseline (α=7.0):              0.4722")
logger.info(f"Best Alpha Tuning:             {best_score:.5f} (α={best_alpha})")
logger.info(
    f"Best Ensemble:                 {best_ensemble_score:.5f} {best_ensemble_weights}"
)
logger.info("=" * 80)
