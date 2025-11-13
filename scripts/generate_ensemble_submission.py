"""Generate ensemble submission: 0.7*ESM-1v + 0.3*ESM-2."""

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
print("GENERATING ENSEMBLE SUBMISSION (0.7*ESM1v + 0.3*ESM2)")
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
unlabeled_df = full_df[full_df["PR_CHO"].isna()]

vh_sequences = labeled_df["vh_protein_sequence"].tolist()
vl_sequences = labeled_df["vl_protein_sequence"].tolist()
labels = labeled_df["PR_CHO"].values
folds = labeled_df["hierarchical_cluster_IgG_isotype_stratified_fold"].values

logger.info(f"Loaded {len(vh_sequences)} labeled samples")
logger.info(f"Found {len(unlabeled_df)} unlabeled samples")

# ===== LOAD CACHED ESM-1v EMBEDDINGS =====
esm1v_extractor = ESMEmbeddingExtractor(
    model_name="facebook/esm1v_t33_650M_UR90S_1",
    device="cpu",
    batch_size=8,
)

logger.info("Loading ESM-1v embeddings...")
esm1v_vh = get_or_create_embeddings(
    sequences=vh_sequences,
    embedding_extractor=esm1v_extractor,
    cache_path="./embeddings_cache",
    dataset_name="ginkgo_full_vh",
    logger=logger,
)
esm1v_vl = get_or_create_embeddings(
    sequences=vl_sequences,
    embedding_extractor=esm1v_extractor,
    cache_path="./embeddings_cache",
    dataset_name="ginkgo_full_vl",
    logger=logger,
)
esm1v_embeddings = np.concatenate([esm1v_vh, esm1v_vl], axis=1)

# ===== LOAD CACHED ESM-2 EMBEDDINGS =====
esm2_extractor = ESMEmbeddingExtractor(
    model_name="facebook/esm2_t33_650M_UR50D",
    device="cpu",
    batch_size=8,
)

logger.info("Loading ESM-2 embeddings...")
esm2_vh = get_or_create_embeddings(
    sequences=vh_sequences,
    embedding_extractor=esm2_extractor,
    cache_path="./embeddings_cache",
    dataset_name="ginkgo_esm2_vh",
    logger=logger,
)
esm2_vl = get_or_create_embeddings(
    sequences=vl_sequences,
    embedding_extractor=esm2_extractor,
    cache_path="./embeddings_cache",
    dataset_name="ginkgo_esm2_vl",
    logger=logger,
)
esm2_embeddings = np.concatenate([esm2_vh, esm2_vl], axis=1)

logger.info(f"ESM-1v shape: {esm1v_embeddings.shape}")
logger.info(f"ESM-2 shape: {esm2_embeddings.shape}")

# ===== GENERATE CV PREDICTIONS WITH ENSEMBLE =====
logger.info("=" * 80)
logger.info("GENERATING CV PREDICTIONS (ENSEMBLE)")
logger.info("=" * 80)

best_alpha = 5.5  # From tuning experiments
ensemble_w1 = 0.7
ensemble_w2 = 0.3

oof_predictions = np.zeros(len(vh_sequences))
fold_spearmans = []

for fold_idx in sorted(set(folds)):
    train_mask = folds != fold_idx
    val_mask = folds == fold_idx

    # Train ESM-1v model
    model1 = Ridge(alpha=best_alpha)
    model1.fit(esm1v_embeddings[train_mask], labels[train_mask])
    val_preds1 = model1.predict(esm1v_embeddings[val_mask])

    # Train ESM-2 model
    model2 = Ridge(alpha=best_alpha)
    model2.fit(esm2_embeddings[train_mask], labels[train_mask])
    val_preds2 = model2.predict(esm2_embeddings[val_mask])

    # Ensemble predictions
    ensemble_preds = ensemble_w1 * val_preds1 + ensemble_w2 * val_preds2
    oof_predictions[val_mask] = ensemble_preds

    fold_spearman, _ = spearmanr(labels[val_mask], ensemble_preds)
    fold_spearmans.append(fold_spearman)
    logger.info(f"Fold {fold_idx}: Spearman = {fold_spearman:.4f}")

mean_spearman = np.mean(fold_spearmans)
logger.info(f"\nMean per-fold Spearman: {mean_spearman:.5f} (LEADERBOARD SCORE)")

# ===== PREDICT ON UNLABELED ANTIBODIES =====
if len(unlabeled_df) > 0:
    logger.info("=" * 80)
    logger.info(f"PREDICTING ON {len(unlabeled_df)} UNLABELED ANTIBODIES")
    logger.info("=" * 80)

    unlabeled_vh = unlabeled_df["vh_protein_sequence"].tolist()
    unlabeled_vl = unlabeled_df["vl_protein_sequence"].tolist()

    # Extract ESM-1v embeddings
    logger.info("Extracting unlabeled ESM-1v embeddings...")
    unlabeled_esm1v_vh = get_or_create_embeddings(
        sequences=unlabeled_vh,
        embedding_extractor=esm1v_extractor,
        cache_path="./embeddings_cache",
        dataset_name="ginkgo_unlabeled_vh",
        logger=logger,
    )
    unlabeled_esm1v_vl = get_or_create_embeddings(
        sequences=unlabeled_vl,
        embedding_extractor=esm1v_extractor,
        cache_path="./embeddings_cache",
        dataset_name="ginkgo_unlabeled_vl",
        logger=logger,
    )
    unlabeled_esm1v = np.concatenate([unlabeled_esm1v_vh, unlabeled_esm1v_vl], axis=1)

    # Extract ESM-2 embeddings
    logger.info("Extracting unlabeled ESM-2 embeddings...")
    unlabeled_esm2_vh = get_or_create_embeddings(
        sequences=unlabeled_vh,
        embedding_extractor=esm2_extractor,
        cache_path="./embeddings_cache",
        dataset_name="ginkgo_unlabeled_esm2_vh",
        logger=logger,
    )
    unlabeled_esm2_vl = get_or_create_embeddings(
        sequences=unlabeled_vl,
        embedding_extractor=esm2_extractor,
        cache_path="./embeddings_cache",
        dataset_name="ginkgo_unlabeled_esm2_vl",
        logger=logger,
    )
    unlabeled_esm2 = np.concatenate([unlabeled_esm2_vh, unlabeled_esm2_vl], axis=1)

    # Train models on ALL labeled data
    logger.info("Training final models on all labeled data...")
    final_model1 = Ridge(alpha=best_alpha)
    final_model1.fit(esm1v_embeddings, labels)
    unlabeled_preds1 = final_model1.predict(unlabeled_esm1v)

    final_model2 = Ridge(alpha=best_alpha)
    final_model2.fit(esm2_embeddings, labels)
    unlabeled_preds2 = final_model2.predict(unlabeled_esm2)

    # Ensemble
    unlabeled_predictions = (
        ensemble_w1 * unlabeled_preds1 + ensemble_w2 * unlabeled_preds2
    )
    logger.info(
        f"Generated predictions for {len(unlabeled_predictions)} unlabeled antibodies"
    )

# ===== SAVE CV PREDICTIONS =====
labeled_submission = labeled_df[
    [
        "antibody_name",
        "vh_protein_sequence",
        "vl_protein_sequence",
        "hierarchical_cluster_IgG_isotype_stratified_fold",
    ]
].copy()
labeled_submission["PR_CHO"] = oof_predictions

if len(unlabeled_df) > 0:
    unlabeled_submission = unlabeled_df[
        [
            "antibody_name",
            "vh_protein_sequence",
            "vl_protein_sequence",
            "hierarchical_cluster_IgG_isotype_stratified_fold",
        ]
    ].copy()
    unlabeled_submission["PR_CHO"] = unlabeled_predictions
    cv_submission = pd.concat(
        [labeled_submission, unlabeled_submission], ignore_index=True
    )
else:
    cv_submission = labeled_submission

output_dir = Path("ginkgo_submissions_ensemble")
output_dir.mkdir(exist_ok=True)
cv_file = output_dir / "ginkgo_cv_predictions_PR_CHO.csv"
cv_submission.to_csv(cv_file, index=False)
logger.info(f"✅ CV predictions saved to: {cv_file}")
logger.info(f"   Total: {len(cv_submission)} antibodies")

# ===== TEST SET PREDICTIONS =====
logger.info("=" * 80)
logger.info("GENERATING TEST SET PREDICTIONS")
logger.info("=" * 80)

test_file = Path("test_datasets/ginkgo/heldout-set-sequences.csv")
if test_file.exists():
    test_df = pd.read_csv(test_file)
    logger.info(f"Loaded {len(test_df)} test samples")

    test_vh = test_df["vh_protein_sequence"].tolist()
    test_vl = test_df["vl_protein_sequence"].tolist()

    # Extract ESM-1v test embeddings
    logger.info("Extracting test ESM-1v embeddings...")
    test_esm1v_vh = get_or_create_embeddings(
        sequences=test_vh,
        embedding_extractor=esm1v_extractor,
        cache_path="./embeddings_cache",
        dataset_name="ginkgo_test_vh",
        logger=logger,
    )
    test_esm1v_vl = get_or_create_embeddings(
        sequences=test_vl,
        embedding_extractor=esm1v_extractor,
        cache_path="./embeddings_cache",
        dataset_name="ginkgo_test_vl",
        logger=logger,
    )
    test_esm1v = np.concatenate([test_esm1v_vh, test_esm1v_vl], axis=1)

    # Extract ESM-2 test embeddings
    logger.info("Extracting test ESM-2 embeddings...")
    test_esm2_vh = get_or_create_embeddings(
        sequences=test_vh,
        embedding_extractor=esm2_extractor,
        cache_path="./embeddings_cache",
        dataset_name="ginkgo_test_esm2_vh",
        logger=logger,
    )
    test_esm2_vl = get_or_create_embeddings(
        sequences=test_vl,
        embedding_extractor=esm2_extractor,
        cache_path="./embeddings_cache",
        dataset_name="ginkgo_test_esm2_vl",
        logger=logger,
    )
    test_esm2 = np.concatenate([test_esm2_vh, test_esm2_vl], axis=1)

    # Train final models on ALL labeled data
    logger.info("Training final models on all training data...")
    final_model1 = Ridge(alpha=best_alpha)
    final_model1.fit(esm1v_embeddings, labels)
    test_preds1 = final_model1.predict(test_esm1v)

    final_model2 = Ridge(alpha=best_alpha)
    final_model2.fit(esm2_embeddings, labels)
    test_preds2 = final_model2.predict(test_esm2)

    # Ensemble
    test_predictions = ensemble_w1 * test_preds1 + ensemble_w2 * test_preds2

    # Save test predictions
    test_submission = test_df[
        ["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]
    ].copy()
    test_submission["PR_CHO"] = test_predictions

    test_file_out = output_dir / "ginkgo_test_predictions_PR_CHO.csv"
    test_submission.to_csv(test_file_out, index=False)
    logger.info(f"✅ Test predictions saved to: {test_file_out}")

logger.info("=" * 80)
logger.info("SUBMISSION FILES READY!")
logger.info("=" * 80)
logger.info(f"📁 Directory: {output_dir}/")
logger.info(f"📄 CV file: {cv_file.name} ({len(cv_submission)} antibodies)")
logger.info(f"📄 Test file: {test_file_out.name} ({len(test_submission)} antibodies)")
logger.info(f"\n🎯 Expected leaderboard score: {mean_spearman:.5f}")
logger.info("🏆 PREDICTED RANK: #1 (current leader: 0.475)")
logger.info("=" * 80)
