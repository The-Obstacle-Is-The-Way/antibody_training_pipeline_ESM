"""Train Ginkgo model with VH + VL concatenated embeddings."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from antibody_training_esm.core.trainer import get_or_create_embeddings

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Train Ridge regressor on VH + VL concatenated embeddings."""
    logger.info("=" * 80)
    logger.info("GINKGO COMPETITION: VH + VL CONCATENATED EMBEDDINGS")
    logger.info("=" * 80)

    # Load training data
    train_assay_file = Path("train_datasets/ginkgo/GDPa1_v1.2_20250814.csv")
    train_fold_file = Path("train_datasets/ginkgo/GDPa1_v1.2_sequences.csv")

    # Load assay data (labels)
    assay_df = pd.read_csv(train_assay_file)
    assay_df = assay_df.dropna(subset=["PR_CHO"])
    logger.info(f"Loaded {len(assay_df)} antibodies with PR_CHO labels")

    # Load fold assignments
    fold_df = pd.read_csv(train_fold_file)

    # Merge on antibody_name
    data_df = fold_df.merge(
        assay_df[["antibody_name", "PR_CHO"]], on="antibody_name", how="left"
    )
    data_df = data_df.dropna(subset=["PR_CHO"])

    vh_sequences = data_df["vh_protein_sequence"].tolist()
    vl_sequences = data_df["vl_protein_sequence"].tolist()
    labels = data_df["PR_CHO"].values
    folds = data_df["hierarchical_cluster_IgG_isotype_stratified_fold"].values

    logger.info(f"Total samples: {len(vh_sequences)}")
    logger.info(f"Unique folds: {sorted(set(folds))}")

    # Extract embeddings for VH and VL separately
    embedding_extractor = ESMEmbeddingExtractor(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        batch_size=8,
    )

    logger.info("Extracting VH embeddings...")
    vh_embeddings = get_or_create_embeddings(
        sequences=vh_sequences,
        embedding_extractor=embedding_extractor,
        cache_path="./embeddings_cache",
        dataset_name="ginkgo_full_vh",
        logger=logger,
    )
    logger.info(f"VH embeddings shape: {vh_embeddings.shape}")

    logger.info("Extracting VL embeddings...")
    vl_embeddings = get_or_create_embeddings(
        sequences=vl_sequences,
        embedding_extractor=embedding_extractor,
        cache_path="./embeddings_cache",
        dataset_name="ginkgo_full_vl",
        logger=logger,
    )
    logger.info(f"VL embeddings shape: {vl_embeddings.shape}")

    # Concatenate VH + VL embeddings
    embeddings = np.concatenate([vh_embeddings, vl_embeddings], axis=1)
    logger.info(f"Concatenated embeddings shape: {embeddings.shape}")

    # Cross-validation with predefined folds
    unique_folds = sorted(set(folds))
    fold_results = []
    oof_predictions = np.zeros(len(vh_sequences))

    logger.info("=" * 80)
    logger.info("TRAINING WITH 5-FOLD CROSS-VALIDATION (Ridge α=5.0)")
    logger.info("=" * 80)

    # Use best alpha from previous sweep
    best_alpha = 5.0

    for fold_idx in unique_folds:
        train_mask = folds != fold_idx
        val_mask = folds == fold_idx

        X_train = embeddings[train_mask]
        y_train = labels[train_mask]
        X_val = embeddings[val_mask]
        y_val = labels[val_mask]

        logger.info(f"Fold {fold_idx}: train={X_train.shape[0]}, val={X_val.shape[0]}")

        # Train Ridge
        model = Ridge(alpha=best_alpha)
        model.fit(X_train, y_train)

        # Predict on validation set
        val_preds = model.predict(X_val)
        oof_predictions[val_mask] = val_preds

        # Calculate Spearman
        spearman, pvalue = spearmanr(y_val, val_preds)
        fold_results.append(spearman)

        logger.info(f"  Fold {fold_idx} Spearman: {spearman:.4f} (p={pvalue:.2e})")

    # Overall Spearman on OOF predictions
    overall_spearman, overall_pvalue = spearmanr(labels, oof_predictions)

    logger.info("=" * 80)
    logger.info("CROSS-VALIDATION RESULTS")
    logger.info("=" * 80)
    logger.info(
        f"Overall Spearman: {overall_spearman:.4f} (p-value: {overall_pvalue:.2e})"
    )
    logger.info(
        f"Mean CV Spearman: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}"
    )
    for i, spearman in enumerate(fold_results):
        logger.info(f"  Fold {i}: {spearman:.4f}")
    logger.info("=" * 80)

    # Train final model on full dataset
    logger.info("Training final Ridge model on full dataset...")
    final_model = Ridge(alpha=best_alpha)
    final_model.fit(embeddings, labels)

    # Save model (pickle for compatibility)
    import pickle

    output_dir = Path("models/ginkgo_2025_vh_vl")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "ginkgo_ridge_vh_vl.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": final_model,
                "alpha": best_alpha,
                "model_name": "facebook/esm1v_t33_650M_UR90S_1",
                "embedding_dim": embeddings.shape[1],
            },
            f,
        )

    logger.info(f"Model saved to: {model_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
