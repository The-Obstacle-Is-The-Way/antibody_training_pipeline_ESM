"""Train Ginkgo competition model with XGBoost instead of Ridge."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from antibody_training_esm.core.trainer import get_or_create_embeddings

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Train XGBoost regressor on Ginkgo GDPa1 dataset."""
    logger.info("=" * 80)
    logger.info("GINKGO COMPETITION: XGBOOST REGRESSOR")
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

    sequences = data_df["vh_protein_sequence"].tolist()
    labels = data_df["PR_CHO"].values
    folds = data_df["hierarchical_cluster_IgG_isotype_stratified_fold"].values

    logger.info(f"Total samples: {len(sequences)}")
    logger.info(f"Unique folds: {sorted(set(folds))}")

    # Extract embeddings (cached)
    embedding_extractor = ESMEmbeddingExtractor(
        model_name="facebook/esm1v_t33_650M_UR90S_1",
        device="cpu",
        batch_size=8,
    )

    logger.info("Extracting ESM-1v embeddings...")
    embeddings = get_or_create_embeddings(
        sequences=sequences,
        embedding_extractor=embedding_extractor,
        cache_path="./embeddings_cache",
        dataset_name="ginkgo_full",
        logger=logger,
    )
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Cross-validation with predefined folds
    unique_folds = sorted(set(folds))
    fold_results = []
    oof_predictions = np.zeros(len(sequences))

    logger.info("=" * 80)
    logger.info("TRAINING WITH 5-FOLD CROSS-VALIDATION")
    logger.info("=" * 80)

    for fold_idx in unique_folds:
        train_mask = folds != fold_idx
        val_mask = folds == fold_idx

        X_train = embeddings[train_mask]
        y_train = labels[train_mask]
        X_val = embeddings[val_mask]
        y_val = labels[val_mask]

        logger.info(f"Fold {fold_idx}: train={X_train.shape[0]}, val={X_val.shape[0]}")

        # Train XGBoost
        xgb_params = {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "objective": "reg:squarederror",
            "random_state": 42,
            "tree_method": "hist",
            "verbosity": 0,
        }

        model = xgb.XGBRegressor(**xgb_params)
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
    logger.info("Training final XGBoost model on full dataset...")
    final_model = xgb.XGBRegressor(**xgb_params)
    final_model.fit(embeddings, labels)

    # Save model
    output_dir = Path("models/ginkgo_2025_xgboost")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "ginkgo_xgboost_regressor.json"
    final_model.save_model(model_path)
    logger.info(f"Model saved to: {model_path}")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
