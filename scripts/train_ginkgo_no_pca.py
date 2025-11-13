"""Train Ginkgo model with VH+VL + TAP + Subtypes (NO PCA).

Test if PCA is hurting performance by using full 2560D embeddings.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from antibody_training_esm.core.trainer import get_or_create_embeddings

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_tap_features(tap_path: Path, antibody_names: list[str]) -> np.ndarray:
    """Load and align TAP features with antibody dataset."""
    df_tap = pd.read_csv(tap_path)
    tap_dict = df_tap.set_index("antibody_name")[
        ["SFvCSP", "PSH", "PPC", "PNC", "CDR Length"]
    ].to_dict("index")

    tap_features = []
    for name in antibody_names:
        if name in tap_dict:
            row = tap_dict[name]
            tap_features.append(
                [row["SFvCSP"], row["PSH"], row["PPC"], row["PNC"], row["CDR Length"]]
            )
        else:
            tap_features.append([0.0, 0.0, 0.0, 0.0, 0.0])

    return np.array(tap_features)


def encode_subtypes(hc_subtypes: list[str], lc_subtypes: list[str]) -> np.ndarray:
    """One-hot encode antibody subtypes."""
    hc_map = {"IgG1": [1, 0, 0], "IgG2": [0, 1, 0], "IgG4": [0, 0, 1]}
    lc_map = {"Kappa": [1, 0], "Lambda": [0, 1]}

    subtype_features = []
    for hc, lc in zip(hc_subtypes, lc_subtypes, strict=False):
        hc_encoded = hc_map.get(hc, [0, 0, 0])
        lc_encoded = lc_map.get(lc, [0, 0])
        subtype_features.append(hc_encoded + lc_encoded)

    return np.array(subtype_features)


def main() -> None:
    """Train Ridge regressor WITHOUT PCA."""
    logger.info("=" * 80)
    logger.info("GINKGO COMPETITION: VH+VL + TAP + Subtypes (NO PCA)")
    logger.info("=" * 80)

    # Load training data
    train_assay_file = Path("train_datasets/ginkgo/GDPa1_v1.2_20250814.csv")
    train_fold_file = Path("train_datasets/ginkgo/GDPa1_v1.2_sequences.csv")
    tap_features_file = Path(
        "reference_repos/abdev-benchmark/data/features/processed_features/GDPa1/TAP.csv"
    )

    assay_df = pd.read_csv(train_assay_file)
    assay_df = assay_df.dropna(subset=["PR_CHO"])
    logger.info(f"Loaded {len(assay_df)} antibodies with PR_CHO labels")

    fold_df = pd.read_csv(train_fold_file)
    data_df = fold_df.merge(
        assay_df[["antibody_name", "PR_CHO"]], on="antibody_name", how="left"
    )
    data_df = data_df.dropna(subset=["PR_CHO"])

    vh_sequences = data_df["vh_protein_sequence"].tolist()
    vl_sequences = data_df["vl_protein_sequence"].tolist()
    antibody_names = data_df["antibody_name"].tolist()
    hc_subtypes = data_df["hc_subtype"].tolist()
    lc_subtypes = data_df["lc_subtype"].tolist()
    labels = data_df["PR_CHO"].values
    folds = data_df["hierarchical_cluster_IgG_isotype_stratified_fold"].values

    logger.info(f"Total samples: {len(vh_sequences)}")

    # Extract embeddings
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

    logger.info("Extracting VL embeddings...")
    vl_embeddings = get_or_create_embeddings(
        sequences=vl_sequences,
        embedding_extractor=embedding_extractor,
        cache_path="./embeddings_cache",
        dataset_name="ginkgo_full_vl",
        logger=logger,
    )

    # Concatenate embeddings (NO PCA!)
    embeddings = np.concatenate([vh_embeddings, vl_embeddings], axis=1)
    logger.info(f"Concatenated embeddings shape: {embeddings.shape}")

    # Load TAP features
    logger.info("Loading TAP features...")
    tap_features = load_tap_features(tap_features_file, antibody_names)

    # Encode subtypes
    logger.info("Encoding antibody subtypes...")
    subtype_features = encode_subtypes(hc_subtypes, lc_subtypes)

    # Combine all features (NO PCA)
    all_features = np.concatenate([embeddings, tap_features, subtype_features], axis=1)
    logger.info(f"Combined features shape: {all_features.shape}")
    logger.info(f"  - Embeddings: {embeddings.shape[1]}D")
    logger.info(f"  - TAP: {tap_features.shape[1]}D")
    logger.info(f"  - Subtypes: {subtype_features.shape[1]}D")

    # Standardize
    scaler = StandardScaler()
    features = scaler.fit_transform(all_features)

    # Cross-validation
    unique_folds = sorted(set(folds))
    fold_results = []
    oof_predictions = np.zeros(len(vh_sequences))

    logger.info("=" * 80)
    logger.info("TRAINING WITH 5-FOLD CROSS-VALIDATION (Ridge α=5.0)")
    logger.info("=" * 80)

    best_alpha = 5.0

    for fold_idx in unique_folds:
        train_mask = folds != fold_idx
        val_mask = folds == fold_idx

        X_train = features[train_mask]
        y_train = labels[train_mask]
        X_val = features[val_mask]
        y_val = labels[val_mask]

        logger.info(f"Fold {fold_idx}: train={X_train.shape[0]}, val={X_val.shape[0]}")

        model = Ridge(alpha=best_alpha)
        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        oof_predictions[val_mask] = val_preds

        spearman, pvalue = spearmanr(y_val, val_preds)
        fold_results.append(spearman)

        logger.info(f"  Fold {fold_idx} Spearman: {spearman:.4f} (p={pvalue:.2e})")

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


if __name__ == "__main__":
    main()
