"""Alpha sweep on VH+VL baseline model."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from antibody_training_esm.core.trainer import get_or_create_embeddings

print("=" * 80)
print("ALPHA SWEEP ON VH+VL BASELINE MODEL")
print("Testing alphas: 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 50.0")
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

# Extract embeddings (cached)
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

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

# Alpha sweep
alphas = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 50.0]
results = []

print("\n" + "=" * 80)
print("RUNNING ALPHA SWEEP")
print("=" * 80)

for alpha in alphas:
    # Cross-validation
    unique_folds = sorted(set(folds))
    oof_predictions = np.zeros(len(vh_sequences))

    for fold_idx in unique_folds:
        train_mask = folds != fold_idx
        val_mask = folds == fold_idx

        X_train = embeddings[train_mask]
        y_train = labels[train_mask]
        X_val = embeddings[val_mask]

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        oof_predictions[val_mask] = model.predict(X_val)

    overall_spearman, _ = spearmanr(labels, oof_predictions)
    results.append((alpha, overall_spearman))
    print(f"Alpha={alpha:6.1f} → Spearman={overall_spearman:.4f}")

# Find best
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
best_alpha, best_spearman = max(results, key=lambda x: x[1])
print(f"\n🏆 BEST: Alpha={best_alpha}, Spearman={best_spearman:.4f}")
print("=" * 80)
