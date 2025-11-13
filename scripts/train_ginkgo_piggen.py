"""Train Ginkgo model with p-IgGen antibody-specific embeddings."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

print("=" * 80)
print("GINKGO COMPETITION: p-IgGen ANTIBODY-SPECIFIC EMBEDDINGS")
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

logger.info(f"Loaded {len(vh_sequences)} samples")

# Prepare p-IgGen tokenization
model_name = "ollieturnbull/p-IgGen"
logger.info(f"Loading p-IgGen model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Tokenize VH + VL pairs (p-IgGen format: "1" + VH + VL + "2")
sequences = [
    "1" + " ".join(heavy) + " " + " ".join(light) + "2"
    for heavy, light in zip(vh_sequences, vl_sequences)
]

logger.info(f"Example sequence: {sequences[0][:100]}...")

# Extract embeddings
logger.info("Extracting p-IgGen embeddings...")
batch_size = 16
mean_pooled_embeddings = []

for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
    batch = tokenizer(
        sequences[i : i + batch_size],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )
    outputs = model(
        batch["input_ids"].to(device),
        return_rep_layers=[-1],
        output_hidden_states=True,
    )
    embeddings = outputs["hidden_states"][-1].detach().cpu().numpy()
    mean_pooled_embeddings.append(embeddings.mean(axis=1))

embeddings = np.concatenate(mean_pooled_embeddings)
logger.info(f"Embeddings shape: {embeddings.shape}")

# Cross-validation
logger.info("=" * 80)
logger.info("TRAINING WITH 5-FOLD CROSS-VALIDATION (Ridge α=7.0)")
logger.info("=" * 80)

unique_folds = sorted(set(folds))
fold_spearmans = []
oof_predictions = np.zeros(len(vh_sequences))

best_alpha = 7.0

for fold_idx in unique_folds:
    train_mask = folds != fold_idx
    val_mask = folds == fold_idx

    X_train = embeddings[train_mask]
    y_train = labels[train_mask]
    X_val = embeddings[val_mask]
    y_val = labels[val_mask]

    logger.info(f"Fold {fold_idx}: train={X_train.shape[0]}, val={X_val.shape[0]}")

    model_ridge = Ridge(alpha=best_alpha)
    model_ridge.fit(X_train, y_train)
    val_preds = model_ridge.predict(X_val)
    oof_predictions[val_mask] = val_preds

    fold_spearman, _ = spearmanr(y_val, val_preds)
    fold_spearmans.append(fold_spearman)
    logger.info(f"  Fold {fold_idx} Spearman: {fold_spearman:.4f}")

overall_oof_spearman, _ = spearmanr(labels, oof_predictions)
mean_of_folds_spearman = np.mean(fold_spearmans)

logger.info("=" * 80)
logger.info("CROSS-VALIDATION RESULTS")
logger.info("=" * 80)
logger.info(
    f"Mean of per-fold Spearmans: {mean_of_folds_spearman:.4f} ± {np.std(fold_spearmans):.4f}"
)
logger.info(f"Overall OOF Spearman: {overall_oof_spearman:.4f}")
for i, spearman in enumerate(fold_spearmans):
    logger.info(f"  Fold {i}: {spearman:.4f}")
logger.info("=" * 80)
