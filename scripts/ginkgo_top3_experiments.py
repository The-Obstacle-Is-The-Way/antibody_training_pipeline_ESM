"""Run top 3 experiments to beat 0.504 leaderboard score.

Experiments:
1. Three-model ensemble (ESM-1v + ESM-2 + p-IgGen)
2. Optimized weighted ensemble
3. Stacked meta-learner
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from transformers import AutoModelForCausalLM, AutoTokenizer

from antibody_training_esm.core.embeddings import ESMEmbeddingExtractor
from antibody_training_esm.core.trainer import get_or_create_embeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

print("=" * 80)
print("TOP 3 EXPERIMENTS - TARGET: 0.504+")
print("=" * 80)

# Load data
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

# ===== LOAD ALL EMBEDDINGS =====
logger.info("Loading all cached embeddings...")

# ESM-1v
esm1v_extractor = ESMEmbeddingExtractor(
    model_name="facebook/esm1v_t33_650M_UR90S_1", device="cpu", batch_size=8
)
esm1v_vh = get_or_create_embeddings(
    vh_sequences, esm1v_extractor, "./embeddings_cache", "ginkgo_full_vh", logger
)
esm1v_vl = get_or_create_embeddings(
    vl_sequences, esm1v_extractor, "./embeddings_cache", "ginkgo_full_vl", logger
)
esm1v_embeddings = np.concatenate([esm1v_vh, esm1v_vl], axis=1)
logger.info(f"ESM-1v loaded: {esm1v_embeddings.shape}")

# ESM-2
esm2_extractor = ESMEmbeddingExtractor(
    model_name="facebook/esm2_t33_650M_UR50D", device="cpu", batch_size=8
)
esm2_vh = get_or_create_embeddings(
    vh_sequences, esm2_extractor, "./embeddings_cache", "ginkgo_esm2_vh", logger
)
esm2_vl = get_or_create_embeddings(
    vl_sequences, esm2_extractor, "./embeddings_cache", "ginkgo_esm2_vl", logger
)
esm2_embeddings = np.concatenate([esm2_vh, esm2_vl], axis=1)
logger.info(f"ESM-2 loaded: {esm2_embeddings.shape}")

# p-IgGen (antibody-specific)
logger.info("Extracting p-IgGen embeddings...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
piggen_model_name = "ollieturnbull/p-IgGen"
piggen_tokenizer = AutoTokenizer.from_pretrained(piggen_model_name)
piggen_model = AutoModelForCausalLM.from_pretrained(piggen_model_name).to(device)


def embed_piggen(vh_seqs, vl_seqs, batch_size=8):
    """Extract p-IgGen embeddings with format: 1 + VH + VL + 2"""
    sequences = [
        "1" + " ".join(vh) + " " + " ".join(vl) + "2"
        for vh, vl in zip(vh_seqs, vl_seqs, strict=True)
    ]

    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        inputs = piggen_tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = piggen_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # Last layer
            # Mean pool
            mean_pooled = hidden_states.mean(dim=1).cpu().numpy()
            embeddings.append(mean_pooled)

    return np.vstack(embeddings)


piggen_cache = Path("./embeddings_cache/ginkgo_piggen_embeddings.npy")
if piggen_cache.exists():
    piggen_embeddings = np.load(piggen_cache)
    logger.info(f"p-IgGen loaded from cache: {piggen_embeddings.shape}")
else:
    piggen_embeddings = embed_piggen(vh_sequences, vl_sequences)
    np.save(piggen_cache, piggen_embeddings)
    logger.info(f"p-IgGen extracted: {piggen_embeddings.shape}")


# ===== EXPERIMENT 1: THREE-MODEL ENSEMBLE =====
logger.info("\n" + "=" * 80)
logger.info("EXPERIMENT 1: THREE-MODEL ENSEMBLE (ESM-1v + ESM-2 + p-IgGen)")
logger.info("=" * 80)

best_alpha = 5.5
embeddings_dict = {
    "esm1v": esm1v_embeddings,
    "esm2": esm2_embeddings,
    "piggen": piggen_embeddings,
}

# Try different weight combinations
weight_configs = [
    (0.6, 0.2, 0.2, "balanced"),
    (0.7, 0.2, 0.1, "esm1v_heavy"),
    (0.5, 0.3, 0.2, "esm2_boost"),
    (0.5, 0.25, 0.25, "equal_minor"),
]

best_three_score = 0.0
best_three_weights = None

for w1, w2, w3, name in weight_configs:
    fold_spearmans = []
    for fold_idx in sorted(set(folds)):
        train_mask = folds != fold_idx
        val_mask = folds == fold_idx

        # Train all three models
        model1 = Ridge(alpha=best_alpha)
        model1.fit(esm1v_embeddings[train_mask], labels[train_mask])
        pred1 = model1.predict(esm1v_embeddings[val_mask])

        model2 = Ridge(alpha=best_alpha)
        model2.fit(esm2_embeddings[train_mask], labels[train_mask])
        pred2 = model2.predict(esm2_embeddings[val_mask])

        model3 = Ridge(alpha=best_alpha)
        model3.fit(piggen_embeddings[train_mask], labels[train_mask])
        pred3 = model3.predict(piggen_embeddings[val_mask])

        # Ensemble
        ensemble_pred = w1 * pred1 + w2 * pred2 + w3 * pred3

        fold_spearman, _ = spearmanr(labels[val_mask], ensemble_pred)
        fold_spearmans.append(fold_spearman)

    mean_spearman = np.mean(fold_spearmans)
    logger.info(f"  {name} ({w1:.1f}, {w2:.1f}, {w3:.1f}): {mean_spearman:.5f}")

    if mean_spearman > best_three_score:
        best_three_score = mean_spearman
        best_three_weights = (w1, w2, w3)

logger.info(f"\n🏆 BEST 3-MODEL: {best_three_weights} → {best_three_score:.5f}")


# ===== EXPERIMENT 2: OPTIMIZED WEIGHTED ENSEMBLE =====
logger.info("\n" + "=" * 80)
logger.info("EXPERIMENT 2: OPTIMIZED WEIGHTED ENSEMBLE")
logger.info("=" * 80)

# Generate OOF predictions for each model
oof_preds = {name: np.zeros(len(labels)) for name in embeddings_dict}

for name, embeddings in embeddings_dict.items():
    for fold_idx in sorted(set(folds)):
        train_mask = folds != fold_idx
        val_mask = folds == fold_idx

        model = Ridge(alpha=best_alpha)
        model.fit(embeddings[train_mask], labels[train_mask])
        oof_preds[name][val_mask] = model.predict(embeddings[val_mask])


# Optimize weights
def objective(weights):
    weights = np.abs(weights)  # Ensure positive
    weights = weights / weights.sum()  # Normalize

    ensemble = sum(
        w * oof_preds[name] for w, name in zip(weights, embeddings_dict.keys(), strict=False)
    )
    return -spearmanr(labels, ensemble)[0]


result = minimize(
    objective,
    x0=np.array([1 / 3, 1 / 3, 1 / 3]),
    method="Nelder-Mead",
    options={"maxiter": 1000},
)

optimal_weights = np.abs(result.x)
optimal_weights = optimal_weights / optimal_weights.sum()
optimal_score = -result.fun

logger.info(f"Optimal weights: {dict(zip(embeddings_dict.keys(), optimal_weights, strict=False))}")
logger.info(f"🏆 OPTIMIZED ENSEMBLE: {optimal_score:.5f}")


# ===== EXPERIMENT 3: STACKED ENSEMBLE =====
logger.info("\n" + "=" * 80)
logger.info("EXPERIMENT 3: STACKED META-LEARNER")
logger.info("=" * 80)

# Stack OOF predictions as features for meta-learner
stacked_features = np.column_stack([oof_preds[name] for name in embeddings_dict])

# Train meta-learner with CV
meta_oof_preds = np.zeros(len(labels))
fold_spearmans = []

for fold_idx in sorted(set(folds)):
    train_mask = folds != fold_idx
    val_mask = folds == fold_idx

    meta_model = Ridge(alpha=1.0)  # Lower alpha for meta-learner
    meta_model.fit(stacked_features[train_mask], labels[train_mask])
    meta_oof_preds[val_mask] = meta_model.predict(stacked_features[val_mask])

    fold_spearman, _ = spearmanr(labels[val_mask], meta_oof_preds[val_mask])
    fold_spearmans.append(fold_spearman)

stacked_score = np.mean(fold_spearmans)
logger.info(f"🏆 STACKED ENSEMBLE: {stacked_score:.5f}")


# ===== FINAL SUMMARY =====
logger.info("\n" + "=" * 80)
logger.info("FINAL RESULTS")
logger.info("=" * 80)
logger.info("Current leader:           0.504")
logger.info("Our baseline:             0.486")
logger.info(f"Best 3-model ensemble:    {best_three_score:.5f}")
logger.info(f"Optimized weights:        {optimal_score:.5f}")
logger.info(f"Stacked meta-learner:     {stacked_score:.5f}")
logger.info("=" * 80)

# Save results
results_df = pd.DataFrame(
    [
        {"experiment": "baseline_ensemble", "score": 0.486},
        {"experiment": "three_model_ensemble", "score": best_three_score},
        {"experiment": "optimized_weights", "score": optimal_score},
        {"experiment": "stacked_meta", "score": stacked_score},
    ]
)
results_df.to_csv("experiment_results.csv", index=False)
logger.info("Results saved to experiment_results.csv")
